"""OME-Zarr / OME-NGFF reader (v0.4 and v0.5).

Hybrid design:

* **Metadata** (axes, per-level scale -> MPP, level dimensions, downsamples) comes
  from ``ngff-zarr`` (``from_ngff_zarr``), which handles NGFF 0.1-0.5 uniformly.
* **Coordinate mapping** (level-0 px -> a pyramid level's index) is per-axis and
  honours each level's ``translation`` - see ``read_level_mapping`` - read straight
  from the multiscales ``coordinateTransformations`` (ngff-zarr surfaces only scale).
* **Hot reads** go straight to the underlying ``zarr-python`` level arrays, so a
  patch read only touches the chunks/shards it overlaps - and stays safe for
  multi-worker DataLoaders (no dask scheduler in the read path).

Handles the canonical single-image layout (multiscales at the store root) and the
``bioformats2raw`` layout (image series under an integer subgroup, e.g. ``0/``).
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict
from functools import partial
from urllib.parse import urlsplit

import numpy as np

from raw2features.core.geometry import Region, Size
from raw2features.core.plugins import register
from raw2features.core.uris import (
    join_uri_path,
    redact_uri_credentials,
    source_uri,
)

from .base import WSISource

# Default bound for the decompressed-chunk cache, in number of 2D chunk planes.
# A 512x512 uint8 plane is 256 KiB, so 512 planes ~= 128 MiB per reader. The
# whole cohort opens one reader at a time, so this is a per-slide working set.
# Overridable via the ``RAW2FEATURES_CHUNK_CACHE`` env var (0 disables the cache).
_DEFAULT_CHUNK_CACHE_PLANES = 512


def _chunk_cache_capacity() -> int:
    raw = os.environ.get("RAW2FEATURES_CHUNK_CACHE")
    if raw is None:
        return _DEFAULT_CHUNK_CACHE_PLANES
    try:
        return max(0, int(raw))
    except ValueError:
        warnings.warn(
            f"invalid RAW2FEATURES_CHUNK_CACHE={raw!r}; using default "
            f"{_DEFAULT_CHUNK_CACHE_PLANES}",
            stacklevel=2,
        )
        return _DEFAULT_CHUNK_CACHE_PLANES


class _ChunkCache:
    """Thread-safe LRU cache of decompressed 2D (y, x) chunk planes.

    Keyed by ``(level, c_index, chunk_y, chunk_x)``; the value is the decompressed
    chunk plane (uint8/source dtype, edge-clipped to the array bounds), normalised
    from the source axis order to ``(y, x)``. Adjacent ~224 px patches read in grid
    order overlap the same 512x512 chunks, so caching the decompressed plane lets
    the second patch reuse it instead of re-decompressing - and an RGB read touches
    three channel chunks, all cached.

    The reads run on the pipeline's parallel read-worker pool, so get/put are
    guarded by a lock. ``capacity`` bounds the cache to N planes (LRU eviction);
    ``capacity == 0`` disables caching (every call misses and reads straight from
    zarr, preserving the pre-cache behaviour exactly).
    """

    def __init__(self, capacity: int = _DEFAULT_CHUNK_CACHE_PLANES) -> None:
        self.capacity = int(capacity)
        self._store: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def get_or_read(self, key: tuple, reader):
        """Return the cached plane for *key*, else call ``reader()`` and cache it.

        Decompression (``reader()``) happens outside the lock so concurrent misses
        on *different* chunks decompress in parallel (the win on a slow FS). A rare
        duplicate decode of the same chunk under contention is harmless: the planes
        are identical, so equivalence holds regardless of which copy wins the slot.
        """
        if self.capacity <= 0:
            return reader()
        with self._lock:
            plane = self._store.get(key)
            if plane is not None:
                self._store.move_to_end(key)
                self.hits += 1
                return plane
            self.misses += 1
        plane = reader()
        with self._lock:
            self._store[key] = plane
            self._store.move_to_end(key)
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)
        return plane


# Conversion factors to micrometers for common UDUNITS spatial units.
_UNIT_TO_UM = {
    None: 1.0,
    "micrometer": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "millimeter": 1_000.0,
    "mm": 1_000.0,
    "nanometer": 1e-3,
    "nm": 1e-3,
    "centimeter": 1e4,
    "cm": 1e4,
    "meter": 1e6,
    "m": 1e6,
}


@register("readers", "omezarr")
class OmeZarrReader(WSISource):
    """Reader for OME-Zarr (OME-NGFF v0.4 / v0.5) pyramids."""

    name = "omezarr"

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._open = False
        self._dims: tuple[str, ...] = ()
        self._mpp: float | None = None
        self._level_dims: list[Size] = []
        self._downsamples: list[float] = []
        # Per-axis read mapping (see read_level_mapping): a level-0 (x, y) maps to
        # level L's array index as round(x / downsample_x[L] + offset_x[L]). offset is
        # the translation correction (in level-L px); both are 0/isotropic for the
        # common pathology pyramid, and only differ for anisotropic or translation-
        # bearing sources - which the old x-only, translation-free mapping mis-read.
        self._downsamples_x: list[float] = []
        self._downsamples_y: list[float] = []
        self._level_off_x: list[float] = []
        self._level_off_y: list[float] = []
        self._arrays: list[object] = []
        self._ngff_version: str | None = None
        self._ome_version: str | None = None  # NGFF 0.5+ keeps version in `ome`
        self._channel_names: list[str] | None = None
        # Self-description of the source coordinate frame (captured as plain VALUES
        # so a coordinate-systems emitter is a re-encode, not a
        # re-extraction). axes = the NGFF axis order; axis_units = per-axis unit
        # string; scale_um = per-axis level-0 physical pixel size in µm; the
        # level-0 translation/origin (µm) is the source's coordinateTransformations
        # translation, or None when it carries none (the common case).
        self._axes: tuple[str, ...] = ()
        self._axis_units: dict[str, str | None] = {}
        self._scale_um: dict[str, float] = {}
        self._level0_translation_um: dict[str, float] | None = None
        # Per-level (chunk_h, chunk_w) along the y/x axes, read from the zarr
        # array's own ``.chunks`` (never hard-coded) so the cache keys align with
        # the on-disk chunk grid for whatever layout the store actually uses.
        self._chunk_hw: list[tuple[int, int]] = []
        self._chunk_cache = _ChunkCache(_chunk_cache_capacity())
        # Float pixels have no dtype range to distinguish normalized [0, 1] from
        # byte-like [0, 255]. This multiplier is inferred once from level 0 at open
        # (255 or 1), then reused for every region and pyramid level. ``None`` means
        # the level-0 dtype is not floating and keeps the integer path untouched.
        self._float_to_uint8_scale: float | None = None
        # Query-authenticated HTTP paths use a custom read-only store so the raw
        # query is attached after every metadata/chunk key. Cache one per child root.
        self._source_stores: dict[str, object] = {}

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> OmeZarrReader:
        import ngff_zarr

        prefix, multiscales = self._resolve_multiscales()
        ms_path = self.path if prefix == "" else join_uri_path(self.path, prefix)

        ms = ngff_zarr.from_ngff_zarr(self._store_for(ms_path))
        images = ms.images
        im0 = images[0]
        self._dims = tuple(im0.dims)
        if "x" not in self._dims or "y" not in self._dims:
            raise ValueError(
                f"{source_uri(self.path)}: multiscales axes lack x/y: {self._dims}"
            )
        self._ngff_version = str(
            multiscales.get("version") or self._ome_version or "unknown"
        )

        # MPP at level 0 from the x/y scale. The physical pixel size is only known
        # when the source declares an axis UNIT - NGFF makes the unit optional, and a
        # unit-less space axis means "arbitrary units" (pixels), NOT micrometers. So a
        # source without an x/y unit is treated as UNCALIBRATED (mpp_level0 = None)
        # rather than silently assumed to be µm - the pipeline then fails loud and asks
        # for an explicit source pixel size (apply_source_mpp / --source-mpp). The
        # per-axis downsample RATIOS used for reads are unit-independent, so reads of a
        # calibrated source are unaffected.
        units = im0.axes_units or {}
        self._axes = self._dims
        self._axis_units = {ax: units.get(ax) for ax in self._dims}
        calibrated = units.get("x") is not None and units.get("y") is not None
        if calibrated:
            sx_um = self._to_um(im0.scale["x"], units.get("x"))
            sy_um = self._to_um(im0.scale["y"], units.get("y"))
            # mpp_level0 is the scalar x/y mean kept for the (scale-isotropic) pathology
            # convention; record the per-axis scale too, and warn rather than silently
            # average when x and y differ (the per-axis scale_um is the faithful value).
            if min(sx_um, sy_um) > 0 and abs(sx_um - sy_um) / min(sx_um, sy_um) > 1e-3:
                warnings.warn(
                    f"{source_uri(self.path)}: anisotropic level-0 pixel size "
                    f"(x={sx_um:g} µm, y={sy_um:g} µm); mpp_level0 reports their mean "
                    f"({(sx_um + sy_um) / 2.0:g}). Per-axis scale in source.scale_um.",
                    stacklevel=2,
                )
            self._mpp = (sx_um + sy_um) / 2.0
            self._scale_um = {"x": sx_um, "y": sy_um}
        else:
            self._mpp = None  # uncalibrated: no x/y axis unit -> physical size unknown
            self._scale_um = {}
        self._level0_translation_um = self._read_level0_translation_um(
            multiscales, units
        )

        # Level dimensions (W, H) and downsamples from x scale ratios.
        xi, yi = self._dims.index("x"), self._dims.index("y")
        self._level_dims = [
            Size(int(im.data.shape[xi]), int(im.data.shape[yi])) for im in images
        ]
        s0 = images[0].scale["x"]
        self._downsamples = [float(im.scale["x"] / s0) for im in images]
        self._build_level_mapping(multiscales["datasets"], xi, yi)

        # Plain zarr arrays for the hot read path.
        import zarr

        self._arrays = [
            zarr.open_array(
                self._store_for(join_uri_path(ms_path, d["path"])), mode="r"
            )
            for d in multiscales["datasets"]
        ]
        if len(self._arrays) != len(self._level_dims):
            raise ValueError(
                f"{source_uri(self.path)}: dataset/level count mismatch "
                f"({len(self._arrays)} arrays vs {len(self._level_dims)} levels)"
            )

        # Cache the per-level y/x chunk extents straight from each array's own
        # ``.chunks`` (zarr is the source of truth for the chunk grid).
        self._chunk_hw = [
            (int(a.chunks[yi]), int(a.chunks[xi]))  # type: ignore[attr-defined]
            for a in self._arrays
        ]
        self._float_to_uint8_scale = self._infer_float_to_uint8_scale(self._arrays[0])
        self._channel_names = self._read_omero_channels(ms_path)
        self._warn_plane_collapse(self._arrays[0].shape)  # type: ignore[attr-defined]
        self._chunk_cache.clear()
        self._open = True
        return self

    def _warn_plane_collapse(self, level0_shape: tuple[int, ...]) -> None:
        """Warn for every extra axis with extent > 1 (only index 0 is read).

        The reader serves a single 2D ``(y, x[, c])`` plane per patch, indexing every
        axis other than x, y, and c at 0. Any non-singleton extra axis is therefore
        reduced to its first plane - fine if intended, a trap if not - so we surface
        it once at open rather than embedding plane 0 without a word.
        """
        for pos, axis in enumerate(self._dims):
            if axis in {"x", "y", "c"}:
                continue
            extent = int(level0_shape[pos])
            if extent > 1:
                warnings.warn(
                    f"{source_uri(self.path)}: source has {axis}={extent}; only "
                    f"{axis}=0 is read (the reader serves one 2D plane per patch).",
                    stacklevel=2,
                )

    def close(self) -> None:
        self._arrays = []
        self._chunk_hw = []
        self._float_to_uint8_scale = None
        self._chunk_cache.clear()
        for store in self._source_stores.values():
            store.close()  # type: ignore[attr-defined]
            fs = getattr(store, "fs", None)
            if fs is not None and hasattr(fs, "close"):
                fs.close()
        self._source_stores.clear()
        self._open = False

    def _store_for(self, path: str):
        """Return a query-safe HTTP store, or the original non-query path."""

        parsed = urlsplit(path)
        if parsed.scheme.casefold() not in {"http", "https"} or not (
            parsed.query or parsed.fragment
        ):
            return path
        store = self._source_stores.get(path)
        if store is None:
            from ._http_store import query_http_store

            store = query_http_store(path)
            self._source_stores[path] = store
        return store

    # -- metadata ------------------------------------------------------------
    @property
    def mpp(self) -> float | None:
        return self._mpp

    @property
    def level_dimensions(self) -> list[Size]:
        return list(self._level_dims)

    def level_downsamples(self) -> list[float]:
        return list(self._downsamples)

    def read_level_mapping(self, level: int) -> tuple[float, float, float, float]:
        """Per-axis ``(downsample_x, downsample_y, offset_x, offset_y)`` for ``level``.

        A level-0 ``(x, y)`` maps to this level's array index as
        ``round(x / downsample_x + offset_x)`` / ``round(y / downsample_y + offset_y)``.
        Unlike the scalar ``level_downsamples()``, this honours per-axis scale
        (anisotropic pyramids) and any per-level ``translation`` (half-pixel
        corrections, stage offsets) in the source's ``coordinateTransformations``. For
        the common isotropic, translation-free pyramid it is exactly
        ``(ds, ds, 0, 0)`` - identical to the previous x-only mapping.
        """
        return (
            self._downsamples_x[level],
            self._downsamples_y[level],
            self._level_off_x[level],
            self._level_off_y[level],
        )

    def apply_source_mpp(self, mpp_level0: float) -> None:
        """Set the level-0 physical pixel size (µm/px) for an *uncalibrated* source.

        Only meaningful when the source declared no x/y axis unit, so ``mpp`` is
        ``None``. This supplies the missing physical scale (the ``--source-mpp``
        override) so MPP-aware extraction can proceed. It sets the absolute scale
        used for level selection / ``level0_patch`` only; the per-axis downsample
        ratios (unit-independent) are unchanged. Assumed isotropic.
        """
        if mpp_level0 <= 0:
            raise ValueError(f"source-mpp must be positive, got {mpp_level0}")
        self._mpp = float(mpp_level0)
        self._scale_um = {"x": float(mpp_level0), "y": float(mpp_level0)}

    @property
    def channel_names(self) -> list[str] | None:
        return list(self._channel_names) if self._channel_names else None

    @property
    def ngff_version(self) -> str | None:
        return self._ngff_version

    @property
    def axes(self) -> tuple[str, ...]:
        """The source NGFF axis order (e.g. ``("c", "y", "x")``)."""
        return self._axes

    @property
    def axis_units(self) -> dict[str, str | None]:
        """Per-axis unit string as declared by the source (may be ``None``)."""
        return dict(self._axis_units)

    @property
    def scale_um(self) -> dict[str, float]:
        """Per-axis level-0 physical pixel size in µm (``{"x": .., "y": ..}``)."""
        return dict(self._scale_um)

    @property
    def level0_translation_um(self) -> dict[str, float] | None:
        """Source level-0 translation/origin in µm, or ``None`` if it carries none.

        This is the translation component of the source's NGFF
        ``coordinateTransformations`` for level 0 (plus any multiscales-level
        translation), converted to µm. It is recorded for relocatability but is
        **not** applied to ``coords`` (which stay level-0 pixels, origin top-left);
        a downstream consumer that needs the source's physical frame combines it
        with the pixel coords. ``None`` (no translation) is the common case.
        """
        if not self._level0_translation_um:
            return None
        return dict(self._level0_translation_um)

    # -- reads ---------------------------------------------------------------
    def read_region(self, region: Region) -> np.ndarray:
        if not self._open:
            raise RuntimeError(
                "reader is not open; use `with OmeZarrReader(path) as r:`"
            )
        arr = self._arrays[region.level]
        dsx, dsy, ox, oy = self.read_level_mapping(region.level)
        x0 = int(round(region.location.x / dsx + ox))
        y0 = int(round(region.location.y / dsy + oy))
        w, h = region.size.width, region.size.height
        return self._read_hwc(arr, region.level, x0, y0, w, h)

    def read_region_channels(self, region: Region) -> np.ndarray:
        """Native multi-channel read -> ``(h, w, C)`` in the source dtype.

        Shares the cached chunk reader with :meth:`read_region` but keeps every
        channel and the native dtype (no RGB collapse / 8-bit cast), for multiplex
        models. Border reads are zero-padded (background).
        """
        if not self._open:
            raise RuntimeError(
                "reader is not open; use `with OmeZarrReader(path) as r:`"
            )
        arr = self._arrays[region.level]
        dsx, dsy, ox, oy = self.read_level_mapping(region.level)
        x0 = int(round(region.location.x / dsx + ox))
        y0 = int(round(region.location.y / dsy + oy))
        w, h = region.size.width, region.size.height
        block, sliced = self._read_block_cached(arr, region.level, x0, y0, w, h)
        target = ["y", "x"] + (["c"] if "c" in sliced else [])
        block = np.transpose(block, [sliced.index(t) for t in target])
        if "c" not in sliced:
            block = block[..., None]
        nc = block.shape[2]
        if block.shape[0] != h or block.shape[1] != w:
            out = np.zeros((h, w, nc), dtype=block.dtype)
            dst_y = min(h, max(0, -y0))
            dst_x = min(w, max(0, -x0))
            ph = min(block.shape[0], h - dst_y)
            pw = min(block.shape[1], w - dst_x)
            if ph > 0 and pw > 0:
                out[dst_y : dst_y + ph, dst_x : dst_x + pw, :] = block[:ph, :pw, :]
            block = out
        return np.ascontiguousarray(block)

    def _read_omero_channels(self, ms_path: str) -> list[str] | None:
        """Marker/channel names from NGFF ``omero.channels`` labels, or None (RGB)."""
        import zarr

        try:
            attrs = dict(zarr.open_group(self._store_for(ms_path), mode="r").attrs)
        except Exception:  # noqa: BLE001 - no omero block -> plain RGB source
            return None
        channels = (attrs.get("omero") or {}).get("channels") or []
        names = [c.get("label") or c.get("name") for c in channels]
        names = [n for n in names if n]
        return names or None

    # -- internals -----------------------------------------------------------
    def _read_hwc(
        self, arr: object, level: int, x0: int, y0: int, w: int, h: int
    ) -> np.ndarray:
        """Slice a multidimensional level array and normalise to (h, w, 3) uint8.

        The (y, x) window is assembled from a decompressed-chunk cache (see
        ``_read_block_cached``) so adjacent/overlapping patches reuse already-
        decompressed 512x512 chunks instead of re-decompressing them. The
        assembled block is byte-identical to a direct ``arr[...]`` multi-chunk
        slice - only the decompression is shared.
        """
        block, sliced = self._read_block_cached(arr, level, x0, y0, w, h)

        # Reorder remaining axes to (y, x, c).
        target = ["y", "x"] + (["c"] if "c" in sliced else [])
        block = np.transpose(block, [sliced.index(t) for t in target])
        if "c" not in sliced:
            block = block[..., None]

        # Normalise to 3 channels.
        c = block.shape[2]
        if c == 1:
            block = np.repeat(block, 3, axis=2)
        elif c == 2:
            block = np.concatenate([block, block[..., :1]], axis=2)
        elif c > 3:
            block = block[..., :3]

        # Convert to 8-bit RGB by rescaling on the source dtype range -- never a
        # truncating ``astype(uint8)``, which would wrap uint16/float pixels
        # mod 256 and silently corrupt them (e.g. uint16 40000 -> 64).
        block = self._to_uint8(
            block,
            float_scale=self._float_to_uint8_scale,
        )

        # Pad to the requested size if the read was clipped at a border.
        if block.shape[0] != h or block.shape[1] != w:
            out = np.full((h, w, 3), 255, dtype=np.uint8)
            dst_y = min(h, max(0, -y0))
            dst_x = min(w, max(0, -x0))
            ph = min(block.shape[0], h - dst_y)
            pw = min(block.shape[1], w - dst_x)
            if ph > 0 and pw > 0:
                out[dst_y : dst_y + ph, dst_x : dst_x + pw, :] = block[:ph, :pw, :]
            block = out
        return np.ascontiguousarray(block)

    def _read_block_cached(
        self, arr: object, level: int, x0: int, y0: int, w: int, h: int
    ) -> tuple[np.ndarray, list[str]]:
        """Assemble the (y, x[, c]) window from cached decompressed chunk planes.

        Returns ``(block, sliced)`` where ``block`` is byte-identical to a direct
        source-order slice with every extra axis indexed at 0 (and the complete c,
        y, and x ranges retained). ``sliced`` lists those surviving dimensions in
        ``self._dims`` order.

        The window is built by copying from per-(channel, chunk_y, chunk_x) planes
        pulled from ``self._chunk_cache``; a miss decompresses exactly one chunk via
        a chunk-aligned zarr slice. Because the array tiles exactly into chunks and
        zarr decompression is deterministic, the assembled pixels equal the
        multi-chunk slice they replace - only redundant re-decompression is avoided.
        Reads are clipped to the array bounds (the caller pads to the requested
        size), matching the prior direct-slice semantics for border patches.
        """
        level_h, level_w = self._level_dims[level].height, self._level_dims[level].width
        ch, cw = self._chunk_hw[level]
        # Clip the request to the array extent (a direct slice clips the same way).
        y_lo, y_hi = max(0, y0), min(level_h, y0 + h)
        x_lo, x_hi = max(0, x0), min(level_w, x0 + w)
        clip_h = max(0, y_hi - y_lo)
        clip_w = max(0, x_hi - x_lo)

        # The surviving dimensions in self._dims order; every extra axis collapses
        # to a single plane just as the direct slice's scalar 0-index did.
        ci = None
        sliced: list[str] = []
        for pos, d in enumerate(self._dims):
            if d in ("y", "x", "c"):
                sliced.append(d)
                if d == "c":
                    ci = pos

        # Channels to fetch (the slice(None) the direct path used over the c axis).
        c_vals = range(int(arr.shape[ci])) if ci is not None else (None,)  # type: ignore[attr-defined,arg-type]

        # Allocate the block in self._dims (minus scalar) axis order.
        nonscalar = sliced  # already in dims order
        _axis_len = {"c": len(c_vals), "y": clip_h, "x": clip_w}
        shape = [_axis_len[d] for d in nonscalar]
        block = np.empty(shape, dtype=arr.dtype)  # type: ignore[attr-defined]

        # Chunk range covering the clipped window along y and x.
        cy0, cy1 = (y_lo // ch, (y_hi - 1) // ch) if clip_h else (0, -1)
        cx0, cx1 = (x_lo // cw, (x_hi - 1) // cw) if clip_w else (0, -1)

        for c_pos, c_val in enumerate(c_vals):
            for cy in range(cy0, cy1 + 1):
                cy_lo, cy_hi = cy * ch, min((cy + 1) * ch, level_h)
                for cx in range(cx0, cx1 + 1):
                    cx_lo, cx_hi = cx * cw, min((cx + 1) * cw, level_w)
                    key = (level, c_val, cy, cx)
                    plane = self._chunk_cache.get_or_read(
                        key,
                        partial(
                            self._read_chunk_plane,
                            arr,
                            c_val,
                            cy_lo,
                            cy_hi,
                            cx_lo,
                            cx_hi,
                        ),
                    )
                    # Intersection of this chunk with the requested window.
                    iy0, iy1 = max(y_lo, cy_lo), min(y_hi, cy_hi)
                    ix0, ix1 = max(x_lo, cx_lo), min(x_hi, cx_hi)
                    dst_y = slice(iy0 - y_lo, iy1 - y_lo)
                    dst_x = slice(ix0 - x_lo, ix1 - x_lo)
                    src_y = slice(iy0 - cy_lo, iy1 - cy_lo)
                    src_x = slice(ix0 - cx_lo, ix1 - cx_lo)
                    sub = plane[src_y, src_x]
                    self._place(block, nonscalar, c_pos, dst_y, dst_x, sub)
        return block, sliced

    def _read_chunk_plane(
        self, arr: object, c_val: int | None, ys: int, ye: int, xs: int, xe: int
    ) -> np.ndarray:
        """Decompress one chunk-aligned (y, x) plane for a single channel.

        Builds a full index over ``self._dims`` (extra axes -> 0, the c axis -> the
        single channel ``c_val``, y/x -> the chunk's bounds), then explicitly
        transposes the two surviving spatial dimensions from source order to
        ``(y, x)``. This keeps the cache contract independent of source axis order.
        """
        idx: list[object] = []
        for d in self._dims:
            if d == "y":
                idx.append(slice(ys, ye))
            elif d == "x":
                idx.append(slice(xs, xe))
            elif d == "c":
                idx.append(c_val)
            else:
                idx.append(0)
        plane = np.asarray(arr[tuple(idx)])  # type: ignore[index]
        spatial_order = [d for d in self._dims if d in {"y", "x"}]
        plane = np.transpose(
            plane,
            (spatial_order.index("y"), spatial_order.index("x")),
        )
        return np.ascontiguousarray(plane)

    @staticmethod
    def _place(
        block: np.ndarray,
        nonscalar: list[str],
        c_pos: int,
        dst_y: slice,
        dst_x: slice,
        sub: np.ndarray,
    ) -> None:
        """Write a 2D (y, x) sub-window into ``block`` at the right axis positions.

        ``block`` axes follow ``nonscalar`` (dims order). The c axis (if present) is
        indexed by the scalar ``c_pos``; y/x take the destination slices. The cached
        ``sub`` plane is always ``(y, x)``, so it is transposed back to the block's
        source spatial order for assignment when necessary.
        """
        sel: list[object] = []
        for d in nonscalar:
            if d == "c":
                sel.append(c_pos)
            elif d == "y":
                sel.append(dst_y)
            else:  # "x"
                sel.append(dst_x)
        spatial_order = [d for d in nonscalar if d in {"y", "x"}]
        source_order = ("y", "x")
        sub_in_block_order = np.transpose(
            sub,
            tuple(source_order.index(d) for d in spatial_order),
        )
        block[tuple(sel)] = sub_in_block_order

    def _infer_float_to_uint8_scale(self, level0: object) -> float | None:
        """Choose one float convention for the opened slide from level zero.

        OME-NGFF does not require float arrays to declare whether pixels use
        normalized ``[0, 1]`` or byte-like ``[0, 255]`` values. Scanning a whole
        level-zero WSI at open would be prohibitively expensive, so inspect a
        deterministic 3x3 lattice of complete source chunks (up to the first three
        channels), plus the array fill value. Any finite value above 1 selects the
        byte-like convention; otherwise the slide is treated as normalized.

        The important invariant is that this bounded, slide-level decision is made
        once. It is never revised from a later patch or lower pyramid level, so two
        regions from one float slide cannot receive different scaling merely because
        their local value ranges differ.
        """
        dtype = np.dtype(level0.dtype)  # type: ignore[attr-defined]
        if not np.issubdtype(dtype, np.floating):
            return None

        fill_value = getattr(level0, "fill_value", None)
        try:
            fill = float(fill_value)
        except (TypeError, ValueError):
            fill = float("nan")
        if np.isfinite(fill) and fill > 1.0:
            return 1.0

        height = self._level_dims[0].height
        width = self._level_dims[0].width
        if height <= 0 or width <= 0:
            return 255.0
        chunk_h, chunk_w = self._chunk_hw[0]
        chunk_rows = (height + chunk_h - 1) // chunk_h
        chunk_cols = (width + chunk_w - 1) // chunk_w
        sample_rows = sorted({0, (chunk_rows - 1) // 2, chunk_rows - 1})
        sample_cols = sorted({0, (chunk_cols - 1) // 2, chunk_cols - 1})

        if "c" in self._dims:
            ci = self._dims.index("c")
            channels: tuple[int | None, ...] = tuple(
                range(min(3, int(level0.shape[ci])))  # type: ignore[attr-defined]
            )
        else:
            channels = (None,)

        for chunk_y in sample_rows:
            y0 = chunk_y * chunk_h
            y1 = min(y0 + chunk_h, height)
            for chunk_x in sample_cols:
                x0 = chunk_x * chunk_w
                x1 = min(x0 + chunk_w, width)
                for channel in channels:
                    sample = self._read_chunk_plane(
                        level0,
                        channel,
                        y0,
                        y1,
                        x0,
                        x1,
                    )
                    finite = sample[np.isfinite(sample)]
                    if finite.size and float(finite.max()) > 1.0:
                        return 1.0
        return 255.0

    @staticmethod
    def _to_uint8(
        block: np.ndarray,
        *,
        float_scale: float | None = None,
    ) -> np.ndarray:
        """Rescale an image to 8-bit by its dtype range (never wrap mod 256).

        ``float_scale`` is the slide-level convention selected at open. Leaving it
        unset preserves the standalone helper's historical local-range behaviour.
        """
        if block.dtype == np.uint8:
            return block
        if np.issubdtype(block.dtype, np.unsignedinteger):
            scale = 255.0 / float(np.iinfo(block.dtype).max)
            scaled = np.rint(block.astype(np.float32) * scale)
            return np.clip(scaled, 0, 255).astype(np.uint8)
        if np.issubdtype(block.dtype, np.signedinteger):
            # Treat as [0, max] -- negative values are not valid pixel intensities.
            scale = 255.0 / float(np.iinfo(block.dtype).max)
            scaled = np.rint(np.clip(block.astype(np.float32), 0, None) * scale)
            return np.clip(scaled, 0, 255).astype(np.uint8)
        if np.issubdtype(block.dtype, np.floating):
            # Float images are usually [0, 1] (normalized) or already [0, 255]. The
            # reader supplies one level-zero decision; the fallback is retained for
            # backwards-compatible direct calls to this internal helper.
            b = np.nan_to_num(block.astype(np.float32))
            scale = float_scale
            if scale is None:
                scale = 255.0 if b.size and float(b.max()) <= 1.0 else 1.0
            if scale != 1.0:
                # Clip before multiplying so +inf (converted to float32 max by
                # nan_to_num) saturates without an overflow warning.
                b = np.clip(b, 0, 255.0 / scale) * scale
            return np.clip(np.rint(b), 0, 255).astype(np.uint8)
        raise NotImplementedError(
            f"unsupported OME-Zarr pixel dtype {block.dtype!r}; expected uint8, "
            "unsigned/signed integer, or float"
        )

    def _resolve_multiscales(self) -> tuple[str, dict]:
        """Find the multiscales group; return (path-prefix, multiscales[0] dict)."""
        import zarr

        def _extract(attrs: dict) -> dict | None:
            if "multiscales" in attrs:
                return attrs["multiscales"][0]
            ome = attrs.get("ome")
            if isinstance(ome, dict) and "multiscales" in ome:
                # NGFF 0.5 moved the spec version into the `ome` wrapper (it is no
                # longer on the multiscales dict), so capture it here for provenance.
                if ome.get("version"):
                    self._ome_version = str(ome["version"])
                return ome["multiscales"][0]
            return None

        try:
            root = zarr.open_group(self._store_for(self.path), mode="r")
        except Exception as exc:  # noqa: BLE001 - re-raised as one actionable error
            raise FileNotFoundError(
                f"could not open {source_uri(self.path)!r} as an OME-Zarr store "
                f"({redact_uri_credentials(str(exc))}). Check the "
                "path/URL exists and points at the .zarr group; for a bioformats2raw "
                "layout, include the image-series subpath (e.g. .../image.zarr/0)."
            ) from exc
        ms = _extract(dict(root.attrs))
        if ms is not None:
            return "", ms

        # bioformats2raw layout: image series under an integer subgroup.
        for key in ("0", "1"):
            try:
                sub = zarr.open_group(
                    self._store_for(join_uri_path(self.path, key)), mode="r"
                )
            except Exception:  # noqa: BLE001
                continue
            ms = _extract(dict(sub.attrs))
            if ms is not None:
                return key, ms
        raise ValueError(
            f"{source_uri(self.path)}: no OME-NGFF multiscales metadata found"
        )

    def _build_level_mapping(self, datasets: list, xi: int, yi: int) -> None:
        """Per-axis downsample + translation offset per level, for ``read_region``.

        For each level L the source's ``coordinateTransformations`` give a scale and
        (optionally) a translation per axis. Mapping a level-0 index to level L is
        ``idx_L = (scale_0 * idx_0 + trans_0 - trans_L) / scale_L``, i.e.
        ``idx_0 / (scale_L/scale_0) + (trans_0 - trans_L)/scale_L``. So the per-axis
        downsample is ``scale_L/scale_0`` and the offset ``(trans_0-trans_L)/scale_L``
        (in level-L px). scale/translation share an axis unit across the multiscales,
        so the ratios are unitless and need no µm conversion.

        x reuses ``self._downsamples`` (already the x ratios) so the common case is
        byte-identical to the previous mapping. If per-axis scale can't be parsed for
        every level we fall back to the isotropic, translation-free behaviour.
        """
        n = len(self._downsamples)
        sx, sy, tx, ty = [], [], [], []
        for d in datasets:
            s = t_ = None
            txi = tyi = 0.0
            for tr in d.get("coordinateTransformations") or []:
                if tr.get("type") == "scale":
                    vec = tr.get("scale") or []
                    s = float(vec[xi]) if xi < len(vec) else None
                    t_ = float(vec[yi]) if yi < len(vec) else None
                elif tr.get("type") == "translation":
                    vec = tr.get("translation") or []
                    txi = float(vec[xi]) if xi < len(vec) else 0.0
                    tyi = float(vec[yi]) if yi < len(vec) else 0.0
            sx.append(s)
            sy.append(t_)
            tx.append(txi)
            ty.append(tyi)

        have = (
            len(sx) == n
            and all(v and v > 0 for v in sx)
            and all(v and v > 0 for v in sy)
        )
        self._downsamples_x = list(self._downsamples)
        if have:
            self._downsamples_y = [v / sy[0] for v in sy]
            self._level_off_x = [(tx[0] - t) / s for t, s in zip(tx, sx, strict=True)]
            self._level_off_y = [(ty[0] - t) / s for t, s in zip(ty, sy, strict=True)]
        else:
            self._downsamples_y = list(self._downsamples)
            self._level_off_x = [0.0] * n
            self._level_off_y = [0.0] * n

    def _read_level0_translation_um(
        self, multiscales: dict, units: dict
    ) -> dict[str, float] | None:
        """Source level-0 translation (x/y) in µm, or ``None`` if there is none.

        Reads the ``translation`` component of the NGFF ``coordinateTransformations``
        for the level-0 dataset (and any multiscales-level translation, applied after
        it), summed per axis and converted to µm. ngff-zarr's ``NgffImage`` surfaces
        only the scale, so this reads the raw multiscales dict (which we already hold)
        rather than going back through the metadata model. Most stores carry no
        translation - returning ``None`` keeps the header clean for that common case.
        """
        acc: dict[str, float] = {}

        def _accumulate(transforms) -> None:
            for t in transforms or []:
                if isinstance(t, dict) and t.get("type") == "translation":
                    vec = t.get("translation") or []
                    for ax, v in zip(self._dims, vec, strict=False):
                        acc[ax] = acc.get(ax, 0.0) + float(v)

        datasets = multiscales.get("datasets") or []
        if datasets:
            _accumulate(datasets[0].get("coordinateTransformations"))
        _accumulate(multiscales.get("coordinateTransformations"))

        out = {
            ax: self._to_um(acc[ax], units.get(ax))
            for ax in ("x", "y")
            if ax in acc and acc[ax] != 0.0
        }
        return out or None

    @staticmethod
    def _to_um(value: float, unit: str | None) -> float:
        factor = _UNIT_TO_UM.get(unit)
        if factor is None:
            warnings.warn(
                f"unknown spatial unit {unit!r}; assuming micrometer", stacklevel=2
            )
            factor = 1.0
        return float(value) * factor
