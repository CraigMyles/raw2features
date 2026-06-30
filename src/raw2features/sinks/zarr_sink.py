"""Write a per-slide ``<slide_id>.embeddings.zarr`` (the raw2features standard).

Patch sets live uniformly under ``grids/<key>/`` -- one subgroup per geometry
``(mpp, patch_px)`` (see :mod:`raw2features.core.store`). Each grid group is
self-describing (a complete format-0.1 header in its attrs) and holds ``coords`` /
``grid_index`` / ``mask`` / ``features/<model>`` / ``slide/<model>``. The root group
carries a ``grids`` index. ``self._group`` is the active grid group, so the write and
slide-encoder paths address it exactly as they addressed the old flat root.
"""

from __future__ import annotations

import json
import os

import numpy as np

from raw2features.core.plugins import register
from raw2features.core.store import GRIDS

from .base import Sink

_CHUNK = 4096  # patches per chunk along the patch axis


def _patch_chunk_size(n: int) -> int:
    """Chunk length along the patch axis (array axis 0), clamped to [1, _CHUNK]."""
    return max(1, min(_CHUNK, n))


def _grids_index_entry(header: dict, model_dims: dict, n: int) -> dict:
    """The root ``grids`` index summary for one grid (geometry + models + grid_hash)."""
    p = header.get("patching", {}) or {}
    return {
        "target_mpp": p.get("target_mpp"),
        "achieved_mpp": p.get("achieved_mpp"),
        "patch_px": p.get("patch_px"),
        "level0_patch": p.get("level0_patch"),
        "n_patches": int(n),
        "grid_hash": header.get("grid_hash"),
        "models": list(model_dims),
    }


def _root_header(grid: str, header: dict, model_dims: dict, n: int) -> dict:
    """The root discovery header: shared slide-level keys + the ``grids`` index.

    The authoritative, schema-conformant header lives per grid; the root carries only
    the slide-level fields (source / provenance / segmentation / thumbnail) + the index.
    """
    root = {"schema_version": header.get("schema_version")}
    for k in ("source", "provenance", "segmentation", "thumbnail"):
        if k in header:
            root[k] = header[k]
    root["grids"] = {grid: _grids_index_entry(header, model_dims, n)}
    return root


@register("sinks", "zarr")
class ZarrSink(Sink):
    """One ``grids/<key>/`` per geometry: coords + grid_index + mask + features."""

    name = "zarr"

    def __init__(self, output_zarr_format: int = 2) -> None:
        self._fmt = output_zarr_format
        self._root = None
        self._group = None  # the active grid group
        self._uri = ""

    def create(
        self,
        out_dir: str,
        slide_id: str,
        *,
        grid: str,
        fresh: bool = True,
        n_patches: int,
        coords: np.ndarray,
        grid_index: np.ndarray,
        grid_tissue: np.ndarray | None,
        model_dims: dict[str, int],
        header: dict,
        features_dtype: str = "float16",
    ) -> None:
        """Write the ``grids/<grid>/`` grid.

        ``fresh=True`` writes a brand-new store (wiping any existing). ``fresh=False``
        ADDS this grid to an existing store (a different geometry written later) without
        touching the other grids -- it merges the grid into the root ``grids`` index.
        """
        import zarr

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
        self._uri = f"file://{os.path.abspath(path)}"
        if fresh:
            root = zarr.open_group(path, mode="w", zarr_format=self._fmt)
            root.attrs["raw2features"] = _root_header(
                grid, header, model_dims, n_patches
            )
        else:
            # use_consolidated=False: read live metadata (a prior run consolidated it).
            root = zarr.open_group(path, mode="r+", use_consolidated=False)
            rh = dict(root.attrs.get("raw2features", {}))
            rh.setdefault("schema_version", header.get("schema_version"))
            for k in ("source", "provenance", "segmentation", "thumbnail"):
                if k not in rh and k in header:
                    rh[k] = header[k]
            grids = dict(rh.get("grids", {}))
            grids[grid] = _grids_index_entry(header, model_dims, n_patches)
            rh["grids"] = grids
            root.attrs["raw2features"] = rh
        g = root.require_group(GRIDS).require_group(grid)
        g.attrs["raw2features"] = header

        coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        grid_index = np.asarray(grid_index, dtype=np.int32).reshape(-1, 2)
        c = g.create_array(
            "coords",
            shape=coords.shape,
            chunks=(_patch_chunk_size(n_patches), 2),
            dtype="int32",
        )
        if n_patches:
            c[:] = coords
        c.attrs["role"] = "coords"
        c.attrs["units"] = "level0_px"

        gi = g.create_array(
            "grid_index",
            shape=grid_index.shape,
            chunks=(_patch_chunk_size(n_patches), 2),
            dtype="int32",
        )
        if n_patches:
            gi[:] = grid_index
        gi.attrs["role"] = "grid_index"

        if grid_tissue is not None:
            mask = (np.clip(grid_tissue, 0.0, 1.0) * 255).astype(np.uint8)
            m = g.create_array(
                "mask", shape=mask.shape, chunks=mask.shape or (1,), dtype="uint8"
            )
            m[:] = mask
            m.attrs["role"] = "tissue_mask"

        feats = g.create_group("features")
        self._dtype = features_dtype
        for model, dim in model_dims.items():
            a = feats.create_array(
                model,
                shape=(n_patches, dim),
                chunks=(_patch_chunk_size(n_patches), dim),
                dtype=features_dtype,
            )
            a.attrs["role"] = "features"
            a.attrs["model"] = model
        self._root = root
        self._group = g

    def open_append(
        self,
        out_dir: str,
        slide_id: str,
        *,
        key: str | None = None,
        new_model_dims: dict[str, int],
        new_model_meta: dict | None = None,
    ) -> int:
        """Open an existing store's grid ``r+`` to add feature arrays, no clobber.

        Creates ``features/<model>`` only for models in *new_model_dims* not already
        present in the grid, matching the grid's existing feature dtype so it stays
        homogeneous. Coords/grid/mask and existing feature arrays are left untouched.
        ``key=None`` targets the sole grid. Returns the grid's patch count. Pass empty
        ``new_model_dims`` to open for slide-only work.
        """
        import zarr

        from raw2features.core.store import grid_keys, open_grid

        path = os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
        self._uri = f"file://{os.path.abspath(path)}"
        # use_consolidated=False: the store was consolidated by a prior run, so its
        # consolidated metadata predates arrays we add here. Read live metadata and
        # re-consolidate in close().
        root = zarr.open_group(path, mode="r+", use_consolidated=False)
        g = open_grid(root, key)
        actual_key = key if key is not None else grid_keys(root)[0]
        feats = g["features"]
        existing = list(feats.keys())
        self._dtype = feats[existing[0]].dtype if existing else "float16"
        n = int(g["coords"].shape[0])
        added: list[str] = []
        for model, dim in new_model_dims.items():
            if model in feats:  # never clobber an existing model
                continue
            a = feats.create_array(
                model,
                shape=(n, dim),
                chunks=(_patch_chunk_size(n), dim),
                dtype=self._dtype,
            )
            a.attrs["role"] = "features"
            a.attrs["model"] = model
            added.append(model)
        if new_model_meta:
            gh = dict(g.attrs.get("raw2features", {}))
            gh.setdefault("models", {}).update(new_model_meta)
            g.attrs["raw2features"] = gh
        if added:
            self._update_root_models(root, actual_key, list(feats.keys()))
        self._root = root
        self._group = g
        return n

    @staticmethod
    def _update_root_models(root, key: str, models: list[str]) -> None:
        """Refresh this grid's ``models`` list in the root ``grids`` index."""
        rh = dict(root.attrs.get("raw2features", {}))
        grids = dict(rh.get("grids", {}))
        if key in grids:
            entry = dict(grids[key])
            entry["models"] = list(models)
            grids[key] = entry
            rh["grids"] = grids
            root.attrs["raw2features"] = rh

    def feature_dims(self) -> dict[str, int]:
        """Map ``model -> embedding_dim`` for every feature array now in the grid."""
        if self._group is None:
            return {}
        feats = self._group["features"]
        return {m: int(feats[m].shape[1]) for m in feats.keys()}

    def write_block(self, model: str, start: int, feats: np.ndarray) -> None:
        if self._group is None:
            raise RuntimeError("sink not created; call create() first")
        block = np.asarray(feats, dtype=self._dtype)
        self._group["features"][model][start : start + block.shape[0]] = block

    def write_slide_embedding(
        self,
        slide_model: str,
        vector: np.ndarray,
        provenance: dict,
    ) -> None:
        """Write a single slide-level vector to the grid's ``slide/<model>/``."""
        if self._group is None:
            raise RuntimeError("sink not created; call create() first")
        slide_group = self._group.require_group("slide")
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        arr = slide_group.create_array(
            slide_model,
            shape=vec.shape,
            chunks=vec.shape,
            dtype="float32",
        )
        arr[:] = vec
        arr.attrs["role"] = "slide_embedding"
        for k, v in provenance.items():
            arr.attrs[k] = v
        # Mirror provenance into the grid header for convenience.
        gh = dict(self._group.attrs.get("raw2features", {}))
        slide_meta = gh.setdefault("slide_embeddings", {})
        slide_meta[slide_model] = provenance
        self._group.attrs["raw2features"] = gh

    def write_qc(
        self,
        tool: str,
        scores: np.ndarray,
        classes: list[str],
        *,
        label: np.ndarray | None = None,
        usable: np.ndarray | None = None,
        legend: dict | None = None,
        provenance: dict | None = None,
    ) -> None:
        """Write an optional per-patch QC layer into the active grid's ``qc/<tool>/``.

        ``scores`` is ``(N, k)`` per-class fractions, 1:1 with the grid's coords;
        ``classes`` names its ``k`` columns (stored on ``scores.attrs["classes"]``).
        Optional ``label`` ((N,) hard label, with an NGFF ``image-label`` legend on the
        group) and ``usable`` ((N,) uint8 keep/drop). Every array carries ``role="qc"``.
        ``provenance`` (tool, version, mpp, threshold, …) is stored on the group
        and mirrored into the grid header's ``qc`` block. This is the hook for any
        per-patch scorer -- it does not run any QC model itself.
        """
        if self._group is None:
            raise RuntimeError("sink not created; call create() first")
        scores = np.asarray(scores, dtype=np.float16).reshape(-1, len(classes))
        n = scores.shape[0]
        qc = self._group.require_group("qc").require_group(tool)

        a = qc.create_array(
            "scores",
            shape=scores.shape,
            chunks=(_patch_chunk_size(n), scores.shape[1]),
            dtype="float16",
        )
        if n:
            a[:] = scores
        a.attrs["role"] = "qc"
        a.attrs["classes"] = list(classes)

        for name, data in (("label", label), ("usable", usable)):
            if data is None:
                continue
            arr = qc.create_array(
                name, shape=(n,), chunks=(_patch_chunk_size(n),), dtype="uint8"
            )
            if n:
                arr[:] = np.asarray(data).reshape(-1).astype("uint8")
            arr.attrs["role"] = "qc"

        if legend is not None:
            qc.attrs["image-label"] = legend
        if provenance:
            for k, v in provenance.items():
                qc.attrs[k] = v
            gh = dict(self._group.attrs.get("raw2features", {}))
            gh.setdefault("qc", {})[tool] = provenance
            self._group.attrs["raw2features"] = gh

    def close(self) -> None:
        if self._root is None:
            return
        try:
            import zarr

            zarr.consolidate_metadata(self._root.store)
        except Exception:  # noqa: BLE001 - consolidation is best-effort
            pass

    @property
    def uri(self) -> str:
        return self._uri


def write_patches_geojson(
    out_dir: str, slide_id: str, coords: np.ndarray, level0_patch: int
) -> str:
    """Write per-patch square polygons in level-0 pixel coords (QuPath-importable)."""
    features = []
    for x, y in np.asarray(coords, dtype=int).reshape(-1, 2):
        x, y, s = int(x), int(y), int(level0_patch)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]
                    ],
                },
                "properties": {"objectType": "tile"},
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    path = os.path.join(out_dir, f"{slide_id}.patches.geojson")
    with open(path, "w") as fh:
        json.dump(fc, fh)
    return path
