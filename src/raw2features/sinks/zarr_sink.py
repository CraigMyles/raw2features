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
import secrets
import stat
import unicodedata

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


def _select_grid_label(root, base: str, grid_hash: str | None) -> str:
    """Choose a deterministic unused label without treating it as grid identity."""

    if GRIDS not in root or base not in root[GRIDS]:
        return base
    if not grid_hash:
        raise ValueError(
            f"grid label {base!r} is occupied and the new grid has no grid_hash"
        )

    def stored_hash(key: str):
        return dict(root[GRIDS][key].attrs.get("raw2features", {})).get("grid_hash")

    if stored_hash(base) == grid_hash:
        raise ValueError(
            f"grid {base!r} already records this grid_hash; reopen it for append"
        )
    lengths = list(range(8, len(grid_hash), 4)) + [len(grid_hash)]
    for length in dict.fromkeys(lengths):
        candidate = f"{base}_{grid_hash[:length]}"
        if candidate not in root[GRIDS]:
            return candidate
        if stored_hash(candidate) == grid_hash:
            raise ValueError(
                f"grid {candidate!r} already records this grid_hash; "
                "reopen it for append"
            )
    raise ValueError(
        f"cannot derive a collision-free label for grid_hash {grid_hash!r}"
    )


def _grid_scaffold_is_usable(
    group,
    *,
    expected_n: int | None = None,
    require_mask: bool = False,
    expected_mask_shape: tuple[int, ...] | None = None,
) -> bool:
    """Whether a grid has the structural scaffold needed for safe model append.

    Missing model arrays are deliberately allowed: adding those arrays is the normal
    resume path.  A malformed coordinate scaffold or an existing feature column with a
    different row count cannot be repaired model-by-model because it breaks the grid's
    1:1 row identity, so the matching grid must instead be rebuilt as a unit.
    """

    try:
        if "coords" not in group:
            return False
        coords = group["coords"]
        if (
            getattr(coords, "ndim", None) != 2
            or tuple(coords.shape[1:]) != (2,)
            or not np.issubdtype(coords.dtype, np.integer)
        ):
            return False
        n = int(coords.shape[0])
        if expected_n is not None and n != int(expected_n):
            return False

        # grid_index is optional in the public format, but when present it must retain
        # the same row identity as coords.
        if "grid_index" in group:
            grid_index = group["grid_index"]
            if (
                getattr(grid_index, "ndim", None) != 2
                or tuple(grid_index.shape) != (n, 2)
                or not np.issubdtype(grid_index.dtype, np.integer)
            ):
                return False

        if "features" not in group or not hasattr(group["features"], "keys"):
            return False
        features = group["features"]
        for model in features.keys():
            array = features[model]
            if getattr(array, "ndim", None) != 2 or int(array.shape[0]) != n:
                return False

        # A mask is required only for a segmented extraction.  Unsegmented and legacy
        # stores may validly omit it.
        if require_mask:
            if "mask" not in group:
                return False
            mask = group["mask"]
            if getattr(mask, "ndim", None) != 2:
                return False
            if expected_mask_shape is not None and tuple(mask.shape) != tuple(
                expected_mask_shape
            ):
                return False
    except Exception:  # noqa: BLE001 - an unreadable scaffold is not appendable
        return False
    return True


def _unappendable_grid_to_rebuild(
    root,
    base: str,
    grid_hash: str | None,
    *,
    expected_n: int,
    required_mask_shape: tuple[int, ...] | None,
) -> str | None:
    """Return the matching target when its grid scaffold cannot be appended safely.

    A same-hash grid is authoritative even when a label collision gave it a suffix.
    The sole hashless legacy grid remains the one explicitly-supported compatibility
    case. Other hashless/mismatched grids are unrelated and must be preserved.
    """

    if GRIDS not in root:
        return None
    keys = list(root[GRIDS].keys())
    for key in keys:
        g = root[GRIDS][key]
        stored_hash = dict(g.attrs.get("raw2features", {})).get("grid_hash")
        is_target = stored_hash == grid_hash and grid_hash is not None
        is_sole_legacy_target = len(keys) == 1 and key == base and stored_hash is None
        if (is_target or is_sole_legacy_target) and not _grid_scaffold_is_usable(
            g,
            expected_n=expected_n,
            require_mask=required_mask_shape is not None,
            expected_mask_shape=required_mask_shape,
        ):
            return key
    return None


@register("sinks", "zarr")
class ZarrSink(Sink):
    """One ``grids/<key>/`` per geometry: coords + grid_index + mask + features."""

    name = "zarr"

    def __init__(self, output_zarr_format: int = 2) -> None:
        self._fmt = output_zarr_format
        self._root = None
        self._group = None  # the active grid group
        self._uri = ""
        self._active_key: str | None = None

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
    ) -> str:
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
            actual_grid = grid
            root.attrs["raw2features"] = _root_header(
                actual_grid, header, model_dims, n_patches
            )
        else:
            # use_consolidated=False: read live metadata (a prior run consolidated it).
            root = zarr.open_group(path, mode="r+", use_consolidated=False)
            # Validate shared source metadata before deleting or rebuilding a target
            # grid.
            rh = dict(root.attrs.get("raw2features", {}))
            rh.setdefault("schema_version", header.get("schema_version"))
            if "source" not in rh and "source" in header:
                rh["source"] = header["source"]
            elif "source" in header:
                source_update = {
                    name: header["source"][name]
                    for name in (
                        "channel_names",
                        "channel_names_source",
                        "omero_channel_names",
                    )
                    if name in header["source"]
                }
                if source_update:
                    rh["source"] = self._merge_source_metadata(
                        rh.get("source", {}), source_update
                    )
            for k in ("provenance", "segmentation", "thumbnail"):
                if k not in rh and k in header:
                    rh[k] = header[k]
            rebuild = _unappendable_grid_to_rebuild(
                root,
                grid,
                header.get("grid_hash"),
                expected_n=n_patches,
                required_mask_shape=(
                    tuple(np.asarray(grid_tissue).shape)
                    if grid_tissue is not None
                    else None
                ),
            )
            if rebuild is not None:
                # Delete only the unusable target. The complete replacement inputs
                # are already in hand, and all unrelated grids remain untouched.
                del root[GRIDS][rebuild]
                actual_grid = rebuild
            else:
                actual_grid = _select_grid_label(root, grid, header.get("grid_hash"))
            grids = dict(rh.get("grids", {}))
            grids[actual_grid] = _grids_index_entry(header, model_dims, n_patches)
            rh["grids"] = grids
            root.attrs["raw2features"] = rh
        g = root.require_group(GRIDS).require_group(actual_grid)
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
        self._active_key = actual_grid
        return actual_grid

    def open_append(
        self,
        out_dir: str,
        slide_id: str,
        *,
        key: str | None = None,
        new_model_dims: dict[str, int],
        new_model_meta: dict | None = None,
        new_panel_meta: dict | None = None,
        new_source_meta: dict | None = None,
        replace_models: list[str] | tuple[str, ...] | set[str] = (),
    ) -> int:
        """Open an existing store's grid ``r+`` to add feature arrays, no clobber.

        Creates ``features/<model>`` only for models in *new_model_dims* not already
        present in the grid, matching the grid's existing feature dtype so it stays
        homogeneous. Names explicitly listed in *replace_models* are deleted and
        recreated; callers use that only after the old output fails its current model
        contract. Coords/grid/mask and unrelated feature arrays are left untouched.
        ``key=None`` targets the sole grid. Returns the grid's patch count. Pass empty
        ``new_model_dims`` to open for slide-only work. ``new_panel_meta`` is merged by
        model key alongside ``new_model_meta``. ``new_source_meta`` may add or fill the
        effective positional channel metadata in both grid and root headers, but cannot
        relabel an already named physical channel.
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
        gh = dict(g.attrs.get("raw2features", {}))
        rh = None
        if new_source_meta:
            gh["source"] = self._merge_source_metadata(
                gh.get("source", {}), new_source_meta
            )
            rh = dict(root.attrs.get("raw2features", {}))
            rh["source"] = self._merge_source_metadata(
                rh.get("source", {}), new_source_meta
            )
        feats = g["features"]
        existing = list(feats.keys())
        self._dtype = feats[existing[0]].dtype if existing else "float16"
        n = int(g["coords"].shape[0])
        added: list[str] = []
        replace = set(replace_models)
        invalidating = replace & set(new_model_dims)
        if invalidating:
            # A corrupt store may have lost features/<model> while retaining a slide
            # vector derived from it, so invalidate dependents even when the patch
            # array itself is already absent.
            self._drop_slide_dependents(g, invalidating)
        for model, dim in new_model_dims.items():
            if model in feats:
                if model not in replace:  # default remains no-clobber
                    continue
                del feats[model]
            a = feats.create_array(
                model,
                shape=(n, dim),
                chunks=(_patch_chunk_size(n), dim),
                dtype=self._dtype,
            )
            a.attrs["role"] = "features"
            a.attrs["model"] = model
            added.append(model)
        if new_model_meta or new_panel_meta or new_source_meta:
            if new_model_meta:
                models = dict(gh.get("models", {}))
                models.update(new_model_meta)
                gh["models"] = models
            if new_panel_meta:
                panel = dict(gh.get("panel", {}))
                panel.update(new_panel_meta)
                gh["panel"] = panel
            g.attrs["raw2features"] = gh
            if rh is not None:
                root.attrs["raw2features"] = rh
        if added:
            self._update_root_models(root, actual_key, list(feats.keys()))
        self._root = root
        self._group = g
        self._active_key = actual_key
        return n

    @staticmethod
    def _merge_source_metadata(existing: dict, update: dict) -> dict:
        """Fill positional channel metadata without changing an established label."""

        merged = dict(existing or {})
        new_names = update.get("channel_names")
        if new_names is None:
            return merged
        new_names = ["" if value is None else str(value) for value in new_names]
        old_names = merged.get("channel_names")
        if old_names is None:
            merged.update(update)
            merged["channel_names"] = new_names
            return merged
        old_names = ["" if value is None else str(value) for value in old_names]
        if len(old_names) > len(new_names):
            raise ValueError(
                "stored source channel metadata has more positional entries than "
                f"the effective physical panel ({len(old_names)} > {len(new_names)})"
            )
        old_names.extend([""] * (len(new_names) - len(old_names)))

        def identity(value: str) -> str:
            return unicodedata.normalize("NFKC", value).strip().casefold()

        completed = []
        filled = False
        for index, (old_name, new_name) in enumerate(
            zip(old_names, new_names, strict=True)
        ):
            old_identity = identity(old_name)
            new_identity = identity(new_name)
            if old_identity and old_identity != new_identity:
                raise ValueError(
                    "stored source channel metadata conflicts at physical C index "
                    f"{index}: {old_name!r} != {new_name!r}"
                )
            if old_identity:
                completed.append(old_name)
            else:
                completed.append(new_name)
                filled = filled or bool(new_identity)
        merged["channel_names"] = completed
        if filled:
            merged["channel_names_source"] = update.get("channel_names_source")
        else:
            merged.setdefault(
                "channel_names_source", update.get("channel_names_source")
            )
        if (
            "omero_channel_names" in update
            and "omero_channel_names" not in merged
            and update["omero_channel_names"] != completed
        ):
            merged["omero_channel_names"] = update["omero_channel_names"]
        return merged

    @staticmethod
    def _drop_slide_dependents(group, patch_models: set[str]) -> None:
        """Remove slide vectors derived from feature arrays being replaced."""

        if "slide" in group:
            slide = group["slide"]
            for name in list(slide.keys()):
                if dict(slide[name].attrs).get("patch_encoder") in patch_models:
                    del slide[name]
        header = dict(group.attrs.get("raw2features", {}))
        slide_meta = dict(header.get("slide_embeddings", {}))
        kept = {
            name: value
            for name, value in slide_meta.items()
            if not isinstance(value, dict)
            or value.get("patch_encoder") not in patch_models
        }
        if kept != slide_meta:
            if kept:
                header["slide_embeddings"] = kept
            else:
                header.pop("slide_embeddings", None)
            group.attrs["raw2features"] = header

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

    def finalize_models(self, contracts: dict[str, dict]) -> None:
        """Commit successfully-written feature arrays with their output fingerprints.

        The array attribute is deliberately absent while rows are being written.  A
        crash therefore leaves the model incomplete even if the old finite values or
        zarr fill values would otherwise pass structural checks.
        """

        if self._group is None:
            raise RuntimeError("sink not created; call create() first")
        from raw2features.embedders.fingerprint import output_fingerprints_equal

        header = dict(self._group.attrs.get("raw2features", {}))
        model_meta = header.get("models", {})
        for model, contract in contracts.items():
            if model not in self._group["features"]:
                raise KeyError(f"cannot finalize missing features/{model}")
            expected_dim = int(contract["embedding_dim"])
            array = self._group["features"][model]
            if array.ndim != 2 or int(array.shape[1]) != expected_dim:
                raise ValueError(
                    f"features/{model} has shape {array.shape}; "
                    f"expected (*, {expected_dim})"
                )
            fingerprint = contract["output_fingerprint"]
            stored_header = (
                model_meta.get(model, {}).get("output_fingerprint")
                if isinstance(model_meta, dict)
                and isinstance(model_meta.get(model), dict)
                else None
            )
            if not output_fingerprints_equal(stored_header, fingerprint):
                raise ValueError(
                    f"header fingerprint for {model!r} does not match the "
                    "output contract"
                )
            # This is the completion commit marker and must be the final model write.
            array.attrs["output_fingerprint"] = fingerprint

    def write_slide_embedding(
        self,
        slide_model: str,
        vector: np.ndarray,
        provenance: dict,
        *,
        output_name: str | None = None,
    ) -> None:
        """Write a single slide-level vector to the grid's ``slide/<model>/``."""
        if self._group is None:
            raise RuntimeError("sink not created; call create() first")
        from raw2features.slide_embedders.encoding import write_slide_embedding

        write_slide_embedding(
            self._group,
            slide_model,
            vector,
            provenance,
            output_name=output_name,
        )

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
        qc_root = self._group.require_group("qc")
        # The runner calls this writer only for an absent or structurally incomplete
        # producer. Replace an interrupted partial group as one unit so a crash that
        # left (for example) a wrong-length ``scores`` array is repairable on resume.
        if tool in qc_root:
            del qc_root[tool]
        qc = qc_root.create_group(tool)

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
        # Final commit marker: a crash before arrays, attrs and any header mirror are
        # complete leaves this absent, so the next produce-if-missing run repairs it.
        qc.attrs["complete"] = True

    def update_thumbnail(self, metadata: dict, *, update_root: bool = True) -> None:
        """Record a grid thumbnail, mirroring the primary grid at the store root."""

        if self._root is None or self._group is None:
            raise RuntimeError("sink not created; call create() or open_append() first")
        grid_header = dict(self._group.attrs.get("raw2features", {}))
        grid_header["thumbnail"] = metadata
        self._group.attrs["raw2features"] = grid_header
        if update_root:
            root_header = dict(self._root.attrs.get("raw2features", {}))
            root_header["thumbnail"] = metadata
            self._root.attrs["raw2features"] = root_header

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
    out_dir: str,
    slide_id: str,
    coords: np.ndarray,
    level0_patch: int,
    *,
    filename: str | None = None,
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
    if filename is not None and os.path.basename(filename) != filename:
        raise ValueError("filename must be a basename, not a path")
    path = os.path.join(out_dir, filename or f"{slide_id}.patches.geojson")
    temporary: str | None = None
    fd: int | None = None
    try:
        try:
            existing_mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            existing_mode = None
        for _ in range(100):
            temporary = os.path.join(
                out_dir,
                f".r2f-sidecar.{secrets.token_hex(8)}.tmp",
            )
            try:
                # Honour the process umask/default ACL, matching a normal output
                # file rather than mkstemp's fixed 0600 permissions.
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                break
            except FileExistsError:
                temporary = None
        else:  # pragma: no cover - 100 cryptographic-name collisions is infeasible
            raise FileExistsError("could not allocate a temporary GeoJSON path")
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        with os.fdopen(fd, mode="w", encoding="utf-8") as fh:
            fd = None  # ownership transferred to ``fh``
            json.dump(fc, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:  # noqa: BLE001 - cleanup on interrupts as well as failures
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise
    return path
