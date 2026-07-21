"""Export a raw2features embedding store to scverse **SpatialData** (`.zarr`).

A post-hoc converter: it reads an existing ``<slide_id>.embeddings.zarr`` (coords +
``features/<model>`` + provenance header) and writes a sibling SpatialData store -
an OME-NGFF-aligned `.zarr` holding a ``tiles`` shapes element (one square polygon per
patch) and a ``table`` AnnData whose ``obsm["X_<model>"]`` holds each model's embedding.
The result round-trips through ``spatialdata_io.experimental.to_legacy_anndata`` and is
directly readable by squidpy / napari-spatialdata.

The tiles carry two coordinate systems: ``global`` (their intrinsic level-0 **pixel**
space) and, when the source is physically calibrated, ``micrometers`` - a Scale (plus a
Translation for a non-zero source origin) mapping pixels to **µm**. The µm system lets
tiles align to other slides/modalities in real units; it is built from the store's
recorded ``source.scale_um`` / ``level0_translation_um``, so it is exact.

For **multiplex** embeddings the marker panel is stored as a queryable table: the
per-channel resolution (which source channel fed which marker, under what id) becomes a
tidy, queryable ``table.uns["raw2features_panel"]`` DataFrame - one row per kept
channel, with a ``model`` column. (It would be lost otherwise: anndata's zarr writer
stringifies a raw list-of-dicts into unusable ``"{...}"`` blobs.) A compact per-model
coverage summary sits in ``uns["raw2features_export"]["panel"]``; the faithful,
columnarised record stays in ``uns["raw2features"]``.

The schema is transcribed from two canonical, working "patches -> SpatialData"
converters that agree with each other - HEST's ``HESTData.to_spatial_data`` and
spatialdata-io's ``from_legacy_anndata`` - plus CLAM/Trident field names for the tile
geometry, so the output is legible to the dominant pathology-FM toolchain.

Dependencies (``spatialdata``, ``anndata``, ``geopandas``) are imported lazily and ship
in the optional ``[spatialdata]`` extra, so importing raw2features never requires them.
"""

from __future__ import annotations

import json
import os

import numpy as np

# Coords are authored top-left in level-0 pixels by the runner.
_COORDS_CONVENTION = "level0_xy"


def _open_store(store: str, grid: str | None = None):
    from raw2features.core.store import open_grid

    # grid=None opens the sole grid; a multi-grid store errors asking for --grid <key>.
    g = open_grid(store, grid)
    if "features" not in g:
        raise ValueError(
            f"{store!r} has no 'features' group - is it a raw2features embedding store?"
        )
    return g


def _slide_id(store: str) -> str:
    base = os.path.basename(os.path.normpath(store))
    for suffix in (".embeddings.zarr", ".zarr"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _tile_ids(coords: np.ndarray) -> list[str]:
    """Stable, equal-width per-tile ids from level-0 pixel coords."""
    width = len(str(int(coords.max()))) if coords.size else 1
    return [f"{int(x):0{width}d}_{int(y):0{width}d}" for x, y in coords]


def _physical_scale(header: dict) -> tuple[float | None, float | None, dict]:
    """Per-axis µm/level-0-px and the source origin (µm) from the store header.

    Prefers the per-axis ``source.scale_um`` (honours anisotropy); falls back to the
    scalar ``mpp_level0``. Returns ``(sx, sy, translation)`` with ``sx``/``sy`` None
    when the source carried no physical calibration (so no µm system can be built).
    """
    source = dict(header.get("source", {}))
    scale_um = source.get("scale_um") or {}
    mpp = source.get("mpp_level0")
    sx = scale_um.get("x", mpp)
    sy = scale_um.get("y", mpp)
    translation = source.get("level0_translation_um") or {}
    return sx, sy, translation


def _marker_field(entry: dict):
    """The marker-name value in a panel ``mapping`` entry, model-agnostically.

    KRONOS records it under ``kronos_marker``; a future multiplex model may use another
    ``*_marker`` key. Falls back through the known names so the table stays generic.
    """
    for k in ("marker", "kronos_marker", "canonical_name", "source_name"):
        if k in entry:
            return entry[k]
    for k in entry:
        if k.endswith("_marker"):
            return entry[k]
    return None


def _panel_dataframe(panel: dict):
    """Tidy, queryable marker map: one row per kept channel across multiplex models.

    Columns: ``model, channel, channel_index, marker, marker_id``. Returns ``None`` when
    no model carries a per-channel ``mapping`` (e.g. a brightfield run). Stored at
    ``uns["raw2features_panel"]`` - a DataFrame round-trips through anndata's zarr
    writer intact and is directly filterable, whereas the raw list-of-dicts gets
    stringified into unusable ``"{...}"`` blobs.
    """
    import pandas as pd

    rows = []
    for model, p in (panel or {}).items():
        for e in (p or {}).get("mapping", []) or []:
            rows.append(
                {
                    "model": model,
                    "channel": e.get("channel", e.get("source_name")),
                    "channel_index": e.get(
                        "channel_index", e.get("source_index")
                    ),
                    "marker": _marker_field(e),
                    "marker_id": e.get("marker_id"),
                }
            )
    if not rows:
        return None
    with pd.option_context("future.infer_string", False):
        return pd.DataFrame(
            rows, columns=["model", "channel", "channel_index", "marker", "marker_id"]
        )


def _panel_summary(panel: dict) -> dict:
    """Per-model coverage summary without the list-of-dicts ``mapping`` (uns-safe).

    Keeps the scalar/list provenance - ``n_markers`` and the ``kept`` / ``dropped`` /
    ``unmatched`` name lists and the ``vocabulary`` pointer - for a quick look in the
    export namespace. The full per-channel resolution lives in
    ``uns["raw2features_panel"]`` (queryable) and ``uns["raw2features"]`` (faithful).
    """
    keys = ("n_markers", "kept", "dropped", "unmatched", "vocabulary")
    return {
        model: {k: p[k] for k in keys if k in (p or {})}
        for model, p in (panel or {}).items()
    }


def _columnarize_records(value):
    """Recursively convert record lists to columns that AnnData can round-trip."""

    if isinstance(value, dict):
        return {key: _columnarize_records(item) for key, item in value.items()}
    if isinstance(value, list) and value and all(
        isinstance(item, dict) for item in value
    ):
        keys = list(dict.fromkeys(key for item in value for key in item))
        return {
            key: [_columnarize_records(item.get(key)) for item in value]
            for key in keys
        }
    if isinstance(value, list):
        return [_columnarize_records(item) for item in value]
    return value


def _uns_safe_header(header: dict) -> dict:
    """Copy multiplex records into an AnnData-safe recursive columnar form.

    AnnData's zarr writer turns a list of dictionaries into stringified reprs. This
    preserves top-level panel mappings, nested normalization bounds, and the mirrored
    model strategy/fingerprint records as reconstructable arrays without changing
    ordinary brightfield model headers.
    """
    panel = header.get("panel")
    if not panel:
        return header
    safe_panel = {}
    for model, p in panel.items():
        safe_panel[model] = _columnarize_records(dict(p or {}))
    result = {**header, "panel": safe_panel}
    models = header.get("models")
    if isinstance(models, dict):
        safe_models = dict(models)
        for model, metadata in models.items():
            if isinstance(metadata, dict) and metadata.get("multiplex") is not None:
                # The fingerprint digest authenticates the exact nested payload.
                # Columnarising its record lists would mutate that payload while
                # retaining the old digest, making the exported contract invalid.
                # Canonical JSON is AnnData-safe, lossless, and directly verifiable;
                # only the separate human-readable multiplex metadata is reshaped.
                fingerprint = metadata.get("output_fingerprint")
                readable = {
                    key: value
                    for key, value in metadata.items()
                    if key != "output_fingerprint"
                }
                safe_metadata = _columnarize_records(readable)
                if fingerprint is not None:
                    safe_metadata["output_fingerprint"] = json.dumps(
                        fingerprint,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                safe_models[model] = safe_metadata
        result["models"] = safe_models
    return result


def _coordinate_systems(sx, sy, translation):
    """SpatialData transforms keyed by coordinate system.

    ``global`` is the intrinsic level-0 **pixel** space (Identity) - kept so existing
    pixel-based tooling and ``to_legacy_anndata`` round-trips are unchanged. When the
    source is physically calibrated we also publish a ``micrometers`` coordinate system
    that maps pixels to µm via a Scale (plus a Translation for a non-zero source origin)
    - so tiles can be aligned to other slides/modalities in real units, which an
    Identity-only export cannot do.
    """
    from spatialdata.transformations import Identity, Scale, Sequence, Translation

    systems = {"global": Identity()}
    if sx and sy and sx > 0 and sy > 0:
        to_um = Scale([float(sx), float(sy)], axes=("x", "y"))
        ox, oy = float(translation.get("x", 0.0)), float(translation.get("y", 0.0))
        if ox or oy:
            to_um = Sequence([to_um, Translation([ox, oy], axes=("x", "y"))])
        systems["micrometers"] = to_um
    return systems


def export_spatialdata(
    store: str,
    out: str | None = None,
    *,
    models: list[str] | None = None,
    geometry: str = "polygon",
    overwrite: bool = False,
    grid: str | None = None,
) -> str:
    """Convert an embedding store to a SpatialData ``.zarr`` and return its path.

    Parameters
    ----------
    store:
        Path to a ``<slide_id>.embeddings.zarr`` written by raw2features.
    out:
        Output SpatialData store path. Defaults to a sibling
        ``<slide_id>.spatialdata.zarr``.
    models:
        Which feature arrays to export (each becomes ``obsm["X_<model>"]``).
        ``None`` exports every model present in the store.
    geometry:
        ``"polygon"`` (default) writes one square per patch in level-0 pixel
        space - faithful to the patch footprint. ``"circle"`` writes inscribed
        circles (the HEST/Visium "spot" convention); both keep ``obsm["spatial"]``
        tile centres, so either round-trips through ``to_legacy_anndata``.
    overwrite:
        Overwrite ``out`` if it already exists.
    """
    if geometry not in ("polygon", "circle"):
        raise ValueError(f"geometry must be 'polygon' or 'circle', got {geometry!r}")

    g = _open_store(store, grid)
    header = dict(g.attrs.get("raw2features", {}))
    patching = dict(header.get("patching", {}))

    coords = np.asarray(g["coords"], dtype=np.int64).reshape(-1, 2)
    grid_index = (
        np.asarray(g["grid_index"], dtype=np.int64).reshape(-1, 2)
        if "grid_index" in g
        else None
    )

    available = list(g["features"].keys())
    selected = list(models) if models else available
    missing = [m for m in selected if m not in available]
    if missing:
        raise ValueError(
            f"models {missing} not in store (available: {available or '<none>'})"
        )

    # Patch side in level-0 pixels - the square footprint of each tile.
    level0_patch = patching.get("level0_patch") or patching.get("patch_px")
    if not level0_patch:
        raise ValueError(
            "store header lacks patching.level0_patch / patch_px; cannot size tiles"
        )
    side = float(level0_patch)
    centers = coords.astype(np.float64) + side / 2.0  # (x, y) tile centres

    # Coordinate systems: pixel (global) + physical µm (micrometers) when calibrated.
    sx, sy, translation = _physical_scale(header)
    transforms = _coordinate_systems(sx, sy, translation)

    # 1) tile geometry first - its index is the table's instance key.
    shapes = _build_shapes(coords, centers, side, geometry, transforms)
    # 2) table, linked to the shapes element by index.
    table = _build_table(
        g, header, patching, coords, grid_index, centers, selected, side,
        instance_ids=shapes.index.values, coordinate_systems=list(transforms),
    )

    from spatialdata import SpatialData

    sdata = SpatialData(shapes={"tiles": shapes}, tables={"table": table})

    out = out or os.path.join(
        os.path.dirname(os.path.abspath(os.path.normpath(store))),
        f"{_slide_id(store)}.spatialdata.zarr",
    )
    if overwrite and os.path.exists(out):
        import shutil

        shutil.rmtree(out)
    sdata.write(out)
    return out


def _build_shapes(coords, centers, side, geometry, transforms):
    """Build the ``tiles`` shapes element (circles or square polygons).

    ``transforms`` maps each coordinate system (``global`` pixel, ``micrometers``) to
    the transform from the tiles' intrinsic level-0-pixel space into it.
    """
    from spatialdata.models import ShapesModel

    if geometry == "circle":
        return ShapesModel.parse(
            centers, geometry=0, radius=side / 2.0, transformations=transforms
        )
    import geopandas as gpd
    from shapely.geometry import box

    polys = [box(x, y, x + side, y + side) for x, y in coords]
    gdf = gpd.GeoDataFrame(geometry=polys)
    return ShapesModel.parse(gdf, transformations=transforms)


def _build_table(
    g, header, patching, coords, grid_index, centers, selected, side, *, instance_ids,
    coordinate_systems,
):
    import anndata as ad
    import pandas as pd
    from spatialdata.models import TableModel

    n = coords.shape[0]
    obs = {
        "x": coords[:, 0],  # level-0 top-left pixel (CLAM/Trident convention)
        "y": coords[:, 1],
        "level": int(patching.get("read_level", 0)),
        "mpp": float(patching.get("achieved_mpp") or patching.get("target_mpp") or 0.0),
    }
    if grid_index is not None:
        obs["array_row"] = grid_index[:, 0]
        obs["array_col"] = grid_index[:, 1]
        if "mask" in g:
            # `mask` is the (n_rows, n_cols) grid tissue-fraction map (0..255); index it
            # by each kept patch's (row, col) to get that patch's tissue fraction.
            mask = np.asarray(g["mask"])
            if mask.ndim == 2:
                obs["tissue_frac"] = (
                    mask[grid_index[:, 0], grid_index[:, 1]].astype(np.float32) / 255.0
                )

    # Build the table with pandas' legacy object strings. Under pandas 3 + pyarrow
    # (the [tables] extra) string columns/indexes/categoricals default to the pyarrow
    # backend (ArrowStringArray), which anndata's zarr writer cannot serialise; building
    # them under future.infer_string=False keeps the tile_id index and the `region`
    # categorical as object dtype so the store round-trips.
    with pd.option_context("future.infer_string", False):
        obs_df = pd.DataFrame(obs, index=pd.Index(_tile_ids(coords), name="tile_id"))
        adata = ad.AnnData(
            X=np.zeros((n, 0), dtype=np.float32),  # embeddings live in obsm
            obs=obs_df,
        )
        adata.obsm["spatial"] = centers.astype(np.float32)  # squidpy/HEST compatibility
        for model in selected:
            adata.obsm[f"X_{model}"] = np.asarray(
                g["features"][model], dtype=np.float32
            ).reshape(n, -1)

        # Full provenance header -> uns (license, model cards, segmentation, source).
        # The multiplex marker panel's per-channel `mapping` is a list of dicts; the
        # zarr writer stringifies list-of-dicts into unusable `"{...}"` blobs, so we
        # columnarise it first (round-trips as plain arrays, stays reconstructable).
        adata.uns["raw2features"] = _uns_safe_header(header)
        sx, sy, translation = _physical_scale(header)
        adata.uns["raw2features_export"] = {
            "exporter": "spatialdata",
            "models": list(selected),
            "patch_size_level0": float(side),
            "coords_convention": patching.get("coords_convention", _COORDS_CONVENTION),
            # The coordinate systems present on the tiles element + the pixel->µm scale
            # behind the physical one, so a consumer can read the µm calibration without
            # re-deriving it. None when the source carried no physical calibration.
            "coordinate_systems": list(coordinate_systems),
            "micrometers_per_pixel": (
                [float(sx), float(sy)] if sx and sy else None
            ),
        }
        panel = header.get("panel")  # multiplex marker-panel resolution (per model)
        if panel:
            # One tidy row per kept channel across all multiplex models ->
            # uns["raw2features_panel"] (channel -> marker + id, with a `model` column).
            # A DataFrame round-trips through anndata cleanly and is filterable, so a
            # consumer can answer "which channel fed marker X, under what id?" without
            # parsing the nested provenance blob.
            table_df = _panel_dataframe(panel)
            if table_df is not None:
                adata.uns["raw2features_panel"] = table_df
            # Compact per-model coverage summary in the export namespace (no dict list).
            adata.uns["raw2features_export"]["panel"] = _panel_summary(panel)
        if "slide" in g:  # carry slide-level vectors as unstructured metadata
            adata.uns["slide_embeddings"] = {
                m: np.asarray(g["slide"][m]).reshape(-1).tolist()
                for m in g["slide"].keys()
            }

        # Link to the shapes element: region name + per-row instance id.
        adata.obs["region"] = pd.Categorical(["tiles"] * n)
        adata.obs["instance_id"] = np.asarray(instance_ids)
        return TableModel.parse(
            adata, region="tiles", region_key="region", instance_key="instance_id"
        )
