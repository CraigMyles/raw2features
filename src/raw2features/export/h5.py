"""Export an embedding store to the pathology-MIL HDF5 layouts (TRIDENT / STAMP).

A **non-default, egress-only** convenience: the native ``.embeddings.zarr`` stays the
FAIR primary output (see the README). This bridges to existing toolchains that read
HDF5 so people already in those pipelines can drop raw2features in as the extractor:

* ``trident`` - ``features`` (N, dim) + ``coords`` (N, 2) in **level-0 pixels**, with
  ``coords.attrs['patch_size_level0']`` (the CLAM / TRIDENT / TITAN convention; TITAN
  reads exactly this to build its patch grid).
* ``clam`` - the same layout as ``trident`` but with ``coords`` as **int32** (the exact
  dtype CLAM's feature ``.h5`` uses). ``trident`` is otherwise a byte-superset of CLAM's
  feature file, so this is a thin alias for users feeding CLAM directly.
* ``stamp`` - ``feats`` (N, dim) fp16 + ``coords`` (N, 2) in **microns**, with
  ``unit='um'``, ``tile_size_um`` and ``tile_size_px`` (KatherLab STAMP convention).

Both layouts hold a *single* encoder, so one ``.h5`` is written per model. The schemas
are transcribed from the projects' own source and implemented with an independent
``h5py`` writer - file formats are not copyrightable and no project's code is copied
(notably CLAM's GPL writer is never vendored). ``h5py`` ships in the ``[h5]`` extra.
"""

from __future__ import annotations

import os

import numpy as np

_LAYOUTS = ("trident", "clam", "stamp")


def _open_store(store: str, grid: str | None = None):
    from raw2features.core.store import open_grid

    # grid=None opens the sole grid; a multi-grid store errors asking for --grid <key>.
    g = open_grid(store, grid)
    if "features" not in g:
        raise ValueError(f"{store!r} has no 'features' group - not an embedding store?")
    return g


def _slide_id(store: str) -> str:
    base = os.path.basename(os.path.normpath(store))
    for suffix in (".embeddings.zarr", ".zarr"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _mag(mpp: float | None) -> float | None:
    """Approximate objective magnification from MPP (0.5 µm/px ≈ 20×)."""
    return round(10.0 / mpp, 1) if mpp else None


def export_h5(
    store: str,
    out_dir: str | None = None,
    *,
    models: list[str] | None = None,
    layout: str = "trident",
    overwrite: bool = False,
    grid: str | None = None,
) -> list[str]:
    """Write one HDF5 file per model and return the written paths.

    Parameters
    ----------
    store:
        Path to a ``<slide_id>.embeddings.zarr`` written by raw2features.
    out_dir:
        Directory for the ``.h5`` file(s). Defaults to the store's directory.
    models:
        Which feature arrays to export (one file each). ``None`` = all present.
    layout:
        ``"trident"`` (features + level-0-pixel coords + ``patch_size_level0``),
        ``"clam"`` (``trident`` with int32 coords), or ``"stamp"`` (feats fp16 +
        micron coords + ``tile_size_um``).
    overwrite:
        Overwrite existing ``.h5`` files.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {_LAYOUTS}, got {layout!r}")
    import h5py

    g = _open_store(store, grid)
    header = dict(g.attrs.get("raw2features", {}))
    patching = dict(header.get("patching", {}))
    source = dict(header.get("source", {}))

    coords = np.asarray(g["coords"], dtype=np.int64).reshape(-1, 2)
    available = list(g["features"].keys())
    selected = list(models) if models else available
    missing = [m for m in selected if m not in available]
    if missing:
        raise ValueError(
            f"models {missing} not in store (have {available or '<none>'})"
        )

    level0_patch = patching.get("level0_patch") or patching.get("patch_px")
    if not level0_patch:
        raise ValueError("store header lacks patching.level0_patch / patch_px")
    patch_px = int(patching.get("patch_px") or level0_patch)
    achieved_mpp = float(
        patching.get("achieved_mpp") or patching.get("target_mpp") or 0
    )
    mpp_level0 = source.get("mpp_level0")

    out_dir = out_dir or os.path.dirname(os.path.abspath(os.path.normpath(store)))
    os.makedirs(out_dir, exist_ok=True)
    slide_id = _slide_id(store)
    single = len(selected) == 1

    written: list[str] = []
    for model in selected:
        feats = np.asarray(g["features"][model]).reshape(coords.shape[0], -1)
        name = f"{slide_id}.h5" if single else f"{slide_id}.{model}.h5"
        path = os.path.join(out_dir, name)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        with h5py.File(path, "w") as fh:
            if layout in ("trident", "clam"):
                _write_trident(
                    fh,
                    feats,
                    coords,
                    level0_patch,
                    achieved_mpp,
                    mpp_level0,
                    model,
                    coord_dtype=np.int32 if layout == "clam" else np.int64,
                )
            else:
                _write_stamp(fh, feats, coords, patch_px, achieved_mpp, model, header)
        written.append(path)
    return written


def _write_trident(
    fh,
    feats,
    coords,
    level0_patch,
    achieved_mpp,
    mpp_level0,
    model,
    coord_dtype=np.int64,
):
    """CLAM / TRIDENT / TITAN layout: features + level-0-pixel coords + attrs.

    ``coord_dtype`` is int64 for ``trident`` and int32 for the ``clam`` alias (CLAM's
    feature ``.h5`` writes coords as int32); the layouts are otherwise identical.
    """
    fh.create_dataset("features", data=feats.astype(np.float32))
    c = fh.create_dataset("coords", data=coords.astype(coord_dtype))
    # patch_size_level0 == the level-0 patch extent; for the default non-overlap grid
    # this is also the inter-patch stride TITAN uses to build its feature grid.
    c.attrs["patch_size_level0"] = int(level0_patch)
    c.attrs["patch_size"] = int(level0_patch)
    if achieved_mpp:
        c.attrs["target_magnification"] = _mag(achieved_mpp)
    if mpp_level0:
        c.attrs["level0_magnification"] = _mag(mpp_level0)
    fh["features"].attrs["encoder"] = model


def _write_stamp(fh, feats, coords, patch_px, achieved_mpp, model, header):
    """KatherLab STAMP layout: feats (fp16) + slide-relative micron coords."""
    source = dict(header.get("source", {}))
    scale_um = source.get("scale_um")
    if (
        isinstance(scale_um, dict)
        and scale_um.get("x") is not None
        and scale_um.get("y") is not None
    ):
        sx, sy = float(scale_um["x"]), float(scale_um["y"])
    else:
        # Legacy stores carry only the isotropic scalar and imply a zero origin.
        mpp_level0 = source.get("mpp_level0")
        if not mpp_level0:
            raise ValueError(
                "STAMP layout needs source.scale_um or source.mpp_level0 to "
                "convert px -> microns"
            )
        sx = sy = float(mpp_level0)

    if sx <= 0 or sy <= 0:
        raise ValueError(
            "STAMP layout needs positive source.scale_um/source.mpp_level0"
        )
    # STAMP coordinates are measured from the top-left of the WSI scan.  An NGFF
    # translation describes the source's physical/stage origin, so adding it here
    # would shift STAMP heatmaps relative to the pixels they are drawn over.
    coords_um = coords.astype(np.float32) * np.asarray([sx, sy], dtype=np.float32)
    fh.create_dataset("feats", data=feats.astype(np.float16))
    fh.create_dataset("coords", data=coords_um)
    fh.attrs["unit"] = "um"
    fh.attrs["tile_size_um"] = float(patch_px) * achieved_mpp  # physical tile size
    fh.attrs["tile_size_px"] = int(patch_px)
    fh.attrs["feat_type"] = "tile"
    fh.attrs["extractor"] = model
    fh.attrs["raw2features_version"] = str(header.get("schema_version", ""))
