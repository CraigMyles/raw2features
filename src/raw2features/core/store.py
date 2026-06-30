"""Embeddings-store layout: the ``grids/<key>/`` nesting and how to navigate it.

A store is a ``<slide_id>.embeddings.zarr`` whose patch sets live UNIFORMLY under
``grids/<key>/`` -- one child group per geometry ``(mpp, patch_px)``. Each grid group
is self-describing: a complete header (the format-0.1 header shape) plus ``coords`` /
``grid_index`` / ``mask`` / ``features/<model>`` / ``slide/<model>``. The root group
carries a ``grids`` index (key -> geometry summary). There is no flat single-grid
special case -- a one-grid store is just a ``grids/`` with one child.

These helpers are the single place that knows the layout, so a reader descends with
``open_grid`` instead of hard-coding paths.
"""

from __future__ import annotations

GRIDS = "grids"  # the root group holding one subgroup per geometry


def grid_key(target_mpp: float, patch_px: int) -> str:
    """Human-readable, filesystem-safe label for one geometry, e.g. ``mpp0.5_px224``.

    ``%g`` trims trailing zeros (0.5 -> ``0.5``, 1.0 -> ``1``, 0.25 -> ``0.25``), so the
    key is stable and readable. The authoritative geometry still lives in the grid's
    header; the key is only an addressable label.
    """
    return f"mpp{float(target_mpp):g}_px{int(patch_px)}"


def open_root(path: str, mode: str = "r"):
    """Open the root group of a store at ``path`` (strips a ``file://`` prefix)."""
    import zarr

    return zarr.open_group(str(path).removeprefix("file://"), mode=mode)


def grid_keys(root) -> list[str]:
    """Sorted grid keys present in an open root group (empty if none)."""
    return sorted(root[GRIDS].keys()) if GRIDS in root else []


def open_grid(path_or_root, key: str | None = None, mode: str = "r"):
    """Open one grid group of a store.

    ``path_or_root`` is a filesystem path (opened per ``mode``) or an already-open root
    group. ``key=None`` returns the SOLE grid, raising ValueError when the store holds
    zero or several grids (the caller must then pass a key). Unknown key -> KeyError.
    """
    root = (
        path_or_root
        if hasattr(path_or_root, "attrs")
        else open_root(path_or_root, mode)
    )
    if GRIDS not in root:
        raise ValueError("store has no grids/ group (not a v0.1 embeddings store)")
    grids = root[GRIDS]
    if key is None:
        keys = grid_keys(root)
        if len(keys) == 1:
            return grids[keys[0]]
        raise ValueError(
            f"store has {len(keys)} grids {keys}; pass a grid key to choose one"
        )
    return grids[key]


def grid_for_model(root, model: str) -> str:
    """Key of the single grid whose ``features`` contains ``model``.

    Raises ValueError when the model is in zero or several grids (the caller then passes
    an explicit key).
    """
    hits = [
        k
        for k in grid_keys(root)
        if "features" in root[GRIDS][k] and model in root[GRIDS][k]["features"]
    ]
    if len(hits) == 1:
        return hits[0]
    raise ValueError(
        f"model {model!r} is present in {len(hits)} grids {hits}; specify a grid key"
    )
