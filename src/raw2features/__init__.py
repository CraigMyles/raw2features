"""raw2features - read OME-Zarr WSIs and emit patch-level FM embeddings.

The storage backend (reader) and model implementation are independently swappable
via six plugin seams: readers, segmenters, patchers, embedders, sinks, and
slide_embedders (see ``raw2features.core.plugins.SEAMS``). Third parties add an
implementation through the matching ``raw2features.<seam>`` entry-point group.
For embedders, that entry point is a loader family used by the packaged model
registry; it does not add a new CLI ``--model`` name. Programmatic callers can pass
external patch-model instances through ``embed_slide(..., embedders=[...])``.
Those group names and the explicit ``embedders=`` hook are the public plugin contract.

Public API (stable import surface - couple to these, not to deep module paths):

* ``embed_slide(slide_path, out_dir, cfg)`` - high-level per-model geometry runner.
* ``run_slide(slide_path, out_dir, cfg)`` / ``RunConfig`` - single-grid primitive.
* ``register`` / ``available`` / ``get`` - the plugin registry.
* ``write_patches_geojson`` - patch grid -> QuPath-readable GeoJSON.
* ``validate_store`` - conformance-check an embeddings store against docs/SPEC.md.

These are re-exported lazily, so ``import raw2features`` stays light (the core
install has no torch/zarr); importing a name pulls only what it needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "embed_slide",
    "run_slide",
    "RunConfig",
    "register",
    "available",
    "get",
    "write_patches_geojson",
    "validate_store",
]

if TYPE_CHECKING:  # import-time only for type checkers; never imported at runtime
    from raw2features.core.plugins import available, get, register
    from raw2features.pipeline.runner import RunConfig, embed_slide, run_slide
    from raw2features.sinks.zarr_sink import write_patches_geojson
    from raw2features.spec import validate_store


def __getattr__(name: str):
    """Resolve a blessed public name on first access (PEP 562 lazy re-export)."""
    if name in ("embed_slide", "run_slide", "RunConfig"):
        from raw2features.pipeline import runner

        return getattr(runner, name)
    if name in ("register", "available", "get"):
        from raw2features.core import plugins

        return getattr(plugins, name)
    if name == "write_patches_geojson":
        from raw2features.sinks.zarr_sink import write_patches_geojson

        return write_patches_geojson
    if name == "validate_store":
        from raw2features.spec import validate_store

        return validate_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
