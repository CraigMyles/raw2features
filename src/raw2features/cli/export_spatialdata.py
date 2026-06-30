"""``raw2features export-spatialdata`` - convert an embedding store to SpatialData.

Post-hoc: reads an existing ``<slide_id>.embeddings.zarr`` and writes a scverse
SpatialData ``.zarr`` (``tiles`` shapes + ``table`` AnnData with ``obsm["X_<model>"]``).
Needs the optional ``[spatialdata]`` extra. Embeddings are never recomputed.
"""

from __future__ import annotations

import typer


def export_spatialdata(
    store: str = typer.Argument(
        ..., help="Path to a <slide_id>.embeddings.zarr written by raw2features."
    ),
    out: str | None = typer.Argument(
        None, help="Output .zarr (default: sibling <slide_id>.spatialdata.zarr)."
    ),
    models: list[str] | None = typer.Option(
        None, "--model", "-m", help="Model(s) to export (repeatable; default: all)."
    ),
    geometry: str = typer.Option(
        "polygon", "--geometry", help="Tile geometry: 'polygon' (square) or 'circle'."
    ),
    grid: str | None = typer.Option(
        None, "--grid",
        help="Grid key (e.g. mpp0.5_px224) for a multi-grid store; omit for a single "
        "grid. A multi-grid store errors listing its keys when --grid is missing.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite the output store if it exists."
    ),
) -> None:
    """Export patch embeddings to a SpatialData store (squidpy/napari-readable)."""
    try:
        from raw2features.export.spatialdata import export_spatialdata as _export
    except ImportError as exc:  # pragma: no cover - only without the extra
        raise typer.BadParameter(
            "SpatialData export needs the optional extra: "
            'pip install "raw2features[spatialdata]"'
        ) from exc

    try:
        path = _export(
            store,
            out,
            models=list(models) if models else None,
            geometry=geometry,
            overwrite=overwrite,
            grid=grid,
        )
    except ImportError as exc:  # spatialdata/anndata/geopandas missing at call time
        raise typer.BadParameter(
            f"{exc}. Install the extra: pip install \"raw2features[spatialdata]\""
        ) from exc
    typer.echo(path)
