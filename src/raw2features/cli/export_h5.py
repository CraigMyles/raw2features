"""``raw2features export-h5`` - export to pathology-MIL HDF5 (TRIDENT / STAMP).

NON-DEFAULT egress convenience for feeding existing toolchains (CLAM/TRIDENT/TITAN or
KatherLab STAMP). The native ``.embeddings.zarr`` stays the FAIR primary output. One
``.h5`` is written per model. Needs the optional ``[h5]`` extra (``h5py``).
"""

from __future__ import annotations

import typer


def export_h5(
    store: str = typer.Argument(
        ..., help="Path to a <slide_id>.embeddings.zarr written by raw2features."
    ),
    out_dir: str | None = typer.Argument(
        None, help="Output directory (default: the store's directory)."
    ),
    layout: str = typer.Option(
        "trident", "--layout",
        help="HDF5 layout: 'trident' (CLAM/TITAN), 'clam' (int32 coords), or 'stamp'.",
    ),
    models: list[str] | None = typer.Option(
        None, "--model", "-m", help="Model(s) to export (one .h5 each; default: all)."
    ),
    grid: str | None = typer.Option(
        None, "--grid",
        help="Grid key (e.g. mpp0.5_px224) for a multi-grid store; omit for a single "
        "grid. A multi-grid store errors listing its keys when --grid is missing.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing .h5 files."
    ),
) -> None:
    """Export patch embeddings to TRIDENT- or STAMP-compatible HDF5 (non-default)."""
    try:
        from raw2features.export.h5 import export_h5 as _export
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter(
            'HDF5 export needs the optional extra: pip install "raw2features[h5]"'
        ) from exc
    try:
        paths = _export(
            store, out_dir, models=list(models) if models else None,
            layout=layout, overwrite=overwrite, grid=grid,
        )
    except ImportError as exc:  # h5py missing at call time
        raise typer.BadParameter(
            f'{exc}. Install the extra: pip install "raw2features[h5]"'
        ) from exc
    for p in paths:
        typer.echo(p)
