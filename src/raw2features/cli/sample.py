"""``raw2features sample`` - write a tiny synthetic OME-Zarr slide to try offline."""

from __future__ import annotations

import typer


def sample(
    out_path: str = typer.Argument(
        "sample.ome.zarr", help="Where to write the synthetic OME-Zarr slide."
    ),
    size: int = typer.Option(1024, "--size", help="Level-0 side in pixels (square)."),
    mpp: float = typer.Option(0.5, "--mpp", help="Level-0 microns/pixel."),
) -> None:
    """Write a small synthetic OME-NGFF slide so you can try the pipeline offline.

    No download, token, or GPU needed:

        raw2features sample sample.ome.zarr
        raw2features embed  sample.ome.zarr out/ -m resnet50 --device auto
    """
    from raw2features.data import write_sample_slide

    path = write_sample_slide(out_path, mpp0=mpp, size=size)
    typer.echo(f"wrote synthetic OME-Zarr slide: {path}")
    typer.echo(f"try:  raw2features embed {path} out/ -m resnet50 --device auto")
