"""``raw2features info <slide.zarr>`` - probe a reader without extracting."""

from __future__ import annotations

import typer

from raw2features.core import plugins
from raw2features.core.uris import source_uri


def info(
    slide: str = typer.Argument(..., help="Path to an OME-Zarr store."),
    mpp: float = typer.Option(
        1.0, "--mpp", help="Show the exact level/resample plan for this target MPP."
    ),
    patch_size: int = typer.Option(
        224, "--patch-size", help="Patch size for the plan."
    ),
    reader: str = typer.Option("omezarr", "--reader"),
) -> None:
    """Print NGFF version, MPP, pyramid levels, and the exact-MPP read plan."""
    reader_cls = plugins.get("readers", reader)
    with reader_cls(slide) as rdr:
        typer.echo(f"path:          {source_uri(rdr.path)}")
        typer.echo(f"ngff_version:  {getattr(rdr, 'ngff_version', None)}")
        typer.echo(f"mpp_level0:    {rdr.mpp}")
        typer.echo("levels:")
        for i, (dim, ds) in enumerate(
            zip(rdr.level_dimensions, rdr.level_downsamples(), strict=True)
        ):
            native = rdr.mpp * ds if rdr.mpp else float("nan")
            typer.echo(
                f"  L{i}: {dim.width:>7} x {dim.height:<7}  "
                f"downsample {ds:>6.2f}  mpp {native:.4f}"
            )
        if rdr.mpp:
            plan = rdr.level_for_mpp(mpp, patch_size)
            typer.echo(
                f"plan @ {mpp} um/px, {patch_size}px: read L{plan.level} "
                f"({plan.read_px}px, resample {plan.resample:.4f}) -> "
                f"achieved {plan.achieved_mpp:.4f} um/px"
                + ("" if plan.needs_resample else " [native, no resize]")
            )
