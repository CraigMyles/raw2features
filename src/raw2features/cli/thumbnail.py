"""``raw2features thumbnail`` - write a slide thumbnail (+ optional QC overlay).

Standalone so thumbnails can be batched separately, before or after the heavy
embed run. ``--overlay`` reruns the (cheap) segmenter + patcher to show the
tissue mask and the patches that tiling would keep for the given grid params.
"""

from __future__ import annotations

import typer

from raw2features.core import plugins
from raw2features.viz import DEFAULT_THUMBNAIL_MPP, write_thumbnails


def thumbnail(
    slide: str = typer.Argument(..., help="Path to an OME-Zarr store."),
    out_dir: str = typer.Argument(..., help="Output directory for the thumbnail."),
    reader: str = typer.Option("omezarr", "--reader"),
    thumbnail_mpp: float = typer.Option(
        DEFAULT_THUMBNAIL_MPP,
        "--thumbnail-mpp",
        help="Thumbnail MPP (default = seg MPP).",
    ),
    max_px: int | None = typer.Option(
        None, "--max-px", help="Cap longest side (overrides --thumbnail-mpp)."
    ),
    overlay: bool = typer.Option(
        False, "--overlay", help="Also tint the tissue mask + outline kept patches."
    ),
    segmenter: str = typer.Option("otsu", "--segmenter", help="Overlay segmenter."),
    mpp: float = typer.Option(1.0, "--mpp", help="Target MPP for the grid (overlay)."),
    patch_size: int = typer.Option(224, "--patch-size", help="Patch px (overlay)."),
    step: int | None = typer.Option(None, "--step", help="Stride (overlay)."),
    tissue_threshold: float = typer.Option(0.1, "--tissue-threshold"),
) -> None:
    """Render a thumbnail; with --overlay, show the tissue mask + kept patches."""
    from raw2features.pipeline.runner import slide_id_from_path

    reader_cls = plugins.get("readers", reader)
    rdr = reader_cls(slide)
    slide_id = slide_id_from_path(slide)

    tissue = None
    coords = None
    level0_patch = None
    with rdr:
        if overlay:
            from raw2features.patcher.grid import GridPatcher

            seg = plugins.get("segmenters", segmenter)()
            tissue = seg.segment(rdr)
            patcher = GridPatcher(target_mpp=mpp, patch_px=patch_size, step_out_px=step)
            grid = patcher.build_grid(rdr)
            coords, _grid_index, _grid_tissue = patcher.tile(
                grid, tissue, tissue_threshold
            )
            level0_patch = grid.level0_patch
        meta = write_thumbnails(
            rdr,
            out_dir,
            slide_id,
            mpp=thumbnail_mpp,
            max_px=max_px,
            tissue=tissue,
            coords=coords,
            level0_patch=level0_patch,
            overlay=overlay,
        )
    for key, value in meta.items():
        typer.echo(f"{key}: {value}")
