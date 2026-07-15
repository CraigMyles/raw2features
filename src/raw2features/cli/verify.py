"""``raw2features verify`` - exit 0 iff a slide's output is complete & valid.

Used by the SLURM array for skip-if-complete. The content-affecting flags must
match the ``embed`` invocation so the config hash lines up.
"""

from __future__ import annotations

import os

import typer

from raw2features.pipeline.receipt import (
    canonical_source_uri,
    is_complete,
)
from raw2features.pipeline.runner import (
    RunConfig,
    expected_model_contracts,
    slide_id_from_path,
)


def verify(
    slide: str = typer.Argument(...),
    receipts_dir: str = typer.Option(..., "--receipts-dir"),
    out_dir: str | None = typer.Option(
        None,
        "--out-dir",
        help="Expected output directory; binds the receipt to this run's target store.",
    ),
    feature_extractor: list[str] = typer.Option(
        ["resnet50"], "--model", "-m", "--feature-extractor", "-f"
    ),
    mpp: float | None = typer.Option(
        None, "--mpp", help="Target µm/px (default: model recommended; matches embed)."
    ),
    patch_size: int | None = typer.Option(
        None, "--patch-size", help="Patch size in px (default: model recommended)."
    ),
    step: int | None = typer.Option(None, "--step"),
    reader: str = typer.Option("omezarr", "--reader"),
    segmenter: str = typer.Option("otsu", "--segmenter"),
    no_seg: bool = typer.Option(False, "--no-seg"),
    tissue_threshold: float = typer.Option(0.1, "--tissue-threshold"),
    features_dtype: str = typer.Option("float16", "--features-dtype"),
    stain_norm: str | None = typer.Option(
        None, "--stain-norm", help="Must match the embed run (content-hash field)."
    ),
    amp: str = typer.Option("auto", "--amp"),
    device: str = typer.Option(
        "auto",
        "--device",
        help="auto | cuda | mps | cpu (must match the effective embed precision)",
    ),
    snap_to_level: bool = typer.Option(False, "--snap-to-level"),
    mpp_tolerance: float = typer.Option(0.001, "--mpp-tolerance"),
    allow_upsample: bool = typer.Option(False, "--allow-upsample"),
    config: str | None = typer.Option(
        None, "--config", help="YAML extraction plan (must match the embed run)."
    ),
    quiet: bool = typer.Option(False, "--quiet"),
) -> None:
    """Exit 0 if the slide is already complete & output-validated, else exit 1."""
    from raw2features.core.device import resolve_device
    from raw2features.pipeline.runner import resolve_run

    geometry_config = None
    if config:
        from raw2features.core.config import load_extractions

        geometry_config = load_extractions(config)
        models = list(dict.fromkeys(e["model"] for e in geometry_config))
    else:
        models = list(feature_extractor)
    cfg = RunConfig(
        models=models,
        reader=reader,
        segmenter=segmenter,
        no_seg=no_seg,
        target_mpp=mpp if mpp is not None else 1.0,  # per-group geometry overrides this
        patch_px=patch_size if patch_size is not None else 224,
        step_px=step,
        tissue_threshold=tissue_threshold,
        features_dtype=features_dtype,
        stain_norm=stain_norm,
        snap_to_level=snap_to_level,
        mpp_tolerance=mpp_tolerance,
        allow_upsample=allow_upsample,
        amp=amp,
        device=resolve_device(device),
    )
    # Hash exactly as embed_slide does, so verify matches the embed that wrote it.
    _, group_cfgs, run_hash = resolve_run(cfg, mpp, patch_size, geometry_config)
    expected_source = canonical_source_uri(slide)
    if expected_source is None:
        if not quiet:
            typer.echo("source URI is malformed; incomplete", err=True)
        raise typer.Exit(code=1)
    try:
        slide_id = slide_id_from_path(slide)
    except Exception:  # noqa: BLE001 - suppress credential-bearing parser errors
        if not quiet:
            typer.echo("source URI is malformed; incomplete", err=True)
        raise typer.Exit(code=1) from None
    expected_path = (
        os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
        if out_dir is not None
        else None
    )
    expected_output = (
        f"file://{os.path.abspath(expected_path)}"
        if expected_path is not None
        else None
    )
    from raw2features.embedders.model_registry import load_registry

    registered = load_registry()
    unknown = [model for model in models if model not in registered]
    if unknown:
        if not quiet:
            typer.echo(
                "cannot derive the current output contract for unregistered "
                f"model(s) {', '.join(unknown)}; incomplete",
                err=True,
            )
        raise typer.Exit(code=1)
    contracts = expected_model_contracts(cfg)
    expected_grid_models = {
        group_cfg.grid_hash(): list(group_cfg.models) for group_cfg in group_cfgs
    }
    compatible_grid_hashes = {
        group_cfg.grid_hash(): group_cfg.compatible_legacy_grid_hashes()
        for group_cfg in group_cfgs
    }
    complete = is_complete(
        receipts_dir,
        slide_id,
        run_hash,
        expected_source_uri=expected_source,
        expected_output_uri=expected_output,
        expected_model_contracts=contracts,
        expected_grid_models=expected_grid_models,
        compatible_grid_hashes=compatible_grid_hashes,
    )
    if not quiet:
        typer.echo(f"{slide_id}: {'complete' if complete else 'incomplete'}")
    raise typer.Exit(code=0 if complete else 1)
