"""``raw2features verify`` - exit 0 iff a slide's output is complete & valid.

Used by the SLURM array for skip-if-complete. The content-affecting flags must
match the ``embed`` invocation so the config hash lines up.
"""

from __future__ import annotations

import os

import typer

from raw2features.core.uris import redact_uri_credentials
from raw2features.pipeline.receipt import (
    canonical_source_uri,
    is_complete,
)
from raw2features.pipeline.runner import (
    RunConfig,
    expected_model_contracts,
    resolve_multiplex_output_contracts,
    resolve_multiplex_source_config,
    slide_id_from_path,
)

from ._validation import (
    parse_channel_names_file,
    parse_json_object,
    validate_amp,
    validate_geometry,
    validate_multiplex_percentiles,
    validate_positive_int,
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
    source_mpp: float | None = typer.Option(
        None,
        "--source-mpp",
        help="Source level-0 µm/px override (must match the embed run).",
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
    multiplex_strategy: str | None = typer.Option(
        None,
        "--multiplex-strategy",
        help="Multiplex strategy used by the embed run (initially: channelwise).",
    ),
    multiplex_markers: list[str] = typer.Option(
        [],
        "--marker",
        help="Multiplex marker; repeat in the exact order used by the embed run.",
    ),
    channel_names_file: str | None = typer.Option(
        None,
        "--channel-names-file",
        help="UTF-8 .txt/.csv/.tsv with exactly one ordered name per physical C-axis "
        "position; must resolve to the same effective panel used by embed.",
    ),
    multiplex_normalization: str = typer.Option(
        "percentile",
        "--multiplex-normalization",
        help="Per-marker whole-image-level normalization used by the embed run.",
    ),
    multiplex_percentile_low: float = typer.Option(
        1.0,
        "--multiplex-percentile-low",
        help="Lower normalization percentile used by the embed run.",
    ),
    multiplex_percentile_high: float = typer.Option(
        99.0,
        "--multiplex-percentile-high",
        help="Upper normalization percentile used by the embed run.",
    ),
    multiplex_normalization_max_side_px: int = typer.Option(
        2048,
        "--multiplex-normalization-max-side-px",
        help="Whole-image normalization level limit used by the embed run.",
    ),
    multiplex_aggregation: str = typer.Option(
        "mean",
        "--multiplex-aggregation",
        help="Marker aggregation used by the embed run: mean | concat.",
    ),
    multiplex_params: str | None = typer.Option(
        None,
        "--multiplex-params",
        help="JSON content parameters used by a third-party multiplex strategy.",
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
    validate_amp(amp)
    validate_geometry(mpp=mpp, patch_size=patch_size, step=step, source_mpp=source_mpp)
    validate_positive_int(
        multiplex_normalization_max_side_px,
        "--multiplex-normalization-max-side-px",
    )
    validate_multiplex_percentiles(multiplex_percentile_low, multiplex_percentile_high)
    strategy_params = parse_json_object(multiplex_params, "--multiplex-params")
    channel_names_override = parse_channel_names_file(channel_names_file)
    from raw2features.core.device import resolve_device
    from raw2features.embedders.model_registry import resolve_geometry
    from raw2features.pipeline.runner import resolve_run

    geometry_config = None
    if config:
        from raw2features.core.config import load_extractions

        geometry_config = load_extractions(config)
        models = list(dict.fromkeys(e["model"] for e in geometry_config))
    else:
        models = list(feature_extractor)
    # Verification has no programmatic external-embedder instance from which to
    # derive a current contract or geometry. Fail closed with the established clean
    # diagnostic before the geometry resolver rejects the unknown name.
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
    groups = resolve_geometry(models, mpp, patch_size, geometry_config)
    if not groups:
        raise typer.BadParameter("at least one model/extraction is required")
    representative = groups[0]
    cfg = RunConfig(
        models=models,
        reader=reader,
        segmenter=segmenter,
        no_seg=no_seg,
        target_mpp=representative.mpp,
        source_mpp=source_mpp,
        patch_px=representative.patch_px,
        step_px=step,
        tissue_threshold=tissue_threshold,
        features_dtype=features_dtype,
        stain_norm=stain_norm,
        multiplex_strategy=multiplex_strategy,
        multiplex_markers=list(multiplex_markers),
        multiplex_normalization=multiplex_normalization,
        multiplex_percentile_low=multiplex_percentile_low,
        multiplex_percentile_high=multiplex_percentile_high,
        multiplex_normalization_max_side_px=multiplex_normalization_max_side_px,
        multiplex_aggregation=multiplex_aggregation,
        multiplex_strategy_params=strategy_params,
        channel_names_override=channel_names_override,
        snap_to_level=snap_to_level,
        mpp_tolerance=mpp_tolerance,
        allow_upsample=allow_upsample,
        amp=amp,
        device=resolve_device(device),
    )
    try:
        cfg = resolve_multiplex_source_config(slide, cfg)
    except (OSError, ValueError) as exc:
        if not quiet:
            typer.echo(
                redact_uri_credentials(
                    f"multiplex panel is invalid: {exc}; incomplete"
                ),
                err=True,
            )
        raise typer.Exit(code=1) from exc
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
    if cfg.multiplex_strategy is None:
        contracts = expected_model_contracts(cfg)
        expected_grid_models = {
            group_cfg.grid_hash(): list(group_cfg.models) for group_cfg in group_cfgs
        }
    else:
        contracts = {}
        expected_grid_models = {}
        for group_cfg in group_cfgs:
            effective_cfg, group_contracts = resolve_multiplex_output_contracts(
                slide,
                group_cfg,
                device=cfg.device,
            )
            for name, contract in group_contracts.items():
                if name in contracts and contracts[name] != contract:
                    raise ValueError(
                        f"multiplex effective model key {name!r} resolved to "
                        "different contracts within one request"
                    )
                contracts[name] = contract
            expected_grid_models[group_cfg.grid_hash()] = list(effective_cfg.models)
    compatible_grid_hashes = {
        group_cfg.grid_hash(): group_cfg.compatible_legacy_grid_hashes()
        for group_cfg in group_cfgs
    }
    compatible_grid_segmenters = {
        group_cfg.grid_hash(): group_cfg.compatible_legacy_grid_segmenters()
        for group_cfg in group_cfgs
    }
    allow_hashless_legacy_grids = {
        group_cfg.grid_hash(): group_cfg.allows_hashless_legacy_grid()
        for group_cfg in group_cfgs
    }
    completion_kwargs = {
        "expected_source_uri": expected_source,
        "expected_output_uri": expected_output,
        "expected_model_contracts": contracts,
        "expected_grid_models": expected_grid_models,
        "compatible_grid_hashes": compatible_grid_hashes,
        "compatible_grid_segmenters": compatible_grid_segmenters,
        "allow_hashless_legacy_grids": allow_hashless_legacy_grids,
    }
    complete = is_complete(
        receipts_dir,
        slide_id,
        run_hash,
        **completion_kwargs,
    )
    if not complete and len(group_cfgs) == 1:
        # ``run_slide`` is the public single-grid primitive and historically writes
        # its direct RunConfig hash, while ``embed``/``embed_slide`` write the whole-
        # request hash above. Accept either only for a genuine one-grid request; both
        # paths still validate source, output, live arrays, and output fingerprints.
        complete = is_complete(
            receipts_dir,
            slide_id,
            group_cfgs[0].content_hash(),
            **completion_kwargs,
        )
    if not quiet:
        typer.echo(f"{slide_id}: {'complete' if complete else 'incomplete'}")
    raise typer.Exit(code=0 if complete else 1)
