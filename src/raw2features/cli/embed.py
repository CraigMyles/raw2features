"""``raw2features embed`` - read a slide and write patch embeddings."""

from __future__ import annotations

import os
import sys

import typer

from raw2features.core.provenance import sanitize_argv
from raw2features.pipeline.runner import RunConfig, embed_slide
from raw2features.viz import DEFAULT_THUMBNAIL_MPP

from ._validation import (
    parse_channel_names_file,
    parse_json_object,
    validate_amp,
    validate_batch_size,
    validate_geometry,
    validate_multiplex_percentiles,
    validate_positive_int,
)


def embed(
    slide: str = typer.Argument(..., help="Path to an OME-Zarr store."),
    out_dir: str = typer.Argument(
        ..., help="Output directory for the embeddings zarr."
    ),
    feature_extractor: list[str] = typer.Option(
        ["resnet50"],
        "--model",
        "-m",
        "--feature-extractor",
        "-f",
        help="Model(s); repeatable. (--feature-extractor/-f are aliases.)",
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help="Target µm/px. Default: the model's recommended MPP (0.5 for pathology "
        "FMs = 20×; 1.0 for scale-agnostic models). Pass a value to override.",
    ),
    source_mpp: float | None = typer.Option(
        None,
        "--source-mpp",
        help="Physical pixel size (µm/px) of the SOURCE image at level 0. ONLY needed "
        "when the OME-Zarr omits it (no axis unit declared), so the run can't tell how "
        "big a pixel is. This is the source's resolution - NOT the extraction scale "
        "(use --mpp for that). Prefer fixing the source metadata over this override.",
    ),
    patch_size: int | None = typer.Option(
        None,
        "--patch-size",
        help="Patch size in px. Default: each model's recommended size (224 for most, "
        "448 CONCH, 512 conch_v1_5, ...). Pass a value to force one size for all.",
    ),
    step: int | None = typer.Option(None, "--step", help="Stride (default = patch)."),
    reader: str = typer.Option("omezarr", "--reader"),
    segmenter: str = typer.Option("otsu", "--segmenter"),
    no_seg: bool = typer.Option(False, "--no-seg", help="Tile everything (no mask)."),
    tissue_threshold: float = typer.Option(0.1, "--tissue-threshold"),
    features_dtype: str = typer.Option("float16", "--features-dtype"),
    stain_norm: str | None = typer.Option(
        None,
        "--stain-norm",
        help="Stain-normalize each patch BEFORE embedding: macenko|reinhard|vahadane. "
        "Changes the "
        "features -- use separate output dirs for with/without-norm experiments.",
    ),
    multiplex_strategy: str | None = typer.Option(
        None,
        "--multiplex-strategy",
        help="Apply a named multiplex strategy (initially: channelwise) around each "
        "ordinary patch encoder.",
    ),
    multiplex_markers: list[str] = typer.Option(
        [],
        "--marker",
        help="Multiplex marker to include; repeat in the required order. Default: all "
        "named channels in source order (concat requires an explicit list).",
    ),
    channel_names_file: str | None = typer.Option(
        None,
        "--channel-names-file",
        help="UTF-8 .txt/.csv/.tsv with exactly one ordered name per physical C-axis "
        "position. Supplies missing labels and verifies existing labels; combine "
        "with --marker to select or order a subset.",
    ),
    multiplex_normalization: str = typer.Option(
        "percentile",
        "--multiplex-normalization",
        help="Per-marker intensity normalization (default: percentile over a "
        "deterministically selected whole-image pyramid level).",
    ),
    multiplex_percentile_low: float = typer.Option(
        1.0,
        "--multiplex-percentile-low",
        help="Lower whole-image-level percentile for per-marker normalization.",
    ),
    multiplex_percentile_high: float = typer.Option(
        99.0,
        "--multiplex-percentile-high",
        help="Upper whole-image-level percentile for per-marker normalization.",
    ),
    multiplex_normalization_max_side_px: int = typer.Option(
        2048,
        "--multiplex-normalization-max-side-px",
        help="Longest side allowed for the deterministic whole-image normalization "
        "level (default: 2048; larger may select a finer level and use more RAM).",
    ),
    multiplex_aggregation: str = typer.Option(
        "mean",
        "--multiplex-aggregation",
        help="Marker embedding aggregation: mean | concat.",
    ),
    multiplex_params: str | None = typer.Option(
        None,
        "--multiplex-params",
        help="JSON object of content parameters for a third-party multiplex strategy. "
        "The built-in channelwise strategy uses its explicit options instead.",
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto | cuda | mps | cpu (auto = best available)"
    ),
    devices: str | None = typer.Option(
        None,
        "--devices",
        help=(
            "Comma-separated devices for in-process multi-GPU (e.g. "
            "'cuda:0,cuda:1'), or 'auto' for every visible GPU. Shards this slide's "
            "patches across them and gathers features back in coord order (output "
            "identical to one device). Mainly helps very large slides - the per-device "
            "model load can outweigh the gain on small ones. Default: the single "
            "--device."
        ),
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    read_workers: int = typer.Option(
        8, "--read-workers", help="Concurrent patch-decode threads (read parallelism)."
    ),
    read_block: int = typer.Option(
        1,
        "--read-block",
        help="Read patches in N x N blocks per read instead of one-at-a-time "
        "(1 = off, the default). Bigger = fewer, larger reads: faster on remote / "
        "slow / HDD storage and on cached re-reads, but costs host RAM and over-reads "
        "non-tissue gaps -- so it helps little on fast local storage and wastes "
        "bandwidth on sparse tissue. Try 8 (local) / 16 (remote); smaller for big "
        "multiplex panels. Bit-identical to per-patch either way.",
    ),
    compile: bool = typer.Option(
        False,
        "--compile",
        help="torch.compile the model once at load (speed only; off by default).",
    ),
    amp: str = typer.Option(
        "auto",
        "--amp",
        help="auto | fp32 | bf16 | fp16 (auto = each model's card precision)",
    ),
    snap_to_level: bool = typer.Option(False, "--snap-to-level"),
    mpp_tolerance: float = typer.Option(0.001, "--mpp-tolerance"),
    allow_upsample: bool = typer.Option(False, "--allow-upsample"),
    emit_geojson: bool = typer.Option(False, "--emit-geojson"),
    emit_thumbnail: bool = typer.Option(
        False, "--emit-thumbnail", help="Also write a thumbnail PNG (+ QC overlay)."
    ),
    thumbnail_mpp: float = typer.Option(
        DEFAULT_THUMBNAIL_MPP,
        "--thumbnail-mpp",
        help="Thumbnail MPP (default = seg MPP).",
    ),
    max_px: int | None = typer.Option(
        None, "--max-px", help="Cap thumbnail longest side (overrides --thumbnail-mpp)."
    ),
    slide_encoder: list[str] = typer.Option(
        [],
        "--slide-encoder",
        "-s",
        help=(
            "Slide-level encoder(s); repeatable. Run after patch embedding using "
            "the patch features already written - no WSI re-read. "
            "Example: -f conch_v1_5 --patch-size 512 -s titan"
        ),
    ),
    qc: list[str] = typer.Option(
        [],
        "--qc",
        help="Per-patch QC producer(s) writing grids/<key>/qc/<tool>/ (e.g. grandqc). "
        'Needs the producer\'s extra, e.g. pip install "raw2features[grandqc]".',
    ),
    qc_stain_norm: str | None = typer.Option(
        None,
        "--qc-stain-norm",
        help="Normalize the QC input (macenko|reinhard|vahadane) for an out-of-domain "
        "stain (eosin-heavy H&E that reads as blur).",
    ),
    qc_artifact_mpp: str = typer.Option(
        "1.5",
        "--qc-artifact-mpp",
        help="GrandQC artifact scale in µm/px (1.0|1.5|2.0). Coarser (2.0) is less "
        "focus-sensitive; raise it if a softer-scanned cohort over-reads as blur.",
    ),
    output_zarr_format: int = typer.Option(2, "--output-zarr-format"),
    receipts_dir: str | None = typer.Option(None, "--receipts-dir"),
    config: str | None = typer.Option(
        None,
        "--config",
        help="YAML extraction plan: an 'extractions:' list of {model, mpp?, patch_px?} "
        "(same model may repeat). Overrides -m/--mpp/--patch-size.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild the store from scratch instead of adding missing models to it.",
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
) -> None:
    """Extract foundation-model patch embeddings from one OME-Zarr slide.

    Re-running with a new ``--model`` adds ``features/<model>`` to an existing
    store in place and skips models already computed (no WSI re-read for them).
    Use ``--force`` to overwrite the store instead.
    """
    validate_amp(amp)
    validate_batch_size(batch_size)
    validate_geometry(mpp=mpp, patch_size=patch_size, step=step, source_mpp=source_mpp)
    validate_positive_int(
        multiplex_normalization_max_side_px,
        "--multiplex-normalization-max-side-px",
    )
    validate_multiplex_percentiles(multiplex_percentile_low, multiplex_percentile_high)
    strategy_params = parse_json_object(multiplex_params, "--multiplex-params")
    channel_names_override = parse_channel_names_file(channel_names_file)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    from raw2features.embedders.model_registry import resolve_geometry

    # A --config extraction plan supersedes -m / --mpp / --patch-size.
    geometry_config = None
    if config:
        from raw2features.core.config import load_extractions

        geometry_config = load_extractions(config)
        models = list(dict.fromkeys(e["model"] for e in geometry_config))
    else:
        models = list(feature_extractor)

    # Resolve and preview the per-model extraction geometry (one grid per group).
    groups = resolve_geometry(models, mpp, patch_size, geometry_config)
    if not groups:
        raise typer.BadParameter("at least one model/extraction is required")
    if len(groups) > 1:
        plan = " · ".join(
            f"{','.join(g.models)} -> {g.mpp:g}/{g.patch_px}" for g in groups
        )
        typer.echo(f"geometry ({len(groups)} grids): {plan}")
    elif groups and groups[0].source != "explicit":
        g0 = groups[0]
        typer.echo(
            f"geometry: {','.join(g0.models)} -> {g0.mpp:g}/{g0.patch_px} ({g0.source})"
        )

    # RunConfig carries one concrete grid geometry for the run_slide primitive.
    # embed_slide replaces it for every resolved group, so seed it with a real group
    # rather than treating an otherwise meaningful 1.0/224 geometry as a sentinel.
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
        device=device,
        devices=devices,
        batch_size=batch_size,
        read_workers=read_workers,
        read_block=read_block,
        compile=compile,
        emit_geojson=emit_geojson,
        emit_thumbnail=emit_thumbnail,
        thumbnail_mpp=thumbnail_mpp,
        thumbnail_max_px=max_px,
        slide_encoders=list(slide_encoder),
        qc=list(qc),
        qc_stain_norm=qc_stain_norm,
        qc_artifact_mpp=qc_artifact_mpp,
        output_zarr_format=output_zarr_format,
    )
    summary = embed_slide(
        slide,
        out_dir,
        cfg,
        requested_mpp=mpp,
        requested_patch_px=patch_size,
        geometry_config=geometry_config,
        receipts_dir=receipts_dir,
        cli=sanitize_argv(sys.argv),
        force=force,
    )
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")
