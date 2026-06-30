"""``raw2features embed`` - read a slide and write patch embeddings."""

from __future__ import annotations

import os
import sys

import typer

from raw2features.pipeline.runner import RunConfig, embed_slide
from raw2features.viz import DEFAULT_THUMBNAIL_MPP


def embed(
    slide: str = typer.Argument(..., help="Path to an OME-Zarr store."),
    out_dir: str = typer.Argument(
        ..., help="Output directory for the embeddings zarr."
    ),
    feature_extractor: list[str] = typer.Option(
        ["resnet50"], "--model", "-m", "--feature-extractor", "-f",
        help="Model(s); repeatable. (--feature-extractor/-f are aliases.)"
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
            "Example: -f uni -s titan"
        ),
    ),
    qc: list[str] = typer.Option(
        [],
        "--qc",
        help="Per-patch QC producer(s) writing grids/<key>/qc/<tool>/ (e.g. grandqc). "
        "Needs the producer's extra, e.g. pip install \"raw2features[grandqc]\".",
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

    cfg = RunConfig(
        models=models,
        reader=reader,
        segmenter=segmenter,
        no_seg=no_seg,
        target_mpp=mpp if mpp is not None else 1.0,  # per-group geometry overrides this
        source_mpp=source_mpp,
        patch_px=patch_size if patch_size is not None else 224,
        step_px=step,
        tissue_threshold=tissue_threshold,
        features_dtype=features_dtype,
        stain_norm=stain_norm,
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
        cli=" ".join(sys.argv),
        force=force,
    )
    for key, value in summary.items():
        typer.echo(f"{key}: {value}")
