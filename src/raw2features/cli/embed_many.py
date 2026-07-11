"""``raw2features embed-many`` - warm worker: load models once, embed many slides.

The one-slide-per-SLURM-task path pays the model-load cost (and Python/CUDA
start-up) on every slide. This command builds the embedders **once** and loops the
per-slide pipeline over a shard of slides, so at cohort scale that fixed cost is
paid once per task instead of once per slide.

Sharding: the SLURM array gives each task ``--shard-index K --num-shards N``; the
task processes ``slides[K::N]`` of the deterministically-sorted slide list. Per-slide
receipts keep it idempotent and resumable - re-submitting the array re-runs only the
missing/failed slides, exactly like the one-slide-per-task path. A slide that errors
is recorded in its receipt and skipped so it can't take down the rest of the shard.
"""

from __future__ import annotations

import glob as globmod
import os
import sys
import time

import typer

from raw2features.pipeline.runner import (
    RunConfig,
    embed_slide,
    load_embedders,
    slide_id_from_path,
)


def _shard(items: list, index: int, num: int) -> list:
    """This task's strided subset. Across ``index in range(num)`` the strided
    subsets partition *items* exactly (disjoint, covering), and load-balance better
    than contiguous blocks when slides vary in size."""
    return items[index::num]


def embed_many(
    slide_dir: str = typer.Argument(..., help="Directory holding the slide stores."),
    out_dir: str = typer.Argument(..., help="Output directory for *.embeddings.zarr."),
    glob: str = typer.Option(
        "*.zarr", "--glob", help="Top-level glob for slides (never recurses)."
    ),
    feature_extractor: list[str] = typer.Option(
        ["resnet50"], "--model", "-m", "--feature-extractor", "-f",
        help="Model(s); repeatable. (--feature-extractor/-f are aliases.)"
    ),
    mpp: float | None = typer.Option(
        None,
        "--mpp",
        help="Target µm/px. Default: the models' recommended MPP (0.5 for pathology "
        "FMs = 20×; 1.0 for scale-agnostic). Pass a value to override.",
    ),
    source_mpp: float | None = typer.Option(
        None,
        "--source-mpp",
        help="Physical pixel size (µm/px) of the SOURCE images at level 0. ONLY needed "
        "for sources whose OME-Zarr omits it (no axis unit). Applies to EVERY slide in "
        "the run, so use it only for a homogeneous batch; mixed-calibration cohorts "
        "should fix their metadata. Not the extraction scale - that's --mpp.",
    ),
    patch_size: int | None = typer.Option(
        None, "--patch-size", help="Patch size px (default: model's recommended)."
    ),
    step: int | None = typer.Option(None, "--step"),
    reader: str = typer.Option("omezarr", "--reader"),
    segmenter: str = typer.Option("otsu", "--segmenter"),
    no_seg: bool = typer.Option(False, "--no-seg"),
    tissue_threshold: float = typer.Option(0.1, "--tissue-threshold"),
    features_dtype: str = typer.Option("float16", "--features-dtype"),
    stain_norm: str | None = typer.Option(
        None, "--stain-norm",
        help="Stain-normalize each patch before embedding (macenko|reinhard|vahadane).",
    ),
    slide_encoder: list[str] = typer.Option(
        [], "--slide-encoder", "-s",
        help="Slide-level encoder(s) to run after patch embedding (e.g. titan).",
    ),
    qc: list[str] = typer.Option(
        [], "--qc", help="Per-patch QC producer(s) writing qc/<tool>/ (e.g. grandqc)."
    ),
    qc_stain_norm: str | None = typer.Option(
        None, "--qc-stain-norm",
        help="Normalize the QC input first (macenko|reinhard|vahadane).",
    ),
    qc_artifact_mpp: str = typer.Option(
        "1.5", "--qc-artifact-mpp",
        help="GrandQC artifact µm/px (1.0|1.5|2.0); coarser = less focus-sensitive.",
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto | cuda | mps | cpu (auto = best available)"
    ),
    devices: str | None = typer.Option(
        None,
        "--devices",
        help=(
            "Comma-separated devices for in-process multi-GPU (e.g. "
            "'cuda:0,cuda:1'), or 'auto' for every visible GPU. Distributes this "
            "shard's slides across them via a shared queue (one model copy + worker "
            "per device, faster GPUs pull more); each slide is fully embedded on one "
            "device, so per-slide output is identical to single-device. Default: "
            "--device. Useful on a multi-GPU box without a scheduler (a SLURM array "
            "already parallelises one GPU per task)."
        ),
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    amp: str = typer.Option(
        "auto",
        "--amp",
        help="auto | fp32 | bf16 | fp16 (auto = each model's card precision)",
    ),
    read_workers: int = typer.Option(
        8, "--read-workers",
        help="Concurrent patch-decode threads (16 suits a shared/parallel FS).",
    ),
    read_block: int = typer.Option(
        1,
        "--read-block",
        help="Read patches in N x N blocks per read instead of one-at-a-time "
        "(1 = off, the default). Bigger = fewer, larger reads: faster on remote / "
        "slow / HDD storage, but costs host RAM and over-reads non-tissue gaps -- so "
        "it helps little on fast local storage and wastes bandwidth on sparse tissue. "
        "Try 8 (local) / 16 (remote); smaller for big multiplex panels. Bit-identical.",
    ),
    compile: bool = typer.Option(
        False,
        "--compile",
        help="torch.compile the models once per worker (speed only; off by default).",
    ),
    snap_to_level: bool = typer.Option(False, "--snap-to-level"),
    mpp_tolerance: float = typer.Option(0.001, "--mpp-tolerance"),
    allow_upsample: bool = typer.Option(False, "--allow-upsample"),
    shard_index: int = typer.Option(
        0, "--shard-index", help="This task's shard (0-based)."
    ),
    num_shards: int = typer.Option(
        1, "--num-shards", help="Total shards = array size."
    ),
    receipts_dir: str | None = typer.Option(None, "--receipts-dir"),
    force: bool = typer.Option(False, "--force"),
    config: str | None = typer.Option(
        None, "--config",
        help="YAML extraction plan ('extractions:' list of {model, mpp?, patch_px?}).",
    ),
    manifest: str | None = typer.Option(
        None, "--manifest",
        help="CSV of slides (path[,source_mpp]) instead of globbing slide_dir; "
        "relative paths resolve against slide_dir.",
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
) -> None:
    """Warm worker: build the models once and embed this shard's slides."""
    from raw2features.slide_embedders.model_registry import (
        validate_slide_encoder_names,
    )

    try:
        validate_slide_encoder_names(slide_encoder)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        typer.echo(f"Error: invalid shard {shard_index}/{num_shards}", err=True)
        raise typer.Exit(2)

    # Deterministic, locale-independent ordering (Python sorts by code point, like
    # LC_COLLATE=C) so a given (shard_index, num_shards) always maps to the same
    # slides regardless of node.
    # Input set: a manifest (curated paths + optional per-slide source_mpp) or the glob.
    if manifest:
        from raw2features.core.config import load_manifest

        all_rows = load_manifest(manifest)
        for r in all_rows:
            if not os.path.isabs(r["path"]):
                r["path"] = os.path.join(slide_dir, r["path"])
        all_rows = sorted(all_rows, key=lambda r: r["path"])
    else:
        all_rows = [
            {"path": p} for p in sorted(globmod.glob(os.path.join(slide_dir, glob)))
        ]
    if not all_rows:
        where = manifest or os.path.join(slide_dir, glob)
        typer.echo(f"Error: no slides found ({where})", err=True)
        raise typer.Exit(1)
    shard = _shard(all_rows, shard_index, num_shards)
    typer.echo(
        f"shard {shard_index}/{num_shards}: {len(shard)} of {len(all_rows)} slides"
    )
    if not shard:
        return

    # A --config extraction plan supersedes -m / --mpp / --patch-size geometry.
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
        source_mpp=source_mpp,
        patch_px=patch_size if patch_size is not None else 224,
        step_px=step,
        tissue_threshold=tissue_threshold,
        features_dtype=features_dtype,
        stain_norm=stain_norm,
        slide_encoders=list(slide_encoder),
        qc=list(qc),
        qc_stain_norm=qc_stain_norm,
        qc_artifact_mpp=qc_artifact_mpp,
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
    )
    cli = " ".join(sys.argv)
    device_list = cfg.device_list()

    t0 = time.time()
    if len(device_list) > 1:
        done, skipped, failed = _embed_shard_parallel(
            shard, out_dir, cfg, device_list, receipts_dir, cli, force,
            mpp, patch_size, geometry_config,
        )
    else:
        done, skipped, failed = _embed_shard_serial(
            shard, out_dir, cfg, receipts_dir, cli, force,
            mpp, patch_size, geometry_config,
        )
    typer.echo(
        f"shard {shard_index}/{num_shards} done: {done} embedded, {skipped} skipped, "
        f"{failed} failed in {time.time() - t0:.0f}s"
    )
    if failed:
        raise typer.Exit(1)


def _classify(summary: dict) -> str:
    """'skipped' | 'done' for the running tally (anything not skipped counts done)."""
    return "skipped" if summary["status"] == "skipped" else "done"


def _with_source_mpp(cfg: RunConfig, row: dict) -> RunConfig:
    """Apply a manifest row's per-slide ``source_mpp`` override (else cfg unchanged)."""
    if "source_mpp" in row:
        import dataclasses

        return dataclasses.replace(cfg, source_mpp=row["source_mpp"])
    return cfg


def _embed_shard_serial(
    shard: list[dict],
    out_dir: str,
    cfg: RunConfig,
    receipts_dir: str | None,
    cli: str,
    force: bool,
    requested_mpp: float | None,
    requested_patch_px: int | None,
    requested_config: list[dict] | None,
) -> tuple[int, int, int]:
    """Single-device warm worker (the original path): build the models ONCE and
    embed the shard serially, reusing the one embedder set across every slide."""
    embedders = load_embedders(cfg)  # the whole point: pay model-load once per task
    done = skipped = failed = 0
    for i, row in enumerate(shard, 1):
        slide = row["path"]
        sid = slide_id_from_path(slide)
        try:
            summary = embed_slide(
                slide,
                out_dir,
                _with_source_mpp(cfg, row),
                requested_mpp=requested_mpp,
                requested_patch_px=requested_patch_px,
                geometry_config=requested_config,
                receipts_dir=receipts_dir,
                cli=cli,
                embedders=embedders,
                force=force,
            )
            if _classify(summary) == "skipped":
                skipped += 1
            else:
                done += 1
            typer.echo(f"[{i}/{len(shard)}] {sid}: {summary['status']}")
        except Exception as exc:  # noqa: BLE001 - record + continue; one bad slide
            failed += 1  # must not abort the rest of the shard (receipt logs it)
            typer.echo(
                f"[{i}/{len(shard)}] {sid}: FAILED {type(exc).__name__}: {exc}",
                err=True,
            )
    return done, skipped, failed


def _embed_shard_parallel(
    shard: list[dict],
    out_dir: str,
    cfg: RunConfig,
    device_list: list[str],
    receipts_dir: str | None,
    cli: str,
    force: bool,
    requested_mpp: float | None,
    requested_patch_px: int | None,
    requested_config: list[dict] | None,
) -> tuple[int, int, int]:
    """Slide-parallel warm worker (throughput mode for a multi-GPU box).

    One worker thread per device, each building its OWN model copies on that device
    (``load_embedders(cfg, device)``) and pulling slide paths off a shared queue, so
    slides are distributed across devices and embedded concurrently. Each slide is
    fully processed on a single device via the unchanged single-device ``run_slide``,
    so its output is byte-identical to a one-GPU run -- this only changes *which*
    device a slide lands on and that several run at once. Per-slide receipts keep it
    idempotent; a slide that errors is recorded and skipped (it can't take down the
    rest of the shard). Distinct slides write distinct ``*.embeddings.zarr`` stores,
    so the concurrent writes never collide.
    """
    import dataclasses
    import queue
    import threading

    work: queue.Queue = queue.Queue()
    for row in shard:
        work.put(row)
    n = len(shard)
    tally = {"done": 0, "skipped": 0, "failed": 0}
    seq = {"i": 0}
    lock = threading.Lock()  # guards the tally, the progress counter, and echo

    def _worker(device: str) -> None:
        # Build this device's model copies once and reuse across the slides it pulls
        # (warm worker, per device). Done inside the thread so the loads run in
        # parallel and each lands on its own device.
        embedders = load_embedders(cfg, device)
        # Each slide is embedded on THIS one device: pin device + drop --devices so
        # run_slide takes the plain single-device path (slide-parallel, not nested
        # patch-parallel). Output is then identical to a one-GPU run of the slide.
        slide_cfg = dataclasses.replace(cfg, device=device, devices=None)
        while True:
            try:
                row = work.get_nowait()
            except queue.Empty:
                return
            slide = row["path"]
            sid = slide_id_from_path(slide)
            try:
                summary = embed_slide(
                    slide,
                    out_dir,
                    _with_source_mpp(slide_cfg, row),
                    requested_mpp=requested_mpp,
                    requested_patch_px=requested_patch_px,
                    geometry_config=requested_config,
                    receipts_dir=receipts_dir,
                    cli=cli,
                    embedders=embedders,
                    force=force,
                )
                with lock:
                    seq["i"] += 1
                    tally[_classify(summary)] += 1
                    typer.echo(
                        f"[{seq['i']}/{n}] {sid}: {summary['status']} ({device})"
                    )
            except Exception as exc:  # noqa: BLE001 - record + continue; one bad slide
                with lock:
                    seq["i"] += 1
                    tally["failed"] += 1
                    typer.echo(
                        f"[{seq['i']}/{n}] {sid}: FAILED {type(exc).__name__}: {exc}"
                        f" ({device})",
                        err=True,
                    )

    threads = [
        threading.Thread(target=_worker, args=(d,), name=f"r2f-slide-{d}")
        for d in device_list
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return tally["done"], tally["skipped"], tally["failed"]
