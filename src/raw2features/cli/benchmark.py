"""``raw2features benchmark`` - per-stage timing + throughput for the embed path.

Runs the *real* pipeline (the same code production uses) with a Profiler attached
and prints where the wall-time goes - model-load / segment / tile / read /
transform / GPU / write / consolidate - plus patches/s and decode MB/s. Use it to
find the bottleneck before optimizing, and ``--check`` to prove an optimization
left the embeddings numerically identical.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import tempfile

import typer

from raw2features.benchmark.equivalence import compare_stores
from raw2features.benchmark.profiler import Profiler
from raw2features.pipeline.runner import (
    RunConfig,
    load_embedders,
    run_slide,
    slide_id_from_path,
)

from ._validation import validate_amp, validate_batch_size, validate_geometry


def _print_run(slide_id: str, rep: dict, idx: int) -> None:
    typer.echo(
        f"\n[{slide_id}] run {idx}: {rep['n_patches']} patches in {rep['wall_s']}s "
        f"-> {rep['patches_per_s']} patches/s, decode {rep['decode_MB_per_s']} MB/s"
    )
    typer.echo(f"  {'stage':<12}{'sec':>10}{'%wall':>8}{'calls':>8}")
    for name, s in rep["stages"].items():
        typer.echo(
            f"  {name:<12}{s['seconds']:>10.3f}{s['pct_wall']:>7.1f}%{s['calls']:>8}"
        )
    typer.echo(f"  {'unaccounted':<12}{rep['unaccounted_s']:>10.3f}")


def benchmark(
    slides: list[str] = typer.Argument(..., help="OME-Zarr slide path(s)."),
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
        help="Target µm/px; default = the model's recommended scale (like embed), so "
        "the benchmark profiles at the same scale production runs at.",
    ),
    patch_size: int = typer.Option(224, "--patch-size"),
    no_seg: bool = typer.Option(False, "--no-seg"),
    device: str = typer.Option(
        "auto", "--device", help="auto | cuda | mps | cpu (auto = best available)"
    ),
    devices: str | None = typer.Option(
        None,
        "--devices",
        help=(
            "Comma-separated devices for patch-parallel multi-GPU (e.g. "
            "'cuda:0,cuda:1'), or 'auto' for every visible GPU, so single-slide "
            "latency can be measured. Wall-time and "
            "patches/s stay accurate; the per-stage breakdown is reported only for "
            "the single-device path (concurrent-worker stage timings aren't "
            "meaningful). Default: the single --device."
        ),
    ),
    amp: str = typer.Option(
        "auto", "--amp", help="auto | fp32 | bf16 | fp16 (auto = card precision)"
    ),
    batch_size: int = typer.Option(256, "--batch-size"),
    read_workers: int = typer.Option(
        8, "--read-workers", help="Concurrent patch-decode threads (read parallelism)."
    ),
    read_block: int = typer.Option(
        1,
        "--read-block",
        help="Read patches in NxN blocks (1=off). Bigger helps "
        "remote/slow storage, costs RAM; bit-identical. Try 8 local / 16 remote.",
    ),
    compile: bool = typer.Option(
        False,
        "--compile",
        help="torch.compile the model once at load (run 0 pays the warmup).",
    ),
    repeat: int = typer.Option(1, "--repeat", help="Runs per slide (run 0 is cold)."),
    out_dir: str | None = typer.Option(
        None, "--out", help="Keep outputs here (default: a temp dir, deleted after)."
    ),
    check: str | None = typer.Option(
        None, "--check", help="Baseline dir to compare outputs against (equivalence)."
    ),
    json_out: str | None = typer.Option(
        None, "--json", help="Write all run summaries to this JSON file."
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
) -> None:
    """Profile the embed pipeline and report per-stage timing + throughput."""
    validate_amp(amp)
    validate_batch_size(batch_size)
    validate_geometry(mpp=mpp, patch_size=patch_size)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    from raw2features.embedders.model_registry import resolve_target_mpp

    target_mpp, mpp_source = resolve_target_mpp(list(feature_extractor), mpp)
    if mpp_source != "explicit":
        typer.echo(f"target_mpp: {target_mpp} ({mpp_source})")

    tmp = out_dir or tempfile.mkdtemp(prefix="r2f-bench-")
    cuda = device.startswith("cuda")
    cfg = RunConfig(
        models=list(feature_extractor),
        no_seg=no_seg,
        target_mpp=target_mpp,
        patch_px=patch_size,
        device=device,
        devices=devices,
        amp=amp,
        batch_size=batch_size,
        read_workers=read_workers,
        read_block=read_block,
        compile=compile,
    )
    # Build (and, with --compile, torch.compile) the models ONCE and reuse them
    # across reps/slides -- so the compile warmup is paid on run 0 only and runs
    # 1+ measure the steady state, matching the warm-worker production path. Under
    # --devices (multi-GPU patch-parallel) the per-device copies are built inside
    # run_slide, so don't pre-build a single-device set here.
    embedders = load_embedders(cfg) if len(cfg.device_list()) == 1 else None
    all_summaries: list[dict] = []
    try:
        for slide in slides:
            sid = slide_id_from_path(slide)
            reps = []
            for i in range(repeat):
                prof = Profiler(cuda=cuda)
                # force=True so every run re-embeds (no skip) for a fair timing.
                summary = run_slide(
                    slide, tmp, cfg, profiler=prof, force=True, embedders=embedders
                )
                rep = prof.summary(
                    n_patches=summary["n_patches"], wall_s=summary["elapsed_s"]
                )
                rep["slide_id"], rep["run"] = sid, i
                reps.append(rep)
                _print_run(sid, rep, i)
            if repeat > 1:
                pps = [r["patches_per_s"] for r in reps]
                typer.echo(
                    f"[{sid}] median {statistics.median(pps)} patches/s over "
                    f"{repeat} runs (run 0 cold)"
                )
            all_summaries.extend(reps)

            if check:
                base = os.path.join(check, f"{sid}.embeddings.zarr")
                got = os.path.join(tmp, f"{sid}.embeddings.zarr")
                report = compare_stores(base, got)
                tag = "OK" if report["ok"] else "MISMATCH"
                detail = report["models"] if report["models"] else report["issues"]
                typer.echo(f"[{sid}] equivalence vs baseline: {tag}  {detail}")
                if not report["ok"]:
                    raise typer.Exit(1)

        if json_out:
            with open(json_out, "w") as fh:
                json.dump(all_summaries, fh, indent=2)
            typer.echo(f"\nwrote {json_out}")
    finally:
        if out_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
