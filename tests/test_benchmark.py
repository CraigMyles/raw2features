"""Benchmark harness: profiler accumulation, output-equivalence, and wiring."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from conftest import MockEmbedder
from raw2features.benchmark.equivalence import compare_stores
from raw2features.benchmark.profiler import NullProfiler, Profiler
from raw2features.pipeline.runner import RunConfig, run_slide

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


# -- profiler (torch-free) -----------------------------------------------------


def test_profiler_accumulates_stages_and_bytes():
    p = Profiler()
    for _ in range(3):
        with p.stage("read"):
            pass
    with p.stage("gpu"):
        pass
    p.add_bytes(1000)
    p.add_bytes(500)
    s = p.summary(n_patches=10, wall_s=1.0)
    assert s["stages"]["read"]["calls"] == 3
    assert s["stages"]["gpu"]["calls"] == 1
    assert s["decoded_MB"] == round(1500 / 1e6, 1)
    assert s["n_patches"] == 10
    assert s["patches_per_s"] == 10.0


def test_null_profiler_is_noop():
    p = NullProfiler()
    with p.stage("read"):
        pass
    p.add_bytes(123)  # must not raise


# -- equivalence (torch-free) --------------------------------------------------


def _store(path, coords, feats: dict):
    from raw2features.core.store import GRIDS

    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    g = root.require_group(GRIDS).require_group("mpp1_px224")  # uniform grid nesting
    c = g.create_array("coords", shape=coords.shape, chunks=coords.shape, dtype="int32")
    c[:] = coords
    fg = g.create_group("features")
    for m, arr in feats.items():
        a = fg.create_array(m, shape=arr.shape, chunks=arr.shape, dtype="float16")
        a[:] = arr
    return str(path)


def test_compare_stores_identical_ok(tmp_path):
    coords = np.arange(6, dtype="int32").reshape(3, 2)
    f = np.ones((3, 4), dtype="float16")
    a = _store(tmp_path / "a.zarr", coords, {"m": f})
    b = _store(tmp_path / "b.zarr", coords, {"m": f.copy()})
    rep = compare_stores(a, b)
    assert rep["ok"] is True
    assert rep["models"]["m"]["ok"] is True


def test_compare_stores_detects_feature_diff(tmp_path):
    coords = np.arange(6, dtype="int32").reshape(3, 2)
    a = _store(tmp_path / "a.zarr", coords, {"m": np.ones((3, 4), "float16")})
    b = _store(tmp_path / "b.zarr", coords, {"m": np.full((3, 4), 2.0, "float16")})
    rep = compare_stores(a, b)
    assert rep["ok"] is False
    assert rep["models"]["m"]["ok"] is False
    assert rep["models"]["m"]["max_abs_diff"] >= 1.0


def test_compare_stores_detects_coords_diff(tmp_path):
    f = np.ones((3, 4), "float16")
    a = _store(tmp_path / "a.zarr", np.zeros((3, 2), "int32"), {"m": f})
    b = _store(tmp_path / "b.zarr", np.ones((3, 2), "int32"), {"m": f})
    rep = compare_stores(a, b)
    assert rep["ok"] is False
    assert any("coords" in i for i in rep["issues"])


# -- profiler wired through the real runner (needs torch) ----------------------


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_profiler_captures_runner_stages(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    prof = Profiler()
    summary = run_slide(
        synthetic_ngff,
        str(tmp_path / "o"),
        cfg,
        profiler=prof,
        embedders=[MockEmbedder(dim=8)],
    )
    s = prof.summary(n_patches=summary["n_patches"], wall_s=summary["elapsed_s"])
    for stage in ("read", "transform", "gpu", "write"):
        assert stage in s["stages"], f"missing stage {stage}"
    assert s["decoded_MB"] >= 0.0
    assert s["patches_per_s"] >= 0.0


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_null_profiler_default_does_not_change_output(synthetic_ngff, tmp_path):
    # Running with no profiler (production default) must behave exactly as before.
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    summary = run_slide(
        synthetic_ngff, str(tmp_path / "o"), cfg, embedders=[MockEmbedder(dim=8)]
    )
    assert summary["status"] == "complete"
    from raw2features.core.store import open_grid

    g = open_grid(summary["output_uri"])  # the sole grid
    assert np.isfinite(np.asarray(g["features"]["mock"][:])).all()
