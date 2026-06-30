"""Dev-data-guarded end-to-end test on a real OME-Zarr slide (marked ``slow``).

Runs the full ``omezarr -> otsu -> grid -> embedder -> zarr`` path on a real
slide in ``dev-data/`` and asserts the exact-MPP behaviour and the spatial-
provenance invariants (every embedding is losslessly relocatable to its level-0
WSI region). Skipped unless a slide is present and CUDA is available.

The automated path uses the open, ungated ``resnet50`` so it needs no HF token;
the gated models (``uni``, ``uni2_h``) and ``dinov2`` are validated manually on
the dev slide. Point the test at a different slide / model set
with ``R2F_TEST_SLIDE`` and ``R2F_TEST_MODELS`` (space-separated).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
zarr = pytest.importorskip("zarr")

from raw2features.embedders.model_registry import get_spec  # noqa: E402
from raw2features.pipeline.runner import RunConfig, run_slide  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _find_dev_slide() -> str | None:
    override = os.environ.get("R2F_TEST_SLIDE")
    if override:
        return override if Path(override).is_dir() else None
    dev = _REPO_ROOT / "dev-data"
    if not dev.is_dir():
        return None
    cands = sorted(p for p in dev.glob("*.zarr") if p.is_dir())
    return str(cands[0]) if cands else None


_SLIDE = _find_dev_slide()
_MODELS = os.environ.get("R2F_TEST_MODELS", "resnet50").split()

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(_SLIDE is None, reason="no dev-data/*.zarr slide present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]


def test_real_slide_e2e_exact_mpp_and_provenance(tmp_path):
    cfg = RunConfig(
        models=_MODELS,
        target_mpp=1.0,
        patch_px=224,
        device="cuda",
        amp="bf16",
        emit_geojson=True,
    )
    out_dir = str(tmp_path / "out")
    rec_dir = str(tmp_path / "rec")
    summary = run_slide(_SLIDE, out_dir, cfg, receipts_dir=rec_dir, cli="pytest")

    assert summary["status"] == "complete"
    n = summary["n_patches"]
    assert n > 0

    g = zarr.open_group(summary["output_uri"].removeprefix("file://"), mode="r")
    header = dict(g.attrs)["raw2features"]

    # exact MPP: achieved is target (resampled) or a native level within tolerance
    achieved = header["patching"]["achieved_mpp"]
    assert abs(achieved - cfg.target_mpp) <= cfg.target_mpp * cfg.mpp_tolerance + 1e-6
    # we only ever downsample, never upscale -> read at native px, never above patch
    assert header["patching"]["read_px"] >= cfg.patch_px

    coords = np.asarray(g["coords"])
    grid_index = np.asarray(g["grid_index"])
    assert coords.shape == (n, 2)
    assert grid_index.shape == (n, 2)
    # 1:1 spatial locatability: each row is a distinct, in-bounds level-0 region
    assert len({tuple(r) for r in coords.tolist()}) == n
    level0_patch = header["patching"]["level0_patch"]
    width, height = header["source"]["level_dimensions"][0]
    assert (coords[:, 0] >= 0).all() and (coords[:, 1] >= 0).all()
    assert (coords[:, 0] + level0_patch <= width).all()
    assert (coords[:, 1] + level0_patch <= height).all()

    # every requested model: (n, dim) float16, finite, aligned 1:1 with coords
    for model in _MODELS:
        feat = g["features"][model]
        assert feat.shape == (n, get_spec(model).embedding_dim)
        assert feat.dtype == np.float16
        sample = np.asarray(feat[: min(256, n)]).astype("float32")
        assert np.isfinite(sample).all()
        assert header["models"][model]["transform_source_url"].startswith("http")

    # geojson written for QuPath relocation
    assert summary["geojson"] is not None
    assert Path(summary["geojson"]).exists()

    # rerun must skip (validate-against-output idempotency)
    again = run_slide(_SLIDE, out_dir, cfg, receipts_dir=rec_dir, cli="pytest")
    assert again["status"] == "skipped"
