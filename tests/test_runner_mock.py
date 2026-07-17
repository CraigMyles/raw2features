"""End-to-end runner test with a mock embedder (no weights, CPU)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed")
from conftest import MockEmbedder
from raw2features.core.store import open_grid
from raw2features.pipeline.runner import RunConfig, run_slide


def test_high_level_external_embedder_uses_its_spec_geometry(tmp_path, monkeypatch):
    """The documented external-instance hook must not silently use 1.0/224."""

    from raw2features.pipeline import runner

    external = MockEmbedder(dim=8, input_size=64, name="external")
    external.spec = replace(
        external.spec,
        recommended_mpp=0.75,
        recommended_patch_px=96,
    )
    seen = []

    def fake_run_slide(slide_path, out_dir, cfg, **kwargs):
        seen.append((cfg.target_mpp, cfg.patch_px, tuple(cfg.models)))
        return {"status": "complete", "grid": "external-grid"}

    monkeypatch.setattr(runner, "run_slide", fake_run_slide)
    # These fields are the concrete geometry for run_slide. The high-level API uses
    # model specs unless requested_mpp/requested_patch_px are passed explicitly.
    cfg = RunConfig(
        models=["external"],
        target_mpp=0.5,
        patch_px=64,
        no_seg=True,
        device="cpu",
        amp="fp32",
    )
    runner.embed_slide(
        str(tmp_path / "source.zarr"),
        str(tmp_path / "out"),
        cfg,
        embedders=[external],
    )

    assert seen == [(0.75, 96, ("external",))]


def test_runner_e2e_mock_no_seg(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        batch_size=8,
    )
    out_dir = str(tmp_path / "out")
    rec_dir = str(tmp_path / "rec")
    summary = run_slide(
        synthetic_ngff, out_dir, cfg, receipts_dir=rec_dir, embedders=[MockEmbedder()]
    )
    assert summary["status"] == "complete"
    n = summary["n_patches"]
    assert n == 12  # 4x3 grid over 300x200 at patch 64

    g = open_grid(summary["output_uri"])
    assert g["coords"].shape == (n, 2)
    assert g["grid_index"].shape == (n, 2)
    assert g["features"]["mock"].shape == (n, 8)
    assert g["features"]["mock"].dtype == np.float16
    header = dict(g.attrs)["raw2features"]
    assert header["patching"]["n_patches"] == n
    assert header["segmentation"]["segmenter"] == "none"
    assert header["models"]["mock"]["transform_source_url"].startswith("http")

    # rerun must skip (validate-against-output idempotency)
    again = run_slide(
        synthetic_ngff, out_dir, cfg, receipts_dir=rec_dir, embedders=[MockEmbedder()]
    )
    assert again["status"] == "skipped"


def test_runner_e2e_mock_with_otsu_and_geojson(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        segmenter="otsu",
        target_mpp=0.5,
        patch_px=64,
        tissue_threshold=0.0,  # keep all cells regardless of the synthetic content
        device="cpu",
        amp="fp32",
        emit_geojson=True,
    )
    out_dir = str(tmp_path / "out")
    summary = run_slide(synthetic_ngff, out_dir, cfg, embedders=[MockEmbedder()])
    assert summary["status"] == "complete"
    assert summary["geojson"] is not None
    g = open_grid(summary["output_uri"])
    assert "mask" in g
