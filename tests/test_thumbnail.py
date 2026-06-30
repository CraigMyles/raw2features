"""Tests for thumbnails + QC overlays (raw2features.viz + the CLI/runner wiring)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from conftest import MockEmbedder
from raw2features.cli.main import app
from raw2features.pipeline.runner import RunConfig, run_slide
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.base import TissueMask
from raw2features.viz import (
    Thumbnail,
    render_overlay,
    render_thumbnail,
    write_thumbnails,
)

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


# -- render_thumbnail -------------------------------------------------------
def test_render_thumbnail_picks_level_by_mpp(synthetic_ngff):
    # synthetic: level mpps 0.5/1.0/2.0; mpp=8 -> nearest is the coarsest (level 2)
    with OmeZarrReader(synthetic_ngff) as r:
        thumb = render_thumbnail(r, mpp=8.0)
    assert thumb.level == 2
    assert thumb.downsample == 4.0
    assert thumb.image.shape == (50, 75, 3)  # level-2 (H, W)
    assert thumb.image.dtype == np.uint8


def test_render_thumbnail_max_px_caps_longest_side(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        thumb = render_thumbnail(r, max_px=100)
    assert max(thumb.image.shape[:2]) == 100  # downscaled to the cap
    assert thumb.downsample > 2.0  # coarser than the level it was read from


# -- render_overlay ---------------------------------------------------------
def test_render_overlay_tints_mask_and_draws_patches():
    base = np.full((20, 30, 3), 100, dtype=np.uint8)
    thumb = Thumbnail(image=base, level=0, downsample=4.0)
    tissue = TissueMask(
        mask=np.ones((20, 30), dtype=np.float32), level=0, downsample=4.0
    )
    coords = np.array([[0, 0]], dtype=np.int32)  # level-0 top-left
    out = render_overlay(thumb, tissue=tissue, coords=coords, level0_patch=40)
    assert out.shape == base.shape
    assert out.dtype == np.uint8
    assert (out != base).any()  # mask tint changed pixels
    assert not np.shares_memory(out, base)  # operated on a copy


# -- write_thumbnails -------------------------------------------------------
def test_write_thumbnails_creates_files_and_meta(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    coords = np.array([[0, 0]], dtype=np.int32)
    with OmeZarrReader(synthetic_ngff) as r:
        meta = write_thumbnails(
            r, out, "syn", coords=coords, level0_patch=40, overlay=True
        )
    assert os.path.exists(os.path.join(out, "syn.thumbnail.png"))
    assert os.path.exists(os.path.join(out, "syn.thumbnail.overlay.png"))
    assert meta["plain"] == "syn.thumbnail.png"
    assert meta["overlay"] == "syn.thumbnail.overlay.png"
    assert meta["mpp"] == 8.0 and meta["max_px"] is None
    assert meta["size_wh"][0] > 0 and meta["size_wh"][1] > 0


# -- standalone CLI ---------------------------------------------------------
def test_cli_thumbnail_overlay(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    result = CliRunner().invoke(
        app,
        [
            "thumbnail",
            synthetic_ngff,
            out,
            "--overlay",
            "--mpp",
            "0.5",
            "--patch-size",
            "64",
            "--tissue-threshold",
            "0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert os.path.exists(os.path.join(out, "synthetic.thumbnail.png"))
    assert os.path.exists(os.path.join(out, "synthetic.thumbnail.overlay.png"))


# -- embed --emit-thumbnail (runner integration) ----------------------------
@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_emit_thumbnail_writes_files_and_header(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        segmenter="otsu",
        target_mpp=0.5,
        patch_px=64,
        tissue_threshold=0.0,
        device="cpu",
        amp="fp32",
        emit_thumbnail=True,
    )
    out = str(tmp_path / "out")
    summary = run_slide(synthetic_ngff, out, cfg, embedders=[MockEmbedder(bias=1.0)])
    assert summary["status"] == "complete"
    assert summary["thumbnail"] is not None
    assert os.path.exists(os.path.join(out, "synthetic.thumbnail.png"))
    assert os.path.exists(os.path.join(out, "synthetic.thumbnail.overlay.png"))
    g = zarr.open_group(summary["output_uri"].removeprefix("file://"), mode="r")
    header = dict(g.attrs)["raw2features"]
    assert header["thumbnail"] is not None
    assert header["thumbnail"]["plain"] == "synthetic.thumbnail.png"
