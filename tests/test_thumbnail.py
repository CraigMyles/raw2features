"""Tests for thumbnails + QC overlays (raw2features.viz + the CLI/runner wiring)."""

from __future__ import annotations

import json
import os
import stat

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
    save_png,
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
def test_save_png_new_file_honours_umask(tmp_path):
    path = tmp_path / "new.png"
    old_umask = os.umask(0o022)
    try:
        save_png(np.zeros((2, 2, 3), dtype=np.uint8), str(path))
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_save_png_replacement_preserves_mode(tmp_path):
    path = tmp_path / "existing.png"
    path.write_bytes(b"old")
    path.chmod(0o664)
    save_png(np.zeros((2, 2, 3), dtype=np.uint8), str(path))
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_save_png_accepts_near_name_max_destination(tmp_path):
    suffix = ".png"
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    path = tmp_path / ("p" * (name_max - len(suffix) - 5) + suffix)
    save_png(np.zeros((2, 2, 3), dtype=np.uint8), str(path))
    assert path.is_file()


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


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_missing_thumbnail_is_produced_without_reembedding_or_overwrite(
    synthetic_ngff, tmp_path, monkeypatch
):
    import raw2features.pipeline.runner as rn
    from raw2features.core.store import open_grid

    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    common = dict(
        models=["mock"],
        segmenter="otsu",
        target_mpp=0.5,
        patch_px=64,
        tissue_threshold=0.0,
        device="cpu",
        amp="fp32",
    )
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common),
        embedders=[MockEmbedder(bias=1.0)],
        receipts_dir=receipts,
    )
    monkeypatch.setattr(
        rn,
        "_embed_patches",
        lambda *args, **kwargs: pytest.fail("thumbnail run re-embedded patches"),
    )

    produced = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, emit_thumbnail=True),
        embedders=[MockEmbedder(bias=1.0)],
        receipts_dir=receipts,
    )
    assert produced["status"] == "complete"
    plain = os.path.join(out, "synthetic.thumbnail.png")
    overlay = os.path.join(out, "synthetic.thumbnail.overlay.png")
    assert os.path.isfile(plain) and os.path.isfile(overlay)
    with open(plain, "rb") as fh:
        original_plain = fh.read()

    root = zarr.open_group(
        produced["output_uri"].removeprefix("file://"),
        mode="r",
        use_consolidated=False,
    )
    root_meta = dict(root.attrs["raw2features"])["thumbnail"]
    grid_meta = dict(open_grid(root).attrs["raw2features"])["thumbnail"]
    assert root_meta == grid_meta

    os.unlink(overlay)
    repaired = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, emit_thumbnail=True, thumbnail_max_px=25),
        embedders=[MockEmbedder(bias=1.0)],
        receipts_dir=receipts,
    )
    assert repaired["status"] == "complete"
    assert os.path.isfile(overlay)
    with open(plain, "rb") as fh:
        assert fh.read() == original_plain
    repaired_root = zarr.open_group(
        repaired["output_uri"].removeprefix("file://"),
        mode="r",
        use_consolidated=False,
    )
    repaired_meta = dict(open_grid(repaired_root).attrs["raw2features"])["thumbnail"]
    assert repaired_meta["max_px"] is None

    again = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, emit_thumbnail=True),
        embedders=[MockEmbedder(bias=1.0)],
        receipts_dir=receipts,
    )
    assert again["status"] == "skipped"


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_missing_thumbnail_is_produced_during_model_append(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    common = dict(
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mockA")],
    )
    appended = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockB"], emit_thumbnail=True, **common),
        embedders=[MockEmbedder(dim=5, input_size=64, name="mockB")],
    )

    assert appended["models_added"] == ["mockB"]
    assert os.path.isfile(os.path.join(out, "synthetic.thumbnail.png"))
    assert os.path.isfile(os.path.join(out, "synthetic.thumbnail.overlay.png"))


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_grid_dependent_sidecars_are_namespaced_on_label_collision(
    synthetic_ngff, tmp_path
):
    from raw2features.core.store import open_grid

    out = str(tmp_path / "out")
    common = dict(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        emit_thumbnail=True,
        emit_geojson=True,
    )
    first = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    primary_overlay = os.path.join(out, "synthetic.thumbnail.overlay.png")
    primary_geojson = os.path.join(out, "synthetic.patches.geojson")
    with open(primary_overlay, "rb") as fh:
        primary_overlay_before = fh.read()
    with open(primary_geojson, "rb") as fh:
        primary_geojson_before = fh.read()

    second_cfg = RunConfig(**common, step_px=32)
    second = run_slide(
        synthetic_ngff,
        out,
        second_cfg,
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    key = f"mpp0.5_px64_{second_cfg.grid_hash()[:8]}"
    secondary_overlay = os.path.join(
        out, f"synthetic.{key}.thumbnail.overlay.png"
    )
    secondary_geojson = os.path.join(out, f"synthetic.{key}.patches.geojson")

    assert second["grid"] == key
    assert second["thumbnail"]["overlay"] == os.path.basename(secondary_overlay)
    assert second["geojson"] == secondary_geojson
    assert os.path.isfile(secondary_overlay) and os.path.isfile(secondary_geojson)
    with open(primary_overlay, "rb") as fh:
        assert fh.read() == primary_overlay_before
    with open(primary_geojson, "rb") as fh:
        assert fh.read() == primary_geojson_before

    root = zarr.open_group(
        first["output_uri"].removeprefix("file://"),
        mode="r",
        use_consolidated=False,
    )
    primary = open_grid(root, "mpp0.5_px64")
    secondary = open_grid(root, key)
    assert dict(root.attrs["raw2features"])["thumbnail"]["overlay"] == (
        "synthetic.thumbnail.overlay.png"
    )
    assert dict(primary.attrs["raw2features"])["thumbnail"]["overlay"] == (
        "synthetic.thumbnail.overlay.png"
    )
    assert dict(secondary.attrs["raw2features"])["thumbnail"]["overlay"] == (
        os.path.basename(secondary_overlay)
    )
    with open(secondary_geojson) as fh:
        feature_collection = json.load(fh)
    assert len(feature_collection["features"]) == secondary["coords"].shape[0]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_force_rebuild_regenerates_grid_sidecars(synthetic_ngff, tmp_path):
    from raw2features.core.store import open_grid

    out = str(tmp_path / "out")
    common = dict(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        emit_thumbnail=True,
        emit_geojson=True,
    )
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    overlay = os.path.join(out, "synthetic.thumbnail.overlay.png")
    geojson = os.path.join(out, "synthetic.patches.geojson")
    with open(overlay, "rb") as fh:
        overlay_before = fh.read()
    with open(geojson) as fh:
        count_before = len(json.load(fh)["features"])

    rebuilt = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, step_px=32),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
        force=True,
    )
    group = open_grid(rebuilt["output_uri"])
    with open(overlay, "rb") as fh:
        assert fh.read() != overlay_before
    with open(geojson) as fh:
        count_after = len(json.load(fh)["features"])
    assert count_after == group["coords"].shape[0]
    assert count_after != count_before


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_aux_thumbnail_reuses_effective_stored_segmenter(
    synthetic_multiplex_ngff, tmp_path, monkeypatch
):
    from dataclasses import replace

    import raw2features.pipeline.runner as runner
    from raw2features.core.store import open_grid

    out = str(tmp_path / "out")
    cfg = RunConfig(
        models=["mock"],
        segmenter="otsu",
        target_mpp=0.5,
        patch_px=64,
        tissue_threshold=0.0,
        device="cpu",
        amp="fp32",
    )
    first = run_slide(
        synthetic_multiplex_ngff,
        out,
        cfg,
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    group = open_grid(first["output_uri"], mode="r+")
    header = dict(group.attrs["raw2features"])
    header["segmentation"] = {"segmenter": "nuclear"}
    group.attrs["raw2features"] = header

    seen = []

    def fake_segment(reader, requested_cfg, segmenter_name=None):
        seen.append(segmenter_name)
        return None, {"segmenter": segmenter_name or requested_cfg.segmenter}

    monkeypatch.setattr(runner, "_segment", fake_segment)
    monkeypatch.setattr(
        runner,
        "_embed_patches",
        lambda *args, **kwargs: pytest.fail("aux-only thumbnail re-embedded patches"),
    )
    produced = run_slide(
        synthetic_multiplex_ngff,
        out,
        replace(cfg, emit_thumbnail=True),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )

    assert produced["status"] == "complete"
    assert seen == ["nuclear"]
