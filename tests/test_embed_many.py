"""Warm worker (`embed-many`): sharding, per-slide equivalence, idempotent resume."""

from __future__ import annotations

import os

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import MockEmbedder, build_ngff_v04
from raw2features.cli.embed_many import _shard
from raw2features.cli.main import app
from raw2features.pipeline.runner import RunConfig, run_slide

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


# -- sharding (torch-free) -----------------------------------------------------


def test_shard_partitions_disjoint_and_complete():
    items = list(range(23))
    n = 4
    shards = [_shard(items, k, n) for k in range(n)]
    seen = [x for s in shards for x in s]
    assert sorted(seen) == items  # every item covered exactly once
    assert shards[0] == items[0::4]  # strided


# -- end-to-end (needs torch; mocks injected so no weight download) ------------


def _two_slides(tmp_path) -> str:
    d = tmp_path / "slides"
    d.mkdir()
    build_ngff_v04(str(d / "A.zarr"))
    build_ngff_v04(str(d / "B.zarr"))
    return str(d)


def _mock(cfg):  # stand-in for load_embedders (no registry / no weights)
    return [MockEmbedder(dim=8, name="mock", bias=1.0)]


_COMMON = [
    "-f",
    "mock",
    "--no-seg",
    "--mpp",
    "0.5",
    "--patch-size",
    "64",
    "--device",
    "cpu",
    "--amp",
    "fp32",
]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_many_matches_individual(tmp_path, monkeypatch):
    import raw2features.cli.embed_many as em

    monkeypatch.setattr(em, "load_embedders", _mock)
    slides, out = _two_slides(tmp_path), str(tmp_path / "out")

    r = CliRunner().invoke(app, ["embed-many", slides, out, *_COMMON])
    assert r.exit_code == 0, r.output
    assert "2 embedded" in r.output

    from raw2features.core.store import open_grid

    ga = open_grid(os.path.join(out, "A.embeddings.zarr"))  # the sole grid
    # a direct single-slide run with the same mock must produce identical features
    s = run_slide(
        os.path.join(slides, "A.zarr"),
        str(tmp_path / "ref"),
        RunConfig(
            models=["mock"],
            no_seg=True,
            target_mpp=0.5,
            patch_px=64,
            device="cpu",
            amp="fp32",
        ),
        embedders=[MockEmbedder(dim=8, name="mock", bias=1.0)],
    )
    gref = open_grid(s["output_uri"])
    assert np.array_equal(
        np.asarray(ga["features"]["mock"][:]),
        np.asarray(gref["features"]["mock"][:]),
    )


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_many_sharding_covers_all(tmp_path, monkeypatch):
    import raw2features.cli.embed_many as em

    monkeypatch.setattr(em, "load_embedders", _mock)
    slides, out = _two_slides(tmp_path), str(tmp_path / "out")
    base = ["embed-many", slides, out, *_COMMON, "--num-shards", "2"]
    r0 = CliRunner().invoke(app, [*base, "--shard-index", "0"])
    r1 = CliRunner().invoke(app, [*base, "--shard-index", "1"])
    assert r0.exit_code == 0 and r1.exit_code == 0
    assert "1 of 2 slides" in r0.output and "1 of 2 slides" in r1.output
    assert os.path.exists(os.path.join(out, "A.embeddings.zarr"))
    assert os.path.exists(os.path.join(out, "B.embeddings.zarr"))


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_many_idempotent(tmp_path, monkeypatch):
    import raw2features.cli.embed_many as em

    monkeypatch.setattr(em, "load_embedders", _mock)
    slides, out = _two_slides(tmp_path), str(tmp_path / "out")
    rcpts = str(tmp_path / "rcpts")
    base = ["embed-many", slides, out, *_COMMON, "--receipts-dir", rcpts]
    CliRunner().invoke(app, base)
    r2 = CliRunner().invoke(app, base)
    assert r2.exit_code == 0
    assert "2 skipped" in r2.output  # both already complete -> skipped on re-run


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_many_manifest_selects_subset(tmp_path, monkeypatch):
    """--manifest picks a curated subset (relative paths resolve against slide_dir)."""
    import raw2features.cli.embed_many as em

    monkeypatch.setattr(em, "load_embedders", _mock)
    slides, out = _two_slides(tmp_path), str(tmp_path / "out")  # A.zarr + B.zarr
    manifest = tmp_path / "m.csv"
    manifest.write_text("path\nA.zarr\n")  # only A, relative to slide_dir
    r = CliRunner().invoke(
        app, ["embed-many", slides, out, "--manifest", str(manifest), *_COMMON]
    )
    assert r.exit_code == 0, r.output
    assert "1 of 1 slides" in r.output
    assert os.path.exists(os.path.join(out, "A.embeddings.zarr"))
    assert not os.path.exists(os.path.join(out, "B.embeddings.zarr"))


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_many_config_writes_multiple_grids(tmp_path, monkeypatch):
    """--config extraction plan drives per-grid geometry across the cohort."""
    import zarr

    import raw2features.cli.embed_many as em
    from raw2features.core.store import grid_keys

    monkeypatch.setattr(em, "load_embedders", _mock)
    slides, out = _two_slides(tmp_path), str(tmp_path / "out")
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "extractions:\n"
        "  - {model: mock, mpp: 0.5, patch_px: 64}\n"
        "  - {model: mock, mpp: 1.0, patch_px: 64}\n"
    )
    r = CliRunner().invoke(
        app, ["embed-many", slides, out, "--config", str(plan),
              "--no-seg", "--device", "cpu", "--amp", "fp32"],
    )
    assert r.exit_code == 0, r.output
    root = zarr.open_group(os.path.join(out, "A.embeddings.zarr"), mode="r")
    assert set(grid_keys(root)) == {"mpp0.5_px64", "mpp1_px64"}
    # the explicit plan is recorded in the store for replay
    job = dict(root.attrs["raw2features"]).get("job", {})
    assert len(job.get("geometry_config", [])) == 2


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_cli_config_drives_grids(tmp_path, monkeypatch):
    """The single-slide `embed` CLI with --config builds one grid per plan entry.

    Same model at two MPPs (the ablation case) -> two grids in one store.
    """
    import zarr

    import raw2features.pipeline.runner as rn
    from raw2features.core.store import grid_keys

    monkeypatch.setattr(
        rn, "_build_embedders_on",
        lambda cfg, names, device: [MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        "extractions:\n"
        "  - {model: mock, mpp: 0.5, patch_px: 64}\n"
        "  - {model: mock, mpp: 1.0, patch_px: 64}\n"
    )
    r = CliRunner().invoke(
        app, ["embed", slide, str(tmp_path / "out"), "--config", str(plan),
              "--no-seg", "--device", "cpu", "--amp", "fp32"],
    )
    assert r.exit_code == 0, r.output
    root = zarr.open_group(str(tmp_path / "out" / "S.embeddings.zarr"), mode="r")
    assert set(grid_keys(root)) == {"mpp0.5_px64", "mpp1_px64"}
    job = dict(root.attrs["raw2features"]).get("job", {})
    assert len(job.get("geometry_config", [])) == 2


def test_with_source_mpp_applies_manifest_override():
    """A manifest row's per-slide source_mpp lands in the slide's RunConfig."""
    from raw2features.cli.embed_many import _with_source_mpp

    base = RunConfig(models=["mock"], source_mpp=None)
    assert _with_source_mpp(base, {"path": "x.zarr"}).source_mpp is None
    got = _with_source_mpp(base, {"path": "x.zarr", "source_mpp": 0.33})
    assert got.source_mpp == 0.33
    assert base.source_mpp is None  # original untouched (replace, not mutate)
