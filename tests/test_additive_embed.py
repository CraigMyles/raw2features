"""Additive, per-model patch embedding.

Re-running ``embed`` with a new model adds ``features/<model>`` to an existing
store in place, skips models already present and valid, and never clobbers the
existing arrays or coords. ``force`` rebuilds from scratch. Resume is therefore
per-(slide x model), not whole-slide.

The resume/inspection logic is exercised without torch (stores are built directly
with zarr); the end-to-end embed paths are gated on torch like the other runner
tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from conftest import MockEmbedder, build_ngff_v04
from raw2features.core.store import GRIDS, open_grid
from raw2features.pipeline.receipt import validate_model
from raw2features.pipeline.runner import (
    RunConfig,
    _assert_store_source,
    _inspect_store,
    embed_slide,
    run_slide,
)
from raw2features.sinks.zarr_sink import ZarrSink

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _make_store(
    out_dir,
    slide_id="s",
    *,
    n=12,
    models=(("a", 4),),
    grid_hash="GH",
    fill=1.0,
    source="file:///source/s.zarr",
):
    """Build a minimal *.embeddings.zarr by hand (no embedding needed)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{slide_id}.embeddings.zarr")
    key = "mpp1_px64"
    root = zarr.open_group(path, mode="w", zarr_format=2)
    g = root.require_group(GRIDS).require_group(key)  # uniform grid nesting
    c = g.create_array("coords", shape=(n, 2), chunks=(n, 2), dtype="int32")
    c[:] = (np.arange(n * 2, dtype="int32") % 7).reshape(n, 2)
    feats = g.create_group("features")
    for name, dim in models:
        arr = feats.create_array(name, shape=(n, dim), chunks=(n, dim), dtype="float16")
        arr[:] = np.full((n, dim), fill, dtype="float16")
    header = {
        "source": {"uri": source},
        "patching": {"read_level": 0, "read_px": 64, "patch_px": 64},
    }
    if grid_hash is not None:
        header["grid_hash"] = grid_hash
    g.attrs["raw2features"] = header
    root.attrs["raw2features"] = {
        "source": {"uri": source},
        "grids": {key: {"models": [m for m, _ in models], "grid_hash": grid_hash}}
    }
    return path


# -- torch-free: per-model validation ------------------------------------------


def test_validate_model_accepts_finite_nonzero(tmp_path):
    p = _make_store(str(tmp_path / "o"), fill=1.0)
    assert validate_model(open_grid(p), "a", 12) is True


def test_validate_model_rejects_all_zero_rows(tmp_path):
    p = _make_store(str(tmp_path / "o"), fill=0.0)
    assert validate_model(open_grid(p), "a", 12) is False


def test_validate_model_allows_legit_mid_zero_row(tmp_path):
    # A model may legitimately emit an all-zero feature row; only a truncated (all-zero
    # last) tail is incomplete, so a zero row mid-array must not fail resume.
    p = _make_store(str(tmp_path / "o"), fill=1.0)
    zarr.open_group(p, mode="r+")[GRIDS]["mpp1_px64"]["features"]["a"][5] = 0.0
    assert validate_model(open_grid(p), "a", 12) is True


def test_validate_model_rejects_missing(tmp_path):
    p = _make_store(str(tmp_path / "o"))
    assert validate_model(open_grid(p), "nope", 12) is False


def test_validate_model_rejects_wrong_length(tmp_path):
    p = _make_store(str(tmp_path / "o"), n=12)
    assert validate_model(open_grid(p), "a", 99) is False


# -- torch-free: store inspection / append decision ----------------------------


def test_inspect_store_reports_present_models(tmp_path):
    p = _make_store(str(tmp_path / "o"), models=(("a", 4), ("b", 8)), grid_hash="GH")
    key, n, valid = _inspect_store(p, "GH", ["a", "b", "c"])
    assert key is not None  # a grid of this geometry exists -> append to it
    assert n == 12
    assert set(valid) == {"a", "b"}  # c is absent -> would be embedded


def test_inspect_store_geometry_mismatch_adds_new_grid(tmp_path):
    p = _make_store(str(tmp_path / "o"), grid_hash="GH1")
    key, _, _ = _inspect_store(p, "GH2", ["a"])
    assert key is None  # no grid of this geometry -> caller adds a new grid


def test_inspect_store_legacy_without_grid_hash_is_compatible(tmp_path):
    p = _make_store(str(tmp_path / "o"), grid_hash=None)
    key, n, valid = _inspect_store(p, "anything", ["a"])
    assert key is not None  # legacy store: coords are authoritative
    assert valid == ["a"]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_same_label_different_grid_hash_uses_deterministic_suffix(
    synthetic_ngff, tmp_path
):
    out = str(tmp_path / "out")
    common = dict(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    base_cfg = RunConfig(**common)
    first = run_slide(
        synthetic_ngff,
        out,
        base_cfg,
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    path = first["output_uri"].removeprefix("file://")
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    original = np.asarray(open_grid(root, "mpp0.5_px64")["features"]["mock"][:])

    overlapping_cfg = RunConfig(**common, step_px=32)
    second = run_slide(
        synthetic_ngff,
        out,
        overlapping_cfg,
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    assert second["status"] == "complete"
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    keys = list(root[GRIDS].keys())
    suffixed = f"mpp0.5_px64_{overlapping_cfg.grid_hash()[:8]}"
    assert second["grid"] == suffixed
    assert set(keys) == {"mpp0.5_px64", suffixed}
    assert np.array_equal(
        np.asarray(open_grid(root, "mpp0.5_px64")["features"]["mock"][:]),
        original,
    )
    root_index = dict(root.attrs["raw2features"])["grids"]
    assert root_index[suffixed]["grid_hash"] == overlapping_cfg.grid_hash()

    again = run_slide(
        synthetic_ngff,
        out,
        overlapping_cfg,
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
    )
    assert again["status"] == "skipped"
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert set(root[GRIDS].keys()) == {"mpp0.5_px64", suffixed}


def test_inspect_store_absent_is_not_appendable(tmp_path):
    key, n, valid = _inspect_store(str(tmp_path / "missing.zarr"), "GH", ["a"])
    assert key is None and n == 0 and valid == []


def test_inspect_store_rejects_ambiguous_hashless_multigrid(tmp_path):
    path = _make_store(str(tmp_path / "out"), grid_hash=None)
    root = zarr.open_group(path, mode="r+")
    other = root[GRIDS].create_group("mpp2_px64")
    coords = other.create_array("coords", shape=(2, 2), dtype="int32")
    coords[:] = 0
    features = other.create_group("features")
    values = features.create_array("a", shape=(2, 4), dtype="float16")
    values[:] = 1
    other.attrs["raw2features"] = {
        "source": {"uri": "file:///source/s.zarr"},
        "patching": {"read_level": 0, "read_px": 64, "patch_px": 64},
    }

    assert _inspect_store(path, "unknown-geometry", ["a"]) == (None, 0, [])


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_multigrid_request_does_not_reuse_one_hashless_legacy_grid(
    synthetic_ngff, tmp_path
):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, device="cpu", amp="fp32")
    legacy = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["legacy"], target_mpp=0.5, patch_px=64, **common),
        embedders=[MockEmbedder(dim=4, input_size=64, name="legacy")],
    )
    path = legacy["output_uri"].removeprefix("file://")
    root = zarr.open_group(path, mode="r+", use_consolidated=False)
    legacy_grid = open_grid(root, "mpp0.5_px64")
    legacy_header = dict(legacy_grid.attrs["raw2features"])
    legacy_header.pop("grid_hash")
    legacy_grid.attrs["raw2features"] = legacy_header
    root_header = dict(root.attrs["raw2features"])
    root_grids = dict(root_header["grids"])
    root_entry = dict(root_grids["mpp0.5_px64"])
    root_entry.pop("grid_hash")
    root_grids["mpp0.5_px64"] = root_entry
    root_header["grids"] = root_grids
    root.attrs["raw2features"] = root_header

    geometry = [
        {"model": "mockA", "mpp": 0.5, "patch_px": 64},
        {"model": "mockB", "mpp": 1.0, "patch_px": 64},
    ]
    summary = embed_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA", "mockB"], **common),
        geometry_config=geometry,
        embedders=[
            MockEmbedder(dim=8, input_size=64, name="mockA"),
            MockEmbedder(dim=5, input_size=64, name="mockB"),
        ],
    )

    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert len(root[GRIDS]) == 3
    assert set(open_grid(root, "mpp0.5_px64")["features"].keys()) == {"legacy"}
    feature_sets = [
        set(open_grid(root, key)["features"].keys())
        for key in root[GRIDS].keys()
        if key != "mpp0.5_px64"
    ]
    assert {"mockA"} in feature_sets and {"mockB"} in feature_sets
    assert len(summary["grids"]) == 2
    assert "mpp0.5_px64" not in summary["grids"]


def test_store_source_binding_checks_root_and_every_grid(tmp_path):
    p = _make_store(str(tmp_path / "o"), source="file:///source/A.zarr")
    _assert_store_source(p, "file:///source/A.zarr")

    root = zarr.open_group(p, mode="r+")
    other = root[GRIDS].require_group("mpp2_px64")
    other.attrs["raw2features"] = {
        "source": {"uri": "file:///source/B.zarr"}
    }

    with pytest.raises(ValueError, match="Refusing to reuse existing store"):
        _assert_store_source(p, "file:///source/A.zarr")


def test_store_source_binding_does_not_leak_malformed_recorded_uri(tmp_path):
    malformed = "https://user:DO_NOT_PRINT@exa／mple.com/image.zarr"
    p = _make_store(str(tmp_path / "o"), source=malformed)

    with pytest.raises(ValueError) as caught:
        _assert_store_source(p, "https://example.org/image.zarr")

    assert "DO_NOT_PRINT" not in str(caught.value)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_runner_rejects_malformed_source_without_leaking_credentials(tmp_path):
    malformed = "https://user:DO_NOT_PRINT@exa／mple.com/image.zarr"

    with pytest.raises(ValueError) as caught:
        run_slide(
            malformed,
            str(tmp_path / "out"),
            RunConfig(models=["mock"], device="cpu"),
            embedders=[],
        )

    assert "DO_NOT_PRINT" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_store_source_binding_accepts_rotated_signed_uri(tmp_path):
    old = (
        "https://user:old@example.org/image.zarr?series=2&"
        "X-Amz-Credential=old&X-Amz-Signature=old"
    )
    new = (
        "https://user:new@example.org/image.zarr?series=2&"
        "X-Amz-Credential=new&X-Amz-Signature=new"
    )
    p = _make_store(str(tmp_path / "o"), source=old)
    _assert_store_source(p, new)


# -- torch-free: sink open_append is additive ----------------------------------


def test_open_append_adds_array_without_touching_existing(tmp_path):
    out = str(tmp_path / "out")
    path = _make_store(out, "s", n=10, models=(("a", 4),), fill=2.0)
    a_before = np.asarray(open_grid(path)["features"]["a"][:])

    sink = ZarrSink()
    n = sink.open_append(
        out, "s", new_model_dims={"b": 6}, new_model_meta={"b": {"x": 1}}
    )
    assert n == 10
    sink.write_block("b", 0, np.full((10, 6), 3.0, dtype="float32"))
    assert sink.feature_dims() == {"a": 4, "b": 6}
    sink.close()

    g = open_grid(path)
    assert set(g["features"].keys()) == {"a", "b"}
    assert np.array_equal(np.asarray(g["features"]["a"][:]), a_before)  # untouched
    assert (np.asarray(g["features"]["b"][:]) == 3.0).all()


def test_open_append_never_clobbers_present_model(tmp_path):
    out = str(tmp_path / "out")
    path = _make_store(out, "s", n=8, models=(("a", 4),), fill=5.0)
    sink = ZarrSink()
    # asking to "add" a model that already exists must not recreate/zero it
    sink.open_append(out, "s", new_model_dims={"a": 4}, new_model_meta={})
    sink.close()
    g = open_grid(path)
    assert (np.asarray(g["features"]["a"][:]) == 5.0).all()


# -- end-to-end (needs torch) --------------------------------------------------


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_then_add_model_is_additive(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")

    sa = run_slide(
        synthetic_ngff, out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    assert sa["status"] == "complete"
    path = sa["output_uri"].removeprefix("file://")
    g = open_grid(path)
    a_before = np.asarray(g["features"]["mockA"][:])
    coords_before = np.asarray(g["coords"][:])
    assert a_before.shape[1] == 8

    sb = run_slide(
        synthetic_ngff, out,
        RunConfig(models=["mockB"], **common),
        embedders=[MockEmbedder(dim=5, name="mockB", bias=2.0)],
    )
    assert sb["status"] == "complete"
    assert sb["models_added"] == ["mockB"]

    g2 = open_grid(path)
    assert set(g2["features"].keys()) == {"mockA", "mockB"}
    assert np.array_equal(np.asarray(g2["features"]["mockA"][:]), a_before)  # untouched
    assert np.array_equal(np.asarray(g2["coords"][:]), coords_before)
    b = np.asarray(g2["features"]["mockB"][:])
    assert b.shape == (a_before.shape[0], 5)
    assert np.isfinite(b).all()


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_rerun_same_model_skips(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    s2 = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    assert s2["status"] == "skipped"
    assert "mockA" in s2.get("models_present", [])


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_partial_overlap_only_adds_missing(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    s = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA", "mockB"], **common),
        embedders=[
            MockEmbedder(dim=8, name="mockA", bias=1.0),
            MockEmbedder(dim=5, name="mockB", bias=2.0),
        ],
    )
    assert s["status"] == "complete"
    assert s["models_added"] == ["mockB"]
    assert set(s["models_skipped"]) == {"mockA"}
    assert set(s["models"]) == {"mockA", "mockB"}


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_force_overwrites(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    s = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockB"], **common),
        embedders=[MockEmbedder(dim=5, name="mockB", bias=2.0)],
        force=True,
    )
    assert s["status"] == "complete"
    g = open_grid(s["output_uri"])
    assert set(g["features"].keys()) == {"mockB"}  # mockA overwritten away


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_run_slide_force_bypasses_valid_receipt(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    first = run_slide(
        synthetic_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[MockEmbedder(name="mock", bias=1.0)],
    )
    before = np.asarray(open_grid(first["output_uri"])["features"]["mock"][:])

    forced = run_slide(
        synthetic_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[MockEmbedder(name="mock", bias=5.0)],
        force=True,
    )
    after = np.asarray(open_grid(forced["output_uri"])["features"]["mock"][:])

    assert forced["status"] == "complete"
    assert not np.array_equal(before, after)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_slide_force_bypasses_whole_request_receipt(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(models=["mock"], no_seg=True, device="cpu", amp="fp32")
    geometry = [{"model": "mock", "mpp": 0.5, "patch_px": 64}]
    first = embed_slide(
        synthetic_ngff,
        out,
        cfg,
        geometry_config=geometry,
        receipts_dir=receipts,
        embedders=[MockEmbedder(name="mock", bias=1.0)],
    )
    before = np.asarray(open_grid(first["output_uri"])["features"]["mock"][:])

    forced = embed_slide(
        synthetic_ngff,
        out,
        cfg,
        geometry_config=geometry,
        receipts_dir=receipts,
        embedders=[MockEmbedder(name="mock", bias=5.0)],
        force=True,
    )
    after = np.asarray(open_grid(forced["output_uri"])["features"]["mock"][:])

    assert forced["status"] == "complete"
    assert not np.array_equal(before, after)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_same_basename_different_source_cannot_reuse_store(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = build_ngff_v04(str(first_dir / "S.zarr"))
    second = build_ngff_v04(str(second_dir / "S.zarr"))
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    summary = run_slide(
        first,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[MockEmbedder(name="mock")],
    )
    before = np.asarray(open_grid(summary["output_uri"])["coords"][:])

    with pytest.raises(ValueError, match="same-named slides"):
        run_slide(
            second,
            out,
            cfg,
            receipts_dir=receipts,
            embedders=[MockEmbedder(name="mock")],
        )

    after = np.asarray(open_grid(summary["output_uri"])["coords"][:])
    assert np.array_equal(before, after)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_different_geometry_adds_a_second_grid(synthetic_ngff, tmp_path):
    """A later embedder at a DIFFERENT geometry adds a NEW grid alongside the first --
    never wiping or erroring (the 'add an embedder a day later' workflow)."""
    out = str(tmp_path / "out")
    common = dict(no_seg=True, patch_px=64, device="cpu", amp="fp32")
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockA"], target_mpp=0.5, **common),
        embedders=[MockEmbedder(dim=8, name="mockA", bias=1.0)],
    )
    sb = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["mockB"], target_mpp=1.0, **common),  # different MPP
        embedders=[MockEmbedder(dim=5, name="mockB", bias=2.0)],
    )
    assert sb["status"] == "complete"
    root = zarr.open_group(sb["output_uri"].removeprefix("file://"), mode="r")
    assert set(root[GRIDS].keys()) == {"mpp0.5_px64", "mpp1_px64"}  # two grids
    # each grid holds only its own model -- the first grid is untouched
    assert "mockA" in open_grid(root, "mpp0.5_px64")["features"]
    assert "mockB" in open_grid(root, "mpp1_px64")["features"]
    assert "mockB" not in open_grid(root, "mpp0.5_px64")["features"]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_slide_writes_multiple_grids_in_one_run(synthetic_ngff, tmp_path):
    """embed_slide resolves per-model geometry and writes one grid per group in a single
    invocation. A geometry config drives two grids from the two mock models."""
    out = str(tmp_path / "out")
    rec = str(tmp_path / "rec")
    cfg = RunConfig(models=["mockA", "mockB"], no_seg=True, device="cpu", amp="fp32")
    geom = [
        {"model": "mockA", "mpp": 0.5, "patch_px": 64},
        {"model": "mockB", "mpp": 1.0, "patch_px": 64},
    ]
    embedders = [MockEmbedder(dim=8, name="mockA"), MockEmbedder(dim=5, name="mockB")]
    summary = embed_slide(
        synthetic_ngff, out, cfg, geometry_config=geom,
        embedders=embedders, receipts_dir=rec,
    )
    assert summary["status"] == "complete"
    assert set(summary["grids"]) == {"mpp0.5_px64", "mpp1_px64"}
    root = zarr.open_group(summary["output_uri"].removeprefix("file://"), mode="r")
    assert set(root[GRIDS].keys()) == {"mpp0.5_px64", "mpp1_px64"}
    assert "mockA" in open_grid(root, "mpp0.5_px64")["features"]
    assert "mockB" in open_grid(root, "mpp1_px64")["features"]

    # whole-request receipt -> an identical re-run skips both grids
    again = embed_slide(
        synthetic_ngff, out, cfg, geometry_config=geom,
        embedders=embedders, receipts_dir=rec,
    )
    assert again["status"] == "skipped"


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_missing_geojson_is_produced_after_complete_receipt_once(
    synthetic_ngff, tmp_path, monkeypatch
):
    import raw2features.pipeline.runner as rn

    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    common = dict(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
        receipts_dir=receipts,
    )
    monkeypatch.setattr(
        rn,
        "_embed_patches",
        lambda *args, **kwargs: pytest.fail("GeoJSON run re-embedded patches"),
    )
    produced = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, emit_geojson=True),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
        receipts_dir=receipts,
    )
    path = os.path.join(out, "synthetic.patches.geojson")
    assert produced["status"] == "complete"
    assert produced["geojson"] == path
    with open(path, "rb") as fh:
        original = fh.read()

    again = run_slide(
        synthetic_ngff,
        out,
        RunConfig(**common, emit_geojson=True),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mock")],
        receipts_dir=receipts,
    )
    assert again["status"] == "skipped"
    with open(path, "rb") as fh:
        assert fh.read() == original


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_slide_runs_pooling_encoder_on_each_grid(synthetic_ngff, tmp_path):
    """In a multi-grid run a model-agnostic pooling encoder runs on EACH grid (its patch
    encoder is whatever that grid holds) -> slide/mean under both grids."""
    out = str(tmp_path / "out")
    cfg = RunConfig(
        models=["mockA", "mockB"], no_seg=True, device="cpu", amp="fp32",
        slide_encoders=["mean"],
    )
    geom = [
        {"model": "mockA", "mpp": 0.5, "patch_px": 64},
        {"model": "mockB", "mpp": 1.0, "patch_px": 64},
    ]
    embs = [MockEmbedder(dim=8, name="mockA"), MockEmbedder(dim=5, name="mockB")]
    summary = embed_slide(
        synthetic_ngff, out, cfg, geometry_config=geom, embedders=embs,
    )
    assert summary["status"] == "complete"
    root = zarr.open_group(summary["output_uri"].removeprefix("file://"), mode="r")
    assert "mean" in open_grid(root, "mpp0.5_px64")["slide"]  # pooled mockA
    assert "mean" in open_grid(root, "mpp1_px64")["slide"]  # pooled mockB
