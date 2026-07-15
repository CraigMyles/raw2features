"""The --qc producer path through the runner: run_slide invokes it and the layer lands.

The GrandQC inference itself needs a GPU + weights (covered by cluster validation, not
the unit suite), so here we stub the producer and assert the *wiring*: when cfg.qc is
set, run_slide calls the qc step on the freshly-built grid (its coords + level0_patch)
and the scores land in grids/<key>/qc/<tool>/.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from conftest import MockEmbedder, build_ngff_v04

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_qc_runs_in_embed_path_and_writes_layer(tmp_path, monkeypatch):
    import raw2features.pipeline.runner as rn
    from raw2features.core.store import open_grid
    from raw2features.pipeline.runner import RunConfig, embed_slide

    seen = {}

    def fake_run_qc(
        qc_tools, reader, sink, coords, level0_patch, device, stain_norm=None,
        artifact_mpp="1.5",
    ):
        seen["tools"] = list(qc_tools)
        seen["artifact_mpp"] = artifact_mpp
        seen["n"] = len(coords)
        seen["level0_patch"] = level0_patch
        scores = np.tile([0.7, 0.3], (len(coords), 1)).astype("float16")
        sink.write_qc("fake", scores, ["clean", "artifact"])

    monkeypatch.setattr(rn, "_run_qc", fake_run_qc)

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    cfg = RunConfig(models=["mock"], no_seg=True, target_mpp=0.5, patch_px=64,
                    device="cpu", amp="fp32", qc=["fake"])
    embed_slide(slide, out, cfg,
                embedders=[MockEmbedder(dim=8, input_size=64, name="mock")])

    # run_slide invoked the qc step with the grid's geometry
    assert seen.get("tools") == ["fake"]
    assert seen["n"] > 0 and seen["level0_patch"] > 0
    # ...and the layer is in the store, 1:1 with coords, role=qc
    g = open_grid(os.path.join(out, "S.embeddings.zarr"))
    assert "qc" in g and "fake" in g["qc"]
    arr = g["qc"]["fake"]["scores"]
    assert arr.shape == (seen["n"], 2)
    assert list(arr.attrs["classes"]) == ["clean", "artifact"]
    assert arr.attrs.get("role") == "qc"


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_missing_qc_is_produced_after_complete_receipt_once(tmp_path, monkeypatch):
    import raw2features.pipeline.runner as rn
    from raw2features.pipeline.runner import RunConfig, run_slide

    calls = []

    def fake_run_qc(
        qc_tools,
        reader,
        sink,
        coords,
        level0_patch,
        device,
        stain_norm=None,
        artifact_mpp="1.5",
    ):
        calls.append(list(qc_tools))
        scores = np.ones((len(coords), 1), dtype="float16")
        sink.write_qc("fake", scores, ["clean"])

    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
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
    embedder = MockEmbedder(dim=8, input_size=64, name="mock")
    run_slide(
        slide,
        out,
        RunConfig(**common),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    monkeypatch.setattr(rn, "_run_qc", fake_run_qc)
    monkeypatch.setattr(
        rn,
        "_embed_patches",
        lambda *args, **kwargs: pytest.fail("aux-only run re-embedded patches"),
    )

    produced = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert produced["status"] == "complete"
    assert calls == [["fake"]]

    # An interrupted QC producer may have left a same-named but incomplete group.
    # Resume replaces that partial group rather than failing at create_array().
    from raw2features.core.store import open_grid

    group = open_grid(produced["output_uri"], mode="r+")
    del group["qc"]["fake"]["scores"]
    group["qc"]["fake"].create_array(
        "scores", shape=(1, 1), chunks=(1, 1), dtype="float16"
    )
    repaired = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert repaired["status"] == "complete"
    assert calls == [["fake"], ["fake"]]

    group = open_grid(repaired["output_uri"], mode="r+")
    del group["qc"]["fake"]["scores"].attrs["role"]
    repaired_attrs = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert repaired_attrs["status"] == "complete"
    assert calls == [["fake"], ["fake"], ["fake"]]

    group = open_grid(repaired_attrs["output_uri"], mode="r+")
    del group["qc"]["fake"].attrs["complete"]
    repaired_marker = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert repaired_marker["status"] == "complete"
    assert calls == [["fake"], ["fake"], ["fake"], ["fake"]]

    group = open_grid(repaired_marker["output_uri"], mode="r+")
    extra = group["qc"]["fake"].create_array(
        "partial_extra", shape=(1,), chunks=(1,), dtype="uint8"
    )
    extra.attrs["role"] = "wrong"
    repaired_extra = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert repaired_extra["status"] == "complete"
    assert calls == [["fake"], ["fake"], ["fake"], ["fake"], ["fake"]]

    again = run_slide(
        slide,
        out,
        RunConfig(**common, qc=["fake"]),
        embedders=[embedder],
        receipts_dir=receipts,
    )
    assert again["status"] == "skipped"
    assert calls == [["fake"], ["fake"], ["fake"], ["fake"], ["fake"]]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_missing_qc_is_produced_during_model_append(tmp_path, monkeypatch):
    import raw2features.pipeline.runner as rn
    from raw2features.pipeline.runner import RunConfig, run_slide

    calls = []

    def fake_run_qc(
        qc_tools,
        reader,
        sink,
        coords,
        level0_patch,
        device,
        stain_norm=None,
        artifact_mpp="1.5",
    ):
        calls.append(list(qc_tools))
        sink.write_qc("fake", np.ones((len(coords), 1), dtype="float16"), ["clean"])

    monkeypatch.setattr(rn, "_run_qc", fake_run_qc)
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    out = str(tmp_path / "out")
    common = dict(
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    run_slide(
        slide,
        out,
        RunConfig(models=["mockA"], **common),
        embedders=[MockEmbedder(dim=8, input_size=64, name="mockA")],
    )
    appended = run_slide(
        slide,
        out,
        RunConfig(models=["mockB"], qc=["fake"], **common),
        embedders=[MockEmbedder(dim=5, input_size=64, name="mockB")],
    )

    assert appended["models_added"] == ["mockB"]
    assert calls == [["fake"]]
