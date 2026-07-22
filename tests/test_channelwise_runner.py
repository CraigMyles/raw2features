"""End-to-end multiplex strategy tests over the real runner/store seams."""

from __future__ import annotations

import os
from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from conftest import MockEmbedder
from raw2features.cli.main import app
from raw2features.core.store import open_grid
from raw2features.pipeline.receipt import canonical_source_uri, is_complete
from raw2features.pipeline.runner import (
    RunConfig,
    _multiplex_normalization_level,
    embed_slide,
    resolve_multiplex_output_contracts,
    resolve_multiplex_source_config,
    resolve_run,
    run_slide,
)


def _cfg(*, markers=("CD3", "DAPI"), aggregation="mean", slide=True):
    return RunConfig(
        models=["mock"],
        target_mpp=0.5,
        patch_px=64,
        step_px=64,
        no_seg=True,
        features_dtype="float32",
        device="cpu",
        batch_size=2,
        read_workers=1,
        multiplex_strategy="channelwise",
        multiplex_markers=list(markers),
        multiplex_normalization="percentile",
        multiplex_aggregation=aggregation,
        slide_encoders=["mean"] if slide else [],
    )


def _root_and_grid(out_dir, summary):
    path = os.path.join(out_dir, "multiplex.embeddings.zarr")
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    return root, open_grid(root, summary["grid"])


def test_channelwise_runner_writes_self_describing_patch_and_slide_outputs(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    out = str(tmp_path / "out")
    summary = run_slide(
        synthetic_multiplex_ngff,
        out,
        _cfg(),
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )

    assert summary["status"] == "complete"
    assert len(summary["requested_models"]) == 1
    effective = summary["requested_models"][0]
    assert effective.startswith("mock__channelwise_mean_")
    assert len(effective.rsplit("_", 1)[1]) == 16

    _root, group = _root_and_grid(out, summary)
    features = group["features"][effective]
    assert features.shape[1] == 4
    assert np.isfinite(np.asarray(features[:])).all()

    fingerprint = dict(features.attrs)["output_fingerprint"]
    payload = fingerprint["payload"]
    assert payload["output"]["modality"] == "multiplex"
    assert payload["multiplex"]["base_model"] == "mock"
    assert payload["multiplex"]["base_pooling"] == "cls"
    assert payload["multiplex"]["normalization"]["source_level"] == 0
    assert [
        row["source_name"] for row in payload["multiplex"]["normalization"]["resolved"]
    ] == ["CD3", "DAPI"]

    header = dict(group.attrs["raw2features"])
    assert header["segmentation"] == {"segmenter": "none"}
    assert "mask" not in group
    model_header = header["models"][effective]
    assert model_header["pooling"] == "mean"
    assert model_header["output_fingerprint"]["payload"]["output"]["pooling"] == "mean"
    assert model_header["multiplex"]["base_pooling"] == "cls"
    panel = header["panel"][effective]
    assert [row["source_name"] for row in panel["source_channels"]] == [
        "DAPI",
        "CD3",
        "CD8",
        "CD20",
        "FOXP3",
    ]
    assert [row["source_name"] for row in panel["excluded"]] == [
        "CD8",
        "CD20",
        "FOXP3",
    ]
    slide_key = f"mean__{effective}"
    assert tuple(group["slide"][slide_key].shape) == (1, 4)
    assert dict(group["slide"][slide_key].attrs)["patch_encoder"] == effective
    assert header["slide_embeddings"][slide_key]["slide_encoder"] == "mean"


def test_dna1_dna2_pair_is_bound_into_grid_and_segmentation_provenance(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    source = zarr.open_group(synthetic_multiplex_ngff, mode="a", use_consolidated=False)
    omero = dict(source.attrs["omero"])
    omero["channels"] = [
        {"label": name}
        for name in ("DNA1(Ir191)", "DNA2(Ir193)", "CD3", "CD20", "FOXP3")
    ]
    source.attrs["omero"] = omero

    cfg = replace(_cfg(markers=("CD3",), slide=False), no_seg=False)
    summary = run_slide(
        synthetic_multiplex_ngff,
        str(tmp_path / "out"),
        cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )

    _root, group = _root_and_grid(str(tmp_path / "out"), summary)
    segmentation = dict(group.attrs["raw2features"])["segmentation"]
    assert segmentation["nuclear_channel_indices"] == [0, 1]
    assert segmentation["nuclear_channel_names"] == [
        "DNA1(Ir191)",
        "DNA2(Ir193)",
    ]
    assert segmentation["nuclear_channel_combination"] == "float32_mean"
    assert segmentation["channel_binding_contract_version"] == 2

    resolved = resolve_multiplex_source_config(
        synthetic_multiplex_ngff,
        cfg,
        model_specs={"mock": MockEmbedder(name="mock").spec},
    )
    assert resolved.resolved_nuclear_channel_indices == [0, 1]
    assert resolved.grid_hash() == dict(group.attrs["raw2features"])["grid_hash"]


def test_channelwise_append_keeps_both_panel_contracts_and_slide_pools(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    out = str(tmp_path / "out")
    first = run_slide(
        synthetic_multiplex_ngff,
        out,
        _cfg(markers=("CD3", "DAPI")),
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )
    second = run_slide(
        synthetic_multiplex_ngff,
        out,
        _cfg(markers=("DAPI", "CD3")),
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )
    first_key = first["requested_models"][0]
    second_key = second["requested_models"][0]
    assert first_key != second_key
    assert first["grid"] == second["grid"]

    _root, group = _root_and_grid(out, second)
    assert {first_key, second_key} <= set(group["features"].keys())
    header = dict(group.attrs["raw2features"])
    assert {first_key, second_key} <= set(header["panel"])
    assert {f"mean__{first_key}", f"mean__{second_key}"} <= set(group["slide"].keys())


def test_channel_names_override_enables_incomplete_ome_metadata(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    root = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}, {}]}
    cfg = _cfg(slide=False)
    cfg.channel_names_override = ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]

    summary = run_slide(
        synthetic_multiplex_ngff,
        str(tmp_path / "out"),
        cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )
    _root, group = _root_and_grid(str(tmp_path / "out"), summary)
    header = dict(group.attrs["raw2features"])
    assert header["source"]["channel_names"] == cfg.channel_names_override
    assert header["source"]["channel_names_source"] == "channel_names_file"
    assert header["source"]["omero_channel_names"] == ["DAPI", ""]


def test_channel_names_override_is_reapplied_for_aux_thumbnail(
    synthetic_multiplex_ngff, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    root = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    root.attrs["omero"] = {"channels": [{"label": "DAPI"}, {}]}
    cfg = replace(_cfg(slide=False), no_seg=False)
    cfg.channel_names_override = ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]
    out = str(tmp_path / "out")

    run_slide(
        synthetic_multiplex_ngff,
        out,
        cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )
    runner = import_module("raw2features.pipeline.runner")
    monkeypatch.setattr(
        runner,
        "_embed_patches",
        lambda *args, **kwargs: pytest.fail("auxiliary run re-embedded patches"),
    )
    run_slide(
        synthetic_multiplex_ngff,
        out,
        replace(cfg, emit_thumbnail=True),
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )

    assert os.path.isfile(os.path.join(out, "multiplex.thumbnail.png"))
    assert os.path.isfile(os.path.join(out, "multiplex.thumbnail.overlay.png"))


@pytest.mark.parametrize("multiplex_mpp", [0.5, 1.0])
def test_multiplex_addition_fills_shared_source_channel_provenance(
    synthetic_multiplex_ngff, tmp_path, multiplex_mpp
):
    pytest.importorskip("torch")
    source = zarr.open_group(synthetic_multiplex_ngff, mode="r+")
    source.attrs["omero"] = {"channels": [{"label": "DAPI"}, {}]}
    out = str(tmp_path / "out")
    brightfield_cfg = RunConfig(
        models=["bright"],
        target_mpp=0.5,
        patch_px=64,
        step_px=64,
        no_seg=True,
        features_dtype="float32",
        device="cpu",
        batch_size=2,
        read_workers=1,
    )
    run_slide(
        synthetic_multiplex_ngff,
        out,
        brightfield_cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="bright")],
    )

    multiplex_cfg = replace(_cfg(slide=False), target_mpp=multiplex_mpp)
    multiplex_cfg.channel_names_override = [
        "DAPI",
        "CD3",
        "CD8",
        "CD20",
        "FOXP3",
    ]
    resolved_cfg = resolve_multiplex_source_config(
        synthetic_multiplex_ngff,
        multiplex_cfg,
        model_specs={"mock": MockEmbedder(name="mock").spec},
    )
    assert resolved_cfg.resolved_original_channel_names == ["DAPI", ""]
    summary = run_slide(
        synthetic_multiplex_ngff,
        out,
        multiplex_cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )

    root, group = _root_and_grid(out, summary)
    for header in (
        dict(root.attrs["raw2features"]),
        dict(group.attrs["raw2features"]),
    ):
        assert header["source"]["channel_names"] == (
            multiplex_cfg.channel_names_override
        )
        assert header["source"]["channel_names_source"] == "channel_names_file"
        assert header["source"]["omero_channel_names"] == ["DAPI", ""]


def test_panel_source_origin_does_not_change_channelwise_output_fingerprint(
    synthetic_multiplex_ngff,
):
    base = MockEmbedder(dim=4, input_size=64, name="mock")
    metadata_cfg = resolve_multiplex_source_config(
        synthetic_multiplex_ngff, _cfg(slide=False), model_specs={"mock": base.spec}
    )
    override_cfg = _cfg(slide=False)
    override_cfg.channel_names_override = ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]
    override_cfg = resolve_multiplex_source_config(
        synthetic_multiplex_ngff, override_cfg, model_specs={"mock": base.spec}
    )
    assert metadata_cfg.resolved_original_channel_names == [
        "DAPI",
        "CD3",
        "CD8",
        "CD20",
        "FOXP3",
    ]
    assert override_cfg.resolved_original_channel_names == [
        "DAPI",
        "CD3",
        "CD8",
        "CD20",
        "FOXP3",
    ]
    _, metadata_contracts = resolve_multiplex_output_contracts(
        synthetic_multiplex_ngff,
        metadata_cfg,
        embedders=[base],
        device="cpu",
    )
    _, override_contracts = resolve_multiplex_output_contracts(
        synthetic_multiplex_ngff,
        override_cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
        device="cpu",
    )
    first = next(iter(metadata_contracts.values()))["output_fingerprint"]
    second = next(iter(override_contracts.values()))["output_fingerprint"]
    assert first == second


@pytest.mark.parametrize("strategy_first", [False, True])
def test_brightfield_otsu_and_channelwise_nuclear_grids_never_alias(
    synthetic_multiplex_ngff, tmp_path, strategy_first
):
    pytest.importorskip("torch")
    out = str(tmp_path / "out")
    brightfield = RunConfig(
        models=["mock"],
        target_mpp=0.5,
        patch_px=64,
        step_px=64,
        segmenter="otsu",
        features_dtype="float32",
        device="cpu",
        batch_size=2,
        read_workers=1,
    )
    strategy = replace(_cfg(slide=False), no_seg=False)
    requests = [
        (strategy, MockEmbedder(dim=4, input_size=64, name="mock")),
        (brightfield, MockEmbedder(dim=4, input_size=64, name="mock")),
    ]
    if not strategy_first:
        requests.reverse()

    summaries = [
        run_slide(synthetic_multiplex_ngff, out, cfg, embedders=[emb])
        for cfg, emb in requests
    ]

    assert summaries[0]["grid"] != summaries[1]["grid"]
    root = zarr.open_group(
        os.path.join(out, "multiplex.embeddings.zarr"),
        mode="r",
        use_consolidated=False,
    )
    headers = [
        dict(root["grids"][key].attrs["raw2features"])
        for key in root["grids"].group_keys()
    ]
    assert {header["segmentation"]["segmenter"] for header in headers} == {
        "otsu",
        "nuclear",
    }
    assert len({header["grid_hash"] for header in headers}) == 2


def test_complete_channelwise_rerun_does_not_load_registry_base_model(
    synthetic_multiplex_ngff, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    loads = []
    unloads = []

    class TrackingEmbedder(MockEmbedder):
        def load(self, device="cpu", dtype=None, compile=False):
            loads.append((device, dtype, compile))
            return super().load(device=device, dtype=dtype, compile=compile)

        def unload(self):
            unloads.append(True)
            return super().unload()

    monkeypatch.setattr(
        "raw2features.pipeline.runner.build_embedder",
        lambda name: TrackingEmbedder(dim=4, input_size=64, name=name),
    )
    cfg = _cfg(slide=False)
    out = str(tmp_path / "out")
    first = run_slide(synthetic_multiplex_ngff, out, cfg)
    second = run_slide(synthetic_multiplex_ngff, out, cfg)

    assert first["status"] == "complete"
    assert second["status"] == "skipped"
    assert len(loads) == 1
    assert len(unloads) == 1


def test_channelwise_does_not_unload_a_caller_owned_warm_base(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    unloads = []

    class TrackingEmbedder(MockEmbedder):
        def unload(self):
            unloads.append(True)
            return super().unload()

    base = TrackingEmbedder(dim=4, input_size=64, name="mock")
    run_slide(
        synthetic_multiplex_ngff,
        str(tmp_path / "out-a"),
        _cfg(slide=False),
        embedders=[base],
    )
    run_slide(
        synthetic_multiplex_ngff,
        str(tmp_path / "out-b"),
        _cfg(slide=False),
        embedders=[base],
    )

    assert unloads == []


def test_effective_patch_batch_preserves_base_rgb_batch_budget(
    synthetic_multiplex_ngff,
):
    cfg = _cfg(markers=("DAPI", "CD3"), slide=False)
    cfg.batch_size = 16
    effective, _contracts = resolve_multiplex_output_contracts(
        synthetic_multiplex_ngff,
        cfg,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
        device="cpu",
    )
    assert effective.batch_size == 8


def test_normalization_refuses_an_unbounded_single_level_read():
    class HugeSingleLevel:
        level_dimensions = [
            type("Dimensions", (), {"width": 50_000, "height": 40_000})()
        ]

    with pytest.raises(ValueError) as exc_info:
        _multiplex_normalization_level(HugeSingleLevel(), max_side=2048)
    message = str(exc_info.value)
    assert "Add a downsampled level" in message
    assert "--multiplex-normalization-max-side-px" in message
    assert "50000" in message
    assert "more RAM" in message


def test_normalization_level_honours_the_configured_maximum_side():
    class Pyramid:
        level_dimensions = [
            type("Dimensions", (), {"width": side, "height": side})()
            for side in (4096, 2048, 1024)
        ]

    assert _multiplex_normalization_level(Pyramid(), max_side=4096) == 0
    assert _multiplex_normalization_level(Pyramid(), max_side=2048) == 1
    assert _multiplex_normalization_level(Pyramid(), max_side=1024) == 2


def test_high_level_receipt_validates_against_source_resolved_effective_contract(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = _cfg(slide=False)
    summary = embed_slide(
        synthetic_multiplex_ngff,
        out,
        cfg,
        requested_mpp=0.5,
        requested_patch_px=64,
        receipts_dir=receipts,
        embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
    )

    resolved_cfg = resolve_multiplex_source_config(
        synthetic_multiplex_ngff,
        cfg,
        model_specs={"mock": MockEmbedder(dim=4, input_size=64).spec},
    )
    _groups, group_cfgs, run_hash = resolve_run(
        resolved_cfg,
        requested_mpp=0.5,
        requested_patch_px=64,
        model_specs={"mock": MockEmbedder(dim=4, input_size=64).spec},
    )
    contracts = {}
    expected_grid_models = {}
    for group_cfg in group_cfgs:
        effective_cfg, group_contracts = resolve_multiplex_output_contracts(
            synthetic_multiplex_ngff,
            group_cfg,
            embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
            device="cpu",
        )
        contracts.update(group_contracts)
        expected_grid_models[group_cfg.grid_hash()] = effective_cfg.models

    assert is_complete(
        receipts,
        "multiplex",
        run_hash,
        expected_source_uri=canonical_source_uri(synthetic_multiplex_ngff),
        expected_output_uri=summary["output_uri"],
        expected_model_contracts=contracts,
        expected_grid_models=expected_grid_models,
        compatible_grid_hashes={
            group_cfg.grid_hash(): group_cfg.compatible_legacy_grid_hashes()
            for group_cfg in group_cfgs
        },
    )


def test_verify_accepts_a_direct_single_grid_channelwise_receipt(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    from raw2features.embedders.model_registry import get_spec

    spec = get_spec("resnet50")
    base = MockEmbedder(
        dim=spec.embedding_dim,
        input_size=spec.input_size,
        name=spec.name,
    )
    base.spec = spec
    cfg = RunConfig(
        models=["resnet50"],
        target_mpp=0.5,
        patch_px=64,
        step_px=64,
        no_seg=True,
        features_dtype="float32",
        device="cpu",
        batch_size=2,
        read_workers=1,
        multiplex_strategy="channelwise",
        multiplex_markers=["CD3", "DAPI"],
    )
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    summary = run_slide(
        synthetic_multiplex_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[base],
    )
    second = run_slide(
        synthetic_multiplex_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[base],
    )
    result = CliRunner().invoke(
        app,
        [
            "verify",
            synthetic_multiplex_ngff,
            "--receipts-dir",
            receipts,
            "--out-dir",
            out,
            "-m",
            "resnet50",
            "--mpp",
            "0.5",
            "--patch-size",
            "64",
            "--step",
            "64",
            "--no-seg",
            "--features-dtype",
            "float32",
            "--device",
            "cpu",
            "--multiplex-strategy",
            "channelwise",
            "--marker",
            "CD3",
            "--marker",
            "DAPI",
            "--quiet",
        ],
    )

    assert summary["status"] == "complete"
    assert second["status"] == "skipped"
    assert second["reason"] == "already complete"
    assert result.exit_code == 0, result.output


def test_channelwise_rejects_learned_slide_encoder_before_embedding(
    synthetic_multiplex_ngff, tmp_path
):
    pytest.importorskip("torch")
    cfg = _cfg(slide=False)
    cfg.slide_encoders = ["titan"]
    with pytest.raises(ValueError, match="model-agnostic slide poolers"):
        run_slide(
            synthetic_multiplex_ngff,
            str(tmp_path / "out"),
            cfg,
            embedders=[MockEmbedder(dim=4, input_size=64, name="mock")],
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"multiplex_markers": ["CD3"]},
        {"multiplex_normalization": "dtype"},
        {"multiplex_aggregation": "attention"},
        {"multiplex_normalization_max_side_px": 1024},
    ],
)
def test_brightfield_rejects_orphaned_multiplex_options(kwargs):
    with pytest.raises(ValueError, match="require multiplex_strategy"):
        RunConfig(models=["mock"], **kwargs)
