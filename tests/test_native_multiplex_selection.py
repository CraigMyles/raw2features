"""Native multiplex marker selection, identity, and execution-safety contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from raw2features.embedders.base import Embedder, ModelSpec
from raw2features.multiplex.panel import resolve_marker_selection
from raw2features.pipeline.runner import (
    RunConfig,
    _effective_batch_size,
    _reject_mixed_native_and_brightfield_models,
    expected_model_contracts,
    resolve_multiplex_source_config,
    resolve_run,
    run_slide,
)


def _native_spec(name: str) -> ModelSpec:
    return ModelSpec(
        name=name,
        family="test",
        source=f"hf_hub:example/{name}",
        embedding_dim=4,
        input_size=32,
        pooling="cls",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        transform_source_url="https://example.org/model",
        license="MIT",
        gated=False,
        modality="multiplex",
    )


class _NativeEmbedder(Embedder):
    def __init__(self, name: str):
        super().__init__(_native_spec(name))
        self.bound_names = None

    def set_panel(self, channel_names):
        self.bound_names = list(channel_names or [])
        return {"kept": self.bound_names}

    def load(self, device="cpu", dtype=None, compile=False):
        return self

    def embed_batch(self, batch):  # pragma: no cover - selection is tested pre-forward
        raise AssertionError("not used")


class _LimitedNativeEmbedder(_NativeEmbedder):
    def __init__(self, name: str, limit):
        super().__init__(name)
        self.limit = limit

    @property
    def max_batch_size(self):
        return self.limit


def test_selection_is_normalized_but_keeps_requested_order_and_physical_indices():
    panel = ["DAPI", "CD3", "PanCK"]
    assert resolve_marker_selection(panel, [" pAnCk ", "dapi"]) == [
        {"source_index": 2, "source_name": "PanCK"},
        {"source_index": 0, "source_name": "DAPI"},
    ]
    assert resolve_marker_selection(["DAPI", "", "CD3"]) == [
        {"source_index": 0, "source_name": "DAPI"},
        {"source_index": 1, "source_name": ""},
        {"source_index": 2, "source_name": "CD3"},
    ]


@pytest.mark.parametrize(
    ("panel", "requested", "message"),
    [
        (["CD3"], ["CD8"], "not present"),
        (["CD3", "cd3"], ["CD3"], "ambiguous"),
        (["CD3"], ["CD3", "cd3"], "more than once"),
    ],
)
def test_selection_refuses_missing_ambiguous_or_duplicate_requests(
    panel, requested, message
):
    with pytest.raises(ValueError, match=message):
        resolve_marker_selection(panel, requested)


def test_native_binding_slices_full_hwc_patches_by_physical_index_in_request_order():
    embedder = _NativeEmbedder("native")
    selection = [
        {"source_index": 2, "source_name": "C"},
        {"source_index": 0, "source_name": "A"},
    ]
    summary = embedder.bind_multiplex_panel(
        ["A", "B", "C"], selected_channels=selection
    )
    patch = np.stack(
        [
            np.full((2, 2), 1, dtype=np.uint16),
            np.full((2, 2), 2, dtype=np.uint16),
            np.full((2, 2), 3, dtype=np.uint16),
        ],
        axis=2,
    )
    selected = embedder.select_multiplex_channels([patch])[0]

    assert summary == {
        "kept": ["C", "A"],
        "source_selection": selection,
    }
    assert embedder.bound_names == ["C", "A"]
    assert selected.shape == (2, 2, 2)
    assert np.all(selected[..., 0] == 3)
    assert np.all(selected[..., 1] == 1)


def test_kronos_v1_explicit_selection_provenance_uses_physical_source_indices():
    from raw2features.embedders.kronos_embedder import KronosEmbedder
    from raw2features.embedders.model_registry import get_spec

    embedder = KronosEmbedder(get_spec("kronos"))
    embedder._vocab = {
        "DAPI": (4, 0.1, 0.2, "DAPI"),
        "CD3": (3, 0.1, 0.2, "CD3"),
    }
    selection = [
        {"source_index": 2, "source_name": "CD3"},
        {"source_index": 0, "source_name": "DAPI"},
    ]
    summary = embedder.bind_multiplex_panel(
        ["DAPI", "ignored", "CD3"], selected_channels=selection
    )

    assert [row["channel_index"] for row in summary["mapping"]] == [2, 0]
    assert [row["channel"] for row in summary["mapping"]] == ["CD3", "DAPI"]
    assert summary["source_selection"] == selection
    assert embedder._panel["idx"].tolist() == [0, 1]


def test_native_transform_signatures_never_cross_model_names():
    first = _NativeEmbedder("native_a")
    second = _NativeEmbedder("native_b")
    assert first.transform_signature != second.transform_signature


def test_model_batch_caps_lower_the_runtime_batch_and_fail_on_invalid_limits():
    ordinary = _NativeEmbedder("ordinary")
    limited = _LimitedNativeEmbedder("limited", 8)
    assert _effective_batch_size(256, [ordinary]) == 256
    assert _effective_batch_size(256, [ordinary, limited]) == 8
    assert _effective_batch_size(4, [limited]) == 4
    with pytest.raises(ValueError, match="positive integer or None"):
        _effective_batch_size(256, [_LimitedNativeEmbedder("bad", 0)])


def test_native_marker_selection_changes_output_identity_not_grid_identity():
    base = RunConfig(
        models=["kronos"],
        no_seg=True,
        resolved_channel_names=["DAPI", "CD3", "CD8"],
        resolved_native_marker_selection=[
            {"source_index": 0, "source_name": "DAPI"},
            {"source_index": 1, "source_name": "CD3"},
            {"source_index": 2, "source_name": "CD8"},
        ],
        device="cpu",
    )
    selected = replace(
        base,
        multiplex_markers=["CD8", "DAPI"],
        resolved_native_marker_selection=[
            {"source_index": 2, "source_name": "CD8"},
            {"source_index": 0, "source_name": "DAPI"},
        ],
    )

    old_payload = expected_model_contracts(base)["kronos"]["output_fingerprint"][
        "payload"
    ]["multiplex_panel"]
    new_payload = expected_model_contracts(selected)["kronos"]["output_fingerprint"][
        "payload"
    ]["multiplex_panel"]
    assert old_payload == {
        "binding_contract_version": 1,
        "channel_axis": "c",
        "physical_channel_count": 3,
        "effective_channel_names": ["DAPI", "CD3", "CD8"],
    }
    assert new_payload["binding_contract_version"] == 2
    assert new_payload["selection"] == {
        "mode": "explicit",
        "channels": [
            {"source_index": 2, "source_name": "CD8"},
            {"source_index": 0, "source_name": "DAPI"},
        ],
    }
    assert base.grid_hash() == selected.grid_hash()
    assert base.content_hash() != selected.content_hash()


def test_kronos2_uses_v2_panel_contract_even_for_the_all_channel_default():
    cfg = RunConfig(
        models=["kronos2"],
        no_seg=True,
        resolved_channel_names=["DAPI", "CD3"],
        resolved_native_marker_selection=[
            {"source_index": 0, "source_name": "DAPI"},
            {"source_index": 1, "source_name": "CD3"},
        ],
        device="cpu",
    )
    panel = expected_model_contracts(cfg)["kronos2"]["output_fingerprint"]["payload"][
        "multiplex_panel"
    ]
    assert panel["binding_contract_version"] == 2
    assert panel["selection"]["mode"] == "all"
    assert panel["selection"]["channels"] == cfg.resolved_native_marker_selection
    assert panel["model_params"] == {}


@pytest.mark.parametrize("amp", ["fp16", "bf16"])
def test_kronos2_rejects_explicit_lower_precision_before_source_work(amp):
    with pytest.raises(ValueError, match="published fp32 inference path"):
        RunConfig(models=["kronos2"], amp=amp)


def test_kronos2_rejects_compile_before_source_work():
    with pytest.raises(ValueError, match="does not support --compile"):
        RunConfig(models=["kronos2"], compile=True)


def test_kronos2_legacy_grid_discovery_does_not_build_forbidden_amp_requests():
    no_seg = RunConfig(models=["kronos2"], no_seg=True, patch_px=256)
    assert no_seg.compatible_legacy_grid_hashes()

    segmented = RunConfig(
        models=["kronos2"],
        patch_px=256,
        resolved_channel_names=["DAPI", "CD3"],
        resolved_nuclear_channel_indices=[0],
        resolved_original_channel_names=["DAPI", "CD3"],
    )
    hashes = segmented.compatible_legacy_grid_hashes()
    evidence = segmented.compatible_legacy_grid_segmenters()
    assert hashes
    assert evidence
    assert set(evidence).issubset(hashes)
    assert set(evidence.values()) == {"nuclear"}


@pytest.mark.parametrize("patch_px", [64, 256])
def test_kronos2_accepts_stride_aligned_patch_geometry(patch_px):
    _groups, group_cfgs, _run_hash = resolve_run(
        RunConfig(models=["kronos2"], no_seg=True, patch_px=patch_px),
        requested_patch_px=patch_px,
    )
    assert group_cfgs[0].patch_px == patch_px


def test_kronos2_rejects_unaligned_patch_geometry_before_execution():
    with pytest.raises(ValueError, match="divisible by 16"):
        resolve_run(
            RunConfig(models=["kronos2"], no_seg=True, patch_px=250),
            requested_patch_px=250,
        )


def test_single_grid_primitive_rejects_unaligned_kronos2_before_store_mutation(
    synthetic_multiplex_ngff, tmp_path
):
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="divisible by 16"):
        run_slide(
            synthetic_multiplex_ngff,
            str(out_dir),
            RunConfig(
                models=["kronos2"],
                no_seg=True,
                patch_px=250,
                device="cpu",
            ),
        )

    assert not out_dir.exists()


def test_source_resolution_selects_native_markers_but_keeps_nuclear_full_panel(
    synthetic_multiplex_ngff,
):
    cfg = RunConfig(
        models=["kronos"],
        multiplex_markers=["foxp3", "DAPI"],
        device="cpu",
    )
    resolved = resolve_multiplex_source_config(synthetic_multiplex_ngff, cfg)
    assert resolved.resolved_native_marker_selection == [
        {"source_index": 4, "source_name": "FOXP3"},
        {"source_index": 0, "source_name": "DAPI"},
    ]
    assert resolved.resolved_channel_names == ["DAPI", "CD3", "CD8", "CD20", "FOXP3"]
    assert resolved.resolved_nuclear_channel_indices == [0]


def test_injected_native_model_can_use_marker_selection(
    synthetic_multiplex_ngff,
):
    spec = _native_spec("external_native")
    cfg = RunConfig(
        models=["external_native"],
        multiplex_markers=["FOXP3", "DAPI"],
        device="cpu",
    )

    resolved = resolve_multiplex_source_config(
        synthetic_multiplex_ngff,
        cfg,
        model_specs={"external_native": spec},
    )

    assert resolved.resolved_native_marker_selection == [
        {"source_index": 4, "source_name": "FOXP3"},
        {"source_index": 0, "source_name": "DAPI"},
    ]


def test_mixed_native_and_brightfield_requests_fail_but_native_models_can_coexist():
    with pytest.raises(ValueError, match="cannot be combined"):
        _reject_mixed_native_and_brightfield_models(["kronos", "uni"])
    _reject_mixed_native_and_brightfield_models(["kronos", "kronos2"])


def test_native_parameter_namespaces_are_finite_and_bound_to_requested_models():
    with pytest.raises(ValueError, match="finite JSON"):
        RunConfig(
            models=["kronos2"],
            native_multiplex_params={"kronos2": {"value": float("nan")}},
        )
    with pytest.raises(ValueError, match="not requested native"):
        resolve_multiplex_source_config(
            "unused.zarr",
            RunConfig(
                models=["kronos2"],
                native_multiplex_params={"kronos": {"value": 1}},
            ),
        )


def test_native_parameters_are_filtered_to_each_geometry_group():
    cfg = RunConfig(
        models=["kronos", "kronos2"],
        no_seg=True,
        native_multiplex_params={"kronos2": {"additional_markers": "panel.csv"}},
    )
    _groups, group_cfgs, _run_hash = resolve_run(cfg)
    by_model = {group.models[0]: group for group in group_cfgs}
    assert by_model["kronos"].native_multiplex_params == {}
    assert by_model["kronos2"].native_multiplex_params == {
        "kronos2": {"additional_markers": "panel.csv"}
    }
