"""Focused contract tests for the channel-wise multiplex strategy."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from raw2features.embedders.base import ModelSpec
from raw2features.embedders.fingerprint import (
    patch_output_fingerprint,
    valid_output_fingerprint,
)
from raw2features.multiplex import build_strategy
from raw2features.multiplex.channelwise import ChannelwiseStrategy
from raw2features.pipeline.runner import _models_header


class TinyRGBEmbedder:
    """A deterministic two-dimensional RGB encoder with no model weights."""

    def __init__(self, *, modality: str = "brightfield") -> None:
        self.spec = ModelSpec(
            name="tiny_rgb",
            family="mock",
            source="mock://tiny-rgb",
            embedding_dim=2,
            input_size=1,
            pooling="cls",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            transform_source_url="https://example.org/tiny-rgb",
            license="MIT",
            gated=False,
            modality=modality,
        )
        self._device = "cpu"
        self._dtype = None
        self.last_rgb: list[np.ndarray] = []

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        return self.spec.modality

    @property
    def embedding_dim(self) -> int:
        return self.spec.embedding_dim

    def transform_batch(self, patches_hwc: list[np.ndarray], device: str):
        assert device == "cpu"
        self.last_rgb = [np.asarray(patch).copy() for patch in patches_hwc]
        intensity = np.asarray(
            [float(patch[..., 0].mean()) / 255.0 for patch in patches_hwc],
            dtype=np.float32,
        )
        values = np.stack([intensity, 1.0 - intensity**2], axis=1)
        try:
            import torch
        except ImportError:
            return values
        return torch.from_numpy(values)

    def embed_batch(self, batch):
        return batch.float().cpu()


class TinyFloatRGBEmbedder(TinyRGBEmbedder):
    @property
    def transform_input_dtype(self) -> str:
        return "float32_0_1"

    def transform_batch(self, patches_hwc: list[np.ndarray], device: str):
        self.last_rgb = [np.asarray(patch).copy() for patch in patches_hwc]
        return np.asarray([[float(patch[..., 0].mean()), 0.0] for patch in patches_hwc])


def _config(
    markers=("A", "B", "C"),
    *,
    normalization="dtype",
    aggregation="mean",
    max_side=2048,
):
    return {
        "multiplex_markers": list(markers) if markers is not None else [],
        "multiplex_normalization": normalization,
        "multiplex_percentile_low": 1.0,
        "multiplex_percentile_high": 99.0,
        "multiplex_aggregation": aggregation,
        "multiplex_normalization_max_side_px": max_side,
    }


def _prepare(
    *,
    panel=("A", "B", "C"),
    channel_count=None,
    config=None,
    base=None,
):
    if channel_count is None and panel is not None:
        channel_count = len(panel)
    return ChannelwiseStrategy().prepare(
        base_embedder=base or TinyRGBEmbedder(),
        channel_names=list(panel) if panel is not None else None,
        channel_count=channel_count,
        config=config or _config(),
    )


def _resolved(prepared, *, values=(0, 128, 255), dtype=np.uint8, level=0):
    values_array = np.asarray(values)
    if prepared.normalization == "percentile":
        # Percentiles are slide-wide per marker, so each channel needs a genuine
        # intensity distribution rather than the single-pixel dtype fixture.
        image = np.stack([values_array, values_array + 20], axis=0)
        image = image.reshape(2, 1, -1).astype(dtype)
    else:
        image = values_array.astype(dtype).reshape(1, 1, -1)
    return prepared.resolve_slide_context(image, level=level, source_dtype=dtype)


def _bound(
    *,
    panel=("A", "B", "C"),
    config=None,
    values=(0, 128, 255),
    dtype=np.uint8,
    base=None,
):
    prepared = _prepare(panel=panel, config=config, base=base)
    return prepared.bind(_resolved(prepared, values=values, dtype=dtype))


def _full_fingerprint(bound, resolved_amp="fp32"):
    base_fingerprint = patch_output_fingerprint(bound.base_embedder.spec, resolved_amp)
    contract = bound.multiplex_fingerprint_payload(base_fingerprint)
    spec = replace(bound.spec, multiplex=contract)
    return patch_output_fingerprint(spec, resolved_amp)


def test_channelwise_is_available_through_the_strategy_registry():
    strategy = build_strategy("channelwise")
    assert isinstance(strategy, ChannelwiseStrategy)
    assert strategy.contract_version == 1


def test_panel_binding_preserves_requested_order_and_physical_indices():
    prepared = _prepare(
        panel=("af", "CD3", "CK", "DAPI"),
        config=_config((" dapi ", "CD3")),
    )

    assert prepared.selected_indices == [3, 1]
    assert prepared.marker_records == [
        {"source_index": 3, "source_name": "DAPI", "canonical_name": "DAPI"},
        {"source_index": 1, "source_name": "CD3", "canonical_name": "CD3"},
    ]

    resolved = prepared.resolve_slide_context(
        np.asarray([[[10, 20, 30, 40]]], dtype=np.uint8),
        level=2,
        source_dtype=np.uint8,
    )
    bound = prepared.bind(resolved)
    assert [row["source_index"] for row in bound.strategy_metadata["markers"]] == [
        3,
        1,
    ]
    assert bound.panel_metadata["kept"] == ["DAPI", "CD3"]


def test_panel_provenance_preserves_exact_source_label_spelling():
    prepared = _prepare(
        panel=("  CD3  ", "DAPI"),
        config=_config(("CD3",)),
    )
    assert prepared.marker_records == [
        {
            "source_index": 0,
            "source_name": "  CD3  ",
            "canonical_name": "CD3",
        }
    ]


def test_explicit_selection_can_exclude_unnamed_physical_channels():
    prepared = _prepare(
        panel=("DAPI", "", "CD3"), config=_config(("CD3", "DAPI"))
    )
    assert prepared.selected_indices == [2, 0]


@pytest.mark.parametrize(
    ("panel", "channel_count", "config", "message"),
    [
        (("A", "B"), 3, _config(("A",)), "does not match the physical C axis"),
        (("A", "B"), 2, _config(("missing",)), "is not present"),
        (("CD3", " cd3 "), 2, _config(("CD3",)), "is ambiguous"),
        (("A", "", "C"), 3, _config(None), "every channel to be named"),
    ],
)
def test_invalid_panel_bindings_fail_closed(panel, channel_count, config, message):
    with pytest.raises(ValueError, match=message):
        _prepare(panel=panel, channel_count=channel_count, config=config)


def test_missing_panel_metadata_fails_closed():
    with pytest.raises(ValueError, match="requires positional marker names"):
        ChannelwiseStrategy().prepare(
            base_embedder=TinyRGBEmbedder(),
            channel_names=None,
            channel_count=3,
            config=_config(),
        )


def test_duplicate_requested_marker_fails_after_identity_normalization():
    with pytest.raises(ValueError, match="requested more than once"):
        _prepare(panel=("CD3", "DAPI"), config=_config(("CD3", " cd3 ")))


def test_non_brightfield_model_cannot_be_used_as_the_base_encoder():
    with pytest.raises(ValueError, match="ordinary RGB encoder"):
        _prepare(base=TinyRGBEmbedder(modality="multiplex"))


def test_unsafe_base_model_name_fails_before_a_zarr_key_is_derived():
    base = TinyRGBEmbedder()
    base.spec = replace(base.spec, name="nested/model")
    with pytest.raises(ValueError, match="safe Zarr path segment"):
        _prepare(base=base)


def test_default_all_marker_panels_have_distinct_cohort_keys():
    first = _bound(
        panel=("CD3", "DAPI"),
        config=_config(None),
        values=(10, 20),
    )
    second = _bound(
        panel=("CK", "DAPI"),
        config=_config(None),
        values=(10, 20),
    )
    assert first.name != second.name


def test_channelwise_rejects_third_party_strategy_params():
    with pytest.raises(ValueError, match="does not accept --multiplex-params"):
        _prepare(
            config={
                **_config(),
                "multiplex_strategy_params": {"adapter": "example"},
            }
        )


def test_reorder_exclusion_and_static_recipe_change_the_output_identity():
    variants = [
        _bound(config=_config(("A", "B", "C"))),
        _bound(config=_config(("C", "B", "A"))),
        _bound(config=_config(("A", "C")), values=(0, 128, 255)),
        _bound(config=_config(("A", "B", "C"), aggregation="concat")),
    ]
    keys = [bound.name for bound in variants]
    digests = [_full_fingerprint(bound)["digest"] for bound in variants]

    assert len(set(keys)) == len(keys)
    assert len(set(digests)) == len(digests)


def test_irrelevant_percentile_bounds_do_not_fragment_dtype_keys():
    dtype_default = _bound(config=_config(normalization="dtype"))
    dtype_other_bounds = _bound(
        config={
            **_config(normalization="dtype"),
            "multiplex_percentile_low": 10.0,
            "multiplex_percentile_high": 90.0,
        }
    )
    assert dtype_default.name == dtype_other_bounds.name


def test_normalization_and_aggregation_change_full_patch_fingerprint():
    percentile = _bound(
        config=_config(normalization="percentile"), values=(0, 100, 200)
    )
    dtype = _bound(config=_config(normalization="dtype"), values=(0, 100, 200))
    concat = _bound(
        config=_config(("A", "B", "C"), normalization="dtype", aggregation="concat"),
        values=(0, 100, 200),
    )

    assert len(
        {
            _full_fingerprint(percentile)["digest"],
            _full_fingerprint(dtype)["digest"],
            _full_fingerprint(concat)["digest"],
        }
    ) == 3


def test_slide_specific_normalization_moves_fingerprint_not_effective_key():
    first = _bound(
        config=_config(normalization="percentile"), values=(0, 100, 200)
    )
    second = _bound(
        config=_config(normalization="percentile"), values=(20, 120, 220)
    )

    assert first.name == second.name
    assert _full_fingerprint(first)["digest"] != _full_fingerprint(second)["digest"]


def test_concat_requires_explicit_order_and_expands_the_output_dimension():
    with pytest.raises(ValueError, match="explicit ordered --marker list"):
        _prepare(config=_config(None, aggregation="concat"))

    bound = _bound(config=_config(("C", "A"), aggregation="concat"))
    assert bound.embedding_dim == 4
    assert bound.spec.embedding_dim == 4
    assert bound.strategy_metadata["aggregation"] == {"name": "concat"}


def test_normalization_max_side_changes_static_and_full_output_identity():
    first = _bound(config=_config(max_side=2048))
    second = _bound(config=_config(max_side=1024))

    assert first.name != second.name
    assert _full_fingerprint(first)["digest"] != _full_fingerprint(second)["digest"]
    assert (
        second.strategy_metadata["normalization"]["level_selection"]["max_side_px"]
        == 1024
    )


def test_channelwise_rejects_unimplemented_attention_aggregation():
    with pytest.raises(ValueError, match="available: mean, concat"):
        _prepare(config=_config(aggregation="attention"))


@pytest.mark.parametrize("aggregation", ["mean", "concat"])
def test_aggregation_matches_the_recorded_numerical_recipe(aggregation):
    torch = pytest.importorskip("torch")
    base = TinyRGBEmbedder()
    bound = _bound(
        base=base,
        config=_config(("A", "B", "C"), aggregation=aggregation),
    )
    patch = np.asarray([[[26, 102, 230]]], dtype=np.uint8)
    transformed = bound.transform_batch([patch], "cpu")
    marker_vectors = transformed[0]
    actual = bound.embed_batch(transformed)

    if aggregation == "mean":
        expected = marker_vectors.mean(dim=0, keepdim=True)
    else:
        expected = marker_vectors.reshape(1, -1)

    assert torch.allclose(actual, expected, atol=1e-7, rtol=0)
    assert all(np.array_equal(rgb[..., 0], rgb[..., 1]) for rgb in base.last_rgb)
    assert all(np.array_equal(rgb[..., 1], rgb[..., 2]) for rgb in base.last_rgb)
    assert [int(rgb[0, 0, 0]) for rgb in base.last_rgb] == [26, 102, 230]


def test_many_markers_chunk_base_transform_and_forward_to_batch_budget():
    torch = pytest.importorskip("torch")

    class TrackingEmbedder(TinyRGBEmbedder):
        def __init__(self):
            super().__init__()
            self.transform_batch_sizes = []
            self.forward_batch_sizes = []

        def transform_batch(self, patches_hwc, device):
            self.transform_batch_sizes.append(len(patches_hwc))
            return super().transform_batch(patches_hwc, device)

        def embed_batch(self, batch):
            self.forward_batch_sizes.append(int(batch.shape[0]))
            return super().embed_batch(batch)

    base = TrackingEmbedder()
    config = {**_config(), "batch_size": 4}
    bound = _bound(base=base, config=config)
    patches = [np.asarray([[[10, 20, 30]]], dtype=np.uint8) for _ in range(3)]
    transformed = bound.transform_batch(patches, "cpu")
    output = bound.embed_batch(transformed)

    assert tuple(output.shape) == (3, 2)
    assert max(base.transform_batch_sizes) <= 4
    assert max(base.forward_batch_sizes) <= 4
    assert sum(base.transform_batch_sizes) == sum(base.forward_batch_sizes) == 9
    assert torch.isfinite(output).all()


def test_percentile_quantization_uses_numpy_ties_to_even_rounding():
    base = TinyRGBEmbedder()
    prepared = _prepare(
        panel=("A",),
        config=_config(("A",), normalization="percentile"),
        base=base,
    )
    resolved = prepared.resolve_slide_context(
        np.asarray([[[0.0]], [[1.0]]], dtype=np.float32),
        level=0,
        source_dtype=np.float32,
    )
    resolved["resolved"][0]["low"] = 0.0
    resolved["resolved"][0]["high"] = 1.0
    bound = prepared.bind(resolved)
    bound.transform_batch([np.asarray([[[2.5 / 255.0]]], dtype=np.float32)], "cpu")
    assert int(base.last_rgb[0][0, 0, 0]) == 2


def test_float_input_transform_receives_unquantized_unit_interval():
    base = TinyFloatRGBEmbedder()
    bound = _bound(
        base=base,
        config=_config(normalization="dtype"),
        dtype=np.uint16,
        values=(0, 32768, 65535),
    )
    bound.transform_batch(
        [np.asarray([[[0, 32768, 65535]]], dtype=np.uint16)], "cpu"
    )

    assert all(rgb.dtype == np.float32 for rgb in base.last_rgb)
    assert [float(rgb[0, 0, 0]) for rgb in base.last_rgb] == pytest.approx(
        [0.0, 32768 / 65535, 1.0]
    )
    assert bound.strategy_metadata["encoder_input_contract"] == "float32_0_1"
    assert bound.strategy_metadata["quantization"]["rounding"] == "none"
    assert "cast_float32_unit_interval" in bound.strategy_metadata["operation_order"]
    assert "quantize_uint8" not in bound.strategy_metadata["operation_order"]


def test_patch_pixels_recheck_nonfinite_and_float_dtype_range():
    percentile = _bound(
        panel=("A",),
        config=_config(("A",), normalization="percentile"),
        values=(0.0,),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="non-finite selected marker pixels"):
        percentile.transform_batch(
            [np.asarray([[[np.nan]]], dtype=np.float32)], "cpu"
        )

    dtype = _bound(
        panel=("A",),
        config=_config(("A",), normalization="dtype"),
        values=(0.5,),
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        dtype.transform_batch([np.asarray([[[1.1]]], dtype=np.float32)], "cpu")


def test_resolved_normalization_rejects_row_count_or_marker_mismatch():
    prepared = _prepare()
    valid = _resolved(prepared)

    with pytest.raises(ValueError, match="does not match selected markers"):
        prepared.bind({**valid, "resolved": valid["resolved"][:-1]})

    wrong = json.loads(json.dumps(valid))
    wrong["resolved"][0]["source_index"] = 2
    with pytest.raises(ValueError, match="does not match"):
        prepared.bind(wrong)


@pytest.mark.parametrize(
    ("low", "high", "message"),
    [
        (float("nan"), 1.0, "finite low < high"),
        (0.0, float("inf"), "finite low < high"),
        (1.0, 1.0, "finite low < high"),
        (2.0, 1.0, "finite low < high"),
    ],
)
def test_bound_normalization_rejects_nonfinite_and_degenerate_ranges(
    low, high, message
):
    prepared = _prepare()
    resolved = _resolved(prepared)
    resolved["resolved"][0]["low"] = low
    resolved["resolved"][0]["high"] = high
    with pytest.raises(ValueError, match=message):
        prepared.bind(resolved)


def test_resolving_normalization_rejects_nonfinite_or_constant_marker():
    prepared = _prepare(
        panel=("A",),
        config=_config(("A",), normalization="percentile"),
    )
    with pytest.raises(ValueError, match="non-finite pixels"):
        prepared.resolve_slide_context(
            np.full((2, 2, 1), np.nan, dtype=np.float32),
            level=0,
            source_dtype=np.float32,
        )
    with pytest.raises(ValueError, match="degenerate normalization"):
        prepared.resolve_slide_context(
            np.ones((2, 2, 1), dtype=np.float32),
            level=0,
            source_dtype=np.float32,
        )


def test_dtype_normalization_handles_signed_range_and_rejects_unscaled_float():
    signed = _prepare(config=_config(normalization="dtype"))
    resolved = signed.resolve_slide_context(
        np.zeros((1, 1, 3), dtype=np.int16),
        level=0,
        source_dtype=np.int16,
    )
    assert resolved["resolved"][0]["low"] == np.iinfo(np.int16).min
    assert resolved["resolved"][0]["high"] == np.iinfo(np.int16).max

    floating = _prepare(config=_config(normalization="dtype"))
    with pytest.raises(ValueError, match=r"requires finite values in \[0, 1\]"):
        floating.resolve_slide_context(
            np.full((1, 1, 3), 2.0, dtype=np.float32),
            level=0,
            source_dtype=np.float32,
        )


def test_bound_panel_cannot_be_reused_with_changed_source_metadata():
    bound = _bound(panel=("A", "B", "C"))
    assert bound.set_panel(["A", "B", "C"])["n_markers"] == 3
    with pytest.raises(ValueError, match="changed after channelwise strategy binding"):
        bound.set_panel(["A", "C", "B"])


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (np.zeros((2, 2), dtype=np.uint8), "HWC"),
        (np.zeros((2, 2, 2), dtype=np.uint8), "C=3"),
        (np.zeros((2, 2, 3), dtype=np.uint16), "dtype"),
    ],
)
def test_patch_shape_channel_count_and_dtype_must_match_bound_source(patch, message):
    bound = _bound(dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        bound.transform_batch([patch], "cpu")


def test_deriving_channelwise_output_does_not_mutate_brightfield_fingerprint():
    base = TinyRGBEmbedder()
    before = patch_output_fingerprint(base.spec, "fp32")
    before_bytes = json.dumps(before, sort_keys=True, separators=(",", ":")).encode()

    derived = _bound(base=base)
    derived_fingerprint = _full_fingerprint(derived)

    after = patch_output_fingerprint(base.spec, "fp32")
    after_bytes = json.dumps(after, sort_keys=True, separators=(",", ":")).encode()
    assert after_bytes == before_bytes
    assert base.spec.multiplex is None
    assert derived_fingerprint != before


def test_readable_model_header_redacts_injected_base_credentials():
    base = TinyRGBEmbedder()
    base.spec = replace(
        base.spec,
        source=(
            "https://user:password@private.example/model?revision=v1&token=SENTINEL"
        ),
        transform_source_url=(
            "https://private.example/transform.py?api_key=SECOND_SENTINEL"
        ),
    )
    bound = _bound(base=base)
    fingerprint = _full_fingerprint(bound)
    bound.spec = replace(
        bound.spec,
        multiplex=bound.multiplex_fingerprint_payload(
            patch_output_fingerprint(base.spec, "fp32")
        ),
    )

    header = _models_header(
        [bound],
        {
            bound.name: {
                "embedding_dim": bound.embedding_dim,
                "output_fingerprint": fingerprint,
            }
        },
    )[bound.name]
    persisted = json.dumps(header, sort_keys=True)

    assert "password" not in persisted
    assert "SENTINEL" not in persisted
    assert "SECOND_SENTINEL" not in persisted
    assert header["source"] == "https://private.example/model?revision=v1"
    assert header["transform_source_url"] == "https://private.example/transform.py"
    assert valid_output_fingerprint(header["output_fingerprint"])


def test_model_name_that_looks_like_a_secret_key_is_not_redacted():
    base = TinyRGBEmbedder()
    base.spec = replace(base.spec, name="token")
    fingerprint = patch_output_fingerprint(base.spec, "fp32")

    header = _models_header(
        [base],
        {
            "token": {
                "embedding_dim": base.embedding_dim,
                "output_fingerprint": fingerprint,
            }
        },
    )

    assert isinstance(header["token"], dict)
    assert valid_output_fingerprint(header["token"]["output_fingerprint"])
