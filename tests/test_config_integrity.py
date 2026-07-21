"""Guards against silent config-hash drift (the skip-if-complete correctness key).

- content_hash is pinned to known values so a refactor can't change the bytes.
- _CONTENT_FIELDS/_RUNTIME_FIELDS must partition every RunConfig field, so a new
  field has to be classified consciously (not silently ignored by the hash).
- embed and verify must agree on every shared content-affecting option default,
  or the SLURM array's verify-based skip would compute a different hash to embed.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, replace

import pytest

from raw2features.cli.embed import embed
from raw2features.cli.embed_many import embed_many
from raw2features.cli.verify import verify
from raw2features.pipeline.runner import RunConfig


def test_content_hash_is_pinned():
    # If these change, skip-if-complete silently invalidates every prior receipt.
    # Changing any hashed field changes these values, so update the expected hashes
    # deliberately.
    assert RunConfig(models=["resnet50"]).content_hash() == "a8c12b66d8558e1b"
    assert (
        RunConfig(
            models=["uni", "resnet50"], no_seg=True, target_mpp=1.0
        ).content_hash()
        == "9be3d1202bcfe08d"
    )
    assert (
        RunConfig(
            models=["dinov2"],
            segmenter="otsu",
            target_mpp=0.5,
            patch_px=256,
            step_px=128,
            tissue_threshold=0.2,
            features_dtype="float32",
            snap_to_level=True,
            mpp_tolerance=0.0,
            allow_upsample=True,
            amp="fp16",
        ).content_hash()
        == "7af8f129093bb671"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_mpp", 0),
        ("target_mpp", float("nan")),
        ("source_mpp", -0.5),
        ("patch_px", 0),
        ("step_px", -1),
    ],
)
def test_runconfig_rejects_invalid_geometry_at_construction(field, value):
    with pytest.raises(ValueError, match=field):
        RunConfig(models=["resnet50"], **{field: value})


def test_content_hash_is_model_order_independent():
    a = RunConfig(models=["uni", "resnet50"]).content_hash()
    b = RunConfig(models=["resnet50", "uni"]).content_hash()
    assert a == b


def test_grid_hash_is_model_independent_but_geometry_sensitive():
    # grid_hash identifies the patch geometry, so it must NOT depend on the model
    # set (that is what lets a new model be appended to an existing store) but MUST
    # change when any geometry field changes.
    base = RunConfig(models=["resnet50"]).grid_hash()
    assert base == RunConfig(models=["resnet50", "uni"]).grid_hash()
    assert base == RunConfig(models=["dinov2"]).grid_hash()
    assert base != RunConfig(models=["resnet50"], target_mpp=0.5).grid_hash()
    assert base != RunConfig(models=["resnet50"], patch_px=256).grid_hash()
    # and it is distinct from the full content hash (which includes the models)
    assert base != RunConfig(models=["resnet50"]).content_hash()


def test_multiplex_settings_affect_receipts_but_never_grid_identity():
    # Disable segmentation so only strategy settings vary; the default channelwise
    # path deliberately uses nuclear rather than brightfield Otsu geometry.
    brightfield = RunConfig(models=["uni"], no_seg=True)
    multiplex = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="channelwise",
        multiplex_markers=["CD3", "DAPI"],
        multiplex_aggregation="mean",
    )
    reordered = replace(multiplex, multiplex_markers=["DAPI", "CD3"])

    assert multiplex.grid_hash() == brightfield.grid_hash()
    assert reordered.grid_hash() == brightfield.grid_hash()
    assert multiplex.content_hash() != brightfield.content_hash()
    assert reordered.content_hash() != multiplex.content_hash()


def test_brightfield_identities_are_unchanged_by_multiplex_support():
    """Release guard: ordinary UNI stores/resume keep their exact v0.1 identity."""
    from raw2features.embedders.fingerprint import patch_output_fingerprint
    from raw2features.embedders.model_registry import get_spec

    cfg = RunConfig(models=["uni"])
    assert cfg.grid_hash() == "a7410e019eac5b45"
    assert cfg.content_hash() == "5543560bf2dc9a3a"
    assert patch_output_fingerprint(get_spec("uni"), "fp32")["digest"] == (
        "4f6b83b5f6958108f0f68c24164c6dad7906a297982248a11472a7b1bec46680"
    )


def test_native_multiplex_accepts_only_its_historical_otsu_grid_alias():
    """Safe native legacy families require evidence of nuclear execution."""

    legacy_otsu = RunConfig(models=["uni"]).grid_hash()
    native = RunConfig(
        models=["kronos"],
        resolved_channel_names=["DAPI", "CD3"],
        resolved_nuclear_channel_indices=[0],
        resolved_original_channel_names=["DAPI", "CD3"],
    )
    derived = RunConfig(models=["uni"], multiplex_strategy="channelwise")
    legacy_nuclear = native.legacy_grid_hash()

    assert native.segmenter == "nuclear"
    assert legacy_otsu in native.compatible_legacy_grid_hashes()
    assert legacy_nuclear in native.compatible_legacy_grid_hashes()
    assert native.compatible_legacy_grid_segmenters()[legacy_otsu] == "nuclear"
    assert native.compatible_legacy_grid_segmenters()[legacy_nuclear] == "nuclear"
    assert native.allows_hashless_legacy_grid()
    assert legacy_otsu not in derived.compatible_legacy_grid_hashes()
    assert not derived.allows_hashless_legacy_grid()
    assert RunConfig(models=["uni"]).allows_hashless_legacy_grid()


@pytest.mark.parametrize(
    ("original_names", "current_indices"),
    [
        (["DNA1", "DNA2", "CD3"], [0, 1]),
        (["DNA-PKcs", "DAPI", "CD3"], [1]),
        (["", "DAPI", "CD3"], [1]),
    ],
)
def test_native_multiplex_rejects_unproven_historical_nuclear_grids(
    original_names, current_indices
):
    legacy_otsu = RunConfig(models=["uni"]).grid_hash()
    native = RunConfig(
        models=["kronos"],
        resolved_channel_names=original_names,
        resolved_nuclear_channel_indices=current_indices,
        resolved_original_channel_names=original_names,
    )
    legacy_nuclear = native.legacy_grid_hash()

    assert legacy_otsu not in native.compatible_legacy_grid_hashes()
    assert legacy_nuclear not in native.compatible_legacy_grid_hashes()
    assert native.compatible_legacy_grid_segmenters() == {}
    assert not native.allows_hashless_legacy_grid()


def test_multiplex_settings_move_content_identity_but_not_grid_identity():
    """A strategy changes model output/receipt identity, never patch geometry."""
    base = RunConfig(models=["uni"], no_seg=True)
    multiplex = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="channelwise",
        multiplex_markers=["CD3", "CK"],
        multiplex_normalization="percentile",
        multiplex_percentile_low=1.0,
        multiplex_percentile_high=99.0,
        multiplex_aggregation="mean",
        multiplex_normalization_max_side_px=2048,
    )
    reordered = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="channelwise",
        multiplex_markers=["CK", "CD3"],
    )

    assert multiplex.grid_hash() == base.grid_hash()
    assert multiplex.content_hash() != base.content_hash()
    assert reordered.content_hash() != multiplex.content_hash()


def test_multiplex_actual_nuclear_segmentation_has_distinct_grid_identity():
    brightfield = RunConfig(models=["uni"])
    strategy = RunConfig(models=["uni"], multiplex_strategy="channelwise")
    native = RunConfig(models=["kronos"])

    assert brightfield.segmenter == "otsu"
    assert strategy.segmenter == native.segmenter == "nuclear"
    assert strategy.grid_hash() == native.grid_hash()
    assert strategy.grid_hash() != brightfield.grid_hash()


def test_runconfig_preserves_the_v01_positional_constructor_prefix():
    expected_prefix = [
        "models",
        "reader",
        "segmenter",
        "no_seg",
        "target_mpp",
        "source_mpp",
        "patch_px",
        "step_px",
        "tissue_threshold",
        "features_dtype",
        "stain_norm",
        "snap_to_level",
        "mpp_tolerance",
        "allow_upsample",
        "amp",
    ]
    assert [field.name for field in fields(RunConfig)[:15]] == expected_prefix

    cfg = RunConfig(
        ["uni"],
        "omezarr",
        "otsu",
        True,
        0.5,
        None,
        128,
        64,
        0.2,
        "float32",
        None,
        True,
        0.01,
        True,
        "bf16",
    )
    assert cfg.snap_to_level is True
    assert cfg.mpp_tolerance == pytest.approx(0.01)
    assert cfg.allow_upsample is True
    assert cfg.amp == "bf16"
    assert cfg.multiplex_strategy is None


def test_irrelevant_multiplex_knobs_do_not_move_receipt_identity():
    mean = RunConfig(models=["uni"], multiplex_strategy="channelwise")
    dtype = replace(mean, multiplex_normalization="dtype")
    changed_bounds = replace(
        dtype,
        multiplex_percentile_low=10.0,
        multiplex_percentile_high=90.0,
    )
    assert dtype.content_hash() == changed_bounds.content_hash()


def test_channel_identity_affects_outputs_but_only_nuclear_index_affects_grid():
    first = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="channelwise",
        resolved_channel_names=["CD3", "DAPI"],
    )
    renamed = replace(first, resolved_channel_names=["CK", "DAPI"])
    assert first.content_hash() != renamed.content_hash()
    assert first.grid_hash() == renamed.grid_hash()

    segmented = replace(
        first,
        no_seg=False,
        resolved_nuclear_channel_indices=[1],
    )
    nonnuclear_rename = replace(segmented, resolved_channel_names=["CK", "DAPI"])
    moved_nucleus = replace(segmented, resolved_nuclear_channel_indices=[0])
    paired_nucleus = replace(segmented, resolved_nuclear_channel_indices=[0, 1])
    different_metadata_origin = replace(
        segmented, resolved_original_channel_names=["", "DAPI"]
    )
    assert segmented.grid_hash() == nonnuclear_rename.grid_hash()
    assert segmented.grid_hash() != moved_nucleus.grid_hash()
    assert segmented.grid_hash() != paired_nucleus.grid_hash()
    assert segmented.content_hash() == different_metadata_origin.content_hash()
    assert segmented.grid_hash() == different_metadata_origin.grid_hash()


def test_normalization_max_side_is_content_not_grid_identity():
    first = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="channelwise",
        multiplex_normalization_max_side_px=2048,
    )
    second = replace(first, multiplex_normalization_max_side_px=1024)
    assert first.content_hash() != second.content_hash()
    assert first.grid_hash() == second.grid_hash()

    with pytest.raises(ValueError, match="multiplex_normalization_max_side_px"):
        replace(first, multiplex_normalization_max_side_px=0)


def test_namespaced_strategy_params_are_content_not_grid_identity():
    first = RunConfig(
        models=["uni"],
        no_seg=True,
        multiplex_strategy="third_party",
        multiplex_strategy_params={"adapter": {"width": 3}},
    )
    second = replace(
        first,
        multiplex_strategy_params={"adapter": {"width": 4}},
    )
    assert first.grid_hash() == second.grid_hash()
    assert first.content_hash() != second.content_hash()


def test_strategy_params_must_be_finite_json_with_string_keys():
    with pytest.raises(ValueError, match="string keys"):
        RunConfig(
            models=["uni"],
            multiplex_strategy="third_party",
            multiplex_strategy_params={1: "bad"},
        )
    with pytest.raises(ValueError, match="finite JSON"):
        RunConfig(
            models=["uni"],
            multiplex_strategy="third_party",
            multiplex_strategy_params={"value": float("nan")},
        )


def test_amp_auto_is_default_and_resolves_per_model():
    from raw2features.embedders.model_registry import get_spec
    from raw2features.pipeline.runner import _resolve_amp

    cfg = RunConfig(models=["uni"])
    assert cfg.amp == "auto"  # follow each provider's card by default
    # models without an inference_amp field default to fp32 (uni, resnet50) ...
    assert _resolve_amp(cfg, get_spec("uni")) == "fp32"
    assert _resolve_amp(cfg, get_spec("resnet50")) == "fp32"
    # ... while models whose card recommends fp16 resolve to fp16 under auto
    # (exercises the non-trivial per-model branch, not just the fp32 default).
    assert _resolve_amp(cfg, get_spec("virchow2")) == "fp16"
    assert _resolve_amp(cfg, get_spec("h_optimus_0")) == "fp16"
    assert _resolve_amp(cfg, get_spec("h0_mini")) == "fp16"
    # an explicit --amp overrides the per-model card precision
    assert (
        _resolve_amp(RunConfig(models=["uni"], amp="bf16"), get_spec("uni")) == "bf16"
    )


def test_runconfig_fields_are_partitioned():
    content = set(RunConfig._CONTENT_FIELDS)
    runtime = set(RunConfig._RUNTIME_FIELDS)
    all_fields = {f.name for f in fields(RunConfig)}
    assert content.isdisjoint(runtime)
    # every field must be classified as either content-affecting or runtime-only
    assert content | runtime == all_fields, (
        f"unclassified RunConfig fields: {all_fields - content - runtime}"
    )


def _opt_default(func, name):
    opt = inspect.signature(func).parameters[name].default
    return getattr(opt, "default", opt)  # unwrap typer OptionInfo


def test_embed_and_verify_share_content_option_defaults():
    e = set(inspect.signature(embed).parameters)
    v = set(inspect.signature(verify).parameters)
    # Exclude paths: verify's optional --out-dir binds the receipt target, while embed
    # requires it positionally; neither it nor the receipt path feeds the content hash.
    shared = (e & v) - {"slide", "out_dir", "receipts_dir"}
    assert shared, "expected embed/verify to share content options"
    for name in sorted(shared):
        assert _opt_default(embed, name) == _opt_default(verify, name), (
            f"default drift for --{name}: embed={_opt_default(embed, name)!r} "
            f"verify={_opt_default(verify, name)!r}"
        )


def test_embed_and_embed_many_share_content_option_defaults():
    # embed-many (the warm worker) must hash identically to embed for the same
    # flags, or a cohort run's receipts wouldn't be recognised by a later resume.
    e = set(inspect.signature(embed).parameters)
    m = set(inspect.signature(embed_many).parameters)
    # content-affecting options both expose (exclude paths / non-content knobs).
    content = {
        "feature_extractor",
        "mpp",
        "patch_size",
        "step",
        "reader",
        "segmenter",
        "no_seg",
        "tissue_threshold",
        "features_dtype",
        "multiplex_strategy",
        "multiplex_markers",
        "multiplex_normalization",
        "multiplex_percentile_low",
        "multiplex_percentile_high",
        "multiplex_aggregation",
        "multiplex_normalization_max_side_px",
        "amp",
        "snap_to_level",
        "mpp_tolerance",
        "allow_upsample",
    }
    shared = (e & m) & content
    assert shared, "expected embed/embed-many to share content options"
    for name in sorted(shared):
        assert _opt_default(embed, name) == _opt_default(embed_many, name), (
            f"default drift for --{name}: embed={_opt_default(embed, name)!r} "
            f"embed_many={_opt_default(embed_many, name)!r}"
        )
