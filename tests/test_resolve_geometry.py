"""`resolve_geometry` - group requested models into per-model extraction grids.

With no `--mpp`/`--patch-size`, each model is extracted at its own card geometry
(`recommended_mpp`, `recommended_patch_px` or `input_size`); models that disagree
get SEPARATE grids instead of raising. A bare `--mpp` preserves model patch sizes;
supplying both `--mpp` and `--patch-size` collapses everything onto one grid. A config
controls geometry per model (incl. repeats).
Pure in (models, mpp, patch_px, config) so every CLI resolves identically.
"""

from __future__ import annotations

import pytest

from raw2features.embedders.base import ModelSpec
from raw2features.embedders.model_registry import (
    DEFAULT_TARGET_MPP,
    get_spec,
    model_geometry,
    resolve_geometry,
)


def _geom_by_models(groups):
    """Map frozenset(models) -> (mpp, patch_px) for order-independent assertions."""
    return {frozenset(g.models): (g.mpp, g.patch_px) for g in groups}


# --- model_geometry --------------------------------------------------------


def test_foundation_model_uses_recommended_mpp_and_input_size():
    assert model_geometry("uni") == (0.5, 224, "recommended")


def test_conch_v1_5_extracts_larger_then_resizes():
    # Its card extracts 512 px @ 20x and resizes to the 448 input -> grid px is 512.
    spec = get_spec("conch_v1_5")
    assert spec.recommended_patch_px == 512 and spec.input_size == 448
    assert model_geometry("conch_v1_5") == (0.5, 512, "recommended")


@pytest.mark.parametrize("name", ["gigapath", "gigapath_flash"])
def test_gigapath_models_extract_256_then_center_crop(name):
    spec = get_spec(name)
    assert spec.input_size == 224
    assert spec.recommended_patch_px == 256
    assert spec.crop_pct == 0.875
    assert spec.crop_mode == "center"
    assert model_geometry(name) == (0.5, 256, "recommended")


def test_scale_agnostic_baseline_defaults():
    assert model_geometry("resnet50") == (DEFAULT_TARGET_MPP, 224, "default")


def _external_spec(
    *, name="external", recommended_mpp=0.75, recommended_patch_px=96
):
    return ModelSpec(
        name=name,
        family="external",
        source="external://weights",
        embedding_dim=8,
        input_size=64,
        pooling="cls",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        transform_source_url="https://example.org/external",
        license="MIT",
        gated=False,
        recommended_mpp=recommended_mpp,
        recommended_patch_px=recommended_patch_px,
    )


def test_unknown_model_requires_an_injected_specification():
    with pytest.raises(ValueError, match="not in the registry or supplied"):
        model_geometry("not_a_real_model")


def test_external_model_geometry_comes_from_its_injected_specification():
    spec = _external_spec()
    assert model_geometry("external", specs={"external": spec}) == (
        0.75,
        96,
        "recommended",
    )
    groups = resolve_geometry(["external"], specs={"external": spec})
    assert [(g.mpp, g.patch_px, g.models) for g in groups] == [
        (0.75, 96, ("external",))
    ]


# --- per-model grouping (no override) --------------------------------------


def test_same_geometry_models_share_one_grid():
    groups = resolve_geometry(["uni", "virchow2"])  # both 0.5/224
    assert len(groups) == 1
    assert (groups[0].mpp, groups[0].patch_px) == (0.5, 224)
    assert set(groups[0].models) == {"uni", "virchow2"}
    assert groups[0].source == "recommended"


def test_disagreeing_models_get_separate_grids_not_an_error():
    # uni 0.5/224, conch 0.5/448, retccl 1.0/256 -> three grids. This is the feature:
    # today resolve_target_mpp would raise on uni+retccl.
    groups = resolve_geometry(["uni", "conch", "retccl"])
    g = _geom_by_models(groups)
    assert g[frozenset({"uni"})] == (0.5, 224)
    assert g[frozenset({"conch"})] == (0.5, 448)
    assert g[frozenset({"retccl"})] == (1.0, 256)


def test_baseline_grouped_separately_from_foundation():
    # resnet50 -> 1.0/224, uni -> 0.5/224 : two grids (baseline never inherits 0.5).
    g = _geom_by_models(resolve_geometry(["uni", "resnet50"]))
    assert g[frozenset({"uni"})] == (0.5, 224)
    assert g[frozenset({"resnet50"})] == (1.0, 224)


def test_per_model_group_order_is_request_order():
    groups = resolve_geometry(["resnet50", "uni"])
    assert groups[0].models == ("resnet50",)  # first-requested geometry first


def test_duplicate_model_names_dedup_within_a_grid():
    groups = resolve_geometry(["uni", "uni"])
    assert len(groups) == 1 and groups[0].models == ("uni",)


def test_high_level_run_resolution_rejects_an_empty_request():
    from raw2features.pipeline.runner import RunConfig, resolve_run

    with pytest.raises(ValueError, match="at least one model/extraction is required"):
        resolve_run(RunConfig(models=[]))


# --- explicit geometry overrides ------------------------------------------


def test_explicit_mpp_preserves_each_models_recommended_patch_size():
    groups = resolve_geometry(["uni", "conch", "retccl"], requested_mpp=0.5)
    assert _geom_by_models(groups) == {
        frozenset({"uni"}): (0.5, 224),
        frozenset({"conch"}): (0.5, 448),
        frozenset({"retccl"}): (0.5, 256),
    }
    assert all(group.source == "explicit" for group in groups)


def test_explicit_mpp_and_patch_size_collapse():
    groups = resolve_geometry(
        ["uni", "conch"], requested_mpp=0.5, requested_patch_px=512
    )
    assert len(groups) == 1
    assert (groups[0].mpp, groups[0].patch_px) == (0.5, 512)


def test_patch_size_only_resolves_shared_mpp():
    # only --patch-size: the shared mpp is resolved from the (agreeing) cards.
    groups = resolve_geometry(["uni", "virchow2"], requested_patch_px=256)
    assert len(groups) == 1
    assert (groups[0].mpp, groups[0].patch_px) == (0.5, 256)


# --- config (per-model, incl. the same model several times) ----------------


def test_config_same_model_at_multiple_mpps():
    cfg = [
        {"model": "uni", "mpp": 0.5},
        {"model": "uni", "mpp": 1.0},
        {"model": "uni", "mpp": 0.25},
    ]
    groups = resolve_geometry(["uni"], config=cfg)
    assert sorted((g.mpp, g.patch_px) for g in groups) == [
        (0.25, 224),
        (0.5, 224),
        (1.0, 224),
    ]
    assert all(g.models == ("uni",) and g.source == "config" for g in groups)


def test_config_defaults_from_registry_when_omitted():
    groups = resolve_geometry([], config=[{"model": "conch"}])
    assert len(groups) == 1
    assert (groups[0].mpp, groups[0].patch_px) == (0.5, 448)


def test_config_merges_same_geometry_into_one_grid():
    groups = resolve_geometry([], config=[{"model": "uni"}, {"model": "virchow2"}])
    assert len(groups) == 1
    assert set(groups[0].models) == {"uni", "virchow2"}
