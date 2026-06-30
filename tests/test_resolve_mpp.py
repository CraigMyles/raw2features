"""`resolve_target_mpp` - auto-select the patch-extraction MPP from the model cards.

Explicit always wins; a single shared `recommended_mpp` is used; scale-agnostic /
unknown models fall back to the default; models that disagree raise (one run = one
scale). The resolver is pure in (models, requested) so embed/embed-many/verify all
resolve identically and their config hashes line up.
"""

from __future__ import annotations

import pytest

from raw2features.embedders import model_registry as mr
from raw2features.embedders.base import ModelSpec
from raw2features.embedders.model_registry import (
    DEFAULT_TARGET_MPP,
    resolve_target_mpp,
)


def test_explicit_request_always_wins():
    assert resolve_target_mpp(["uni"], 0.25) == (0.25, "explicit")
    # explicit wins even if it disagrees with the card (the warning fires elsewhere).
    assert resolve_target_mpp(["uni", "resnet50"], 2.0) == (2.0, "explicit")


def test_single_foundation_model_auto_resolves_to_its_recommended_mpp():
    mpp, source = resolve_target_mpp(["uni"], None)
    assert source == "auto"
    assert mpp == mr.get_spec("uni").recommended_mpp == 0.5


def test_scale_agnostic_model_falls_back_to_default():
    # resnet50 declares no recommended MPP - it is scale-agnostic.
    assert mr.get_spec("resnet50").recommended_mpp is None
    assert resolve_target_mpp(["resnet50"], None) == (
        DEFAULT_TARGET_MPP,
        "auto-default",
    )


def test_foundation_plus_scale_agnostic_uses_the_foundation_mpp():
    # resnet50 contributes no recommendation, so uni's 0.5 wins for the shared grid.
    assert resolve_target_mpp(["uni", "resnet50"], None) == (0.5, "auto")


def test_unknown_model_is_ignored_not_an_error():
    assert resolve_target_mpp(["mock_not_in_registry"], None) == (
        DEFAULT_TARGET_MPP,
        "auto-default",
    )


def test_conflicting_recommendations_raise(monkeypatch):
    def fake_spec(name, mpp):
        return ModelSpec(
            name=name, family="x", source="x", embedding_dim=1, input_size=1,
            pooling="cls", mean=(0.0,), std=(1.0,), transform_source_url="x",
            license="MIT", gated=False, recommended_mpp=mpp,
        )

    monkeypatch.setattr(
        mr, "load_registry",
        lambda: {"a": fake_spec("a", 0.5), "b": fake_spec("b", 1.0)},
    )
    with pytest.raises(ValueError, match="different MPPs"):
        resolve_target_mpp(["a", "b"], None)
    # explicit still bypasses the conflict.
    assert resolve_target_mpp(["a", "b"], 0.75) == (0.75, "explicit")
