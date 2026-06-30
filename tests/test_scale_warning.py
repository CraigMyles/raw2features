"""The target_mpp vs default-MPP scale note (weight-free, informational)."""

from __future__ import annotations

import warnings

import pytest

from raw2features.embedders.model_registry import get_spec
from raw2features.pipeline.runner import RunConfig, _warn_scale_mismatch


def test_warns_when_extraction_scale_differs():
    cfg = RunConfig(models=["uni"], target_mpp=1.0)  # uni's default is 0.5 (20x)
    # An informational note (not a "you're wrong"): names the chosen + the common scale.
    with pytest.warns(UserWarning, match="commonly run at"):
        _warn_scale_mismatch(cfg, get_spec("uni"))


def test_no_warning_when_scale_matches():
    cfg = RunConfig(models=["uni"], target_mpp=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        _warn_scale_mismatch(cfg, get_spec("uni"))


def test_no_warning_for_scale_agnostic_baseline():
    # resnet50 / dinov2 carry no recommended_mpp -> never warn, at any target_mpp.
    cfg = RunConfig(models=["resnet50"], target_mpp=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_scale_mismatch(cfg, get_spec("resnet50"))
