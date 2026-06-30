"""Unit tests for exact-MPP level selection and resample maths."""

from __future__ import annotations

import math

import pytest

from raw2features.core.mpp import level0_step_px, level_for_mpp, level_mpps

# Typical /2 pyramid with 7 levels.
PYRAMID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def test_exact_1mpp_downsamples_from_finer_level():
    # level-0 = 0.1112 um/px; target 1.0 -> level 3 (0.8896), downsample.
    c = level_for_mpp(1.0, 0.1112, PYRAMID, 224)
    assert c.level == 3
    assert math.isclose(c.level_mpp, 0.8896, rel_tol=1e-9)
    assert c.needs_resample is True
    assert math.isclose(c.resample, 1.0 / 0.8896, rel_tol=1e-9)
    assert c.read_px == 252  # round(224 * 1.12416), downscaled to patch_px (224)
    assert math.isclose(c.achieved_mpp, 1.0)  # exact


def test_float_imperfect_level_read_native_within_tolerance():
    # iSyntax-converted scales are slightly off round numbers; a level at
    # 1.0001652 um/px must read natively for target 1.0 (not over-read level 1).
    downsamples = [1.0, 2.000097, 4.0006608]  # -> mpps 0.25, 0.50002, 1.0001652
    c = level_for_mpp(1.0, 0.25, downsamples, 224)
    assert c.level == 2
    assert c.needs_resample is False
    assert c.read_px == 224
    assert abs(c.achieved_mpp - 1.0001652) < 1e-6


def test_quarter_mpp_lands_exactly_on_level():
    # 0.25 um/px source; target 1.0 -> level 2 (1.0), no resample needed.
    c = level_for_mpp(1.0, 0.25, PYRAMID, 224)
    assert c.level == 2
    assert math.isclose(c.level_mpp, 1.0)
    assert c.needs_resample is False
    assert c.read_px == 224
    assert math.isclose(c.achieved_mpp, 1.0)


def test_target_equals_level0_reads_level0_natively():
    c = level_for_mpp(0.25, 0.25, PYRAMID, 224)
    assert c.level == 0
    assert c.needs_resample is False
    assert c.read_px == 224
    assert math.isclose(c.achieved_mpp, 0.25)


def test_never_upsamples_by_default():
    # target finer than level-0 -> must raise unless allow_upsample.
    with pytest.raises(ValueError):
        level_for_mpp(0.25, 0.5, PYRAMID, 224)


def test_allow_upsample_uses_level0():
    c = level_for_mpp(0.25, 0.5, PYRAMID, 224, allow_upsample=True)
    assert c.level == 0
    assert c.resample < 1.0  # upsample
    assert c.read_px == 112  # round(224 * 0.5)
    assert math.isclose(c.achieved_mpp, 0.25)


def test_snap_to_level_reads_native_nearest():
    # target 1.0; nearest level by |mpp - target| is level 3 (0.8896).
    c = level_for_mpp(1.0, 0.1112, PYRAMID, 224, snap_to_level=True)
    assert c.level == 3
    assert c.needs_resample is False
    assert math.isclose(c.achieved_mpp, 0.8896, rel_tol=1e-9)


def test_tolerance_reads_native_when_close_enough():
    # Within 15% of target -> read native (achieved == level mpp).
    c = level_for_mpp(1.0, 0.1112, PYRAMID, 224, tolerance=0.15)
    assert c.level == 3
    assert c.needs_resample is False
    assert math.isclose(c.achieved_mpp, 0.8896, rel_tol=1e-9)
    # Tighter tolerance -> resample to exact.
    c2 = level_for_mpp(1.0, 0.1112, PYRAMID, 224, tolerance=0.05)
    assert c2.needs_resample is True
    assert math.isclose(c2.achieved_mpp, 1.0)


def test_level0_step_px_no_overlap():
    assert level0_step_px(224, 1.0, 0.25) == 896
    assert level0_step_px(224, 1.0, 0.1112) == 2014  # round(2014.388)


def test_level_mpps_helper():
    assert level_mpps(0.1112, [1.0, 2.0, 4.0]) == pytest.approx(
        [0.1112, 0.2224, 0.4448]
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        level_for_mpp(bad, 0.25, PYRAMID, 224)
    with pytest.raises(ValueError):
        level_for_mpp(1.0, bad, PYRAMID, 224)
    with pytest.raises(ValueError):
        level_for_mpp(1.0, 0.25, PYRAMID, int(bad))
