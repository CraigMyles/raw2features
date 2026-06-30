"""Per-patch QC scores from a per-pixel class raster (the GrandQC -> hook step)."""

from __future__ import annotations

import numpy as np

from raw2features.core.qc import patch_qc_scores


def test_patch_qc_scores_class_coverage():
    # 8x8 raster at downsample 2 (level-0 16x16): rows 0-3 class 1, rows 4-7 class 2.
    raster = np.zeros((8, 8), dtype="uint8")
    raster[:4] = 1
    raster[4:] = 2
    coords = np.array([[0, 0], [0, 8], [0, 4]], dtype="int32")  # level-0 (x, y)
    scores = patch_qc_scores(
        coords, level0_patch=8, raster=raster, class_values=[1, 2], raster_downsample=2
    )
    assert scores.shape == (3, 2)
    np.testing.assert_allclose(scores[0], [1.0, 0.0])  # fully in the class-1 band
    np.testing.assert_allclose(scores[1], [0.0, 1.0])  # fully in the class-2 band
    np.testing.assert_allclose(scores[2], [0.5, 0.5])  # straddles the boundary


def test_patch_qc_scores_clips_at_border():
    raster = np.ones((4, 4), dtype="uint8")  # all class 1
    # cell = 2 raster px; a patch at level-0 (6, 6) -> raster (3, 3): the window clips
    # to the 1x1 corner, still all class 1.
    scores = patch_qc_scores(
        np.array([[6, 6]], dtype="int32"), level0_patch=4, raster=raster,
        class_values=[1, 2], raster_downsample=2,
    )
    np.testing.assert_allclose(scores[0], [1.0, 0.0])


def test_patch_qc_scores_empty_coords():
    out = patch_qc_scores(
        np.empty((0, 2), "int32"), 4, np.ones((4, 4), "uint8"),
        [1, 2], raster_downsample=2,
    )
    assert out.shape == (0, 2)


def test_patch_qc_scores_patch_off_raster_is_empty():
    # A patch whose footprint is entirely above/left the raster: cell=2, ry=rx=-4 so the
    # window end is negative -- the old code read a wrong window; now it is empty.
    raster = np.ones((4, 4), "uint8")  # all class 1
    scores = patch_qc_scores(
        np.array([[-8, -8]], "int32"), level0_patch=4, raster=raster,
        class_values=[1, 2], raster_downsample=2,
    )
    np.testing.assert_array_equal(scores[0], [0.0, 0.0])


# -- GrandQC producer wiring (no smp / weights needed for these) ----------------


def test_grandqc_provenance_and_class_vocab():
    # The module imports without the optional smp/weights (lazy inside the forward).
    from raw2features.qc.grandqc import QC_CLASSES, GrandQC

    assert QC_CLASSES[1] == "clean_tissue" and QC_CLASSES[7] == "background"
    prov = GrandQC(device="cpu", artifact_mpp="1.5").provenance()
    assert prov["tool"] == "grandqc"
    assert prov["non_commercial"] is True
    assert prov["model_mpp"] == 1.5
    assert prov["license"] == "CC-BY-NC-SA-4.0"


def test_grandqc_segmenter_registers():
    import raw2features.segmenters.grandqc  # noqa: F401 - importing runs @register
    from raw2features.core.plugins import get

    assert get("segmenters", "grandqc").name == "grandqc"
