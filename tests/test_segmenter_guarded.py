"""Tests for the guarded-union segmenter (``otsu | canny`` minus the scanner canvas).

The shared ``synthetic_ngff`` fixture is a smooth ramp with no edges and no canvas, so
assertions about guard behaviour would pass vacuously on it. Every behavioural test here
uses ``canvas_ngff``: a **dark, saturated** canvas block (saturated so Otsu genuinely
claims it -- a pure-black canvas is invisible to a saturation threshold and would let a
broken guard pass) with a separate pale specimen inside the scanned region.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from conftest import build_ngff_v04
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.guarded import GuardedUnionSegmenter
from raw2features.segmenters.otsu import OtsuSegmenter

SIZES = ((200, 300), (100, 150), (50, 75))

# The fixture slide is 200x300 px at 0.5 um/px = 100x150 um TOTAL -- smaller than the
# production guard's 200 um border band, which would kill the whole image. Scale the
# guard to the fixture and segment at the finest level so Canny has pixels to work with.
SEG_MPP = 0.5
GUARD = {"border_um": 4.0, "frame_um": 1.0}


@pytest.fixture
def canvas_ngff(tmp_path) -> str:
    """Dark saturated canvas on the left; a pale textured specimen on the right."""
    path = build_ngff_v04(str(tmp_path / "canvas.zarr"))
    g = zarr.open_group(path, mode="r+", zarr_format=2)
    rng = np.random.default_rng(0)
    for i, (h, w) in enumerate(SIZES):
        # light background
        r = np.full((h, w), 235, np.uint8)
        gr = np.full((h, w), 232, np.uint8)
        b = np.full((h, w), 238, np.uint8)
        # canvas: dark AND saturated (deep blue) so Otsu claims it
        cw = w // 3
        r[:, :cw], gr[:, :cw], b[:, :cw] = 8, 10, 70
        # specimen: pale pink with texture, well inside the scanned region
        y0, y1 = h // 3, 2 * h // 3
        x0, x1 = w // 2, w - w // 8
        tex = rng.integers(0, 26, size=(y1 - y0, x1 - x0), dtype=np.uint8)
        r[y0:y1, x0:x1] = 214 + (tex // 6)
        gr[y0:y1, x0:x1] = 176 + tex
        b[y0:y1, x0:x1] = 202 + (tex // 4)
        a = g[str(i)]
        a[0, 0, 0], a[0, 1, 0], a[0, 2, 0] = r, gr, b
    return path


def _mask(seg, path):
    with OmeZarrReader(path) as r:
        return seg.segment(r)


def test_guarded_returns_valid_binary_tissuemask(canvas_ngff):
    tm = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, **GUARD), canvas_ngff)
    assert tm.mask.dtype == np.float32
    assert set(np.unique(tm.mask)).issubset({0.0, 1.0})
    assert tm.mask.ndim == 2
    assert tm.downsample > 0


def test_guard_removes_the_canvas_that_otsu_claims(canvas_ngff):
    """The load-bearing behaviour: Otsu keeps the dark canvas, the guard removes it."""
    otsu = _mask(OtsuSegmenter(seg_mpp=SEG_MPP), canvas_ngff).mask > 0
    guarded = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, **GUARD), canvas_ngff).mask > 0

    cw = otsu.shape[1] // 3
    otsu_canvas = otsu[:, :cw].mean()
    guarded_canvas = guarded[:, :cw].mean()

    # Otsu must actually claim the canvas, or this fixture proves nothing.
    assert otsu_canvas > 0.5, "fixture too weak: otsu did not claim the canvas"
    assert guarded_canvas < 0.05, "guard failed to remove the canvas"


def test_guard_off_reinstates_the_canvas(canvas_ngff):
    """guard=False must be a real ablation, not a no-op."""
    on = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, guard=True, **GUARD), canvas_ngff).mask > 0
    off = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, guard=False, **GUARD), canvas_ngff).mask > 0
    cw = on.shape[1] // 3
    assert off[:, :cw].mean() > on[:, :cw].mean() + 0.4


def test_guarded_recovers_specimen_otsu_misses_entirely(canvas_ngff):
    """The union half, asserted as a *delta* rather than a tuned absolute level.

    This fixture reproduces the production failure exactly: Otsu claims 100% of the
    saturated canvas and **0%** of the pale specimen (the canvas drags its global
    threshold up). The guarded union must invert both. Asserting ``otsu == 0`` keeps
    the test honest -- if a change ever makes Otsu find the specimen on its own, this
    fails loudly rather than passing for the wrong reason.
    """
    otsu = _mask(OtsuSegmenter(seg_mpp=SEG_MPP), canvas_ngff).mask > 0
    guarded = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, **GUARD), canvas_ngff).mask > 0
    h, w = guarded.shape
    sy, sx = slice(h // 3, 2 * h // 3), slice(w // 2, w - w // 8)

    assert otsu[sy, sx].mean() == 0.0, "fixture too weak: otsu already finds the specimen"
    assert guarded[sy, sx].mean() > 0.05, "union failed to recover the specimen"
    # ...and it is the Canny term, not a relaxed guard, that supplies it.
    assert guarded[sy, sx].sum() > otsu[sy, sx].sum()


def test_guarded_needs_no_grandqc_weights(canvas_ngff, monkeypatch):
    """It must never import/instantiate the GrandQC model (no GPU, no download)."""
    import raw2features.qc.grandqc as gq

    def _boom(*a, **k):  # pragma: no cover - fails the test if reached
        raise AssertionError("guarded must not touch GrandQC")

    monkeypatch.setattr(gq, "GrandQC", _boom)
    tm = _mask(GuardedUnionSegmenter(seg_mpp=SEG_MPP, **GUARD), canvas_ngff)
    assert tm.mask.any()


def test_guarded_records_provenance(canvas_ngff):
    seg = GuardedUnionSegmenter(seg_mpp=SEG_MPP, **GUARD)
    _mask(seg, canvas_ngff)
    assert "kill_frac" in seg.last_stats
    assert seg.last_stats["veto"] == "none"


def test_guarded_is_registered_and_no_arg_constructible():
    from raw2features.core import plugins

    seg = plugins.get("segmenters", "guarded")()
    assert seg.name == "guarded"
