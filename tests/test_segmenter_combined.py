"""Tests for the combined (Otsu OR/AND Canny) tissue segmenter.

These use a dedicated *textured* fixture, not the shared ``synthetic_ngff`` smooth
ramp: a ramp has no edges, so Canny returns an empty mask and the AND/OR tests pass
trivially (an all-zero AND satisfies "subset of both"). The fixture below has zones
that make Otsu (saturation) and Canny (edges) both fire AND disagree, so AND is a
non-empty *proper* subset of OR.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.canny import CannySegmenter
from raw2features.segmenters.combined import CombinedSegmenter
from raw2features.segmenters.otsu import OtsuSegmenter


def _zones(h: int, w: int) -> np.ndarray:
    """(h, w, 3) uint8 with four zones: saturated+textured (Otsu AND Canny),
    saturated+smooth (Otsu only), desaturated+textured (Canny only), and a
    desaturated smooth background (neither)."""
    img = np.full((h, w, 3), 205, np.uint8)  # desaturated smooth background, low S
    hw, hh = w // 2, h // 2
    sq = max(2, h // 16)
    # left half: saturated checkerboard -> high saturation + many edges
    yy, xx = np.mgrid[0:h, 0:hw]
    ck = (((yy // sq) + (xx // sq)) % 2).astype(bool)
    img[:, :hw, 0] = np.where(ck, 230, 20)
    img[:, :hw, 1] = 20
    img[:, :hw, 2] = np.where(ck, 20, 230)
    # top-right: saturated smooth red -> Otsu fires, no interior edges for Canny
    img[:hh, hw:, 0] = 230
    img[:hh, hw:, 1] = 15
    img[:hh, hw:, 2] = 15
    # bottom-right: desaturated checkerboard -> Canny fires, low saturation for Otsu
    yy2, xx2 = np.mgrid[0:h - hh, 0:w - hw]
    ck2 = (((yy2 // sq) + (xx2 // sq)) % 2).astype(bool)
    gray = np.where(ck2, 70, 200).astype(np.uint8)
    for c in range(3):
        img[hh:, hw:, c] = gray
    return img


@pytest.fixture
def textured_ngff(tmp_path) -> str:
    """OME-NGFF v0.4 store whose content makes Otsu and Canny both fire and disagree."""
    path = str(tmp_path / "textured.zarr")
    sizes = ((240, 240), (120, 120), (60, 60))
    g = zarr.open_group(path, mode="w", zarr_format=2)
    for i, (h, w) in enumerate(sizes):
        im = _zones(h, w)
        a = g.create_array(
            str(i), shape=(1, 3, 1, h, w),
            chunks=(1, 1, 1, min(64, h), min(64, w)), dtype="uint8",
        )
        a[0, 0, 0], a[0, 1, 0], a[0, 2, 0] = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [
        {"path": str(i), "coordinateTransformations":
            [{"type": "scale", "scale": [1.0, 1.0, 1.0, 0.5 * 2 ** i, 0.5 * 2 ** i]}]}
        for i in range(len(sizes))
    ]
    g.attrs["multiscales"] = [
        {"version": "0.4", "name": "textured", "axes": axes, "datasets": datasets}
    ]
    return path


def test_combined_or_is_superset_of_both(textured_ngff):
    with OmeZarrReader(textured_ngff) as r:
        o = OtsuSegmenter(seg_mpp=2.0).segment(r).mask > 0
        c = CannySegmenter(seg_mpp=2.0).segment(r).mask > 0
        comb = CombinedSegmenter(seg_mpp=2.0, mode="or").segment(r)
        m = comb.mask > 0
    assert comb.mask.dtype == np.float32
    assert o.sum() > 0 and c.sum() > 0  # inputs are non-trivial (Canny not empty)
    assert m.shape == o.shape
    assert np.array_equal(m, o | c)  # OR is exactly the union ...
    assert np.all(m[o]) and np.all(m[c])  # ... so it contains both inputs
    assert m.sum() >= o.sum() and m.sum() >= c.sum()


def test_combined_and_is_subset_of_both(textured_ngff):
    with OmeZarrReader(textured_ngff) as r:
        o = OtsuSegmenter(seg_mpp=2.0).segment(r).mask > 0
        c = CannySegmenter(seg_mpp=2.0).segment(r).mask > 0
        m = CombinedSegmenter(seg_mpp=2.0, mode="and").segment(r).mask > 0
        m_or = CombinedSegmenter(seg_mpp=2.0, mode="or").segment(r).mask > 0
    assert o.sum() > 0 and c.sum() > 0  # inputs non-trivial - guards the false-green
    assert m.sum() > 0  # AND recovers the real overlap, not an all-zero mask
    assert np.array_equal(m, o & c)  # AND is exactly the intersection
    assert m.sum() <= o.sum() and m.sum() <= c.sum()
    assert not np.any(m & ~o) and not np.any(m & ~c)  # AND lies inside both
    assert m.sum() < m_or.sum()  # strictly tighter than OR (Otsu, Canny disagree here)


def test_combined_registered_and_rejects_bad_mode():
    from raw2features.core import plugins

    seg = plugins.get("segmenters", "combined")()  # no-arg construct (pipeline path)
    assert seg.name == "combined"
    with pytest.raises(ValueError):
        CombinedSegmenter(mode="xor")
