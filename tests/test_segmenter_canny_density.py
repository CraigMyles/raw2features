"""Tests for the per-patch Canny edge-density segmenter."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from conftest import build_ngff_v04
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.canny import CannySegmenter
from raw2features.segmenters.canny_density import CannyDensitySegmenter


@pytest.fixture
def edged_ngff(tmp_path) -> str:
    """A synthetic store containing a hard-edged block.

    The shared ``synthetic_ngff`` fixture is a smooth ramp, which by construction has
    no Canny edges -- fine for shape/dtype assertions, useless for edge behaviour. This
    paints a solid dark square on a light field so there is a real contour to find.
    """
    path = build_ngff_v04(str(tmp_path / "edged.zarr"))
    g = zarr.open_group(path, mode="r+", zarr_format=2)
    for i, (h, w) in enumerate(((200, 300), (100, 150), (50, 75))):
        a = g[str(i)]
        for c in range(3):
            plane = np.full((h, w), 230, dtype="uint8")
            plane[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 30  # hard-edged block
            a[0, c, 0] = plane
    return path


def test_canny_density_returns_binary_edge_map_at_level(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        # mpps are [0.5, 1.0, 2.0]; seg_mpp 2.0 -> level 2.
        tm = CannyDensitySegmenter(seg_mpp=2.0).segment(r)
        assert tm.level == 2
        assert tm.mask.ndim == 2
        assert tm.mask.dtype == np.float32
        assert set(np.unique(tm.mask)).issubset({0.0, 1.0})
        dim = r.level_dimensions[tm.level]
        assert tm.mask.shape == (dim.height, dim.width)
        assert tm.downsample == 4.0


def test_canny_density_finds_edges_but_does_not_fill_them(edged_ngff):
    """The whole point: keep the thin edge map, never fill it into a solid region."""
    with OmeZarrReader(edged_ngff) as r:
        density = CannyDensitySegmenter(seg_mpp=2.0).segment(r).mask
        filled = CannySegmenter(seg_mpp=2.0).segment(r).mask

    assert density.mean() > 0.0, "should detect the block's edges"
    # The block covers ~1/4 of the field; a *filled* mask approaches that, while an
    # edge map stays a thin outline. This is the property that stops background
    # between tissue fragments from being tiled.
    assert density.mean() < filled.mean()
    assert density.mean() < 0.15


def test_canny_density_dilate_thickens_edges(edged_ngff):
    with OmeZarrReader(edged_ngff) as r:
        thin = CannyDensitySegmenter(seg_mpp=2.0, dilate=0).segment(r).mask
        thick = CannyDensitySegmenter(seg_mpp=2.0, dilate=3).segment(r).mask
    assert thick.mean() > thin.mean()


def test_canny_density_lower_threshold_is_more_sensitive(edged_ngff):
    with OmeZarrReader(edged_ngff) as r:
        low = CannyDensitySegmenter(seg_mpp=2.0, low=0.02, high=0.06).segment(r).mask
        high = CannyDensitySegmenter(seg_mpp=2.0, low=0.5, high=0.9).segment(r).mask
    assert low.mean() >= high.mean()


def test_canny_density_is_registered_and_no_arg_constructible():
    from raw2features.core import plugins

    seg = plugins.get("segmenters", "canny_density")()  # pipeline builds with no args
    assert seg.name == "canny_density"
