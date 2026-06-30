"""Tests for the Canny edge-density tissue segmenter against the synthetic fixture."""

from __future__ import annotations

import numpy as np

from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.canny import CannySegmenter


def test_canny_returns_binary_mask_at_level(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        # mpps are [0.5, 1.0, 2.0]; seg_mpp 2.0 -> level 2.
        tm = CannySegmenter(seg_mpp=2.0).segment(r)
        assert tm.level == 2
        assert tm.mask.ndim == 2
        assert tm.mask.dtype == np.float32
        assert set(np.unique(tm.mask)).issubset({0.0, 1.0})
        dim = r.level_dimensions[tm.level]
        assert tm.mask.shape == (dim.height, dim.width)
        assert tm.downsample == 4.0


def test_canny_is_registered_and_no_arg_constructible():
    from raw2features.core import plugins

    seg = plugins.get("segmenters", "canny")()  # built with no args by the pipeline
    assert seg.name == "canny"
