"""read_level_capped: the OOM guard for thumbnail / segmentation reads.

A deficient or non-pyramidal slide can have a coarsest level that is still enormous;
reading it whole OOMs. read_level_capped tiles-and-downsamples instead, with factor 1
(read whole) for normal levels so there is no behaviour change there.
"""

from __future__ import annotations

import numpy as np

from conftest import build_ngff_v04
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.otsu import OtsuSegmenter
from raw2features.viz import read_level_capped, render_thumbnail


def test_read_whole_when_level_fits(tmp_path):
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))  # level 0 is 200 x 300
    with OmeZarrReader(slide) as r:
        img, factor = read_level_capped(r, 0)  # well under the cap
        assert factor == 1
        assert img.shape[:2] == (200, 300)


def test_tile_downsamples_when_level_too_big(tmp_path):
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    with OmeZarrReader(slide) as r:
        d = r.level_dimensions[0]  # 300 x 200 (w, h)
        img, factor = read_level_capped(r, 0, max_px=100)  # force the tiled path
        assert factor == int(np.ceil(max(d.width, d.height) / 100))  # == 3
        assert max(img.shape[0], img.shape[1]) <= 100
        # same as a whole read downsampled by `factor` (the tiled path is faithful)
        import cv2
        whole, _ = read_level_capped(r, 0)
        ref = cv2.resize(whole, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_AREA)
        assert np.abs(img.astype(int) - ref.astype(int)).mean() < 12  # ~equal


def test_render_thumbnail_caps_a_non_pyramidal_huge_level(tmp_path):
    # A single-level slide wider than SAFE_READ_PX: the old path read it whole (OOM on a
    # real 100k-px slide); now it is tile-downsampled and the downsample reflects it.
    slide = build_ngff_v04(str(tmp_path / "wide.zarr"), sizes=((200, 13000),))
    with OmeZarrReader(slide) as r:
        thumb = render_thumbnail(r)  # default mpp path, no max_px
        assert max(thumb.image.shape[0], thumb.image.shape[1]) <= 12000
        assert thumb.downsample >= 2.0  # level-0 downsample (1.0) x factor (>=2)


def test_otsu_downsample_tracks_the_cap(tmp_path, monkeypatch):
    # Force the tiled path (cap below the seg level's size) so the mask's downsample
    # picks up the extra factor. otsu imports read_level_capped from viz at call time.
    import raw2features.viz as viz

    real = viz.read_level_capped
    monkeypatch.setattr(
        viz, "read_level_capped",
        lambda reader, level, max_px=40: real(reader, level, 40),
    )
    slide = build_ngff_v04(str(tmp_path / "S.zarr"))
    with OmeZarrReader(slide) as r:
        mask = OtsuSegmenter(seg_mpp=8.0).segment(r)
        assert mask.downsample > r.level_downsamples()[mask.level]  # *= factor
