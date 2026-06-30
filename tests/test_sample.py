"""`raw2features sample` writes a readable synthetic slide the pipeline can embed."""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.data import write_sample_slide
from raw2features.readers.omezarr import OmeZarrReader


def test_sample_slide_is_readable_ngff(tmp_path):
    p = write_sample_slide(str(tmp_path / "sample.ome.zarr"), mpp0=0.5, size=512)
    with OmeZarrReader(p) as r:
        assert len(r.level_dimensions) == 3
        w, h = r.level_dimensions[0]
        assert (w, h) == (512, 512)
        assert abs(r.mpp - 0.5) < 1e-9
        # tissue blob is present: the centre patch is not the bright background.
        from raw2features.core.geometry import Point, Region, Size

        patch = r.read_region(Region(0, Point(192, 192), Size(64, 64)))
        assert patch.shape == (64, 64, 3)
        assert patch.mean() < 240  # blob is darker/saturated vs the 245 background


def test_sample_slide_embeds_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from conftest import MockEmbedder
    from raw2features.pipeline.runner import RunConfig, run_slide

    slide = write_sample_slide(str(tmp_path / "s.ome.zarr"), size=512)
    s = run_slide(
        slide,
        str(tmp_path / "out"),
        RunConfig(models=["mock"], no_seg=True, target_mpp=1.0, patch_px=64,
                  device="cpu", amp="fp32"),
        embedders=[MockEmbedder(dim=8)],
    )
    assert s["status"] == "complete"
    assert s["n_patches"] > 0
    assert np.isfinite(np.array([s["n_patches"]])).all()
