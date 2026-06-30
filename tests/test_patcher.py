"""Tests for the MPP-aware grid patcher against the synthetic fixture."""

from __future__ import annotations

import numpy as np

from raw2features.patcher.grid import GridPatcher
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.base import TissueMask


def test_build_grid_exact_native(synthetic_ngff):
    # mpp0 = 0.5; patch 64 @ target 0.5 -> level-0 patch 64, read level 0, no resize.
    with OmeZarrReader(synthetic_ngff) as r:
        g = GridPatcher(target_mpp=0.5, patch_px=64).build_grid(r)
        assert g.read_level == 0 and not g.needs_resample
        assert g.level0_patch == 64 and g.level0_step == 64
        assert g.xs == tuple(range(0, 300 - 64 + 1, 64))  # 0,64,128,192
        assert g.ys == tuple(range(0, 200 - 64 + 1, 64))  # 0,64,128
        assert (g.n_rows, g.n_cols) == (3, 4)


def test_build_grid_exact_with_resample(synthetic_ngff):
    # target 0.75: nearest finer-or-equal level is level 0 (0.5) -> resample 1.5.
    with OmeZarrReader(synthetic_ngff) as r:
        g = GridPatcher(target_mpp=0.75, patch_px=64).build_grid(r)
        assert g.read_level == 0 and g.needs_resample
        assert g.read_px == round(64 * 1.5)  # 96
        assert g.level0_patch == 96
        assert abs(g.achieved_mpp - 0.75) < 1e-9


def test_tile_keep_all_without_mask(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        p = GridPatcher(target_mpp=0.5, patch_px=64)
        g = p.build_grid(r)
        coords, grid_index, grid_tissue = p.tile(g, None, threshold=0.1)
        assert coords.shape == (12, 2)
        assert grid_index.shape == (12, 2)
        assert grid_tissue.shape == (3, 4) and np.all(grid_tissue == 1.0)
        # level-0 coords within bounds; row-major order.
        assert coords[:, 0].max() <= 300 - 64
        assert coords[:, 1].max() <= 200 - 64
        assert tuple(grid_index[0]) == (0, 0)
        assert tuple(grid_index[1]) == (0, 1)


def test_tile_threshold_with_manual_mask(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        p = GridPatcher(target_mpp=0.5, patch_px=64)
        g = p.build_grid(r)
        ones = TissueMask(np.ones((200, 300), np.float32), level=0, downsample=1.0)
        coords, _, gt = p.tile(g, ones, threshold=0.5)
        assert coords.shape[0] == 12 and np.allclose(gt, 1.0)

        zeros = TissueMask(np.zeros((200, 300), np.float32), level=0, downsample=1.0)
        c2, gi2, gt2 = p.tile(g, zeros, threshold=0.5)
        assert c2.shape == (0, 2) and gi2.shape == (0, 2)
        assert np.allclose(gt2, 0.0)
