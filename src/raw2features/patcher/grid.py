"""Regular MPP-aware grid patcher."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.mpp import level0_step_px
from raw2features.core.plugins import register
from raw2features.segmenters.base import TissueMask

from .base import Patcher, PatchGrid

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("patchers", "grid")
class GridPatcher(Patcher):
    """Tile a slide on a regular grid at an exact target MPP."""

    name = "grid"

    def __init__(
        self,
        target_mpp: float = 1.0,
        patch_px: int = 224,
        step_out_px: int | None = None,
        snap_to_level: bool = False,
        mpp_tolerance: float = 0.001,
        allow_upsample: bool = False,
    ) -> None:
        self.target_mpp = target_mpp
        self.patch_px = patch_px
        self.step_out_px = step_out_px if step_out_px is not None else patch_px
        self.snap_to_level = snap_to_level
        self.mpp_tolerance = mpp_tolerance
        self.allow_upsample = allow_upsample

    def build_grid(self, reader: WSISource) -> PatchGrid:
        mpp0 = reader.mpp
        if mpp0 is None:
            raise ValueError(
                "source physical pixel size is unknown: this OME-Zarr declares no "
                "x/y axis unit, so its scale is in arbitrary (pixel) units, not µm. "
                "Pass --source-mpp <µm/px> to supply the source's level-0 pixel size "
                "(e.g. 0.25 for a 40x scan) - this is the source's physical "
                "resolution, NOT the extraction scale (--mpp). Better still, fix the "
                "source metadata to declare the axis unit."
            )
        plan = reader.level_for_mpp(
            self.target_mpp,
            self.patch_px,
            snap_to_level=self.snap_to_level,
            tolerance=self.mpp_tolerance,
            allow_upsample=self.allow_upsample,
        )
        level0_patch = round(self.patch_px * plan.achieved_mpp / mpp0)
        level0_step = level0_step_px(self.step_out_px, plan.achieved_mpp, mpp0)
        w0, h0 = reader.level_dimensions[0]

        xs = tuple(range(0, max(w0 - level0_patch, 0) + 1, level0_step))
        ys = tuple(range(0, max(h0 - level0_patch, 0) + 1, level0_step))
        return PatchGrid(
            target_mpp=self.target_mpp,
            achieved_mpp=plan.achieved_mpp,
            patch_px=self.patch_px,
            step_out_px=self.step_out_px,
            level0_patch=level0_patch,
            level0_step=level0_step,
            read_level=plan.level,
            read_px=plan.read_px,
            resample=plan.resample,
            needs_resample=plan.needs_resample,
            n_rows=len(ys),
            n_cols=len(xs),
            xs=xs,
            ys=ys,
        )

    def tile(
        self,
        grid: PatchGrid,
        tissue: TissueMask | None,
        threshold: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_tissue = np.zeros((grid.n_rows, grid.n_cols), dtype=np.float32)

        if tissue is None:
            grid_tissue[:] = 1.0
            keep = np.ones((grid.n_rows, grid.n_cols), dtype=bool)
        else:
            grid_tissue = self._cell_tissue_fractions(grid, tissue)
            keep = grid_tissue >= threshold

        coords = []
        grid_index = []
        for r, y0 in enumerate(grid.ys):
            for c, x0 in enumerate(grid.xs):
                if keep[r, c]:
                    coords.append((x0, y0))
                    grid_index.append((r, c))
        coords_arr = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        grid_index_arr = np.asarray(grid_index, dtype=np.int32).reshape(-1, 2)
        return coords_arr, grid_index_arr, grid_tissue

    @staticmethod
    def _cell_tissue_fractions(grid: PatchGrid, tissue: TissueMask) -> np.ndarray:
        mask = tissue.mask
        ds = tissue.downsample
        cell = max(1, round(grid.level0_patch / ds))
        frac = np.zeros((grid.n_rows, grid.n_cols), dtype=np.float32)
        for r, y0 in enumerate(grid.ys):
            my0 = int(round(y0 / ds))
            for c, x0 in enumerate(grid.xs):
                mx0 = int(round(x0 / ds))
                window = mask[my0 : my0 + cell, mx0 : mx0 + cell]
                frac[r, c] = float(window.mean()) if window.size else 0.0
        return frac


def resample_patch(patch_hwc: np.ndarray, out_px: int) -> np.ndarray:
    """Resample an HWC uint8 patch to exactly ``out_px`` square.

    Realises the exact-MPP contract independently of any model: the reader
    returns ``read_px`` = ``round(patch_px * resample)`` pixels covering the
    target field of view, and this resamples them to exactly ``patch_px`` at the
    target MPP. The per-model ``input_size`` resize happens later, in the
    embedder's transform -- so the stored ``patch_px``/``achieved_mpp`` describe
    the patch faithfully even when ``patch_px != input_size``.
    """
    h, w = patch_hwc.shape[:2]
    if h == out_px and w == out_px:
        return patch_hwc
    import cv2

    interp = cv2.INTER_AREA if out_px < max(h, w) else cv2.INTER_LINEAR
    # cv2.resize handles at most 4 channels; multiplex stacks (C>4) resize per channel
    # (the equal-size fast path above means native-MPP reads never reach here).
    if patch_hwc.ndim == 3 and patch_hwc.shape[2] > 4:
        return np.stack(
            [cv2.resize(patch_hwc[:, :, c], (out_px, out_px), interpolation=interp)
             for c in range(patch_hwc.shape[2])],
            axis=-1,
        )
    return cv2.resize(patch_hwc, (out_px, out_px), interpolation=interp)
