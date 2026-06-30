"""``Patcher`` - the MPP-aware tiling seam.

Builds a tiling grid at an exact target MPP, then selects tiles overlapping
tissue. Coordinates are emitted in level-0 pixels (row-major), 1:1 with the
``grid_index`` - the spatial-provenance contract the sink relies on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from raw2features.segmenters.base import TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@dataclass(frozen=True)
class PatchGrid:
    """A resolved tiling plan over a slide."""

    target_mpp: float
    achieved_mpp: float
    patch_px: int  # output patch side (e.g. 224)
    step_out_px: int  # stride in output pixels (== patch_px for no overlap)
    level0_patch: int  # patch extent in level-0 pixels
    level0_step: int  # stride in level-0 pixels
    read_level: int  # pyramid level to read from
    read_px: int  # pixels to read per side at read_level (before resize)
    resample: float
    needs_resample: bool
    n_rows: int
    n_cols: int
    xs: tuple[int, ...]  # level-0 x of each column
    ys: tuple[int, ...]  # level-0 y of each row


class Patcher(ABC):
    """Abstract patcher."""

    name: str = "patcher"

    @abstractmethod
    def build_grid(self, reader: WSISource) -> PatchGrid:
        """Resolve the tiling grid for ``reader``."""

    @abstractmethod
    def tile(
        self,
        grid: PatchGrid,
        tissue: TissueMask | None,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select tiles.

        Returns ``(coords, grid_index, grid_tissue)``:

        * ``coords``: ``(N, 2)`` int32 level-0 ``(x, y)`` of kept tiles.
        * ``grid_index``: ``(N, 2)`` int32 ``(row, col)`` of kept tiles.
        * ``grid_tissue``: ``(n_rows, n_cols)`` float32 per-cell tissue fraction.
        """
