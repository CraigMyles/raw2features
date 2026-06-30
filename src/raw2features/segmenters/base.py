"""``Segmenter`` - the tissue-detection seam.

A segmenter returns a low-resolution tissue mask at a chosen pyramid level. The
patcher maps the tiling grid onto this mask to decide which patches to keep, so
the segmenter is decoupled from patch size / target MPP.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@dataclass(frozen=True)
class TissueMask:
    """A tissue mask at one pyramid level.

    Attributes
    ----------
    mask:
        2D float32 array in [0, 1] (1 == tissue), at ``level`` resolution.
    level:
        Pyramid level the mask was computed at.
    downsample:
        ``level``'s downsample factor relative to level 0 (maps level-0 px to
        mask px: ``mask_px = level0_px / downsample``).
    """

    mask: np.ndarray
    level: int
    downsample: float


class Segmenter(ABC):
    """Abstract tissue segmenter."""

    name: str = "segmenter"

    @abstractmethod
    def segment(self, reader: WSISource) -> TissueMask:
        """Return a :class:`TissueMask` for the slide behind ``reader``."""
