"""Combined tissue segmenter: fuse Otsu-on-saturation with Canny edge-density.

The literature-backed best use of edges on H&E: run the colour-based Otsu
segmenter and the texture-based Canny edge-density segmenter and fuse their masks.

- ``or`` (default): tissue = otsu OR canny -> recovers faint / low-saturation
  tissue that still has texture, while keeping Otsu's solid regions.
- ``and``: tissue = otsu AND canny -> suppresses smooth high-saturation artefacts
  (some pen smears / coloured background) that Otsu alone would keep.

Standalone Canny is unreliable on WSIs (it over- or under-segments depending on the
slide); fused with the colour threshold is where edges add real value. Pure OpenCV
+ numpy, permissive - it just composes our own Otsu and Canny segmenters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.plugins import register

from .base import Segmenter, TissueMask
from .canny import CannySegmenter
from .otsu import OtsuSegmenter

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("segmenters", "combined")
class CombinedSegmenter(Segmenter):
    """Otsu-on-saturation fused with Canny edge-density (``or`` / ``and``)."""

    name = "combined"

    def __init__(self, seg_mpp: float = 8.0, mode: str = "or") -> None:
        if mode not in ("or", "and"):
            raise ValueError(f"mode must be 'or' or 'and', got {mode!r}")
        self.mode = mode
        self.otsu = OtsuSegmenter(seg_mpp=seg_mpp)
        self.canny = CannySegmenter(seg_mpp=seg_mpp)

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        o = self.otsu.segment(reader)
        c = self.canny.segment(reader)
        om, cm = o.mask > 0, c.mask > 0
        if om.shape != cm.shape:  # both default to the same seg_mpp; guard anyway
            cm = cv2.resize(
                cm.astype(np.uint8),
                (om.shape[1], om.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        fused = (om & cm) if self.mode == "and" else (om | cm)
        return TissueMask(
            mask=fused.astype(np.float32), level=o.level, downsample=o.downsample
        )
