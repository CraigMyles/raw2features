"""Classical Otsu tissue segmenter.

Permissively-licensed (we implement the standard HSV-saturation + Otsu +
morphology technique ourselves - no GPL/CLAM code, no model weights). Operates on
a low-resolution pyramid level so it is cheap and dependency-light. Pluggable: a
deep/licensed segmenter can be added later as an opt-in plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.mpp import nearest_level
from raw2features.core.plugins import register

from .base import Segmenter, TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("segmenters", "otsu")
class OtsuSegmenter(Segmenter):
    """Otsu-on-saturation tissue detection at ~``seg_mpp`` resolution."""

    name = "otsu"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        median_blur: int = 7,
        morph_kernel: int = 4,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.median_blur = median_blur if median_blur % 2 == 1 else median_blur + 1
        self.morph_kernel = morph_kernel

    def _pick_level(self, reader: WSISource) -> int:
        return nearest_level(reader.mpp, reader.level_downsamples(), self.seg_mpp)

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        from raw2features.viz import read_level_capped

        level = self._pick_level(reader)
        # Bounded read: a deficient/non-pyramidal slide whose seg level is still huge is
        # tile-downsampled instead of loaded whole (factor>1 scales the downsample).
        img, factor = read_level_capped(reader, level)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = cv2.medianBlur(hsv[:, :, 1], self.median_blur)
        _, binary = cv2.threshold(sat, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if self.morph_kernel > 0:
            k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        mask = (binary > 0).astype(np.float32)
        downsample = reader.level_downsamples()[level] * factor
        return TissueMask(mask=mask, level=level, downsample=float(downsample))
