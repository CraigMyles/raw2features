"""Nuclear-channel tissue segmenter for MULTIPLEX slides (CODEX/Phenocycler).

Multiplex slides have no H&E saturation/colour to threshold; instead, tissue is where
the nuclear stain (DAPI / Hoechst) is present. This segmenter finds the nuclear marker
channel by name (via the reader's ``channel_names``), reads it at a low-res level, and
runs Otsu + morphology - the multiplex analogue of the default Otsu-on-saturation
segmenter. Pure OpenCV + numpy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.mpp import nearest_level
from raw2features.core.plugins import register

from .base import Segmenter, TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource

_NUCLEAR_ALIASES = ("dapi", "hoechst", "hochst", "dna")


@register("segmenters", "nuclear")
class NuclearSegmenter(Segmenter):
    """Otsu on the nuclear (DAPI/Hoechst) marker channel - for multiplex slides."""

    name = "nuclear"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        morph_kernel: int = 4,
        nuclear_aliases: tuple[str, ...] = _NUCLEAR_ALIASES,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.morph_kernel = morph_kernel
        self.nuclear_aliases = tuple(a.lower() for a in nuclear_aliases)

    def _pick_level(self, reader: WSISource) -> int:
        return nearest_level(reader.mpp, reader.level_downsamples(), self.seg_mpp)

    def _nuclear_index(self, channel_names: list[str] | None) -> int:
        if not channel_names:
            raise ValueError(
                "NuclearSegmenter needs a multiplex reader with channel_names "
                "(no DAPI/Hoechst channel to threshold)"
            )
        for i, name in enumerate(channel_names):
            low = name.lower()
            if any(alias in low for alias in self.nuclear_aliases):
                return i
        raise ValueError(
            f"no nuclear channel (aliases {self.nuclear_aliases}) found in the panel"
        )

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        idx = self._nuclear_index(reader.channel_names)
        level = self._pick_level(reader)
        dim: Size = reader.level_dimensions[level]
        block = reader.read_region_channels(
            Region(level=level, location=Point(0, 0), size=Size(dim.width, dim.height))
        )
        nuclear = block[:, :, idx]
        # Scale the (possibly uint16) nuclear channel to 8-bit for Otsu.
        chan8 = cv2.normalize(nuclear, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(chan8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if self.morph_kernel > 0:
            k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        mask = (binary > 0).astype(np.float32)
        downsample = reader.level_downsamples()[level]
        return TissueMask(mask=mask, level=level, downsample=float(downsample))
