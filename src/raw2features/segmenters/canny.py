"""Canny edge tissue segmenter (classical, permissive) - tuned for H&E.

A **low** Canny threshold catches the faint edges of pale / sparse / low-contrast
tissue (e.g. cervical); we then close the edge map and **fill its external contours**
into solid tissue regions. This is far more reliable on faint slides than thresholding
edge *density*, which misses low-contrast tissue (see docs/SEGMENTATION.md for the
on-slide comparison). Opt-in alternative to the default Otsu segmenter
(``--segmenter canny``); particularly good on cervical.

Thresholds are given as **fractions of the 0-255 range** (so ``low=0.05`` is the
~0.05 setting that works well on cervical, ``= a raw Canny low of ~13``), which is how
the threshold is usually reasoned about, rather than raw 8-bit values.

Pure OpenCV + numpy (Canny + morphology + contour fill) - no GPL/CLAM code, no model
weights. Runs on the same cheap low-res level as the Otsu segmenter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.mpp import nearest_level
from raw2features.core.plugins import register

from .base import Segmenter, TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("segmenters", "canny")
class CannySegmenter(Segmenter):
    """Low-threshold Canny -> close -> fill external contours into tissue regions.

    Parameters
    ----------
    seg_mpp:
        Target microns/px for the (cheap, low-res) level the mask is computed at.
    blur:
        Gaussian-blur kernel (odd) before Canny, to denoise without losing edges.
    low, high:
        Canny hysteresis thresholds as **fractions of 0-255** (0..1). The default
        ``low=0.05`` is the low-sensitivity setting that works well on cervical H&E;
        ``high`` is the upper hysteresis bound.
    dilate_kernel:
        Connect adjacent edge fragments before contour finding.
    close_kernel:
        Close the edge map so tissue outlines become closed, fillable contours.
    min_component_frac:
        Drop filled contours smaller than this fraction of the image (dust, debris).
        Lower it to keep small wispy tissue islands Otsu tends to miss.
    max_hole_frac:
        Internal cavities (lumens / glands / fat) larger than this fraction of the
        image are kept as **background** rather than filled in; smaller holes (gaps in
        the edge map) are filled. ``1.0`` fills all holes; ``0.0`` keeps all holes open.
    edge_filter:
        Optional ``f(edges, img, level_mpp) -> edges`` applied to the **raw Canny edge
        map**, before the dilate/close/contour-fill. This is the only useful place to
        suppress an edge you do not want: once a scan-region rectangle or a slide-edge
        ring has been *filled*, it is the outer contour and the real tissue inside it
        becomes a ``max_hole_frac``-sized hole that gets carved back to background --
        i.e. filtering the finished mask instead loses the tissue as well as the ring.
        ``level_mpp`` is the microns/px of the level the edges were computed at (or
        ``None`` when the slide has no MPP), so a filter can size itself in microns.
        See :func:`raw2features.segmenters.grandqc_veto.scan_region_kill`.
    """

    name = "canny"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        blur: int = 5,
        low: float = 0.05,
        high: float = 0.15,
        dilate_kernel: int = 3,
        close_kernel: int = 9,
        min_component_frac: float = 0.001,
        max_hole_frac: float = 0.01,
        edge_filter: Callable[[np.ndarray, np.ndarray, float | None], np.ndarray]
        | None = None,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.blur = blur if blur % 2 == 1 else blur + 1
        self.low = low
        self.high = high
        self.dilate_kernel = dilate_kernel
        self.close_kernel = close_kernel
        self.min_component_frac = min_component_frac
        self.max_hole_frac = max_hole_frac
        self.edge_filter = edge_filter

    def _pick_level(self, reader: WSISource) -> int:
        return nearest_level(reader.mpp, reader.level_downsamples(), self.seg_mpp)

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        level = self._pick_level(reader)
        dim: Size = reader.level_dimensions[level]
        img = reader.read_region(
            Region(level=level, location=Point(0, 0), size=Size(dim.width, dim.height))
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)

        lo = int(round(self.low * 255))
        hi = int(round(self.high * 255))
        edges = cv2.Canny(gray, lo, hi)

        # Suppress unwanted edges HERE, while they are still thin lines -- see the
        # ``edge_filter`` docstring for why doing it after the fill does not work.
        if self.edge_filter is not None:
            mpp = reader.mpp
            level_mpp = mpp * reader.level_downsamples()[level] if mpp else None
            edges = self.edge_filter(edges, img, level_mpp)

        # Connect fragments, then close outlines so they enclose fillable regions.
        if self.dilate_kernel > 0:
            edges = cv2.dilate(
                edges, np.ones((self.dilate_kernel, self.dilate_kernel), np.uint8),
                iterations=2,
            )
        if self.close_kernel > 0:
            k = np.ones((self.close_kernel, self.close_kernel), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

        # Fill contours into solid tissue regions, but KEEP large internal cavities
        # (lumens / glands / fat) as background -- only small holes (gaps in the edge
        # map) get filled. RETR_CCOMP gives a 2-level hierarchy: outer contours
        # (parent == -1) are tissue; their children are holes.
        contours, hier = cv2.findContours(
            edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros(gray.shape, np.uint8)
        if hier is not None:
            hier = hier[0]
            min_area = self.min_component_frac * mask.size
            max_hole = self.max_hole_frac * mask.size
            for i, c in enumerate(contours):  # outer contours -> tissue
                if hier[i][3] == -1 and cv2.contourArea(c) >= min_area:
                    cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)
            for i, c in enumerate(contours):  # large holes -> carve back to background
                if hier[i][3] != -1 and cv2.contourArea(c) >= max_hole:
                    cv2.drawContours(mask, [c], -1, 0, cv2.FILLED)

        downsample = reader.level_downsamples()[level]
        return TissueMask(
            mask=(mask > 0).astype(np.float32),
            level=level,
            downsample=float(downsample),
        )
