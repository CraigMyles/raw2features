"""Per-patch Canny **edge-density** tissue segmenter (classical, permissive).

The sibling ``canny`` segmenter takes a *region* view: low-threshold Canny ->
dilate -> close -> **fill external contours** into solid tissue regions. Filling is
what recovers faint tissue, but it also fills the gaps *between* fragments, so
sparse/fragmented slides gain background and boundaries come out blobby.

This segmenter takes the *per-patch* view instead, the recipe practitioners use on
cervical H&E: run Canny at a **low** threshold (``low=0.05``, i.e. a raw Canny low of
~13) and keep the **raw edge map** -- no dilation, no closing, no contour fill. The
patcher averages the mask over each patch footprint
(:meth:`~raw2features.patcher.grid.GridPatcher._cell_tissue_fractions` takes
``window.mean()``), so with this mask that average *is* the patch's **edge density**
and ``--tissue-threshold`` *is* the minimum edge density to keep a patch. No pipeline
change is needed: the decision is made independently for every patch.

Two consequences worth knowing:

* **Sharp, gap-respecting boundaries.** Nothing grows or fills, so background
  between tissue fragments stays background.
* **The threshold is a density, not a coverage fraction.** Edge *pixels* are a small
  minority of even solid tissue, so useful values are far lower than the 0.1 default
  that suits a filled/binary mask -- see ``docs/SEGMENTATION.md``. Density also scales
  with ``seg_mpp`` (finer masks put more, thinner edges in the same field of view), so
  re-tune ``--tissue-threshold`` if you change it.

``seg_mpp`` defaults to 2.0 rather than the 8.0 used by ``otsu``/``canny``: the mask is
the resolution at which the keep/drop decision is made, and 2.0 µm/px gives a 224 px
patch at 0.5 µm/px roughly 56x56 mask pixels to average over. Pure OpenCV + numpy - no
model weights, no GPL/CLAM code.
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


@register("segmenters", "canny_density")
class CannyDensitySegmenter(Segmenter):
    """Low-threshold Canny kept as a raw edge map -> per-patch edge density.

    Parameters
    ----------
    seg_mpp:
        Target microns/px for the level the edge map is computed at. Finer than the
        classical segmenters (default 2.0) because this mask *is* the per-patch
        decision surface rather than a coarse region outline.
    blur:
        Gaussian-blur kernel (odd) applied before Canny, to denoise without
        losing faint edges.
    low, high:
        Canny hysteresis thresholds as **fractions of 0-255**. ``low=0.05`` is the
        low-sensitivity setting that works well on cervical H&E.
    dilate:
        Optional edge thickening (0 = off, the default). Leave at 0 for a faithful
        density; a small value (1-2) stabilises the density on very thin edges at
        fine ``seg_mpp``, at the cost of slightly inflating it.
    """

    name = "canny_density"

    def __init__(
        self,
        seg_mpp: float = 2.0,
        blur: int = 3,
        low: float = 0.05,
        high: float = 0.15,
        dilate: int = 0,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.blur = blur if blur % 2 == 1 else blur + 1
        self.low = low
        self.high = high
        self.dilate = dilate

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
        if self.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)

        lo = int(round(self.low * 255))
        hi = int(round(self.high * 255))
        edges = cv2.Canny(gray, lo, hi)

        # Deliberately NO close / no contour fill: the raw edge map is the signal.
        # Optional thickening only, off by default.
        if self.dilate > 0:
            edges = cv2.dilate(
                edges, np.ones((self.dilate, self.dilate), np.uint8), iterations=1
            )

        mask = (edges > 0).astype(np.float32)  # patcher averages this -> edge density
        ds = float(reader.level_downsamples()[level])
        return TissueMask(mask=mask, level=level, downsample=ds)
