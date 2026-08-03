"""Guarded union tissue segmenter: ``otsu | canny`` minus the scanner canvas.

Pure OpenCV + numpy, no model weights, no GPU -- the same footing as ``otsu``/``canny``.

## What it fixes

Two classical-segmenter failures on real WSIs share one cause: the **scanner canvas**
(the dark no-signal fill outside the scanned region) and the **scan-region border**.

* **Otsu** thresholds HSV saturation *globally*, so a dark saturated canvas corner drags
  the threshold up and real tissue falls below it. On the worst observed slide Otsu put
  **all** of its patches on a corner artifact and **none** on the biopsy.
* **Canny** is worse in a subtler way: the scan-region rectangle becomes the outer
  contour, so the real tissue inside it is then a ``max_hole_frac`` *hole* that gets
  carved back to background -- the fill does not merely add junk, it **deletes** tissue.

Removing the canvas from the **raw edge map** (before Canny's dilate/close/fill) and from
the Otsu mask fixes both. Measured on 101 cervical WSIs: on the artifact slides the tiles
move off the artifact and onto the biopsy (one slide 2 -> 1172 patches, another 0 -> 1097),
pale slides gain ~2x (Canny recovers what Otsu's saturation threshold misses), and the
unflagged controls grow only ~1.2x.

## What it does NOT fix

**Out-of-focus tissue.** Blur is invisible to both a colour threshold and an edge filter;
OOF slides simply gain patches here. Use a dedicated focus metric if that matters.

## Relationship to ``grandqc_veto``

``grandqc_veto`` is this segmenter plus a GrandQC-tissue veto. On the same 101-slide panel
the veto changed nothing at all on **79/101** slides and removed only **1.2%** of patches
overall (median relative difference 0.0000), its one real contribution being on
out-of-focus slides. Prefer this segmenter unless you specifically want that margin: it
needs no GPU, no ~25 MB download, and no CC-BY-NC-SA weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.plugins import register

from .base import Segmenter, TissueMask
from .grandqc_veto import GrandQCVetoSegmenter

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("segmenters", "guarded")
class GuardedUnionSegmenter(Segmenter):
    """``(otsu | canny) & ~scanner_canvas`` -- classical, permissive, no weights.

    Parameters mirror the guard/union half of
    :class:`~raw2features.segmenters.grandqc_veto.GrandQCVetoSegmenter`; see
    :func:`~raw2features.segmenters.grandqc_veto.scan_region_kill` for the canvas
    detector itself.

    Parameters
    ----------
    seg_mpp:
        Target microns/px for the (cheap, low-res) level the masks are computed at.
    guard:
        Remove the detected scanner canvas / scan-region border. ``False`` reduces this
        to a plain ``otsu | canny`` union (i.e. ``combined or``) and reinstates both
        failures above -- it exists so the guard's contribution can be measured.
    canvas_dark, canvas_bright, canvas_var, border_um, frame_um:
        Canvas-detector parameters, passed through unchanged.
    canny_min_component_frac, canny_max_area_frac:
        Canny tuning; ``canny_max_area_frac`` is the safety valve for a filled outer
        ring swallowing the slide, in which case the Canny term is dropped and Otsu
        alone is used.
    """

    name = "guarded"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        *,
        guard: bool = True,
        canvas_dark: int = 20,
        canvas_bright: int | None = None,
        canvas_var: float = 0.5,
        border_um: float = 200.0,
        frame_um: float = 24.0,
        canny_min_component_frac: float = 0.001,
        canny_max_area_frac: float = 0.90,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.guard = guard
        # Compose rather than duplicate: the union half is shared with grandqc_veto, so
        # a fix to the canvas detector or the edge-map guard lands in both. The veto
        # half is never reached -- `segment` below returns the union directly and no
        # GrandQC import, weight download or forward pass happens.
        self._impl = GrandQCVetoSegmenter(
            seg_mpp=seg_mpp,
            guard=guard,
            canvas_dark=canvas_dark,
            canvas_bright=canvas_bright,
            canvas_var=canvas_var,
            border_um=border_um,
            frame_um=frame_um,
            canny_min_component_frac=canny_min_component_frac,
            canny_max_area_frac=canny_max_area_frac,
        )

    @property
    def last_stats(self) -> dict:
        """Provenance from the most recent :meth:`segment` (``kill_frac`` etc.)."""
        return self._impl.last_stats

    def segment(self, reader: WSISource) -> TissueMask:
        self._impl.last_stats = {}
        u, level, ds, _kill = self._impl._union(reader)
        self._impl.last_stats.update(
            union_mpp=(reader.mpp * ds if reader.mpp else None),
            level=level,
            veto="none",
        )
        return TissueMask(mask=u.astype(np.float32), level=level, downsample=ds)
