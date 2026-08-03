"""Vetoed-union segmenter: ``(otsu | canny)``, scan-region guarded, GrandQC vetoed.

Three cheap, classical stages and one deep stage, each used only where it is reliable.

**What each term contributes**

* ``otsu`` (HSV-saturation Otsu @ 8 µm/px) -- *sharp boundaries and correct holes*.
  It defines the reference grid. It is the conservative term: on a clean slide the
  output's outline is Otsu's outline.
* ``canny`` (low-threshold Canny -> close -> fill contours @ 8 µm/px) -- *recall on
  pale / low-saturation tissue*, which thresholding saturation drops (cervical stroma,
  curettings). Filled, so it is a *region* mask: the union and the per-component
  algebra below both need regions, which is why ``canny_density``'s raw edge map is
  deliberately not used here (its mask is a density, and OR-ing a density into a binary
  mask silently rescales ``--tissue-threshold``).
* the **scan-region guard** (pure OpenCV, ~10 ms) -- *removes the synthetic canvas fill,
  the scan-region border and the image frame*. This is not cosmetic. Canny latches onto
  the canvas/scan-region step edge, that rectangle becomes the outer contour, and the
  real tissue inside it then becomes a ``max_hole_frac``-sized hole that
  :class:`~raw2features.segmenters.canny.CannySegmenter` carves back to background --
  so on canvas-heavy slides the border does not merely add junk, it *destroys* the
  canny term. The guard therefore runs on the **edge map**, before the morphology
  (via ``CannySegmenter(edge_filter=...)``); guarding the finished mask instead is
  measurably useless.
* ``grandqc`` (stage-1 tissue UNet++ @ its trained 10 µm/px) -- **veto only, never a
  source of tissue**. Its mask bleeds into glass and fills lumens (one mask pixel spans
  20x20 grid px at 0.5 µm/px), so it must never *add* area; but it is excellent at
  saying "that block is not tissue". Used per connected component of the union, so its
  coarse resolution stops mattering.

**What the veto does NOT fix**

* **Out-of-focus tissue.** Neither GrandQC stage separates focus (the artifact stage
  scores ``out_of_focus`` .027 on controls vs .024 on flagged slides -- no separation;
  the tissue stage happily calls blurred tissue tissue). Defocused regions are also
  low-contrast, so the union terms are weak there, but *nothing here measures focus*.
  Slides that are tiled because they are blurry stay tiled. That needs a separate
  per-patch focus gate (e.g. variance-of-Laplacian), not this segmenter.
* **A GrandQC false negative.** A tissue fragment GrandQC misses forms its own
  component with coverage ~0 and is dropped wholesale -- no slack or width guard
  protects that path (they only guard the pixel-wise clip). ``veto_removed_frac`` in
  :attr:`GrandQCVetoSegmenter.last_stats` exists so a cohort can be sorted by it and
  the top of the distribution eyeballed.
* **Otsu's threshold being dragged up by an artifact.** Where a dark block pushes the
  global Otsu threshold above real tissue, the veto can only *remove*; it cannot restore
  what Otsu already lost. Recall on those slides is carried entirely by the guarded
  canny term. (The principled fix -- re-estimating Otsu on the veto-endorsed region --
  is deliberately out of scope here.)

**The combination algebra** is a hybrid that contains the two obvious designs as corner
cases, so they can be compared without a second implementation:

* pure pixel-wise ``AND``: ``veto_cov=0.0, keep_cov=1.1, clip_min_area_mm2=0.0,
  veto_slack_um=0.0, removal_min_width_um=0.0``
* pure component veto: ``clip_min_area_mm2=math.inf`` (the clip branch never fires)
* the default: drop a component GrandQC rejects outright, keep one it endorses, and
  pixel-wise clip only the *large, mixed-evidence* ones -- because boundary-erosion cost
  scales as ~1/diameter (cheap on big components, destructive on thin ones) while the
  risk of an artifact being fused into a real component grows with component size.

Two guards make that clip safe: the veto mask is dilated by ``veto_slack_um`` first (so
quantisation and ~one mask pixel of model error cannot erode anything), and the removal
set is filtered by opening-by-reconstruction at ``removal_min_width_um`` (a real
artifact is a blob; a mis-registration is a sliver, and slivers are put back).

The output is **strictly binary**, so ``--tissue-threshold`` keeps its usual
coverage meaning (default 0.1) and tile counts stay comparable with existing
``otsu`` / ``canny`` runs.

Needs the optional ``[grandqc]`` extra; the weights are CC-BY-NC-SA (non-commercial) and
are fetched on first use -- see ``docs/SEGMENTATION.md``, ``docs/MODEL_LICENSES.md``.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from raw2features.core.mpp import nearest_level
from raw2features.core.plugins import register

from .base import Segmenter, TissueMask
from .canny import CannySegmenter
from .grandqc import GrandQCSegmenter
from .otsu import OtsuSegmenter

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource

__all__ = ["GrandQCVetoSegmenter", "scan_region_kill"]


# -- small OpenCV helpers ------------------------------------------------------


def _disk(radius: int) -> np.ndarray:
    import cv2

    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _px(um: float, mpp: float | None) -> int:
    """``um`` microns as a whole number of mask pixels at ``mpp`` µm/px.

    ``0`` only when the caller asked for no margin at all (``um <= 0``) -- callers
    treat that as "skip this morphology entirely", which is what makes the documented
    pixel-wise-AND corner case exact. Any positive request rounds up to >= 1 px.
    """
    if um <= 0:
        return 0
    if not mpp:  # unknown MPP: fall back to a single pixel rather than guessing
        return 1
    return max(1, int(round(um / mpp)))


def _align(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample a boolean mask onto ``shape`` (H, W), preserving thin structures.

    Upsampling uses nearest neighbour; **downsampling uses ``INTER_AREA`` on a float
    mask thresholded at > 0**, i.e. OR-pooling -- a structure occupying any part of a
    destination pixel survives. ``INTER_NEAREST`` on the downsample path deletes thin
    structures at random, which is exactly the tissue this segmenter exists to recover.
    """
    import cv2

    if m.shape == shape:
        return m
    h, w = shape
    interp = (
        cv2.INTER_NEAREST if (m.shape[0] <= h and m.shape[1] <= w) else cv2.INTER_AREA
    )
    return cv2.resize(m.astype(np.float32), (w, h), interpolation=interp) > 0


def _frame_connected(m: np.ndarray) -> np.ndarray:
    """Components of ``m`` (8-connected) that touch the image perimeter."""
    import cv2

    if not m.any():
        return np.zeros(m.shape, bool)
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(m.shape, bool)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    keep = np.unique(border)
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros(m.shape, bool)
    lut = np.zeros(n, np.uint8)
    lut[keep] = 1
    return lut[lab].astype(bool)


def _reconstruct(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Morphological reconstruction: components of ``mask`` containing a seed pixel."""
    import cv2

    if not seeds.any():
        return np.zeros(mask.shape, bool)
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    hit = np.unique(lab[seeds])
    hit = hit[hit != 0]
    if hit.size == 0:
        return np.zeros(mask.shape, bool)
    lut = np.zeros(n, np.uint8)
    lut[hit] = 1
    return lut[lab].astype(bool)


# -- the scan-region / canvas guard -------------------------------------------


def scan_region_kill(
    img: np.ndarray,
    level_mpp: float | None,
    *,
    canvas_dark: int = 20,
    canvas_bright: int | None = None,
    canvas_var: float = 0.5,
    border_um: float = 200.0,
    frame_um: float = 24.0,
) -> np.ndarray:
    """Pixels that are scanner canvas, scan-region border, or image frame.

    The border a Canny segmenter picks up on a WSI is not an abstract "edge of the
    image" -- it is the step edge at the boundary of the **synthetic canvas fill** the
    store pads the scan region with (black in OME-Zarr v0.5; white in some SVS/CZI
    exports). Detecting the fill itself is exact and free, whereas a fixed border
    *margin* both misses interior scan-region rectangles and clips genuine tissue on
    slides where the specimen runs to the frame.

    A pixel is canvas when it is (a) a **no-signal fill** -- near-constant in a 5x5
    neighbourhood *and* dark (or bright, if ``canvas_bright`` is set) -- and (b)
    **connected to the image perimeter**. The local-variance term is what keeps a dark
    *tissue* artifact out of the flood: real material is textured. The perimeter
    connectivity is what keeps an interior dark lumen out of it.

    Returns a boolean mask of canvas | a ``border_um`` band dilated from the canvas
    boundary (the ring the filled canny mask actually occupies) | a ``frame_um``
    perimeter ring (for slides with no canvas at all).
    """
    import cv2

    gray = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2GRAY)
    g32 = gray.astype(np.float32)
    mean = cv2.boxFilter(g32, -1, (5, 5))
    var = np.maximum(cv2.boxFilter(g32 * g32, -1, (5, 5)) - mean * mean, 0.0)

    fill = gray <= canvas_dark
    if canvas_bright is not None:
        fill |= gray >= canvas_bright
    canvas = _frame_connected(fill & (var <= canvas_var))

    kill = canvas.copy()
    r = _px(border_um, level_mpp)
    if canvas.any() and r > 0:
        grad = cv2.morphologyEx(
            canvas.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
        )
        kill |= cv2.dilate(grad, _disk(r)) > 0
    f = _px(frame_um, level_mpp)
    if f > 0:
        kill[:f], kill[-f:], kill[:, :f], kill[:, -f:] = True, True, True, True
    return kill


# -- the segmenter -------------------------------------------------------------


@register("segmenters", "grandqc_veto")
class GrandQCVetoSegmenter(Segmenter):
    """``(otsu | canny)``, scan-region guarded, vetoed component-wise by GrandQC.

    Parameters
    ----------
    seg_mpp:
        Microns/px of the **union grid** -- the level otsu and canny are computed at,
        and the grid of the returned mask. 8.0 is a region-outline grid, not a decision
        surface; do not lower it expecting sharper patches.
    device:
        Torch device for the GrandQC veto (``"auto"`` resolves to the GPU). The veto is
        a UNet++ forward and wants a GPU, though it is only ~2-4% of a slide's
        embedding cost either way and runs acceptably on CPU (~5-20 s/slide).
    canny_min_component_frac:
        Passthrough to :class:`~raw2features.segmenters.canny.CannySegmenter`. Left at
        canny's own default so a comparison attributes changes to the fusion rather
        than to a retuned term. With the veto acting as the junk filter, lowering it
        (0.0005 / 0.0002) to recover small pale fragments becomes much safer.
    canny_max_area_frac:
        Degenerate-fill backstop. If the filled canny mask still covers more than this
        fraction of the level after the guard, treat the canny term as having collapsed
        into one giant contour and fall back to otsu alone for that slide.
    guard:
        Run the scan-region / canvas guard (see :func:`scan_region_kill`). On by
        default: it is ~10 ms, it only ever removes, and on canvas-heavy slides it is
        the difference between the canny term working and the canny term being empty.
    canvas_dark, canvas_bright, canvas_var, border_um, frame_um:
        Guard parameters, see :func:`scan_region_kill`. ``canvas_bright`` is ``None``
        (white-fill detection **off**) by default -- a JPEG-flattened slide with a
        uniform white background connected to the frame would otherwise be flooded as
        canvas and lose a ``border_um`` ring around every tissue island. Enable it per
        format only after checking how much of the Otsu mask the guard removes.
    veto_cov:
        Drop a union component outright when GrandQC endorses less than this fraction
        of it. Deliberately low: only remove what GrandQC *confidently* rejects, because
        this branch has no safety net (a GrandQC false negative is deleted wholesale).
    keep_cov:
        Skip the clip when GrandQC endorses essentially all of a component -- a
        short-circuit, not a safety mechanism. High on purpose, so an artifact attached
        to a 95%-endorsed component still gets clipped.
    clip_min_area_mm2:
        Only pixel-wise clip components at least this large. 0.25 mm² is a 500x500 µm
        square; erosion of a compact component costs about ``4*slack/diameter``, so at
        ``veto_slack_um=16`` this keeps the loss under ~10%. ``math.inf`` disables the
        clip entirely (pure component veto); ``0.0`` clips everything.
    veto_slack_um:
        Dilate the veto mask by this much **before** any pixel-wise clip, so the clip
        can only remove pixels that are this far from anything GrandQC calls tissue.
        Absorbs grid quantisation (half a 10 µm mask pixel) plus a full mask pixel of
        model boundary error. Cheap, because GrandQC's error direction is documented
        over-inclusion and the union term -- not the veto -- defines the outer boundary.
    removal_min_width_um:
        A clipped-away region must be at least this wide *somewhere*
        (opening-by-reconstruction) or it is returned to tissue. This protects a thin
        tissue wisp of any length, which a pure area threshold on removed pieces would
        not. The trade is explicit: a genuine artifact thinner than this everywhere
        (a hairline pen line or fold) is put back.
    connectivity:
        Component connectivity, 4 or 8. 4 on purpose: a single diagonal pixel chain
        must not merge an artifact block into the tissue component.
    veto_abort_cov:
        Safety valve. If GrandQC endorses less than this fraction of the whole union,
        treat the veto as degenerate (empty prediction / model or stain failure), warn,
        and emit the union unclipped rather than silently returning zero tiles. Set low
        on purpose: on a genuine artifact slide the veto legitimately removes most of
        the union's *area*.
    min_area_mm2:
        Optional dust filter on the final mask. ``0.0`` (off) by default so it does not
        confound an evaluation of the fusion itself.
    on_veto_error:
        ``"raise"`` (default) if the GrandQC term cannot run -- missing ``[grandqc]``
        extra, download failure, OOM. ``"warn"`` degrades to the bare guarded union.
        Never silent: a silent fallback reintroduces every failure mode this exists to
        fix.

    Notes
    -----
    After :meth:`segment`, :attr:`last_stats` holds a provenance/observability dict
    (``veto_removed_frac``, per-branch component counts, whether the canny term or the
    veto degenerated, the grid MPPs). Sorting a cohort by ``veto_removed_frac`` is the
    cheapest way to find slides where the veto is wrong.
    """

    name = "grandqc_veto"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        device: str = "auto",
        *,
        canny_min_component_frac: float = 0.001,
        canny_max_area_frac: float = 0.90,
        guard: bool = True,
        canvas_dark: int = 20,
        canvas_bright: int | None = None,
        canvas_var: float = 0.5,
        border_um: float = 200.0,
        frame_um: float = 24.0,
        veto_cov: float = 0.10,
        keep_cov: float = 0.98,
        clip_min_area_mm2: float = 0.25,
        veto_slack_um: float = 16.0,
        removal_min_width_um: float = 32.0,
        connectivity: int = 4,
        veto_abort_cov: float = 0.02,
        min_area_mm2: float = 0.0,
        on_veto_error: str = "raise",
    ) -> None:
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity!r}")
        if on_veto_error not in ("raise", "warn"):
            raise ValueError(
                f"on_veto_error must be 'raise' or 'warn', got {on_veto_error!r}"
            )
        self.seg_mpp = seg_mpp
        self.device = device
        self.canny_min_component_frac = canny_min_component_frac
        self.canny_max_area_frac = canny_max_area_frac
        self.guard = guard
        self.canvas_dark = canvas_dark
        self.canvas_bright = canvas_bright
        self.canvas_var = canvas_var
        self.border_um = border_um
        self.frame_um = frame_um
        self.veto_cov = veto_cov
        self.keep_cov = keep_cov
        self.clip_min_area_mm2 = clip_min_area_mm2
        self.veto_slack_um = veto_slack_um
        self.removal_min_width_um = removal_min_width_um
        self.connectivity = connectivity
        self.veto_abort_cov = veto_abort_cov
        self.min_area_mm2 = min_area_mm2
        self.on_veto_error = on_veto_error
        self.last_stats: dict[str, Any] = {}

    # -- the single veto call site (subclass / monkeypatch seam) ---------------

    def _veto_mask(self, reader: WSISource) -> TissueMask:
        """GrandQC's stage-1 tissue mask, **always at its trained 10 µm/px**.

        ``mpp`` is deliberately not forwarded: running the checkpoint finer is
        off-domain and measured non-monotonic (on one cohort a control went
        6210 -> 6343 -> 4194 tiles and an artifact slide went 1405 -> 2329 -> 1956 as
        the MPP was refined), so a "sharper" veto is not a better veto.

        Everything deep sits behind this one call, so a caller that wants to memoise,
        batch or gate the model can override it without touching the algebra.
        """
        return GrandQCSegmenter(device=self.device).segment(reader)

    # -- terms -----------------------------------------------------------------

    def _union(self, reader: WSISource) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Guarded ``otsu | canny`` on otsu's grid; returns (U, level, ds, kill)."""
        from raw2features.viz import read_level_capped

        level = nearest_level(reader.mpp, reader.level_downsamples(), self.seg_mpp)
        # Otsu reads through read_level_capped, so its downsample may be the level's
        # downsample times an extra factor on a deficient pyramid. Always take the
        # reference grid from the Otsu result, never from level_downsamples()[level].
        img, factor = read_level_capped(reader, level)
        ds = float(reader.level_downsamples()[level]) * factor
        union_mpp = reader.mpp * ds if reader.mpp else None

        kill = (
            scan_region_kill(
                img,
                union_mpp,
                canvas_dark=self.canvas_dark,
                canvas_bright=self.canvas_bright,
                canvas_var=self.canvas_var,
                border_um=self.border_um,
                frame_um=self.frame_um,
            )
            if self.guard
            else np.zeros(img.shape[:2], bool)
        )

        otsu = OtsuSegmenter(seg_mpp=self.seg_mpp).segment(reader)
        shape = otsu.mask.shape
        kill_ref = _align(kill, shape)

        def _edge_filter(edges: np.ndarray, _img: np.ndarray, _mpp: float | None):
            k = _align(kill_ref, edges.shape)
            out = edges.copy()
            out[k] = 0
            return out

        canny = CannySegmenter(
            seg_mpp=self.seg_mpp,
            min_component_frac=self.canny_min_component_frac,
            edge_filter=_edge_filter if self.guard else None,
        ).segment(reader)
        cm = _align(canny.mask > 0, shape)
        degenerate = bool(cm.mean() > self.canny_max_area_frac)
        if degenerate:  # a filled outer ring swallowed the slide -> otsu alone
            cm = np.zeros(shape, bool)
        self.last_stats["canny_degenerate"] = degenerate

        u = ((otsu.mask > 0) | cm) & ~kill_ref
        self.last_stats["kill_frac"] = float(kill_ref.mean())
        return u, otsu.level, otsu.downsample, kill_ref

    # -- the seam --------------------------------------------------------------

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        self.last_stats = {}
        u, level, ds, _kill = self._union(reader)
        union_mpp = reader.mpp * ds if reader.mpp else None
        self.last_stats.update(union_mpp=union_mpp, level=level)

        if not u.any():  # nothing to veto; skip the model entirely (provably exact)
            self.last_stats.update(
                n_components=0, n_dropped=0, n_clipped=0, n_kept_intact=0,
                n_kept_small=0, veto_aborted=False, veto_removed_frac=0.0,
            )
            return TissueMask(np.zeros(u.shape, np.float32), level, ds)

        try:
            veto = self._veto_mask(reader)
            # Derived, never asserted: if _veto_mask is ever changed to run the
            # checkpoint off its trained 10 um/px this records the real value (and
            # the test that pins it to 10.0 then fails, as it should).
            self.last_stats["veto_mpp"] = (
                reader.mpp * veto.downsample if reader.mpp else None
            )
        except Exception as e:  # noqa: BLE001 - re-raised or explicitly degraded
            if self.on_veto_error == "raise":
                raise
            warnings.warn(
                f"{self.name}: GrandQC veto unavailable ({e!r}); emitting the "
                "unvetoed union -- artifacts will NOT be removed.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.last_stats.update(veto_aborted=True, veto_removed_frac=0.0)
            return TissueMask(u.astype(np.float32), level, ds)

        g = _align(veto.mask > 0, u.shape)
        cov_all = float(g[u].mean())
        self.last_stats["union_cov"] = cov_all
        if cov_all < self.veto_abort_cov:
            warnings.warn(
                f"{self.name}: GrandQC endorses only {cov_all:.3f} of the union; "
                "treating the veto as degenerate and emitting the union unclipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.last_stats.update(veto_aborted=True, veto_removed_frac=0.0)
            return TissueMask(u.astype(np.float32), level, ds)
        self.last_stats["veto_aborted"] = False

        # Slack-dilated veto: the clip may only remove pixels this far from anything
        # GrandQC calls tissue, so quantisation + ~1 mask px of model error is absorbed.
        r_slack = _px(self.veto_slack_um, union_mpp)
        gd = cv2.dilate(g.astype(np.uint8), _disk(r_slack)) > 0 if r_slack else g

        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            u.astype(np.uint8), connectivity=self.connectivity
        )
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        # Endorsed pixels per component, measured on the UNDILATED veto: this is a
        # statistic, not a clip, and dilating it would inflate every coverage.
        sums = np.bincount(lab[g], minlength=n).astype(np.float64)
        cov = np.zeros(n)
        cov[1:] = sums[1:] / np.maximum(areas[1:], 1.0)
        # A slide with no MPP has no physical area, so every component reads as "small"
        # and the segmenter degrades to a pure component veto -- the safe direction,
        # since the clip is the only branch that can erode a boundary.
        px_mm2 = (union_mpp / 1000.0) ** 2 if union_mpp else 0.0
        area_mm2 = areas * px_mm2

        drop = cov < self.veto_cov
        small = area_mm2 < self.clip_min_area_mm2
        keep_whole = ~drop & ((cov >= self.keep_cov) | small)
        clip = ~drop & ~keep_whole
        for arr in (drop, keep_whole, clip):
            arr[0] = False  # label 0 is background

        lut = np.zeros(n, np.uint8)
        lut[keep_whole] = 1
        out = lut[lab].astype(bool)

        r_open = (
            max(1, math.ceil(self.removal_min_width_um / 2.0 / union_mpp))
            if (self.removal_min_width_um > 0 and union_mpp)
            else 0
        )
        for c in np.nonzero(clip)[0]:
            x, y, w, h = (
                stats[c, cv2.CC_STAT_LEFT], stats[c, cv2.CC_STAT_TOP],
                stats[c, cv2.CC_STAT_WIDTH], stats[c, cv2.CC_STAT_HEIGHT],
            )
            sl = (slice(y, y + h), slice(x, x + w))
            m = lab[sl] == c
            removed = m & ~gd[sl]
            if r_open:
                # A real artifact is a blob; a mis-registration is a sliver. Keep only
                # removal regions that are >= removal_min_width_um wide somewhere.
                seeds = cv2.morphologyEx(
                    removed.astype(np.uint8), cv2.MORPH_OPEN, _disk(r_open)
                ).astype(bool)
                removed = _reconstruct(removed, seeds)
            out[sl] |= m & ~removed

        if self.min_area_mm2 > 0 and out.any() and px_mm2 > 0:
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(
                out.astype(np.uint8), connectivity=self.connectivity
            )
            big = st2[:, cv2.CC_STAT_AREA].astype(np.float64) * px_mm2 >= (
                self.min_area_mm2
            )
            big[0] = False
            lut2 = np.zeros(n2, np.uint8)
            lut2[big] = 1
            out = lut2[lab2].astype(bool)

        self.last_stats.update(
            n_components=int(n - 1),
            n_dropped=int(drop.sum()),
            n_clipped=int(clip.sum()),
            n_kept_intact=int((keep_whole & (cov >= self.keep_cov)).sum()),
            n_kept_small=int((keep_whole & small & (cov < self.keep_cov)).sum()),
            veto_removed_frac=float(1.0 - out.sum() / max(int(u.sum()), 1)),
        )
        return TissueMask(out.astype(np.float32), level, ds)
