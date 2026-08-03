"""Tests for the vetoed-union segmenter ``(otsu | canny)`` + guard + GrandQC veto.

Two things make these tests non-vacuous:

* **The fixtures have real edges.** The shared ``synthetic_ngff`` fixture is a smooth
  ramp, so Canny returns nothing on it and every edge/union assertion would pass
  trivially. ``zoned_ngff`` below paints three separated blocks whose colour/texture
  make Otsu and Canny *disagree*, and ``canvas_ngff`` paints a black scanner canvas
  around the scan region.
* **The GrandQC veto is injected.** The real veto needs the ``[grandqc]`` extra, a
  GPU and CC-BY-NC-SA weights, so tests patch the single ``_veto_mask`` call site with
  a hand-drawn mask on a *coarser* grid (10 µm/px on a 4 µm/px slide), which also
  exercises the veto-to-union resampling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import zarr

from conftest import build_ngff_v04
from raw2features.readers.omezarr import OmeZarrReader
from raw2features.segmenters.base import TissueMask
from raw2features.segmenters.canny import CannySegmenter
from raw2features.segmenters.grandqc_veto import (
    GrandQCVetoSegmenter,
    _align,
    scan_region_kill,
)
from raw2features.segmenters.otsu import OtsuSegmenter

# Level-0 MPP 4.0 with a /2 pyramid -> level MPPs 4 / 8 / 16, so the production
# ``seg_mpp=8.0`` default resolves to level 1 (240x240) and every micron-denominated
# parameter (border_um, veto_slack_um, clip_min_area_mm2, ...) lands in its designed
# regime rather than being scaled into absurdity by a toy MPP.
SIZES = ((480, 480), (240, 240), (120, 120))
MPP0 = 4.0
VETO_DS = 2.5  # 10.0 µm/px veto grid / 4.0 µm/px level 0
VETO_N = int(round(SIZES[0][0] / VETO_DS))  # 192

# (row0, row1, col0, col1) as fractions of the level, so they hold at every level.
BOXES = {
    "A": (0.10, 0.40, 0.10, 0.40),  # saturated + smooth -> otsu YES, canny outline
    "B": (0.10, 0.40, 0.60, 0.90),  # desaturated + textured -> otsu NO, canny YES
    "C": (0.60, 0.90, 0.60, 0.90),  # saturated junk -> union YES, veto will reject it
}


def _slice(name: str, h: int, w: int):
    r0, r1, c0, c1 = BOXES[name]
    return slice(int(r0 * h), int(r1 * h)), slice(int(c0 * w), int(c1 * w))


def _frac(mask: np.ndarray, name: str) -> float:
    h, w = mask.shape
    return float(mask[_slice(name, h, w)].mean())


def _paint_zones(h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), 230, np.uint8)  # desaturated smooth background
    img[_slice("A", h, w)] = (200, 40, 60)  # saturated block
    ys, xs = _slice("B", h, w)
    hh, ww = ys.stop - ys.start, xs.stop - xs.start
    sq = max(2, h // 40)
    yy, xx = np.mgrid[0:hh, 0:ww]
    checks = np.where((((yy // sq) + (xx // sq)) % 2).astype(bool), 120, 200)
    for c in range(3):  # grey checkerboard: no saturation, plenty of edges
        img[ys, xs, c] = checks.astype(np.uint8)
    img[_slice("C", h, w)] = (40, 60, 200)  # a second saturated block
    return img


def _write(path: str, painter) -> str:
    p = build_ngff_v04(path, mpp0=MPP0, sizes=SIZES)
    g = zarr.open_group(p, mode="r+", zarr_format=2)
    for i, (h, w) in enumerate(SIZES):
        img = painter(h, w)
        for c in range(3):
            g[str(i)][0, c, 0] = img[:, :, c]
    return p


@pytest.fixture
def zoned_ngff(tmp_path) -> str:
    """Three well-separated blocks that make Otsu and Canny disagree."""
    return _write(str(tmp_path / "zoned.zarr"), _paint_zones)


CANVAS_MARGIN = 0.06
TISSUE_BOX = (0.30, 0.70)


def _paint_canvas(h: int, w: int) -> np.ndarray:
    img = np.full((h, w, 3), 230, np.uint8)
    r0, r1 = int(TISSUE_BOX[0] * h), int(TISSUE_BOX[1] * h)
    c0, c1 = int(TISSUE_BOX[0] * w), int(TISSUE_BOX[1] * w)
    img[r0:r1, c0:c1] = (200, 40, 60)
    m = int(CANVAS_MARGIN * h)  # black synthetic canvas around the scan region
    img[:m] = img[-m:] = 0
    img[:, :m] = img[:, -m:] = 0
    return img


@pytest.fixture
def canvas_ngff(tmp_path) -> str:
    """A scan region padded with black canvas -- the slide-edge failure mode."""
    return _write(str(tmp_path / "canvas.zarr"), _paint_canvas)


@pytest.fixture
def blank_ngff(tmp_path) -> str:
    """Uniform white glass: no saturation, no edges, so the union is empty."""
    return _write(
        str(tmp_path / "blank.zarr"),
        lambda h, w: np.full((h, w, 3), 255, np.uint8),
    )


def _veto(monkeypatch, arr: np.ndarray, counter: list | None = None):
    """Patch the single veto call site with a hand-drawn 10 µm/px mask."""

    def fake(self, reader):
        if counter is not None:
            counter.append(1)
        return TissueMask(mask=arr.astype(np.float32), level=0, downsample=VETO_DS)

    monkeypatch.setattr(GrandQCVetoSegmenter, "_veto_mask", fake)


def _boxes(*names: str, pad: float = 0.04) -> np.ndarray:
    m = np.zeros((VETO_N, VETO_N), np.float32)
    for n in names:
        r0, r1, c0, c1 = BOXES[n]
        m[
            int((r0 - pad) * VETO_N) : int((r1 + pad) * VETO_N),
            int((c0 - pad) * VETO_N) : int((c1 + pad) * VETO_N),
        ] = 1.0
    return m


# -- the seam ------------------------------------------------------------------


def test_returns_a_valid_binary_tissue_mask_on_the_otsu_grid(zoned_ngff, monkeypatch):
    _veto(monkeypatch, _boxes("A", "B"))
    seg = GrandQCVetoSegmenter()
    with OmeZarrReader(zoned_ngff) as r:
        ref = OtsuSegmenter(seg_mpp=8.0).segment(r)
        tm = seg.segment(r)

    assert isinstance(tm, TissueMask)
    assert tm.mask.dtype == np.float32 and tm.mask.ndim == 2
    # Strictly binary: --tissue-threshold must keep its coverage meaning, so tile
    # counts stay comparable with plain otsu / canny runs.
    assert set(np.unique(tm.mask)).issubset({0.0, 1.0})
    # The reference grid is Otsu's, including any read_level_capped downscale factor.
    assert tm.mask.shape == ref.mask.shape
    assert tm.level == ref.level and tm.downsample == ref.downsample
    assert seg.last_stats["veto_mpp"] == 10.0  # never run off its trained scale


def test_registered_and_no_arg_constructible():
    from raw2features.core import plugins

    seg = plugins.get("segmenters", "grandqc_veto")()  # pipeline builds with no args
    assert seg.name == "grandqc_veto"


def test_rejects_nonsense_construction():
    with pytest.raises(ValueError):
        GrandQCVetoSegmenter(connectivity=6)
    with pytest.raises(ValueError):
        GrandQCVetoSegmenter(on_veto_error="ignore")


# -- the union term ------------------------------------------------------------


def test_union_recovers_pale_tissue_that_otsu_alone_misses(zoned_ngff, monkeypatch):
    """Zone B is grey (no saturation) but textured: Otsu drops it, Canny fills it."""
    _veto(monkeypatch, _boxes("A", "B"))
    with OmeZarrReader(zoned_ngff) as r:
        otsu = OtsuSegmenter(seg_mpp=8.0).segment(r).mask > 0
        out = GrandQCVetoSegmenter().segment(r).mask > 0

    assert _frac(otsu, "B") < 0.01, "fixture must be one Otsu genuinely misses"
    assert _frac(out, "B") > 0.90, "the canny term should recover it"
    assert _frac(otsu, "A") > 0.90 and _frac(out, "A") > 0.90  # otsu's own kept


# -- the veto ------------------------------------------------------------------


def test_veto_drops_a_component_the_union_kept(zoned_ngff, monkeypatch):
    _veto(monkeypatch, _boxes("A", "B"))
    seg = GrandQCVetoSegmenter()
    with OmeZarrReader(zoned_ngff) as r:
        union, _, _, _ = GrandQCVetoSegmenter()._union(r)
        out = seg.segment(r).mask > 0

    assert _frac(union, "C") > 0.90, "the union must actually keep the junk block"
    assert _frac(out, "C") == 0.0, "the veto must remove it"
    assert _frac(out, "A") > 0.90 and _frac(out, "B") > 0.90  # endorsed zones survive
    assert not np.any(out & ~union), "the veto may only remove, never add"
    assert seg.last_stats["n_dropped"] == 1
    assert 0.2 < seg.last_stats["veto_removed_frac"] < 0.5


def test_clip_branch_erodes_a_large_mixed_component(zoned_ngff, monkeypatch):
    """A big component that is half-endorsed is clipped; a pure component veto is not.

    This is the whole point of the hybrid: boundary erosion is cheap on large
    components, and large components are exactly the ones that can fuse real tissue
    with an attached artifact.
    """
    half = np.zeros((VETO_N, VETO_N), np.float32)
    r0, r1, c0, c1 = BOXES["A"]
    half[int((r0 - 0.04) * VETO_N) : int((r1 + 0.04) * VETO_N),
         int((c0 - 0.04) * VETO_N) : int(((c0 + c1) / 2) * VETO_N)] = 1.0  # left half
    r0, r1, c0, c1 = BOXES["B"]
    half[int((r0 - 0.04) * VETO_N) : int((r1 + 0.04) * VETO_N),
         int((c0 - 0.04) * VETO_N) : int((c1 + 0.04) * VETO_N)] = 1.0
    _veto(monkeypatch, half)

    hybrid, component = GrandQCVetoSegmenter(), GrandQCVetoSegmenter(
        clip_min_area_mm2=math.inf  # documented corner case: clip never fires
    )
    with OmeZarrReader(zoned_ngff) as r:
        h = hybrid.segment(r).mask > 0
        c = component.segment(r).mask > 0

    assert 0.3 < _frac(h, "A") < 0.7, "clipped back to roughly the endorsed half"
    assert _frac(c, "A") > 0.90, "component veto keeps the mixed component whole"
    assert hybrid.last_stats["n_clipped"] == 1
    assert component.last_stats["n_clipped"] == 0
    assert _frac(h, "C") == 0.0 and _frac(c, "C") == 0.0  # both still drop the junk


def test_slack_and_width_guards_protect_a_thin_unendorsed_sliver(
    zoned_ngff, monkeypatch
):
    """A sliver of disagreement is mis-registration, not an artifact -- keep it."""
    sliver = _boxes("A", "B")
    sliver[:, int(0.24 * VETO_N) : int(0.24 * VETO_N) + 2] = 0.0  # ~2 veto px wide
    _veto(monkeypatch, sliver)

    with OmeZarrReader(zoned_ngff) as r:
        default = GrandQCVetoSegmenter().segment(r).mask > 0
        # keep_cov high enough to force the clip branch in all three variants
        no_guards = GrandQCVetoSegmenter(
            keep_cov=1.1, veto_slack_um=0.0, removal_min_width_um=0.0
        ).segment(r)
        width_only = GrandQCVetoSegmenter(keep_cov=1.1, veto_slack_um=0.0).segment(r)

    assert _frac(default, "A") > 0.95, "veto_slack_um must absorb the sliver"
    assert _frac(no_guards.mask > 0, "A") < _frac(default, "A"), (
        "with both guards off the sliver is eroded away -- proves the guards act"
    )
    assert _frac(width_only.mask > 0, "A") == _frac(default, "A"), (
        "opening-by-reconstruction alone must also put the sliver back"
    )


def test_pure_pixelwise_and_is_exactly_union_and_veto(zoned_ngff, monkeypatch):
    """The documented corner case (a) must be a bit-exact pixel-wise AND."""
    g10 = _boxes("A", "B", pad=0.0)
    _veto(monkeypatch, g10)
    seg = GrandQCVetoSegmenter(
        veto_cov=0.0,
        keep_cov=1.1,
        clip_min_area_mm2=0.0,
        veto_slack_um=0.0,
        removal_min_width_um=0.0,
    )
    with OmeZarrReader(zoned_ngff) as r:
        union, _, _, _ = GrandQCVetoSegmenter()._union(r)
        out = seg.segment(r).mask > 0

    assert np.array_equal(out, union & _align(g10 > 0, union.shape))
    assert out.sum() < union.sum()  # non-trivial: the AND actually removes something


def test_degenerate_veto_emits_the_union_and_warns(zoned_ngff, monkeypatch):
    """An empty veto prediction must never silently return zero tiles."""
    _veto(monkeypatch, np.zeros((VETO_N, VETO_N), np.float32))
    seg = GrandQCVetoSegmenter()
    with OmeZarrReader(zoned_ngff) as r:
        union, _, _, _ = GrandQCVetoSegmenter()._union(r)
        with pytest.warns(RuntimeWarning, match="degenerate"):
            out = seg.segment(r).mask > 0

    assert np.array_equal(out, union)
    assert seg.last_stats["veto_aborted"] is True


def test_veto_failure_raises_by_default_and_can_degrade_to_the_union(
    zoned_ngff, monkeypatch
):
    def boom(self, reader):
        raise RuntimeError("no [grandqc] extra")

    monkeypatch.setattr(GrandQCVetoSegmenter, "_veto_mask", boom)
    with OmeZarrReader(zoned_ngff) as r:
        with pytest.raises(RuntimeError, match="grandqc"):
            GrandQCVetoSegmenter().segment(r)
        with pytest.warns(RuntimeWarning, match="artifacts will NOT be removed"):
            out = GrandQCVetoSegmenter(on_veto_error="warn").segment(r).mask > 0
        union, _, _, _ = GrandQCVetoSegmenter()._union(r)
    assert np.array_equal(out, union)


def test_empty_union_short_circuits_without_running_the_veto(blank_ngff, monkeypatch):
    calls: list = []
    _veto(monkeypatch, _boxes("A"), counter=calls)
    seg = GrandQCVetoSegmenter()
    with OmeZarrReader(blank_ngff) as r:
        tm = seg.segment(r)
    assert tm.mask.sum() == 0
    assert calls == [], "the GPU term must be skipped when there is nothing to veto"
    assert seg.last_stats["n_components"] == 0


# -- the scan-region / canvas guard --------------------------------------------


def test_scan_region_kill_finds_the_canvas_not_the_tissue():
    img = _paint_canvas(240, 240)
    kill = scan_region_kill(img, 8.0)
    h = w = 240
    canvas = np.zeros((h, w), bool)
    m = int(CANVAS_MARGIN * h)
    canvas[:m] = canvas[-m:] = True
    canvas[:, :m] = canvas[:, -m:] = True
    tissue = np.zeros((h, w), bool)
    r0, r1 = int(TISSUE_BOX[0] * h), int(TISSUE_BOX[1] * h)
    tissue[r0:r1, r0:r1] = True

    assert kill[canvas].all(), "the synthetic fill itself must be killed"
    assert not kill[tissue].any(), "and it must not reach the specimen"
    # No canvas at all -> only the thin perimeter ring, so the guard is a no-op inside.
    plain = scan_region_kill(_paint_zones(240, 240), 8.0)
    assert plain.mean() < 0.06


def test_guard_rescues_the_canny_term_on_a_canvas_slide(canvas_ngff, monkeypatch):
    """The load-bearing property: guard the EDGE MAP, not the finished mask.

    Unguarded, the scan-region rectangle is Canny's outer contour and the real tissue
    inside it becomes a ``max_hole_frac`` hole that gets carved back to background --
    so the border does not merely add junk, it deletes the tissue.
    """
    with OmeZarrReader(canvas_ngff) as r:
        unguarded = CannySegmenter(seg_mpp=8.0).segment(r).mask > 0
        guarded = GrandQCVetoSegmenter(guard=True)
        _veto(monkeypatch, np.ones((VETO_N, VETO_N), np.float32))
        out = guarded.segment(r).mask > 0
        union_off, _, _, _ = GrandQCVetoSegmenter(guard=False)._union(r)

    h = w = out.shape[0]
    canvas = np.zeros((h, w), bool)
    m = int(CANVAS_MARGIN * h)
    canvas[:m] = canvas[-m:] = True
    canvas[:, :m] = canvas[:, -m:] = True
    tissue = np.zeros((h, w), bool)
    r0, r1 = int(TISSUE_BOX[0] * h), int(TISSUE_BOX[1] * h)
    tissue[r0:r1, r0:r1] = True

    assert unguarded[tissue].mean() == 0.0, "unguarded canny loses the tissue entirely"
    assert union_off[canvas].mean() > 0.05, "and keeps a bogus border ring"
    assert out[canvas].mean() == 0.0, "the guard removes the canvas and its ring"
    assert out[tissue].mean() > 0.90, "while the specimen survives"


def test_canny_edge_filter_hook_sees_the_raw_edge_map(zoned_ngff):
    """The hook must run on the edges, with the level's own image and MPP."""
    seen: list = []

    def kill_all(edges, img, level_mpp):
        seen.append((edges.shape, img.shape[:2], level_mpp, int((edges > 0).sum())))
        return np.zeros_like(edges)

    with OmeZarrReader(zoned_ngff) as r:
        plain = CannySegmenter(seg_mpp=8.0).segment(r).mask
        blanked = CannySegmenter(seg_mpp=8.0, edge_filter=kill_all).segment(r).mask

    assert plain.sum() > 0 and blanked.sum() == 0
    (eshape, ishape, mpp, n_edges) = seen[0]
    assert eshape == ishape == plain.shape
    assert mpp == pytest.approx(8.0)  # level MPP, so a filter can size itself in µm
    assert n_edges > 0, "the hook must see the edge map, not an already-filled mask"


def test_align_or_pools_on_the_downsample_path():
    """INTER_NEAREST on the downsample path deletes thin structures; ours must not."""
    m = np.zeros((64, 64), bool)
    m[:, 30] = True  # a 1-px wide line
    small = _align(m, (16, 16))
    assert small.any(), "a thin structure must survive an 4x downsample"
    big = _align(m, (128, 128))
    assert big.sum() > m.sum()
