"""--read-block (block reads) must be BIT-IDENTICAL to the per-patch decode path.

If the decoded patches match byte-for-byte, the embeddings match for any deterministic
embedder, so this is the real equivalence proof. Covers: ds==1 (native read) and ds>1
(downsampled read, exercising the round(x/ds) offset mapping), a zero-padded edge patch,
the multichannel path, and the parallel (executor) path. The end-to-end smoke just
confirms the flag threads through run_slide.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import zarr

from raw2features.patcher.grid import GridPatcher
from raw2features.pipeline.runner import _decode_batch
from raw2features.readers.omezarr import OmeZarrReader


def _grid_coords(g, W, H):
    """All in-bounds grid coords + one deliberate edge-overhang coord (zero-pad)."""
    step, lp = g.level0_step, g.level0_patch
    coords = [(x, y)
              for y in range(0, max(H - lp, 0) + 1, step)
              for x in range(0, max(W - lp, 0) + 1, step)]
    coords.append((max(W - lp // 2, 0), max(H - lp // 2, 0)))  # overhangs -> zero-pad
    return np.asarray(coords, dtype=np.int32)


# ds==1 & ds==2 (downsampled read) x several block sizes (the tunable --read-block)
@pytest.mark.parametrize("target_mpp", [0.5, 1.0])
@pytest.mark.parametrize("read_block", [4, 8, 16])
def test_blocked_decode_bit_identical_rgb(synthetic_ngff, target_mpp, read_block):
    pp = 64
    with OmeZarrReader(synthetic_ngff) as r:
        g = GridPatcher(target_mpp=target_mpp, patch_px=pp).build_grid(r)
        W, H = r.level_dimensions[0]
        coords = _grid_coords(g, W, H)
        args = (coords, g.read_level, g.read_px, pp)
        per = _decode_batch(r, *args, None, False, read_block=1)
        blk = _decode_batch(r, *args, None, False, read_block=read_block)
        with ThreadPoolExecutor(4) as ex:
            blk_par = _decode_batch(r, *args, ex, False, read_block=read_block)
    assert len(per) == len(blk) == len(blk_par) == len(coords)
    for i in range(len(coords)):
        assert np.array_equal(per[i], blk[i]), f"serial patch {i} @ {coords[i]} differs"
        assert np.array_equal(per[i], blk_par[i]), f"parallel patch {i} differs"


@pytest.mark.parametrize("read_block", [4, 8])
def test_blocked_decode_bit_identical_multichannel(
    synthetic_multiplex_ngff, read_block
):
    pp = 32
    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        g = GridPatcher(target_mpp=0.5, patch_px=pp).build_grid(r)
        W, H = r.level_dimensions[0]
        coords = _grid_coords(g, W, H)
        args = (coords, g.read_level, g.read_px, pp)
        per = _decode_batch(r, *args, None, True, read_block=1)
        blk = _decode_batch(r, *args, None, True, read_block=read_block)
    assert len(per) == len(blk) == len(coords)
    for i in range(len(coords)):
        p, c = per[i], blk[i]
        assert p.shape == c.shape and p.dtype == c.dtype
        assert np.array_equal(p, c), f"multichannel patch {i} @ {coords[i]} differs"


def _make_anisotropic_translated(store: str) -> None:
    """Give levels >=1 an anisotropic x scale and a per-level translation.

    Exercises the per-axis + translation read mapping (read_level_mapping): the block
    path and the per-patch path must still agree byte-for-byte, since both go through
    the one mapping. Level 0 is left as the identity reference.
    """
    g = zarr.open_group(store, mode="r+")
    attrs = dict(g.attrs)
    ms = attrs["multiscales"][0]
    for level in range(1, len(ms["datasets"])):
        cts = ms["datasets"][level]["coordinateTransformations"]
        for t in cts:
            if t["type"] == "scale":
                s = list(t["scale"])
                s[-1] *= 1.25  # x coarser than y -> anisotropic downsample
                t["scale"] = s
        n = len(cts[0]["scale"])
        # sub-pixel translation (x, y) that differs per level
        cts.append({"type": "translation",
                    "translation": [0.0] * (n - 2) + [0.4 * level, 0.7 * level]})
    attrs["multiscales"][0] = ms
    g.attrs.update(attrs)


@pytest.mark.parametrize("read_block", [4, 8])
def test_blocked_decode_bit_identical_aniso_translated(synthetic_ngff, read_block):
    # A downsampled read at a level that carries anisotropy + translation: the offset
    # mapping is non-trivial, and block vs per-patch must still match exactly.
    _make_anisotropic_translated(synthetic_ngff)
    pp, read_level, read_px = 32, 1, 48
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # anisotropy warning is expected
        with OmeZarrReader(synthetic_ngff) as r:
            dsx, dsy, ox, oy = r.read_level_mapping(read_level)
            assert dsx != dsy and (ox, oy) != (0.0, 0.0)  # mapping is non-trivial
            W, H = r.level_dimensions[0]
            coords = np.asarray(
                [(x, y) for y in range(0, H - 80, 40) for x in range(0, W - 80, 40)],
                dtype=np.int32,
            )
            args = (coords, read_level, read_px, pp)
            per = _decode_batch(r, *args, None, False, read_block=1)
            blk = _decode_batch(r, *args, None, False, read_block=read_block)
            with ThreadPoolExecutor(4) as ex:
                blk_par = _decode_batch(r, *args, ex, False, read_block=read_block)
    assert len(per) == len(blk) == len(coords)
    for i in range(len(coords)):
        assert np.array_equal(per[i], blk[i]), f"serial patch {i} @ {coords[i]} differs"
        assert np.array_equal(per[i], blk_par[i]), f"parallel patch {i} differs"


def test_run_slide_read_block_threads_through(synthetic_ngff, tmp_path):
    """End-to-end smoke: run_slide accepts read_block and produces a valid store with
    the same shape (the bit-identity of the features follows from the decode tests
    above - identical decoded patches -> identical embeddings)."""
    pytest.importorskip("torch")

    from conftest import MockEmbedder
    from raw2features.pipeline.runner import RunConfig, run_slide

    common = dict(models=["mock"], no_seg=True, target_mpp=0.5, patch_px=64,
                  device="cpu", amp="fp32", batch_size=4)
    s = run_slide(synthetic_ngff, str(tmp_path / "out"),
                  RunConfig(read_block=8, **common), embedders=[MockEmbedder()])
    assert s["status"] == "complete"
    from raw2features.core.store import open_grid

    g = open_grid(s["output_uri"])  # the sole grid
    assert g["features"]["mock"].shape == (s["n_patches"], 8)
