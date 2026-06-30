"""Regression tests for the four correctness fixes from the adversarial review.

A: omezarr reader rescales non-uint8 sources (never truncates mod 256).
B: the patcher resamples reads to exactly patch_px (exact-MPP, model-independent).
C: validate_output scans the full array + rejects all-zero fill tails.
D: --no-seg omits the mask array (matches the documented schema).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from conftest import MockEmbedder
from raw2features.core.geometry import Point, Region, Size
from raw2features.core.store import GRIDS, open_grid
from raw2features.patcher.grid import resample_patch
from raw2features.pipeline.receipt import validate_output
from raw2features.pipeline.runner import RunConfig, run_slide
from raw2features.readers.omezarr import OmeZarrReader

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _build_uint16_ngff(path, value: int, *, size: int = 32) -> str:
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    a = g.create_array(
        "0", shape=(1, 3, 1, size, size), chunks=(1, 3, 1, size, size), dtype="uint16"
    )
    a[0, 0, 0] = value
    a[0, 1, 0] = value
    a[0, 2, 0] = value
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    datasets = [
        {
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0, 0.5, 0.5]}
            ],
        }
    ]
    g.attrs["multiscales"] = [{"version": "0.4", "axes": axes, "datasets": datasets}]
    return str(path)


# -- A: dtype-safe pixel conversion ----------------------------------------
def test_reader_uint16_rescales_not_truncates(tmp_path):
    value = 60000
    path = _build_uint16_ngff(tmp_path / "u16.zarr", value)
    with OmeZarrReader(path) as r:
        patch = r.read_region(Region(0, Point(0, 0), Size(16, 16)))
    assert patch.dtype == np.uint8
    expected = round(value * 255 / 65535)  # ~233 (bright tissue)
    assert abs(int(patch[0, 0, 0]) - expected) <= 1
    assert int(patch[0, 0, 0]) != value % 256  # NOT the truncating value (96)


def test_reader_handles_float_and_rejects_truly_unsupported_dtype():
    # float is now read (range-detected; see test_reader_omezarr), but a genuinely
    # unsupported dtype (e.g. complex) still raises rather than silently wrap.
    assert OmeZarrReader._to_uint8(np.zeros((2, 2, 3), np.float32)).dtype == np.uint8
    with pytest.raises(NotImplementedError):
        OmeZarrReader._to_uint8(np.zeros((2, 2, 3), dtype=np.complex64))


# -- B: resample to exactly patch_px ---------------------------------------
def test_resample_patch_downscale_and_noop():
    rng = np.random.RandomState(0)
    patch = (rng.rand(96, 96, 3) * 255).astype(np.uint8)
    out = resample_patch(patch, 64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    same = resample_patch(patch, 96)
    assert same is patch  # exact-size read is a no-op (no needless resample)


# -- C: full-array + all-zero-tail validation ------------------------------
def test_validate_output_rejects_unwritten_tail(tmp_path):
    n, dim = 300, 8  # tail (rows 256-299) is beyond the old 256-row window
    path = str(tmp_path / "x.embeddings.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=2)
    g = root.require_group(GRIDS).require_group("mpp1_px224")  # uniform grid nesting
    coords = g.create_array("coords", shape=(n, 2), chunks=(n, 2), dtype="int32")
    coords[:] = np.zeros((n, 2), dtype="int32")
    feats = g.create_group("features")
    arr = feats.create_array("m", shape=(n, dim), chunks=(n, dim), dtype="float16")
    arr[:256] = np.ones((256, dim), dtype="float16")  # tail left at fill 0.0
    uri = f"file://{path}"
    assert validate_output(uri, ["m"], n) is False  # all-zero tail -> invalid
    arr[:] = np.ones((n, dim), dtype="float16")
    assert validate_output(uri, ["m"], n) is True  # fully written -> valid


# -- D: --no-seg omits the mask --------------------------------------------
@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_no_seg_omits_mask(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        batch_size=8,
    )
    summary = run_slide(
        synthetic_ngff, str(tmp_path / "out"), cfg, embedders=[MockEmbedder(bias=1.0)]
    )
    assert summary["status"] == "complete"
    g = open_grid(summary["output_uri"])  # the sole grid
    assert "mask" not in g
