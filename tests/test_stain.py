"""Macenko stain normalization (core/stain.py) + the patch-embedding wiring."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from raw2features.core.stain import (
    macenko_fit,
    macenko_normalize,
    make_normalizer,
    vahadane_fit,
)

# vahadane needs scikit-learn, an *optional* dep (it gates with a clear RuntimeError).
# CI's core/full envs don't install it, so skip that parametrization there.
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _he_image(h=64, w=64):
    """A tiny H&E-ish RGB image: purple + pink stains on a light background."""
    img = np.full((h, w, 3), 230, np.uint8)
    img[: h // 2, : w // 2] = [150, 80, 160]  # purple (hematoxylin-ish)
    img[h // 2 :, w // 2 :] = [220, 120, 180]  # pink (eosin-ish)
    return img


def test_macenko_returns_same_shape_uint8_and_remaps():
    img = _he_image()
    out = macenko_normalize(img)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)  # colour was remapped


def test_macenko_blank_image_unchanged():
    img = np.full((32, 32, 3), 245, np.uint8)  # all background -> no tissue to estimate
    np.testing.assert_array_equal(macenko_normalize(img), img)


def test_black_padding_does_not_poison_the_fit():
    # OME-Zarr canvas padding (black) is high-OD; without a near-black drop it counts as
    # tissue and skews the whole-slide fit. With the drop, the fit is unchanged.
    img = _he_image()
    he_clean, _ = macenko_fit(img)
    padded = np.zeros((img.shape[0] + 40, img.shape[1], 3), np.uint8)
    padded[: img.shape[0]] = img  # a black strip below the tissue
    he_pad, _ = macenko_fit(padded)
    assert he_clean is not None and he_pad is not None
    np.testing.assert_allclose(he_clean, he_pad, atol=1e-6)  # black dropped -> same fit


def test_fit_excludes_black_padding_as_nontissue():
    # an almost-all-black image has too few real-tissue px to fit -> None (must not fit
    # on black). Covers both the macenko and vahadane tissue selection.
    img = np.zeros((64, 64, 3), np.uint8)
    img[:1, :8] = [150, 80, 160]  # a handful of tissue px, << the 100 floor
    assert macenko_fit(img)[0] is None
    if _HAS_SKLEARN:
        assert vahadane_fit(img)[0] is None


# -- make_normalizer: fit the slide stain once, apply per patch ------------------


def test_make_normalizer_fits_and_applies_to_patches():
    norm = make_normalizer("macenko", _he_image())  # fit from a "thumbnail"
    assert callable(norm)
    patch = np.full((16, 16, 3), 200, np.uint8)
    patch[:8] = [150, 80, 160]
    out = norm(patch)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert not np.array_equal(out, patch)  # the slide's stain was applied


@pytest.mark.parametrize("method", [
    "macenko",
    "reinhard",
    pytest.param(
        "vahadane",
        marks=pytest.mark.skipif(not _HAS_SKLEARN, reason="needs scikit-learn"),
    ),
])
def test_make_normalizer_all_methods_change_patches(method):
    norm = make_normalizer(method, _he_image())  # fit the slide stain once
    assert callable(norm)
    patch = np.full((16, 16, 3), 200, np.uint8)
    patch[:8] = [150, 80, 160]
    out = norm(patch)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert not np.array_equal(out, patch)


def test_make_normalizer_none_when_off_or_degenerate():
    ref = _he_image()
    assert make_normalizer(None, ref) is None
    assert make_normalizer("unknown", ref) is None
    # a blank (tissue-free) reference can't be fitted -> no normalizer
    assert make_normalizer("macenko", np.full((32, 32, 3), 245, np.uint8)) is None


# -- end to end: --stain-norm changes the embedded features ---------------------


def _color_slide(path):
    """build_ngff_v04, but with H&E-ish colour content so Macenko can fit a stain."""
    import zarr

    from conftest import build_ngff_v04

    build_ngff_v04(path)
    g = zarr.open_group(path, mode="r+")
    for k in ("0", "1", "2"):
        a = g[k]
        a[0, :, 0] = _he_image(a.shape[3], a.shape[4]).transpose(2, 0, 1)
    return path


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_stain_norm_changes_features_and_records(tmp_path):
    """Embedding with --stain-norm macenko changes the features and is recorded."""
    import zarr

    from conftest import MockEmbedder
    from raw2features.core.store import open_grid
    from raw2features.pipeline.runner import RunConfig, embed_slide

    slide = _color_slide(str(tmp_path / "C.zarr"))

    def run(out, stain):
        cfg = RunConfig(models=["mock"], no_seg=True, target_mpp=0.5, patch_px=64,
                        device="cpu", amp="fp32", stain_norm=stain)
        embed_slide(slide, out, cfg,
                    embedders=[MockEmbedder(dim=8, input_size=64, name="mock")])
        return os.path.join(out, "C.embeddings.zarr")

    raw = run(str(tmp_path / "raw"), None)
    norm = run(str(tmp_path / "norm"), "macenko")
    fr = np.asarray(open_grid(raw)["features"]["mock"][:])
    fn = np.asarray(open_grid(norm)["features"]["mock"][:])
    assert fr.shape == fn.shape and fr.size > 0
    assert not np.allclose(fr, fn)  # stain-norm changed patches -> changed features
    job = dict(zarr.open_group(norm, mode="r").attrs["raw2features"]).get("job", {})
    assert job.get("stain_norm") == "macenko"
