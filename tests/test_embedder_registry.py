"""Tests for the model registry and embedder construction (no weights needed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

import numpy as np  # noqa: E402

from raw2features.embedders.model_registry import (  # noqa: E402
    build_embedder,
    get_spec,
    load_registry,
)
from raw2features.embedders.timm_embedder import TimmEmbedder  # noqa: E402
from raw2features.embedders.torchvision_embedder import (  # noqa: E402
    TorchvisionEmbedder,
)


def test_registry_has_v1_models_with_sourced_transforms():
    reg = load_registry()
    for name in ("resnet50", "dinov2", "uni", "uni2_h"):
        assert name in reg
        spec = reg[name]
        # provenance is mandatory
        assert spec.transform_source_url.startswith("http")
        assert len(spec.mean) == 3 and len(spec.std) == 3
        assert spec.input_size == 224
    assert reg["uni"].gated and reg["uni2_h"].gated
    assert not reg["resnet50"].gated and not reg["dinov2"].gated
    assert reg["uni"].embedding_dim == 1024
    assert reg["uni2_h"].embedding_dim == 1536
    assert reg["resnet50"].embedding_dim == 2048


def test_build_embedder_resolves_family_class():
    assert isinstance(build_embedder("resnet50"), TorchvisionEmbedder)
    assert isinstance(build_embedder("uni"), TimmEmbedder)
    assert isinstance(build_embedder("dinov2"), TimmEmbedder)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_spec("does-not-exist")


def test_transform_normalises_to_chw_float():
    emb = build_embedder("resnet50")
    patch = np.full((224, 224, 3), 128, dtype=np.uint8)
    t = emb.transform(patch)
    assert tuple(t.shape) == (3, 224, 224)
    assert t.dtype.is_floating_point
    # (128/255 - 0.485)/0.229 ~= 0.073 on R channel
    expected = (128 / 255 - 0.485) / 0.229
    assert abs(float(t[0].mean()) - expected) < 1e-3


def test_transform_resizes_when_patch_size_mismatches():
    emb = build_embedder("resnet50")
    patch = np.full((128, 128, 3), 200, dtype=np.uint8)
    t = emb.transform(patch)
    assert tuple(t.shape) == (3, 224, 224)


def test_transform_batch_matches_stacked_transform_no_resize():
    """The batched (on-device) transform must equal stacking the per-patch CPU
    transform -- the equivalence contract the GPU fast path relies on. Checked on
    CPU here so it runs without CUDA; the only cross-device delta is fp last-ULP."""
    emb = build_embedder("resnet50")
    rng = np.random.RandomState(0)
    patches = [rng.randint(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(5)]
    stacked = torch.stack([emb.transform(p) for p in patches])
    batched = emb.transform_batch(patches, "cpu")
    assert tuple(batched.shape) == (5, 3, 224, 224)
    assert torch.allclose(stacked, batched, rtol=1e-6, atol=1e-6)


def test_transform_batch_resize_fallback_matches_stacked_transform():
    """When patch_px != input_size the batch path falls back to the PIL-bilinear
    per-patch transform, so it stays exactly equivalent to ``transform``."""
    emb = build_embedder("resnet50")
    rng = np.random.RandomState(1)
    patches = [rng.randint(0, 256, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
    stacked = torch.stack([emb.transform(p) for p in patches])
    batched = emb.transform_batch(patches, "cpu")
    assert tuple(batched.shape) == (3, 3, 224, 224)
    assert torch.equal(stacked, batched)
