"""Midnight (transformers family) integration: spec + (slow, ungated) forward."""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.embedders.model_registry import build_embedder, get_spec


def test_midnight_spec_is_sourced_and_consistent():
    s = get_spec("midnight")
    assert s.family == "transformers"
    assert s.embedding_dim == 3072  # concat(CLS[1536], mean-patch[1536])
    assert s.pooling == "cls_concat_meanpatch"
    assert s.reg_tokens == 0
    assert s.input_size == 224
    assert s.gated is False
    assert "MIT" in s.license
    assert tuple(s.mean) == (0.5, 0.5, 0.5)


def test_transformers_family_class_resolves():
    from raw2features.core import plugins

    assert plugins.get("embedders", "transformers").__name__ == "TransformersEmbedder"


@pytest.mark.slow
def test_midnight_forward_3072_finite():
    import torch

    pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    emb = build_embedder("midnight").load("cuda")
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    out = emb.embed_batch(emb.transform_batch(patches, "cuda"))
    assert tuple(out.shape) == (4, 3072)
    assert bool(torch.isfinite(out).all())
