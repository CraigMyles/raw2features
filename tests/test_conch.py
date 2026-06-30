"""CONCH integration: registry contract + (slow, gated) forward pass.

CONCH is an OPTIONAL family (the ``conch`` package). The spec/contract tests run
everywhere; the forward test skips unless the package, a CUDA GPU, and the gated
weights are all present.
"""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.embedders.model_registry import build_embedder, get_spec


def test_conch_spec_is_sourced_and_consistent():
    spec = get_spec("conch")
    assert spec.family == "conch"
    assert spec.embedding_dim == 512
    assert spec.input_size == 448  # CONCH operates at 448 px, not 224
    assert spec.pooling == "pooled"
    assert "CC-BY-NC-ND" in spec.license
    assert spec.gated is True
    # OpenAI-CLIP normalisation, sourced from CONCH's own preprocess transform.
    assert spec.mean[0] == pytest.approx(0.48145466)
    assert spec.std[0] == pytest.approx(0.26862954)
    # card documents fp16 AMP for TRAINING only -> we default to faithful fp32.
    assert spec.inference_amp == "fp32"


def test_conch_family_class_resolves():
    # The family class is importable even though `conch` is only imported inside
    # .load() - this is what keeps the dependency optional without breaking
    # discovery. (If the package is truly absent, the entry-point loader skips it.)
    from raw2features.core import plugins

    cls = plugins.get("embedders", "conch")
    assert cls.__name__ == "ConchEmbedder"


@pytest.mark.slow
def test_conch_forward_512d_finite():
    import torch

    pytest.importorskip("conch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    emb = build_embedder("conch").load("cuda")
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (448, 448, 3), dtype=np.uint8) for _ in range(4)]
    batch = emb.transform_batch(patches, "cuda")
    assert tuple(batch.shape) == (4, 3, 448, 448)
    with torch.no_grad():
        out = emb.embed_batch(batch)
    assert tuple(out.shape) == (4, 512)
    assert bool(torch.isfinite(out).all())
