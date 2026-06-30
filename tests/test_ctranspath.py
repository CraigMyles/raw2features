"""CTransPath integration: spec + ConvStem architecture + (slow) forward."""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.embedders.model_registry import build_embedder, get_spec


def test_ctranspath_spec_is_sourced_and_consistent():
    s = get_spec("ctranspath")
    assert s.family == "timm"
    assert s.embedding_dim == 768
    assert s.input_size == 224
    assert s.pooling == "pooled"
    assert s.gated is False
    assert "GPL" in s.license  # copyleft weights, flagged
    assert s.timm_kwargs["embed_layer"] == "raw2features.embedders.convstem.ConvStem"


def test_convstem_architecture_matches_checkpoint_shape():
    torch = pytest.importorskip("torch")

    from raw2features.embedders.convstem import ConvStem

    # timm calls ConvStem with the host model's embed_dim (96 for Swin-Tiny).
    stem = ConvStem(img_size=224, patch_size=4, embed_dim=96)
    # Positional bookkeeping the checkpoint relies on.
    assert stem.grid_size == (56, 56)
    assert stem.num_patches == 3136
    # Conv channel widths the checkpoint keys (proj.0/.3/.6) require: 3->12->24->96,
    # final a 1x1 projection (embed_dim//8, embed_dim//4, embed_dim).
    assert stem.proj[0].out_channels == 12
    assert stem.proj[3].out_channels == 24
    assert stem.proj[6].out_channels == 96
    assert stem.proj[6].kernel_size == (1, 1)
    # two stride-2 blocks: 224 -> 56 grid; BHWC with 96 channels.
    out = stem(torch.randn(2, 3, 224, 224))
    assert tuple(out.shape) == (2, 56, 56, 96)


def test_convstem_loads_into_timm_swin():
    """The real contract behind 'matches checkpoint shape': timm's Swin-Tiny accepts
    ConvStem as embed_layer, passes embed_dim=96 to it, consumes the BHWC output, and
    forwards to 768-d. CPU + pretrained=False, so it catches a timm-API drift (the
    embed_layer kwargs or BHWC/BNC contract) without needing the GPU or the weights."""
    pytest.importorskip("torch")
    timm = pytest.importorskip("timm")
    import torch

    from raw2features.embedders.convstem import ConvStem

    m = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        num_classes=0,
        pretrained=False,
    ).eval()
    assert isinstance(m.patch_embed, ConvStem)
    assert m.patch_embed.proj[0].out_channels == 12  # timm passed embed_dim=96
    with torch.inference_mode():
        out = m(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 768)


@pytest.mark.slow
def test_ctranspath_forward_768_finite():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    emb = build_embedder("ctranspath").load("cuda")
    rng = np.random.default_rng(0)
    patches = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8) for _ in range(4)]
    out = emb.embed_batch(emb.transform_batch(patches, "cuda"))
    assert tuple(out.shape) == (4, 768)
    assert bool(torch.isfinite(out).all())
