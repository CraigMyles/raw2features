"""Virchow2 - registry contract + the pooling math, all weight-free.

The real gated forward (download of ``hf-hub:paige-ai/Virchow2``) is exercised by
``test_embedder_forward.py`` under the ``slow`` mark and skips cleanly without a
token. What we *can* gate here, with no weights, is the part that is bespoke to
Virchow2 and therefore the actual extensibility risk:

* the registry builds a complete, consistent ModelSpec for ``virchow2``;
* the new ``cls_concat_meanpatch`` pooling reduces a synthetic ``[B,261,1280]``
  token tensor to exactly ``cat([cls, patch_tokens.mean(1)])`` of width 2560,
  skipping the 4 register tokens - i.e. it equals the verbatim recipe from the
  card. This is the correctness gate the forward then confirms end-to-end.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from raw2features.embedders.model_registry import (  # noqa: E402
    build_embedder,
    get_spec,
)
from raw2features.embedders.timm_embedder import TimmEmbedder  # noqa: E402


def test_virchow2_registry_entry_is_complete_and_sourced():
    spec = get_spec("virchow2")
    assert spec.family == "timm"
    assert spec.source == "hf-hub:paige-ai/Virchow2"
    assert spec.embedding_dim == 2560
    assert spec.input_size == 224
    assert spec.gated is True
    assert spec.inference_amp == "fp16"  # card highly recommends fp16 autocast
    assert spec.pooling == "cls_concat_meanpatch"
    assert spec.reg_tokens == 4  # tokens 1-4 are register tokens
    assert spec.transform_source == "pretrained_cfg"
    assert "CC-BY-NC-ND" in spec.license  # non-commercial
    assert spec.transform_source_url == "https://huggingface.co/paige-ai/Virchow2"
    # callable kwargs are encoded as resolvable dotted strings (like uni2_h)
    assert spec.timm_kwargs["mlp_layer"] == "timm.layers.SwiGLUPacked"
    assert spec.timm_kwargs["act_layer"] == "torch.nn.SiLU"
    # crucially NOT num_classes=0: we need the token sequence, not a pooled vector
    assert "num_classes" not in spec.timm_kwargs
    # family resolves to the generic timm driver
    assert isinstance(build_embedder("virchow2"), TimmEmbedder)


def test_virchow2_callable_kwargs_resolve_to_real_classes():
    # The dotted strings in the registry must import to the real timm/torch classes
    # the card passes; this is what create_model would receive at load time.
    from raw2features.embedders.timm_embedder import _resolve_callables

    resolved = _resolve_callables(dict(get_spec("virchow2").timm_kwargs))
    import timm.layers

    assert resolved["mlp_layer"] is timm.layers.SwiGLUPacked
    assert resolved["act_layer"] is torch.nn.SiLU


def _make_embedder():
    """A virchow2 embedder whose spec is wired from the registry (no weights)."""
    emb = build_embedder("virchow2")
    emb._device = "cpu"
    emb._dtype = torch.float32
    return emb


def test_pooling_math_matches_card_recipe_exactly():
    """The hard correctness gate: synthetic [B,261,1280] -> [B,2560] == the card.

    Verbatim from the Virchow2 card:
        class_token  = output[:, 0]      # [B,1280]
        patch_tokens = output[:, 5:]     # [B,256,1280]  (skip 4 register tokens)
        embedding    = cat([class_token, patch_tokens.mean(1)], dim=-1)  # [B,2560]
    """
    emb = _make_embedder()
    torch.manual_seed(0)
    B, T, D = 3, 261, 1280
    out = torch.randn(B, T, D)

    pooled = emb._pool(out)

    assert tuple(pooled.shape) == (B, 2 * D) == (3, 2560)
    # the exact reference computed straight from the card snippet
    class_token = out[:, 0]
    patch_tokens = out[:, 5:]
    expected = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    assert torch.equal(pooled, expected)
    # and it is genuinely the concat of two halves: CLS then mean-patch
    assert torch.equal(pooled[:, :D], class_token)
    assert torch.equal(pooled[:, D:], patch_tokens.mean(1))


def test_pooling_respects_reg_tokens_field():
    """reg_tokens drives where patch tokens start (1 + reg_tokens), not a constant.

    With reg_tokens patched to 0 the patch block must start at index 1, proving the
    '5:' in the card is reg_tokens(4)+CLS(1) and not hard-coded.
    """
    emb = _make_embedder()
    object.__setattr__(emb.spec, "reg_tokens", 0)
    out = torch.randn(2, 10, 4)
    pooled = emb._pool(out)
    expected = torch.cat([out[:, 0], out[:, 1:].mean(1)], dim=-1)
    assert torch.equal(pooled, expected)


def test_pooling_rejects_non_token_output():
    """cls_concat_meanpatch needs the token sequence; a 2-D output is a misconfig
    (e.g. someone added num_classes=0 / global pool) and must fail loudly."""
    emb = _make_embedder()
    with pytest.raises(ValueError, match="token tensor"):
        emb._pool(torch.randn(2, 1280))


def test_other_poolings_unchanged():
    """The generic CLS / pass-through behaviour for the existing models is intact:
    a 2-D output passes straight through, a stray token tensor falls back to CLS."""
    emb = build_embedder("uni")  # pooling == "cls"
    emb._device, emb._dtype = "cpu", torch.float32
    two_d = torch.randn(4, 1024)
    assert torch.equal(emb._pool(two_d), two_d)  # 2-D pass-through
    tokens = torch.randn(4, 7, 1024)
    assert torch.equal(emb._pool(tokens), tokens[:, 0])  # safety -> CLS
