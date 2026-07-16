"""H0-mini registry, loader, and card-recipe contracts without model weights."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from raw2features.embedders.fingerprint import (  # noqa: E402
    expected_patch_outputs,
)
from raw2features.embedders.model_registry import (  # noqa: E402
    build_embedder,
    get_spec,
)
from raw2features.embedders.timm_embedder import (  # noqa: E402
    TimmEmbedder,
    _resolve_callables,
)


def test_h0_mini_registry_matches_pinned_card_and_config():
    spec = get_spec("h0_mini")

    assert spec.family == "timm"
    assert spec.source == "hf-hub:bioptimus/H0-mini"
    assert spec.embedding_dim == 768
    assert spec.input_size == 224
    assert spec.recommended_mpp == 0.5
    assert spec.pooling == "cls"
    assert spec.reg_tokens == 4
    assert spec.transform_source == "pretrained_cfg"
    assert spec.mean == (0.707223, 0.578729, 0.703617)
    assert spec.std == (0.211883, 0.230117, 0.177517)
    assert spec.interpolation == "bicubic"
    assert spec.inference_amp == "fp16"
    assert spec.gated is True
    assert "CC-BY-NC-ND-4.0" in spec.license
    assert spec.weights_filename == "model.safetensors"
    assert spec.weights_revision == "5b5cc0505d19ae558270045eb0df8c34df4d9609"
    assert spec.weights_sha256 == (
        "5e4de45a6527a8160f7c21fe3105c987757efe30ce581e1f2cf5809476069ade"
    )
    # Keep the token sequence: global_pool="" makes timm return all 261 tokens,
    # then raw2features selects output[:, 0] exactly as the card recommends.
    assert spec.timm_kwargs["global_pool"] == ""
    assert spec.timm_kwargs["num_classes"] == 0
    assert isinstance(build_embedder("h0_mini"), TimmEmbedder)


def test_h0_mini_card_constructor_kwargs_resolve():
    resolved = _resolve_callables(dict(get_spec("h0_mini").timm_kwargs))

    import timm.layers

    assert resolved["mlp_layer"] is timm.layers.SwiGLUPacked
    assert resolved["act_layer"] is torch.nn.SiLU


def test_h0_mini_pooling_selects_card_recommended_cls_token():
    emb = build_embedder("h0_mini")
    # The card documents [B,261,768]: CLS + 4 register + 256 patch tokens.
    tokens = torch.randn(2, 261, 768)

    pooled = emb._pool(tokens)

    assert tuple(pooled.shape) == (2, 768)
    assert torch.equal(pooled, tokens[:, 0])


def test_h0_mini_fingerprint_binds_loader_preprocessing_and_output():
    spec = get_spec("h0_mini")
    contract = expected_patch_outputs(["h0_mini"], "auto", device="cuda")["h0_mini"]
    assert contract["embedding_dim"] == 768
    payload = contract["output_fingerprint"]["payload"]

    assert payload["loader"] == {
        "family": "timm",
        "contract_version": 1,
        "source": "hf-hub:bioptimus/H0-mini",
        "constructor": {
            "entrypoint": "timm.create_model",
            "timm_kwargs": {
                "act_layer": "torch.nn.SiLU",
                "dynamic_img_size": True,
                "global_pool": "",
                "img_size": 224,
                "init_values": 1e-5,
                "mlp_layer": "timm.layers.SwiGLUPacked",
                "mlp_ratio": 5.33334,
                "num_classes": 0,
                "reg_tokens": 4,
            },
            "checkpoint_load": None,
        },
    }
    assert payload["checkpoint"] == {
        "effective": {
            "repo": "bioptimus/H0-mini",
            "filename": "model.safetensors",
            "mechanism": "loader_managed_snapshot",
        },
        "weights_revision": spec.weights_revision,
        "weights_sha256": spec.weights_sha256,
    }
    assert payload["preprocessing"] == {
        "input_size": 224,
        "mean": [0.707223, 0.578729, 0.703617],
        "std": [0.211883, 0.230117, 0.177517],
        "interpolation": "bicubic",
        "transform_source": "pretrained_cfg",
    }
    assert payload["output"] == {
        "pooling": "cls",
        "embedding_dim": 768,
        "reg_tokens": 4,
        "modality": "brightfield",
        "resolved_amp": "fp16",
    }
