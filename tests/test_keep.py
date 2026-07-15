"""KEEP's controlled image-only loader and immutable output contract."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from raw2features.embedders.fingerprint import (
    KEEP_CONSTRUCTOR_CONTRACT,
    patch_output_fingerprint,
)
from raw2features.embedders.keep_embedder import (
    KEEPEmbedder,
    _build_keep_image_model,
    _load_keep_image_state,
    _validate_keep_spec,
)
from raw2features.embedders.model_registry import build_embedder, get_spec


def test_keep_registry_records_the_primary_source_contract_without_invented_mpp():
    spec = get_spec("keep")
    assert spec.family == "keep"
    assert spec.source == "hf-hub:Astaxanthin/KEEP"
    assert spec.embedding_dim == 768
    assert spec.input_size == 224
    assert spec.pooling == "pooled"
    assert spec.mean == (0.485, 0.456, 0.406)
    assert spec.std == (0.229, 0.224, 0.225)
    assert spec.interpolation == "bicubic"
    assert spec.recommended_mpp is None
    assert spec.gated is False
    assert spec.license.startswith("MIT")
    assert spec.weights_revision == "28a25d95cc6ba27a7e6fab3f144e13dbafd8b21e"
    assert spec.weights_sha256 == (
        "82f610d5359aca67b5fd5d841009f26db430ae78d0693743589c0a727b0a146d"
    )
    assert spec.weights_filename == "model.safetensors"
    assert isinstance(build_embedder("keep"), KEEPEmbedder)


def test_keep_fingerprint_covers_checkpoint_preprocessing_and_local_constructor():
    spec = get_spec("keep")
    fingerprint = patch_output_fingerprint(spec, "fp32")
    payload = fingerprint["payload"]
    assert payload["checkpoint"] == {
        "effective": {
            "repo": "Astaxanthin/KEEP",
            "filename": "model.safetensors",
            "mechanism": "pinned_safetensors_local_image_wrapper",
        },
        "weights_revision": spec.weights_revision,
        "weights_sha256": spec.weights_sha256,
    }
    assert payload["loader"]["constructor"] == {
        "timm_kwargs": {},
        "checkpoint_load": None,
        **KEEP_CONSTRUCTOR_CONTRACT,
    }
    assert payload["loader"]["constructor"]["trust_remote_code"] is False
    assert payload["preprocessing"] == {
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "interpolation": "bicubic",
        "transform_source": "registry",
    }
    assert payload["output"]["embedding_dim"] == 768
    assert payload["output"]["resolved_amp"] == "fp32"


def test_keep_constructor_uses_compatible_layer_scale_and_normalises_output():
    torch = pytest.importorskip("torch")
    nn = torch.nn
    calls = []

    class ModernLayerScale(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gamma = nn.Parameter(torch.full((1024,), 1e-5))
            self.inplace = False

        def forward(self, x):
            return x * self.gamma

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ls1 = ModernLayerScale()
            self.ls2 = ModernLayerScale()

    class TinyVisual(nn.Module):
        num_features = 1024

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block() for _ in range(24)])
            self.input_projection = nn.Linear(3, 1024)

        def forward(self, x):
            return self.input_projection(x)

    def create_model(architecture, **kwargs):
        calls.append((architecture, kwargs))
        return TinyVisual()

    model = _build_keep_image_model(SimpleNamespace(create_model=create_model), torch)
    assert calls == [
        (
            "vit_large_patch16_224",
            {
                "pretrained": False,
                "img_size": 224,
                "patch_size": 16,
                "init_values": 1e-5,
                "num_classes": 0,
            },
        )
    ]
    state_keys = set(model.state_dict())
    assert "visual.blocks.0.ls1.weight" in state_keys
    assert "visual.blocks.23.ls2.weight" in state_keys
    assert not any(key.endswith(".gamma") for key in state_keys)
    output = model(torch.randn(2, 3))
    assert tuple(output.shape) == (2, 768)
    assert torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-6)


def test_keep_safetensors_reader_loads_only_the_exact_image_state():
    requested = []
    loaded = []

    class Model:
        def state_dict(self):
            return {
                "visual.weight": object(),
                "visual_head.0.weight": object(),
            }

        def load_state_dict(self, state, *, strict):
            loaded.append((state, strict))

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return (
                "text.embeddings.weight",
                "logit_scale",
                "visual.weight",
                "visual_head.0.weight",
            )

        def get_tensor(self, key):
            requested.append(key)
            return f"tensor:{key}"

    calls = []

    def safe_open(path, **kwargs):
        calls.append((path, kwargs))
        return Handle()

    _load_keep_image_state("checkpoint.safetensors", Model(), safe_open_fn=safe_open)
    assert calls == [("checkpoint.safetensors", {"framework": "pt", "device": "cpu"})]
    assert requested == ["visual.weight", "visual_head.0.weight"]
    assert loaded == [
        (
            {
                "visual.weight": "tensor:visual.weight",
                "visual_head.0.weight": "tensor:visual_head.0.weight",
            },
            True,
        )
    ]


def test_keep_safetensors_reader_rejects_missing_or_extra_image_keys():
    class Model:
        def state_dict(self):
            return {"visual.expected": object()}

        def load_state_dict(self, *_args, **_kwargs):
            pytest.fail("mismatched image state must not be loaded")

    class Handle:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return ("visual.unexpected", "text.weight")

        def get_tensor(self, _key):
            pytest.fail("mismatched image tensors must not be read")

    with pytest.raises(ValueError, match="missing 1 image keys"):
        _load_keep_image_state(
            "bad.safetensors", Model(), safe_open_fn=lambda *_a, **_k: Handle()
        )


def _install_fake_loader_modules(monkeypatch, checkpoint, events):
    torch = ModuleType("torch")
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "timm", ModuleType("timm"))

    safetensors = ModuleType("safetensors")
    safetensors.safe_open = object()
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)

    hub = ModuleType("huggingface_hub")

    def hf_hub_download(**kwargs):
        events.append(("download", kwargs))
        return str(checkpoint)

    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return safetensors.safe_open


def test_keep_loader_verifies_before_construction_and_safe_loading(
    monkeypatch, tmp_path
):
    import raw2features.embedders.keep_embedder as module

    events = []
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"pinned KEEP safetensors")
    safe_open = _install_fake_loader_modules(monkeypatch, checkpoint, events)

    class Model:
        def eval(self):
            events.append("eval")
            return self

        def to(self, device):
            events.append(("to", device))
            return self

    original_verify = module.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    model = Model()

    def build(_timm, _torch):
        events.append("build")
        return model

    def load_state(path, model, *, safe_open_fn):
        events.append(("safe_load", path, model, safe_open_fn))

    monkeypatch.setattr(module, "verify_sha256", verify)
    monkeypatch.setattr(module, "_build_keep_image_model", build)
    monkeypatch.setattr(module, "_load_keep_image_state", load_state)
    spec = replace(
        get_spec("keep"),
        weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    sentinel_dtype = object()
    embedder = KEEPEmbedder(spec).load(device="cpu", dtype=sentinel_dtype)

    assert events[0] == (
        "download",
        {
            "repo_id": "Astaxanthin/KEEP",
            "filename": "model.safetensors",
            "revision": spec.weights_revision,
            "cache_dir": None,
        },
    )
    assert events[1:3] == ["verify", "build"]
    assert events[3] == ("safe_load", str(checkpoint), model, safe_open)
    assert events[4:] == ["eval", ("to", "cpu")]
    assert embedder._dtype is sentinel_dtype


def test_keep_bad_digest_never_constructs_or_opens_checkpoint(monkeypatch, tmp_path):
    import raw2features.embedders.keep_embedder as module

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"wrong bytes")
    _install_fake_loader_modules(monkeypatch, checkpoint, [])
    monkeypatch.setattr(
        module,
        "_build_keep_image_model",
        lambda *_args: pytest.fail("model construction must follow verification"),
    )
    monkeypatch.setattr(
        module,
        "_load_keep_image_state",
        lambda *_args, **_kwargs: pytest.fail("safe_open must follow verification"),
    )

    with pytest.raises(ValueError, match="sha256"):
        KEEPEmbedder(get_spec("keep")).load(device="cpu", dtype=object())


def test_keep_fixed_wrapper_rejects_unhonoured_registry_constructor_changes():
    spec = get_spec("keep")
    _validate_keep_spec(spec)
    with pytest.raises(ValueError, match="embedding_dim"):
        _validate_keep_spec(replace(spec, embedding_dim=1024))
    with pytest.raises(ValueError, match="timm_kwargs"):
        _validate_keep_spec(replace(spec, timm_kwargs={"dynamic_img_size": True}))
