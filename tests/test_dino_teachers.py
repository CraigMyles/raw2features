"""Offline contracts for the OpenMidnight/OpenPath DINOv2 teacher loader."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace

import pytest

from raw2features.embedders.dino_teacher_embedder import (
    DinoTeacherEmbedder,
    _convert_dinov2_state,
    _extract_teacher_state,
    _flatten_chunked_blocks,
    _validate_model_contract,
)
from raw2features.embedders.fingerprint import (
    DINO_TEACHER_CONSTRUCTOR_CONTRACT,
    patch_output_fingerprint,
)
from raw2features.embedders.model_registry import get_spec


@dataclass(frozen=True)
class _Tensor:
    """Tiny tensor-expression recorder; keeps these tests torch-free."""

    expression: str

    def __getitem__(self, key):
        return _Tensor(f"{self.expression}[{key!r}]")

    def __add__(self, other):
        return _Tensor(f"({self.expression}+{other.expression})")


def _typed_stub(module: str, name: str, **attributes):
    instance = type(name, (), {"__module__": module})()
    for key, value in attributes.items():
        setattr(instance, key, value)
    return instance


def _fake_ffn():
    return _typed_stub(
        "timm.layers.mlp",
        "GluMlp",
        act=_typed_stub("torch.nn.modules.activation", "SiLU"),
        gate_last=False,
        chunk_dim=-1,
        fc1=_typed_stub(
            "torch.nn.modules.linear",
            "Linear",
            in_features=1536,
            out_features=8192,
            bias=object(),
        ),
        norm=_typed_stub("torch.nn.modules.linear", "Identity"),
        fc2=_typed_stub(
            "torch.nn.modules.linear",
            "Linear",
            in_features=4096,
            out_features=1536,
            bias=object(),
        ),
        drop1=_typed_stub("torch.nn.modules.dropout", "Dropout", p=0.0),
        drop2=_typed_stub("torch.nn.modules.dropout", "Dropout", p=0.0),
    )


class _FakeBlock:
    def __init__(self):
        self.attn = SimpleNamespace(num_heads=24)
        self.mlp = _fake_ffn()


class _FakeModel:
    num_features = 1536
    num_reg_tokens = 4
    no_embed_class = True
    global_pool = "token"
    patch_embed = SimpleNamespace(patch_size=(14, 14))

    def __init__(self, events, loaded):
        self.blocks = [_FakeBlock() for _ in range(40)]
        self._events = events
        self._loaded = loaded

    def load_state_dict(self, state, strict):
        self._events.append("strict_load")
        self._loaded.append((state, strict))

    def eval(self):
        self._events.append("eval")
        return self

    def to(self, device):
        self._events.append(f"to:{device}")
        return self


def _payload(name: str):
    state = {
        "register_tokens": _Tensor("register_tokens"),
        "cls_token": _Tensor("cls_token"),
        "pos_embed": _Tensor("pos_embed"),
        "mask_token": _Tensor("mask_token"),
    }
    if name == "openpath":
        nested = {f"backbone.{key}": value for key, value in state.items()}
        nested["backbone.blocks.2.20.mlp.w12.weight"] = _Tensor("w12")
        nested["dino_head.last_layer.weight"] = _Tensor("not_backbone")
        return {"teacher": nested, "iteration": 316250}
    state["blocks.0.mlp.w12.weight"] = _Tensor("w12")
    return state


def _install_fake_runtime(monkeypatch, payload, events, loaded, *, fail_safe=False):
    torch = ModuleType("torch")
    torch.float32 = object()

    def torch_load(_path, *, map_location, weights_only):
        events.append(f"torch_load:{weights_only}")
        assert map_location == "cpu"
        if fail_safe and weights_only:
            raise RuntimeError("checkpoint contains author metadata")
        return payload

    torch.load = torch_load
    monkeypatch.setitem(sys.modules, "torch", torch)

    timm = ModuleType("timm")

    def create_model(architecture, **kwargs):
        events.append("create_model")
        assert architecture == "vit_giant_patch14_reg4_dinov2"
        assert kwargs == {
            "pretrained": False,
            "img_size": 224,
            "num_classes": 0,
            "global_pool": "token",
        }
        return _FakeModel(events, loaded)

    timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", timm)


def test_openmidnight_and_openpath_specs_are_exact_and_sourced():
    openmidnight = get_spec("openmidnight")
    assert openmidnight.family == "dino_teacher"
    assert openmidnight.source == "hf-hub:SophontAI/OpenMidnight"
    assert openmidnight.embedding_dim == 1536
    assert openmidnight.pooling == "cls"
    assert openmidnight.reg_tokens == 4
    assert openmidnight.input_size == 224
    assert openmidnight.recommended_mpp is None
    assert openmidnight.gated is True
    assert openmidnight.weights_revision == ("87189e6674d397a14a5cd342c97b1a1615a185aa")
    assert openmidnight.weights_sha256 == (
        "b57121fc10e2b04c9e85514fa5e4c23b75329301e05bfc756c3bca57312505be"
    )
    assert openmidnight.weights_filename == "teacher_checkpoint_load.pt"

    openpath = get_spec("openpath")
    assert openpath.family == "dino_teacher"
    assert openpath.source == "hf-hub:taejoon89/openpath"
    assert openpath.embedding_dim == 1536
    assert openpath.pooling == "cls"
    assert openpath.reg_tokens == 4
    assert openpath.input_size == 224
    assert openpath.recommended_mpp == 0.5
    assert openpath.gated is False
    assert openpath.weights_revision == ("22b339237195931eb56ea6c86653362edd2dd155")
    assert openpath.weights_sha256 == (
        "1b389360b4867b5159ce7534ff231f51e87d3cf140daa4b37483d0a178af3da3"
    )
    assert openpath.weights_filename == "teacher_checkpoint.pth"
    assert openpath.checkpoint == {
        "repo": "taejoon89/openpath",
        "filename": "teacher_checkpoint.pth",
        "state_dict_key": "teacher",
        "state_dict_prefix": "backbone.",
        "flatten_block_chunks": True,
    }

    for spec in (openmidnight, openpath):
        assert spec.mean == (0.485, 0.456, 0.406)
        assert spec.std == (0.229, 0.224, 0.225)
        assert spec.inference_amp == "fp32"
        assert spec.experimental is False
        assert spec.doi is None


def test_openpath_checkpoint_extraction_and_conversion_is_deterministic():
    spec = get_spec("openpath")
    state = _extract_teacher_state(_payload("openpath"), spec.checkpoint or {})
    assert "blocks.20.mlp.w12.weight" in state
    assert all(not key.startswith("backbone.") for key in state)
    assert "dino_head.last_layer.weight" not in state

    converted = _convert_dinov2_state(state)
    assert "reg_token" in converted
    assert "register_tokens" not in converted
    assert "mask_token" not in converted
    assert "blocks.20.mlp.fc1.weight" in converted
    assert "blocks.20.mlp.w12.weight" not in converted
    assert converted["cls_token"].expression.startswith("(cls_token+")
    assert converted["pos_embed"].expression.startswith("pos_embed[")


def test_checkpoint_conversion_fails_closed_on_malformed_or_ambiguous_state():
    with pytest.raises(ValueError, match="missing required keys"):
        _convert_dinov2_state({"cls_token": _Tensor("cls")})

    with pytest.raises(ValueError, match="duplicate DINOv2 state key"):
        _flatten_chunked_blocks(
            {
                "blocks.0.0.norm.weight": _Tensor("first"),
                "blocks.1.0.norm.weight": _Tensor("duplicate"),
            }
        )

    with pytest.raises(ValueError, match="state_dict_key"):
        _extract_teacher_state({}, {"state_dict_key": "teacher"})


@pytest.mark.parametrize(
    ("name", "expected_source", "expected_block"),
    [
        ("openmidnight", "hf-hub:SophontAI/OpenMidnight", "blocks.0.mlp.fc1.weight"),
        ("openpath", "hf-hub:taejoon89/openpath", "blocks.20.mlp.fc1.weight"),
    ],
)
def test_loader_pins_verifies_converts_and_loads_strictly(
    monkeypatch, tmp_path, name, expected_source, expected_block
):
    import raw2features.embedders.dino_teacher_embedder as module

    events = []
    loaded = []
    calls = []
    weight_bytes = f"pinned {name} weights".encode()
    path = tmp_path / get_spec(name).weights_filename
    path.write_bytes(weight_bytes)
    digest = hashlib.sha256(weight_bytes).hexdigest()
    spec = replace(get_spec(name), weights_sha256=digest)

    def download(source, filename, revision):
        events.append("download")
        calls.append((source, filename, revision))
        return str(path)

    original_verify = module.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(module, "download_pinned_hf_file", download)
    monkeypatch.setattr(module, "verify_sha256", verify)
    _install_fake_runtime(monkeypatch, _payload(name), events, loaded)

    embedder = DinoTeacherEmbedder(spec).load(device="cpu", dtype=object())
    assert embedder._model is not None
    assert calls == [
        (expected_source, spec.weights_filename, spec.weights_revision),
    ]
    assert events == [
        "download",
        "verify",
        "torch_load:True",
        "create_model",
        "strict_load",
        "eval",
        "to:cpu",
    ]
    assert len(loaded) == 1
    state, strict = loaded[0]
    assert strict is True
    assert expected_block in state
    assert "mask_token" not in state


def test_bad_checksum_refuses_before_deserialisation_or_model_construction(
    monkeypatch, tmp_path
):
    import raw2features.embedders.dino_teacher_embedder as module

    spec = get_spec("openpath")
    path = tmp_path / spec.weights_filename
    path.write_bytes(b"not the pinned checkpoint")
    events = []
    loaded = []
    monkeypatch.setattr(
        module, "download_pinned_hf_file", lambda *_args, **_kwargs: str(path)
    )
    _install_fake_runtime(monkeypatch, _payload("openpath"), events, loaded)

    with pytest.raises(ValueError, match="does not match the pinned"):
        DinoTeacherEmbedder(spec).load(device="cpu", dtype=object())
    assert events == []
    assert loaded == []


def test_weights_only_fallback_still_happens_after_one_verification(
    monkeypatch, tmp_path
):
    import raw2features.embedders.dino_teacher_embedder as module

    events = []
    loaded = []
    weight_bytes = b"pinned OpenPath checkpoint with metadata"
    path = tmp_path / "teacher_checkpoint.pth"
    path.write_bytes(weight_bytes)
    spec = replace(
        get_spec("openpath"),
        weights_sha256=hashlib.sha256(weight_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        module, "download_pinned_hf_file", lambda *_args, **_kwargs: str(path)
    )
    original_verify = module.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(module, "verify_sha256", verify)
    _install_fake_runtime(
        monkeypatch, _payload("openpath"), events, loaded, fail_safe=True
    )

    DinoTeacherEmbedder(spec).load(device="cpu", dtype=object())
    assert events[:4] == [
        "verify",
        "torch_load:True",
        "torch_load:False",
        "create_model",
    ]
    assert events.count("verify") == 1


def test_model_architecture_drift_fails_before_state_load(monkeypatch, tmp_path):
    import raw2features.embedders.dino_teacher_embedder as module

    events = []
    loaded = []
    weight_bytes = b"pinned OpenMidnight weights"
    path = tmp_path / "teacher_checkpoint_load.pt"
    path.write_bytes(weight_bytes)
    spec = replace(
        get_spec("openmidnight"),
        weights_sha256=hashlib.sha256(weight_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        module, "download_pinned_hf_file", lambda *_args, **_kwargs: str(path)
    )
    _install_fake_runtime(monkeypatch, _payload("openmidnight"), events, loaded)
    timm = sys.modules["timm"]

    def drifted_model(*_args, **_kwargs):
        model = _FakeModel(events, loaded)
        model.num_reg_tokens = 0
        return model

    timm.create_model = drifted_model
    with pytest.raises(RuntimeError, match="does not match the loader contract"):
        DinoTeacherEmbedder(spec).load(device="cpu", dtype=object())
    assert loaded == []


@pytest.mark.parametrize(
    ("component", "attribute", "drifted_value"),
    [
        (None, "gate_last", True),
        (None, "chunk_dim", 1),
        ("fc1", "out_features", 4096),
        ("fc2", "in_features", 8192),
        ("drop1", "p", 0.1),
    ],
)
def test_model_ffn_semantic_drift_fails_closed(component, attribute, drifted_value):
    model = _FakeModel([], [])
    ffn = model.blocks[23].mlp
    target = getattr(ffn, component) if component else ffn
    setattr(target, attribute, drifted_value)

    with pytest.raises(RuntimeError, match="block 23 FFN"):
        _validate_model_contract(model, get_spec("openpath"))


def test_model_ffn_implementation_and_activation_drift_fail_closed():
    model = _FakeModel([], [])
    model.blocks[0].mlp = _typed_stub(
        "timm.layers.mlp",
        "SwiGLU",
        **vars(_fake_ffn()),
    )
    with pytest.raises(RuntimeError, match="block 0 FFN"):
        _validate_model_contract(model, get_spec("openpath"))

    model = _FakeModel([], [])
    model.blocks[0].mlp.act = _typed_stub("torch.nn.modules.activation", "GELU")
    with pytest.raises(RuntimeError, match="block 0 FFN"):
        _validate_model_contract(model, get_spec("openpath"))


def test_fingerprint_covers_local_architecture_and_checkpoint_layout():
    for name in ("openmidnight", "openpath"):
        spec = get_spec(name)
        payload = patch_output_fingerprint(spec, "fp32")["payload"]
        constructor = payload["loader"]["constructor"]
        for key, value in DINO_TEACHER_CONSTRUCTOR_CONTRACT.items():
            assert constructor[key] == value
        assert constructor["checkpoint_load"] == spec.checkpoint
        assert constructor["remote_code"] is False
        assert payload["checkpoint"]["weights_revision"] == spec.weights_revision
        assert payload["checkpoint"]["weights_sha256"] == spec.weights_sha256
        assert payload["checkpoint"]["effective"]["repo"] == spec.checkpoint["repo"]
        assert payload["checkpoint"]["effective"]["filename"] == spec.weights_filename


def test_dino_teacher_family_registers_without_optional_model_imports():
    from raw2features.core import plugins

    assert plugins.get("embedders", "dino_teacher") is DinoTeacherEmbedder
