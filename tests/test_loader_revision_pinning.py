"""Loader-level tests that enforce every registry revision at download time."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from raw2features.embedders._hub import pinned_model_cache_dir
from raw2features.embedders.model_registry import get_spec


class _Model:
    def eval(self):
        return self

    def to(self, _device):
        return self


def _fake_torch(monkeypatch) -> None:
    torch = ModuleType("torch")
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)


def _fake_hub(monkeypatch, *, file_download=None, snapshot_download=None) -> None:
    hub = ModuleType("huggingface_hub")
    if file_download is not None:
        hub.hf_hub_download = file_download
    if snapshot_download is not None:
        hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_clip_hf_passes_registry_revision_to_transformers(monkeypatch):
    from raw2features.embedders.clip_hf_embedder import ClipHFEmbedder

    _fake_torch(monkeypatch)
    calls = []
    transformers = ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return _Model()

    transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    spec = get_spec("plip")
    ClipHFEmbedder(spec).load(device="cpu", dtype=object())
    assert calls == [
        (
            spec.source,
            {"trust_remote_code": True, "revision": spec.weights_revision},
        )
    ]


def test_open_clip_downloads_exact_files_to_app_cache_and_verifies_before_load(
    monkeypatch, tmp_path
):
    import raw2features.embedders.open_clip_embedder as open_clip_embedder

    _fake_torch(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    downloads = []
    loads = []
    events = []
    weight_bytes = b"pinned QuiltNet weights"

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "open_clip_config.json").write_text("{}")
        (local_dir / "open_clip_pytorch_model.bin").write_bytes(weight_bytes)
        return str(local_dir)

    _fake_hub(monkeypatch, snapshot_download=snapshot_download)
    open_clip = ModuleType("open_clip")

    def create_model_from_pretrained(source):
        events.append("load")
        loads.append(source)
        return _Model(), SimpleNamespace(transforms=[])

    open_clip.create_model_from_pretrained = create_model_from_pretrained
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)

    original_verify = open_clip_embedder.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(open_clip_embedder, "verify_sha256", verify)
    spec = replace(get_spec("quiltnet"), weights_sha256=_digest(weight_bytes))
    local_dir = pinned_model_cache_dir(spec.source, spec.weights_revision)
    open_clip_embedder.OpenClipEmbedder(spec).load(device="cpu", dtype=object())
    assert downloads == [
        {
            "repo_id": "wisdomik/QuiltNet-B-32",
            "revision": spec.weights_revision,
            "allow_patterns": (
                "open_clip_config.json",
                "open_clip_pytorch_model.bin",
            ),
            "local_dir": local_dir,
        }
    ]
    assert events == ["verify", "load"]
    assert loads == [f"local-dir:{local_dir}"]


def test_biomedclip_pins_nested_text_config_before_open_clip_boundary(
    monkeypatch, tmp_path
):
    import raw2features.embedders.open_clip_embedder as open_clip_embedder

    _fake_torch(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    downloads = []
    loads = []
    weight_bytes = b"pinned BiomedCLIP weights"
    remote_text_id = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        if kwargs["repo_id"].startswith("microsoft/BiomedCLIP-"):
            config = {
                "model_cfg": {
                    "text_cfg": {
                        "hf_model_name": remote_text_id,
                        "hf_tokenizer_name": remote_text_id,
                    }
                }
            }
            (local_dir / "open_clip_config.json").write_text(json.dumps(config))
            (local_dir / "open_clip_pytorch_model.bin").write_bytes(weight_bytes)
        else:
            (local_dir / "config.json").write_text("{}")
            (local_dir / "tokenizer_config.json").write_text("{}")
            (local_dir / "vocab.txt").write_text("token\n")
        return str(local_dir)

    _fake_hub(monkeypatch, snapshot_download=snapshot_download)
    open_clip = ModuleType("open_clip")

    expected_nested = pinned_model_cache_dir(
        open_clip_embedder._BIOMEDCLIP_TEXT_SOURCE,
        open_clip_embedder._BIOMEDCLIP_TEXT_REVISION,
    )

    def create_model_from_pretrained(source):
        loads.append(source)
        local_dir = Path(source.removeprefix("local-dir:"))
        config = json.loads((local_dir / "open_clip_config.json").read_text())
        text_cfg = config["model_cfg"]["text_cfg"]
        assert text_cfg["hf_model_name"] == expected_nested
        assert text_cfg["hf_tokenizer_name"] == expected_nested
        assert Path(text_cfg["hf_model_name"]).is_absolute()
        assert remote_text_id not in json.dumps(config)
        return _Model(), SimpleNamespace(transforms=[])

    open_clip.create_model_from_pretrained = create_model_from_pretrained
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)

    spec = replace(get_spec("biomedclip"), weights_sha256=_digest(weight_bytes))
    local_dir = pinned_model_cache_dir(spec.source, spec.weights_revision)
    open_clip_embedder.OpenClipEmbedder(spec).load(device="cpu", dtype=object())

    assert downloads == [
        {
            "repo_id": ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
            "revision": spec.weights_revision,
            "allow_patterns": (
                "open_clip_config.json",
                "open_clip_pytorch_model.bin",
            ),
            "local_dir": local_dir,
        },
        {
            "repo_id": remote_text_id,
            "revision": "d673b8835373c6fa116d6d8006b33d48734e305d",
            "allow_patterns": (
                "config.json",
                "tokenizer_config.json",
                "vocab.txt",
            ),
            "local_dir": expected_nested,
        },
    ]
    assert loads == [f"local-dir:{local_dir}"]


def test_biomedclip_construction_dependency_pin_is_auditable():
    import raw2features.embedders.open_clip_embedder as open_clip_embedder

    notes = get_spec("biomedclip").notes
    nested_repo = open_clip_embedder._BIOMEDCLIP_TEXT_SOURCE.removeprefix("hf-hub:")
    assert nested_repo in notes
    assert open_clip_embedder._BIOMEDCLIP_TEXT_REVISION in notes
    assert "construction-only" in notes
    assert "output fingerprint includes it" in notes
    assert "separate from grid identity" in notes


def test_conch_verifies_pinned_checkpoint_then_uses_local_path(monkeypatch, tmp_path):
    import raw2features.embedders.conch_embedder as conch_embedder

    _fake_torch(monkeypatch)
    downloads = []
    loads = []
    events = []
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"pinned CONCH weights")

    def file_download(**kwargs):
        downloads.append(kwargs)
        return str(checkpoint)

    _fake_hub(monkeypatch, file_download=file_download)
    conch = ModuleType("conch")
    conch.__path__ = []
    custom = ModuleType("conch.open_clip_custom")

    def create_model_from_pretrained(arch, checkpoint):
        events.append("load")
        loads.append((arch, checkpoint))
        return _Model(), SimpleNamespace(transforms=[])

    custom.create_model_from_pretrained = create_model_from_pretrained
    conch.open_clip_custom = custom
    monkeypatch.setitem(sys.modules, "conch", conch)
    monkeypatch.setitem(sys.modules, "conch.open_clip_custom", custom)

    original_verify = conch_embedder.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(conch_embedder, "verify_sha256", verify)
    spec = replace(get_spec("conch"), weights_sha256=_digest(checkpoint.read_bytes()))
    conch_embedder.ConchEmbedder(spec).load(device="cpu", dtype=object())
    assert downloads == [
        {
            "repo_id": "MahmoodLab/conch",
            "filename": "pytorch_model.bin",
            "revision": spec.weights_revision,
            "cache_dir": None,
        }
    ]
    assert events == ["verify", "load"]
    assert loads == [("conch_ViT-B-16", str(checkpoint))]


def test_kronos_downloads_model_and_metadata_from_same_revision(monkeypatch, tmp_path):
    import raw2features.embedders.kronos_embedder as kronos_embedder

    _fake_torch(monkeypatch)
    monkeypatch.setattr(
        kronos_embedder, "_patch_kronos_attention_to_sdpa", lambda: None
    )
    metadata = tmp_path / "marker_metadata.csv"
    metadata.write_text("marker_name,marker_id,marker_mean,marker_std\nCD3,1,0.1,0.2\n")
    checkpoint = tmp_path / "kronos_vits16_model.pt"
    checkpoint.write_bytes(b"pinned KRONOS weights")
    downloads = []
    loads = []
    events = []

    def file_download(**kwargs):
        downloads.append(kwargs)
        if kwargs["filename"] == "marker_metadata.csv":
            return str(metadata)
        return str(checkpoint)

    _fake_hub(monkeypatch, file_download=file_download)
    kronos = ModuleType("kronos")

    def create_model_from_pretrained(**kwargs):
        events.append("load")
        loads.append(kwargs)
        return _Model(), "fp32", 384

    kronos.create_model_from_pretrained = create_model_from_pretrained
    monkeypatch.setitem(sys.modules, "kronos", kronos)

    original_verify = kronos_embedder.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(kronos_embedder, "verify_sha256", verify)
    spec = replace(get_spec("kronos"), weights_sha256=_digest(checkpoint.read_bytes()))
    kronos_embedder.KronosEmbedder(spec).load(device="cpu")
    assert [call["revision"] for call in downloads] == [
        spec.weights_revision,
        spec.weights_revision,
    ]
    assert [call["filename"] for call in downloads] == [
        "kronos_vits16_model.pt",
        "marker_metadata.csv",
    ]
    assert loads == [
        {
            "checkpoint_path": str(checkpoint),
            "cache_dir": None,
            "cfg": {"model_type": "vits16", "token_overlap": False},
        }
    ]
    assert events == ["verify", "load"]


def test_madeleine_downloads_pinned_snapshot_then_loads_local_files(
    monkeypatch, tmp_path
):
    import raw2features.slide_embedders.madeleine as madeleine_embedder

    (tmp_path / "model_config.json").write_text(json.dumps({"precision": "float32"}))
    (tmp_path / "model.pt").write_bytes(b"test checkpoint")
    downloads = []
    loads = []
    events = []

    def snapshot_download(**kwargs):
        downloads.append(kwargs)
        return str(tmp_path)

    _fake_hub(monkeypatch, snapshot_download=snapshot_download)
    local_dir = tmp_path / "madeleine-cache"
    monkeypatch.setenv("RAW2FEATURES_MADELEINE_DIR", str(local_dir))
    madeleine = ModuleType("madeleine")
    madeleine.__path__ = []
    models = ModuleType("madeleine.models")
    models.__path__ = []
    model_module = ModuleType("madeleine.models.Model")

    def create_model(model_cfg, *, device, checkpoint_path):
        events.append("load")
        loads.append((model_cfg, device, checkpoint_path))
        return _Model()

    model_module.create_model = create_model
    monkeypatch.setitem(sys.modules, "madeleine", madeleine)
    monkeypatch.setitem(sys.modules, "madeleine.models", models)
    monkeypatch.setitem(sys.modules, "madeleine.models.Model", model_module)

    original_verify = madeleine_embedder.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(madeleine_embedder, "verify_sha256", verify)
    monkeypatch.setattr(
        madeleine_embedder,
        "_SPEC",
        replace(
            madeleine_embedder._SPEC,
            weights_sha256=_digest((tmp_path / "model.pt").read_bytes()),
        ),
    )
    emb = madeleine_embedder.MadeleineSlideEmbedder()
    emb.load(device="cpu")
    assert downloads == [
        {
            "repo_id": "MahmoodLab/madeleine",
            "revision": emb.spec.weights_revision,
            "allow_patterns": ("model_config.json", "model.pt"),
            "local_dir": str(local_dir),
        }
    ]
    cfg, device, checkpoint = loads[0]
    assert events == ["verify", "load"]
    assert cfg.precision == "float32"
    assert device == "cpu"
    assert checkpoint == str(tmp_path / "model.pt")


def _fake_seal_stack(monkeypatch, model, captured):
    seal = ModuleType("seal")
    seal.__path__ = []
    models = ModuleType("seal.models")
    models.__path__ = []
    load_model = ModuleType("seal.models.load_model")
    constants = ModuleType("seal.utils.constants")
    utils = ModuleType("seal.utils")
    utils.__path__ = []

    class ModelMixin:
        def get_img_model(self, backbone, **kwargs):
            captured.append((backbone, dict(self.conf), kwargs))
            return model, None, object()

    load_model.ModelMixin = ModelMixin
    constants.EMB_DICT = {"conch": 512, "univ2": 1536}
    monkeypatch.setitem(sys.modules, "seal", seal)
    monkeypatch.setitem(sys.modules, "seal.models", models)
    monkeypatch.setitem(sys.modules, "seal.models.load_model", load_model)
    monkeypatch.setitem(sys.modules, "seal.utils", utils)
    monkeypatch.setitem(sys.modules, "seal.utils.constants", constants)


def test_seal_verifies_adapter_before_deserialising_and_enforces_constructor(
    monkeypatch, tmp_path
):
    import raw2features.embedders.seal_embedder as seal_embedder
    from raw2features.embedders.fingerprint import SEAL_CONSTRUCTOR_CONTRACT

    events = []
    captured = []
    checkpoint = tmp_path / "seal_conch_vision.pth"
    checkpoint.write_bytes(b"pinned SEAL adapter")
    keys = {
        "encoder.base_model.model.trunk.blocks.11.attn.qkv.lora_A.default.weight": 1,
        "encoder.base_model.model.trunk.blocks.11.attn.qkv.lora_B.default.weight": 2,
    }

    class Model(_Model):
        def state_dict(self):
            return dict(keys)

        def load_state_dict(self, state, strict=False):
            events.append("load_state_dict")
            assert state == keys
            assert strict is False
            return [], []

    model = Model()
    _fake_seal_stack(monkeypatch, model, captured)
    downloads = []

    torch = ModuleType("torch")
    torch.float32 = object()

    def torch_load(path, **kwargs):
        events.append("torch.load")
        assert path == str(checkpoint)
        assert kwargs["map_location"] == "cpu"
        return {"state_dict": keys}

    torch.load = torch_load
    monkeypatch.setitem(sys.modules, "torch", torch)
    def file_download(**kwargs):
        downloads.append(kwargs)
        return str(checkpoint)

    _fake_hub(monkeypatch, file_download=file_download)

    original_verify = seal_embedder.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(seal_embedder, "verify_sha256", verify)
    spec = replace(
        get_spec("seal_conch"),
        weights_sha256=_digest(checkpoint.read_bytes()),
    )
    seal_embedder.SealEmbedder(spec).load(device="cpu", dtype=object())

    assert events == ["verify", "torch.load", "load_state_dict"]
    backbone, conf, kwargs = captured[0]
    assert backbone == "conch"
    assert {k: conf[k] for k in SEAL_CONSTRUCTOR_CONTRACT} == SEAL_CONSTRUCTOR_CONTRACT
    assert conf["encoder"] == "conch"
    assert conf["out_dim"] == 512
    assert kwargs["partial_blocks"] == 1
    assert kwargs["use_adapter"] is False
    assert downloads == [
        {
            "repo_id": "MahmoodLab/SEAL",
            "filename": "seal_conch_vision.pth",
            "revision": spec.weights_revision,
            "cache_dir": None,
        }
    ]


def test_seal_bad_digest_never_constructs_or_deserialises(monkeypatch, tmp_path):
    import raw2features.embedders.seal_embedder as seal_embedder

    checkpoint = tmp_path / "seal_conch_vision.pth"
    checkpoint.write_bytes(b"wrong bytes")
    captured = []
    _fake_seal_stack(monkeypatch, _Model(), captured)
    torch = ModuleType("torch")
    torch.float32 = object()
    torch.load = lambda *_args, **_kwargs: pytest.fail("torch.load must not run")
    monkeypatch.setitem(sys.modules, "torch", torch)
    _fake_hub(monkeypatch, file_download=lambda **_kwargs: str(checkpoint))

    with pytest.raises(ValueError, match="sha256"):
        seal_embedder.SealEmbedder(get_spec("seal_conch")).load(
            device="cpu", dtype=object()
        )
    assert captured == []


def test_seal_rejects_a_lora_delta_that_matches_no_model_keys(monkeypatch, tmp_path):
    import raw2features.embedders.seal_embedder as seal_embedder

    checkpoint = tmp_path / "seal_conch_vision.pth"
    checkpoint.write_bytes(b"adapter")
    adapter = {
        "wrong.lora_A.default.weight": 1,
        "wrong.lora_B.default.weight": 2,
    }

    class Model(_Model):
        def state_dict(self):
            return {"right.lora_A.default.weight": 1, "right.lora_B.default.weight": 2}

        def load_state_dict(self, *_args, **_kwargs):
            pytest.fail("load_state_dict must not accept a non-matching adapter")

    _fake_seal_stack(monkeypatch, Model(), [])
    torch = ModuleType("torch")
    torch.float32 = object()
    torch.load = lambda *_args, **_kwargs: {"state_dict": adapter}
    monkeypatch.setitem(sys.modules, "torch", torch)
    _fake_hub(monkeypatch, file_download=lambda **_kwargs: str(checkpoint))
    spec = replace(get_spec("seal_conch"), weights_sha256=_digest(b"adapter"))

    with pytest.raises(ValueError, match="do not match"):
        seal_embedder.SealEmbedder(spec).load(device="cpu", dtype=object())


def test_seal_rejects_a_partial_lora_delta(monkeypatch, tmp_path):
    import raw2features.embedders.seal_embedder as seal_embedder

    checkpoint = tmp_path / "seal_conch_vision.pth"
    checkpoint.write_bytes(b"partial adapter")
    adapter = {
        "block.0.lora_A.default.weight": 1,
        "block.0.lora_B.default.weight": 2,
        "block.1.lora_A.default.weight": 3,
    }
    complete_model = {
        **adapter,
        "block.1.lora_B.default.weight": 4,
    }

    class Model(_Model):
        def state_dict(self):
            return complete_model

        def load_state_dict(self, *_args, **_kwargs):
            pytest.fail("load_state_dict must not accept a partial adapter")

    _fake_seal_stack(monkeypatch, Model(), [])
    torch = ModuleType("torch")
    torch.float32 = object()
    torch.load = lambda *_args, **_kwargs: {"state_dict": adapter}
    monkeypatch.setitem(sys.modules, "torch", torch)
    _fake_hub(monkeypatch, file_download=lambda **_kwargs: str(checkpoint))
    spec = replace(
        get_spec("seal_conch"),
        weights_sha256=_digest(checkpoint.read_bytes()),
    )

    with pytest.raises(ValueError, match="missing 1 LoRA adapter key"):
        seal_embedder.SealEmbedder(spec).load(device="cpu", dtype=object())


def _fake_gigapath_stack(monkeypatch, create_model):
    gigapath = ModuleType("gigapath")
    gigapath.__path__ = []
    slide_encoder = ModuleType("gigapath.slide_encoder")
    slide_encoder.create_model = create_model
    torchscale = ModuleType("torchscale")
    torchscale.__path__ = []
    architecture = ModuleType("torchscale.architecture")
    architecture.__path__ = []
    config = ModuleType("torchscale.architecture.config")

    class EncoderConfig:
        pass

    EncoderConfig.__module__ = config.__name__
    config.EncoderConfig = EncoderConfig
    monkeypatch.setitem(sys.modules, "gigapath", gigapath)
    monkeypatch.setitem(sys.modules, "gigapath.slide_encoder", slide_encoder)
    monkeypatch.setitem(sys.modules, "torchscale", torchscale)
    monkeypatch.setitem(sys.modules, "torchscale.architecture", architecture)
    monkeypatch.setitem(sys.modules, "torchscale.architecture.config", config)


def test_gigapath_slide_verifies_before_create_model(monkeypatch, tmp_path):
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide
    from raw2features.slide_embedders.model_registry import get_slide_spec

    checkpoint = tmp_path / "slide_encoder.pth"
    checkpoint.write_bytes(b"pinned GigaPath slide weights")
    events = []
    calls = []
    downloads = []

    def create_model(*args, **kwargs):
        events.append("create_model")
        calls.append((args, kwargs))
        return _Model()

    _fake_gigapath_stack(monkeypatch, create_model)
    def file_download(**kwargs):
        downloads.append(kwargs)
        return str(checkpoint)

    _fake_hub(monkeypatch, file_download=file_download)
    original_verify = gigapath_slide.verify_sha256

    def verify(*args, **kwargs):
        events.append("verify")
        original_verify(*args, **kwargs)

    monkeypatch.setattr(gigapath_slide, "verify_sha256", verify)
    emb = gigapath_slide.GigapathSlideEmbedder()
    emb.spec = replace(
        get_slide_spec("gigapath_slide"),
        weights_sha256=_digest(checkpoint.read_bytes()),
    )
    emb.load(device="cpu")

    assert events == ["verify", "create_model"]
    assert calls == [
        (
            (str(checkpoint), "gigapath_slide_enc12l768d", 1536),
            {"global_pool": True},
        )
    ]
    assert downloads == [
        {
            "repo_id": "prov-gigapath/prov-gigapath",
            "filename": "slide_encoder.pth",
            "revision": emb.spec.weights_revision,
            "cache_dir": None,
        }
    ]


def test_gigapath_bad_digest_never_reaches_create_model(monkeypatch, tmp_path):
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide
    from raw2features.slide_embedders.model_registry import get_slide_spec

    checkpoint = tmp_path / "slide_encoder.pth"
    checkpoint.write_bytes(b"wrong bytes")
    _fake_gigapath_stack(
        monkeypatch,
        lambda *_args, **_kwargs: pytest.fail("create_model must not run"),
    )
    _fake_hub(monkeypatch, file_download=lambda **_kwargs: str(checkpoint))
    emb = gigapath_slide.GigapathSlideEmbedder()
    emb.spec = get_slide_spec("gigapath_slide")

    with pytest.raises(ValueError, match="sha256"):
        emb.load(device="cpu")


@pytest.mark.parametrize(
    "model_name", ["quiltnet", "conch", "kronos", "seal_conch", "seal_univ2"]
)
def test_loader_checksum_contract_is_recorded(model_name):
    """Every file-deserialising loader covered above has a registry digest."""
    assert len(get_spec(model_name).weights_sha256 or "") == 64
