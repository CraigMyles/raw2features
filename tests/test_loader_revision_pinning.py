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
    assert "not part of grid/model identity" in notes


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


@pytest.mark.parametrize("model_name", ["quiltnet", "conch", "kronos"])
def test_loader_checksum_contract_is_recorded(model_name):
    """Every file-deserialising loader covered above has a registry digest."""
    assert len(get_spec(model_name).weights_sha256 or "") == 64
