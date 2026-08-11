"""PRISM2 registry, loader, and forward-contract tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from raw2features.embedders.fingerprint import (
    patch_output_fingerprint,
    resolved_slide_amp,
    slide_output_fingerprint,
)
from raw2features.embedders.model_registry import get_spec
from raw2features.slide_embedders.model_registry import get_slide_spec
from raw2features.slide_embedders.prism2 import (
    PRISM2_CHECKPOINT_SHA256,
    PRISM2_CODE_FILES,
    PRISM2_FLASH_ATTN_VERSION,
    PRISM2_PHI3_ARTIFACT_SHA256,
    PRISM2_PHI3_REVISION,
    PRISM2_PHI3_SOURCE,
    PRISM2_REVISION,
    PRISM2_TRANSFORMERS_VERSION,
    Prism2DiagnosticSlideEmbedder,
    Prism2SlideEmbedder,
    _assert_snapshot_files,
    _require_prism2_runtime,
)


def test_prism2_registry_records_both_published_embeddings():
    base = get_slide_spec("prism2")
    diagnostic = get_slide_spec("prism2_diagnostic")

    for spec in (base, diagnostic):
        assert spec.family == "prism2"
        assert spec.patch_encoder == "virchow2"
        # Stored virchow2 is [CLS(1280), mean-patch(1280)]; the loader slices CLS.
        assert spec.patch_dim == 2560
        assert spec.gated is True
        assert spec.license == "CC-BY-NC-ND-4.0"
        assert spec.weights_revision == PRISM2_REVISION
        assert spec.weights_filename == "model.safetensors.index.json"
        assert spec.weights_manifest == PRISM2_CHECKPOINT_SHA256
        assert (
            spec.weights_sha256
            == PRISM2_CHECKPOINT_SHA256["model.safetensors.index.json"]
        )
        assert spec.doi == "10.1038/s41591-026-04521-4"
    assert base.embedding_dim == 2560
    assert diagnostic.embedding_dim == 3072


@pytest.mark.parametrize(
    ("embedder_cls", "method", "dimension"),
    [
        (Prism2SlideEmbedder, "base", 2560),
        (Prism2DiagnosticSlideEmbedder, "diagnostic", 3072),
    ],
)
def test_prism2_forward_slices_preserved_virchow2_cls(embedder_cls, method, dimension):
    torch = pytest.importorskip("torch")
    calls = []

    class Model:
        def _forward(self, name, tile_embeddings, attention_mask):
            calls.append((name, tile_embeddings.cpu(), attention_mask.cpu()))
            return torch.full((1, dimension), 3.0)

        def get_base_embedding(self, tile_embeddings, attention_mask):
            return self._forward("base", tile_embeddings, attention_mask)

        def get_diagnostic_embedding(self, tile_embeddings, attention_mask):
            return self._forward("diagnostic", tile_embeddings, attention_mask)

    features = np.empty((3, 2560), dtype=np.float32)
    features[:, :1280] = np.arange(3, dtype=np.float32)[:, None]
    features[:, 1280:] = 1000.0
    embedder = embedder_cls()
    embedder._model = Model()
    embedder._device = "cpu"

    output = embedder.encode(features)

    assert output.shape == (dimension,)
    assert np.isfinite(output).all()
    name, received, mask = calls.pop()
    assert name == method
    np.testing.assert_array_equal(received.numpy(), features[:, :1280][None])
    assert received.shape == (1, 3, 1280)
    assert mask.dtype == torch.int32
    np.testing.assert_array_equal(mask.numpy(), np.ones((1, 3), dtype=np.int32))


def test_prism2_rejects_an_incompatible_patch_array():
    pytest.importorskip("torch")
    embedder = Prism2SlideEmbedder()
    embedder._model = SimpleNamespace()
    with pytest.raises(ValueError, match=r"virchow2 features shaped \(N, 2560\)"):
        embedder.encode(np.zeros((4, 1280), dtype=np.float32))


def test_prism2_load_uses_only_verified_local_snapshots(monkeypatch):
    calls = []

    def verified(source, revision, artifacts, *, extra_files=(), what):
        calls.append((source, revision, dict(artifacts), tuple(extra_files), what))
        return "/verified/phi3" if "Phi-3" in what else "/verified/prism2"

    config = SimpleNamespace(text_decoder_model_id="mutable-upstream-name")

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == "/verified/prism2"
            assert kwargs == {"trust_remote_code": True, "local_files_only": True}
            return config

    class Model:
        def eval(self):
            return self

        def to(self, device):
            assert device == "cuda:3"
            return self

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == "/verified/prism2"
            assert kwargs["config"] is config
            assert kwargs["trust_remote_code"] is True
            assert kwargs["local_files_only"] is True
            assert kwargs["torch_dtype"] == "auto"
            return Model()

    import raw2features.slide_embedders.prism2 as module

    monkeypatch.setattr(module, "_verified_snapshot", verified)
    monkeypatch.setattr(module, "_require_prism2_runtime", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel),
    )

    loaded = Prism2SlideEmbedder().load(device="cuda:3")

    assert loaded._device == "cuda:3"
    assert config.text_decoder_model_id == "/verified/phi3"
    assert calls[0][1:] == (
        PRISM2_REVISION,
        PRISM2_CHECKPOINT_SHA256,
        PRISM2_CODE_FILES,
        "prism2",
    )
    assert calls[1][0:4] == (
        PRISM2_PHI3_SOURCE,
        PRISM2_PHI3_REVISION,
        PRISM2_PHI3_ARTIFACT_SHA256,
        (),
    )


def test_prism2_rejects_cpu_before_downloading(monkeypatch):
    import raw2features.slide_embedders.prism2 as module

    monkeypatch.setattr(
        module,
        "_verified_snapshot",
        lambda *args, **kwargs: pytest.fail("download attempted"),
    )
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        Prism2SlideEmbedder().load(device="cpu")


def test_prism2_runtime_pin_fails_clearly(monkeypatch):
    import raw2features.slide_embedders.prism2 as module

    versions = {"transformers": "4.57.6", "flash-attn": None}

    def version(distribution):
        value = versions[distribution]
        if value is None:
            raise module.importlib_metadata.PackageNotFoundError(distribution)
        return value

    monkeypatch.setattr(module.importlib_metadata, "version", version)
    with pytest.raises(
        RuntimeError,
        match=(
            r"transformers==4\.51\.3 \(found 4\.57\.6\).*"
            r"flash-attn==2\.8\.3 \(found missing\)"
        ),
    ):
        _require_prism2_runtime()


def test_prism2_snapshot_allowlist_fails_closed(tmp_path):
    for filename in ("config.json", "model.safetensors.index.json"):
        (tmp_path / filename).write_bytes(b"test")
    _assert_snapshot_files(
        str(tmp_path),
        ("config.json", "model.safetensors.index.json"),
        what="prism2",
    )

    (tmp_path / "stale.py").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="unexpected=.*stale.py"):
        _assert_snapshot_files(
            str(tmp_path),
            ("config.json", "model.safetensors.index.json"),
            what="prism2",
        )


def test_prism2_fingerprint_binds_cls_projection_shards_and_phi3():
    patch = patch_output_fingerprint(get_spec("virchow2"), "fp16")
    base_spec = get_slide_spec("prism2")
    diagnostic_spec = get_slide_spec("prism2_diagnostic")
    base = slide_output_fingerprint(
        base_spec,
        patch_model="virchow2",
        patch_output_fingerprint=patch,
        patch_dim=2560,
        resolved_amp=resolved_slide_amp(base_spec, "cuda:0"),
    )
    diagnostic = slide_output_fingerprint(
        diagnostic_spec,
        patch_model="virchow2",
        patch_output_fingerprint=patch,
        patch_dim=2560,
        resolved_amp=resolved_slide_amp(diagnostic_spec, "cuda:0"),
    )

    assert base["payload"]["checkpoint"]["weights_manifest"] == (
        PRISM2_CHECKPOINT_SHA256
    )
    constructor = base["payload"]["loader"]["constructor"]
    assert constructor["model_input_projection"] == {
        "operation": "slice",
        "axis": 1,
        "start": 0,
        "stop": 1280,
        "meaning": "Virchow2 CLS token",
    }
    assert constructor["forward"] == "get_base_embedding"
    assert constructor["transformers_version"] == PRISM2_TRANSFORMERS_VERSION
    assert constructor["flash_attn_version"] == PRISM2_FLASH_ATTN_VERSION
    phi3 = constructor["phi3_construction_dependency"]
    assert phi3["repo"] == "microsoft/Phi-3-mini-128k-instruct"
    assert phi3["revision"] == PRISM2_PHI3_REVISION
    assert phi3["artifacts"] == PRISM2_PHI3_ARTIFACT_SHA256
    assert base["payload"]["output"]["resolved_amp"] == "bf16"
    assert diagnostic["payload"]["loader"]["constructor"]["forward"] == (
        "get_diagnostic_embedding"
    )
    assert diagnostic["payload"]["output"]["embedding_dim"] == 3072
    assert base["digest"] != diagnostic["digest"]


def test_prism2_fields_do_not_change_existing_prism_fingerprint_shape():
    patch = patch_output_fingerprint(get_spec("virchow"), "fp16")
    prism = get_slide_spec("prism")
    fingerprint = slide_output_fingerprint(
        prism,
        patch_model="virchow",
        patch_output_fingerprint=patch,
        patch_dim=2560,
        resolved_amp="fp16",
    )
    assert "weights_manifest" not in fingerprint["payload"]["checkpoint"]
    assert fingerprint["payload"]["loader"]["constructor"]["forward"] == (
        "slide_representations"
    )
