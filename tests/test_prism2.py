"""PRISM2 registry, loader, and forward-contract tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from raw2features.embedders.fingerprint import (
    patch_output_fingerprint,
    resolved_slide_amp,
    slide_output_dim,
    slide_output_fingerprint,
)
from raw2features.embedders.model_registry import get_spec
from raw2features.pipeline.runner import _preflight_slide_encoders_for_grid
from raw2features.slide_embedders.encoding import write_slide_embedding
from raw2features.slide_embedders.model_registry import (
    get_slide_spec,
    validate_slide_encoder_runtime,
)
from raw2features.slide_embedders.prism2 import (
    _VERIFIED_SNAPSHOT_CACHE,
    PRISM2_CHECKPOINT_SHA256,
    PRISM2_CODE_SHA256,
    PRISM2_FLASH_ATTN_VERSION,
    PRISM2_PHI3_ARTIFACT_SHA256,
    PRISM2_PHI3_CONFIG_SHA256,
    PRISM2_PHI3_MASK_COMPAT_VERSION,
    PRISM2_PHI3_REVISION,
    PRISM2_PHI3_SOURCE,
    PRISM2_REVISION,
    PRISM2_RUNTIME_SHA256,
    PRISM2_SNAPSHOT_ALLOW_PATTERNS,
    PRISM2_TRANSFORMERS_VERSION,
    Prism2DiagnosticSlideEmbedder,
    Prism2SlideEmbedder,
    _assert_snapshot_files,
    _bind_phi3_mask_compatibility,
    _prepare_phi3_4d_causal_mask,
    _require_prism2_runtime,
    _verified_snapshot,
)


def _write_prism2_patch_grid(root, key: str):
    group = root["grids"].create_group(key)
    coords = group.create_array("coords", shape=(2, 2), dtype="int32")
    coords[:] = np.asarray([[0, 0], [224, 0]], dtype=np.int32)
    patch_spec = get_spec("virchow2")
    patch_fingerprint = patch_output_fingerprint(patch_spec, "fp16")
    features = group.create_group("features").create_array(
        "virchow2",
        shape=(2, patch_spec.embedding_dim),
        chunks=(2, patch_spec.embedding_dim),
        dtype="float32",
    )
    features[:] = 1.0
    features.attrs.update(
        {
            "role": "features",
            "model": "virchow2",
            "output_fingerprint": patch_fingerprint,
        }
    )
    group.attrs["raw2features"] = {
        "schema_version": "0.1",
        "models": {
            "virchow2": {
                "embedding_dim": patch_spec.embedding_dim,
                "output_fingerprint": patch_fingerprint,
            }
        },
        "patching": {"level0_patch": 224},
    }
    return group, patch_spec, patch_fingerprint


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

    def verified(source, revision, artifacts, *, allowed_files=None, what):
        calls.append(
            (source, revision, dict(artifacts), tuple(allowed_files or ()), what)
        )
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
        PRISM2_RUNTIME_SHA256,
        (),
        "prism2",
    )
    assert calls[1] == (
        PRISM2_PHI3_SOURCE,
        PRISM2_PHI3_REVISION,
        PRISM2_PHI3_CONFIG_SHA256,
        tuple(PRISM2_PHI3_ARTIFACT_SHA256),
        "prism2:Phi-3",
    )


def test_prism2_diagnostic_load_binds_helper_and_full_phi3_assets(monkeypatch):
    calls = []
    model = SimpleNamespace(
        eval=lambda: model,
        to=lambda device: model,
    )

    def verified(source, revision, artifacts, *, allowed_files=None, what):
        calls.append((dict(artifacts), tuple(allowed_files or ()), what))
        return "/verified/phi3" if "Phi-3" in what else "/verified/prism2"

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return SimpleNamespace(text_decoder_model_id="mutable")

    class AutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model

    import raw2features.slide_embedders.prism2 as module

    monkeypatch.setattr(module, "_verified_snapshot", verified)
    monkeypatch.setattr(module, "_require_prism2_runtime", lambda: None)
    monkeypatch.setattr(
        module,
        "_bind_phi3_mask_compatibility",
        lambda received: calls.append(("bind", received)),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=AutoConfig, AutoModel=AutoModel),
    )

    Prism2DiagnosticSlideEmbedder().load("cuda:0")

    assert calls[1] == (
        PRISM2_PHI3_ARTIFACT_SHA256,
        tuple(PRISM2_PHI3_ARTIFACT_SHA256),
        "prism2_diagnostic:Phi-3",
    )
    assert calls[2] == ("bind", model)


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
            r"install flash-attn==2\.8\.3 with --no-build-isolation.*"
            r"transformers==4\.56\.0 \(found 4\.57\.6\).*"
            r"flash-attn==2\.8\.3 \(found missing\)"
        ),
    ):
        _require_prism2_runtime()


def test_prism2_runtime_pin_accepts_pep440_equivalent_versions(monkeypatch):
    import raw2features.slide_embedders.prism2 as module

    versions = {"transformers": "4.56", "flash-attn": "2.8.3+cu128"}
    monkeypatch.setattr(
        module.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    monkeypatch.setattr(module.importlib, "import_module", lambda name: object())

    _require_prism2_runtime()


def test_prism2_runtime_preflight_rejects_an_incompatible_flash_attn_abi(
    monkeypatch,
):
    import raw2features.slide_embedders.prism2 as module

    versions = {"transformers": "4.56.0", "flash-attn": "2.8.3"}
    monkeypatch.setattr(
        module.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    def import_module(name):
        assert name == "flash_attn"
        raise ImportError("undefined symbol: _ZN3c105ErrorC2E")

    monkeypatch.setattr(module.importlib, "import_module", import_module)

    with pytest.raises(
        RuntimeError,
        match=(
            r"found flash-attn==2\.8\.3, but it could not be imported.*"
            r"Rebuild flash-attn against the active Torch/CUDA environment.*"
            r"undefined symbol"
        ),
    ):
        _require_prism2_runtime()


def test_prism2_runtime_pin_rejects_unvalidated_prereleases(monkeypatch):
    import raw2features.slide_embedders.prism2 as module

    versions = {"transformers": "4.56.0.dev0", "flash-attn": "2.8.3"}
    monkeypatch.setattr(
        module.importlib_metadata,
        "version",
        lambda distribution: versions[distribution],
    )

    with pytest.raises(RuntimeError, match=r"found 4\.56\.0\.dev0"):
        _require_prism2_runtime()


def test_prism2_runtime_preflight_checks_packages_and_device(monkeypatch):
    import raw2features.slide_embedders.prism2 as module

    calls = []
    monkeypatch.setattr(module, "_require_prism2_runtime", lambda: calls.append(True))

    validate_slide_encoder_runtime(["mean"], devices=["cpu"])
    assert calls == []
    validate_slide_encoder_runtime(["prism2"], devices=["cuda:0"])
    assert calls == [True]
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        validate_slide_encoder_runtime(
            ["prism2_diagnostic"],
            devices=["cpu"],
        )


def test_prism2_restores_released_phi3_mask_contract():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    mask = _prepare_phi3_4d_causal_mask(
        attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.int8),
        sequence_length=3,
        target_length=3,
        dtype=torch.float32,
        device=torch.device("cpu"),
        cache_position=torch.arange(3),
        batch_size=1,
        config=SimpleNamespace(sliding_window=None),
        past_key_values=None,
    )
    minimum = torch.finfo(torch.float32).min
    expected = torch.tensor(
        [[[[0.0, minimum, minimum], [0.0, 0.0, minimum], [0.0, 0.0, minimum]]]]
    )
    torch.testing.assert_close(mask, expected, rtol=0, atol=0)

    supplied = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    assert (
        _prepare_phi3_4d_causal_mask(
            supplied,
            2,
            2,
            torch.float32,
            torch.device("cpu"),
            torch.arange(2),
            1,
            SimpleNamespace(sliding_window=None),
            None,
        )
        is supplied
    )


def test_prism2_binds_phi3_helper_only_to_verified_dynamic_module(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.models.phi3.modeling_phi3 import Phi3Model

    sentinel = object()
    shared_before = getattr(
        Phi3Model,
        "_prepare_4d_causal_attention_mask_with_cache_position",
        sentinel,
    )
    module_name = "transformers_modules.paige_prism2.modeling_prism2"
    original = type(
        "Phi3Model",
        (),
        {"__module__": "transformers.models.phi3.modeling_phi3"},
    )
    dynamic_module = SimpleNamespace(Phi3Model=original)
    model_cls = type("Prism2Model", (), {"__module__": module_name})
    monkeypatch.setitem(sys.modules, module_name, dynamic_module)

    _bind_phi3_mask_compatibility(model_cls())

    helper = dynamic_module.Phi3Model
    assert helper is not original
    assert callable(helper._prepare_4d_causal_attention_mask_with_cache_position)
    assert (
        getattr(
            Phi3Model,
            "_prepare_4d_causal_attention_mask_with_cache_position",
            sentinel,
        )
        is shared_before
    )


def test_prism2_rejects_unknown_dynamic_module_binding(monkeypatch):
    module_name = "transformers_modules.paige_prism2.modeling_prism2"
    dynamic_module = SimpleNamespace(Phi3Model=object)
    model_cls = type("Prism2Model", (), {"__module__": module_name})
    monkeypatch.setitem(sys.modules, module_name, dynamic_module)

    with pytest.raises(RuntimeError, match="does not expose the expected Phi3Model"):
        _bind_phi3_mask_compatibility(model_cls())


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


def test_prism2_runtime_code_is_sha_bound_and_cached(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    artifact = cache / "modeling_prism2.py"
    downloads = []
    hashes = []

    def download(*args, **kwargs):
        downloads.append(kwargs["local_dir"])
        cache.mkdir()
        artifact.write_bytes(b"verified code")
        return str(cache)

    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.download_pinned_hf_snapshot", download
    )
    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.pinned_model_cache_dir",
        lambda *args: str(cache),
    )

    def verify(path, expected, *, what):
        hashes.append((Path(path).name, expected, what))

    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.verify_sha256",
        verify,
    )
    _VERIFIED_SNAPSHOT_CACHE.clear()
    artifacts = {"modeling_prism2.py": "a" * 64}

    _verified_snapshot("hf-hub:test/model", "a" * 40, artifacts, what="test")
    _verified_snapshot("hf-hub:test/model", "a" * 40, artifacts, what="test")
    assert len(downloads) == 1
    assert len(hashes) == 1

    artifact.write_bytes(b"changed code")
    _verified_snapshot("hf-hub:test/model", "a" * 40, artifacts, what="test")
    assert len(downloads) == 1
    assert len(hashes) == 2


def test_prism2_rejects_bad_runtime_digest_before_custom_code(monkeypatch, tmp_path):
    for filename in PRISM2_SNAPSHOT_ALLOW_PATTERNS:
        (tmp_path / filename).write_bytes(b"untrusted")
    events = []

    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.download_pinned_hf_snapshot",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.pinned_model_cache_dir",
        lambda *args: str(tmp_path),
    )
    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2._require_prism2_runtime",
        lambda: None,
    )

    def reject(*args, **kwargs):
        events.append("verify")
        raise ValueError("digest mismatch")

    monkeypatch.setattr(
        "raw2features.slide_embedders.prism2.verify_sha256",
        reject,
    )
    _VERIFIED_SNAPSHOT_CACHE.clear()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *a, **k: pytest.fail("custom code executed")
            ),
            AutoModel=SimpleNamespace(
                from_pretrained=lambda *a, **k: pytest.fail("custom code executed")
            ),
        ),
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        Prism2SlideEmbedder().load("cuda:0")
    assert events == ["verify"]


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
    assert phi3["artifacts"] == PRISM2_PHI3_CONFIG_SHA256
    assert "mask_compatibility" not in phi3
    assert constructor["snapshot"] == {
        "allow_patterns": list(PRISM2_SNAPSHOT_ALLOW_PATTERNS),
        "custom_code_sha256": PRISM2_CODE_SHA256,
    }
    assert base["payload"]["output"]["resolved_amp"] == "bf16"
    diagnostic_constructor = diagnostic["payload"]["loader"]["constructor"]
    assert diagnostic_constructor["forward"] == "get_diagnostic_embedding"
    diagnostic_phi3 = diagnostic_constructor["phi3_construction_dependency"]
    assert diagnostic_phi3["artifacts"] == PRISM2_PHI3_ARTIFACT_SHA256
    assert diagnostic_phi3["mask_compatibility"]["version"] == (
        PRISM2_PHI3_MASK_COMPAT_VERSION
    )
    assert diagnostic_phi3["mask_compatibility"]["binding_scope"] == (
        "verified_prism2_dynamic_module"
    )
    assert diagnostic["payload"]["output"]["embedding_dim"] == 3072
    assert base["digest"] != diagnostic["digest"]


def test_prism2_amp_contract_is_device_independent():
    for name in ("prism2", "prism2_diagnostic"):
        spec = get_slide_spec(name)
        assert resolved_slide_amp(spec, "cpu") == "bf16"
        assert resolved_slide_amp(spec, "cuda:0") == "bf16"


def test_prism2_grid_preflight_uses_exact_grid_and_slide_device(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "complete.embeddings.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=2)
    root.create_group("grids")
    group, patch_spec, patch_fingerprint = _write_prism2_patch_grid(
        root, "mpp0.5_px224"
    )
    root.attrs["raw2features"] = {
        "schema_version": "0.1",
        "grids": {"mpp0.5_px224": {}},
    }

    slide_spec = get_slide_spec("prism2")
    slide_fingerprint = slide_output_fingerprint(
        slide_spec,
        patch_model="virchow2",
        patch_output_fingerprint=patch_fingerprint,
        patch_dim=patch_spec.embedding_dim,
        resolved_amp="bf16",
    )
    write_slide_embedding(
        group,
        "prism2",
        np.ones(slide_output_dim(slide_spec, patch_spec.embedding_dim)),
        {
            "patch_encoder": "virchow2",
            "embedding_dim": slide_spec.embedding_dim,
            "output_fingerprint": slide_fingerprint,
        },
    )

    assert (
        _preflight_slide_encoders_for_grid(
            path,
            "mpp0.5_px224",
            ["prism2"],
            ["virchow2"],
            [],
            "cpu",
        )
        == []
    )

    _write_prism2_patch_grid(root, "mpp1_px224")
    import raw2features.slide_embedders.prism2 as module

    monkeypatch.setattr(module, "_require_prism2_runtime", lambda: None)
    with pytest.raises(ValueError, match="requires a CUDA GPU"):
        _preflight_slide_encoders_for_grid(
            path,
            "mpp1_px224",
            ["prism2"],
            ["virchow2"],
            [],
            "cpu",
        )


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
