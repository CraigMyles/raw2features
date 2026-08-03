"""KRONOS2's pinned loader, marker identity, and output contract."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from raw2features.embedders.fingerprint import (
    patch_output_fingerprint,
    resolved_patch_amp,
)
from raw2features.embedders.kronos2_embedder import (
    KRONOS2_ADDITIONAL_MARKER_REGISTRATION_CONTRACT,
    KRONOS2_BIOLINKBERT_ARTIFACT_SHA256,
    KRONOS2_BIOLINKBERT_REVISION,
    KRONOS2_BIOLINKBERT_SOURCE,
    KRONOS2_FORWARD_CONTRACT,
    KRONOS2_MARKER_MATCHING_CONTRACT,
    KRONOS2_SCALING_CONTRACT,
    KRONOS2_SNAPSHOT_ALLOW_PATTERNS,
    KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST,
    KRONOS2_XFORMERS_VERSION,
    Kronos2Embedder,
    _allows_pinned_xformers,
    _marker_match_key,
    _scaling_factor,
)
from raw2features.embedders.kronos2_metadata import (
    KRONOS2_ADDITIONAL_MARKER_COLUMNS,
    parse_kronos2_additional_markers,
)
from raw2features.embedders.model_registry import get_spec
from raw2features.pipeline.runner import RunConfig, expected_model_contracts


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(
    canonical: str,
    *,
    pretraining: bool = True,
    family: str = "immune_marker",
) -> dict:
    return {
        "canonical": canonical,
        "pretraining": pretraining,
        "compartment": "membrane",
        "family": family,
    }


def _additional_contract(
    tmp_path,
    *,
    marker_name: str = "Novel-X",
    marker_full_name: str = "Novel marker X",
    compartment: str = "membrane",
    family: str = "immune_marker",
    family_desc: str = "immune marker",
    mean: float = 0.25,
    std: float = 0.5,
    filename: str = "additional.csv",
):
    path = tmp_path / filename
    path.write_text(
        ",".join(KRONOS2_ADDITIONAL_MARKER_COLUMNS)
        + "\n"
        + ",".join(
            [
                marker_name,
                marker_full_name,
                compartment,
                family,
                family_desc,
                str(mean),
                str(std),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return parse_kronos2_additional_markers(path)


def _panel_only_embedder() -> Kronos2Embedder:
    emb = Kronos2Embedder(get_spec("kronos2"))
    base = {
        _marker_match_key("CD3"): _record("CD3"),
        _marker_match_key("DAPI"): _record("DAPI", family="dna_stain")
        | {"compartment": "nucleus"},
    }
    emb._base_metadata_index = base
    emb._metadata_index = dict(base)
    emb._base_compartments = frozenset({"membrane", "nucleus"})
    emb._base_families = frozenset({"immune_marker", "dna_stain"})
    return emb


def test_kronos2_registry_and_fingerprint_cover_the_complete_static_contract():
    spec = get_spec("kronos2")
    assert spec.family == "kronos2"
    assert spec.modality == "multiplex"
    assert spec.source == "hf_hub:MahmoodLab/KRONOS2"
    assert spec.embedding_dim == 768
    assert spec.input_size == spec.extract_px == 256
    assert spec.pooling == "cls"
    assert spec.inference_amp == "fp32"
    assert spec.weights_revision == "e2a5473206c02d9973885b6445ef783f7e082951"
    assert spec.weights_sha256 == (
        "afa261ef8b63bb0acea62612d7594ac11fbdb89fbdd23c513f046a034725d2b4"
    )
    assert spec.timm_kwargs["marker_metadata_sha256"] == (
        "1e6791448a792c0edec9b2ba74efceaefba84d96852351cdc58d67f80b539074"
    )

    payload = patch_output_fingerprint(spec, "fp32")["payload"]
    assert payload["checkpoint"]["effective"] == {
        "repo": "MahmoodLab/KRONOS2",
        "filename": "kronos2_vitb16_teacher.pth",
        "mechanism": "pinned_local_snapshot",
    }
    constructor = payload["loader"]["constructor"]
    assert constructor["input"] == "app_owned_allowlisted_pinned_local_snapshot"
    assert constructor["transformers_local_files_only"] is True
    assert constructor["upstream_nested_download_guard"] == "local_directory_input"
    assert (
        constructor["upstream_sys_path_cleanup"]
        == "remove_new_entry_after_construction"
    )
    assert constructor["trust_remote_code"] is True
    assert constructor["snapshot_allow_patterns"] == list(
        KRONOS2_SNAPSHOT_ALLOW_PATTERNS
    )
    assert constructor["snapshot_top_level_allowlist"] == list(
        KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST
    )
    assert "*.py" not in KRONOS2_SNAPSHOT_ALLOW_PATTERNS
    assert "test.py" not in KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST
    assert "sp_image.py" not in KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST
    assert constructor["marker_matching"] == KRONOS2_MARKER_MATCHING_CONTRACT
    assert constructor["preferred_dapi_policy"] == (
        "canonical_dapi_else_draq5_else_none"
    )
    assert constructor["novel_marker_registration"] == (
        "explicit_additional_markers_parameter_with_conditional_contract"
    )
    assert constructor["forward"] == {
        **KRONOS2_FORWARD_CONTRACT,
        "method": "__call__",
        "marker_names": "canonical_marker_metadata_names",
        "result": "x_norm_clstoken",
    }
    assert constructor["forward"]["attention_backend_policy"] == {
        "selection": (
            "pinned_xformers_in_reference_environment_else_forced_fallback"
        ),
        "pinned_reference": {
            "implementation": "xformers.ops.memory_efficient_attention",
            "package_version": KRONOS2_XFORMERS_VERSION,
            "install_environment": "linux_x86_64_cpython_3_11_or_3_12",
        },
        "fallback": {
            "implementation": "upstream_dinov2_pytorch_attention",
            "condition": (
                "outside_reference_environment_or_xformers_unavailable_or_disabled"
            ),
            "enforcement": "temporary_XFORMERS_DISABLED_during_construction",
            "preimport_conflict": "fail_closed",
        },
    }
    assert constructor["marker_metadata"] == {
        "filename": "marker_metadata.csv",
        "sha256": spec.timm_kwargs["marker_metadata_sha256"],
    }
    assert constructor["published_marker_text_embeddings"] == "checkpoint_buffer"
    assert payload["preprocessing"]["native_multiplex"] == KRONOS2_SCALING_CONTRACT
    assert KRONOS2_FORWARD_CONTRACT["forward_microbatch_size"] == 8
    assert payload["output"] == {
        "pooling": "cls",
        "embedding_dim": 768,
        "reg_tokens": 0,
        "modality": "multiplex",
        "resolved_amp": "fp32",
    }
    assert KRONOS2_BIOLINKBERT_SOURCE not in json.dumps(payload, sort_keys=True)
    with pytest.raises(ValueError, match="does not accept requested AMP"):
        resolved_patch_amp(spec, "fp16", "cuda")
    with pytest.raises(ValueError, match="does not accept requested AMP"):
        resolved_patch_amp(spec, "bf16", "cuda")


def test_novel_marker_contract_conditionally_enters_the_output_fingerprint(tmp_path):
    additional = _additional_contract(tmp_path)
    cfg = RunConfig(
        models=["kronos2"],
        no_seg=True,
        device="cpu",
        resolved_channel_names=["DAPI", "Novel-X"],
        resolved_native_marker_selection=[
            {"source_index": 0, "source_name": "DAPI"},
            {"source_index": 1, "source_name": "Novel-X"},
        ],
        native_multiplex_params={
            "kronos2": {"additional_markers": additional}
        },
    )

    payload = expected_model_contracts(cfg)["kronos2"]["output_fingerprint"][
        "payload"
    ]
    dynamic = payload["multiplex_panel"]["model_params"]["additional_markers"]

    assert dynamic["content_sha256"] == additional["content_sha256"]
    assert dynamic["registration"]["text_encoder"]["source"] == (
        KRONOS2_BIOLINKBERT_SOURCE
    )
    assert dynamic["registration"]["text_encoder"]["revision"] == (
        KRONOS2_BIOLINKBERT_REVISION
    )
    assert dynamic["registration"]["text_encoder"]["artifact_sha256"] == (
        KRONOS2_BIOLINKBERT_ARTIFACT_SHA256
    )


def test_marker_matching_is_separator_only_and_never_uses_v1_aliases():
    assert _marker_match_key("CD-8") == _marker_match_key("CD_8") == "cd8"
    assert _marker_match_key("TCR-Vα7.2") == "tcrva72"

    emb = Kronos2Embedder(get_spec("kronos2"))
    emb._metadata_index = {
        _marker_match_key("CD8"): _record("CD8"),
        _marker_match_key("panCK"): _record("panCK"),
        _marker_match_key("CYTOKERATIN"): _record("CYTOKERATIN"),
        _marker_match_key("DAPI"): _record("DAPI", family="dna_stain"),
    }
    summary = emb.set_panel(["CD-8", "panCK", "DAPI", ""])
    assert summary["matched"] == ["CD8", "panCK", "DAPI"]
    assert summary["preferred_dapi"] == "DAPI"
    assert summary["dropped"] == [""]
    assert [row["channel_index"] for row in summary["mapping"]] == [0, 1, 2]

    # KRONOSv1 aliases CK -> CYTOKERATIN and Hoechst -> DAPI. KRONOS2 must not.
    with pytest.raises(ValueError, match=r"\['CK', 'Hoechst1'\]"):
        emb.set_panel(["CD8", "CK", "Hoechst1"])


def test_panel_provenance_marks_published_non_pretraining_entries():
    spec = get_spec("kronos2")
    emb = Kronos2Embedder(spec)
    emb._metadata_index = {
        _marker_match_key("CD3"): _record("CD3"),
        _marker_match_key("DRAQ5"): _record(
            "DRAQ5", pretraining=False, family="dna_stain"
        ),
    }

    summary = emb.set_panel(["CD3", "DRAQ-5"])
    assert summary["matched"] == ["CD3", "DRAQ5"]
    assert summary["non_pretraining"] == ["DRAQ5"]
    assert summary["preferred_dapi"] == "DRAQ5"
    assert summary["defaulted"] == []
    assert summary["vocabulary"] == {
        "source": spec.source,
        "filename": "marker_metadata.csv",
        "revision": spec.weights_revision,
        "sha256": spec.timm_kwargs["marker_metadata_sha256"],
    }
    assert summary["novel_markers"] == {
        "upstream_registration_available": True,
        "enabled": False,
    }


def test_normal_panel_binding_never_resolves_biolinkbert(monkeypatch):
    import raw2features.embedders.kronos2_embedder as module

    emb = _panel_only_embedder()
    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "ordinary published-marker binding downloaded BioLinkBERT"
        ),
    )

    summary = emb.bind_multiplex_panel(
        ["DAPI", "CD3"],
        selected_channels=[{"source_index": 1, "source_name": "CD3"}],
        model_params={},
    )

    assert summary["matched"] == ["CD3"]
    assert summary["novel_markers"]["enabled"] is False
    assert emb._raw2features_multiplex_source_indices == (1,)


def test_novel_registration_is_pinned_exact_ordered_and_instance_idempotent(
    monkeypatch, tmp_path
):
    import raw2features.embedders.kronos2_embedder as module

    torch = ModuleType("torch")

    class Device:
        def __init__(self, name):
            self.type = str(name)

        def __eq__(self, other):
            return isinstance(other, Device) and self.type == other.type

    class InferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    class FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array, dtype=np.float32)

        @property
        def shape(self):
            return self._array.shape

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self._array

    torch.device = Device
    torch.inference_mode = InferenceMode
    monkeypatch.setitem(sys.modules, "torch", torch)

    contract = _additional_contract(tmp_path)
    snapshot = tmp_path / "biolinkbert"
    snapshot.mkdir()
    for filename in KRONOS2_BIOLINKBERT_ARTIFACT_SHA256:
        (snapshot / filename).write_bytes(filename.encode())

    events = []
    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda source, revision, *, allow_patterns: (
            events.append(("download", source, revision, allow_patterns))
            or str(snapshot)
        ),
    )
    monkeypatch.setattr(
        module,
        "verify_sha256",
        lambda path, sha256, *, what: events.append(
            ("verify", Path(path).name, sha256, what)
        ),
    )

    builder_module_name = "_raw2features_test_kronos2_builder"
    builder_module = ModuleType(builder_module_name)
    encoder_inits = []
    prompt_inputs = []

    class InnerEncoder:
        def float(self):
            events.append(("encoder_float",))
            return self

        def eval(self):
            events.append(("encoder_eval",))
            return self

        def to(self, device):
            events.append(("encoder_to", device.type))
            return self

    class BioLinkBERTTextEncoder:
        def __init__(self, **kwargs):
            events.append(("encoder_init",))
            encoder_inits.append(kwargs)
            self.enc = InnerEncoder()

        def __call__(self, prompts):
            events.append(("encode", tuple(prompts)))
            return FakeTensor(np.ones((len(prompts), 1024), dtype=np.float32))

    def build_prompt(*values):
        prompt_inputs.append(values)
        return "prompt::" + "::".join(values)

    builder_module.BioLinkBERTTextEncoder = BioLinkBERTTextEncoder
    builder_module.build_prompt = build_prompt
    monkeypatch.setitem(sys.modules, builder_module_name, builder_module)
    Builder = type("MarkerEmbeddingBuilder", (), {})
    Builder.__module__ = builder_module_name
    marker_builder = Builder()
    original_cache = tmp_path / "original-cache"
    marker_builder.cache_dir = original_cache
    registration_calls = []
    temporary_dirs = []

    def register_additional_markers(csv_path):
        temporary_dirs.append(Path(marker_builder.cache_dir))
        array = np.load(
            Path(marker_builder.cache_dir) / "additional_marker_text_embeddings.npy",
            allow_pickle=False,
        )
        registration_calls.append(
            {
                "csv": Path(csv_path).read_text(encoding="utf-8"),
                "shape": tuple(array.shape),
                "dtype": array.dtype,
            }
        )
        events.append(("register",))

    emb = _panel_only_embedder()
    emb._model = SimpleNamespace(
        backbone=SimpleNamespace(marker_builder=marker_builder),
        register_additional_markers=register_additional_markers,
    )
    selection = [
        {"source_index": 2, "source_name": "Novel-X"},
        {"source_index": 0, "source_name": "DAPI"},
    ]
    params = {"additional_markers": contract}

    summary = emb.bind_multiplex_panel(
        ["DAPI", "CD3", "Novel-X"],
        selected_channels=selection,
        model_params=params,
    )

    assert events[0] == (
        "download",
        KRONOS2_BIOLINKBERT_SOURCE,
        KRONOS2_BIOLINKBERT_REVISION,
        tuple(KRONOS2_BIOLINKBERT_ARTIFACT_SHA256),
    )
    assert [event[1] for event in events if event[0] == "verify"] == list(
        KRONOS2_BIOLINKBERT_ARTIFACT_SHA256
    )
    assert max(
        index for index, event in enumerate(events) if event[0] == "verify"
    ) < next(index for index, event in enumerate(events) if event[0] == "encoder_init")
    assert encoder_inits == [
        {
            "model_id": str(snapshot),
            "device": torch.device("cpu"),
            "cache_dir": encoder_inits[0]["cache_dir"],
            "max_length": 128,
            "fp16_on_cuda": False,
        }
    ]
    assert Path(encoder_inits[0]["cache_dir"]).name == "transformers-cache"
    assert prompt_inputs == [("novel-x", "Novel marker X", "membrane", "immune marker")]
    assert registration_calls[0]["shape"] == (1, 1024)
    assert registration_calls[0]["dtype"] == np.dtype("float32")
    assert registration_calls[0]["csv"].splitlines()[0] == ",".join(
        KRONOS2_ADDITIONAL_MARKER_COLUMNS
    )
    assert marker_builder.cache_dir == original_cache
    assert not temporary_dirs[0].exists()
    assert summary["matched"] == ["Novel-X", "DAPI"]
    assert summary["preferred_dapi"] == "DAPI"
    assert summary["novel_markers"] == {
        "upstream_registration_available": True,
        "enabled": True,
        "content_sha256": contract["content_sha256"],
        "markers": ["Novel-X"],
        "registration": KRONOS2_ADDITIONAL_MARKER_REGISTRATION_CONTRACT,
    }
    assert emb._raw2features_multiplex_source_indices == (2, 0)
    full_patch = np.stack(
        [
            np.full((16, 16), 10, dtype=np.uint16),
            np.full((16, 16), 20, dtype=np.uint16),
            np.full((16, 16), 30, dtype=np.uint16),
        ],
        axis=2,
    )
    selected = emb.select_multiplex_channels([full_patch])[0]
    assert selected[0, 0].tolist() == [30, 10]

    # The same semantic contract is a true no-op on a warm model.
    second = emb.bind_multiplex_panel(
        ["DAPI", "CD3", "Novel-X"],
        selected_channels=selection,
        model_params=params,
    )
    assert second["matched"] == ["Novel-X", "DAPI"]
    assert len(encoder_inits) == len(registration_calls) == 1

    # Removing the parameter also removes the novel marker from the active
    # raw2features vocabulary even though upstream buffers remain registered.
    with pytest.raises(ValueError, match="Novel-X"):
        emb.bind_multiplex_panel(
            ["Novel-X"],
            model_params={},
        )

    changed = _additional_contract(
        tmp_path,
        mean=0.3,
        filename="changed.csv",
    )
    with pytest.raises(ValueError, match="cannot be reused with different contract"):
        emb.bind_multiplex_panel(
            ["Novel-X"],
            model_params={"additional_markers": changed},
        )


@pytest.mark.parametrize(
    ("marker_name", "compartment", "family", "message"),
    [
        ("CD-3", "membrane", "immune_marker", "collides"),
        ("Novel-X", "cytoplasm", "immune_marker", "compartment"),
        ("Novel-X", "membrane", "unknown_family", "family"),
    ],
)
def test_novel_contract_collisions_and_categories_fail_before_biolink_execution(
    monkeypatch,
    tmp_path,
    marker_name,
    compartment,
    family,
    message,
):
    import raw2features.embedders.kronos2_embedder as module

    contract = _additional_contract(
        tmp_path,
        marker_name=marker_name,
        compartment=compartment,
        family=family,
    )
    emb = _panel_only_embedder()
    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "BioLinkBERT resolved before marker metadata validation"
        ),
    )

    with pytest.raises(ValueError, match=message):
        emb.bind_multiplex_panel(
            [marker_name],
            model_params={"additional_markers": contract},
        )


def test_only_published_dapi_or_draq5_gets_the_preferred_dapi_override():
    emb = _panel_only_embedder()
    emb._metadata_index[_marker_match_key("NovelDNA")] = _record(
        "NovelDNA", pretraining=False, family="dna_stain"
    )
    emb._binding_native_panel = True
    try:
        summary = emb.set_panel(["NovelDNA"])
    finally:
        emb._binding_native_panel = False

    assert summary["preferred_dapi"] is None


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.dtype("uint8"), 255.0),
        (np.dtype("uint16"), 65535.0),
        (np.dtype("uint32"), 65535.0),
        (np.dtype("float32"), 400.0),
        (np.dtype("float64"), 400.0),
    ],
)
def test_kronos2_uses_the_upstream_dtype_divisor(dtype, expected):
    assert _scaling_factor(dtype) == expected


def test_kronos2_rejects_a_dtype_without_published_scaling():
    with pytest.raises(ValueError, match="no published input scaling"):
        _scaling_factor(np.dtype("int16"))


def test_transform_delegates_zscore_to_upstream_with_canonical_markers():
    torch = pytest.importorskip("torch")
    emb = Kronos2Embedder(get_spec("kronos2"))
    emb._metadata_index = {
        _marker_match_key("CD3"): _record("CD3"),
        _marker_match_key("DRAQ5"): _record(
            "DRAQ5", pretraining=False, family="dna_stain"
        ),
    }
    emb.set_panel(["CD-3", "DRAQ-5"])
    calls = []

    class Model:
        def preprocess(self, patches, names, *, preferred_dapi):
            calls.append((patches.copy(), list(names), preferred_dapi))
            return patches + np.float32(1.0)

    emb._model = Model()
    patches = [
        np.stack(
            [
                np.full((16, 16), 32767, dtype=np.uint16),
                np.full((16, 16), 65535, dtype=np.uint16),
            ],
            axis=2,
        )
    ]
    output = emb.transform_batch(patches, "cpu")

    scaled, names, preferred_dapi = calls[0]
    assert scaled.dtype == np.float32
    assert tuple(scaled.shape) == (1, 2, 16, 16)
    assert np.allclose(scaled[0, 0], np.float32(32767 / 65535))
    assert np.allclose(scaled[0, 1], 1.0)
    assert names == ["CD3", "DRAQ5"]
    assert preferred_dapi == "DRAQ5"
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert np.allclose(output.numpy(), scaled + 1.0)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (np.zeros((16, 16), dtype=np.uint16), "must be HWC"),
        (np.zeros((16, 32, 1), dtype=np.uint16), "must be square"),
        (np.zeros((15, 15, 1), dtype=np.uint16), "divisible by"),
        (np.zeros((16, 16, 2), dtype=np.uint16), "bound selected panel"),
        (
            np.full((16, 16, 1), np.nan, dtype=np.float32),
            "non-finite float",
        ),
    ],
)
def test_transform_rejects_invalid_native_patch_contract(patch, message):
    emb = Kronos2Embedder(get_spec("kronos2"))
    emb._metadata_index = {_marker_match_key("CD3"): _record("CD3")}
    emb.set_panel(["CD3"])
    emb._model = type(
        "Model",
        (),
        {"preprocess": staticmethod(lambda patches, names, *, preferred_dapi: patches)},
    )()

    with pytest.raises(ValueError, match=message):
        emb.transform_batch([patch], "cuda")


@pytest.mark.parametrize("n_rows", [1, 7, 8, 9])
def test_forward_uses_padded_fixed_eight_row_microbatches(n_rows):
    torch = pytest.importorskip("torch")
    emb = Kronos2Embedder(get_spec("kronos2"))
    emb._metadata_index = {_marker_match_key("CD3"): _record("CD3")}
    emb.set_panel(["CD3"])
    emb._device = "cpu"
    calls = []

    class Model:
        def __call__(self, batch, marker_names):
            calls.append(
                {
                    "shape": tuple(batch.shape),
                    "rows": batch[:, 0, 0, 0].detach().cpu().tolist(),
                    "marker_names": list(marker_names),
                }
            )
            values = batch[:, 0, 0, 0].reshape(8, 1)
            return values.repeat(1, emb.embedding_dim)

    emb._model = Model()
    batch = (
        torch.arange(n_rows, dtype=torch.float32)
        .reshape(n_rows, 1, 1, 1)
        .expand(n_rows, 1, 16, 16)
        .clone()
    )

    output = emb.embed_batch(batch)

    assert emb.max_batch_size == 8
    assert tuple(output.shape) == (n_rows, emb.embedding_dim)
    assert output[:, 0].tolist() == list(range(n_rows))
    assert all(call["shape"] == (8, 1, 16, 16) for call in calls)
    assert all(call["marker_names"] == ["CD3"] for call in calls)
    assert len(calls) == (n_rows + 7) // 8
    assert all(len(call["rows"]) == 8 for call in calls)
    if n_rows % 8:
        final_rows = calls[-1]["rows"]
        valid = n_rows % 8
        assert final_rows[valid:] == [final_rows[valid - 1]] * (8 - valid)


@pytest.mark.parametrize(
    ("system", "machine", "implementation", "python_version", "version", "expected"),
    [
        ("Linux", "x86_64", "CPython", (3, 11), KRONOS2_XFORMERS_VERSION, True),
        ("Linux", "AMD64", "CPython", (3, 12), KRONOS2_XFORMERS_VERSION, True),
        ("Linux", "x86_64", "CPython", (3, 13), KRONOS2_XFORMERS_VERSION, False),
        ("Linux", "aarch64", "CPython", (3, 12), KRONOS2_XFORMERS_VERSION, False),
        ("Darwin", "x86_64", "CPython", (3, 12), KRONOS2_XFORMERS_VERSION, False),
        ("Linux", "x86_64", "PyPy", (3, 12), KRONOS2_XFORMERS_VERSION, False),
        ("Linux", "x86_64", "CPython", (3, 12), "0.0.30", False),
    ],
)
def test_only_the_pinned_reference_xformers_environment_is_allowed(
    monkeypatch, system, machine, implementation, python_version, version, expected
):
    import raw2features.embedders.kronos2_embedder as module

    monkeypatch.setattr(module.importlib_metadata, "version", lambda _name: version)

    assert (
        _allows_pinned_xformers(
            system=system,
            machine=machine,
            implementation=implementation,
            python_version=python_version,
        )
        is expected
    )


def test_forced_fallback_rejects_an_already_imported_xformers_backend(monkeypatch):
    import raw2features.embedders.kronos2_embedder as module

    monkeypatch.setitem(
        sys.modules,
        "dinov2.layers.attention",
        SimpleNamespace(XFORMERS_AVAILABLE=True),
    )
    with pytest.raises(RuntimeError, match="already-imported DINOv2"):
        module._assert_forced_fallback_is_not_preimported_with_xformers()


@pytest.mark.parametrize("snapshot_preexisting", [False, True])
def test_loader_verifies_both_artifacts_then_loads_only_the_local_snapshot(
    monkeypatch, tmp_path, snapshot_preexisting
):
    import raw2features.embedders.kronos2_embedder as module

    weight_bytes = b"pinned KRONOS2 weights"
    metadata_bytes = (
        b",marker_name,compartment,family,mean,std,pretraining\n"
        b"0,DAPI,nucleus,dna_stain,0.1,0.2,yes\n"
        b"1,DRAQ5,nucleus,dna_stain,0.1,0.2,no\n"
    )
    weights = tmp_path / "kronos2_vitb16_teacher.pth"
    metadata = tmp_path / "marker_metadata.csv"
    weights.write_bytes(weight_bytes)
    metadata.write_bytes(metadata_bytes)
    snapshot_calls = []
    load_calls = []
    backend_env = []
    events = []

    isolated_dir = tmp_path / "isolated-runtime"
    monkeypatch.setattr(
        module,
        "pinned_model_cache_dir",
        lambda source, revision: str(isolated_dir),
    )

    def snapshot_download(source, revision, *, allow_patterns, local_dir):
        snapshot_calls.append((source, revision, allow_patterns, local_dir))
        return str(tmp_path)

    monkeypatch.setattr(module, "download_pinned_hf_snapshot", snapshot_download)
    monkeypatch.setattr(module, "_allows_pinned_xformers", lambda: False)
    monkeypatch.delenv("XFORMERS_DISABLED", raising=False)
    original_verify = module.verify_sha256

    def verify(path, expected, *, what):
        events.append(("verify", Path(path).name, what))
        original_verify(path, expected, what=what)

    monkeypatch.setattr(module, "verify_sha256", verify)

    class Model:
        def float(self):
            return self

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

    transformers = ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            events.append(("load", Path(source).name, "AutoModel"))
            load_calls.append((source, kwargs))
            backend_env.append(module.os.environ.get("XFORMERS_DISABLED"))
            sys.path.insert(0, source)
            return Model()

    transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch = ModuleType("torch")
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    base = get_spec("kronos2")
    spec = replace(
        base,
        weights_sha256=_digest(weight_bytes),
        timm_kwargs={
            **base.timm_kwargs,
            "marker_metadata_sha256": _digest(metadata_bytes),
        },
    )
    emb = Kronos2Embedder(spec)
    emb._registered_additional_markers_digest = "stale"
    emb._novel_registration_failed = True
    emb._raw2features_multiplex_source_indices = (9,)
    if snapshot_preexisting:
        monkeypatch.setattr(sys, "path", [*sys.path, str(tmp_path)])
    original_sys_path = list(sys.path)
    emb.load(device="cpu")

    assert snapshot_calls == [
        (
            spec.source,
            spec.weights_revision,
            KRONOS2_SNAPSHOT_ALLOW_PATTERNS,
            str(isolated_dir),
        )
    ]
    assert events == [
        ("verify", "kronos2_vitb16_teacher.pth", "kronos2"),
        ("verify", "marker_metadata.csv", "kronos2 marker metadata"),
        ("load", tmp_path.name, "AutoModel"),
    ]
    assert load_calls == [
        (
            str(tmp_path),
            {
                "trust_remote_code": True,
                "local_files_only": True,
                "device": "cpu",
            },
        )
    ]
    assert emb._dtype is torch.float32
    assert set(emb._metadata_index) == {"dapi", "draq5"}
    assert not hasattr(emb, "_registered_additional_markers_digest")
    assert not hasattr(emb, "_novel_registration_failed")
    assert not hasattr(emb, "_raw2features_multiplex_source_indices")
    assert sys.path == original_sys_path
    assert backend_env == ["1"]
    assert "XFORMERS_DISABLED" not in module.os.environ


def test_loader_rejects_stale_unrequested_snapshot_code_before_execution(
    monkeypatch, tmp_path
):
    import raw2features.embedders.kronos2_embedder as module

    weight_bytes = b"pinned KRONOS2 weights"
    metadata_bytes = (
        b",marker_name,compartment,family,mean,std,pretraining\n"
        b"0,DAPI,nucleus,dna_stain,0.1,0.2,yes\n"
    )
    (tmp_path / "kronos2_vitb16_teacher.pth").write_bytes(weight_bytes)
    (tmp_path / "marker_metadata.csv").write_bytes(metadata_bytes)
    (tmp_path / "test.py").write_text(
        "raise AssertionError('must never execute')\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        module,
        "pinned_model_cache_dir",
        lambda *_args: str(tmp_path / "isolated-runtime"),
    )
    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda *_args, **_kwargs: str(tmp_path),
    )

    transformers = ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            pytest.fail("unexpected snapshot code reached AutoModel")

    transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    torch = ModuleType("torch")
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)

    base = get_spec("kronos2")
    spec = replace(
        base,
        weights_sha256=_digest(weight_bytes),
        timm_kwargs={
            **base.timm_kwargs,
            "marker_metadata_sha256": _digest(metadata_bytes),
        },
    )
    with pytest.raises(ValueError, match=r"unexpected.*test\.py"):
        Kronos2Embedder(spec).load(device="cpu")


def test_loader_rejects_compile_before_import_or_download(monkeypatch):
    import raw2features.embedders.kronos2_embedder as module

    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "KRONOS2 downloaded a snapshot before rejecting compile"
        ),
    )

    with pytest.raises(ValueError, match="does not support --compile"):
        Kronos2Embedder(get_spec("kronos2")).load(device="cpu", compile=True)


def test_loader_refuses_bad_marker_metadata_before_custom_code_executes(
    monkeypatch, tmp_path
):
    import raw2features.embedders.kronos2_embedder as module

    weights = tmp_path / "kronos2_vitb16_teacher.pth"
    metadata = tmp_path / "marker_metadata.csv"
    weights.write_bytes(b"weights")
    metadata.write_text(
        ",marker_name,compartment,family,mean,std,pretraining\n"
        "0,DAPI,nucleus,dna_stain,0.1,0.2,yes\n"
    )
    monkeypatch.setattr(
        module,
        "download_pinned_hf_snapshot",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    torch = ModuleType("torch")
    torch.float32 = object()
    monkeypatch.setitem(sys.modules, "torch", torch)
    transformers = ModuleType("transformers")

    class AutoModel:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            pytest.fail("custom model code ran before marker metadata verification")

    transformers.AutoModel = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    base = get_spec("kronos2")
    spec = replace(
        base,
        weights_sha256=_digest(b"weights"),
        timm_kwargs={**base.timm_kwargs, "marker_metadata_sha256": "0" * 64},
    )

    with pytest.raises(ValueError, match="marker metadata"):
        Kronos2Embedder(spec).load(device="cpu")
