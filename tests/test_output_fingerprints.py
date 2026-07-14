"""Per-output model contracts: identity, resume, and replacement safety."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import MockEmbedder
from raw2features.core.store import open_grid
from raw2features.embedders.base import ModelSpec
from raw2features.embedders.fingerprint import (
    expected_patch_outputs,
    patch_output_fingerprint,
    slide_output_fingerprint,
    valid_output_fingerprint,
)
from raw2features.pipeline.receipt import validate_model, validate_store_models
from raw2features.pipeline.runner import RunConfig, embed_slide, run_slide
from raw2features.sinks.zarr_sink import ZarrSink
from raw2features.slide_embedders.encoding import slide_embedding_is_complete

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _spec(**overrides) -> ModelSpec:
    values = {
        "name": "contract_model",
        "family": "timm",
        "source": "hf-hub:example/model",
        "embedding_dim": 8,
        "input_size": 224,
        "pooling": "cls",
        "mean": (0.1, 0.2, 0.3),
        "std": (0.4, 0.5, 0.6),
        "interpolation": "bicubic",
        "transform_source_url": "https://example.org/model",
        "license": "MIT",
        "gated": False,
        "transform_source": "registry",
        "inference_amp": "bf16",
        "reg_tokens": 0,
        "modality": "brightfield",
        "timm_kwargs": {"num_classes": 0},
        "checkpoint": {
            "repo": "example/weights",
            "filename": "model.pth",
            "state_dict_key": "teacher",
            "strip_prefixes": ["module."],
            "strict": False,
        },
        "weights_revision": "1" * 40,
        "weights_sha256": "a" * 64,
        "weights_filename": "model.pth",
    }
    values.update(overrides)
    return ModelSpec(**values)


def test_patch_fingerprint_is_canonical_and_covers_the_output_contract(monkeypatch):
    import raw2features.embedders.fingerprint as module

    base = _spec()
    original = patch_output_fingerprint(base, "bf16")
    assert valid_output_fingerprint(original)
    assert len(original["digest"]) == 64
    assert patch_output_fingerprint(base, "bf16") == original
    assert original["payload"]["checkpoint"]["effective"] == {
        "repo": "example/weights",
        "filename": "model.pth",
        "url": None,
        "mechanism": "explicit_checkpoint",
    }

    variants = [
        replace(base, weights_revision="2" * 40),
        replace(base, weights_sha256="b" * 64),
        replace(base, weights_filename="other-weights.pth"),
        replace(base, checkpoint={**base.checkpoint, "repo": "other/repo"}),
        replace(base, checkpoint={**base.checkpoint, "filename": "other.pth"}),
        replace(base, mean=(0.11, 0.2, 0.3)),
        replace(base, std=(0.41, 0.5, 0.6)),
        replace(base, interpolation="bilinear"),
        replace(base, input_size=256),
        replace(base, pooling="pooled"),
        replace(base, embedding_dim=16),
        replace(base, family="transformers"),
        replace(base, timm_kwargs={"num_classes": 0, "dynamic_img_size": True}),
        replace(base, reg_tokens=4),
        replace(base, modality="multiplex"),
    ]
    assert all(
        patch_output_fingerprint(candidate, "bf16")["digest"] != original["digest"]
        for candidate in variants
    )
    assert patch_output_fingerprint(base, "fp32")["digest"] != original["digest"]

    monkeypatch.setattr(module, "PATCH_LOADER_CONTRACT_VERSION", 2)
    assert patch_output_fingerprint(base, "bf16")["digest"] != original["digest"]


def test_fingerprint_never_persists_uri_credentials_and_ignores_rotation():
    first = _spec(
        checkpoint={
            "url": "https://weights.example/model.pt?part=1&X-Amz-Signature=SECRET_A",
            "filename": "model.pt",
        }
    )
    second = replace(
        first,
        checkpoint={
            **first.checkpoint,
            "url": "https://weights.example/model.pt?part=1&X-Amz-Signature=SECRET_B",
        },
    )
    left = patch_output_fingerprint(first, "fp32")
    right = patch_output_fingerprint(second, "fp32")
    assert left == right
    assert "SECRET" not in repr(left)
    assert left["payload"]["checkpoint"]["effective"]["url"] == (
        "https://weights.example/model.pt?part=1"
    )
    leaked_payload = {
        **left["payload"],
        "checkpoint": {
            **left["payload"]["checkpoint"],
            "effective": {
                **left["payload"]["checkpoint"]["effective"],
                "url": "https://weights.example/model.pt?token=DO_NOT_PERSIST",
            },
        },
    }
    assert not valid_output_fingerprint({**left, "payload": leaked_payload})


def test_auto_amp_resolves_and_grid_hash_stays_geometry_only():
    spec = _spec(name="x", inference_amp="bf16")
    automatic = expected_patch_outputs(["x"], "auto", specs={"x": spec})["x"]
    explicit = expected_patch_outputs(["x"], "bf16", specs={"x": spec})["x"]
    fp32 = expected_patch_outputs(["x"], "fp32", specs={"x": spec})["x"]
    cpu_auto = expected_patch_outputs(
        ["x"], "auto", "cpu", specs={"x": spec}
    )["x"]
    assert automatic == explicit
    assert automatic["output_fingerprint"] != fp32["output_fingerprint"]
    assert cpu_auto == fp32

    common = dict(target_mpp=0.5, patch_px=224, amp="auto")
    assert RunConfig(models=["x"], **common).grid_hash() == RunConfig(
        models=["another"], **common
    ).grid_hash()
    assert RunConfig(models=["x"], **common).grid_hash() == RunConfig(
        models=["x"], target_mpp=0.5, patch_px=224, amp="fp32"
    ).grid_hash()
    legacy_auto = RunConfig(models=["x"], **common).legacy_grid_hash()
    explicit_cfg = RunConfig(
        models=["x"], target_mpp=0.5, patch_px=224, amp="fp32"
    )
    assert legacy_auto in explicit_cfg.compatible_legacy_grid_hashes()


@pytest.mark.parametrize(
    ("name", "base_repo"),
    [("seal_conch", "MahmoodLab/conch"), ("seal_univ2", "MahmoodLab/UNI2-h")],
)
def test_seal_fingerprint_is_an_honest_experimental_composite(name, base_repo):
    from raw2features.embedders.model_registry import get_spec

    spec = get_spec(name)
    payload = patch_output_fingerprint(spec, "fp32")["payload"]
    composite = payload["composite"]
    assert spec.experimental is True
    assert composite["experimental"] is True
    assert composite["adapter"] == {
        "repo": "MahmoodLab/SEAL",
        "filename": f"seal_{spec.source}_vision.pth",
        "revision": spec.weights_revision,
        "sha256": spec.weights_sha256,
    }
    assert composite["base"]["repo"] == base_repo
    assert composite["base"]["revision"] is None
    assert composite["base"]["sha256"] is None
    assert composite["base"]["integrity"] == "upstream_factory_unpinned"


def test_every_registry_model_has_a_serializable_output_contract():
    from raw2features.embedders.model_registry import load_registry
    from raw2features.slide_embedders.model_registry import load_slide_registry

    patch_fingerprints = {}
    for name, spec in load_registry().items():
        fingerprint = patch_output_fingerprint(spec, spec.inference_amp)
        assert valid_output_fingerprint(fingerprint), name
        assert fingerprint["payload"]["checkpoint"]["effective"]["filename"], name
        patch_fingerprints[name] = fingerprint

    fallback = patch_output_fingerprint(_spec(name="fallback"), "fp32")
    for name, spec in load_slide_registry().items():
        patch_name = spec.patch_encoder if spec.patch_encoder != "any" else "fallback"
        patch_dim = spec.patch_dim if spec.patch_dim > 0 else 8
        fingerprint = slide_output_fingerprint(
            spec,
            patch_model=patch_name,
            patch_output_fingerprint=patch_fingerprints.get(patch_name, fallback),
            patch_dim=patch_dim,
            resolved_amp="fp32",
        )
        assert valid_output_fingerprint(fingerprint), name
        effective = fingerprint["payload"]["checkpoint"]["effective"]
        if spec.family != "pool":
            assert effective["filename"], name


def test_slide_fingerprint_records_device_resolved_precision():
    from raw2features.embedders.fingerprint import resolved_slide_amp
    from raw2features.slide_embedders.model_registry import get_slide_spec

    spec = get_slide_spec("titan")
    patch = patch_output_fingerprint(_spec(name="conch_v1_5"), "fp16")
    cpu = slide_output_fingerprint(
        spec,
        patch_model="conch_v1_5",
        patch_output_fingerprint=patch,
        patch_dim=768,
        resolved_amp=resolved_slide_amp(spec, "cpu"),
    )
    cuda = slide_output_fingerprint(
        spec,
        patch_model="conch_v1_5",
        patch_output_fingerprint=patch,
        patch_dim=768,
        resolved_amp=resolved_slide_amp(spec, "cuda:0"),
    )
    assert cpu["payload"]["output"]["resolved_amp"] == "fp32"
    assert cuda["payload"]["output"]["resolved_amp"] == "fp16"
    assert cpu["digest"] != cuda["digest"]


def test_validate_model_requires_current_fingerprint_and_expected_dimension(tmp_path):
    import zarr

    spec = _spec(name="m", embedding_dim=3)
    fingerprint = patch_output_fingerprint(spec, "fp32")
    group = zarr.open_group(str(tmp_path / "g.zarr"), mode="w", zarr_format=2)
    features = group.create_group("features")
    array = features.create_array("m", shape=(4, 3), dtype="float32")
    array[:] = 1
    array.attrs["output_fingerprint"] = fingerprint
    group.attrs["raw2features"] = {
        "models": {"m": {"embedding_dim": 3, "output_fingerprint": fingerprint}}
    }

    assert validate_model(
        group,
        "m",
        4,
        expected_dim=3,
        expected_fingerprint=fingerprint,
    )
    assert not validate_model(
        group,
        "m",
        4,
        expected_dim=99,
        expected_fingerprint=fingerprint,
    )

    del array.attrs["output_fingerprint"]
    assert not validate_model(
        group,
        "m",
        4,
        expected_dim=3,
        expected_fingerprint=fingerprint,
    )
    # A current-looking legacy header must never be used to reconstruct the marker.
    stored = group.attrs["raw2features"]["models"]["m"]["output_fingerprint"]
    assert stored == fingerprint


def test_array_fingerprint_is_a_post_write_commit_marker(tmp_path):
    spec = _spec(name="m", embedding_dim=3)
    fingerprint = patch_output_fingerprint(spec, "fp32")
    contract = {"embedding_dim": 3, "output_fingerprint": fingerprint}
    model_meta = {
        "m": {"embedding_dim": 3, "output_fingerprint": fingerprint}
    }
    coords = np.arange(8, dtype=np.int32).reshape(4, 2)
    sink = ZarrSink()
    sink.create(
        str(tmp_path),
        "s",
        grid="mpp1_px224",
        n_patches=4,
        coords=coords,
        grid_index=coords,
        grid_tissue=None,
        model_dims={"m": 3},
        header={"schema_version": "0.1", "models": model_meta},
    )
    sink.write_block("m", 0, np.ones((4, 3), np.float32))
    sink.finalize_models({"m": contract})
    sink.close()
    path = tmp_path / "s.embeddings.zarr"
    assert validate_model(
        open_grid(str(path)),
        "m",
        4,
        expected_dim=3,
        expected_fingerprint=fingerprint,
    )

    # Simulate a contract-invalid model being prepared for replacement, followed by
    # a process death before any rows/finalization. The old commit marker is gone.
    replacement = ZarrSink()
    replacement.open_append(
        str(tmp_path),
        "s",
        new_model_dims={"m": 3},
        new_model_meta=model_meta,
        replace_models=["m"],
    )
    array = replacement._group["features"]["m"]
    assert "output_fingerprint" not in dict(array.attrs)
    assert not validate_model(
        replacement._group,
        "m",
        4,
        expected_dim=3,
        expected_fingerprint=fingerprint,
    )


def _pinned_mock(*, name="mock", dim=8, bias=0.0, revision="1") -> MockEmbedder:
    embedder = MockEmbedder(name=name, dim=dim, bias=bias)
    embedder.spec = replace(
        embedder.spec,
        weights_revision=revision * 40,
        weights_sha256=revision * 64,
    )
    return embedder


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_loaded_precision_must_match_the_contract_before_writing():
    import torch

    from raw2features.pipeline.runner import _assert_loaded_model_contracts

    embedder = _pinned_mock()
    embedder._device = "cuda:0"
    embedder._dtype = torch.float32
    expected = expected_patch_outputs(
        ["mock"],
        "bf16",
        "cuda:0",
        specs={"mock": embedder.spec},
    )
    with pytest.raises(ValueError, match="effective AMP"):
        _assert_loaded_model_contracts([embedder], expected)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_high_level_embed_accepts_a_custom_multidevice_factory(
    synthetic_ngff, tmp_path
):
    calls = []

    def factory(device):
        calls.append(device)
        return [_pinned_mock(name="custom").load(device=device)]

    result = embed_slide(
        synthetic_ngff,
        str(tmp_path / "out"),
        RunConfig(
            models=["custom"],
            no_seg=True,
            device="cpu",
            devices="cpu,cpu",
            amp="fp32",
            batch_size=2,
        ),
        requested_mpp=0.5,
        requested_patch_px=64,
        embedder_factory=factory,
    )
    group = open_grid(result["output_uri"])
    assert group["features"]["custom"].shape[1] == 8
    assert len(calls) >= 4  # high-level probe, header copy, and two workers


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_multidevice_contract_uses_actual_worker_devices(
    synthetic_ngff, tmp_path, monkeypatch
):
    monkeypatch.setattr("raw2features.core.device._accelerators", lambda: (True, False))
    calls = []

    def factory(device):
        calls.append(device)
        embedder = _pinned_mock(name="custom")
        embedder.spec = replace(embedder.spec, inference_amp="fp16")
        return [embedder.load(device=device)]

    result = embed_slide(
        synthetic_ngff,
        str(tmp_path / "out"),
        RunConfig(
            models=["custom"],
            no_seg=True,
            device="auto",
            devices="cpu,cpu",
            amp="auto",
            batch_size=2,
        ),
        requested_mpp=0.5,
        requested_patch_px=64,
        embedder_factory=factory,
    )

    group = open_grid(result["output_uri"])
    fingerprint = dict(group["features"]["custom"].attrs)["output_fingerprint"]
    assert fingerprint["payload"]["output"]["resolved_amp"] == "fp32"
    assert calls[0] == "cpu"


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_multidevice_rejects_different_output_contracts_before_store_creation(
    synthetic_ngff, tmp_path, monkeypatch
):
    monkeypatch.setattr("raw2features.core.device._accelerators", lambda: (True, False))
    out = tmp_path / "out"
    cfg = RunConfig(
        models=["virchow2"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        devices="cuda:0,cpu",
        amp="auto",
    )

    with pytest.raises(
        ValueError,
        match=r"virchow2 \(cuda:0=fp16, cpu=fp32\)",
    ):
        run_slide(synthetic_ngff, str(out), cfg)

    assert not (out / "synthetic.embeddings.zarr").exists()


def test_multidevice_allows_heterogeneous_devices_with_one_contract():
    from raw2features.pipeline.runner import _expected_contracts_for_devices

    contracts = _expected_contracts_for_devices(
        RunConfig(models=["virchow2"], amp="fp32"),
        None,
        ["cuda:0", "cpu"],
    )

    fingerprint = contracts["virchow2"]["output_fingerprint"]
    assert fingerprint["payload"]["output"]["resolved_amp"] == "fp32"


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_fingerprint_change_replaces_only_that_model_and_invalidates_slide(
    synthetic_ngff, tmp_path
):
    out = str(tmp_path / "out")
    common = dict(no_seg=True, target_mpp=0.5, patch_px=64, device="cpu", amp="fp32")
    first_cfg = RunConfig(models=["a"], slide_encoders=["mean"], **common)
    first = run_slide(
        synthetic_ngff,
        out,
        first_cfg,
        embedders=[_pinned_mock(name="a", bias=1.0, revision="1")],
    )
    path = first["output_uri"].removeprefix("file://")
    run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["b"], **common),
        embedders=[_pinned_mock(name="b", dim=5, bias=2.0, revision="1")],
    )
    group = open_grid(path)
    coords_before = np.asarray(group["coords"][:]).copy()
    a_before = np.asarray(group["features"]["a"][:]).copy()
    b_before = np.asarray(group["features"]["b"][:]).copy()
    b_attrs_before = dict(group["features"]["b"].attrs)
    assert "mean" in group["slide"]

    second = run_slide(
        synthetic_ngff,
        out,
        RunConfig(models=["a"], **common),
        embedders=[_pinned_mock(name="a", bias=9.0, revision="2")],
    )
    assert second["models_added"] == ["a"]
    group = open_grid(path)
    np.testing.assert_array_equal(group["coords"][:], coords_before)
    assert not np.array_equal(group["features"]["a"][:], a_before)
    np.testing.assert_array_equal(group["features"]["b"][:], b_before)
    assert dict(group["features"]["b"].attrs) == b_attrs_before
    assert "slide" not in group or "mean" not in group["slide"]


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_wrong_dimension_array_is_replaced_on_resume(synthetic_ngff, tmp_path):
    out = str(tmp_path / "out")
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    first = run_slide(
        synthetic_ngff,
        out,
        cfg,
        embedders=[_pinned_mock(dim=8, revision="1")],
    )
    path = first["output_uri"].removeprefix("file://")
    coords = np.asarray(open_grid(path)["coords"][:]).copy()

    resumed = run_slide(
        synthetic_ngff,
        out,
        cfg,
        embedders=[_pinned_mock(dim=5, revision="1")],
    )
    group = open_grid(path)
    assert resumed["models_added"] == ["mock"]
    assert group["features"]["mock"].shape == (coords.shape[0], 5)
    np.testing.assert_array_equal(group["coords"][:], coords)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_receipt_and_legacy_marker_cannot_skip_a_recompute(
    synthetic_ngff, tmp_path
):
    out = str(tmp_path / "out")
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    first = run_slide(
        synthetic_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[_pinned_mock(bias=1.0, revision="1")],
    )
    path = first["output_uri"].removeprefix("file://")
    before = np.asarray(open_grid(path)["features"]["mock"][:]).copy()

    # Same config hash, different model contract: the receipt fast path must fail.
    changed = run_slide(
        synthetic_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[_pinned_mock(bias=4.0, revision="2")],
    )
    assert changed["status"] == "complete"
    assert changed["models_added"] == ["mock"]
    assert not np.array_equal(open_grid(path)["features"]["mock"][:], before)

    # Even with a matching current header, removing the concrete array commit marker
    # makes this a legacy/unknown output and forces one more recompute.
    import zarr

    # Preserve the old consolidated marker, then remove only the live marker to
    # simulate a replacement crash before close() refreshes consolidated metadata.
    zarr.consolidate_metadata(path)
    root = zarr.open_group(path, mode="r+", use_consolidated=False)
    group = open_grid(root)
    del group["features"]["mock"].attrs["output_fingerprint"]
    previous = np.asarray(group["features"]["mock"][:]).copy()
    expected = expected_patch_outputs(
        ["mock"],
        "fp32",
        "cpu",
        specs={"mock": _pinned_mock(revision="2").spec},
    )
    assert not validate_store_models(
        f"file://{path}",
        ["mock"],
        expected_model_contracts=expected,
    )
    repaired = run_slide(
        synthetic_ngff,
        out,
        cfg,
        receipts_dir=receipts,
        embedders=[_pinned_mock(bias=8.0, revision="2")],
    )
    assert repaired["models_added"] == ["mock"]
    assert not np.array_equal(open_grid(path)["features"]["mock"][:], previous)


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_slide_completion_is_bound_to_the_patch_output_fingerprint(
    synthetic_ngff, tmp_path
):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        slide_encoders=["mean"],
    )
    summary = run_slide(
        synthetic_ngff,
        str(tmp_path / "out"),
        cfg,
        embedders=[_pinned_mock(revision="1")],
    )
    group = open_grid(summary["output_uri"], mode="r+")
    assert slide_embedding_is_complete(group, "mean", patch_model="mock")

    changed_spec = replace(_pinned_mock(revision="2").spec, name="mock")
    changed_fingerprint = patch_output_fingerprint(changed_spec, "fp32")
    group["features"]["mock"].attrs["output_fingerprint"] = changed_fingerprint
    header = dict(group.attrs["raw2features"])
    header["models"]["mock"]["output_fingerprint"] = changed_fingerprint
    group.attrs["raw2features"] = header
    assert not slide_embedding_is_complete(group, "mean", patch_model="mock")
