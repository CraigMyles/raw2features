"""Unit tests for the weight-pin helpers (``embedders/_hub.py``)."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType

import pytest

from raw2features.embedders._hub import (
    download_pinned_hf_file,
    download_pinned_hf_snapshot,
    hf_repo_id,
    pin_source,
    pinned_model_cache_dir,
    verify_sha256,
)


def test_hf_repo_id_accepts_both_registry_prefixes():
    assert hf_repo_id("hf-hub:owner/repo") == "owner/repo"
    assert hf_repo_id("hf_hub:owner/repo") == "owner/repo"
    with pytest.raises(ValueError, match="expected an hf-hub"):
        hf_repo_id("owner/repo")


def test_pinned_model_cache_dir_is_app_owned_and_revision_specific(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = pinned_model_cache_dir("hf-hub:owner/model", "abc123")
    assert path == str(tmp_path / "raw2features" / "models" / "owner--model" / "abc123")


def test_download_pinned_hf_file_threads_revision(monkeypatch):
    calls = []
    hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/model.bin"

    hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    path = download_pinned_hf_file(
        "hf-hub:owner/repo", "model.bin", "abc123", cache_dir="/cache"
    )
    assert path == "/cache/model.bin"
    assert calls == [
        {
            "repo_id": "owner/repo",
            "filename": "model.bin",
            "revision": "abc123",
            "cache_dir": "/cache",
        }
    ]


def test_download_pinned_hf_snapshot_threads_revision(monkeypatch):
    calls = []
    hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        return "/cache/snapshot"

    hub.snapshot_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    path = download_pinned_hf_snapshot(
        "hf_hub:owner/repo", "abc123", allow_patterns=("config.json", "*.bin")
    )
    assert path == "/cache/snapshot"
    assert calls == [
        {
            "repo_id": "owner/repo",
            "revision": "abc123",
            "allow_patterns": ("config.json", "*.bin"),
        }
    ]


@pytest.mark.parametrize("kind", ["file", "snapshot"])
def test_pinned_hf_download_requires_recorded_revision(kind):
    with pytest.raises(ValueError, match="no weights_revision"):
        if kind == "file":
            download_pinned_hf_file("hf-hub:owner/repo", "model.bin", None)
        else:
            download_pinned_hf_snapshot("hf-hub:owner/repo", None)


def test_pinned_model_cache_dir_requires_recorded_revision():
    with pytest.raises(ValueError, match="no weights_revision"):
        pinned_model_cache_dir("hf-hub:owner/repo", None)


def test_pin_source_appends_revision_for_hf_hub():
    assert pin_source("hf-hub:owner/repo", "abc123") == "hf-hub:owner/repo@abc123"
    assert pin_source("hf_hub:owner/repo", "deadbeef") == "hf_hub:owner/repo@deadbeef"


def test_pin_source_passthrough():
    # No revision recorded -> unchanged.
    assert pin_source("hf-hub:owner/repo", None) == "hf-hub:owner/repo"
    assert pin_source("hf-hub:owner/repo", "") == "hf-hub:owner/repo"
    # Non-hub sources (torchvision URI, bare arch name) -> unchanged.
    src = "torchvision://resnet50?weights=IMAGENET1K_V2"
    assert pin_source(src, "IMAGENET1K_V2") == src
    assert pin_source("vit_large_patch14_dinov2.lvd142m", "abc") == (
        "vit_large_patch14_dinov2.lvd142m"
    )
    # Already pinned -> not double-pinned.
    assert pin_source("hf-hub:owner/repo@x", "y") == "hf-hub:owner/repo@x"


def test_verify_sha256_matches(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"some weight bytes")
    digest = hashlib.sha256(b"some weight bytes").hexdigest()
    verify_sha256(str(p), digest, what="testmodel")  # no raise


def test_verify_sha256_mismatch_raises(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match the pinned"):
        verify_sha256(str(p), "0" * 64, what="testmodel")


def test_verify_sha256_none_is_noop(tmp_path):
    # Nothing recorded to check against (e.g. torchvision-managed weights).
    verify_sha256(str(tmp_path / "missing.bin"), None, what="resnet50")
