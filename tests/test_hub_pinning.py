"""Unit tests for the weight-pin helpers (``embedders/_hub.py``)."""

from __future__ import annotations

import hashlib

import pytest

from raw2features.embedders._hub import pin_source, verify_sha256


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
