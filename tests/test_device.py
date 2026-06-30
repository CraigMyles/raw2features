"""Device resolution: `auto` picks the best backend; an absent explicit one errors."""

from __future__ import annotations

import os

import pytest

from raw2features.core import device as dev
from raw2features.core.device import resolve_device


def test_cpu_is_passthrough():
    assert resolve_device("cpu") == "cpu"


def test_auto_returns_a_real_backend():
    # Env-agnostic: whatever this machine has, auto resolves to a concrete backend.
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch):
    monkeypatch.setattr(dev, "_accelerators", lambda: (False, False))
    assert resolve_device("auto") == "cpu"


def test_explicit_cuda_without_cuda_raises(monkeypatch):
    monkeypatch.setattr(dev, "_accelerators", lambda: (False, False))
    with pytest.raises(ValueError, match="CUDA requested"):
        resolve_device("cuda")


def test_explicit_mps_without_mps_raises(monkeypatch):
    monkeypatch.setattr(dev, "_accelerators", lambda: (False, False))
    with pytest.raises(ValueError, match="MPS requested"):
        resolve_device("mps")


def test_auto_prefers_cuda_then_mps(monkeypatch):
    monkeypatch.setattr(dev, "_accelerators", lambda: (True, True))
    assert resolve_device("auto") == "cuda"
    monkeypatch.setattr(dev, "_accelerators", lambda: (False, True))
    assert resolve_device("auto") == "mps"


def test_cuda_index_passthrough_when_available(monkeypatch):
    monkeypatch.setattr(dev, "_accelerators", lambda: (True, False))
    assert resolve_device("cuda:1") == "cuda:1"


def test_mps_sets_cpu_fallback_env(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(dev, "_accelerators", lambda: (False, True))
    resolve_device("auto")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
