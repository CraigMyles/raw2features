"""Live validation that every Hugging Face registry pin exists.

Loader-contract tests mock the Hub so they can stay fast and deterministic.  This
separate network check catches a different failure mode: a well-formed but nonexistent
``weights_revision`` that would 404 only when a real loader enforces it.

Run just this check in a model-enabled environment with::

    pytest -q -m network tests/test_registry_revisions_network.py
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import pytest

from raw2features.embedders._hub import download_pinned_hf_file, verify_sha256
from raw2features.embedders.model_registry import load_registry
from raw2features.embedders.open_clip_embedder import (
    _BIOMEDCLIP_TEXT_REVISION,
    _BIOMEDCLIP_TEXT_SOURCE,
)
from raw2features.embedders.seal_embedder import _SEAL_REPO
from raw2features.slide_embedders.model_registry import load_slide_registry

_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class _HubPin:
    repo: str
    revision: str
    entries: tuple[str, ...]


def _hf_repo(spec) -> str | None:
    """Return the HF repository that owns ``spec.weights_revision``, if any."""
    checkpoint = getattr(spec, "checkpoint", None) or {}
    if checkpoint.get("repo"):
        return str(checkpoint["repo"])

    source = str(spec.source)
    for prefix in ("hf-hub:", "hf_hub:"):
        if source.startswith(prefix):
            return source.removeprefix(prefix).split("@", 1)[0]
    if spec.family in {"transformers", "clip_hf"}:
        return source
    if spec.family == "seal":
        return _SEAL_REPO
    return None


def _is_non_hf_pin(spec) -> bool:
    """Identify the registry's deliberately non-Hub revision identifiers."""
    checkpoint = getattr(spec, "checkpoint", None) or {}
    return str(spec.source).startswith("torchvision://") or (
        bool(checkpoint.get("url")) and spec.weights_revision == "pretrained-weights"
    )


def _registry_hf_pins() -> list[_HubPin]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for kind, registry in (
        ("patch", load_registry()),
        ("slide", load_slide_registry()),
    ):
        for name, spec in registry.items():
            revision = spec.weights_revision
            if not revision:
                continue
            repo = _hf_repo(spec)
            if repo is None:
                assert _is_non_hf_pin(spec), (
                    f"{kind}:{name} has weights_revision={revision!r}, but its "
                    "loader is not classified as Hugging Face or a known non-HF source"
                )
                continue
            grouped[(repo, str(revision))].append(f"{kind}:{name}")

    # BiomedCLIP constructs a text tower even though raw2features exposes only its
    # image encoder. Keep that separately recorded construction dependency live too.
    text_repo = _BIOMEDCLIP_TEXT_SOURCE.removeprefix("hf-hub:")
    grouped[(text_repo, _BIOMEDCLIP_TEXT_REVISION)].append(
        "construction:biomedclip-pubmedbert"
    )

    return [
        _HubPin(repo, revision, tuple(sorted(entries)))
        for (repo, revision), entries in sorted(grouped.items())
    ]


_HF_PINS = _registry_hf_pins()
assert _HF_PINS, "expected at least one Hugging Face registry pin"


@pytest.mark.network
@pytest.mark.parametrize(
    "pin",
    _HF_PINS,
    ids=lambda pin: f"{pin.repo}@{pin.revision[:12]}",
)
def test_huggingface_registry_revision_resolves(pin: _HubPin):
    label = ", ".join(pin.entries)
    assert _FULL_COMMIT.fullmatch(pin.revision), (
        f"{label}: weights_revision must be a full 40-character lowercase commit"
    )
    huggingface_hub = pytest.importorskip(
        "huggingface_hub", reason="install raw2features[models] for live pin validation"
    )
    info = huggingface_hub.HfApi().model_info(
        repo_id=pin.repo,
        revision=pin.revision,
        timeout=15,
        expand=["sha"],
        token=os.environ.get("HF_TOKEN") or False,
    )
    assert info.sha == pin.revision, (
        f"{label}: {pin.repo}@{pin.revision} resolved to {info.sha!r}"
    )


@pytest.mark.network
@pytest.mark.parametrize(
    ("kind", "name", "source"),
    [
        ("patch", "seal_conch", "hf-hub:MahmoodLab/SEAL"),
        ("patch", "seal_univ2", "hf-hub:MahmoodLab/SEAL"),
        ("slide", "gigapath_slide", None),
    ],
)
def test_direct_loader_file_checksum_matches_registry(kind, name, source):
    """The files newly verified before SEAL/GigaPath loader boundaries match pins."""

    registry = load_registry() if kind == "patch" else load_slide_registry()
    spec = registry[name]
    path = download_pinned_hf_file(
        source or spec.source,
        spec.weights_filename,
        spec.weights_revision,
    )
    verify_sha256(path, spec.weights_sha256, what=name)
