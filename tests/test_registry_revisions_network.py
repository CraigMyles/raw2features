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
from collections.abc import Mapping
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
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _HubPin:
    repo: str
    revision: str
    entries: tuple[str, ...]


@dataclass(frozen=True)
class _HubArtifactPin:
    name: str
    repo: str
    revision: str
    filename: str
    sha256: str


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


def _new_model_artifact_pins() -> list[_HubArtifactPin]:
    registry = load_registry()
    pins: list[_HubArtifactPin] = []
    for name in ("h0_mini", "keep", "openmidnight", "openpath"):
        spec = registry[name]
        repo = _hf_repo(spec)
        assert repo is not None, (
            f"patch:{name} must resolve to a Hugging Face repository"
        )
        assert spec.weights_revision is not None
        assert spec.weights_filename is not None
        assert spec.weights_sha256 is not None
        pins.append(
            _HubArtifactPin(
                name=name,
                repo=repo,
                revision=str(spec.weights_revision),
                filename=str(spec.weights_filename),
                sha256=str(spec.weights_sha256),
            )
        )
    return pins


_NEW_MODEL_ARTIFACT_PINS = _new_model_artifact_pins()


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
    "pin",
    _NEW_MODEL_ARTIFACT_PINS,
    ids=lambda pin: pin.name,
)
def test_new_model_artifact_checksum_matches_huggingface_metadata(pin: _HubArtifactPin):
    """The new multi-GB checkpoints match their recorded Hub LFS digests."""

    assert _FULL_COMMIT.fullmatch(pin.revision), (
        f"patch:{pin.name}: weights_revision must be a full commit"
    )
    assert _SHA256.fullmatch(pin.sha256), (
        f"patch:{pin.name}: weights_sha256 must be a lowercase SHA-256 digest"
    )
    huggingface_hub = pytest.importorskip(
        "huggingface_hub", reason="install raw2features[models] for live pin validation"
    )
    info = huggingface_hub.HfApi().model_info(
        repo_id=pin.repo,
        revision=pin.revision,
        timeout=15,
        files_metadata=True,
        token=os.environ.get("HF_TOKEN") or False,
    )
    matches = [
        sibling
        for sibling in (info.siblings or ())
        if sibling.rfilename == pin.filename
    ]
    assert matches, (
        f"patch:{pin.name}: {pin.filename!r} is missing from "
        f"{pin.repo}@{pin.revision}"
    )
    assert len(matches) == 1, (
        f"patch:{pin.name}: expected one metadata entry for {pin.filename!r}, "
        f"found {len(matches)}"
    )

    lfs = matches[0].lfs
    assert lfs is not None, (
        f"patch:{pin.name}: {pin.filename!r} has no Git LFS metadata; "
        "cannot validate its SHA-256 without downloading the checkpoint"
    )
    metadata_sha256 = (
        lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
    )
    assert isinstance(metadata_sha256, str) and _SHA256.fullmatch(metadata_sha256), (
        f"patch:{pin.name}: {pin.filename!r} has invalid or missing SHA-256 "
        f"in its Git LFS metadata: {metadata_sha256!r}"
    )
    assert metadata_sha256 == pin.sha256, (
        f"patch:{pin.name}: registry SHA-256 {pin.sha256} does not match "
        f"{pin.repo}@{pin.revision}/{pin.filename} metadata {metadata_sha256}"
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
