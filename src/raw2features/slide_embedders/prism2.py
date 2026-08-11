"""PRISM2 slide encoders (paige-ai/Prism2, CC-BY-NC-ND-4.0).

The gated model consumes 1280-d Virchow2 CLS tokens from 224 px tissue tiles at
0.5 micrometres per pixel. raw2features already preserves those tokens as the first
half of its 2560-d ``virchow2`` output, so the adapter slices them losslessly instead
of requiring a duplicate patch extraction. The base and diagnostic representations
are separate outputs because the authors give them different downstream roles.
"""

from __future__ import annotations

from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

from raw2features.core.plugins import register
from raw2features.embedders._hub import (
    download_pinned_hf_snapshot,
    pinned_model_cache_dir,
    verify_sha256,
)

from .base import SlideEmbedder, SlideModelSpec

PRISM2_SOURCE = "hf-hub:paige-ai/Prism2"
PRISM2_REVISION = "450352d0ddc6b42b21ce20794ce0fbefe6b5a47a"
PRISM2_TRANSFORMERS_VERSION = "4.51.3"
PRISM2_FLASH_ATTN_VERSION = "2.8.3"
PRISM2_CHECKPOINT_SHA256 = {
    "model.safetensors.index.json": (
        "bda099bfdad33ee0e2fe49b6101feb58a0389d8c8a000cc6636c7a24b8c41bbd"
    ),
    "model-00001-of-00004.safetensors": (
        "3eec1f46c9dab8ac0dc744ef6f7b2db19028b5dd3bc722499e25a802f9a95556"
    ),
    "model-00002-of-00004.safetensors": (
        "8cba90ae79ea934770673e9461720b0bf37665e0ac00e86124d116d9dfa526a4"
    ),
    "model-00003-of-00004.safetensors": (
        "f899aefe28c8111bbd6a14ef6e295925e1588c88b1741d06935015162f0d4dd5"
    ),
    "model-00004-of-00004.safetensors": (
        "da79790c9a73cf1ce85867aaa57c47cfd8d69e13b57ac86252d24a2687c2998c"
    ),
}
PRISM2_CODE_FILES = (
    "config.json",
    "configuration_prism2.py",
    "modeling_prism2.py",
    "processing_prism2.py",
)
PRISM2_SNAPSHOT_ALLOW_PATTERNS = (
    *PRISM2_CODE_FILES,
    *PRISM2_CHECKPOINT_SHA256,
)

PRISM2_PHI3_SOURCE = "hf-hub:microsoft/Phi-3-mini-128k-instruct"
PRISM2_PHI3_REVISION = "f3c06aed622e14ca0abf5115094e4fc9a9948f36"
PRISM2_PHI3_LICENSE = "MIT"
PRISM2_PHI3_ARTIFACT_SHA256 = {
    "added_tokens.json": (
        "f8e5a880c6c563a126d9efacb654e105fc4b66f21e5a540f04436e9487a955fb"
    ),
    "config.json": ("af70441b867458d18788c564ea69894adddaa4de47ab6534391604fc4893f03b"),
    "special_tokens_map.json": (
        "810adc6e6c6ef2f56c285ef930d243358a3a9f05e36a01c5a10bafc6fac4609b"
    ),
    "tokenizer.json": (
        "072ab882d6c7192a42f78790945d16c064691321a73251a4b18f6a380f0fbe39"
    ),
    "tokenizer.model": (
        "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347"
    ),
    "tokenizer_config.json": (
        "aaa87217a0f61c684cdc8703d3d4030a1f5b1077183610b61a14d7f28addbb58"
    ),
}

_BASE_SPEC = SlideModelSpec(
    name="prism2",
    family="prism2",
    source=PRISM2_SOURCE,
    embedding_dim=2560,
    patch_encoder="virchow2",
    patch_dim=2560,
    gated=True,
    license="CC-BY-NC-ND-4.0",
    transform_source_url="https://huggingface.co/paige-ai/Prism2",
    doi="10.1038/s41591-026-04521-4",
    weights_sha256=PRISM2_CHECKPOINT_SHA256["model.safetensors.index.json"],
    weights_revision=PRISM2_REVISION,
    weights_filename="model.safetensors.index.json",
    weights_manifest=PRISM2_CHECKPOINT_SHA256,
    notes=(
        "Base perceiver embedding over the 1280-d CLS half of raw2features' "
        "Virchow2 output. CUDA and flash-attn are required."
    ),
)

_DIAGNOSTIC_SPEC = SlideModelSpec(
    name="prism2_diagnostic",
    family="prism2",
    source=PRISM2_SOURCE,
    embedding_dim=3072,
    patch_encoder="virchow2",
    patch_dim=2560,
    gated=True,
    license="CC-BY-NC-ND-4.0",
    transform_source_url="https://huggingface.co/paige-ai/Prism2",
    doi="10.1038/s41591-026-04521-4",
    weights_sha256=PRISM2_CHECKPOINT_SHA256["model.safetensors.index.json"],
    weights_revision=PRISM2_REVISION,
    weights_filename="model.safetensors.index.json",
    weights_manifest=PRISM2_CHECKPOINT_SHA256,
    notes=(
        "Diagnostic Phi-3 hidden-state embedding over the 1280-d CLS half of "
        "raw2features' Virchow2 output. CUDA and flash-attn are required."
    ),
)


def _assert_snapshot_files(
    snapshot: str,
    expected: tuple[str, ...],
    *,
    what: str,
) -> None:
    """Reject a stale managed directory before its custom code can execute."""

    root = Path(snapshot)
    allowed = {*expected, ".cache", "__pycache__"}
    unexpected = sorted(
        path.name for path in root.iterdir() if path.name not in allowed
    )
    missing = sorted(name for name in expected if not (root / name).is_file())
    if unexpected or missing:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        joined = "; ".join(details)
        raise ValueError(f"{what}: invalid pinned runtime snapshot ({joined})")


def _verified_snapshot(
    source: str,
    revision: str,
    artifacts: dict[str, str],
    *,
    extra_files: tuple[str, ...] = (),
    what: str,
) -> str:
    expected = (*extra_files, *artifacts)
    snapshot = download_pinned_hf_snapshot(
        source,
        revision,
        allow_patterns=expected,
        local_dir=pinned_model_cache_dir(source, revision),
    )
    _assert_snapshot_files(snapshot, expected, what=what)
    for filename, sha256 in artifacts.items():
        verify_sha256(str(Path(snapshot) / filename), sha256, what=f"{what}:{filename}")
    return snapshot


def _require_prism2_runtime() -> None:
    required = {
        "transformers": PRISM2_TRANSFORMERS_VERSION,
        "flash-attn": PRISM2_FLASH_ATTN_VERSION,
    }
    mismatches = []
    for distribution, expected in required.items():
        try:
            installed = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            found = installed or "missing"
            mismatches.append(f"{distribution}=={expected} (found {found})")
    if mismatches:
        raise RuntimeError(
            "PRISM2 requires its validated runtime; install raw2features[prism2]: "
            + ", ".join(mismatches)
        )


class _Prism2SlideEmbedder(SlideEmbedder):
    def __init__(self, spec: SlideModelSpec) -> None:
        super().__init__(spec)
        self._model = None
        self._device = "cpu"

    def load(self, device: str = "cuda", dtype=None) -> _Prism2SlideEmbedder:
        if not str(device).startswith("cuda"):
            raise ValueError("PRISM2 requires a CUDA GPU and flash-attn==2.8.3")
        _require_prism2_runtime()
        del dtype  # the published path keeps fp32 weights and uses bf16 autocast

        if self.spec.weights_manifest != PRISM2_CHECKPOINT_SHA256:
            raise ValueError(
                f"{self.spec.name}: registry weights_manifest does not match the "
                "pinned PRISM2 checkpoint"
            )
        prism_snapshot = _verified_snapshot(
            self.spec.source,
            str(self.spec.weights_revision),
            dict(self.spec.weights_manifest),
            extra_files=PRISM2_CODE_FILES,
            what=self.spec.name,
        )
        phi3_snapshot = _verified_snapshot(
            PRISM2_PHI3_SOURCE,
            PRISM2_PHI3_REVISION,
            PRISM2_PHI3_ARTIFACT_SHA256,
            what=f"{self.spec.name}:Phi-3",
        )

        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            prism_snapshot,
            trust_remote_code=True,
            local_files_only=True,
        )
        # Pinned upstream code constructs Phi-3 from this field. Point it at the
        # verified local config/tokenizer snapshot instead of mutable Hub main.
        config.text_decoder_model_id = phi3_snapshot
        model = AutoModel.from_pretrained(
            prism_snapshot,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype="auto",
        )
        model.eval().to(device)
        self._model = model
        self._device = str(device)
        return self

    def encode(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,  # noqa: ARG002 - PRISM2 is non-spatial
        patch_size_lv0: int | None = None,  # noqa: ARG002 - fixed upstream tile scale
    ) -> np.ndarray:
        import torch

        if self._model is None:
            raise RuntimeError("call load() before encode()")
        if features.ndim != 2 or int(features.shape[1]) != 2560:
            raise ValueError(
                "PRISM2 requires raw2features virchow2 features shaped (N, 2560); "
                f"got {tuple(features.shape)}"
            )

        # virchow2 is stored as [CLS(1280), mean-patch(1280)]. PRISM2 consumes CLS.
        cls = np.ascontiguousarray(features[:, :1280], dtype=np.float32)
        tile_embeddings = torch.from_numpy(cls).unsqueeze(0).to(self._device)
        attention_mask = torch.ones(
            tile_embeddings.shape[:2],
            dtype=torch.int32,
            device=self._device,
        )
        autocast = (
            torch.autocast("cuda", torch.bfloat16)
            if self._device.startswith("cuda")
            else torch.autocast("cpu", enabled=False)
        )
        forwards: dict[str, Any] = {
            "prism2": self._model.get_base_embedding,
            "prism2_diagnostic": self._model.get_diagnostic_embedding,
        }
        try:
            forward = forwards[self.spec.name]
        except KeyError as exc:  # pragma: no cover - registry construction guards this
            raise ValueError(f"unknown PRISM2 output {self.spec.name!r}") from exc
        with autocast, torch.inference_mode():
            vector = forward(
                tile_embeddings=tile_embeddings,
                attention_mask=attention_mask,
            )
        return vector.reshape(-1).float().cpu().numpy()

    def unload(self) -> None:
        import torch

        if self._model is not None:
            self._model.cpu()
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


@register("slide_embedders", "prism2")
class Prism2SlideEmbedder(_Prism2SlideEmbedder):
    """PRISM2's general 2560-dimensional base representation."""

    def __init__(self) -> None:
        super().__init__(_BASE_SPEC)


@register("slide_embedders", "prism2_diagnostic")
class Prism2DiagnosticSlideEmbedder(_Prism2SlideEmbedder):
    """PRISM2's diagnosis-oriented 3072-dimensional representation."""

    def __init__(self) -> None:
        super().__init__(_DIAGNOSTIC_SPEC)
