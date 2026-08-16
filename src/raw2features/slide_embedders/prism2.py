"""PRISM2 slide encoders (paige-ai/Prism2, CC-BY-NC-ND-4.0).

The gated model consumes 1280-d Virchow2 CLS tokens from 224 px tissue tiles at
0.5 micrometres per pixel. raw2features already preserves those tokens as the first
half of its 2560-d ``virchow2`` output, so the adapter slices them losslessly instead
of requiring a duplicate patch extraction. The base and diagnostic representations
are separate outputs because the authors give them different downstream roles.
"""

from __future__ import annotations

import importlib
import sys
import threading
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
PRISM2_TRANSFORMERS_VERSION = "4.56.0"
PRISM2_FLASH_ATTN_VERSION = "2.8.3"
PRISM2_PHI3_MASK_COMPAT_VERSION = "phi3_4_51_3_causal_mask_v1"
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
PRISM2_CODE_SHA256 = {
    "config.json": "2170566eeb216810e32f69293e07429694fc308e4290efb4ca2157dc7438f7a9",
    "configuration_prism2.py": (
        "24874c747af4ad499db5ce4ac9e6113c0289fb937badce4a6ccf37b3e6f8ede4"
    ),
    "modeling_prism2.py": (
        "57546d28c94d4e8fd0fd29600f0b0190d32cc8fedf7c82ec85f525183f28669a"
    ),
    "processing_prism2.py": (
        "a3dcc7b466234fcc96b47b32f774f1ffbc7271b8af37b6352b13d537b51c9bc9"
    ),
}
PRISM2_CODE_FILES = tuple(PRISM2_CODE_SHA256)
PRISM2_SNAPSHOT_ALLOW_PATTERNS = (
    *PRISM2_CODE_FILES,
    *PRISM2_CHECKPOINT_SHA256,
)
PRISM2_RUNTIME_SHA256 = {**PRISM2_CODE_SHA256, **PRISM2_CHECKPOINT_SHA256}

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
PRISM2_PHI3_CONFIG_SHA256 = {"config.json": PRISM2_PHI3_ARTIFACT_SHA256["config.json"]}

_VERIFIED_SNAPSHOT_CACHE: dict[
    tuple[str, tuple[tuple[str, str], ...]],
    tuple[tuple[str, int, int, int, int, int], ...],
] = {}
_VERIFIED_SNAPSHOT_LOCK = threading.Lock()


def _prepare_phi3_4d_causal_mask(
    attention_mask,
    sequence_length: int,
    target_length: int,
    dtype,
    device,
    cache_position,
    batch_size: int,
    config,
    past_key_values,
):
    """Restore the Phi-3 mask contract called by the pinned PRISM2 code."""

    # This is the mask algorithm expected by the pinned PRISM2 custom code. Its
    # Phi3Model entry point was present in Transformers 4.51.3 (Apache-2.0) but
    # removed before the shared 4.56.0 runtime.
    import torch
    from transformers.cache_utils import SlidingWindowCache

    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    min_dtype = torch.finfo(dtype).min
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    target_positions = torch.arange(target_length, device=device)
    diagonal_attend_mask = target_positions > cache_position.reshape(-1, 1)
    if config.sliding_window is not None:
        if (
            not isinstance(past_key_values, SlidingWindowCache)
            or sequence_length > target_length
        ):
            sliding_attend_mask = target_positions <= (
                cache_position.reshape(-1, 1) - config.sliding_window
            )
            diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
    causal_mask *= diagonal_attend_mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        if attention_mask.shape[-1] > target_length:
            attention_mask = attention_mask[:, :target_length]
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length]
            + attention_mask[:, None, None, :].to(causal_mask.device)
        ) == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)
    return causal_mask


class _Prism2Phi3MaskCompatibility:
    """Private namespace for the Phi-3 helper used by pinned PRISM2 code."""

    _prepare_4d_causal_attention_mask_with_cache_position = staticmethod(
        _prepare_phi3_4d_causal_mask
    )


def _bind_phi3_mask_compatibility(model) -> None:
    """Bind the helper only inside PRISM2's verified dynamic-code module."""

    module_name = type(model).__module__
    module = sys.modules.get(module_name)
    if module is None or not module_name.startswith("transformers_modules."):
        raise RuntimeError(
            "PRISM2 loaded outside its expected Transformers dynamic-code module; "
            "refusing to install the Phi-3 compatibility contract"
        )
    current = getattr(module, "Phi3Model", None)
    if current is _Prism2Phi3MaskCompatibility:
        return
    if (
        current is None
        or getattr(current, "__name__", None) != "Phi3Model"
        or getattr(current, "__module__", None)
        != "transformers.models.phi3.modeling_phi3"
    ):
        raise RuntimeError(
            "PRISM2's verified module does not expose the expected Phi3Model import; "
            "refusing to alter an unknown runtime"
        )
    # modeling_prism2.py calls its imported Phi3Model name directly during the
    # diagnostic forward. Rebind that module global rather than changing the shared
    # Transformers class process-wide.
    module.Phi3Model = _Prism2Phi3MaskCompatibility


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
    allowed_files: tuple[str, ...] | None = None,
    what: str,
) -> None:
    """Reject a stale managed directory before its custom code can execute."""

    root = Path(snapshot)
    allowed = {*(allowed_files or expected), ".cache", "__pycache__"}
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
    allowed_files: tuple[str, ...] | None = None,
    what: str,
) -> str:
    expected = tuple(artifacts)
    local_dir = pinned_model_cache_dir(source, revision)
    if Path(local_dir).is_dir() and all(
        (Path(local_dir) / filename).is_file() for filename in expected
    ):
        _assert_snapshot_files(
            local_dir,
            expected,
            allowed_files=allowed_files,
            what=what,
        )
        _verify_snapshot_artifacts_once(local_dir, artifacts, what=what)
        return local_dir

    snapshot = download_pinned_hf_snapshot(
        source,
        revision,
        allow_patterns=expected,
        local_dir=local_dir,
    )
    _assert_snapshot_files(
        snapshot,
        expected,
        allowed_files=allowed_files,
        what=what,
    )
    _verify_snapshot_artifacts_once(snapshot, artifacts, what=what)
    return snapshot


def _snapshot_signature(
    snapshot: str,
    artifacts: dict[str, str],
) -> tuple[tuple[str, int, int, int, int, int], ...]:
    root = Path(snapshot)
    signature = []
    for filename in sorted(artifacts):
        stat = (root / filename).stat()
        signature.append(
            (
                filename,
                int(stat.st_dev),
                int(stat.st_ino),
                int(stat.st_size),
                int(stat.st_mtime_ns),
                int(stat.st_ctime_ns),
            )
        )
    return tuple(signature)


def _verify_snapshot_artifacts_once(
    snapshot: str,
    artifacts: dict[str, str],
    *,
    what: str,
) -> None:
    """Hash an immutable snapshot once per process, unless any file changes."""

    key = (str(Path(snapshot).resolve()), tuple(sorted(artifacts.items())))
    with _VERIFIED_SNAPSHOT_LOCK:
        before = _snapshot_signature(snapshot, artifacts)
        if _VERIFIED_SNAPSHOT_CACHE.get(key) == before:
            return
        for filename, sha256 in artifacts.items():
            verify_sha256(
                str(Path(snapshot) / filename),
                sha256,
                what=f"{what}:{filename}",
            )
        after = _snapshot_signature(snapshot, artifacts)
        if after != before:
            raise ValueError(f"{what}: pinned runtime snapshot changed while hashing")
        _VERIFIED_SNAPSHOT_CACHE[key] = after


def _require_prism2_runtime() -> None:
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:  # pragma: no cover - declared by the optional extra
        raise RuntimeError(
            "PRISM2 requires its optional runtime; install raw2features[prism2], "
            "then install flash-attn==2.8.3 with --no-build-isolation"
        ) from exc

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
        if installed is None or not SpecifierSet(f"=={expected}").contains(installed):
            found = installed or "missing"
            mismatches.append(f"{distribution}=={expected} (found {found})")
    if mismatches:
        raise RuntimeError(
            "PRISM2 requires its validated runtime. Install raw2features[prism2], "
            "then install flash-attn==2.8.3 with --no-build-isolation. Mismatch: "
            + ", ".join(mismatches)
        )
    try:
        importlib.import_module("flash_attn")
    except Exception as exc:  # noqa: BLE001 - surface binary/ABI import failures early
        raise RuntimeError(
            "PRISM2 found flash-attn==2.8.3, but it could not be imported. "
            "Rebuild flash-attn against the active Torch/CUDA environment with "
            "--no-build-isolation. Import error: "
            f"{exc}"
        ) from exc


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
            PRISM2_RUNTIME_SHA256,
            what=self.spec.name,
        )
        phi3_artifacts = (
            PRISM2_PHI3_ARTIFACT_SHA256
            if self.spec.name == "prism2_diagnostic"
            else PRISM2_PHI3_CONFIG_SHA256
        )
        phi3_snapshot = _verified_snapshot(
            PRISM2_PHI3_SOURCE,
            PRISM2_PHI3_REVISION,
            phi3_artifacts,
            allowed_files=tuple(PRISM2_PHI3_ARTIFACT_SHA256),
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
        if self.spec.name == "prism2_diagnostic":
            _bind_phi3_mask_compatibility(model)
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
