"""Pinned DINOv2 ViT-g teacher checkpoints (OpenMidnight and OpenPath).

The authors publish PyTorch teacher checkpoints rather than self-contained timm or
Transformers models. Their examples construct DINOv2 with ``torch.hub``; resolving a
mutable GitHub branch at inference time would leave the executable loader code
unpinned. This loader instead constructs timm's installed DINOv2 ViT-g/14-reg4
implementation and applies the small, deterministic state-dict conversion used by
timm itself.

Checkpoint bytes are downloaded at the registry's immutable Hugging Face revision
and SHA-256 verified *before* ``torch.load``. Loading is strict, including after
OpenPath's chunked training-block keys are flattened, so architecture or checkpoint
drift fails closed instead of silently producing different embeddings.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from raw2features.core.plugins import register

from ._hub import download_pinned_hf_file, verify_sha256
from .base import Embedder
from .fingerprint import DINO_TEACHER_CONSTRUCTOR_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    import torch


_CHUNKED_BLOCK = re.compile(r"^blocks\.\d+\.(\d+)\.(.+)$")
_DINOV2_W12 = re.compile(r"^blocks\.(\d+)\.mlp\.w12\.(weight|bias)$")
_DINOV2_W3 = re.compile(r"^blocks\.(\d+)\.mlp\.w3\.(weight|bias)$")


def _flatten_chunked_blocks(state: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten DINOv2 ``BlockChunk`` keys to timm's sequential block keys.

    OpenPath was built with ``block_chunks=4``. Its parameter names are therefore
    ``blocks.<chunk>.<global-block>.*`` (padding identities keep the second index
    global). OpenMidnight is already flat and does not request this conversion.
    """

    flattened: dict[str, Any] = {}
    for key, value in state.items():
        match = _CHUNKED_BLOCK.match(key)
        converted = f"blocks.{match.group(1)}.{match.group(2)}" if match else key
        if converted in flattened:
            raise ValueError(
                f"duplicate DINOv2 state key after flattening: {converted}"
            )
        flattened[converted] = value
    return flattened


def _extract_teacher_state(
    payload: Any, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the author-documented backbone state from a checkpoint payload."""

    state = payload
    state_dict_key = checkpoint.get("state_dict_key")
    if state_dict_key is not None:
        if not isinstance(state, Mapping) or state_dict_key not in state:
            raise ValueError(
                f"checkpoint has no documented state_dict_key {state_dict_key!r}"
            )
        state = state[state_dict_key]
    if not isinstance(state, Mapping):
        raise ValueError("DINOv2 teacher checkpoint did not contain a state dict")

    prefix = checkpoint.get("state_dict_prefix")
    if prefix:
        state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if not state:
            raise ValueError(f"checkpoint contains no keys with prefix {prefix!r}")
    else:
        state = dict(state)

    if checkpoint.get("flatten_block_chunks", False):
        state = _flatten_chunked_blocks(state)
    return dict(state)


def _convert_dinov2_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Convert official DINOv2 names and position embeddings to timm's form.

    DINOv2 gives the class token a position embedding but gives register tokens no
    positional embedding. timm represents the same computation with
    ``no_embed_class=True``: fold the class position into ``cls_token``, remove it
    from ``pos_embed``, and rename ``register_tokens``. The SwiGLU parameters are
    the only remaining architectural rename.
    """

    source = dict(state)
    source.pop("mask_token", None)  # training-only token, absent from timm inference
    missing = [
        key
        for key in ("register_tokens", "cls_token", "pos_embed")
        if key not in source
    ]
    if missing:
        raise ValueError(
            "DINOv2 teacher checkpoint is missing required keys: " + ", ".join(missing)
        )

    converted: dict[str, Any] = {"reg_token": source.pop("register_tokens")}
    cls_token = source.pop("cls_token")
    pos_embed = source.pop("pos_embed")
    converted["cls_token"] = cls_token + pos_embed[:, :1]
    converted["pos_embed"] = pos_embed[:, 1:]

    for key, value in source.items():
        if _DINOV2_W12.match(key):
            key = key.replace(".mlp.w12.", ".mlp.fc1.")
        elif _DINOV2_W3.match(key):
            key = key.replace(".mlp.w3.", ".mlp.fc2.")
        if key in converted:
            raise ValueError(f"duplicate DINOv2 state key after conversion: {key}")
        converted[key] = value
    return converted


def _type_path(value: Any) -> str | None:
    """Return a stable concrete type name for a runtime contract value."""

    if value is None:
        return None
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _linear_contract(layer: Any) -> dict[str, Any]:
    return {
        "implementation": _type_path(layer),
        "in_features": getattr(layer, "in_features", None),
        "out_features": getattr(layer, "out_features", None),
        "bias": getattr(layer, "bias", None) is not None,
    }


def _dropout_contract(layer: Any) -> dict[str, Any]:
    return {
        "implementation": _type_path(layer),
        "p": getattr(layer, "p", None),
    }


def _ffn_contract(ffn: Any) -> dict[str, Any]:
    """Describe timm's concrete packed-SwiGLU computation."""

    return {
        "implementation": _type_path(ffn),
        "activation": _type_path(getattr(ffn, "act", None)),
        "gate_last": getattr(ffn, "gate_last", None),
        "chunk_dim": getattr(ffn, "chunk_dim", None),
        "fc1": _linear_contract(getattr(ffn, "fc1", None)),
        "norm": _type_path(getattr(ffn, "norm", None)),
        "fc2": _linear_contract(getattr(ffn, "fc2", None)),
        "drop1": _dropout_contract(getattr(ffn, "drop1", None)),
        "drop2": _dropout_contract(getattr(ffn, "drop2", None)),
    }


def _validate_model_contract(model: Any, spec) -> None:
    """Fail closed if timm's named architecture no longer means our contract."""

    contract = DINO_TEACHER_CONSTRUCTOR_CONTRACT
    blocks = getattr(model, "blocks", ())
    first_block = blocks[0] if len(blocks) else None
    patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    checks = {
        "embedding_dim": getattr(model, "num_features", None),
        "depth": len(blocks),
        "num_heads": getattr(getattr(first_block, "attn", None), "num_heads", None),
        "register_tokens": getattr(model, "num_reg_tokens", None),
        "patch_size": list(patch_size) if patch_size is not None else None,
        "no_embed_class": getattr(model, "no_embed_class", None),
        "global_pool": getattr(model, "global_pool", None),
    }
    expected = {
        key: contract[key]
        for key in (
            "embedding_dim",
            "depth",
            "num_heads",
            "register_tokens",
            "patch_size",
            "no_embed_class",
            "global_pool",
        )
    }
    if checks != expected:
        raise RuntimeError(
            f"{spec.name}: timm architecture does not match the loader contract "
            f"(got {checks}, expected {expected})"
        )
    expected_ffn = contract["ffn"]
    for index, block in enumerate(blocks):
        actual_ffn = _ffn_contract(getattr(block, "mlp", None))
        if actual_ffn != expected_ffn:
            raise RuntimeError(
                f"{spec.name}: timm block {index} FFN does not match the loader "
                f"contract (got {actual_ffn}, expected {expected_ffn})"
            )
    if spec.input_size != contract["input_size"]:
        raise ValueError(
            f"{spec.name}: registry input_size={spec.input_size} does not match "
            f"the DINOv2 teacher contract ({contract['input_size']})"
        )
    if spec.embedding_dim != contract["embedding_dim"] or spec.pooling != "cls":
        raise ValueError(
            f"{spec.name}: DINOv2 teacher output must be "
            f"{contract['embedding_dim']}-d CLS"
        )


@register("embedders", "dino_teacher")
class DinoTeacherEmbedder(Embedder):
    """Strict loader for SHA-verified DINOv2 ViT-g teacher checkpoints."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> DinoTeacherEmbedder:
        import timm
        import torch

        checkpoint = self.spec.checkpoint or {}
        filename = checkpoint.get("filename") or self.spec.weights_filename
        if not filename or filename != self.spec.weights_filename:
            raise ValueError(
                f"{self.spec.name}: checkpoint filename must equal the registry's "
                "weights_filename"
            )
        repo = checkpoint.get("repo")
        source = f"hf-hub:{repo}" if repo else self.spec.source
        path = download_pinned_hf_file(
            source,
            filename,
            self.spec.weights_revision,
        )
        # Keep this check before both torch.load calls. The fallback may unpickle
        # metadata, but only after the bytes match the immutable registry pin.
        verify_sha256(path, self.spec.weights_sha256, what=self.spec.name)
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - author checkpoints may contain metadata
            payload = torch.load(path, map_location="cpu", weights_only=False)

        state = _convert_dinov2_state(_extract_teacher_state(payload, checkpoint))
        contract = DINO_TEACHER_CONSTRUCTOR_CONTRACT
        model = timm.create_model(
            contract["architecture"],
            pretrained=False,
            img_size=contract["input_size"],
            num_classes=0,
            global_pool=contract["global_pool"],
        )
        _validate_model_contract(model, self.spec)
        model.load_state_dict(state, strict=True)
        model.eval().to(device)

        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            output = self._model(batch.to(self._device))
        if output.ndim != 2 or output.shape[-1] != self.spec.embedding_dim:
            raise ValueError(
                f"{self.spec.name}: expected [B, {self.spec.embedding_dim}] CLS "
                f"embeddings, got {tuple(output.shape)}"
            )
        return output.float().cpu()
