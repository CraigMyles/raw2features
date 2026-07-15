"""KEEP vision-language pathology encoder -- controlled image-only loader.

The upstream Hugging Face repository exposes KEEP through ``trust_remote_code``.
Its small wrapper constructs a timm ViT-L/16, renames timm's LayerScale parameter
from ``gamma`` to ``weight``, adds a two-layer 768-d projection head, and L2
normalises the result. The published remote code mutates timm globally and its
LayerScale constructor is incompatible with current timm releases.

This module reproduces only that documented image path locally. It downloads one
immutable, SHA-256-pinned ``model.safetensors`` file, reads only ``visual.*`` and
``visual_head.*`` tensors, and loads them strictly. No remote Python is executed,
the unused BERT text tower is never constructed, and pickle deserialisation is not
used.

Primary source (pinned model card/code):
https://huggingface.co/Astaxanthin/KEEP/tree/28a25d95cc6ba27a7e6fab3f144e13dbafd8b21e
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from raw2features.core.plugins import register

from ._hub import download_pinned_hf_file, verify_sha256
from .base import Embedder
from .fingerprint import KEEP_CONSTRUCTOR_CONTRACT

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _replace_keep_layer_scales(visual: Any, torch_module: Any) -> None:
    """Give current-timm LayerScale modules KEEP's checkpoint key contract.

    Modern timm stores each scale as ``gamma``. KEEP's published code replaces
    the class globally so the same parameter is named ``weight``. Replace only
    this model's 48 LayerScale instances, preserving their values, device, dtype,
    and in-place behaviour without mutating timm process-wide.
    """

    nn = torch_module.nn

    class _KeepLayerScale(nn.Module):
        def __init__(self, value: Any, *, inplace: bool) -> None:
            super().__init__()
            self.inplace = bool(inplace)
            self.weight = nn.Parameter(value.detach().clone())

        def forward(self, x):
            return x.mul_(self.weight) if self.inplace else x * self.weight

    blocks = getattr(visual, "blocks", None)
    if blocks is None or len(blocks) != 24:
        raise ValueError(
            "KEEP: expected the pinned ViT-L/16 constructor to make 24 blocks"
        )

    replaced = 0
    for block_index, block in enumerate(blocks):
        for attribute in ("ls1", "ls2"):
            layer = getattr(block, attribute, None)
            value = getattr(layer, "gamma", None)
            if value is None:
                value = getattr(layer, "weight", None)
            if value is None or getattr(value, "ndim", None) != 1:
                raise ValueError(
                    f"KEEP: block {block_index} {attribute} is not a supported "
                    "timm LayerScale module"
                )
            setattr(
                block,
                attribute,
                _KeepLayerScale(value, inplace=getattr(layer, "inplace", False)),
            )
            replaced += 1

    if replaced != 48:  # pragma: no cover - guarded by the loops above
        raise ValueError(f"KEEP: replaced {replaced} LayerScale modules, expected 48")


def _build_keep_image_model(timm_module: Any, torch_module: Any):
    """Construct the exact image path recorded in ``KEEP_CONSTRUCTOR_CONTRACT``."""

    nn = torch_module.nn
    contract = KEEP_CONSTRUCTOR_CONTRACT
    create = contract["create_model"]
    visual = timm_module.create_model(
        contract["architecture"],
        pretrained=create["pretrained"],
        img_size=create["img_size"],
        patch_size=create["patch_size"],
        init_values=create["init_values"],
        num_classes=create["num_classes"],
    )
    expected_width = int(contract["vision_width"])
    if int(getattr(visual, "num_features", -1)) != expected_width:
        raise ValueError(
            "KEEP: constructed vision width "
            f"{getattr(visual, 'num_features', None)!r}, expected {expected_width}"
        )
    _replace_keep_layer_scales(visual, torch_module)

    projection_dim = int(contract["projection_dim"])

    class _KEEPImageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = visual
            self.visual_head = nn.Sequential(
                nn.Linear(expected_width, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )

        def forward(self, image_inputs):
            vision_features = self.visual(image_inputs)
            return torch_module.nn.functional.normalize(
                self.visual_head(vision_features), dim=-1
            )

    return _KEEPImageModel()


def _load_keep_image_state(
    path: str,
    model: Any,
    *,
    safe_open_fn: Callable[..., Any] | None = None,
) -> None:
    """Read exactly the image tensors from a verified safetensors checkpoint."""

    if safe_open_fn is None:
        from safetensors import safe_open

        safe_open_fn = safe_open

    expected = set(model.state_dict())
    if not expected or any(
        not key.startswith(("visual.", "visual_head.")) for key in expected
    ):
        raise ValueError("KEEP: local image model exposed an unexpected state contract")

    with safe_open_fn(path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        image_keys = {
            key for key in available if key.startswith(("visual.", "visual_head."))
        }
        missing = sorted(expected - image_keys)
        unexpected = sorted(image_keys - expected)
        if missing or unexpected:
            detail = []
            if missing:
                detail.append(f"missing {len(missing)} image keys (e.g. {missing[0]})")
            if unexpected:
                detail.append(
                    f"found {len(unexpected)} unexpected image keys "
                    f"(e.g. {unexpected[0]})"
                )
            raise ValueError(
                "KEEP: pinned checkpoint does not match the local image wrapper: "
                + "; ".join(detail)
            )
        # Deliberately do not read text.*, logit_scale, or future non-image keys.
        state: Mapping[str, Any] = {
            key: handle.get_tensor(key) for key in sorted(expected)
        }

    model.load_state_dict(state, strict=True)


def _validate_keep_spec(spec: Any) -> None:
    """Reject registry drift that the fixed local wrapper cannot honour."""

    expected = KEEP_CONSTRUCTOR_CONTRACT
    errors = []
    for field, wanted in (
        ("family", "keep"),
        ("input_size", expected["create_model"]["img_size"]),
        ("embedding_dim", expected["projection_dim"]),
        ("pooling", "pooled"),
    ):
        got = getattr(spec, field)
        if got != wanted:
            errors.append(f"{field}={got!r} (expected {wanted!r})")
    if spec.timm_kwargs:
        errors.append("timm_kwargs must be empty (constructor is a versioned contract)")
    if not spec.weights_filename or not spec.weights_filename.endswith(".safetensors"):
        errors.append("weights_filename must name the pinned .safetensors artifact")
    if not spec.weights_revision:
        errors.append("weights_revision is required")
    if not spec.weights_sha256:
        errors.append("weights_sha256 is required")
    if errors:
        raise ValueError("KEEP registry contract mismatch: " + "; ".join(errors))


@register("embedders", "keep")
class KEEPEmbedder(Embedder):
    """KEEP ViT-L/16 image tower -> 768-d L2-normalised projection."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> KEEPEmbedder:
        try:
            import timm
            import torch
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'KEEP needs the model stack: pip install "raw2features[models]"'
            ) from exc

        _validate_keep_spec(self.spec)
        checkpoint = download_pinned_hf_file(
            self.spec.source,
            self.spec.weights_filename,
            self.spec.weights_revision,
        )
        verify_sha256(checkpoint, self.spec.weights_sha256, what=self.spec.name)

        model = _build_keep_image_model(timm, torch)
        _load_keep_image_state(checkpoint, model, safe_open_fn=safe_open)
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model(batch.to(self._device))
        if out.ndim != 2 or out.shape[-1] != self.spec.embedding_dim:
            raise ValueError(
                f"KEEP: expected [B, {self.spec.embedding_dim}] output, "
                f"got {tuple(out.shape)}"
            )
        return out.float().cpu()
