"""torchvision ResNet-50 (ImageNet) - the open, ungated floor baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "torchvision")
class TorchvisionEmbedder(Embedder):
    """ResNet-50 IMAGENET1K_V2 with the classifier head removed (2048-d)."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> TorchvisionEmbedder:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        if "resnet50" not in self.spec.source:
            raise ValueError(
                f"TorchvisionEmbedder currently supports resnet50 only; "
                f"got {self.spec.source!r}"
            )
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
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
        return out.float().cpu()
