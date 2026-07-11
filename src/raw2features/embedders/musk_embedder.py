"""MUSK vision-language pathology encoder (Xiang et al., Nature 2024) - image tower.

MUSK is a BEiT3 multimodal model; we use only its image tower. Its package
(``lilab-stanford/MUSK``) timm-registers ``musk_large_patch16_384``, so it loads via
timm plus the ``musk`` package -- its own family, with that import deferred to
:meth:`load` so the dependency stays optional (the ``[musk]`` extra).

Gated: needs a HuggingFace token with accepted ``xiangjx/musk`` access + the ``[musk]``
extra. 384 px, inception (``0.5``) normalisation, 1024-d output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from ._hub import verify_sha256
from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "musk")
class MuskEmbedder(Embedder):
    """MUSK image tower (BEiT3-large @ 384 px) -> 1024-d vision embedding."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> MuskEmbedder:
        import torch

        try:
            import timm
            from huggingface_hub import hf_hub_download
            from musk import modeling, utils  # noqa: F401 - registers the timm arch
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "MUSK needs the optional `musk` package. Install the stack, then the "
                "(non-PyPI) package:\n"
                '  pip install "raw2features[musk]"\n'
                "  pip install git+https://github.com/lilab-stanford/MUSK@"
                "714b666969c1911e5efe70d991140a21030f4ef3"
            ) from exc

        model = timm.create_model("musk_large_patch16_384")
        # Pin the download to the recorded immutable commit; verify bytes before load.
        repo = self.spec.source.removeprefix("hf-hub:").removeprefix("hf_hub:")
        path = hf_hub_download(
            repo, "model.safetensors", revision=self.spec.weights_revision
        )
        verify_sha256(path, self.spec.weights_sha256, what=self.spec.name)
        utils.load_model_and_may_interpolate(path, model, "model|module", "")
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float16  # the card runs MUSK in fp16
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            # Image tower only: with_head=False -> the pre-projection feature; out_norm
            # L2-normalises it; ms_aug=False -> single-scale (the exact-MPP patch).
            out = self._model(
                image=batch.to(self._device),
                with_head=False,
                out_norm=True,
                ms_aug=False,
            )[0]
        return out.float().cpu()
