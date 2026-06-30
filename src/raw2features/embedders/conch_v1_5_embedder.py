"""CONCH v1.5 patch encoder (MahmoodLab) - the tile encoder TITAN consumes.

CONCH v1.5 is the 768-d patch encoder that TITAN's slide features are built from
(512 px @ 20x). It is a custom ViT tower (not a plain timm backbone), and MahmoodLab
ships it *inside* the gated TITAN model: ``AutoModel.from_pretrained('MahmoodLab/TITAN',
trust_remote_code=True).return_conch()`` returns the encoder (and its eval transform).
Loading it this way - rather than re-implementing the architecture - guarantees the
patch features are byte-for-byte what TITAN expects.

Preprocessing is the standard pipeline path: the runner delivers a patch at
``patch_px`` µm/px and the shared transform resizes it to ``input_size`` (448, BILINEAR)
and applies ImageNet normalisation - exactly CONCH v1.5's documented eval transform
(``Resize(448)+CenterCrop(448)+Normalize(ImageNet)``). Extract at 512 px to match TITAN
(``--patch-size 512``); the store's ``level0_patch`` then equals TITAN's
``patch_size_lv0``.

Gated: needs a HuggingFace token with accepted ``MahmoodLab/TITAN`` access, plus the
``[models]`` extra (transformers). Paired slide encoder: ``-s titan``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "conch_v1_5")
class ConchV1_5Embedder(Embedder):
    """CONCH v1.5 tile encoder (custom ViT tower) -> 768-d patch features."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> ConchV1_5Embedder:
        import torch
        from transformers import AutoModel

        # CONCH v1.5 is bundled in the gated TITAN model; return_conch() is the
        # card-documented way to obtain the matched patch encoder.
        titan = AutoModel.from_pretrained(
            "MahmoodLab/TITAN",
            trust_remote_code=True,
            revision=self.spec.weights_revision,  # pin the immutable HF commit
        )
        model, _eval_transform = titan.return_conch()
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float16  # the TITAN card runs CONCH v1.5 in fp16
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model(batch.to(self._device))
        return out.float().cpu()
