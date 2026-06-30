"""HF-transformers CLIP-style image towers (e.g. PLIP) - optional.

These vision-language pathology models load through ``transformers`` (not ``open_clip``)
and expose their image embedding by one of two documented method names, so this family
duck-types on which the loaded model provides:

* ``get_image_features(pixel_values=x)`` - a ``transformers.CLIPModel``'s projected
  image embedding (PLIP), returned un-normalised (PLIP's downstream patch feature).
* ``encode_image(x)`` - a model's own projected (often L2-normalised) image embedding.

``transformers`` is already in the ``[models]`` extra; the import is still deferred to
:meth:`load` so the entry-point loader skips cleanly when it is absent.
``trust_remote_code`` is enabled so a model with custom modeling code loads (a no-op for
PLIP's vanilla CLIP config). (KEEP fits the ``encode_image`` path but is currently
blocked by a timm-version clash in its remote code - see registry.yaml.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "clip_hf")
class ClipHFEmbedder(Embedder):
    """A transformers CLIP-style image tower; ``-> (B, embedding_dim)``.

    ``spec.source`` is the Hugging Face repo id (e.g. ``vinid/plip``,
    ``Astaxanthin/KEEP``).
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> ClipHFEmbedder:
        import torch

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'clip_hf models (plip, keep) need transformers: '
                'pip install "raw2features[models]"'
            ) from exc

        model = AutoModel.from_pretrained(self.spec.source, trust_remote_code=True)
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        model = self._model
        x = batch.to(self._device)
        with _forward_ctx(self._device, self._dtype):
            if hasattr(model, "encode_image"):  # KEEP (L2-normalised projection)
                out = model.encode_image(x)
            else:  # transformers CLIPModel (PLIP)
                out = model.get_image_features(pixel_values=x)
        return out.float().cpu()
