"""open_clip image towers (CLIP-style vision-language pathology models) - optional.

Models like QuiltNet and BiomedCLIP are CLIP image-text models; we use only the image
tower, whose ``encode_image`` returns the projected image embedding (the standard
PLIP/QuiltNet downstream patch feature). They load through the upstream ``open_clip``
package rather than ``timm``/``transformers``, so they get their own family with the
dependency **deferred** to :meth:`load` - the entry-point loader skips this family
cleanly when ``open_clip`` is absent.

The embedding is returned **un-normalised** (raw ``encode_image``): open_clip L2-
normalises only inside its contrastive loss / cosine retrieval, so the faithful feature
for downstream linear probes / MIL is the pre-normalisation projection.

Install::

    pip install "raw2features[open_clip]"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "open_clip")
class OpenClipEmbedder(Embedder):
    """An open_clip image tower; ``encode_image(x)`` -> ``(B, embedding_dim)``.

    ``spec.source`` is the open_clip model name, e.g. ``hf-hub:wisdomik/QuiltNet-B-32``
    (open_clip resolves the ``hf-hub:`` prefix to the Hugging Face checkpoint + its
    ``open_clip_config.json``).
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> OpenClipEmbedder:
        import torch

        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "open_clip models (quiltnet, biomedclip) need the optional "
                '`open_clip_torch` package: pip install "raw2features[open_clip]"'
            ) from exc

        model, preprocess = open_clip.create_model_from_pretrained(self.spec.source)
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._assert_transform_matches_preprocess(preprocess)
        self._maybe_compile(compile)
        return self

    def _assert_transform_matches_preprocess(self, preprocess) -> None:
        """Cross-check the registry norm against open_clip's own preprocess Normalize.

        open_clip carries the authoritative mean/std in its ``preprocess`` Compose; we
        source those into the registry and assert here so a drift fails loudly rather
        than silently embedding under the wrong normalisation (same guard as CONCH).
        """
        norm = next(
            (t for t in getattr(preprocess, "transforms", [])
             if t.__class__.__name__ == "Normalize"),
            None,
        )
        if norm is None:  # pragma: no cover - defensive
            return
        got_mean = tuple(float(x) for x in norm.mean)
        got_std = tuple(float(x) for x in norm.std)
        for got, want, field in (
            (got_mean, self.spec.mean, "mean"),
            (got_std, self.spec.std, "std"),
        ):
            if any(abs(a - b) > 1e-6 for a, b in zip(got, want, strict=True)):
                raise ValueError(
                    f"{self.spec.name}: open_clip preprocess {field}={got} disagrees "
                    f"with registry {field}={want}; fix registry.yaml (do not guess)."
                )

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model.encode_image(batch.to(self._device))
        return out.float().cpu()
