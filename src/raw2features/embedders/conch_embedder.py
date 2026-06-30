"""CONCH image encoder (MahmoodLab) - optional, needs the ``conch`` package.

CONCH is a vision-language pathology model; we use only its image tower. Unlike the
timm backbones it loads through the project's own ``conch`` package rather than
``timm``, so it lives in its own family and the dependency is **optional** - the
``conch`` import is deferred to :meth:`load`, and the entry-point loader skips this
family cleanly when the package is absent.

Install::

    pip install "raw2features[conch]"
    # or: pip install git+https://github.com/Mahmoodlab/CONCH.git
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch

# CONCH's single published vision architecture (paired with the hf_hub checkpoint).
_ARCH = "conch_ViT-B-16"


@register("embedders", "conch")
class ConchEmbedder(Embedder):
    """CONCH ViT-B/16 image tower; ``encode_image(proj_contrast=False)`` -> 512-d.

    The 512-d non-contrastive image features are CONCH's recommended representation
    for downstream tasks (the contrastive projection is for image-text retrieval).
    """

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> ConchEmbedder:
        import torch

        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "CONCH needs the optional `conch` package. Install the stack, then "
                "the (non-PyPI, gated) package:\n"
                '  pip install "raw2features[conch]"\n'
                "  pip install git+https://github.com/Mahmoodlab/CONCH.git"
            ) from exc

        model, preprocess = create_model_from_pretrained(_ARCH, self.spec.source)
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._assert_transform_matches_preprocess(preprocess)
        self._maybe_compile(compile)
        return self

    def _assert_transform_matches_preprocess(self, preprocess) -> None:
        """Cross-check the registry norm against CONCH's own preprocess transform.

        The card documents no fixed mean/std; the authoritative numbers live in the
        ``Normalize`` step of the ``preprocess`` returned by the loader. We source
        those into the registry and assert here so a CONCH/registry drift fails
        loudly rather than silently embedding under the wrong normalisation.
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
                    f"conch: preprocess {field}={got} disagrees with the registry "
                    f"{field}={want}; update registry.yaml (do not guess)."
                )

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model.encode_image(
                batch.to(self._device), proj_contrast=False, normalize=False
            )
        return out.float().cpu()
