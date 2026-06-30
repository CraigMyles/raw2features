"""HuggingFace ``transformers`` image encoders (``AutoModel``) - e.g. Midnight.

A non-timm family for pathology models published as transformers checkpoints. It
loads via ``AutoModel.from_pretrained`` and pools the ``last_hidden_state`` per
``spec.pooling`` (CLS, or CLS + mean-patch concat like Virchow2/Midnight). The
``transformers`` import is deferred to :meth:`load`, so the dependency stays soft
and the guarded entry-point loader skips the family cleanly when it is absent.

``spec.timm_kwargs`` doubles as the generic load-kwargs bag here (e.g.
``trust_remote_code: true`` for models that ship custom modeling code).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Embedder

if TYPE_CHECKING:  # pragma: no cover
    import torch


@register("embedders", "transformers")
class TransformersEmbedder(Embedder):
    """transformers ``AutoModel`` image tower; pools ``last_hidden_state``."""

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        compile: bool = False,
    ) -> TransformersEmbedder:
        import torch

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                'transformers models need: pip install "raw2features[models]"'
            ) from exc

        trust_remote_code = bool(self.spec.timm_kwargs.get("trust_remote_code", False))
        try:
            model = AutoModel.from_pretrained(
                self.spec.source,
                trust_remote_code=trust_remote_code,
                revision=self.spec.weights_revision,  # pin the immutable HF commit
            )
        except (ImportError, ModuleNotFoundError) as exc:
            # e.g. Hibou's remote code imports the removed transformers.onnx (gone in
            # transformers 5). Point at the pinned extra rather than the cryptic error.
            if trust_remote_code and "onnx" in str(exc):
                raise ImportError(
                    f"{self.spec.name}: its trust_remote_code modeling needs "
                    f'transformers<5. Install: pip install "raw2features[hibou]".'
                ) from exc
            raise
        model.eval().to(device)
        self._model = model
        self._device = device
        self._dtype = dtype or torch.float32
        self._maybe_compile(compile)
        return self

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        from .base import _forward_ctx

        with _forward_ctx(self._device, self._dtype):
            out = self._model(pixel_values=batch.to(self._device))
        return self._pool(out.last_hidden_state).float().cpu()

    def _pool(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Reduce ``last_hidden_state`` ``[B, T, D]`` to one vector per item."""
        import torch

        if self.spec.pooling == "cls_concat_meanpatch":
            cls = last_hidden_state[:, 0]
            patches = last_hidden_state[:, 1 + self.spec.reg_tokens :]
            return torch.cat([cls, patches.mean(dim=1)], dim=-1)
        if self.spec.pooling in ("cls", "pooled"):
            return last_hidden_state[:, 0]
        raise ValueError(
            f"transformers family: unsupported pooling {self.spec.pooling!r}"
        )
