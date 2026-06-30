"""Simple pooling slide encoders (mean / max / concat).

These require no model weights and no HuggingFace access. They are useful
as fast baselines and for testing the slide-embedding pipeline end-to-end
without gated models.

``mean``   - global mean pool across patches: (N, dim) → (dim,)
``max``    - global max pool across patches:  (N, dim) → (dim,)
``meanmax``- concat mean and max:             (N, dim) → (2·dim,)
"""

from __future__ import annotations

import numpy as np

from raw2features.core.plugins import register

from .base import SlideEmbedder, SlideModelSpec


def _make_spec(
    name: str, dim: int, patch_encoder: str, patch_dim: int
) -> SlideModelSpec:
    return SlideModelSpec(
        name=name,
        family="pool",
        source="builtin",
        embedding_dim=dim,
        patch_encoder=patch_encoder,
        patch_dim=patch_dim,
        gated=False,
        license="MIT",
        transform_source_url="https://github.com/CraigMyles/raw2features",
        notes=f"Simple {name} pooling - no weights, no HF token required.",
    )


class _PoolEmbedder(SlideEmbedder):
    def load(self, device="cpu", dtype=None) -> _PoolEmbedder:
        return self

    def encode(self, features: np.ndarray, coords=None, patch_size_lv0=None):
        raise NotImplementedError


@register("slide_embedders", "mean")
class MeanPoolSlideEmbedder(_PoolEmbedder):
    """Global mean pool: (N, dim) → (dim,)."""

    def __init__(self, patch_encoder: str = "resnet50", patch_dim: int = 2048) -> None:
        super().__init__(_make_spec("mean", patch_dim, patch_encoder, patch_dim))

    def encode(self, features: np.ndarray, coords=None, patch_size_lv0=None):
        return features.astype(np.float32).mean(axis=0)


@register("slide_embedders", "max")
class MaxPoolSlideEmbedder(_PoolEmbedder):
    """Global max pool: (N, dim) → (dim,)."""

    def __init__(self, patch_encoder: str = "resnet50", patch_dim: int = 2048) -> None:
        super().__init__(_make_spec("max", patch_dim, patch_encoder, patch_dim))

    def encode(self, features: np.ndarray, coords=None, patch_size_lv0=None):
        return features.astype(np.float32).max(axis=0)


@register("slide_embedders", "meanmax")
class MeanMaxPoolSlideEmbedder(_PoolEmbedder):
    """Concat of mean and max: (N, dim) → (2·dim,)."""

    def __init__(self, patch_encoder: str = "resnet50", patch_dim: int = 2048) -> None:
        super().__init__(_make_spec("meanmax", 2 * patch_dim, patch_encoder, patch_dim))

    def encode(self, features: np.ndarray, coords=None, patch_size_lv0=None):
        f = features.astype(np.float32)
        return np.concatenate([f.mean(axis=0), f.max(axis=0)])
