"""``SlideEmbedder`` - the slide-level encoding seam.

A slide embedder takes the patch-level feature matrix for one slide and
returns a single slide-level vector. It reads from an already-written
``*.embeddings.zarr`` store - no pixel access, no WSI required.

The seam is intentionally separate from ``Embedder`` because the I/O
contract is fundamentally different: input is a ``(N, patch_dim)`` float
array from disk, not raw pixel patches from a WSI reader.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class SlideModelSpec:
    """Provenance descriptor for one slide-level encoder.

    ``patch_encoder`` names the patch-level model whose features this
    encoder was trained on (e.g. ``"conch_v1_5"`` for TITAN). The pipeline
    uses this to auto-select the right ``features/<patch_encoder>``
    array from the embeddings zarr.
    """

    name: str
    family: str
    source: str
    embedding_dim: int
    patch_encoder: str
    patch_dim: int
    gated: bool
    license: str
    transform_source_url: str
    notes: str = ""
    # Resolvable DOI for the model's paper (FAIR). None for the weight-free pooling
    # baselines. Flows into the slide-embedding provenance.
    doi: str | None = None
    # sha256 + pinned HuggingFace commit of the weights (None for pooling baselines).
    weights_sha256: str | None = None
    weights_revision: str | None = None


class SlideEmbedder(ABC):
    """Abstract slide-level encoder.

    Subclasses implement :meth:`encode` which receives the patch feature
    matrix for a single slide and returns one slide-level vector.
    """

    def __init__(self, spec: SlideModelSpec) -> None:
        self.spec = spec
        self.name = spec.name

    @abstractmethod
    def load(self, device: str = "cuda", dtype=None) -> SlideEmbedder:
        """Load weights and move to ``device``. Returns ``self``."""

    @abstractmethod
    def encode(
        self,
        features: np.ndarray,
        coords: np.ndarray | None = None,
        patch_size_lv0: int | None = None,
    ) -> np.ndarray:
        """Encode one slide.

        Parameters
        ----------
        features:
            ``(N, patch_dim)`` float32 patch feature matrix. The caller
            loads this from ``features/<patch_encoder>`` in the zarr and
            casts to float32.
        coords:
            ``(N, 2)`` int32 level-0 ``(x, y)`` coordinates, for encoders
            that use spatial position. May be ``None`` for pooling models.
        patch_size_lv0:
            Side of one patch in level-0 pixels (the store's
            ``patching.level0_patch``) - i.e. the spacing between adjacent
            patch coordinates at level 0. Position-aware encoders (e.g.
            TITAN, which builds a feature grid from ``coords`` and this
            spacing) require it; pooling models ignore it.

        Returns
        -------
        np.ndarray
            1-D float32 array of length ``spec.embedding_dim``.
        """

    def unload(self) -> None:  # noqa: B027 - intentional no-op default
        """Release GPU memory. Override if the model holds device tensors."""
