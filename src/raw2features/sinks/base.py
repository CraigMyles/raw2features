"""``Sink`` - the output seam.

Streaming, per-block writes so a slide never holds all features in RAM. The
spatial-provenance contract (1:1 ``coords``/``grid_index`` <-> ``features`` rows)
is the sink's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Sink(ABC):
    """Abstract per-slide embedding writer."""

    name: str = "sink"

    @abstractmethod
    def create(
        self,
        out_dir: str,
        slide_id: str,
        *,
        n_patches: int,
        coords: np.ndarray,
        grid_index: np.ndarray,
        grid_tissue: np.ndarray | None,
        model_dims: dict[str, int],
        header: dict,
        features_dtype: str = "float16",
    ) -> None:
        """Create the output store, write coords/grid_index/mask + header, and
        allocate one ``features/<model>`` array per model."""

    @abstractmethod
    def write_block(self, model: str, start: int, feats: np.ndarray) -> None:
        """Write ``feats`` (B, dim) at patch offset ``start`` for ``model``."""

    @abstractmethod
    def close(self) -> None:
        """Finalise (consolidate metadata)."""

    @property
    @abstractmethod
    def uri(self) -> str:
        """Filesystem URI of the created store."""
