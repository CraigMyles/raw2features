"""``WSISource`` - the reader seam.

A reader exposes a pyramidal whole-slide image as level dimensions, downsamples,
a level-0 microns-per-pixel, and ``read_region``. Coordinates follow the OpenSlide
convention: ``location`` is in level-0 pixels, ``size`` in the read level's pixels.
``read_region`` always returns an HWC uint8 RGB ndarray.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from raw2features.core.geometry import Region, Size
from raw2features.core.mpp import LevelChoice, level_for_mpp


class WSISource(ABC):
    """Abstract pyramidal WSI reader."""

    name: str = "wsisource"

    def __init__(self, path: str) -> None:
        self.path = str(path)

    @abstractmethod
    def open(self) -> WSISource:
        """Open the source and return ``self``."""

    @abstractmethod
    def close(self) -> None:
        """Release any handles."""

    def __enter__(self) -> WSISource:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    @abstractmethod
    def mpp(self) -> float | None:
        """Microns-per-pixel at level 0, or ``None`` if unknown."""

    @property
    @abstractmethod
    def level_dimensions(self) -> list[Size]:
        """``(width, height)`` per pyramid level, level 0 first."""

    @abstractmethod
    def level_downsamples(self) -> list[float]:
        """Downsample factor per level relative to level 0 (level 0 == 1.0)."""

    def read_level_mapping(self, level: int) -> tuple[float, float, float, float]:
        """Per-axis ``(downsample_x, downsample_y, offset_x, offset_y)`` for ``level``.

        Maps a level-0 ``(x, y)`` to this level's array index as
        ``round(x / downsample_x + offset_x)`` (and likewise for ``y``). The default
        is isotropic with no translation - ``(ds, ds, 0, 0)`` from
        ``level_downsamples()`` - so existing readers behave exactly as before. A
        reader whose source declares per-axis scale (anisotropy) or a per-level
        ``translation`` overrides this so the block-read fast path and ``read_region``
        share one mapping. The block path consults this when present.
        """
        ds = float(self.level_downsamples()[level])
        return ds, ds, 0.0, 0.0

    @abstractmethod
    def read_region(self, region: Region) -> np.ndarray:
        """Read ``region``; return an HWC uint8 RGB ndarray of ``region.size``."""

    # -- multi-channel / multiplex (optional; brightfield readers need not) --
    @property
    def channel_names(self) -> list[str] | None:
        """Channel/marker names for multiplex sources (e.g. CODEX), else ``None``.

        ``None`` means a plain RGB brightfield source (H&E) - the default. A
        multiplex reader returns one name per channel (the marker panel), which the
        multiplex embedders (e.g. KRONOS) map to per-marker normalisation + IDs.
        """
        return None

    def read_region_channels(self, region: Region) -> np.ndarray:
        """Native multi-channel read -> ``(h, w, C)`` in the source dtype.

        Unlike :meth:`read_region` (which collapses to 8-bit RGB for brightfield
        models), this preserves every channel and the native dtype, for multiplex
        models. Override in multi-channel-capable readers.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support multi-channel reads"
        )

    def level_for_mpp(
        self, target_mpp: float, patch_px: int, **kwargs: object
    ) -> LevelChoice:
        """Resolve the exact-MPP read plan for this source (see ``core.mpp``)."""
        if self.mpp is None:
            raise ValueError(
                f"{self.path}: source MPP is unknown; cannot target {target_mpp} um/px"
            )
        return level_for_mpp(
            target_mpp,
            self.mpp,
            self.level_downsamples(),
            patch_px,
            **kwargs,  # type: ignore[arg-type]
        )
