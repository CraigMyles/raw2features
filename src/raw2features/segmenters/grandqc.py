"""GrandQC tissue segmenter (opt-in ``[grandqc]`` extra).

Wraps GrandQC's stage-1 tissue UNet++ as a ``--segmenter grandqc`` plugin -- a deep
alternative to the classical otsu/canny segmenters for dirty H&E. Weights are
CC-BY-NC-SA (non-commercial), fetched on first use and never bundled; see
``docs/SEGMENTATION.md`` / ``docs/MODEL_LICENSES.md``. The inference lives in
:mod:`raw2features.qc.grandqc`, shared with the ``--qc grandqc`` artifact stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raw2features.core.plugins import register

from .base import Segmenter, TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource


@register("segmenters", "grandqc")
class GrandQCSegmenter(Segmenter):
    """Tissue detection via GrandQC's stage-1 UNet++ (needs the ``[grandqc]`` extra)."""

    name = "grandqc"

    def __init__(self, device: str = "auto") -> None:
        # "auto" resolves to the GPU when available; GrandQC is a UNet++ that runs on
        # three-channel (RGB) patches, so it benefits from a GPU.
        self.device = device

    def segment(self, reader: WSISource) -> TissueMask:
        from raw2features.core.device import resolve_device
        from raw2features.qc.grandqc import GrandQC

        return GrandQC(device=resolve_device(self.device)).tissue_mask(reader)
