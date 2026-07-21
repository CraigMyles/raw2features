"""Nuclear-channel tissue segmenter for named-channel multiplex slides.

Multiplex slides have no H&E saturation/colour to threshold; instead, tissue is where
the nuclear stain (DAPI / Hoechst) is present. This segmenter finds the nuclear marker
channel(s) by name (via the reader's ``channel_names``), reads them at a low-res level,
and runs Otsu + morphology - the multiplex analogue of the default
Otsu-on-saturation segmenter. Pure OpenCV + numpy.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.mpp import nearest_level
from raw2features.core.plugins import register

from .base import Segmenter, TissueMask

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource

_NUCLEAR_ALIASES = ("dapi", "hoechst", "hochst", "dna")
# Numbered DNA channels occur both as plain labels (``DNA1``) and as delimited
# tokens in IMC exports (``191Ir_DNA1`` / ``Ir193_DNA2``). Keep the boundaries
# strict so biomarkers and prose containing ``DNA`` never match.
_DNA_NUMBERED_RE = re.compile(r"(?<![a-z])dna[\s_-]*([12])(?![a-z0-9])")


@register("segmenters", "nuclear")
class NuclearSegmenter(Segmenter):
    """Otsu on recognized DAPI/Hoechst/DNA channels in a multiplex slide."""

    name = "nuclear"

    def __init__(
        self,
        seg_mpp: float = 8.0,
        morph_kernel: int = 4,
        nuclear_aliases: tuple[str, ...] = _NUCLEAR_ALIASES,
    ) -> None:
        self.seg_mpp = seg_mpp
        self.morph_kernel = morph_kernel
        self.nuclear_aliases = tuple(
            unicodedata.normalize("NFKC", a).strip().casefold() for a in nuclear_aliases
        )

    def _pick_level(self, reader: WSISource) -> int:
        return nearest_level(reader.mpp, reader.level_downsamples(), self.seg_mpp)

    def _nuclear_matches(
        self, channel_names: list[str] | None
    ) -> list[tuple[int, str]]:
        if not channel_names:
            raise ValueError(
                "NuclearSegmenter needs a multiplex reader with channel_names "
                "(no recognized DAPI/Hoechst/DNA channel to threshold)"
            )
        matches: list[tuple[int, str]] = []
        for i, name in enumerate(channel_names):
            normalized = unicodedata.normalize("NFKC", str(name)).strip().casefold()
            match_kind = None
            for alias in self.nuclear_aliases:
                if alias == "dna":
                    if normalized == "dna":
                        match_kind = "dna"
                    else:
                        numbered_dna = _DNA_NUMBERED_RE.search(normalized)
                        if numbered_dna is not None:
                            match_kind = f"dna{numbered_dna.group(1)}"
                elif (
                    re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", normalized)
                    is not None
                ):
                    match_kind = alias
                if match_kind is not None:
                    break
            if match_kind is not None:
                matches.append((i, match_kind))
        if not matches:
            raise ValueError(
                "no nuclear channel "
                f"(aliases {self.nuclear_aliases}) found in the panel"
            )
        return matches

    def _nuclear_indices(self, channel_names: list[str] | None) -> list[int]:
        matches = self._nuclear_matches(channel_names)
        if len(matches) == 1:
            return [matches[0][0]]
        # Repeated acquisitions of one stain (for example DAPI1/DAPI2 in CODEX)
        # represent one nuclear signal. Numbered DNA1/DNA2 channels are likewise
        # one family and may contain a canonical pair or repeated acquisitions.
        # Never silently mix distinct stains such as DAPI and Hoechst/DNA.
        family_by_kind = {
            "hochst": "hoechst",
            "dna1": "dna-numbered",
            "dna2": "dna-numbered",
        }
        families = {family_by_kind.get(kind, kind) for _, kind in matches}
        if len(families) == 1:
            return [index for index, _ in matches]
        indices = [index for index, _ in matches]
        raise ValueError(
            "nuclear segmentation requires one nuclear stain family; matched mixed "
            f"families at physical C indices {indices}. Use --no-seg or provide "
            "unambiguous channel names."
        )

    def _nuclear_index(self, channel_names: list[str] | None) -> int:
        """Return the first bound index (legacy private helper)."""

        return self._nuclear_indices(channel_names)[0]

    def nuclear_channels(
        self, channel_names: list[str] | None
    ) -> list[tuple[int, str]]:
        """Return physical nuclear-channel bindings in source order."""

        indices = self._nuclear_indices(channel_names)
        assert channel_names is not None
        return [(index, channel_names[index]) for index in indices]

    def nuclear_channel(self, channel_names: list[str] | None) -> tuple[int, str]:
        """Return the first physical binding (kept for source compatibility)."""

        return self.nuclear_channels(channel_names)[0]

    @staticmethod
    def _combine_nuclear_channels(block: np.ndarray, indices: list[int]) -> np.ndarray:
        """Return one signal; repeated channels are averaged in float32."""

        if len(indices) == 1:
            # Preserve the v0.1 single-channel arithmetic exactly; OpenCV
            # normalizes in the source dtype before the established uint8 cast.
            return np.asarray(block[..., indices[0]])
        selected = np.asarray(block[..., indices], dtype=np.float32)
        return selected.mean(axis=2, dtype=np.float32)

    def segment(self, reader: WSISource) -> TissueMask:
        import cv2

        indices = self._nuclear_indices(reader.channel_names)
        level = self._pick_level(reader)
        dim: Size = reader.level_dimensions[level]
        block = reader.read_region_channels(
            Region(level=level, location=Point(0, 0), size=Size(dim.width, dim.height))
        )
        nuclear = self._combine_nuclear_channels(block, indices)
        # Scale the (possibly uint16) nuclear channel to 8-bit for Otsu.
        chan8 = cv2.normalize(nuclear, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, binary = cv2.threshold(chan8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        if self.morph_kernel > 0:
            k = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
        mask = (binary > 0).astype(np.float32)
        downsample = reader.level_downsamples()[level]
        return TissueMask(mask=mask, level=level, downsample=float(downsample))
