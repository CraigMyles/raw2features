"""Interfaces for adapting ordinary RGB patch encoders to multiplex images.

A multiplex strategy has two phases.  ``prepare`` validates the source marker panel
and resolves every configuration choice that is shared across a cohort.  ``bind``
then attaches slide-specific inputs (for example resolved intensity percentiles).
Keeping those phases separate lets output array names remain stable across slides
while the complete per-slide production contract is still fingerprinted.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from raw2features.embedders.base import Embedder


class BoundMultiplexStrategy(Embedder, ABC):
    """An embedder-compatible, fully resolved multiplex production contract."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Effective feature-array key for the strategy-derived output."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Width of the aggregated patch representation."""

    @property
    @abstractmethod
    def strategy_metadata(self) -> dict[str, Any]:
        """JSON-safe, fully resolved strategy provenance."""

    @property
    @abstractmethod
    def panel_metadata(self) -> dict[str, Any]:
        """JSON-safe positional marker-panel provenance for the store header."""

    @property
    @abstractmethod
    def transform_signature(self) -> tuple[Any, ...]:
        """Identity of every strategy and slide-specific transform input."""

    @abstractmethod
    def set_panel(self, channel_names: Sequence[str | None] | None) -> dict[str, Any]:
        """Verify that source marker metadata still matches the bound panel."""

    @abstractmethod
    def transform_batch(self, patches_hwc: list[Any], device: str) -> Any:
        """Adapt multiplex HWC patches to a marker-aware base-model batch."""

    @abstractmethod
    def multiplex_fingerprint_payload(
        self, base_output_fingerprint: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Return the strategy block for the model-output fingerprint.

        The complete, self-validating base fingerprint is required (a digest alone
        is insufficient). The runner supplies it after resolving the base encoder's
        effective AMP and any load-time preprocessing contract.
        """


class PreparedMultiplexStrategy(ABC):
    """A panel/config-bound strategy awaiting slide-specific resolved inputs."""

    @property
    @abstractmethod
    def effective_model_key(self) -> str:
        """Cohort-stable output key, independent of per-slide measured values."""

    @property
    @abstractmethod
    def config_metadata(self) -> dict[str, Any]:
        """JSON-safe strategy configuration used to derive the output key."""

    @property
    @abstractmethod
    def context_max_side_px(self) -> int:
        """Maximum side of the whole-slide context level requested from the reader."""

    @property
    @abstractmethod
    def base_inputs_per_patch(self) -> int:
        """Number of ordinary base-encoder inputs produced for one source patch."""

    @abstractmethod
    def resolve_slide_context(
        self,
        image_hwc: Any,
        *,
        level: int,
        source_dtype: Any,
    ) -> Any:
        """Resolve slide-specific state before completion checks or array creation."""

    @abstractmethod
    def bind(self, resolved: Any) -> BoundMultiplexStrategy:
        """Attach slide-specific inputs and return an embedder-compatible object."""


class MultiplexStrategy(ABC):
    """Pluggable strategy for using an RGB encoder on named multiplex channels.

    Future implementations can use the same seam for learned N-to-3 adapters or
    marker-triplet pseudo-RGB mappings.  A strategy owns its configuration and
    fingerprint schema; the runner only asks it to prepare and bind.
    """

    name: str
    contract_version: int

    @abstractmethod
    def prepare(
        self,
        *,
        base_embedder: Any,
        channel_names: Sequence[str | None],
        channel_count: int,
        config: Mapping[str, Any] | Any | None = None,
    ) -> PreparedMultiplexStrategy:
        """Validate a source panel and resolve the cohort-level strategy config."""

    def bind(
        self,
        *,
        base_embedder: Any,
        channel_names: Sequence[str | None],
        channel_count: int,
        resolved: Any,
        config: Mapping[str, Any] | Any | None = None,
    ) -> BoundMultiplexStrategy:
        """Convenience wrapper around ``prepare(...).bind(resolved)``."""

        return self.prepare(
            base_embedder=base_embedder,
            channel_names=channel_names,
            channel_count=channel_count,
            config=config,
        ).bind(resolved)
