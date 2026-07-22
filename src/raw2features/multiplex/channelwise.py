"""Channel-wise adaptation of ordinary RGB encoders to named multiplex images.

The RGB vision-encoder baselines evaluated in the public KRONOS paper provide
structural inspiration. This strategy adapts ordinary RGB encoders rather than
implementing the marker-aware KRONOS architecture. Every marker is normalized
independently, repeated to RGB, embedded independently, then aggregated with a fully
recorded deterministic recipe.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from raw2features.core.plugins import register
from raw2features.embedders.base import ModelSpec

from .base import BoundMultiplexStrategy, MultiplexStrategy, PreparedMultiplexStrategy


def _value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _identity(name: str) -> str:
    return unicodedata.normalize("NFKC", str(name)).strip().casefold()


def _json_digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_safe_model_segment(name: Any) -> str:
    value = str(name)
    if (
        not value
        or value in {".", ".."}
        or any(character in value for character in ("/", "\\", "\x00"))
    ):
        raise ValueError(f"base model name {value!r} is not a safe Zarr path segment")
    return value


class PreparedChannelwise(PreparedMultiplexStrategy):
    @property
    def context_max_side_px(self) -> int:
        return self._context_max_side_px

    @property
    def base_inputs_per_patch(self) -> int:
        return len(self.marker_records)

    def __init__(self, base_embedder: Any, channel_names, channel_count: int, config):
        if getattr(base_embedder, "modality", "brightfield") != "brightfield":
            raise ValueError(
                f"channelwise requires an ordinary RGB encoder; "
                f"{base_embedder.name!r} is {base_embedder.modality!r}"
            )
        if len(channel_names) != int(channel_count):
            raise ValueError(
                "multiplex marker metadata does not match the physical C axis: "
                f"omero.channels has {len(channel_names)} entries but C={channel_count}"
            )

        base_name = _require_safe_model_segment(base_embedder.name)
        panel = ["" if value is None else str(value) for value in channel_names]
        self.channel_count = int(channel_count)
        self.source_panel = panel
        self.source_panel_records = [
            {
                "source_index": index,
                "source_name": name,
                "canonical_name": unicodedata.normalize("NFKC", name).strip(),
            }
            for index, name in enumerate(panel)
        ]
        requested = list(_value(config, "multiplex_markers", []) or [])
        requested = [str(value).strip() for value in requested]
        if any(not value for value in requested):
            raise ValueError("selected multiplex marker names must be non-empty")

        positions: dict[str, list[int]] = {}
        for index, source_name in enumerate(panel):
            identity = _identity(source_name)
            if identity:
                positions.setdefault(identity, []).append(index)

        if requested:
            records = []
            seen: set[str] = set()
            for requested_name in requested:
                key = _identity(requested_name)
                if key in seen:
                    raise ValueError(
                        f"multiplex marker {requested_name!r} was requested more "
                        "than once"
                    )
                seen.add(key)
                matches = positions.get(key, [])
                if not matches:
                    raise ValueError(
                        f"requested multiplex marker {requested_name!r} is not present"
                    )
                if len(matches) != 1:
                    raise ValueError(
                        f"requested multiplex marker {requested_name!r} is ambiguous; "
                        f"it matches source channels {matches}"
                    )
                index = matches[0]
                records.append(
                    {
                        "source_index": index,
                        "source_name": panel[index],
                        "canonical_name": unicodedata.normalize(
                            "NFKC", panel[index]
                        ).strip(),
                    }
                )
        else:
            unnamed = [index for index, name in enumerate(panel) if not _identity(name)]
            if unnamed:
                raise ValueError(
                    "default marker selection requires every channel to be named; "
                    f"unnamed source channel indices: {unnamed}. Pass --marker "
                    "explicitly to exclude them."
                )
            duplicate = {
                key: indices for key, indices in positions.items() if len(indices) > 1
            }
            if duplicate:
                raise ValueError(
                    "default marker selection is ambiguous because normalized marker "
                    f"names are duplicated: {duplicate}"
                )
            records = [
                {
                    "source_index": index,
                    "source_name": name,
                    "canonical_name": unicodedata.normalize("NFKC", name).strip(),
                }
                for index, name in enumerate(panel)
            ]

        normalization = str(
            _value(config, "multiplex_normalization", "percentile")
        ).lower()
        if normalization not in {"percentile", "dtype"}:
            raise ValueError(
                "unknown multiplex normalization "
                f"{normalization!r}; available: percentile, dtype"
            )
        aggregation = str(_value(config, "multiplex_aggregation", "mean")).lower()
        if aggregation not in {"mean", "concat"}:
            raise ValueError(
                f"unknown multiplex aggregation {aggregation!r}; "
                "available: mean, concat"
            )
        if aggregation == "concat" and not requested:
            raise ValueError(
                "channelwise concat requires an explicit ordered --marker list"
            )
        low = float(_value(config, "multiplex_percentile_low", 1.0))
        high = float(_value(config, "multiplex_percentile_high", 99.0))
        if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
            raise ValueError(
                "multiplex percentiles must satisfy 0 <= low < high <= 100"
            )
        max_side = _value(config, "multiplex_normalization_max_side_px", 2048)
        if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side <= 0:
            raise ValueError(
                "multiplex_normalization_max_side_px must be a positive integer"
            )
        strategy_params = dict(_value(config, "multiplex_strategy_params", {}) or {})
        if strategy_params:
            raise ValueError(
                "channelwise does not accept --multiplex-params; use its explicit "
                "marker, normalization, and aggregation options"
            )

        self.base_embedder = base_embedder
        self.marker_records = records
        self.selected_indices = [int(r["source_index"]) for r in records]
        self.normalization = normalization
        self.percentile_low = low
        self.percentile_high = high
        self.aggregation = aggregation
        self._context_max_side_px = max_side
        batch_limit = _value(config, "batch_size", 256)
        if (
            isinstance(batch_limit, bool)
            or not isinstance(batch_limit, int)
            or batch_limit <= 0
        ):
            raise ValueError("channelwise requires a positive integer batch_size")
        self.base_image_batch_limit = batch_limit
        input_contract = str(getattr(base_embedder, "transform_input_dtype", "uint8"))
        if input_contract not in {"uint8", "float32_0_1"}:
            raise ValueError(
                f"channelwise does not support base transform input contract "
                f"{input_contract!r}; expected 'uint8' or 'float32_0_1'"
            )
        self.transform_input_dtype = input_contract
        # The effective key represents cohort-level configuration, not how a marker
        # happened to be capitalised in one source's metadata. Exact source spelling
        # and indices remain in each slide's full fingerprint/provenance below.
        marker_request = {
            "mode": "explicit" if requested else "all_named_source_order",
            "markers": [_identity(record["canonical_name"]) for record in records],
        }
        quantization = (
            {
                "dtype": "uint8",
                "scale": 255,
                "clipping": "unit_interval",
                "rounding": "numpy.rint_ties_to_even",
            }
            if input_contract == "uint8"
            else {
                "dtype": "float32",
                "range": "unit_interval",
                "clipping": "unit_interval",
                "rounding": "none",
            }
        )
        self._config_metadata = {
            "strategy": "channelwise",
            "contract_version": ChannelwiseStrategy.contract_version,
            "strategy_params": strategy_params,
            "base_model": base_name,
            "base_pooling": base_embedder.spec.pooling,
            "marker_request": marker_request,
            "normalization": {
                "name": normalization,
                "scope": "whole_slide_downsample_level",
                "nonfinite_policy": "error",
                "statistics_compute_dtype": "float64",
                "applied_bounds_dtype": "float32",
                "patch_compute_dtype": "float32",
                "level_selection": {
                    "rule": "first_pyramid_level_with_max_side_lte",
                    "max_side_px": self.context_max_side_px,
                },
                **(
                    {"lower_percentile": low, "upper_percentile": high}
                    if normalization == "percentile"
                    else {}
                ),
            },
            "channel_expansion": "repeat_single_channel_to_rgb",
            "execution_policy": {
                "configured_batch_unit": "base_encoder_rgb_inputs",
                "patch_batch_rule": (
                    "max(1,floor(configured_batch_size/selected_marker_count))"
                ),
                "base_transform_and_forward_chunking": (
                    "at_most_configured_batch_size_rgb_images"
                ),
            },
            "encoder_input_contract": input_contract,
            "quantization": quantization,
            "encoder_output": {
                "pooling": base_embedder.spec.pooling,
                "dimension": int(base_embedder.embedding_dim),
            },
            "aggregation": {"name": aggregation},
        }
        suffix = _json_digest(self._config_metadata)[:16]
        self._effective_model_key = f"{base_name}__channelwise_{aggregation}_{suffix}"

    @property
    def effective_model_key(self) -> str:
        return self._effective_model_key

    @property
    def config_metadata(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._config_metadata))

    def resolve_slide_context(
        self, image_hwc: np.ndarray, *, level: int, source_dtype: Any
    ) -> dict[str, Any]:
        image = np.asarray(image_hwc)
        if image.ndim != 3 or image.shape[2] != self.channel_count:
            raise ValueError(
                "normalization image must be HWC with exactly the source C-axis "
                f"extent ({self.channel_count}); got shape {image.shape}"
            )
        dtype = np.dtype(source_dtype)
        if image.dtype != dtype:
            raise ValueError(
                f"normalization image dtype {image.dtype} does not match declared "
                f"source dtype {dtype}"
            )
        resolved = []
        for record in self.marker_records:
            index = int(record["source_index"])
            values = image[..., index].astype(np.float64, copy=False)
            if not np.isfinite(values).all():
                raise ValueError(
                    f"marker {record['source_name']!r} contains non-finite pixels; "
                    "channelwise normalization has nonfinite_policy='error'"
                )
            if self.normalization == "percentile":
                low, high = np.percentile(
                    values,
                    [self.percentile_low, self.percentile_high],
                    method="linear",
                )
            elif np.issubdtype(dtype, np.bool_):
                low, high = (0.0, 1.0)
            elif np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                low, high = (float(info.min), float(info.max))
            else:
                observed_low = float(values.min())
                observed_high = float(values.max())
                if observed_low < 0.0 or observed_high > 1.0:
                    raise ValueError(
                        "dtype normalization for floating-point multiplex images "
                        "requires finite values in [0, 1]; observed "
                        f"[{observed_low:g}, {observed_high:g}] for marker "
                        f"{record['source_name']!r}"
                    )
                low, high = (0.0, 1.0)
            if not (math.isfinite(float(low)) and math.isfinite(float(high))):
                raise ValueError(
                    f"marker {record['source_name']!r} normalization is not finite"
                )
            if float(high) <= float(low):
                raise ValueError(
                    f"marker {record['source_name']!r} has degenerate normalization "
                    f"range ({float(low):g}, {float(high):g})"
                )
            # Patch normalization runs in float32. Persist the exact cast bounds that
            # are applied rather than higher-precision statistics that never reach the
            # encoder input.
            low = float(np.float32(low))
            high = float(np.float32(high))
            if high <= low:
                raise ValueError(
                    f"marker {record['source_name']!r} normalization range collapses "
                    "when represented as float32"
                )
            resolved.append(
                {
                    **record,
                    "low": float(low),
                    "high": float(high),
                }
            )
        return {
            "name": self.normalization,
            "scope": "whole_slide_downsample_level",
            "level_selection": {
                "rule": "first_pyramid_level_with_max_side_lte",
                "max_side_px": self.context_max_side_px,
            },
            "source_level": int(level),
            "source_level_shape_hwc": [
                int(image.shape[0]),
                int(image.shape[1]),
                int(image.shape[2]),
            ],
            "source_dtype": dtype.str,
            "nonfinite_policy": "error",
            "clipping": "unit_interval",
            "statistics_compute_dtype": "float64",
            "applied_bounds_dtype": "float32",
            "patch_compute_dtype": "float32",
            "percentile_method": (
                "numpy_linear" if self.normalization == "percentile" else None
            ),
            "lower_percentile": (
                self.percentile_low if self.normalization == "percentile" else None
            ),
            "upper_percentile": (
                self.percentile_high if self.normalization == "percentile" else None
            ),
            "resolved": resolved,
        }

    def bind(self, resolved: Any) -> BoundChannelwise:
        return BoundChannelwise(self, resolved)


class BoundChannelwise(BoundMultiplexStrategy):
    def __init__(self, prepared: PreparedChannelwise, resolved: dict[str, Any]):
        self.prepared = prepared
        self.base_embedder = prepared.base_embedder
        self._normalization = json.loads(json.dumps(resolved))
        if self._normalization.get("name") != prepared.normalization:
            raise ValueError(
                "resolved normalization method does not match the prepared strategy"
            )
        if self._normalization.get("nonfinite_policy") != "error":
            raise ValueError(
                "resolved normalization must retain nonfinite_policy='error'"
            )
        expected_contract = {
            "scope": "whole_slide_downsample_level",
            "level_selection": {
                "rule": "first_pyramid_level_with_max_side_lte",
                "max_side_px": prepared.context_max_side_px,
            },
            "clipping": "unit_interval",
            "statistics_compute_dtype": "float64",
            "applied_bounds_dtype": "float32",
            "patch_compute_dtype": "float32",
        }
        if any(
            self._normalization.get(field) != value
            for field, value in expected_contract.items()
        ):
            raise ValueError(
                "resolved normalization execution contract does not match "
                "channelwise v1"
            )
        level = self._normalization.get("source_level")
        if isinstance(level, bool) or not isinstance(level, int) or level < 0:
            raise ValueError("resolved normalization has no valid source level")
        shape = self._normalization.get("source_level_shape_hwc")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in shape
            )
            or any(value <= 0 for value in shape)
            or shape[2] != prepared.channel_count
        ):
            raise ValueError(
                "resolved normalization has no valid source-level HWC shape"
            )
        if prepared.normalization == "percentile":
            if (
                self._normalization.get("percentile_method") != "numpy_linear"
                or self._normalization.get("lower_percentile")
                != prepared.percentile_low
                or self._normalization.get("upper_percentile")
                != prepared.percentile_high
            ):
                raise ValueError(
                    "resolved percentile recipe does not match the prepared strategy"
                )
        elif any(
            self._normalization.get(field) is not None
            for field in (
                "percentile_method",
                "lower_percentile",
                "upper_percentile",
            )
        ):
            raise ValueError("dtype normalization must not carry a percentile recipe")
        try:
            self._source_dtype = np.dtype(self._normalization["source_dtype"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "resolved normalization has no valid source dtype"
            ) from exc
        rows = self._normalization.get("resolved", [])
        if len(rows) != len(prepared.marker_records):
            raise ValueError("resolved normalization does not match selected markers")
        lows: list[float] = []
        highs: list[float] = []
        for expected, row in zip(prepared.marker_records, rows, strict=True):
            if not isinstance(row, Mapping) or any(
                row.get(field) != expected[field]
                for field in ("source_index", "source_name", "canonical_name")
            ):
                raise ValueError(
                    "resolved normalization marker order/identity does not match "
                    f"the prepared panel at {expected!r}"
                )
            try:
                low = float(row["low"])
                high = float(row["high"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"resolved normalization for {expected['source_name']!r} "
                    "has no numeric low/high range"
                ) from exc
            if not (math.isfinite(low) and math.isfinite(high) and high > low):
                raise ValueError(
                    f"resolved normalization for {expected['source_name']!r} must "
                    "have finite low < high"
                )
            lows.append(low)
            highs.append(high)
        self._low = np.asarray(lows, dtype=np.float32)
        self._high = np.asarray(highs, dtype=np.float32)
        base_dim = int(self.base_embedder.embedding_dim)
        self._embedding_dim = (
            base_dim * len(rows) if prepared.aggregation == "concat" else base_dim
        )
        spec: ModelSpec = replace(
            self.base_embedder.spec,
            name=prepared.effective_model_key,
            embedding_dim=self._embedding_dim,
            modality="multiplex",
        )
        super().__init__(spec)
        self._device = getattr(self.base_embedder, "_device", "cpu")
        self._dtype = getattr(self.base_embedder, "_dtype", None)
        # The runner sets ownership after binding. Caller-injected warm embedders are
        # deliberately never owned by this wrapper and must survive across slides.
        self._raw2features_owns_base = False
        self._base_load_attempted_by_wrapper = False

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def modality(self) -> str:
        return "multiplex"

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def transform_signature(self) -> tuple[Any, ...]:
        # Never share a transformed tensor across a different panel/statistics recipe.
        return ("multiplex", self.name, _json_digest(self.strategy_metadata))

    @property
    def strategy_metadata(self) -> dict[str, Any]:
        input_conversion = (
            "quantize_uint8"
            if self.prepared.transform_input_dtype == "uint8"
            else "cast_float32_unit_interval"
        )
        return {
            **self.prepared.config_metadata,
            "markers": json.loads(json.dumps(self.prepared.marker_records)),
            "normalization": json.loads(json.dumps(self._normalization)),
            "output_dimension": self.embedding_dim,
            "operation_order": [
                "select_marker",
                "normalize_to_unit_interval",
                input_conversion,
                "repeat_to_rgb",
                "base_encoder_preprocessing",
                "base_encoder_output",
                "aggregate_markers",
            ],
        }

    @property
    def panel_metadata(self) -> dict[str, Any]:
        selected = {int(row["source_index"]) for row in self.prepared.marker_records}
        return {
            "strategy": "channelwise",
            "physical_channel_count": self.prepared.channel_count,
            "n_markers": len(selected),
            "kept": [row["source_name"] for row in self.prepared.marker_records],
            "excluded": json.loads(
                json.dumps(
                    [
                        row
                        for row in self.prepared.source_panel_records
                        if int(row["source_index"]) not in selected
                    ]
                )
            ),
            "source_channels": json.loads(
                json.dumps(self.prepared.source_panel_records)
            ),
            "mapping": json.loads(json.dumps(self.prepared.marker_records)),
            "normalization": json.loads(json.dumps(self._normalization)),
        }

    def multiplex_fingerprint_payload(
        self, base_output_fingerprint: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(base_output_fingerprint, Mapping):
            raise ValueError(
                "channelwise fingerprint requires the full base fingerprint"
            )
        return {
            **self.strategy_metadata,
            "base_output_fingerprint": dict(base_output_fingerprint),
        }

    def set_panel(self, channel_names) -> dict[str, Any]:
        values = [] if channel_names is None else channel_names
        panel = ["" if value is None else str(value) for value in values]
        if panel != self.prepared.source_panel:
            raise ValueError(
                "source marker panel changed after channelwise strategy binding; "
                f"expected {self.prepared.source_panel!r}, got {panel!r}"
            )
        return self.panel_metadata

    def load(self, device="cuda", dtype=None, compile=False):
        if self._raw2features_owns_base:
            # Mark before loading so a partially failed load is still cleaned up by
            # the runner's finally block.
            self._base_load_attempted_by_wrapper = True
        self.base_embedder.load(device=device, dtype=dtype, compile=compile)
        self._device = device
        self._dtype = dtype
        return self

    def unload(self) -> None:
        if self._raw2features_owns_base and self._base_load_attempted_by_wrapper:
            self.base_embedder.unload()
            self._base_load_attempted_by_wrapper = False

    def transform_batch(self, patches_hwc: list[np.ndarray], device: str):
        if not patches_hwc:
            raise ValueError("channelwise transform requires at least one patch")
        indices = self.prepared.selected_indices
        rgb: list[np.ndarray] = []
        for patch_index, patch in enumerate(patches_hwc):
            patch = np.asarray(patch)
            if patch.ndim != 3 or int(patch.shape[2]) != self.prepared.channel_count:
                raise ValueError(
                    f"multiplex patch {patch_index} must be HWC with "
                    f"C={self.prepared.channel_count}; got shape {patch.shape}"
                )
            if patch.dtype != self._source_dtype:
                raise ValueError(
                    f"multiplex patch {patch_index} dtype {patch.dtype} does not "
                    f"match bound source dtype {self._source_dtype}"
                )
            selected = patch[..., indices].astype(np.float32)
            if not np.isfinite(selected).all():
                raise ValueError(
                    f"multiplex patch {patch_index} contains non-finite selected "
                    "marker pixels; channelwise normalization has "
                    "nonfinite_policy='error'"
                )
            if (
                self.prepared.normalization == "dtype"
                and np.issubdtype(self._source_dtype, np.floating)
                and (float(selected.min()) < 0.0 or float(selected.max()) > 1.0)
            ):
                raise ValueError(
                    f"multiplex patch {patch_index} violates float dtype "
                    "normalization: selected marker pixels must be in [0, 1]"
                )
            unit = (selected - self._low) / (self._high - self._low)
            unit = np.clip(unit, 0.0, 1.0)
            if self.prepared.transform_input_dtype == "uint8":
                encoder_input = np.rint(unit * 255.0).astype(np.uint8)
            else:
                encoder_input = unit.astype(np.float32, copy=False)
            for marker in range(encoder_input.shape[2]):
                rgb.append(np.repeat(encoder_input[..., marker, None], 3, axis=2))
        chunks = [
            self.base_embedder.transform_batch(
                rgb[start : start + self.prepared.base_image_batch_limit], device
            )
            for start in range(0, len(rgb), self.prepared.base_image_batch_limit)
        ]
        transformed = chunks[0]
        if len(chunks) > 1:
            try:
                import torch

                if torch.is_tensor(transformed):
                    transformed = torch.cat(chunks, dim=0)
                else:
                    transformed = np.concatenate(chunks, axis=0)
            except ImportError:
                transformed = np.concatenate(chunks, axis=0)
        if int(transformed.shape[0]) != len(rgb):
            raise ValueError(
                "base transform output row count does not match expanded marker "
                f"images ({int(transformed.shape[0])} != {len(rgb)})"
            )
        return transformed.reshape(
            len(patches_hwc), len(indices), *transformed.shape[1:]
        )

    def embed_batch(self, batch):
        import torch

        if getattr(batch, "ndim", 0) < 3:
            raise ValueError(
                "channelwise transformed batch must have batch, marker, and base "
                "transform dimensions"
            )
        batch_size, markers = int(batch.shape[0]), int(batch.shape[1])
        expected_markers = len(self.prepared.selected_indices)
        if batch_size <= 0 or markers != expected_markers:
            raise ValueError(
                "channelwise transformed batch marker count does not match the "
                f"bound panel ({markers} != {expected_markers})"
            )
        flat = batch.reshape(batch_size * markers, *batch.shape[2:])
        embedded_parts = [
            self.base_embedder.embed_batch(
                flat[start : start + self.prepared.base_image_batch_limit]
            ).float()
            for start in range(
                0,
                int(flat.shape[0]),
                self.prepared.base_image_batch_limit,
            )
        ]
        embedded = torch.cat(embedded_parts, dim=0)
        expected_rows = batch_size * markers
        expected_dim = int(self.base_embedder.embedding_dim)
        if embedded.ndim != 2 or tuple(embedded.shape) != (
            expected_rows,
            expected_dim,
        ):
            raise ValueError(
                "base encoder output does not match the bound channelwise contract: "
                f"expected ({expected_rows}, {expected_dim}), got "
                f"{tuple(embedded.shape)}"
            )
        marker_embeddings = embedded.reshape(batch_size, markers, -1)
        aggregation = self.prepared.aggregation
        if aggregation == "mean":
            output = marker_embeddings.mean(dim=1)
        elif aggregation == "concat":
            output = marker_embeddings.reshape(batch_size, -1)
        else:
            raise RuntimeError(
                f"unbound channelwise aggregation {aggregation!r}; "
                "expected mean or concat"
            )
        return output.float().cpu()


@register("multiplex_strategies", "channelwise")
class ChannelwiseStrategy(MultiplexStrategy):
    name = "channelwise"
    contract_version = 1

    def prepare(
        self,
        *,
        base_embedder,
        channel_names,
        channel_count: int,
        config=None,
    ) -> PreparedChannelwise:
        if channel_names is None:
            raise ValueError(
                "channelwise requires positional marker names in omero.channels"
            )
        return PreparedChannelwise(base_embedder, channel_names, channel_count, config)
