"""KRONOS2 native multiplex patch encoder.

KRONOS2 is distributed as a gated Hugging Face repository containing custom
``transformers`` code, a marker-aware DINOv2 ViT-B/16 checkpoint, and the marker
metadata that defines both normalisation and marker identity. raw2features downloads
only the files required for inference from one immutable revision into a managed cache,
verifies both content-affecting data artifacts, and gives only that local directory to
``AutoModel``. This prevents the pinned custom loader from silently resolving a newer
copy of its own weights.

Published markers use text embeddings already baked into the checkpoint.  The upstream
``register_additional_markers`` path runs BioLinkBERT for genuinely novel markers.
raw2features enables it only when the user supplies a complete, fingerprinted marker
table.  BioLinkBERT is then downloaded at an immutable revision, every model and
tokenizer artifact is verified, and the pinned upstream prompt/encoder path runs
locally in CPU float32.  Ordinary published-marker inference never downloads or
instantiates that additional 1.3 GB model.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import sys
import tempfile
import unicodedata
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from raw2features.core.plugins import register

from ._hub import (
    download_pinned_hf_snapshot,
    pinned_model_cache_dir,
    verify_sha256,
)
from .base import Embedder
from .kronos2_metadata import (
    KRONOS2_ADDITIONAL_MARKER_COLUMNS,
    KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION,
    KRONOS2_BIOLINKBERT_ARTIFACT_SHA256,
    KRONOS2_BIOLINKBERT_REVISION,
    KRONOS2_BIOLINKBERT_SOURCE,
    KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT,
    materialize_kronos2_additional_markers_csv,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


KRONOS2_MARKER_METADATA_FILENAME = "marker_metadata.csv"
KRONOS2_XFORMERS_VERSION = "0.0.29.post3"
KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST = (
    "config.json",
    "configuration_kronos2.py",
    "dinov2",
    KRONOS2_MARKER_METADATA_FILENAME,
    "marker_utils.py",
    "modeling_kronos2.py",
    "kronos2_vitb16_teacher.pth",
)
KRONOS2_SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "configuration_kronos2.py",
    "dinov2/**",
    KRONOS2_MARKER_METADATA_FILENAME,
    "marker_utils.py",
    "modeling_kronos2.py",
    "kronos2_vitb16_teacher.pth",
)
KRONOS2_SCALING_CONTRACT: dict[str, Any] = {
    "stage_1_dtype_divisor": {
        "uint8": 255.0,
        "other_unsigned_integer": 65535.0,
        "floating": 400.0,
    },
    "stage_2": "upstream_model.preprocess_per_marker_zscore",
    "arithmetic_dtype": "float32",
}
KRONOS2_FORWARD_CONTRACT: dict[str, Any] = {
    "backbone": {
        "architecture": "dinov2_vit_base",
        "patch_stride_px": 16,
        "register_tokens": 16,
        "positional_reference_size_px": 224,
    },
    "reference_inference_patch_size_px": 256,
    "accepted_patch_shape": "square_hwc_positive_side_multiple_of_16",
    "upstream_patch_stride_px": 16,
    "runner_max_batch_size": 8,
    "transform_staging_device": "cpu",
    "forward_microbatch_size": 8,
    "h2d_transfer": "one_forward_microbatch_at_a_time",
    "short_microbatch_padding": {
        "method": "repeat_last_sample",
        "target_rows": 8,
        "output": "slice_back_to_unpadded_row_count",
    },
    "output_order": "input_order",
    "attention_backend_policy": {
        "selection": "pinned_xformers_in_reference_environment_else_forced_fallback",
        "pinned_reference": {
            "implementation": "xformers.ops.memory_efficient_attention",
            "package_version": KRONOS2_XFORMERS_VERSION,
            "install_environment": "linux_x86_64_cpython_3_11_or_3_12",
        },
        "fallback": {
            "implementation": "upstream_dinov2_pytorch_attention",
            "condition": (
                "outside_reference_environment_or_xformers_unavailable_or_disabled"
            ),
            "enforcement": "temporary_XFORMERS_DISABLED_during_construction",
            "preimport_conflict": "fail_closed",
        },
    },
}
KRONOS2_MARKER_MATCHING_CONTRACT: dict[str, Any] = {
    "name": "upstream_separator_insensitive_exact",
    "cleaning": {
        "lowercase": True,
        "translate_to_underscore": ["-", " ", ":", "(", ")"],
        "drop": ["/"],
        "translate": {"α": "a"},
        "strip": True,
    },
    "match_key_drop": ["_", "."],
    "aliases": False,
    "compound_token_fallback": False,
    "forward_names": "canonical_marker_metadata_names",
}

# Re-exported here because fingerprint construction already obtains KRONOS2's
# loader contracts from this module.  The conditional value itself lives beside
# the canonical additional-marker parser so CLI, runner, and loader share one
# identity rather than independently spelling the BioLinkBERT recipe.
KRONOS2_ADDITIONAL_MARKER_REGISTRATION_CONTRACT = (
    KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_additional_marker_contract(
    value: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Validate the exact semantic contract emitted by ``kronos2_metadata``."""

    expected_keys = {
        "contract_version",
        "consumed_columns",
        "registration",
        "rows",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(
            "KRONOS2 additional_markers must be the canonical parsed contract "
            f"with exactly these fields: {sorted(expected_keys)}"
        )
    if value["contract_version"] != KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION:
        raise ValueError("KRONOS2 additional_markers has an unknown contract version")
    if value["consumed_columns"] != list(KRONOS2_ADDITIONAL_MARKER_COLUMNS):
        raise ValueError(
            "KRONOS2 additional_markers has a different consumed-column contract"
        )
    if value["registration"] != KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT:
        raise ValueError(
            "KRONOS2 additional_markers has a different BioLinkBERT registration "
            "contract"
        )

    rows = value["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("KRONOS2 additional_markers must contain at least one row")
    expected_row_keys = set(KRONOS2_ADDITIONAL_MARKER_COLUMNS)
    seen_casefold: set[str] = set()
    seen_match_keys: set[str] = set()
    canonical_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        if not isinstance(row, dict) or set(row) != expected_row_keys:
            raise ValueError(
                "KRONOS2 additional_markers row "
                f"{row_number} must contain exactly "
                f"{list(KRONOS2_ADDITIONAL_MARKER_COLUMNS)}"
            )
        canonical: dict[str, Any] = {}
        for column in KRONOS2_ADDITIONAL_MARKER_COLUMNS[:5]:
            item = row[column]
            if (
                not isinstance(item, str)
                or not item
                or item != item.strip()
                or any(unicodedata.category(character) == "Cc" for character in item)
            ):
                raise ValueError(
                    "KRONOS2 additional_markers row "
                    f"{row_number} column {column!r} is not canonical text"
                )
            canonical[column] = item
        for column in ("mean", "std"):
            item = row[column]
            if not isinstance(item, float):
                raise ValueError(
                    "KRONOS2 additional_markers row "
                    f"{row_number} column {column!r} must be a canonical float"
                )
            number = item
            if not math.isfinite(number) or (column == "std" and number <= 0):
                raise ValueError(
                    "KRONOS2 additional_markers row "
                    f"{row_number} column {column!r} is invalid"
                )
            canonical[column] = 0.0 if number == 0 else number

        marker_name = canonical["marker_name"]
        casefold_key = marker_name.casefold()
        match_key = _marker_match_key(marker_name)
        if casefold_key in seen_casefold or match_key in seen_match_keys:
            raise ValueError(
                "KRONOS2 additional_markers contains duplicate marker identities"
            )
        seen_casefold.add(casefold_key)
        seen_match_keys.add(match_key)
        canonical_rows.append(canonical)

    semantic = {
        "contract_version": KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION,
        "consumed_columns": list(KRONOS2_ADDITIONAL_MARKER_COLUMNS),
        "registration": KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT,
        "rows": canonical_rows,
    }
    digest = hashlib.sha256(_canonical_json(semantic).encode("utf-8")).hexdigest()
    if value["content_sha256"] != digest:
        raise ValueError(
            "KRONOS2 additional_markers content_sha256 does not match its "
            "canonical rows and registration contract"
        )
    return canonical_rows, digest


def _clean_marker_name(name: str) -> str:
    """Mirror the immutable upstream ``clean_marker_name`` contract."""

    return (
        str(name)
        .lower()
        .translate(
            str.maketrans(
                {
                    "-": "_",
                    " ": "_",
                    ":": "_",
                    "α": "a",
                    "(": "_",
                    ")": "_",
                    "/": "",
                }
            )
        )
        .strip()
    )


def _marker_match_key(name: str) -> str:
    """Return the exact separator-insensitive key used by pinned KRONOS2."""

    return _clean_marker_name(name).replace("_", "").replace(".", "")


def _scaling_factor(dtype: np.dtype) -> float:
    """Return the upstream raw-pixel divisor for one multiplex source dtype."""

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return 255.0
    if np.issubdtype(dtype, np.unsignedinteger):
        return 65535.0
    if np.issubdtype(dtype, np.floating):
        return 400.0
    raise ValueError(
        f"KRONOS2 has no published input scaling for dtype {dtype!r}; "
        "expected uint8, a wider unsigned integer, or floating point"
    )


def _metadata_sha256(spec) -> str:
    value = spec.timm_kwargs.get("marker_metadata_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(
            f"{spec.name}: registry timm_kwargs.marker_metadata_sha256 must be "
            "a 64-character SHA-256 digest"
        )
    return value


def _assert_allowlisted_runtime_snapshot(snapshot: str) -> None:
    """Refuse a cached runtime directory containing unrequested top-level code."""

    allowed = {
        *KRONOS2_SNAPSHOT_TOP_LEVEL_ALLOWLIST,
        ".cache",  # huggingface_hub local-directory download metadata
        "__pycache__",  # may appear after a previous successful local load
    }
    actual = {path.name for path in Path(snapshot).iterdir()}
    unexpected = sorted(actual - allowed)
    if unexpected:
        raise ValueError(
            "KRONOS2's raw2features-managed cache contains unexpected top-level "
            f"entries: {unexpected}. Refusing to add it to the Python import path."
        )


def _allows_pinned_xformers(
    *,
    system: str | None = None,
    machine: str | None = None,
    implementation: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> bool:
    """Whether this process may use the one validated xFormers distribution."""

    runtime = tuple(sys.version_info[:2]) if python_version is None else python_version
    if (
        (system or platform.system()).casefold() != "linux"
        or (machine or platform.machine()).casefold() not in {"x86_64", "amd64"}
        or (implementation or platform.python_implementation()).casefold() != "cpython"
        or runtime not in {(3, 11), (3, 12)}
    ):
        return False
    try:
        return importlib_metadata.version("xformers") == KRONOS2_XFORMERS_VERSION
    except importlib_metadata.PackageNotFoundError:
        return False


def _assert_forced_fallback_is_not_preimported_with_xformers() -> None:
    """Fail if an already-imported upstream module makes fallback unenforceable."""

    conflicts = [
        name
        for name in ("dinov2.layers.attention", "dinov2.layers.block")
        if bool(getattr(sys.modules.get(name), "XFORMERS_AVAILABLE", False))
    ]
    if conflicts:
        raise RuntimeError(
            "KRONOS2 must use its PyTorch attention fallback in this environment, "
            "but an already-imported DINOv2 module enabled xFormers: "
            f"{conflicts}. Start a fresh Python process before loading KRONOS2."
        )


@register("embedders", "kronos2")
class Kronos2Embedder(Embedder):
    """KRONOS2 marker-aware ViT-B/16, returning the 768-dimensional CLS token."""

    @property
    def max_batch_size(self) -> int:
        """Cap decoded batches before the CPU-staged, fixed-size forward path."""

        return 8

    def load(
        self,
        device: str = "cuda",
        dtype: torch.dtype | None = None,  # noqa: ARG002 - KRONOS2 is fixed fp32
        compile: bool = False,
    ) -> Kronos2Embedder:
        if compile:
            raise ValueError(
                "KRONOS2 does not support --compile in raw2features v0.2.1; "
                "use the validated eager reference path"
            )

        import torch

        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "KRONOS2 needs its pinned optional inference stack. Install "
                "`raw2features[kronos2]`, log in to Hugging Face, and request "
                "MahmoodLab/KRONOS2 access with an institutional account."
            ) from exc

        snapshot = download_pinned_hf_snapshot(
            self.spec.source,
            self.spec.weights_revision,
            allow_patterns=KRONOS2_SNAPSHOT_ALLOW_PATTERNS,
            local_dir=pinned_model_cache_dir(
                self.spec.source, self.spec.weights_revision
            ),
        )
        _assert_allowlisted_runtime_snapshot(snapshot)
        weights = os.path.join(snapshot, str(self.spec.weights_filename))
        metadata = os.path.join(snapshot, KRONOS2_MARKER_METADATA_FILENAME)
        # Both checks happen before Transformers imports or executes snapshot code.
        verify_sha256(weights, self.spec.weights_sha256, what=self.spec.name)
        verify_sha256(
            metadata,
            _metadata_sha256(self.spec),
            what=f"{self.spec.name} marker metadata",
        )

        # Passing a local directory makes the pinned upstream override skip its own
        # snapshot_download call. The outer Transformers resolver also receives
        # local_files_only=True. Upstream temporarily inserts the directory at
        # sys.path[0] for its bundled dinov2 package; remove that new entry after
        # construction so the directory is not left at the front of the caller's
        # import search path.
        force_attention_fallback = not _allows_pinned_xformers()
        previous_xformers_disabled = os.environ.get("XFORMERS_DISABLED")
        if force_attention_fallback:
            _assert_forced_fallback_is_not_preimported_with_xformers()
            os.environ["XFORMERS_DISABLED"] = "1"
        snapshot_path_count = sys.path.count(snapshot)
        try:
            model = AutoModel.from_pretrained(
                snapshot,
                trust_remote_code=True,
                local_files_only=True,
                device=device,
            )
        finally:
            added_count = max(0, sys.path.count(snapshot) - snapshot_path_count)
            for _ in range(added_count):
                # Upstream inserts at the front, and list.remove() removes the first
                # occurrence. Preserve any entry that belonged to the caller.
                sys.path.remove(snapshot)
            if force_attention_fallback:
                if previous_xformers_disabled is None:
                    os.environ.pop("XFORMERS_DISABLED", None)
                else:
                    os.environ["XFORMERS_DISABLED"] = previous_xformers_disabled
        model = model.float().eval().to(device)
        self._model = model
        self._device = device
        self._dtype = torch.float32
        self._metadata_path = metadata
        self._load_marker_metadata(metadata)
        self._panel: dict[str, Any] | None = None
        # A successful load owns a fresh upstream model. Per-instance registration and
        # panel-selection state from any prior unload/load cycle must not leak into it:
        # a remembered digest without the corresponding appended model buffers would
        # make the active vocabulary disagree with the actual forward path.
        for attribute in (
            "_registered_additional_markers_digest",
            "_novel_registration_failed",
            "_raw2features_multiplex_source_indices",
            "_binding_native_panel",
        ):
            self.__dict__.pop(attribute, None)
        return self

    def _load_marker_metadata(self, path: str) -> None:
        """Build the first-wins exact-key vocabulary from the verified CSV."""

        index: dict[str, dict[str, Any]] = {}
        compartments: set[str] = set()
        families: set[str] = set()
        with open(path, newline="", encoding="utf-8") as fh:
            for row_number, row in enumerate(csv.DictReader(fh)):
                canonical = str(row.get("marker_name") or "").strip()
                if not canonical:
                    raise ValueError(
                        f"{self.spec.name}: {KRONOS2_MARKER_METADATA_FILENAME} "
                        f"contains a blank marker_name at data row {row_number + 1}"
                    )
                key = _marker_match_key(canonical)
                record = {
                    "canonical": canonical,
                    "pretraining": str(row.get("pretraining") or "").strip().casefold()
                    == "yes",
                    "compartment": str(row.get("compartment") or ""),
                    "family": str(row.get("family") or ""),
                }
                compartments.add(record["compartment"])
                families.add(record["family"])
                # This mirrors upstream build_marker_index: CSV order wins when
                # separator variants collapse to the same exact match key.
                index.setdefault(key, record)
        if not index:
            raise ValueError(
                f"{self.spec.name}: {KRONOS2_MARKER_METADATA_FILENAME} is empty"
            )
        self._base_metadata_index = index
        self._metadata_index = dict(index)
        self._base_compartments = frozenset(compartments)
        self._base_families = frozenset(families)

    def _published_metadata(self) -> dict[str, dict[str, Any]]:
        """Return the verified shipped vocabulary, including in lightweight tests."""

        return getattr(self, "_base_metadata_index", self._metadata_index)

    def _validated_novel_records(
        self, contract: Any
    ) -> tuple[list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
        """Validate novel rows against the verified base marker/category contract."""

        rows, digest = _validate_additional_marker_contract(contract)
        base_index = self._published_metadata()
        allowed_compartments = getattr(
            self,
            "_base_compartments",
            frozenset(record["compartment"] for record in base_index.values()),
        )
        allowed_families = getattr(
            self,
            "_base_families",
            frozenset(record["family"] for record in base_index.values()),
        )
        additions: dict[str, dict[str, Any]] = {}
        for row in rows:
            canonical = str(row["marker_name"])
            match_key = _marker_match_key(canonical)
            if match_key in base_index:
                raise ValueError(
                    f"KRONOS2 additional marker {canonical!r} collides with the "
                    "verified published vocabulary under its exact matching contract"
                )
            compartment = str(row["compartment"])
            if compartment not in allowed_compartments:
                raise ValueError(
                    f"KRONOS2 additional marker {canonical!r} uses compartment "
                    f"{compartment!r}; it must exactly reuse one of the published "
                    f"categories: {sorted(allowed_compartments)}"
                )
            family = str(row["family"])
            if family not in allowed_families:
                raise ValueError(
                    f"KRONOS2 additional marker {canonical!r} uses family "
                    f"{family!r}; it must exactly reuse one of the published "
                    f"categories: {sorted(allowed_families)}"
                )
            additions[match_key] = {
                "canonical": canonical,
                "pretraining": False,
                "compartment": compartment,
                "family": family,
            }
        return rows, digest, additions

    def _download_verified_biolinkbert_snapshot(self) -> str:
        """Resolve and verify the complete conditional BioLinkBERT snapshot."""

        snapshot = download_pinned_hf_snapshot(
            KRONOS2_BIOLINKBERT_SOURCE,
            KRONOS2_BIOLINKBERT_REVISION,
            allow_patterns=tuple(KRONOS2_BIOLINKBERT_ARTIFACT_SHA256),
        )
        for filename, sha256 in KRONOS2_BIOLINKBERT_ARTIFACT_SHA256.items():
            verify_sha256(
                os.path.join(snapshot, filename),
                sha256,
                what=f"KRONOS2 BioLinkBERT {filename}",
            )
        return snapshot

    def _register_novel_markers(
        self, contract: dict[str, Any]
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Run the pinned upstream registration path once for one semantic table."""

        rows, digest, additions = self._validated_novel_records(contract)
        if getattr(self, "_novel_registration_failed", False):
            raise RuntimeError(
                "this KRONOS2 instance had a failed novel-marker registration and "
                "cannot be reused safely; unload it and construct a fresh model"
            )
        registered_digest = getattr(self, "_registered_additional_markers_digest", None)
        if registered_digest is not None:
            if registered_digest != digest:
                raise ValueError(
                    "this loaded KRONOS2 instance is already registered with "
                    f"additional-marker contract {registered_digest}; it cannot be "
                    f"reused with different contract {digest}. Load a fresh model."
                )
            return digest, additions

        snapshot = self._download_verified_biolinkbert_snapshot()
        marker_builder = self._model.backbone.marker_builder
        builder_module_name = type(marker_builder).__module__
        try:
            builder_module = importlib.import_module(builder_module_name)
            encoder_cls = builder_module.BioLinkBERTTextEncoder
            build_prompt = builder_module.build_prompt
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "the pinned KRONOS2 marker-builder module does not expose its "
                "BioLinkBERTTextEncoder and build_prompt registration contract"
            ) from exc

        import torch

        with tempfile.TemporaryDirectory(
            prefix="raw2features-kronos2-markers-"
        ) as temp_dir:
            work_dir = Path(temp_dir)
            canonical_csv = work_dir / "additional_markers.csv"
            materialize_kronos2_additional_markers_csv(contract, canonical_csv)

            cpu = torch.device("cpu")
            encoder = encoder_cls(
                model_id=snapshot,
                device=cpu,
                cache_dir=str(work_dir / "transformers-cache"),
                max_length=128,
                fp16_on_cuda=False,
            )
            if not hasattr(encoder, "enc"):
                raise RuntimeError(
                    "the pinned BioLinkBERTTextEncoder no longer exposes its encoder"
                )
            encoder.enc.float().eval().to(cpu)
            prompts = [
                build_prompt(
                    str(row["marker_name"]).lower(),
                    str(row["marker_full_name"]),
                    str(row["compartment"]),
                    str(row["family_desc"]),
                )
                for row in rows
            ]
            with torch.inference_mode():
                encoded = encoder(prompts).detach().cpu().float()
            expected_shape = (len(rows), 1024)
            if tuple(encoded.shape) != expected_shape:
                raise ValueError(
                    "KRONOS2 BioLinkBERT returned shape "
                    f"{tuple(encoded.shape)}, expected {expected_shape}"
                )
            embeddings = np.asarray(encoded.numpy(), dtype=np.float32)
            if not np.isfinite(embeddings).all():
                raise ValueError(
                    "KRONOS2 BioLinkBERT returned non-finite marker embeddings"
                )
            np.save(
                work_dir / "additional_marker_text_embeddings.npy",
                np.ascontiguousarray(embeddings),
                allow_pickle=False,
            )
            del encoded, encoder
            gc.collect()

            missing = object()
            previous_cache = getattr(marker_builder, "cache_dir", missing)
            marker_builder.cache_dir = work_dir
            try:
                self._model.register_additional_markers(str(canonical_csv))
            except Exception:
                # Upstream updates the normalisation table before replacing all
                # marker-builder buffers.  A failure can therefore leave the live
                # instance partially changed even though no store has been opened.
                self._novel_registration_failed = True
                raise
            finally:
                if previous_cache is missing:
                    delattr(marker_builder, "cache_dir")
                else:
                    marker_builder.cache_dir = previous_cache

        self._registered_additional_markers_digest = digest
        return digest, additions

    def bind_multiplex_panel(
        self,
        channel_names: list[str] | None,
        *,
        selected_channels: list[dict[str, Any]] | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> dict:
        """Bind a physical panel and conditionally register fingerprinted markers."""

        if getattr(self, "_novel_registration_failed", False):
            raise RuntimeError(
                "this KRONOS2 instance had a failed novel-marker registration and "
                "cannot be reused safely; unload it and construct a fresh model"
            )
        params = {} if model_params is None else model_params
        if not isinstance(params, dict) or set(params) - {"additional_markers"}:
            keys = sorted(params) if isinstance(params, dict) else []
            raise ValueError(
                "KRONOS2 native model parameters accept only "
                f"'additional_markers'; received {keys or type(params).__name__}"
            )

        novel_summary: dict[str, Any] = {
            "upstream_registration_available": True,
            "enabled": False,
        }
        active_index = dict(self._published_metadata())
        if "additional_markers" in params:
            contract = params["additional_markers"]
            if not isinstance(contract, dict):
                raise ValueError(
                    "KRONOS2 additional_markers must be a canonical parsed marker "
                    "contract, not a path, null value, or empty list"
                )
            digest, additions = self._register_novel_markers(contract)
            active_index.update(additions)
            novel_summary = {
                "upstream_registration_available": True,
                "enabled": True,
                "content_sha256": digest,
                "markers": [str(row["marker_name"]) for row in contract["rows"]],
                "registration": KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT,
            }

        # ``super`` validates and stores the selected physical source indices, then
        # passes only those names to set_panel.  Keep _panel indices local to that
        # already-sliced tensor so the physical channels are never sliced twice.
        self._metadata_index = active_index
        self._binding_native_panel = True
        try:
            summary = super().bind_multiplex_panel(
                channel_names,
                selected_channels=selected_channels,
                model_params=None,
            )
        finally:
            self._binding_native_panel = False
        summary["novel_markers"] = novel_summary
        return summary

    def set_panel(self, channel_names: list[str] | None) -> dict:
        """Bind physical channels to published KRONOS2 markers before patch work.

        Matching is exact after the pinned separator-only normalisation.  No KRONOSv1
        biological aliases or embedded-CD-token fallbacks are used.  Named unknowns
        fail now rather than receiving default statistics and crashing during forward.
        """

        if not channel_names:
            raise ValueError(
                "KRONOS2 requires one marker name for every multiplex channel"
            )
        if not hasattr(self, "_metadata_index"):
            raise RuntimeError("load KRONOS2 before binding a marker panel")
        if not getattr(self, "_binding_native_panel", False):
            # The compatibility API remains published-vocabulary-only.  A prior
            # explicit registration on a warm model must not make a later direct
            # set_panel call silently accept a novel marker without its contract.
            self._metadata_index = dict(self._published_metadata())

        indices: list[int] = []
        marker_names: list[str] = []
        mapping: list[dict[str, Any]] = []
        dropped: list[str] = []
        unknown: list[str] = []
        non_pretraining: list[str] = []

        for channel_index, value in enumerate(channel_names):
            source_name = "" if value is None else str(value)
            if not source_name.strip():
                dropped.append(source_name)
                continue
            record = self._metadata_index.get(_marker_match_key(source_name))
            if record is None:
                unknown.append(source_name)
                continue
            canonical = str(record["canonical"])
            indices.append(channel_index)
            marker_names.append(canonical)
            if not record["pretraining"]:
                non_pretraining.append(canonical)
            mapping.append(
                {
                    "channel": source_name,
                    "channel_index": channel_index,
                    "kronos2_marker": canonical,
                    "pretraining": bool(record["pretraining"]),
                }
            )

        if unknown:
            raise ValueError(
                "KRONOS2 could not exactly match these named channels to its pinned "
                f"marker_metadata.csv: {unknown}. No biological aliases are applied. "
                "The published vocabulary includes DAPI and DRAQ5 but not common "
                "Hoechst or DNA1/DNA2 labels. Such a channel can be omitted from an "
                "explicit --marker selection while remaining available to nuclear "
                "segmentation, or registered with dataset-specific statistics. "
                "For intentionally novel markers, supply the complete additional-"
                "marker CSV through KRONOS2's explicit additional-markers option; "
                "raw2features will validate and fingerprint its statistics and pinned "
                "BioLinkBERT registration before patch execution."
            )
        if not indices:
            raise ValueError(
                "no non-empty channels matched the KRONOS2 marker vocabulary"
            )

        # Upstream's preferred_dapi argument forces DAPI statistics.  Only the two
        # published, explicitly reviewed nuclear stains are selected automatically;
        # assigning an arbitrary novel ``dna_stain`` that override would change its
        # user-supplied normalisation silently.
        if "DAPI" in marker_names:
            preferred_dapi = "DAPI"
        elif "DRAQ5" in marker_names:
            preferred_dapi = "DRAQ5"
        else:
            preferred_dapi = None

        self._panel = {
            "idx": np.asarray(indices, dtype=np.int64),
            "marker_names": marker_names,
            "preferred_dapi": preferred_dapi,
            "source_channel_count": len(channel_names),
        }
        return {
            "n_markers": len(marker_names),
            "kept": [str(channel_names[i]) for i in indices],
            "dropped": dropped,
            "unmatched": [],
            "matched": marker_names,
            "defaulted": [],
            "non_pretraining": non_pretraining,
            "preferred_dapi": preferred_dapi,
            "mapping": mapping,
            "matching": KRONOS2_MARKER_MATCHING_CONTRACT,
            "vocabulary": {
                "source": self.spec.source,
                "filename": KRONOS2_MARKER_METADATA_FILENAME,
                "revision": self.spec.weights_revision,
                "sha256": _metadata_sha256(self.spec),
            },
            "novel_markers": {
                "upstream_registration_available": True,
                "enabled": False,
            },
        }

    def transform_batch(
        self,
        patches_hwc: list[np.ndarray],
        device: str,  # noqa: ARG002 - KRONOS2 stages transforms on CPU
    ) -> torch.Tensor:
        """Scale raw HWC marker patches, then delegate z-scoring upstream."""

        if self._panel is None:
            raise RuntimeError("call set_panel() before embedding multiplex patches")
        if not patches_hwc:
            raise ValueError("KRONOS2 transform_batch received an empty patch batch")
        expected_channels = int(self._panel["source_channel_count"])
        source_dtype: np.dtype | None = None
        side_px: int | None = None
        validated: list[np.ndarray] = []
        for patch_index, patch in enumerate(patches_hwc):
            array = np.asarray(patch)
            if array.ndim != 3:
                raise ValueError(
                    f"KRONOS2 patch {patch_index} must be HWC, got shape "
                    f"{tuple(array.shape)}"
                )
            height, width, channels = array.shape
            if height <= 0 or height != width or height % 16:
                raise ValueError(
                    f"KRONOS2 patch {patch_index} must be square with a positive "
                    f"side divisible by its 16-pixel stride, got {height}x{width}"
                )
            if channels != expected_channels:
                raise ValueError(
                    f"KRONOS2 patch {patch_index} has {channels} channels but the "
                    f"bound selected panel has {expected_channels}"
                )
            if side_px is not None and height != side_px:
                raise ValueError(
                    "KRONOS2 requires one spatial patch size across a batch"
                )
            dtype = np.dtype(array.dtype)
            if source_dtype is not None and dtype != source_dtype:
                raise ValueError(
                    "KRONOS2 requires one source dtype across a patch batch"
                )
            if np.issubdtype(dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(
                    f"KRONOS2 patch {patch_index} contains non-finite float values"
                )
            source_dtype = dtype
            side_px = height
            validated.append(array)
        if source_dtype is None:  # pragma: no cover - guarded by non-empty input
            raise AssertionError("KRONOS2 source dtype was not resolved")
        selected = np.stack(
            [
                patch[:, :, self._panel["idx"]].astype(np.float32, copy=False)
                for patch in validated
            ]
        )
        selected = np.transpose(selected, (0, 3, 1, 2))
        scaled = selected / _scaling_factor(source_dtype)
        normalized = self._model.preprocess(
            scaled,
            self._panel["marker_names"],
            preferred_dapi=self._panel["preferred_dapi"],
        )
        import torch

        return torch.from_numpy(
            np.ascontiguousarray(np.asarray(normalized, dtype=np.float32))
        )

    def embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Return CLS vectors through the upstream's fixed eight-row regime."""

        import torch

        if self._panel is None:
            raise RuntimeError("call set_panel() before embedding multiplex patches")
        if batch.ndim != 4 or batch.shape[0] <= 0:
            raise ValueError("KRONOS2 embed_batch expects a non-empty BCHW tensor")
        if batch.shape[1] != len(self._panel["marker_names"]):
            raise ValueError(
                f"KRONOS2 batch has {batch.shape[1]} marker channels but the bound "
                f"panel has {len(self._panel['marker_names'])}"
            )
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, int(batch.shape[0]), 8):
                microbatch = batch[start : start + 8]
                valid_rows = int(microbatch.shape[0])
                if valid_rows < 8:
                    repeats = microbatch[-1:].expand(
                        8 - valid_rows, *microbatch.shape[1:]
                    )
                    microbatch = torch.cat((microbatch, repeats), dim=0)
                cls = self._model(
                    microbatch.to(self._device, dtype=torch.float32),
                    self._panel["marker_names"],
                )
                if tuple(cls.shape) != (8, self.embedding_dim):
                    raise ValueError(
                        "KRONOS2 returned shape "
                        f"{tuple(cls.shape)}, expected {(8, self.embedding_dim)}"
                    )
                outputs.append(cls[:valid_rows].float().cpu())
        return torch.cat(outputs, dim=0)
