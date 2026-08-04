"""Canonical input contract for user-supplied KRONOS2 marker metadata.

The upstream KRONOS2 novel-marker path consumes a small CSV table.  This module
normalises only the fields that path reads, validates the parts that can be
checked without the gated base vocabulary, and hashes the semantic table together
with the immutable text-registration recipe rather than incidental CSV formatting.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import unicodedata
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TextIO

KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION = 1
KRONOS2_ADDITIONAL_MARKER_COLUMNS = (
    "marker_name",
    "marker_full_name",
    "compartment",
    "family",
    "family_desc",
    "mean",
    "std",
)
_TEXT_COLUMNS = KRONOS2_ADDITIONAL_MARKER_COLUMNS[:5]

KRONOS2_BIOLINKBERT_SOURCE = "hf_hub:michiyasunaga/BioLinkBERT-large"
KRONOS2_BIOLINKBERT_REVISION = "1eb6d81c5fc1c42d3a43c71956b0e526558ae053"
KRONOS2_BIOLINKBERT_ARTIFACT_SHA256 = {
    "pytorch_model.bin": (
        "fed75e5716547b54198d4dd123e7a3f3c64a82e1172b3492a11deebd6ab4cd4d"
    ),
    "config.json": ("ba7be72cf4013c5a69166e88d51f922aee44bfde6927cf9315580c23a6e020a6"),
    "tokenizer.json": (
        "5a797027c356bfdc779fd021b4a7a2b2f341242ddd7035d385f56b8e242dac2a"
    ),
    "tokenizer_config.json": (
        "98632482a9851173e796e4366ede775ddc2900564a0ac659acb89248e39fbbd3"
    ),
    "special_tokens_map.json": (
        "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"
    ),
    "vocab.txt": ("7b36651908a88bc38bda41b728b2a598191e0d3b553cbacf7b1e5f026d5b5b9f"),
}

# This conditional contract is included only when a user supplies an
# additional-marker table.  Published-marker inference therefore retains its smaller
# fingerprint and never needs BioLinkBERT, while a novel-marker result binds every
# part of the upstream text-embedding recipe.
KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT: dict[str, Any] = {
    "contract_version": 1,
    "activation": "additional_markers_parameter_only",
    "registration": {
        "entrypoint": "pinned_kronos2_model.register_additional_markers",
        "canonical_csv_columns": list(KRONOS2_ADDITIONAL_MARKER_COLUMNS),
        "statistics": "user_supplied_per_marker_mean_std",
        "category_policy": "reuse_exact_base_compartment_and_family_only",
        "base_marker_collision_policy": "reject_separator_insensitive_exact",
        "text_embedding_handoff": "temporary_additional_marker_text_embeddings.npy",
    },
    "prompt": {
        "builder": "pinned_kronos2_builder_module.build_prompt",
        "inputs": [
            "marker_name_lowercase",
            "marker_full_name",
            "compartment",
            "family_desc",
        ],
        "template": (
            "Marker: {marker_name.upper()} ({marker_full_name}) | "
            "It is located in the {compartment} compartment and "
            "belongs to the {family_desc} family"
        ),
    },
    "text_encoder": {
        "class": "pinned_kronos2_builder_module.BioLinkBERTTextEncoder",
        "source": KRONOS2_BIOLINKBERT_SOURCE,
        "revision": KRONOS2_BIOLINKBERT_REVISION,
        "license": "Apache-2.0",
        "artifact_sha256": deepcopy(KRONOS2_BIOLINKBERT_ARTIFACT_SHA256),
        "input": "verified_pinned_local_snapshot",
        "device": "cpu",
        "precision": "fp32",
        "embedding_dim": 1024,
        "tokenizer": {
            "use_fast": True,
            "padding": True,
            "truncation": True,
            "max_length": 128,
        },
        "pooling": "attention_mask_mean_last_hidden_state_including_special_tokens",
        "normalization": "l2",
        "fp16_on_cuda": False,
    },
}


class Kronos2MarkerMetadataError(ValueError):
    """Raised when an additional-marker CSV cannot form a safe contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _contains_control(value: str) -> bool:
    return any(unicodedata.category(character) == "Cc" for character in value)


def _canonical_text(value: str | None, *, column: str, row_number: int) -> str:
    if value is None:
        value = ""
    # Prompt text is content, not CSV formatting. Preserve its Unicode after
    # trimming so BioLinkBERT receives what the user supplied and the semantic digest
    # moves whenever that text changes.
    value = str(value)
    if _contains_control(value):
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column {column!r} contains a control character"
        )
    value = value.strip()
    if not value:
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column {column!r} must not be blank"
        )
    return value


def _canonical_number(value: str | None, *, column: str, row_number: int) -> float:
    if value is None:
        value = ""
    value = str(value)
    if _contains_control(value):
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column {column!r} contains a control character"
        )
    try:
        number = float(value.strip())
    except ValueError as exc:
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column {column!r} must be a finite number"
        ) from exc
    if not math.isfinite(number):
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column {column!r} must be a finite number"
        )
    if column == "std" and number <= 0:
        raise Kronos2MarkerMetadataError(
            f"row {row_number} column 'std' must be greater than zero"
        )
    # Treat signed zero as one canonical numeric value.
    return 0.0 if number == 0 else number


def _kronos2_marker_match_key(value: str) -> str:
    """Mirror KRONOS2's separator-insensitive forward-lookup identity."""

    cleaned = (
        value.lower()
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
    return cleaned.replace("_", "").replace(".", "")


def _validate_headers(fieldnames: list[str | None] | None) -> None:
    if not fieldnames:
        raise Kronos2MarkerMetadataError("additional-marker CSV has no header")
    if any(name is None or not name for name in fieldnames):
        raise Kronos2MarkerMetadataError(
            "additional-marker CSV contains a blank column name"
        )
    duplicate_headers = sorted(
        {name for name in fieldnames if fieldnames.count(name) > 1}
    )
    if duplicate_headers:
        raise Kronos2MarkerMetadataError(
            f"additional-marker CSV has duplicate columns: {duplicate_headers}"
        )
    missing = [
        column
        for column in KRONOS2_ADDITIONAL_MARKER_COLUMNS
        if column not in fieldnames
    ]
    if missing:
        raise Kronos2MarkerMetadataError(
            f"additional-marker CSV is missing required columns: {missing}"
        )


def _canonical_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "contract_version": KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION,
        "consumed_columns": list(KRONOS2_ADDITIONAL_MARKER_COLUMNS),
        "registration": deepcopy(KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT),
        "rows": rows,
    }


def parse_kronos2_additional_markers(
    path: str | os.PathLike[str],
    *,
    include_original_file_sha256: bool = False,
) -> dict[str, Any]:
    """Parse *path* into a JSON-safe, fingerprintable marker-table contract.

    Extra CSV columns are deliberately ignored: the contract lists and hashes
    exactly the columns consumed by KRONOS2.  Row order is semantic because the
    upstream registration path appends marker buffers in that order.  Validation
    against the gated base vocabulary's compartment/family categories is left to
    the loader after that independently verified metadata is available.  The
    original file-byte digest is optional audit data and is omitted by default so
    formatting-only CSV changes do not change an embedding's output identity.
    """

    try:
        raw = os.fsdecode(path)
        with open(raw, "rb") as handle:
            file_bytes = handle.read()
    except OSError as exc:
        raise Kronos2MarkerMetadataError(
            f"could not read additional-marker CSV {os.fspath(path)!r}: {exc}"
        ) from exc

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Kronos2MarkerMetadataError(
            "additional-marker CSV must be UTF-8 encoded"
        ) from exc

    rows: list[dict[str, Any]] = []
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        _validate_headers(reader.fieldnames)
        for row_number, source in enumerate(reader, start=2):
            if None in source:
                raise Kronos2MarkerMetadataError(
                    f"row {row_number} contains more fields than the CSV header"
                )
            row = {
                column: _canonical_text(
                    source.get(column), column=column, row_number=row_number
                )
                for column in _TEXT_COLUMNS
            }
            row["mean"] = _canonical_number(
                source.get("mean"), column="mean", row_number=row_number
            )
            row["std"] = _canonical_number(
                source.get("std"), column="std", row_number=row_number
            )
            rows.append(row)
    except csv.Error as exc:
        raise Kronos2MarkerMetadataError(
            f"could not parse additional-marker CSV: {exc}"
        ) from exc

    if not rows:
        raise Kronos2MarkerMetadataError(
            "additional-marker CSV must contain at least one marker row"
        )

    casefold_seen: dict[str, tuple[int, str]] = {}
    match_seen: dict[str, tuple[int, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        marker_name = row["marker_name"]
        casefold_key = marker_name.casefold()
        match_key = _kronos2_marker_match_key(marker_name)
        if previous := casefold_seen.get(casefold_key):
            raise Kronos2MarkerMetadataError(
                "marker_name values must be unique case-insensitively: "
                f"rows {previous[0]} ({previous[1]!r}) and "
                f"{row_number} ({marker_name!r})"
            )
        if previous := match_seen.get(match_key):
            raise Kronos2MarkerMetadataError(
                "marker_name values must be unique under KRONOS2's "
                "separator-insensitive lookup: "
                f"rows {previous[0]} ({previous[1]!r}) and "
                f"{row_number} ({marker_name!r})"
            )
        casefold_seen[casefold_key] = (row_number, marker_name)
        match_seen[match_key] = (row_number, marker_name)

    semantic_payload = _canonical_payload(rows)
    payload = deepcopy(semantic_payload)
    payload["content_sha256"] = hashlib.sha256(
        _canonical_json(semantic_payload).encode("utf-8")
    ).hexdigest()
    if include_original_file_sha256:
        payload["original_file_sha256"] = hashlib.sha256(file_bytes).hexdigest()
    return payload


def materialize_kronos2_additional_markers_csv(
    contract: Mapping[str, Any],
    destination: str | os.PathLike[str] | TextIO,
) -> None:
    """Write a parsed contract as the exact canonical CSV consumed upstream."""

    if contract.get("contract_version") != KRONOS2_ADDITIONAL_MARKER_CONTRACT_VERSION:
        raise Kronos2MarkerMetadataError(
            "cannot materialize an unknown additional-marker contract version"
        )
    if contract.get("consumed_columns") != list(KRONOS2_ADDITIONAL_MARKER_COLUMNS):
        raise Kronos2MarkerMetadataError(
            "cannot materialize a contract with different consumed columns"
        )
    if contract.get("registration") != KRONOS2_NOVEL_MARKER_REGISTRATION_CONTRACT:
        raise Kronos2MarkerMetadataError(
            "cannot materialize a contract with a different registration recipe"
        )
    rows = contract.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Kronos2MarkerMetadataError(
            "cannot materialize a contract without marker rows"
        )

    close = False
    if hasattr(destination, "write"):
        handle = destination
    else:
        handle = open(destination, "w", encoding="utf-8", newline="")
        close = True
    try:
        writer = csv.DictWriter(
            handle,
            fieldnames=KRONOS2_ADDITIONAL_MARKER_COLUMNS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if close:
            handle.close()
