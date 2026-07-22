"""Shared validation for content/runtime options exposed by several commands."""

from __future__ import annotations

import csv
import json
import math
import os
import unicodedata
from typing import Any

import typer

AMP_CHOICES = ("auto", "fp32", "bf16", "fp16")


def validate_amp(value: str) -> None:
    """Reject an unknown precision before model discovery or output mutation."""

    if value not in AMP_CHOICES:
        choices = ", ".join(AMP_CHOICES)
        raise typer.BadParameter(f"must be one of: {choices}", param_hint="--amp")


def validate_batch_size(value: int) -> None:
    """Reject empty/negative batches before they reach the embedding loop."""

    if value <= 0:
        raise typer.BadParameter("must be greater than zero", param_hint="--batch-size")


def validate_positive_float(value: float | None, param_hint: str) -> None:
    """Reject zero, negative, NaN, and infinite physical scales."""

    if value is not None and (not math.isfinite(value) or value <= 0):
        raise typer.BadParameter(
            "must be finite and greater than zero", param_hint=param_hint
        )


def validate_positive_int(value: int | None, param_hint: str) -> None:
    """Reject zero or negative pixel sizes/strides."""

    if value is not None and value <= 0:
        raise typer.BadParameter("must be greater than zero", param_hint=param_hint)


def parse_channel_names_file(path: str | None) -> list[str]:
    """Read a complete positional channel panel from a small text table.

    ``.txt`` files contain one name per line. ``.csv`` and ``.tsv`` files contain
    exactly one column and may start with a conventional channel-name header. Blank
    lines and ``#`` comments in text files are ignored; the resulting ordered list is
    validated against the physical C-axis length after the reader opens.
    """

    if path is None:
        return []
    suffix = os.path.splitext(path)[1].casefold()
    if suffix not in {".txt", ".csv", ".tsv"}:
        raise typer.BadParameter(
            "must be a .txt, .csv, or .tsv file",
            param_hint="--channel-names-file",
        )
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            if suffix == ".txt":
                rows = [
                    [line]
                    for line in handle.read().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]
            else:
                rows = [
                    row
                    for row in csv.reader(
                        handle,
                        delimiter="," if suffix == ".csv" else "\t",
                        strict=True,
                    )
                    if row
                ]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise typer.BadParameter(
            f"could not read file ({exc})", param_hint="--channel-names-file"
        ) from exc

    header_names = {
        "channel",
        "channel_name",
        "channel name",
        "marker",
        "marker_name",
        "marker name",
        "name",
        "label",
    }
    if rows and len(rows[0]) == 1 and rows[0][0].strip().casefold() in header_names:
        rows = rows[1:]
    if not rows:
        raise typer.BadParameter(
            "must contain at least one channel name",
            param_hint="--channel-names-file",
        )

    names: list[str] = []
    for index, row in enumerate(rows, start=1):
        if len(row) != 1:
            raise typer.BadParameter(
                f"row {index} must contain exactly one column",
                param_hint="--channel-names-file",
            )
        name = row[0].strip()
        if not name:
            raise typer.BadParameter(
                f"row {index} has an empty channel name",
                param_hint="--channel-names-file",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise typer.BadParameter(
                f"row {index} contains a control character",
                param_hint="--channel-names-file",
            )
        names.append(name)
    identities = [unicodedata.normalize("NFKC", name).casefold() for name in names]
    duplicates = sorted(
        {identity for identity in identities if identities.count(identity) > 1}
    )
    if duplicates:
        raise typer.BadParameter(
            "channel names must be unique after Unicode/case normalization",
            param_hint="--channel-names-file",
        )
    return names


def validate_geometry(
    *,
    mpp: float | None = None,
    patch_size: int | None = None,
    step: int | None = None,
    source_mpp: float | None = None,
) -> None:
    """Validate shared CLI geometry before discovery, loading, or output mutation."""

    validate_positive_float(mpp, "--mpp")
    validate_positive_int(patch_size, "--patch-size")
    validate_positive_int(step, "--step")
    validate_positive_float(source_mpp, "--source-mpp")


def validate_multiplex_percentiles(low: float, high: float) -> None:
    """Require a finite, ordered percentile interval within ``[0, 100]``."""

    if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 100):
        raise typer.BadParameter(
            "must be finite and satisfy 0 <= low < high <= 100",
            param_hint="--multiplex-percentile-low/--multiplex-percentile-high",
        )


def parse_json_object(value: str | None, param_hint: str) -> dict[str, Any]:
    """Parse a finite JSON object for a namespaced plugin configuration."""

    if value is None:
        return {}
    try:
        parsed = json.loads(value, parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant {token}")
        ))
    except (json.JSONDecodeError, ValueError) as exc:
        raise typer.BadParameter(
            f"must be a valid finite JSON object ({exc})", param_hint=param_hint
        ) from exc
    if not isinstance(parsed, dict) or any(not isinstance(key, str) for key in parsed):
        raise typer.BadParameter(
            "must be a JSON object with string keys", param_hint=param_hint
        )
    return parsed
