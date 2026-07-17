"""Shared validation for content/runtime options exposed by several commands."""

from __future__ import annotations

import math

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
