"""Microns-per-pixel (MPP) aware pyramid-level selection and resample maths.

The guiding decision (see plan): we hit the requested MPP **exactly** by default,
not "close enough". Foundation-model embeddings are scale-sensitive, so a
*consistent* input scale across slides and datasets is what makes them
comparable.

Strategy: pick the nearest **finer-or-equal** pyramid level (native MPP <=
target), so we only ever *downsample* - never upsample / invent detail - then
resample by the exact factor to land on precisely ``target_mpp`` at ``patch_px``.

This module is pure maths: no I/O, no array libraries beyond plain Python, and is
the most heavily unit-tested part of the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

# Relative tolerance for treating a resample factor as "already exact" (1.0).
_EXACT_EPS = 1e-6


@dataclass(frozen=True)
class LevelChoice:
    """How to read one patch at a target MPP from a pyramidal source.

    Attributes
    ----------
    level:
        Pyramid level to read from.
    level_mpp:
        Native microns-per-pixel at ``level``.
    target_mpp:
        The requested microns-per-pixel.
    achieved_mpp:
        The MPP actually delivered. Equals ``target_mpp`` when resampling is
        applied (the default), or ``level_mpp`` when reading natively
        (``snap_to_level`` / within ``tolerance``).
    resample:
        ``target_mpp / level_mpp``. ``>= 1`` means downsample (read more pixels
        then shrink); ``< 1`` means upsample (only with ``allow_upsample``).
    read_px:
        Number of pixels to read per side at ``level`` before resizing to
        ``patch_px``.
    needs_resample:
        Whether a resize from ``read_px`` to ``patch_px`` is required.
    """

    level: int
    level_mpp: float
    target_mpp: float
    achieved_mpp: float
    resample: float
    read_px: int
    needs_resample: bool


def level_mpps(mpp_level0: float, level_downsamples: list[float]) -> list[float]:
    """Native MPP at each pyramid level."""
    return [mpp_level0 * float(d) for d in level_downsamples]


def nearest_level(
    mpp_level0: float | None, level_downsamples: list[float], target_mpp: float
) -> int:
    """Index of the pyramid level whose native MPP is nearest ``target_mpp``.

    Falls back to the coarsest level when ``mpp_level0`` is unknown (``None``).
    Shared by tissue segmentation and thumbnail level selection so the rule
    lives in one place.
    """
    if mpp_level0 is None:
        return len(level_downsamples) - 1
    mpps = level_mpps(mpp_level0, level_downsamples)
    return min(range(len(mpps)), key=lambda i: abs(mpps[i] - target_mpp))


def level_for_mpp(
    target_mpp: float,
    mpp_level0: float,
    level_downsamples: list[float],
    patch_px: int,
    *,
    tolerance: float = 0.001,
    snap_to_level: bool = False,
    allow_upsample: bool = False,
) -> LevelChoice:
    """Choose a pyramid level and resample plan for ``target_mpp``.

    Parameters
    ----------
    target_mpp:
        Desired output microns-per-pixel (e.g. 1.0).
    mpp_level0:
        Microns-per-pixel at level 0 of the source.
    level_downsamples:
        Per-level downsample factors relative to level 0 (level 0 == 1.0).
        Assumed monotonically non-decreasing.
    patch_px:
        Output patch side length (e.g. 224).
    tolerance:
        Relative tolerance (default 0.001 = 0.1%). A level whose MPP is within
        this of ``target_mpp`` is read natively instead of resampling - this
        absorbs floating-point pyramid scales (e.g. an iSyntax-converted level at
        1.0001652 is read natively for target 1.0 rather than over-reading the
        2x-finer level and downsampling).
    snap_to_level:
        If True, pick the level whose native MPP is *nearest* the target and
        read natively (achieved MPP == that level's MPP). Faster, not exact.
    allow_upsample:
        If True, permit choosing level 0 and upsampling when the target is finer
        than the finest available level. Off by default (raises instead).

    Raises
    ------
    ValueError:
        If inputs are invalid, or the target is finer than level 0 and
        ``allow_upsample`` is False.
    """
    if target_mpp <= 0 or mpp_level0 <= 0:
        raise ValueError("target_mpp and mpp_level0 must be positive")
    if patch_px <= 0:
        raise ValueError("patch_px must be positive")
    if not level_downsamples:
        raise ValueError("level_downsamples must be non-empty")

    mpps = level_mpps(mpp_level0, level_downsamples)

    if snap_to_level:
        level = nearest_level(mpp_level0, level_downsamples, target_mpp)
        return LevelChoice(
            level=level,
            level_mpp=mpps[level],
            target_mpp=target_mpp,
            achieved_mpp=mpps[level],
            resample=target_mpp / mpps[level],
            read_px=patch_px,
            needs_resample=False,
        )

    # Nearest finer-or-equal level: the coarsest level whose MPP is <= target
    # (within tolerance), so the downsample factor is minimal. The tolerance lets
    # a level fractionally above target (float pyramid scales) still qualify.
    bound = target_mpp * (1 + max(tolerance, _EXACT_EPS))
    finer_or_equal = [i for i, m in enumerate(mpps) if m <= bound]
    if finer_or_equal:
        level = finer_or_equal[-1]
    elif allow_upsample:
        level = 0  # finest available; we will upsample
    else:
        raise ValueError(
            f"target_mpp={target_mpp} is finer than level-0 MPP={mpp_level0}; "
            f"pass allow_upsample=True to upsample"
        )

    level_mpp = mpps[level]

    # Within tolerance of target -> read natively, achieved MPP is the level's.
    if abs(level_mpp - target_mpp) <= max(tolerance, _EXACT_EPS) * target_mpp:
        return LevelChoice(
            level=level,
            level_mpp=level_mpp,
            target_mpp=target_mpp,
            achieved_mpp=level_mpp,
            resample=target_mpp / level_mpp,
            read_px=patch_px,
            needs_resample=False,
        )

    resample = target_mpp / level_mpp
    needs_resample = abs(resample - 1.0) > _EXACT_EPS
    read_px = round(patch_px * resample) if needs_resample else patch_px
    return LevelChoice(
        level=level,
        level_mpp=level_mpp,
        target_mpp=target_mpp,
        achieved_mpp=target_mpp if needs_resample else level_mpp,
        resample=resample,
        read_px=read_px,
        needs_resample=needs_resample,
    )


def level0_step_px(step_out_px: int, target_mpp: float, mpp_level0: float) -> int:
    """Tiling stride in level-0 pixels.

    ``step_out_px`` is the stride in output (target-MPP) pixels; the physical
    extent ``step_out_px * target_mpp`` converted to level-0 pixels.
    """
    if step_out_px <= 0:
        raise ValueError("step_out_px must be positive")
    return round(step_out_px * target_mpp / mpp_level0)
