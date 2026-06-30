"""Derive per-patch QC scores from a per-pixel class raster.

A QC tool whose native output is a coarse per-pixel class map (GrandQC's tissue/artifact
segmentation is the motivating case) becomes the per-patch layer the store holds
(``qc/<tool>/scores``) by projecting each kept patch's level-0 footprint into the raster
and counting class-coverage fractions -- the same footprint math the patcher uses for
tissue fraction (``GridPatcher._cell_tissue_fractions``), for N classes at any raster
resolution.

Pure and producer-agnostic: it needs only the kept-patch ``coords`` (level-0 px),
the patch side, and the raster + how its pixels map to level 0. The QC *model* that made
the raster is external (see ``docs/SEGMENTATION.md``); only the raster reaches here.
"""

from __future__ import annotations

import numpy as np


def patch_qc_scores(
    coords: np.ndarray,
    level0_patch: int,
    raster: np.ndarray,
    class_values: list[int],
    *,
    raster_downsample: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Per-patch class-coverage fractions from a per-pixel class-index ``raster``.

    Parameters
    ----------
    coords:
        ``(N, 2)`` level-0 ``(x, y)`` top-left of each kept patch.
    level0_patch:
        Patch side in level-0 pixels (``patching.level0_patch``).
    raster:
        ``(h, w)`` integer class-index map at a coarse MPP (e.g. GrandQC's 7-class
        artifact map). Border-padded reads are clipped to the raster bounds.
    class_values:
        The ``k`` raster integer values, in column order (GrandQC's ``[1, 2, …, 7]``).
        Column ``j`` of the output is the fraction of the patch footprint equal to
        ``class_values[j]``.
    raster_downsample:
        Level-0 pixels per raster pixel (``raster_mpp / source_mpp``).
    origin_xy:
        Level-0 ``(x, y)`` of the raster's top-left, when it does not start at the slide
        origin (default ``(0, 0)``).

    Returns
    -------
    ``(N, k)`` float32 fractions in ``[0, 1]``, 1:1 with ``coords`` -- ready for
    ``ZarrSink.write_qc(tool, scores, classes, …)``.
    """
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 2)
    n = coords.shape[0]
    k = len(class_values)
    out = np.zeros((n, k), dtype=np.float32)
    if n == 0 or raster.size == 0:
        return out
    ox, oy = origin_xy
    cell = max(1, round(level0_patch / raster_downsample))
    h, w = raster.shape[:2]
    for i in range(n):
        x0, y0 = int(coords[i, 0]), int(coords[i, 1])
        rx = int(round((x0 - ox) / raster_downsample))
        ry = int(round((y0 - oy) / raster_downsample))
        # Clamp both ends to [0, dim]: an above/left patch has a negative end, which
        # numpy reads as a wrap-around slice (a wrong window) rather than empty.
        y1, x1 = max(0, min(ry + cell, h)), max(0, min(rx + cell, w))
        win = raster[max(ry, 0) : y1, max(rx, 0) : x1]
        total = win.size
        if total == 0:
            continue
        for j, cv in enumerate(class_values):
            out[i, j] = float(np.count_nonzero(win == cv)) / total
    return out
