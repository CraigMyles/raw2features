"""Slide thumbnails and QC overlays.

A thumbnail is a small RGB overview read from a coarse pyramid level (never the
full-resolution level 0). By default it renders at the segmenter's MPP
(:data:`DEFAULT_THUMBNAIL_MPP`), so the thumbnail, the tissue mask and the patch
grid share a pixel grid and the QC overlay needs no resampling. ``mpp`` / ``max_px``
override the resolution; level indices are never exposed (they are not portable
across slides -- the tool derives the level from MPP, as everywhere else).
"""

from __future__ import annotations

import os
import secrets
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from raw2features.core.geometry import Point, Region, Size
from raw2features.core.mpp import nearest_level

if TYPE_CHECKING:
    from raw2features.readers.base import WSISource

# Matches OtsuSegmenter.seg_mpp so the default thumbnail aligns with the mask.
DEFAULT_THUMBNAIL_MPP = 8.0


@dataclass
class Thumbnail:
    """An RGB thumbnail and the pyramid geometry it was read at."""

    image: np.ndarray  # (H, W, 3) uint8 RGB
    level: int
    downsample: float  # level-0 px per thumbnail px


def _nearest_level_for_mpp(reader: WSISource, target_mpp: float) -> int:
    return nearest_level(reader.mpp, reader.level_downsamples(), target_mpp)


def _coarsest_level_at_least(reader: WSISource, max_px: int) -> int:
    """Coarsest level whose longest side is still >= ``max_px`` (so the final
    downscale never upscales). Levels run fine -> coarse."""
    chosen = 0
    for i, d in enumerate(reader.level_dimensions):
        if max(d.width, d.height) >= max_px:
            chosen = i
        else:
            break
    return chosen


# Read a single level whole only if it fits this many px on its longest side; otherwise
# tile-and-downsample (below). Generous, so a normal coarse level reads whole (factor 1,
# unchanged) and only a deficient / non-pyramidal slide's huge level trips the guard.
SAFE_READ_PX = 12000


def read_level_capped(
    reader: WSISource, level: int, max_px: int = SAFE_READ_PX
) -> tuple[np.ndarray, int]:
    """Read ``level`` as RGB without ever holding more than one bounded tile in RAM.

    Returns ``(rgb, factor)``. A level fitting ``max_px`` is read whole (``factor`` 1 --
    unchanged). A larger one is read tile-by-tile, each tile downsampled by ``factor``
    (result <= ~``max_px``); the caller scales the level's downsample by ``factor``.
    Prevents an OOM on a slide whose coarsest level is still huge (deficient pyramid).
    """
    d: Size = reader.level_dimensions[level]
    w, h = int(d.width), int(d.height)
    if max(w, h) <= max_px:
        whole = reader.read_region(
            Region(level=level, location=Point(0, 0), size=Size(w, h))
        )
        return np.asarray(whole)[..., :3], 1

    import cv2

    factor = int(np.ceil(max(w, h) / max_px))
    oh, ow = (h + factor - 1) // factor, (w + factor - 1) // factor
    out = np.zeros((oh, ow, 3), np.uint8)
    in_tile = max(factor, (4096 // factor) * factor)  # bounded, a multiple of factor
    for y in range(0, h, in_tile):
        for x in range(0, w, in_tile):
            tw, th = min(in_tile, w - x), min(in_tile, h - y)
            block = np.asarray(
                reader.read_region(Region(level, Point(x, y), Size(tw, th)))
            )[..., :3]
            dw, dh = max(1, tw // factor), max(1, th // factor)
            small = cv2.resize(block, (dw, dh), interpolation=cv2.INTER_AREA)
            out[y // factor : y // factor + dh, x // factor : x // factor + dw] = small
    return out, factor


def render_thumbnail(
    reader: WSISource, *, mpp: float = DEFAULT_THUMBNAIL_MPP, max_px: int | None = None
) -> Thumbnail:
    """Read a coarse pyramid level as an RGB thumbnail.

    With ``max_px`` the longest side is capped at ``max_px`` (read the coarsest
    level still >= max_px, then downscale). Otherwise the nearest level to ``mpp``
    is used -- the segmenter's rule, so the default aligns 1:1 with the mask. The level
    read is itself bounded (:func:`read_level_capped`), so a huge level can't OOM.
    """
    if max_px is not None:
        level = _coarsest_level_at_least(reader, max_px)
    else:
        level = _nearest_level_for_mpp(reader, mpp)
    img, factor = read_level_capped(reader, level)
    downsample = float(reader.level_downsamples()[level]) * factor
    if max_px is not None and max(img.shape[0], img.shape[1]) > max_px:
        import cv2

        h, w = img.shape[:2]
        scale = max_px / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downsample = downsample / scale
    return Thumbnail(image=img, level=level, downsample=downsample)


def render_overlay(
    thumb: Thumbnail,
    *,
    tissue=None,
    coords: np.ndarray | None = None,
    level0_patch: int | None = None,
    mask_rgb: tuple[int, int, int] = (220, 20, 60),
    mask_alpha: float = 0.30,
    patch_rgb: tuple[int, int, int] = (0, 200, 0),
) -> np.ndarray:
    """Tint the tissue mask + outline kept patches on a copy of the thumbnail.

    Coords are level-0 (x, y) top-left; they are mapped to thumbnail pixels via
    ``thumb.downsample``. The tissue mask is resized to the thumbnail only if it
    was computed at a different resolution (none, at the default MPP).
    """
    import cv2

    img = np.ascontiguousarray(thumb.image.copy())
    h, w = img.shape[:2]
    ds = thumb.downsample

    if tissue is not None:
        m = tissue.mask
        if m.shape[:2] != (h, w):
            # cv2.resize is float-safe and channel-agnostic for single-channel arrays
            m = cv2.resize(
                m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST
            )
        sel = m > 0.5
        if sel.any():
            # mask_rgb is in RGB order (same as img); no cv2 colour interpretation here
            blended = img[sel].astype(np.float32) * (1.0 - mask_alpha) + (
                np.array(mask_rgb, dtype=np.float32) * mask_alpha
            )
            img[sel] = blended.astype(np.uint8)

    if coords is not None and len(coords) and level0_patch:
        side = max(1, int(round(level0_patch / ds)))
        thickness = max(1, round(min(h, w) / 1500))
        # cv2.rectangle treats the colour tuple as BGR, but img is RGB.
        # Reverse the tuple so the drawn colour matches the RGB patch_rgb spec.
        patch_bgr = patch_rgb[::-1]
        for x0, y0 in np.asarray(coords):
            x = int(round(int(x0) / ds))
            y = int(round(int(y0) / ds))
            cv2.rectangle(img, (x, y), (x + side, y + side), patch_bgr, thickness)
    return img


def save_png(image: np.ndarray, path: str) -> str:
    from PIL import Image

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    temporary: str | None = None
    fd: int | None = None
    try:
        try:
            existing_mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            existing_mode = None
        for _ in range(100):
            temporary = os.path.join(
                directory,
                f".r2f-sidecar.{secrets.token_hex(8)}.tmp",
            )
            try:
                # Honour the process umask/default ACL, matching a normal output
                # file rather than mkstemp's fixed 0600 permissions.
                fd = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666
                )
                break
            except FileExistsError:
                temporary = None
        else:  # pragma: no cover - 100 cryptographic-name collisions is infeasible
            raise FileExistsError("could not allocate a temporary PNG path")
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        with os.fdopen(fd, mode="wb") as fh:
            fd = None  # ownership transferred to ``fh``
            Image.fromarray(image).save(fh, format="PNG")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    except BaseException:  # noqa: BLE001 - cleanup on interrupts as well as failures
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise
    return path


def write_thumbnails(
    reader: WSISource,
    out_dir: str,
    slide_id: str,
    *,
    mpp: float = DEFAULT_THUMBNAIL_MPP,
    max_px: int | None = None,
    tissue=None,
    coords: np.ndarray | None = None,
    level0_patch: int | None = None,
    overlay: bool = False,
    overwrite: bool = True,
    overlay_name: str | None = None,
) -> dict:
    """Render + save the plain thumbnail (and the QC overlay if requested).

    ``overwrite=False`` repairs only missing assets, preserving an existing plain
    thumbnail while (for example) recreating a deleted overlay. ``overlay_name``
    gives a non-primary grid its own namespaced overlay (a basename, not a path).
    Returns provenance metadata (basenames + the read geometry) for ``.zattrs``.
    """
    thumb = render_thumbnail(reader, mpp=mpp, max_px=max_px)
    plain_path = os.path.join(out_dir, f"{slide_id}.thumbnail.png")
    if overwrite or not os.path.exists(plain_path):
        save_png(thumb.image, plain_path)

    overlay_path = None
    if overlay and (tissue is not None or (coords is not None and len(coords))):
        composed = render_overlay(
            thumb, tissue=tissue, coords=coords, level0_patch=level0_patch
        )
        if overlay_name is not None and os.path.basename(overlay_name) != overlay_name:
            raise ValueError("overlay_name must be a basename, not a path")
        overlay_path = os.path.join(
            out_dir, overlay_name or f"{slide_id}.thumbnail.overlay.png"
        )
        if overwrite or not os.path.exists(overlay_path):
            save_png(composed, overlay_path)

    return {
        "plain": os.path.basename(plain_path),
        "overlay": os.path.basename(overlay_path) if overlay_path else None,
        "mpp": None if max_px else mpp,
        "max_px": max_px,
        "read_level": thumb.level,
        "downsample": thumb.downsample,
        "size_wh": [int(thumb.image.shape[1]), int(thumb.image.shape[0])],
    }
