"""Patch -> model-input tensor.

The runner delivers patches at exactly ``patch_px`` and the target MPP. Here we
normalise and apply the per-model resize to ``spec.input_size`` -- a no-op in the
common case where ``patch_px == input_size``, and the model-fit step when a user
samples at a ``patch_px`` that differs from the model's fixed input size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ModelSpec

if TYPE_CHECKING:  # pragma: no cover
    import torch


def _pil_resample(interpolation: str):
    """Map a spec ``interpolation`` name to a PIL resample filter.

    The resize filter is part of each model's documented eval transform (and of
    ``transform_signature``), so it must follow the card - several models specify
    bicubic - rather than a hard-coded bilinear. Unknown names fall back to
    bilinear with a warning so a typo cannot silently change preprocessing.
    """
    from PIL import Image

    table = {
        "nearest": Image.NEAREST,
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
    }
    key = (interpolation or "bilinear").lower()
    resample = table.get(key)
    if resample is None:
        import warnings

        warnings.warn(
            f"unknown interpolation {interpolation!r}; using bilinear", stacklevel=2
        )
        resample = Image.BILINEAR
    return resample


def to_model_tensor(patch_hwc_uint8: np.ndarray, spec: ModelSpec) -> torch.Tensor:
    """HWC uint8 RGB -> normalised CHW float32 tensor."""
    import torch

    img = patch_hwc_uint8
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HWC RGB patch, got shape {img.shape}")
    if img.shape[0] != spec.input_size or img.shape[1] != spec.input_size:
        from PIL import Image

        img = np.asarray(
            Image.fromarray(img).resize(
                (spec.input_size, spec.input_size),
                _pil_resample(spec.interpolation),
            )
        )
    arr = np.array(img, dtype=np.uint8)  # writable, contiguous copy
    t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t.to(torch.float32).div_(255.0)
    mean = torch.tensor(spec.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(spec.std, dtype=torch.float32).view(3, 1, 1)
    t.sub_(mean).div_(std)
    return t


def to_model_batch(
    patches_hwc_uint8: list[np.ndarray], spec: ModelSpec, device: str
) -> torch.Tensor:
    """Batched counterpart of :func:`to_model_tensor`, normalising on ``device``.

    Stacks a batch of HWC uint8 RGB patches, copies them to ``device`` *once*, and
    does the permute / ``/255`` / ``(x-mean)/std`` as batched kernels there. The
    output is a ``(B, 3, input_size, input_size)`` float32 tensor on ``device``.

    When every patch already matches ``spec.input_size`` (the ``patch_px ==
    input_size`` case) no resize is needed and the result is bit-for-bit the same
    arithmetic as stacking :func:`to_model_tensor` over the batch -- just done in
    one device transfer and one set of vectorised ops. When a resize *is* needed we
    fall back to the per-patch CPU path (PIL, with the model's own interpolation) so
    the resize stays exactly equivalent to :func:`to_model_tensor`; only the cheap
    normalise differs in where it runs, and the result is then moved to ``device``.
    """
    import torch

    if not patches_hwc_uint8:
        raise ValueError("to_model_batch requires at least one patch")

    needs_resize = any(
        p.shape[0] != spec.input_size or p.shape[1] != spec.input_size
        for p in patches_hwc_uint8
    )
    if needs_resize:
        # Keep the (rare) resize on the CPU PIL path (model's own interpolation) for
        # exact equivalence with the per-patch transform; stack and move once. The
        # normalise is folded in.
        stacked = torch.stack([to_model_tensor(p, spec) for p in patches_hwc_uint8])
        return stacked.to(device, non_blocking=True)

    for p in patches_hwc_uint8:
        if p.ndim != 3 or p.shape[2] != 3:
            raise ValueError(f"expected HWC RGB patch, got shape {p.shape}")

    # One contiguous (B, H, W, 3) uint8 host buffer -> one H2D copy.
    batch_np = np.ascontiguousarray(np.stack(patches_hwc_uint8, axis=0))
    t = torch.from_numpy(batch_np).to(device, non_blocking=True)
    # (B, H, W, 3) uint8 -> (B, 3, H, W) float32 / 255, normalised, all on device.
    t = t.permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    mean = torch.tensor(spec.mean, dtype=torch.float32, device=t.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor(spec.std, dtype=torch.float32, device=t.device).view(1, 3, 1, 1)
    t.sub_(mean).div_(std)
    return t
