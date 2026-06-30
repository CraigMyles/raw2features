"""Device resolution so raw2features runs on whatever hardware is present.

``--device auto`` (the default) picks the best available backend - CUDA, then Apple
MPS, then CPU - instead of assuming an NVIDIA GPU. The forward pass itself is
backend-agnostic (fp32 works everywhere); the only thing missing was *selecting* a
non-CUDA device and failing clearly when a specific one was asked for but isn't there.
"""

from __future__ import annotations

import os


def _accelerators() -> tuple[bool, bool]:
    """Return ``(cuda_available, mps_available)``; both False without an accelerator."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch installed -> CPU only
        return (False, False)
    cuda = bool(torch.cuda.is_available())
    backend = getattr(torch.backends, "mps", None)
    mps = bool(backend) and torch.backends.mps.is_available()
    return (cuda, bool(mps))


def resolve_device(requested: str = "auto") -> str:
    """Resolve a ``--device`` value to a concrete backend.

    ``"auto"`` → the best available of ``cuda`` → ``mps`` → ``cpu``. A concrete
    request (``cuda``, ``cuda:1``, ``mps``, ``cpu``) is returned as-is, but raises
    ``ValueError`` with an actionable message if that accelerator is unavailable -
    replacing torch's opaque mid-pipeline ``RuntimeError``. Selecting MPS also enables
    ``PYTORCH_ENABLE_MPS_FALLBACK`` so ops without an MPS kernel fall back to CPU.
    """
    req = (requested or "auto").strip().lower()
    cuda, mps = _accelerators()
    if req == "auto":
        if cuda:
            return "cuda"
        if mps:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
        return "cpu"
    base = req.split(":")[0]
    if base == "cuda" and not cuda:
        raise ValueError(
            "CUDA requested (--device cuda) but no CUDA GPU is available. "
            "Use --device auto (picks the best backend) or --device cpu."
        )
    if base == "mps":
        if not mps:
            raise ValueError(
                "MPS requested but unavailable (needs Apple Silicon + a recent torch). "
                "Use --device auto or --device cpu."
            )
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return req
