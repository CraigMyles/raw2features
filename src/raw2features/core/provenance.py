"""Capture machine-readable provenance for an embedding run.

FAIR: every output records exactly how it was produced - the CLI invocation, the
git commit, host/arch/GPU, and tool versions - so a result is reproducible and
re-traceable months later.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _package_version() -> str:
    try:
        return version("raw2features")
    except PackageNotFoundError:  # pragma: no cover - editable/uninstalled
        return "0+unknown"


def _gpu_info() -> dict[str, object] | None:
    """Best-effort GPU description without importing torch unconditionally."""
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    if not torch.cuda.is_available():
        return {"cuda_available": False, "torch": torch.__version__}
    return {
        "cuda_available": True,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
    }


def now_utc_iso() -> str:
    """Current UTC time, ISO-8601 with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture(cli: str | None = None) -> dict[str, object]:
    """Return a provenance dict for embedding in output ``.zattrs`` / receipts."""
    prov: dict[str, object] = {
        "raw2features_version": _package_version(),
        "created_utc": now_utc_iso(),
        "cli": cli if cli is not None else " ".join(sys.argv),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "arch": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    gpu = _gpu_info()
    if gpu is not None:
        prov["gpu"] = gpu
    return prov
