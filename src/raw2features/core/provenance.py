"""Capture machine-readable provenance for an embedding run.

FAIR: every output records exactly how it was produced - the CLI invocation, the
git commit, host/arch/GPU, and tool versions - so a result is reproducible and
re-traceable months later.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version

from raw2features.core.uris import (
    is_qualified_uri,
    redact_uri_credentials,
    source_uri,
)

_SECRET_OPTION_RE = re.compile(
    r"(?<!\S)(?P<flag>--(?:access-token|api-key|authorization|client-secret|"
    r"credential|hf-token|password|secret|token))"
    r"(?P<separator>\s*=\s*|\s+)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|\S+)",
    flags=re.IGNORECASE,
)

_SECRET_OPTIONS = {
    "--access-token",
    "--api-key",
    "--authorization",
    "--client-secret",
    "--credential",
    "--hf-token",
    "--password",
    "--secret",
    "--token",
}


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


def sanitize_cli(cli: str) -> str:
    """Redact secret option values and credentials embedded in source URIs."""

    redacted = _SECRET_OPTION_RE.sub(
        lambda match: (
            f"{match.group('flag')}{match.group('separator')}<redacted>"
        ),
        str(cli),
    )
    return redact_uri_credentials(redacted)


def sanitize_argv(argv: Sequence[str]) -> str:
    """Redact structured command arguments, then render a shell-safe invocation.

    Redacting before joining is important: once argument boundaries are flattened,
    a secret containing whitespace or quotes can be only partly matched and leaked.
    ``sanitize_cli`` remains the best-effort boundary for caller-supplied strings.
    """

    redacted: list[str] = []
    index = 0
    while index < len(argv):
        value = str(argv[index])
        option, separator, _ = value.partition("=")
        if option.casefold() in _SECRET_OPTIONS:
            redacted.append(f"{option}=<redacted>" if separator else option)
            if not separator and index + 1 < len(argv):
                redacted.append("<redacted>")
                index += 1
        else:
            try:
                # argv boundaries are authoritative, so sanitize a whole URI rather
                # than asking the arbitrary-text scanner to guess where it ends.
                sanitized = source_uri(value) if is_qualified_uri(value) else value
            except Exception:  # noqa: BLE001 - provenance must fail closed
                sanitized = "<redacted-uri>" if "://" in value else value
            redacted.append(redact_uri_credentials(sanitized))
        index += 1
    return shlex.join(redacted)


def capture(cli: str | None = None) -> dict[str, object]:
    """Return a provenance dict for embedding in output ``.zattrs`` / receipts."""
    invocation = cli if cli is not None else sanitize_argv(sys.argv)
    prov: dict[str, object] = {
        "raw2features_version": _package_version(),
        "created_utc": now_utc_iso(),
        "cli": sanitize_cli(invocation),
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
