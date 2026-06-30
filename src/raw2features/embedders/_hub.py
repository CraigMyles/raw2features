"""Enforce the weight pins raw2features records (``weights_revision`` / ``sha256``).

Every model in the registry records ``weights_revision`` - an immutable HuggingFace
commit - and ``weights_sha256``. These helpers make those records *enforced* rather
than merely written into provenance:

- :func:`pin_source` threads the revision into the loader so a download resolves to
  one exact commit even though HF repos are mutable (timm parses ``hf-hub:repo@rev``;
  ``from_pretrained`` / ``hf_hub_download`` take ``revision=`` directly).
- :func:`verify_sha256` checks a directly-downloaded checkpoint's bytes against the
  recorded digest *before* it is deserialised - which also closes the
  ``torch.load(weights_only=False)`` arbitrary-code-execution surface, since only
  bytes matching the pinned digest are ever unpickled.
"""

from __future__ import annotations

import hashlib


def pin_source(source: str, revision: str | None) -> str:
    """Return ``source`` with ``@<revision>`` appended for ``hf(-|_)hub:`` ids.

    timm downloads the exact commit from ``hf-hub:owner/repo@<rev>``. Non-hub
    sources (torchvision URIs, bare arch names) and an unset/blank revision pass
    through unchanged, so this is safe to call unconditionally.
    """
    if not revision:
        return source
    if source.startswith(("hf-hub:", "hf_hub:")) and "@" not in source:
        return f"{source}@{revision}"
    return source


def verify_sha256(path: str, expected: str | None, *, what: str) -> None:
    """Raise ``ValueError`` if the file at ``path`` doesn't match ``expected``.

    A no-op when ``expected`` is falsy (nothing recorded to check against). Reads
    in 1 MiB chunks so a multi-GB checkpoint is never held in memory.
    """
    if not expected:
        return
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != expected:
        raise ValueError(
            f"{what}: downloaded weight file sha256 {got} does not match the "
            f"pinned {expected} (registry weights_sha256). Refusing to load - the "
            f"download is corrupt or the pinned revision no longer matches."
        )


def download_pinned_url(url: str, sha256: str | None, *, what: str) -> str:
    """Download ``url`` to a cache dir, verify its sha256, and return the local path.

    For weights pinned to a stable URL outside HuggingFace (e.g. a GitHub release
    asset). Cached under ``$XDG_CACHE_HOME/raw2features/weights`` keyed by sha256 and
    reused on later runs; :func:`verify_sha256` runs every time before the path is
    returned, so the URL + sha256 together are the immutable pin.
    """
    import os
    import shutil
    import tempfile
    import urllib.request

    cache = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "raw2features",
        "weights",
    )
    os.makedirs(cache, exist_ok=True)
    prefix = f"{sha256[:16]}-" if sha256 else ""
    dst = os.path.join(cache, prefix + url.rsplit("/", 1)[-1])
    if not os.path.exists(dst):
        with urllib.request.urlopen(url, timeout=120) as resp:
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
                shutil.copyfileobj(resp, tmp)
                tmp_path = tmp.name
        os.replace(tmp_path, dst)
    verify_sha256(dst, sha256, what=what)
    return dst
