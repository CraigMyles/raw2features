"""Source-URI handling shared by the pipeline and cohort CLI.

Plain local paths deliberately keep the v0.1 behaviour: provenance records an
absolute ``file://`` URI and slide IDs come from the final path component.  Explicit
``file://`` and remote URIs remain byte-for-byte intact in provenance.  Remote output
names also need a stable fingerprint because basenames such as ``image.zarr`` (or the
bioformats2raw series component ``0``) are common across unrelated sources.
"""

from __future__ import annotations

import hashlib
import os
import re
from urllib.parse import unquote, urlsplit

_STORE_SUFFIXES = (".embeddings.zarr", ".ome.zarr", ".zarr")
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
# Leaves ample room below the common 255-byte filename-component limit for the
# separator, 16-hex fingerprint, and final ``.embeddings.zarr`` suffix.
_MAX_READABLE_PREFIX = 160


def is_qualified_uri(source: str) -> bool:
    """Return whether *source* has an explicit ``scheme://`` URI prefix."""

    value = str(source)
    parsed = urlsplit(value)
    return "://" in value and bool(parsed.scheme)


def is_remote_uri(source: str) -> bool:
    """Return whether *source* is a qualified, non-file URI.

    Requiring ``://`` avoids mistaking Windows drive letters or colon-bearing local
    filenames for URI schemes.  ``file://`` remains on the local compatibility path.
    """

    value = str(source)
    return is_qualified_uri(value) and urlsplit(value).scheme.lower() != "file"


def source_uri(source: str) -> str:
    """Return the provenance URI without mangling an already-qualified URI.

    The plain-path branch is intentionally identical to the v0.1 expression so
    existing stores and receipts keep their recorded values.
    """

    value = str(source)
    if is_qualified_uri(value):
        return value
    return f"file://{os.path.abspath(value)}"


def slide_id_from_source(source: str) -> str:
    """Derive a readable output ID, fingerprinting collision-prone sources.

    Ordinary local paths use the v0.1 basename rule exactly.  Every remote URI gets a
    short hash of its exact preserved text so two buckets/hosts/queries with the same
    basename cannot overwrite one another.  A local ``.../image.zarr/0`` input is the
    sole local exception: ``0`` is not a useful identity, so it uses the parent store
    name and a path fingerprint just like the equivalent remote form.  Explicit file
    URIs derive their readable name from the decoded URI path.
    """

    value = str(source)
    qualified = is_qualified_uri(value)
    remote = is_remote_uri(value)
    parsed = urlsplit(value) if qualified else None
    # A file URI represents a local path, so decode its URI path before deriving the
    # same basename-style ID as a plain local source.  Remote path segmentation stays
    # encoded; only its selected readable component is decoded below.
    if parsed and not remote:
        path = unquote(parsed.path)
    elif parsed:
        path = parsed.path
    else:
        path = value
    parts = [part for part in path.rstrip("/").split("/") if part]
    bare_index = (
        len(parts) >= 2
        and parts[-1].isdigit()
        and (remote or _has_store_suffix(parts[-2]))
    )

    if not remote and not bare_index:
        return _strip_local_store_suffix(os.path.basename(path.rstrip("/")))

    if bare_index:
        readable = parts[-2]
    elif parts:
        readable = parts[-1]
    else:
        readable = urlsplit(value).netloc or "slide"
    readable = _safe_id(_strip_store_suffix(unquote(readable)))
    fingerprint = _fingerprint_source(value, qualified)
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    return f"{readable}-{digest}"


def _has_store_suffix(value: str) -> bool:
    lower = value.lower()
    return any(lower.endswith(suffix) for suffix in _STORE_SUFFIXES)


def _strip_local_store_suffix(value: str) -> str:
    """The original v0.1 suffix rule, including its case-sensitive semantics."""

    for suffix in _STORE_SUFFIXES:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _strip_store_suffix(value: str) -> str:
    lower = value.lower()
    for suffix in _STORE_SUFFIXES:
        if lower.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _safe_id(value: str) -> str:
    cleaned = _UNSAFE_ID_CHARS.sub("-", value).strip("._-")
    cleaned = cleaned[:_MAX_READABLE_PREFIX].rstrip("._-")
    return cleaned or "slide"


def _fingerprint_source(value: str, qualified: bool) -> str:
    # Qualified URIs are preserved exactly in provenance, so hash those same bytes:
    # queries, fragments, trailing slashes, and case-sensitive userinfo all matter.
    if qualified:
        return value
    return os.path.abspath(value.rstrip("/"))
