"""Source-URI handling shared by the pipeline and cohort CLI.

Plain local paths deliberately keep the v0.1 behaviour: provenance records an
absolute ``file://`` URI and slide IDs come from the final path component. Qualified
URIs are kept intact for opening the source, but credentials are removed from the
copy used for provenance and output identity. Remote output names also need a stable
fingerprint because basenames such as ``image.zarr`` (or the bioformats2raw series
component ``0``) are common across unrelated sources.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, unquote_plus, urlsplit, urlunsplit

_STORE_SUFFIXES = (".embeddings.zarr", ".ome.zarr", ".zarr")
_UNSAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_GREEDY_QUALIFIED_URI_IN_TEXT = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"<>`]+"
)
_QUALIFIED_URI_IN_TEXT = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://"
    r"(?:(?![A-Za-z][A-Za-z0-9+.-]*://)[^\s\"<>`])+"
)
# Leaves ample room below the common 255-byte filename-component limit for the
# separator, 16-hex fingerprint, and final ``.embeddings.zarr`` suffix.
_MAX_READABLE_PREFIX = 160

# Query parameters whose values are credentials regardless of storage provider.
# Deliberately omit ambiguous names such as ``key`` and ``code`` so genuine source
# selectors survive. Matching is case-insensitive and percent-decoded.
_GENERIC_AUTH_QUERY_KEYS = frozenset(
    {
        "access-token",
        "access_token",
        "api-key",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "awsaccesskeyid",
        "bearer_token",
        "client_secret",
        "credential",
        "googleaccessid",
        "hf-token",
        "hf_token",
        "id_token",
        "key-pair-id",
        "oauth_token",
        "password",
        "passwd",
        "refresh_token",
        "sas_token",
        "secret",
        "security-token",
        "shared_access_signature",
        "sig",
        "signature",
        "subscription-key",
        "token",
        "x-api-key",
    }
)
_NORMALIZED_GENERIC_AUTH_QUERY_KEYS = frozenset(
    re.sub(r"[-_]", "", key) for key in _GENERIC_AUTH_QUERY_KEYS
)

# Azure Shared Access Signature fields. These short names are stripped only for an
# Azure Storage host or when the query otherwise has the shape of an SAS token.
_AZURE_SAS_QUERY_KEYS = frozenset(
    {
        "epk",
        "erk",
        "saoid",
        "scid",
        "sdd",
        "se",
        "ses",
        "si",
        "sig",
        "sip",
        "skdutid",
        "skoid",
        "sks",
        "skt",
        "sktid",
        "ske",
        "skv",
        "sp",
        "spk",
        "spr",
        "sr",
        "srh",
        "srk",
        "srq",
        "srt",
        "ss",
        "st",
        "suoid",
        "sv",
    }
)
_AZURE_STORAGE_HOST_SUFFIXES = (
    ".blob.core.windows.net",
    ".dfs.core.windows.net",
    ".file.core.windows.net",
    ".queue.core.windows.net",
    ".table.core.windows.net",
)

# Legacy S3 and Google Cloud Storage signed URLs predate the X-Amz-/X-Goog-
# parameter families. ``Expires`` is stripped only when one of these signed-URL
# profiles is present because it can otherwise be a meaningful source selector.
_AWS_LEGACY_QUERY_KEYS = frozenset(
    {"awsaccesskeyid", "expires", "security-token", "signature"}
)
_GOOGLE_LEGACY_QUERY_KEYS = frozenset(
    {"expires", "googleaccessid", "signature"}
)
_CLOUDFRONT_QUERY_KEYS = frozenset(
    {"expires", "key-pair-id", "policy", "signature"}
)


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
    """Return the credential-free URI safe to persist in provenance.

    The plain-path branch is intentionally identical to the v0.1 expression so
    existing local stores and receipts keep their recorded values. For a qualified
    URI, userinfo and known authentication query parameters are removed while path,
    semantic query parameters, fragment, ordering, and raw encoding are preserved.
    The caller retains the original URI separately for opening the source.
    """

    value = str(source)
    if is_remote_uri(value):
        return _credential_free_qualified_uri(value)
    if is_qualified_uri(value):
        return value
    return f"file://{os.path.abspath(value)}"


def join_uri_path(source: str, *children: str) -> str:
    """Append path components without moving a URI query onto a child name.

    ``os.path.join("https://host/image.zarr?sig=…", "0")`` produces the invalid
    ``...image.zarr?sig=…/0``. Qualified URIs instead need the child inserted into
    the parsed path, before the unchanged query and fragment. Plain local paths keep
    normal platform path semantics.
    """

    value = str(source)
    if not children:
        return value
    if not is_qualified_uri(value):
        return os.path.join(value, *(str(child) for child in children))

    parsed = urlsplit(value)
    path = parsed.path
    for child in children:
        # NGFF dataset paths are relative to the multiscales group. A leading slash
        # must not discard that group as it would with plain posixpath.join.
        path = posixpath.join(path, str(child).lstrip("/"))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment)
    )


def redact_uri_credentials(text: str) -> str:
    """Remove credentials from qualified URIs embedded in arbitrary text."""

    # The greedy pass keeps an outer URI's query intact when it contains an
    # unescaped nested URL, so auth fields later in that outer query are removed.
    # The split pass then handles credentials in nested or punctuation-adjacent
    # URLs that the greedy parse necessarily treated as part of the outer value.
    redacted = _GREEDY_QUALIFIED_URI_IN_TEXT.sub(_redact_uri_match, str(text))
    return _QUALIFIED_URI_IN_TEXT.sub(_redact_uri_match, redacted)


def redact_metadata_credentials(value: Any) -> Any:
    """Recursively make untrusted plugin metadata safe to persist.

    URI userinfo and signed/auth query values are removed from every string. Values
    under unambiguously credential-bearing keys are replaced even when they are plain
    tokens rather than URLs. Tuples become JSON-equivalent lists.
    """

    if isinstance(value, Mapping):
        safe = {}
        for key, item in value.items():
            name = str(key)
            normalized = re.sub(r"[-_]", "", name).casefold()
            safe[name] = (
                "<redacted>"
                if normalized in _NORMALIZED_GENERIC_AUTH_QUERY_KEYS
                else redact_metadata_credentials(item)
            )
        return safe
    if isinstance(value, (list, tuple)):
        return [redact_metadata_credentials(item) for item in value]
    if isinstance(value, str):
        return redact_uri_credentials(value)
    return value


def redact_metadata_uri_credentials(value: Any) -> Any:
    """Recursively sanitize URI strings without interpreting semantic key names.

    Model constructors and strategy contracts may legitimately use names such as
    ``token`` or ``signature``. Those values affect output identity and must not be
    mistaken for credentials merely because of their key. This narrower helper strips
    credentials only when they occur inside a URI value.
    """

    if isinstance(value, Mapping):
        return {
            str(key): redact_metadata_uri_credentials(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_metadata_uri_credentials(item) for item in value]
    if isinstance(value, str):
        return redact_uri_credentials(value)
    return value


def slide_id_from_source(source: str) -> str:
    """Derive a readable output ID, fingerprinting collision-prone sources.

    Ordinary local paths use the v0.1 basename rule exactly. Every remote URI gets a
    short hash of its credential-free form so two buckets/hosts/semantic queries with
    the same basename cannot overwrite one another, while rotating a signed URL does
    not rename the output. A local ``.../image.zarr/0`` input is the sole local
    exception: ``0`` is not a useful identity, so it uses the parent store name and a
    path fingerprint just like the equivalent remote form. Explicit file URIs derive
    their readable name from the decoded URI path.
    """

    value = str(source)
    qualified = is_qualified_uri(value)
    remote = is_remote_uri(value)
    identity_source = source_uri(value) if qualified else value
    parsed = urlsplit(identity_source) if qualified else None
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
        readable = urlsplit(identity_source).netloc or "slide"
    readable = _safe_id(_strip_store_suffix(unquote(readable)))
    fingerprint = _fingerprint_source(identity_source, qualified)
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
    # Qualified URIs arrive credential-free, matching the value persisted in
    # provenance. Semantic queries, fragments, trailing slashes, and case still
    # matter; rotating a signed URL does not.
    if qualified:
        return value
    return os.path.abspath(value.rstrip("/"))


def _credential_free_qualified_uri(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.casefold() == "file":
        return value
    # The final literal @ separates RFC 3986 userinfo from host. Keeping the raw
    # host/port text avoids otherwise normalising case or percent-encoding.
    has_userinfo = "@" in parsed.netloc
    netloc = parsed.netloc.rsplit("@", 1)[-1]

    raw_fields = _query_fields(parsed.query)
    keys = {_query_key(field) for field in raw_fields}
    host = (parsed.hostname or "").casefold()
    azure_sas = (
        any(host.endswith(suffix) for suffix in _AZURE_STORAGE_HOST_SUFFIXES)
        or ("sig" in keys and bool(keys & (_AZURE_SAS_QUERY_KEYS - {"sig"})))
    )
    aws_legacy = "signature" in keys and "awsaccesskeyid" in keys
    google_legacy = "signature" in keys and "googleaccessid" in keys
    cloudfront = "signature" in keys and "key-pair-id" in keys

    kept_query = [
        field
        for field in raw_fields
        if not _is_auth_query_key(
            _query_key(field),
            azure_sas=azure_sas,
            aws_legacy=aws_legacy,
            google_legacy=google_legacy,
            cloudfront=cloudfront,
        )
    ]
    query = _join_kept_query_fields(parsed.query, raw_fields, kept_query)

    # OAuth-style implicit grants can carry access_token in a query-shaped fragment.
    # Preserve ordinary semantic fragments (``#series-1`` or ``#series=1``) exactly.
    raw_fragment_fields = (
        _query_fields(parsed.fragment) if "=" in parsed.fragment else []
    )
    fragment_keys = {_query_key(field) for field in raw_fragment_fields}
    fragment_azure_sas = "sig" in fragment_keys and bool(
        fragment_keys & (_AZURE_SAS_QUERY_KEYS - {"sig"})
    )
    kept_fragment = [
        field
        for field in raw_fragment_fields
        if not _is_auth_query_key(
            _query_key(field),
            azure_sas=fragment_azure_sas,
            aws_legacy=(
                "signature" in fragment_keys and "awsaccesskeyid" in fragment_keys
            ),
            google_legacy=(
                "signature" in fragment_keys and "googleaccessid" in fragment_keys
            ),
            cloudfront=(
                "signature" in fragment_keys and "key-pair-id" in fragment_keys
            ),
        )
    ]
    fragment = (
        _join_kept_query_fields(
            parsed.fragment, raw_fragment_fields, kept_fragment
        )
        if raw_fragment_fields
        else parsed.fragment
    )

    if (
        not has_userinfo
        and len(kept_query) == len(raw_fields)
        and fragment == parsed.fragment
    ):
        # Avoid normalising an otherwise unauthenticated URI (notably scheme case),
        # which would change the output ID of an existing remote source.
        return value
    raw_scheme = value.partition(":")[0]
    return urlunsplit((raw_scheme, netloc, parsed.path, query, fragment))


def _redact_uri_match(match: re.Match[str]) -> str:
    """Redact one URI while retaining punctuation that merely surrounds it."""

    value = match.group(0)
    suffix = ""
    while value:
        last = value[-1]
        surrounding = last in ".,;!?'" or (
            (last == ")" and value.count(")") > value.count("("))
            or (last == "]" and value.count("]") > value.count("["))
            or (last == "}" and value.count("}") > value.count("{"))
        )
        if not surrounding:
            break
        value = value[:-1]
        suffix = last + suffix
    try:
        return _credential_free_qualified_uri(value) + suffix
    except Exception:  # noqa: BLE001 - this failure boundary must always fail closed
        # This helper runs while handling other failures. A malformed URI must not
        # replace the original exception or fail open with credentials in the log.
        return "<redacted-uri>" + suffix


def _query_key(raw_field: str) -> str:
    return unquote_plus(raw_field.partition("=")[0]).casefold()


def _query_fields(raw: str) -> list[str]:
    """Return legacy ``&``/``;``-delimited fields without decoding their bytes."""

    return re.split(r"[&;]", raw) if raw else []


def _join_kept_query_fields(
    raw: str, fields: list[str], kept_fields: list[str]
) -> str:
    """Remove filtered fields while retaining surviving raw separators and bytes."""

    if len(kept_fields) == len(fields):
        return raw
    keep_counts: dict[str, int] = {}
    for field in kept_fields:
        keep_counts[field] = keep_counts.get(field, 0) + 1

    parts = re.split(r"([&;])", raw)
    selected_indices: list[int] = []
    for index, field in enumerate(fields):
        remaining = keep_counts.get(field, 0)
        if remaining:
            selected_indices.append(index)
            keep_counts[field] = remaining - 1

    if not selected_indices:
        return ""
    output = [fields[selected_indices[0]]]
    for index in selected_indices[1:]:
        output.extend((parts[2 * index - 1], fields[index]))
    return "".join(output)


def _is_auth_query_key(
    key: str,
    *,
    azure_sas: bool,
    aws_legacy: bool,
    google_legacy: bool,
    cloudfront: bool,
) -> bool:
    if key.startswith(("x-amz-", "x-goog-")):
        return True
    if (
        key in _GENERIC_AUTH_QUERY_KEYS
        or re.sub(r"[-_]", "", key) in _NORMALIZED_GENERIC_AUTH_QUERY_KEYS
    ):
        return True
    if azure_sas and key in _AZURE_SAS_QUERY_KEYS:
        return True
    if aws_legacy and key in _AWS_LEGACY_QUERY_KEYS:
        return True
    if google_legacy and key in _GOOGLE_LEGACY_QUERY_KEYS:
        return True
    return cloudfront and key in _CLOUDFRONT_QUERY_KEYS
