"""Per-slide receipts + idempotent skip-if-complete.

Receipts can lie when a run is interrupted, so "complete" is confirmed by
validating the ACTUAL output store, not just by reading the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from dataclasses import asdict, dataclass, field

from raw2features.core.uris import source_uri

# The embeddings-store format version, written into every store header and fed into the
# config hash. It is independent of the package version and changes only when the
# normative store contract changes; package v0.2.0 therefore continues to write schema
# 0.1. The matching JSON Schema ships under schema/embeddings_store-<version>.json.
SCHEMA_VERSION = "0.1"


@dataclass
class Receipt:
    slide_id: str
    status: str  # pending | running | complete | failed
    source_uri: str
    output_uri: str
    reader: str
    models: list[str]
    config_hash: str
    n_patches: int = 0
    model_dims: dict[str, int] = field(default_factory=dict)
    started_utc: str | None = None
    finished_utc: str | None = None
    elapsed_s: float | None = None
    host: str | None = None
    raw2features_version: str | None = None
    schema_version: str = SCHEMA_VERSION
    error: str | None = None

    def to_json(self) -> dict:
        return asdict(self)


def config_hash(config: dict) -> str:
    """Stable hash of the run configuration (order-independent)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def receipt_path(receipts_dir: str, slide_id: str) -> str:
    return os.path.join(receipts_dir, f"{slide_id}.json")


def write_receipt(receipts_dir: str, receipt: Receipt) -> str:
    """Atomically replace one receipt with a complete JSON document.

    The temporary file lives beside the destination so ``os.replace`` is atomic on
    the target filesystem.  A failed write therefore leaves the previous receipt
    intact instead of truncating it into a permanently unreadable resume marker.
    """
    os.makedirs(receipts_dir, exist_ok=True)
    path = receipt_path(receipts_dir, receipt.slide_id)
    tmp_path: str | None = None
    fd: int | None = None
    try:
        try:
            existing_mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            existing_mode = None
        for _ in range(100):
            tmp_path = os.path.join(
                receipts_dir,
                f".receipt.{secrets.token_hex(8)}.tmp",
            )
            try:
                # Unlike NamedTemporaryFile's fixed 0600, this honours the process
                # umask/default ACL just like the old open(path, "w") implementation.
                fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
                break
            except FileExistsError:
                tmp_path = None
        else:  # pragma: no cover - 100 cryptographic-name collisions is infeasible
            raise FileExistsError("could not allocate a temporary receipt path")
        if existing_mode is not None:
            os.chmod(tmp_path, existing_mode)
        with os.fdopen(fd, mode="w", encoding="utf-8") as fh:
            fd = None  # ownership transferred to ``fh``
            json.dump(receipt.to_json(), fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:  # noqa: BLE001 - cleanup on interrupts as well as failures
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise
    return path


def read_receipt(receipts_dir: str, slide_id: str) -> dict | None:
    path = receipt_path(receipts_dir, slide_id)
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _normalise_output_uri(value: str) -> str:
    """Canonical comparison form for a receipt's output target."""

    value = str(value)
    if value.startswith("file://"):
        # raw2features receipts intentionally use ``file://`` + the raw absolute path,
        # not an RFC-percent-encoded URI. Preserve literal ``?``, ``#`` and ``%2F`` in
        # filenames exactly as the store-opening code does.
        path = os.path.realpath(os.path.abspath(value.removeprefix("file://")))
        return f"file://{path}"
    return source_uri(value)


def canonical_source_uri(value: str | os.PathLike[str]) -> str | None:
    """Credential-free comparison form, or ``None`` for malformed provenance."""

    try:
        value = os.fspath(value)
    except TypeError:
        return None
    if not isinstance(value, str) or not value:
        return None
    try:
        return source_uri(value)
    except Exception:  # noqa: BLE001 - invalid provenance must fail closed
        return None


def store_source_bindings(output_uri: str) -> list[tuple[str, str | None]]:
    """Return live root/grid source bindings from an embeddings store.

    Reading live metadata matters after an additive write: consolidated metadata may
    predate a newly-added grid. Each v0.1 root and grid is required to identify its
    source, so callers can reject missing or mutually inconsistent provenance.
    """

    import zarr

    from raw2features.core.store import GRIDS, grid_keys

    path = str(output_uri).removeprefix("file://")
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    bindings: list[tuple[str, str | None]] = []

    def _record(label: str, attrs) -> None:
        header = dict(attrs.get("raw2features", {}))
        source = header.get("source")
        uri = source.get("uri") if isinstance(source, dict) else None
        bindings.append((label, uri if isinstance(uri, str) and uri else None))

    _record("root", root.attrs)
    for key in grid_keys(root):
        _record(f"grids/{key}", root[GRIDS][key].attrs)
    return bindings


def validate_store_source(output_uri: str, expected_source_uri: str) -> bool:
    """True iff the root and every grid bind to exactly the expected source."""

    try:
        expected = canonical_source_uri(expected_source_uri)
        bindings = store_source_bindings(output_uri)
    except Exception:  # noqa: BLE001 - unreadable output is not a valid binding
        return False
    if expected is None or not bindings:
        return False
    return all(
        recorded is not None and canonical_source_uri(recorded) == expected
        for _, recorded in bindings
    )


def validate_model(
    group,
    model: str,
    n_patches: int,
    *,
    expected_dim: int | None = None,
    expected_fingerprint: dict | None = None,
) -> bool:
    """True iff ``features/<model>`` in an open zarr group holds complete data.

    Streams the array in row-blocks to bound memory: every row must be finite. A zarr
    shape is metadata only; an unwritten tail reads back as the finite fill value (0.0),
    so a finiteness check alone would pass a truncated store as "complete". Writes go in
    coord order, so the unwritten part is always a contiguous *suffix* -- an all-zero
    **last** row signals truncation. Checking only the last row (not every row) lets a
    model emit a legitimately all-zero feature row mid-array without making resume loop.
    """
    import numpy as np

    try:
        if "features" not in group or model not in group["features"]:
            return False
        arr = group["features"][model]
        if arr.ndim != 2 or arr.shape[0] != n_patches:
            return False
        if expected_dim is not None and arr.shape[1] != int(expected_dim):
            return False
        if expected_fingerprint is not None:
            from raw2features.embedders.fingerprint import (
                output_fingerprints_equal,
            )

            # The array record is the post-write commit marker.  The mirrored header
            # is independently required so an old provenance block is never used to
            # reconstruct/bless a legacy array whose actual loader contract is unknown.
            array_fingerprint = dict(arr.attrs).get("output_fingerprint")
            header = dict(group.attrs.get("raw2features", {}))
            models = header.get("models", {})
            model_meta = models.get(model, {}) if isinstance(models, dict) else {}
            header_fingerprint = (
                model_meta.get("output_fingerprint")
                if isinstance(model_meta, dict)
                else None
            )
            if not output_fingerprints_equal(array_fingerprint, expected_fingerprint):
                return False
            if not output_fingerprints_equal(header_fingerprint, expected_fingerprint):
                return False
        block = 8192
        for s in range(0, n_patches, block):
            rows = np.asarray(arr[s : s + block]).astype(np.float32, copy=False)
            if not np.isfinite(rows).all():
                return False
        if n_patches > 0 and not np.asarray(arr[n_patches - 1]).any():  # all-zero tail
            return False
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return True


def validate_output(
    output_uri: str,
    models: list[str],
    n_patches: int,
    *,
    expected_model_contracts: dict[str, dict] | None = None,
) -> bool:
    """Open the output zarr and confirm it actually matches the receipt."""
    import zarr

    from raw2features.core.store import open_grid

    if expected_model_contracts is not None and not set(models) <= set(
        expected_model_contracts
    ):
        return False
    path = output_uri.removeprefix("file://")
    if not os.path.exists(path):
        return False
    try:
        root = zarr.open_group(path, mode="r", use_consolidated=False)
        g = open_grid(root)  # the sole grid this single-grid validator checks
        if g["coords"].shape[0] != n_patches:
            return False
        for model in models:
            contract = (expected_model_contracts or {}).get(model, {})
            if not validate_model(
                g,
                model,
                n_patches,
                expected_dim=contract.get("embedding_dim"),
                expected_fingerprint=contract.get("output_fingerprint"),
            ):
                return False
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return True


def validate_store_models(
    output_uri: str,
    models: list[str],
    *,
    expected_model_contracts: dict[str, dict] | None = None,
    expected_grid_models: dict[str, list[str]] | None = None,
    compatible_grid_hashes: dict[str, tuple[str, ...]] | None = None,
    compatible_grid_segmenters: dict[str, dict[str, str]] | None = None,
    allow_hashless_legacy_grids: dict[str, bool] | None = None,
) -> bool:
    """True iff the requested model outputs are complete in their requested grids.

    ``expected_grid_models`` binds each requested model set to its authoritative full
    ``grid_hash``.  A valid same-named model in another grid must not satisfy that
    requirement. ``compatible_grid_hashes`` retains explicitly-supported legacy
    identities. ``compatible_grid_segmenters`` can require live segmentation evidence
    for an otherwise ambiguous alias. A hashless store is accepted only for the
    unambiguous one-grid case, when its per-grid policy permits it, and, when evidence
    is required, a matching live header. Missing policy entries retain the historical
    permissive fallback for API compatibility.

    When no per-grid mapping is supplied, the historical cross-grid union behaviour is
    retained for API compatibility.  Production resume/verify callers supply the map.
    """
    import zarr

    from raw2features.core.store import GRIDS, grid_keys

    if expected_model_contracts is not None and not set(models) <= set(
        expected_model_contracts
    ):
        return False
    path = output_uri.removeprefix("file://")
    if not os.path.exists(path):
        return False
    try:
        # Never let a stale consolidated view bless a replacement that crashed before
        # its live array fingerprint (the post-write commit marker) was restored.
        root = zarr.open_group(path, mode="r", use_consolidated=False)
        keys = grid_keys(root)
        if not keys:
            return False
        if expected_grid_models is not None:
            required_models = {
                model
                for requested in expected_grid_models.values()
                for model in requested
            }
            if set(models) != required_models:
                return False

            stored_hashes = {
                key: dict(root[GRIDS][key].attrs.get("raw2features", {})).get(
                    "grid_hash"
                )
                for key in keys
            }
            sole_hashless = (
                len(expected_grid_models) == 1
                and len(keys) == 1
                and stored_hashes[keys[0]] is None
            )
            used: set[str] = set()
            for expected_hash, requested in expected_grid_models.items():
                segmenter_requirements = (compatible_grid_segmenters or {}).get(
                    expected_hash, {}
                )
                candidates = tuple(
                    dict.fromkeys(
                        (
                            expected_hash,
                            *(compatible_grid_hashes or {}).get(expected_hash, ()),
                        )
                    )
                )
                matches: list[str] = []
                # Prefer the current hash, then each explicitly-supported legacy
                # candidate. A store can legitimately contain an old and a rebuilt
                # current grid; the current one must win instead of making the
                # receipt permanently ambiguous.
                for candidate in candidates:
                    required_segmenter = segmenter_requirements.get(candidate)
                    matches = [
                        key
                        for key in keys
                        if key not in used
                        and stored_hashes[key] == candidate
                        and (
                            required_segmenter is None
                            or (
                                dict(
                                    root[GRIDS][key].attrs.get("raw2features", {})
                                ).get("segmentation")
                                or {}
                            ).get("segmenter")
                            == required_segmenter
                        )
                    ]
                    if matches:
                        break
                hashless_allowed = (allow_hashless_legacy_grids or {}).get(
                    expected_hash, True
                )
                if not matches and sole_hashless and hashless_allowed:
                    required = set(segmenter_requirements.values())
                    if not required or (
                        len(required) == 1
                        and (
                            dict(
                                root[GRIDS][keys[0]].attrs.get("raw2features", {})
                            ).get("segmentation")
                            or {}
                        ).get("segmenter")
                        == next(iter(required))
                    ):
                        matches = [keys[0]]
                if len(matches) != 1:
                    return False
                key = matches[0]
                used.add(key)
                group = root[GRIDS][key]
                n_patches = int(group["coords"].shape[0])
                for model in requested:
                    contract = (expected_model_contracts or {}).get(model, {})
                    if not validate_model(
                        group,
                        model,
                        n_patches,
                        expected_dim=contract.get("embedding_dim"),
                        expected_fingerprint=contract.get("output_fingerprint"),
                    ):
                        return False
            return True

        for m in models:
            found = False
            for k in keys:
                g = root[GRIDS][k]
                if "features" in g and m in g["features"]:
                    contract = (expected_model_contracts or {}).get(m, {})
                    if validate_model(
                        g,
                        m,
                        int(g["coords"].shape[0]),
                        expected_dim=contract.get("embedding_dim"),
                        expected_fingerprint=contract.get("output_fingerprint"),
                    ):
                        found = True
                        break
            if not found:
                return False
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return True


def is_complete(
    receipts_dir: str,
    slide_id: str,
    expected_hash: str,
    *,
    expected_source_uri: str | None = None,
    expected_output_uri: str | None = None,
    expected_model_contracts: dict[str, dict] | None = None,
    expected_grid_models: dict[str, list[str]] | None = None,
    compatible_grid_hashes: dict[str, tuple[str, ...]] | None = None,
    compatible_grid_segmenters: dict[str, dict[str, str]] | None = None,
    allow_hashless_legacy_grids: dict[str, bool] | None = None,
) -> bool:
    """True iff a request-bound, output-validated ``complete`` receipt exists.

    The source argument remains optional for call compatibility, but omitting it
    fails closed: a receipt cannot safely short-circuit without the current source.
    """
    rec = read_receipt(receipts_dir, slide_id)
    if not rec or rec.get("status") != "complete":
        return False
    if rec.get("slide_id") != slide_id:
        return False
    if rec.get("config_hash") != expected_hash:
        return False
    recorded_source = rec.get("source_uri")
    if not recorded_source or expected_source_uri is None:
        return False
    expected_source = canonical_source_uri(expected_source_uri)
    recorded_source_canonical = canonical_source_uri(recorded_source)
    if expected_source is None or recorded_source_canonical != expected_source:
        return False
    recorded_output = rec.get("output_uri")
    if not recorded_output:
        return False
    if expected_output_uri is not None:
        try:
            if _normalise_output_uri(recorded_output) != _normalise_output_uri(
                expected_output_uri
            ):
                return False
        except Exception:  # noqa: BLE001 - malformed target cannot be complete
            return False
    if not validate_store_source(recorded_output, expected_source):
        return False
    models = rec.get("models", [])
    if not isinstance(models, list) or not all(isinstance(m, str) for m in models):
        return False
    if expected_model_contracts is not None and set(models) != set(
        expected_model_contracts
    ):
        return False
    return validate_store_models(
        recorded_output,
        models,
        expected_model_contracts=expected_model_contracts,
        expected_grid_models=expected_grid_models,
        compatible_grid_hashes=compatible_grid_hashes,
        compatible_grid_segmenters=compatible_grid_segmenters,
        allow_hashless_legacy_grids=allow_hashless_legacy_grids,
    )
