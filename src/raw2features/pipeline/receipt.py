"""Per-slide receipts + idempotent skip-if-complete.

Receipts can lie when a run is interrupted, so "complete" is confirmed by
validating the ACTUAL output store, not just by reading the receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

# The embeddings-store format version, written into every store header and fed into the
# config hash. Tracks the 0.x release line while the format is alpha (one version line:
# package 0.x == format 0.x == schema 0.x). Bumps with each release that changes the
# format; the matching JSON Schema ships under schema/embeddings_store-<version>.json
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
    os.makedirs(receipts_dir, exist_ok=True)
    path = receipt_path(receipts_dir, receipt.slide_id)
    with open(path, "w") as fh:
        json.dump(receipt.to_json(), fh, indent=2)
    return path


def read_receipt(receipts_dir: str, slide_id: str) -> dict | None:
    path = receipt_path(receipts_dir, slide_id)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def validate_model(group, model: str, n_patches: int) -> bool:
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
        if arr.shape[0] != n_patches:
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


def validate_output(output_uri: str, models: list[str], n_patches: int) -> bool:
    """Open the output zarr and confirm it actually matches the receipt."""
    from raw2features.core.store import open_grid

    path = output_uri.removeprefix("file://")
    if not os.path.exists(path):
        return False
    try:
        g = open_grid(path)  # the sole grid this single-grid validator checks
        if g["coords"].shape[0] != n_patches:
            return False
        for model in models:
            if not validate_model(g, model, n_patches):
                return False
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return True


def validate_store_models(output_uri: str, models: list[str]) -> bool:
    """True iff every model in *models* is present and fully written in some grid.

    Multi-grid generalisation of :func:`validate_output`: a store may hold several
    ``grids/<key>/`` (one per geometry), and a model is 'complete' if some
    ``grids/<key>/features/<model>`` is finite and free of all-zero fill rows.
    """
    from raw2features.core.store import GRIDS, grid_keys, open_root

    path = output_uri.removeprefix("file://")
    if not os.path.exists(path):
        return False
    try:
        root = open_root(path)
        keys = grid_keys(root)
        if not keys:
            return False
        for m in models:
            found = False
            for k in keys:
                g = root[GRIDS][k]
                if "features" in g and m in g["features"]:
                    if validate_model(g, m, int(g["coords"].shape[0])):
                        found = True
                        break
            if not found:
                return False
    except Exception:  # noqa: BLE001 - any failure means "not valid"
        return False
    return True


def is_complete(receipts_dir: str, slide_id: str, expected_hash: str) -> bool:
    """True iff a matching, output-validated 'complete' receipt exists."""
    rec = read_receipt(receipts_dir, slide_id)
    if not rec or rec.get("status") != "complete":
        return False
    if rec.get("config_hash") != expected_hash:
        return False
    return validate_store_models(rec.get("output_uri", ""), rec.get("models", []))
