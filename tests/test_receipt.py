"""Tests for receipts and validate-against-output idempotency."""

from __future__ import annotations

import numpy as np

from raw2features.pipeline.receipt import (
    Receipt,
    config_hash,
    is_complete,
    write_receipt,
)
from raw2features.sinks.zarr_sink import ZarrSink


def test_config_hash_order_independent_and_sensitive():
    a = config_hash({"models": ["a", "b"], "x": 1})
    b = config_hash({"x": 1, "models": ["a", "b"]})
    assert a == b
    assert a != config_hash({"models": ["a", "c"], "x": 1})


def _make_output(tmp_path, n=4, dim=3):
    coords = np.zeros((n, 2), "int32")
    sink = ZarrSink()
    sink.create(
        str(tmp_path),
        "s",
        grid="mpp1_px224",
        n_patches=n,
        coords=coords,
        grid_index=coords,
        grid_tissue=None,
        model_dims={"m": dim},
        header={"schema_version": "0.1"},
    )
    sink.write_block("m", 0, np.ones((n, dim), "float32"))
    sink.close()
    return sink.uri


def test_is_complete_true_then_false_on_mismatch(tmp_path):
    out_dir = tmp_path / "out"
    rec_dir = str(tmp_path / "rec")
    uri = _make_output(out_dir, n=4, dim=3)
    chash = "deadbeef"
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri="file:///x",
            output_uri=uri,
            reader="omezarr",
            models=["m"],
            config_hash=chash,
            n_patches=4,
        ),
    )
    assert is_complete(rec_dir, "s", chash) is True
    # wrong config hash -> not complete
    assert is_complete(rec_dir, "s", "other") is False


def test_is_complete_false_when_output_missing_model(tmp_path):
    out_dir = tmp_path / "out"
    rec_dir = str(tmp_path / "rec")
    uri = _make_output(out_dir, n=4, dim=3)
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri="file:///x",
            output_uri=uri,
            reader="omezarr",
            models=["m", "missing"],  # 'missing' not in output
            config_hash="h",
            n_patches=4,
        ),
    )
    assert is_complete(rec_dir, "s", "h") is False
