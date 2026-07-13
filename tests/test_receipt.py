"""Tests for receipts and validate-against-output idempotency."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from raw2features.pipeline.receipt import (
    Receipt,
    _normalise_output_uri,
    canonical_source_uri,
    config_hash,
    is_complete,
    read_receipt,
    write_receipt,
)
from raw2features.sinks.zarr_sink import ZarrSink


def test_config_hash_order_independent_and_sensitive():
    a = config_hash({"models": ["a", "b"], "x": 1})
    b = config_hash({"x": 1, "models": ["a", "b"]})
    assert a == b
    assert a != config_hash({"models": ["a", "c"], "x": 1})


def test_output_uri_normalisation_preserves_literal_path_characters(tmp_path):
    base = f"file://{tmp_path / 'target'}"
    assert _normalise_output_uri(base + "?query") != _normalise_output_uri(base)
    assert _normalise_output_uri(base + "#fragment") != _normalise_output_uri(base)
    assert _normalise_output_uri(base + "%20name") != _normalise_output_uri(
        base + " name"
    )
    assert _normalise_output_uri(base + "%2Fchild") != _normalise_output_uri(
        base + "/child"
    )


def test_canonical_source_accepts_pathlike_but_rejects_non_string_metadata(tmp_path):
    assert canonical_source_uri(tmp_path / "slide.zarr") == (
        f"file://{tmp_path / 'slide.zarr'}"
    )
    assert canonical_source_uri({"token": "DO_NOT_PRINT"}) is None


def _make_output(tmp_path, n=4, dim=3, source="file:///x"):
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
        header={"schema_version": "0.1", "source": {"uri": source}},
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
    assert is_complete(
        rec_dir,
        "s",
        chash,
        expected_source_uri="file:///x",
        expected_output_uri=uri,
    ) is True
    # Legacy callers no longer raise, but fail closed without the current source.
    assert is_complete(rec_dir, "s", chash) is False
    # wrong config hash -> not complete
    assert is_complete(
        rec_dir, "s", "other", expected_source_uri="file:///x"
    ) is False


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
    assert is_complete(
        rec_dir, "s", "h", expected_source_uri="file:///x"
    ) is False


def test_is_complete_binds_receipt_to_source_and_requested_output(tmp_path):
    rec_dir = str(tmp_path / "rec")
    uri = _make_output(tmp_path / "out")
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri="file:///x",
            output_uri=uri,
            reader="omezarr",
            models=["m"],
            config_hash="h",
            n_patches=4,
        ),
    )

    assert not is_complete(
        rec_dir, "s", "h", expected_source_uri="file:///other"
    )
    assert not is_complete(
        rec_dir,
        "s",
        "h",
        expected_source_uri="file:///x",
        expected_output_uri=f"file://{tmp_path / 'elsewhere' / 's.embeddings.zarr'}",
    )

    receipt = read_receipt(rec_dir, "s")
    assert receipt is not None
    receipt["slide_id"] = "copied-from-another-slide"
    with open(os.path.join(rec_dir, "s.json"), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh)
    assert not is_complete(
        rec_dir, "s", "h", expected_source_uri="file:///x"
    )


def test_is_complete_checks_live_store_source_binding(tmp_path):
    import zarr

    from raw2features.core.store import GRIDS

    rec_dir = str(tmp_path / "rec")
    uri = _make_output(tmp_path / "out")
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri="file:///x",
            output_uri=uri,
            reader="omezarr",
            models=["m"],
            config_hash="h",
            n_patches=4,
        ),
    )
    root = zarr.open_group(uri.removeprefix("file://"), mode="r+")
    key = next(iter(root[GRIDS].keys()))
    header = dict(root[GRIDS][key].attrs["raw2features"])
    header["source"] = {"uri": "file:///different"}
    root[GRIDS][key].attrs["raw2features"] = header

    assert not is_complete(
        rec_dir, "s", "h", expected_source_uri="file:///x"
    )


def test_is_complete_canonicalises_rotated_remote_credentials(tmp_path):
    old = (
        "https://user:old@example.org/image.zarr?series=2&"
        "X-Amz-Credential=old&X-Amz-Signature=old"
    )
    new = (
        "https://user:new@example.org/image.zarr?series=2&"
        "X-Amz-Credential=new&X-Amz-Signature=new"
    )
    rec_dir = str(tmp_path / "rec")
    uri = _make_output(tmp_path / "out", source=old)
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri=old,
            output_uri=uri,
            reader="omezarr",
            models=["m"],
            config_hash="h",
            n_patches=4,
        ),
    )

    assert is_complete(
        rec_dir,
        "s",
        "h",
        expected_source_uri=new,
        expected_output_uri=uri,
    )


def test_corrupt_or_non_object_receipt_is_incomplete(tmp_path):
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    path = rec_dir / "s.json"
    path.write_text('{"status": "complete"')
    assert read_receipt(str(rec_dir), "s") is None
    assert not is_complete(
        str(rec_dir), "s", "h", expected_source_uri="file:///x"
    )

    path.write_text("[]")
    assert read_receipt(str(rec_dir), "s") is None


def test_malformed_recorded_source_fails_closed(tmp_path):
    malformed = "https://user:DO_NOT_PRINT@exa／mple.com/image.zarr"
    rec_dir = str(tmp_path / "rec")
    uri = _make_output(tmp_path / "out", source=malformed)
    write_receipt(
        rec_dir,
        Receipt(
            slide_id="s",
            status="complete",
            source_uri=malformed,
            output_uri=uri,
            reader="omezarr",
            models=["m"],
            config_hash="h",
            n_patches=4,
        ),
    )

    assert not is_complete(
        rec_dir, "s", "h", expected_source_uri="https://example.org/image.zarr"
    )


def test_atomic_write_failure_preserves_previous_receipt(tmp_path, monkeypatch):
    import raw2features.pipeline.receipt as receipt_module

    rec_dir = str(tmp_path / "rec")
    original = Receipt(
        slide_id="s",
        status="complete",
        source_uri="file:///x",
        output_uri="file:///out/s.embeddings.zarr",
        reader="omezarr",
        models=["m"],
        config_hash="old",
    )
    write_receipt(rec_dir, original)

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(receipt_module.os, "replace", fail_replace)
    replacement = Receipt(**{**original.to_json(), "config_hash": "new"})
    with pytest.raises(OSError, match="simulated"):
        write_receipt(rec_dir, replacement)

    with open(os.path.join(rec_dir, "s.json"), encoding="utf-8") as fh:
        assert json.load(fh)["config_hash"] == "old"
    assert not [name for name in os.listdir(rec_dir) if name.endswith(".tmp")]


def test_atomic_replacement_preserves_existing_receipt_mode(tmp_path):
    rec_dir = str(tmp_path / "rec")
    receipt = Receipt(
        slide_id="s",
        status="complete",
        source_uri="file:///x",
        output_uri="file:///out/s.embeddings.zarr",
        reader="omezarr",
        models=["m"],
        config_hash="old",
    )
    path = write_receipt(rec_dir, receipt)
    os.chmod(path, 0o640)

    receipt.config_hash = "new"
    write_receipt(rec_dir, receipt)

    assert os.stat(path).st_mode & 0o777 == 0o640


def test_atomic_temp_name_does_not_overflow_near_name_max(tmp_path):
    slide_id = "x" * 250  # destination is exactly 255 bytes with the .json suffix
    receipt = Receipt(
        slide_id=slide_id,
        status="failed",
        source_uri="file:///x",
        output_uri="",
        reader="omezarr",
        models=["m"],
        config_hash="h",
    )

    path = write_receipt(str(tmp_path), receipt)

    assert len(os.path.basename(path).encode()) == 255
    assert read_receipt(str(tmp_path), slide_id)["slide_id"] == slide_id
