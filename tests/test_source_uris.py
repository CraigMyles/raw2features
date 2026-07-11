"""Remote provenance, manifest resolution, and collision-resistant slide IDs."""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace

import pytest

from raw2features.cli.embed_many import _resolve_manifest_sources
from raw2features.core.geometry import Size
from raw2features.core.uris import is_qualified_uri, is_remote_uri, source_uri
from raw2features.pipeline import runner
from raw2features.pipeline.runner import RunConfig, slide_id_from_path


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.org/cohort/image.ome.zarr/0?token=a%2Fb",
        "http://example.org/cohort/image.zarr",
        "s3://bucket/cohort/image.zarr/0",
        "gs://bucket/cohort/image.zarr",
    ],
)
def test_remote_source_uris_are_recognised_and_preserved_verbatim(uri):
    assert is_remote_uri(uri)
    assert source_uri(uri) == uri


def test_plain_local_source_uri_keeps_v01_abspath_behaviour(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = os.path.join("cohort", "image.ome.zarr")
    assert not is_remote_uri(source)
    assert source_uri(source) == f"file://{os.path.abspath(source)}"


def test_explicit_file_uri_is_preserved_and_id_uses_decoded_path():
    uri = "file:///data/cohort/case%2001.ome.zarr"
    assert is_qualified_uri(uri)
    assert not is_remote_uri(uri)
    assert source_uri(uri) == uri
    assert slide_id_from_path(uri) == "case 01"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("/data/case-01.ome.zarr", "case-01"),
        ("relative/case-02.zarr", "case-02"),
        ("/data/case-03.embeddings.zarr", "case-03"),
        ("/data/a slide.svs", "a slide.svs"),
        ("/data/CASE.ZARR", "CASE.ZARR"),
    ],
)
def test_ordinary_local_slide_ids_are_backward_compatible(source, expected):
    assert slide_id_from_path(source) == expected


def test_bare_series_index_uses_readable_parent_and_fingerprint(tmp_path):
    source = str(tmp_path / "image.ome.zarr" / "0")
    slide_id = slide_id_from_path(source)
    assert slide_id.startswith("image-")
    assert slide_id != "0"
    assert slide_id == slide_id_from_path(source)


def test_numeric_file_uri_uses_decoded_parent_and_exact_uri_fingerprint():
    uri = "file:///data/cohort/case%2001.ome.zarr/0?version=1#series"
    slide_id = slide_id_from_path(uri)
    digest = hashlib.sha256(uri.encode()).hexdigest()[:16]
    assert slide_id == f"case-01-{digest}"


def test_distinct_remote_sources_with_same_basename_get_distinct_readable_ids():
    first = "s3://bucket-a/cohort/image.ome.zarr/0"
    second = "s3://bucket-b/cohort/image.ome.zarr/0"
    first_id = slide_id_from_path(first)
    second_id = slide_id_from_path(second)
    assert first_id.startswith("image-")
    assert second_id.startswith("image-")
    assert len(first_id.rsplit("-", 1)[-1]) == 16
    assert first_id != second_id
    assert first_id == slide_id_from_path(first)


def test_remote_bare_index_without_zarr_suffix_uses_parent_name():
    slide_id = slide_id_from_path("https://example.org/tiled/case-42/0")
    assert slide_id.startswith("case-42-")


def test_remote_query_is_excluded_from_readable_name_but_included_in_identity():
    base = "https://example.org/cohort/case%2001.ome.zarr/0"
    first_id = slide_id_from_path(f"{base}?version=1&token=a%2Fb")
    second_id = slide_id_from_path(f"{base}?version=2&token=a%2Fb")
    assert first_id.startswith("case-01-")
    assert "?" not in first_id and "&" not in first_id
    assert first_id != second_id


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (
            "https://example.org/cohort/case.ome.zarr#series-a",
            "https://example.org/cohort/case.ome.zarr#series-b",
        ),
        (
            "https://example.org/cohort/case.ome.zarr",
            "https://example.org/cohort/case.ome.zarr/",
        ),
        (
            "https://User:Secret@example.org/cohort/case.ome.zarr",
            "https://user:secret@example.org/cohort/case.ome.zarr",
        ),
    ],
)
def test_remote_fingerprint_hashes_exact_preserved_uri_bytes(first, second):
    first_id = slide_id_from_path(first)
    second_id = slide_id_from_path(second)
    assert first_id.endswith(hashlib.sha256(first.encode()).hexdigest()[:16])
    assert second_id.endswith(hashlib.sha256(second.encode()).hexdigest()[:16])
    assert first_id != second_id


def test_remote_readable_prefix_is_capped_for_safe_store_filename_length():
    uri = f"https://example.org/cohort/{'a' * 1000}.ome.zarr"
    slide_id = slide_id_from_path(uri)
    readable, digest = slide_id.rsplit("-", 1)
    assert len(readable.encode()) <= 160
    assert digest == hashlib.sha256(uri.encode()).hexdigest()[:16]
    assert len(f"{slide_id}.embeddings.zarr".encode()) <= 255


def test_manifest_resolution_preserves_remote_and_resolves_only_relative_local():
    remote = "https://example.org/cohort/image.ome.zarr/0?token=abc"
    file_uri = "file:///mounted/cohort/image%202.ome.zarr"
    rows = [
        {"path": remote},
        {"path": file_uri},
        {"path": "relative/image.zarr", "source_mpp": 0.5},
        {"path": "/absolute/image.zarr"},
    ]
    got = _resolve_manifest_sources(rows, "/slides")
    assert got == [
        {"path": remote},
        {"path": file_uri},
        {"path": "/slides/relative/image.zarr", "source_mpp": 0.5},
        {"path": "/absolute/image.zarr"},
    ]


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.org/cohort/image.ome.zarr/0?token=a%2Fb",
        "file:///mounted/cohort/image%202.ome.zarr",
    ],
)
def test_store_header_preserves_qualified_source_uri_verbatim(uri):
    reader = SimpleNamespace(
        path=uri,
        ngff_version="0.4",
        name="omezarr",
        mpp=0.5,
        level_dimensions=[Size(100, 80)],
        level_downsamples=lambda: [1.0],
        axes=(),
        axis_units={},
        scale_um={},
        level0_translation_um=None,
    )
    grid = SimpleNamespace(
        target_mpp=0.5,
        achieved_mpp=0.5,
        patch_px=224,
        step_out_px=224,
        read_level=0,
        read_px=224,
        resample=1.0,
        needs_resample=False,
        level0_patch=448,
        level0_step=448,
        n_rows=1,
        n_cols=1,
    )
    header = runner._build_header(
        reader, grid, {"segmenter": "none"}, [], "image", 1, None, "hash", {}
    )
    assert header["source"]["uri"] == uri


def test_multigrid_receipt_preserves_remote_source_uri(monkeypatch, tmp_path):
    uri = "s3://bucket/cohort/image.ome.zarr/0?version=7"
    captured = []
    monkeypatch.setattr(runner, "is_complete", lambda *args: False)
    monkeypatch.setattr(
        runner,
        "run_slide",
        lambda *args, **kwargs: {
            "status": "complete",
            "slide_embeddings": {},
        },
    )
    monkeypatch.setattr(
        runner, "write_receipt", lambda _directory, rec: captured.append(rec)
    )

    summary = runner.embed_slide(
        uri,
        str(tmp_path / "out"),
        RunConfig(models=["resnet50"]),
        receipts_dir=str(tmp_path / "receipts"),
    )

    assert summary["status"] == "complete"
    assert len(captured) == 1
    assert captured[0].source_uri == uri
    assert captured[0].slide_id == summary["slide_id"]
