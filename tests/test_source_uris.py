"""Remote provenance, manifest resolution, and collision-resistant slide IDs."""

from __future__ import annotations

import hashlib
import importlib
import os
from types import SimpleNamespace

import numpy as np
import pytest
import typer

from raw2features.cli.embed_many import _manifest_sort_key, _resolve_manifest_sources
from raw2features.cli.info import info as info_command
from raw2features.core import plugins
from raw2features.core.geometry import Size
from raw2features.core.uris import (
    is_qualified_uri,
    is_remote_uri,
    join_uri_path,
    redact_uri_credentials,
    source_uri,
)
from raw2features.pipeline import runner
from raw2features.pipeline.receipt import read_receipt
from raw2features.pipeline.runner import RunConfig, slide_id_from_path
from raw2features.readers.base import WSISource


@pytest.mark.parametrize(
    "uri",
    [
        "http://example.org/cohort/image.zarr",
        "HTTPS://Example.org/cohort/image.zarr?series=1&empty=#view",
        "s3://bucket/cohort/image.zarr/0",
        "gs://bucket/cohort/image.zarr",
    ],
)
def test_unauthenticated_remote_source_uris_are_preserved_verbatim(uri):
    assert is_remote_uri(uri)
    assert source_uri(uri) == uri


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "https://User:Secret@[2001:db8::1]:8443/image.zarr?series=1&token=S",
            "https://[2001:db8::1]:8443/image.zarr?series=1",
        ),
        (
            "https://s3.example/image.zarr?versionId=v%2F1&X-Amz-Algorithm=A&"
            "x-amz-credential=C&X-Amz-Signature=S#series-a",
            "https://s3.example/image.zarr?versionId=v%2F1#series-a",
        ),
        (
            "https://storage.example/image.zarr?X-Goog-Date=D&generation=42&"
            "x-goog-signature=S",
            "https://storage.example/image.zarr?generation=42",
        ),
        (
            "https://account.blob.core.windows.net/c/image.zarr?snapshot=7&sv=V&"
            "st=T&se=E&sp=r&spr=https&sr=c&sig=S",
            "https://account.blob.core.windows.net/c/image.zarr?snapshot=7",
        ),
        (
            "https://s3.example/image.zarr?versionId=7&AWSAccessKeyId=K&"
            "Signature=S&Expires=9",
            "https://s3.example/image.zarr?versionId=7",
        ),
        (
            "https://storage.example/image.zarr?generation=7&GoogleAccessId=I&"
            "Signature=S&Expires=9",
            "https://storage.example/image.zarr?generation=7",
        ),
        (
            "https://cdn.example/image.zarr?series=7&Policy=P&Signature=S&"
            "Key-Pair-Id=K",
            "https://cdn.example/image.zarr?series=7",
        ),
        (
            "https://example.org/image.zarr?series=2&To%6Ben=A&ACCESS_TOKEN=B&"
            "api_key=C&accessToken=D&clientSecret=E&channel=H%26E&series=3&empty=",
            "https://example.org/image.zarr?series=2&channel=H%26E&series=3&empty=",
        ),
        (
            "https://example.org/image.zarr#access_token=S&state=kept",
            "https://example.org/image.zarr#state=kept",
        ),
        (
            "https://example.org/image.zarr?series=1;token=S;channel=2&empty=",
            "https://example.org/image.zarr?series=1;channel=2&empty=",
        ),
    ],
)
def test_remote_source_uri_removes_credentials_and_keeps_selectors(uri, expected):
    assert source_uri(uri) == expected
    assert "Secret" not in source_uri(uri)
    assert "=S" not in source_uri(uri)


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


def test_explicit_file_uri_remains_exact_even_with_query_text():
    uri = "file:///data/case.ome.zarr?token=local-selector#series"
    assert source_uri(uri) == uri


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
            "https://example.org/case.zarr?versionId=7&X-Amz-Date=1&"
            "X-Amz-Signature=first",
            "https://example.org/case.zarr?X-Amz-Signature=second&versionId=7&"
            "X-Amz-Date=2",
        ),
        (
            "https://first:secret@example.org/case.zarr?series=1",
            "https://second:rotated@example.org/case.zarr?series=1",
        ),
        (
            "https://example.org/case.zarr#access_token=first&state=1",
            "https://example.org/case.zarr#access_token=second&state=1",
        ),
    ],
)
def test_credential_rotation_does_not_change_remote_slide_id(first, second):
    assert source_uri(first) == source_uri(second)
    assert slide_id_from_path(first) == slide_id_from_path(second)


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


def test_remote_manifest_sort_key_ignores_rotating_credentials():
    first = {"path": "https://example.org/image.zarr?series=2&token=first"}
    second = {"path": "https://user:second@example.org/image.zarr?series=2&token=second"}
    assert _manifest_sort_key(first) == _manifest_sort_key(second)


def test_info_prints_credential_free_source_uri(monkeypatch):
    sentinel = "R2F_INFO_SECRET"
    uri = f"https://user:{sentinel}@example.org/image.zarr?series=2&token={sentinel}"
    output = []

    class FakeReader:
        path = uri
        ngff_version = "0.4"
        mpp = None
        level_dimensions = []

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def level_downsamples(self):
            return []

    monkeypatch.setattr(plugins, "get", lambda *_args: FakeReader)
    monkeypatch.setattr(typer, "echo", output.append)

    info_command(uri, mpp=1.0, patch_size=224, reader="fake")

    rendered = "\n".join(str(line) for line in output)
    assert "path:          https://example.org/image.zarr?series=2" in rendered
    assert sentinel not in rendered


@pytest.mark.parametrize(
    ("source", "children", "expected"),
    [
        (
            "https://example.org/image.zarr?token=S#view",
            ("0", "2"),
            "https://example.org/image.zarr/0/2?token=S#view",
        ),
        (
            "s3://bucket/image.zarr?X-Amz-Signature=S",
            ("0",),
            "s3://bucket/image.zarr/0?X-Amz-Signature=S",
        ),
        ("/data/image.zarr", ("0", "2"), "/data/image.zarr/0/2"),
    ],
)
def test_join_uri_path_inserts_children_before_query(source, children, expected):
    assert join_uri_path(source, *children) == expected


def test_redact_uri_credentials_sanitizes_urls_inside_error_text():
    text = (
        "failed (https://user:password@example.org/image.zarr?"
        "series=1&token=SENTINEL). retry"
    )
    got = redact_uri_credentials(text)
    assert got == "failed (https://example.org/image.zarr?series=1). retry"
    assert "password" not in got and "SENTINEL" not in got


def test_redact_uri_credentials_never_raises_for_malformed_uri():
    got = redact_uri_credentials("failed http://[invalid/path?token=R2F_SECRET")
    assert got == "failed <redacted-uri>"
    assert "R2F_SECRET" not in got


def test_redact_uri_credentials_handles_adjacent_uris_without_truncating_commas():
    text = (
        "failed https://safe.example/a,"
        "https://user:R2F_SECRET@other.example/b?selector=a,b&token=R2F_SECRET"
    )
    got = redact_uri_credentials(text)
    assert got == (
        "failed https://safe.example/a,https://other.example/b?selector=a,b"
    )
    assert "R2F_SECRET" not in got


def test_redact_uri_credentials_sanitizes_outer_and_nested_url_credentials():
    text = (
        "failed https://host.example/login?"
        "redirect=https://user:NESTED_SECRET@safe.example/path&token=OUTER_SECRET"
    )
    got = redact_uri_credentials(text)
    assert got == (
        "failed https://host.example/login?redirect=https://safe.example/path"
    )
    assert "NESTED_SECRET" not in got and "OUTER_SECRET" not in got


def test_cli_boundary_redacts_signed_url_errors_but_preserves_other_tracebacks(
    monkeypatch, capsys
):
    main_module = importlib.import_module("raw2features.cli.main")
    sentinel = "R2F_CLI_ERROR_SECRET"
    remote_error = RuntimeError(
        f"403 for https://user:{sentinel}@example.org/image.zarr/0/.zattrs?"
        f"series=2;token={sentinel}"
    )

    def fail_with_remote_url():
        raise remote_error

    monkeypatch.setattr(main_module, "app", fail_with_remote_url)
    with pytest.raises(SystemExit) as caught:
        main_module.main()
    assert caught.value.code == 2
    rendered = capsys.readouterr().err
    assert rendered == (
        "Error: 403 for https://example.org/image.zarr/0/.zattrs?series=2\n"
    )
    assert sentinel not in rendered

    ordinary_error = RuntimeError("ordinary programming error")

    def fail_without_credentials():
        raise ordinary_error

    monkeypatch.setattr(main_module, "app", fail_without_credentials)
    with pytest.raises(RuntimeError) as uncaught:
        main_module.main()
    assert uncaught.value is ordinary_error


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "https://user:secret@example.org/cohort/image.ome.zarr/0?"
            "series=1&token=a%2Fb",
            "https://example.org/cohort/image.ome.zarr/0?series=1",
        ),
        (
            "file:///mounted/cohort/image%202.ome.zarr",
            "file:///mounted/cohort/image%202.ome.zarr",
        ),
    ],
)
def test_store_header_persists_credential_free_source_uri(uri, expected):
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
    assert header["source"]["uri"] == expected


def test_multigrid_receipt_persists_credential_free_source_uri(monkeypatch, tmp_path):
    uri = "s3://user:secret@bucket/cohort/image.ome.zarr/0?version=7&token=S"
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
    assert captured[0].source_uri == (
        "s3://bucket/cohort/image.ome.zarr/0?version=7"
    )
    assert captured[0].slide_id == summary["slide_id"]


def test_failed_receipt_redacts_uri_from_source_and_error(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from conftest import MockEmbedder

    sentinel = "R2F_FAILURE_SECRET"
    uri = f"https://user:{sentinel}@example.org/image.zarr?series=2&token={sentinel}"

    class FailingReader:
        name = "failing"

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            raise FileNotFoundError(f"could not open {self.path}")

        def __exit__(self, *exc):
            return None

    original_get = runner.plugins.get
    monkeypatch.setattr(
        runner.plugins,
        "get",
        lambda component, name: (
            FailingReader if component == "readers" else original_get(component, name)
        ),
    )
    receipts = str(tmp_path / "receipts")
    cfg = RunConfig(models=["mock"], reader="failing", no_seg=True, device="cpu")

    with pytest.raises(FileNotFoundError):
        runner.run_slide(
            uri,
            str(tmp_path / "out"),
            cfg,
            receipts_dir=receipts,
            embedders=[MockEmbedder(name="mock")],
        )

    receipt = read_receipt(receipts, slide_id_from_path(uri))
    assert receipt is not None
    assert receipt["source_uri"] == "https://example.org/image.zarr?series=2"
    assert sentinel not in str(receipt)


def test_raw_authenticated_uri_reaches_reader_but_no_persisted_artifact(
    monkeypatch, tmp_path
):
    pytest.importorskip("torch")
    from conftest import MockEmbedder

    sentinel = "R2F_ARTIFACT_SECRET"
    uri = (
        f"https://user:{sentinel}@example.org/image.zarr?"
        f"series=2&token={sentinel}"
    )
    opened = []

    class AuthenticatedFixtureReader(WSISource):
        name = "authenticated-fixture"

        def open(self):
            opened.append(self.path)
            return self

        def close(self):
            return None

        @property
        def mpp(self):
            return 0.5

        @property
        def level_dimensions(self):
            return [Size(64, 64)]

        def level_downsamples(self):
            return [1.0]

        def read_region(self, region):
            return np.full(
                (region.size.height, region.size.width, 3), 128, dtype=np.uint8
            )

    original_get = runner.plugins.get
    monkeypatch.setattr(
        runner.plugins,
        "get",
        lambda component, name: (
            AuthenticatedFixtureReader
            if component == "readers"
            else original_get(component, name)
        ),
    )
    receipts = tmp_path / "receipts"
    cfg = RunConfig(
        models=["mock"],
        reader="authenticated-fixture",
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    summary = runner.run_slide(
        uri,
        str(tmp_path / "out"),
        cfg,
        receipts_dir=str(receipts),
        cli=f"raw2features embed {uri} out --hf-token={sentinel}",
        embedders=[MockEmbedder(name="mock")],
    )

    assert opened == [uri]
    store = summary["output_uri"].removeprefix("file://")
    artifacts = [path for path in receipts.rglob("*") if path.is_file()]
    artifacts += [
        path for path in tmp_path.joinpath("out").rglob("*") if path.is_file()
    ]
    assert artifacts
    for artifact in artifacts:
        assert sentinel.encode() not in artifact.read_bytes(), artifact
    assert store.endswith(".embeddings.zarr")
