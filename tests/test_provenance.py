"""Security boundaries for command provenance written into output metadata."""

from __future__ import annotations

import json
import shlex

import pytest

from raw2features.core import provenance


@pytest.mark.parametrize(
    "cli",
    [
        "raw2features embed slide.zarr out --hf-token R2F_SECRET",
        "raw2features embed slide.zarr out --hf-token=R2F_SECRET",
        "raw2features embed slide.zarr out --HF-TOKEN 'R2F_SECRET'",
        "raw2features embed slide.zarr out --token R2F_SECRET --api-key=R2F_SECRET",
    ],
)
def test_sanitize_cli_redacts_secret_option_forms(cli):
    got = provenance.sanitize_cli(cli)
    assert "R2F_SECRET" not in got
    assert "<redacted>" in got
    assert "raw2features embed slide.zarr out" in got


def test_sanitize_cli_redacts_authenticated_uri_and_keeps_semantic_selector():
    cli = (
        "raw2features embed "
        "https://user:password@example.org/image.zarr?series=2&token=R2F_SECRET "
        "out --model uni"
    )
    got = provenance.sanitize_cli(cli)
    assert got == (
        "raw2features embed https://example.org/image.zarr?series=2 "
        "out --model uni"
    )
    assert "password" not in got and "R2F_SECRET" not in got


def test_sanitize_cli_redacts_apostrophes_inside_uri_credentials():
    sentinel = "SEC'RET"
    cli = (
        "raw2features embed "
        f"https://user:{sentinel}@example.org/s.zarr?series=2&token={sentinel} out"
    )
    got = provenance.sanitize_cli(cli)
    assert got == "raw2features embed https://example.org/s.zarr?series=2 out"
    assert sentinel not in got


def test_capture_sanitizes_explicit_cli(monkeypatch):
    monkeypatch.setattr(provenance, "_gpu_info", lambda: None)
    monkeypatch.setattr(provenance, "_git_sha", lambda: None)
    captured = provenance.capture(
        "raw2features embed https://example.org/s.zarr?token=R2F_SECRET out "
        "--hf-token=R2F_SECRET"
    )
    serialized = json.dumps(captured)
    assert "R2F_SECRET" not in serialized
    assert captured["cli"] == (
        "raw2features embed https://example.org/s.zarr out --hf-token=<redacted>"
    )


def test_capture_sanitizes_sys_argv_fallback(monkeypatch):
    monkeypatch.setattr(provenance, "_gpu_info", lambda: None)
    monkeypatch.setattr(provenance, "_git_sha", lambda: None)
    monkeypatch.setattr(
        provenance.sys,
        "argv",
        [
            "raw2features",
            "benchmark",
            "https://user:R2F_SECRET@example.org/s.zarr?generation=2&token=R2F_SECRET",
            "--hf-token",
            "R2F_SECRET",
        ],
    )
    captured = provenance.capture()
    serialized = json.dumps(captured)
    assert "R2F_SECRET" not in serialized
    assert "https://example.org/s.zarr?generation=2" in captured["cli"]
    assert shlex.split(captured["cli"])[-2:] == ["--hf-token", "<redacted>"]


def test_sanitize_argv_redacts_a_whole_secret_argument_before_joining():
    sentinel = "first second' third"
    uri_sentinel = "URI_SEC'RET"
    got = provenance.sanitize_argv(
        [
            "raw2features",
            "embed",
            f"https://user:{uri_sentinel}@example.org/s.zarr?"
            f"series=2&token={uri_sentinel}",
            "output dir",
            "--password",
            sentinel,
            "--api-key=equal value' secret",
            "--model",
            "uni",
        ]
    )

    assert sentinel not in got
    assert uri_sentinel not in got
    assert shlex.split(got) == [
        "raw2features",
        "embed",
        "https://example.org/s.zarr?series=2",
        "output dir",
        "--password",
        "<redacted>",
        "--api-key=<redacted>",
        "--model",
        "uni",
    ]
