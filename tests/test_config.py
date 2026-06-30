"""Loaders for the declarative job inputs: the extraction plan + the slide manifest."""

from __future__ import annotations

import pytest

from raw2features.core.config import load_extractions, load_manifest

# -- extraction plan -----------------------------------------------------------


def test_load_extractions_normalises_entries(tmp_path):
    p = tmp_path / "plan.yaml"
    p.write_text(
        "extractions:\n"
        "  - {model: uni, mpp: 0.5, patch_px: 224}\n"
        "  - {model: uni, mpp: 1.0}\n"
        "  - {model: conch}\n"  # omitted geometry -> registry defaults downstream
    )
    assert load_extractions(str(p)) == [
        {"model": "uni", "mpp": 0.5, "patch_px": 224},
        {"model": "uni", "mpp": 1.0},
        {"model": "conch"},
    ]


def test_load_extractions_rejects_missing_list(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("models: [uni]\n")
    with pytest.raises(ValueError, match="extractions"):
        load_extractions(str(p))


def test_load_extractions_rejects_entry_without_model(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("extractions:\n  - {mpp: 0.5}\n")
    with pytest.raises(ValueError, match="model"):
        load_extractions(str(p))


# -- slide manifest ------------------------------------------------------------


def test_load_manifest_csv_with_header(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,source_mpp\n/a/x.zarr,0.25\n/b/y.zarr\n")
    assert load_manifest(str(p)) == [
        {"path": "/a/x.zarr", "source_mpp": 0.25},
        {"path": "/b/y.zarr"},
    ]


def test_load_manifest_bare_txt_with_comments_and_blanks(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("# a curated subset\n/a/x.zarr\n\n/b/y.zarr\n")
    assert load_manifest(str(p)) == [{"path": "/a/x.zarr"}, {"path": "/b/y.zarr"}]


def test_load_manifest_positional_source_mpp(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("/a/x.zarr,0.5\n")
    assert load_manifest(str(p)) == [{"path": "/a/x.zarr", "source_mpp": 0.5}]


def test_load_manifest_empty_errors(tmp_path):
    p = tmp_path / "m.txt"
    p.write_text("# nothing but a comment\n")
    with pytest.raises(ValueError, match="empty"):
        load_manifest(str(p))


# -- loader edge cases ---------------------------------------------------------


def test_load_extractions_rejects_non_dict_root(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- {model: uni}\n")  # a list at the root, not a mapping
    with pytest.raises(ValueError, match="mapping"):
        load_extractions(str(p))


def test_load_extractions_accepts_json(tmp_path):
    p = tmp_path / "plan.json"  # JSON is valid YAML; the loader handles it
    p.write_text('{"extractions": [{"model": "uni", "mpp": 0.5}]}')
    assert load_extractions(str(p)) == [{"model": "uni", "mpp": 0.5}]


def test_load_manifest_rejects_non_numeric_source_mpp(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,source_mpp\n/a/x.zarr,notanumber\n")
    with pytest.raises(ValueError):  # float("notanumber") fails loudly
        load_manifest(str(p))
