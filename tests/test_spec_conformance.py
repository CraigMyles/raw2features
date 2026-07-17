"""Conformance to the embeddings-store spec (docs/SPEC.md / raw2features.spec).

The load-bearing test is `test_real_pipeline_output_conforms`: it validates the actual
runner output, so docs/SPEC.md and the code cannot drift. The rest check the validator
catches the violations the spec calls out.
"""

from __future__ import annotations

import pytest
import zarr

from raw2features.core.store import GRIDS, open_grid
from raw2features.spec import SPEC_VERSION, validate_store


def _model_meta(feat_dim):
    """A fully-populated per-model provenance block (every SPEC-required key)."""
    return {
        "source": "hf-hub:org/model",
        "embedding_dim": feat_dim,
        "input_size": 224,
        "pooling": "cls",
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "interpolation": "bilinear",
        "transform_source_url": "https://example/card",
        "license": "MIT",
        "gated": False,
        "weights_sha256": None,
        "weights_revision": "deadbeef",
        "doi": None,
    }


def _store(
    tmp_path,
    *,
    name="s",
    header="default",
    coords_n=2,
    feat_n=2,
    feat_dim=4,
    model="mock",
    model_in_header=True,
):
    """Build a store by hand so each test can break exactly one rule."""
    path = str(tmp_path / f"{name}.embeddings.zarr")
    key = "mpp0.5_px224"
    root = zarr.open_group(path, mode="w", zarr_format=2)
    if header == "default":
        header = {
            "schema_version": SPEC_VERSION,
            "source": {
                "uri": "file:///x",
                "slide_id": name,
                "mpp_level0": 0.25,
                "ngff_version": "0.4",
                "reader": "omezarr",
                "level_dimensions": [[1024, 1024]],
                "level_downsamples": [1.0],
            },
            "patching": {
                "target_mpp": 0.5,
                "achieved_mpp": 0.5,
                "patch_px": 224,
                "level0_patch": 448,
                "level0_step": 448,
                "read_level": 0,
                "step_out_px": 224,
                "n_patches": coords_n,
                "grid_shape": [1, coords_n],
                "coords_convention": "level0_xy",
            },
            "models": ({model: _model_meta(feat_dim)} if model_in_header else {}),
            "grid_hash": "abc123",
            "provenance": {
                "raw2features_version": "0",
                "created_utc": "1970-01-01T00:00:00Z",
                "cli": "raw2features embed",
                "git_sha": None,
                "host": "h",
                "arch": "x86_64",
                "platform": "test",
                "python": "3.12",
            },
        }
    # Root discovery header + grids index; the authoritative header is per grid.
    root.attrs["raw2features"] = {
        "schema_version": SPEC_VERSION,
        "grids": {
            key: {
                "target_mpp": 0.5,
                "patch_px": 224,
                "n_patches": coords_n,
                "models": [model] if model_in_header else [],
                "grid_hash": "abc123",
            }
        },
    }
    g = root.require_group(GRIDS).require_group(key)
    if header is not None:
        g.attrs["raw2features"] = header
    c = g.create_array("coords", shape=(coords_n, 2), dtype="int32")
    c.attrs["role"] = "coords"
    c.attrs["units"] = "level0_px"
    feats = g.create_group("features")
    a = feats.create_array(model, shape=(feat_n, feat_dim), dtype="float16")
    a.attrs["role"] = "features"
    a.attrs["model"] = model
    return path


def test_valid_store_conforms(tmp_path):
    assert validate_store(_store(tmp_path)) == []


def test_validation_reads_live_metadata_not_stale_consolidated_view(tmp_path):
    path = _store(tmp_path)
    zarr.consolidate_metadata(path)
    root = zarr.open_group(path, mode="r+", use_consolidated=False)
    del root[GRIDS]["mpp0.5_px224"]["coords"]

    violations = validate_store(path)

    assert any("required array 'coords' is missing" in item for item in violations)


def test_missing_header(tmp_path):
    v = validate_store(_store(tmp_path, header=None))
    assert any("raw2features" in x for x in v)


def test_length_mismatch_breaks_1to1(tmp_path):
    v = validate_store(_store(tmp_path, coords_n=2, feat_n=3))
    assert any("1:1 invariant" in x for x in v)


def test_header_patch_count_must_match_coords(tmp_path):
    path = _store(tmp_path, coords_n=2)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    h["patching"]["n_patches"] = 3
    g.attrs["raw2features"] = h

    v = validate_store(path)
    assert any("header.patching.n_patches 3 != coords length 2" in item for item in v)


def test_every_header_model_must_have_a_feature_array(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    h["models"]["missing"] = _model_meta(7)
    g.attrs["raw2features"] = h

    v = validate_store(path)
    assert any(
        "header.models.missing has no features/missing array" in item for item in v
    )


def test_model_without_header_entry(tmp_path):
    v = validate_store(_store(tmp_path, model_in_header=False))
    assert any("no header.models" in x for x in v)


def test_not_a_store(tmp_path):
    assert validate_store(str(tmp_path / "nope.zarr"))  # non-empty == violation(s)


def test_missing_model_provenance_key(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    del h["models"]["mock"]["weights_revision"]  # drop a required pin
    g.attrs["raw2features"] = h
    v = validate_store(path)
    assert any("header.models.mock.weights_revision is missing" in x for x in v)


def test_missing_source_subkey(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    del h["source"]["mpp_level0"]
    g.attrs["raw2features"] = h
    v = validate_store(path)
    assert any("header.source.mpp_level0 is missing" in x for x in v)


def test_missing_coords_role_and_units(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    g["coords"].attrs.clear()
    v = validate_store(path)
    assert any("role='coords'" in x for x in v)
    assert any("units='level0_px'" in x for x in v)


def test_non_float_features_rejected(tmp_path):
    path = str(tmp_path / "i.embeddings.zarr")
    _store(tmp_path, name="i")
    g = open_grid(path, mode="r+")
    del g["features"]["mock"]
    a = g["features"].create_array("mock", shape=(2, 4), dtype="int16")
    a.attrs["role"] = "features"
    v = validate_store(path)
    assert any("must be a float dtype" in x for x in v)


def test_qc_layer_conforms(tmp_path):
    # An optional qc/<tool>/ layer 1:1 with coords + tagged role='qc' conforms.
    path = _store(tmp_path, coords_n=2)
    qc = open_grid(path, mode="r+").require_group("qc").require_group("grandqc")
    a = qc.create_array("scores", shape=(2, 3), dtype="float16")
    a.attrs["role"] = "qc"
    a.attrs["classes"] = ["clean_tissue", "out_of_focus", "pen_marking"]
    assert validate_store(path) == []


def test_qc_wrong_length_breaks_1to1(tmp_path):
    path = _store(tmp_path, coords_n=2)
    qc = open_grid(path, mode="r+").require_group("qc").require_group("grandqc")
    a = qc.create_array("scores", shape=(3, 3), dtype="float16")  # != coords length 2
    a.attrs["role"] = "qc"
    v = validate_store(path)
    assert any("qc/grandqc/scores" in x and "1:1 invariant" in x for x in v)


def test_qc_missing_role_is_reported(tmp_path):
    path = _store(tmp_path, coords_n=2)
    qc = open_grid(path, mode="r+").require_group("qc").require_group("grandqc")
    qc.create_array("scores", shape=(2, 3), dtype="float16")  # role attr omitted
    assert any(
        "qc/grandqc/scores" in x and "role='qc'" in x for x in validate_store(path)
    )


def test_packaged_schema_is_valid_jsonschema():
    import jsonschema

    from raw2features.spec import _load_header_schema

    schema = _load_header_schema(SPEC_VERSION)
    assert schema is not None  # the schema for this build's version is packaged
    jsonschema.Draft202012Validator.check_schema(schema)  # raises if malformed


def test_unknown_header_keys_are_tolerated(tmp_path):
    # additionalProperties: true -> a store written by a newer version (extra keys)
    # still validates against this schema. Forward-compatibility by construction.
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    h["future_top_level_key"] = {"anything": [1, 2, 3]}
    h["source"]["future_subkey"] = "ignored"
    g.attrs["raw2features"] = h
    assert validate_store(path) == []


def test_unknown_schema_version_is_reported(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    h["schema_version"] = "9.9"  # no packaged schema for this
    g.attrs["raw2features"] = h
    assert any("no packaged JSON Schema" in x for x in validate_store(path))


def test_bad_coords_convention_is_reported(tmp_path):
    path = _store(tmp_path)
    g = open_grid(path, mode="r+")
    h = dict(g.attrs)["raw2features"]
    h["patching"]["coords_convention"] = "yx"  # schema pins const "level0_xy"
    g.attrs["raw2features"] = h
    v = validate_store(path)
    assert any("coords_convention" in x and "level0_xy" in x for x in v)


def test_built_header_aligns_with_schema(synthetic_ngff):
    """The header the runner actually builds must validate against the packaged schema.

    Closes the schema<->code drift gap in the cheap (no-torch) CI lane: if the schema
    grows a required field the builder doesn't emit (or vice versa), this fails here,
    not only in the torch-gated real-pipeline test below. Uses the real `_build_header`
    with a real reader + grid + a mock embedder (no weights)."""
    import jsonschema

    from conftest import MockEmbedder
    from raw2features.core import provenance
    from raw2features.patcher.grid import GridPatcher
    from raw2features.pipeline.runner import _build_header
    from raw2features.readers.omezarr import OmeZarrReader
    from raw2features.spec import _load_header_schema

    with OmeZarrReader(synthetic_ngff) as r:
        grid = GridPatcher(target_mpp=0.5, patch_px=64).build_grid(r)
        header = _build_header(
            r,
            grid,
            {"segmenter": "none"},
            [MockEmbedder().load()],
            "s",
            0,
            None,
            "gridhash",
            provenance.capture("test"),
        )
    schema = _load_header_schema(SPEC_VERSION)
    jsonschema.Draft202012Validator(schema).validate(header)  # raises on any mismatch


def test_real_pipeline_output_conforms(synthetic_ngff, tmp_path):
    """The actual runner output MUST conform - this is the spec<->code guarantee."""
    pytest.importorskip("torch")
    from conftest import MockEmbedder
    from raw2features.pipeline.runner import RunConfig, run_slide

    cfg = RunConfig(
        models=["mock"],
        segmenter="otsu",
        target_mpp=0.5,
        patch_px=64,
        tissue_threshold=0.0,
        device="cpu",
        amp="fp32",
        batch_size=8,
    )
    summary = run_slide(
        synthetic_ngff, str(tmp_path / "out"), cfg, embedders=[MockEmbedder()]
    )
    store = summary["output_uri"].removeprefix("file://")
    assert validate_store(store) == [], validate_store(store)
