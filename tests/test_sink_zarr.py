"""Tests for the embeddings-zarr sink."""

from __future__ import annotations

import os
import stat

import numpy as np
import zarr

from raw2features.sinks.zarr_sink import ZarrSink, write_patches_geojson


def test_sink_writes_full_structure(tmp_path):
    n = 5
    coords = np.arange(n * 2).reshape(n, 2).astype("int32")
    grid_index = np.zeros((n, 2), "int32")
    grid_tissue = np.ones((2, 3), "float32")
    sink = ZarrSink()
    sink.create(
        str(tmp_path),
        "slideX",
        grid="mpp1_px224",
        n_patches=n,
        coords=coords,
        grid_index=grid_index,
        grid_tissue=grid_tissue,
        model_dims={"m": 4},
        header={"schema_version": "0.1"},
        features_dtype="float16",
    )
    sink.write_block("m", 0, np.ones((n, 4), "float32"))
    sink.close()

    from raw2features.core.store import open_grid

    store = str(tmp_path / "slideX.embeddings.zarr")
    # Uniform nesting: the patch set lives under grids/<key>/; root carries the index.
    root = zarr.open_group(store, mode="r")
    assert "grids" in root and "mpp1_px224" in root["grids"]
    assert dict(root.attrs)["raw2features"]["grids"]["mpp1_px224"]["models"] == ["m"]

    g = open_grid(store)  # the sole grid
    assert g["coords"].shape == (n, 2)
    assert np.array_equal(np.asarray(g["coords"]), coords)
    assert g["grid_index"].shape == (n, 2)
    assert g["mask"].shape == (2, 3)
    feats = g["features"]["m"]
    assert feats.shape == (n, 4)
    assert feats.dtype == np.float16
    assert np.allclose(np.asarray(feats), 1.0)
    assert dict(g.attrs)["raw2features"]["schema_version"] == "0.1"
    assert g["features"]["m"].attrs["model"] == "m"


def test_sink_write_qc_layer(tmp_path):
    from raw2features.core.store import open_grid

    n = 5
    coords = np.arange(n * 2).reshape(n, 2).astype("int32")
    sink = ZarrSink()
    sink.create(
        str(tmp_path), "qcX", grid="mpp1_px224", n_patches=n,
        coords=coords, grid_index=np.zeros((n, 2), "int32"), grid_tissue=None,
        model_dims={"m": 4}, header={"schema_version": "0.1"},
    )
    sink.write_block("m", 0, np.ones((n, 4), "float32"))
    classes = ["clean_tissue", "out_of_focus", "pen_marking"]
    sink.write_qc(
        "grandqc",
        np.tile([0.7, 0.2, 0.1], (n, 1)),
        classes,
        label=np.zeros(n, "uint8"),
        usable=np.ones(n, "uint8"),
        legend={"properties": [{"label-value": 1, "name": "clean_tissue"}]},
        provenance={"tool": "grandqc", "model_mpp": 1.5},
    )
    sink.close()

    qc = open_grid(str(tmp_path / "qcX.embeddings.zarr"))["qc"]["grandqc"]
    assert qc["scores"].shape == (n, 3)  # packed (N, k), 1:1 with coords
    assert list(qc["scores"].attrs["classes"]) == classes
    assert qc["scores"].attrs["role"] == "qc"
    assert qc["label"].shape == (n,) and qc["label"].attrs["role"] == "qc"
    assert qc["usable"].attrs["role"] == "qc"
    assert dict(qc.attrs)["image-label"]["properties"][0]["name"] == "clean_tissue"
    g = open_grid(str(tmp_path / "qcX.embeddings.zarr"))
    assert dict(g.attrs)["raw2features"]["qc"]["grandqc"]["model_mpp"] == 1.5


def test_geojson_polygons(tmp_path):
    coords = np.array([[0, 0], [100, 50]], dtype="int32")
    path = write_patches_geojson(str(tmp_path), "slideX", coords, level0_patch=64)
    import json

    fc = json.loads(open(path).read())
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 2
    poly = fc["features"][0]["geometry"]["coordinates"][0]
    assert poly[0] == [0, 0] and poly[2] == [64, 64]


def test_geojson_new_file_honours_umask(tmp_path):
    old_umask = os.umask(0o022)
    try:
        path = write_patches_geojson(
            str(tmp_path), "new", np.empty((0, 2), dtype="int32"), level0_patch=64
        )
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o644


def test_geojson_replacement_preserves_mode(tmp_path):
    path = tmp_path / "existing.patches.geojson"
    path.write_text("old")
    path.chmod(0o664)
    replaced = write_patches_geojson(
        str(tmp_path),
        "unused",
        np.empty((0, 2), dtype="int32"),
        level0_patch=64,
        filename=path.name,
    )
    assert replaced == str(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_geojson_accepts_near_name_max_destination(tmp_path):
    suffix = ".geojson"
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    filename = "g" * (name_max - len(suffix) - 5) + suffix
    path = write_patches_geojson(
        str(tmp_path),
        "unused",
        np.empty((0, 2), dtype="int32"),
        level0_patch=64,
        filename=filename,
    )
    assert os.path.isfile(path)
