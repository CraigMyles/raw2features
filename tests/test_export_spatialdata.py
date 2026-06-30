"""SpatialData exporter: embeddings.zarr -> SpatialData .zarr (schema + round-trip).

The heavy scverse stack (``spatialdata``/``anndata``/``geopandas``) is optional, so the
write/read assertions are guarded by ``importorskip``. The pure-validation path (model
selection, missing-store errors) is exercised without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.export.spatialdata import export_spatialdata
from raw2features.sinks.zarr_sink import ZarrSink

_LEVEL0_PATCH = 224
_COORDS = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.int32)
_GRID_INDEX = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
# Realistic (n_rows, n_cols) grid tissue-fraction map (matches what the patcher writes).
# The three kept patches sit at grid cells (0,0), (0,1), (1,0); (1,1) is unkept.
_GRID_TISSUE = np.array([[1.0, 0.5], [0.8, 0.0]], dtype=np.float32)
_TISSUE = np.array([1.0, 0.5, 0.8], dtype=np.float32)  # expected per-patch values
_FEATS = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)


def _make_store(tmp_path, slide_id="slideA", *, source=None, panel=None):
    """Write a minimal but valid embedding store and return its path.

    ``source`` overrides the source header block - pass one without ``scale_um`` /
    ``mpp_level0`` to model an uncalibrated source (no µm coordinate system). ``panel``
    sets the multiplex marker-panel block (per model), to exercise the export's
    marker-panel surfacing.
    """
    header = {
        "schema_version": 1,
        "patching": {
            "level0_patch": _LEVEL0_PATCH,
            "patch_px": _LEVEL0_PATCH,
            "achieved_mpp": 0.5,
            "target_mpp": 0.5,
            "read_level": 0,
            "coords_convention": "level0_xy",
            "n_patches": 3,
        },
        "source": source or {
            "uri": "file:///x", "slide_id": slide_id, "mpp_level0": 0.25,
            "scale_um": {"x": 0.25, "y": 0.25}, "level0_translation_um": None,
        },
        "models": {"mock": {"embedding_dim": 4, "license": "MIT"}},
    }
    if panel is not None:
        header["panel"] = panel
    sink = ZarrSink()
    sink.create(
        str(tmp_path),
        slide_id,
        grid="mpp0.5_px224",
        n_patches=3,
        coords=_COORDS,
        grid_index=_GRID_INDEX,
        grid_tissue=_GRID_TISSUE,
        model_dims={"mock": 4},
        header=header,
    )
    sink.write_block("mock", 0, _FEATS.astype(np.float16))
    sink.close()
    return str(tmp_path / f"{slide_id}.embeddings.zarr")


def test_missing_model_raises_without_spatialdata(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="not in store"):
        export_spatialdata(store, str(tmp_path / "out.zarr"), models=["nope"])


def test_bad_geometry_raises(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="geometry must be"):
        export_spatialdata(store, str(tmp_path / "out.zarr"), geometry="triangle")


def test_export_roundtrip_polygon(tmp_path):
    spatialdata = pytest.importorskip("spatialdata")
    pytest.importorskip("anndata")
    pytest.importorskip("geopandas")

    store = _make_store(tmp_path)
    out = str(tmp_path / "slideA.spatialdata.zarr")
    written = export_spatialdata(store, out, overwrite=True)
    assert written == out

    sdata = spatialdata.read_zarr(out)
    assert "tiles" in sdata.shapes
    assert "table" in sdata.tables

    table = sdata.tables["table"]
    assert table.n_obs == 3
    # Embeddings live in obsm["X_<model>"], float32, row-aligned to tiles.
    assert "X_mock" in table.obsm
    np.testing.assert_allclose(table.obsm["X_mock"], _FEATS, rtol=0, atol=1e-3)
    # squidpy/HEST-compatible tile centres.
    centers = _COORDS.astype(np.float32) + _LEVEL0_PATCH / 2.0
    np.testing.assert_allclose(table.obsm["spatial"], centers, atol=1e-3)
    # obs carries CLAM/Trident geometry fields + tissue fraction.
    for col in ("x", "y", "array_row", "array_col", "tissue_frac"):
        assert col in table.obs.columns
    np.testing.assert_allclose(
        np.sort(table.obs["tissue_frac"].to_numpy()), np.sort(_TISSUE), atol=2e-2
    )
    # provenance survived into uns.
    assert table.uns["raw2features"]["patching"]["level0_patch"] == _LEVEL0_PATCH
    assert table.uns["raw2features_export"]["models"] == ["mock"]

    # square tiles in level-0 pixel space: first tile bounds == (0,0,224,224).
    tiles = sdata.shapes["tiles"]
    assert len(tiles) == 3
    bounds = tiles.geometry.iloc[0].bounds
    np.testing.assert_allclose(bounds, (0, 0, _LEVEL0_PATCH, _LEVEL0_PATCH))


@pytest.mark.parametrize("geometry", ["polygon", "circle"])
def test_official_inverse_roundtrip(tmp_path, geometry):
    """spatialdata-io's `to_legacy_anndata` (the canonical inverse) must read our store
    and reconstruct embeddings + spatial coords - i.e. we conform to the spec the
    ecosystem enforces, for both tile geometries."""
    pytest.importorskip("spatialdata")
    pytest.importorskip("spatialdata_io")
    import spatialdata
    from spatialdata_io.experimental import to_legacy_anndata

    store = _make_store(tmp_path)
    out = str(tmp_path / f"{geometry}.spatialdata.zarr")
    export_spatialdata(store, out, geometry=geometry, overwrite=True)

    # A calibrated export carries two coordinate systems (pixel `global` + physical
    # `micrometers`); the legacy converter needs one named - `global` is the pixel space
    # the reconstructed `obsm["spatial"]` coords live in.
    adata = to_legacy_anndata(
        spatialdata.read_zarr(out), table_name="table", include_images=False,
        coordinate_system="global",
    )
    np.testing.assert_allclose(adata.obsm["X_mock"], _FEATS, atol=1e-3)
    centers = _COORDS.astype(np.float32) + _LEVEL0_PATCH / 2.0
    np.testing.assert_allclose(adata.obsm["spatial"], centers, atol=1e-3)


def test_export_publishes_micrometers_coordinate_system(tmp_path):
    """A calibrated source must yield a physical µm coordinate system (not just pixels),
    so tiles can be aligned in real units - the point of the SpatialData export."""
    sd = pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")
    from spatialdata.transformations import Scale, get_transformation

    store = _make_store(tmp_path)  # calibrated at 0.25 µm/px
    out = str(tmp_path / "cs.spatialdata.zarr")
    export_spatialdata(store, out, overwrite=True)
    sdata = sd.read_zarr(out)

    systems = get_transformation(sdata.shapes["tiles"], get_all=True)
    assert set(systems) == {"global", "micrometers"}
    um = systems["micrometers"]
    assert isinstance(um, Scale)
    np.testing.assert_allclose(um.to_affine_matrix(("x", "y"), ("x", "y")),
                               [[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]])
    mpp = sdata.tables["table"].uns["raw2features_export"]["micrometers_per_pixel"]
    np.testing.assert_allclose(np.asarray(mpp), [0.25, 0.25])


def test_export_uncalibrated_source_has_no_micrometers_system(tmp_path):
    sd = pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")
    from spatialdata.transformations import get_transformation

    # Source without scale_um / mpp_level0 -> only the pixel (global) system.
    store = _make_store(tmp_path, source={"uri": "file:///x", "slide_id": "slideA"})
    out = str(tmp_path / "uncal.spatialdata.zarr")
    export_spatialdata(store, out, overwrite=True)
    sdata = sd.read_zarr(out)
    assert set(get_transformation(sdata.shapes["tiles"], get_all=True)) == {"global"}
    assert (sdata.tables["table"].uns["raw2features_export"]["micrometers_per_pixel"]
            is None)


def test_export_anisotropic_with_origin_translation(tmp_path):
    """Anisotropic per-axis scale + a non-zero source origin compose into the µm map."""
    sd = pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")
    from spatialdata.transformations import Sequence, get_transformation

    store = _make_store(tmp_path, source={
        "uri": "file:///x", "slide_id": "slideA", "mpp_level0": 0.3,
        "scale_um": {"x": 0.25, "y": 0.35},
        "level0_translation_um": {"x": 10.0, "y": 20.0},
    })
    out = str(tmp_path / "aniso.spatialdata.zarr")
    export_spatialdata(store, out, overwrite=True)
    sdata = sd.read_zarr(out)
    um = get_transformation(sdata.shapes["tiles"], get_all=True)["micrometers"]
    assert isinstance(um, Sequence)  # Scale then Translation
    # pixel (x, y) -> (0.25x + 10, 0.35y + 20) µm
    affine = um.to_affine_matrix(("x", "y"), ("x", "y"))
    np.testing.assert_allclose(affine, [[0.25, 0, 10.0], [0, 0.35, 20.0], [0, 0, 1]])


# A realistic multiplex panel block (as KRONOS's set_panel records it): a per-channel
# `mapping` list-of-dicts plus coverage lists. The mapping is exactly what anndata's
# zarr writer mangles into stringified blobs if exported raw.
_PANEL = {
    "mock": {
        "n_markers": 2,
        "kept": ["DAPI", "pancytokeratin"],
        "dropped": ["blank", "cd62l"],
        "unmatched": ["cd62l"],
        "mapping": [
            {"channel": "DAPI", "channel_index": 0,
             "kronos_marker": "DAPI", "marker_id": 4},
            {"channel": "pancytokeratin", "channel_index": 1,
             "kronos_marker": "Cytokeratin", "marker_id": 322},
        ],
        "vocabulary": "MahmoodLab/kronos marker_metadata.csv@abc123",
    }
}


def test_export_surfaces_multiplex_marker_panel(tmp_path):
    """A multiplex run's marker map must survive as a first-class, queryable table - not
    the stringified ``"{...}"`` blobs anndata makes of a raw list-of-dicts."""
    spatialdata = pytest.importorskip("spatialdata")
    pytest.importorskip("anndata")
    pytest.importorskip("geopandas")
    import pandas as pd

    store = _make_store(tmp_path, panel=_PANEL)
    out = str(tmp_path / "mx.spatialdata.zarr")
    export_spatialdata(store, out, overwrite=True)
    uns = spatialdata.read_zarr(out).tables["table"].uns

    # 1) queryable tidy table: one row per kept channel, real (non-string) id dtype.
    df = pd.DataFrame(uns["raw2features_panel"])
    assert list(df.columns) == [
        "model", "channel", "channel_index", "marker", "marker_id",
    ]
    assert len(df) == 2
    row = df[df["marker"] == "Cytokeratin"].iloc[0]  # the synonym-resolved channel
    assert row["channel"] == "pancytokeratin"
    assert int(row["marker_id"]) == 322  # an int, not "322" - the mapping is usable
    assert df["channel_index"].dtype.kind == "i"

    # 2) compact coverage summary carries the lists, never the list-of-dicts mapping.
    summ = uns["raw2features_export"]["panel"]["mock"]
    assert int(summ["n_markers"]) == 2
    assert list(summ["unmatched"]) == ["cd62l"]
    assert "mapping" not in summ
    assert "marker_metadata.csv" in summ["vocabulary"]

    # 3) faithful header keeps the mapping, columnarised (arrays, not "{...}" strings).
    hdr_map = uns["raw2features"]["panel"]["mock"]["mapping"]
    assert isinstance(hdr_map, dict)  # dict-of-lists, not a list of stringified dicts
    assert list(hdr_map["marker_id"]) == [4, 322]
    assert list(hdr_map["channel"]) == ["DAPI", "pancytokeratin"]


def test_export_brightfield_has_no_panel(tmp_path):
    """No marker panel on a brightfield store -> no panel artefacts in the export."""
    pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")
    import spatialdata

    store = _make_store(tmp_path)  # no panel
    out = str(tmp_path / "bf.spatialdata.zarr")
    export_spatialdata(store, out, overwrite=True)
    uns = spatialdata.read_zarr(out).tables["table"].uns
    assert "raw2features_panel" not in uns
    assert "panel" not in uns["raw2features_export"]


def test_export_circle_geometry(tmp_path):
    pytest.importorskip("spatialdata")
    pytest.importorskip("anndata")

    store = _make_store(tmp_path)
    out = str(tmp_path / "circ.spatialdata.zarr")
    export_spatialdata(store, out, geometry="circle", overwrite=True)

    import spatialdata

    tiles = spatialdata.read_zarr(out).shapes["tiles"]
    # circle radius == half the patch side.
    assert "radius" in tiles.columns
    np.testing.assert_allclose(tiles["radius"].to_numpy(), _LEVEL0_PATCH / 2.0)
