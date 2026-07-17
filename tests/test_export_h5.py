"""HDF5 export: TRIDENT (features + level-0-px coords) and STAMP (feats + µm coords).

A non-default egress convenience; the native zarr stays primary. h5py is light and a
test/runtime dep here, so these run unguarded.
"""

from __future__ import annotations

import numpy as np
import pytest

from raw2features.export.h5 import export_h5
from raw2features.sinks.zarr_sink import ZarrSink

_LEVEL0_PATCH = 224
_PATCH_PX = 224
_ACHIEVED_MPP = 0.5
_MPP_LEVEL0 = 0.25
_COORDS = np.array([[0, 0], [224, 0], [0, 224]], dtype=np.int32)
_FEATS = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)


def _make_store(tmp_path, slide_id="slideA", models=("mock",)):
    header = {
        "schema_version": 1,
        "patching": {
            "level0_patch": _LEVEL0_PATCH,
            "patch_px": _PATCH_PX,
            "achieved_mpp": _ACHIEVED_MPP,
            "target_mpp": _ACHIEVED_MPP,
            "read_level": 0,
            "coords_convention": "level0_xy",
            "n_patches": 3,
        },
        "source": {"uri": "file:///x", "slide_id": slide_id, "mpp_level0": _MPP_LEVEL0},
        "models": {m: {"embedding_dim": 4} for m in models},
    }
    sink = ZarrSink()
    sink.create(
        str(tmp_path),
        slide_id,
        grid="mpp0.5_px224",
        n_patches=3,
        coords=_COORDS,
        grid_index=_COORDS,
        grid_tissue=np.ones(3, dtype=np.float32),
        model_dims={m: 4 for m in models},
        header=header,
    )
    for m in models:
        sink.write_block(m, 0, _FEATS.astype(np.float16))
    sink.close()
    return str(tmp_path / f"{slide_id}.embeddings.zarr")


def test_trident_layout(tmp_path):
    import h5py

    store = _make_store(tmp_path)
    (path,) = export_h5(store, str(tmp_path / "out"), layout="trident")
    assert path.endswith("slideA.h5")
    with h5py.File(path, "r") as fh:
        assert fh["features"].dtype == np.float32
        np.testing.assert_allclose(fh["features"][:], _FEATS)
        assert fh["coords"].dtype == np.int64
        np.testing.assert_array_equal(fh["coords"][:], _COORDS)  # level-0 px
        assert fh["coords"].attrs["patch_size_level0"] == _LEVEL0_PATCH
        assert fh["coords"].attrs["target_magnification"] == 20.0  # 10 / 0.5
        assert fh["features"].attrs["encoder"] == "mock"


def test_clam_layout_is_trident_with_int32_coords(tmp_path):
    import h5py

    store = _make_store(tmp_path)
    (path,) = export_h5(store, str(tmp_path / "out"), layout="clam")
    with h5py.File(path, "r") as fh:
        # Same datasets as trident, but coords as int32 (CLAM's feature .h5 dtype).
        assert fh["features"].dtype == np.float32
        np.testing.assert_allclose(fh["features"][:], _FEATS)
        assert fh["coords"].dtype == np.int32
        np.testing.assert_array_equal(fh["coords"][:], _COORDS)  # level-0 px
        assert fh["coords"].attrs["patch_size_level0"] == _LEVEL0_PATCH


def test_stamp_layout(tmp_path):
    import h5py

    store = _make_store(tmp_path)
    (path,) = export_h5(store, str(tmp_path / "out"), layout="stamp")
    with h5py.File(path, "r") as fh:
        assert fh["feats"].dtype == np.float16  # STAMP stores fp16
        np.testing.assert_allclose(fh["feats"][:], _FEATS.astype(np.float16))
        # coords in microns = level-0 px * mpp_level0
        np.testing.assert_allclose(fh["coords"][:], _COORDS * _MPP_LEVEL0, atol=1e-3)
        assert fh.attrs["unit"] == "um"
        # physical tile size = patch_px * achieved_mpp = 112 µm
        assert fh.attrs["tile_size_um"] == pytest.approx(_PATCH_PX * _ACHIEVED_MPP)
        assert fh.attrs["tile_size_px"] == _PATCH_PX
        assert fh.attrs["extractor"] == "mock"


def test_stamp_honours_per_axis_scale_but_remains_slide_relative(tmp_path):
    import h5py
    import zarr

    from raw2features.core.store import open_grid

    store = _make_store(tmp_path)
    g = open_grid(store, mode="r+")
    header = dict(g.attrs["raw2features"])
    header["source"]["mpp_level0"] = 0.3  # scalar mean must not mask anisotropy
    header["source"]["scale_um"] = {"x": 0.2, "y": 0.4}
    header["source"]["level0_translation_um"] = {"x": 10.0, "y": 20.0}
    g.attrs["raw2features"] = header
    zarr.consolidate_metadata(store)

    (path,) = export_h5(store, str(tmp_path / "out"), layout="stamp")
    with h5py.File(path, "r") as fh:
        # STAMP defines coords from the scan's top-left, so the NGFF physical/stage
        # origin must not be added even though it remains in native store metadata.
        expected = _COORDS.astype(np.float32) * np.array([0.2, 0.4])
        np.testing.assert_allclose(fh["coords"][:], expected, atol=1e-5)


def test_multi_model_writes_one_file_each(tmp_path):
    store = _make_store(tmp_path, models=("uni", "resnet50"))
    paths = export_h5(store, str(tmp_path / "out"), layout="trident")
    assert {p.split("/")[-1] for p in paths} == {"slideA.uni.h5", "slideA.resnet50.h5"}


def test_errors(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(ValueError, match="layout must be"):
        export_h5(store, str(tmp_path / "o"), layout="nope")
    with pytest.raises(ValueError, match="not in store"):
        export_h5(store, str(tmp_path / "o"), models=["absent"])


def _hdr(slide_id, mpp, px, l0):
    return {
        "schema_version": "0.1",
        "patching": {
            "achieved_mpp": mpp,
            "target_mpp": mpp,
            "patch_px": px,
            "level0_patch": l0,
            "read_level": 0,
            "coords_convention": "level0_xy",
            "n_patches": 3,
        },
        "source": {"uri": "file:///x", "slide_id": slide_id, "mpp_level0": _MPP_LEVEL0},
        "models": {"mock": {"embedding_dim": 4}},
    }


def test_grid_selector_required_on_multigrid(tmp_path):
    pytest.importorskip("h5py")
    from raw2features.sinks.zarr_sink import ZarrSink

    slide = "s"
    # Two grids in one store: gridA (0.5/224), gridB (1.0/256), both holding 'mock'.
    a = ZarrSink()
    a.create(
        str(tmp_path),
        slide,
        grid="mpp0.5_px224",
        fresh=True,
        n_patches=3,
        coords=_COORDS,
        grid_index=_COORDS,
        grid_tissue=np.ones(3, np.float32),
        model_dims={"mock": 4},
        header=_hdr(slide, 0.5, 224, 448),
    )
    a.write_block("mock", 0, _FEATS.astype(np.float16))
    a.close()
    b = ZarrSink()
    b.create(
        str(tmp_path),
        slide,
        grid="mpp1_px256",
        fresh=False,
        n_patches=3,
        coords=_COORDS,
        grid_index=_COORDS,
        grid_tissue=np.ones(3, np.float32),
        model_dims={"mock": 4},
        header=_hdr(slide, 1.0, 256, 256),
    )
    b.write_block("mock", 0, _FEATS.astype(np.float16))
    b.close()
    store = str(tmp_path / f"{slide}.embeddings.zarr")

    # No --grid on a multi-grid store: a clear error listing the grids.
    with pytest.raises(ValueError, match="grids"):
        export_h5(store, str(tmp_path / "o"))
    # --grid selects one geometry's patches.
    import h5py

    paths = export_h5(store, str(tmp_path / "o"), grid="mpp1_px256")
    assert len(paths) == 1
    with h5py.File(paths[0], "r") as fh:
        assert fh["coords"].attrs["patch_size_level0"] == 256  # gridB's level0_patch
