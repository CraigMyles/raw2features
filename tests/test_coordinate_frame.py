"""Coordinate-frame self-description: axes, units, per-axis scale, translation.

Covers the optional ``source.*`` values the store records so a consumer can re-express
level-0-pixel ``coords`` in the source's physical frame (docs/SPEC.md), plus
the two fail-loud warnings (anisotropy, brightfield-on-multichannel). All offline / no
torch - these exercise the reader + a runner guard directly.
"""

from __future__ import annotations

import warnings

import pytest
import zarr

from conftest import build_ngff_v04
from raw2features.patcher.grid import GridPatcher
from raw2features.pipeline.runner import _warn_channel_collapse
from raw2features.readers.omezarr import OmeZarrReader


def _strip_axis_units(store: str) -> None:
    """Remove the x/y axis units, making the source uncalibrated (pixel units)."""
    g = zarr.open_group(store, mode="r+")
    attrs = dict(g.attrs)
    for ax in attrs["multiscales"][0]["axes"]:
        ax.pop("unit", None)
    g.attrs.update(attrs)


def _patch_level0(store: str, *, scale_xy=None, translation_xy=None) -> None:
    """Rewrite the level-0 dataset's scale and/or add a translation (x, y order)."""
    g = zarr.open_group(store, mode="r+")
    attrs = dict(g.attrs)
    ms = attrs["multiscales"][0]
    cts = ms["datasets"][0]["coordinateTransformations"]
    if scale_xy is not None:
        for t in cts:
            if t["type"] == "scale":
                s = list(t["scale"])
                s[-2], s[-1] = scale_xy[1], scale_xy[0]  # axes end ...y, x
                t["scale"] = s
    if translation_xy is not None:
        n = len(cts[0]["scale"])
        cts.append(
            {"type": "translation",
             "translation": [0.0] * (n - 2) + [translation_xy[1], translation_xy[0]]}
        )
    attrs["multiscales"][0] = ms
    g.attrs.update(attrs)


def test_reader_records_axes_units_and_per_axis_scale(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        assert r.axes == ("t", "c", "z", "y", "x")
        assert r.axis_units["x"] == "micrometer"
        assert r.axis_units["y"] == "micrometer"
        assert r.scale_um == {"x": 0.5, "y": 0.5}
        # The common case carries no translation.
        assert r.level0_translation_um is None


def test_reader_reads_level0_translation(tmp_path):
    store = build_ngff_v04(str(tmp_path / "translated.zarr"))
    _patch_level0(store, translation_xy=(20.0, 10.0))
    with OmeZarrReader(store) as r:
        assert r.level0_translation_um == {"x": 20.0, "y": 10.0}


def test_anisotropic_source_warns_and_keeps_per_axis_scale(tmp_path):
    store = build_ngff_v04(str(tmp_path / "aniso.zarr"))
    _patch_level0(store, scale_xy=(0.75, 0.5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with OmeZarrReader(store) as r:
            assert r.scale_um == {"x": 0.75, "y": 0.5}
            assert r.mpp == 0.625  # mpp_level0 is the x/y mean, for back-compat
    assert any("anisotropic" in str(w.message) for w in caught)


def _set_level(store: str, level: int, *, scale_xy, translation_xy=(0.0, 0.0)) -> None:
    """Set the (x, y) scale and translation of a specific pyramid level."""
    g = zarr.open_group(store, mode="r+")
    attrs = dict(g.attrs)
    ms = attrs["multiscales"][0]
    cts = ms["datasets"][level]["coordinateTransformations"]
    for t in cts:
        if t["type"] == "scale":
            s = list(t["scale"])
            s[-2], s[-1] = scale_xy[1], scale_xy[0]
            t["scale"] = s
    n = len(cts[0]["scale"])
    cts.append(
        {"type": "translation",
         "translation": [0.0] * (n - 2) + [translation_xy[1], translation_xy[0]]}
    )
    attrs["multiscales"][0] = ms
    g.attrs.update(attrs)


def test_read_level_mapping_common_case_is_isotropic_no_offset(synthetic_ngff):
    # No anisotropy, no translation -> exactly (ds, ds, 0, 0), so reads are unchanged.
    with OmeZarrReader(synthetic_ngff) as r:
        for level, ds in enumerate(r.level_downsamples()):
            assert r.read_level_mapping(level) == (ds, ds, 0.0, 0.0)


def test_read_level_mapping_honours_anisotropy_and_translation(tmp_path):
    store = build_ngff_v04(str(tmp_path / "map.zarr"))  # level-0 scale 0.5, no offset
    _set_level(store, 1, scale_xy=(1.2, 1.0), translation_xy=(0.5, 0.3))  # x, y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # anisotropy warning expected
        with OmeZarrReader(store) as r:
            dsx, dsy, ox, oy = r.read_level_mapping(1)
    sx0 = sy0 = 0.5
    assert abs(dsx - 1.2 / sx0) < 1e-9
    assert abs(dsy - 1.0 / sy0) < 1e-9
    assert abs(ox - (0.0 - 0.5) / 1.2) < 1e-9  # (trans0 - transL) / scaleL
    assert abs(oy - (0.0 - 0.3) / 1.0) < 1e-9
    # Level 0 itself is always the identity reference.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with OmeZarrReader(store) as r:
            assert r.read_level_mapping(0) == (1.0, 1.0, 0.0, 0.0)


def test_uncalibrated_source_has_no_mpp(tmp_path):
    # No declared axis unit -> physical pixel size unknown -> mpp is None (not a silent
    # assumption of µm), and scale_um is empty.
    store = build_ngff_v04(str(tmp_path / "uncal.zarr"))
    _strip_axis_units(store)
    with OmeZarrReader(store) as r:
        assert r.mpp is None
        assert r.scale_um == {}
        assert r.axis_units["x"] is None


def test_build_grid_fails_loud_on_uncalibrated_source(tmp_path):
    store = build_ngff_v04(str(tmp_path / "uncal2.zarr"))
    _strip_axis_units(store)
    with OmeZarrReader(store) as r:
        with pytest.raises(ValueError, match="--source-mpp"):
            GridPatcher(target_mpp=0.5, patch_px=64).build_grid(r)


def test_apply_source_mpp_lets_an_uncalibrated_source_proceed(tmp_path):
    store = build_ngff_v04(str(tmp_path / "uncal3.zarr"))
    _strip_axis_units(store)
    with OmeZarrReader(store) as r:
        r.apply_source_mpp(0.25)
        assert r.mpp == 0.25
        assert r.scale_um == {"x": 0.25, "y": 0.25}
        # With the override in place, build_grid no longer raises.
        grid = GridPatcher(target_mpp=0.5, patch_px=64).build_grid(r)
        assert grid.level0_patch > 0
        with pytest.raises(ValueError):  # non-positive override is rejected
            r.apply_source_mpp(0.0)


def test_plane_collapse_warns_on_t_or_z_stack(synthetic_ngff):
    # dims are (t, c, z, y, x); a >1 extent on t or z means only plane 0 is read.
    with OmeZarrReader(synthetic_ngff) as r:
        for shape, token in (
            ((2, 3, 1, 300, 200), "t=2"),
            ((1, 3, 4, 300, 200), "z=4"),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                r._warn_plane_collapse(shape)
            assert any(token in str(w.message) for w in caught)
        # A single-plane source (the common case) must not warn.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r._warn_plane_collapse((1, 3, 1, 300, 200))
        assert not caught


def test_brightfield_on_multichannel_source_warns(synthetic_multiplex_ngff):
    with OmeZarrReader(synthetic_multiplex_ngff) as r:
        assert len(r.channel_names) > 3  # the multiplex fixture has 5 markers
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_channel_collapse(r, multiplex=False)
        assert any("first three" in str(w.message) for w in caught)
        # A multiplex run reads channels natively, so it must not warn here.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_channel_collapse(r, multiplex=True)
        assert not caught
