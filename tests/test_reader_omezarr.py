"""Tests for the OME-Zarr reader against the synthetic NGFF fixture."""

from __future__ import annotations

import functools
import http.server
import re
import threading
from urllib.parse import urlsplit

import numpy as np
import pytest
import zarr

from conftest import build_ngff_v04
from raw2features.core.geometry import Point, Region, Size
from raw2features.readers.omezarr import OmeZarrReader, _ChunkCache


def test_query_auth_is_attached_to_every_http_zarr_request(tmp_path):
    root = tmp_path / "remote.zarr"
    zarr.open_group(str(root), mode="w", zarr_format=2)
    build_ngff_v04(str(root / "0"), sizes=((64, 64), (32, 32)))
    expected_query = "token=R2F%2FSECRET%2BVALUE%3D&series=1&series=2&empty="
    seen = []

    class AuthenticatedHandler(http.server.SimpleHTTPRequestHandler):
        def _authorized(self):
            seen.append(self.path)
            if urlsplit(self.path).query == expected_query:
                return True
            self.send_error(403, "missing query authentication")
            return False

        def do_GET(self):  # noqa: N802 - stdlib handler API
            if self._authorized():
                super().do_GET()

        def do_HEAD(self):  # noqa: N802 - stdlib handler API
            if self._authorized():
                super().do_HEAD()

        def log_message(self, *args):
            return None

    handler = functools.partial(AuthenticatedHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    uri = f"http://{host}:{port}/remote.zarr?{expected_query}"
    try:
        with OmeZarrReader(uri) as reader:
            image = reader.read_region(Region.patch(x=0, y=0, size=16, level=0))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert image.shape == (16, 16, 3)
    assert seen
    assert all(urlsplit(path).query == expected_query for path in seen)
    assert any(".zattrs" in path or "zarr.json" in path for path in seen)
    assert any("/remote.zarr/0/0/" in urlsplit(path).path for path in seen)
    assert any(
        (request_path := urlsplit(path).path).startswith("/remote.zarr/0/0/")
        and re.fullmatch(r"\d+(?:\.\d+)+", request_path.rsplit("/", 1)[-1]) is not None
        for path in seen
    )


def _direct_slice_block(r: OmeZarrReader, level: int, x0: int, y0: int, w: int, h: int):
    """Reference direct slice, with every extra axis indexed at zero."""
    arr = r._arrays[level]
    idx: list[object] = []
    for d in r._dims:
        if d == "y":
            idx.append(slice(y0, y0 + h))
        elif d == "x":
            idx.append(slice(x0, x0 + w))
        elif d == "c":
            idx.append(slice(None))
        else:
            idx.append(0)
    return np.asarray(arr[tuple(idx)])


def _write_single_level_ngff(
    path,
    *,
    data: np.ndarray,
    chunks: tuple[int, ...],
    axes: list[dict[str, str]],
    scale: list[float],
) -> str:
    """Write a minimal offline NGFF v0.4 store with an explicit axis layout."""
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    arr = g.create_array("0", shape=data.shape, chunks=chunks, dtype=data.dtype)
    arr[...] = data
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "axis-order-regression",
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": scale}],
                }
            ],
        }
    ]
    return str(path)


def _write_float_pyramid_ngff(
    path,
    *,
    level0: np.ndarray,
    level1: np.ndarray,
) -> str:
    """Write a two-level c,y,x float pyramid for scaling regressions."""
    g = zarr.open_group(str(path), mode="w", zarr_format=2)
    for key, data in (("0", level0), ("1", level1)):
        chunks = (1, min(4, data.shape[1]), min(4, data.shape[2]))
        arr = g.create_array(key, shape=data.shape, chunks=chunks, dtype=data.dtype)
        arr[...] = data
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "float-scaling-regression",
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 0.5, 0.5]}
                    ],
                },
                {
                    "path": "1",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                    ],
                },
            ],
        }
    ]
    return str(path)


def test_metadata_mpp_dims_downsamples(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        assert r.mpp == 0.5
        # sizes are (H, W) = (200,300) -> level_dimensions are Size(width, height)
        assert r.level_dimensions[0] == Size(300, 200)
        assert r.level_dimensions[1] == Size(150, 100)
        assert r.level_downsamples() == [1.0, 2.0, 4.0]
        assert r.ngff_version == "0.4"


def test_read_region_returns_hwc_uint8_rgb(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        img = r.read_region(Region.patch(x=0, y=0, size=32, level=0))
        assert img.shape == (32, 32, 3)
        assert img.dtype == np.uint8
        # R channel is a vertical ramp (increases with y); B is constant 128.
        assert img[0, 0, 0] < img[31, 0, 0]
        assert np.all(img[:, :, 2] == 128)


def test_level0_location_maps_through_downsample(synthetic_ngff):
    # A patch at level-0 (x=64,y=0) read at level 1 (downsample 2) starts at (32,0).
    with OmeZarrReader(synthetic_ngff) as r:
        img = r.read_region(Region(level=1, location=Point(64, 0), size=Size(16, 16)))
        assert img.shape == (16, 16, 3)


def test_border_read_is_padded_to_requested_size(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        # level-0 width is 300; request a patch overrunning the right edge.
        img = r.read_region(Region.patch(x=290, y=0, size=32, level=0))
        assert img.shape == (32, 32, 3)  # padded with white
        assert np.all(img[:, 20:, :] == 255)


def test_negative_translated_border_padding_keeps_source_offset(tmp_path):
    """A translated level's clipped pixels land after the top/left padding."""
    store = build_ngff_v04(
        str(tmp_path / "translated-border.zarr"), sizes=((8, 10), (4, 5))
    )
    root = zarr.open_group(store, mode="r+")
    attrs = dict(root.attrs)
    datasets = attrs["multiscales"][0]["datasets"]
    # Level 1 is 1 µm/px. A (+2,+1) µm source translation maps level-0 (0,0)
    # to level-1 index (-2,-1), so the returned 4x4 window needs two white/zero
    # columns and one white/zero row before the clipped source data.
    datasets[1]["coordinateTransformations"].append(
        {"type": "translation", "translation": [0.0, 0.0, 0.0, 1.0, 2.0]}
    )
    root.attrs.update(attrs)

    region = Region(level=1, location=Point(0, 0), size=Size(4, 4))
    with OmeZarrReader(store) as reader:
        assert reader.read_level_mapping(1)[2:] == (-2.0, -1.0)
        native = reader.read_region_channels(region)
        rgb = reader.read_region(region)

    source = np.asarray(root["1"][0, :, 0, :3, :2]).transpose(1, 2, 0)
    expected_native = np.zeros((4, 4, 3), dtype=np.uint8)
    expected_native[1:, 2:, :] = source
    expected_rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
    expected_rgb[1:, 2:, :] = source
    np.testing.assert_array_equal(native, expected_native)
    np.testing.assert_array_equal(rgb, expected_rgb)


def test_exact_mpp_plan_via_reader(synthetic_ngff):
    with OmeZarrReader(synthetic_ngff) as r:
        plan = r.level_for_mpp(1.0, 224)  # mpp0=0.5 -> level 1 is exactly 1.0
        assert plan.level == 1
        assert plan.achieved_mpp == 1.0
        assert plan.needs_resample is False


# -- decompressed-chunk cache -------------------------------------------------
def test_cached_block_equals_direct_zarr_slice(synthetic_ngff):
    """The chunk-cache assembly is byte-identical to a direct multi-chunk slice.

    The fixture chunks y/x at 64 px, so windows wider/taller than 64 px span
    several chunks - exactly the case the cache assembles from per-chunk planes.
    """
    with OmeZarrReader(synthetic_ngff) as r:
        arr = r._arrays[0]
        cases = [
            (0, 0, 32, 32),  # within a single chunk
            (0, 0, 130, 100),  # spans 3x2 chunks
            (50, 40, 120, 90),  # unaligned start, spans chunk borders
            (200, 150, 80, 60),  # clipped at the right/bottom edge (W=300, H=200)
        ]
        for x0, y0, w, h in cases:
            got, _ = r._read_block_cached(arr, 0, x0, y0, w, h)
            assert np.array_equal(got, _direct_slice_block(r, 0, x0, y0, w, h)), (
                x0,
                y0,
                w,
                h,
            )


def test_cached_read_region_equals_uncached(synthetic_ngff):
    """Full read_region output is identical with the cache on vs disabled."""
    region = Region(level=0, location=Point(40, 30), size=Size(150, 120))
    with OmeZarrReader(synthetic_ngff) as r:
        cached = r.read_region(region)
        # Re-read the same region: now served from the cache (must be unchanged).
        cached_again = r.read_region(region)
    with OmeZarrReader(synthetic_ngff) as r:
        r._chunk_cache.capacity = 0  # disable: every call reads straight from zarr
        uncached = r.read_region(region)
    assert np.array_equal(cached, uncached)
    assert np.array_equal(cached_again, uncached)


def test_cached_read_normalises_noncanonical_x_c_y_axis_order(tmp_path):
    """An x,c,y source is read with correct orientation across unequal chunks."""
    height, width = 9, 13
    yy, xx = np.indices((height, width))
    expected = np.stack(
        [xx + 10 * yy, 80 + xx + 10 * yy, 160 + xx + 10 * yy], axis=-1
    ).astype(np.uint8)
    # Source order is deliberately x,c,y rather than a conventional c,y,x. Unequal
    # dimensions and chunks make an accidental x/y swap observable.
    source = np.transpose(expected, (1, 2, 0))
    store = _write_single_level_ngff(
        tmp_path / "x-c-y.zarr",
        data=source,
        chunks=(5, 1, 3),
        axes=[
            {"name": "x", "type": "space", "unit": "micrometer"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
        ],
        scale=[0.5, 1.0, 0.5],
    )
    region = Region(level=0, location=Point(3, 2), size=Size(8, 6))

    with OmeZarrReader(store) as reader:
        assert reader.axes == ("x", "c", "y")
        native = reader.read_region_channels(region)
        rgb = reader.read_region(region)
        assert reader._chunk_cache.misses > 1

    wanted = expected[2:8, 3:11, :]
    np.testing.assert_array_equal(native, wanted)
    np.testing.assert_array_equal(rgb, wanted)


def test_non_singleton_custom_axis_warns_and_reads_index_zero(tmp_path):
    """Every extent-bearing extra axis warns; its index-zero plane is selected."""
    height, width = 7, 11
    yy, xx = np.indices((height, width))
    phase0 = np.stack(
        [xx + 10 * yy, 70 + xx + 10 * yy, 140 + xx + 10 * yy], axis=0
    ).astype(np.uint8)
    phase1 = np.full_like(phase0, 255)
    store = _write_single_level_ngff(
        tmp_path / "custom-axis.zarr",
        data=np.stack([phase0, phase1], axis=0),
        chunks=(1, 1, 3, 5),
        axes=[
            {"name": "phase", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        scale=[1.0, 1.0, 0.5, 0.5],
    )

    with pytest.warns(UserWarning, match=r"phase=2; only phase=0 is read"):
        with OmeZarrReader(store) as reader:
            got = reader.read_region_channels(
                Region(level=0, location=Point(2, 1), size=Size(7, 5))
            )

    wanted = np.transpose(phase0[:, 1:6, 2:9], (1, 2, 0))
    np.testing.assert_array_equal(got, wanted)


def test_overlapping_reads_hit_the_cache(synthetic_ngff):
    """Adjacent/overlapping reads reuse decompressed chunks (cache hits > 0)."""
    with OmeZarrReader(synthetic_ngff) as r:
        # Two horizontally adjacent 64-px windows share the chunk column at x=64.
        r.read_region(Region(level=0, location=Point(40, 40), size=Size(64, 64)))
        misses_after_first = r._chunk_cache.misses
        r.read_region(Region(level=0, location=Point(56, 40), size=Size(64, 64)))
        assert r._chunk_cache.hits > 0
        # The second (overlapping) read decompressed fewer new chunks than the first.
        assert r._chunk_cache.misses < 2 * misses_after_first


def test_chunk_cache_lru_eviction_and_disabled():
    """The standalone cache bounds itself (LRU) and no-ops at capacity 0."""
    calls = {"n": 0}

    def make_reader(val):
        def _read():
            calls["n"] += 1
            return np.full((4, 4), val, dtype=np.uint8)

        return _read

    cache = _ChunkCache(capacity=2)
    a = cache.get_or_read(("k", 0), make_reader(1))
    cache.get_or_read(("k", 1), make_reader(2))
    # A repeat hit does not re-read.
    again = cache.get_or_read(("k", 0), make_reader(99))
    assert np.array_equal(a, again)
    assert cache.hits == 1 and calls["n"] == 2
    # Inserting a third key evicts the LRU entry (k,1); (k,0) was just used.
    cache.get_or_read(("k", 2), make_reader(3))
    assert len(cache._store) == 2
    assert ("k", 1) not in cache._store and ("k", 0) in cache._store

    # capacity 0: never stores, always invokes the reader (pre-cache behaviour).
    off = _ChunkCache(capacity=0)
    off.get_or_read(("k", 0), make_reader(1))
    off.get_or_read(("k", 0), make_reader(1))
    assert len(off._store) == 0 and calls["n"] == 5  # 4 + 1 (only the first off read)


def test_to_uint8_handles_unsigned_signed_and_float():
    f = OmeZarrReader._to_uint8
    u8 = np.full((2, 2, 3), 100, np.uint8)
    np.testing.assert_array_equal(f(u8), u8)  # uint8 passthrough
    assert (f(np.full((1, 1, 3), 65535, np.uint16)) == 255).all()  # uint16 rescaled
    # signed int: negatives clamp to 0, positive rescales by dtype max
    np.testing.assert_array_equal(
        f(np.array([[[-5, 32767, 0]]], np.int16))[0, 0], [0, 255, 0]
    )
    # float in [0, 1] -> *255; float already in [0, 255] -> clip passthrough
    np.testing.assert_array_equal(
        f(np.array([[[0.0, 1.0, 0.5]]], np.float32))[0, 0], [0, 255, 128]
    )
    np.testing.assert_array_equal(
        f(np.array([[[0.0, 255.0, 300.0]]], np.float32))[0, 0], [0, 255, 255]
    )


def test_float_byte_range_is_chosen_once_from_level_zero(tmp_path):
    """A locally dark patch must not be mistaken for normalized float pixels."""
    level0 = np.full((3, 12, 12), 0.75, dtype=np.float32)
    # A byte-range value in a distant level-zero chunk establishes the convention.
    level0[:, -2:, -2:] = 200.0
    level1 = np.full((3, 6, 6), 0.75, dtype=np.float32)
    store = _write_float_pyramid_ngff(
        tmp_path / "float-byte.zarr",
        level0=level0,
        level1=level1,
    )

    with OmeZarrReader(store) as reader:
        assert reader._float_to_uint8_scale == 1.0
        dark_level0 = reader.read_region(
            Region(level=0, location=Point(0, 0), size=Size(3, 3))
        )
        dark_level1 = reader.read_region(
            Region(level=1, location=Point(0, 0), size=Size(3, 3))
        )
        native = reader.read_region_channels(
            Region(level=0, location=Point(0, 0), size=Size(3, 3))
        )

    # Per-region inference used to turn these local 0.75 values into 191. The
    # level-zero byte-range decision now applies to both levels, while the native
    # multiplex-style read remains untouched.
    assert np.all(dark_level0 == 1)
    assert np.all(dark_level1 == 1)
    assert native.dtype == np.float32
    assert np.all(native == np.float32(0.75))


def test_float_unit_range_decision_is_reused_by_lower_levels(tmp_path):
    """Lower-level local extrema cannot revise a normalized slide's convention."""
    level0 = np.full((3, 12, 12), 0.5, dtype=np.float32)
    level0[:, -2:, -2:] = 1.0
    # Deliberate out-of-contract sentinel: if level 1 reclassified itself from its
    # local max, this would become 2 instead of saturating under level zero's choice.
    level1 = np.full((3, 6, 6), 2.0, dtype=np.float32)
    store = _write_float_pyramid_ngff(
        tmp_path / "float-unit.zarr",
        level0=level0,
        level1=level1,
    )

    with OmeZarrReader(store) as reader:
        assert reader._float_to_uint8_scale == 255.0
        normalized = reader.read_region(
            Region(level=0, location=Point(0, 0), size=Size(3, 3))
        )
        lower_level = reader.read_region(
            Region(level=1, location=Point(0, 0), size=Size(3, 3))
        )

    assert np.all(normalized == 128)
    assert np.all(lower_level == 255)
