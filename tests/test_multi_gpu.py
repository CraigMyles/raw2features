"""Optional in-process multi-GPU (``--devices``): equivalence to single-device.

Two modes are exercised, both on a single box using duplicate CPU "devices"
(``cpu,cpu``) so the equivalence gate runs without real multi-GPU hardware:

* patch-parallel (``embed`` / ``run_slide``) -- shard one slide's patches across
  devices and gather them back; features must be ``np.array_equal`` to the
  single-device run (the gather has to preserve coord order exactly).
* slide-parallel (``embed-many``) -- distribute slides across devices, each slide
  fully embedded on one device; every slide's features must equal the
  single-device result.

Plus: the default (no ``--devices``) path is byte-identical to before, and
``--devices`` is runtime-only (no content/grid-hash change).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from raw2features.pipeline.runner import RunConfig, _contiguous_shards

pytest.importorskip("torch", reason="torch not installed")

from conftest import MockEmbedder  # noqa: E402
from raw2features.pipeline.runner import run_slide  # noqa: E402


# --------------------------------------------------------------------------- #
# A patch-distinguishing mock + a slide whose every patch differs.            #
# --------------------------------------------------------------------------- #
class PosMockEmbedder(MockEmbedder):
    """Mock whose feature row is the patch's exact mean (float32), per dim.

    Unlike a constant embedder, distinct patches map to distinct rows, so a
    gather that reordered rows would change the array and be caught by
    ``np.array_equal``. Mirrors a real embedder by returning a CPU tensor (so the
    ``cuda:0,cuda:0`` path works: ``transform_batch`` puts the input on CUDA, and
    ``_run_batches`` calls ``.numpy()`` on the result, which requires CPU).
    """

    def load(self, device="cpu", dtype=None):
        self._device = device  # record the device so the CUDA path is exercised
        return self

    def embed_batch(self, batch):
        v = batch.float().mean(dim=(1, 2, 3))
        return (v.unsqueeze(1).repeat(1, self.spec.embedding_dim) + self._bias).cpu()


def _ramp2d_store(path: str, *, h: int = 256, w: int = 192, mpp0: float = 0.5) -> str:
    """An OME-NGFF v0.4 store whose pixels encode a 2-D ramp ``(x + y)``.

    Every patch then has a distinct mean, so the patch-parallel gather is genuinely
    order-sensitive (adjacent patches differ, in both axes). One level is enough --
    the patcher reads level 0 at the matching target MPP.
    """
    g = zarr.open_group(path, mode="w", zarr_format=2)
    a = g.create_array(
        "0", shape=(1, 3, 1, h, w), chunks=(1, 1, 1, 64, 64), dtype="uint8"
    )
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    ramp = ((xx + yy) % 256).astype("uint8") * np.ones((h, w), "uint8")
    a[0, 0, 0] = ramp
    a[0, 1, 0] = (255 - ramp).astype("uint8")
    a[0, 2, 0] = ((xx * 2) % 256).astype("uint8") * np.ones((h, w), "uint8")
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    g.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": "ramp2d",
            "axes": axes,
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0, 1.0, 1.0, mpp0, mpp0]}
                    ],
                }
            ],
        }
    ]
    return path


def _features(store_path: str, model: str) -> np.ndarray:
    from raw2features.core.store import open_grid

    g = open_grid(store_path)  # the sole grid
    return np.asarray(g["features"][model][:])


def _coords(store_path: str) -> np.ndarray:
    from raw2features.core.store import open_grid

    g = open_grid(store_path)
    return np.asarray(g["coords"][:])


def _cfg(devices: str | None = None, **over) -> RunConfig:
    base = dict(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        batch_size=8,
        features_dtype="float32",  # max sensitivity for the order check
    )
    base.update(over)
    return RunConfig(devices=devices, **base)


def _factory(_device: str) -> list:
    """Per-device embedder copies for the patch-parallel path (a fresh mock each)."""
    return [PosMockEmbedder(dim=8, name="mock")]


# --------------------------------------------------------------------------- #
# device_list parsing + contiguous sharding (pure logic).                     #
# --------------------------------------------------------------------------- #
def test_device_list_default_is_single_device(monkeypatch):
    # device_list tests the parse/expand logic, so make CUDA appear present to keep it
    # hardware-independent (it must pass in CPU-only CI too). resolve_device's real
    # accelerator checks are covered separately in test_device.py.
    monkeypatch.setattr("raw2features.core.device._accelerators", lambda: (True, False))
    assert RunConfig(models=["m"], device="cuda").device_list() == ["cuda"]
    assert RunConfig(models=["m"], device="cpu", devices=None).device_list() == ["cpu"]
    assert RunConfig(models=["m"], device="cpu", devices="").device_list() == ["cpu"]


def test_device_list_parses_and_preserves_order_and_dups(monkeypatch):
    monkeypatch.setattr("raw2features.core.device._accelerators", lambda: (True, False))
    c = RunConfig(models=["m"], devices="cuda:0,cuda:1")
    assert c.device_list() == ["cuda:0", "cuda:1"]
    # whitespace tolerated; duplicates kept (used to test on one physical GPU)
    assert RunConfig(models=["m"], devices=" cpu , cpu ").device_list() == [
        "cpu",
        "cpu",
    ]
    assert RunConfig(models=["m"], devices="cuda:1,cuda:0").device_list() == [
        "cuda:1",
        "cuda:0",
    ]


def test_contiguous_shards_partition_in_order():
    # contiguous, in order, and concatenation reproduces range(n) exactly
    assert _contiguous_shards(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert _contiguous_shards(9, 3) == [(0, 3), (3, 6), (6, 9)]
    # k > n -> no empty shards (one per item, the rest dropped)
    assert _contiguous_shards(2, 4) == [(0, 1), (1, 2)]
    for n, k in [(10, 3), (9, 3), (1, 1), (100, 7), (5, 5), (2, 4)]:
        shards = _contiguous_shards(n, k)
        flat = [i for lo, hi in shards for i in range(lo, hi)]
        assert flat == list(range(n))
        assert all(hi > lo for lo, hi in shards)  # never an empty worker


def test_devices_is_runtime_only_hashes_unchanged():
    # --devices must not change the store identity (content + grid hash).
    base = RunConfig(models=["resnet50"])
    multi = RunConfig(models=["resnet50"], devices="cuda:0,cuda:1")
    assert base.content_hash() == multi.content_hash()
    assert base.grid_hash() == multi.grid_hash()
    # and the pinned value is still exactly the documented one
    assert multi.content_hash() == "a8c12b66d8558e1b"


# --------------------------------------------------------------------------- #
# Patch-parallel (embed / run_slide): cpu,cpu == single device.               #
# --------------------------------------------------------------------------- #
def test_patch_parallel_features_identical_to_single_device(tmp_path):
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"))

    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(),
        embedders=[PosMockEmbedder(dim=8, name="mock")],
    )
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(devices="cpu,cpu"),
        embedder_factory=_factory,
    )

    fs = _features(single["output_uri"], "mock")
    fm = _features(multi["output_uri"], "mock")
    assert fs.shape == fm.shape and fs.shape[0] == single["n_patches"]
    # the gather MUST preserve coord order exactly -> bit-identical features
    assert np.array_equal(fm, fs)
    assert np.array_equal(_coords(multi["output_uri"]), _coords(single["output_uri"]))
    # the slide genuinely distinguishes patches (else the test would be vacuous)
    assert np.unique(fs, axis=0).shape[0] > 1


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="no CUDA device"
)
def test_patch_parallel_on_real_cuda_duplicate_device(tmp_path):
    """Exercise the real CUDA transform + per-device threads + gather on one GPU.

    Equivalence note: across *distinct* GPUs each patch is fully processed on one
    device exactly as in a single-device run, so the result is bit-identical (the
    CPU ``cpu,cpu`` tests pin that). Listing the *same* physical GPU twice
    (``cuda:0,cuda:0``) is only a 1-GPU test convenience: the two worker threads then
    submit to the same CUDA stream concurrently, so float reductions can reorder and
    differ by ~1 ULP. We therefore assert ``allclose`` here (a gather/order bug would
    show large diffs, not sub-ULP), and rely on the CPU tests for the exact-equality
    gate. Real multi-GPU has no shared-stream contention.
    """
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"))
    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(device="cuda:0"),
        embedders=[PosMockEmbedder(dim=8, name="mock").load("cuda:0")],
    )
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(device="cuda:0", devices="cuda:0,cuda:0"),
        embedder_factory=lambda d: [PosMockEmbedder(dim=8, name="mock").load(d)],
    )
    fm = _features(multi["output_uri"], "mock")
    fs = _features(single["output_uri"], "mock")
    assert fm.shape == fs.shape
    assert np.allclose(fm, fs, rtol=0, atol=1e-6)  # sub-ULP shared-stream jitter only
    assert np.array_equal(_coords(multi["output_uri"]), _coords(single["output_uri"]))


def test_patch_parallel_uneven_shards_and_three_devices(tmp_path):
    # 3 devices over a patch count not divisible by 3 -> uneven contiguous shards.
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"))
    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(),
        embedders=[PosMockEmbedder(dim=8, name="mock")],
    )
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(devices="cpu,cpu,cpu"),
        embedder_factory=_factory,
    )
    assert np.array_equal(
        _features(multi["output_uri"], "mock"), _features(single["output_uri"], "mock")
    )


def test_patch_parallel_multi_model_gather(tmp_path):
    # Two models with different dims/bias: each model's column is gathered
    # independently and must match single-device for both.
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"))

    def two(_device: str) -> list:
        return [
            PosMockEmbedder(dim=8, name="a", bias=0.0),
            PosMockEmbedder(dim=4, name="b", bias=10.0),
        ]

    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(models=["a", "b"]),
        embedders=two("cpu"),
    )
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(models=["a", "b"], devices="cpu,cpu"),
        embedder_factory=two,
    )
    for m in ("a", "b"):
        assert np.array_equal(
            _features(multi["output_uri"], m), _features(single["output_uri"], m)
        ), m


def test_patch_parallel_zero_patches(tmp_path):
    # An impossible tissue threshold keeps 0 cells; the multi-device path must
    # produce the same empty store as single-device (no concatenate-of-empty crash).
    slide = _ramp2d_store(str(tmp_path / "r.zarr"))
    zero = dict(segmenter="otsu", no_seg=False, tissue_threshold=2.0)  # >1 -> nothing
    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(**zero),
        embedders=[PosMockEmbedder(dim=8, name="mock")],
    )
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(devices="cpu,cpu", **zero),
        embedder_factory=_factory,
    )
    assert single["n_patches"] == 0 and multi["n_patches"] == 0
    assert _features(multi["output_uri"], "mock").shape == (0, 8)
    assert np.array_equal(
        _features(multi["output_uri"], "mock"), _features(single["output_uri"], "mock")
    )


def test_more_devices_than_patches_does_not_spawn_empty_workers(tmp_path):
    # Tiny slide, many devices: contiguous-shard drops empties; still equivalent.
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"), h=64, w=64)  # 1 patch
    single = run_slide(
        slide,
        str(tmp_path / "single"),
        _cfg(),
        embedders=[PosMockEmbedder(dim=8, name="mock")],
    )
    assert single["n_patches"] == 1
    multi = run_slide(
        slide,
        str(tmp_path / "multi"),
        _cfg(devices="cpu,cpu,cpu,cpu"),
        embedder_factory=_factory,
    )
    assert np.array_equal(
        _features(multi["output_uri"], "mock"), _features(single["output_uri"], "mock")
    )


# --------------------------------------------------------------------------- #
# Default (no --devices) path is unchanged.                                    #
# --------------------------------------------------------------------------- #
def test_default_path_byte_identical_to_explicit_single_device(tmp_path):
    # devices=None and devices="cpu" (one device) must both take the single-device
    # path and produce identical output.
    slide = _ramp2d_store(str(tmp_path / "ramp.zarr"))
    a = run_slide(
        slide, str(tmp_path / "a"), _cfg(), embedders=[PosMockEmbedder(name="mock")]
    )
    b = run_slide(
        slide,
        str(tmp_path / "b"),
        _cfg(devices="cpu"),
        embedders=[PosMockEmbedder(name="mock")],
    )
    assert np.array_equal(
        _features(a["output_uri"], "mock"), _features(b["output_uri"], "mock")
    )


# --------------------------------------------------------------------------- #
# Slide-parallel (embed-many): cpu,cpu == single device, per slide.           #
# --------------------------------------------------------------------------- #
def _build_slides(tmp_path, n: int = 5) -> str:
    d = tmp_path / "slides"
    d.mkdir()
    for i in range(n):
        # vary size so the slides differ (distinct features per slide)
        _ramp2d_store(str(d / f"s{i}.zarr"), h=128 + 16 * i, w=96 + 16 * i)
    return str(d)


def test_slide_parallel_matches_single_device(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import raw2features.cli.embed_many as em
    from raw2features.cli.main import app

    # load_embedders is called as load_embedders(cfg, device) by the parallel path;
    # replicate a fresh mock per device (no registry / no weights).
    monkeypatch.setattr(em, "load_embedders", lambda cfg, device: [_factory(device)[0]])

    slides = _build_slides(tmp_path, n=5)
    out = str(tmp_path / "out")
    common = [
        "-f",
        "mock",
        "--no-seg",
        "--mpp",
        "0.5",
        "--patch-size",
        "64",
        "--features-dtype",
        "float32",
    ]
    r = CliRunner().invoke(
        app, ["embed-many", slides, out, *common, "--devices", "cpu,cpu"]
    )
    assert r.exit_code == 0, r.output
    assert "5 embedded" in r.output

    # Each slide's features must equal a direct single-device run of that slide.
    for name in sorted(os.listdir(slides)):
        sid = name.removesuffix(".zarr")
        got = _features(os.path.join(out, f"{sid}.embeddings.zarr"), "mock")
        ref = run_slide(
            os.path.join(slides, name),
            str(tmp_path / "ref" / sid),
            _cfg(),
            embedders=[PosMockEmbedder(dim=8, name="mock")],
        )
        assert np.array_equal(got, _features(ref["output_uri"], "mock")), sid


def test_slide_parallel_covers_all_and_is_idempotent(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import raw2features.cli.embed_many as em
    from raw2features.cli.main import app

    monkeypatch.setattr(em, "load_embedders", lambda cfg, device: [_factory(device)[0]])
    slides = _build_slides(tmp_path, n=4)
    out, rcpts = str(tmp_path / "out"), str(tmp_path / "rcpts")
    base = [
        "embed-many",
        slides,
        out,
        "-f",
        "mock",
        "--no-seg",
        "--mpp",
        "0.5",
        "--patch-size",
        "64",
        "--features-dtype",
        "float32",
        "--devices",
        "cpu,cpu",
        "--receipts-dir",
        rcpts,
    ]
    r1 = CliRunner().invoke(app, base)
    assert r1.exit_code == 0, r1.output
    for i in range(4):
        assert os.path.exists(os.path.join(out, f"s{i}.embeddings.zarr"))
    # re-run: every slide already complete -> all skipped (idempotent resume)
    r2 = CliRunner().invoke(app, base)
    assert r2.exit_code == 0, r2.output
    assert "4 skipped" in r2.output


def test_device_list_auto_uses_all_visible_gpus(monkeypatch):
    """`--devices auto` expands to every visible CUDA GPU, and falls back to the
    single device when 0/1 GPU is present (so it's safe on a laptop / 1-GPU box)."""
    torch = pytest.importorskip("torch")
    from raw2features.pipeline.runner import RunConfig

    # Make CUDA appear present so device resolution is hardware-independent (CPU-only
    # CI included); device_count is mocked per-case below to vary the GPU count.
    monkeypatch.setattr("raw2features.core.device._accelerators", lambda: (True, False))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert RunConfig(models=["m"], devices="auto").device_list() == [
        "cuda:0",
        "cuda:1",
        "cuda:2",
    ]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert RunConfig(models=["m"], devices="auto", device="cuda").device_list() == [
        "cuda"
    ]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert RunConfig(models=["m"], devices="auto").device_list() == ["cuda"]
