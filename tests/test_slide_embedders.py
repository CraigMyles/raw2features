"""Tests for the slide-embedder seam and the mean/max pooling baselines."""

from __future__ import annotations

import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from conftest import MockEmbedder
from raw2features.cli.main import app
from raw2features.core.store import open_grid
from raw2features.pipeline.runner import RunConfig, run_slide
from raw2features.slide_embedders.model_registry import (
    build_slide_embedder,
    get_slide_spec,
    load_slide_registry,
    resolve_patch_encoder,
)
from raw2features.slide_embedders.pool import (
    MaxPoolSlideEmbedder,
    MeanMaxPoolSlideEmbedder,
    MeanPoolSlideEmbedder,
)

try:
    import torch as _torch_mod  # noqa: F401

    _TORCH = True
except ImportError:
    _TORCH = False


def _write_slide_store(tmp_path, grids):
    """Create a minimal multi-grid v0.1 store for standalone CLI tests."""
    path = str(tmp_path / "slide.embeddings.zarr")
    root = zarr.open_group(path, mode="w", zarr_format=2)
    grid_group = root.create_group("grids")
    root.attrs["raw2features"] = {
        "schema_version": "0.1",
        "grids": {key: {} for key in grids},
    }
    for key, (level0_patch, models) in grids.items():
        group = grid_group.create_group(key)
        group.attrs["raw2features"] = {
            "schema_version": "0.1",
            "patching": {"level0_patch": level0_patch},
        }
        n = next(iter(models.values())).shape[0]
        coords = np.column_stack(
            [np.arange(n, dtype=np.int32) * level0_patch, np.zeros(n, np.int32)]
        )
        coords_array = group.create_array(
            "coords", shape=coords.shape, chunks=coords.shape, dtype="int32"
        )
        coords_array[:] = coords
        feature_group = group.create_group("features")
        for model, values in models.items():
            values = np.asarray(values, dtype=np.float32)
            array = feature_group.create_array(
                model,
                shape=values.shape,
                chunks=values.shape,
                dtype="float32",
            )
            array[:] = values
    return path


@pytest.fixture
def recording_slide_embedder(monkeypatch):
    """Replace weighted/pooling encoders with a deterministic call recorder."""
    state = {"loads": 0, "encodes": []}

    class _Recorder:
        def __init__(self, name):
            self.spec = get_slide_spec(name)
            self.ordinal = 0

        def load(self, device="cpu", dtype=None):
            state["loads"] += 1
            self.ordinal = state["loads"]
            return self

        def encode(self, features, coords=None, patch_size_lv0=None):
            state["encodes"].append(
                {
                    "features": np.asarray(features).copy(),
                    "coords": None if coords is None else np.asarray(coords).copy(),
                    "patch_size_lv0": patch_size_lv0,
                }
            )
            return np.asarray(features, dtype=np.float32).mean(axis=0) + self.ordinal

        def unload(self):
            return None

    monkeypatch.setattr(
        "raw2features.slide_embedders.model_registry.build_slide_embedder",
        lambda name: _Recorder(name),
    )
    return state


# -- registry ------------------------------------------------------------------


def test_patch_registry_ignores_slide_encoders_section():
    # The slide encoders live in the same registry.yaml under a reserved
    # 'slide_encoders' key. The PATCH registry loader must skip it, or it tries
    # to build a patch ModelSpec from the slide section and raises
    # KeyError('family') - which silently broke every embed run once.
    from raw2features.embedders.model_registry import load_registry

    patch_reg = load_registry()
    assert "slide_encoders" not in patch_reg
    assert set(patch_reg) == {
        "resnet50", "dinov2", "uni", "uni2_h", "path_orchestra", "virchow2",
        "gigapath", "conch",
        "conch_v1_5", "h_optimus_0", "gpfm", "midnight", "ctranspath", "hibou_l",
        "hibou_b", "kronos", "phikon", "phikon_v2",
        "lunit_dino", "lunit_dino8", "lunit_bt", "lunit_mocov2", "lunit_swav",
        "sp22m", "retccl", "hipt", "h_optimus_1", "virchow", "musk", "mstar",
        "kaiko_vitl", "quiltnet", "biomedclip", "plip", "seal_conch", "seal_univ2",
    }


def test_slide_registry_has_pooling_baselines():
    reg = load_slide_registry()
    for name in ("mean", "max", "meanmax"):
        assert name in reg
        spec = reg[name]
        assert spec.gated is False
        assert spec.license == "MIT"


def test_slide_registry_has_titan():
    reg = load_slide_registry()
    assert "titan" in reg
    spec = reg["titan"]
    # TITAN consumes CONCH v1.5 (768-d) patch features, not UNI.
    assert spec.patch_encoder == "conch_v1_5"
    assert spec.patch_dim == 768
    assert spec.embedding_dim == 768
    assert spec.gated is True
    assert spec.doi == "10.1038/s41591-025-03982-3"


def test_get_slide_spec_unknown_raises():
    with pytest.raises(KeyError, match="Unknown slide encoder"):
        get_slide_spec("does_not_exist")


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [n for n in sorted(load_slide_registry()) if n not in ("mean", "max", "meanmax")],
)
def test_slide_forward_output_matches_declared_dim(name):
    """Each weighted slide encoder's forward must return (embedding_dim,) and be finite,
    under real weights. Gated encoders skip cleanly when their weights aren't available
    (no HF token / offline). Feeds a random (N, patch_dim) feature matrix + coords; the
    position-aware ones (TITAN) use the coords, the rest ignore them."""
    torch = pytest.importorskip("torch")
    spec = get_slide_spec(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        emb = build_slide_embedder(name).load(device=device)
    except Exception as exc:  # noqa: BLE001 - gated weights / no token / offline
        pytest.skip(f"{name}: weights unavailable ({type(exc).__name__})")
    rng = np.random.RandomState(0)
    n = 96
    feats = rng.rand(n, spec.patch_dim).astype(np.float32)
    coords = rng.randint(0, 20000, (n, 2)).astype(np.int32)
    out = emb.encode(feats, coords=coords, patch_size_lv0=512)
    assert out.shape == (spec.embedding_dim,)
    assert bool(np.isfinite(out).all())


# -- resolve_patch_encoder -----------------------------------------------------


def test_resolve_patch_encoder_auto():
    pm = resolve_patch_encoder("titan", ["conch_v1_5"])
    assert pm == "conch_v1_5"


def test_resolve_patch_encoder_missing_raises():
    with pytest.raises(ValueError, match="requires patch features from"):
        resolve_patch_encoder("titan", ["resnet50"])


def test_resolve_patch_encoder_any_picks_sole_model():
    # Pooling encoders declare patch_encoder="any" -> accept whatever is present.
    assert resolve_patch_encoder("mean", ["mock"]) == "mock"
    assert resolve_patch_encoder("max", ["uni"]) == "uni"


def test_resolve_patch_encoder_any_requires_disambiguation():
    with pytest.raises(ValueError, match="multiple patch models"):
        resolve_patch_encoder("mean", ["uni", "resnet50"])


def test_resolve_patch_encoder_empty_raises():
    with pytest.raises(ValueError, match="No patch features"):
        resolve_patch_encoder("mean", [])


# -- pooling baselines ---------------------------------------------------------


def test_mean_pool_shape_and_values():
    emb = MeanPoolSlideEmbedder(patch_encoder="resnet50", patch_dim=8)
    emb.load()
    feats = np.ones((10, 8), dtype=np.float32) * 2.0
    out = emb.encode(feats)
    assert out.shape == (8,)
    assert np.allclose(out, 2.0)


def test_max_pool_shape():
    emb = MaxPoolSlideEmbedder(patch_encoder="resnet50", patch_dim=8)
    emb.load()
    rng = np.random.RandomState(0)
    feats = rng.rand(20, 8).astype(np.float32)
    out = emb.encode(feats)
    assert out.shape == (8,)
    assert np.allclose(out, feats.max(axis=0))


def test_meanmax_pool_dim_is_doubled():
    emb = MeanMaxPoolSlideEmbedder(patch_encoder="resnet50", patch_dim=8)
    emb.load()
    feats = np.ones((5, 8), dtype=np.float32)
    out = emb.encode(feats)
    assert out.shape == (16,)


def test_build_slide_embedder_resolves_pool():
    emb = build_slide_embedder("mean")
    assert isinstance(emb, MeanPoolSlideEmbedder)


# -- inline runner integration -------------------------------------------------


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_embed_with_inline_slide_encoder(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
        slide_encoders=["mean"],
    )
    summary = run_slide(
        synthetic_ngff,
        str(tmp_path / "out"),
        cfg,
        embedders=[MockEmbedder(dim=8, bias=1.0)],
    )
    assert summary["status"] == "complete"
    assert "slide_embeddings" in summary
    assert "mean" in summary["slide_embeddings"]

    g = open_grid(summary["output_uri"])  # the sole grid
    assert "slide" in g
    sv = np.asarray(g["slide"]["mean"][:])
    assert sv.shape == (1, 8)
    assert np.isfinite(sv).all()
    # value invariant: the slide vector IS the mean of the persisted patch features
    # (the point of mean pooling - shape/finite/non-zero alone wouldn't catch a pooling
    # regression that returns a different finite vector, e.g. a sum or the wrong axis).
    pf = np.asarray(g["features"]["mock"][:]).astype(np.float32)
    np.testing.assert_allclose(sv[0], pf.mean(axis=0), rtol=0, atol=1e-3)
    # Provenance recorded in the grid header
    hdr = dict(g.attrs)["raw2features"]
    assert "slide_embeddings" in hdr
    assert hdr["slide_embeddings"]["mean"]["patch_encoder"] == "mock"

    # A no-receipt rerun follows the public append path. The existing slide vector
    # must count as complete instead of reaching create_array() and failing.
    rerun = run_slide(
        synthetic_ngff,
        str(tmp_path / "out"),
        cfg,
        embedders=[MockEmbedder(dim=8, bias=1.0)],
    )
    assert rerun["status"] == "complete"
    assert rerun["slide_embeddings"] == {"mean": "slide/mean"}


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_runner_threads_patch_size_lv0(synthetic_ngff, tmp_path, monkeypatch):
    """Position-aware encoders (TITAN) need patch_size_lv0; the runner must pass the
    store's level0_patch. A recording encoder captures what it actually receives."""
    from raw2features.slide_embedders.base import SlideEmbedder

    captured: dict = {}

    class _Recorder(SlideEmbedder):
        def __init__(self) -> None:
            super().__init__(get_slide_spec("mean"))

        def load(self, device="cpu", dtype=None) -> _Recorder:
            return self

        def encode(self, features, coords=None, patch_size_lv0=None):
            captured["patch_size_lv0"] = patch_size_lv0
            captured["coords_is_none"] = coords is None
            return features.astype(np.float32).mean(axis=0)

    monkeypatch.setattr(
        "raw2features.slide_embedders.model_registry.build_slide_embedder",
        lambda name: _Recorder(),
    )

    cfg = RunConfig(
        models=["mock"], no_seg=True, target_mpp=0.5, patch_px=64,
        device="cpu", amp="fp32", slide_encoders=["mean"],
    )
    s = run_slide(
        synthetic_ngff, str(tmp_path / "out"), cfg, embedders=[MockEmbedder(dim=8)]
    )
    g = open_grid(s["output_uri"])  # the sole grid
    expected = dict(g.attrs)["raw2features"]["patching"]["level0_patch"]
    assert captured["patch_size_lv0"] == expected
    assert captured["coords_is_none"] is False  # coords passed through too


# -- standalone slide-embed CLI ------------------------------------------------


def test_cli_slide_embed_threads_patch_size_lv0(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (137, {"mock": np.ones((3, 4), dtype=np.float32)})},
    )

    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "mean",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    call = recording_slide_embedder["encodes"][0]
    assert call["patch_size_lv0"] == 137
    assert call["coords"] is not None
    np.testing.assert_array_equal(call["coords"][:, 0], [0, 137, 274])


def test_cli_slide_embed_selects_explicit_grid(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {
            "mpp0.5_px64": (128, {"mock_a": np.ones((2, 3), np.float32)}),
            "mpp1_px64": (64, {"mock_b": np.full((2, 3), 2, np.float32)}),
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "mean",
            "--grid",
            "mpp1_px64",
            "--patch-model",
            "mock_b",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert "slide" not in open_grid(root, "mpp0.5_px64")
    assert "mean" in open_grid(root, "mpp1_px64")["slide"]
    assert "mpp1_px64" in result.output
    assert recording_slide_embedder["encodes"][0]["patch_size_lv0"] == 64


def test_cli_slide_embed_infers_grid_for_required_model(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {
            "mpp0.5_px512": (
                1024,
                {"conch_v1_5": np.ones((2, 3), np.float32)},
            ),
            "mpp1_px224": (224, {"uni": np.full((2, 3), 2, np.float32)}),
        },
    )

    result = CliRunner().invoke(
        app,
        ["slide-embed", path, "-s", "titan", "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    assert "titan" in open_grid(root, "mpp0.5_px512")["slide"]
    assert "slide" not in open_grid(root, "mpp1_px224")
    call = recording_slide_embedder["encodes"][0]
    assert call["patch_size_lv0"] == 1024
    assert np.all(call["features"] == 1)


def test_cli_slide_embed_requires_grid_for_ambiguous_pooling(tmp_path):
    path = _write_slide_store(
        tmp_path,
        {
            "mpp0.5_px64": (128, {"mock_a": np.ones((2, 3), np.float32)}),
            "mpp1_px64": (64, {"mock_b": np.ones((2, 3), np.float32)}),
        },
    )

    result = CliRunner().invoke(
        app,
        ["slide-embed", path, "-s", "mean", "--device", "cpu"],
    )

    assert result.exit_code == 1
    assert "multiple grids" in result.output
    assert "--grid" in result.output
    assert "mpp0.5_px64" in result.output
    assert "mpp1_px64" in result.output


def test_cli_slide_embed_force_recomputes(tmp_path, recording_slide_embedder):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (128, {"mock": np.ones((2, 3), np.float32)})},
    )
    command = [
        "slide-embed",
        path,
        "-s",
        "mean",
        "--patch-model",
        "mock",
        "--device",
        "cpu",
    ]

    first = CliRunner().invoke(app, command)
    assert first.exit_code == 0, first.output
    first_vector = np.asarray(open_grid(path)["slide"]["mean"][:]).copy()

    second = CliRunner().invoke(app, command)
    assert second.exit_code == 0, second.output
    assert "skipping" in second.output
    assert recording_slide_embedder["loads"] == 1
    np.testing.assert_array_equal(
        np.asarray(open_grid(path)["slide"]["mean"][:]), first_vector
    )

    forced = CliRunner().invoke(app, [*command, "--force"])
    assert forced.exit_code == 0, forced.output
    assert recording_slide_embedder["loads"] == 2
    assert not np.array_equal(
        np.asarray(open_grid(path)["slide"]["mean"][:]), first_vector
    )


def test_cli_slide_embed_recomputes_when_patch_model_changes(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {
            "mpp0.5_px64": (
                128,
                {
                    "mock_a": np.ones((2, 3), np.float32),
                    "mock_b": np.full((2, 3), 2, np.float32),
                },
            )
        },
    )
    base = ["slide-embed", path, "-s", "mean", "--device", "cpu"]

    first = CliRunner().invoke(app, [*base, "--patch-model", "mock_a"])
    second = CliRunner().invoke(app, [*base, "--patch-model", "mock_b"])

    assert first.exit_code == second.exit_code == 0
    assert recording_slide_embedder["loads"] == 2
    array = open_grid(path)["slide"]["mean"]
    assert array.attrs["patch_encoder"] == "mock_b"
    assert dict(open_grid(path).attrs)["raw2features"]["slide_embeddings"]["mean"][
        "patch_encoder"
    ] == "mock_b"


def test_cli_slide_embed_rejects_incompatible_patch_model(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px224": (224, {"uni": np.ones((2, 3), np.float32)})},
    )

    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "titan",
            "--patch-model",
            "uni",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 1
    assert "requires patch model 'conch_v1_5'" in result.output
    assert "incompatible" in result.output
    assert recording_slide_embedder["loads"] == 0


def test_cli_slide_embed_repairs_malformed_existing_output(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (128, {"mock": np.ones((2, 3), np.float32)})},
    )
    group = open_grid(path, mode="r+")
    slide_group = group.create_group("slide")
    malformed = slide_group.create_array(
        "mean", shape=(2, 3), chunks=(2, 3), dtype="float32"
    )
    malformed[:] = 1
    malformed.attrs.update(
        {
            "role": "slide_embedding",
            "patch_encoder": "mock",
            "embedding_dim": 3,
        }
    )
    header = dict(group.attrs["raw2features"])
    header["slide_embeddings"] = {
        "mean": {"patch_encoder": "mock", "embedding_dim": 3}
    }
    group.attrs["raw2features"] = header

    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "mean",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recording_slide_embedder["loads"] == 1
    repaired = open_grid(path)["slide"]["mean"]
    assert repaired.shape == (1, 3)
    assert repaired.attrs["role"] == "slide_embedding"


def test_cli_slide_embed_validates_all_encoder_names_before_work(
    tmp_path, recording_slide_embedder
):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (128, {"mock": np.ones((2, 3), np.float32)})},
    )

    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "mean",
            "-s",
            "does_not_exist",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 1
    assert "Unknown slide encoder" in result.output
    assert recording_slide_embedder["loads"] == 0
    assert "slide" not in open_grid(path, mode="r+")


def test_cli_slide_embed_consolidates_once_after_all_writes(
    tmp_path, recording_slide_embedder, monkeypatch
):
    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (128, {"mock": np.ones((2, 3), np.float32)})},
    )
    consolidate = zarr.consolidate_metadata
    calls = []

    def _record_consolidation(store):
        calls.append(store)
        return consolidate(store)

    monkeypatch.setattr(zarr, "consolidate_metadata", _record_consolidation)
    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            path,
            "-s",
            "mean",
            "-s",
            "max",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0, result.output
    assert recording_slide_embedder["loads"] == 2
    assert len(calls) == 1


def test_embed_slide_validates_all_encoder_names_before_work(tmp_path, monkeypatch):
    from raw2features.pipeline.runner import embed_slide

    cfg = RunConfig(
        models=["resnet50"],
        slide_encoders=["mean", "does_not_exist", "max"],
    )
    monkeypatch.setattr(
        "raw2features.pipeline.runner.run_slide",
        lambda *args, **kwargs: pytest.fail("run_slide must not be called"),
    )

    with pytest.raises(ValueError, match="does_not_exist"):
        embed_slide("unused.zarr", str(tmp_path), cfg)


def test_inline_slide_encoder_rerun_is_idempotent(
    tmp_path, recording_slide_embedder
):
    from raw2features.pipeline.runner import _run_slide_encoders
    from raw2features.sinks.zarr_sink import ZarrSink

    path = _write_slide_store(
        tmp_path,
        {"mpp0.5_px64": (128, {"mock": np.ones((2, 3), np.float32)})},
    )
    sink = ZarrSink()
    sink._group = open_grid(path, mode="r+")

    first = _run_slide_encoders(sink, ["mean"], "cpu", ["mock"])
    second = _run_slide_encoders(sink, ["mean"], "cpu", ["mock"])

    assert first == second == {"mean": "slide/mean"}
    assert recording_slide_embedder["loads"] == 1


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_cli_slide_embed_on_existing_zarr(synthetic_ngff, tmp_path):
    # First produce patch embeddings.
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    out = str(tmp_path / "out")
    summary = run_slide(synthetic_ngff, out, cfg, embedders=[MockEmbedder(dim=8)])
    zarr_path = summary["output_uri"].removeprefix("file://")

    # Now run slide-embed on the zarr - no WSI re-read.
    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            zarr_path,
            "-s",
            "mean",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "done" in result.output

    g = open_grid(zarr_path)  # the sole grid
    sv = np.asarray(g["slide"]["mean"][:])
    assert sv.shape == (1, 8)
    assert np.isfinite(sv).all()


@pytest.mark.skipif(not _TORCH, reason="torch not installed")
def test_cli_slide_embed_skips_if_complete(synthetic_ngff, tmp_path):
    cfg = RunConfig(
        models=["mock"],
        no_seg=True,
        target_mpp=0.5,
        patch_px=64,
        device="cpu",
        amp="fp32",
    )
    out = str(tmp_path / "out")
    summary = run_slide(synthetic_ngff, out, cfg, embedders=[MockEmbedder(dim=8)])
    zarr_path = summary["output_uri"].removeprefix("file://")

    # Run once.
    CliRunner().invoke(
        app,
        [
            "slide-embed",
            zarr_path,
            "-s",
            "mean",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )
    # Run again - should skip.
    result = CliRunner().invoke(
        app,
        [
            "slide-embed",
            zarr_path,
            "-s",
            "mean",
            "--patch-model",
            "mock",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0
    assert "skipping" in result.output


def test_slide_encoder_skips_empty_tissue(tmp_path):
    """0 kept patches must skip slide encoding (no NaN mean / zero-size max crash)."""
    from raw2features.pipeline.runner import _run_slide_encoders

    g = zarr.open_group(str(tmp_path / "empty.embeddings.zarr"), mode="w",
                        zarr_format=2)
    feats = g.create_group("features")
    feats.create_array("resnet50", shape=(0, 2048), dtype="float16")
    g.create_array("coords", shape=(0, 2), dtype="int32")
    g.attrs["raw2features"] = {"patching": {"level0_patch": 448}}

    class _Sink:
        pass

    sink = _Sink()
    sink._group = g
    with pytest.warns(UserWarning, match="0 patch features"):
        res = _run_slide_encoders(sink, ["mean"], "cpu", ["resnet50"])
    assert res == {}  # nothing written, no crash
