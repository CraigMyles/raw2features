"""GigaPath-Flash registry and loader contracts, without downloading weights."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from raw2features.embedders.fingerprint import (  # noqa: E402
    patch_output_fingerprint,
    slide_output_fingerprint,
)
from raw2features.embedders.model_registry import build_embedder, get_spec  # noqa: E402
from raw2features.embedders.timm_embedder import TimmEmbedder  # noqa: E402
from raw2features.slide_embedders.model_registry import (  # noqa: E402
    build_slide_embedder,
    get_slide_spec,
)


def test_gigapath_flash_registry_pair_is_matched_and_pinned():
    # Register the source-tree implementation explicitly. Test environments may
    # still expose entry-point metadata from the previously installed release.
    import raw2features.slide_embedders.gigapath_slide  # noqa: F401

    tile = get_spec("gigapath_flash")
    slide = get_slide_spec("gigapath_flash_slide")

    assert isinstance(build_embedder("gigapath_flash"), TimmEmbedder)
    assert tile.source == "gigapath_tile_enc_dinov2s"
    assert tile.embedding_dim == slide.patch_dim == slide.embedding_dim == 384
    assert slide.patch_encoder == tile.name
    assert slide.architecture == "gigapath_slide_enc12l384d"
    assert tile.weights_revision == slide.weights_revision
    assert tile.weights_filename == "pytorch_model.bin"
    assert slide.weights_filename == "slide_encoder.pth"
    assert tile.checkpoint["strict"] is True
    assert tile.registration_modules == ("gigapath.tile_encoder",)

    slide_embedder = build_slide_embedder("gigapath_flash_slide")
    assert slide_embedder.name == "gigapath_flash_slide"
    assert slide_embedder.spec is slide

    fingerprint = slide_output_fingerprint(
        slide,
        patch_model=tile.name,
        patch_output_fingerprint=patch_output_fingerprint(tile, "fp16"),
        patch_dim=tile.embedding_dim,
        resolved_amp="fp16",
    )
    constructor = fingerprint["payload"]["loader"]["constructor"]
    assert constructor["constructor_tile_size"] == 256
    assert constructor["slide_ngrids"] == 1000
    assert (
        constructor["runtime_position_divisor_source"] == "patching.level0_patch"
    )
    assert constructor["coords_frame"] == "stored_level0_xy"
    assert constructor["coords_transform"] == "none"
    assert constructor["position_mapping"] == (
        "floor(x / divisor) * slide_ngrids + floor(y / divisor) + 1"
    )
    assert "tile_size" not in constructor


@pytest.mark.parametrize("patch_size_lv0", [256, 512])
def test_gigapath_slide_forward_normalizes_level0_coordinate_frame(
    patch_size_lv0,
):
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide

    captured: dict[str, torch.Tensor] = {}

    class RecordingLongNet:
        tile_size = 0
        slide_ngrids = 1000

        def __call__(self, features, coords, all_layer_embed=False):
            captured["features"] = features.detach().cpu().clone()
            captured["coords"] = coords.detach().cpu().clone()
            cells = torch.floor(coords / self.tile_size)
            captured["positions"] = (
                cells[..., 0] * self.slide_ngrids + cells[..., 1] + 1
            ).long()
            captured["all_layer_embed"] = all_layer_embed
            return [torch.ones((1, 384), dtype=torch.float32)]

    features = np.arange(8 * 384, dtype=np.float32).reshape(8, 384)
    coords = np.asarray(
        [
            [0, 0],
            [patch_size_lv0 // 2, 0],
            [patch_size_lv0 - 1, 0],
            [patch_size_lv0, 0],
            [patch_size_lv0 + 1, 0],
            [2 * patch_size_lv0, 0],
            [0, patch_size_lv0],
            [patch_size_lv0, patch_size_lv0],
        ],
        dtype=np.float32,
    )
    model = RecordingLongNet()
    embedder = gigapath_slide.GigapathSlideEmbedder()
    embedder.spec = get_slide_spec("gigapath_flash_slide")
    embedder._model = model
    embedder._device = "cpu"

    output = embedder.encode(features, coords, patch_size_lv0=patch_size_lv0)

    assert model.tile_size == patch_size_lv0
    assert captured["features"].shape == (1, 8, 384)
    assert captured["coords"].shape == (1, 8, 2)
    assert captured["positions"].tolist() == [
        [1, 1, 1, 1001, 1001, 2001, 2, 1002]
    ]
    assert captured["all_layer_embed"] is True
    np.testing.assert_array_equal(captured["features"][0].numpy(), features)
    np.testing.assert_array_equal(captured["coords"][0].numpy(), coords)
    assert output.shape == (384,)


@pytest.mark.parametrize("patch_size_lv0", [None, 0, -256, True, 256.5])
def test_gigapath_slide_requires_positive_integer_level0_tile_extent(
    patch_size_lv0,
):
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide

    embedder = gigapath_slide.GigapathSlideEmbedder()
    embedder.spec = get_slide_spec("gigapath_flash_slide")
    embedder._model = type("LongNet", (), {"slide_ngrids": 1000})()
    embedder._device = "cpu"

    with pytest.raises(ValueError, match="patch_size_lv0|positive integer"):
        embedder.encode(
            np.zeros((1, 384), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            patch_size_lv0=patch_size_lv0,
        )


@pytest.mark.parametrize(
    "coords",
    [
        [[-1, 0]],
        [[0, -1]],
        [[256_000, 0]],
        [[0, 256_000]],
        [[np.nan, 0]],
    ],
)
def test_gigapath_slide_rejects_coords_outside_positional_grid(coords):
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide

    embedder = gigapath_slide.GigapathSlideEmbedder()
    embedder.spec = get_slide_spec("gigapath_flash_slide")
    embedder._model = type("LongNet", (), {"slide_ngrids": 1000})()
    embedder._device = "cpu"

    with pytest.raises(ValueError, match="positional grid|finite"):
        embedder.encode(
            np.zeros((1, 384), dtype=np.float32),
            np.asarray(coords, dtype=np.float32),
            patch_size_lv0=256,
        )


def test_timm_registration_module_is_imported_before_checkpoint_construction(
    monkeypatch,
):
    events: list[object] = []
    real_import = importlib.import_module

    def import_module(name):
        if name == "gigapath.tile_encoder":
            events.append(("import", name))
            return ModuleType(name)
        return real_import(name)

    class Model:
        def eval(self):
            return self

        def to(self, device):
            events.append(("to", device))
            return self

    timm = ModuleType("timm")

    def create_model(name, **kwargs):
        events.append(("create", name, kwargs))
        return Model()

    timm.create_model = create_model
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setattr(
        "raw2features.embedders.timm_embedder.importlib.import_module",
        import_module,
    )
    monkeypatch.setattr(
        TimmEmbedder,
        "_load_checkpoint",
        lambda self, model, checkpoint: events.append(("load", checkpoint)),
    )

    embedder = build_embedder("gigapath_flash")
    embedder.load(device="cpu", dtype=torch.float32)

    assert events[:3] == [
        ("import", "gigapath.tile_encoder"),
        (
            "create",
            "gigapath_tile_enc_dinov2s",
            {"pretrained": False, "num_classes": 0},
        ),
        ("load", get_spec("gigapath_flash").checkpoint),
    ]
