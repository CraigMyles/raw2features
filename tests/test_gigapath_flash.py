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
    assert constructor["tile_size"] == 256
    assert constructor["coords_frame"] == "stored_level0_xy"
    assert constructor["coords_transform"] == "none"
    assert "tile_size_source" not in constructor


def test_gigapath_slide_forward_uses_authors_fixed_coordinate_frame():
    import raw2features.slide_embedders.gigapath_slide as gigapath_slide

    captured: dict[str, torch.Tensor] = {}

    class RecordingLongNet:
        tile_size = 256

        def __call__(self, features, coords, all_layer_embed=False):
            captured["features"] = features.detach().cpu().clone()
            captured["coords"] = coords.detach().cpu().clone()
            captured["all_layer_embed"] = all_layer_embed
            return [torch.ones((1, 384), dtype=torch.float32)]

    features = np.arange(4 * 384, dtype=np.float32).reshape(4, 384)
    coords = np.asarray(
        [[0, 0], [512, 0], [0, 512], [512, 512]], dtype=np.float32
    )
    model = RecordingLongNet()
    embedder = gigapath_slide.GigapathSlideEmbedder()
    embedder.spec = get_slide_spec("gigapath_flash_slide")
    embedder._model = model
    embedder._device = "cpu"

    output = embedder.encode(features, coords, patch_size_lv0=512)

    assert model.tile_size == 256
    assert captured["features"].shape == (1, 4, 384)
    assert captured["coords"].shape == (1, 4, 2)
    assert captured["all_layer_embed"] is True
    np.testing.assert_array_equal(captured["features"][0].numpy(), features)
    np.testing.assert_array_equal(captured["coords"][0].numpy(), coords)
    assert output.shape == (384,)


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
