"""Registry contract: every model entry is complete and internally consistent.

This is fast and weight-free - it catches a malformed ``registry.yaml`` edit
(missing field, zero std, bad URL) before it reaches a GPU job. The matching
forward-pass dimension check (output dim == declared embedding_dim) lives in
``test_embedder_forward.py`` and is marked slow because it downloads weights.
"""

from __future__ import annotations

from raw2features.core import plugins
from raw2features.embedders.model_registry import load_registry
from raw2features.slide_embedders.model_registry import load_slide_registry


def test_every_patch_spec_is_complete_and_consistent():
    reg = load_registry()
    assert reg, "patch registry is empty"
    for name, spec in reg.items():
        assert spec.name == name
        assert spec.embedding_dim > 0, f"{name}: embedding_dim must be positive"
        assert spec.input_size > 0, f"{name}: input_size must be positive"
        assert len(spec.mean) == 3 and len(spec.std) == 3, (
            f"{name}: mean/std must be RGB"
        )
        assert all(s > 0 for s in spec.std), f"{name}: zero std would divide by zero"
        assert spec.pooling in {"cls", "pooled", "cls_concat_meanpatch"}, (
            f"{name}: bad pooling {spec.pooling!r}"
        )
        assert spec.transform_source_url.startswith("http"), f"{name}: needs source URL"
        assert spec.license, f"{name}: license must be recorded"
        assert spec.recommended_mpp is None or spec.recommended_mpp > 0, (
            f"{name}: recommended_mpp must be positive or None"
        )
        assert spec.recommended_patch_px is None or spec.recommended_patch_px > 0, (
            f"{name}: recommended_patch_px must be positive or None"
        )
        assert spec.crop_pct is None or 0 < spec.crop_pct <= 1, (
            f"{name}: crop_pct must be in (0, 1] or None"
        )
        assert spec.crop_mode in ({None} if spec.crop_pct is None else {"center"}), (
            f"{name}: crop_mode must be 'center' exactly when crop_pct is set"
        )
        assert all(spec.registration_modules), (
            f"{name}: registration_modules must not contain blank module names"
        )
        # extract_px is always a usable patch size (falls back to input_size).
        assert spec.extract_px > 0, f"{name}: extract_px must be positive"
        # FAIR findability: a resolvable DOI, unless the model is an open-weights
        # release with no paper (doi may be None then).
        assert spec.doi is None or spec.doi.startswith("10."), (
            f"{name}: doi must be a bare resolvable DOI (got {spec.doi!r})"
        )
        # Weights are pinned: a sha256 + the HF commit they were taken at. torchvision
        # pins by its weight enum (IMAGENET1K_V2) and verifies its own hash.
        if spec.family == "torchvision":
            assert spec.weights_revision, f"{name}: needs a weight revision/enum"
        else:
            assert spec.weights_sha256 and len(spec.weights_sha256) == 64, (
                f"{name}: needs a 64-hex weights_sha256"
            )
            assert spec.weights_revision, f"{name}: needs a pinned HF weights_revision"
        assert spec.weights_filename, f"{name}: needs the exact weights_filename"
        # family must resolve to a registered embedder plugin
        plugins.get("embedders", spec.family)


def test_every_slide_spec_is_complete():
    reg = load_slide_registry()
    assert reg, "slide registry is empty"
    for name, spec in reg.items():
        assert spec.name == name
        assert spec.license, f"{name}: license must be recorded"
        assert spec.transform_source_url.startswith("http"), f"{name}: needs source URL"
        # pooling baselines carry -1 (resolved to patch_dim at runtime); real
        # encoders declare a concrete positive dimension.
        assert spec.embedding_dim == -1 or spec.embedding_dim > 0
        # Weight-bearing slide encoders (not the weight-free pooling baselines) carry a
        # FAIR DOI; pooling baselines legitimately have none.
        if spec.family != "pool":
            assert spec.doi and spec.doi.startswith("10."), f"{name}: needs a DOI"
            assert spec.weights_filename, f"{name}: needs the exact weights_filename"


def test_optional_fields_round_trip_through_loader():
    """The loader cherry-picks fields; guard that optional ones aren't dropped.

    (Regression: ``checkpoint`` was once omitted, silently loading the base arch's
    pretrained weights instead of the model's checkpoint.)
    """
    from raw2features.embedders.model_registry import get_spec

    gpfm = get_spec("gpfm")
    assert gpfm.checkpoint and gpfm.checkpoint["repo"] == "majiabo/GPFM", (
        "gpfm.checkpoint dropped -> it would load plain DINOv2, not GPFM"
    )
    assert get_spec("uni").recommended_mpp == 0.5
    assert get_spec("resnet50").recommended_mpp is None
    # recommended_patch_px is set where extraction px differs from the fixed model
    # input (for example CONCH v1.5 and the centre-cropped GigaPath tiles).
    assert get_spec("conch_v1_5").recommended_patch_px == 512
    assert get_spec("conch_v1_5").extract_px == 512
    assert get_spec("gigapath").recommended_patch_px == 256
    assert get_spec("gigapath").crop_pct == 0.875
    assert get_spec("gigapath_flash").registration_modules == (
        "gigapath.tile_encoder",
    )
    assert get_spec("gigapath_flash").checkpoint == {
        "repo": "prov-gigapath/prov-gigapath-flash",
        "filename": "pytorch_model.bin",
        "strict": True,
    }
    assert get_spec("hipt").checkpoint.get("repo") is None
    assert get_spec("hipt").checkpoint["url"].startswith(
        "https://media.githubusercontent.com/media/mahmoodlab/HIPT/"
    )
    assert get_spec("hipt").mean == (0.5, 0.5, 0.5)
    assert get_spec("hipt").std == (0.5, 0.5, 0.5)
    assert "Commons Clause" in get_spec("hipt").license
    assert get_spec("gigapath").inference_amp == "fp16"
    assert get_spec("gigapath_flash").inference_amp == "fp16"
    assert get_spec("uni").recommended_patch_px is None
    assert get_spec("uni").extract_px == get_spec("uni").input_size == 224
    # doi is cherry-picked by the loader too -> guard it survives.
    assert get_spec("uni").doi == "10.1038/s41591-024-02857-3"
    assert get_spec("h_optimus_0").doi is None  # open-weights release, no paper
    assert get_spec("seal_conch").experimental is True
    assert get_spec("seal_univ2").experimental is True
    assert get_spec("uni").experimental is False
