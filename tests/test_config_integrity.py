"""Guards against silent config-hash drift (the skip-if-complete correctness key).

- content_hash is pinned to known values so a refactor can't change the bytes.
- _CONTENT_FIELDS/_RUNTIME_FIELDS must partition every RunConfig field, so a new
  field has to be classified consciously (not silently ignored by the hash).
- embed and verify must agree on every shared content-affecting option default,
  or the SLURM array's verify-based skip would compute a different hash to embed.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

from raw2features.cli.embed import embed
from raw2features.cli.embed_many import embed_many
from raw2features.cli.verify import verify
from raw2features.pipeline.runner import RunConfig


def test_content_hash_is_pinned():
    # If these change, skip-if-complete silently invalidates every prior receipt.
    # Changing any hashed field changes these values, so update the expected hashes
    # deliberately.
    assert RunConfig(models=["resnet50"]).content_hash() == "a8c12b66d8558e1b"
    assert (
        RunConfig(
            models=["uni", "resnet50"], no_seg=True, target_mpp=1.0
        ).content_hash()
        == "9be3d1202bcfe08d"
    )
    assert (
        RunConfig(
            models=["dinov2"],
            segmenter="otsu",
            target_mpp=0.5,
            patch_px=256,
            step_px=128,
            tissue_threshold=0.2,
            features_dtype="float32",
            snap_to_level=True,
            mpp_tolerance=0.0,
            allow_upsample=True,
            amp="fp16",
        ).content_hash()
        == "7af8f129093bb671"
    )


def test_content_hash_is_model_order_independent():
    a = RunConfig(models=["uni", "resnet50"]).content_hash()
    b = RunConfig(models=["resnet50", "uni"]).content_hash()
    assert a == b


def test_grid_hash_is_model_independent_but_geometry_sensitive():
    # grid_hash identifies the patch geometry, so it must NOT depend on the model
    # set (that is what lets a new model be appended to an existing store) but MUST
    # change when any geometry field changes.
    base = RunConfig(models=["resnet50"]).grid_hash()
    assert base == RunConfig(models=["resnet50", "uni"]).grid_hash()
    assert base == RunConfig(models=["dinov2"]).grid_hash()
    assert base != RunConfig(models=["resnet50"], target_mpp=0.5).grid_hash()
    assert base != RunConfig(models=["resnet50"], patch_px=256).grid_hash()
    # and it is distinct from the full content hash (which includes the models)
    assert base != RunConfig(models=["resnet50"]).content_hash()


def test_amp_auto_is_default_and_resolves_per_model():
    from raw2features.embedders.model_registry import get_spec
    from raw2features.pipeline.runner import _resolve_amp

    cfg = RunConfig(models=["uni"])
    assert cfg.amp == "auto"  # follow each provider's card by default
    # models without an inference_amp field default to fp32 (uni, resnet50) ...
    assert _resolve_amp(cfg, get_spec("uni")) == "fp32"
    assert _resolve_amp(cfg, get_spec("resnet50")) == "fp32"
    # ... while models whose card recommends fp16 resolve to fp16 under auto
    # (exercises the non-trivial per-model branch, not just the fp32 default).
    assert _resolve_amp(cfg, get_spec("virchow2")) == "fp16"
    assert _resolve_amp(cfg, get_spec("h_optimus_0")) == "fp16"
    assert _resolve_amp(cfg, get_spec("h0_mini")) == "fp16"
    # an explicit --amp overrides the per-model card precision
    assert (
        _resolve_amp(RunConfig(models=["uni"], amp="bf16"), get_spec("uni")) == "bf16"
    )


def test_runconfig_fields_are_partitioned():
    content = set(RunConfig._CONTENT_FIELDS)
    runtime = set(RunConfig._RUNTIME_FIELDS)
    all_fields = {f.name for f in fields(RunConfig)}
    assert content.isdisjoint(runtime)
    # every field must be classified as either content-affecting or runtime-only
    assert content | runtime == all_fields, (
        f"unclassified RunConfig fields: {all_fields - content - runtime}"
    )


def _opt_default(func, name):
    opt = inspect.signature(func).parameters[name].default
    return getattr(opt, "default", opt)  # unwrap typer OptionInfo


def test_embed_and_verify_share_content_option_defaults():
    e = set(inspect.signature(embed).parameters)
    v = set(inspect.signature(verify).parameters)
    # Exclude paths: verify's optional --out-dir binds the receipt target, while embed
    # requires it positionally; neither it nor the receipt path feeds the content hash.
    shared = (e & v) - {"slide", "out_dir", "receipts_dir"}
    assert shared, "expected embed/verify to share content options"
    for name in sorted(shared):
        assert _opt_default(embed, name) == _opt_default(verify, name), (
            f"default drift for --{name}: embed={_opt_default(embed, name)!r} "
            f"verify={_opt_default(verify, name)!r}"
        )


def test_embed_and_embed_many_share_content_option_defaults():
    # embed-many (the warm worker) must hash identically to embed for the same
    # flags, or a cohort run's receipts wouldn't be recognised by a later resume.
    e = set(inspect.signature(embed).parameters)
    m = set(inspect.signature(embed_many).parameters)
    # content-affecting options both expose (exclude paths / non-content knobs).
    content = {
        "feature_extractor",
        "mpp",
        "patch_size",
        "step",
        "reader",
        "segmenter",
        "no_seg",
        "tissue_threshold",
        "features_dtype",
        "amp",
        "snap_to_level",
        "mpp_tolerance",
        "allow_upsample",
    }
    shared = (e & m) & content
    assert shared, "expected embed/embed-many to share content options"
    for name in sorted(shared):
        assert _opt_default(embed, name) == _opt_default(embed_many, name), (
            f"default drift for --{name}: embed={_opt_default(embed, name)!r} "
            f"embed_many={_opt_default(embed_many, name)!r}"
        )
