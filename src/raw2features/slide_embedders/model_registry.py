"""Slide-encoder registry and builder.

Built-in slide encoders live in the ``slide_encoders`` section of
``embedders/registry.yaml`` (the same file as patch encoders, to keep all
model provenance in one place). Third-party slide encoders can be added
via the ``raw2features.slide_embedders`` entry-point group.
"""

from __future__ import annotations

import importlib.resources

import yaml

from .base import SlideEmbedder, SlideModelSpec

_REGISTRY: dict[str, SlideModelSpec] | None = None


def load_slide_registry() -> dict[str, SlideModelSpec]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    data = importlib.resources.files("raw2features.embedders").joinpath("registry.yaml")
    raw = yaml.safe_load(data.read_text())
    _REGISTRY = {
        name: SlideModelSpec(name=name, **cfg)
        for name, cfg in raw.get("slide_encoders", {}).items()
    }
    return _REGISTRY


def get_slide_spec(name: str) -> SlideModelSpec:
    reg = load_slide_registry()
    if name not in reg:
        raise KeyError(
            f"Unknown slide encoder {name!r}. "
            f"Available: {sorted(reg)}. "
            f"Check 'raw2features list slide_embedders'."
        )
    return reg[name]


def build_slide_embedder(name: str) -> SlideEmbedder:
    from raw2features.core.plugins import get

    spec = get_slide_spec(name)
    cls = get("slide_embedders", name)
    emb = cls()
    # Overwrite spec from registry so provenance is always registry-sourced.
    emb.spec = spec
    return emb


def resolve_patch_encoder(
    slide_model: str,
    available_patch_models: list[str],
) -> str:
    """Return the patch model name to feed into ``slide_model``.

    The registry's ``patch_encoder`` field names the required model. A
    value of ``"any"`` (the pooling baselines) means the encoder is
    model-agnostic: it accepts the single available patch model, or - if
    several are present - requires the caller to disambiguate with
    ``--patch-model``. A specific value (e.g. ``"conch_v1_5"`` for TITAN)
    must be present in the embeddings zarr.
    """
    spec = get_slide_spec(slide_model)
    required = spec.patch_encoder

    if not available_patch_models:
        raise ValueError(
            f"No patch features in the embeddings zarr; run "
            f"'raw2features embed' first before slide encoding with "
            f"{slide_model!r}."
        )

    if required == "any":
        if len(available_patch_models) == 1:
            return available_patch_models[0]
        raise ValueError(
            f"Slide encoder {slide_model!r} is model-agnostic but the zarr "
            f"has multiple patch models {available_patch_models}; pass "
            f"--patch-model to choose one."
        )

    if required not in available_patch_models:
        raise ValueError(
            f"Slide encoder {slide_model!r} requires patch features from "
            f"{required!r}, but the embeddings zarr contains only: "
            f"{available_patch_models}. "
            f"Re-run 'raw2features embed ... -f {required}' first."
        )
    return required
