"""Load the model registry (``registry.yaml``) and build embedders by name."""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass

import yaml

from .base import Embedder, ModelSpec


def load_registry() -> dict[str, ModelSpec]:
    """Parse ``registry.yaml`` (packaged with this module) into ModelSpecs."""
    text = (
        importlib.resources.files("raw2features.embedders")
        .joinpath("registry.yaml")
        .read_text()
    )
    raw = yaml.safe_load(text) or {}
    specs: dict[str, ModelSpec] = {}
    for name, d in raw.items():
        if name == "slide_encoders":  # reserved section, parsed by slide registry
            continue
        specs[name] = ModelSpec(
            name=name,
            family=d["family"],
            source=d["source"],
            embedding_dim=int(d["embedding_dim"]),
            input_size=int(d["input_size"]),
            pooling=d["pooling"],
            mean=tuple(float(x) for x in d["mean"]),
            std=tuple(float(x) for x in d["std"]),
            transform_source_url=d["transform_source_url"],
            license=d["license"],
            gated=bool(d["gated"]),
            reg_tokens=int(d.get("reg_tokens", 0)),
            transform_source=d.get("transform_source", "registry"),
            inference_amp=d.get("inference_amp", "fp32"),
            interpolation=d.get("interpolation", "bilinear"),
            recommended_mpp=d.get("recommended_mpp"),
            recommended_patch_px=d.get("recommended_patch_px"),
            modality=d.get("modality", "brightfield"),
            timm_kwargs=dict(d.get("timm_kwargs") or {}),
            checkpoint=d.get("checkpoint"),
            weights_sha256=d.get("weights_sha256"),
            weights_revision=d.get("weights_revision"),
            notes=d.get("notes", ""),
            doi=d.get("doi"),
        )
    return specs


def get_spec(name: str) -> ModelSpec:
    registry = load_registry()
    if name not in registry:
        raise KeyError(f"model {name!r} not in registry; available: {sorted(registry)}")
    return registry[name]


# Scale-agnostic fallback when no requested model declares a recommended MPP.
DEFAULT_TARGET_MPP = 1.0
# Extraction patch size (px) for models that declare none (scale-agnostic / unknown).
DEFAULT_PATCH_PX = 224


def resolve_target_mpp(
    models: list[str], requested: float | None
) -> tuple[float, str]:
    """Resolve the patch-extraction MPP for a run, returning ``(mpp, source)``.

    An explicit ``requested`` MPP always wins (``source="explicit"``). Otherwise
    the models' card-sourced ``recommended_mpp`` decides: a single shared value is
    used (``"auto"`` - the common case, all pathology FMs want 0.5 µm/px = 20×);
    if no requested model declares one (scale-agnostic, e.g. resnet50 / dinov2) it
    falls back to :data:`DEFAULT_TARGET_MPP` (``"auto-default"``). Models that
    *disagree* raise - one run extracts patches at one scale, so the caller must
    pass ``--mpp`` to pick. Unknown names (e.g. test mocks) contribute nothing
    rather than erroring, so they resolve to the default.

    Pure in ``(models, requested)`` so every CLI (embed / embed-many / verify)
    resolves identically and their config hashes line up.
    """
    if requested is not None:
        return float(requested), "explicit"
    registry = load_registry()
    recs = {
        name: float(registry[name].recommended_mpp)
        for name in models
        if name in registry and registry[name].recommended_mpp
    }
    distinct = sorted(set(recs.values()))
    if len(distinct) == 1:
        return distinct[0], "auto"
    if not distinct:
        return DEFAULT_TARGET_MPP, "auto-default"
    detail = ", ".join(f"{n}={v}" for n, v in sorted(recs.items()))
    raise ValueError(
        f"requested models recommend different MPPs ({detail}); "
        "pass --mpp explicitly to extract at one scale."
    )


@dataclass(frozen=True)
class GeometryGroup:
    """One ``(mpp, patch_px)`` extraction geometry and the models that share it.

    A "grid" in the store: every model in ``models`` is decoded once at this geometry
    (decode-once fan-out) and writes ``features/<model>`` against the same coords. The
    field of view is ``patch_px * mpp`` µm. ``source`` records how the geometry was
    chosen -- ``"explicit"`` (a global --mpp/--patch-size collapse), ``"config"`` (a
    per-model config file), ``"recommended"`` (the models' card geometry), or
    ``"default"`` (the scale-agnostic ``DEFAULT_TARGET_MPP`` / ``DEFAULT_PATCH_PX``
    fallback).
    """

    mpp: float
    patch_px: int
    models: tuple[str, ...]
    source: str


def _dedup(names: list[str]) -> tuple[str, ...]:
    """First-seen-order de-duplication (a model can't appear twice in one grid)."""
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return tuple(seen)


def model_geometry(name: str) -> tuple[float, int, str]:
    """The default ``(mpp, patch_px, source)`` one model is extracted at.

    ``recommended_mpp`` (card-sourced) sets the scale; ``recommended_patch_px`` (or
    ``input_size``, via :attr:`ModelSpec.extract_px`) the size. Models with no
    ``recommended_mpp`` are scale-agnostic and fall back to :data:`DEFAULT_TARGET_MPP`.
    Unknown names (e.g. test mocks) resolve to the scale-agnostic default rather than
    raising, mirroring :func:`resolve_target_mpp`.
    """
    spec = load_registry().get(name)
    if spec is None:
        return DEFAULT_TARGET_MPP, DEFAULT_PATCH_PX, "default"
    if spec.recommended_mpp:
        return float(spec.recommended_mpp), spec.extract_px, "recommended"
    return DEFAULT_TARGET_MPP, spec.extract_px, "default"


def resolve_geometry(
    models: list[str],
    requested_mpp: float | None = None,
    requested_patch_px: int | None = None,
    config: list[dict] | None = None,
) -> list[GeometryGroup]:
    """Group requested models into extraction geometries -- one grid per group.

    Resolution order:

    * ``config`` -- a list of ``{model, mpp?, patch_px?}`` entries: explicit per-model
      geometry, missing values falling back to the model's defaults. Supports the SAME
      model at several geometries (distinct groups -- the MPP-ablation case).
    * an explicit ``requested_mpp`` and/or ``requested_patch_px`` -- a GLOBAL collapse:
      every model onto one grid at that geometry (today's behaviour, and the "all at
      one scale" switch). A bare ``requested_patch_px`` resolves the shared mpp via
      :func:`resolve_target_mpp` (which raises if the cards disagree).
    * neither -- PER-MODEL geometry: each model at its own geometry ``(recommended_mpp
      or default, recommended_patch_px or input_size)``, grouped by identical geometry
      in request order. Models that disagree no longer raise; they get their own grids.

    Pure in its arguments so every CLI resolves identically (config hashes line up).
    """
    if config is not None:
        return _groups_from_config(config)
    if requested_mpp is not None or requested_patch_px is not None:
        mpp = (
            float(requested_mpp)
            if requested_mpp is not None
            else resolve_target_mpp(models, None)[0]
        )
        px = (
            int(requested_patch_px)
            if requested_patch_px is not None
            else DEFAULT_PATCH_PX
        )
        return [GeometryGroup(mpp, px, _dedup(models), "explicit")]
    groups: dict[tuple[float, int], list[str]] = {}
    src: dict[tuple[float, int], str] = {}
    for m in models:
        mpp, px, s = model_geometry(m)
        groups.setdefault((mpp, px), []).append(m)
        src.setdefault((mpp, px), s)
    return [
        GeometryGroup(mpp, px, _dedup(ms), src[(mpp, px)])
        for (mpp, px), ms in groups.items()
    ]


def _groups_from_config(extractions: list[dict]) -> list[GeometryGroup]:
    """Build geometry groups from a parsed config: ``[{model, mpp?, patch_px?}, ...]``.

    Each entry resolves mpp/patch_px (explicit, else the model's defaults); entries with
    the same ``(mpp, patch_px)`` merge into one grid. A model may appear in several
    groups (different geometries).
    """
    groups: dict[tuple[float, int], list[str]] = {}
    for e in extractions:
        name = e["model"]
        d_mpp, d_px, _ = model_geometry(name)
        mpp = float(e.get("mpp", d_mpp))
        px = int(e.get("patch_px", d_px))
        groups.setdefault((mpp, px), []).append(name)
    return [
        GeometryGroup(mpp, px, _dedup(ms), "config")
        for (mpp, px), ms in groups.items()
    ]


def build_embedder(name: str) -> Embedder:
    """Resolve a model name -> its family Embedder class -> a ready instance."""
    from raw2features.core import plugins

    spec = get_spec(name)
    cls = plugins.get("embedders", spec.family)
    return cls(spec)
