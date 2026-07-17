"""Load the model registry (``registry.yaml``) and build embedders by name."""

from __future__ import annotations

import importlib.resources
import math
from collections.abc import Mapping
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
            weights_filename=d.get("weights_filename"),
            experimental=bool(d.get("experimental", False)),
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
# Extraction patch size (px) for registered scale-agnostic models that declare none.
DEFAULT_PATCH_PX = 224


def _spec_catalog(specs: Mapping[str, ModelSpec] | None = None) -> dict[str, ModelSpec]:
    """Packaged registry plus any explicitly injected model specifications."""

    catalog = load_registry()
    catalog.update(specs or {})
    return catalog


def _require_known_models(
    models: list[str], catalog: Mapping[str, ModelSpec]
) -> None:
    unknown = sorted(set(models) - set(catalog))
    if unknown:
        available = sorted(catalog)
        raise ValueError(
            f"model(s) {', '.join(unknown)} not in the registry or supplied model "
            f"specifications; available: {available}"
        )


def _positive_mpp(value: float, *, field: str = "mpp") -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and greater than zero")
    return value


def _positive_patch_px(value: int, *, field: str = "patch_px") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer greater than zero")
    integer = int(value)
    if integer != value or integer <= 0:
        raise ValueError(f"{field} must be an integer greater than zero")
    return integer


def resolve_target_mpp(
    models: list[str],
    requested: float | None,
    *,
    specs: Mapping[str, ModelSpec] | None = None,
) -> tuple[float, str]:
    """Resolve the patch-extraction MPP for a run, returning ``(mpp, source)``.

    An explicit ``requested`` MPP always wins (``source="explicit"``). Otherwise
    the models' card-sourced ``recommended_mpp`` decides: a single shared value is
    used (``"auto"`` - the common case, all pathology FMs want 0.5 µm/px = 20×);
    if no requested model declares one (scale-agnostic, e.g. resnet50 / dinov2) it
    falls back to :data:`DEFAULT_TARGET_MPP` (``"auto-default"``). Models that
    *disagree* raise - one run extracts patches at one scale, so the caller must
    pass ``--mpp`` to pick. Programmatic external model names are accepted only
    when their authoritative :class:`ModelSpec` is supplied through ``specs``;
    unknown uninjected names fail instead of silently receiving generic geometry.

    Pure in its inputs so every CLI (embed / embed-many / verify) resolves
    identically and their config hashes line up.
    """
    registry = _spec_catalog(specs)
    _require_known_models(models, registry)
    if requested is not None:
        return _positive_mpp(requested), "explicit"
    recs = {
        name: float(registry[name].recommended_mpp)
        for name in models
        if registry[name].recommended_mpp
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
    chosen -- ``"explicit"`` (a command-line mpp and/or patch-size override),
    ``"config"`` (a per-model config file), ``"recommended"`` (the models' card
    geometry), or ``"default"`` (the scale-agnostic ``DEFAULT_TARGET_MPP`` /
    ``DEFAULT_PATCH_PX`` fallback).
    """

    mpp: float
    patch_px: int
    models: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        _positive_mpp(self.mpp)
        _positive_patch_px(self.patch_px)


def _dedup(names: list[str]) -> tuple[str, ...]:
    """First-seen-order de-duplication (a model can't appear twice in one grid)."""
    seen: dict[str, None] = {}
    for n in names:
        seen.setdefault(n, None)
    return tuple(seen)


def model_geometry(
    name: str, *, specs: Mapping[str, ModelSpec] | None = None
) -> tuple[float, int, str]:
    """The default ``(mpp, patch_px, source)`` one model is extracted at.

    ``recommended_mpp`` (card-sourced) sets the scale; ``recommended_patch_px`` (or
    ``input_size``, via :attr:`ModelSpec.extract_px`) the size. Models with no
    ``recommended_mpp`` are scale-agnostic and fall back to :data:`DEFAULT_TARGET_MPP`.
    Programmatic external models must be provided through ``specs``; an unknown name
    without a specification is rejected rather than silently receiving 1.0/224.
    """
    catalog = _spec_catalog(specs)
    _require_known_models([name], catalog)
    spec = catalog.get(name)
    if spec is None:
        raise AssertionError("known model unexpectedly missing from the catalog")
    if spec.recommended_mpp:
        return (
            _positive_mpp(spec.recommended_mpp, field=f"{name}.recommended_mpp"),
            _positive_patch_px(spec.extract_px, field=f"{name}.extract_px"),
            "recommended",
        )
    return (
        DEFAULT_TARGET_MPP,
        _positive_patch_px(spec.extract_px, field=f"{name}.extract_px"),
        "default",
    )


def resolve_geometry(
    models: list[str],
    requested_mpp: float | None = None,
    requested_patch_px: int | None = None,
    config: list[dict] | None = None,
    *,
    specs: Mapping[str, ModelSpec] | None = None,
) -> list[GeometryGroup]:
    """Group requested models into extraction geometries -- one grid per group.

    Resolution order:

    * ``config`` -- a list of ``{model, mpp?, patch_px?}`` entries: explicit per-model
      geometry, missing values falling back to the model's defaults. Supports the SAME
      model at several geometries (distinct groups -- the MPP-ablation case).
    * both ``requested_mpp`` and ``requested_patch_px`` -- a GLOBAL collapse: every
      model onto one grid at that geometry.
    * a bare ``requested_mpp`` -- override scale while retaining each model's
      recommended extraction size; models with different patch sizes get separate
      grids at the requested scale. A bare ``requested_patch_px`` resolves the shared
      mpp via :func:`resolve_target_mpp` (which raises if the cards disagree).
    * neither -- PER-MODEL geometry: each model at its own geometry ``(recommended_mpp
      or default, recommended_patch_px or input_size)``, grouped by identical geometry
      in request order. Models that disagree no longer raise; they get their own grids.

    ``specs`` supplies authoritative metadata for programmatic external embedders.
    Unknown names without a registry entry or injected spec fail instead of silently
    receiving the generic 1.0/224 geometry. Pure in its arguments so every CLI
    resolves identically (config hashes line up).
    """
    if config is not None:
        return _groups_from_config(config, specs=specs)
    catalog = _spec_catalog(specs)
    _require_known_models(models, catalog)
    if requested_mpp is not None and requested_patch_px is None:
        mpp = _positive_mpp(requested_mpp)
        groups: dict[int, list[str]] = {}
        for model in models:
            _default_mpp, patch_px, _source = model_geometry(model, specs=specs)
            groups.setdefault(patch_px, []).append(model)
        return [
            GeometryGroup(mpp, patch_px, _dedup(group_models), "explicit")
            for patch_px, group_models in groups.items()
        ]
    if requested_mpp is not None or requested_patch_px is not None:
        mpp = (
            _positive_mpp(requested_mpp)
            if requested_mpp is not None
            else resolve_target_mpp(models, None, specs=specs)[0]
        )
        px = (
            _positive_patch_px(requested_patch_px)
            if requested_patch_px is not None
            else DEFAULT_PATCH_PX
        )
        return [GeometryGroup(mpp, px, _dedup(models), "explicit")]
    groups: dict[tuple[float, int], list[str]] = {}
    src: dict[tuple[float, int], str] = {}
    for m in models:
        mpp, px, s = model_geometry(m, specs=specs)
        groups.setdefault((mpp, px), []).append(m)
        src.setdefault((mpp, px), s)
    return [
        GeometryGroup(mpp, px, _dedup(ms), src[(mpp, px)])
        for (mpp, px), ms in groups.items()
    ]


def _groups_from_config(
    extractions: list[dict], *, specs: Mapping[str, ModelSpec] | None = None
) -> list[GeometryGroup]:
    """Build geometry groups from a parsed config: ``[{model, mpp?, patch_px?}, ...]``.

    Each entry resolves mpp/patch_px (explicit, else the model's defaults); entries with
    the same ``(mpp, patch_px)`` merge into one grid. A model may appear in several
    groups (different geometries).
    """
    groups: dict[tuple[float, int], list[str]] = {}
    for e in extractions:
        name = e["model"]
        d_mpp, d_px, _ = model_geometry(name, specs=specs)
        mpp = _positive_mpp(e.get("mpp", d_mpp))
        px = _positive_patch_px(e.get("patch_px", d_px))
        groups.setdefault((mpp, px), []).append(name)
    return [
        GeometryGroup(mpp, px, _dedup(ms), "config")
        for (mpp, px), ms in groups.items()
    ]


def build_embedder(name: str) -> Embedder:
    """Build a packaged-registry model with its family implementation plugin.

    An ``embedders`` entry point supplies a loader family; it does not by itself add
    *name* to the CLI model catalogue. Programmatic callers may instead pass external
    :class:`Embedder` instances through ``embed_slide(..., embedders=[...])``.
    """
    from raw2features.core import plugins

    spec = get_spec(name)
    cls = plugins.get("embedders", spec.family)
    return cls(spec)
