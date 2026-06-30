"""Plugin discovery for the swappable seams.

Each seam (readers, segmenters, patchers, embedders, sinks, slide_embedders)
is resolved by name from two sources, in order:

1. An in-process registry populated by the ``@register`` decorator (used by
   built-in implementations and tests).
2. Python entry-points in the group ``raw2features.<seam>`` (used by third-party
   packages - ``pip install raw2features-myreader`` and it appears here with no
   forking).

Entry-points are imported lazily and guarded: a plugin whose optional dependency
is missing simply does not appear, rather than breaking discovery.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

# group name -> {plugin name -> object/class}
_REGISTRY: dict[str, dict[str, Any]] = {}

SEAMS = ("readers", "segmenters", "patchers", "embedders", "sinks", "slide_embedders")


def register(seam: str, name: str):
    """Decorator registering ``cls`` as ``name`` within ``seam``."""
    if seam not in SEAMS:
        raise ValueError(f"unknown seam {seam!r}; expected one of {SEAMS}")

    def decorator(obj: Any) -> Any:
        _REGISTRY.setdefault(seam, {})[name] = obj
        return obj

    return decorator


def _entry_point_objects(seam: str) -> dict[str, Any]:
    """Load entry-point plugins for ``seam``, skipping any that fail to import."""
    found: dict[str, Any] = {}
    try:
        eps = entry_points(group=f"raw2features.{seam}")
    except TypeError:  # pragma: no cover - very old importlib API
        eps = entry_points().get(f"raw2features.{seam}", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception:  # noqa: BLE001 - a missing optional dep must not break discovery
            continue
    return found


def available(seam: str) -> list[str]:
    """Sorted names available for ``seam`` (in-process + entry-points)."""
    if seam not in SEAMS:
        raise ValueError(f"unknown seam {seam!r}; expected one of {SEAMS}")
    names = set(_REGISTRY.get(seam, {})) | set(_entry_point_objects(seam))
    return sorted(names)


def get(seam: str, name: str) -> Any:
    """Resolve ``name`` within ``seam`` (in-process takes precedence)."""
    if seam not in SEAMS:
        raise ValueError(f"unknown seam {seam!r}; expected one of {SEAMS}")
    in_process = _REGISTRY.get(seam, {})
    if name in in_process:
        return in_process[name]
    obj = _entry_point_objects(seam).get(name)
    if obj is None:
        choices = available(seam)
        raise KeyError(f"{seam} plugin {name!r} not found; available: {choices}")
    return obj
