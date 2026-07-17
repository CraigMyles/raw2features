"""Loaders for the two declarative job inputs: the extraction plan and the manifest.

* ``load_extractions`` reads a YAML/JSON **extraction plan** -- an ordered
  ``extractions:`` list of ``{model, mpp?, patch_px?}`` entries. The per-model geometry
  plan: the same model may appear several times (the MPP-ablation case), and an omitted
  ``mpp``/``patch_px`` falls back to the model's registry default. Passed through as
  ``geometry_config`` to :func:`raw2features.pipeline.runner.embed_slide`.

* ``load_manifest`` reads a slide **manifest** -- a CSV ``path[,source_mpp]`` (a bare
  one-path-per-line ``.txt`` is a degenerate single-column CSV). The input set for
  ``embed-many`` when a directory glob is not enough (scattered paths, remote URLs, a
  curated subset, mixed-calibration per-slide ``source_mpp``).

These are *just* the compute plan and the input set -- no output routing or templating.
"""

from __future__ import annotations

import math


def _positive_float(value, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number greater than zero") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a finite number greater than zero")
    return number


def _positive_int(value, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer greater than zero")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer greater than zero") from exc
    if integer != value or integer <= 0:
        raise ValueError(f"{field} must be an integer greater than zero")
    return integer


def load_extractions(path: str) -> list[dict]:
    """Parse an extraction-plan config; return the validated ``extractions`` list.

    Each entry is normalised to ``{"model": str, "mpp"?: float, "patch_px"?: int}``.
    Raises ``ValueError`` on a malformed plan (no list, or an entry without model).
    """
    import yaml

    with open(path) as fh:
        data = yaml.safe_load(fh.read()) or {}
    if not isinstance(data, dict) or not isinstance(data.get("extractions"), list):
        raise ValueError(
            f"{path}: config must be a mapping with a non-empty 'extractions:' list"
        )
    exts = data["extractions"]
    if not exts:
        raise ValueError(f"{path}: 'extractions' is empty")
    out: list[dict] = []
    for i, e in enumerate(exts):
        if not isinstance(e, dict) or "model" not in e:
            raise ValueError(
                f"{path}: extractions[{i}] must be a mapping with a 'model' key"
            )
        entry: dict = {"model": str(e["model"])}
        if e.get("mpp") is not None:
            entry["mpp"] = _positive_float(
                e["mpp"], field=f"{path}: extractions[{i}].mpp"
            )
        if e.get("patch_px") is not None:
            entry["patch_px"] = _positive_int(
                e["patch_px"], field=f"{path}: extractions[{i}].patch_px"
            )
        out.append(entry)
    return out


def load_manifest(path: str) -> list[dict]:
    """Parse a slide manifest; return rows ``{"path": str, "source_mpp"?: float}``.

    Accepts a CSV with a ``path``-first header (``path``, optional ``source_mpp``)
    or a bare file (``path`` or ``path,source_mpp`` per line). Blank lines and
    ``#`` comments are ignored. Raises ``ValueError`` when no slide paths are found.
    """
    import csv

    with open(path, newline="") as fh:
        lines = [
            ln for ln in fh.read().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
    if not lines:
        raise ValueError(f"{path}: manifest is empty")
    header = [c.strip().lower() for c in lines[0].split(",")]
    out: list[dict] = []
    if header[0] == "path":
        for r in csv.DictReader(lines):
            p = (r.get("path") or "").strip()
            if not p:
                continue
            row: dict = {"path": p}
            sm = (r.get("source_mpp") or "").strip()
            if sm:
                row["source_mpp"] = _positive_float(
                    sm, field=f"{path}: source_mpp"
                )
            out.append(row)
    else:
        for ln in lines:
            parts = [c.strip() for c in ln.split(",")]
            row = {"path": parts[0]}
            if len(parts) > 1 and parts[1]:
                row["source_mpp"] = _positive_float(
                    parts[1], field=f"{path}: source_mpp"
                )
            out.append(row)
    if not out:
        raise ValueError(f"{path}: manifest has no slide paths")
    return out
