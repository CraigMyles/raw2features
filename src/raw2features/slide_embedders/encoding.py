"""Shared slide-encoding workflow for one embeddings-store grid.

Both ``raw2features embed -s`` and the standalone ``slide-embed`` command read
the same grid layout.  Keeping feature selection, spatial arguments,
provenance, and writes here prevents the two entry points from drifting.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlideEncoding:
    """One computed slide vector and the metadata needed to persist it."""

    patch_model: str
    vector: np.ndarray
    provenance: dict


def slide_output_key(group, slide_model: str, patch_model: str) -> str:
    """Storage key for a slide output, preserving multiplex coexistence.

    Historical brightfield/native outputs remain ``slide/<slide_model>`` exactly.
    Strategy-derived multiplex patch arrays can coexist on one geometry grid, so a
    model-agnostic pool is qualified by that cohort-stable effective patch key and
    cannot overwrite a pool derived from another marker/aggregation recipe.
    """

    try:
        fingerprint = dict(group["features"][patch_model].attrs).get(
            "output_fingerprint"
        )
        payload = fingerprint.get("payload", {})
        strategy_derived = (
            payload.get("loader", {}).get("family") == "multiplex_strategy"
        )
    except (AttributeError, KeyError, TypeError):
        strategy_derived = False
    if not strategy_derived:
        return slide_model
    if "/" in patch_model:
        raise ValueError(
            f"multiplex patch model key {patch_model!r} is not a safe zarr path segment"
        )
    return f"{slide_model}__{patch_model}"


def _validated_patch_fingerprint(group, patch_model: str) -> dict:
    """Return a committed, complete patch-output fingerprint.

    Slide outputs inherit the exact patch-output identity, so a self-consistent
    metadata record is not enough: the patch array must still pass the same
    shape/data completion checks used by resume before it can be consumed.
    """

    from raw2features.embedders.fingerprint import output_fingerprints_equal
    from raw2features.pipeline.receipt import validate_model

    array = group["features"][patch_model]
    array_fingerprint = dict(array.attrs).get("output_fingerprint")
    header = dict(group.attrs.get("raw2features", {}))
    models = header.get("models", {})
    metadata = models.get(patch_model, {}) if isinstance(models, dict) else {}
    header_fingerprint = (
        metadata.get("output_fingerprint") if isinstance(metadata, dict) else None
    )
    if not output_fingerprints_equal(array_fingerprint, header_fingerprint):
        raise ValueError(
            f"Patch model {patch_model!r} has no matching committed output "
            "fingerprint in its array and grid header; rerun raw2features embed for "
            "that patch model before slide encoding."
        )
    if array.ndim != 2:
        raise ValueError(
            f"Patch model {patch_model!r} does not have a two-dimensional feature "
            "array; rerun raw2features embed before slide encoding."
        )
    payload = array_fingerprint.get("payload", {})
    output = payload.get("output", {}) if isinstance(payload, dict) else {}
    embedding_dim = output.get("embedding_dim") if isinstance(output, dict) else None
    if (
        payload.get("kind") != "patch_features"
        or payload.get("model") != patch_model
        or embedding_dim != int(array.shape[1])
    ):
        raise ValueError(
            f"Patch model {patch_model!r} has a fingerprint that does not match its "
            "array name/dimension; rerun raw2features embed before slide encoding."
        )
    if (
        "coords" not in group
        or group["coords"].ndim != 2
        or int(group["coords"].shape[1]) != 2
    ):
        raise ValueError(
            "The selected grid has no valid coordinate array; rerun raw2features "
            "embed before slide encoding."
        )
    if not validate_model(
        group,
        patch_model,
        int(group["coords"].shape[0]),
        expected_dim=int(embedding_dim),
        expected_fingerprint=array_fingerprint,
    ):
        raise ValueError(
            f"Patch model {patch_model!r} is incomplete or contains invalid data; "
            "rerun raw2features embed before slide encoding."
        )
    return array_fingerprint


def slide_embedding_is_complete(
    group,
    slide_model: str,
    *,
    patch_model: str | None = None,
    device: str = "cpu",
    output_name: str | None = None,
) -> bool:
    """Return whether ``slide/<model>`` is valid for the requested patch model."""
    stored_name = output_name or slide_model
    if "slide" not in group or stored_name not in group["slide"]:
        return False
    array = group["slide"][stored_name]
    attrs = dict(array.attrs)
    stored_patch_model = attrs.get("patch_encoder")
    if attrs.get("role") != "slide_embedding" or not stored_patch_model:
        return False
    if patch_model is not None and stored_patch_model != patch_model:
        return False
    if "features" not in group or stored_patch_model not in group["features"]:
        return False

    from raw2features.embedders.fingerprint import (
        output_fingerprints_equal,
        resolved_slide_amp,
        slide_output_dim,
        slide_output_fingerprint,
    )
    from raw2features.slide_embedders.model_registry import get_slide_spec

    patch_array = group["features"][stored_patch_model]
    try:
        patch_fingerprint = _validated_patch_fingerprint(group, stored_patch_model)
        spec = get_slide_spec(slide_model)
        expected_dim = slide_output_dim(spec, int(patch_array.shape[1]))
        expected_fingerprint = slide_output_fingerprint(
            spec,
            patch_model=stored_patch_model,
            patch_output_fingerprint=patch_fingerprint,
            patch_dim=int(patch_array.shape[1]),
            resolved_amp=resolved_slide_amp(spec, device),
        )
    except (KeyError, TypeError, ValueError):
        return False

    shape = tuple(int(size) for size in array.shape)
    if shape != (1, expected_dim):
        return False
    if np.dtype(array.dtype) != np.dtype(np.float32):
        return False
    if not output_fingerprints_equal(
        attrs.get("output_fingerprint"), expected_fingerprint
    ):
        return False
    try:
        if int(attrs.get("embedding_dim", -1)) != shape[1]:
            return False
    except (TypeError, ValueError):
        return False

    header = dict(group.attrs.get("raw2features", {}))
    mirrored = header.get("slide_embeddings", {}).get(stored_name, {})
    if not isinstance(mirrored, dict):
        return False
    if mirrored.get("patch_encoder") != stored_patch_model:
        return False
    if not output_fingerprints_equal(
        mirrored.get("output_fingerprint"), expected_fingerprint
    ):
        return False
    try:
        if int(mirrored.get("embedding_dim", -1)) != shape[1]:
            return False
    except (TypeError, ValueError):
        return False

    vector = np.asarray(array[:], dtype=np.float32)
    return bool(vector.size and np.isfinite(vector).all() and (vector != 0).any())


def resolve_slide_grid(
    root,
    slide_model: str,
    *,
    grid: str | None = None,
    patch_model: str | None = None,
):
    """Select the store grid and patch model for one slide encoder request.

    This is the shared selection contract for standalone ``slide-embed`` and inline
    ``embed -s`` fallback. A specific slide model is discovered by its required patch
    encoder across all grids. Model-agnostic encoders require an explicit choice when
    the store has several grids.
    """
    from raw2features.core.store import grid_for_model, grid_keys, open_grid
    from raw2features.slide_embedders.model_registry import get_slide_spec

    keys = grid_keys(root)
    if not keys:
        raise ValueError("store has no grids/ group; expected a v0.1 embeddings store")
    if grid is not None:
        if grid not in keys:
            raise ValueError(f"Unknown grid {grid!r}. Available grid keys: {keys}")
        selected_grid = grid
    elif len(keys) == 1:
        selected_grid = keys[0]
    elif patch_model is not None:
        selected_grid = grid_for_model(root, patch_model)
    else:
        required = get_slide_spec(slide_model).patch_encoder
        if required == "any":
            raise ValueError(
                f"Slide encoder {slide_model!r} is model-agnostic and the store has "
                f"multiple grids {keys}; pass --grid or --patch-model to choose one."
            )
        selected_grid = grid_for_model(root, required)

    group = open_grid(root, selected_grid)
    selected_patch_model = resolve_slide_patch_model(
        group,
        slide_model,
        patch_model=patch_model,
    )
    return selected_grid, group, selected_patch_model


def resolve_slide_patch_model(
    group,
    slide_model: str,
    *,
    patch_model: str | None = None,
    available_patch_models: list[str] | None = None,
) -> str:
    """Resolve and validate the patch model for one slide encoder and grid."""
    from raw2features.slide_embedders.model_registry import (
        get_slide_spec,
        resolve_patch_encoder,
    )

    feature_group = group.get("features")
    if feature_group is None:
        raise ValueError(
            "Embeddings grid has no 'features' group; run "
            "'raw2features embed' first."
        )

    available = (
        sorted(feature_group.keys())
        if available_patch_models is None
        else list(available_patch_models)
    )
    if patch_model is None:
        return resolve_patch_encoder(slide_model, available)

    required = get_slide_spec(slide_model).patch_encoder
    if required != "any" and patch_model != required:
        raise ValueError(
            f"Slide encoder {slide_model!r} requires patch model {required!r}; "
            f"--patch-model {patch_model!r} is incompatible."
        )
    if patch_model not in available or patch_model not in feature_group:
        raise ValueError(
            f"Patch model {patch_model!r} is not in the selected grid. "
            f"Available: {sorted(feature_group.keys())}"
        )
    return patch_model


def encode_slide_embedding(
    group,
    slide_model: str,
    device: str,
    *,
    patch_model: str | None = None,
    available_patch_models: list[str] | None = None,
) -> SlideEncoding | None:
    """Encode ``slide_model`` from patch features in one open grid group.

    ``None`` means the selected patch array is empty, so there is no meaningful
    slide vector to write.
    """
    from raw2features import __version__
    from raw2features.core.provenance import now_utc_iso
    from raw2features.embedders.fingerprint import (
        fingerprint_digest,
        resolved_slide_amp,
        slide_output_dim,
        slide_output_fingerprint,
    )
    from raw2features.slide_embedders.model_registry import build_slide_embedder

    feature_group = group.get("features")
    selected_patch_model = resolve_slide_patch_model(
        group,
        slide_model,
        patch_model=patch_model,
        available_patch_models=available_patch_models,
    )

    patch_array = feature_group[selected_patch_model]
    if int(patch_array.shape[0]) == 0:
        warnings.warn(
            f"slide encoder {slide_model!r}: 0 patch features (no tissue kept) - "
            "skipping slide-level encoding for this slide.",
            stacklevel=2,
        )
        return None

    # Refuse to mint current-looking slide provenance over a legacy/unknown patch
    # array. The patch model must first be recomputed under the current contract.
    slide_embedder = build_slide_embedder(slide_model)
    patch_fingerprint = _validated_patch_fingerprint(group, selected_patch_model)
    output_fingerprint = slide_output_fingerprint(
        slide_embedder.spec,
        patch_model=selected_patch_model,
        patch_output_fingerprint=patch_fingerprint,
        patch_dim=int(patch_array.shape[1]),
        resolved_amp=resolved_slide_amp(slide_embedder.spec, device),
    )
    expected_dim = slide_output_dim(slide_embedder.spec, int(patch_array.shape[1]))

    slide_embedder.load(device=device)
    try:
        patch_features = np.asarray(patch_array[:], dtype=np.float32)
        coords = np.asarray(group["coords"][:]) if "coords" in group else None
        header = dict(group.attrs.get("raw2features", {}))
        patch_size_lv0 = header.get("patching", {}).get("level0_patch")
        vector = np.asarray(
            slide_embedder.encode(patch_features, coords, patch_size_lv0),
            dtype=np.float32,
        ).reshape(-1)
        if int(vector.size) != expected_dim:
            raise ValueError(
                f"slide encoder {slide_model!r} returned {vector.size} values; "
                f"its current output contract requires {expected_dim}."
            )
        provenance = {
            "patch_encoder": selected_patch_model,
            "source": slide_embedder.spec.source,
            "embedding_dim": int(vector.size),
            "license": slide_embedder.spec.license,
            "transform_source_url": slide_embedder.spec.transform_source_url,
            "doi": slide_embedder.spec.doi,
            "weights_sha256": slide_embedder.spec.weights_sha256,
            "weights_revision": slide_embedder.spec.weights_revision,
            "weights_filename": slide_embedder.spec.weights_filename,
            "patch_output_fingerprint": fingerprint_digest(patch_fingerprint),
            "resolved_amp": resolved_slide_amp(slide_embedder.spec, device),
            "output_fingerprint": output_fingerprint,
            "computed_utc": now_utc_iso(),
            "raw2features_version": __version__,
        }
    finally:
        slide_embedder.unload()

    return SlideEncoding(selected_patch_model, vector, provenance)


def write_slide_embedding(
    group,
    slide_model: str,
    vector: np.ndarray,
    provenance: dict,
    *,
    output_name: str | None = None,
) -> None:
    """Replace ``slide/<model>`` with one vector and refresh grid metadata."""
    stored_name = output_name or slide_model
    stored_provenance = dict(provenance)
    if stored_name != slide_model:
        stored_provenance["slide_encoder"] = slide_model
    slide_group = group.require_group("slide")
    if stored_name in slide_group:
        del slide_group[stored_name]

    vector_2d = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    array = slide_group.create_array(
        stored_name,
        shape=vector_2d.shape,
        chunks=vector_2d.shape,
        dtype="float32",
    )
    array[:] = vector_2d
    array.attrs["role"] = "slide_embedding"
    for key, value in stored_provenance.items():
        array.attrs[key] = value

    header = dict(group.attrs.get("raw2features", {}))
    slide_metadata = dict(header.get("slide_embeddings", {}))
    slide_metadata[stored_name] = stored_provenance
    header["slide_embeddings"] = slide_metadata
    group.attrs["raw2features"] = header
