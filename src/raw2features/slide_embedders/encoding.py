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


def slide_embedding_is_complete(
    group,
    slide_model: str,
    *,
    patch_model: str | None = None,
) -> bool:
    """Return whether ``slide/<model>`` is valid for the requested patch model."""
    if "slide" not in group or slide_model not in group["slide"]:
        return False
    array = group["slide"][slide_model]
    shape = tuple(int(size) for size in array.shape)
    if len(shape) != 2 or shape[0] != 1 or shape[1] <= 0:
        return False
    if np.dtype(array.dtype) != np.dtype(np.float32):
        return False

    attrs = dict(array.attrs)
    stored_patch_model = attrs.get("patch_encoder")
    if attrs.get("role") != "slide_embedding" or not stored_patch_model:
        return False
    if patch_model is not None and stored_patch_model != patch_model:
        return False
    try:
        if int(attrs.get("embedding_dim", -1)) != shape[1]:
            return False
    except (TypeError, ValueError):
        return False

    header = dict(group.attrs.get("raw2features", {}))
    mirrored = header.get("slide_embeddings", {}).get(slide_model, {})
    if not isinstance(mirrored, dict):
        return False
    if mirrored.get("patch_encoder") != stored_patch_model:
        return False
    try:
        if int(mirrored.get("embedding_dim", -1)) != shape[1]:
            return False
    except (TypeError, ValueError):
        return False

    vector = np.asarray(array[:], dtype=np.float32)
    return bool(vector.size and np.isfinite(vector).all() and (vector != 0).any())


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

    slide_embedder = build_slide_embedder(slide_model).load(device=device)
    try:
        patch_features = np.asarray(patch_array[:], dtype=np.float32)
        coords = np.asarray(group["coords"][:]) if "coords" in group else None
        header = dict(group.attrs.get("raw2features", {}))
        patch_size_lv0 = header.get("patching", {}).get("level0_patch")
        vector = np.asarray(
            slide_embedder.encode(patch_features, coords, patch_size_lv0),
            dtype=np.float32,
        ).reshape(-1)
        provenance = {
            "patch_encoder": selected_patch_model,
            "source": slide_embedder.spec.source,
            "embedding_dim": int(vector.size),
            "license": slide_embedder.spec.license,
            "transform_source_url": slide_embedder.spec.transform_source_url,
            "doi": slide_embedder.spec.doi,
            "weights_sha256": slide_embedder.spec.weights_sha256,
            "weights_revision": slide_embedder.spec.weights_revision,
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
) -> None:
    """Replace ``slide/<model>`` with one vector and refresh grid metadata."""
    slide_group = group.require_group("slide")
    if slide_model in slide_group:
        del slide_group[slide_model]

    vector_2d = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    array = slide_group.create_array(
        slide_model,
        shape=vector_2d.shape,
        chunks=vector_2d.shape,
        dtype="float32",
    )
    array[:] = vector_2d
    array.attrs["role"] = "slide_embedding"
    for key, value in provenance.items():
        array.attrs[key] = value

    header = dict(group.attrs.get("raw2features", {}))
    slide_metadata = dict(header.get("slide_embeddings", {}))
    slide_metadata[slide_model] = dict(provenance)
    header["slide_embeddings"] = slide_metadata
    group.attrs["raw2features"] = header

    # Slide encoding can run several requested models in sequence. Consolidate
    # after each successful mutation so a later model failure cannot leave this
    # completed result invisible to readers that use consolidated metadata.
    try:
        import zarr

        zarr.consolidate_metadata(group.store)
    except Exception:  # noqa: BLE001 - consolidation is best-effort
        pass
