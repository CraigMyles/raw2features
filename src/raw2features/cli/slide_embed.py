"""``raw2features slide-embed`` - encode slide-level vectors from patch features.

Reads an existing ``*.embeddings.zarr`` produced by ``raw2features embed``
and writes a ``slide/<model>/`` array containing a single slide-level
embedding vector. No WSI access is needed - the patch features are read
directly from the zarr store.

Skip-if-complete: if ``slide/<model>`` already exists in the zarr and its
vector is finite, the slide is silently skipped. Delete the array (or the
whole zarr) to force re-encoding.

Two-stage example
-----------------
Stage 1 (already done, may be on a different machine / day)::

    raw2features embed slide.zarr out/ -f uni

Stage 2 (reads from the zarr, no WSI)::

    raw2features slide-embed out/slide_id.embeddings.zarr -s titan

Inline (single pass, both done together)::

    raw2features embed slide.zarr out/ -f uni -s titan
"""

from __future__ import annotations

import os

import typer


def slide_embed(
    embeddings_zarr: str = typer.Argument(
        ...,
        help="Path to an existing *.embeddings.zarr produced by 'raw2features embed'.",
    ),
    slide_encoder: list[str] = typer.Option(
        ..., "--slide-encoder", "-s", help="Slide encoder name(s); repeatable."
    ),
    patch_model: str | None = typer.Option(
        None,
        "--patch-model",
        help=(
            "Patch model whose features to use (e.g. 'uni'). "
            "Auto-detected from the registry if the zarr contains only one "
            "compatible patch model."
        ),
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto | cuda | mps | cpu (auto = best available)"
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
    force: bool = typer.Option(
        False, "--force", help="Re-encode even if slide/<model> already exists."
    ),
) -> None:
    """Encode slide-level vectors from patch features in an embeddings zarr."""
    import numpy as np
    import zarr

    from raw2features.core.device import resolve_device
    from raw2features.core.provenance import now_utc_iso
    from raw2features.slide_embedders.model_registry import (
        build_slide_embedder,
        resolve_patch_encoder,
    )

    try:
        device = resolve_device(device)  # "auto" -> cuda/mps/cpu; clear error otherwise
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    path = embeddings_zarr.rstrip("/")
    if not os.path.exists(path):
        typer.echo(f"Error: zarr store not found: {path}", err=True)
        raise typer.Exit(1)

    # use_consolidated=False: the store was consolidated by `embed`, so its
    # consolidated metadata predates any slide/ group we add here. Reading the
    # live metadata avoids stale-key KeyErrors; we re-consolidate at the end.
    from raw2features.core.store import open_grid

    root = zarr.open_group(path, mode="r+", use_consolidated=False)
    g = open_grid(root)  # the sole grid (one geometry per store)
    feat_group = g.get("features")
    if feat_group is None:
        typer.echo(
            "Error: zarr has no 'features' group - run 'raw2features embed' first.",
            err=True,
        )
        raise typer.Exit(1)

    available = sorted(feat_group.keys())
    coords = np.asarray(g["coords"][:]) if "coords" in g else None

    from raw2features import __version__

    for slide_model_name in slide_encoder:
        # Skip-if-complete check.
        if not force and "slide" in g and slide_model_name in g["slide"]:
            arr = g["slide"][slide_model_name]
            vec = np.asarray(arr[:]).astype(np.float32)
            if np.isfinite(vec).all() and (vec != 0).any():
                typer.echo(f"{slide_model_name}: already complete (skipping)")
                continue

        # Resolve which patch features to use.
        if patch_model is not None:
            pm = patch_model
            if pm not in available:
                typer.echo(
                    f"Error: --patch-model {pm!r} not in zarr. Available: {available}",
                    err=True,
                )
                raise typer.Exit(1)
        else:
            pm = resolve_patch_encoder(slide_model_name, available)

        typer.echo(f"{slide_model_name}: encoding from '{pm}' patch features …")
        patch_features = np.asarray(feat_group[pm][:]).astype(np.float32)

        slide_emb = build_slide_embedder(slide_model_name).load(device=device)
        try:
            vector = slide_emb.encode(patch_features, coords)
        finally:
            slide_emb.unload()

        # Write to zarr (open r+ so we can add without clobbering patch data).
        prov = {
            "patch_encoder": pm,
            "source": slide_emb.spec.source,
            "embedding_dim": int(len(vector)),
            "license": slide_emb.spec.license,
            "transform_source_url": slide_emb.spec.transform_source_url,
            "computed_utc": now_utc_iso(),
            "raw2features_version": __version__,
        }
        slide_grp = g.require_group("slide")
        if slide_model_name in slide_grp:
            del slide_grp[slide_model_name]
        vec2d = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        arr = slide_grp.create_array(
            slide_model_name, shape=vec2d.shape, chunks=vec2d.shape, dtype="float32"
        )
        arr[:] = vec2d
        arr.attrs["role"] = "slide_embedding"
        for k, v in prov.items():
            arr.attrs[k] = v

        # Mirror into the grid header for quick inspection.
        grid_meta = dict(g.attrs.get("raw2features", {}))
        grid_meta.setdefault("slide_embeddings", {})[slide_model_name] = prov
        g.attrs["raw2features"] = grid_meta

        typer.echo(f"{slide_model_name}: done  shape={vec2d.shape}")

    # Refresh consolidated metadata so the new slide/ arrays are visible to
    # readers that use it (and to a later `slide-embed` skip-check).
    try:
        zarr.consolidate_metadata(g.store)
    except Exception:  # noqa: BLE001 - consolidation is best-effort
        pass
