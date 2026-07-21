"""``raw2features slide-embed`` - encode slide-level vectors from patch features.

Reads an existing ``*.embeddings.zarr`` produced by ``raw2features embed``
and writes a ``slide/<model>/`` array containing a single slide-level
embedding vector. No WSI access is needed - the patch features are read
directly from the zarr store.

Skip-if-complete: if ``slide/<model>`` already contains a finite, non-zero
vector, the slide is skipped. Pass ``--force`` to replace it.

Two-stage example
-----------------
Stage 1 (already done, may be on a different machine / day)::

    raw2features embed slide.zarr out/ -f conch_v1_5 --patch-size 512

Stage 2 (reads from the zarr, no WSI)::

    raw2features slide-embed out/slide_id.embeddings.zarr -s titan

Inline (single pass, both done together)::

    raw2features embed slide.zarr out/ -f conch_v1_5 --patch-size 512 -s titan
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
    grid: str | None = typer.Option(
        None,
        "--grid",
        help=(
            "Grid key under grids/ (e.g. 'mpp0.5_px512'). Required when the "
            "grid cannot be inferred unambiguously from the encoder or --patch-model."
        ),
    ),
    patch_model: str | None = typer.Option(
        None,
        "--patch-model",
        help=(
            "Patch model whose features to use for a model-agnostic encoder "
            "(e.g. 'uni' with 'mean'). Specific encoders require their registry "
            "model; otherwise it is auto-detected."
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
    import zarr

    from raw2features.core.device import resolve_device
    from raw2features.slide_embedders.encoding import (
        encode_slide_embedding,
        resolve_slide_grid,
        slide_embedding_is_complete,
        slide_output_key,
        write_slide_embedding,
    )
    from raw2features.slide_embedders.model_registry import (
        validate_slide_encoder_names,
    )

    try:
        validate_slide_encoder_names(slide_encoder)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

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
    root = zarr.open_group(path, mode="r+", use_consolidated=False)

    for slide_model_name in slide_encoder:
        try:
            selected_grid, group, selected_patch_model = resolve_slide_grid(
                root,
                slide_model_name,
                grid=grid,
                patch_model=patch_model,
            )
        except (KeyError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc

        # Strategy-derived multiplex pools are patch-qualified so several marker
        # recipes can coexist on the same grid without replacing one another.
        output_name = slide_output_key(
            group, slide_model_name, selected_patch_model
        )
        # Skip only an output produced from the requested patch model.
        if not force and slide_embedding_is_complete(
            group,
            slide_model_name,
            patch_model=selected_patch_model,
            device=device,
            output_name=output_name,
        ):
            typer.echo(
                f"{slide_model_name} [{selected_grid}]: already complete (skipping)"
            )
            continue

        typer.echo(
            f"{slide_model_name} [{selected_grid}]: encoding from "
            f"'{selected_patch_model}' patch features …"
        )
        try:
            encoding = encode_slide_embedding(
                group,
                slide_model_name,
                device,
                patch_model=selected_patch_model,
            )
        except (KeyError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1) from exc
        if encoding is None:
            typer.echo(
                f"{slide_model_name} [{selected_grid}]: 0 patch features (skipping)"
            )
            continue

        write_slide_embedding(
            group,
            slide_model_name,
            encoding.vector,
            encoding.provenance,
            output_name=output_name,
        )
        typer.echo(
            f"{slide_model_name} [{selected_grid}]: done  "
            f"shape={(1, encoding.vector.size)}"
        )

    # Refresh consolidated metadata so the new slide/ arrays are visible to
    # readers that use it (and to a later `slide-embed` skip-check).
    try:
        zarr.consolidate_metadata(root.store)
    except Exception:  # noqa: BLE001 - consolidation is best-effort
        pass
