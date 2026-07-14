"""raw2features CLI entry point.

Wires every command onto the Typer app and maps user-facing errors (bad paths,
missing optional extras) to one clean line instead of a traceback.
"""

from __future__ import annotations

import typer

from raw2features import __version__
from raw2features.cli.benchmark import benchmark
from raw2features.cli.embed import embed
from raw2features.cli.embed_many import embed_many
from raw2features.cli.export_h5 import export_h5
from raw2features.cli.export_spatialdata import export_spatialdata
from raw2features.cli.info import info
from raw2features.cli.sample import sample
from raw2features.cli.slide_embed import slide_embed
from raw2features.cli.thumbnail import thumbnail
from raw2features.cli.validate_store import validate_store
from raw2features.cli.verify import verify
from raw2features.core import plugins
from raw2features.core.uris import redact_uri_credentials

app = typer.Typer(
    help="raw2features - OME-Zarr → foundation-model patch embeddings.",
    no_args_is_help=True,
    add_completion=False,
    # Surface user errors (e.g. a bad slide path) as one actionable line rather than
    # a rich traceback; main() catches them below. Real bugs still raise normally.
    pretty_exceptions_enable=False,
)

app.command()(info)
app.command()(sample)
app.command()(embed)
app.command("embed-many")(embed_many)
app.command()(verify)
app.command()(thumbnail)
app.command("slide-embed")(slide_embed)
app.command()(benchmark)
app.command("export-spatialdata")(export_spatialdata)
app.command("export-h5")(export_h5)
app.command("validate-store")(validate_store)


@app.command()
def version() -> None:
    """Print the raw2features version."""
    typer.echo(__version__)


@app.command("list")
def list_plugins(
    component: str = typer.Argument(
        ...,
        help="Component type to list: readers, segmenters, patchers, embedders, "
        "sinks, slide_embedders.",
    ),
) -> None:
    """List the available plugins of a given component type."""
    names = plugins.available(component)
    if not names:
        typer.echo(f"(no {component} registered yet)")
        return
    for name in names:
        typer.echo(name)
    if component == "embedders":
        # These are embedder *families*, not the names passed to --model. Point users
        # at the model registry so model discovery isn't misled.
        typer.echo(
            "\nThese are embedder families. For the model names you pass to "
            "--model (with dims/gating/source), run: raw2features models"
        )


@app.command()
def models() -> None:
    """List feature extractors in the model registry (name, dim, gated, source)."""
    from raw2features.embedders.model_registry import load_registry

    registry = load_registry()
    for name in sorted(registry):
        spec = registry[name]
        gate = "gated" if spec.gated else "open"
        stability = " experimental" if spec.experimental else ""
        typer.echo(
            f"{name:<10} dim={spec.embedding_dim:<5} {gate:<5}{stability:<13} "
            f"family={spec.family:<11} {spec.source}"
        )


# Which optional extra provides each third-party top-level module - used to turn a
# bare ModuleNotFoundError (clean-core install) into an actionable "install the extra".
_EXTRA_FOR_MODULE = {
    "zarr": "zarr", "numcodecs": "zarr", "ngff_zarr": "zarr", "ome_zarr": "zarr",
    "fsspec": "zarr",
    "cv2": "image", "shapely": "image", "PIL": "image",
    "torch": "torch", "torchvision": "torch",
    "timm": "models", "transformers": "models", "huggingface_hub": "models",
    "einops": "models",
    "h5py": "h5", "spatialdata": "spatialdata", "geopandas": "spatialdata",
    "s3fs": "s3", "gcsfs": "s3", "psutil": "benchmark", "pynvml": "benchmark",
}


def main() -> None:
    # A clearly user-facing error should print one actionable line and exit non-zero,
    # not a Python traceback. A bad slide path raises FileNotFoundError; a missing
    # optional dependency raises ModuleNotFoundError, which we map to its extra.
    try:
        app()
    except FileNotFoundError as exc:
        typer.secho(
            redact_uri_credentials(f"Error: {exc}"),
            fg=typer.colors.RED,
            err=True,
        )
        raise SystemExit(2) from exc
    except ModuleNotFoundError as exc:
        extra = _EXTRA_FOR_MODULE.get((exc.name or "").split(".")[0])
        if extra is None:
            raise  # genuine/unknown import error - don't mask it behind a hint
        typer.secho(
            f'Error: this needs the optional "{extra}" dependencies (missing '
            f'{exc.name!r}). Install: pip install "raw2features[{extra}]".',
            fg=typer.colors.RED,
            err=True,
        )
        raise SystemExit(2) from exc
    except Exception as exc:
        # Remote backends may put the complete signed child-object URL in an
        # otherwise ordinary HTTP/Zarr exception. Hide that specific class of
        # failure at the CLI boundary, while preserving tracebacks for unrelated
        # programming errors whose message needs no redaction.
        message = str(exc)
        redacted = redact_uri_credentials(message)
        if redacted == message:
            raise
        typer.secho(f"Error: {redacted}", fg=typer.colors.RED, err=True)
        raise SystemExit(2) from exc
