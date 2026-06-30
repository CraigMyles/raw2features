"""``raw2features validate-store`` - check a store conforms to the format spec.

Exits 0 if the store conforms to the spec (docs/SPEC.md), else 1 with the
violations. Works on a local path or a remote URL.
"""

from __future__ import annotations

import typer


def validate_store(
    store: str = typer.Argument(
        ..., help="Path or URL of a <slide_id>.embeddings.zarr to check."
    ),
) -> None:
    """Validate a store against the embeddings-store spec; exit 0 if conformant."""
    from raw2features.spec import SPEC_VERSION
    from raw2features.spec import validate_store as _validate

    violations = _validate(store)
    if not violations:
        typer.echo(f"OK - conforms to embeddings-store spec v{SPEC_VERSION}")
        raise typer.Exit(0)
    typer.echo(f"NONCONFORMANT - {len(violations)} issue(s):", err=True)
    for item in violations:
        typer.echo(f"  - {item}", err=True)
    raise typer.Exit(1)
