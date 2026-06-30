"""Conformance check for the embeddings-store format (see ``docs/SPEC.md``).

``validate_store(path)`` returns a list of human-readable violations; an empty list
means the store conforms. The **header** is validated against the packaged JSON Schema
(``schema/embeddings_store-<version>.schema.json`` - the normative definition); the
**array-level** invariants a JSON Schema cannot express (array shapes/dtypes, the
``role``/``units`` attrs, and the 1:1 ``coords``/``features`` rows) are checked here
too. It uses zarr + jsonschema, so any consumer can run it on a local path or a
remote URL. The test suite validates real pipeline output against this, so the format,
the schema and ``docs/SPEC.md`` cannot drift.
"""

from __future__ import annotations

import json
from importlib.resources import files

SPEC_VERSION = "0.1"

_INT_DTYPES = (
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
)

def _load_header_schema(version: str | None) -> dict | None:
    """Load the packaged JSON Schema for a store's ``schema_version`` (or ``None``)."""
    if not version:
        return None
    name = f"embeddings_store-{version}.schema.json"
    try:
        text = files("raw2features.schema").joinpath(name).read_text()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        return None
    return json.loads(text)


def _format_schema_error(err) -> str:
    """Render a jsonschema error in the established ``header.<path> …`` style."""
    path = "".join(f".{p}" for p in err.absolute_path)
    if err.validator == "required":
        # Name every missing required key under this object (clear, stable messages).
        missing = [k for k in err.validator_value if k not in err.instance]
        return "; ".join(f"header{path}.{k} is missing" for k in missing) or (
            f"header{path}: {err.message}"
        )
    if err.validator == "const":
        return f"header{path} must be {err.validator_value!r}, got {err.instance!r}"
    if err.validator == "minProperties" and path == ".models":
        return "header.models has no model entries"
    return f"header{path}: {err.message}"


def _header_schema_violations(header: dict) -> list[str]:
    """Validate ``header`` against its JSON Schema; return de-duplicated violations."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a core dependency
        return [
            "jsonschema is not installed; cannot validate the header against the "
            "format schema (install raw2features's core dependencies)"
        ]
    version = header.get("schema_version")
    schema = _load_header_schema(version)
    if schema is None:
        return [
            f"no packaged JSON Schema for schema_version {version!r} "
            f"(this build ships {SPEC_VERSION!r})"
        ]
    validator = jsonschema.Draft202012Validator(schema)
    seen: dict[str, None] = {}  # ordered de-dup (a `required` error may repeat per key)
    errors = sorted(validator.iter_errors(header), key=lambda e: list(e.absolute_path))
    for err in errors:
        for line in _format_schema_error(err).split("; "):
            seen.setdefault(line, None)
    return list(seen)


def validate_store(path: str) -> list[str]:
    """Return a list of spec violations for ``path`` (empty == conformant).

    Patch sets live under ``grids/<key>/``; each grid has a complete, self-describing
    header validated independently (violations are prefixed ``grids/<key>:``).
    The root must carry the ``raw2features`` attr and at least one grid.
    """
    import zarr

    from raw2features.core.store import GRIDS, grid_keys

    try:
        root = zarr.open_group(path, mode="r")
    except Exception as exc:  # noqa: BLE001 - any open failure is one clear violation
        return [f"cannot open {path!r} as a zarr group: {exc}"]

    if not isinstance(dict(root.attrs).get("raw2features"), dict):
        # The header is the whole contract; without it nothing else can run.
        return ["missing group attr 'raw2features' (the header)"]
    keys = grid_keys(root)
    if not keys:
        return ["store has no grids/ subgroups (not a v0.1 embeddings store)"]

    out: list[str] = []
    for key in keys:
        g = root[GRIDS][key]
        header = dict(g.attrs).get("raw2features")
        if not isinstance(header, dict):
            out.append(f"grids/{key}: missing 'raw2features' header")
            continue
        out += [f"grids/{key}: {v}" for v in _validate_grid(g, header)]
    return out


def _validate_grid(g, header: dict) -> list[str]:
    """Spec violations for one grid group: its header (schema) + array invariants."""
    # 1) The header, against the normative JSON Schema (the source of truth).
    out: list[str] = list(_header_schema_violations(header))

    # 2) Array-level invariants the JSON Schema cannot express.
    n = None
    if "coords" not in g:
        out.append("required array 'coords' is missing")
    else:
        c = g["coords"]
        if c.ndim != 2 or tuple(c.shape[1:]) != (2,):
            out.append(f"coords must be (N, 2), got shape {tuple(c.shape)}")
        if str(c.dtype) not in _INT_DTYPES:
            out.append(f"coords must be an integer dtype, got {c.dtype}")
        if dict(c.attrs).get("role") != "coords":
            out.append("coords must carry attr role='coords'")
        if dict(c.attrs).get("units") != "level0_px":
            out.append("coords must carry attr units='level0_px'")
        n = int(c.shape[0])

    if "features" not in g:
        out.append("required group 'features' is missing")
    else:
        feats = g["features"]
        hdr_models = header.get("models")
        hdr_models = hdr_models if isinstance(hdr_models, dict) else {}
        models = list(feats.keys())
        if not models:
            out.append("'features' group has no model arrays")
        for m in models:
            a = feats[m]
            if a.ndim != 2:
                out.append(f"features/{m} must be 2-D (N, dim), got {tuple(a.shape)}")
                continue
            if n is not None and int(a.shape[0]) != n:
                out.append(
                    f"features/{m} length {a.shape[0]} != coords length {n} "
                    "(1:1 invariant)"
                )
            if not str(a.dtype).startswith("float"):
                out.append(f"features/{m} must be a float dtype, got {a.dtype}")
            if dict(a.attrs).get("role") != "features":
                out.append(f"features/{m} must carry attr role='features'")
            if dict(a.attrs).get("model") != m:
                out.append(f"features/{m} must carry attr model='{m}'")
            if m in hdr_models:
                entry = hdr_models[m] if isinstance(hdr_models[m], dict) else {}
                dim = entry.get("embedding_dim")
                if dim is not None and int(a.shape[1]) != int(dim):
                    out.append(
                        f"features/{m} dim {a.shape[1]} != "
                        f"header.models.{m}.embedding_dim {dim}"
                    )
            else:
                out.append(f"features/{m} has no header.models['{m}'] entry")

    # optional arrays - shape sanity + role when present
    if "grid_index" in g:
        if n is not None and tuple(g["grid_index"].shape) != (n, 2):
            got = tuple(g["grid_index"].shape)
            out.append(f"grid_index must be (N, 2)=({n}, 2), got {got}")
        if dict(g["grid_index"].attrs).get("role") != "grid_index":
            out.append("grid_index must carry attr role='grid_index'")
    if "mask" in g:
        if g["mask"].ndim != 2:
            out.append(
                f"mask must be 2-D (rows, cols), got shape {tuple(g['mask'].shape)}"
            )
        if dict(g["mask"].attrs).get("role") != "tissue_mask":
            out.append("mask must carry attr role='tissue_mask'")
    if "slide" in g:
        for m in g["slide"].keys():
            s = g["slide"][m]
            if s.ndim != 2 or int(s.shape[0]) != 1:
                out.append(f"slide/{m} must be (1, dim), got shape {tuple(s.shape)}")
            if dict(s.attrs).get("role") != "slide_embedding":
                out.append(f"slide/{m} must carry attr role='slide_embedding'")

    # Optional per-patch QC layer: ONE generic rule for every tool/class (values, not
    # schema), so a new scorer adds a qc/<tool>/ subgroup with no format change -- every
    # array under any qc/ group is 1:1 with coords and carries role='qc'.
    if "qc" in g:
        for tool in g["qc"].keys():
            tg = g["qc"][tool]
            for name in tg.keys():
                arr = tg[name]
                if not hasattr(arr, "shape"):
                    continue  # a nested group, not a qc array
                if n is not None and int(arr.shape[0]) != n:
                    out.append(
                        f"qc/{tool}/{name} length {arr.shape[0]} != coords length {n} "
                        "(1:1 invariant)"
                    )
                if dict(arr.attrs).get("role") != "qc":
                    out.append(f"qc/{tool}/{name} must carry attr role='qc'")

    return out
