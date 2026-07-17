# Contributing to raw2features

raw2features is meant to be built *with* the community. Its six implementation seams -
readers, segmenters, patchers, embedders, sinks, and slide embedders - are Python
entry-point plugin groups. A third-party package can add an implementation or backend
without forking raw2features.

The patch-model names accepted by the CLI are deliberately narrower: they come from the
provenance-complete registry bundled with raw2features. An external
`raw2features.embedders` entry point adds a loader family, not a new `--model` row by
itself. Python callers can pass external `Embedder` instances explicitly through
`embed_slide(..., embedders=[...])`; proposing a built-in CLI model still means adding a
reviewed registry row here.

## Dev setup

```bash
uv sync --extra zarr --extra image --extra torch --extra models
uv run pytest -m "not slow and not network"
uv run ruff check
```

## Adding a feature extractor (the common case)

Every model entry must be **provenance-complete** - its preprocessing is
transcribed from the model's authoritative card/repo, and should never be guessed. If the
preprocessing is uncertain, the model is not added.

1. Add a row to `src/raw2features/embedders/registry.yaml` with **all** required fields:
   - `source` (HF repo id / `torchvision://…` / local path)
   - `weights_sha256`, `weights_revision`, and `weights_filename` (immutable artifact
     identity; verified at load where raw2features handles the bytes)
   - `transform_source_url` (the model card / paper / repo the transform came from)
   - `input_size`, `mean`, `std`, `resize`, `interpolation`
   - `embedding_dim`, `pooling`, model family/constructor parameters, and any
     recommended extraction MPP/patch size
   - `license`, `gated` (Hugging Face access gate?), paper/source metadata, and DOI
2. If the model loads via timm/torchvision with a standard transform, that's all -
   the generic `TimmEmbedder` handles it. Only write a new `Embedder` subclass for
   bespoke loaders (e.g. CTransPath's modified Swin).
3. If it needs a new loader family, register that implementation under
   `[project.entry-points."raw2features.embedders"]`. Models using an existing family
   need only the registry row and tests.
4. Add a test (a `MockEmbedder`-style shape/contract test is fine; a real-weights
   test should be marked `slow`).

See `docs/adding_a_model.md` for a worked example.

## Adding another implementation plugin

Subclass the relevant ABC and register it through the matching
`raw2features.<seam>` entry-point group. The guaranteed discovery contract is small:
an importable entry point appears in `available()` and is returned by `get()`; an entry
point that cannot import (for example because its own optional dependency is absent) is
skipped without breaking discovery. The seam ABC defines the runtime methods and return
types. raw2features does not claim that an arbitrary third-party object conforms merely
because its entry point imports, so plugin packages should add their own end-to-end
contract test.

## Conventions

- `ruff` (line length 88) for lint + format; `pytest` for tests.
- Coordinates are **level-0 (x, y)** everywhere; patch `size` is in the read
  level's pixels.
- Keep core (`src/raw2features/core/`) dependency-light and I/O-free.
