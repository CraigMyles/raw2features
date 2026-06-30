# Contributing to raw2features

raw2features is meant to be built *with* the community. The five seams - readers,
segmenters, patchers, embedders, sinks - are plugin points: you can add a model or
a storage backend by shipping a small package, no fork required.

## Dev setup

```bash
uv sync --extra zarr --extra image --extra torch --extra models
uv run pytest -m "not slow"
uv run ruff check
```

## Adding a feature extractor (the common case)

Every model entry must be **provenance-complete** - its preprocessing is
transcribed from the model's authoritative card/repo, and should never be guessed. If the
preprocessing is uncertain, the model is not added.

1. Add a row to `models/registry.yaml` with **all** required fields:
   - `source` (HF repo id / `torchvision://…` / local path)
   - `weights_sha256` (verified at load; load fails loudly on mismatch)
   - `transform_source_url` (the model card / paper / repo the transform came from)
   - `input_size`, `mean`, `std`, `resize`, `interpolation`
   - `embedding_dim`, `pooling` (`cls` / `mean` / `cls_mean_cat` / `pooled`)
   - `license`, `gated` (HuggingFace gating?)
2. If the model loads via timm/torchvision with a standard transform, that's all -
   the generic `TimmEmbedder` handles it. Only write a new `Embedder` subclass for
   bespoke loaders (e.g. CTransPath's modified Swin).
3. Register it: add an entry-point under `[project.entry-points."raw2features.embedders"]`
   (in this repo) or ship it from your own package's entry-points.
4. Add a test (a `MockEmbedder`-style shape/contract test is fine; a real-weights
   test should be marked `slow`).

See `docs/adding_a_model.md` for a worked example.

## Adding a reader / segmenter / sink

Same shape: subclass the seam's ABC in `src/raw2features/<seam>/base.py`, register
via entry-points. A missing optional dependency must degrade gracefully (the
plugin simply does not appear) - never break discovery.

## Conventions

- `ruff` (line length 88) for lint + format; `pytest` for tests.
- Coordinates are **level-0 (x, y)** everywhere; patch `size` is in the read
  level's pixels.
- Keep core (`src/raw2features/core/`) dependency-light and I/O-free.
