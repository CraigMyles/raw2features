# Adding a feature extractor - worked example

> **Provenance rule:** every model's preprocessing is transcribed from its
> authoritative model card / paper / repo, and the source URL is recorded. If the
> preprocessing is uncertain, **do not guess** - the model is not added.

## 1. Add a registry row

`src/raw2features/embedders/registry.yaml`:

```yaml
resnet50:
  family: torchvision
  source: "torchvision://resnet50?weights=IMAGENET1K_V2"
  weights_sha256: "<sha256 of the weights file>"
  transform_source_url: "https://pytorch.org/vision/stable/models.html"
  input_size: 224
  resize: 256          # resize-then-centre-crop, per the torchvision recipe
  interpolation: bilinear
  mean: [0.485, 0.456, 0.406]
  std:  [0.229, 0.224, 0.225]
  embedding_dim: 2048
  pooling: pooled       # global average pool, fc removed
  license: "BSD-3-Clause (torchvision)"
  gated: false
```

For a gated pathology model, set `gated: true` and cite the HF card, e.g.:

```yaml
uni:
  family: timm
  source: "MahmoodLab/UNI"
  transform_source_url: "https://huggingface.co/MahmoodLab/UNI"
  ...
  gated: true
```

Gated weights are downloaded with the user's `--hf-token` / `HF_TOKEN`.

## 2. Reuse or extend the embedder

If the model loads through timm/torchvision with a standard transform, the generic
`TimmEmbedder` handles it directly from the registry row. Only write a
new `Embedder` subclass (`src/raw2features/embedders/<name>.py`) for bespoke
loaders such as CTransPath's modified Swin checkpoint.

## 3. Register + test

- Entry-point under `[project.entry-points."raw2features.embedders"]`, or ship it
  from your own package.
- Add a contract test (shape/dim/normalisation). Real-weights tests are marked
  `slow` (downloads real weights + runs a forward; skipped in CI).
