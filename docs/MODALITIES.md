# Modalities - brightfield (H&E) + multiplex (spatial proteomics)

raw2features is **H&E-first**, but the same pipeline also handles a second modality:
**multiplex** imaging (CODEX / Phenocycler / spatial proteomics), via marker-aware
encoders like KRONOS. H&E runs are completely unchanged.

## How a model declares its modality
Each registry entry carries `modality: brightfield` (default) or `modality: multiplex`.
The runner reads the modality off the requested embedders and routes the per-slide path:

| stage | brightfield (H&E) | multiplex |
|---|---|---|
| reader | `read_region` → RGB `[H,W,3]` uint8 | `read_region_channels` → native `[H,W,C]` + `channel_names` |
| segmenter | `otsu` (saturation), `canny`, … | `nuclear` - Otsu on the DAPI/Hoechst channel |
| embedder | RGB transform → `[B,3,H,W]` | per-marker norm → `[B,M,H,W]` + marker ids |

## Multiplex data → OME-Zarr
Multiplex slides are usually OME-TIFF / TIFF stacks. Convert to a channel-named OME-Zarr:

```bash
pip install "raw2features[kronos]"
python scripts/codex_to_omezarr.py SLIDE.tif SLIDE.ome.zarr --markers markers.json --mpp 0.5
```

`markers.json` is a list (or `{"raw_markers": [...]}`) of one marker name per channel. The
names are written to `omero.channels[].label`; the reader surfaces them as `channel_names`.

## KRONOS - the first multiplex model
`kronos` (MahmoodLab **KRONOSv1**) is a panel-agnostic spatial-proteomics encoder. Needs
the `[kronos]` extra (the public `kronos` package). ViT-S/16 → **384-d** patch embedding.

- **Marker matching - your panel → KRONOS's vocabulary.** KRONOS doesn't ship a
  name-resolver, only a vocabulary: the **public** `marker_metadata.csv` (175 markers,
  each with a marker id + per-marker mean/std). Mapping *your* channels to it is the
  pipeline's job (`Embedder.set_panel`), and KRONOS is panel-agnostic - it embeds whatever
  subset of its vocabulary your slide carries. Each `channel_name` is matched: exact name →
  common **synonyms** (`pancytokeratin`→`CYTOKERATIN`, `granzymeb`→`GZMB`) → a **CD-number**
  in a compound name (`cla_cd162`→`CD162`); `hoechst*`/`dapi` → DAPI; blank/empty cycles
  drop silently.
- **The mapping is recorded in the store.** The header's
  `panel.kronos` block records the full per-channel resolution under `mapping` - each kept
  channel as `{channel, channel_index, kronos_marker, marker_id}` (e.g.
  `pancytokeratin → CYTOKERATIN (id 322)`) - plus `kept` / `dropped` / `unmatched` and a
  `vocabulary` pointer pinned to the exact `marker_metadata.csv` revision. The SpatialData
  export (`INTEROP.md`) surfaces this same map as a tidy, queryable `uns["raw2features_panel"]`
  table.
- **Unmatched named channels are surfaced.** A *named* marker that
  doesn't resolve (genuinely outside KRONOS's vocabulary, or a synonym we don't yet know)
  is dropped, **warned** about (with the `matched/total` coverage), and listed in
  `panel.kronos.unmatched`. (Zero matches is a hard error; partial panels are fine.) If an
  unmatched marker *is* a KRONOS marker under another name, add it to `_resolve_marker`'s
  synonym map.
- Run it like any model:

  ```bash
  raw2features embed SLIDE.ome.zarr OUT -m kronos --mpp 0.5
  ```

  The runner detects the multiplex modality and routes the `nuclear` segmenter + native
  N-channel reads automatically.

**Performance.** A multiplex patch is an M-marker stack (~M× the tokens of a single-channel
ViT). The `kronos` family uses **torch SDPA (flash attention) by default** - no xFormers,
fully portable - which on an A100 (M=41) runs at **~41 patches/s in ~1.3 GB at batch 8**,
vs ~20 p/s / 26 GB with KRONOS's upstream naive-attention fallback (it materialises the full
8k×8k attention matrix). Memory is low and predictable, so the default `--batch-size`
fits comfortably; lower it only on small GPUs or very large panels. (Opt out via the
registry `sdpa: false`.) Validated full-slide on GPU against real CODEX multiplex data - a
56-channel panel (44 markers matched, the rest empty/blank cycles) → 28 tissue patches →
`(28, 384)`, all finite.

> KRONOS weights are **CC-BY-NC-ND** (non-commercial), like UNI / Virchow2 / CONCH - so its
> embeddings are non-commercial too. See `MODEL_LICENSES.md`.

### Marker intensity scaling
The transform brings marker intensities to `[0,1]` **by the source dtype** (uint16 → `/65535`,
uint8 → `/255`, float assumed already normalised), then standardises each marker with its
`marker_mean`/`marker_std`. For the uint16 CODEX/Phenocycler case this matches KRONOS's own
feature-extraction pipeline exactly - its loader divides by `max_value` (default `65535`) then
applies the same per-marker `(x − mean) / std`, so the published marker statistics are defined
on the same `[0,1]` scale. Scaling by the dtype range (rather than a hard-coded `65535`) keeps
uint8 / pre-normalised sources correct instead of silently squashing them.

### Adding more multiplex models
Future marker-conditioned models (e.g. KRONOS successors) are added the same way as any
encoder: a registry entry with `modality: multiplex` + a small family class (or reuse the
`kronos` family). The reader / segmenter / runner routing is shared.
