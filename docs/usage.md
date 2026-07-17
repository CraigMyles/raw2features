# raw2features - Usage Guide

`raw2features` reads a whole-slide image **in OME-Zarr / OME-NGFF** and writes
**patch-level foundation-model embeddings** to its own `*.embeddings.zarr`
store, with the storage backend and the model(s) independently swappable. Every
embedding records the exact slide region it came from, so it maps back to its precise location.

This guide explains each command, **what actually happens when you run it**, and
the behaviours that are easy to get wrong (idempotent re-runs, exact MPP,
multi-model fan-out, thumbnails).

## Contents

1. [Mental model](#1-mental-model)
2. [Install](#2-install)
3. [Commands at a glance](#3-commands-at-a-glance)
4. [`info` - probe a slide](#4-info--probe-a-slide)
5. [`embed` - extract embeddings](#5-embed--extract-embeddings)
6. [What `embed` actually does](#6-what-embed-actually-does)
7. [Re-runs & idempotency (the "already-embedded" behaviour)](#7-re-runs--idempotency)
8. [`thumbnail` - previews & QC overlays](#8-thumbnail--previews--qc-overlays)
9. [Cohort runs on SLURM](#9-cohort-runs-on-slurm)
10. [Output reference](#10-output-reference)
11. [FAQ / gotchas](#11-faq--gotchas)

---

## 1. Mental model

The pipeline is a chain of swappable **seams**; each `embed` run flows left to right:

```
OME-Zarr  ──reader──▶  segmenter ──▶  patcher ──▶  embedder(s) ──▶  sink ──▶  *.embeddings.zarr
(omezarr)             (otsu/none)   (grid,        (resnet50/dinov2/  (zarr)    + receipt (optional)
                                     exact MPP)    uni/uni2_h, …)
```

| Seam | Default | What it does |
|------|---------|--------------|
| **reader** | `omezarr` | resolves NGFF metadata (MPP, pyramid levels) + does random patch reads |
| **segmenter** | `otsu` (or `--no-seg`) | low-res tissue mask so only tissue patches are kept |
| **patcher** | `grid` | tiles a regular grid at an **exact** target MPP |
| **embedder** | `resnet50` (repeatable) | patch → feature vector; one or many, run together |
| **slide_embedder** | none | patch-feature matrix → optional slide-level vector |
| **sink** | `zarr` | writes `coords`/`grid_index`/`mask`/`features/<model>` + provenance |

List what's installed for any component: `raw2features list embedders` (or `readers`,
`segmenters`, `patchers`, `sinks`, `slide_embedders`).

---

## 2. Install

```bash
git clone https://github.com/CraigMyles/raw2features && cd raw2features
uv sync --extra zarr --extra image --extra torch --extra models
```

Build the venv **on the machine you will run on** - Torch wheels are
architecture-specific, so a venv built on one arch (e.g. aarch64) will not import
on another (e.g. an x86 cluster). Verify with `raw2features version`.

---

## 3. Commands at a glance

| Command | Purpose |
|---------|---------|
| `raw2features info <slide.zarr>` | print NGFF version, MPP, pyramid levels, and the exact-MPP read plan - extracts nothing |
| `raw2features sample <out.zarr>` | write a tiny synthetic OME-Zarr slide (quickstart / testing) |
| `raw2features embed <slide.zarr> <out_dir>` | extract patch embeddings for one slide |
| `raw2features embed-many <slide_dir> <out_dir>` | extract embeddings for a directory/cohort of slides (sharded, resumable) |
| `raw2features verify <slide.zarr> --receipts-dir DIR --out-dir OUT` | exit 0 iff the requested output is already complete, source-bound, and validated |
| `raw2features thumbnail <slide.zarr> <out_dir>` | write a thumbnail PNG (+ optional QC overlay) |
| `raw2features export-spatialdata <store.embeddings.zarr>` | convert an embedding store to scverse SpatialData (squidpy/napari) - see [`INTEROP.md`](INTEROP.md) |
| `raw2features export-h5 <store.embeddings.zarr> <out_dir>` | export features to HDF5 (one `.h5` per model); needs the `[h5]` extra |
| `raw2features validate-store <store.embeddings.zarr>` | check a store conforms to the embeddings-store spec |
| `raw2features models` | list the model registry (name, dim, gated, source) |
| `raw2features list <component>` | list installed plugins for a component type (readers, segmenters, patchers, embedders, sinks, slide_embedders) |
| `raw2features version` | print the version |

Every option below is also visible via `raw2features <command> --help`.

---

## 4. `info` - probe a slide

```bash
raw2features info slide.zarr --mpp 1.0 --patch-size 224
```

Reads only metadata (no pixels extracted) and prints the NGFF version, level-0
MPP, every pyramid level (size, downsample, native MPP), and the **read plan** for
your target MPP - which level it would read, at what pixel size, the resample
factor, and the achieved MPP. Use it to sanity-check a slide before a run. Example:

```
mpp_level0:    0.2500
levels:
  L0: 102917 x 43525    downsample   1.00  mpp 0.2500
  ...
  L2:  25729 x 10881    downsample   4.00  mpp 1.0002
plan @ 1.0 um/px, 224px: read L2 (224px, resample 0.9998) -> achieved 1.0002 um/px [native, no resize]
```

---

## 5. `embed` - extract embeddings

Simplest run (open model, no token needed):

```bash
raw2features embed slide.zarr out/ -m resnet50 --mpp 1.0 --patch-size 224
```

**Multiple extractors** (decode each patch once, run every model on it):

```bash
raw2features embed slide.zarr out/ -m uni -m resnet50 -m dinov2 --mpp 1.0
```

**Gated models** (UNI, UNI2-h) need a HuggingFace token *and* access granted on
the model page:

```bash
hf auth login                   # cached login; alternatively set HF_TOKEN
raw2features embed slide.zarr out/ -m uni
```

`--hf-token` remains available for automation, but cached login or the `HF_TOKEN`
environment variable avoids exposing a token through shell history or process listings.
Any token passed on the command line is redacted from stored command provenance.

### Key options

| Option | Default | Meaning |
|--------|---------|---------|
| `-m, --model` | `resnet50` | model name; **repeatable** for multi-model (`-f`/`--feature-extractor` are aliases) |
| `--mpp` | per-model | target µm/px; default = each model's **recommended** MPP (0.5 for most pathology FMs, 1.0 for scale-agnostic). A value overrides the scale while retaining each model's recommended patch size; combine it with `--patch-size` to force one shared grid |
| `--source-mpp` | source metadata | level-0 source µm/px override for an uncalibrated OME-Zarr whose spatial axes declare no physical unit; normally leave unset |
| `--patch-size` | per-model | patch side in px; default = each model's recommended size (224 / 448 CONCH / 512 conch_v1_5). A value overrides the size; combine it with `--mpp` to force one shared grid |
| `--step` | = patch | stride in output px; `< patch` gives overlap |
| `--no-seg` | off | tile the whole slide (skip tissue masking) |
| `--segmenter` | `otsu` | tissue segmenter to use |
| `--tissue-threshold` | `0.1` | keep a grid cell if ≥ this fraction is tissue |
| `--features-dtype` | `float16` | stored feature dtype |
| `--stain-norm` | off | stain-normalize each patch before embedding (`macenko`\|`reinhard`\|`vahadane`; `vahadane` needs `raw2features[stain]`); **changes the features** - use separate output dirs for with/without-norm experiments |
| `--config FILE` | none | YAML extraction plan (`extractions:` list of `{model, mpp?, patch_px?}`); supersedes `-m`/`--mpp`/`--patch-size`. The same model may repeat (one grid each - the MPP-ablation case) |
| `-s, --slide-encoder` | none | slide-level encoder(s) run after patch embedding (e.g. `titan`); reads patch features from the store, no WSI re-read. Repeatable. |
| `--qc` | none | per-patch QC producer(s) writing `grids/<key>/qc/<tool>/` (e.g. `grandqc`); needs the producer's extra (`raw2features[grandqc]`) |
| `--qc-stain-norm` | off | normalize the QC input first (`macenko`\|`reinhard`\|`vahadane`; `vahadane` needs `raw2features[stain]`) - for a stain outside the QC model's domain |
| `--device` | `auto` | `auto` (best of cuda→mps→cpu) \| `cuda` \| `mps` \| `cpu` |
| `--devices` | = `--device` | opt-in in-process multi-GPU, e.g. `cuda:0,cuda:1`; shards this slide's patches across them and gathers in coord order (output identical to one device). See _In-process multi-GPU_ in §6. |
| `--batch-size` | `256` | patches per forward pass (lower it on small GPUs) |
| `--amp` | `auto` | precision: `auto` (each model's card precision) \| `fp32` \| `bf16` \| `fp16` |
| `--snap-to-level` | off | read a pyramid level natively (no resample); MPP = that level |
| `--mpp-tolerance` | `0.001` | read a level natively if within this fraction of target MPP |
| `--allow-upsample` | off | permit upsampling when the target is finer than level 0 |
| `--emit-geojson` | off | also write `<id>.patches.geojson` (QuPath patch polygons) |
| `--emit-thumbnail` | off | also write a thumbnail + QC overlay (see §8) |
| `--receipts-dir DIR` | none | enable receipts + **skip-if-complete** (see §7) |
| `--hf-token` | `$HF_TOKEN` | token for gated models; cached login or the environment is preferred, and command provenance is redacted |

See `raw2features models` for available extractors and which are gated.

### Authenticated remote sources

The reader receives the complete remote URI in memory so query-authenticated OME-Zarr
stores can be opened. The persisted `source.uri`, receipts, output identity, and
captured CLI remove userinfo and known AWS, Google, Azure, and generic authentication
parameters, while retaining semantic selectors such as object versions, generations,
series, and fragments. Credential rotation therefore does not rename the output.

Query credentials must be valid for the Zarr prefix and all child metadata/chunk objects.
A presigned S3 or GCS URL for one object usually cannot authorize the whole hierarchy;
prefer native `s3://` or `gs://` credential-provider configuration in that case.

---

## 6. What `embed` actually does

For one slide, in order:

1. **(skip check)** If `--receipts-dir` is set and the source-bound target is already
   complete for this exact config, the run returns `skipped` immediately (see §7).
2. **Read metadata** via the `omezarr` reader (NGFF version, MPP, pyramid).
3. **Build the grid** at the **exact target MPP** (below).
4. **Segment** the tissue (unless `--no-seg`) on a low-res level and keep only the
   grid cells whose tissue fraction ≥ `--tissue-threshold`.
5. **(optional)** Render the thumbnail/overlay if `--emit-thumbnail`.
6. **Embed**, decode-once: for each batch, read the patches **once**, then run
   **every** requested model on that shared batch.
7. **Write** the `*.embeddings.zarr` store + provenance, optional sidecars.
8. **(optional)** Write a `complete` receipt if `--receipts-dir` is set.

### Exact MPP (why patches are comparable)

Foundation-model embeddings are scale-sensitive, so every patch is sampled at the
**exact** MPP you ask for - not "close enough". The patcher:

- picks the **nearest finer-or-equal** pyramid level (native MPP ≤ target), so it
  only ever **downsamples** - it never invents detail by upscaling;
- reads `read_px = round(patch_px × resample)` pixels at that level
  (`resample = target_mpp / level_mpp ≥ 1`);
- downsamples that read to **exactly** `patch_px` (the stored patch is `patch_px`
  at the target MPP, independent of any model's input size - each model then
  resizes the patch to *its* input size internally).

If a level's native MPP is within `--mpp-tolerance` (default 0.1%) of the target,
it is read **natively** with no resize (the common case for pyramids built at the
target resolution), and the recorded `achieved_mpp` is that level's native MPP.
Otherwise the patch is resampled and `achieved_mpp == target`. `--snap-to-level`
forces a native level read (faster, MPP = that level); `--allow-upsample` permits
going finer than level 0. `raw2features info` prints the exact plan.

### Float source pixels

OME-NGFF float arrays do not necessarily declare whether RGB values use a normalized
`[0, 1]` range or a byte-like `[0, 255]` range. The OME-Zarr reader therefore chooses
one convention when the slide is opened and reuses it for every patch and pyramid level;
it never rescales one patch differently because that patch happens to be locally dark.
The decision uses the array fill value and a deterministic 3×3 lattice of level-0
chunks (up to the first three channels): any finite observed value above 1 selects the
byte-like convention, otherwise the slide is treated as normalized. This bounded sample
avoids scanning an entire WSI at open. `read_region_channels` remains native-dtype and
unscaled for multiplex models.

### Segmentation

The default `otsu` segmenter runs Otsu-on-saturation + morphology on a low-res
level (~8 µm/px) to produce a tissue mask, and the patcher keeps a grid cell only
if its tissue fraction ≥ `--tissue-threshold`. `--no-seg` tiles the whole slide
(and omits the `mask` array). Other segmenters (incl. the opt-in deep `grandqc`)
are pluggable via the `segmenters` seam - see
[`SEGMENTATION.md`](SEGMENTATION.md) for the methods and the GrandQC QC layer.

### Multi-model fan-out (decode-once)

With several `-m` flags (`-f` is an alias), each patch is **read and decoded once**
and reused across every model in the batch - the slow part (IO + decode) is not
repeated per model.
Each model applies **its own** preprocessing (mean/std/input size, sourced from
its model card), and writes into its own `features/<model>` array.

### In-process multi-GPU (`--devices`, opt-in)

By default the pipeline runs on the single `--device` (CPU or one GPU) and is fully
portable. `--devices` is an **opt-in** runtime flag that spreads work across several
devices in one process; it never changes the embeddings (it is excluded from the
config hash), so a store built with it is identical to a single-device one and
resumes interchangeably.

- **`embed` (one slide) - patch-parallel, latency mode.** `embed … --devices
  cuda:0,cuda:1` shards the slide's patches into contiguous blocks, embeds each
  block on its own device (one model copy + reader per device), and **gathers the
  features back in exact coord order** - so the output is byte-identical to a
  one-GPU run, just faster for a single large slide.
- **`embed-many` (a shard of slides) - slide-parallel, throughput mode.** `embed-many
  … --devices cuda:0,cuda:1` runs one warm worker per device, each pulling slides
  off a shared queue; every slide is embedded **entirely on one device**, so its
  output equals a one-GPU run. This is the path for a multi-GPU workstation **without
  a scheduler** (a SLURM array already parallelises one GPU per task - see §9).
- **`benchmark … --devices …`** measures the patch-parallel single-slide path
  (wall-time / patches/s stay accurate; the per-stage breakdown is single-device
  only).

Equivalence is exact on CPU and across distinct GPUs (each patch is processed on one
device exactly as in the single-device path). Listing the *same* GPU twice
(`cuda:0,cuda:0`) is only a single-GPU test convenience and may differ by ~1 ULP due
to shared-stream float reordering.

---

## 7. Re-runs & idempotency

> This section explains the already-embedded behaviour: how re-runs detect existing work and what they do
> with it. Worth reading before a cohort run.

### Adding a model / resuming is automatic (store-aware)

`embed` inspects any existing `<slide>.embeddings.zarr` before doing work, so
re-runs are **additive and idempotent - no `--receipts-dir` required**:

- A requested model already present and output-validated against its current model
  fingerprint is **skipped**.
- A **missing** model is embedded and **appended in place**: `features/<model>` is
  added next to the existing arrays when it shares that grid's geometry. Existing
  models and `coords` are left untouched.
- A requested geometry not already in the store is added as another
  `grids/<key>/` group. A difference in `--mpp`, `--patch-size`, segmentation, or
  another geometry field is **not** an error and does not wipe existing grids or
  mix incompatible rows in one grid.
- If **all** requested models and auxiliary outputs are already present, no patch
  features are recomputed (a model-only run returns `status: skipped`).

Use `--force` only when you intentionally want to rebuild the whole store from
scratch.

```bash
raw2features embed slide.zarr out/ -m uni              # writes features/uni
raw2features embed slide.zarr out/ -m virchow2         # ADDS features/virchow2; uni untouched
raw2features embed slide.zarr out/ -m uni -m virchow2  # both present -> nothing to do
raw2features embed slide.zarr out/ -m conch_v1_5       # ADDS its 0.5/512 grid; other grids untouched
```

The shared extraction/storage grid is identified by a **`grid_hash`**, recorded in
the header. Model-specific settings such as resolved AMP live in each output
fingerprint instead, so changing them replaces that model in the same grid. "Present"
is strict: a model counts only if `features/<model>` has the
current expected dimension and a matching output fingerprint in both the array and
grid header, is fully finite, has no unwritten (all-zero) tail, and has
`len(coords) == n_patches`. The fingerprint covers the effective weights,
preprocessing, pooling, resolved AMP, and loader construction. A truncated, damaged,
legacy-unfingerprinted, or stale-contract array is re-embedded; unrelated model arrays
and coordinates remain untouched. A derived `slide/<model>` fingerprint includes its
patch input fingerprint, so stale slide vectors are recomputed too.

### Receipts - the fast path for cohort runs

`--receipts-dir DIR` adds a cheap short-circuit on top of the store inspection
above. After confirming that any existing target store belongs to the requested
source, `embed` checks for a validated `complete` receipt for **this exact
configuration** (including the model set), source, and output target. If found it
returns `skipped`; otherwise it processes and writes a `complete` receipt at the
end. This is what the SLURM array uses to skip finished slides cheaply.
Completeness is validated against the **actual output store**, never just the
receipt file.

The receipt's config hash covers the **content-affecting** settings *and* the
model set:

```
reader, models (order-independent), segmenter / --no-seg, --mpp, --patch-size,
--source-mpp (when set), --step, --tissue-threshold, --features-dtype,
--stain-norm, --snap-to-level, --mpp-tolerance, --allow-upsample, --amp
```

Runtime-only knobs do **not** change the hash:
`--device`, `--devices`, `--batch-size`, `--read-workers`, `--read-block`,
`--compile`, `--emit-geojson`, `--emit-thumbnail`, `--thumbnail-mpp`, `--max-px`,
`--slide-encoder`, `--qc`, `--qc-stain-norm`, `--qc-artifact-mpp`,
`--output-zarr-format`, `--force`.
`--force` is the exception for control flow: it deliberately bypasses skipping and
rebuilds the target without changing the content identity.

### `verify` - the standalone skip check

`raw2features verify <slide> --receipts-dir DIR --out-dir OUT <same content flags>`
runs exactly the same completeness check and exits `0` (complete) or `1`
(incomplete). It is how the SLURM array skips finished slides. A receipt is trusted
only when its slide ID, credential-free source, requested output target, and the
actual store's root/grid source provenance all agree. **Pass `--out-dir` and the same
content-affecting flags you pass to `embed`**, or the check is not target-aware (or
the config hash will not match).

```bash
raw2features verify slide.zarr --receipts-dir receipts/ --out-dir embeddings/ -m uni -m resnet50 --mpp 1.0 --quiet
```

### Forcing a rebuild

Pass `--force` to ignore an existing store and rebuild it from scratch (all
requested models recomputed, the store overwritten). It also bypasses a valid
receipt's fast path. Deleting the output store (and any receipt) has the same effect.

### Auxiliary outputs are produce-if-missing

An explicit `--emit-thumbnail`, `--emit-geojson`, `--qc`, or `-s/--slide-encoder`
request is checked against the matching grid even when its patch features are already
complete. Missing outputs are produced without re-embedding patches; existing valid
outputs are left alone. These runtime products do not change the patch-grid identity or
receipt content hash. The standalone [`thumbnail`](#8-thumbnail--previews--qc-overlays)
and `slide-embed` commands remain convenient when you only want to add those products.

In a multi-grid invocation, a slide encoder runs on the grid containing its required
patch encoder (for example, TITAN uses the `conch_v1_5` grid). Model-agnostic pooling
encoders run on each requested grid. Standalone `slide-embed` can infer an unambiguous
grid from the required patch model; otherwise pass `--grid` (and `--patch-model` when
needed).

---

## 8. `thumbnail` - previews & QC overlays

Two ways to get a thumbnail; both are **optional and off by default** so they
never slow a large feature-extraction run.

**Standalone** (run before or after the heavy embed run; one cheap coarse read):

```bash
raw2features thumbnail slide.zarr out/                 # plain preview
raw2features thumbnail slide.zarr out/ --overlay       # + tissue mask + kept-patch grid
```

**Inline** with embeddings (piggybacks on the already-open + segmented slide):

```bash
raw2features embed slide.zarr out/ -m uni --emit-thumbnail
```

### What it does / sizing

A thumbnail is an RGB overview read from a **coarse pyramid level** (never the
full-resolution level 0). By **default it renders at the segmentation MPP**
(8 µm/px), chosen by the same nearest-level rule the segmenter uses. That single
choice gives two things at once:

- the thumbnail, the tissue mask, and the patch grid share a pixel grid, so the
  `--overlay` needs **no resampling** - it is a faithful QC view ("did
  segmentation + patching do the right thing on this slide?");
- it is also a sensible size (≈3 k px wide for a 0.25 µm/px slide).

Override the resolution with `--thumbnail-mpp <µm/px>` or `--max-px <N>` (cap the
longest side; overrides `--thumbnail-mpp`). Pyramid-level indices are never
exposed - they aren't portable across slides, and the whole tool derives the
level from MPP.

For the standalone `--overlay`, the segmenter + patcher are re-run (cheaply) so
the overlay reflects the grid params you pass (`--mpp`, `--patch-size`, `--step`,
`--tissue-threshold`, `--segmenter`).

### Output

```
<slide_id>.thumbnail.png            # plain overview
<slide_id>.thumbnail.overlay.png    # tissue tint + kept-patch outlines (with --overlay / --emit-thumbnail)
```

Inline (`--emit-thumbnail`) also records the thumbnail metadata in the store's
`.zattrs` (`raw2features.thumbnail`).

---

## 9. Cohort runs on SLURM

`slurm/` ships a one-slide-per-task array, a per-batch ("shard") array, and a pre-flight
checker - see [`slurm/README.md`](../slurm/README.md) for a plain-language overview and
which script to use. Each task is **idempotent**: it `verify`s the slide first and skips if
already complete, so re-submitting the same array safely resumes only the missing/failed
slides.

Before `embed-many` shards or loads models, it requires every resolved cohort input to
derive a unique output ID. Ordinary local IDs remain basename-based for v0.1
compatibility, so same-named slides from different directories must be renamed or run
as separate commands with different output directories. Repeated identical manifest
rows are also rejected; a per-output lock across independent invocations is not implied.

```bash
cd raw2features && uv sync --extra zarr --extra image --extra torch --extra models
# Add any model-specific extras / pinned git packages from docs/MODELS.md.
export SLIDE_DIR=/path/to/raw            # dir of *.zarr stores (top-level only)
export OUT_DIR=/path/to/embeddings
export MODELS="uni resnet50"             # space-separated; each becomes a -m flag
export HF_TOKEN=hf_...                    # only if MODELS includes a gated model

bash slurm/preflight.sh                  # validates venv/arch, models/HF, slide count
mkdir -p logs                            # SLURM opens --output here before the job runs
shopt -s nullglob
SLIDES=("$SLIDE_DIR"/*.zarr)
shopt -u nullglob
N=${#SLIDES[@]}
(( N > 0 )) || { echo "no *.zarr slides found" >&2; exit 1; }
sbatch --array=0-$((N-1))%64 --export=ALL,SLIDE_DIR,OUT_DIR,MODELS,HF_TOKEN slurm/embed_array.sbatch
```

Notes:

- Set `--partition` in `embed_array.sbatch` to your cluster's GPU partition (the
  template ships a `gpu` placeholder).
- Lower `BATCH_SIZE` on smaller GPUs (UNI2-h is a ViT-H and is memory-hungry).
- The slide list is a **top-level glob** (`$SLIDE_DIR/*.zarr`) - never a recursive
  `find` into the stores - and is sorted byte-order (`LC_COLLATE=C`) so task *i*
  maps to the same slide on every node.
- The pre-flight catches the most common cohort-killer: a venv built on the wrong
  architecture (it imports Torch to check).
- **No scheduler?** On a multi-GPU workstation, `embed-many … --devices
  cuda:0,cuda:1,…` is the equivalent of the array: one warm worker per GPU, slides
  distributed across them, per-slide receipts for idempotent resume (see _In-process
  multi-GPU_ in §6).

---

## 10. Output reference

```
<slide_id>.embeddings.zarr/
├── .zattrs                     # root header: source, provenance + a grids index
└── grids/<key>/                # one per geometry (mpp, patch_px) - usually just one
    ├── .zattrs                 # this grid's full header (below)
    ├── coords/                 # (N,2) int32  level-0 (x,y) top-left of each patch
    ├── grid_index/             # (N,2) int32  (row,col) on the tiling grid
    ├── mask/                   # (n_rows,n_cols) uint8  fraction of each cell that is tissue; /255 for [0,1] (omitted with --no-seg)
    ├── features/<model>/       # (N,dim) float16  one array per extractor
    └── slide/<model>/          # (1,dim) float32  optional slide-level vectors

<slide_id>.patches.geojson          # optional (--emit-geojson): per-patch polygons for QuPath
<slide_id>.thumbnail.png            # optional (--emit-thumbnail / thumbnail cmd)
<slide_id>.thumbnail.overlay.png    # optional QC overlay
<slide_id>.<key>.patches.geojson              # corresponding sidecar for each additional grid
<slide_id>.<key>.thumbnail.overlay.png        # corresponding overlay for each additional grid
```

A store nests its patch sets under `grids/<key>/` (one child per extraction geometry -
`mpp0.5_px224`, etc.); a one-geometry run is just a single grid. Open one grid with the
reader helper (`raw2features.core.store.open_grid(store)` returns the sole grid, or pass a
key), and `export`/`info` default to the sole grid (a multi-grid store wants `--grid`). The
first-created grid keeps the backward-compatible unsuffixed GeoJSON/overlay names; later
grids use their key as shown above. The plain thumbnail is shared by the slide.

**Spatial-provenance contract (within a grid).** Row `i` refers to the same patch across
**every** array of that grid: `coords[i]`, `grid_index[i]`, and `features/<model>[i]` are
aligned 1:1 in identical order. Combined with `patching.level0_patch` (the patch's level-0
pixel extent), any embedding is invertible to the exact slide region
`[x, y, x+level0_patch, y+level0_patch]` at level 0.

**Each grid's `.zattrs["raw2features"]`** records the full, self-describing header:
`schema_version`, `provenance` (version, CLI, git SHA, host, GPU), `thumbnail` (if emitted),
`source` (uri, ngff_version, reader, mpp_level0, dimensions/downsamples, source axes,
per-axis physical scale, and source translation/origin),
`patching` (target/achieved MPP, patch_px, read_level, read_px, resample, level0_patch,
level0_step, n_patches, grid_shape, `coords_convention: level0_xy`), `segmentation`, and
per-model `models.<name>` (source, embedding_dim, input_size, pooling, mean, std,
`transform_source_url`, license, gated, weights_sha256). The **root** `.zattrs` carries the
shared `source`/`provenance` plus a `grids` index (`key → {mpp, patch_px, n_patches,
models, grid_hash}`) for discovery without opening each grid.

Read it back:

```python
from raw2features.core.store import open_grid

# Omit the key only when the store contains exactly one grid.
g = open_grid("out/<slide_id>.embeddings.zarr", key="mpp0.5_px224")
print(g["features/uni"].shape)          # (N, 1024)
print(dict(g.attrs)["raw2features"]["patching"]["achieved_mpp"])
```

---

## 11. FAQ / gotchas

**`embed --emit-thumbnail` produced no thumbnail.** A missing thumbnail should be
produced even when the patch arrays are complete, while an existing valid thumbnail is
left unchanged. Check the command's error and output path; in a multi-grid store,
non-primary overlays have the grid key in their filename. The standalone
`raw2features thumbnail … --overlay` command is also available when only a preview is
needed.

**What happens when I add a model?** A model with the same extraction geometry is
appended to that grid. A model with a different geometry gets a new grid. Existing
valid arrays are preserved; only a missing, damaged, legacy-unfingerprinted, or
stale-fingerprint model output is recomputed.

**`verify` always says incomplete.** It must be given the **same content-affecting
flags** as the `embed` that produced the store (same `-m` models, `--mpp`,
`--patch-size`, etc.) so the config hash matches. Pass the same output directory as
`--out-dir` so the receipt is also bound to the intended target store.

**`import torch` / model load fails on the cluster.** The venv was built for a
different CPU architecture. Re-run `uv sync` on the target machine.

**Gated model 401 / access error.** You need both a token (`--hf-token` /
`HF_TOKEN` / cached login) **and** access granted on the model's HuggingFace page.
Open models (`resnet50`, `dinov2`) need neither. Prefer `HF_TOKEN` or `hf auth login`
so the token never appears in shell history or a process listing.

**Embeddings are `float16`.** Default storage dtype (inode/byte-light). Use
`--features-dtype float32` if you need full precision on disk.

**`achieved_mpp` isn't exactly my `--mpp`.** A level within `--mpp-tolerance`
(0.1%) of the target is read natively (no resize), so `achieved_mpp` is that
level's native MPP. Tighten `--mpp-tolerance 0` to force exact resampling, or use
`--snap-to-level` to read a level natively on purpose.
