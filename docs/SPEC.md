# raw2features embeddings store - format specification

**Format version 0.1** - the **read contract** below (the `grids/<key>/` nesting + per-grid
header, the 1:1 coords↔features invariant within a grid, `coords` units, and "tolerate
unknown keys") is stable; the full on-disk layout and the coordinate convention may still
change while raw2features itself is pre-1.0 (`0.x`). See [Stability](#stability). A
machine-readable [JSON Schema](src/raw2features/schema/embeddings_store-0.1.schema.json)
ships with each version and is the normative definition of each grid's header.

This document defines the on-disk format raw2features writes: a per-slide
`<slide_id>.embeddings.zarr`. It is the contract other tools read against. You can
implement a reader from this document alone, with a plain Zarr library and no
dependency on raw2features.

The keywords MUST, SHOULD and MAY are used as in RFC 2119.

## Stability

raw2features is alpha (`0.x`); this format ships with it on **one version line** - package
`0.x`, format `0.x`, schema `0.x` all move together (so today's store says
`schema_version: "0.1"`). To let readers depend on the format now during the 0.x line, the stable parts are scoped explicitly:

- **Stable (the read contract).** Patch sets live under `grids/<key>/`, each a complete,
  self-describing grid carrying its header under that group's `raw2features` attr; **within
  a grid** the **1:1 invariant** (row `i` of every `features/<model>` ⇔ `coords[i]`) holds;
  `coords` are level-0 `(x, y)` pixels carrying `units = "level0_px"`; and a reader **MUST
  tolerate keys/arrays it does not recognise**. Code written against these will keep
  working across `0.x`.
- **Still revisable during `0.x`.** The exact on-disk array layout, the set of header
  keys, and the coordinate *convention* (e.g. if/when the store grows a richer, physical
  coordinate frame - see [Coordinates](#coordinates--relocatability)) may change between
  `0.x` releases. Each such change bumps `schema_version` and ships an updated JSON Schema
  (`embeddings_store-<version>.schema.json`); the read contract above is held across them.

## Scope

The store holds **patch-level feature vectors** extracted from one whole-slide image,
the **coordinates** that place each vector on the slide, and the **provenance** needed to
reproduce and trust them. It is a *derived* artifact that references its source image; it
does not contain the image pixels. It is not a segmentation format, a training format, or
a viewer format - those are downstream of it (see [Related formats](#related-formats)).

It is deliberately a separate store, not data written back into the source OME-Zarr. See
[Why a separate store](#why-a-separate-store).

## Layout

A store is a Zarr group (v2 by default; v3 permitted). Patch sets live **uniformly** under
`grids/<key>/` - one child group per extraction geometry `(mpp, patch_px)`. There is **no
flat single-grid special case**: a one-geometry store is just a `grids/` with one child.
Tree:

```
<slide_id>.embeddings.zarr/                  # Zarr group
├── (group attrs)["raw2features"]            # ROOT header: shared source/provenance + grids index
└── grids/
    └── <key>/                               # e.g. mpp0.5_px224 - one geometry (mpp, patch_px)
        ├── (group attrs)["raw2features"]    # this grid's complete header - see below
        ├── coords          (N, 2) int32     # required - level-0 (x, y) top-left of each patch
        ├── grid_index      (N, 2) int32     # optional - (row, col) of each patch in the grid
        ├── mask        (rows, cols) uint8   # optional - fraction of each cell that is tissue; /255 for [0,1]
        ├── features/                        # required group
        │   ├── <model>     (N, dim) f16     # one array per model; 1:1 with this grid's coords
        │   └── ...
        └── slide/                           # optional - slide-level vectors from this grid
            └── <model>     (1, dim) f32
```

A **grid key** is a human-readable label `mpp{mpp}_px{patch_px}` (e.g. `mpp0.5_px224`,
`mpp1_px256`); the authoritative geometry lives in the grid's header, so the key is only an
addressable handle. Most stores have a single grid; a store grows a second grid when a model
with a different recommended geometry is requested (e.g. CONCH at 0.5/448 alongside UNI at
0.5/224), or the same model is extracted at several scales. Every array carries a `role`
attribute (`coords`, `grid_index`, `tissue_mask`, `features`, `slide_embedding`, `qc`).
`coords` carries `units = "level0_px"`; each `features` array carries `model = <name>`.

### The 1:1 invariant

**Within a grid**, row `i` of every `features/<model>` array is the embedding of the patch
whose top-left level-0 pixel is `coords[i]`, of side `patching.level0_patch` pixels. A
grid's feature arrays and its `coords` MUST have the same length `N` and the same row order.
Different grids carry different geometries, so their `N` and `coords` differ.

## Required arrays (per grid)

Under each `grids/<key>/`:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `coords` | (N, 2) | int32 | Patch top-left in **source level-0 pixels**, `(x, y)` = `(column, row)`, origin top-left. |
| `features/<model>` | (N, dim) | float16¹ | Patch embeddings for one model. One sub-array per model. |

¹ Default `float16`; the producer MAY use another float dtype, recorded per model.

## Optional arrays (per grid)

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `grid_index` | (N, 2) | int32 | `(row, col)` of each patch in the regular grid. |
| `mask` | (rows, cols) | uint8 | Fraction of each grid cell that is tissue (the mean of the segmentation mask over the cell), stored as `0-255` - divide by 255 for the `[0,1]` value. Index a kept patch's cell via `grid_index[i]`. Absent when tiling without segmentation. |
| `slide/<model>` | (1, dim) | float32 | A slide-level vector (e.g. from a slide encoder or pooling), derived from this grid's features. |

## Optional QC layer (per grid)

A grid MAY carry per-patch quality-control scores under `grids/<key>/qc/<tool>/` - one
subgroup per QC tool (e.g. `qc/grandqc/`). Every array is 1:1 with the grid's `coords` and
carries `role = "qc"`:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `scores` | (N, k) | float16 | Per-class fraction in `[0, 1]`; the `k` class names are on `scores.attrs["classes"]`. |
| `label` | (N,) | uint8 | Optional hard label; an NGFF-style `image-label` legend (label-value → name) is on the `qc/<tool>` group attrs. |
| `usable` | (N,) | uint8 | Optional keep/drop flag at a recorded threshold. |

The `qc/<tool>` group records the producing tool, version, the MPP it ran at, and any
threshold; the grid header mirrors this under a `qc` block. The convention is **values, not
schema** - tool and class names are *data*, so a new scorer adds a `qc/<tool>/` subgroup with
no format change. The one rule `validate-store` enforces is that every array under any `qc/`
group is length `N` and carries `role = "qc"`.

A QC tool whose native output is a coarse per-pixel raster (the common case - e.g. GrandQC)
MAY also store it once per slide, grid-independent, under a root `qc_raster/<tool>/`
(`artifact` / `tissue` `(h, w)` uint8 at the tool's MPP, with an `image-label` legend); the
per-patch `scores` are then derived by projecting each patch's footprint into it. This root
layer is **reserved but not yet produced**: the shipped GrandQC producer derives per-patch
`scores` directly and does not persist the raster. QC models and weights are
external/optional - raw2features ships the layout, not the scorers.

## Header

Headers live under the `raw2features` group attr at two levels: a **grid header** on each
`grids/<key>/` (the authoritative, self-describing record of that grid) and a lightweight
**root header** on the store root (shared slide-level fields + a discovery index).

### Grid header (`grids/<key>/`)

A JSON object; the normative definition is the packaged JSON Schema (see
[Conformance](#conformance)). Required keys:

- `schema_version` - string, `"0.1"` for this spec.
- `source` - the slide this was extracted from:
  `uri`, `slide_id`, `mpp_level0` (µm/px at level 0), `ngff_version`, `reader`,
  `level_dimensions` (`[[w, h], …]`), `level_downsamples`.
  `uri` is safe-to-persist provenance: plain local paths use their absolute `file://`
  form, while remote URIs omit userinfo and recognised authentication parameters but
  retain semantic selectors. It is therefore an identity/provenance URI and may need
  credentials supplied separately before it can be reopened.
  Optional coordinate-frame self-description (present when the reader can supply it):
  `axes` (the source NGFF axis order, e.g. `["c", "y", "x"]`), `axis_units` (per-axis
  unit string), `scale_um` (per-axis level-0 pixel size in µm, e.g.
  `{"x": 0.25, "y": 0.25}` - the faithful value when `mpp_level0`'s x/y mean hides
  anisotropy), and `level0_translation_um` (the source's level-0 translation/origin in
  µm, or `null` when it carries none). These are recorded **values**, not an NGFF
  transform object; they let a consumer re-express `coords` in the source's physical
  frame (see [Coordinates](#coordinates--relocatability)).
- `patching` - **this grid's** geometry: `target_mpp`, `achieved_mpp`, `patch_px`,
  `level0_patch` (patch side in level-0 px), `level0_step`, `read_level`, `step_out_px`,
  `n_patches`, `grid_shape` (`[rows, cols]`), `coords_convention` (`"level0_xy"`).
- `models` - map of model name → `{ source, embedding_dim, input_size, pooling, mean,
  std, interpolation, license, gated, weights_sha256, weights_revision,
  transform_source_url, inference_amp, doi }` for the models in **this grid**. These pin
  the exact preprocessing and weights per model. `weights_revision` is the immutable
  HuggingFace commit the loader pins the download to (or a torchvision weights enum such as
  `IMAGENET1K_V2`, which is itself immutable, for which `weights_sha256` is `null`; or a
  release tag for weights pinned by a stable URL + `weights_sha256`); `weights_sha256` is
  the weight file's digest, recorded in every output and verified before load for the
  bare-checkpoint and URL-pinned models - so each feature set is traceable to the exact
  weights. `doi` is a resolvable DOI for the model's paper (`null` for an open-weights
  release with no paper).
- `grid_hash` - a hash of the patch geometry. Two grids with the same `grid_hash` share an
  identical grid, so feature arrays from different runs are row-comparable.
- `provenance` - `raw2features_version`, `created_utc`, `cli`, `git_sha`, `host`, `arch`,
  `platform`, `python`, and `gpu` when applicable. Secret option values and credentials
  embedded in URIs are redacted from `cli`.

Optional grid keys: `segmentation` (segmenter name + parameters), `thumbnail` (thumbnail
metadata when one was written), `slide_embeddings` (provenance for any `slide/<model>`).

### Root header (store root)

A JSON object carrying `schema_version`, the shared `source` / `provenance` /
`segmentation`, and a **`grids` index** - a map `key → { target_mpp, achieved_mpp,
patch_px, level0_patch, n_patches, models, grid_hash }` summarising each grid for discovery
without opening it. An explicit `--config` run also records the plan under
`job.geometry_config` for replay. The authoritative geometry is always the grid header; the
index is a convenience.

A reader MUST tolerate keys it does not recognise.

## Coordinates & relocatability

`coords` are in the **source slide's level-0 pixel grid** (the OpenSlide / CLAM
convention), so an embedding is placeable on the slide independently of which pyramid
level was actually read. To map a patch onto a downscaled level `L`, divide by
`source.level_downsamples[L]`.

To map a patch to **physical space**, multiply by `source.mpp_level0` (µm/px). This is
exact **only when the source carries no non-zero translation and its origin is `(0, 0)`** -
the common case for pathology pyramids. When the source declares an origin/translation,
that simple multiply lands in a pixel frame offset from the source's true physical frame:
use `source.scale_um` (per-axis, so anisotropy is honoured) and add
`source.level0_translation_um` to recover the source's physical coordinates. raw2features
**records** these values but does not bake them into `coords` (which stay level-0 pixels,
origin top-left) - keeping the pixel grid that the pathology-FM ecosystem reads, while
preserving everything needed to relocate into physical space.

Together with `source.uri` (and any stable identifier it carries - a DOI, archive
accession, or checksum), this makes the store relocatable to its source even when that
source is remote or read-only.

> The coordinate frame is described with plain **values** (axes, units, per-axis scale,
> translation) rather than an NGFF `coordinateTransformations` object. The community's
> coordinate-systems work (OME-NGFF RFC-5) is still in review; recording the ingredients
> now means a future NGFF-native emitter is a re-encode, not a re-extraction, without
> committing the store to a shape the spec has not settled.

## Versioning

`schema_version` is `MAJOR.MINOR` and tracks the 0.x release line (one version line -
package, format and schema move together). Adding optional arrays or header keys is a
MINOR bump; removing or repurposing anything is a MAJOR bump. Each version ships its
JSON Schema as `embeddings_store-<version>.schema.json`. Readers MUST check
`schema_version`, MUST ignore unknown keys/arrays, and SHOULD warn on an unknown MAJOR.

## Conformance

Each **grid header** is defined normatively by a **JSON Schema**
([`embeddings_store-0.1.schema.json`](src/raw2features/schema/embeddings_store-0.1.schema.json)):
its `required` set is the stable read contract, everything else is optional, and
`additionalProperties: true` makes it forward-compatible. Any JSON-Schema validator can
check a grid's header against it - no dependency on raw2features.

`raw2features validate-store <path>` is the reference implementation: for **every grid** it
applies that schema to the grid header **and** adds the array-level checks a JSON Schema
cannot express (array shapes/dtypes, the `role` attrs, and the 1:1 `coords`↔`features`
invariant within the grid), prefixing each grid's violations `grids/<key>:`. It works on a
local path or a remote URL, and the test suite runs it against real pipeline output, so the
format, the schema and this document stay in sync.

## Minimal example

A conforming store with one grid, one model, two patches, no segmentation:

```
slideX.embeddings.zarr/
  attrs["raw2features"] = {                          # ROOT header (discovery)
    "schema_version": "0.1",
    "source": {…}, "provenance": {…},
    "grids": {"mpp0.5_px224": {"target_mpp": 0.5, "patch_px": 224, "n_patches": 2,
                               "models": ["uni"], "grid_hash": "…"}}
  }
  grids/mpp0.5_px224/
    attrs["raw2features"] = {                        # GRID header (authoritative)
      "schema_version": "0.1",
      "source":   {"uri": "…/slideX.ome.zarr", "slide_id": "slideX",
                   "mpp_level0": 0.25, "ngff_version": "0.4", "reader": "omezarr",
                   "level_dimensions": [[40000, 30000]], "level_downsamples": [1.0],
                   "axes": ["c", "y", "x"], "scale_um": {"x": 0.25, "y": 0.25},
                   "level0_translation_um": null},
      "patching": {"target_mpp": 0.5, "achieved_mpp": 0.5, "patch_px": 224,
                   "level0_patch": 448, "level0_step": 448, "read_level": 0,
                   "n_patches": 2, "grid_shape": [1, 2], "coords_convention": "level0_xy"},
      "models":   {"uni": {"embedding_dim": 1024, "input_size": 224, "license": "…",
                           "doi": "10.1038/s41591-024-02857-3", …}},
      "grid_hash": "…", "provenance": {"raw2features_version": "…", …}
    }
    coords        = [[0, 0], [448, 0]]               # int32
    features/uni  = <(2, 1024) float16>
```

## Why a separate store

Embeddings are not written back into the source OME-Zarr, for one decisive reason and
several supporting ones:

- **The source is often read-only.** Feature extraction commonly runs against public
  archives (e.g. the EBI BioImage Archive over S3) or shared, immutable cohort datasets.
  You cannot write into them. A separate store is the only thing that works everywhere.
- **Different lifecycles.** The source is raw, immutable, often citable; embeddings are
  derived and regenerated as models improve. They should not share a lifecycle.
- **Many extractors, one slide.** Several models (and people) extract from the same slide;
  a separate store per run avoids mutating a shared artifact.

The store is, however, shaped to sit *beside* an OME-NGFF image (it carries no pixels and
references the source by coordinate), so it could be embedded as an NGFF group where the
source is owned and that is desired.

## Related formats

- **Source image** - OME-Zarr / OME-NGFF. Referenced by `source`, never copied.
- **SpatialData** (scverse) and **HDF5** (TRIDENT / STAMP) - downstream exports produced
  by `raw2features export-spatialdata` / `export-h5`. One-way; this store is the source of
  truth.
