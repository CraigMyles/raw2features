# Interoperability - exporting embeddings to other ecosystems

raw2features writes a compact, inode-light per-slide store - *inode-light* meaning few
files (chunks coarsely), so it stays under the file-count limits of parallel/HPC
filesystems - (`<slide_id>.embeddings.zarr`: `coords` + `grid_index` + `mask` +
`features/<model>` + a provenance header). That is the canonical output. **Exporters** are post-hoc
converters that re-emit an existing store in another community's format - they never
recompute embeddings, and their heavy dependencies stay optional.

| Target | Command | Extra | Reads in |
|---|---|---|---|
| scverse **SpatialData** | `raw2features export-spatialdata` | `[spatialdata]` | squidpy, napari-spatialdata, scanpy |
| pathology-MIL **HDF5** | `raw2features export-h5` | `[h5]` | TRIDENT, CLAM, TITAN, THREADS, STAMP |
| QuPath tiles (GeoJSON) | `from raw2features import write_patches_geojson` (lib) | `[image]` | QuPath |

## SpatialData

```bash
pip install "raw2features[spatialdata]"
raw2features export-spatialdata SLIDE.embeddings.zarr            # -> SLIDE.spatialdata.zarr
raw2features export-spatialdata SLIDE.embeddings.zarr OUT.zarr --model uni --model conch
raw2features export-spatialdata SLIDE.embeddings.zarr --geometry circle --overwrite
```

The output `<slide>.spatialdata.zarr` is an OME-NGFF-aligned [SpatialData](https://spatialdata.scverse.org)
store:

- **`shapes["tiles"]`** - one square polygon per patch, authored in **level-0 (full-res)
  pixel** coordinates. Tiles carry two coordinate systems: `global` (pixel space, an
  `Identity` transform) and - when the source is physically calibrated - `micrometers`, a
  `Scale` (plus a `Translation` for a non-zero source origin) that maps pixels to **µm**.
  Align across slides/modalities in real units via the `micrometers` system; the pixel→µm
  scale is also in `uns["raw2features_export"]["micrometers_per_pixel"]`.
  `--geometry circle` instead writes inscribed circles (the HEST / Visium "spot"
  convention).
- **`tables["table"]`** - an [AnnData](https://anndata.readthedocs.io) with one row per
  tile, linked to `tiles` via `region`/`instance_id`:
  - `obsm["X_<model>"]` - the patch embedding for each model (`X_uni`, `X_conch`, …),
    `float32`. This holds the embeddings; `X` is intentionally empty (`n_var = 0`).
  - `obsm["spatial"]` - `(N, 2)` tile **centres** in full-res pixels (squidpy / HEST
    compatibility; this is what `to_legacy_anndata` reconstructs from).
  - `obs` - `x`, `y` (level-0 top-left px, CLAM/Trident convention), `array_row`,
    `array_col`, `level`, `mpp`, `tissue_frac`, plus the link columns.
  - `uns["raw2features"]` - the full provenance header (per-model licence + card,
    segmentation, patching, source). `uns["slide_embeddings"]` carries any slide-level
    vectors.
  - **Multiplex only** - `uns["raw2features_panel"]` is a tidy, **queryable** table (one
    row per kept channel: `model`, `channel`, `channel_index`, `marker`, `marker_id`)
    recording how each source channel was matched to the model's marker vocabulary. A
    compact per-model coverage summary
    (`kept` / `dropped` / `unmatched` + the `vocabulary` revision) sits in
    `uns["raw2features_export"]["panel"]`. (Without this, the per-channel map would be
    lost: anndata's zarr writer stringifies a raw list-of-dicts into unusable blobs.)

### Why this schema

It is the intersection of two existing "patches → SpatialData" converters
that agree - HEST's
[`HESTData.to_spatial_data`](https://github.com/mahmoodlab/HEST/blob/main/src/hest/HESTData.py)
and spatialdata-io's
[`from_legacy_anndata`](https://github.com/scverse/spatialdata-io/blob/main/src/spatialdata_io/converters/legacy_anndata.py)
- while borrowing CLAM/Trident `obs`/`uns` field names so the metadata is legible to the
dominant pathology-FM toolchain. The result **round-trips** through
`spatialdata_io.experimental.to_legacy_anndata` and loads directly in squidpy and
napari-spatialdata.

### Consuming it

```python
import spatialdata
sdata = spatialdata.read_zarr("SLIDE.spatialdata.zarr")
table = sdata.tables["table"]
X = table.obsm["X_uni"]            # (N, D) embeddings
xy = table.obsm["spatial"]        # (N, 2) tile centres, full-res px
tiles = sdata.shapes["tiles"]     # GeoDataFrame of square tile polygons

panel = table.uns.get("raw2features_panel")   # multiplex: tidy channel->marker table
# e.g. which channel fed marker "Cytokeratin", under what id:
#   panel[panel["marker"] == "Cytokeratin"][["channel", "marker_id"]]
```

### Notes / pins

- `spatialdata` pulls `anndata`, `ome-zarr`/`zarr` and pins `anndata` to exclude the
  broken `1.17.0`; we only add `spatialdata>=0.2` + `geopandas`. Output defaults to
  **Zarr v2** for maximum interoperability.
- A **calibrated** export carries two coordinate systems (`global` + `micrometers`), so
  `to_legacy_anndata` needs one named - `to_legacy_anndata(sdata, table_name="table",
  coordinate_system="global", …)` (`global` is the pixel space `obsm["spatial"]` lives in).
- The WSI image is referenced, not duplicated: the source OME-Zarr is already NGFF, so the
  store points to it via `uns["raw2features"]["source"]["uri"]` and it loads alongside - no
  second copy of the pyramid. (Embedding the pyramid directly as an `images["wsi"]` element
  is a possible future addition.)

## HDF5 (TRIDENT / CLAM / TITAN, and STAMP)

> **Non-default, export only.** The native `.embeddings.zarr` stays the FAIR primary
> output (rich provenance, multi-model, additive resume). HDF5 export is a one-way bridge
> so you can feed existing pathology-MIL toolchains - for full FAIR provenance, keep and
> share the `.embeddings.zarr`.

```bash
pip install "raw2features[h5]"
raw2features export-h5 SLIDE.embeddings.zarr --layout trident   # CLAM/TRIDENT/TITAN/THREADS
raw2features export-h5 SLIDE.embeddings.zarr --layout clam      # CLAM (int32 coords)
raw2features export-h5 SLIDE.embeddings.zarr --layout stamp     # KatherLab STAMP
raw2features export-h5 SLIDE.embeddings.zarr --model uni --overwrite
```

One `.h5` is written **per model** (their datasets hold a single encoder). The layouts
disagree on dataset names and coordinate units, so pick the one your downstream reads
(`clam` is `trident` with `coords` as int32 - the exact dtype CLAM's feature `.h5` uses):

| | `--layout trident` / `clam` | `--layout stamp` |
|---|---|---|
| features dataset | `features` (float32) | `feats` (float16) |
| coords dataset | `coords` - **level-0 pixels** (int64; int32 for `clam`) | `coords` - **microns (µm)** |
| size metadata | `coords.attrs['patch_size_level0']` | `attrs['tile_size_um']`, `tile_size_px`, `unit='um'` |
| read by | CLAM, TRIDENT, TITAN, THREADS, CHIEF, Patho-Bench | KatherLab STAMP |

We have every field for both natively (level-0 coords + `level0_patch` for TRIDENT;
per-axis `source.scale_um` to convert px→µm, and `patch_px × achieved_mpp` for STAMP's
`tile_size_um`). STAMP coordinates remain relative to the WSI scan's top-left, so an
NGFF physical/stage origin in `source.level0_translation_um` is deliberately not added.
Legacy stores without per-axis scale fall back to isotropic `mpp_level0`. The schemas
are transcribed from each project's own source and written with an independent `h5py`
writer - file formats are not copyrightable, and no project's code (e.g. CLAM's GPL
writer) is copied.

### Verified loading in each tool - a point-in-time claim

Each export was fed through the consuming tool's **own, unmodified feature loader** (the
tool pinned to the commit shown, run as a CPU job on an HPC cluster) and confirmed to
load: the `features` and `coords` come through with the shapes, dtypes and metadata that
tool's pipeline expects. This verifies **ingestion** - the tool's data path reads our export
and yields the tensors/dict it feeds downstream; it does *not* run a full train/encode (that
needs the tool's model weights, out of scope here). These formats aren't standardised and
the tools can change theirs, so read each row as *"loaded as of the date shown"* - re-check
against a newer version before relying on it. The self-describing `.embeddings.zarr` is the
durable FAIR primary output and depends on none of this.

| export | feeds | loader verified (tool @ commit) | as of |
|---|---|---|---|
| `--layout trident` | CLAM | `Generic_MIL_Dataset` `.h5` path, `mahmoodlab/CLAM` @ `53e2409` ¹ | 2026-06-30 |
| `--layout clam` | CLAM | `Generic_MIL_Dataset` `.h5` path, `mahmoodlab/CLAM` @ `53e2409` (int32 coords) ¹ | 2026-06-30 |
| `--layout trident` | TRIDENT | `read_coords` + `WSI.extract_slide_features`, `mahmoodlab/TRIDENT` v0.3.1 (`a91fa84`) | 2026-06-30 |
| `--layout stamp` | STAMP | `BagDataset` / `get_coords`, `KatherLab/STAMP` 2.5.0 (`a410d27`) | 2026-06-30 |

¹ Verified through CLAM's coords-carrying `.h5` feature-bag path (`load_from_h5(True)`); its
default training path reads `.pt`. CLAM @ `53e2409` also requires `pandas < 3` (a CLAM-side
constraint in its CSV bookkeeping, independent of the export).

`--layout trident` **also feeds CLAM** directly - CLAM reads the same `features` +
level-0 `coords`, and h5py reads the int64 coords fine; `--layout clam` only narrows
`coords` to int32 to byte-match CLAM's own output. TITAN / THREADS / CHIEF / Patho-Bench
read the same trident-style `features` + `coords`.

**TIAToolbox isn't an export *layout* target** - but not because it can't use our
features. Its feature-consuming downstream (SlideGraph, the patch-graph slide model) takes
features + coordinates as **in-memory arrays**, not a required on-disk file, so there's no
fixed format to match - you feed it programmatically from the native store (below).
**Functionally verified** against `TissueImageAnalytics/tiatoolbox` v2.1.2:
`SlideGraphConstructor.build` accepts the native store's `coords` (int32) and
`features/<model>` (float16) arrays directly and returns a graph.

## Programmatic use - the native store as a universal handoff

The export layouts above exist only for tools whose pipelines *require* a specific file
(CLAM, STAMP). Otherwise the `.embeddings.zarr` is a plain, self-describing zarr: read
`features/<model>` and `coords` and pass them straight into any tool that takes precomputed
patch features as arrays - clustering, UMAP, a custom MIL/GNN, or e.g. TIAToolbox's
SlideGraph (`SlideGraphConstructor.build(points, features)`). No export, no WSI re-read.
