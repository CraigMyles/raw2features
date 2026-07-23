# Changelog

Notable changes to raw2features, newest first. This project follows
[Semantic Versioning](https://semver.org).

## [0.2.0] - Unreleased

This release extends named-channel multiplex support with a `channelwise` strategy, adds
several pinned patch and slide encoders, and includes a set of correctness and
release-safety fixes. The v0.1 store layout and existing brightfield workflows remain
unchanged.

### Correctness and reproducibility

- Redact command-line secrets and persist credential-free canonical source URIs while
  still supporting query-authenticated HTTP Zarr stores in memory.
- Bind stores and receipts to their source and requested output, reject duplicate cohort
  output IDs before dispatch, make receipt writes atomic, and propagate worker-startup
  failures to the command exit status.
- Enforce immutable model revisions at load time. Patch and slide outputs now carry a
  per-model fingerprint covering their effective weights, preprocessing, pooling,
  dimension, resolved AMP, and loader contract; a stale or legacy-unfingerprinted output
  is recomputed without changing `grid_hash` or unrelated model arrays.
- Verify composite checkpoints before deserialisation, validate expected embedding
  dimensions during completion checks, and mark SEAL experimental until its upstream
  base-model fetch can be pinned end to end.
- Make colliding human-readable grid labels unique, produce explicitly requested
  thumbnails/QC/GeoJSON/slide embeddings when missing, and validate receipts against the
  requested grid rather than unioning same-named models across grids.
- Harden partial-store recovery and conformance validation, preserve per-axis scale in
  slide-relative STAMP coordinates, place clipped border pixels at their true offsets,
  reject invalid AMP/batch/geometry settings before model loading, and avoid attributing
  an enclosing unrelated Git checkout to an installed wheel.

### Models and interoperability

- Add a pluggable multiplex-strategy layer and a built-in `channelwise` strategy for
  applying registered RGB patch encoders to named-channel multiplex tissue images. It
  records positional channel identity (including a strict metadata-file fallback), marker
  selection and order, normalization, RGB conversion, base-model identity, mean/concat
  aggregation, and model-agnostic slide pooling without changing brightfield outputs.
- Bind native multiplex outputs to the complete effective positional panel. Nuclear
  masking accepts one unambiguous nuclear channel or a recognized same-stain group, such
  as DNA1/DNA2 or repeated DAPI channels. It averages each multi-channel group in float32
  and binds every physical index into grid identity; `--no-seg` remains supported.
- Add pinned, forward-validated H0-mini, KEEP, OpenMidnight, and OpenPath patch encoders,
  with their licenses, access requirements, preprocessing, and weight hashes recorded.
- Add the paired GigaPath-Flash tile and slide encoders with immutable weight and loader
  pins, and align original GigaPath's default extraction with its released whole-slide
  path: non-overlapping 256-pixel tiles on a 256-pixel lattice, followed by a centre crop
  to 224. LongNet receives the stored level-0 coordinates and uses each tile's level-0
  extent as its runtime positional divisor, normalizing raw2features' coordinate units
  into the tile-index frame of upstream's default 256-pixel tiling.
- Repair HIPT downloads by replacing the unavailable unofficial mirror with Mahmood Lab's
  commit-pinned, SHA-verified Git LFS checkpoint and record its Commons Clause accurately.
- Correct cached OME-Zarr reads for noncanonical spatial axis order. The reader accepts
  x/y in either order (with an optional channel axis) and warns whenever another
  non-singleton axis is reduced to index zero. Unnamed `omero.channels` entries now
  retain their physical positions instead of shifting later marker identities.
- Refresh the development lock to current `ngff-zarr`/`ome-zarr` releases while retaining
  the existing declared compatibility floor.
- Choose the float-pixel `[0,1]` versus `[0,255]` convention once per opened slide instead
  of independently for every patch.

### API, documentation, and release safety

- Bless `embed_slide` as the high-level Python entry point while retaining `run_slide` as
  the explicit single-grid primitive, resolve injected embedders from their own model
  specifications, and make inline `-s/--slide-encoder` requests produce missing outputs
  on otherwise-complete runs.
- Correct the multi-grid, additive-append, readback, and SLURM documentation, and use the
  compact 4x PNG project banner so GitHub and PyPI render it consistently.
- Add a genuine dependency-lean import/CLI check, built-wheel and source-distribution
  smoke installs, package metadata validation, and a release tag/version gate.

### Compatibility and migration

- Existing v0.1 stores remain readable; the store layout/schema and model-independent
  grid-identity contract are unchanged. Multiplex nuclear masking now binds the resolved
  physical nuclear-channel indices. Averaging a recognized same-stain group is an
  intentional value change from v0.1 and therefore creates a new grid. Compatible
  historical native-multiplex grids remain appendable only when the original OME panel
  proves that the v0.1 selector used the same single physical channel.
- Original GigaPath now defaults to the released 256-pixel non-overlapping extraction
  lattice followed by a centre crop to 224. Existing 224-pixel grids remain readable and
  distinct; an unmodified default request creates or selects the correctly keyed
  256-pixel grid.
- The first request for a model output created before fingerprints were introduced will
  recompute that model in place. The old header is deliberately not treated as proof that
  pre-v0.2.0 revision pins were enforced.
- Plain local output IDs remain basename-based for v0.1 compatibility. Source comparison
  prevents accidental reuse, and manifest preflight rejects duplicate derived IDs.
- Credentials already persisted by v0.1.0 are not rewritten by this release and should be
  rotated separately if the store was shared.

Tracked by [#6](https://github.com/CraigMyles/raw2features/issues/6); this release resolves
[#2](https://github.com/CraigMyles/raw2features/issues/2),
[#3](https://github.com/CraigMyles/raw2features/issues/3),
[#4](https://github.com/CraigMyles/raw2features/issues/4), and
[#5](https://github.com/CraigMyles/raw2features/issues/5).

## [0.1.0] - 2026-06-30

First public release.

raw2features reads whole-slide images in OME-Zarr / OME-NGFF - local or remote - and emits
patch- and slide-level foundation-model embeddings as a self-describing `.embeddings.zarr`
store. The reader, segmenter, patcher, embedder and sink are pluggable seams, so a new model
or storage backend ships as a small package with no fork.

What 0.1.0 provides:

- **Cloud-direct reads** of NGFF v0.4 and v0.5 (`http(s)`, `s3`, `gs`): the whole
  segment -> tile -> embed pipeline runs against a remote store with nothing downloaded.
- **Exact-MPP patch extraction** by downsampling from the nearest finer pyramid level, so
  embeddings are comparable across slides and scanners.
- **30+ patch encoders** (ImageNet baselines plus pathology foundation models) and several
  **slide encoders**, each with preprocessing transcribed from its model card and weights
  pinned to a sha256 + Hugging Face revision for full provenance.
- **Decode-once multi-model fan-out**: many encoders in one pass, one store, 1:1 aligned to
  the patch coordinates; resumable and additive across re-runs.
- **Optional quality control** (GrandQC artifact scoring) and **stain normalization**
  (Macenko, Reinhard, Vahadane).
- **Multiplex** (spatial proteomics) alongside H&E on the same seams: channel-aware reads,
  nuclear segmentation, and the marker-aware KRONOS encoder.
- A **specified, self-describing store** with a JSON Schema and a `validate-store` command,
  plus one-way exports to scverse SpatialData, pathology-MIL HDF5, and QuPath GeoJSON.
- **Cohort tooling**: `embed-many` (sharded, resumable, with SLURM templates), a benchmark
  harness, and six runnable tutorial notebooks.

[0.1.0]: https://github.com/CraigMyles/raw2features/releases/tag/v0.1.0
[0.2.0]: https://github.com/CraigMyles/raw2features/compare/v0.1.0...v0.2.0
