# Brightfield and named-channel multiplex tissue images

raw2features supports RGB brightfield images and named-channel multiplex tissue images.
Multiplex sources use native `(H, W, C)` reads and preserve positional channel identity.
Registered RGB encoders can be applied through a multiplex strategy, while native
multiplex encoders consume the channel stack directly. Ordinary brightfield execution is
unchanged.

## Execution paths

Each registry model declares `modality: brightfield` (the default) or
`modality: multiplex`. A strategy wraps a brightfield model without changing its registry
entry.

| stage | brightfield | `channelwise` RGB strategy | native multiplex encoder |
|---|---|---|---|
| reader | `read_region` → RGB `[H,W,3]` uint8 | `read_region_channels` → native `[H,W,C]` | `read_region_channels` → native `[H,W,C]` |
| channel identity | none | selected positional names | selected positional names resolved by the model |
| segmentation | `otsu`, `canny`, … | one nuclear channel, a recognized same-stain group, or `--no-seg` | one nuclear channel, a recognized same-stain group, or `--no-seg` |
| embedding | one RGB input per patch | one RGB input per selected marker, then mean/concat | model-specific marker stack |

Panel binding happens before receipt or store completion checks. Native multiplex and
`channelwise` fingerprints bind the selected physical channel identities and order; the
complete effective panel remains in source/panel provenance. When nuclear masking is
enabled, the resolved physical nuclear-channel index or same-stain index group is also
part of grid identity.

## Channel metadata

The preferred source of channel identity is `omero.channels` in the OME-Zarr metadata.
The reader preserves unnamed entries as empty positional slots rather than shifting later
labels onto the wrong pixels.

For a source whose labels are absent or incomplete, pass `--channel-names-file` with a
complete ordered panel. UTF-8 `.txt`, `.csv`, and `.tsv` files are accepted, with one
unique name per physical C-axis position. Existing non-empty OME labels are treated as
assertions and must agree with the supplied name at the same index. The override is
in-memory only and never rewrites the source. Identity uses only the resolved names.
Provenance also records whether effective names came from OME metadata or the supplied
file and preserves any differing original OME labels. See
[usage.md](usage.md#rgb-encoders-on-named-channel-multiplex-tissue-images) for examples
and marker-selection rules.

## Converting a multiplex TIFF

Multiplex images are often supplied as OME-TIFF or TIFF channel stacks. The included
conversion helper accepts a `(C,Y,X)` TIFF and writes a multiscale OME-Zarr with one
`omero.channels[].label` per channel:

```bash
pip install "raw2features[zarr]" tifffile imagecodecs
python scripts/codex_to_omezarr.py SLIDE.tif SLIDE.ome.zarr \
  --markers markers.json --mpp 0.5
```

`markers.json` is a list, or an object containing `raw_markers`, `channels`, or `markers`.
The converter requires one name per channel. Prefer a domain converter that preserves all
available OME metadata when one is available; this helper is intentionally small.

## RGB encoders through `channelwise`

`channelwise` normalizes selected channels independently, repeats each single-channel
patch across RGB, runs the registered RGB encoder and pooling, then combines the marker
vectors with mean or ordered concatenation. It records the effective panel, normalization
level and values, RGB conversion, base-model output fingerprint, pooling, aggregation,
and output dimension. Model-agnostic slide poolers can consume the result.

This makes it possible to compare models such as UNI or Virchow2 on named-channel data
without representing them as native multiplex models. The strategy's assumptions and
full CLI contract are documented in [usage.md](usage.md#rgb-encoders-on-named-channel-multiplex-tissue-images).

## Native multiplex encoders

Native multiplex models use the same positional-panel plumbing but consume the marker
stack directly. `kronos` and `kronos2` map effective source names to their pinned marker
metadata, record the physical mapping, and bind the selected indices and order into the
output fingerprint before resume. Repeated `--marker` options select and order a subset;
without them, the source channels are offered to the model in physical order.
Installation and model-specific license/access details are in
[MODELS.md](MODELS.md) and [MODEL_LICENSES.md](MODEL_LICENSES.md).

```bash
raw2features embed SLIDE.ome.zarr OUT -m kronos --mpp 0.5
raw2features embed SLIDE.ome.zarr OUT -m kronos2 \
  --marker CD3 --marker CD8 --marker DAPI
```

Use `--channel-names-file` when the source lacks a complete panel, or `--no-seg` when the
entire image should be tiled without a nuclear mask.

KRONOS2 performs exact separator-insensitive marker matching and does not reuse
KRONOSv1's biological aliases. Its released metadata contains 288 usable entries, of
which 268 are marked as pretraining markers. Their text vectors are already stored in
the checkpoint, so BioLinkBERT is neither downloaded nor run.

The released KRONOS2 vocabulary includes DAPI and DRAQ5, but not Hoechst or
DNA1/DNA2 labels found in many CODEX panels. raw2features does not silently map those
stains to DAPI. A Hoechst channel may be excluded from the encoder with an explicit
`--marker` list while still being used by nuclear segmentation, which always sees the
complete source panel. To include it in the KRONOS2 marker stack, register it as a novel
marker with statistics computed for the relevant dataset.

For a marker outside the released metadata, pass a complete CSV through
`--kronos2-additional-markers`. Each row requires `marker_name`, `marker_full_name`,
`compartment`, `family`, `family_desc`, `mean`, and `std`. The statistics describe the
marker after KRONOS2's published dtype scaling (uint8 / 255, wider unsigned integers /
65535, or floating values / 400) and are supplied by the user; signed integers are
rejected and raw2features does not infer cohort statistics. Novel names must not collide
with the pinned vocabulary, while `compartment` and `family` must reuse its categories
exactly. This optional path downloads the separately pinned BioLinkBERT model to create
the marker text vector. The CSV content, statistics, and text-model identity become part
of the output fingerprint.

Following the published inference example, raw2features passes DRAQ5 as
`preferred_dapi` when DAPI is absent. The upstream preprocessor therefore applies its
DAPI normalization statistics to that selected DRAQ5 channel. This policy is recorded
in provenance and is not configurable in v0.2.1.

## Extending multiplex support

The `raw2features.multiplex_strategies` entry point separates panel/config preparation
from slide-specific binding. Future marker-to-RGB mappings or learned channel adapters
can use this contract if there is demand. A new native multiplex model instead declares
`modality: multiplex` and implements the standard embedder panel-binding seam.
