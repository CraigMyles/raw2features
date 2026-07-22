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
| channel identity | none | selected positional names | positional names resolved by the model |
| segmentation | `otsu`, `canny`, … | one nuclear channel, a recognized same-stain group, or `--no-seg` | one nuclear channel, a recognized same-stain group, or `--no-seg` |
| embedding | one RGB input per patch | one RGB input per selected marker, then mean/concat | model-specific marker stack |

Panel binding happens before receipt or store completion checks. Native multiplex
fingerprints bind the complete effective positional panel; `channelwise` fingerprints
bind the selected physical channel identities and order (all channels when default
selection is used). The full effective panel remains in source/panel provenance. When
nuclear masking is enabled, the resolved physical nuclear-channel index or same-stain
index group is also part of grid identity.

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
stack directly. The current built-in example, `kronos`, maps effective source names to
its pinned marker vocabulary, records the kept/dropped physical mapping, and binds the
complete effective panel into its output fingerprint before resume. Installation and
model-specific license/access details are in [MODELS.md](MODELS.md) and
[MODEL_LICENSES.md](MODEL_LICENSES.md).

```bash
raw2features embed SLIDE.ome.zarr OUT -m kronos --mpp 0.5
```

Use `--channel-names-file` when the source lacks a complete panel, or `--no-seg` when the
entire image should be tiled without a nuclear mask.

## Extending multiplex support

The `raw2features.multiplex_strategies` entry point separates panel/config preparation
from slide-specific binding. Future marker-to-RGB mappings or learned channel adapters
can use this contract if there is demand. A new native multiplex model instead declares
`modality: multiplex` and implements the standard embedder panel-binding seam.
