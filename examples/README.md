# Example inputs

Ready-to-adapt files for the two declarative `raw2features` inputs.

- **[`extractions.yaml`](extractions.yaml)** - an extraction plan
  (`embed … --config extractions.yaml`): one grid per entry; the same model may repeat at
  different MPPs (an ablation); omit `mpp`/`patch_px` to use the model's registry default.
- **[`cohort.csv`](cohort.csv)** - a slide manifest
  (`embed-many <dir> out/ --manifest cohort.csv`): a curated path list with an optional
  per-slide `source_mpp` (only needed when the OME-Zarr omits its level-0 pixel size).

Stain-normalization **experiments** use the CLI flag, not a config file - run with and
without into separate output dirs (the features differ):

```bash
raw2features embed slide.ome.zarr out_raw/     -f uni
raw2features embed slide.ome.zarr out_macenko/ -f uni --stain-norm macenko
```
