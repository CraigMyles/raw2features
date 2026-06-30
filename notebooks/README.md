# Notebooks

A short, runnable tour of raw2features. Every notebook runs **CPU-only, no GPU, no
HuggingFace token** (patch features use the open `resnet50` encoder; 05-06 add GrandQC's
openly-licensed QC weights), and starts with a bootstrap cell that installs raw2features
only where it isn't already present - so the same file runs on **Colab** or a local checkout.

| # | Notebook | What it shows | Needs network |
|---|----------|---------------|---------------|
| 01 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | Fully **offline** visual tour on a synthetic slide: slide → tissue segmentation → patch tiles → feature map. No data, no token. | model download only |
| 02 | [`02_visual_walkthrough.ipynb`](02_visual_walkthrough.ipynb) | **Visual, cloud-direct tour** on a real SurGen H&E slide (resolved from the BioImage Archive): thumbnail → tissue segmentation → patch tiles → a ResNet-50 feature map of the slide. Nothing downloaded. | yes (BIA store) |
| 03 | [`03_the_embeddings_store.ipynb`](03_the_embeddings_store.ipynb) | **Inside the `.embeddings.zarr`** - mask, feature matrix and patch grid visualised; features relocated onto the slide; pinned-weights provenance; spec validation. | model download only |
| 04 | [`04_spatialdata.ipynb`](04_spatialdata.ipynb) | Export to scverse **SpatialData** and view tiles with `spatialdata-plot` | model download + optional `spatialdata-plot` |
| 05 | [`05_quality_control.ipynb`](05_quality_control.ipynb) | **GrandQC quality control** - `--qc grandqc` scores every patch for artifacts; rank slides by artifact burden, map blur/folds/pen/bubbles onto the slide, gallery the worst patches, and filter features by QC. | model + GrandQC weight download |
| 06 | [`06_cohort_quality_control.ipynb`](06_cohort_quality_control.ipynb) | **Cohort QC triage** - scan a folder of slides with GrandQC, rank the cohort by artifact burden (out-of-focus, folds, bubbles, pen), and inspect the worst. Scales to thousands of slides via batch processing. | model + GrandQC weight download |

**Start with [`02_visual_walkthrough.ipynb`](02_visual_walkthrough.ipynb)** for the picture-led tour; it's the one with baked-in figures you can view here on GitHub without running anything.

All six run end-to-end on CPU (05-06 are also GPU-validated). The figure outputs are committed so you can preview results on GitHub without running
anything; text outputs are kept minimal.

**Where they run:**
- **Colab** - recommended; torch / matplotlib / pandas are preinstalled, so the bootstrap
  cell only adds raw2features. (A free GPU speeds up 05-06's GrandQC but isn't required.)
- **Local** - any Python env with the `[zarr,image,torch,models]` extras; the bootstrap is
  a no-op once raw2features is importable.
