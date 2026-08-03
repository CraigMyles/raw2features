# Tissue segmentation

A segmenter returns a low-resolution tissue mask (1 = tissue) at a cheap pyramid
level; the patcher overlays the tiling grid on it to keep/drop patches. The seam is
pluggable (`--segmenter NAME`, or `--no-seg` to tile everything). All built-ins are
pure OpenCV + numpy, run on the CPU on a small low-res image, and are re-implemented
from standard techniques - no GPL/CLAM code, no model weights.

## Options

| `--segmenter` | method | when to use |
|---|---|---|
| `otsu` *(default)* | Otsu threshold on HSV **saturation** + morphology | general default - clean & robust |
| `canny` | **low**-threshold Canny (`low=0.05`) → close → **hole-aware contour fill** (`max_hole_frac` keeps large cavities as background) | good on **cervical / faint** tissue - recovers pale tissue Otsu can under-segment |
| `canny_density` | the same low-threshold Canny kept as a **raw edge map** (no dilate/close/fill) at a finer `seg_mpp` (2.0) → the patcher's per-cell mean *is* the patch's **edge density** | **per-patch** decisions with sharp, gap-respecting boundaries - fragmented / sparse slides where filling would swallow the background between fragments. **`--tissue-threshold` becomes a density** (try 0.01-0.05), not a coverage fraction |
| `combined` | fuse `otsu` and `canny` - `or` (default) recovers faint tissue, `and` suppresses smooth high-saturation artefacts | when a cohort needs faint-tissue recovery (`or`) or artefact suppression (`and`) |
| `guarded` | `otsu` OR `canny`, minus the detected **scanner canvas / scan-region border** (the canvas is what drags Otsu's global threshold up *and* becomes Canny's outer contour, so the real tissue gets carved out as a hole) | **dirty cohorts, no weights needed** - corner/edge artefacts being tiled *and* pale tissue being missed. Pure OpenCV; this is `grandqc_veto` without the model |
| `grandqc_veto` *(`[grandqc]` extra)* | `otsu` OR `canny`, minus the scanner canvas / scan-region border, then **vetoed** component-wise by GrandQC's tissue stage | dirty cohorts: corner/edge artefacts being tiled *and* pale tissue being missed. Needs the non-commercial GrandQC weights |

## Choosing a segmenter (evidence)

Compared on real cervical WSIs (8 µm/px masks), with `canny` tuned to the
low-threshold contour-fill recipe (`low=0.05`):

| slide | otsu | canny | combined (or) | IoU(otsu,canny) |
|---|---|---|---|---|
| Slide A | 9.8% | 19.1% | 19.1% | 0.52 |
| Slide B | 18.5% | 25.7% | 25.7% | 0.72 |

The **threshold is what matters**: a *high*/auto Canny threshold (or thresholding edge
*density*) under-segments faint cervical tissue badly - an earlier auto version found
**0.3%** on Slide B. Dropping to a **low** threshold (~0.05) and **filling
contours** instead makes it consistent (19-26%) and recovers pale tissue Otsu can miss,
which matches practitioners' experience on cervical H&E. So:

- **`otsu` stays the default** - it is the field standard (CLAM), parameter-free, and
  clean on the majority of slides.
- **`canny` is a strong alternative for cervical / faint-tissue cohorts**; tune `low`
  (lower = more sensitive) per cohort.
- **`combined or`** is a safe superset of `otsu` (never loses tissue) that folds in
  canny's faint-tissue recovery; **`combined and`** is stricter (artefact suppression).

### `canny` (contour fill) vs `canny_density` (per-patch)

The finding above - that a *high/auto* threshold or **edge-density** thresholding
under-segments faint tissue - was measured against a **region**-style mask: one coarse
whole-slide mask that then has to outline whole tissue regions. Filling contours is the
right fix *there*.

`canny_density` is a different use of the same edges: it keeps the raw edge map and lets
the patcher average it per patch, so **each patch is judged on its own edge content** and
nothing is grown or filled. That combination - a **low** threshold (0.05) with a
**per-patch** density - is the recipe practitioners report working well on cervical H&E,
and it is not the configuration the 0.3% result above came from (that was a high/auto
threshold). Pick by failure mode:

- tissue is *faint but contiguous*, and you want solid regions → **`canny`**
- tissue is *fragmented/sparse*, and filling the gaps between fragments would tile
  background → **`canny_density`**

Because the mask is a thin edge map, its per-patch mean is small: start around
`--tissue-threshold 0.02` and tune per cohort (and re-tune if you change `seg_mpp`,
since finer masks carry more, thinner edges per field of view).

## Provenance of the built-in segmenters

The built-in segmenters are independent implementations of standard, textbook techniques
(Otsu thresholding, HSV saturation, morphology) and contain no code copied from other tools.
The `otsu` default mirrors the same standard HSV-saturation recipe used widely in the field,
including by CLAM. The opt-in `grandqc` weights are fetched under their original
CC-BY-NC-SA-4.0 (non-commercial) licence and are never bundled.

## Other methods that fit this seam

The segmenter is a plugin seam, so new methods drop in without touching the pipeline. From
the methods review, the highest-value candidates, all permissive-implementable:

- `otsu_multichannel` - selectable channel (saturation | optical-density | luminance)
  + a grayspace pre-filter (drop near-gray dust/shadow before thresholding) +
  optional CLAM-style black-background repaint. The robust path for faint/fatty
  slides and odd scanner backgrounds.
- `texture_entropy` - local-entropy density (top-ranked single channel in the
  literature) with an `or`/`and` fuse option (a cleaner cousin of `canny`).
- `deep` *(optional, off by default, extra-install)* - only an Apache/MIT model
  (e.g. a Slideflow-style U-Net) or a user-supplied external; never bundle GrandQC
  (CC-BY-NC-SA) or PathML/CLAM (GPL).

Morphology kernel sizes could be specified in microns (resolution-independent)
rather than pixels.

### Sources
Foucart - best channel for tissue segmentation (saturation/entropy top); EntropyMasker
(Sci Rep 2023); TCGA classical-methods benchmark (PMC12427738); CLAM; HistoQC;
TIAToolbox `tissuemask.py`; Slideflow; GrandQC (Nat Commun 2024).

## GrandQC (`[grandqc]` extra)

`--segmenter grandqc` (tissue) and `--qc grandqc` (artifact) reimplement GrandQC's two
stages with `segmentation_models_pytorch`: the **tissue** stage is a UNet++ (2 classes),
the **artifact** stage a UNet (7 labelled classes + background); the architecture and
class count are read from each checkpoint. Weights are pinned from Zenodo
(CC-BY-NC-SA, non-commercial), fetched on first use, never bundled.

GrandQC's **artifact** checkpoints (`GrandQC_MPP*.pth`) are *full-model*
pickles saved against `smp==0.3.1` + `timm==0.4.12`, so they cannot be unpickled in a
modern env. Convert each once to a plain state_dict (cached beside the download as
`<sha16>-GrandQC_MPP15.statedict.pt`, which the loader then prefers):

```bash
uv venv /tmp/gqc && uv pip install --python /tmp/gqc/bin/python \
    "segmentation_models_pytorch==0.3.1" "torch==2.0.1" six
W=~/.cache/raw2features/weights
/tmp/gqc/bin/python -c "import torch; m=torch.load('$W/<sha16>-GrandQC_MPP15.pth', \
    map_location='cpu', weights_only=False); torch.save(m.state_dict(), \
    '$W/<sha16>-GrandQC_MPP15.statedict.pt')"
```

The tissue checkpoint is already a clean state_dict and needs no conversion.

**Domain limits - verify before trusting cohort-wide QC.** GrandQC was trained on specific
H&E cohorts and does not generalise to every stain/scanner. It scored a set of cervical and
endometrial slides sensibly (mostly clean tissue, real folds/bubbles/blur), but on a
700-slide colorectal cohort (SurGen, ~0.11 µm/px) it labelled **~85% of slides as mostly
out-of-focus** despite visibly crisp tissue - a domain gap, not a scale bug (reading from a
finer pyramid level and downscaling changed it by ~2%). It's a real limit, not a missing
step: GrandQC's own inference applies no stain normalization either, only the same ImageNet
preprocessing we use. Run GrandQC on a sample of your own data and eyeball the class rasters
(see notebook 06) before relying on it across a cohort.

`--qc-stain-norm macenko` Macenko-normalizes the input first and largely recovers the
out-of-domain case: on those colorectal slides it cut a ~100%-out-of-focus call to ~30-40%,
and a milder 28% case to <1%. It's opt-in (GrandQC's default does no such thing); enable it
when your stain sits outside GrandQC's training domain.

### `grandqc_veto` - classical union, GrandQC as a veto

`--segmenter grandqc_veto` exists for cohorts where Otsu fails in two directions at once:
a dark corner artefact is densely tiled (and drags the global threshold up, so real
tissue drops out), while pale stroma is missed. Each term does one job:

- **`otsu`** gives sharp boundaries and correct holes, and defines the mask's grid.
- **`canny`** (contour fill) adds recall on pale, low-saturation tissue.
- a **scan-region guard** removes the synthetic canvas fill the store pads the scan
  region with, the border ring around it, and the image frame. It is applied to the
  *Canny edge map*, before the dilate/close/fill - once that rectangle has been filled
  it is the outer contour, and the real tissue inside becomes a `max_hole_frac` hole
  that gets carved back to background, so filtering the finished mask loses the tissue
  along with the ring.
- **GrandQC's tissue stage @ 10 µm/px** is used *only as a veto*, per connected
  component of the union, so its coarse blobs never add area and its over-inclusion is
  clipped by the classical terms. Its trained scale is not configurable here: running
  the checkpoint finer is off-domain and measured non-monotonic.

A component GrandQC rejects is dropped whole; one it endorses is kept intact; only
*large, mixed-evidence* components are clipped pixel-wise - and that clip first dilates
the veto by `veto_slack_um` (so quantisation and ~one mask pixel of model error cannot
erode anything) and then puts back any removed region narrower than
`removal_min_width_um` (a real artefact is a blob, a mis-registration is a sliver). The
two obvious designs are corner cases of the same parameters: `clip_min_area_mm2=inf` is
a pure component veto; `veto_cov=0, keep_cov=1.1, clip_min_area_mm2=0, veto_slack_um=0,
removal_min_width_um=0` is a pure pixel-wise AND.

The output is binary, so `--tissue-threshold` keeps its usual coverage meaning and tile
counts stay comparable with `otsu`/`canny` runs. After `segment()`, `last_stats` carries
`veto_removed_frac` and per-branch component counts; sorting a cohort by
`veto_removed_frac` is the cheapest way to find slides where the veto is wrong.

**It does not fix out-of-focus tissue.** Neither GrandQC stage separates focus (the
artifact stage scores `out_of_focus` .027 on controls vs .024 on flagged slides), and
nothing else here measures it. That needs a separate per-patch focus gate.

If the veto cannot run (missing extra, download failure) it raises by default;
`on_veto_error="warn"` degrades to the bare guarded union, never silently.
