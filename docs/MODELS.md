# Models

Every encoder `raw2features` can run, with the facts verified against each model's
primary sources (Hugging Face card, source repo, paper). The machine-readable source of
truth is the registry (`src/raw2features/embedders/registry.yaml`); per-weight licensing
is in `MODEL_LICENSES.md`; mean/std/input-size are transcribed from each model's own card
(see the preprocessing caveat below).

Two kinds of model live here: **patch encoders** turn an image tile into a vector;
**slide encoders** turn a slide's matrix of patch vectors into one slide vector (so each
slide encoder *requires* a specific patch encoder upstream). They are in separate tables.

We record each model's **exact licence** as a fact and stop there - what it permits for
*your* use (commercial use, redistribution, derivatives) is the licence's own terms to
read and decide on. raw2features makes no commercial-use determination.

## Patch encoders

| model | family | dim | px | µm/px | basis | license | gated | links |
|---|---|---:|---:|:--|:--|---|:--|---|
| `resnet50` | torchvision | 2048 | 224 | - | n/a | BSD-3-Clause | no | [GH](https://github.com/pytorch/vision) · [paper](https://arxiv.org/abs/1512.03385) |
| `dinov2` | timm | 1024 | 224¹ | - | n/a | Apache-2.0 | no | [HF](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m) · [GH](https://github.com/facebookresearch/dinov2) · [paper](https://arxiv.org/abs/2304.07193) |
| `uni` | timm | 1024 | 224 | 0.5 | mag→ | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/UNI) · [GH](https://github.com/mahmoodlab/UNI) · [paper](https://doi.org/10.1038/s41591-024-02857-3) |
| `uni2_h` | timm | 1536 | 224 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/UNI2-h) · [GH](https://github.com/mahmoodlab/UNI) · [paper](https://doi.org/10.1038/s41591-024-02857-3) |
| `path_orchestra` | timm | 1024 | 224 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/AI4Pathology/PathOrchestra) · [paper](https://arxiv.org/abs/2503.24345) |
| `virchow` | timm | 2560 | 224 | 0.5 | stated | Apache-2.0 | yes | [HF](https://huggingface.co/paige-ai/Virchow) · [paper](https://doi.org/10.1038/s41591-024-03141-0) |
| `virchow2` | timm | 2560 | 224 | 0.5² | stated | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/paige-ai/Virchow2) · [paper](https://arxiv.org/abs/2408.00738) |
| `gigapath` | timm | 1536 | 224³ | 0.5 | stated | Apache-2.0 ⁴ | yes | [HF](https://huggingface.co/prov-gigapath/prov-gigapath) · [GH](https://github.com/prov-gigapath/prov-gigapath) · [paper](https://doi.org/10.1038/s41586-024-07441-w) |
| `gigapath_flash` | timm | 384 | 224³ | 0.5 | stated | Apache-2.0 ⁴ | yes | [HF](https://huggingface.co/prov-gigapath/prov-gigapath-flash) · [GH](https://github.com/prov-gigapath/prov-gigapath) · [paper](https://doi.org/10.48550/arXiv.2607.18218) |
| `conch` | conch | 512 | 448 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/CONCH) · [GH](https://github.com/mahmoodlab/CONCH) · [paper](https://doi.org/10.1038/s41591-024-02856-4) |
| `conch_v1_5` | conch_v1_5 | 768 | 448⁵ | 0.5 | mag→ | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/conchv1_5) · [GH](https://github.com/mahmoodlab/TITAN) · [paper](https://arxiv.org/abs/2411.19666) |
| `h_optimus_0` | timm | 1536 | 224 | 0.5 | stated | Apache-2.0 | yes | [HF](https://huggingface.co/bioptimus/H-optimus-0) · [GH](https://github.com/bioptimus/releases) |
| `h0_mini` | timm | 768 | 224 | 0.5 | mag→ | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/bioptimus/H0-mini) · [paper](https://doi.org/10.1007/978-3-032-04981-0_16) |
| `h_optimus_1` | timm | 1536 | 224 | 0.5 | stated | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/bioptimus/H-optimus-1) |
| `gpfm` | timm | 1024 | 224 | 0.5⁶ | mag→ | MIT / CC-BY-NC-ND-4.0 ⁷ | no | [HF](https://huggingface.co/majiabo/GPFM) · [GH](https://github.com/birkhoffkiki/GPFM) · [paper](https://doi.org/10.1038/s41551-025-01488-4) |
| `midnight` | transformers | 3072 | 224 | 0.5² | stated | MIT | no | [HF](https://huggingface.co/kaiko-ai/midnight) · [GH](https://github.com/kaiko-ai/Midnight) · [paper](https://arxiv.org/abs/2504.05186) |
| `openmidnight` | dino_teacher | 1536 | 224 | - | unspec. | Apache-2.0 | yes | [HF](https://huggingface.co/SophontAI/OpenMidnight) · [GH](https://github.com/MedARC-AI/OpenMidnight) |
| `openpath` | dino_teacher | 1536 | 224 | 0.5 | stated | Apache-2.0 | no | [HF](https://huggingface.co/taejoon89/openpath) · [GH](https://github.com/taejoon89/openpath) |
| `phikon` | transformers | 768 | 224 | 0.5 | stated | Owkin NC ⁸ | no | [HF](https://huggingface.co/owkin/phikon) · [GH](https://github.com/owkin/HistoSSLscaling) · [paper](https://doi.org/10.1101/2023.07.21.23292757) |
| `phikon_v2` | transformers | 1024 | 224 | 0.5 | stated | Owkin NC ⁸ | no | [HF](https://huggingface.co/owkin/phikon-v2) · [GH](https://github.com/owkin/HistoSSLscaling) · [paper](https://arxiv.org/abs/2409.09173) |
| `lunit_dino` | timm | 384 | 224 | 0.5² | stated | Lunit NC ⁹ | no | [HF](https://huggingface.co/1aurent/vit_small_patch16_224.lunit_dino) · [GH](https://github.com/lunit-io/benchmark-ssl-pathology) · [paper](https://arxiv.org/abs/2212.04690) |
| `lunit_dino8` | timm | 384 | 224 | 0.5² | stated | Lunit NC ⁹ | no | [HF](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) · [GH](https://github.com/lunit-io/benchmark-ssl-pathology) · [paper](https://arxiv.org/abs/2212.04690) |
| `lunit_bt` | timm | 2048 | 224 | 0.5² | stated | Lunit NC ⁹ | no | [HF](https://huggingface.co/1aurent/resnet50.lunit_bt) · [GH](https://github.com/lunit-io/benchmark-ssl-pathology) · [paper](https://arxiv.org/abs/2212.04690) |
| `lunit_mocov2` | timm | 2048 | 224 | 0.5² | stated | Lunit NC ⁹ | no | [HF](https://huggingface.co/1aurent/resnet50.lunit_mocov2) · [GH](https://github.com/lunit-io/benchmark-ssl-pathology) · [paper](https://arxiv.org/abs/2212.04690) |
| `lunit_swav` | timm | 2048 | 224 | 0.5² | stated | Lunit NC ⁹ | no | [HF](https://huggingface.co/1aurent/resnet50.lunit_swav) · [GH](https://github.com/lunit-io/benchmark-ssl-pathology) · [paper](https://arxiv.org/abs/2212.04690) |
| `sp22m` | timm | 384 | 224 | 0.5 | stated | CC-BY-NC-SA-4.0 | no | [HF](https://huggingface.co/MountSinaiCompPath/SP22M) · [GH](https://github.com/sinai-computational-pathology/SSL_tile_benchmarks) · [paper](https://arxiv.org/abs/2310.07033) |
| `retccl` | timm | 2048 | 256⁵ | 1.0 | stated | GPL-3.0 ¹⁰ | no | [HF](https://huggingface.co/jamesdolezal/RetCCL) · [GH](https://github.com/Xiyue-Wang/RetCCL) · [paper](https://doi.org/10.1016/j.media.2022.102645) |
| `hipt` | timm | 384 | 256⁵ | 0.5 | mag→ | Apache-2.0 + Commons Clause | no | [GH](https://github.com/mahmoodlab/HIPT) · [paper](https://arxiv.org/abs/2206.02647) |
| `mstar` | timm | 1024 | 224 | 0.5 | mag→ | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/Wangyh/mSTAR) · [GH](https://github.com/Innse/mSTAR) · [paper](https://arxiv.org/abs/2407.15362) |
| `kaiko_vitl` | timm | 1024 | 224 | 0.5 | conv. | Kaiko NC ¹³ | no | [HF](https://huggingface.co/1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms) · [GH](https://github.com/kaiko-ai/towards_large_pathology_fms) · [paper](https://arxiv.org/abs/2404.15217) |
| `ctranspath` | timm | 768 | 224 | 0.5 | mag→¹¹ | GPL-3.0 ¹⁰ | no | [HF](https://huggingface.co/1aurent/swin_tiny_patch4_window7_224.CTransPath) · [GH](https://github.com/Xiyue-Wang/TransPath) · [paper](https://doi.org/10.1016/j.media.2022.102559) |
| `hibou_l` | transformers | 1024 | 224 | 0.5 | conv. | Apache-2.0 | yes | [HF](https://huggingface.co/histai/hibou-L) · [GH](https://github.com/HistAI/hibou) · [paper](https://arxiv.org/abs/2406.05074) |
| `hibou_b` | transformers | 768 | 224 | 0.5 | conv. | Apache-2.0 | yes | [HF](https://huggingface.co/histai/hibou-b) · [GH](https://github.com/HistAI/hibou) · [paper](https://arxiv.org/abs/2406.05074) |
| `musk` | musk | 1024 | 384 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/xiangjx/musk) · [GH](https://github.com/lilab-stanford/MUSK) · [paper](https://doi.org/10.1038/s41586-024-08378-w) |
| `quiltnet` | open_clip | 512 | 224 | 0.5 | conv. | MIT | no | [HF](https://huggingface.co/wisdomik/QuiltNet-B-32) · [GH](https://github.com/wisdomikezogwo/quilt1m) · [paper](https://arxiv.org/abs/2306.11207) |
| `biomedclip` | open_clip | 512 | 224 | 0.5 | conv. | MIT ¹⁴ | no | [HF](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) · [paper](https://arxiv.org/abs/2303.00915) |
| `plip` | clip_hf | 512 | 224 | 0.5 | conv. | MIT ¹⁵ | no | [HF](https://huggingface.co/vinid/plip) · [GH](https://github.com/PathologyFoundation/plip) · [paper](https://doi.org/10.1038/s41591-023-02504-3) |
| `keep` | keep | 768 | 224 | - | unspec. ¹⁶ | MIT | no | [HF](https://huggingface.co/Astaxanthin/KEEP) · [GH](https://github.com/MAGIC-AI4Med/KEEP) · [paper](https://doi.org/10.1016/j.ccell.2026.01.019) |
| `seal_conch` ⚠ | seal | 512 | 224 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/SEAL) · [GH](https://github.com/mahmoodlab/SEAL) · [paper](https://arxiv.org/abs/2602.14177) |
| `seal_univ2` ⚠ | seal | 1536 | 224 | 0.5 | conv. | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/SEAL) · [GH](https://github.com/mahmoodlab/SEAL) · [paper](https://arxiv.org/abs/2602.14177) |
| `kronos` | kronos | 384 | 224 | -¹² | n/a | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/KRONOS) · [GH](https://github.com/mahmoodlab/KRONOS) · [paper](https://arxiv.org/abs/2506.03373) |

⚠ **SEAL is experimental in v0.2.0 and is outside the exact-weight pinning
guarantee.** raw2features pins and SHA-256 verifies the SEAL LoRA adapter, freezes
the adapter constructor, and records its composite identity. The pinned upstream
factory still downloads the frozen CONCH/UNI2-h base from mutable HEAD, however, so
the complete base+adapter output cannot yet be reproduced from immutable weights.

## Slide encoders

These consume a slide's `(N_patches, patch_dim)` feature matrix and return one slide
vector. Run them with `raw2features slide-embed`, or `-s <name>` on `raw2features embed`.
Each needs the matching patch encoder run first (`-f <patch_encoder>`).

| model | requires | dim | license | gated | links |
|---|---|---:|---|:--|---|
| `mean` | any patch encoder | = patch dim | MIT | no | built-in (no weights) |
| `max` | any patch encoder | = patch dim | MIT | no | built-in (no weights) |
| `meanmax` | any patch encoder | 2 × patch dim | MIT | no | built-in (no weights) |
| `titan` | `conch_v1_5` (768) | 768 | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/TITAN) · [GH](https://github.com/mahmoodlab/TITAN) · [paper](https://arxiv.org/abs/2411.19666) |
| `prism` | `virchow` (2560) | 1280 | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/paige-ai/Prism) · [paper](https://arxiv.org/abs/2405.10254) |
| `madeleine` | `conch` (512) | 512 | MIT / CC-BY-NC-ND ‡ | yes | [HF](https://huggingface.co/MahmoodLab/madeleine) · [GH](https://github.com/mahmoodlab/MADELEINE) · [paper](https://arxiv.org/abs/2408.02859) |
| `feather_conch_v15` | `conch_v1_5` (768) | 512 | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k) · [GH](https://github.com/mahmoodlab/MIL-Lab) · [paper](https://arxiv.org/abs/2506.09022) |
| `feather_uni_v2` | `uni2_h` (1536) | 512 | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/abmil.base.uni_v2.pc108-24k) · [GH](https://github.com/mahmoodlab/MIL-Lab) · [paper](https://arxiv.org/abs/2506.09022) |
| `feather_uni` | `uni` (1024) | 512 | CC-BY-NC-ND-4.0 | yes | [HF](https://huggingface.co/MahmoodLab/abmil.base.uni.pc108-24k) · [GH](https://github.com/mahmoodlab/MIL-Lab) · [paper](https://arxiv.org/abs/2506.09022) |
| `chief` | `ctranspath` (768) | 768 | GPL-3.0 ◊ | no | [GH](https://github.com/hms-dbmi/CHIEF) · [paper](https://doi.org/10.1038/s41586-024-07894-z) |
| `tangle` | `uni` (1024) | 512 | CC-BY-NC-ND-4.0 ✦ | no | [GH](https://github.com/mahmoodlab/TANGLE) · [paper](https://arxiv.org/abs/2405.11618) |
| `gigapath_slide` | `gigapath` (1536) | 768 | Apache-2.0 ⁴ ✪ | yes | [HF](https://huggingface.co/prov-gigapath/prov-gigapath) · [GH](https://github.com/prov-gigapath/prov-gigapath) · [paper](https://doi.org/10.1038/s41586-024-07441-w) |
| `gigapath_flash_slide` | `gigapath_flash` (384) | 384 | Apache-2.0 ⁴ ✪ | yes | [HF](https://huggingface.co/prov-gigapath/prov-gigapath-flash) · [GH](https://github.com/prov-gigapath/prov-gigapath) · [paper](https://doi.org/10.48550/arXiv.2607.18218) |

TITAN runs on CONCH v1.5 patch features, **not** UNI:
`raw2features embed slide.ome.zarr out -f conch_v1_5 --patch-size 512 -s titan`.

For standalone encoding, `slide-embed` selects the sole grid automatically. In a
multi-grid store it can infer the grid from a slide encoder's required patch model when
that model occurs once. Use `--grid <key>` when a model appears in several grids, or for
a model-agnostic pool such as `mean`; use `--patch-model` as well when that grid contains
several patch models. An unchanged rerun is skipped, while `--force` replaces the existing
`slide/<model>` vector.

‡ `madeleine`'s licence is contested - the HF card tags MIT but the GitHub repo's `LICENSE`
is CC-BY-NC-ND-4.0; we record both and leave the determination to you.

◊ `chief`'s aggregator is our own MIT re-implementation; the **weights** are GPL-3.0
(non-commercial academic). The weights live on Google Drive (no HF mirror); raw2features
fetches them by their public Drive file id and verifies a pinned SHA-256 before loading.
`chief` needs the `[chief]` extra (`gdown`).

✦ `tangle` is the **pan-cancer TANGLE v2** checkpoint (multi-head ABMIL, trained across 27
TCGA cohorts) - *not* the breast-specific releases. The aggregator is our own
re-implementation; the weights are on Google Drive (no HF mirror), fetched by their public
Drive file id and verified against a pinned SHA-256. `tangle` needs the `[tangle]` extra
(`gdown`).

✪ `gigapath_slide` and `gigapath_flash_slide` need **`flash-attn`** (no SDPA fallback).
Prebuilt wheels exist for
x86 + torch 2.7-2.9; for torch 2.6 or aarch64 torch ≥2.10, build it from source (needs
`nvcc`). GPU-validated with a source-built flash-attn (torch 2.12). Where flash-attn is
not installed, these slide encoders are unavailable (their tests skip cleanly), so the
rest of the toolkit is unaffected. Both need the
`[gigapath_slide]` extra + the gigapath git package + `fairscale` + a flash-attn matching
your torch/ABI. The Flash tile encoder also needs the pinned git package to register its
architecture, but it does not itself use flash-attn:

```bash
pip install "raw2features[gigapath_slide]"
pip install --no-deps fairscale "git+https://github.com/prov-gigapath/prov-gigapath.git@9d42a60babe04359978d5ad2eb94e7b3bcf9ca39"
pip install flash-attn  # slide encoders only; use a wheel/build matching torch + CUDA
```

### Legend

- **dim** - embedding dimension. **px** - the input tile size after our transform
  (`input_size`); field of view = `px × µm/px`.
- **µm/px** - the model-specific physical scale raw2features extracts at when you omit
  `--mpp` (it resamples every slide to this). `-` = no author-sourced model default;
  raw2features uses its 1.0 µm/px fallback unless another requested model or `--mpp`
  supplies the scale.
- **basis** - how that default is grounded, so you can see assumption vs. certainty:
  - **stated** - a primary source (card, repo config, or paper) prints a µm/px figure.
  - **mag→** - only a *magnification* is stated; `0.5` is the conventional 20×→µm/px
    conversion, not a printed µm/px value.
  - **conv.** - neither µm/px nor magnification is stated for the model; pure field
    convention.
  - **unspec.** - the pathology model's primary sources state no scale; unlike
    **conv.**, raw2features does not assign a pathology-field convention.
  - **n/a** - scale-agnostic baseline, or multiplex (no single µm/px).
- **license** - the model's exact *weights* licence (SPDX id or precise name). Read its
  terms to decide what your use permits; we make no such call. Footnoted rows are where the
  primary sources disagree or add conditions. raw2features' own code is MIT regardless
  (`MODEL_LICENSES.md`).
- **gated** - `yes` means you must accept the model's terms on Hugging Face before the
  weights will download.

### Footnotes

1. `dinov2` - the timm card lists a native input of 518×518; we run it at 224 (`input_size:
   224`, dynamic image size) as an ImageNet-domain control, not at 518.
2. Multi-scale training - `virchow2` (2.0 / 1.0 / 0.5 / 0.25 µm/px), `midnight` (2 / 1 /
   0.5 / 0.25 µm/px), the `lunit_*` family (20× = 0.5 **and** 40× = 0.25 µm/px). `0.5` is
   one of several training scales; raw2features defaults to it.
3. `gigapath` / `gigapath_flash` - the released whole-slide path extracts
   non-overlapping 256 px tiles at 0.5 µm/px on a 256 px lattice, then centre-crops
   each tile to 224 px for the tile encoder. The crop discards a 16 px margin on every
   side; those margins are not seen by the encoder. Coordinates remain those of the
   original 256 px tiles. raw2features passes the stored level-0 coordinates unchanged
   to the paired LongNet and leaves its upstream fixed 256 px positional lattice intact.
4. `gigapath` / `gigapath_flash` - the `LICENSE` file is Apache-2.0; the model cards
   additionally state "any
   deployed use case … commercial or otherwise … is out of scope" and restricts use to
   research/reproducibility. Both apply - read them and decide.
5. Larger / different tiles - extract `conch_v1_5` at 512 px @ 20× (resized to 448:
   `--patch-size 512`); `retccl` at 256 px @ 1.0 µm/px (`--patch-size 256 --mpp 1.0`);
   `hipt` at 256 px.
6. `gpfm` - the card recommends 512 px tiles at 40× (~0.25 µm/px); raw2features defaults
   to 0.5 (the 20×-equivalent convention). To follow the card literally,
   `--patch-size 512 --mpp 0.25`.
7. `gpfm` - **licence conflict.** The Hugging Face card metadata says `mit`, but the
   source repo's `LICENSE` file (birkhoffkiki/GPFM) is verbatim CC-BY-NC-ND-4.0. The two
   disagree; verify before relying on either.
8. `phikon` / `phikon_v2` - Owkin's own non-commercial licence (HF tag `other`; research
   use by non-profit only). `phikon_v2` ships a distinct licence file from `phikon`.
9. `lunit_*` - the "Lunit. Inc Public License for Benchmarking Self-supervised Learning on
   Diverse Pathology Datasets" (HF tag `lunit-non-commercial`).
10. GPL-3.0 weights (`retccl`, `ctranspath`) - GPL-3.0 is copyleft (it carries derivative
    -code obligations); the authors' repos (RetCCL, TransPath) additionally state the
    weights are "available for non-commercial academic purposes." Read both before use.
11. `ctranspath` - pretrained at 20×, but the authors recommend **1.0 µm/px (10×)**
    downstream; raw2features defaults to 0.5. Override with `--mpp 1.0` to follow them.
12. `kronos` - a multiplex (spatial-proteomics) encoder, not H&E; it has no single µm/px.
    See `MODALITIES.md`.
13. `kaiko_vitl` - kaiko.ai's earlier ViT-L (distinct from `midnight`); norm is symmetric
    `[0.5]` from the card, **not** the pretrained_cfg ImageNet default (the usual base-arch
    trap), so it's set explicitly in the registry.
14. `biomedclip` - a **general-biomedical** CLIP (trained on PubMed Central figures), not a
    pathology-native H&E encoder; useful as a VLM baseline, weaker as a tissue patch
    extractor than the pathology-trained models. Its embedding is the raw (un-L2-normalised)
    `encode_image` projection, like `quiltnet`.
15. `plip` - its MIT licence is declared only in the project `setup.py` (no `LICENSE` file or
    HF YAML `license` field); confirm before relying on it. Loads via transformers
    `CLIPModel.get_image_features` (the `clip_hf` family), distinct from the `open_clip` pair.
16. `keep` - the authors specify ViT-L/16, 224 px, bicubic interpolation, and ImageNet
    normalisation, but neither µm/px nor magnification. The registry therefore leaves
    `recommended_mpp` unset; choose `--mpp` explicitly for a pathology protocol. The
    output is the official 768-d projected, L2-normalised image feature.

## Scale: the µm/px default

raw2features extracts at a physical scale (µm/px), not a magnification - and the **basis**
column above records, per model, whether that scale is `stated` (the authors printed a
µm/px figure), `mag→` (a magnification we converted by convention), or `conv.` (a field
default with no source). Omit `--mpp` and each model runs at its `µm/px` default and the
CLI echoes `target_mpp: 0.5 (auto)`; pass `--mpp` to override (it prints an informational
note if it differs). Requesting models whose defaults
disagree stops the run and asks you to pick one.

> **Why µm/px, not magnification.** "20×" is not a fixed physical scale - it depends on the
> scanner's optics and sensor pixel size, so the same nominal magnification lands at
> different µm/px on different scanners (and even different slides in one cohort). In a
> metadata scan of real public slides, those labelled "20×" ranged **0.23-0.50 µm/px**
> (TCGA/CPTAC/ccRCC/sarcoma, n≈1,290) - some "20×" slides are physically 40× sampling.
> Microns-per-pixel is the objective measurement; DICOM WSI and OME-NGFF record physical
> *pixel spacing* and treat magnification as advisory. So raw2features takes **0.5 µm/px as
> the default and extracts at exactly that**, resampling every slide to one comparable
> physical scale regardless of what its "20×" happens to mean.

`resnet50` and `dinov2` are ImageNet baselines and scale-agnostic. `keep` and
`openmidnight` are different: they are pathology-specific, but their authors publish no
physical scale. All four have no model-specific MPP in the registry and therefore fall
back to **1.0 µm/px** unless run alongside a model that supplies one or given an explicit
`--mpp`; for KEEP and OpenMidnight, explicitly choosing the protocol's MPP is recommended.

> **Adding a model - preprocessing caveat.** Source mean/std from the authors' actual usage
> code, not a model's timm `pretrained_cfg` field: that field can be an un-overridden
> base-arch default (several ViTs default to `[0.5, 0.5, 0.5]`) rather than the real
> training norm. The two disagreed for mSTAR - its config reported `[0.5]`, its README used
> ImageNet - and we follow the README. `transform_source: pretrained_cfg` only guards
> against drift between our value and the config; it can't catch a config that's wrong to
> begin with.

## Install notes for the special families

- `conch` - needs the `conch` extra **plus** its non-PyPI git package, pinned to the
  audited revision:
  `pip install "raw2features[conch]" && pip install git+https://github.com/Mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62`.
- `conch_v1_5` / `titan` - the CONCH v1.5 tile encoder is loaded via the gated TITAN model
  (`return_conch`), so it needs the `[models]` extra and an accepted `MahmoodLab/TITAN`
  gate. Extract at `--patch-size 512`, then `-s titan` for slide embeddings.
- `hibou_l` / `hibou_b` - need the `[hibou]` extra, which pins `transformers<5` because the
  models' remote code imports the removed `transformers.onnx`. Validated under that extra;
  available but not validated on transformers 5.
- `musk` - needs the `[musk]` extra; run at `--patch-size 384`.
- `keep` - included in `[models]`. Its local image-only wrapper reads the pinned,
  SHA-256-verified safetensors file and does not execute the repository's remote code or
  construct its unused BERT text tower.
- `kronos` - needs the `[kronos]` extra; multiplex only (see `MODALITIES.md`).
- `ctranspath` - custom `ConvStem` patch-embed (`embedders/convstem.py`), loaded via the
  1aurent modern-timm mirror (no pinned timm fork needed).
- `gpfm`, `retccl`, `sp22m`, `hipt`, `lunit_bt/mocov2/swav` - built from a base timm arch +
  a pinned checkpoint (`checkpoint:` in the registry), URL/sha256-verified before load.

More encoders are planned. The model registry (`src/raw2features/embedders/registry.yaml`) is the live list of what ships today.
