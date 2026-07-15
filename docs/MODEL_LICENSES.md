# Model licences

`raw2features` (this code) is **MIT**. It downloads model weights at runtime from each
model's source - Hugging Face, or Google Drive (by a pinned file id + verified SHA-256) for
`chief` / `tangle` - under *your* acceptance of that model's licence/gate, and
**redistributes no weights**. The inference recipe for each model is re-implemented in our
own code with the source cited (`transform_source_url` in
`src/raw2features/embedders/registry.yaml`); we vendor no model code.

**You are responsible for complying with each model's own licence where applicable.** The
per-model `license:` field in `registry.yaml` is the machine-readable source of truth; the
table below lists each model's weights licence.

## Patch encoders

| model | weights licence | gated |
|---|---|---|
| `resnet50` | BSD-3-Clause (torchvision) | no |
| `dinov2` | Apache-2.0 | no |
| `gigapath` | Apache-2.0 | yes (login) |
| `h_optimus_0` | Apache-2.0 | yes (login) |
| `h0_mini` | CC-BY-NC-ND-4.0 | yes (approval) |
| `gpfm` | MIT (HF metadata) / CC-BY-NC-ND-4.0 (repo `LICENSE`) - conflict, verify | no |
| `midnight` | MIT | no |
| `openmidnight` | Apache-2.0 | yes (terms acceptance) |
| `openpath` | Apache-2.0 | no |
| `hibou_l` | Apache-2.0 | yes |
| `hibou_b` | Apache-2.0 | yes |
| `ctranspath` | GPL-3.0 | no |
| `uni` | CC-BY-NC-ND-4.0 | yes |
| `uni2_h` | CC-BY-NC-ND-4.0 | yes |
| `path_orchestra` | CC-BY-NC-ND-4.0 (PathOrchestra, non-commercial) | yes |
| `virchow2` | CC-BY-NC-ND-4.0 | yes |
| `conch` | CC-BY-NC-ND-4.0 | yes |
| `conch_v1_5` | CC-BY-NC-ND-4.0 (CONCH v1.5, bundled in gated TITAN) | yes |
| `phikon` | Owkin non-commercial | no |
| `phikon_v2` | Owkin non-commercial | no |
| `lunit_{dino,dino8,bt,mocov2,swav}` | Lunit non-commercial | no |
| `sp22m` | CC-BY-NC-SA-4.0 | no |
| `retccl` | GPL-3.0 | no |
| `hipt` | Apache-2.0 + Commons Clause | no |
| `h_optimus_1` | CC-BY-NC-ND-4.0 | yes (login) |
| `virchow` | Apache-2.0 | yes (login) |
| `musk` | CC-BY-NC-ND-4.0 | yes (login) |
| `mstar` | CC-BY-NC-ND-4.0 | yes (login) |
| `kaiko_vitl` | Kaiko Non-Commercial Public License v1 | no |
| `seal_conch` / `seal_univ2` | CC-BY-NC-ND-4.0 | yes |
| `quiltnet` | MIT | no |
| `biomedclip` | MIT | no |
| `plip` | MIT (project `setup.py`; no LICENSE file) | no |
| `keep` | MIT | no |
| `kronos` | CC-BY-NC-ND-4.0 (KRONOS, MahmoodLab; multiplex, non-commercial) | yes |

## Slide encoders

These aggregate a slide's patch features into one slide vector. Their **code** is our own
re-implementation (MIT, like the rest of raw2features); the **weights** carry the licences
below. `mean` / `max` / `meanmax` are weightless pooling baselines (MIT).

| model | weights licence | gated |
|---|---|---|
| `titan` | CC-BY-NC-ND-4.0 | yes |
| `prism` | CC-BY-NC-ND-4.0 | yes |
| `madeleine` | MIT (HF card) / CC-BY-NC-ND-4.0 (GitHub `LICENSE`) - conflict, verify | yes |
| `feather_conch_v15` / `feather_uni_v2` / `feather_uni` | CC-BY-NC-ND-4.0 | yes |
| `chief` | GPL-3.0 (weights; aggregator code MIT) | no (Google Drive) |
| `tangle` | CC-BY-NC-ND-4.0 | no (Google Drive) |

Check each model's licence on its model card before use.

## Quality control / preprocessing

QC tools run before or alongside extraction; like the encoders, our **code is an MIT
re-implementation** (a generic UNet++ via `segmentation-models-pytorch`) and the **weights**
carry the licence below. GrandQC weights are fetched on first use and never bundled.

| tool | weights licence | gated |
|---|---|---|
| `grandqc` (`--segmenter grandqc` / `--qc grandqc`) | CC-BY-NC-SA-4.0 (non-commercial, share-alike) | no (Zenodo) |

GrandQC: Weng et al., *Nat Commun* 2024 (`10.1038/s41467-024-54769-y`). The CC-BY-NC-SA-4.0
term is the licence declared on the **Zenodo weight deposits** (records 14507273 tissue /
14041538 artifact) - the GrandQC *code* is MIT, but the *weights* we fetch carry the
non-commercial share-alike terms. Its outputs (masks / QC scores) inherit those terms, so
writing them into a shared or redistributed store is opt-in.
