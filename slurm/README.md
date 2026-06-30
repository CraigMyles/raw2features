# Running raw2features on a SLURM cluster

These scripts batch-embed a folder of slides on an HPC cluster (GPU nodes managed by
[SLURM](https://slurm.schedmd.com/)). **In plain terms:** you point them at a directory of
`*.zarr` slides and an output directory, and the cluster embeds them in parallel - one job
per slide (or per batch of slides) - picking up where it left off if anything is interrupted.

You don't need to be a SLURM expert. You set a few environment variables (where the slides
are, where output goes, which models), adjust the `#SBATCH` lines at the top of a script to
match your cluster (partition, GPU, time limit), and submit it. Everything else is handled.

## Which script do I use?

| Script | What it does | Use it when |
|--------|--------------|-------------|
| **`preflight.sh`** | A read-only sanity check, run on the **login node**: is the venv working, do the slides exist, is HuggingFace access OK? | **Always run this first** - it catches a misconfiguration before you spend GPU hours discovering it. |
| **`embed_array.sbatch`** | One cluster job **per slide**. Simplest mapping; one log file per slide. | The default. Small-to-medium cohorts. |
| **`embed_cohort.sbatch`** | One cluster job **per batch ("shard") of slides**, loading the models **once** per job instead of once per slide. | Large cohorts, where re-loading the model for every slide would waste time. |

**"Resumable" / "idempotent"** just means: if a run dies (hit the time limit, a node failed),
**re-submit the exact same command**. Slides that already finished are skipped - checked
against the *actual output store*, not just a marker file - and only the missing ones re-run.
You can't accidentally double-process a slide.

## Typical workflow

```bash
# 1. Tell the scripts where things are (gated models also need a HuggingFace token).
export SLIDE_DIR=/path/to/slides        # a directory of *.zarr slide stores
export OUT_DIR=/path/to/embeddings      # where the *.embeddings.zarr outputs go
export MODELS="uni resnet50"            # space-separated model names
export HF_TOKEN=hf_...                  # only if a model is gated (e.g. uni)

# 2. Sanity-check on the LOGIN node (read-only, no jobs submitted).
bash slurm/preflight.sh

# 3. Submit one job per slide (array size = number of slides).
mkdir -p logs
N=$(ls -d "$SLIDE_DIR"/*.zarr | wc -l)
sbatch --array=0-$((N-1))%64 \
       --export=ALL,SLIDE_DIR,OUT_DIR,MODELS,HF_TOKEN \
       slurm/embed_array.sbatch
```

For a large cohort, use `embed_cohort.sbatch` instead and set `NUM_SHARDS` to the array size
- the exact submit command is in the header of that file.

## Tuning (optional)

Everything is configured through environment variables with sensible defaults, listed at the
top of each script. The ones you're most likely to touch:

- `MODELS` - which encoders to run (space-separated).
- `MPP` - extraction scale; leave blank to use each model's recommended scale.
- `AMP` - precision (`bf16` is a good default on modern GPUs; `auto` uses each model's card).
- `BATCH_SIZE` - lower it if a GPU runs out of memory.
- `READ_BLOCK` - read patches in NxN blocks; bigger means fewer, larger reads, which helps on
  a slow shared filesystem (try `16`) at the cost of host RAM.

> **Cluster-specific:** adjust the `#SBATCH` directives (`--partition`, `--gres=gpu:1`,
> `--time`, `--mem`) near the top of each script to match your site - the values shipped are
> examples, not universal.
