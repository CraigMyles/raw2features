#!/bin/bash
# raw2features - warm-worker cohort array. Each task embeds a SHARD of slides,
# loading the models ONCE (amortises the per-slide model-load at cohort scale)
# via `raw2features embed-many`. Idempotent: per-slide receipts make re-submitting
# the array resume only the missing/failed slides.
#
# Plain terms: embed a folder of slides on the cluster, each job handling a BATCH of slides
# (faster than one-job-per-slide for big cohorts); resumable. New here? See slurm/README.md.
#
# Submit (NUM_SHARDS must equal the array size; %K caps concurrency to ~GPU count):
#
#   export SLIDE_DIR=/path/to/slides/raw
#   export OUT_DIR=/path/to/embeddings
#   export MODELS="uni resnet50"     # gated models need a cached HF login / HF_TOKEN
#   export NUM_SHARDS=48
#   mkdir -p logs
#   sbatch --array=0-$((NUM_SHARDS-1))%24 \
#          --partition=<your-gpu-partition> \
#          --export=ALL,SLIDE_DIR,OUT_DIR,MODELS,NUM_SHARDS,HF_TOKEN \
#          slurm/embed_cohort.sbatch
#
#SBATCH --job-name=r2f-cohort
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/r2fc-%A_%a.out
#SBATCH --error=logs/r2fc-%A_%a.err

set -euo pipefail

: "${SLIDE_DIR:?set SLIDE_DIR to the directory holding the slide stores}"
: "${OUT_DIR:?set OUT_DIR for the *.embeddings.zarr outputs}"
: "${NUM_SHARDS:?set NUM_SHARDS to the submitted array size}"
: "${GLOB:=*.zarr}"
: "${RECEIPTS_DIR:=$OUT_DIR/receipts}"
: "${REPO_DIR:=$HOME/raw2features}"
: "${VENV:=$REPO_DIR/.venv}"
: "${MODELS:=uni resnet50}"
: "${MPP:=}"                    # empty => auto (each model's recommended MPP)
: "${SOURCE_MPP:=}"             # level-0 MPP override for uncalibrated sources
: "${PATCH:=}"                  # empty => each model's recommended patch size
: "${STEP:=}"                   # empty => no overlap (step == patch)
: "${TISSUE_THRESHOLD:=0.1}"
: "${AMP:=auto}"               # each model's declared inference precision
: "${READ_WORKERS:=16}"        # 16 suits a latency-bound shared FS; 8 on local NVMe
: "${READ_BLOCK:=8}"           # read patches in NxN blocks (1=off); 16 helps on a latency-bound shared FS, costs host RAM
: "${BATCH_SIZE:=256}"

mkdir -p "$OUT_DIR" "$RECEIPTS_DIR" logs

set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
set -u

MODEL_FLAGS=()
for m in $MODELS; do MODEL_FLAGS+=(-m "$m"); done

echo "[$(date -u +%FT%TZ)] shard ${SLURM_ARRAY_TASK_ID:-0}/${NUM_SHARDS} on $(hostname)"
CMD=(
  raw2features embed-many "$SLIDE_DIR" "$OUT_DIR"
  --glob "$GLOB"
  "${MODEL_FLAGS[@]}"
  --tissue-threshold "$TISSUE_THRESHOLD"
  --amp "$AMP"
  --read-workers "$READ_WORKERS"
  --read-block "$READ_BLOCK"
  --batch-size "$BATCH_SIZE"
  --receipts-dir "$RECEIPTS_DIR"
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
  --num-shards "$NUM_SHARDS"
)
[[ -n "$MPP" ]] && CMD+=(--mpp "$MPP")
[[ -n "$SOURCE_MPP" ]] && CMD+=(--source-mpp "$SOURCE_MPP")
[[ -n "$PATCH" ]] && CMD+=(--patch-size "$PATCH")
[[ -n "$STEP" ]] && CMD+=(--step "$STEP")
srun "${CMD[@]}"
echo "[$(date -u +%FT%TZ)] shard ${SLURM_ARRAY_TASK_ID:-0} finished"
