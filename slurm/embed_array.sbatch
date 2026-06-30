#!/bin/bash
# raw2features - SLURM array: one slide == one array task.
#
# Plain terms: embed a whole folder of slides on the cluster, ONE GPU job per slide;
# resumable -- re-submit to finish what's left. New here? slurm/README.md explains these
# scripts in plain language and which one to use.
#
# Each task is IDEMPOTENT: it first runs `raw2features verify`, which validates
# the ACTUAL output store (every requested features/<model> at (n_patches, dim),
# finite, coords aligned) - not merely the receipt. If the slide is already
# complete it exits 0 without recomputing, so re-submitting the same array
# safely resumes only the missing/failed slides.
#
# Heavy IO + GPU work runs on the compute node only. The slide list is built by a
# TOP-LEVEL glob of $SLIDE_DIR/*.zarr - never a recursive `find` over the stores.
#
# ---------------------------------------------------------------------------
# Submit (array range must match the slide count; create logs/ first because
# SLURM opens the --output/--error files before this script runs):
#
#   export SLIDE_DIR=/path/to/slides/raw
#   export OUT_DIR=/path/to/embeddings
#   export MODELS="uni resnet50"          # gated models need HF_TOKEN exported
#   export HF_TOKEN=hf_...                 # only if MODELS includes a gated one
#   mkdir -p logs
#   N=$(ls -d "$SLIDE_DIR"/*.zarr 2>/dev/null | wc -l)
#   sbatch --array=0-$((N-1))%64 \
#          --export=ALL,SLIDE_DIR,OUT_DIR,MODELS,HF_TOKEN \
#          slurm/embed_array.sbatch
# ---------------------------------------------------------------------------
#
#SBATCH --job-name=r2f-embed
#SBATCH --partition=gpu                 # adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/r2f-%A_%a.out
#SBATCH --error=logs/r2f-%A_%a.err

set -euo pipefail

# Deterministic slide ordering across ALL array tasks, regardless of node
# locale: pathname-expansion results are sorted by LC_COLLATE, so pin it to
# byte order -- otherwise task i could map to a different slide on a node with
# a different locale, breaking the one-slide-per-task contract.
export LC_COLLATE=C

# ---- configuration (override via the environment / --export) --------------
: "${SLIDE_DIR:?set SLIDE_DIR to the directory holding the *.zarr stores}"
: "${OUT_DIR:?set OUT_DIR for the *.embeddings.zarr outputs}"
: "${RECEIPTS_DIR:=$OUT_DIR/receipts}"
: "${REPO_DIR:=$HOME/raw2features}"
: "${VENV:=$REPO_DIR/.venv}"
: "${MODELS:=uni resnet50}"             # space-separated; each becomes a -f flag
: "${MPP:=}"                            # empty => auto (each model's recommended MPP)
: "${PATCH:=224}"
: "${STEP:=}"                           # empty => no overlap (step == patch)
: "${TISSUE_THRESHOLD:=0.1}"
: "${AMP:=bf16}"
: "${BATCH_SIZE:=256}"
: "${READ_BLOCK:=8}"                    # read patches in NxN blocks (1=off); bump to 16 on a latency-bound shared FS
: "${EMIT_GEOJSON:=0}"                  # 1 => also write <slide>.patches.geojson
# For gated models (uni, uni2_h) export HF_TOKEN, or have a cached HF login on
# the shared filesystem so it is visible on the compute node.

mkdir -p "$OUT_DIR" "$RECEIPTS_DIR"

# ---- slide list: TOP-LEVEL glob ONLY (never recurse into the stores) ------
shopt -s nullglob
SLIDES=("$SLIDE_DIR"/*.zarr)
shopt -u nullglob
N=${#SLIDES[@]}
if (( N == 0 )); then
  echo "ERROR: no *.zarr stores found in $SLIDE_DIR" >&2
  exit 1
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX >= N )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range (have $N slides, 0..$((N-1)))" >&2
  exit 1
fi
SLIDE="${SLIDES[$IDX]}"
echo "[$(date -u +%FT%TZ)] task ${IDX}/$((N-1)) on $(hostname): ${SLIDE}"

# ---- activate the project venv --------------------------------------------
# venv activate scripts reference unset vars; relax -u just around the source.
set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
set -u

# ---- repeated -f flags + the content-affecting flags shared by verify+embed
# These MUST be identical for `verify` and `embed` so the config hash matches.
MODEL_FLAGS=()
for m in $MODELS; do MODEL_FLAGS+=(-f "$m"); done

# Only pin --mpp when MPP is set; empty => let the CLI auto-resolve the model's
# recommended MPP (identical for verify + embed, so the config hash still matches).
MPP_FLAG=()
[ -n "$MPP" ] && MPP_FLAG=(--mpp "$MPP")

COMMON=(
  "${MODEL_FLAGS[@]}"
  --reader omezarr
  "${MPP_FLAG[@]}"
  --patch-size "$PATCH"
  --tissue-threshold "$TISSUE_THRESHOLD"
  --amp "$AMP"
)
[[ -n "$STEP" ]] && COMMON+=(--step "$STEP")

# ---- skip-if-complete (validates the actual output, not just the receipt) -
if raw2features verify "$SLIDE" --receipts-dir "$RECEIPTS_DIR" "${COMMON[@]}" --quiet; then
  echo "[$(date -u +%FT%TZ)] already complete; skipping ${SLIDE}"
  exit 0
fi

# ---- embed (GPU) ----------------------------------------------------------
# --read-block is runtime-only (doesn't affect the config hash), so it lives on
# EMBED_FLAGS, not COMMON -- `verify` neither needs nor accepts it.
EMBED_FLAGS=("${COMMON[@]}" --batch-size "$BATCH_SIZE" --read-block "$READ_BLOCK" --receipts-dir "$RECEIPTS_DIR")
[[ "$EMIT_GEOJSON" == "1" ]] && EMBED_FLAGS+=(--emit-geojson)

srun raw2features embed "$SLIDE" "$OUT_DIR" "${EMBED_FLAGS[@]}"
echo "[$(date -u +%FT%TZ)] done ${SLIDE}"
