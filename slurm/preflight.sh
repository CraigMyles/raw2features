#!/bin/bash
# raw2features SLURM pre-flight: run this on your cluster's LOGIN node before
# submitting embed_array.sbatch. It validates the environment so you don't burn
# a whole array on a misconfiguration (wrong-arch venv, missing HF access, empty
# slide dir, ...). Read-only except for `mkdir -p` of OUT_DIR/logs.
#
# New here? slurm/README.md explains the cluster workflow + which script to use.
#
#   export SLIDE_DIR=/path/to/slides/raw  OUT_DIR=/path/to/embeddings
#   export MODELS="uni resnet50"   HF_TOKEN=hf_...   # token only if gated
#   bash slurm/preflight.sh
#
# Exits 0 if everything is ready, 1 if any check fails.

export LC_COLLATE=C
: "${REPO_DIR:=$HOME/raw2features}"
: "${VENV:=$REPO_DIR/.venv}"
: "${MODELS:=uni resnet50}"
: "${OUT_DIR:=}"
: "${SLIDE_DIR:=}"

FAIL=0
ok()   { printf '  \033[32mPASS\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; FAIL=1; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$1"; }

echo "raw2features pre-flight on $(hostname) ($(uname -m))"
echo

echo "[1] project venv ($VENV)"
if [[ -f "$VENV/bin/activate" ]]; then
  set +u; # shellcheck disable=SC1091
  source "$VENV/bin/activate"; set -u
  if VER=$(raw2features version 2>&1); then ok "raw2features $VER"; else bad "raw2features not runnable: $VER"; fi
  # importing torch catches an aarch64 venv copied onto x86 (or vice-versa)
  if TI=$(python -c "import torch;print(torch.__version__, 'cuda_build='+str(torch.version.cuda))" 2>&1); then
    ok "torch imports ($TI)"
  else
    bad "torch import failed -- rebuild the venv on THIS arch ($(uname -m)) with 'uv sync': ${TI##*Error}"
  fi
else
  bad "no venv at $VENV -- create it on the target machine: cd $REPO_DIR && uv sync --extra zarr --extra image --extra torch --extra models"
fi

echo "[2] slide directory (SLIDE_DIR=$SLIDE_DIR)"
if [[ -z "$SLIDE_DIR" ]]; then bad "SLIDE_DIR is unset"
elif [[ ! -d "$SLIDE_DIR" ]]; then bad "SLIDE_DIR does not exist"
else
  shopt -s nullglob; SLIDES=("$SLIDE_DIR"/*.zarr); shopt -u nullglob
  N=${#SLIDES[@]}
  if (( N == 0 )); then bad "no *.zarr stores directly under SLIDE_DIR (top-level only; no recursion)"
  else
    ok "$N slide store(s); first=$(basename "${SLIDES[0]}") last=$(basename "${SLIDES[N-1]}")"
    for s in "${SLIDES[@]}"; do [[ -d "$s" ]] || warn "$(basename "$s") is a FILE, not a store dir"; done
  fi
fi

echo "[3] models ($MODELS) + HuggingFace access"
if command -v python >/dev/null 2>&1; then
  GATED=$(python - <<PY 2>/dev/null
from raw2features.embedders.model_registry import load_registry
reg = load_registry(); req = "$MODELS".split()
unknown = [m for m in req if m not in reg]
gated = [m for m in req if m in reg and reg[m].gated]
print("UNKNOWN " + " ".join(unknown))
print("GATED " + " ".join(gated))
PY
)
  UNK=$(sed -n 's/^UNKNOWN //p' <<<"$GATED"); GAT=$(sed -n 's/^GATED //p' <<<"$GATED")
  [[ -n "${UNK// }" ]] && bad "unknown model(s): $UNK (see 'raw2features models')" || ok "all models known"
  if [[ -n "${GAT// }" ]]; then
    if WHO=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null); then
      ok "gated models [$GAT] -> HF authenticated as '$WHO' (ensure access is granted on the model pages)"
    else
      bad "gated models [$GAT] but no HF login -- export HF_TOKEN (and pass it via --export) or 'huggingface-cli login'"
    fi
  else
    ok "no gated models requested (no HF token needed)"
  fi
fi

echo "[4] output + logs"
if [[ -n "$OUT_DIR" ]]; then
  if mkdir -p "$OUT_DIR" 2>/dev/null && [[ -w "$OUT_DIR" ]]; then ok "OUT_DIR writable ($OUT_DIR)"; else bad "OUT_DIR not writable ($OUT_DIR)"; fi
else warn "OUT_DIR unset (required at submit time)"; fi
mkdir -p logs && ok "logs/ ready (SLURM opens --output here before the job runs)"

echo
if (( FAIL == 0 )) && [[ -n "$SLIDE_DIR" && -d "$SLIDE_DIR" ]]; then
  shopt -s nullglob; SLIDES=("$SLIDE_DIR"/*.zarr); shopt -u nullglob; N=${#SLIDES[@]}
  echo "READY. Submit with:"
  echo "  sbatch --array=0-$((N-1))%64 --export=ALL,SLIDE_DIR,OUT_DIR,MODELS,HF_TOKEN slurm/embed_array.sbatch"
else
  echo "NOT READY -- resolve the FAIL items above, then re-run this pre-flight."
fi
exit $FAIL
