#!/bin/bash
# Launch script for the cell_state_HCMA end-to-end benchmark (SHAPE reference run).
#   bash benchmarks/cell_state_HCMA/run_hcma.sh <crop-manifest.parquet> <labels.parquet> [out-dir]
#
# The agent produces a label-free AnnData (embedding + Leiden cluster); the FIXED
# evaluator (evaluate.py) scores it against the held-out cell-type labels with the
# multi-source subset + image-grouped GroupKFold guards. Labels never reach the embedder.
#
# GPU is strongly recommended (DINOv3 contrastive training). Long-lived GPU work must be
# launched by the USER (the tool sandbox SIGKILLs background CUDA) -- run this via `!`.
set -eo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

CROPS="${1:?usage: run_hcma.sh <crop-manifest.parquet> <labels.parquet> [out-dir]}"
LABELS="${2:?need the held-out labels parquet [cell_id, cell_type]}"
OUT="${3:-solver_cache/HCMA/run1}"
PY="${PYTHON:-$REPO/.venv/bin/python}"

echo "== HCMA run =="
echo "repo=$REPO head=$(git rev-parse --short HEAD)"
echo "crops=$CROPS labels=$LABELS out=$OUT python=$PY"

exec "$PY" -m benchmarks.cell_state_HCMA.run_shape_hcma \
    --crop-manifest "$CROPS" \
    --labels "$LABELS" --join-key cell_id \
    --out-dir "$OUT" "${@:4}"
