#!/bin/bash
# Full REPLICA run on a field-replicate selection (data/CPJUMP1_seed<SEED>): prepare, then the arms
# control / shape / codex / claude, each into results/<arm>_seed<SEED>. Run on the GPU node -- segment
# delegates to the remote SAM pool ($SHAPE_SAM_SERVER_URL, else $SAM_URLS from gpu_services.sh), embed
# trains on the local GPU. The shape arm runs the FULL cohort (segment + crop) AND SSL early stopping
# ($FULL_CPJUMP1_EARLY_STOP=1); the other arms are unchanged. control is scored first (every arm's
# verdict is vs this replica's own control). Continue-on-error + resumable (re-run to pick up).
#
#   export SAM_URLS="http://gpu-node.example.net:8137,http://gpu-node.example.net:8138,..."   # from: gpu_services.sh urls
#   bash benchmarks/full_CPJUMP1/runners/run_replica.sh 20261009 [arms...]
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a

SEED="${1:?usage: run_replica.sh SEED [arms...]}"; shift || true
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(control shape codex claude)
export SHAPE_SAM_SERVER_URL="${SHAPE_SAM_SERVER_URL:-${SAM_URLS:?set SAM_URLS or SHAPE_SAM_SERVER_URL (gpu_services.sh urls)}}"

DATA="data/CPJUMP1_seed${SEED}"
PREP="$DATA/prepared"
ASSETS="data/CPJUMP1/assets"                 # shared: same genes/wells -> same DINOv3/CPSAM weights
STAGED="benchmarks/full_CPJUMP1/runners/run_staged.sh"
PY="${PYTHON:-$PWD/.venv/bin/python}"
FINALIZE() { "$PY" -m benchmarks.full_CPJUMP1.harness.finalize_run "$@"; }
CTRL="benchmarks/full_CPJUMP1/results/control_seed${SEED}"
AENV="$PWD/benchmarks/full_CPJUMP1/runners/.claude_api.env"
STAGES=(segment crop embed batch cluster)

echo "###### replica seed${SEED}: ${ARMS[*]} | SAM=$SHAPE_SAM_SERVER_URL ######"

# 1) prepare (idempotent)
[ -f "$PREP/PREPARE.json" ] || "$PY" -m benchmarks.full_CPJUMP1.harness.cli prepare --data "$DATA" --out "$PREP"

# 2) control baseline FIRST -- needed for every arm's delta/verdict
if printf '%s\n' "${ARMS[@]}" | grep -qx control; then
  echo "=== control -> $CTRL ($(date -u +%H:%M:%SZ)) ==="
  bash "$STAGED" control "$PREP" "$CTRL" "$ASSETS" "${STAGES[@]}" \
    && FINALIZE --prep "$PREP" --run "$CTRL" \
    || echo "### control FAILED -- re-run this script to resume"
fi

# 3) the arms (shape: cohort seg+crop + SSL early stopping; codex/claude: native)
for ARM in "${ARMS[@]}"; do
  [ "$ARM" = control ] && continue
  RUN="benchmarks/full_CPJUMP1/results/${ARM}_seed${SEED}"
  echo "=== $ARM -> $RUN ($(date -u +%H:%M:%SZ)) ==="
  (
    [ "$ARM" = shape ]  && export FULL_CPJUMP1_EARLY_STOP=1       # SSL early stopping for shape only
    [ "$ARM" = claude ] && export ARM_ENV="$AENV" SHAPE_API_ENV_FILE="$AENV"
    bash "$STAGED" "$ARM" "$PREP" "$RUN" "$ASSETS" "${STAGES[@]}"
  ) \
    && FINALIZE --prep "$PREP" --run "$RUN" --control "$CTRL" \
    && "$PY" -m benchmarks.full_CPJUMP1.harness.figures --run "$RUN" --level both \
    || echo "### $ARM FAILED (exit $?) -- re-run this script to resume"
done

echo "###### replica seed${SEED} done -> results/{control,shape,codex,claude}_seed${SEED}/RESULT.md ######"
