#!/bin/bash
# Phase A (CPU + remote SAM, NO local GPU): prepare the replica seeds and run segment + crop for every
# arm across them, fanned out $JOBS at a time against the SAM pool -- cranks parallelism so the slow
# per-image segment/crop of several arms overlap. embed + SSL + scoring is phase B on the GPU
# (replica_embed_score.sh). Continue-on-error + resumable (run_staged skips a completed stage's work by
# reusing its *_COMPLETE marker only when re-invoked for later stages; re-run to pick up a failed one).
#
#   export SAM_URLS="http://gpu-node.example.net:8137,..."              # from: gpu_services.sh urls
#   JOBS=4 ARMS="control shape codex claude" \
#     bash benchmarks/full_CPJUMP1/runners/replica_segcrop.sh 20261009 20261010
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
export SHAPE_SAM_SERVER_URL="${SHAPE_SAM_SERVER_URL:-${SAM_URLS:?set SAM_URLS or SHAPE_SAM_SERVER_URL (gpu_services.sh urls)}}"
# codex CLI on PATH for the codex arm
CODEX_BIN="${CODEX_BIN:-$(ls -d "$HOME"/shape/cache/envs/codex-*/bin 2>/dev/null | sort -V | tail -1 || true)}"
[ -n "$CODEX_BIN" ] && [ -d "$CODEX_BIN" ] && export PATH="$CODEX_BIN:$PATH"

JOBS="${JOBS:-4}"
read -r -a ARMS <<< "${ARMS:-control shape codex claude}"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && { echo "usage: replica_segcrop.sh SEED [SEED...]"; exit 2; }
ASSETS="data/CPJUMP1/assets"
STAGED="benchmarks/full_CPJUMP1/runners/run_staged.sh"
PY="${PYTHON:-$PWD/.venv/bin/python}"
AENV="$PWD/benchmarks/full_CPJUMP1/runners/.claude_api.env"
LOG="benchmarks/full_CPJUMP1/results/.replica_segcrop_logs"; mkdir -p "$LOG"

echo "###### replica segment+crop | seeds=${SEEDS[*]} arms=${ARMS[*]} JOBS=$JOBS | SAM=$SHAPE_SAM_SERVER_URL ######"

# prepare each seed ONCE first (idempotent) so the per-arm jobs do not race on it
for SEED in "${SEEDS[@]}"; do
  PREP="data/CPJUMP1_seed${SEED}/prepared"
  [ -f "$PREP/PREPARE.json" ] || "$PY" -m benchmarks.full_CPJUMP1.harness.cli prepare \
      --data "data/CPJUMP1_seed${SEED}" --out "$PREP"
done

one() {  # arm seed
  local arm="$1" seed="$2" prep="data/CPJUMP1_seed${2}/prepared" run="benchmarks/full_CPJUMP1/results/${1}_seed${2}"
  (
    [ "$arm" = claude ] && export ARM_ENV="$AENV" SHAPE_API_ENV_FILE="$AENV"
    bash "$STAGED" "$arm" "$prep" "$run" "$ASSETS" segment crop
  ) >"$LOG/${arm}_seed${seed}.log" 2>&1 \
    && echo "### OK   segcrop $arm seed$seed" \
    || echo "### FAIL segcrop $arm seed$seed (exit $?) -> $LOG/${arm}_seed${seed}.log"
}

# fan out (arm x seed) with a $JOBS-wide pool
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do
    one "$arm" "$seed" &
    while [ "$(jobs -r | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done
done
wait
echo "###### segment+crop done. Phase B (GPU): replica_embed_score.sh ${SEEDS[*]} ######"
