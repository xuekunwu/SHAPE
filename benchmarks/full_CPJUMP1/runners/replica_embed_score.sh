#!/bin/bash
# Phase B (GPU): embed + batch + cluster + finalize + figures per arm, for the given replica seeds.
# Run AFTER replica_segcrop.sh (phase A, which did segment + crop). SSL training is GPU-bound, so arms
# run sequentially on the one GPU. control is scored first (every arm's verdict is vs this replica's own
# control). shape embeds with SSL early stopping ($FULL_CPJUMP1_EARLY_STOP=1). Continue-on-error + resumable.
#
#   JOBS=1 ARMS="control shape codex claude" \
#     bash benchmarks/full_CPJUMP1/runners/replica_embed_score.sh 20261009 20261010
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
CODEX_BIN="${CODEX_BIN:-$(ls -d "$HOME"/shape/cache/envs/codex-*/bin 2>/dev/null | sort -V | tail -1 || true)}"
[ -n "$CODEX_BIN" ] && [ -d "$CODEX_BIN" ] && export PATH="$CODEX_BIN:$PATH"

read -r -a ARMS <<< "${ARMS:-control shape codex claude}"
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && { echo "usage: replica_embed_score.sh SEED [SEED...]"; exit 2; }
ASSETS="data/CPJUMP1/assets"
STAGED="benchmarks/full_CPJUMP1/runners/run_staged.sh"
PY="${PYTHON:-$PWD/.venv/bin/python}"
AENV="$PWD/benchmarks/full_CPJUMP1/runners/.claude_api.env"
FINALIZE() { "$PY" -m benchmarks.full_CPJUMP1.harness.finalize_run "$@"; }
STAGES=(embed batch cluster)

for SEED in "${SEEDS[@]}"; do
  PREP="data/CPJUMP1_seed${SEED}/prepared"
  CTRL="benchmarks/full_CPJUMP1/results/control_seed${SEED}"
  echo "###### replica seed${SEED}: embed+score ${ARMS[*]} ######"
  # control first (the baseline every arm is scored against)
  if printf '%s\n' "${ARMS[@]}" | grep -qx control && [ ! -f "$CTRL/RESULT.json" ]; then
    echo "=== control -> $CTRL ($(date -u +%H:%M:%SZ)) ==="
    bash "$STAGED" control "$PREP" "$CTRL" "$ASSETS" "${STAGES[@]}" \
      && FINALIZE --prep "$PREP" --run "$CTRL" \
      && "$PY" -m benchmarks.full_CPJUMP1.harness.figures --run "$CTRL" --level both \
      || echo "### control seed$SEED FAILED -- re-run to resume"
  fi
  for ARM in "${ARMS[@]}"; do
    [ "$ARM" = control ] && continue
    RUN="benchmarks/full_CPJUMP1/results/${ARM}_seed${SEED}"
    [ -f "$RUN/RESULT.json" ] && { echo "=== $ARM seed$SEED already scored -- skip ==="; continue; }
    echo "=== $ARM -> $RUN ($(date -u +%H:%M:%SZ)) ==="
    (
      [ "$ARM" = shape ]  && export FULL_CPJUMP1_EARLY_STOP=1
      [ "$ARM" = claude ] && export ARM_ENV="$AENV" SHAPE_API_ENV_FILE="$AENV"
      bash "$STAGED" "$ARM" "$PREP" "$RUN" "$ASSETS" "${STAGES[@]}"
    ) \
      && FINALIZE --prep "$PREP" --run "$RUN" --control "$CTRL" \
      && "$PY" -m benchmarks.full_CPJUMP1.harness.figures --run "$RUN" --level both \
      || echo "### $ARM seed$SEED FAILED (exit $?) -- re-run to resume"
  done
done
echo "###### embed+score done -> results/{control,shape,codex,claude}_seed*/RESULT.md ######"
