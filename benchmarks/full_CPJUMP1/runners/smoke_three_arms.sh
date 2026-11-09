#!/bin/bash
# Smoke-validate the three non-control arms (shape, codex, claude) end-to-end on the staged
# segment+crop path with the REMOTE SAM pool, BEFORE committing tokens to full 1000-image runs.
# Each arm runs on its own tiny --smoke image subset in its own run dir. Continue-on-error so one
# arm's failure does not hide the others' results.
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
# codex 0.162.0 must win over the stale ~/.local/bin/codex 0.155.1
# codex CLI PATH is set by run_staged.sh (newest codex-*/bin; override $CODEX_BIN) -- not pinned here
# The parser splits SHAPE_SAM_SERVER_URL on "," expecting FULL urls per process -- build them
# (a "http://IP:p1,p2" shorthand would parse "p2" as a bare, invalid url).
SAM_IP="${SAM_IP:-204.12.190.241}"
SHAPE_SAM_SERVER_URL=""
for p in 8137 8138 8139 8140; do
  SHAPE_SAM_SERVER_URL="${SHAPE_SAM_SERVER_URL:+$SHAPE_SAM_SERVER_URL,}http://$SAM_IP:$p"
done
export SHAPE_SAM_SERVER_URL

PREP="data/CPJUMP1/prepared"
ASSETS="data/CPJUMP1/assets"
STAGED="benchmarks/full_CPJUMP1/runners/run_staged.sh"

echo "###### SMOKE validation: shape, codex, claude ######"
echo "SAM=$SHAPE_SAM_SERVER_URL  codex=$(command -v codex)  claude=$(command -v claude)"

for ARM in shape codex claude; do
  RUN="benchmarks/full_CPJUMP1/results/smoke_${ARM}"
  rm -rf "$RUN"
  echo
  echo "============================================================"
  echo "=== SMOKE $ARM -> $RUN ==="
  echo "============================================================"
  SMOKE=1 bash "$STAGED" "$ARM" "$PREP" "$RUN" "$ASSETS" segment crop \
    && echo "### SMOKE $ARM: OK ($(ls "$RUN"/*_COMPLETE.json 2>/dev/null | wc -l) stage markers)" \
    || echo "### SMOKE $ARM: FAILED (exit $?) -- see $RUN"
done

echo
echo "###### SMOKE validation done ######"
for ARM in shape codex claude; do
  RUN="benchmarks/full_CPJUMP1/results/smoke_${ARM}"
  echo "$ARM: $(ls "$RUN"/*_COMPLETE.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done
