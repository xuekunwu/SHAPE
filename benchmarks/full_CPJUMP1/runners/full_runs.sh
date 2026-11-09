#!/bin/bash
# Full 1000-image segment+crop for the validated arms, run SEQUENTIALLY (one arm at a time, per
# the user's "顺序跑") against the REMOTE SAM pool on the GPU node. embed runs later on the GPU
# node. Pass the arm list as args (default: shape codex). Continue-on-error so one arm's failure
# does not block the next; each arm keeps its own run dir under results/<arm>.
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
# codex CLI PATH is set by run_staged.sh (newest codex-*/bin; override $CODEX_BIN) -- not pinned here

# Full urls (one per SAM process); a "http://IP:p1,p2" shorthand parses "p2" as an invalid url.
SAM_IP="${SAM_IP:-204.12.190.241}"
SHAPE_SAM_SERVER_URL=""
for p in 8137 8138 8139 8140; do
  SHAPE_SAM_SERVER_URL="${SHAPE_SAM_SERVER_URL:+$SHAPE_SAM_SERVER_URL,}http://$SAM_IP:$p"
done
export SHAPE_SAM_SERVER_URL

PREP="data/CPJUMP1/prepared"
ASSETS="data/CPJUMP1/assets"
STAGED="benchmarks/full_CPJUMP1/runners/run_staged.sh"
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(shape codex)

echo "###### FULL segment+crop: ${ARMS[*]} ######"
echo "SAM=$SHAPE_SAM_SERVER_URL"

for ARM in "${ARMS[@]}"; do
  RUN="benchmarks/full_CPJUMP1/results/${ARM}"
  echo
  echo "============================================================"
  echo "=== FULL $ARM -> $RUN  ($(date -u +%H:%M:%SZ)) ==="
  echo "============================================================"
  bash "$STAGED" "$ARM" "$PREP" "$RUN" "$ASSETS" segment crop \
    && echo "### FULL $ARM: segment+crop OK" \
    || echo "### FULL $ARM: FAILED (exit $?) -- see $RUN"
done

echo
echo "###### FULL segment+crop done -- next: embed on the GPU node per arm ######"
for ARM in "${ARMS[@]}"; do
  RUN="benchmarks/full_CPJUMP1/results/${ARM}"
  echo "$ARM: $(ls "$RUN"/*_COMPLETE.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done
