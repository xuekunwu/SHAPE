#!/bin/bash
# ModalityProbe-family staged orchestration -- same stage-by-stage mechanics as run_staged.sh
# (refresh the CONFIG.json deadline and inject stage_env before each `cli stage` call, so a human can
# run GPU stages on a GPU node and CPU stages anywhere, each reading the previous stage's on-disk
# artifacts), but inits via init_run_modality_probe.py (dataset-aware: ModalityProbe/2/3) instead of
# harness/init_run.py, which is hardcoded to CPJUMP1's own manifest-gated loader and cannot be reused
# unmodified for a ModalityProbe prep.
#
#   bash benchmarks/full_CPJUMP1/runners/run_staged_modality_probe.sh DATASET ARM PREP RUN ASSETS [STAGES...]
#     DATASET  ModalityProbe | ModalityProbe2 | ModalityProbe3
#     ARM      control | shape | codex | claude | biomni
#     PREP     prepared data dir (output of prepare_modality_probeN.py)
#     RUN      fresh (or in-progress) run dir
#     ASSETS   assets root (verified cpsam/DINOv3 cache)
#     STAGES   subset to run this invocation (default: all five, in order)
#
# Same env contract as run_staged.sh: MODEL / ARM_ENV / CORES / SMOKE / SHAPE_SAM_SERVER_URL are read
# from the environment if set, not required. segment can delegate to a remote SAM service via
# $SHAPE_SAM_SERVER_URL; embed's SSL training needs a local GPU.
set -eo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env
[ -n "${ARM_ENV:-}" ] && [ -f "$ARM_ENV" ] && . "$ARM_ENV"
set +a
CODEX_BIN="${CODEX_BIN:-$(ls -d "$HOME"/shape/cache/envs/codex-*/bin 2>/dev/null | sort -V | tail -1 || true)}"
if [ -n "$CODEX_BIN" ] && [ -d "$CODEX_BIN" ]; then export PATH="$CODEX_BIN:$PATH"; fi
PY="${PYTHON:-python}"
CLI="benchmarks.full_CPJUMP1.harness.cli"

DATASET="${1:?usage: run_staged_modality_probe.sh DATASET ARM PREP RUN ASSETS [STAGES...]}"
ARM="${2:?need ARM}"
PREP="${3:?need PREP (prepared data dir)}"
RUN="${4:?need RUN (run dir)}"
ASSETS="${5:?need ASSETS (assets root)}"
shift 5 || true
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(segment crop embed batch cluster)
# SHAPE runs cohort + SSL early stopping (user decision, matches seed_run.sh/run_replica.sh's established
# convention for the shape arm); opt-in at the tool level ($FULL_CPJUMP1_EARLY_STOP, pipeline.py), always on
# here rather than left to the caller so every ModalityProbe-family staged run matches that convention.
[ "$ARM" = shape ] && export FULL_CPJUMP1_EARLY_STOP=1

COMMIT="$(git rev-parse HEAD)"
echo "== $DATASET staged: arm=$ARM run=$RUN stages=${STAGES[*]} commit=${COMMIT:0:8} =="

# ---- initialise the run (writes CONFIG.json, images, reasoning, ssl budget) if absent ----
if [ ! -f "$RUN/CONFIG.json" ]; then
  echo "[init] creating run dir + CONFIG.json (init_run_modality_probe, no stages)"
  "$PY" -m benchmarks.full_CPJUMP1.runners.init_run_modality_probe --arm "$ARM" --dataset "$DATASET" \
    ${MODEL:+--model "$MODEL"} --assets-root "$ASSETS" --prep "$PREP" --out "$RUN" \
    ${CORES:+--cores "$CORES"} ${SMOKE:+--smoke}
fi

# ---- run each requested stage against the shared run dir ----
for st in "${STAGES[@]}"; do
  echo "== STAGE $st =="
  "$PY" - "$RUN" <<'PYBUMP'
import json, sys, time
cfg = sys.argv[1] + "/CONFIG.json"
c = json.load(open(cfg))
c["deadline"] = time.time() + float(c.get("budget_hours", 11)) * 3600
json.dump(c, open(cfg, "w"), indent=2)
print(f"[deadline] refreshed -> now + {c.get('budget_hours', 11)}h")
PYBUMP
  eval "$("$PY" - "$ARM" "$st" <<'PYENV'
import sys
from benchmarks.full_CPJUMP1.harness.pipeline import stage_env
for k, v in stage_env(sys.argv[1], sys.argv[2]).items():
    print(f'export {k}={v!r}')
PYENV
)"
  t0=$(date +%s)
  "$PY" -m "$CLI" stage "$st" --prep "$PREP" --run "$RUN"
  unset SHAPE_REASONING_EFFORT SHAPE_LLM_TIMEOUT
  echo "[stage $st] done in $(( $(date +%s) - t0 ))s -> $(ls "$RUN/${st}_COMPLETE.json" 2>/dev/null || echo 'NO COMPLETE MARKER')"
done

echo "== DONE. run dir: $RUN =="
echo "   artifacts: $(ls "$RUN"/*_COMPLETE.json 2>/dev/null | wc -l) stage(s) complete"
