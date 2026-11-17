#!/bin/bash
# CPJUMP1 staged orchestration — run the five stages ONE AT A TIME against a shared run dir,
# so GPU-bound stages (segment, embed) go on a GPU node while CPU stages (crop, batch,
# cluster) run anywhere, each reading the previous stage's on-disk artifacts. This is the
# staged pattern applied to CPJUMP1, using the harness's OWN
# single-stage worker (`cli stage <name> --prep --run`) rather than the monolithic
# `cli run`, so no stage logic is duplicated.
#
#   bash benchmarks/full_CPJUMP1/runners/run_staged.sh ARM PREP RUN ASSETS [STAGES...]
#     ARM      control | shape | codex | claude | biomni
#     PREP     prepared data dir (output of `cli prepare`)
#     RUN      fresh run dir (CONFIG.json is created on the first call)
#     ASSETS   assets root (verified cpsam/DINOv3 cache)
#     STAGES   subset to run this invocation (default: all five, in order)
#
# IMPORTANT: put RUN on the SHARED disk (e.g. benchmarks/full_CPJUMP1/results/<run>), NOT
# /tmp. cells.json records ABSOLUTE crop paths; if the segment/crop stages write them under
# /tmp on the CPU node, the embed stage on the GPU node resolves /tmp to ITS OWN local /tmp
# and fails FileNotFound. A shared-disk RUN keeps every stage's paths valid across nodes.
#
# Stage GPU needs (see pipeline.py): segment=GPU(SAM), crop=CPU, embed=GPU(SSL training),
# batch=CPU(Harmony), cluster=CPU. Segment can delegate to a remote SAM service if
# $SHAPE_SAM_SERVER_URL is set AND the harness is patched to honor it (see run_staged notes);
# embed's SSL training must run on a local GPU (the DINOv3 inference service cannot train).
#
# LLM creds come from the repo .env (LLM_API_KEY / LLM_BASE_URL for shape/codex/biomni;
# ANTHROPIC_API_KEY for claude once claude_backend is pointed at the same gateway).
set -eo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env
# Optional arm-specific override sourced AFTER .env so it wins (e.g. the claude arm routing to the
# Anthropic endpoint with its own LLM_BASE_URL/LLM_API_KEY, which the repo .env's OpenAI gateway
# values would otherwise clobber). Unset -> no-op, every other arm unchanged.
[ -n "${ARM_ENV:-}" ] && [ -f "$ARM_ENV" ] && . "$ARM_ENV"
set +a
# Put the newest locally-installed codex CLI first (needed over the stale ~/.local/bin/codex). Never
# pin a version here -- it breaks the moment the CLI is updated; discover the latest codex-*/bin,
# overridable with $CODEX_BIN (an absolute .../bin dir).
CODEX_BIN="${CODEX_BIN:-$(ls -d "$HOME"/shape/cache/envs/codex-*/bin 2>/dev/null | sort -V | tail -1 || true)}"
if [ -n "$CODEX_BIN" ] && [ -d "$CODEX_BIN" ]; then export PATH="$CODEX_BIN:$PATH"; fi
PY="${PYTHON:-python}"
CLI="benchmarks.full_CPJUMP1.harness.cli"

ARM="${1:?usage: run_staged.sh ARM PREP RUN ASSETS [STAGES...]}"
PREP="${2:?need PREP (prepared data dir)}"
RUN="${3:?need RUN (run dir)}"
ASSETS="${4:?need ASSETS (assets root)}"
shift 4 || true
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=(segment crop embed batch cluster)

COMMIT="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "== CPJUMP1 staged: arm=$ARM run=$RUN stages=${STAGES[*]} commit=${COMMIT:0:8} =="

# ---- initialise the run (writes CONFIG.json, images, reasoning, ssl budget) if absent ----
# init_run.py writes CONFIG.json WITHOUT running stages (unlike `cli run`, which is monolithic).
if [ ! -f "$RUN/CONFIG.json" ]; then
  echo "[init] creating run dir + CONFIG.json (init_run, no stages)"
  "$PY" -m benchmarks.full_CPJUMP1.harness.init_run --arm "$ARM" ${MODEL:+--model "$MODEL"} \
    --assets-root "$ASSETS" --prep "$PREP" --out "$RUN" ${CORES:+--cores "$CORES"} ${SMOKE:+--smoke}
fi

# ---- run each requested stage against the shared run dir ----
# Each stage worker checks that its per-stage reasoner env is present (check_stage_env):
# SHAPE_REASONING_EFFORT / SHAPE_LLM_TIMEOUT, which run_stages() normally injects. Here we
# ask the harness for the exact pair and export it before the stage, so a standalone stage
# matches a monolithic run byte-for-byte.
# The arm deadline in CONFIG.json is an ABSOLUTE timestamp set at init (budget_hours from then). In
# staged mode the stages run far apart in wall-clock (a human runs embed on the GPU node between
# them), so that single init-relative deadline expires mid-run ("arm budget exhausted" in batch/
# cluster). Each staged stage is its own process run on its own schedule, so refresh the deadline to
# now + budget_hours at the START of every stage -- each stage gets its full time budget, which is
# exactly the staged intent (the monolithic "one shared budget across all stages" does not apply
# when stages are launched separately, possibly on different nodes/days).
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
