#!/bin/bash
# Run ONE arm's embed stage on the GPU node (local GPU for SSL training/inference). embed needs the
# LLM gateway (build_engine: shape's own engine in-process, or the codex/claude CLI for the SSL
# search) but NOT a segmenter, so no SHAPE_SAM_SERVER_URL here. Per-arm env:
#   shape  -> repo .env OpenAI gateway (gpt-5.6-luna), SHAPE engine in-process
#   codex  -> repo .env gateway + newest installed codex on PATH
#   claude -> Anthropic route via .claude_api.env (ARM_ENV + SHAPE_API_ENV_FILE) + claude CLI
#
#   bash benchmarks/full_CPJUMP1/runners/run_arm_embed.sh ARM
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# codex CLI PATH is set by run_staged.sh (newest codex-*/bin; override $CODEX_BIN) -- not pinned here

ARM="${1:?usage: run_arm_embed.sh ARM (shape|codex|claude)}"
RUN="benchmarks/full_CPJUMP1/results/${ARM}"
[ -f "$RUN/crop_COMPLETE.json" ] || { echo "crop not complete for $ARM ($RUN) -- run segment+crop first"; exit 1; }

if [ "$ARM" = claude ]; then
  AENV="${CLAUDE_API_ENV:-$PWD/benchmarks/full_CPJUMP1/runners/.claude_api.env}"
  [ -f "$AENV" ] || { echo "missing $AENV (anthropic api.env)"; exit 1; }
  export ARM_ENV="$AENV" SHAPE_API_ENV_FILE="$AENV"
fi

echo "###### EMBED $ARM -> $RUN  ($(date -u +%H:%M:%SZ)) ######"
bash benchmarks/full_CPJUMP1/runners/run_staged.sh "$ARM" \
  "data/CPJUMP1/prepared" "$RUN" "data/CPJUMP1/assets" embed \
  && echo "### EMBED $ARM OK -> $(ls "$RUN"/embed_COMPLETE.json 2>/dev/null)" \
  || echo "### EMBED $ARM FAILED (exit $?)"
