#!/bin/bash
# Full 1000-image segment+crop for the claude arm, routed to the REAL Anthropic endpoint (the repo
# .env's OpenAI gateway serves gpt-5.6-luna for shape/codex but NOT claude-sonnet-5 -> 404). The
# claude backend sources SHAPE_API_ENV_FILE for its key and checks LLM_BASE_URL==protocol.BASE_URL,
# so both the harness env (ARM_ENV, sourced after .env inside run_staged) and the key helper point
# at api.anthropic.com with the sk-ant key. SAM stays on the remote GPU pool like the other arms.
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# codex CLI PATH is set by run_staged.sh (newest codex-*/bin; override $CODEX_BIN) -- not pinned here

# The Anthropic api.env (export LLM_BASE_URL=https://api.anthropic.com + export LLM_API_KEY=your-api-key).
# Override CLAUDE_API_ENV to point elsewhere; it holds a secret so keep it OUT of the repo.
AENV="${CLAUDE_API_ENV:-$PWD/benchmarks/full_CPJUMP1/runners/.claude_api.env}"
[ -f "$AENV" ] || { echo "missing $AENV -- write 'export LLM_BASE_URL=https://api.anthropic.com' + 'export LLM_API_KEY=<sk-ant>' there, or set CLAUDE_API_ENV"; exit 1; }
export ARM_ENV="$AENV"               # run_staged sources this AFTER .env -> LLM_BASE_URL=anthropic wins
export SHAPE_API_ENV_FILE="$AENV"    # the claude key helper sources the same file

SAM_IP="${SAM_IP:-204.12.190.241}"
SHAPE_SAM_SERVER_URL=""
for p in 8137 8138 8139 8140; do
  SHAPE_SAM_SERVER_URL="${SHAPE_SAM_SERVER_URL:+$SHAPE_SAM_SERVER_URL,}http://$SAM_IP:$p"
done
export SHAPE_SAM_SERVER_URL

RUN="benchmarks/full_CPJUMP1/results/claude"
echo "###### FULL claude (Anthropic-direct) -> $RUN  ($(date -u +%H:%M:%SZ)) ######"
echo "SAM=$SHAPE_SAM_SERVER_URL  ARM_ENV=$ARM_ENV"
bash benchmarks/full_CPJUMP1/runners/run_staged.sh claude \
  "data/CPJUMP1/prepared" "$RUN" "data/CPJUMP1/assets" segment crop \
  && echo "### FULL claude: segment+crop OK" \
  || echo "### FULL claude: FAILED (exit $?) -- see $RUN"
echo "claude: $(ls "$RUN"/*_COMPLETE.json 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
