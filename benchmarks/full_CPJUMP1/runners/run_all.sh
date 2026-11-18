#!/bin/bash
# The one entry point for full_CPJUMP1: pick a dataset (cpjump1's own CRISPR prep, or ModalityProbe3),
# a phase, and which arms to run. Replaces seed_run.sh / probe3_segcrop.sh / probe3_embed.sh -- same
# underlying per-stage engines (run_staged.sh / run_staged_modality_probe.sh), one script.
#
#   bash benchmarks/full_CPJUMP1/runners/run_all.sh --dataset cpjump1|probe3 PHASE [SLUG ...]
#     PHASE    segcrop | embed | all
#                segcrop  CPU: segment + crop for the given arms, $JOBS concurrent (max parallelism;
#                         remote SAM via $SHAPE_SAM_SERVER_URL, $CORES crop workers)
#                embed    GPU: embed + batch + cluster + finalize (+ figures for cpjump1), arms ONE
#                         AT A TIME (SSL training needs the local GPU -- do not parallelise)
#                all      segcrop then embed
#     SLUG...  subset of arm dirs (default: every arm in the dataset's matrix below)
#
#   bash benchmarks/full_CPJUMP1/runners/run_all.sh --dataset cpjump1 segcrop
#   SHAPE_SAM_SERVER_URL=http://gpu-node.example.net:8137,... \
#     bash benchmarks/full_CPJUMP1/runners/run_all.sh --dataset probe3 all shape_luna shape_sonnet55
#
# Arm routing (slug|arm|model|route): shape/biomni claude-* -> Anthropic-direct (arms/anthropic_direct.env,
# LLM_BASE_URL unset, ANTHROPIC_API_KEY); cc -> Claude Code CLI on .claude_api.env; GLM/DeepSeek ->
# Together.ai (arms/together_glm.env, arms/together_deepseek.env); everything else -> the OpenAI-compatible
# gateway in .env.
set -uo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
HERE="benchmarks/full_CPJUMP1/runners"
PY="${PYTHON:-python}"
ASSETS="data/CPJUMP1/assets"
CLAUDE_ENV="$PWD/$HERE/.claude_api.env"
ANTHROPIC_ENV="$PWD/$HERE/arms/anthropic_direct.env"
TOGETHER_GLM_ENV="$PWD/$HERE/arms/together_glm.env"
TOGETHER_DEEPSEEK_ENV="$PWD/$HERE/arms/together_deepseek.env"
[ -n "${CORES:-}" ] && export CORES
[ -n "${SHAPE_SAM_SERVER_URL:-}" ] && export SHAPE_SAM_SERVER_URL

DATASET=""
if [ "${1:-}" = "--dataset" ]; then DATASET="$2"; shift 2; fi
PHASE="${1:?usage: run_all.sh --dataset cpjump1|probe3 PHASE [SLUG...]}"; shift || true

case "$DATASET" in
  cpjump1)
    STAGED="$HERE/run_staged.sh"
    PREP="data/CPJUMP1/prepared"
    OUTDIR="benchmarks/full_CPJUMP1/results/cpjump1"
    FINALIZE() { "$PY" -m benchmarks.full_CPJUMP1.harness.finalize_run "$@"; }
    FIGURES() { :; }   # no figures module in this release
    # slug | arm | model | route(-|anthropic|claude|together-glm|together-deepseek)
    MATRIX=(
      "control|control|-|-"
      "shape_luna|shape|gpt-5.6-luna|-"
      "shape_gpt6sol|shape|gpt-6-sol|-"
      "shape_opus5|shape|claude-opus-5|anthropic"
      "shape_sonnet55|shape|claude-sonnet-5-5|anthropic"
      "shape_glm53flash|shape|zai-org/GLM-5.3-Flash|together-glm"
      "shape_deepseek|shape|deepseek-ai/DeepSeek-V4.1-Flash|together-deepseek"
      "codex_luna|codex|gpt-5.6-luna|-"
      "codex_gpt6sol|codex|gpt-6-sol|-"
      "cc_opus5|claude|claude-opus-5|claude"
      "cc_sonnet55|claude|claude-sonnet-5-5|claude"
      "biomni_luna|biomni|gpt-5.6-luna|-"
      "biomni_gpt6sol|biomni|gpt-6-sol|-"
      "biomni_opus5|biomni|claude-opus-5|anthropic"
      "biomni_sonnet55|biomni|claude-sonnet-5-5|anthropic"
      "biomni_glm53flash|biomni|zai-org/GLM-5.3-Flash|together-glm"
    )
    DEFAULT_WANT=(shape_gpt6sol shape_opus5 codex_gpt6sol cc_opus5 biomni_gpt6sol biomni_opus5)
    ;;
  probe3)
    STAGED="$HERE/run_staged_modality_probe.sh"
    PREP="data/CPJUMP1_modality_probe3/prepared"
    OUTDIR="benchmarks/full_CPJUMP1/results/modality_probe3"
    FINALIZE() { "$PY" -m benchmarks.full_CPJUMP1.harness.finalize_run_modality_probe "$@"; }
    FIGURES() { :; }   # no figures.py for the probe family
    MATRIX=(
      "control|control|-|-"
      "shape_luna|shape|gpt-5.6-luna|-"
      "shape_sonnet55|shape|claude-sonnet-5-5|anthropic"
      "shape_deepseek|shape|deepseek-ai/DeepSeek-V4.1-Flash|together-deepseek"
      "biomni_luna|biomni|gpt-5.6-luna|-"
      "biomni_sonnet55|biomni|claude-sonnet-5-5|anthropic"
      "cc_sonnet55|claude|claude-sonnet-5-5|claude"
      "codex_luna|codex|gpt-5.6-luna|-"
    )
    DEFAULT_WANT=(control shape_luna shape_sonnet55 biomni_luna biomni_sonnet55 cc_sonnet55 codex_luna)
    ;;
  *) echo "usage: run_all.sh --dataset cpjump1|probe3 PHASE [SLUG...]"; exit 2 ;;
esac
[ -d "$PREP" ] || { echo "prepared dir missing: $PREP (run the dataset's prepare_*.py first)"; exit 2; }

WANT=("$@"); [ ${#WANT[@]} -eq 0 ] && WANT=("${DEFAULT_WANT[@]}")

row_for() { for r in "${MATRIX[@]}"; do [ "${r%%|*}" = "$1" ] && { echo "$r"; return; }; done; }

# Run run_staged(_modality_probe).sh for one arm with the route's env wired. Args: slug arm model route out stages...
launch() {
  local slug="$1" arm="$2" model="$3" route="$4" out="$5"; shift 5
  local argenv="" apienv=""
  case "$route" in
    anthropic)          argenv="$ANTHROPIC_ENV" ;;
    claude)             argenv="$CLAUDE_ENV"; apienv="$CLAUDE_ENV" ;;
    together-glm)       argenv="$TOGETHER_GLM_ENV" ;;
    together-deepseek)  argenv="$TOGETHER_DEEPSEEK_ENV" ;;
    -)                  argenv="" ;;
  esac
  echo "=== $slug ($arm / $model / route=$route) -> $out  ($(date -u +%H:%M:%SZ)) ==="
  (
    [ "$model" != "-" ] && export MODEL="$model"
    export ARM_ENV="$argenv" SHAPE_API_ENV_FILE="$apienv"
    # SHAPE runs cohort + SSL early stopping; opt-in, shape arm only.
    [ "$arm" = shape ] && export FULL_CPJUMP1_EARLY_STOP=1
    if [ "$DATASET" = cpjump1 ]; then
      bash "$STAGED" "$arm" "$PREP" "$out" "$ASSETS" "$@"
    else
      bash "$STAGED" ModalityProbe3 "$arm" "$PREP" "$out" "$ASSETS" "$@"
    fi
  )
}

do_segcrop() {
  local pids=() jobs="${JOBS:-4}"
  for slug in "${WANT[@]}"; do
    local row; row="$(row_for "$slug")"; [ -z "$row" ] && { echo "unknown slug $slug"; continue; }
    IFS='|' read -r s arm model route <<< "$row"
    local out="$OUTDIR/$slug"
    mkdir -p "$OUTDIR"   # not $out: init_run creates the run dir and refuses one that exists
    [ -f "$out/crop_COMPLETE.json" ] && { echo "=== $slug crop already done -- skip ==="; continue; }
    launch "$slug" "$arm" "$model" "$route" "$out" segment crop > "$out.segcrop.log" 2>&1 &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do sleep 5; done
  done
  wait
  echo "###### $DATASET segcrop done ######"
}

do_embed() {
  local ctrl="$OUTDIR/control"
  local have_ctrl=""; [ -f "$ctrl/RESULT.json" ] && have_ctrl=1 || echo "WARN: $ctrl not scored; deltas will be absent"
  for slug in "${WANT[@]}"; do
    local row; row="$(row_for "$slug")"; [ -z "$row" ] && { echo "unknown slug $slug"; continue; }
    IFS='|' read -r s arm model route <<< "$row"
    local out="$OUTDIR/$slug"
    if [ ! -f "$out/crop_COMPLETE.json" ]; then
      echo "### $slug: crop not done yet -- run the segcrop phase first -- skip ###"; continue
    fi
    [ -f "$out/RESULT.json" ] && { echo "=== $slug already scored -- skip ==="; continue; }
    extra=(); [ "$slug" != control ] && [ -n "$have_ctrl" ] && extra=(--control "$ctrl")
    if [ -n "${SMOKE:-}" ]; then      # a smoke is a technical check of every stage; finalize scores full deliveries only
      launch "$slug" "$arm" "$model" "$route" "$out" embed batch cluster \
        && echo "=== $slug smoke: every stage complete (a smoke is not scored) ===" \
        || echo "### $slug FAILED (exit $?) -- re-run to resume ###"
      continue
    fi
    launch "$slug" "$arm" "$model" "$route" "$out" embed batch cluster \
      && FINALIZE --prep "$PREP" --run "$out" "${extra[@]}" \
      && FIGURES --run "$out" --level both \
      || { echo "### $slug FAILED (exit $?) -- re-run to resume ###"; continue; }
    echo "=== $slug DONE -> $out/RESULT.json ==="
  done
  echo "###### $DATASET embed+score done ######"
}

case "$PHASE" in
  segcrop) do_segcrop ;;
  embed)   do_embed ;;
  all)     do_segcrop && do_embed ;;
  *) echo "PHASE must be segcrop|embed|all"; exit 2 ;;
esac
