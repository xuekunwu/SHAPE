#!/usr/bin/env bash
# Every post-hoc metric call of full_CPJUMP1, in order, for one experiment directory:
#   1. export_run_records.py   the curated text-only results folder (then --check of it)
#   2. runners/pairwise_deltas.py   paired differences between runs (only with --pair)
#   3. runners/process_readouts.py  L4 protocol friction and L5 efficiency (JSON, CSV, markdown)
# Each step that ran appends its exact command line, the commit it ran at and the UTC time to
# <OUT>/analysis/COMMANDS.txt, so a results folder carries the calls that produced it.
set -eo pipefail

usage() {
    cat <<'TEXT'
usage: bash benchmarks/full_CPJUMP1/runners/run_post_hoc_metrics.sh --run EXPERIMENT_DIR [options]
  --run DIR         the experiment directory (arm run dirs, rerun dirs, comparison dirs, driver logs)
  --out DIR         the results folder (default: <repo>/benchmarks/full_CPJUMP1/results/<name of the experiment dir>);
                    the post-hoc outputs and COMMANDS.txt go to <out>/analysis/
  --pair A:B        run A against run B in pairwise_deltas.py (repeatable; env PAIRS, space separated, is read too)
  --extra-log FILE  a driver log outside the experiment dir, copied by the exporter (repeatable)
  --skip-export     the results folder already exists (the exporter refuses a non-empty --out): skip step 1
Environment: FULL_CPJUMP1_REPO (the repo root; default $PWD), PYTHON (the interpreter; default
  ~/shape/cache/envs/shape_bench/bin/python on the VM, else python3 / python on PATH).
Example:
  bash benchmarks/full_CPJUMP1/runners/run_post_hoc_metrics.sh --run ~/shape/runs/full_CPJUMP1/full_18f7285f_x \
      --extra-log ~/shape/run_after_codex.log --pair shape_rerun_e2e_21f63d2b:shape
TEXT
}

die() { printf 'POST_HOC_FAILED: %s\n' "$*" >&2; exit 2; }

EXP=""; OUT=""; SKIP_EXPORT=0
PAIRS_LIST=(${PAIRS:-})
EXTRA_LOGS=()
while (( $# )); do
    case "$1" in
        --run) EXP=${2:?--run needs a directory}; shift 2 ;;
        --out) OUT=${2:?--out needs a directory}; shift 2 ;;
        --pair) PAIRS_LIST+=("${2:?--pair needs A:B}"); shift 2 ;;
        --extra-log) EXTRA_LOGS+=("${2:?--extra-log needs a file}"); shift 2 ;;
        --skip-export) SKIP_EXPORT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done
[[ -n "$EXP" && -d "$EXP" ]] || { usage >&2; die '--run must be an existing experiment directory'; }
EXP=$(cd "$EXP" && pwd)

REPO=${FULL_CPJUMP1_REPO:-$PWD}
[[ -f "$REPO/benchmarks/full_CPJUMP1/export_run_records.py" ]] \
    || REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
[[ -f "$REPO/benchmarks/full_CPJUMP1/export_run_records.py" ]] || die "no repo at $REPO (set FULL_CPJUMP1_REPO)"
BENCH=$REPO/benchmarks/full_CPJUMP1

if [[ -z "${PYTHON:-}" ]]; then
    VM_ENV=${FULL_CPJUMP1_ENV:-$HOME/shape/cache/envs/shape_bench}
    if [[ -x "$VM_ENV/bin/python" ]]; then PYTHON=$VM_ENV/bin/python
    else PYTHON=$(command -v python3 || command -v python || true); fi
fi
[[ -n "$PYTHON" ]] || die 'no interpreter: set PYTHON'

[[ -n "$OUT" ]] || OUT=$BENCH/results/$(basename "$EXP")
mkdir -p "$(dirname "$OUT")"
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
COMMANDS=$OUT/analysis/COMMANDS.txt
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)
for log in "${EXTRA_LOGS[@]}"; do [[ -f "$log" ]] || die "--extra-log $log: not a file"; done

# record NAME CMD...: append the exact command line, the commit and the UTC time of a step that succeeded.
record() {
    local name=$1; shift
    mkdir -p "$OUT/analysis"
    {
        printf '# %s commit=%s step=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$COMMIT" "$name"
        printf 'PYTHONPATH=%q' "$REPO"
        printf ' %q' "$@"
        printf '\n'
    } >> "$COMMANDS"
}
# step NAME CMD...: run the command with the repo on PYTHONPATH, then record it.
step() {
    local name=$1; shift
    printf '== %s\n' "$name"
    env PYTHONPATH="$REPO" "$@"
    record "$name" "$@"
}
skipped() {
    printf '== %s: skipped (%s)\n' "$1" "$2"
    mkdir -p "$OUT/analysis"
    printf '# %s commit=%s step=%s skipped: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$COMMIT" "$1" "$2" >> "$COMMANDS"
}

if (( SKIP_EXPORT )); then
    skipped export_run_records '--skip-export'
else
    EXPORT_ARGS=("$PYTHON" "$BENCH/export_run_records.py" --run "$EXP" --out "$OUT")
    for log in "${EXTRA_LOGS[@]}"; do EXPORT_ARGS+=(--extra-log "$(cd "$(dirname "$log")" && pwd)/$(basename "$log")"); done
    step export_run_records "${EXPORT_ARGS[@]}"
    step export_run_records_check "$PYTHON" "$BENCH/export_run_records.py" --check "$OUT"
fi

if (( ${#PAIRS_LIST[@]} )); then
    PAIR_ARGS=()
    for pair in "${PAIRS_LIST[@]}"; do PAIR_ARGS+=(--pair "$pair"); done
    step pairwise_deltas "$PYTHON" "$BENCH/runners/pairwise_deltas.py" --run "$EXP" "${PAIR_ARGS[@]}" \
        --out "$OUT/analysis/pairwise_deltas.json"
else
    skipped pairwise_deltas 'no --pair given'
fi

step process_readouts "$PYTHON" "$BENCH/runners/process_readouts.py" --run "$EXP" \
    --out "$OUT/analysis/process_readouts.json" --csv "$OUT/analysis/process_readouts.csv" --md
printf 'done: %s\n' "$COMMANDS"
