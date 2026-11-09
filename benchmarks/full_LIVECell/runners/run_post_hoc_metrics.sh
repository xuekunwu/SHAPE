#!/usr/bin/env bash
# Every post-hoc metric call of full_LIVECell, in order, for one experiment directory:
#   1. runners/extra_readouts.py   per arm (control, shape, codex): classification, AUROC / AUPRC, morphology,
#                                  stage leakage, batch integration (needs the prepared tree: --prep)
#   2. runners/process_readouts.py L4 protocol friction and L5 efficiency (JSON, CSV, markdown)
# An arm whose run dir is absent or incomplete (no LIVECELL_COMPLETE marker, no RESULT.json) is skipped and the skip is
# said. Each step that ran appends its exact command line, the commit it ran at and the UTC time to
# <OUT>/analysis/COMMANDS.txt, so a results folder carries the calls that produced it.
set -eo pipefail

usage() {
    cat <<'TEXT'
usage: bash benchmarks/full_LIVECell/runners/run_post_hoc_metrics.sh --run EXPERIMENT_DIR [options]
  --run DIR     the experiment directory (control/ shape/ codex/ run dirs)
  --out DIR     where analysis/ is written (default: the experiment dir)
  --prep DIR    the prepared LIVECell tree with the private labels (default: $FULL_LIVECELL_PREP, else
                ~/shape/data/LIVECell/prepared); needed by extra_readouts.py
  --arms LIST   the arms to read, space separated (default: control shape codex)
Environment: FULL_LIVECELL_REPO (the repo root; default $PWD), PYTHON (the interpreter; default
  ~/shape/cache/envs/shape_bench/bin/python on the VM, else python3 / python on PATH), WORKERS (processes of the
  GT matching in extra_readouts.py; default all cores).
Example:
  bash benchmarks/full_LIVECell/runners/run_post_hoc_metrics.sh --run ~/shape/runs/full_LIVECell/full_<commit8>_<utc>_x
TEXT
}

die() { printf 'POST_HOC_FAILED: %s\n' "$*" >&2; exit 2; }

EXP=""; OUT=""; PREP_DIR=""; ARMS_LIST="control shape codex"
while (( $# )); do
    case "$1" in
        --run) EXP=${2:?--run needs a directory}; shift 2 ;;
        --out) OUT=${2:?--out needs a directory}; shift 2 ;;
        --prep) PREP_DIR=${2:?--prep needs a directory}; shift 2 ;;
        --arms) ARMS_LIST=${2:?--arms needs a list}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done
[[ -n "$EXP" && -d "$EXP" ]] || { usage >&2; die '--run must be an existing experiment directory'; }
EXP=$(cd "$EXP" && pwd)

REPO=${FULL_LIVECELL_REPO:-$PWD}
[[ -f "$REPO/benchmarks/full_LIVECell/runners/extra_readouts.py" ]] \
    || REPO=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
[[ -f "$REPO/benchmarks/full_LIVECell/runners/extra_readouts.py" ]] || die "no repo at $REPO (set FULL_LIVECELL_REPO)"
BENCH=$REPO/benchmarks/full_LIVECell

if [[ -z "${PYTHON:-}" ]]; then
    VM_ENV=${FULL_LIVECELL_ENV:-$HOME/shape/cache/envs/shape_bench}
    if [[ -x "$VM_ENV/bin/python" ]]; then PYTHON=$VM_ENV/bin/python
    else PYTHON=$(command -v python3 || command -v python || true); fi
fi
[[ -n "$PYTHON" ]] || die 'no interpreter: set PYTHON'

[[ -n "$PREP_DIR" ]] || PREP_DIR=${FULL_LIVECELL_PREP:-$HOME/shape/data/LIVECell/prepared}
[[ -n "$OUT" ]] || OUT=$EXP
mkdir -p "$OUT/analysis"
OUT=$(cd "$OUT" && pwd)
COMMANDS=$OUT/analysis/COMMANDS.txt
COMMIT=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)

# record NAME CMD...: append the exact command line, the commit and the UTC time of a step that succeeded.
record() {
    local name=$1; shift
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
    printf '# %s commit=%s step=%s skipped: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$COMMIT" "$1" "$2" >> "$COMMANDS"
}

for arm in $ARMS_LIST; do
    if [[ ! -d "$EXP/$arm" ]]; then
        skipped "extra_readouts_$arm" "run dir $EXP/$arm is absent"
    elif [[ ! -f "$EXP/$arm/LIVECELL_COMPLETE" || ! -f "$EXP/$arm/RESULT.json" ]]; then
        skipped "extra_readouts_$arm" "run dir $EXP/$arm is incomplete (needs LIVECELL_COMPLETE and RESULT.json)"
    else
        [[ -d "$PREP_DIR" ]] || die "--prep $PREP_DIR: not a directory (extra_readouts.py needs the prepared tree)"
        ARGS=("$PYTHON" "$BENCH/runners/extra_readouts.py" --prep "$PREP_DIR" --run "$EXP/$arm"
              --out "$OUT/analysis/extra_readouts_$arm.json" --md)
        [[ -z "${WORKERS:-}" ]] || ARGS+=(--workers "$WORKERS")
        step "extra_readouts_$arm" "${ARGS[@]}"
    fi
done

step process_readouts "$PYTHON" "$BENCH/runners/process_readouts.py" --run "$EXP" \
    --out "$OUT/analysis/process_readouts.json" --csv "$OUT/analysis/process_readouts.csv" --md
printf 'done: %s\n' "$COMMANDS"
