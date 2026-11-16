#!/usr/bin/env bash
# Run every NeurIPS22-CellSeg arm (the 7 of cellseg_arms.ARMS plus the cohort set-arm,
# shape_cohort) into one run dir, on this machine. The 3 domain-tool baselines
# (cellpose3/stardist/microsam) need their own envs -- build them first with
# setup_seg_base_envs.sh build. The SAM-backed arms (cellpose_sam, react, shape,
# shape_cohort) need the shared Cellpose-SAM service up.
#
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/setup_seg_base_envs.sh build
#   export SHAPE_SAM_SERVER_URL=http://gpu-node.example.net:8137,...   # the shared Cellpose-SAM service
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh lists [n-per-band]
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh cellpose_sam|cellpose3|microsam|stardist|gpt|react|shape|shape_cohort
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh baselines   # cellpose3 + stardist + microsam
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh all         # every arm above, in sequence
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh compare
#
#   RUN_DIR=... to use a run dir other than the default (results/final).
set -eo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
CLI=benchmarks/segmentation_NeurIPS22-CellSeg/run_cellseg_benchmark.py
RUN="${RUN_DIR:-benchmarks/segmentation_NeurIPS22-CellSeg/results/final}"
DATA=data/NeurIPS22-CellSeg
MODEL="${MODEL:-gpt-5.6-luna}"
ENVROOT="${SEG_ENVROOT:-$HOME/shape/cache/envs}"
CP3="$ENVROOT/seg_cellpose3"; SP="$ENVROOT/seg_stardist"; MS="$ENVROOT/seg_microsam"

need_sam() { [ -n "${SHAPE_SAM_SERVER_URL:-}" ] || { echo "set SHAPE_SAM_SERVER_URL (the shared Cellpose-SAM service)"; exit 1; }; }

run_lists() {
  n="${1:-616}"   # large enough to draw every image of every band (hard 213 / medium 171 / easy 616 = 1000)
  .venv/bin/python "$CLI" lists --run-dir "$RUN" --data-root "$DATA" --n-per-band "$n" --seed 0
  for b in hard medium easy; do echo "  $b: $(wc -l < "$RUN"/lists/$b.txt) images"; done
}

run_domain() { .venv/bin/python "$CLI" run --run-dir "$RUN" --arm cellpose_sam --data-root "$DATA"; }   # needs SAM, same .venv

run_baseline() {   # $1: arm name, $2: env dir
  local py="$2/bin/python"
  [ -x "$py" ] || { echo "### $1: env $py missing -- run setup_seg_base_envs.sh build first"; return 1; }
  echo "===== $1 ($py) $(date -u +%H:%M:%SZ) ====="
  "$py" "$CLI" run --run-dir "$RUN" --arm "$1" --data-root "$DATA"
}

run_gpt() { .venv/bin/python "$CLI" run --run-dir "$RUN" --arm gpt --data-root "$DATA" --model "$MODEL"; }

run_react() { need_sam; .venv/bin/python "$CLI" run --run-dir "$RUN" --arm react --data-root "$DATA" --model "$MODEL" --workers 8; }

run_shape() { need_sam; .venv/bin/python "$CLI" run --run-dir "$RUN" --arm shape --data-root "$DATA" --model "$MODEL"; }

run_shape_cohort() { need_sam; .venv/bin/python "$CLI" run-set --run-dir "$RUN" --arm shape_cohort --data-root "$DATA" --model "$MODEL"; }

run_compare() { .venv/bin/python "$CLI" compare --run-dir "$RUN"; }

case "${1:?usage: run_all_arms.sh lists|cellpose_sam|cellpose3|microsam|stardist|baselines|gpt|react|shape|shape_cohort|all|compare}" in
  lists)         run_lists "${2:-}" ;;
  cellpose_sam)  need_sam; run_domain ;;
  cellpose3)     run_baseline cellpose3 "$CP3" ;;
  microsam)      run_baseline microsam "$MS" ;;
  stardist)      run_baseline stardist "$SP" ;;
  baselines)
    for pair in "cellpose3:$CP3" "stardist:$SP" "microsam:$MS"; do
      arm="${pair%%:*}"; run_baseline "$arm" "${pair#*:}" || echo "### $arm FAILED"
    done ;;
  gpt)           run_gpt ;;
  react)         run_react ;;
  shape)         run_shape ;;
  shape_cohort)  run_shape_cohort ;;
  compare)       run_compare ;;
  all)
    run_lists
    need_sam; run_domain
    for pair in "cellpose3:$CP3" "stardist:$SP" "microsam:$MS"; do
      arm="${pair%%:*}"; run_baseline "$arm" "${pair#*:}" || echo "### $arm FAILED"
    done
    run_gpt
    run_react
    run_shape
    run_shape_cohort
    run_compare ;;
  *) echo "usage: run_all_arms.sh lists|cellpose_sam|cellpose3|microsam|stardist|baselines|gpt|react|shape|shape_cohort|all|compare"; exit 2 ;;
esac
