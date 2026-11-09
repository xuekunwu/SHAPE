#!/bin/bash
# Set up the SSL early-stop A/B: a new run (results/shape_es) that REUSES the shape arm's cohort
# segmentation + cohort crop (cells.json / segmentation.json carry ABSOLUTE paths into results/shape,
# so the embed stage reads shape's exact crops) and differs ONLY in the SSL embed stage, run with
# $FULL_CPJUMP1_EARLY_STOP=1. Isolates "early-stopping SSL vs the original SSL" on the SAME crops;
# the baseline of the comparison is the existing results/shape (mAP 0.103).
#
#   bash benchmarks/full_CPJUMP1/runners/setup_ssl_es_ab.sh
# then follow the printed embed -> batch/cluster -> finalize commands (embed on the GPU node).
set -euo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
SRC=benchmarks/full_CPJUMP1/results/shape
DST=benchmarks/full_CPJUMP1/results/shape_es
[ -f "$SRC/crop_COMPLETE.json" ] || { echo "no completed shape crop at $SRC -- run the shape arm first"; exit 1; }

mkdir -p "$DST"
# segment + crop reuse: the inputs the embed stage reads (cells.json's crop_path and segmentation.json's
# mask are abs paths into results/shape) + the stage markers finalize_run requires.
for f in CONFIG.json segmentation.json segment_COMPLETE.json segment_COHORT.json \
         cells.json excluded_cells.json crop_COMPLETE.json crop_USAGE.json; do
  [ -e "$SRC/$f" ] && cp -v "$SRC/$f" "$DST/$f"
done
mkdir -p "$DST/crop" && [ -e "$SRC/crop/COHORT_RECORD.json" ] && cp "$SRC/crop/COHORT_RECORD.json" "$DST/crop/"

cat <<EOF

[setup_ssl_es_ab] results/shape_es prepared (reuses shape's cohort segment + crop).

Next (embed on the GPU node; batch/cluster + finalize anywhere):
  # 1) SSL embed WITH early stopping on the reused cohort crops
  FULL_CPJUMP1_EARLY_STOP=1 bash benchmarks/full_CPJUMP1/runners/run_staged.sh shape \\
      data/CPJUMP1/prepared $DST data/CPJUMP1/assets embed
  # 2) batch + cluster
  bash benchmarks/full_CPJUMP1/runners/run_staged.sh shape \\
      data/CPJUMP1/prepared $DST data/CPJUMP1/assets batch cluster
  # 3) score vs the control baseline
  python -m benchmarks.full_CPJUMP1.harness.finalize_run --prep data/CPJUMP1/prepared \\
      --run $DST --control benchmarks/full_CPJUMP1/results/control_v2

Then compare: $DST/RESULT.md  vs  results/shape/RESULT.md (mAP 0.103, the early-stop-OFF baseline).
EOF
