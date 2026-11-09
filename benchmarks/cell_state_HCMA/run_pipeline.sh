#!/bin/bash
# HCMA Config-A end-to-end pipeline: segment -> crop -> embed -> batch-correct, every
# stage through its VLRR loop. Run this on a CPU-RICH machine (the segmentation VLRR is
# CPU-bound: cellpose flow->mask reconstruction holds the GIL, so throughput scales with
# SAM server PROCESSES + worker processes, not GPU).
#
#   bash benchmarks/cell_state_HCMA/run_pipeline.sh [WORKERS] [SAM_PROCS] [LIMIT]
#     WORKERS   seg/crop worker processes            (default: nproc-4)
#     SAM_PROCS SAM server processes (throughput)    (default: WORKERS/4, capped 8)
#     LIMIT     cap #images for a smoke test          (default: all 1059)
#
# PREREQUISITES on this machine:
#   * GPU visible (torch.cuda.is_available() == True) for the SAM + DINOv3 services
#   * data/HCMA extracted (tarball from SHAPE-Morph/HCMA) + configA_images symlinks:
#       python -c "import pandas as pd,os; m=pd.read_parquet('benchmarks/cell_state_HCMA/results/hcma_configA_manifest.parquet'); d='data/HCMA/configA_images/images'; os.makedirs(d,exist_ok=True); [os.path.exists(os.path.join(d,os.path.basename(p))) or os.symlink(os.path.abspath(p),os.path.join(d,os.path.basename(p))) for p in m.image_path]"
#   * .env with LLM_API_KEY / LLM_BASE_URL (VLRR verifier LLM) + HF_TOKEN
#
# Services are LONG-LIVED CUDA processes -> the Claude sandbox SIGKILLs them; launch this
# whole script from a REAL shell (it starts + stops the services itself).
set -eo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"
PY=.venv/bin/python
RES=benchmarks/cell_state_HCMA/results

WORKERS="${1:-$(( $(nproc) - 4 ))}"; [ "$WORKERS" -lt 1 ] && WORKERS=1
SAM_PROCS="${2:-$(( WORKERS/4 > 8 ? 8 : (WORKERS/4 < 1 ? 1 : WORKERS/4) ))}"
LIMIT="${3:-}"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
echo "== HCMA pipeline: WORKERS=$WORKERS SAM_PROCS=$SAM_PROCS LIMIT=${LIMIT:-all} =="

# ---- services: SAM (multi-process for throughput) + DINOv3 ----
# EXTERNAL mode: if SAM_URLS + DINO_URL are already exported (services on a remote GPU
# node), use them as-is and do NOT launch/kill anything locally. Otherwise start local
# GPU services (this node must have a GPU) and clean them up on exit.
if [ -n "${SAM_URLS:-}" ] && [ -n "${DINO_URL:-}" ]; then
  echo "[services] using EXTERNAL services SAM=$SAM_URLS DINO=$DINO_URL (no local launch)"
else
  SAM_URLS=""
  for i in $(seq 0 $((SAM_PROCS-1))); do p=$((8137+i));
    setsid nohup $PY -u -m shape.tools.cell_segmenter.sam_server --port $p --replicas 2 > sam_$p.log 2>&1 &
    SAM_URLS="${SAM_URLS:+$SAM_URLS,}http://127.0.0.1:$p"; done
  setsid nohup $PY -u -m shape.tools.cell_state_analyzer.dinov3_server --port 8237 --replicas 2 > dinov3_8237.log 2>&1 &
  DINO_URL="http://127.0.0.1:8237"
  trap 'pkill -f "cell_segmenter.sam_server"; pkill -f "cell_state_analyzer.dinov3_server"' EXIT
fi
echo "[services] waiting for SAM ($SAM_URLS) + DINOv3 to load ..."
for u in ${SAM_URLS//,/ } $DINO_URL; do
  for _ in $(seq 1 60); do $PY - "$u" <<'PY' 2>/dev/null && break || sleep 3
import sys,urllib.request,pickle
try: pickle.loads(urllib.request.urlopen(sys.argv[1].rstrip("/")+"/health",timeout=3).read()); print("up",sys.argv[1])
except Exception: sys.exit(1)
PY
  done; done
export SHAPE_SAM_SERVER_URL="${SAM_URLS%%,*}"

# ---- STAGE 1: VLRR segmentation ----
echo "== STAGE 1: VLRR segmentation =="
rm -rf "$RES/stage1_seg"
$PY -m benchmarks.segmentation_NeurIPS22-CellSeg.runners.run_concurrent_benchmark \
  --data-root data/HCMA/configA_images --image-subdir images \
  --output-root "$RES/stage1_seg" --sam-server-url "$SAM_URLS" \
  --workers "$WORKERS" --max-steps 8 $LIMIT_ARG
SEG_RUN=$(ls -dt "$RES/stage1_seg"/*_concurrent | head -1)
echo "[stage1] -> $SEG_RUN"

# ---- STAGE 2: VLRR cropper (tight bbox) on the agentic masks ----
echo "== STAGE 2: VLRR cropper =="
rm -rf "$RES/stage2_crops"
$PY -m benchmarks.cropper_NeurIPS22-CellSeg.runners.run_concurrent_benchmark \
  --data-root data/HCMA/configA_images --image-subdir images \
  --output-root "$RES/stage2_crops" --masks-from-seg-run "$SEG_RUN" \
  --workers "$WORKERS" --crop-max-iters 8 $LIMIT_ARG
CROP_RUN=$(ls -dt "$RES/stage2_crops"/*_concurrent | head -1)
echo "[stage2] -> $CROP_RUN"

# ---- glue: crop manifest (join organ_category + vendor) ----
$PY -m benchmarks.cell_state_HCMA.build_crop_manifest \
  --crop-run "$CROP_RUN" --out "$RES/hcma_configA_crops.parquet"

# ---- STAGE 3: VLRR DINOv3 embedding + probe ----
echo "== STAGE 3: VLRR representation (embed + RepresentationVerifier) =="
$PY -m benchmarks.cell_state_HCMA.stage3_embed \
  --manifest "$RES/hcma_configA_crops.parquet" --service-url "$DINO_URL" --out-dir "$RES"

# ---- STAGE 4: VLRR Harmony cross-vendor batch correction ----
echo "== STAGE 4: VLRR batch correction (Harmony + BatchCorrectionVerifier) =="
$PY -m benchmarks.cell_state_HCMA.stage4_harmony \
  --h5ad "$RES/hcma_configA_stage3.h5ad" --out-dir "$RES"

echo "== DONE. Readouts in $RES/ (stage3_scores.json, stage4_batchcorr.json, *.png) =="
