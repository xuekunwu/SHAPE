#!/usr/bin/env bash
# Build the 3 domain-baseline seg envs ON THE GPU NODE, then run those arms on the SAME run dir as the
# cellpose_sam/shape/react 3-arm run (benchmarks/.../results/seg3arm_luna), so all baselines score the
# same 30 images. /home is node-local here, so these envs must be created on the node that runs them.
#
#   ssh <gpu-node> ; cd /path/to/SHAPE
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/setup_seg_base_envs.sh        # build + cache weights
#   bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/setup_seg_base_envs.sh run    # run the 3 arms
#
# Uses uv (stardist/cellpose3, pure pip) and
# micromamba (micro-sam is conda-forge only). stardist uses CPU TensorFlow (no CUDA clash with torch).
set -eo pipefail
cd "$(cd "$(dirname "$0")/../../.." && pwd)"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"; export PYTHONNOUSERSITE=1
ENVROOT="${SEG_ENVROOT:-$HOME/shape/cache/envs}"; mkdir -p "$ENVROOT"
SP="$ENVROOT/seg_stardist"; CP3="$ENVROOT/seg_cellpose3"; MS="$ENVROOT/seg_microsam"   # where run_all_arms.sh looks
SCORE="numpy scipy scikit-image tifffile pillow"
CLI=benchmarks/segmentation_NeurIPS22-CellSeg/run_cellseg_benchmark.py
RUN=benchmarks/segmentation_NeurIPS22-CellSeg/results/seg3arm_luna
DATA=data/NeurIPS22-CellSeg

build() {
  # --- stardist (uv, CPU TensorFlow) ---
  if "$SP/bin/python" -c "import stardist" 2>/dev/null; then echo "seg_stardist present"; else
    rm -rf "$SP"; uv venv --python 3.10 "$SP"
    uv pip install --python "$SP/bin/python" \
      "tensorflow-cpu==2.15.1" "numpy<2" "stardist==0.9.1" "setuptools<81" $SCORE
    CUDA_VISIBLE_DEVICES= "$SP/bin/python" - <<'PY'
from stardist.models import StarDist2D
for n in ("2D_versatile_he","2D_versatile_fluo"): StarDist2D.from_pretrained(n)
import stardist,tensorflow as tf,numpy;print("stardist",stardist.__version__,"tf",tf.__version__,"np",numpy.__version__,"- weights cached")
PY
  fi
  # --- cellpose3 (uv) ---
  if "$CP3/bin/python" -c "import cellpose,sys; v=cellpose.version if hasattr(cellpose,'version') else ''; sys.exit(0 if str(v).startswith('3') else 1)" 2>/dev/null; then
    echo "seg_cellpose3 present"; else
    rm -rf "$CP3"; uv venv --python 3.10 "$CP3"
    uv pip install --python "$CP3/bin/python" "cellpose>=3,<4" "numpy<2" $SCORE
    "$CP3/bin/python" -c "import cellpose;print('cellpose',getattr(cellpose,'version','?'))"
  fi
  # --- micro-sam (micromamba, conda-forge) ---
  if "$MS/bin/python" -c "import micro_sam" 2>/dev/null; then echo "seg_microsam present"; else
    MAMBA="${MAMBA_BIN:-$HOME/bin/micromamba}"
    [ -x "$MAMBA" ] || { echo "NEED micromamba at $MAMBA (or set MAMBA_BIN). Install: 'curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba'"; exit 1; }
    rm -rf "$MS"
    CONDA_OVERRIDE_CUDA=12.1 "$MAMBA" create -y -p "$MS" -c conda-forge python=3.11 micro_sam "pytorch=*=*cuda*" $SCORE
    "$MS/bin/python" - <<'PY'
from micro_sam.util import get_sam_model
get_sam_model(model_type="vit_b_lm")
import micro_sam;print("micro_sam",micro_sam.__version__,"- vit_b_lm cached")
PY
  fi
  echo "SETUP_OK"
}

run() {
  for pair in "seg_cellpose3:cellpose3:$CP3" "seg_stardist:stardist:$SP" "seg_microsam:microsam:$MS"; do
    env="${pair#*:}"; arm="${env%%:*}"; py="${pair##*:}/bin/python"
    [ -x "$py" ] || { echo "### $arm: env $py missing -- run build first"; continue; }
    echo "===== $arm ($py) $(date -u +%H:%M:%SZ) ====="
    "$py" "$CLI" run --run-dir "$RUN" --arm "$arm" --data-root "$DATA" || echo "### $arm FAILED (exit $?)"
  done
  echo "===== seg baselines done; compare: python $CLI compare --run-dir $RUN ====="
}

case "${1:-build}" in build) build;; run) run;; both) build && run;; *) echo "usage: $0 [build|run|both]";; esac
