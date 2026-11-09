# SHAPE

**SHAPE** (*Self-supervised morpHology Agent for cellular PhenotypE*) is an
agentic framework for biological image analysis.

This public release carries the framework, the benchmark harnesses and the
case-study pipelines. The per-run records (the `results/` directories referenced
below) and run logs are kept out of it.

## Install

Requires **Python 3.10+**. Benchmark assets (`benchmarks/**/*.tif`, `*.npy`) are
stored with [Git LFS](https://git-lfs.com); without it those files clone as
text pointers rather than data.

```bash
git lfs install
git clone <this repository>
cd SHAPE
python -m pip install -e .
```

The segmentation tool needs Cellpose-SAM (`cellpose>=4.0`) and a
CUDA-compatible PyTorch build for anything beyond small images.

## Quick start

Configure the LLM gateway:

```bash
export LLM_MODEL="<your-model>"
export LLM_API_KEY="<your-api-key>"
export LLM_BASE_URL="<your-api-base-url>"
```

Then:

```python
import os
from shape import solve

result = solve(
    question="What cell states are present in this image?",
    image_path="path/to/image.tif",
    llm_engine_name=os.environ["LLM_MODEL"],
)

print(result["direct_output"])
```

## Benchmarks

Evaluation harnesses live in [`benchmarks/`](benchmarks/README.md), one
directory per tool and dataset. All of them run the **concurrent** harness
(many CPU worker processes) and delegate GPU work to **shared, separately
deployed model services**, which is the recommended setup for scale.

### Recommended setup: shared GPU model services

Run the GPU models as long-lived services and point every benchmark's workers at
them over HTTP. Scale throughput with independent **server processes** (a single
process plateaus at ~2× serial on the Python GIL), and put several on consecutive
ports; clients round-robin a comma-separated URL list. Launch from a real shell
(a sandbox may SIGKILL background CUDA).

```bash
# Cellpose-SAM (segmentation) — 4 processes × 2 replicas on 8137–8140
for p in 8137 8138 8139 8140; do
  setsid nohup .venv/bin/python -u -m shape.tools.cell_segmenter.sam_server \
    --host 0.0.0.0 --allow-remote --port $p --replicas 2 > sam_$p.log 2>&1 &
done

# DINOv3 (embedding) — 4 processes × 2 replicas on 8237–8240
export HUGGINGFACE_TOKEN=$HF_TOKEN
for p in 8237 8238 8239 8240; do
  setsid nohup .venv/bin/python -u -m shape.tools.cell_state_analyzer.dinov3_server \
    --host 0.0.0.0 --allow-remote --port $p --replicas 2 > dinov3_$p.log 2>&1 &
done
```

`--host 0.0.0.0 --allow-remote` exposes the service on the private network so
workers on other nodes can reach it (e.g. `http://gpu-node.example.net:8137`). **Security:**
the transport is pickle (a remote-code-execution surface) — only enable
`--allow-remote` on a trusted, firewalled network, or keep the default localhost
bind and SSH-tunnel. See
[`shape/tools/cell_segmenter/SAM_SERVICE.md`](shape/tools/cell_segmenter/SAM_SERVICE.md).

### Running each benchmark

Set `SAM=http://HOST:8137,http://HOST:8138,http://HOST:8139,http://HOST:8140`
and `DINO=http://HOST:8237,http://HOST:8238,http://HOST:8239,http://HOST:8240`
(or `127.0.0.1` for local services), and `DATA` to the dataset root.

**Segmentation** — agentic VLRR Cellpose-SAM
([`benchmarks/segmentation_NeurIPS22-CellSeg`](benchmarks/segmentation_NeurIPS22-CellSeg)):

```bash
python -m benchmarks.segmentation_NeurIPS22-CellSeg.runners.run_concurrent_benchmark \
  --data-root "$DATA/NeurIPS22-CellSeg" --image-subdir images \
  --sam-server-url "$SAM" --workers 32 --max-steps 8
# self-contained alternative (one GPU-resident model for the run): drop
# --sam-server-url and add --autostart-sam --sam-replicas 4
```

**Cropper** — the formal five-arm benchmark (MaskCrop, CellProfiler as a control, gpt-5.6-luna alone,
ReAct, SHAPE) on the ground-truth masks
([`benchmarks/cropper_NeurIPS22-CellSeg`](benchmarks/cropper_NeurIPS22-CellSeg)):

```bash
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py prepare --run-dir R \
  --data-dir "$DATA/NeurIPS22-CellSeg" --n-images 30
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py run --run-dir R --arm shape --cores 8
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py compare --run-dir R
```

The legacy many-process SHAPE runner on predicted segmentation masks (used by the cell-state pipeline):

```bash
python -m benchmarks.cropper_NeurIPS22-CellSeg.runners.run_concurrent_benchmark \
  --data-root "$DATA/NeurIPS22-CellSeg" --image-subdir images \
  --masks-from-seg-run <segmentation-run-dir> --workers 32 --crop-max-iters 8
```

**Cell-state (HCMA)** — one-shot 4-stage pipeline (segment → crop → DINOv3
embed → Harmony), every stage through its VLRR loop
([`benchmarks/cell_state_HCMA`](benchmarks/cell_state_HCMA)):

```bash
SAM_URLS="$SAM" DINO_URL="$DINO" \
  bash benchmarks/cell_state_HCMA/run_pipeline.sh 48 8      # WORKERS SAM_PROCS [LIMIT]
# stages are also runnable standalone: run_shape_hcma.py, stage3_embed.py,
# stage4_harmony.py; scoring: evaluate.py / score_label.py / score_scib.py
```

**Batch correction (JUMP)** — Harmony tuned by the budgeted VRR (`BatchCorrectionVerifier`
through `StepVerifier`; the reasoner chooses every tool call and the final one), scored with the
Arevalo composite in `quant.py` (CPU-only)
([`benchmarks/batch_correction_JUMP`](benchmarks/batch_correction_JUMP/README.md)):
the driver is `run_batch_benchmark.py` (`prepare` / `control` / `suite` / `report`; the implementation in
`runners/`) and the correction itself is `Batch_Corrector_Tool.execute`.

**Baselines** — external coding agents (Claude Code / Codex) driven through the
same segmentation/cropper tasks with a prolonged feedback loop, for a head-to-head
against SHAPE ([`benchmarks/baselines`](benchmarks/baselines)):

```bash
python -m benchmarks.baselines.run_baseline \
  --agent claude-code --benchmark segmentation \
  --data-root "$DATA/NeurIPS22-CellSeg" --iters 10 --min-iters 3 --sam-url "$SAM"
# --agent codex for the Codex CLI; --benchmark cropper for the cropper task.
# Requires the external agent CLI on PATH; GPU still comes from the shared SAM service.
```

## Experimental Case Studies

[`case studies/`](case%20studies/README.md) contains end-to-end experimental
applications, distinct from public-dataset benchmarks. The
[cardiomyocyte studies](case%20studies/cardiomyocytes/README.md) define
brightfield-only prediction of single-cell cTnT positivity from paired staining
and well-level cTnT-positive proportions from flow cytometry. Each has `harness/`,
`runners/`, with authenticated metadata access and pinned data
contracts. Full image pipelines remain to be implemented. Cardiac organoid and
[tissue](case%20studies/tissues/README.md) applications have the same directory structure.

## Development

```bash
python -m pip install -e ".[test]"
pytest
```

Tests that need a GPU, the LLM gateway or a private dataset are skipped or fail
without them; run the suite on a machine that has what the part under test uses.

## License

SHAPE is released under the [CC BY 4.0 license](LICENSE). Redistributed
third-party source sits under `benchmarks/batch_correction_JUMP/runners/vendor/`
with its own licence and its provenance in that benchmark's README.
