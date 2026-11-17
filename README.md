# SHAPE

**SHAPE** (*Self-supervised morpHology Agent for cellular PhenotypE*) is an agentic
framework for biological image analysis. A reasoner drives a fixed set of imaging tools
— segment, crop, learn a representation, correct batch effects, cluster — and a
verification loop decides, from label-free evidence alone, whether a tool call is good
enough or has to be tuned and run again.

This public release carries the framework, the benchmark harnesses and the case-study
pipelines. Per-run records, logs and the cluster launch scripts of the internal research
repository are not part of it.

## How it works

- **Tools** (`shape/tools/`) wrap the models: Cellpose-SAM segmentation, organoid
  segmentation, single-cell cropping, DINOv3 representation learning, Harmony batch
  correction, plus clustering, visualisation and literature helpers. Each tool ships a
  `CARD.md` stating its levers, its verdict declaration and its calibration — the
  contract the agent reads, not prose for humans only.
- **Verification** (`shape/agent/verification/`) is the verify-reason-refine (VRR) loop:
  quantitative readouts and, where it helps, a vision judge produce a verdict per call;
  admission rules decide whether a proposed retune may run at all; a budget bounds the
  search. No ground truth and no held-out label enters it.
- **The agent** (`shape/agent/`, `shape/solver.py`) plans the call chain, executes it and
  delivers one answer with its evidence. `shape/engine/` is the LLM transport (OpenAI,
  Anthropic and LiteLLM backends); `shape/agent/arms/` lets an external coding agent
  (Codex CLI, Claude Code, Biomni) drive the same stage protocol as a comparison arm.

## Layout

| Path | What it holds |
|---|---|
| [`shape/`](shape/) | the framework: tools, agent, verification, LLM engines, solver |
| [`benchmarks/`](benchmarks/README.md) | public-dataset evaluations, one directory per tool and dataset |
| [`case studies/`](case%20studies/README.md) | end-to-end applications on experimental data |
| [`environments/`](environments/README.md) | the pinned runtimes a run records itself against |

## Install

Requires **Python 3.10+**.

```bash
git clone <this repository>
cd SHAPE
python -m pip install -e .
```

The segmentation tool needs Cellpose-SAM (`cellpose>=4.0`) and a CUDA-capable PyTorch
build for anything beyond small images. To reproduce a benchmark runtime exactly, install
a pinned lock from [`environments/`](environments/README.md) instead of resolving
dependencies yourself.

## Quick start

Configure the LLM endpoint:

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

`solve` takes the budget and the loop's switches as keyword arguments (`max_steps`,
`enable_verification`, `verify_refine`, `image_workers`, `segmentation_cohort`,
`cropping_cohort`); `shape.get_available_tools()` lists the tools it may call.

## Benchmarks

Evaluation harnesses live in [`benchmarks/`](benchmarks/README.md), one directory per tool
and dataset. The segmentation benchmark and the segment and crop stages of the end-to-end
benchmark call a **shared, separately deployed Cellpose-SAM service**; the embed stage and the
RxRx1 representation benchmark need a local GPU; the cropper and batch-correction benchmarks
run on CPU.

### Shared GPU model services

Run the GPU models as long-lived services and point every benchmark's workers at them over
HTTP. Scale throughput with independent **server processes** (a single process plateaus at
about 2x serial on the Python GIL), and put several on consecutive ports; clients
round-robin a comma-separated URL list. Launch from a real shell (a sandbox may SIGKILL
background CUDA).

```bash
bash benchmarks/gpu_services.sh start     # 4 SAM processes x 2 replicas + DINOv3
bash benchmarks/gpu_services.sh urls      # prints the SAM_URLS / DINO_URL to export
```

Or start them by hand:

```bash
# Cellpose-SAM (segmentation) - 4 processes x 2 replicas on 8137-8140
for p in 8137 8138 8139 8140; do
  setsid nohup .venv/bin/python -u -m shape.tools.cell_segmenter.sam_server \
    --host 0.0.0.0 --allow-remote --port $p --replicas 2 > sam_$p.log 2>&1 &
done

# DINOv3 (embedding) - 4 processes x 2 replicas on 8237-8240
export HUGGINGFACE_TOKEN=$HF_TOKEN
for p in 8237 8238 8239 8240; do
  setsid nohup .venv/bin/python -u -m shape.tools.cell_state_analyzer.dinov3_server \
    --host 0.0.0.0 --allow-remote --port $p --replicas 2 > dinov3_$p.log 2>&1 &
done
```

`--host 0.0.0.0 --allow-remote` exposes the service on the private network so workers on
other nodes can reach it. **Security:** the transport is pickle (a remote-code-execution
surface) — only enable `--allow-remote` on a trusted, firewalled network, or keep the
default localhost bind and SSH-tunnel. See
[`shape/tools/cell_segmenter/SAM_SERVICE.md`](shape/tools/cell_segmenter/SAM_SERVICE.md).

### Running each benchmark

The inputs are on Hugging Face,
[`SHAPE-agent/benchmark-data`](https://huggingface.co/datasets/SHAPE-agent/benchmark-data);
download the folder a benchmark needs into `data/` from the repository root
([`benchmarks/README.md`](benchmarks/README.md#data) lists the folders):

```bash
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "NeurIPS22-CellSeg/*" --local-dir data
```

Each benchmark's README gives its dataset, the GPU it needs and the commands for every arm:

| Benchmark | Entry point |
|---|---|
| [Segmentation](benchmarks/segmentation_NeurIPS22-CellSeg/README.md): agentic VRR over Cellpose-SAM against domain tools, an LLM alone and ReAct | `benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh` |
| [Cropper](benchmarks/cropper_NeurIPS22-CellSeg/README.md): MaskCrop, CellProfiler, an LLM alone, ReAct and SHAPE on the ground-truth masks | `benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py` |
| [Batch correction (JUMP)](benchmarks/batch_correction_JUMP/README.md): Harmony tuned by the budgeted VRR, scored with the Arevalo metrics (CPU only) | `benchmarks/batch_correction_JUMP/run_batch_benchmark.py` |
| [Representation (RxRx1)](benchmarks/representation_RxRx1/README.md): the self-supervised DINOv3 fine-tune tuned call by call | `benchmarks/representation_RxRx1/run_rxrx1.py` |
| [End to end (CPJUMP1, ModalityProbe3)](benchmarks/full_CPJUMP1/README.md): the whole chain against a default-parameter control and external coding agents (Codex, Claude Code, Biomni) | `benchmarks/full_CPJUMP1/runners/run_all.sh` |

## Case studies

[`case studies/`](case%20studies/README.md) holds end-to-end applications on experimental
datasets, as distinct from the public-dataset benchmarks. The
[cardiomyocyte studies](case%20studies/cardiomyocytes/README.md) predict single-cell cTnT
positivity from brightfield alone against paired staining, and well-level cTnT-positive
proportions against flow cytometry. The
[cardiac organoid](case%20studies/cardiac%20organoids/README.md) and
[tissue](case%20studies/tissues/README.md) applications have the same structure: a
`harness/` with the pinned data contract and a `runners/` with the entry points. Raw images
and labels are not committed to Git; the cardiomyocyte datasets are public on Hugging Face
(linked from that study's README).

## Development

```bash
python -m pip install -e ".[test]"
pytest
```

Tests that need a GPU, the LLM gateway or a private dataset are skipped or fail without
them; run the suite on a machine that has what the part under test uses.

## License

SHAPE is released under the [CC BY 4.0 license](LICENSE). Redistributed third-party source
sits under `benchmarks/batch_correction_JUMP/runners/vendor/` with its own licence and its
provenance in that benchmark's README. The benchmark datasets on Hugging Face keep their
upstream licences, listed on the
[dataset card](https://huggingface.co/datasets/SHAPE-agent/benchmark-data).
