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

## Install

Requires **Python 3.10+**.

```bash
git clone <this repository>
cd SHAPE
python -m pip install -e .
python -m pip install -e ".[batch-metrics]"   # for the benchmarks: copairs, scib-metrics, scanorama
```

The segmentation tool needs Cellpose-SAM (`cellpose>=4.0`) and a CUDA-capable PyTorch
build for anything beyond small images.

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
and dataset.

### Running each benchmark

The inputs are on Hugging Face,
[`SHAPE-agent/benchmark-data`](https://huggingface.co/datasets/SHAPE-agent/benchmark-data);
download the folder a benchmark needs into `data/` from the repository root
([`benchmarks/README.md`](benchmarks/README.md#data) lists the folders):

```bash
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "NeurIPS22-CellSeg/*" --local-dir data
```

Each benchmark's README gives its dataset and the commands for every arm:

| Benchmark | Entry point |
|---|---|
| [Segmentation](benchmarks/segmentation_NeurIPS22-CellSeg/README.md): agentic VRR over Cellpose-SAM against domain tools, an LLM alone and ReAct | `benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh` |
| [Cropper](benchmarks/cropper_NeurIPS22-CellSeg/README.md): MaskCrop, CellProfiler, an LLM alone, ReAct and SHAPE on the ground-truth masks | `benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py` |
| [Batch correction (JUMP)](benchmarks/batch_correction_JUMP/README.md): Harmony tuned by the budgeted VRR, scored with the Arevalo metrics (CPU only) | `benchmarks/batch_correction_JUMP/run_batch_benchmark.py` |
| [Representation (RxRx1)](benchmarks/representation_RxRx1/README.md): the self-supervised DINOv3 fine-tune tuned call by call | `benchmarks/representation_RxRx1/run_rxrx1.py` |
| [End to end (CPJUMP1, ModalityProbe3)](benchmarks/full_CPJUMP1/README.md): the whole chain against a default-parameter control and external coding agents (Codex, Claude Code, Biomni) | `benchmarks/full_CPJUMP1/runners/run_all.sh` |

## Case studies

[`case studies/`](case%20studies/README.md) holds end-to-end applications on experimental
data, one directory per study, each README giving the data and the commands: cTnT positivity of iPSC-derived cardiomyocytes from brightfield alone, against
staining (per cell) and flow cytometry (per well); cardiac-organoid segmentation, a
time course and a drug screen; and cell-class probes of SSL embeddings of Xenium Prime
tissue sections. The cardiomyocyte and organoid time-course and drug-screen data are public
on Hugging Face under [SHAPE-agent](https://huggingface.co/SHAPE-agent).

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
