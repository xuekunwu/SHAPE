# Xenium lung SSL runners (control / shape / codex)

Evaluation code of the three-arm SSL embedding comparison on the Xenium Prime human lung cancer crops.
The study code itself is the shared harness of `benchmarks/full_CPJUMP1/harness` (the embed stage
`pipeline.representation`, SHAPE's `RepresentationAgenticVerifier` through `FixedEpochVerifier`, and
the Codex stage protocol `codex_arm.py`), with the cardiac organoid FC time-course runner supplying the
reasoning protocol, the agent settings and the environment record.

## What is here

| Path | Content |
|---|---|
| `evaluation/linear_probe.py` | the cell-type probe with the chance group |
| `evaluation/per_class_predictive_values.py` | PPV, NPV, sensitivity and specificity per class |

## Run

One GPU per arm. `LLM_API_KEY` and `LLM_BASE_URL` come from the environment for shape and codex; codex
also needs the Codex CLI and Node on `PATH`. `--assets-root` is the pinned DINOv3 asset root of the
shared harness.

For the cell-type probe, export `cell_id,celltype_final` of the CellTypist h5ad to `celltype_final.csv.gz`,
run `evaluation/linear_probe.py` over the three arms' embeddings, then `evaluation/per_class_predictive_values.py`
inside the probe output folder.

## Fixed settings of a run

Seed 20261025; reasoner `gpt-5.6-luna` at high effort (max 16,000 tokens, 900 s per reply) for shape and codex; eight
agent decision rounds; neutral tool-API context for codex (`tool_api_v1`); 9 SSL epochs per training call;
`train_cells_per_site` 1 (every development crop enters SSL); inference in chunks of 2,048 crops; SHAPE's expert
start and the replicate-tier proxy `sibling_site_map` on 90 val wells. Each arm writes its own `CONFIG.json`.

## Speed

On RTX 2080 Ti cards: per training call about 2.2 min of training and 3 min of validation per epoch, plus about
8 min between calls on the CPU (clustering, the verifier's readouts, the reasoner). The shape arm runs out of GPU
memory at batch 64 and trains at batch 32; a larger card and 16 CPUs avoid that.
