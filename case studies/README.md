# SHAPE Case Studies

End-to-end applications of SHAPE on experimental data, one directory per study. Each study keeps its
data contract (`harness/dataset.json` where it has one), its preparation step, the arms and one run
entry point; its README gives the data, the GPU it needs and the commands that run every arm.

| Study | Question | Data | Entry point |
|---|---|---|---|
| [`cardiomyocytes/ctnt_staining`](cardiomyocytes/ctnt_staining/README.md) | per-cell cTnT positivity from brightfield, against cTnT staining | [SHAPE-agent/iPSC-CM-cTnT_staining](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_staining) | `runners/run.py` |
| [`cardiomyocytes/ctnt_flowcytometry`](cardiomyocytes/ctnt_flowcytometry/README.md) | the cTnT-positive percentage of a well from brightfield, against flow cytometry | [SHAPE-agent/iPSC-CM-cTnT_flowcytometry](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_flowcytometry) | `runners/run.py` |
| [`cardiac organoids/fc_timecourse`](cardiac%20organoids/fc_timecourse/README.md) | culture day and Control vs FC from phase contrast (23 wells x 8 time points) | [SHAPE-agent/CO_timecourse](https://huggingface.co/datasets/SHAPE-agent/CO_timecourse) | `runners/e2e.py` |
| [`cardiac organoids/runners/define_candidates_96well`](cardiac%20organoids/runners/define_candidates_96well/README.md) | segmentation, cropping and two-channel SSL embedding of a 1,141-image drug screen | [SHAPE-agent/CO_drugscreen](https://huggingface.co/datasets/SHAPE-agent/CO_drugscreen) | `full_driver.py` |
| [`cardiac organoids/runners/segmentation`](cardiac%20organoids/runners/segmentation/README.md) | whole-organoid instance segmentation: SHAPE against non-LLM baselines, an LLM alone and ReAct | private (in-house images and instance masks) | `run_organoid_segmentation.py` |
| [`tissues/runners/xenium_lineage_annotation`](tissues/runners/xenium_lineage_annotation/README.md) | cross-tissue cell classes of seven Xenium Prime sections | 10x Genomics Xenium Prime FFPE sections | `annotate.py` |
| [`tissues/runners/xenium_lung_ssl`](tissues/runners/xenium_lung_ssl/README.md), [`xenium_organ_ssl`](tissues/runners/xenium_organ_ssl/README.md) | cell-type and cell-class linear probes of the arms' SSL embeddings | the arms' delivered embeddings | `evaluation/linear_probe*.py` |

`_shared/metadata.py` is the Hugging Face metadata admission the two cardiomyocyte studies share
(`run.py inspect`, `fetch-metadata`, `audit-metadata`). `basic_usage.py` is a minimal `solve()` example on
the two bundled images, `iPSC-cardiomyocyte.tif` and `baseball.png`.

Every LLM arm reads `LLM_API_KEY` and `LLM_BASE_URL` from the environment; the Codex arm also needs the
Codex CLI and Node on `PATH`. The pinned model assets (DINOv3 source and weights, Cellpose-SAM weights)
that `--assets-root` points at are built by `python -m benchmarks.full_CPJUMP1.harness.cli assets`.
