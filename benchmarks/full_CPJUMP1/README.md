# full_CPJUMP1

End-to-end benchmark on real Cell Painting microscopy: a SHAPE agent (or a peer agent arm --
Codex, Claude Code, Biomni) must independently discover and tune a 5-stage pipeline (segment,
crop, embed, batch-correct, cluster) to deliver per-well profiles, scored against ground truth.
Two datasets share this same harness and pipeline:

- **CPJUMP1** -- the benchmark's own CRISPR dataset (A549 Cas9, 2 plates, 1000 multichannel
  views / 5000 channel TIFFs).
- **ModalityProbe3** -- 6 (cell_type, perturbation) groups x 100 images = 600 images, mixed
  modality/cell-type/perturbation structure.

Arms: `control` (no agent), `shape` (the SHAPE agent), `codex`, `claude` (Claude Code), `biomni`
-- each accepting several reasoner models (gpt-6-sol, gpt-5.6-luna, claude-opus-5,
claude-sonnet-5-5, zai-org/GLM-5.3-Flash, deepseek-ai/DeepSeek-V4.1-Flash; see `harness/cli.py`'s
`ARM_ALLOWED` / `shape/agent/arms`'s `ARM_ALLOWED` for which model routes where).

## Dataset

CPJUMP1, from Hugging Face (the 5,000 TIFFs under `raw/` plus `manifest.json`, the same file as
`runners/manifest.json`):
```
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "CPJUMP1/*" --local-dir data
python -m benchmarks.full_CPJUMP1.harness.cli prepare --data data/CPJUMP1 --out data/CPJUMP1/prepared
python -m benchmarks.full_CPJUMP1.harness.cli assets --root data/CPJUMP1/assets \
    --dinov3-repo DINOV3 --weights DINOV3_WEIGHTS --cpsam CPSAM
```
`DINOV3` is a checkout of facebookresearch/dinov3 at commit `6876159a11b4df116f30f667f8c9888617df0751`,
`DINOV3_WEIGHTS` the ViT-B/16 weights `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` of Meta's DINOv3 release
(sha256 `73cec8be...`) and `CPSAM` the Cellpose-SAM weights file (`cpsam`, which Cellpose 4 downloads to
`~/.cellpose/models/`). `assets` copies them into a pinned, offline root; a compute node then needs no network.
`runners/download_cpjump1.py` fetches the same files from the Cell Painting Gallery instead
(`--manifest benchmarks/full_CPJUMP1/runners/manifest.json --out data/CPJUMP1`).

ModalityProbe3:
```
python -m benchmarks.full_CPJUMP1.runners.download_modality_probe3 --out data/CPJUMP1_modality_probe3
python -m benchmarks.full_CPJUMP1.runners.prepare_modality_probe3 \
    --raw data/CPJUMP1_modality_probe3 --out data/CPJUMP1_modality_probe3/prepared
python benchmarks/full_CPJUMP1/runners/fetch_modality_probe3_identities.py   # each well's treatment identity, for scoring
```

## Running the arms

One script picks the dataset, the phase (segcrop / embed / all) and which arm slugs to run:

```
export SHAPE_SAM_SERVER_URL=http://...   # optional: a Cellpose-SAM service (benchmarks/gpu_services.sh); unset, segment runs on the local GPU
bash benchmarks/full_CPJUMP1/runners/run_all.sh --dataset cpjump1 all shape_gpt6sol codex_gpt6sol
bash benchmarks/full_CPJUMP1/runners/run_all.sh --dataset probe3 all shape_luna shape_sonnet55
```

`JOBS=N` controls how many arms run concurrently during `segcrop` (CPU); `embed` always runs one
arm at a time (GPU). See `run_all.sh`'s own header for the full arm matrix per dataset.
