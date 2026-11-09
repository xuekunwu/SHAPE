# Xenium organ SSL runners (control / shape / codex / scportrait)

Evaluation code of the multi-organ SSL embedding comparison on Xenium Prime single-cell crops (the study package
defines seven sections: lung, skin melanoma, prostate, breast, lymph node, cervical, ovarian). The embedding stage
is the shared harness of `benchmarks/full_CPJUMP1/harness`; this folder holds the cell-class probe and its summaries.

## Arms and budgets

- **control**: frozen DINOv3 ViT-B/16, zero-shot (user decision: no tuning).
- **shape**: SHAPE's representation verify-reason-refine loop (`gpt-5.6-luna`, reasoning effort high).
- **codex**: Codex CLI through the v2.0 agent stage protocol (`gpt-5.6-luna`, high; 32 decision rounds, 8 tool-call rounds).
- **scportrait**: `facebook/vit-mae-base` (revision 25b184be) MAE-fine-tuned as in the scPortrait preprint's Figure 2 (3 channels without DAPI, 128 px crop resized to 224, batch 128, mask ratio 0.75, AdamW lr 1.5e-4 x batch / 256, mean of all encoder tokens) on the draw's development_train crops; epoch ceiling 8 x the SHAPE rule value, SHAPE's `AdaptiveEarlyStopper` on the training loss with patience only (user decisions 2026-10-27).
- Shared: per training call the fixed epoch ceiling `clamp(round(8 + 6 * sqrt(128 / N_train)), 8, 20)` (no extension), 8 training calls, a 48 h arm deadline, the same prepared draw (cells, split, wells) for every arm.

## Evaluation

| Script | Role |
|---|---|
| `linear_probe_v2.py` | The frozen lung probe (`../xenium_lung_ssl/evaluation/linear_probe.py`, sha256 9252001f) with the label column read from `PROBE_LABEL`; nothing else changed. L2-normalised embeddings, StandardScaler and multinomial logistic regression fitted on development_train, C from {0.01, 0.1, 1, 10} by validation balanced accuracy, scored once on the held-out tiles; chance group, 1000 tile-bootstrap draws shared by all groups. |
| `linear_probe_v3.py` | v2 plus `PROBE_EXCLUDE` (label values left out of every split; breast `Others`, user decision 2026-10-27); identical to v2 without it (checked on skin draw 1). |
| `summarize_probe.py` | Per draw: delivered representation, training calls, tokens, stage minutes, run errors (tracebacks; shape execution errors, failed trials, reasoner failures, fallback rounds; codex rejected rounds, failed calls) and the probe metrics. |
| `merge_draws.py` | Per organ: mean and 95 percent t interval over the three draws (df 2) of every metric and paired difference, the cost per arm, and the pooled held-out cells of the three draws with a (draw, tile) bootstrap. |
| `export_h5ad.py` | One h5ad per organ and arm with the cells of the three draws: embedding (768), cell records, split, class labels, held-out predictions, a UMAP (Fig4b protocol, fitted on the development_train rows). |

Labels: the cross-tissue classes of [xenium_lineage_annotation](../xenium_lineage_annotation/README.md)
(Epithelial / Endothelial / Stromal / Immune / Tumor / Others).
