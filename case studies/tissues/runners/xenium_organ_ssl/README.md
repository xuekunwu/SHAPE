# tissues/runners/xenium_organ_ssl

Cell-class probe of four arms' embeddings of Xenium Prime single-cell crops, three disjoint 10% draws per organ:
`control` (frozen DINOv3 ViT-B/16, zero-shot), `shape` (SHAPE's representation verify-reason-refine loop,
`gpt-5.6-luna` at high effort), `codex` (the Codex CLI through the agent-arm stage protocol) and `scportrait`
(`facebook/vit-mae-base` MAE-fine-tuned as in scPortrait). Every arm gets the same prepared draw, 8 training calls
with the same epoch ceiling and a 48 h deadline. The labels are the cross-tissue classes of
[`xenium_lineage_annotation`](../xenium_lineage_annotation/README.md).

## Dataset

Per organ and draw, the prepared draw (`PREP`), the arms' delivered embeddings (`RUN` per arm) and the class
labels (`class.csv.gz` of `xenium_lineage_annotation`).

## GPU

Not required for the evaluation (CPU).

## Running

```
PROBE_LABEL=class python "case studies/tissues/runners/xenium_organ_ssl/evaluation/linear_probe_v3.py" \
    PREP class.csv.gz OUT control=RUN_CONTROL shape=RUN_SHAPE codex=RUN_CODEX scportrait=RUN_SCPORTRAIT
python "case studies/tissues/runners/xenium_organ_ssl/evaluation/summarize_probe.py" ORGAN DRAW [PROBE_NAME]
python "case studies/tissues/runners/xenium_organ_ssl/evaluation/merge_draws.py" ORGAN [PROBE_NAME]
python "case studies/tissues/runners/xenium_organ_ssl/evaluation/export_h5ad.py" ROOT ORGAN LABELS_NAME PROBE [PROBE ...] OUT_DIR
```

`linear_probe_v3.py` is `linear_probe_v2.py` (the frozen lung probe with the label column read from
`PROBE_LABEL`) plus `PROBE_EXCLUDE`, the label values left out of every split (breast `Others`); without it the two
agree. `summarize_probe.py` reports, per draw, the delivered representation, training calls, tokens, stage
minutes and run errors; `merge_draws.py` the mean and 95% t interval over the three draws per organ (both read
the layout `~/shape/runs/xenium_ssl/<ORGAN>/draw<N>/runs` and `.../eval/<ORGAN>/draw<N>/<PROBE_NAME>`);
`export_h5ad.py` one h5ad per organ and arm with embeddings, splits, labels, held-out predictions and a UMAP.
