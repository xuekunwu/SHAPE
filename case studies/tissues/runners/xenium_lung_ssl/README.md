# tissues/runners/xenium_lung_ssl

Cell-type probe of three arms' (`control`, `shape`, `codex`) self-supervised DINOv3 embeddings of four-channel
single-cell crops of a Xenium Prime human lung cancer section, on held-out spatial tiles. The embeddings come
from the representation stage of the shared harness (`benchmarks/full_CPJUMP1/harness`); this folder holds the
probe.

## Dataset

The prepared section (`PREP`, with `public/images.json` giving each cell's split), the arms' delivered
embeddings (`RUN` per arm) and the CellTypist labels exported as `cell_id,celltype_final` to
`celltype_final.csv.gz`.

## GPU

Not required for the probe (CPU).

## Running

```
python "case studies/tissues/runners/xenium_lung_ssl/evaluation/linear_probe.py" PREP celltype_final.csv.gz OUT \
    control=RUN_CONTROL shape=RUN_SHAPE codex=RUN_CODEX
cd OUT && python "$REPO/case studies/tissues/runners/xenium_lung_ssl/evaluation/per_class_predictive_values.py"
```

`per_class_predictive_values.py` reads the probe's `metrics.json` and `predictions.csv` from the current
directory (`REPO` is the repository root).

The probe: L2-normalised embeddings, StandardScaler and multinomial logistic regression fitted on the training
cells, C chosen on the validation cells, scored once on the held-out tiles with 1,000 tile bootstrap draws
(seed 20261026) shared by every arm, plus a chance group; `per_class_predictive_values.py` adds PPV, NPV,
sensitivity and specificity per class.
