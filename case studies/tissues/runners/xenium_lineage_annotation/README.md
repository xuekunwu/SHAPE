# tissues/runners/xenium_lineage_annotation

Cross-tissue cell classes (Epithelial, Endothelial, Stromal, Immune, Tumor, Others) for seven Xenium Prime FFPE
sections: lung cancer, skin melanoma, prostate, breast cancer, reactive lymph node, cervical cancer and ovarian
cancer. They are the labels of the multi-organ SSL cell-class probe (`../xenium_organ_ssl`). A CellTypist opinion
and a panel-marker opinion are combined per cluster, and the tumor call uses markers specific to each tumor type
(`markers.py`: Xenium Prime 5K panel gene sets, CellTypist label maps and models per organ). The protocol is in
the docstring of `annotate.py`.

## Dataset

Each section's transcript h5ad with raw counts and its 10x `cells.parquet` (10x Genomics Xenium Prime datasets),
and the CellTypist models named in `markers.py`.

## Running

```
python "case studies/tissues/runners/xenium_lineage_annotation/annotate.py" ORGAN H5AD CELLS_PARQUET MODEL_DIR OUT
```

Per organ it writes `class.csv.gz` (`cell_id,class`), `cells.csv.gz` (per-cell evidence and scores), the cluster
and tumor-subcluster tables, `summary.json` (counts, gene sets, input and model sha256) and two figures.
