# Tissues Runners

Preparation, smoke, full-run and evaluation entry points of the tissue studies.

| Runner | Content |
|---|---|
| [xenium_lineage_annotation](xenium_lineage_annotation/README.md) | Cross-tissue cell classes (Epithelial / Endothelial / Stromal / Immune / Tumor / Others) of the seven Xenium Prime sections: CellTypist and panel-marker opinions combined per cluster, tumor-type-specific tumor markers, Slurm launch script and end-to-end tests. |
| [xenium_organ_ssl](xenium_organ_ssl/README.md) | Code record of the Xenium organ SSL study: pinned source (`SHAPE-v2.0@436ff4e3` + `source_delta/`, `source_sha256` 94f841f3), the VM launch scripts (queue, arms, scPortrait ViT-MAE arm, auto-stop), the cell-class probe, the draw merge and the h5ad export. |
| [xenium_lung_ssl](xenium_lung_ssl/README.md) | Code record of the Xenium lung SSL study: pinned source (`cs-co-timecourse@64a18ffc` + `source_delta/`, `source_sha256` 29864c81), Slurm launch scripts, the cell-type probe and per-class predictive values. |
