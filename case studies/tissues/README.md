# Tissues

End-to-end studies on experimental tissue datasets using SHAPE.
Follow the [shared case-study design](../README.md) when adding an application.

| Study | Content |
|---|---|
| Xenium lung cancer crops | Three-arm (control / shape / codex) SSL embeddings of four-channel single-cell crops of a Xenium Prime human lung cancer section, scored by a cell-type linear probe on held-out spatial tiles. [Runners](runners/xenium_lung_ssl/README.md). |
| Xenium organ SSL | Four-arm (control / shape / codex / scportrait) SSL embeddings of the crops of seven Xenium Prime sections, three disjoint 10 percent draws per organ, scored by a linear probe of the cross-tissue cell classes on held-out spatial tiles. [Runners](runners/xenium_organ_ssl/README.md). |
| Xenium cross-tissue cell classes | Epithelial / Endothelial / Stromal / Immune / Tumor / Others labels for the seven Xenium Prime sections (lung, skin melanoma, prostate, breast, lymph node, cervical, ovarian), from CellTypist and panel markers combined, with tumor-type-specific tumor markers. [Runners](runners/xenium_lineage_annotation/README.md). |

- [Harness](harness/): data and pipeline adapters.
- [Runners](runners/): execution entry points and code records.
