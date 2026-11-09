# Xenium cross-tissue class annotation

Coarse cell classes for the seven Xenium Prime FFPE sections (lung cancer, skin melanoma, prostate, breast cancer,
reactive lymph node, cervical cancer, ovarian cancer): **Epithelial, Endothelial, Stromal, Immune, Tumor, Others**.
They are the labels of the multi-organ SSL cell-class probe. The method extends the lung notebook
(`20251208_Xenium_lungcancer_transcript_adata.ipynb`: CellTypist `Human_Lung_Atlas`, fine labels collapsed to
lineages, the top 20% of a summed tumor-marker count set to Tumor). Here it is made organ-general and coarse: a CellTypist
opinion and a panel-marker opinion are combined, and the tumor call uses markers specific to each tumor type.

## What is here

| Path | Content |
|---|---|
| `annotate.py` | One section: QC, normalisation, clustering, the two opinions, their combination, the tumor step, outputs and figures. The protocol is in the module docstring. |
| `markers.py` | Lineage gene sets (Xenium Prime 5K panel genes only), CellTypist label -> lineage maps, CellTypist model per organ, tumor and normal-counterpart genes per organ. |
| `test_annotate.py` | End-to-end tests of `annotate.main` on a synthetic section with a CellTypist model trained in the test (planted classes recovered; the marker call is kept when the model lacks a lineage; a cell_id mismatch with `cells.parquet` is refused), a test of every branch of the combination rule, and tests of the v3 gene filter and tumor background. |

## Method

1. Cells with fewer than 10 transcripts become `Others` (evidence `low_counts`). The rest are normalised (normalize_total 1e4
   + log1p, the notebook and CellTypist convention), then PCA (50 components, all panel genes, no scaling), a 15-NN graph and Leiden at resolution 2.0.
2. Marker opinion: `sc.tl.score_genes` for Epithelial, Endothelial, Stromal, Immune, Melanocytic and Adipocyte gene
   sets; a cluster takes the lineage with the highest cluster-mean score.
3. CellTypist opinion: per-cell predictions of the organ's model (no majority voting), mapped to lineages; a cluster
   takes the majority lineage.
4. Combination per cluster, in this order:
   - No marker set has a positive cluster-mean score -> the CellTypist call (`celltypist_only`), or `Others` without a
     model (`no_evidence`).
   - The marker call is a lineage the model has no class for, or the organ has no model -> the marker call (`marker_only`).
   - The two calls agree -> that lineage (`agree`).
   - They disagree and the marker margin (best minus second cluster-mean score) is at least 0.2 -> the marker call
     (`marker_strong`). On the lung section, CellTypist reached only a 30-50% plurality, almost always Epithelial, in
     every such cluster, while the markers clearly separated Stromal, Endothelial or Immune.
   - Otherwise each cell keeps the lineage that both opinions give it (`conflict_cell_agree`) or becomes `Others`
     (`unresolved`).
5. Tumor: cells of the organ's parent lineage are re-clustered (Leiden 1.0). A subcluster is `Tumor` when its mean
   tumor score is above the background (95th percentile of the tumor score in main-lineage cells outside the parent)
   and above its mean normal-counterpart score (when the organ has one).
6. Melanocytic, Adipocyte and CellTypist minor classes (mesothelium, Schwann cells) end in `Others`.

### Method versions (`markers.METHOD`)

The lung labels were delivered with v2. The other six sections failed QC under v2 and use v3 (user decision,
2026-10-27: keep lung at v2). v3 adds three rules, applied to all six:

- **Section-adaptive gene sets.** A set gene detected in fewer than 1% of the section's cells is dropped from the
  lineage, tumor and normal sets. Undetected genes only diluted the scores: squamous genes, for example, diluted the
  epithelial score of the ovarian tumor cells.
- **CellTypist abstains without a majority.** A cluster whose CellTypist plurality is below 50% takes the marker call
  (`ct_no_majority`).
- **Cluster-level tumor background.** The background is at least the highest mean tumor score of a main-lineage
  cluster outside the parent (100 or more cells). In the cervical section, HPV16 E6/E7 and CDKN2A are so sparse that
  the cell-level 95th percentile was 0: cells without counts in either the program or its matched control genes score
  exactly 0. The cell-level threshold alone then called 91% of the epithelium Tumor.

Also changed for v3 sections: ovarian, cervical and lymph node are markers only (no CellTypist model, see the table),
and ERG is not a prostate tumor gene, because endothelial cells express it and it raised the background (the section is ERG-negative).
After the first v3 run (user decisions, 2026-10-27), two organ-specific fixes were added. The lymph node has no epithelial
lineage (`markers.ABSENT`): with an epithelial set, 28,946 EPCAM/CLDN7/GRHL2-negative T and myeloid cells were called
Epithelial on CDH1 and AGER alone. WT1 and MEIS1 are not ovarian tumor genes: ovarian stroma expresses both, which raised
the background above PAX8+/MUC16+ cells inside the tumor nests (about 63,000 cells were left as Epithelial).
In v3, a CellTypist plurality below 50% takes precedence over agreement, so `ct_no_majority` also covers clusters where the
marker call equals that plurality: the label is the same, but the `agree` share in `summary.json` understates agreement.

| Organ (10x section) | Method | CellTypist model | Tumor parent | Tumor genes (normal counterpart) |
|---|---|---|---|---|
| Lung cancer | v2 | `Human_Lung_Atlas` (HLCA v2) | Epithelial | NKX2-1, FOLR1, MUC1, LAMB3, SLC7A11, CEACAM1, MET, SOX2, TP63, DSG3, MKI67, TOP2A, CA9 (AT1, AT2, ciliated genes) |
| Skin (melanoma) | v3 | `Adult_Human_Skin` | Melanocytic | PRAME, SOX10, MLANA, DCT, TRPM1, MITF |
| Prostate | v3 | none in the catalogue: markers only | Epithelial | AMACR, PCA3, FOLH1, TMEFF2, SIM2, ETV1, CACNA1D (TP63 basal) |
| Breast cancer | v3 | `Cells_Adult_Breast` | Epithelial | ESR1, GATA3, FOXA1, ERBB2, MUC1, AGR2, AGR3, CA12, MLPH, TBX3, KIAA1324, MKI67, TOP2A (myoepithelial, luminal progenitor) |
| Lymph node (reactive) | v3 | markers only (`Cells_Human_Tonsil` called >= 97% of every non-immune cluster Immune in v2); no epithelial lineage | no tumor | - |
| Cervical cancer | v3 | markers only (`Human_Endometrium_Atlas` rejected, as for ovarian) | Epithelial | HPV16/18 E6, E7 probes, CDKN2A |
| Ovarian cancer | v3 | markers only (`Human_Endometrium_Atlas` called PAX8+/MUC16+ tumor cells Immune_Lymphoid or stroma in v2) | Epithelial | PAX8, MUC16, MSLN, FOLR1, CP, SOX17, CDKN2A, CCNE1, MAL2, MKI67, TOP2A (ciliated, mesothelium) |

Model files (CellTypist catalogue, downloaded 2026-10-27), sha256: `Human_Lung_Atlas.pkl` 6b073053...,
`Adult_Human_Skin.pkl` 65bbd15b..., `Cells_Adult_Breast.pkl` 9293ca6f... (full values in each run's `summary.json`).

## Outputs (per organ, `runs/<ORGAN>/`)

| File | Content |
|---|---|
| `class.csv.gz` | `cell_id,class` for every cell of the section (cell_id as in `cells.parquet`, checked before writing). |
| `cells.csv.gz` | Per cell: class, lineage, evidence, cluster, marker cell call, CellTypist label, confidence and lineage, tumor subcluster, tumor and normal scores, total counts. |
| `clusters.csv`, `tumor_subclusters.csv` | The per-cluster decisions and the tumor subcluster table (scores and the background `tau` are in `summary.json`). |
| `summary.json` | Class, lineage and evidence counts, gene sets used and missing, inputs and model sha256, parameters, versions. |
| `spatial_class.png`, `class_scores.png` | Section map coloured by class; mean lineage and tumor scores per class. |

## Reproduce

1. Run folder: copy this folder to `code/` and the three models to `models/`. Inputs: each section's
   transcript h5ad with raw counts and its 10x `cells.parquet`.
2. Environment: conda env `shape_bench` (Python 3.10, scanpy 1.11.5, anndata 0.11.4, leidenalg 0.12.0) plus
   `celltypist==1.7.1`, `openpyxl==3.1.5` and `et_xmlfile==2.0.0` installed beside it on `PYTHONPATH`, which
   leaves `shape_bench` unchanged.
3. Tests: `python -s -m pytest -q test_annotate.py` (6 tests).
4. Per organ: `python annotate.py <ORGAN>`.

## Limits

- The tumor call is a marker heuristic, not a pathology or copy-number call. Benign epithelium that shares the tumor
  program (for example AT2 cells for lung, hormone-receptor luminal cells for breast) can be called Tumor when the
  panel has no gene that separates it; `cells.csv.gz` keeps the scores so the call can be re-thresholded.
- Several classic markers are not on the 5K panel (PTPRC, KRT8/18/19, COL1A1, VWF, ACTA2, KLK3, MSMB, SFTPC, NAPSA).
- Ovarian: without WT1 and MEIS1 the stroma carries no tumor-program signal, so the tumor background can be zero
  even though the QC asks for a positive one. Read the per-cell scores in `cells.csv.gz` before trusting that call.
- A sparse section (few transcripts per cell) raises the unresolved fraction above the QC limit and makes
  single-cell calls noisier than cluster-level calls.
- Prostate, ovarian, cervical and lymph node are markers only (no matched CellTypist reference).
- The 15-NN graph uses approximate, multithreaded neighbour search, so a rerun on other nodes can shift cluster
  boundaries slightly between runs of the same clustering code. The delivered labels are identified by the
  `class_csv_sha256` in each `summary.json`.
