"""Marker sets, CellTypist label maps and per-organ tumor programs of the Xenium cross-tissue class annotation.

Genes are restricted to the Xenium Prime 5K panels (several classic markers such as PTPRC, KRT8/18/19, COL1A1, VWF and
ACTA2 are not on them); a set is intersected with each section's panel at run time. ERG is in no set (endothelial
cells express it, and the prostate section is ERG-negative), MCAM and KIT (expressed by melanoma) are left out of the
stromal and immune sets.
"""

CLASSES = ('Epithelial', 'Endothelial', 'Stromal', 'Immune', 'Tumor', 'Others')

# Marker lineages. Melanocytic and Adipocyte are called so that those cells do not fall into a main class;
# they end in Others unless the melanoma tumor step claims them.
LINEAGE_GENES = {
    'Epithelial': ['EPCAM', 'CDH1', 'CLDN7', 'GRHL2', 'CDH3', 'DSG3', 'SPRR3', 'MUC1', 'AGER'],
    'Endothelial': ['PECAM1', 'CDH5', 'CLDN5', 'KDR', 'PLVAP', 'EGFL7', 'TIE1', 'ROBO4', 'MMRN2', 'PROX1'],
    'Stromal': ['PDGFRA', 'PDGFRB', 'FAP', 'DCN', 'LUM', 'FBLN2', 'MFAP5', 'SFRP4', 'COL5A1', 'POSTN', 'THBS2',
                'PI16', 'RGS5', 'NOTCH3', 'MYH11', 'CNN1', 'SYNPO2', 'PLN', 'SMTN'],
    'Immune': ['CD3E', 'CD2', 'CD247', 'CD7', 'MS4A1', 'CD79A', 'CD19', 'MZB1', 'CD68', 'CD14', 'CSF1R', 'CD163',
               'ITGAX', 'CSF3R'],
    'Melanocytic': ['SOX10', 'MLANA', 'DCT', 'TRPM1'],  # not MITF: mast cells and macrophages express it
    'Adipocyte': ['PLIN1', 'ADIPOQ', 'LEP'],
}
MAIN = ('Epithelial', 'Endothelial', 'Stromal', 'Immune')


def _labels(**groups):
    return {label: lineage for lineage, labels in groups.items() for label in labels}


# CellTypist label -> lineage. Others = a real minor class (mesothelium, Schwann cells); labels mapped to None are
# ambiguous at lineage level and do not vote.
CELLTYPIST_LINEAGE = {
    'Human_Lung_Atlas.pkl': _labels(
        Epithelial=['AT0', 'AT1', 'AT2', 'AT2 proliferating', 'Basal resting', 'Club (nasal)', 'Club (non-nasal)',
                    'Deuterosomal', 'Goblet (bronchial)', 'Goblet (nasal)', 'Goblet (subsegmental)', 'Hillock-like',
                    'Ionocyte', 'Multiciliated (nasal)', 'Multiciliated (non-nasal)', 'Neuroendocrine', 'SMG duct',
                    'SMG mucous', 'SMG serous (bronchial)', 'SMG serous (nasal)', 'Suprabasal', 'Tuft',
                    'pre-TB secretory'],
        Endothelial=['EC aerocyte capillary', 'EC arterial', 'EC general capillary', 'EC venous pulmonary',
                     'EC venous systemic', 'Lymphatic EC differentiating', 'Lymphatic EC mature',
                     'Lymphatic EC proliferating'],
        Stromal=['Adventitial fibroblasts', 'Alveolar fibroblasts', 'Myofibroblasts', 'Peribronchial fibroblasts',
                 'Pericytes', 'SM activated stress response', 'Smooth muscle', 'Smooth muscle FAM83D+',
                 'Subpleural fibroblasts'],
        Immune=['Alveolar Mph CCL3+', 'Alveolar Mph MT-positive', 'Alveolar Mph proliferating', 'Alveolar macrophages',
                'B cells', 'CD4 T cells', 'CD8 T cells', 'Classical monocytes', 'DC1', 'DC2',
                'Hematopoietic stem cells', 'Interstitial Mph perivascular', 'Mast cells', 'Migratory DCs',
                'Monocyte-derived Mph', 'NK cells', 'Non-classical monocytes', 'Plasma cells', 'Plasmacytoid DCs',
                'T cells proliferating'],
        Others=['Mesothelium']),
    'Adult_Human_Skin.pkl': _labels(
        Epithelial=['Differentiated_KC', 'Undifferentiated_KC'],
        Endothelial=['LE1', 'LE2', 'VE1', 'VE2', 'VE3'],
        Stromal=['F1', 'F2', 'F3', 'Pericyte_1', 'Pericyte_2'],
        Immune=['DC1', 'DC2', 'ILC1_3', 'ILC1_NK', 'ILC2', 'Inf_mac', 'LC', 'Macro_1', 'Macro_2', 'Mast_cell', 'MigDC',
                'Mono_mac', 'NK', 'Plasma', 'Tc', 'Th', 'Treg', 'migLC', 'moDC'],
        Melanocytic=['Melanocyte'],
        Others=['Schwann_1', 'Schwann_2']),
    'Cells_Adult_Breast.pkl': _labels(
        Epithelial=['LummHR-SCGB', 'LummHR-active', 'LummHR-major', 'Lumsec-HLA', 'Lumsec-KIT', 'Lumsec-basal',
                    'Lumsec-lac', 'Lumsec-major', 'Lumsec-myo', 'Lumsec-prol', 'basal'],
        Endothelial=['Lymph-immune', 'Lymph-major', 'Lymph-valve1', 'Lymph-valve2', 'Vas-arterial', 'Vas-capillary',
                     'Vas-venous'],
        Stromal=['Fibro-SFRP4', 'Fibro-major', 'Fibro-matrix', 'Fibro-prematrix', 'pericytes', 'vsmc'],
        Immune=['CD4-Tem', 'CD4-Th', 'CD4-Th-like', 'CD4-Treg', 'CD4-activated', 'CD4-naive', 'CD8-Tem', 'CD8-Trm',
                'CD8-activated', 'GD', 'Macro-IFN', 'Macro-lipo', 'Macro-m1', 'Macro-m1-CCL', 'Macro-m2',
                'Macro-m2-CXCL', 'Mast', 'Mono-classical', 'Mono-non-classical', 'NK', 'NK-ILCs', 'NKT', 'Neutrophil',
                'T_prol', 'b_naive', 'bmem_switched', 'bmem_unswitched', 'cDC1', 'cDC2', 'mDC', 'mye-prol', 'pDC',
                'plasma_IgA', 'plasma_IgG']),
}

# Tumor program per organ: parent marker lineage, tumor genes (sensitive markers of that tumor type) and matched
# normal-counterpart genes (empty: none on the panel or the normal counterpart is rare in the section).
TUMOR = {
    'HumanLungCancer': dict(  # adenocarcinoma-like section (NKX2-1, FOLR1), squamous genes kept for mixed areas
        parent='Epithelial',
        tumor=['NKX2-1', 'FOLR1', 'MUC1', 'LAMB3', 'SLC7A11', 'CEACAM1', 'MET', 'SOX2', 'TP63', 'DSG3', 'MKI67',
               'TOP2A', 'CA9'],
        normal=['AGER', 'EMP2', 'RTKN2', 'ABCA3', 'LAMP3', 'CEBPA', 'ETV5', 'FOXJ1', 'TPPP3']),
    'HumanBreastCancer': dict(  # ER+ (ESR1, GATA3, FOXA1); normal counterpart: myoepithelial / luminal progenitor
        parent='Epithelial',
        tumor=['ESR1', 'GATA3', 'FOXA1', 'ERBB2', 'MUC1', 'AGR2', 'AGR3', 'CA12', 'MLPH', 'TBX3', 'KIAA1324',
               'MKI67', 'TOP2A'],
        normal=['TP63', 'OXTR', 'MYH11', 'KIT', 'SFRP1']),
    'HumanProstate': dict(  # ERG-negative adenocarcinoma; normal counterpart: basal cells
        parent='Epithelial',
        tumor=['AMACR', 'PCA3', 'FOLH1', 'TMEFF2', 'SIM2', 'ETV1', 'CACNA1D'],  # not ERG: endothelial cells express it
        normal=['TP63']),
    'HumanOvarianCancer': dict(  # high-grade serous-like (PAX8, WT1, MUC16, CDKN2A); normal: ciliated, mesothelium
        parent='Epithelial',
        # not WT1, MEIS1: ovarian stroma expresses both, which raised the background above PAX8+/MUC16+ tumor nests
        tumor=['PAX8', 'MUC16', 'MSLN', 'FOLR1', 'CP', 'SOX17', 'CDKN2A', 'CCNE1', 'MAL2', 'MKI67', 'TOP2A'],
        normal=['FOXJ1', 'UPK3B', 'CALB2']),
    'HumanCervicalCancer': dict(  # HPV16-positive: viral E6/E7 transcripts and p16 (CDKN2A)
        parent='Epithelial',
        tumor=['HPV16-E6', 'HPV16-E7', 'HPV18-E6', 'HPV18-E7', 'CDKN2A'],
        normal=[]),
    'HumanSkinMelanoma': dict(  # melanoma: melanocytic program plus PRAME
        parent='Melanocytic',
        tumor=['PRAME', 'SOX10', 'MLANA', 'DCT', 'TRPM1', 'MITF'],
        normal=[]),
    'HumanLymphNode': None,  # 10x reactive lymph node: no tumor
}

CELLTYPIST_MODEL = {
    'HumanLungCancer': 'Human_Lung_Atlas.pkl',
    'HumanSkinMelanoma': 'Adult_Human_Skin.pkl',
    'HumanBreastCancer': 'Cells_Adult_Breast.pkl',
    # v2 runs used Cells_Human_Tonsil (lymph node) and Human_Endometrium_Atlas (ovarian, cervical): the tonsil model
    # called >= 97% of every non-immune cluster Immune, the endometrium model called HGSOC tumor cells Immune_Lymphoid or
    # stroma; neither is a usable second opinion, so these sections are markers only.
    'HumanLymphNode': None,
    'HumanOvarianCancer': None,
    'HumanCervicalCancer': None,
    'HumanProstate': None,  # no prostate model in the CellTypist catalogue: markers only
}

# Lineages a section cannot contain: their marker sets are not scored there. The reactive lymph node has no
# epithelium; with an epithelial set, EPCAM/CLDN7/GRHL2-negative T and myeloid clusters were called Epithelial on CDH1
# and AGER alone.
ABSENT = {'HumanLymphNode': ['Epithelial']}

# Method version per organ (annotate.py docstring): v2 produced the delivered lung labels; v3 for the other sections.
METHOD = {'HumanLungCancer': 'v2', 'HumanSkinMelanoma': 'v3', 'HumanProstate': 'v3', 'HumanBreastCancer': 'v3',
          'HumanLymphNode': 'v3', 'HumanCervicalCancer': 'v3', 'HumanOvarianCancer': 'v3'}
