"""Shared HCMA label helpers: normalize the messy cell_type strings into canonical
morphological CELL STATES, and attach any label to an AnnData's obs by image-filename stem.

Why normalize: HCMA cell_type is free text with case/plural/-like variants and vendor-specific
naming (ATCC 'epithelial cell' vs ScienCell 'Epithelial Cells'), so the raw 127 values are
fragmented and (as strings) never span vendors. Collapsing to canonical stems yields ~86
cell states, of which ~23 span >=2 organs and ~10 span >=2 vendors -- making cell state the
biology to recover and organ (or vendor) a correctable batch.
"""
from __future__ import annotations

import os
import re

import pandas as pd

# canonical morphological stems, checked as substrings after light cleaning
_STEMS = ("epithelial", "fibroblast", "endothelial", "smooth muscle", "melanocyte",
          "lymphocyte", "lymphoblast", "myocyte", "neuroblast", "hematopoietic",
          "keratinocyte", "macrophage", "monocyte", "astrocyte", "neuron", "hepatocyte",
          "schwann", "mesenchymal", "myoblast", "chondrocyte", "osteoblast", "adipocyte",
          "pericyte", "stem")


def normalize_cell_state(s) -> str:
    s = str(s).lower().strip()
    if s in ("nan", "none", "", "na", "null"):
        return "unknown"
    s = re.sub(r"[-_]like", "", s)
    s = re.sub(r"\bcells?\b", "", s)
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for k in _STEMS:
        if k in s:
            return k
    return s or "unknown"


def attach_label(obs, meta_path, label_key, image_key="image_name"):
    """Return an object array of `label_key` aligned to obs rows, joined from HCMA_metadata
    by image-filename stem. label_key='cell_state' is derived from cell_type via
    normalize_cell_state; any real metadata column is used verbatim. Unmatched rows -> None."""
    df = pd.read_csv(meta_path, encoding="latin-1")
    df["__stem"] = df["image_filename"].map(lambda s: os.path.splitext(str(s))[0])
    src_col = "cell_type" if label_key == "cell_state" else label_key
    if src_col not in df.columns:
        raise KeyError(f"metadata has no column {src_col!r}")
    vals = df[src_col].map(normalize_cell_state) if label_key == "cell_state" else df[src_col]
    m = pd.Series(vals.to_numpy(), index=df["__stem"]).groupby(level=0).first()
    lab = obs[image_key].astype(str).map(m)
    n_miss = int(lab.isna().sum())
    if n_miss:
        print(f"[labels] WARNING: {n_miss}/{len(lab)} cells have no {label_key} "
              "(image not in metadata); caller should drop them.")
    return lab.to_numpy()
