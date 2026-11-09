"""The committed class map (morph_classes.csv): it loads, uses only the 7 lineage classes plus
`unscored`, covers every known cell_state of the HCMA labels, is LF-only, and TASK.md names its classes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmarks.cell_state_HCMA.hcma_labels import normalize_cell_state
from benchmarks.full_HCMA.harness.score import CLASS_MAP, load_class_map

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CLASSES = ["endothelial", "epithelial", "immune", "melanocytic", "mesenchymal", "muscle", "neural"]


def test_the_committed_map_loads_with_the_seven_classes():
    record = load_class_map()
    assert record["ok"], record["problems"]
    assert record["classes"] == CLASSES
    assert set(record["map"].values()) == set(CLASSES) | {"unscored"}


def test_the_map_covers_every_known_cell_state():
    labels = pd.read_parquet(REPO / "benchmarks" / "cell_state_HCMA" / "results" / "hcma_labels.parquet")
    states = set(labels["cell_type"].map(normalize_cell_state)) - {"unknown"}
    assert states == set(load_class_map()["listed"])


def test_the_map_is_lf_only_and_task_names_its_classes():
    assert b"\r" not in Path(CLASS_MAP).read_bytes()
    task = (REPO / "benchmarks" / "full_HCMA" / "workspace_template" / "TASK.md").read_text(encoding="utf-8")
    assert all(name in task for name in CLASSES)
