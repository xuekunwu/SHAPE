"""Scoring of one arm (never shown to the agent; reads $ROOT/private).

Expected images = the run's prep `images.csv` ($ROOT/prep for prep_name `full`, $ROOT/prep_<name>
otherwise); labels from private/labels.csv; delivered cells = the chosen cluster call's cells and
their states at the chosen resolution. Cells of images labelled `unknown` are left out. A run has a
metadata level (`full` | `none`, workspace.blank_metadata); the comparison uses the control of the
same prep, metadata level and GPU model.

Labels at two levels: the coarse `morph_class` (primary; the versioned mapping
benchmarks/_shared/harness/morph_classes.csv, cell_state -> morph_class, its sha256 recorded; a missing
file, a cell_state of the prep without a row or an empty morph_class -> status NO_CLASS_MAP, nothing
scored) and the fine `cell_state`. The reserved morph_class `unscored` (cell_states of mixed lineage)
leaves its images out of the coarse level entirely (neither voters nor scored cells, not a class);
they stay in the fine level.

Primary endpoint `knn_transfer_macro_f1` (cell-level cross-micrograph kNN label transfer, labels =
morph_class; `_fine` = cell_state): <= 100 sampled cells per image (delivered and excluded), each
sampled delivered cell predicted by DINO's weighted vote (Caron et al. 2021, eval_knn.py) of its
k = 20 nearest sampled delivered cells (cosine) of other parent micrographs (VOTE_RULE,
knn_predict_levels); sampled excluded cells
and, per image with neither delivered nor excluded cells, 100 queries count as misses; macro F1 over
the labels, SE by a bootstrap over labels. At each level a label with fewer than 2 parent
micrographs among the expected images is unscorable there (UNSCORABLE_RULE; also for the
cluster-level transfer). The cluster-level transfer below is a secondary.

Secondary `transfer_macro_f1` (cross-micrograph cluster label transfer, after the
inter-dataset evaluation of Abdelaal et al. 2019, Genome Biology), labels = morph_class:
  - a cell of image i in cluster k is predicted the most frequent label among the cells of cluster k
    from OTHER parent micrographs (every panel of i's parent_id is left out; for the pairs prep,
    whose images all have different parents, this is "other images"); ties: the lexicographically
    smallest label; a cluster with no cell of another micrograph leaves the cell `unlabeled` (a miss
    for its label, a prediction of no label);
  - the cells the agent excluded (embed --exclude-cells, from the chain) count as unlabeled cells
    of their image's label: excluding cannot remove a recall miss;
  - per label of the prep's images: precision, recall and F1; a label without delivered cells has
    F1 = 0;
  - value = the mean F1 over the labels; SE: bootstrap over labels (2000 resamples, seed 0).
  `transfer_macro_f1_fine` (secondary): the same with labels = cell_state. `_delivered_only`
  variants leave the excluded cells out.
Secondary, at both label levels: transfer accuracy and median F1; majority accuracy (cluster
purity); NMI / ARI of the states; scIB-style NMI / ARI (the delivered embedding re-clustered by
cluster.py's code at resolutions 0.1 ... 2.0, the maximum NMI and the ARI there); the linear probe
`state_probe_balanced` (<= 100 cells per image, see sample_cells; the prep images of scorable.csv
with its fixed folds). Label-free: the unlabeled fraction, the number of clusters, the source
predictability of the embedding (<= 20 cells per image); coverage. A comparison with the control:
the paired per-class difference of the macro F1 (bootstrap over classes; the fine level's paired
difference reported too) and the difference of the transfer accuracy (bootstrap over images within
classes), only when the control scored the same images with the same class map.
"""
from __future__ import annotations

import json
import math
import posixpath
import re
import shlex
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ledger import (COMPLETED, TEMPLATE_TOOLS, _excluded_cells, calls, chain, input_calls, read_ledger, sha256,
                     state_columns, traced_tool_uses)
from .subset import all_preps, prep_dir, prep_problems

from benchmarks._shared import full_metrics as fm

REPO = Path(__file__).resolve().parents[3]

PRIMARY = "knn_transfer_macro_f1"       # the endpoint: cell-level cross-micrograph kNN label transfer
CLUSTER = "transfer_macro_f1"           # secondary: the cluster-level cross-micrograph transfer
KNN_K = 20
KNN_TEMPERATURE = 0.07
NO_PREDICTION = "no prediction"
UNLABELED = "unlabeled"
UNKNOWN = "unknown"
SCIB_RESOLUTIONS = tuple(round(0.1 * k, 1) for k in range(1, 21))
SCIB_NEIGHBORS = 15
SCIB_RULE = (f"the kNN primary's sampled delivered cells (<= 100 per image), one cosine kNN graph (n_neighbors "
             f"{SCIB_NEIGHBORS}, exact cosine neighbours, not the tool's approximate pynndescent graph; the analyzer's "
             "representation rule), Leiden at 0.1 ... 2.0, no UMAP; "
             "per level the maximum NMI and the ARI at that resolution")
ISOLATED_RULE = ("scIB isolated-label F1 (Luecken et al. 2022), secondary: isolated labels = the level's labels present "
                 "in the fewest sources among the expected images (batch = source); per label the best F1 of any "
                 "single cluster of the scIB-style re-clustering (the kNN sample's delivered cells, Leiden at 0.1 ... "
                 "2.0), 'cell in that cluster' predicting the label; isolated_label_f1 = the mean over the isolated "
                 "labels (a label without delivered cells: 0). The best-cluster F1 of every label is in "
                 "best_cluster_f1. Single-batch purity: it cannot tell a class from its micrograph")
NSC_RULE = ("image-level NSC / NSCB-style 1-NN (Ljosa 2013; Ando 2017), secondary: per expected image of a "
            "label scorable at this level with sampled delivered cells, its profile = the mean of those cells' "
            "L2-normalised embeddings; cosine 1-NN among the profiles of images of OTHER micrographs (NSC), also "
            "excluding images of the same source (NSCB); equal similarities: the lower image id; correct when the "
            "neighbour's label matches; a query whose label has no candidate image under the rule is removed from "
            "the denominator (Ando 2017)")
CROSS = "knn_transfer_cross_source_macro_f1"     # the hard guard of the verdict (A3.2; pinned 2026-10-05)
CROSS_RULE = ("the primary's sample and DINO-weighted vote at the morph_class level, but a cell's neighbours come "
              "from images of another SOURCE only (its own micrograph is within its source); a morph_class whose "
              "expected images span fewer than 2 sources has no cross-source partner: removed from the denominator "
              "and listed (Ando 2017, NSCB); excluded cells and missing images count as misses as in the primary")
REPLICATION_SHARE = 0.05
REPLICATION_RULE = ("at the chosen resolution, over the delivered cells: a state is replicated when >= 2 parent "
                    "micrographs each contribute >= 5 % of its cells; share = the delivered cells in replicated "
                    "states; effective micrographs of a state = exp(Shannon entropy, natural log, of its micrograph "
                    "shares), the median over states")
COMPOSITION_RULE = ("per expected image of the level's scored labels with delivered cells, its state-composition "
                    "vector (shares of its delivered cells per state at the chosen resolution); per image pair the "
                    "Jensen-Shannon similarity 1 - JS distance (base 2); AUC (rank-based, ties 0.5) of the pairs "
                    "'same label, other micrograph' against 'different label'; cross-source: positives also from "
                    "another source; images without delivered cells are left out and counted")
NEAR_MISS_RULE = ("among the fine-level kNN errors (a predicted cell_state other than the true one; no prediction "
                  "is not an error here), the share whose predicted cell_state has the true one's morph_class "
                  "(`unscored` is no class)")
REFERENCE_RULE = ("on the run's own kNN sample (the same images, sampled cells and delivered / excluded / missing "
                  "cells) the primary and the cross-source guard recomputed with reference embeddings in place of "
                  "the delivered one: `oracle` = one-hot of the cell's morph_class (an `unscored` cell: a zero vector "
                  "plus one fixed noise vector, seed 0, scale 1e-3), the best any embedding reaches with this "
                  "delivery (misses still count); `chance` = Gaussian vectors (seed 0, d = 64), an embedding without "
                  "information; normalized_primary = (primary - chance) / (oracle - chance), null when the "
                  "denominator <= 0")
STRATA = ("image_modality", "source", "magnification")
STRATA_RULE = ("secondary, no SE: the primary's predictions unchanged (neighbours from any stratum), the queries "
               "restricted to one stratum of the image's image_modality / source / magnification (the private "
               "labels or the prep's images.csv, full values under every metadata level); per stratum the macro F1 "
               "over the morph_classes present in its queries, misses counted as in the primary, and the accuracy")
PROXY_RULE = ("Spearman correlation over the candidates of one tool between each numeric field of the call's "
              "result.json summary (flattened; a cluster call's per-resolution fields matched to the row's "
              "resolution; fields present in >= 3 candidates and not constant) and the hidden score: the primary "
              "for embed / batch_correct calls, state_composition_auc for cluster rows; sorted by |rho|")
GAIN, LOSS, NO_RESOLVABLE = "GAIN", "LOSS", "NO_RESOLVABLE_DIFFERENCE"
SHORTCUT = "SOURCE_SHORTCUT_SUSPECTED"
VERDICT_RULE = (f"pinned 2026-10-05: {GAIN} only if the primary {PRIMARY} delta > 2 SE (bootstrap over classes) AND "
                f"the cross-source guard {CROSS} delta (point estimate, paired over the classes of both) >= 0; a "
                f"primary gain with a negative guard delta is {SHORTCUT}; {LOSS} if the primary delta < -2 SE; else "
                f"{NO_RESOLVABLE} (also when SE is 0); a control without the guard's per-class table: guard "
                "`unavailable`, the verdict from the primary alone, with a note")
PROBE = "state_probe_balanced"
MAX_CELLS = 100           # per image, linear probe
PROBE_CELLS = 20          # per image, source residue
SAMPLE_SEED = 0
SAMPLE_RULE = ("per image: its cells sorted by cell_id, permuted by numpy.random.default_rng(0).permutation "
               "(a fresh generator per image), the first N of that permutation; the N = 20 sample is the "
               "first 20 of the same permutation, a subset of the N = 100 sample")
N_BOOT = 2000
BOOT_SEED = 0
TYPE_BOOT_RULE = (f"paired cluster bootstrap over the fine cell_state types ({N_BOOT}, seed {BOOT_SEED}): the "
                  "types resampled with replacement with all their images' sampled cells and kNN predictions "
                  "(misses included), the morph_class macro F1 recomputed for both arms (over the classes present "
                  "in the resample); SE = sd, CI = the 2.5 / 97.5 percentiles; reported next to the class "
                  "bootstrap, which keeps the verdict")
FIGURE = "umap_scored.png"
CLASS_MAP = Path(__file__).resolve().parent / "morph_classes.csv"     # versioned; written by the master
NO_CLASS_MAP = "NO_CLASS_MAP"
UNSCORED = "unscored"     # reserved morph_class: left out of the coarse level, kept in the fine level
UNSCORABLE = "unscorable"     # a label with fewer than 2 parent micrographs among the expected images
UNSCORABLE_RULE = ("at each level a label with fewer than 2 distinct parent micrographs among the expected images "
                   "(the prep's, or the smoke's; never the delivered ones) is not scored there: its images are "
                   "neither queries nor references nor voters, and it is not in the macro F1")
VOTE = "dino_weighted"
VOTE_RULE = (f"DINO's weighted kNN vote (Caron et al. 2021, eval_knn.py): each of the k neighbours votes for its "
             f"label with weight exp(cosine similarity / {KNN_TEMPERATURE}), the weights summed per label")
NAME_NEEDLES = ("Human_Cell_Morphology_Atlas", "HCMA_metadata", "cell_state_HCMA", "hcma_labels",
                "_shared/harness", "_shared.harness", "shape.agent", "shape/agent",
                "labels.csv", "scorable.csv", "folds.csv", "PREPARE_PRIVATE", "SUBSET_", "orig_image_id", "private/",
                "FULL_HCMA_TOOL_STUBS", "allow_tool_stubs", "key_helper", ".claude-full-hcma", "api.sh",
                ".deadline", "TOOLS_SHA256", "DATA_SHA256", "readlink", "realpath", "morph_classes")
CLIMB = "a path climbing above the workspace"
WRITE = "a write into calls/, tools/, data/ or the run's logs/"
PROTECTED_DIRS = ("calls", "tools", "data")
WRITE_COMMANDS = {"rm", "mv", "cp", "ln", "rsync", "install", "chmod", "chown", "touch", "truncate", "tee",
                  "unlink", "rmdir", "mkdir", "dd"}
SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{16,}", r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",
    r"(?i)((?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{16,}")]
INPUT_CHARS = 300
AUDIT_LISTED = 5          # hits per needle shown in RESULT.md (all of them are in RESULT.json)


# ---------------------------------------------------------------- the prep and the private tables

def run_settings(run: Path) -> dict:
    """{"prep_name", "metadata", "prep", "settings_from"}: each from the workspace tools/_config.json,
    else RUN_META.json, else the default (`full` prep, `full` metadata: runs set up before preps and
    metadata levels were named). The expected images are read from $ROOT's prep of that name."""
    run = Path(run)
    found: dict[str, tuple] = {}
    for path, label in ((run / "workspace" / "tools" / "_config.json", "tools/_config.json"),
                        (run / "RUN_META.json", "RUN_META.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("prep_name", "metadata", "prep"):
            if isinstance(record, dict) and record.get(key) and key not in found:
                found[key] = (str(record[key]), label)
    defaults = {"prep_name": "full", "metadata": "full", "prep": None}
    return {**{k: found[k][0] if k in found else v for k, v in defaults.items()},
            "settings_from": {k: found[k][1] if k in found else "default" for k in defaults}}


def load_private(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    private = Path(root) / "private"
    labels = pd.read_csv(private / "labels.csv", dtype=str, keep_default_na=False)
    scorable = pd.read_csv(private / "scorable.csv", dtype=str, keep_default_na=False)
    scorable["fold"] = scorable["fold"].astype(int)
    folds = pd.read_csv(private / "folds.csv", dtype=str, keep_default_na=False)
    folds["fold"] = folds["fold"].astype(int)
    return labels, scorable, folds


def _prep_table(root: Path, prep_name: str) -> pd.DataFrame | None:
    """The prep's images.csv ($ROOT's copy, the full metadata), None when unreadable."""
    try:
        return pd.read_csv(prep_dir(root, prep_name) / "images.csv", dtype=str, keep_default_na=False)
    except (OSError, KeyError, ValueError):
        return None


def prep_images(root: Path, prep_name: str) -> list[str]:
    return pd.read_csv(prep_dir(root, prep_name) / "images.csv", dtype=str, keep_default_na=False)["image_id"].tolist()


# ---------------------------------------------------------------- the coarse class map

def load_class_map(path: Path | None = None) -> dict:
    """The versioned cell_state -> morph_class mapping (columns cell_state, morph_class, note):
    {"ok", "problems", "path", "sha256", "map", "classes" (without UNSCORED), "unscored_cell_states",
    "n_rows"}. A missing or unreadable file, a duplicated cell_state or an empty morph_class is a
    problem; nothing falls back silently. The reserved class UNSCORED marks cell_states of mixed
    lineage: their images are left out of the coarse level only."""
    path = Path(path or CLASS_MAP)
    out = {"ok": False, "problems": [], "path": str(path), "sha256": None, "map": {}, "classes": [], "n_rows": 0,
           "listed": [], "unscored_cell_states": []}
    if not path.is_file():
        out["problems"].append(f"no class map at {path} (cell_state,morph_class,note)")
        return out
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as error:
        out["problems"].append(f"{path} unreadable ({type(error).__name__}: {error})")
        return out
    out["sha256"], out["n_rows"] = sha256(path), int(len(table))
    missing = [c for c in ("cell_state", "morph_class", "note") if c not in table.columns]
    if missing:
        out["problems"].append(f"{path.name}: missing column(s) {missing}")
        return out
    table = table.assign(cell_state=table["cell_state"].str.strip(), morph_class=table["morph_class"].str.strip())
    duplicated = sorted(set(table.loc[table["cell_state"].duplicated(), "cell_state"]))
    empty = sorted(table.loc[table["morph_class"] == "", "cell_state"])
    if duplicated:
        out["problems"].append(f"{path.name}: cell_state listed more than once: {duplicated[:5]}")
    if empty:
        out["problems"].append(f"{path.name}: empty morph_class for {len(empty)} cell_state(s), e.g. {empty[:5]}")
    out["map"] = {s: c for s, c in zip(table["cell_state"], table["morph_class"]) if c}
    out["listed"] = sorted(set(table["cell_state"]))
    out["classes"] = sorted(set(out["map"].values()) - {UNSCORED})
    out["unscored_cell_states"] = sorted(s for s, c in out["map"].items() if c == UNSCORED)
    out["ok"] = not out["problems"]
    return out


def class_map_check(root: Path, prep_name: str, path: Path | None = None) -> dict:
    """load_class_map plus a row for every cell_state of the prep's images (`unknown` images are not
    scored and need none). Never raises; {"ok": False, "problems": [...]} -> status NO_CLASS_MAP."""
    check = load_class_map(path)
    try:
        labels = load_private(root)[0]
        states = set(labels.loc[labels["image_id"].isin(set(prep_images(root, prep_name))), "cell_state"]) - {UNKNOWN}
    except (OSError, KeyError, ValueError) as error:
        check["problems"].append(f"the prep's cell_states are unreadable ({type(error).__name__}: {error})")
        check["ok"] = False
        return check
    unmapped = sorted(states - set(check["listed"]))           # (rows with an empty class are reported above)
    if unmapped and check["sha256"]:
        check["problems"].append(f"{len(unmapped)} cell_state(s) of prep {prep_name} have no morph_class row, "
                                 f"e.g. {unmapped[:5]}")
    check["ok"] = not check["problems"]
    return check


# ---------------------------------------------------------------- the transfer endpoint

def _first_max(counts: np.ndarray) -> int:
    return int(np.argmax(counts))          # the first maximum: types are indexed in sorted order


def transfer_predictions(groups: np.ndarray, types: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Per cell: the most frequent type among its cluster's cells of other groups (the cell's parent
    micrograph; ties: the lexicographically smallest type), UNLABELED when the cluster has no cell
    of another group."""
    order = sorted(set(types))
    t_index = {t: n for n, t in enumerate(order)}
    frame = pd.DataFrame({"group": groups, "t": [t_index[t] for t in types], "state": states})
    totals = {k: np.bincount(g["t"].to_numpy(), minlength=len(order)) for k, g in frame.groupby("state")}
    pred_of: dict[tuple, str] = {}
    for (k, group), cells in frame.groupby(["state", "group"]):
        other = totals[k] - np.bincount(cells["t"].to_numpy(), minlength=len(order))
        pred_of[(k, group)] = order[_first_max(other)] if other.sum() else UNLABELED
    return np.array([pred_of[(k, g)] for k, g in zip(states, groups)], dtype=object)


def majority_accuracy(types: np.ndarray, states: np.ndarray) -> float | None:
    """Cluster purity: the share of cells whose type is their cluster's most frequent type."""
    if not len(types):
        return None
    frame = pd.DataFrame({"t": types, "state": states})
    correct = 0
    for _, group in frame.groupby("state"):
        counts = group["t"].value_counts()
        correct += int(counts.max())
    return correct / len(types)


def per_type_table(all_types: list[str], truth: np.ndarray, pred: np.ndarray, image_types: pd.Series,
                   n_cells: pd.Series, excluded: dict[str, int] | None = None, key: str = "cell_state",
                   missing: dict[str, int] | None = None) -> pd.DataFrame:
    """Per label (column `key`): precision, recall, F1 over the delivered cells plus `excluded`
    ({label: cells the agent excluded}) and, when given, `missing` ({label: the misses charged for
    images with neither delivered nor excluded cells}, column n_missing_queries): both are recall
    misses and predict nothing."""
    excluded = excluded or {}
    rows = []
    for c in all_types:
        lost = int(excluded.get(c, 0)) + int((missing or {}).get(c, 0))
        actual, predicted = int((truth == c).sum()) + lost, int((pred == c).sum())
        tp = int(((truth == c) & (pred == c)).sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        images = image_types[image_types == c].index
        row = {key: c, "n_images": int(len(images)),
               "n_images_with_cells": int((n_cells.reindex(images).fillna(0) > 0).sum()),
               "n_cells": actual - lost, "n_excluded": int(excluded.get(c, 0))}
        if missing is not None:
            row["n_missing_queries"] = int(missing.get(c, 0))
        rows.append({**row, "n_predicted": predicted, "n_true_positive": tp,
                     "n_unlabeled": int(((truth == c) & (pred == UNLABELED)).sum()) + lost,
                     "precision": precision, "recall": recall, "f1": f1})
    columns = [key, "n_images", "n_images_with_cells", "n_cells", "n_excluded",
               *(["n_missing_queries"] if missing is not None else []),
               "n_predicted", "n_true_positive", "n_unlabeled", "precision", "recall", "f1"]
    return pd.DataFrame(rows, columns=columns)


def images_sha256(images: list[str]) -> str:
    """The identity of an expected image set: sha256 of its sorted image ids, one per line."""
    import hashlib

    return hashlib.sha256("\n".join(sorted(images)).encode()).hexdigest()


def type_bootstrap(values: np.ndarray, n: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Bootstrap distribution of a mean over types (types resampled with replacement)."""
    values = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(n, len(values)))
    return values[idx].mean(axis=1)


def _se(boot: np.ndarray | None) -> float | None:
    if boot is None or len(boot) < 2:
        return None
    return 0.0 if np.ptp(boot) <= 1e-12 else float(np.std(boot, ddof=1))     # identical resamples (float noise): 0


# ---------------------------------------------------------------- re-clustering (scIB-style NMI / ARI)

class ExactCosineNeighbors:
    """scanpy's kNN transformer interface (get_params()["n_neighbors"], fit_transform): exact cosine
    neighbours by the kNN primary's code (_top: order (-similarity, index)); per cell itself first
    (distance 0), then its n_neighbors - 1 nearest other cells; distance = 1 - cosine similarity."""

    def __init__(self, n_neighbors: int):
        self.n_neighbors = int(n_neighbors)

    def get_params(self, deep: bool = True) -> dict:
        return {"n_neighbors": self.n_neighbors}

    def fit_transform(self, X, y=None):
        import torch
        from scipy import sparse

        v = _unit_rows(np.asarray(X))
        n, kk = len(v), self.n_neighbors - 1
        idx, sims = np.zeros((n, kk), dtype=np.int64), np.zeros((n, kk), dtype=np.float32)
        with torch.no_grad():
            for q in _query_blocks(n):
                sim = v[q] @ v.T
                sim.masked_fill_(torch.isnan(sim), -np.inf)
                own = torch.arange(q.start, q.stop)
                sim[own - q.start, own] = -np.inf                       # itself: the first column below
                idx[q], sims[q] = _top(sim, kk)
        indices = np.hstack([np.arange(n)[:, None], idx])
        distances = np.hstack([np.zeros((n, 1), dtype=np.float32), np.clip(1.0 - sims, 0.0, None)])
        indptr = np.arange(0, n * (kk + 1) + 1, kk + 1)
        return sparse.csr_matrix((distances.ravel(), indices.ravel(), indptr), shape=(n, n))


def recluster(x: np.ndarray, cell_ids: list[str], resolutions=SCIB_RESOLUTIONS,
              n_neighbors: int = SCIB_NEIGHBORS, timings: dict | None = None) -> dict[float, np.ndarray]:
    """Leiden states of `x` at each resolution with the cluster tool's graph settings (the analyzer's
    _compute_umap_and_clustering: its representation rule, a cosine kNN graph with n_neighbors capped
    below the number of cells, scanpy's UMAP connectivities and Leiden), the graph built once with
    exact cosine neighbours (ExactCosineNeighbors; the tool's pynndescent graph is approximate above
    4096 cells, so this is not the tool's own graph there), no UMAP. `timings`, when given, gets the
    seconds of the graph and of the Leiden runs."""
    import anndata as ad
    import scanpy as sc

    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    started = time.time()
    adata = ad.AnnData(X=np.asarray(x, dtype=np.float32))
    adata.obs_names = [str(c) for c in cell_ids]
    representation = Cell_State_Analyzer_Tool()._resolve_clustering_representation(adata, None)
    if representation["representation"] == "pca":
        sc.pp.pca(adata, n_comps=representation["n_pcs"])
    neighbors = min(max(2, int(n_neighbors)), max(2, adata.n_obs - 1))
    sc.pp.neighbors(adata, use_rep="X_pca" if representation["representation"] == "pca" else "X",
                    n_neighbors=neighbors, transformer=ExactCosineNeighbors(neighbors))
    graph = time.time()
    out = {}
    for resolution in resolutions:
        sc.tl.leiden(adata, resolution=resolution, key_added="scib")
        out[float(resolution)] = adata.obs["scib"].astype(str).to_numpy()
    if timings is not None:
        timings.update(graph=round(graph - started, 2), leiden=round(time.time() - graph, 2))
    return out


def scib_nmi_ari(x: np.ndarray, cell_ids: list[str], truths: dict[str, np.ndarray]) -> dict:
    """scIB-style NMI / ARI per label level ({level: per-cell labels, None = not scored at that
    level}): the delivered embedding re-clustered once (recluster, SCIB_RESOLUTIONS); per level, on
    its cells, the maximum NMI, the ARI at that resolution and the resolution (the lowest among
    equal maxima)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    usable = {}
    for level, truth in truths.items():          # None = the cell is not scored at this level
        truth = np.asarray(truth, dtype=object)
        mask = np.array([t is not None for t in truth], dtype=bool)
        if mask.sum() >= 3 and len(set(truth[mask])) >= 2:
            usable[level] = (mask, truth[mask])
    out: dict = {k: {"nmi_max": None, "ari_at_nmi_max": None, "resolution": None,
                     "note": "fewer than 3 cells or 2 labels delivered"} for k in truths if k not in usable}
    out["n_neighbors"], out["per_resolution"], out["n_cells"] = SCIB_NEIGHBORS, [], int(len(cell_ids))
    out["seconds"] = {}
    if not usable:
        return out
    clusters = recluster(np.asarray(x, dtype=np.float32), cell_ids, timings=out["seconds"])
    for resolution, states in sorted(clusters.items()):
        entry = {"resolution": resolution, "n_clusters": int(len(set(states)))}
        for level, (mask, truth) in usable.items():
            entry[f"nmi_{level}"] = float(normalized_mutual_info_score(truth, states[mask]))
            entry[f"ari_{level}"] = float(adjusted_rand_score(truth, states[mask]))
        out["per_resolution"].append(entry)
    for level in usable:
        best = max(out["per_resolution"], key=lambda e: e[f"nmi_{level}"])
        out[level] = {"nmi_max": best[f"nmi_{level}"], "ari_at_nmi_max": best[f"ari_{level}"],
                      "resolution": best["resolution"]}
    out["best_cluster_f1"] = {level: best_cluster_f1(truth, {r: c[mask] for r, c in clusters.items()})
                              for level, (mask, truth) in usable.items()}
    return out


def best_cluster_f1(truth: np.ndarray, clusterings: dict[float, np.ndarray]) -> pd.DataFrame:
    """Per label: the best F1 of any single cluster of any clustering ({resolution: per-cell
    cluster}), "cell in that cluster" predicting the label; F1 = 2 tp / (cluster size + label size).
    Ties: the lowest resolution, then the first cluster in sorted order. Columns label, n_cells,
    best_f1, resolution, cluster."""
    truth = np.asarray(truth, dtype=object)
    labels = sorted(set(truth))
    size = pd.Series(truth).value_counts()
    best = {c: (0.0, None, None) for c in labels}
    for resolution in sorted(clusterings):
        table = pd.crosstab(pd.Series(truth, name="label"), pd.Series(np.asarray(clusterings[resolution]), name="cluster"))
        table = table.reindex(columns=sorted(table.columns, key=str))
        f1 = 2 * table.to_numpy(float) / (table.sum(axis=0).to_numpy(float)[None, :] + size[table.index].to_numpy(float)[:, None])
        for row, label in enumerate(table.index):
            j = int(np.argmax(f1[row]))
            if f1[row, j] > best[label][0]:
                best[label] = (float(f1[row, j]), float(resolution), str(table.columns[j]))
    return pd.DataFrame([{"label": c, "n_cells": int(size[c]), "best_f1": best[c][0], "resolution": best[c][1],
                          "cluster": best[c][2]} for c in labels],
                        columns=["label", "n_cells", "best_f1", "resolution", "cluster"])


def image_nsc(profiles: np.ndarray, labels: np.ndarray, parents: np.ndarray, sources: np.ndarray) -> dict:
    """NSC_RULE on image profiles (rows L2-normalised, sorted by image id): {"nsc", "nscb"}: accuracy,
    n_evaluated, n_removed, n_correct."""
    labels, parents, sources = (np.asarray(a, dtype=object) for a in (labels, parents, sources))
    n = len(labels)
    sim = profiles @ profiles.T if n else np.zeros((0, 0))
    same_label = labels[:, None] == labels[None, :]
    out = {}
    for rule, excluded in (("nsc", parents[:, None] == parents[None, :]),
                           ("nscb", (parents[:, None] == parents[None, :]) | (sources[:, None] == sources[None, :]))):
        candidate = ~excluded
        evaluated = (candidate & same_label).any(axis=1)
        nearest = np.argmax(np.where(candidate, sim, -np.inf), axis=1) if n else np.zeros(0, dtype=int)
        correct = evaluated & (labels[nearest] == labels) if n else np.zeros(0, dtype=bool)
        out[rule] = {"accuracy": float(correct.sum() / evaluated.sum()) if evaluated.any() else None,
                     "n_evaluated": int(evaluated.sum()), "n_removed": int(n - evaluated.sum()),
                     "n_correct": int(correct.sum())}
    return out


# ---------------------------------------------------------------- the linear probe and the source residue

def sample_cells(cells: pd.DataFrame, max_cells: int = MAX_CELLS) -> np.ndarray:
    """Row indices of <= max_cells cells per image (SAMPLE_RULE): an image's sample depends only on
    its own cells, and a smaller max_cells gives a subset of a larger one's sample."""
    order = cells.assign(_row=np.arange(len(cells))).sort_values(["image_id", "cell_id"], kind="stable")
    picked = []
    for _, group in order.groupby("image_id", sort=True):
        rows = group["_row"].to_numpy()
        if len(rows) > max_cells:
            rows = rows[np.sort(np.random.default_rng(SAMPLE_SEED).permutation(len(rows))[:max_cells])]
        picked.append(rows)
    return np.concatenate(picked) if picked else np.zeros(0, dtype=int)


def _fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """StandardScaler + class-balanced logistic regression; a constant with one training class,
    no prediction (None) without training cells."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y_train)
    if len(classes) < 2:
        return np.full(len(x_test), classes[0] if len(classes) else None, dtype=object)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    model.fit(scaler.transform(x_train), y_train)
    return model.predict(scaler.transform(x_test)).astype(object)


def class_bootstrap(values: np.ndarray, classes: np.ndarray, n: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Bootstrap distribution of the class-balanced mean: images resampled within each class."""
    rng = np.random.default_rng(seed)
    values, classes = np.asarray(values, dtype=float), np.asarray(classes)
    means = []
    for c in sorted(set(classes)):
        v = values[classes == c]
        means.append(v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1))
    return np.mean(means, axis=0)


def _balanced(frame: pd.DataFrame) -> float | None:
    return float(frame.groupby("cell_state")["accuracy"].mean().mean()) if len(frame) else None


def state_probe(s_images: np.ndarray, xs: np.ndarray, n_cells: pd.Series, labels: pd.DataFrame,
                scorable: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """The linear probe (secondary): class-balanced image-level accuracy of a cell_state logistic
    regression with the fixed folds of scorable.csv. `s_images`, `xs`: image id and embedding of
    each sampled cell; `n_cells`: delivered cells per image; `labels`: the images to report."""
    fold_of = scorable.set_index("image_id")["fold"]
    class_of = scorable.set_index("image_id")["cell_state"]
    folds_of_class = scorable.groupby("cell_state")["fold"].agg(set)
    unscorable = {i for i, c, f in zip(scorable["image_id"], scorable["cell_state"], scorable["fold"])
                  if not folds_of_class[c] - {f}}
    images = pd.Series(s_images, dtype=object)
    in_set = images.isin(fold_of.index).to_numpy()
    cell_fold = images.map(fold_of).to_numpy()
    cell_class = images.map(class_of).to_numpy(dtype=object)
    tested = in_set & ~images.isin(unscorable).to_numpy()
    pred = np.full(len(images), None, dtype=object)
    for f in sorted(scorable["fold"].unique()):
        train, test = in_set & (cell_fold != f), tested & (cell_fold == f)
        if test.any():
            pred[test] = _fit_predict(xs[train], cell_class[train], xs[test])
    correct = pd.Series((pred == cell_class).astype(float)).where(tested)

    table = labels[["image_id", "cell_state"]].copy()
    table["n_sampled"] = table["image_id"].map(images.value_counts()).fillna(0).astype(int)
    table["scorable"] = table["image_id"].isin(fold_of.index) & ~table["image_id"].isin(unscorable)
    accuracy = correct.groupby(s_images).mean()
    table["accuracy"] = np.where(table["scorable"], table["image_id"].map(accuracy).fillna(0.0), np.nan)
    scored = table[table["scorable"]]
    boot = class_bootstrap(scored["accuracy"], scored["cell_state"]) if len(scored) else None
    covered = scored[scored["image_id"].map(n_cells).fillna(0) > 0]
    summary = {"value": _balanced(scored), "se": _se(boot), "covered_only": _balanced(covered),
               "n_scored_images": int(len(scored)), "n_unscorable_images": len(unscorable),
               "n_classes": int(scored["cell_state"].nunique()), "max_cells_per_image": MAX_CELLS,
               "n_predicted_cells": int(tested.sum()),
               "note": "secondary: the prep's images that are in scorable.csv, its fixed folds; an image "
                       "whose class has no scorable image of the prep in another fold is unscorable"}
    return summary, table[["image_id", "n_sampled", "accuracy", "scorable"]]


def source_residue(s_images: np.ndarray, xs: np.ndarray, labels: pd.DataFrame, folds: pd.DataFrame) -> dict | None:
    """Balanced accuracy of a logistic regression predicting `source` from the embedding, the prep
    folds of folds.csv (by parent micrograph); lower = less vendor signal."""
    from sklearn.metrics import balanced_accuracy_score

    if not len(s_images):
        return None
    meta = labels.set_index("image_id").loc[s_images, ["parent_id", "source"]]
    fold = meta["parent_id"].map(folds.set_index("parent_id")["fold"]).fillna(-1).astype(int).to_numpy()
    source = meta["source"].to_numpy(dtype=object)
    pred = np.full(len(source), None, dtype=object)
    for f in sorted(set(fold)):
        pred[fold == f] = _fit_predict(xs[fold != f], source[fold != f], xs[fold == f])
    n_sources = int(len(set(source)))
    return {"balanced_accuracy": float(balanced_accuracy_score(source.astype(str), pred.astype(str))),
            "chance": 1.0 / n_sources, "n_sources": n_sources, "n_cells": int(len(source)),
            "max_cells_per_image": PROBE_CELLS,
            "note": "source and class are confounded in HCMA; reported, not an endpoint"}


# ---------------------------------------------------------------- one delivery

KNN_ROWS_BUDGET = 2.5e7      # query rows x cells per similarity block (float32: ~100 MB)


def _unit_rows(x: np.ndarray):
    """The rows of `x` as a float32 torch tensor scaled to unit length (once)."""
    import torch

    v = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _query_blocks(n: int) -> list[slice]:
    chunk = max(64, min(4096, int(KNN_ROWS_BUDGET // max(n, 1))))
    return [slice(s, min(s + chunk, n)) for s in range(0, n, chunk)]


def _top(sim, kk: int) -> tuple[np.ndarray, np.ndarray]:
    """The kk largest entries of each row of `sim` (torch, rows x n; -inf = never), ordered by
    (-similarity, column index): torch.topk, then a row whose kk-th value is also held by entries
    outside the top kk takes the lowest column indices among the equal values."""
    import torch

    vals, idx = torch.topk(sim, kk, dim=1)
    thr = vals[:, -1:]
    fix = ((sim == thr).sum(dim=1) > (vals == thr).sum(dim=1)) & torch.isfinite(thr[:, 0])
    if bool(fix.any()):
        rows = torch.nonzero(fix).flatten()
        s, t = sim[rows], thr[rows]
        greater, equal = s > t, s == t
        keep = greater | (equal & (torch.cumsum(equal, dim=1) <= kk - greater.sum(dim=1, keepdim=True)))
        cols = torch.nonzero(keep)[:, 1].reshape(len(rows), kk)           # row-major: ascending columns
        idx[rows], vals[rows] = cols, torch.gather(s, 1, cols)
    vals, idx = vals.numpy(), idx.numpy()
    order = np.lexsort((idx, -vals), axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


def knn_predict_levels(x: np.ndarray, groups: np.ndarray, levels: dict[str, np.ndarray],
                       k: int = KNN_K, temperature: float = KNN_TEMPERATURE) -> dict[str, np.ndarray]:
    """DINO-weighted kNN label transfer (Caron et al. 2021, eval_knn.py) at several label levels in
    one pass over the queries. `levels`: {name: per-cell label, None = the cell is neither a query
    nor a reference at that level}. Per level and cell q: its k nearest cells of the level by cosine
    similarity among the cells of OTHER groups (micrographs); fewer than k such cells: all of them;
    none: UNLABELED (no prediction); None outside the level. Each neighbour votes for its label with
    weight exp(similarity / temperature), summed per label; the prediction is the label with the
    largest summed weight, a tie goes to the label of the nearest neighbour among the tied labels.
    Neighbour order: (-cosine similarity, index), i.e. exactly
    equal similarities (as computed) are ordered by the lower input row. Exact cosine on rows scaled
    to unit length once (float32, torch on CPU), the similarity of a block of queries computed once
    for all levels; a NaN similarity is never a neighbour."""
    import torch

    n = len(groups)
    out = {name: np.full(n, None, dtype=object) for name in levels}
    if n == 0:
        return out
    prepared = {}
    for name, labels in levels.items():
        labels = np.asarray(labels, dtype=object)
        member = np.array([label is not None for label in labels], dtype=bool)
        order = np.array(sorted(set(labels[member])), dtype=object)
        codes = np.zeros(n, dtype=int)
        codes[member] = np.searchsorted(order, labels[member])
        out[name][member] = UNLABELED
        if member.any():
            prepared[name] = (member, torch.from_numpy(member), order, codes, bool(member.all()))
    group = torch.from_numpy(pd.factorize(pd.Series(list(groups), dtype=object))[0])
    v = _unit_rows(x)
    kk = min(k, n)
    with torch.no_grad():
        for q in _query_blocks(n):
            sim = v[q] @ v.T
            sim.masked_fill_(torch.isnan(sim), -np.inf)
            sim.masked_fill_(group[q, None] == group[None, :], -np.inf)     # the cell's own micrograph never votes
            for name, (member, member_t, order, codes, everyone) in prepared.items():
                queries = np.flatnonzero(member[q])
                if not len(queries):
                    continue
                block = sim if everyone else sim[torch.from_numpy(queries)].masked_fill(~member_t[None, :], -np.inf)
                idx, s = _top(block, kk)
                valid = np.isfinite(s)
                neighbour = codes[idx]
                votes = np.zeros((len(neighbour), len(order)), dtype=np.float64)
                rows = np.broadcast_to(np.arange(len(neighbour))[:, None], neighbour.shape)
                weight = np.exp(np.where(valid, s, 0.0).astype(np.float64) / temperature)
                np.add.at(votes, (rows[valid], neighbour[valid]), weight[valid])       # nearest first: a fixed order
                tied = votes == votes.max(axis=1, keepdims=True)
                hit = np.take_along_axis(tied, neighbour, axis=1) & valid       # neighbours carrying a tied label
                first = np.argmax(hit, axis=1)                                  # the nearest of them
                chosen = order[neighbour[np.arange(len(neighbour)), first]]
                out[name][q.start + queries] = np.where(valid.any(axis=1), chosen, UNLABELED)
    return out


def knn_predict(x: np.ndarray, groups: np.ndarray, labels: np.ndarray, k: int = KNN_K,
                temperature: float = KNN_TEMPERATURE) -> np.ndarray:
    """knn_predict_levels at one level that holds every cell."""
    return knn_predict_levels(x, groups, {"label": labels}, k, temperature)["label"]


def _knn_level(column: str, images_of: pd.DataFrame, sample: pd.DataFrame, n_cells: pd.Series,
               pred_column: str | None = None) -> dict:
    """The kNN transfer at one level over the sampled cells (`sample`: image_id, row = the delivered
    cell's row in the embedding, -1 for an excluded cell, `pred_<column>` = its kNN prediction) of
    the images of `images_of` (the level's scored images). Sampled delivered cells are the queries;
    sampled excluded cells are misses; an image of the level with neither delivered nor excluded
    cells counts as MAX_CELLS (the per-image sample cap) misses. The delivered-only value leaves
    both kinds of misses out."""
    label_of = images_of.set_index("image_id")[column]
    mine = sample[sample["image_id"].isin(label_of.index)]
    delivered, excluded = mine[mine["row"] >= 0], mine[mine["row"] < 0]
    rows = delivered["row"].to_numpy(dtype=int)
    truth = delivered["image_id"].map(label_of).to_numpy(dtype=object)
    pred = delivered[pred_column or f"pred_{column}"].to_numpy(dtype=object)
    missing = sorted(set(label_of.index) - set(mine["image_id"]))
    misses = excluded["image_id"].map(label_of).value_counts().to_dict()
    missing_misses = (label_of.loc[missing].value_counts() * MAX_CELLS).to_dict() if missing else {}
    names = sorted(set(images_of[column]))
    table = per_type_table(names, truth, pred, label_of, n_cells, misses, key=column, missing=missing_misses)
    only = per_type_table(names, truth, pred, label_of, n_cells, key=column)["f1"].to_numpy(float)
    f1 = table["f1"].to_numpy(float)
    boot = type_bootstrap(f1) if len(f1) else None
    n_missing = MAX_CELLS * len(missing)
    n_scored = len(truth) + len(excluded) + n_missing
    correct = pred == truth
    image = delivered["image_id"].to_numpy(dtype=object)
    scored_per_image = pd.concat([delivered["image_id"], excluded["image_id"]]).value_counts()
    if missing:
        lost = pd.Series(MAX_CELLS, index=missing, dtype=int)
        scored_per_image = pd.concat([scored_per_image, lost]) if len(scored_per_image) else lost
    return {"rows": rows, "truth": truth, "pred": pred, "table": table,
            "value": float(f1.mean()) if len(f1) else None, "se": _se(boot),
            "median_f1": float(np.median(f1)) if len(f1) else None,
            "delivered_only": float(only.mean()) if len(only) else None,
            "accuracy": float(correct.sum() / n_scored) if n_scored else None,
            "n_queries": int(len(truth)), "n_excluded_sampled": int(len(excluded)),
            "n_no_prediction": int((pred == UNLABELED).sum()),
            "n_images_missing": int(len(missing)), "n_missing_queries": int(n_missing),
            "query_images": image, "excluded_images": excluded["image_id"].to_numpy(dtype=object),
            "missing_images": missing,
            "correct_per_image": pd.Series(correct.astype(int)).groupby(image).sum(),
            "scored_per_image": scored_per_image}


def scorable_images(images_of: pd.DataFrame, column: str) -> tuple[pd.DataFrame, list[dict]]:
    """(the images of the labels scorable at this level, the unscorable labels [{label, n_images,
    n_micrographs}]) by UNSCORABLE_RULE over `images_of` (expected images: image_id, parent_id,
    `column`)."""
    n_parents = images_of.groupby(column)["parent_id"].nunique()
    off = sorted(n_parents[n_parents < 2].index)
    n_images = images_of[column].value_counts()
    listed = [{column: c, "n_images": int(n_images[c]), "n_micrographs": int(n_parents[c])} for c in off]
    return images_of[~images_of[column].isin(off)], listed


def _level(column: str, known: pd.DataFrame, cells: pd.DataFrame, parents: np.ndarray, states: np.ndarray,
           n_cells: pd.Series, n_excl: pd.Series) -> dict:
    """The transfer endpoint at one label level (`morph_class` or `cell_state`) over the images of
    `known` and their `cells` (row-aligned with `parents` and `states`)."""
    label_of = known.set_index("image_id")[column]
    truth = cells["image_id"].map(label_of).to_numpy(dtype=object)
    pred = transfer_predictions(parents, truth, states)
    names = sorted(set(known[column]))
    n_excl = n_excl[n_excl.index.isin(label_of.index)]
    by_label = n_excl.groupby(n_excl.index.map(label_of)).sum().to_dict() if len(n_excl) else {}
    table = per_type_table(names, truth, pred, label_of, n_cells, by_label, key=column)
    delivered = per_type_table(names, truth, pred, label_of, n_cells, key=column)["f1"].to_numpy(float)
    f1 = table["f1"].to_numpy(float)
    boot = type_bootstrap(f1) if len(f1) else None
    return {"truth": truth, "pred": pred, "states": states, "table": table, "label_of": label_of,
            "n_excluded": int(n_excl.sum()) if len(n_excl) else 0,
            "value": float(f1.mean()) if len(f1) else None, "se": _se(boot),
            "delivered_only": float(delivered.mean()) if len(delivered) else None,
            "median_f1": float(np.median(f1)) if len(f1) else None}


def cross_source_images(images_of: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """(the images of the morph_classes whose expected images span >= 2 sources, the others
    [{morph_class, n_images, n_sources}]) over `images_of` (the coarse level's scored images)."""
    n_sources = images_of.groupby("morph_class")["source"].nunique()
    off = sorted(n_sources[n_sources < 2].index)
    n_images = images_of["morph_class"].value_counts()
    listed = [{"morph_class": c, "n_images": int(n_images[c]), "n_sources": int(n_sources[c])} for c in off]
    return images_of[~images_of["morph_class"].isin(off)], listed


def expected_sets(root: Path, images: list[str], class_map: dict[str, str]) -> dict:
    """The private tables and the expected image sets of every level: expected, known (no `unknown`
    type), coarse_labelled (no `unscored`), coarse_images / fine_images (scorable, UNSCORABLE_RULE),
    cross_images (CROSS_RULE) and the listed unscorable labels."""
    labels, scorable, folds = load_private(root)
    expected = labels[labels["image_id"].isin(set(images))].reset_index(drop=True)      # (inputs_check reports others)
    expected["morph_class"] = expected["cell_state"].map(class_map)
    known = expected[expected["cell_state"] != UNKNOWN]
    coarse_labelled = known[known["morph_class"] != UNSCORED]
    coarse_images, unscorable_coarse = scorable_images(coarse_labelled, "morph_class")
    fine_images, unscorable_fine = scorable_images(known, "cell_state")
    cross_images, unscorable_cross = cross_source_images(coarse_images)
    return {"labels": labels, "scorable": scorable, "folds": folds, "expected": expected, "known": known,
            "coarse_labelled": coarse_labelled, "coarse_images": coarse_images, "unscorable_coarse": unscorable_coarse,
            "fine_images": fine_images, "unscorable_fine": unscorable_fine, "cross_images": cross_images,
            "unscorable_cross": unscorable_cross}


def excluded_frame(excluded: list[str] | None, known: pd.DataFrame) -> pd.DataFrame:
    """cell_id, image_id of the excluded cells of known-type expected images (<image_id>_<mask call>_<label>)."""
    frame = pd.DataFrame({"cell_id": pd.Series(sorted(set(excluded or [])), dtype=object)})
    frame["image_id"] = [c.rsplit("_", 2)[0] for c in frame["cell_id"]]
    return frame[frame["image_id"].isin(set(known["image_id"]))]


def knn_all(sets: dict, cells: pd.DataFrame, x: np.ndarray, excluded_cells: pd.DataFrame, n_cells: pd.Series,
            fine: bool = True) -> dict:
    """The kNN sample (<= MAX_CELLS per image of delivered and excluded cells) and the kNN transfer
    on it: the primary (morph_class) and the fine level in one pass with the micrograph as the
    group, then the cross-source guard with the source as the group. {"sample", "coarse", "fine",
    "cross"}."""
    expected = sets["expected"]
    union = pd.concat([cells.assign(row=np.arange(len(cells))), excluded_cells.assign(row=-1)], ignore_index=True)
    sample = union.iloc[sample_cells(union, MAX_CELLS)].reset_index(drop=True)
    queries = sample[sample["row"] >= 0]
    rows = queries["row"].to_numpy(dtype=int)
    def label(images_of: pd.DataFrame, column: str) -> np.ndarray:      # None outside the level's images
        values = queries["image_id"].map(images_of.set_index("image_id")[column])
        return np.array([None if pd.isna(v) else v for v in values], dtype=object)
    predicted = knn_predict_levels(
        x[rows], queries["image_id"].map(expected.set_index("image_id")["parent_id"]).to_numpy(dtype=object),
        {"morph_class": label(sets["coarse_images"], "morph_class"),
         **({"cell_state": label(sets["fine_images"], "cell_state")} if fine else {})})
    predicted["cross_source"] = knn_predict_levels(
        x[rows], queries["image_id"].map(expected.set_index("image_id")["source"]).to_numpy(dtype=object),
        {"cross_source": label(sets["cross_images"], "morph_class")})["cross_source"]
    for column, values in predicted.items():
        sample[f"pred_{column}"] = None
        sample.loc[queries.index, f"pred_{column}"] = values
    return {"sample": sample, "coarse": _knn_level("morph_class", sets["coarse_images"], sample, n_cells),
            "fine": _knn_level("cell_state", sets["fine_images"], sample, n_cells) if fine else None,
            "cross": _knn_level("morph_class", sets["cross_images"], sample, n_cells, pred_column="pred_cross_source")}


def state_replication(states: np.ndarray, parents: np.ndarray, min_share: float = REPLICATION_SHARE) -> dict:
    """REPLICATION_RULE over cells (their states and parent micrographs)."""
    frame = pd.DataFrame({"state": np.asarray(states, dtype=object), "parent": np.asarray(parents, dtype=object)})
    if not len(frame):
        return {"share_cells_in_replicated_states": None, "median_effective_micrographs": None, "n_states": 0,
                "n_replicated_states": 0, "min_share": min_share, "rule": REPLICATION_RULE}
    counts = frame.groupby(["state", "parent"]).size()
    replicated, effective, size = {}, [], counts.groupby(level=0).sum()
    for state, per in counts.groupby(level=0):
        shares = per.to_numpy(float) / per.sum()
        replicated[state] = int((shares >= min_share).sum()) >= 2
        effective.append(float(np.exp(-(shares * np.log(shares)).sum())))
    in_replicated = sum(int(size[k]) for k, v in replicated.items() if v)
    return {"share_cells_in_replicated_states": in_replicated / len(frame),
            "median_effective_micrographs": float(np.median(effective)), "n_states": int(len(size)),
            "n_replicated_states": int(sum(replicated.values())), "min_share": min_share, "rule": REPLICATION_RULE}


def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    if not len(pos) or not len(neg):
        return None
    ranks = rankdata(np.concatenate([pos, neg]))           # average ranks: ties count 0.5
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def js_similarity(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """1 - the Jensen-Shannon distance (base 2) between the distribution `p` and each row of `q`."""
    p = np.asarray(p, dtype=float)[None, :]
    q = np.atleast_2d(np.asarray(q, dtype=float))
    m = (p + q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log2(p / m), 0.0).sum(axis=1)
        kl_q = np.where(q > 0, q * np.log2(q / m), 0.0).sum(axis=1)
    return 1.0 - np.sqrt(np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0))


def state_composition_auc(image_ids: np.ndarray, states: np.ndarray, images_of: pd.DataFrame, column: str,
                          cross_source: bool = False) -> dict:
    """COMPOSITION_RULE: `image_ids` / `states` per delivered cell; `images_of` (image_id, `column`,
    parent_id, source) the level's scored expected images."""
    frame = pd.DataFrame({"image_id": np.asarray(image_ids, dtype=object), "state": np.asarray(states, dtype=object)})
    frame = frame[frame["image_id"].isin(set(images_of["image_id"]))]
    meta = images_of.set_index("image_id").sort_index()
    used = meta[meta.index.isin(set(frame["image_id"]))]
    out = {"auc": None, "n_images": int(len(used)), "n_images_left_out": int(len(meta) - len(used)),
           "n_positive_pairs": 0, "n_negative_pairs": 0, "cross_source": cross_source, "rule": COMPOSITION_RULE}
    if len(used) < 2:
        return out
    comp = pd.crosstab(frame["image_id"], frame["state"]).reindex(used.index).to_numpy(float)
    comp = comp / comp.sum(axis=1, keepdims=True)
    label, parent, source = (used[c].to_numpy(dtype=object) for c in (column, "parent_id", "source"))
    pos, neg = [], []
    for i in range(len(comp) - 1):
        similarity = js_similarity(comp[i], comp[i + 1:])
        same = label[i + 1:] == label[i]
        positive = same & (parent[i + 1:] != parent[i])
        if cross_source:
            positive &= source[i + 1:] != source[i]
        pos.append(similarity[positive])
        neg.append(similarity[~same])
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    return {**out, "auc": _rank_auc(pos, neg), "n_positive_pairs": int(len(pos)), "n_negative_pairs": int(len(neg))}


def near_miss(truth: np.ndarray, pred: np.ndarray, class_map: dict[str, str]) -> dict:
    """NEAR_MISS_RULE over the fine-level kNN queries."""
    truth, pred = np.asarray(truth, dtype=object), np.asarray(pred, dtype=object)
    errors = (pred != truth) & (pred != UNLABELED)
    true_class = np.array([class_map.get(t) for t in truth[errors]], dtype=object)
    pred_class = np.array([class_map.get(t) for t in pred[errors]], dtype=object)
    near = (true_class == pred_class) & (true_class != UNSCORED) & np.array([c is not None for c in true_class], dtype=bool)
    return {"share": float(near.mean()) if errors.any() else None, "n_errors": int(errors.sum()),
            "n_near_miss": int(near.sum()), "rule": NEAR_MISS_RULE}


def knn_rows(level: dict, class_of: pd.Series) -> pd.DataFrame:
    """A kNN level's scored queries (image_id, morph_class, predicted, n): the delivered queries, the
    sampled excluded cells and MAX_CELLS per missing image (both predicted NO_PREDICTION)."""
    images = np.concatenate([np.asarray(level["query_images"], dtype=object),
                             np.asarray(level["excluded_images"], dtype=object),
                             np.asarray(level["missing_images"], dtype=object)])
    predicted = np.concatenate([np.where(level["pred"] == UNLABELED, NO_PREDICTION, level["pred"]).astype(object),
                                np.full(len(level["excluded_images"]) + len(level["missing_images"]), NO_PREDICTION,
                                        dtype=object)])
    weight = np.concatenate([np.ones(len(level["query_images"]) + len(level["excluded_images"]), dtype=int),
                             np.full(len(level["missing_images"]), MAX_CELLS, dtype=int)])
    return pd.DataFrame({"image_id": images, "morph_class": pd.Series(images, dtype=object).map(class_of).to_numpy(),
                         "predicted": predicted, "n": weight})


def macro_f1_of_rows(rows: pd.DataFrame) -> tuple[float | None, float | None, list[str]]:
    """(macro F1 over the classes present in the rows' truth, accuracy, the classes) of knn_rows."""
    classes = sorted(set(rows.loc[rows["n"] > 0, "morph_class"].dropna()))
    total = int(rows["n"].sum())
    if not classes:
        return None, None, classes
    f1 = []
    for c in classes:
        tp = rows.loc[(rows["morph_class"] == c) & (rows["predicted"] == c), "n"].sum()
        both = rows.loc[rows["predicted"] == c, "n"].sum() + rows.loc[rows["morph_class"] == c, "n"].sum()
        f1.append(2 * tp / both if both else 0.0)
    correct = rows.loc[rows["morph_class"] == rows["predicted"], "n"].sum()
    return float(np.mean(f1)), float(correct / total) if total else None, classes


def strata_table(rows: pd.DataFrame, images_of: pd.DataFrame, meta: dict[str, pd.Series]) -> pd.DataFrame:
    """STRATA_RULE: per stratum column of `meta` ({column: image_id -> value}) and value."""
    out = []
    for column, value_of in meta.items():
        value = rows["image_id"].map(value_of).fillna("").astype(str)
        image_values = images_of["image_id"].map(value_of).fillna("").astype(str)
        for v in sorted(set(image_values) | set(value)):
            mine = rows[value == v]
            f1, accuracy, classes = macro_f1_of_rows(mine)
            delivered = mine["predicted"] != NO_PREDICTION
            out.append({"column": column, "value": v, "n_images": int((image_values == v).sum()),
                        "n_queries": int(mine["n"].sum()), "n_misses_and_no_prediction": int(mine.loc[~delivered, "n"].sum()),
                        "n_classes": len(classes), "classes": ";".join(classes), "macro_f1": f1, "accuracy": accuracy})
    return pd.DataFrame(out, columns=["column", "value", "n_images", "n_queries", "n_misses_and_no_prediction",
                                      "n_classes", "classes", "macro_f1", "accuracy"])


def reference_scores(sets: dict, cells: pd.DataFrame, excluded_cells: pd.DataFrame, n_cells: pd.Series,
                     primary: float | None) -> dict:
    """REFERENCE_RULE: the oracle and chance embeddings on the delivery's own sample."""
    class_of = sets["known"].set_index("image_id")["morph_class"]
    classes = sorted(set(sets["coarse_labelled"]["morph_class"]))
    truth = cells["image_id"].map(class_of).to_numpy(dtype=object)
    oracle = np.zeros((len(cells), max(1, len(classes))), dtype=np.float32)
    index = {c: n for n, c in enumerate(classes)}
    for row, c in enumerate(truth):
        if c in index:
            oracle[row, index[c]] = 1.0
    blank = ~np.isin(truth, classes)
    oracle[blank] += 1e-3 * np.random.default_rng(0).normal(size=oracle.shape[1]).astype(np.float32)
    chance = np.random.default_rng(0).normal(size=(len(cells), 64)).astype(np.float32)
    out = {}
    for name, x in (("oracle", oracle), ("chance", chance)):
        knn = knn_all(sets, cells, x, excluded_cells, n_cells, fine=False)
        out[name] = {"primary": knn["coarse"]["value"], "cross_source": knn["cross"]["value"]}
    o, c = out["oracle"]["primary"], out["chance"]["primary"]
    denominator = None if o is None or c is None else o - c
    out["normalized_primary"] = (float((primary - c) / denominator)
                                 if primary is not None and denominator is not None and denominator > 0 else None)
    out["rule"] = REFERENCE_RULE
    return out


def knn_type_counts(level: dict, type_of: pd.Series, class_of: pd.Series) -> pd.DataFrame:
    """The morph_class kNN level as counts (cell_state, morph_class, predicted, n): the delivered
    queries, the sampled excluded cells and MAX_CELLS per missing image (both predicted
    NO_PREDICTION). The unit of the fine-type bootstrap."""
    images = np.concatenate([np.asarray(level["query_images"], dtype=object),
                             np.asarray(level["excluded_images"], dtype=object),
                             np.asarray(level["missing_images"], dtype=object)])
    predicted = np.concatenate([np.where(level["pred"] == UNLABELED, NO_PREDICTION, level["pred"]).astype(object),
                                np.full(len(level["excluded_images"]) + len(level["missing_images"]), NO_PREDICTION,
                                        dtype=object)])
    weight = np.concatenate([np.ones(len(level["query_images"]) + len(level["excluded_images"]), dtype=int),
                             np.full(len(level["missing_images"]), MAX_CELLS, dtype=int)])
    frame = pd.DataFrame({"cell_state": pd.Series(images, dtype=object).map(type_of),
                          "morph_class": pd.Series(images, dtype=object).map(class_of),
                          "predicted": predicted, "n": weight})
    return frame.groupby(["cell_state", "morph_class", "predicted"], as_index=False)["n"].sum()


FULL_RULE = ("benchmarks/_shared/FULL_METRICS_CONTRACT.md: label = morph_class, unit = image (the coarse level's "
             "scorable images, also those without delivered cells: misses); main split = grouped folds by parent_id "
             "(no micrograph on both sides), guard = grouped folds by source over the classes spanning >= 2 sources; "
             "each the fewest StratifiedGroupKFold folds (shuffle, random_state 0) that keep every class of a fold's "
             f"test side on its train side (2 folds: every class on both sides); <= {fm.CELL_CAP} cells per image in "
             "the cell-level probes, sampled from the delivered and the agent-excluded cells (cap_cells = "
             "SAMPLE_RULE); a sampled excluded cell is a miss, as in knn_transfer; unit profile = the mean of the "
             "L2-normalised delivered cells, L2-normalised")


def full_layout(sets: dict) -> tuple[fm.Layout, pd.DataFrame]:
    """The full_metrics layout over the coarse level's scorable images (sorted by image_id)."""
    units = sets["coarse_images"].sort_values("image_id").reset_index(drop=True)
    labels = units["morph_class"].to_numpy(dtype=str)
    main, n_main = fm.grouped_folds(labels, units["parent_id"].to_numpy(dtype=str))
    cross = np.flatnonzero(units["image_id"].isin(set(sets["cross_images"]["image_id"])).to_numpy())
    guard, n_guard = [], 0
    if len(cross):
        folds, n_guard = fm.grouped_folds(labels[cross], units["source"].to_numpy(dtype=str)[cross])
        guard = [[(cross[train], cross[test]) for train, test in fold] for fold in folds]
    describe = {"main": {"rule": f"cross-micrograph: {n_main} grouped folds by parent_id (the fewest that keep every "
                                 "class of a fold's test side on its train side)", "n_folds_chosen": n_main},
                "guard": {"rule": f"cross-source: {n_guard} grouped folds by source over the morph_classes spanning >= "
                                  "2 sources" if guard else "unavailable: no morph_class spans >= 2 sources",
                          "n_folds_chosen": n_guard,
                          "unscorable_classes": [c["morph_class"] for c in sets["unscorable_cross"]]}}
    return fm.Layout(units["image_id"].tolist(), labels.tolist(), {"main": main, "guard": guard}, describe), units


def full_metrics_reading(sets: dict, cells: pd.DataFrame, x: np.ndarray,
                         excluded_cells: pd.DataFrame | None = None) -> dict:
    """{"layout", "reading", "record"} of the full metrics for one delivery (record = RESULT.json `full_metrics`
    without a control); {"record": {"status": "unavailable", ...}} when no layout can be built."""
    try:
        layout, units = full_layout(sets)
    except ValueError as error:
        return {"layout": None, "reading": None, "record": {"status": "unavailable", "reason": str(error)}}
    position = pd.Series(np.arange(len(units)), index=units["image_id"].to_numpy())
    cell_unit = cells["image_id"].map(position).fillna(-1).astype(np.int64).to_numpy()
    excluded = excluded_cells if excluded_cells is not None else pd.DataFrame({"cell_id": [], "image_id": []})
    reading = fm.evaluate(layout, x, cell_unit, cell_ids=cells["cell_id"].to_numpy(),
                          excluded_unit=excluded["image_id"].map(position).fillna(-1).astype(np.int64).to_numpy(),
                          excluded_ids=excluded["cell_id"].astype(str).to_numpy())
    return {"layout": layout, "reading": reading, "record": full_record(layout, reading, None)}


def full_record(layout: fm.Layout, reading: dict, control: dict | None) -> dict:
    return {"status": "OK", **fm.record(layout, reading, control,
                                        settings={"label": "morph_class (scorer only)", "unit": "image",
                                                  "rule": FULL_RULE})}


def evaluate(cells: pd.DataFrame, x: np.ndarray, states: np.ndarray, root: Path, images: list[str],
             excluded: list[str] | None = None, *, class_map: dict[str, str],
             strata_meta: pd.DataFrame | None = None) -> tuple[dict, dict]:
    """Score one delivery against the expected `images` (the prep's) at both label levels: the
    coarse morph_class (primary; `class_map` = {cell_state: morph_class}, see class_map_check) and
    the fine cell_state. Images whose cell_state maps to UNSCORED are left out of the coarse level
    (neither voters, references nor scored cells, no class) and stay in the fine level. At each
    level a label with fewer than 2 parent micrographs among the expected images (never decided
    from the delivery) is unscorable there (UNSCORABLE_RULE): its images are left out of that
    level's kNN and cluster-level transfer the same way and it is listed. `cells`
    (cell_id, image_id) is row-aligned with the embedding `x` and the delivered `states`;
    `excluded`: the cell ids the agent excluded (the chain's excluded_cells; <image_id>_<mask
    call>_<label>). Returns (result, tables): per_class / per_type (the kNN primary at morph_class /
    cell_state), per_class_cluster / per_type_cluster (the cluster-level transfer), per_image and
    cells (per cell: both labels, the state, the predictions and the statuses)."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    started = time.time()
    timings: dict = {}
    clock = [started]

    def lap(name: str) -> None:
        now = time.time()
        timings[name], clock[0] = round(now - clock[0], 2), now

    sets = expected_sets(root, images, class_map)
    labels, scorable, folds, expected = sets["labels"], sets["scorable"], sets["folds"], sets["expected"]
    cells = cells[["cell_id", "image_id"]].astype(str).reset_index(drop=True)
    states = np.asarray(states, dtype=object).astype(str)
    if cells["cell_id"].duplicated().any():
        raise ValueError("the delivery lists a cell twice")
    if len(x) != len(cells) or len(states) != len(cells):
        raise ValueError(f"{len(cells)} cells, {len(x)} embedding rows, {len(states)} states")
    outside = sorted(set(cells["image_id"]) - set(expected["image_id"]))
    if outside:
        raise ValueError(f"{len(outside)} delivered image ids are not images of the prep, e.g. {outside[:3]}")
    type_of = expected.set_index("image_id")["cell_state"]
    keep = (cells["image_id"].map(type_of) != UNKNOWN).to_numpy()
    cells, x, states = cells[keep].reset_index(drop=True), x[np.flatnonzero(keep)], states[keep]
    known = expected[expected["cell_state"] != UNKNOWN]
    unmapped = sorted(set(known.loc[known["morph_class"].isna(), "cell_state"]))
    if unmapped:                     # class_map_check runs before; this guards direct calls
        raise ValueError(f"{len(unmapped)} cell_state(s) without a morph_class, e.g. {unmapped[:3]}")
    img = cells["image_id"].to_numpy(dtype=object)
    parents = cells["image_id"].map(expected.set_index("image_id")["parent_id"]).to_numpy(dtype=object)
    n_cells = cells["image_id"].value_counts()
    excluded_cells = excluded_frame(excluded, known)
    n_excl = excluded_cells["image_id"].value_counts().astype(int)
    coarse_labelled, coarse_images, fine_images = sets["coarse_labelled"], sets["coarse_images"], sets["fine_images"]
    unscorable_coarse, unscorable_fine = sets["unscorable_coarse"], sets["unscorable_fine"]
    lap("setup")
    in_coarse = cells["image_id"].isin(set(coarse_images["image_id"])).to_numpy()
    in_fine = cells["image_id"].isin(set(fine_images["image_id"])).to_numpy()
    coarse = _level("morph_class", coarse_images, cells[in_coarse].reset_index(drop=True), parents[in_coarse],
                    states[in_coarse], n_cells, n_excl)
    fine = _level("cell_state", fine_images, cells[in_fine].reset_index(drop=True), parents[in_fine],
                  states[in_fine], n_cells, n_excl)
    # per cell over all cells of known type: both labels; a level's prediction is empty where the
    # cell's label is not scored at that level (unscored or unscorable)
    coarse_truth = cells["image_id"].map(known.set_index("image_id")["morph_class"]).to_numpy(dtype=object)
    fine_truth = cells["image_id"].map(type_of).to_numpy(dtype=object)
    coarse_pred, fine_pred = np.full(len(img), "", dtype=object), np.full(len(img), "", dtype=object)
    coarse_pred[in_coarse], fine_pred[in_fine] = coarse["pred"], fine["pred"]
    coarse_correct = in_coarse & (coarse_pred == coarse_truth)
    fine_correct = in_fine & (fine_pred == fine_truth)
    off_coarse = np.where(coarse_truth == UNSCORED, UNSCORED, UNSCORABLE).astype(object)
    status = np.where(~in_coarse, off_coarse, np.where(coarse_pred == UNLABELED, "unlabeled",
                                                       np.where(coarse_correct, "correct", "wrong")))
    cell_table = pd.DataFrame({
        "cell_id": cells["cell_id"], "image_id": img, "cell_state": fine_truth, "morph_class": coarse_truth,
        "state": states, "predicted": coarse_pred, "predicted_fine": fine_pred, "status": status,
        "source": cells["image_id"].map(expected.set_index("image_id")["source"]).to_numpy()})
    per_image = expected[["image_id", "cell_state", "morph_class", "source"]].copy()
    per_image["n_cells"] = per_image["image_id"].map(n_cells).fillna(0).astype(int)
    per_image["n_excluded"] = per_image["image_id"].map(n_excl).fillna(0).astype(int)
    count = lambda flags: per_image["image_id"].map(pd.Series(flags).groupby(img).sum()).fillna(0).astype(int)  # noqa: E731
    scored_coarse = per_image["image_id"].isin(set(coarse_images["image_id"]))
    scored_fine = per_image["image_id"].isin(set(fine_images["image_id"]))
    per_image["n_unlabeled"] = np.where(scored_coarse, count((coarse_pred == UNLABELED).astype(int))
                                        + per_image["n_excluded"], 0)
    per_image["n_unlabeled_fine"] = np.where(scored_fine, count((fine_pred == UNLABELED).astype(int))
                                             + per_image["n_excluded"], 0)
    per_image["n_correct"] = count(coarse_correct.astype(int))
    per_image["n_correct_fine"] = count(fine_correct.astype(int))
    scored = (per_image["n_cells"] + per_image["n_excluded"]).clip(lower=1)
    has = per_image["n_cells"] + per_image["n_excluded"] > 0
    per_image["transfer_accuracy"] = np.where(has & scored_coarse, per_image["n_correct"] / scored, np.nan)
    per_image["transfer_accuracy_fine"] = np.where(has & scored_fine, per_image["n_correct_fine"] / scored, np.nan)
    lap("cluster_transfer")

    sampled, small = sample_cells(cells, MAX_CELLS), sample_cells(cells, PROBE_CELLS)
    xs = np.asarray(x[sampled], dtype=np.float64)
    probes = {}
    for images_of, column, suffix in ((coarse_labelled, "morph_class", ""), (known, "cell_state", "_fine")):
        label_of = images_of.set_index("image_id")[column]
        in_prep = scorable[scorable["image_id"].isin(images_of["image_id"])]
        probe, table = state_probe(img[sampled], xs, n_cells, images_of.assign(cell_state=images_of[column]),
                                   in_prep.assign(cell_state=in_prep["image_id"].map(label_of)))
        probes[column] = probe
        table = table.rename(columns={"accuracy": f"probe_accuracy{suffix}", "scorable": f"probe_scorable{suffix}",
                                      "n_sampled": "probe_n_sampled"})
        per_image = per_image.merge(table.drop(columns="probe_n_sampled"), on="image_id", how="left")
        per_image[f"probe_scorable{suffix}"] = per_image[f"probe_scorable{suffix}"].fillna(False).astype(bool)
    per_image["probe_n_sampled"] = per_image["image_id"].map(pd.Series(img[sampled]).value_counts()).fillna(0).astype(int)
    lap("linear_probe")
    # the kNN primary's sample (<= MAX_CELLS per image, delivered and excluded cells); the scIB-style
    # re-clustering runs on its delivered cells
    knn = knn_all(sets, cells, x, excluded_cells, n_cells)
    knn_sample, knn_coarse, knn_fine, knn_cross = knn["sample"], knn["coarse"], knn["fine"], knn["cross"]
    lap("knn")
    # the state level at the chosen resolution (A5.2, A5.3) and the fine-level near misses (B2)
    state_level = {"replication": state_replication(states, parents),
                   "composition": {"morph_class": state_composition_auc(img, states, coarse_images, "morph_class"),
                                   "cell_state": state_composition_auc(img, states, fine_images, "cell_state"),
                                   "morph_class_cross_source": state_composition_auc(img, states, coarse_images,
                                                                                     "morph_class", cross_source=True)}}
    misses = near_miss(knn_fine["truth"], knn_fine["pred"], class_map)
    type_counts = knn_type_counts(knn_coarse, type_of, known.set_index("image_id")["morph_class"])
    type_counts_cross = knn_type_counts(knn_cross, type_of, known.set_index("image_id")["morph_class"])
    lap("state_level")
    reference = reference_scores(sets, cells, excluded_cells, n_cells, knn_coarse["value"])
    lap("reference")
    stratum_of = {}
    for column in STRATA:          # the private labels first, else the prep's images.csv (never the blanked copy)
        if column in expected.columns:
            stratum_of[column] = expected.set_index("image_id")[column]
        elif strata_meta is not None and column in strata_meta.columns:
            stratum_of[column] = strata_meta.astype({"image_id": str}).drop_duplicates("image_id").set_index("image_id")[column]
    strata = strata_table(knn_rows(knn_coarse, known.set_index("image_id")["morph_class"]), coarse_images, stratum_of)
    lap("strata")
    scib_rows = knn_sample.loc[knn_sample["row"] >= 0, "row"].to_numpy(dtype=int)
    scib = scib_nmi_ari(x[scib_rows], cells["cell_id"].to_numpy()[scib_rows].tolist(),
                        {"morph_class": np.where(coarse_truth != UNSCORED, coarse_truth, None)[scib_rows],
                         "cell_state": fine_truth[scib_rows]})
    lap("scib")
    # scIB isolated-label F1 and the best-cluster F1 of every label (secondary)
    best_tables, isolated = [], {}
    for images_of, column in ((coarse_labelled, "morph_class"), (known, "cell_state")):
        n_sources = images_of.groupby(column)["source"].nunique()
        found = scib.get("best_cluster_f1", {}).get(column)
        base = found if found is not None else pd.DataFrame(columns=["label", "n_cells", "best_f1", "resolution", "cluster"])
        table = pd.DataFrame({"label": sorted(n_sources.index)}).merge(base, on="label", how="left")
        table["n_cells"] = table["n_cells"].astype(float).fillna(0).astype(int)
        table["best_f1"] = table["best_f1"].astype(float).fillna(0.0)      # a label without delivered cells: 0
        table["n_sources"] = table["label"].map(n_sources).astype(int)
        table["isolated"] = table["n_sources"] == (int(n_sources.min()) if len(n_sources) else 0)
        table.insert(0, "level", column)
        best_tables.append(table)
        chosen = table[table["isolated"]]
        isolated[column] = {"value": float(chosen["best_f1"].mean()) if found is not None and len(chosen) else None,
                            "isolated_labels": chosen["label"].tolist(),
                            "n_sources_min": int(n_sources.min()) if len(n_sources) else None, "rule": ISOLATED_RULE,
                            "best_cluster_f1": table.drop(columns="level").to_dict(orient="records")}
    best_cluster = pd.concat(best_tables, ignore_index=True)
    # image-level NSC / NSCB-style 1-NN on mean profiles of the sampled delivered cells (secondary)
    sampled_rows = knn_sample[knn_sample["row"] >= 0]
    unit = np.asarray(x[sampled_rows["row"].to_numpy(dtype=int)], dtype=np.float64)
    unit = unit / np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12)
    means = pd.DataFrame(unit).groupby(sampled_rows["image_id"].to_numpy()).mean()
    nsc = {}
    for images_of, column in ((coarse_images, "morph_class"), (fine_images, "cell_state")):
        meta = images_of.set_index("image_id").sort_index()
        meta = meta[meta.index.isin(means.index)]
        profile = means.loc[meta.index].to_numpy() if len(meta) else np.zeros((0, unit.shape[1]))
        profile = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
        found = image_nsc(profile, meta[column].to_numpy(), meta["parent_id"].to_numpy(), meta["source"].to_numpy())
        nsc[column] = {"image_nsc_accuracy": found["nsc"]["accuracy"], "image_nscb_accuracy": found["nscb"]["accuracy"],
                       "image_nsc": {**found, "n_images": int(len(meta)), "rule": NSC_RULE}}
    lap("isolated_label_and_nsc")

    def secondary_of(level: dict, column: str) -> dict:
        n = len(level["truth"])
        n_scored = n + level["n_excluded"]
        correct = int((level["pred"] == level["truth"]).sum())
        return {f"{CLUSTER}": level["value"], f"{CLUSTER}_se": level["se"],
                f"{CLUSTER}_delivered_only": level["delivered_only"],
                "transfer_accuracy": correct / n_scored if n_scored else None,
                "transfer_median_f1": level["median_f1"],
                "unlabeled_fraction": float(((level["pred"] == UNLABELED).sum() + level["n_excluded"]) / n_scored)
                if n_scored else None,
                "majority_accuracy": majority_accuracy(level["truth"], level["states"]),
                "states_vs_label": {"NMI": float(normalized_mutual_info_score(level["truth"], level["states"])),
                                    "ARI": float(adjusted_rand_score(level["truth"], level["states"]))} if n else None,
                "scib": {**scib.get(column, {}), "n_cells": scib.get("n_cells"), "sampled": True, "rule": SCIB_RULE},
                "linear_probe": probes[column], "n_cells": n,
                "isolated_label_f1": isolated[column]["value"], "isolated_label": isolated[column], **nsc[column]}

    knn_status = np.where(in_coarse, "not sampled", off_coarse).astype(object)
    knn_status[knn_coarse["rows"]] = np.where(knn_coarse["pred"] == UNLABELED, NO_PREDICTION,
                                              np.where(knn_coarse["pred"] == knn_coarse["truth"], "correct", "wrong"))
    cell_table["knn_status"] = knn_status
    for level, suffix in ((knn_coarse, ""), (knn_fine, "_fine")):
        per_image[f"n_knn_scored{suffix}"] = per_image["image_id"].map(level["scored_per_image"]).fillna(0).astype(int)
        per_image[f"n_knn_correct{suffix}"] = per_image["image_id"].map(level["correct_per_image"]).fillna(0).astype(int)
        per_image[f"knn_accuracy{suffix}"] = np.where(per_image[f"n_knn_scored{suffix}"] > 0, per_image[f"n_knn_correct{suffix}"]
                                                      / per_image[f"n_knn_scored{suffix}"].clip(lower=1), np.nan)
    knn_rule = (f"<= {MAX_CELLS} cells per image (delivered and excluded, SAMPLE_RULE); each sampled delivered cell is "
                f"given the {{label}} with the largest weighted vote among its k = {KNN_K} nearest sampled "
                "delivered cells by cosine distance among the cells of other parent micrographs (" + VOTE_RULE + "; "
                "fewer: all; none: no prediction; a tie: the label of the nearest tied neighbour; equal similarities: "
                "the lower sample position); {label}s unscorable at this level are neither queries nor references "
                "(" + UNSCORABLE_RULE + "); a sampled excluded cell is a "
                f"miss; an image with neither delivered nor excluded cells counts as {MAX_CELLS} misses (the per-image "
                "sample cap); macro F1 over the prep's {label}s, a {label} without delivered cells has F1 0; SE: "
                f"bootstrap over {{label}}s ({N_BOOT}, seed {BOOT_SEED})")

    def knn_block(level: dict, label: str, names: str) -> dict:
        table = level["table"]
        return {"value": level["value"], "se": level["se"], "level": label, "k": KNN_K,
                "max_cells_per_image": MAX_CELLS, "vote": VOTE, "temperature": KNN_TEMPERATURE,
                "median_f1": level["median_f1"], f"n_{names}": int(len(table)),
                f"n_{names}_without_cells": int((table["n_cells"] == 0).sum()),
                "accuracy": level["accuracy"], "delivered_only": level["delivered_only"],
                "n_queries": level["n_queries"], "n_excluded_sampled": level["n_excluded_sampled"],
                "n_no_prediction": level["n_no_prediction"], "n_images_missing": level["n_images_missing"],
                "n_missing_queries": level["n_missing_queries"],
                f"per_{'class' if label == 'morph_class' else 'type'}": table.to_dict(orient="records"),
                "rule": knn_rule.format(label=label)}

    counts = per_image["n_cells"]
    unscored_images = known[known["morph_class"] == UNSCORED]
    rule = ("a cell is predicted the most frequent {label} of its cluster's cells from other parent micrographs "
            "(ties: the lexicographically smallest; none: unlabeled); the cells the agent excluded are unlabeled "
            "cells of their image's {label}; {label}s unscorable at this level are left out (" + UNSCORABLE_RULE
            + "); macro F1 over the prep's {label}s, a {label} without delivered cells "
            f"has F1 0; SE: bootstrap over {{label}}s ({N_BOOT}, seed {BOOT_SEED})")
    coarse_secondary = secondary_of(coarse, "morph_class")
    residue = source_residue(img[small], np.asarray(x[small], dtype=np.float64), labels, folds)
    lap("source_residue")
    full = full_metrics_reading(sets, cells, x, excluded_cells)
    lap("full_metrics")
    result = {
        "endpoint": PRIMARY, "endpoint_level": "morph_class",
        PRIMARY: knn_coarse["value"], f"{PRIMARY}_se": knn_coarse["se"],
        f"{PRIMARY}_delivered_only": knn_coarse["delivered_only"],
        f"{PRIMARY}_fine": knn_fine["value"], f"{PRIMARY}_fine_se": knn_fine["se"],
        f"{PRIMARY}_fine_delivered_only": knn_fine["delivered_only"],
        "knn_transfer_median_f1": knn_coarse["median_f1"], "knn_transfer_median_f1_fine": knn_fine["median_f1"],
        "knn_transfer_accuracy": knn_coarse["accuracy"], "knn_transfer_accuracy_fine": knn_fine["accuracy"],
        CLUSTER: coarse["value"], f"{CLUSTER}_fine": fine["value"],
        "expected_images_sha256": images_sha256(list(expected["image_id"])), "n_expected_images": int(len(expected)),
        "unscorable_classes": {"morph_class": unscorable_coarse, "cell_state": unscorable_fine,
                               "rule": UNSCORABLE_RULE},
        "primary": {**knn_block(knn_coarse, "morph_class", "classes"),
                    "note": f"images of a cell_state mapped to `{UNSCORED}` are neither queries nor references here"},
        "fine": knn_block(knn_fine, "cell_state", "types"),
        CROSS: knn_cross["value"], f"{CROSS}_se": knn_cross["se"],
        "cross_source": {**{k: v for k, v in knn_block(knn_cross, "morph_class", "classes").items() if k != "rule"},
                         "unscorable_classes": sets["unscorable_cross"], "rule": CROSS_RULE},
        "reference": reference,
        "strata": {**{column: strata[strata["column"] == column].drop(columns="column").to_dict(orient="records")
                      for column in stratum_of},
                   "absent_columns": [c for c in STRATA if c not in stratum_of], "level": "morph_class",
                   "rule": STRATA_RULE},
        "cluster_transfer": {
            "morph_class": {"value": coarse["value"], "se": coarse["se"], "delivered_only": coarse["delivered_only"],
                            "n_classes": int(len(coarse["table"])),
                            "n_excluded_cells_counted_unlabeled": coarse["n_excluded"],
                            "per_class": coarse["table"].to_dict(orient="records"),
                            "rule": rule.format(label="morph_class") + f"; images of a cell_state mapped to "
                                    f"`{UNSCORED}` are left out of this level (neither voters nor scored cells)"},
            "cell_state": {"value": fine["value"], "se": fine["se"], "delivered_only": fine["delivered_only"],
                           "n_types": int(len(fine["table"])), "n_excluded_cells_counted_unlabeled": fine["n_excluded"],
                           "per_type": fine["table"].to_dict(orient="records"), "rule": rule.format(label="cell_state")}},
        "secondary": {
            "unlabeled_fraction": coarse_secondary["unlabeled_fraction"],
            "n_clusters": int(len(set(states))),
            "source_residue": residue,
            "morph_class": coarse_secondary, "cell_state": secondary_of(fine, "cell_state"),
            "scib_per_resolution": scib.get("per_resolution", []), "scib_n_neighbors": scib.get("n_neighbors"),
            "scib_n_cells": scib.get("n_cells"), "scib_sampled": True, "scib_rule": SCIB_RULE,
            "state_replication": state_level["replication"], "state_composition": state_level["composition"],
            "near_miss": misses},
        "coverage": {"n_images": int(len(expected)), "n_images_known_type": int(len(known)),
                     "n_images_with_cells": int((counts > 0).sum()),
                     "share_images_with_cells": float((counts > 0).mean()) if len(counts) else None,
                     "n_cells": int(len(img)), "n_cells_unknown_type_left_out": int((~keep).sum()),
                     "n_images_unscored_coarse": int(len(unscored_images)),
                     "n_cells_unscored_coarse": int((coarse_truth == UNSCORED).sum()),
                     "n_images_unscorable": {"morph_class": int(sum(c["n_images"] for c in unscorable_coarse)),
                                             "cell_state": int(sum(c["n_images"] for c in unscorable_fine))},
                     "unscored_cell_states_in_prep": sorted(set(unscored_images["cell_state"])),
                     "n_cells_excluded_by_the_agent": int(n_excl.sum()) if len(n_excl) else 0,
                     "cells_per_image": {"median": float(np.median(counts)) if len(counts) else None,
                                         "p10": float(np.percentile(counts, 10)) if len(counts) else None,
                                         "p90": float(np.percentile(counts, 90)) if len(counts) else None,
                                         "note": "over the prep's images, images without cells counted as 0"}},
        "sampling": {"rule": SAMPLE_RULE, "seed": SAMPLE_SEED,
                     "knn": {"max_cells_per_image": MAX_CELLS, "k": KNN_K, "n_cells": int(len(knn_sample)),
                             "n_excluded_cells_sampled": int((knn_sample["row"] < 0).sum())},
                     "linear_probe": {"max_cells_per_image": MAX_CELLS, "n_cells": int(len(sampled))},
                     "source_residue": {"max_cells_per_image": PROBE_CELLS, "n_cells": int(len(small))}},
        "embedding_dim": int(x.shape[1]) if np.ndim(x) == 2 else None,
        "full_metrics": full["record"]}
    result["scoring_seconds"] = round(time.time() - started, 1)
    result["timings"] = {**timings, "scib_graph": scib["seconds"].get("graph"), "scib_leiden": scib["seconds"].get("leiden"),
                         "note": "seconds per scoring component (scib = scib_graph + scib_leiden + imports and the NMI / ARI)"}
    return result, {"per_class": knn_coarse["table"], "per_type": knn_fine["table"], "per_class_cluster": coarse["table"],
                    "per_type_cluster": fine["table"], "per_image": per_image, "cells": cell_table,
                    "best_cluster_f1": best_cluster, "per_class_cross_source": knn_cross["table"],
                    "knn_type_counts": type_counts, "knn_type_counts_cross_source": type_counts_cross,
                    "strata": strata, "full_metrics": full if full.get("reading") is not None else None}


# ---------------------------------------------------------------- the comparison with the control

def _paired(mine: pd.DataFrame, control: pd.DataFrame, key: str) -> tuple[dict, pd.DataFrame]:
    """The macro-F1 difference over the labels of both tables, SE of the paired per-label difference."""
    both = mine.merge(control[[key, "f1"]], on=key, suffixes=("", "_control"))
    out = {"n": int(len(both)), "n_only_this_arm": int(len(set(mine[key]) - set(both[key]))),
           "n_only_control": int(len(set(control[key]) - set(both[key])))}
    if not len(both):
        return {**out, "value": None, "control_value": None, "delta": None, "se": None}, both
    diff = both["f1"].to_numpy(float) - both["f1_control"].to_numpy(float)
    return {**out, "value": float(both["f1"].mean()), "control_value": float(both["f1_control"].mean()),
            "delta": float(diff.mean()), "se": _se(type_bootstrap(diff))}, both[[key, "n_cells", "f1", "f1_control"]]


def _type_matrices(counts: pd.DataFrame, types: list[str], classes: list[str]) -> tuple[np.ndarray, ...]:
    """Per type (rows, `types`) and class (columns): predictions of the class, cells of the class,
    true positives -- from a knn_type_counts table."""
    t_index, c_index = {t: n for n, t in enumerate(types)}, {c: n for n, c in enumerate(classes)}
    predicted, actual, hits = (np.zeros((len(types), len(classes))) for _ in range(3))
    for cell_state, morph_class, guess, n in counts[["cell_state", "morph_class", "predicted", "n"]].itertuples(index=False):
        t = t_index[cell_state]
        if morph_class in c_index:
            actual[t, c_index[morph_class]] += n
        if guess in c_index:
            predicted[t, c_index[guess]] += n
            if guess == morph_class:
                hits[t, c_index[guess]] += n
    return predicted, actual, hits


def _macro_f1(weights: np.ndarray, predicted: np.ndarray, actual: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Macro F1 per row of `weights` (resample x type multiplicities) over the classes present."""
    p, a, tp = weights @ predicted, weights @ actual, weights @ hits
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(p + a > 0, 2 * tp / (p + a), 0.0)
    present = a > 0
    return (f1 * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1)


def type_draws(n_types: int) -> np.ndarray:
    """The fine-type bootstrap's resample multiplicities (N_BOOT x n_types; seed BOOT_SEED)."""
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_types, size=(N_BOOT, n_types))
    weights = np.zeros((N_BOOT, n_types))
    np.add.at(weights, (np.repeat(np.arange(N_BOOT), n_types), draws.ravel()), 1)
    return weights


def type_bootstrap_compare(mine: pd.DataFrame, theirs: pd.DataFrame) -> dict:
    """TYPE_BOOT_RULE: the paired fine-type cluster bootstrap of the morph_class macro-F1 difference
    from the two arms' knn_type_counts tables."""
    types = sorted(set(mine["cell_state"]) | set(theirs["cell_state"]))
    classes = sorted(set(mine["morph_class"]) | set(theirs["morph_class"]))
    if not types or not classes:
        return {"delta": None, "se": None, "ci95": None, "n_types": len(types), "rule": TYPE_BOOT_RULE}
    arms = [_type_matrices(counts, types, classes) for counts in (mine, theirs)]
    observed = _macro_f1(np.ones((1, len(types))), *arms[0]) - _macro_f1(np.ones((1, len(types))), *arms[1])
    weights = type_draws(len(types))
    boot = _macro_f1(weights, *arms[0]) - _macro_f1(weights, *arms[1])
    return {"delta": float(observed[0]), "se": _se(boot),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_types": len(types), "n_classes": len(classes), "rule": TYPE_BOOT_RULE}


def compare(mine: dict, control: dict) -> tuple[dict, pd.DataFrame]:
    """`mine` / `control`: {"per_class", "per_type", "per_image"} tables. The primary (morph_class)
    macro-F1 difference with the SE of the paired per-class difference (bootstrap over classes) and
    the verdict; the fine (cell_state) difference likewise, without a verdict; the transfer-accuracy
    difference (morph_class) with a bootstrap over images within classes."""
    coarse, table = _paired(mine["per_class"], control["per_class"], "morph_class")
    fine, _ = _paired(mine["per_type"], control["per_type"], "cell_state")
    fine["n_types"] = fine.pop("n")
    out = {"level": "morph_class", "n_classes": coarse.pop("n"),
           "n_classes_only_this_arm": coarse.pop("n_only_this_arm"),
           "n_classes_only_control": coarse.pop("n_only_control"),
           "rule": f"difference of {PRIMARY} (morph_class) over the classes of both; SE = sd of {N_BOOT} bootstrap "
                   f"resamples of classes of the per-class F1 difference (seed {BOOT_SEED}). The fine level likewise, "
                   "without a verdict. kNN transfer accuracy: pooled over the sampled cells (delivered + excluded), "
                   "SE from resampling images within classes",
           "verdict_rule": VERDICT_RULE,
           "fine": fine}
    if mine.get("per_class_cross_source") is not None and control.get("per_class_cross_source") is not None:
        guard, _ = _paired(mine["per_class_cross_source"], control["per_class_cross_source"], "morph_class")
        guard = {"status": "available", "delta": guard["delta"], "se": guard["se"], "value": guard["value"],
                 "control_value": guard["control_value"], "n_classes": guard["n"]}
    else:
        guard = {"status": "unavailable", "delta": None, "se": None, "value": None, "control_value": None,
                 "n_classes": 0, "note": "the control has no per_class_cross_source table (scored before the guard)"}
    out["cross_source_guard"] = guard
    if coarse["delta"] is None:
        return {**out, **coarse, "verdict": None, "verdict_note": None, "knn_transfer_accuracy": None}, table
    note = None
    if not coarse["se"]:            # 0 or None: one class, or the same difference for every class
        verdict, note = NO_RESOLVABLE, "se_zero"
    elif coarse["delta"] > 2 * coarse["se"]:
        if guard["delta"] is None:
            verdict, note = GAIN, "cross-source guard unavailable: the verdict from the primary alone"
        else:
            verdict = GAIN if guard["delta"] >= 0 else SHORTCUT
    else:
        verdict = LOSS if coarse["delta"] < -2 * coarse["se"] else NO_RESOLVABLE
    pair = ["n_knn_correct", "n_knn_scored"]
    images = mine["per_image"][["image_id", "morph_class", *pair]].merge(
        control["per_image"][["image_id", *pair]], on="image_id", suffixes=("", "_control"))
    images = images[images["morph_class"].isin(set(mine["per_class"]["morph_class"]))]    # the scored classes
    rng = np.random.default_rng(BOOT_SEED)
    sums = np.zeros((N_BOOT, 4))
    columns = ["n_knn_correct", "n_knn_scored", "n_knn_correct_control", "n_knn_scored_control"]
    for _, group in images.groupby("morph_class", sort=True):
        values = group[columns].to_numpy(float)
        sums += values[rng.integers(0, len(values), size=(N_BOOT, len(values)))].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sums[:, 0] / sums[:, 1] - sums[:, 2] / sums[:, 3]
    total = images[columns].sum()
    acc = total["n_knn_correct"] / total["n_knn_scored"] if total["n_knn_scored"] else None
    acc_control = (total["n_knn_correct_control"] / total["n_knn_scored_control"]
                   if total["n_knn_scored_control"] else None)
    accuracy = {"value": acc, "control_value": acc_control,
                "delta": float(acc - acc_control) if acc is not None and acc_control is not None else None,
                "se": _se(boot[np.isfinite(boot)]), "n_images": int(len(images))}
    types = (type_bootstrap_compare(mine["knn_type_counts"], control["knn_type_counts"])
             if mine.get("knn_type_counts") is not None and control.get("knn_type_counts") is not None
             else {"delta": None, "se": None, "ci95": None, "note": "a table has no knn_type_counts", "rule": TYPE_BOOT_RULE})
    return {**out, **coarse, "verdict": verdict, "verdict_note": note, "knn_transfer_accuracy": accuracy,
            "fine_type_bootstrap": types}, table


def control_slug(run: Path) -> dict:
    """{"slug", "source", ...}: the GPU model of the run's own node, whose control it is compared
    with: RUN_META.json `gpu` (the launcher's nvidia-smi `name, memory.total` line; the name before
    the first comma, normalised by hardware.slug like hardware.gpu_slug). Only without
    RUN_META.json: the scoring node's slug."""
    from .hardware import gpu_slug, slug

    path = Path(run) / "RUN_META.json"
    if path.exists():
        meta = json.loads(path.read_text(encoding="utf-8"))
        gpu = str(meta.get("gpu") or "")
        name = gpu.strip().splitlines()[0].split(",")[0].strip() if gpu.strip() else ""
        return {"slug": slug(name), "source": "RUN_META.json gpu", "gpu": gpu,
                "run_meta_gpu_slug": meta.get("gpu_slug")}
    return {"slug": gpu_slug(), "source": "the scoring node (the run has no RUN_META.json)"}


def control_dir(root: Path, prep_name: str, metadata: str, slug: str) -> Path:
    return Path(root) / "control" / prep_name / metadata / slug


CONTROL_TABLES = ("RESULT.json", "per_class.csv", "per_type.csv", "per_image.csv")
FULL_DRAWS = "full_metrics_draws.npz"     # the full_metrics reading (optional in a control scored before it)


def _with_control(result: dict, tables: dict, control: Path | None, here: Path) -> pd.DataFrame | None:
    """Add result["comparison"] when the control has a result, its tables, the same expected image set
    and the same class map (else result["comparison_skipped"] says why)."""
    result["comparison"], result["verdict"], result["knn_transfer_verdict"] = None, None, None
    if control is None or Path(control).resolve() == Path(here).resolve():
        return None
    control = Path(control)
    if not all((control / name).exists() for name in CONTROL_TABLES):
        return None
    try:
        theirs = json.loads((control / "RESULT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        theirs = {}
    if theirs.get("expected_images_sha256") != result.get("expected_images_sha256"):
        result["comparison_skipped"] = (f"the control at {control} scored another expected image set "
                                        f"(sha256 {theirs.get('expected_images_sha256')} vs "
                                        f"{result.get('expected_images_sha256')})")
        return None
    mine_map, their_map = (result.get("class_map") or {}).get("sha256"), (theirs.get("class_map") or {}).get("sha256")
    if mine_map != their_map:
        result["comparison_skipped"] = (f"the control at {control} was scored with another class map "
                                        f"(sha256 {their_map} vs {mine_map})")
        return None
    if theirs.get("expected_set", "prep") != "prep":
        result["comparison_skipped"] = f"the result at {control} is a smoke (scored over its own images); never a control"
        return None
    primary = theirs.get("primary") if isinstance(theirs.get("primary"), dict) else {}
    endpoint = (theirs.get("endpoint"), primary.get("k"), primary.get("max_cells_per_image"), primary.get("vote"),
                primary.get("temperature"))
    if endpoint != (PRIMARY, KNN_K, MAX_CELLS, VOTE, KNN_TEMPERATURE):
        result["comparison_skipped"] = (f"the control at {control} was scored with another endpoint (endpoint, k, "
                                        f"max_cells_per_image, vote, temperature) = {endpoint}, not "
                                        f"{(PRIMARY, KNN_K, MAX_CELLS, VOTE, KNN_TEMPERATURE)}; recompute the control")
        return None
    chain_of = theirs.get("control_chain") if isinstance(theirs.get("control_chain"), dict) else {}
    if chain_of.get("embed") != "zero_shot":
        result["comparison_skipped"] = (f"the control at {control} is a fine-tune (pre-2026-10-03) control; recompute "
                                        "the zero-shot control")
        return None
    read = lambda name, key: pd.read_csv(control / name, dtype={key: str, "image_id": str})   # noqa: E731
    try:
        counts = control / "knn_type_counts.csv"
        comparison, table = compare(tables, {"per_class": read("per_class.csv", "morph_class"),
                                             "per_type": read("per_type.csv", "cell_state"),
                                             "per_image": read("per_image.csv", "morph_class"),
                                             "per_class_cross_source": read("per_class_cross_source.csv", "morph_class")
                                             if (control / "per_class_cross_source.csv").exists() else None,
                                             "knn_type_counts": pd.read_csv(counts, dtype={"cell_state": str,
                                                                                           "morph_class": str,
                                                                                           "predicted": str},
                                                                            keep_default_na=False)
                                             if counts.exists() else None})
    except Exception as error:          # an unreadable or older control table: no comparison, the score stands
        result["comparison_skipped"] = f"the control at {control} could not be compared: {type(error).__name__}: {error}"
        return None
    result["comparison"] = {"control": str(control), **comparison}
    result["knn_transfer_verdict"] = comparison["verdict"]
    _full_vs_control(result, tables, control)
    return table


def _full_vs_control(result: dict, tables: dict, control: Path) -> None:
    """result["full_metrics"] against the control's saved reading (FULL_DRAWS) and result["verdict"] = its three
    verdicts; a control without the reading (scored before the full metrics) or on another layout: no verdicts, a
    note."""
    mine = tables.get("full_metrics")
    if mine is None or not isinstance(result.get("full_metrics"), dict):
        return
    path = Path(control) / FULL_DRAWS
    if not path.exists():
        result["full_metrics"]["control_note"] = (f"the control at {control} has no {FULL_DRAWS} (scored before the "
                                                  "full metrics): no full_metrics verdicts")
        return
    theirs = fm.load(path)
    if theirs.get("fingerprint") != mine["layout"].fingerprint:
        result["full_metrics"]["control_note"] = f"the control at {control} was scored on another full_metrics layout"
        return
    result["full_metrics"] = full_record(mine["layout"], mine["reading"], theirs)
    result["verdict"] = result["full_metrics"]["verdicts"]


# ---------------------------------------------------------------- run mode: inputs, tools, audit, cost

def inputs_check(run: Path, root: Path, prep_name: str = "full", metadata: str = "full") -> dict:
    """The run's prep against its record (subset.prep_problems: PREPARE_PRIVATE.json for `full`,
    SUBSET_<name>.json for a subset; no file changed, missing or added), the run's workspace/data
    against <run>/DATA_SHA256.json, data/images.csv equal to the prep's images.csv after the run's
    metadata blanking (workspace.blank_metadata, the function setup used), and every prep image in
    labels.csv. {"ok", "problems"}; never raises."""
    from .workspace import blank_metadata

    run, root = Path(run), Path(root)
    try:
        found = prep_problems(root, prep_name)
    except (OSError, ValueError) as error:
        found = [f"unreadable ({type(error).__name__}: {error})"]
    problems = [f"prep {prep_name}: {len(found)} problem(s): " + "; ".join(found[:5])] if found else []
    data = run / "workspace" / "data"
    try:
        for rel, digest in json.loads((run / "DATA_SHA256.json").read_text(encoding="utf-8")).items():
            path = run / "workspace" / rel if rel.startswith("data/") else data / rel
            if not path.is_file() or sha256(path) != digest:
                problems.append(f"workspace data: {rel} missing or changed since setup")
    except (OSError, ValueError, AttributeError) as error:
        problems.append(f"workspace data: no readable DATA_SHA256.json ({type(error).__name__})")
    try:
        mine = pd.read_csv(data / "images.csv", dtype=str, keep_default_na=False)
        prep_table = pd.read_csv(prep_dir(root, prep_name) / "images.csv", dtype=str, keep_default_na=False)
        ids, expected = mine["image_id"], set(prep_table["image_id"])
        unknown, missing = sorted(set(ids) - expected), sorted(expected - set(ids))
        if unknown or missing or ids.duplicated().any():
            problems.append(f"workspace data/images.csv: {len(unknown)} unknown, {len(missing)} missing, "
                            f"{int(ids.duplicated().sum())} duplicated image ids (against the prep's images.csv)")
        else:
            want = blank_metadata(prep_table, metadata).astype(str)
            if list(mine.columns) != list(want.columns):
                problems.append(f"workspace data/images.csv: columns {list(mine.columns)} differ from the prep's "
                                f"{list(want.columns)}")
            else:
                aligned = mine.set_index("image_id").loc[want["image_id"]].reset_index()[list(want.columns)]
                differ = (aligned.astype(str).to_numpy() != want.to_numpy()).any(axis=1)
                if differ.any():
                    problems.append(f"workspace data/images.csv: {int(differ.sum())} row(s) differ from the prep's "
                                    f"images.csv after the run's metadata blanking (`{metadata}`), e.g. "
                                    f"{list(want.loc[differ, 'image_id'][:3])}")
        unlabelled = sorted(expected - set(pd.read_csv(root / "private" / "labels.csv", dtype=str,
                                                       keep_default_na=False)["image_id"]))
        if unlabelled:
            problems.append(f"prep: {len(unlabelled)} image ids not in labels.csv, e.g. {unlabelled[:3]}")
    except (OSError, ValueError, KeyError) as error:
        problems.append(f"workspace data/images.csv unreadable or not comparable ({type(error).__name__}: {error})")
    return {"ok": not problems, "problems": problems}


def tool_intact(run: Path) -> dict:
    """workspace/tools against the repository template's files, and its _config.json against
    <run>/TOOLS_SHA256.json (written by setup) without `allow_tool_stubs`."""
    run = Path(run)
    tools = run / "workspace" / "tools"
    expected = {p.relative_to(TEMPLATE_TOOLS).as_posix(): sha256(p) for p in sorted(TEMPLATE_TOOLS.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts}
    problem = None
    try:
        expected["_config.json"] = json.loads((run / "TOOLS_SHA256.json").read_text(encoding="utf-8"))["_config.json"]
    except (OSError, ValueError, KeyError):
        problem = "no _config.json hash in TOOLS_SHA256.json"
    actual = {p.relative_to(tools).as_posix(): sha256(p) for p in sorted(tools.rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts} if tools.exists() else {}
    modified = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
    missing, added = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    try:
        if json.loads((tools / "_config.json").read_text(encoding="utf-8")).get("allow_tool_stubs"):
            problem = "tools/_config.json allows the test stubs"
    except (OSError, ValueError):
        pass
    return {"intact": not (modified or missing or added or problem), "modified": modified, "missing": missing,
            "added": added, "problem": problem}


def _redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
    return text


def _excerpt(text: str, needle: str) -> str:
    """At most INPUT_CHARS characters around the first occurrence of the needle, secrets masked."""
    text = _redact(text)
    at = text.find(needle)
    start = max(0, at - INPUT_CHARS // 3) if at >= 0 else 0
    return ("..." if start else "") + text[start:start + INPUT_CHARS] + ("..." if start + INPUT_CHARS < len(text) else "")


def tool_inputs(run: Path) -> list[tuple[int, int, str, str]]:
    """(round, line, tool name, input as JSON text) of every tool call in traces/round_*.jsonl."""
    return [(r, line, name, json.dumps(data)) for r, line, name, data in traced_tool_uses(run)]


PATH_END = r"(?=$|[/\s'\"\\;&|):,\]])"   # a path needle ends at a path boundary (`prep` is not `prep_pairs`)


def needle_patterns(run: Path, root: Path, forbidden: list[str], prep_name: str = "full",
                    metadata: str = "full") -> dict[str, re.Pattern]:
    """{needle: pattern} of what the agent's own inputs must not name. Paths, matched up to a path
    boundary: the given ones (the HCMA raw root), $ROOT/private, control, smoke, every other run in
    $ROOT/runs, every other entry beside the run (other runs, logs, legacy; not the one holding $ROOT,
    workspace._forbidden tells the agent the same), every prep but the run's own, under
    metadata `none` also the run's own prep tables (images.csv, PREPARE.json: they hold the withheld
    metadata), the SHAPE repo (this checkout and the one in
    tools/_config.json). Names, matched anywhere: the HCMA names, the labels' code, the harness, the
    stubs, the key."""
    run, root = Path(run), Path(root)
    own = prep_dir(root, prep_name).name
    paths = [root / "private", root / "control", root / "smoke", REPO,
             *[p for p in all_preps(root) if p.name != own]]
    if metadata != "full":
        paths += [prep_dir(root, prep_name) / "images.csv", prep_dir(root, prep_name) / "PREPARE.json"]
    if (root / "runs").is_dir():
        paths += [p for p in sorted((root / "runs").iterdir()) if p.resolve() != run.resolve()]
    if run.parent.is_dir():
        skip = {run.resolve(), root.resolve(), *root.resolve().parents}
        paths += [p for p in sorted(run.parent.iterdir()) if p.resolve() not in skip]
    try:
        repo = json.loads((run / "workspace" / "tools" / "_config.json").read_text(encoding="utf-8")).get("repo")
    except (OSError, ValueError):
        repo = None
    if repo:
        paths.append(Path(repo))
    path_needles = [str(f).rstrip("/") for f in forbidden if f]
    for p in paths:
        path_needles += [p.as_posix(), p.resolve().as_posix()]
    out = {n: re.compile(re.escape(n) + PATH_END) for n in dict.fromkeys(path_needles) if n}
    out.update({n: re.compile(re.escape(n)) for n in NAME_NEEDLES})
    return out


def needles(run: Path, root: Path, forbidden: list[str], prep_name: str = "full", metadata: str = "full") -> list[str]:
    return list(needle_patterns(run, root, forbidden, prep_name, metadata))


def _bash_write_targets(command: str) -> list[str]:
    """Paths a shell command may write: redirection targets and the operands of file-changing
    commands (cp / ln / rsync / install: the destination; the others: every operand)."""
    targets = re.findall(r">{1,2}\|?\s*([^\s;&|<>()]+)", command)
    for segment in re.split(r"[;&|\n]+", command):
        try:
            words = shlex.split(segment)
        except ValueError:
            words = segment.split()
        for i, word in enumerate(words):
            base = word.rsplit("/", 1)[-1]
            if base in WRITE_COMMANDS or (base == "sed" and any(w.startswith("-i") for w in words[i + 1:])):
                operands = [w for w in words[i + 1:] if not w.startswith("-")]
                targets += operands[-1:] if base in ("cp", "ln", "rsync", "install") else operands
                break
    return targets


def _resolve(path: str, workspace: str) -> str:
    """A path as the agent's tool would see it from the workspace (lexically normalised)."""
    path = path.strip("'\"")
    absolute = path.startswith("/") or re.match(r"[A-Za-z]:/", path)          # (a drive letter: tests on Windows)
    return posixpath.normpath(path if absolute else posixpath.join(workspace, path))


def _protected(path: str, run: str, workspace: str) -> bool:
    resolved = _resolve(path, workspace)
    inside = lambda base: resolved == base or resolved.startswith(base + "/")          # noqa: E731
    return any(inside(f"{workspace}/{d}") for d in PROTECTED_DIRS) or inside(f"{run}/logs") \
        or inside(f"{run}/RUN_META.json") or posixpath.basename(resolved) == "tool_calls.jsonl"


def _structural_hits(name: str, data: dict, run: str, workspace: str) -> list[tuple[str, str]]:
    """(needle, offending part) for climbs above the workspace and writes to the evidence."""
    hits = []
    if name == "Bash":
        command = str(data.get("command", ""))
        if "../.." in command:              # the shell's cwd is unknown: any double climb
            hits.append((CLIMB, "../.."))
        if "images/.." in command:          # data/images is a symlink into the prep: `..` leaves the workspace
            hits.append((CLIMB, "images/.."))
        hits += [(WRITE, target) for target in _bash_write_targets(command) if _protected(target, run, workspace)]
    paths = [str(data[k]) for k in ("file_path", "path", "notebook_path") if data.get(k)]
    for path in paths:                      # Read / Write / Edit / Glob / Grep: relative to the workspace
        climbs = ".." in path.split("/") and not (_resolve(path, workspace) + "/").startswith(workspace + "/")
        if climbs or "images/.." in path:   # data/images is a symlink into $ROOT/prep: `..` leaves the workspace
            hits.append((CLIMB, path))
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit") and _protected(path, run, workspace):
            hits.append((WRITE, path))
    return hits


def audit(run: Path, root: Path, forbidden: list[str], prep_name: str = "full", metadata: str = "full") -> dict:
    """Static audit of the agent's tool-call inputs (traces) and of the tool command lines in the
    ledger: forbidden names and paths, paths that climb above the workspace, and writes into
    calls/, tools/, data/ or the run's logs/."""
    run = Path(run)
    run_posix = Path(run).absolute().as_posix()
    workspace = f"{run_posix}/workspace"
    wanted = needle_patterns(run, root, forbidden, prep_name, metadata)
    hits = []
    for r, line, tool, data in traced_tool_uses(run):
        text = json.dumps(data)
        hits += [{"needle": n, "round": r, "line": line, "tool": tool, "input": _excerpt(text, n)}
                 for n, pattern in wanted.items() if pattern.search(text)]
        hits += [{"needle": needle, "round": r, "line": line, "tool": tool, "target": part,
                  "input": _excerpt(text, part)} for needle, part in _structural_hits(tool, data, run_posix, workspace)]
    for call_id, record in calls(run).items():
        argv = (record["start"] or {}).get("argv")
        if argv is None and record["dir"] is not None and (record["dir"] / "meta.json").exists():
            try:
                argv = json.loads((record["dir"] / "meta.json").read_text(encoding="utf-8")).get("argv")
            except ValueError:
                argv = None
        text = json.dumps(argv) if argv is not None else ""
        hits += [{"needle": n, "call": call_id, "input": _excerpt(text, n)}
                 for n, pattern in wanted.items() if pattern.search(text)]
    by_needle: dict[str, int] = {}
    for hit in hits:
        by_needle[hit["needle"]] = by_needle.get(hit["needle"], 0) + 1
    return {"needles": [*wanted, CLIMB, WRITE], "hits": hits, "hits_by_needle": by_needle, "clean": not hits,
            "note": "static audit of the agent's tool-call inputs (traces/round_*.jsonl) and of the ledger "
                    "argv; a script the agent writes and runs is seen only through its own text; every hit "
                    "needs human review; hits do not stop scoring"}


def cost(run: Path) -> dict:
    """Wall hours (AGENT_RUN.json), tool calls and seconds per tool (ledger), tokens (result events)."""
    run = Path(run)
    try:
        wall = json.loads((run / "AGENT_RUN.json").read_text(encoding="utf-8")).get("wall_hours_used")
    except (OSError, ValueError):
        wall = None
    tools: dict[str, dict] = {}
    ends = {e.get("call"): e for e in read_ledger(run) if e.get("event") == "end"}
    for e in read_ledger(run):
        if e.get("event") != "start":
            continue
        entry = tools.setdefault(str(e.get("tool")), {"n_calls": 0, "seconds": 0.0, "statuses": {}})
        end = ends.get(e.get("call")) or {}
        entry["n_calls"] += 1
        status = str(end.get("status") or "not_ended")
        entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
        if isinstance(end.get("seconds"), (int, float)):
            entry["seconds"] += float(end["seconds"])
    usage: dict[str, float] = {}
    cost_usd, n_results = 0.0, 0
    for path in sorted((run / "traces").glob("round_*.jsonl")):
        for raw in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "result":
                continue
            n_results += 1
            for key, value in (event.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            if isinstance(event.get("total_cost_usd"), (int, float)):
                cost_usd += float(event["total_cost_usd"])
    return {"wall_hours": wall, "tool_calls": tools,
            "tool_seconds": round(sum(t["seconds"] for t in tools.values()), 1),
            "tokens": usage or None, "cost_usd": round(cost_usd, 4) if n_results else None,
            "n_result_events": n_results}


def load_delivery(run: Path, verified: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    """The cells, embedding, states and UMAP (the cluster call's own X_umap, None when absent) of a
    verified chain: the K call's input embedding and its states at the chosen resolution."""
    records = calls(run)
    src, k = records[verified["embedding_call"]]["dir"], records[verified["cluster_call"]]["dir"]
    cells = pd.read_csv(src / "cells.csv", dtype=str, keep_default_na=False)
    x = np.load(src / "embedding.npy", mmap_mode="r")
    states = pd.read_csv(k / "states.csv", dtype=str, keep_default_na=False).set_index("cell_id")[verified["state_column"]]
    return cells, x, cells["cell_id"].map(states).to_numpy(dtype=object), _umap(k / "states.h5ad", cells["cell_id"])


def _umap(h5ad: Path, cell_ids: pd.Series) -> np.ndarray | None:
    """X_umap of an AnnData in the order of `cell_ids` (obs `cell_id`, else obs_names)."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(h5ad, backed="r")
        coords = np.asarray(adata.obsm["X_umap"])
        ids = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs else pd.Series(adata.obs_names.astype(str))
    except (OSError, KeyError, ValueError):
        return None
    position = {c: n for n, c in enumerate(ids)}
    if not all(c in position for c in cell_ids):
        return None
    return coords[[position[c] for c in cell_ids]]


def _figure(out: Path, tables: dict, umap: np.ndarray | None, cell_ids: pd.Series, root: Path,
            class_map: dict, title: str) -> dict:
    """umap_scored.png in `out` from the scored cells (those of known type) and the UMAP rows; panel (a)
    by morph_class, coloured from the sorted list of every class of the class map."""
    from .figures import umap_scored

    if umap is None:
        return {"path": None, "note": "no X_umap in the cluster call's states.h5ad"}
    cells = tables["cells"]
    position = {c: n for n, c in enumerate(cell_ids)}
    labels = load_private(root)[0]
    umap_scored(Path(out) / FIGURE, umap[[position[c] for c in cells["cell_id"]]], cells["morph_class"],
                cells["state"], cells["knn_status"], cells["source"], all_classes=class_map["classes"],
                all_sources=list(labels["source"]), class_title="morph_class", title=title)
    return {"path": str(Path(out) / FIGURE)}


# ---------------------------------------------------------------- every candidate of a run (D3, A4.1, A5.5, E2)

EMBEDDING_METRICS = (PRIMARY, f"{PRIMARY}_fine", CROSS)
STATE_METRICS = (CLUSTER, "state_replication_share", "state_effective_micrographs_median", "state_composition_auc",
                 "state_composition_auc_fine", "state_composition_auc_cross_source")
CANDIDATE_COLUMNS = ("call", "tool", "parent_embedding_call", "resolution", *EMBEDDING_METRICS, *STATE_METRICS,
                     "n_cells", "eligible", "delivered", "end_time", "end_seconds")
CANDIDATES_RULE = ("every completed embed / batch_correct call scored with the kNN primary, its fine level and the "
                   "cross-source guard on its own cells (and its embed call's excluded cells); every completed "
                   "cluster call at each resolution with the cluster-level transfer, the state replication and "
                   "the state-composition AUCs, its kNN columns those of its input embedding; the same expected "
                   "image set, images a candidate lacks count as missing as usual. eligible = a cluster call that "
                   "ended done on a scored embedding (the per-candidate chain sha256 check is not repeated). Never "
                   "shown to the agent")


def _embedding_inputs(run: Path, records: dict, call_id: str) -> tuple[str | None, set[str]]:
    """(the parent embed call of a batch_correct call, None for an embed call; the embed call's
    excluded cells)."""
    start = records[call_id]["start"]
    embed_id = call_id
    parent = None
    if start.get("tool") == "batch_correct":
        found = [c for c in input_calls(start) if c.startswith("E")]
        parent = embed_id = found[0] if found else None
    if embed_id is None or embed_id not in records or records[embed_id]["start"] is None:
        return parent, set()
    return parent, _excluded_cells(Path(run) / "workspace", records[embed_id]["start"])[0]


def _flatten(summary, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a summary (bools and lists left out), keys joined by dots."""
    out = {}
    if isinstance(summary, dict):
        for key, value in summary.items():
            out.update(_flatten(value, f"{prefix}{key}."))
    elif isinstance(summary, (int, float)) and not isinstance(summary, bool) and np.isfinite(summary):
        out[prefix[:-1]] = float(summary)
    return out


def _summary(record: dict) -> dict:
    """The call's result.json summary (the ledger end line's when result.json is unreadable)."""
    try:
        return json.loads((record["dir"] / "result.json").read_text(encoding="utf-8")).get("summary") or {}
    except (OSError, ValueError, AttributeError):
        return (record.get("end") or {}).get("summary") or {}


def proxy_validity(fields: pd.DataFrame, frame: pd.DataFrame) -> list[dict]:
    """PROXY_RULE: `fields` (one row per candidate row of `frame`, same index; columns tool + fields)."""
    from scipy.stats import spearmanr

    out = []
    for tool, group in fields.groupby("tool", sort=True):
        metric = "state_composition_auc" if tool == "cluster" else PRIMARY
        target = frame.loc[group.index, metric].astype(float)
        for field in sorted(c for c in group.columns if c != "tool"):
            pair = pd.DataFrame({"x": group[field].astype(float), "y": target}).dropna()
            if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
                continue
            rho = spearmanr(pair["x"], pair["y"]).statistic
            out.append({"tool": tool, "field": field, "n": int(len(pair)),
                        "spearman": None if not np.isfinite(rho) else float(rho), "metric": metric})
    return sorted(out, key=lambda e: -abs(e["spearman"]) if e["spearman"] is not None else 1.0)


def _same_resolution(key, resolution: float) -> bool:
    try:
        return math.isclose(float(key), float(resolution), rel_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _completed(record: dict, files: tuple[str, ...]) -> bool:
    return (record["start"] is not None and record["end"] is not None and record["end"].get("status") in COMPLETED
            and record["dir"] is not None and all((record["dir"] / f).is_file() for f in files))


def _best(frame: pd.DataFrame, metric: str, delivered: pd.Series) -> dict:
    values = frame[metric].astype(float)
    if not values.notna().any():
        return {"best": None, "best_call": None, "delivered": None, "regret": None}
    at = values.idxmax()
    mine = values[delivered].dropna()
    best = {"best": float(values[at]), "best_call": str(frame.at[at, "call"]),
            "best_resolution": None if pd.isna(frame.at[at, "resolution"]) else float(frame.at[at, "resolution"]),
            "delivered": float(mine.iloc[0]) if len(mine) else None}
    return {**best, "regret": best["best"] - best["delivered"] if best["delivered"] is not None else None}


def score_candidates(run: Path, root: Path, images: list[str], class_map: dict[str, str], verified: dict) -> tuple[pd.DataFrame, dict]:
    """CANDIDATES_RULE: candidates.csv rows and the RESULT.json `candidates` summary (selection regret
    per metric, the batch decision regret, the resolution regret and the anytime curve)."""
    started = time.time()
    run = Path(run)
    records = calls(run)
    sets = expected_sets(root, images, class_map)
    known = sets["known"]
    known_ids = set(known["image_id"])
    parent_of = known.set_index("image_id")["parent_id"]
    t0 = min((r["start"].get("time") for r in records.values() if r["start"] and r["start"].get("time") is not None),
             default=None)
    rows, embedded, proxies = [], {}, []

    def timing(record: dict) -> dict:
        end = record["end"].get("time")
        return {"end_time": end, "end_seconds": round(end - t0, 1) if end is not None and t0 is not None else None}

    for call_id in sorted(records):
        record = records[call_id]
        if record["start"] is None or record["start"].get("tool") not in ("embed", "batch_correct") \
                or not _completed(record, ("cells.csv", "embedding.npy")):
            continue
        cells = pd.read_csv(record["dir"] / "cells.csv", dtype=str, keep_default_na=False)[["cell_id", "image_id"]]
        x = np.load(record["dir"] / "embedding.npy")
        keep = cells["image_id"].isin(known_ids).to_numpy()
        cells, x = cells[keep].reset_index(drop=True), x[np.flatnonzero(keep)]
        parent, excluded = _embedding_inputs(run, records, call_id)
        knn = knn_all(sets, cells, x, excluded_frame(sorted(excluded), known), cells["image_id"].value_counts())
        embedded[call_id] = {PRIMARY: knn["coarse"]["value"], f"{PRIMARY}_fine": knn["fine"]["value"],
                             CROSS: knn["cross"]["value"], "excluded": excluded}
        proxies.append({"tool": record["start"]["tool"], **_flatten(_summary(record))})
        rows.append({"call": call_id, "tool": record["start"]["tool"], "parent_embedding_call": parent,
                     "resolution": None, **{m: embedded[call_id][m] for m in EMBEDDING_METRICS},
                     "n_cells": int(len(cells)), "eligible": False, **timing(record)})

    coarse_images, fine_images = sets["coarse_images"], sets["fine_images"]
    for call_id in sorted(records):
        record = records[call_id]
        if record["start"] is None or record["start"].get("tool") != "cluster" or not _completed(record, ("states.csv",)):
            continue
        found = [c for c in input_calls(record["start"]) if c[0] in "EB"]
        source = found[0] if found else None
        table = pd.read_csv(record["dir"] / "states.csv", dtype=str, keep_default_na=False)
        table = table[table["image_id"].isin(known_ids)].reset_index(drop=True)
        img = table["image_id"].to_numpy(dtype=object)
        parents = table["image_id"].map(parent_of).to_numpy(dtype=object)
        n_cells = table["image_id"].value_counts()
        excl = excluded_frame(sorted(embedded.get(source, {}).get("excluded", set())), known)
        n_excl = excl["image_id"].value_counts().astype(int)
        in_coarse = table["image_id"].isin(set(coarse_images["image_id"])).to_numpy()
        summary = _summary(record)
        call_fields = _flatten({k: v for k, v in summary.items() if k != "resolutions"})
        per_resolution = summary.get("resolutions") if isinstance(summary.get("resolutions"), dict) else {}
        for resolution, column in sorted(state_columns(table.columns).items()):
            own = next((v for k, v in per_resolution.items() if _same_resolution(k, resolution)), {})
            proxies.append({"tool": "cluster", **call_fields, **_flatten(own, "resolution.")})
            states = table[column].astype(str).to_numpy(dtype=object)
            cluster = _level("morph_class", coarse_images, table[in_coarse].reset_index(drop=True), parents[in_coarse],
                             states[in_coarse], n_cells, n_excl)
            replication = state_replication(states, parents)
            rows.append({
                "call": call_id, "tool": "cluster", "parent_embedding_call": source, "resolution": float(resolution),
                **{m: embedded.get(source, {}).get(m) for m in EMBEDDING_METRICS},
                CLUSTER: cluster["value"],
                "state_replication_share": replication["share_cells_in_replicated_states"],
                "state_effective_micrographs_median": replication["median_effective_micrographs"],
                "state_composition_auc": state_composition_auc(img, states, coarse_images, "morph_class")["auc"],
                "state_composition_auc_fine": state_composition_auc(img, states, fine_images, "cell_state")["auc"],
                "state_composition_auc_cross_source": state_composition_auc(img, states, coarse_images, "morph_class",
                                                                            cross_source=True)["auc"],
                "n_cells": int(len(table)), "eligible": record["end"].get("status") == "done" and source in embedded,
                **timing(record)})

    frame = pd.DataFrame(rows, columns=[c for c in CANDIDATE_COLUMNS if c != "delivered"])
    validity = proxy_validity(pd.DataFrame(proxies, index=frame.index), frame) if len(frame) else []
    ok = bool(verified.get("ok"))
    is_embedding = frame["tool"].isin(["embed", "batch_correct"])
    is_cluster = frame["tool"] == "cluster"
    frame["delivered"] = ok & ((is_embedding & (frame["call"] == verified.get("embedding_call")))
                               | (is_cluster & (frame["call"] == verified.get("cluster_call"))
                                  & np.isclose(frame["resolution"].astype(float), float(verified.get("resolution") or -1))))
    frame = frame[list(CANDIDATE_COLUMNS)]
    selection = {m: _best(frame[is_embedding], m, frame.loc[is_embedding, "delivered"]) for m in EMBEDDING_METRICS}
    selection.update({m: _best(frame[is_cluster], m, frame.loc[is_cluster, "delivered"]) for m in STATE_METRICS})
    # A4.1: the delivered embed call and the batch_correct calls on it, by the cross-source guard
    chosen, e_id = verified.get("embedding_call"), verified.get("embed_call")
    family = is_embedding & ((frame["call"] == e_id) | ((frame["tool"] == "batch_correct") & (frame["parent_embedding_call"] == e_id)))
    batch = {**_best(frame[family], CROSS, frame.loc[family, "call"] == chosen), "bio": CROSS,
             "chosen": chosen if ok else None, "embed_call": e_id if ok else None,
             "n_batch_correct_candidates": int((family & (frame["tool"] == "batch_correct")).sum()),
             "rule": "max(bio(E), bio(B)) - bio(chosen) over the delivered embed call and the batch_correct calls "
                     "on it, bio = the cross-source guard; with no batch_correct call the counterfactual B is not run"}
    if not ok:
        batch["regret"] = None
    # A5.5: over the delivered cluster call's resolutions
    own = is_cluster & (frame["call"] == verified.get("cluster_call"))
    resolution = {m: _best(frame[own], m, frame.loc[own, "delivered"]) for m in (CLUSTER, "state_composition_auc")}
    # E2: the best eligible choice available at each cluster call's end
    curve, best = [], None
    calls_k = frame[is_cluster & frame["eligible"]].drop_duplicates("call").sort_values(["end_time", "call"], kind="stable")
    for _, row in calls_k.iterrows():
        value = row[PRIMARY]
        if value is not None and not pd.isna(value) and (best is None or value > best[0]):
            best = (float(value), row["call"])
        curve.append({"call": row["call"], "end_time": row["end_time"], "end_seconds": row["end_seconds"],
                      "best_so_far": best[0] if best else None, "best_call": best[1] if best else None})
    summary = {"n_embedding_candidates": int(is_embedding.sum()), "n_cluster_calls": int(frame.loc[is_cluster, "call"].nunique()),
               "n_cluster_candidates": int(is_cluster.sum()), "delivered_found": bool(frame["delivered"].any()),
               "selection_regret": selection, "batch_decision_regret": batch, "resolution_regret": resolution,
               "anytime_curve": {"metric": PRIMARY, "points": curve,
                                 "rule": "at each eligible cluster call's end: the best hidden primary (of its input "
                                         "embedding) over the eligible cluster calls ended so far"},
               "proxy_validity": validity, "proxy_validity_rule": PROXY_RULE,
               "rule": CANDIDATES_RULE, "seconds": round(time.time() - started, 1)}
    return frame, summary


def _public(check: dict) -> dict:
    """The class map as RESULT.json records it."""
    return {k: check[k] for k in ("ok", "problems", "path", "sha256", "n_rows", "classes", "unscored_cell_states",
                                  "map")}


def score_choice(run: Path, root: Path, choice: dict, *, forbidden: list[str], control: Path | None,
                 lever_coverage: bool = True, prep_name: str | None = None, metadata: str | None = None,
                 figure: bool = True, smoke_images: list[str] | None = None,
                 all_candidates: bool = False) -> tuple[dict, dict | None, pd.DataFrame | None]:
    """Everything of run mode for one choice (the figure, when asked, is written into the run dir).
    The expected image set is the prep's images (expected_set `prep`); a smoke (`smoke_images`, the
    smoke's selected image ids) is scored over its own images only (expected_set `smoke`: missing
    images, coverage and classes only over those) and is never a comparison control.
    Status NO_CLASS_MAP (the class map is missing or does not cover the prep: nothing is scored),
    INVALID_INPUTS (the prep, the workspace data or the delivered image ids do not match the prep)
    or NO_VALID_CHOICE (the chain does not verify): the run delivers no cells."""
    run = Path(run)
    settings = run_settings(run)
    for key, value in (("prep_name", prep_name), ("metadata", metadata)):
        if value:
            settings[key], settings["settings_from"][key] = value, "the caller"
    check = class_map_check(root, settings["prep_name"])
    inputs = inputs_check(run, root, settings["prep_name"], settings["metadata"])
    verified = chain(run, choice)
    result = {"mode": "run", "run": str(run), **settings, "class_map": _public(check), "inputs": inputs,
              "choice": choice, "chain": {k: v for k, v in verified.items() if k != "excluded_cells"}}
    tables = both = None
    if not check["ok"]:
        result.update(status=NO_CLASS_MAP, endpoint=PRIMARY, figure=None)
    else:
        status = "INVALID_INPUTS" if not inputs["ok"] else "OK" if verified["ok"] else "NO_VALID_CHOICE"
        empty = pd.DataFrame({"cell_id": [], "image_id": []}), np.zeros((0, 1)), np.array([], dtype=object), None
        loading = time.time()
        cells, x, states, umap = load_delivery(run, verified) if status == "OK" else empty
        loaded = round(time.time() - loading, 2)
        try:
            images = prep_images(root, settings["prep_name"])
        except (OSError, KeyError, ValueError):
            images = []
        known = set(load_private(root)[0]["image_id"])
        images = [i for i in images if i in known]
        if smoke_images is not None:
            chosen = set(smoke_images)
            images = [i for i in images if i in chosen]
        result.update(expected_set="prep" if smoke_images is None else "smoke",
                      expected_image_ids=None if smoke_images is None else list(images))
        outside = sorted(set(cells["image_id"]) - set(images))
        if outside:
            inputs["ok"] = False
            inputs["problems"].append(f"the delivery names {len(outside)} image ids outside the prep, e.g. {outside[:3]}")
            status, (cells, x, states, umap) = "INVALID_INPUTS", empty
        scored, tables = evaluate(cells, x, states, root, images,
                                  verified.get("excluded_cells") if status == "OK" else None,
                                  class_map=check["map"], strata_meta=_prep_table(root, settings["prep_name"]))
        result.update(status=status, **scored)
        both = _with_control(result, tables, control, run)
        drawing = time.time()
        result["figure"] = _figure(run, tables, umap, cells["cell_id"], root, check,
                                   f"{run.name}: {PRIMARY} {_f(result[PRIMARY], '.3f')}") if figure else None
        result["timings"].update(load_delivery=loaded, figure=round(time.time() - drawing, 2))
        if all_candidates:
            tables["candidates"], result["candidates"] = score_candidates(run, root, images, check["map"], verified)
            result["timings"]["candidates"] = result["candidates"]["seconds"]
    auditing = time.time()
    result.update(cost=cost(run), tool_intact=tool_intact(run),
                  audit=audit(run, root, forbidden, settings["prep_name"], settings["metadata"]))
    if "timings" in result:
        result["timings"]["cost_tools_audit"] = round(time.time() - auditing, 2)
    if lever_coverage:
        from .coverage import final_check

        result["lever_coverage"] = final_check(run)
    return result, tables, both


def write_outputs(out: Path, result: dict, tables: dict | None, both: pd.DataFrame | None, suffix: str = "") -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if tables is not None:
        tables["per_image"].to_csv(out / f"per_image{suffix}.csv", index=False)
        tables["per_class"].to_csv(out / f"per_class{suffix}.csv", index=False)
        tables["per_type"].to_csv(out / f"per_type{suffix}.csv", index=False)
        tables["per_class_cluster"].to_csv(out / f"per_class_cluster{suffix}.csv", index=False)
        tables["per_type_cluster"].to_csv(out / f"per_type_cluster{suffix}.csv", index=False)
        tables["best_cluster_f1"].to_csv(out / f"best_cluster_f1{suffix}.csv", index=False)
        tables["per_class_cross_source"].to_csv(out / f"per_class_cross_source{suffix}.csv", index=False)
        tables["knn_type_counts"].to_csv(out / f"knn_type_counts{suffix}.csv", index=False)
        tables["knn_type_counts_cross_source"].to_csv(out / f"knn_type_counts_cross_source{suffix}.csv", index=False)
        tables["strata"].to_csv(out / f"strata{suffix}.csv", index=False)
        if tables.get("full_metrics") is not None:
            fm.save(out / f"full_metrics_draws{suffix}.npz", tables["full_metrics"]["reading"])
        if tables.get("candidates") is not None:
            tables["candidates"].to_csv(out / f"candidates{suffix}.csv", index=False)
    if both is not None:
        both.to_csv(out / f"per_class_vs_control{suffix}.csv", index=False)
    (out / f"RESULT{suffix}.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    (out / f"RESULT{suffix}.md").write_text(markdown(result), encoding="utf-8")


def recorded_root(run: Path, root: Path) -> Path:
    """The $ROOT of the run's RUN_META.json (`root`) when it resolves to `root`, else `root`: a run
    moved with a symlink left at its old $ROOT is scored, and audited, against the paths its traces
    name; a recorded root elsewhere (RUN_META.json is agent-writable, its writes are audited) is ignored."""
    try:
        recorded = json.loads((Path(run) / "RUN_META.json").read_text(encoding="utf-8")).get("root")
    except (OSError, ValueError, AttributeError):
        recorded = None
    if recorded and Path(recorded).is_dir() and Path(recorded).resolve() == Path(root).resolve():
        return Path(recorded)
    return Path(root)


def score_run(run: Path, root: Path, *, forbidden: list[str], all_candidates: bool = False) -> dict:
    """`score --run`: class map, inputs, chain, tool_intact, audit, then the score of the chosen states
    and embedding, compared with $ROOT/control/<prep-name>/<metadata>/<gpu-slug>/ when it has a
    result for the same images and class map. `all_candidates` (cli `--all-candidates`): also
    score every candidate of the ledger (score_candidates: candidates.csv, RESULT.json
    `candidates`)."""
    from .coverage import read_choice

    run, root = Path(run), recorded_root(run, Path(root))
    which, settings = control_slug(run), run_settings(run)
    result, tables, both = score_choice(
        run, root, read_choice(run) or {}, forbidden=list(forbidden),
        control=control_dir(root, settings["prep_name"], settings["metadata"], which["slug"]),
        all_candidates=all_candidates)
    result["control_slug"] = which
    write_outputs(run, result, tables, both)
    return result


def score_h5ad(h5ad: Path, root: Path, out: Path, *, state_key: str, embedding_key: str,
               prep_name: str = "pairs", metadata: str = "full") -> dict:
    """`score --h5ad` (future arms): obs `cell_id`, `image_id`, obs[state_key], the embedding in
    obsm[embedding_key] ("X" = adata.X), the UMAP in obsm["X_umap"] when present; the same scoring
    and control comparison (prep `prep_name`, metadata level `metadata`, this node's GPU model), no
    chain / audit."""
    import anndata as ad

    from .hardware import gpu_slug

    which = {"slug": gpu_slug(), "source": "the scoring node (h5ad mode)"}
    check = class_map_check(root, prep_name)
    result = {"mode": "h5ad", "h5ad": str(h5ad), "state_key": state_key, "embedding_key": embedding_key,
              "prep_name": prep_name, "metadata": metadata, "class_map": _public(check), "control_slug": which,
              "expected_set": "prep", "expected_image_ids": None}
    tables = both = None
    if not check["ok"]:
        result.update(status=NO_CLASS_MAP, endpoint=PRIMARY, figure=None)
    else:
        adata = ad.read_h5ad(h5ad)
        x = adata.X if embedding_key == "X" else adata.obsm[embedding_key]
        x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
        cells = pd.DataFrame({"cell_id": adata.obs["cell_id"].astype(str).to_numpy(),
                              "image_id": adata.obs["image_id"].astype(str).to_numpy()})
        scored, tables = evaluate(cells, x, adata.obs[state_key].astype(str).to_numpy(dtype=object), root,
                                  prep_images(root, prep_name), class_map=check["map"],
                                  strata_meta=_prep_table(root, prep_name))
        result.update(status="OK", **scored)
        both = _with_control(result, tables, control_dir(root, prep_name, metadata, which["slug"]), Path(out))
        umap = np.asarray(adata.obsm["X_umap"]) if "X_umap" in adata.obsm else None
        result["figure"] = _figure(out, tables, umap, cells["cell_id"], root, check, Path(h5ad).name)
    write_outputs(out, result, tables, both)
    return result


# ---------------------------------------------------------------- RESULT.md

def _f(value, spec: str = ".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def _level_line(name: str, sec: dict) -> str:
    st, sc, lp = sec["states_vs_label"] or {}, sec["scib"] or {}, sec["linear_probe"]
    return (f"{name}: cluster transfer `{CLUSTER}` {_f(sec[CLUSTER], '.3f')} +- {_f(sec[CLUSTER + '_se'], '.3f')} "
            f"(accuracy {_f(sec['transfer_accuracy'], '.3f')}, unlabeled {_f(sec['unlabeled_fraction'], '.3f')}); majority "
            f"accuracy (purity) {_f(sec['majority_accuracy'], '.3f')}; states NMI {_f(st.get('NMI'), '.3f')}, ARI "
            f"{_f(st.get('ARI'), '.3f')}; scIB-style NMI {_f(sc.get('nmi_max'), '.3f')} / ARI "
            f"{_f(sc.get('ari_at_nmi_max'), '.3f')} at resolution {sc.get('resolution')}; linear probe `{PROBE}` "
            f"{_f(lp['value'], '.3f')} +- {_f(lp['se'], '.3f')} ({lp['n_scored_images']} images)")


def _extra_line(name: str, sec: dict) -> str:
    iso, nsc = sec.get("isolated_label") or {}, (sec.get("image_nsc") or {})
    one = lambda r: (f"{_f((nsc.get(r) or {}).get('accuracy'), '.3f')} ({(nsc.get(r) or {}).get('n_evaluated')} "  # noqa: E731
                     f"evaluated, {(nsc.get(r) or {}).get('n_removed')} removed)")
    return (f"{name}: secondary `isolated_label_f1` (scIB; single-batch purity, not cross-micrograph) "
            f"{_f(iso.get('value'), '.3f')} over {', '.join(map(str, iso.get('isolated_labels') or [])) or 'none'} "
            f"(present in {iso.get('n_sources_min')} source(s)), every label's best-cluster F1 in best_cluster_f1.csv; "
            f"secondary image-level 1-NN: NSC {one('nsc')}, NSCB {one('nscb')}")


def markdown(result: dict) -> str:
    lines = [f"# full_HCMA result ({result['mode']}, prep `{result.get('prep_name')}`, metadata "
             f"`{result.get('metadata')}`): {result.get('status')}", ""]
    cmap = result.get("class_map") or {}
    if result.get("expected_set") == "smoke":
        lines += [f"smoke: scored over its {len(result.get('expected_image_ids') or [])} images only", ""]
    if "primary" not in result:
        lines.append(f"Not scored ({result.get('status')}): " + "; ".join(cmap.get("problems") or ["no score"]) + ".")
    else:
        p, fine, cov, sec = result["primary"], result["fine"], result["coverage"], result["secondary"]
        lines.append(f"Primary `{PRIMARY}` (morph_class, {p['n_classes']} classes, k = {p['k']}, <= "
                     f"{p['max_cells_per_image']} cells per image): **{_f(p['value'])} +- {_f(p['se'])}** (SE, bootstrap "
                     f"over classes); median per-class F1 `knn_transfer_median_f1` {_f(p.get('median_f1'))}; "
                     f"accuracy {_f(p['accuracy'], '.3f')}; delivered cells only (the {p['n_excluded_sampled']} "
                     f"sampled cells the agent excluded left out instead of counted as misses): {_f(p['delivered_only'])}. "
                     f"{p['n_queries']} sampled cells, {p['n_no_prediction']} without a prediction; "
                     f"{p.get('n_images_missing', 0)} images with neither delivered nor excluded cells "
                     f"({p.get('n_missing_queries', 0)} misses, {MAX_CELLS} each); "
                     f"{p['n_classes_without_cells']} classes without delivered cells (F1 0). Each sampled cell is given "
                     "the class with the largest weighted vote among its nearest sampled cells from other parent "
                     f"micrographs ({VOTE_RULE}).")
        cross = result.get("cross_source") or {}
        if cross:
            off_cross = ", ".join(f"{c['morph_class']} ({c['n_sources']} source)" for c in cross.get("unscorable_classes") or [])
            lines.append(f"Cross-source guard (secondary) `{CROSS}`: {_f(cross.get('value'))} +- {_f(cross.get('se'))} "
                         f"over {cross.get('n_classes')} classes; neighbours from other sources only; classes without a "
                         f"cross-source partner removed: {off_cross or 'none'}.")
        ref = result.get("reference") or {}
        if ref:
            lines.append(f"Reference (same sample): oracle {_f((ref.get('oracle') or {}).get('primary'))} / chance "
                         f"{_f((ref.get('chance') or {}).get('primary'))} on the primary, oracle "
                         f"{_f((ref.get('oracle') or {}).get('cross_source'))} / chance "
                         f"{_f((ref.get('chance') or {}).get('cross_source'))} on the cross-source guard; "
                         f"normalized_primary {_f(ref.get('normalized_primary'))}.")
        lines.append(f"Fine `{PRIMARY}_fine` (cell_state, {fine['n_types']} types): {_f(fine['value'])} +- "
                     f"{_f(fine['se'])}; accuracy {_f(fine['accuracy'], '.3f')}; delivered only {_f(fine['delivered_only'])}; "
                     f"{fine['n_types_without_cells']} types without delivered cells. Class map {cmap.get('path')} "
                     f"(sha256 {cmap.get('sha256')}).")
        off = result.get("unscorable_classes") or {}
        for column, name in (("morph_class", "classes"), ("cell_state", "types")):
            listed = off.get(column) or []
            lines.append(f"Unscorable {name} at the {column} level (fewer than 2 micrographs in the expected set; not "
                         f"scored there): " + (", ".join(f"{c[column]} ({c['n_images']} images)" for c in listed)
                                               if listed else "none") + ".")
        seeds = result.get("seed_spread")
        if seeds:
            lines.append(f"Seeds: {seeds['values']} (mean {_f(seeds['mean'])}, sd {_f(seeds['sd'])}).")
        c = result.get("comparison")
        if c:
            acc, cf = c.get("knn_transfer_accuracy") or {}, c.get("fine") or {}
            tb = c.get("fine_type_bootstrap") or {}
            lines.append(f"Versus the control ({c['control']}; GPU model from {result['control_slug']['source']}): "
                         f"{_f(c['value'])} vs {_f(c['control_value'])}, delta {_f(c['delta'], '+.4f')} +- "
                         f"{_f(c['se'])} (SE, paired over {c['n_classes']} classes) -> knn_transfer verdict **{c['verdict']}**"
                         f"{' (SE 0: no verdict beyond ' + NO_RESOLVABLE + ')' if c.get('verdict_note') == 'se_zero' else ''}. "
                         f"Cross-source guard delta {_f((c.get('cross_source_guard') or {}).get('delta'), '+.4f')} "
                         f"({(c.get('cross_source_guard') or {}).get('status', 'unavailable')}). "
                         f"Verdict rule: {c.get('verdict_rule', VERDICT_RULE)}"
                         f"{' Note: ' + c['verdict_note'] + '.' if c.get('verdict_note') and c['verdict_note'] != 'se_zero' else ''} "
                         f"Fine: delta {_f(cf.get('delta'), '+.4f')} +- {_f(cf.get('se'))} ({cf.get('n_types')} types). "
                         f"kNN accuracy {_f(acc.get('value'))} vs {_f(acc.get('control_value'))}, delta "
                         f"{_f(acc.get('delta'), '+.4f')} +- {_f(acc.get('se'))} (SE, images within classes). "
                         f"Fine-type cluster bootstrap of the primary delta: SE {_f(tb.get('se'))}, 95 % CI "
                         f"{[round(v, 4) for v in tb['ci95']] if tb.get('ci95') else 'n/a'} ({tb.get('n_types')} types; "
                         "the verdict stays on the class bootstrap).")
        elif result.get("comparison_skipped"):
            lines.append(f"Versus the control: not compared ({result['comparison_skipped']}).")
        elif result.get("control_slug"):
            which = result["control_slug"]
            lines.append(f"Versus the control: not compared (no control result for prep `{result.get('prep_name')}`, "
                         f"metadata `{result.get('metadata')}` and GPU model `{which['slug']}`, from {which['source']}).")
        else:
            lines.append("Versus the control: not compared (this is the control).")
        full = result.get("full_metrics") or {}
        if full.get("status") == "OK":
            lines.append("Full metrics (verdict metrics, benchmarks/_shared/FULL_METRICS_CONTRACT.md; "
                         f"{full['n_units_without_cells']} images without delivered cells counted as misses):\n\n"
                         + fm.markdown(full)
                         + (f"\n\nNote: {full['control_note']}." if full.get("control_note") else ""))
        elif full:
            lines.append(f"Full metrics: unavailable ({full.get('reason')}).")
        src = sec["source_residue"] or {}
        rep_, comp, nm = sec.get("state_replication") or {}, sec.get("state_composition") or {}, sec.get("near_miss") or {}
        if rep_:
            auc = lambda k: _f((comp.get(k) or {}).get("auc"), ".3f")          # noqa: E731
            lines.append(f"State level (secondary, chosen resolution): {_f(rep_.get('share_cells_in_replicated_states'), '.3f')} "
                         f"of the cells in states replicated across >= 2 micrographs; median effective micrographs per "
                         f"state {_f(rep_.get('median_effective_micrographs'), '.2f')}; state-composition replicate AUC "
                         f"morph_class {auc('morph_class')}, cell_state {auc('cell_state')}, morph_class cross-source "
                         f"{auc('morph_class_cross_source')}. Fine-level kNN near misses (same morph_class): "
                         f"{_f(nm.get('share'), '.3f')} of {nm.get('n_errors')} errors.")
        lines.append("Secondary. " + _extra_line("morph_class", sec["morph_class"]) + ". "
                     + _extra_line("cell_state", sec["cell_state"]) + ".")
        lines.append("Secondary. " + _level_line("morph_class", sec["morph_class"]) + ". "
                     + _level_line("cell_state", sec["cell_state"]) + f". {sec['n_clusters']} clusters; source "
                     f"balanced accuracy {_f(src.get('balanced_accuracy'), '.3f')} "
                     f"(chance {_f(src.get('chance'), '.3f')}).")
        lines.append(f"Coverage: {cov['n_images_with_cells']}/{cov['n_images']} prep images with cells; {cov['n_cells']} "
                     f"cells; cells per image median {_f(cov['cells_per_image']['median'], 'g')} (p10 "
                     f"{_f(cov['cells_per_image']['p10'], 'g')}, p90 {_f(cov['cells_per_image']['p90'], 'g')})"
                     + (f"; {result['chain']['n_excluded']} cells excluded by the agent" if result.get("chain") else "")
                     + ".")
    if result["mode"] == "run":
        cost_ = result["cost"]
        tools = ", ".join(f"{t} {v['n_calls']} ({v['seconds'] / 3600:.2f} h)" for t, v in cost_["tool_calls"].items())
        lines.append(f"Cost: wall {_f(cost_['wall_hours'], '.2f')} h; tool calls {tools or 'none'}; tokens "
                     f"{cost_['tokens'] or 'n/a'}; cost {_f(cost_['cost_usd'], '.2f')} USD.")
        if not result["inputs"]["ok"]:
            lines.append("Inputs: INVALID (no cells scored): " + "; ".join(result["inputs"]["problems"]))
        ch = result["chain"]
        lines.append(f"Chain: verified ({ch['cluster_call']} at {ch['resolution']} <- {ch['embedding_call']}"
                     f"{' <- ' + ch['embed_call'] if ch['embed_call'] != ch['embedding_call'] else ''}; "
                     f"{ch['n_cells']} cells, {ch['n_excluded']} excluded)." if ch["ok"] else
                     "Chain: NOT VALID (no cells scored): " + "; ".join(ch["problems"]))
        ti = result["tool_intact"]
        lines.append("Tools intact: yes." if ti["intact"] else
                     f"Tools intact: NO (modified {ti['modified']}, missing {ti['missing']}, added {ti['added']}"
                     + (f"; {ti['problem']}" if ti.get('problem') else "") + ").")
        au = result["audit"]
        if au["clean"]:
            lines.append("Audit: no tool-call input names a forbidden path.")
        else:
            listed = []
            for needle, n in au["hits_by_needle"].items():
                hits = [h for h in au["hits"] if h["needle"] == needle]
                listed.append(f"- `{needle}` ({n}):\n" + "\n".join(
                    f"  - {'round ' + str(h['round']) if 'round' in h else 'ledger ' + h['call']}: "
                    f"`{h['input'].replace('`', chr(39))}`" for h in hits[:AUDIT_LISTED]))
            lines.append(f"Audit: {len(au['hits'])} hits (needs review; all in RESULT.json `audit.hits`):\n\n"
                         + "\n".join(listed))
        lc = result.get("lever_coverage")
        if lc:
            lines.append("Lever coverage: complete." if not lc["missing"] else
                         f"Lever coverage: untested without a reason: {', '.join(lc['missing'])}.")
    fig = result.get("figure") or {}
    lines.append(f"Figure: {fig.get('path') or 'none'}" + (f" ({fig['note']})" if fig.get("note") else "") + ".")
    return "\n\n".join(lines) + "\n"
