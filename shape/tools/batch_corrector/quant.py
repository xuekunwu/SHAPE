"""Label-dependent batch-correction metrics (calibration readouts and benchmark evaluation).

The one metric implementation of the batch corrector: the budgeted VLRR loop scores its calibration
rows with it (`verifiers/batch_correction_unified.py`) and the JUMP benchmark evaluates with it
(`benchmarks/batch_correction_JUMP/`). It is not a GT-free readout: every metric needs a biological
label.

Definitions: Arevalo et al., Nat Commun 15, 6516 (2024), doi:10.1038/s41467-024-50613-5, with the
authors' choices (vendored `metrics/scib.py` and `metrics/map.py` at cd9bcf99): `scib-metrics` for
the scib family (kBET omitted: no maintained Python implementation) and `copairs` for the two mAP
metrics; DMSO dropped before the scib metrics but kept as the mAP-negcon reference; cosine
silhouettes rescaled (asw+1)/2 and subsampled to 20000 rows with a fixed seed; one Leiden clustering
at resolution 1.0 for NMI/ARI; scaled LISI on a k = 25 neighbour graph; copairs null_size 10000.
The external agentic benchmark (BatchAgentic `agentic_jump/harness/metric.py`) is a copy of this
module; its tests check parity.

Adaptive overall: a metric whose preconditions do not hold on the scored rows (fewer than two
batches -> the three batch metrics; no DMSO -> mAP-negcon; fewer than two replicated compounds, or no
plate / perturbation-type columns -> the mAPs; fewer than two labels -> the label metrics) is left
out and listed in `skipped`; nothing is imputed. overall = w_b * mean(available batch) + w_o *
mean(available bio), with (w_b, w_o) = (0.4, 0.6) renormalised over the groups that have at least
one metric.

Views: a split's rows are that split's treatment compounds plus every control well (DMSO and
positive controls). Uncertainty: B fixed random halves of the plates, drawn within each batch group
(same halves for every representation); the 2SE band of a paired full-view delta is approximated as
2 * sd(half deltas) / sqrt(2). The halves depend only on the plate list and the seed.
"""
from __future__ import annotations

from importlib.metadata import version as package_version

import numpy as np
import pandas as pd

LABEL_KEY = "Metadata_JCP2022"
DMSO = "DMSO"
CONTROL_SPLITS = ("negcon", "poscon")
MAP_COLUMNS = ("Metadata_PlateType", "Metadata_PertType", "Metadata_Plate", "Metadata_Well")
MAX_SIL_CELLS = 20000
NULL_SIZE = 10000
BATCH_METRICS = ["silhouette_batch", "lisi_batch", "graph_conn"]
BIO_METRICS = ["lisi_label", "ari", "nmi", "asw", "negcon_mean_map", "nonrep_mean_map"]
WEIGHTS = {"batch": 0.4, "bio": 0.6}


class MetricUnavailable(ValueError):
    """A metric whose mathematical preconditions are absent from this view."""


# ---------------------------------------------------------------- metrics
def _sil_subsample_idx(n: int, max_cells: int | None = None, seed: int = 0):
    max_cells = MAX_SIL_CELLS if max_cells is None else max_cells
    return None if n <= max_cells else np.random.default_rng(seed).choice(n, max_cells, replace=False)


def filter_dmso(meta: pd.DataFrame, feats: np.ndarray, label_key: str):
    non_dmso = (meta[label_key] != DMSO).to_numpy()
    return meta.loc[non_dmso].reset_index(drop=True), np.asarray(feats)[non_dmso]


def _neighbors(feats: np.ndarray, n_neighbors: int):
    from scib_metrics.nearest_neighbors import pynndescent
    return pynndescent(np.asarray(feats, dtype=np.float64), n_neighbors=n_neighbors)


def silhouette_batch(feats, labels, batch, metric: str = "cosine") -> float:
    from scib_metrics import silhouette_batch as _silhouette_batch
    feats, labels, batch = np.asarray(feats), np.asarray(labels), np.asarray(batch)
    idx = _sil_subsample_idx(feats.shape[0])
    if idx is not None:
        feats, labels, batch = feats[idx], labels[idx], batch[idx]
    # scib computes a silhouette within each biological label. A label needs at
    # least two batches and more cells than batches; otherwise sklearn cannot
    # define a silhouette. Drop only those labels, and report an unavailable
    # metric when none remain instead of letting scib fail in pd.concat.
    frame = pd.DataFrame({"label": labels, "batch": batch})
    counts = frame.groupby("label", observed=True)["batch"].agg(["size", "nunique"])
    eligible = counts.index[(counts["nunique"] >= 2) & (counts["size"] > counts["nunique"])]
    keep = frame["label"].isin(eligible).to_numpy()
    if not keep.any():
        raise MetricUnavailable("no label has enough cells in at least two batches")
    feats, labels, batch = feats[keep], labels[keep], batch[keep]
    return float(_silhouette_batch(feats, labels, batch, metric=metric))


def asw(feats, labels, metric: str = "cosine") -> float:
    from scib_metrics import silhouette_label
    feats, labels = np.asarray(feats), np.asarray(labels)
    idx = _sil_subsample_idx(feats.shape[0])
    if idx is not None:
        feats, labels = feats[idx], labels[idx]
    return float(silhouette_label(feats, labels, rescale=True, metric=metric))


def graph_conn(feats, labels, n_neighbors: int = 25) -> float:
    from scib_metrics import graph_connectivity
    return float(graph_connectivity(_neighbors(feats, n_neighbors), np.asarray(labels)))


def lisi_batch(feats, batch, n_neighbors: int = 25) -> float:
    from scib_metrics import ilisi_knn
    return float(ilisi_knn(_neighbors(feats, n_neighbors), np.asarray(batch), scale=True))


def lisi_label(feats, labels, n_neighbors: int = 25) -> float:
    from scib_metrics import clisi_knn
    return float(clisi_knn(_neighbors(feats, n_neighbors), np.asarray(labels), scale=True))


def nmi_ari(feats, labels, n_neighbors: int = 25, resolution: float = 1.0) -> dict:
    from scib_metrics import nmi_ari_cluster_labels_leiden
    result = nmi_ari_cluster_labels_leiden(_neighbors(feats, n_neighbors), np.asarray(labels),
                                            optimize_resolution=False, resolution=resolution)
    return {"nmi": float(result["nmi"]), "ari": float(result["ari"])}


def _map_index(meta: pd.DataFrame, plate_types, label_key: str, ignore_dmso: bool = False) -> np.ndarray:
    """Vendored `metrics/map.py::_index`: plate-type filter, drop poscon, drop singleton compounds,
    drop the over-replicated outlier compound, optional DMSO drop."""
    index = meta["Metadata_PlateType"].isin(plate_types)
    index &= meta["Metadata_PertType"] != "poscon"
    valid = meta.loc[index, label_key].value_counts()
    index &= meta[label_key].isin(valid[valid > 1].index)
    index &= meta[label_key] != "JCP2022_033954"
    if ignore_dmso:
        index &= meta[label_key] != DMSO
    return index.to_numpy()


def _group_negcons(meta: pd.DataFrame, label_key: str) -> pd.DataFrame:
    """Every DMSO well gets a unique pseudo-compound id: no DMSO-DMSO positive pairs."""
    meta = meta.copy()
    negcon_ix = meta[label_key] == DMSO
    negcon_ids = [f"{DMSO}_{i}" for i in range(int(negcon_ix.sum()))]
    pert_id = meta[label_key].astype("category").cat.add_categories(negcon_ids)
    pert_id[negcon_ix.to_numpy()] = negcon_ids
    meta[label_key] = pert_id
    return meta


def _copairs_map_sameby(label_key: str):
    try:
        major, minor = (int(part) for part in package_version("copairs").split(".")[:2])
    except Exception:
        major, minor = 0, 4
    return [label_key] if (major, minor) >= (0, 5) else label_key


def average_precision_negcon(meta, feats, label_key: str, plate_types, well_key: str = "Metadata_Well",
                             plate_key: str = "Metadata_Plate", null_size: int = NULL_SIZE, seed: int = 0,
                             threshold: float = 0.05) -> float:
    import copairs.map as copairs
    ix = _map_index(meta, plate_types, label_key)
    sub_meta, sub_feats = _group_negcons(meta.loc[ix].reset_index(drop=True).copy(), label_key), np.asarray(feats)[ix]
    ap = copairs.average_precision(sub_meta, sub_feats, pos_sameby=[label_key], pos_diffby=[well_key],
                                   neg_sameby=[plate_key], neg_diffby=["Metadata_PertType", label_key],
                                   batch_size=20000, progress_bar=False)
    ap = ap.query("Metadata_PertType != 'negcon'") if "Metadata_PertType" in ap.columns else ap
    scores = copairs.mean_average_precision(ap, _copairs_map_sameby(label_key), threshold=threshold,
                                            null_size=null_size, seed=seed, progress_bar=False)
    return float(scores["mean_average_precision"].mean())


def average_precision_nonrep(meta, feats, label_key: str, plate_types, plate_key: str = "Metadata_Plate",
                             null_size: int = NULL_SIZE, seed: int = 0, threshold: float = 0.05) -> float:
    import copairs.map as copairs
    ix = _map_index(meta, plate_types, label_key, ignore_dmso=True)
    sub_meta, sub_feats = meta.loc[ix].reset_index(drop=True).copy(), np.asarray(feats)[ix]
    ap = copairs.average_precision(sub_meta, sub_feats, pos_sameby=[label_key], pos_diffby=[],
                                   neg_sameby=[plate_key], neg_diffby=[label_key],
                                   batch_size=20000, progress_bar=False)
    scores = copairs.mean_average_precision(ap, _copairs_map_sameby(label_key), threshold=threshold,
                                            null_size=null_size, seed=seed, progress_bar=False)
    return float(scores["mean_average_precision"].mean())


# ---------------------------------------------------------------- adaptive composite
def _replicated(meta, label_key: str, plate_types, ignore_dmso: bool) -> int:
    ix = _map_index(meta, plate_types, label_key, ignore_dmso=ignore_dmso)
    return int(meta.loc[ix & (meta[label_key] != DMSO).to_numpy(), label_key].nunique())


def _safe(fn, *args, **kwargs):
    """Run one metric; an unmet mathematical precondition is not a pipeline crash."""
    try:
        return fn(*args, **kwargs), None
    except Exception as error:  # noqa: BLE001
        if isinstance(error, MetricUnavailable):
            return None, f"{getattr(fn, '__name__', 'metric')}: {error}"
        if type(error).__name__ == "UnpairedException":
            return None, f"copairs: {error}"
        raise


def composite_all(meta, feats, label_key: str, batch_key: str, plate_types, n_neighbors: int = 25) -> dict:
    meta_f, feats_f = filter_dmso(meta, feats, label_key)
    n_batch = int(meta_f[batch_key].nunique())
    n_label = int(meta_f[label_key].nunique())
    scores, skipped = {}, {}

    def put(name, fn, *args, **kwargs):
        value, reason = _safe(fn, *args, **kwargs)
        if reason is None:
            scores[name] = float(value)
        else:
            skipped[name] = reason

    if n_batch >= 2:
        put("silhouette_batch", silhouette_batch, feats_f, meta_f[label_key], meta_f[batch_key])
        put("lisi_batch", lisi_batch, feats_f, meta_f[batch_key], n_neighbors=n_neighbors)
        put("graph_conn", graph_conn, feats_f, meta_f[label_key], n_neighbors=n_neighbors)
    else:
        skipped.update({m: f"{n_batch} batch in the scored rows" for m in BATCH_METRICS})
    if n_label >= 2:
        put("lisi_label", lisi_label, feats_f, meta_f[label_key], n_neighbors=n_neighbors)
        put("asw", asw, feats_f, meta_f[label_key])
        value, reason = _safe(nmi_ari, feats_f, meta_f[label_key], n_neighbors=n_neighbors)
        if reason is None:
            scores.update(value)
        else:
            skipped.update({"nmi": reason, "ari": reason})
    else:
        skipped.update({m: f"{n_label} label in the scored rows" for m in ("lisi_label", "asw", "nmi", "ari")})
    missing = [c for c in MAP_COLUMNS if c not in meta.columns]
    if missing:
        skipped.update({m: f"no {missing} columns" for m in ("negcon_mean_map", "nonrep_mean_map")})
        return aggregate(scores, skipped)
    if not (meta[label_key] == DMSO).any():
        skipped["negcon_mean_map"] = "no DMSO wells in the scored rows"
    elif _replicated(meta, label_key, plate_types, ignore_dmso=False) < 1:
        skipped["negcon_mean_map"] = "no replicated compound"
    else:
        put("negcon_mean_map", average_precision_negcon, meta, feats, label_key, plate_types)
    if _replicated(meta, label_key, plate_types, ignore_dmso=True) < 2:
        skipped["nonrep_mean_map"] = "fewer than two replicated compounds"
    else:
        put("nonrep_mean_map", average_precision_nonrep, meta, feats, label_key, plate_types)
    return aggregate(scores, skipped)


def aggregate(scores: dict, skipped: dict) -> dict:
    groups = {"batch": [scores[m] for m in BATCH_METRICS if m in scores],
              "bio": [scores[m] for m in BIO_METRICS if m in scores]}
    means = {g: (float(np.mean(v)) if v else None) for g, v in groups.items()}
    live = {g: WEIGHTS[g] for g, m in means.items() if m is not None}
    overall = (sum(WEIGHTS[g] * means[g] for g in live) / sum(live.values())) if live else None
    return {**scores, "batch_mean": means["batch"], "bio_mean": means["bio"], "overall": overall,
            "skipped": skipped, "weights_used": {g: w / sum(live.values()) for g, w in live.items()} if live else {}}


# ---------------------------------------------------------------- views and band
def split_mask(meta, split: str | None, plates: set | None = None) -> np.ndarray:
    """`split` (val | test) treatment rows plus every control row; split None = all rows."""
    mask = np.ones(len(meta), bool) if split is None else \
        (meta["split"].isin([split, *CONTROL_SPLITS])).to_numpy()
    if plates is not None:
        mask &= meta["Metadata_Plate"].astype(str).isin(plates).to_numpy()
    return mask


def subset(meta, mask: np.ndarray) -> pd.DataFrame:
    """The rows of `mask`, re-indexed, with unused categories dropped (they would count as labels)."""
    m = meta.loc[mask].reset_index(drop=True)
    for column in m.columns:
        if str(m[column].dtype) == "category":
            m[column] = m[column].cat.remove_unused_categories()
    return m


def composite(meta, feats: np.ndarray, mask: np.ndarray, *, batch_key: str, plate_types) -> dict:
    return composite_all(subset(meta, mask), np.asarray(feats, dtype=np.float64)[mask], LABEL_KEY, batch_key,
                         list(plate_types))


def score_split(meta, feats, split: str | None, *, batch_key: str, plate_types) -> dict:
    return composite(meta, feats, split_mask(meta, split), batch_key=batch_key, plate_types=plate_types)


def plate_halves(meta, n: int, *, batch_key: str, seed: int = 0) -> list[list[str]]:
    """B halves of the plates, drawn within each batch group (a group of one plate always stays)."""
    rng = np.random.default_rng(seed)
    groups = {g: sorted(p.astype(str).unique()) for g, p in meta.groupby(batch_key, observed=True)["Metadata_Plate"]}
    halves = []
    for _ in range(n):
        chosen = []
        for g in sorted(groups):
            plates = groups[g]
            chosen += rng.choice(plates, max(1, round(len(plates) / 2)), replace=False).tolist()
        halves.append(sorted(chosen))
    return halves


def score_halves(meta, feats, split: str, halves: list[list[str]], *, batch_key: str, plate_types) -> list[dict]:
    return [composite(meta, feats, split_mask(meta, split, set(h)), batch_key=batch_key, plate_types=plate_types)
            for h in halves]


def paired_band(ref_full: dict, new_full: dict, ref_halves: list[dict], new_halves: list[dict],
                key: str = "overall") -> dict:
    if ref_full.get(key) is None or new_full.get(key) is None:
        return {"delta": None, "band_2se": None, "significant": None, "n_halves": 0, "note": f"{key} not computable"}
    delta = new_full[key] - ref_full[key]
    diffs = np.array([n[key] - r[key] for r, n in zip(ref_halves, new_halves)
                      if r.get(key) is not None and n.get(key) is not None])
    if len(diffs) < 2:
        return {"delta": float(delta), "band_2se": None, "significant": None, "n_halves": int(len(diffs))}
    band = float(2 * diffs.std(ddof=1) / np.sqrt(2))
    return {"delta": float(delta), "band_2se": band, "significant": bool(abs(delta) > band),
            "half_delta_mean": float(diffs.mean()), "n_halves": int(len(diffs))}


# ---------------------------------------------------------------- evaluator evidence
def replicate_retrieval(meta, feats, *, batch_key: str, plate_types, cross: bool) -> float | None:
    """Treatment-compound replicate retrieval against other compounds on the same plate, with the
    positives restricted to the same batch group (cross=False) or to other batch groups (cross=True).
    None when no such positive pairs exist."""
    import copairs.map as copairs
    ix = _map_index(meta, plate_types, LABEL_KEY, ignore_dmso=True)
    sub_meta, sub_feats = meta.loc[ix].reset_index(drop=True).copy(), np.asarray(feats, dtype=np.float64)[ix]
    pos = {"pos_sameby": [LABEL_KEY], "pos_diffby": [batch_key]} if cross else \
        {"pos_sameby": [LABEL_KEY, batch_key], "pos_diffby": ["Metadata_Plate"]}
    value, reason = _safe(lambda: copairs.average_precision(
        sub_meta, sub_feats, **pos, neg_sameby=["Metadata_Plate"], neg_diffby=[LABEL_KEY],
        batch_size=20000, progress_bar=False))
    if reason is not None or value is None or not len(value):
        return None
    scores = copairs.mean_average_precision(value, _copairs_map_sameby(LABEL_KEY), threshold=0.05,
                                            null_size=NULL_SIZE, seed=0, progress_bar=False)
    return float(scores["mean_average_precision"].mean())
