"""Verifier 6: Morphological state (scanpy, tunable).

Terminal verifier, strictest bar. Two upgrades over the original:
  - dual stability (seeds + resolutions) is a JOINT criterion that pinpoints the
    root cause: both low = no real structure (data); only-resolution = tune
    resolution; only-seed = tune neighbors.
  - the final biological credibility gate: significant enrichment WHILE
    batch-dominated is a batch false positive and must FAIL, not pass as a
    discovery.
"""

from __future__ import annotations

import numpy as np

from shape.utils.logger import logger
from ..loaders import _codes, _load_adata
from shape.utils.schemas import VLRRResult

from ..base import BaseVerifier, batch_dominates_group, normalize, weighted_score


# --- morphological metric computation (folded out of the adapters bridge, Phase 2b).
# The audit's "quantify should own the metrics" case: the heavy scanpy/Leiden work that
# produces this stage's metrics now lives with its verifier. _load_adata / _codes stay
# in adapters.py as shared loaders. ---

def _silhouette_safe(emb, codes) -> float:
    try:
        from sklearn.metrics import silhouette_score
        uniq = np.unique(codes)
        if len(uniq) < 2 or len(uniq) >= len(codes):
            return 0.0
        return float(silhouette_score(emb, codes))
    except Exception as e:
        logger.warning(f"silhouette failed: {e}")
        return 0.0


def _enrichment_stats(cluster_codes, group_codes):
    """Return (FDR-ish p-value, Cramer's V effect size) for cluster x group."""
    try:
        from scipy.stats import chi2_contingency
        n_clusters = len(np.unique(cluster_codes))
        n_groups = len(np.unique(group_codes))
        if n_clusters < 2 or n_groups < 2:
            return 1.0, 0.0
        table = np.zeros((n_clusters, n_groups), dtype=float)
        cl_index = {c: i for i, c in enumerate(np.unique(cluster_codes))}
        gp_index = {g: i for i, g in enumerate(np.unique(group_codes))}
        for c, g in zip(cluster_codes, group_codes):
            table[cl_index[c], gp_index[g]] += 1
        chi2, p, _, _ = chi2_contingency(table)
        n = table.sum()
        k = min(n_clusters, n_groups) - 1
        cramers_v = float(np.sqrt(chi2 / (n * k))) if n > 0 and k > 0 else 0.0
        return float(p), min(1.0, cramers_v)
    except Exception as e:
        logger.warning(f"enrichment stats failed: {e}")
        return 1.0, 0.0


def _within_cluster_variance(emb, labels) -> float:
    try:
        vals = []
        for c in np.unique(labels):
            pts = emb[labels == c]
            if len(pts) > 1:
                vals.append(float(np.mean(np.var(pts, axis=0))))
        return float(np.mean(vals)) if vals else 0.0
    except Exception:
        return 0.0


def _replicate_consistency(labels, adata) -> float:
    """1 - mean CV of per-image cluster proportions, averaged over groups."""
    try:
        if "image_name" not in adata.obs:
            return 0.0
        images = np.asarray(adata.obs["image_name"].astype(str))
        groups = (np.asarray(adata.obs["group"].astype(str))
                  if "group" in adata.obs else np.full(len(images), "default"))
        clusters = np.unique(labels)
        scores = []
        for group in np.unique(groups):
            group_mask = groups == group
            group_images = np.unique(images[group_mask])
            if len(group_images) < 2:
                continue
            props = []
            for image in group_images:
                mask = group_mask & (images == image)
                row = np.array([np.mean(labels[mask] == cluster) for cluster in clusters])
                props.append(row)
            props = np.asarray(props)
            mean = props.mean(0)
            present = mean > 1e-9
            if not np.any(present):
                continue
            cv = props[:, present].std(0) / mean[present]
            scores.append(max(0.0, 1.0 - float(np.mean(cv))))
        # No within-group replication means consistency is unverified, not perfect.
        return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        # Fail to the CONSERVATIVE value, not the passing one: a swallowed exception
        # here must not read as perfect replicate consistency (which would let a
        # multi-group "finding" pass its credibility check on a computation that never
        # ran). 0.0 forces the replicate gate to fail instead.
        logger.warning(f"replicate consistency failed ({e}); returning 0.0 (conservative)")
        return 0.0


def adapt_morphological(adata_path: str, resolution: float = 0.5,
                        seeds=(0, 1, 2), context=None):
    """Analyzer h5ad -> (state-metrics dict, context).

    Computes clustering-stability and biological-credibility metrics the
    morphological_state verifier needs, by re-running Leiden on the saved neighbour
    graph across seeds and resolutions (no re-embedding, no GPU).
    """
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    adata = _load_adata(adata_path)
    emb = np.asarray(adata.X, dtype=np.float64)
    group = _codes(adata.obs["group"]) if "group" in adata.obs else np.zeros(emb.shape[0], dtype=int)
    batch = _codes(adata.obs["batch"]) if "batch" in adata.obs else np.zeros(emb.shape[0], dtype=int)

    def _leiden(res, seed):
        key = f"_vlrr_{res:.3f}_{seed}"
        try:
            sc.tl.leiden(adata, resolution=float(res), random_state=int(seed),
                         key_added=key, flavor="igraph", n_iterations=2, directed=False)
        except (TypeError, ImportError, ValueError):
            # older scanpy / no python-igraph: fall back to the legacy leidenalg path
            sc.tl.leiden(adata, resolution=float(res), random_state=int(seed), key_added=key)
        return np.asarray(adata.obs[key].astype(int))

    # base clustering + multi-seed stability at the chosen resolution
    seed_labels = [_leiden(resolution, s) for s in seeds]
    base = seed_labels[0]
    ari_seeds = np.mean([adjusted_rand_score(seed_labels[i], seed_labels[j])
                         for i in range(len(seed_labels)) for j in range(i + 1, len(seed_labels))]) \
        if len(seed_labels) > 1 else 1.0

    # multi-resolution stability vs the base clustering
    res_grid = [resolution * 0.8, resolution, resolution * 1.25]
    res_labels = [_leiden(r, seeds[0]) for r in res_grid]
    ari_res = np.mean([adjusted_rand_score(base, rl) for rl in res_labels])

    n_clusters = len(np.unique(base))
    sizes = np.array([np.sum(base == c) for c in np.unique(base)], dtype=float)
    small_thresh = max(3.0, 0.02 * base.size)
    small_cluster_fraction = float(np.mean(sizes < small_thresh)) if n_clusters else 0.0

    p_enrich, effect = _enrichment_stats(base, group)
    # replicate consistency: stability of cluster proportions across images in a group
    replicate_consistency = _replicate_consistency(base, adata)

    metrics = {
        "n_clusters": float(n_clusters),
        # number of distinct experimental groups; the verifier reads this to decide
        # whether the multi-group credibility gate applies (fires without a flag).
        "n_groups": float(len(np.unique(group))),
        "small_cluster_fraction": small_cluster_fraction,
        "ARI_across_seeds": float(ari_seeds),
        "ARI_across_resolutions": float(ari_res),
        "within_cluster_variance": _within_cluster_variance(emb, base),
        "batch_silhouette": _silhouette_safe(emb, batch),
        "group_silhouette": _silhouette_safe(emb, group),
        "group_enrichment_FDR": float(p_enrich),
        "effect_size": float(effect),
        "replicate_consistency": float(replicate_consistency),
    }
    return metrics, dict(context or {})


def state_qc(adata_metrics: dict) -> dict[str, float]:
    """Expects precomputed scanpy outputs passed in as a dict:
    n_clusters, small_cluster_fraction, ARI_across_seeds, ARI_across_resolutions,
    within_cluster_variance, batch_silhouette, group_silhouette,
    group_enrichment_FDR, effect_size, replicate_consistency."""
    return {k: float(v) for k, v in adata_metrics.items()}


class MorphologicalStateVerifier(BaseVerifier):
    name = "morphological_state"

    # config-declared gates (soft priors), not inline literals in localize()/adjudicate.
    # batch dominates the biology when its silhouette clears max(FLOOR, group + MARGIN)
    # -- see `batch_dominates_group` in base.py for the shared FLOOR/MARGIN defaults.
    _SMALL_CLUSTER_MAX = 0.30

    def _batch_dominates(self, metrics) -> bool:
        return batch_dominates_group(metrics["batch_silhouette"], metrics["group_silhouette"])

    def __init__(self, multi_group: bool, ari_min: float = 0.7):
        self.multi_group = multi_group
        self.ari_min = ari_min

    def _is_multi_group(self, metrics) -> bool:
        """True when the task was constructed multi-group OR the data itself carries
        >=2 groups. The data-driven half is what makes the biological-credibility gate
        fire in the LIVE agent path, where the verifier is constructed with
        multi_group=False (the stage_map default) regardless of the actual data."""
        return bool(self.multi_group) or metrics.get("n_groups", 1.0) > 1.0

    def quantify(self, tool_output, context) -> dict[str, float]:
        return state_qc(tool_output)

    def localize(self, metrics, context):
        triggered = []
        seed_bad = metrics["ARI_across_seeds"] < self.ari_min
        res_bad = metrics["ARI_across_resolutions"] < self.ari_min

        if seed_bad and res_bad:
            triggered.append("no_structure")
            return "data_intrinsic", triggered, []     # clustering structure absent
        if self._batch_dominates(metrics):
            triggered.append("batch_driven")
            return "upstream", triggered, []            # batch dominates -> upstream
        if res_bad:
            triggered.append("resolution_unstable")
        if seed_bad:
            triggered.append("seed_unstable")
        if metrics["small_cluster_fraction"] > self._SMALL_CLUSTER_MAX:
            triggered.append("fragmented")
        return "this_step", triggered, []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        parts = {
            "seed_stab": normalize(metrics["ARI_across_seeds"], 1.0, self.ari_min),
            "res_stab": normalize(metrics["ARI_across_resolutions"], 1.0, self.ari_min),
            "compact": normalize(metrics["small_cluster_fraction"], 0.0, 0.30),
        }
        score = weighted_score(parts, {k: 1.0 for k in parts})

        if locus == "data_intrinsic":
            return VLRRResult("FAIL", score, {}, action="conceded", fault="ceiling",
                              notes="Both seed and resolution ARI low: no stable "
                                    "morphological structure exists; tuning won't help.")

        # final biological credibility gate (highest priority among passes)
        if self._is_multi_group(metrics):
            sig = metrics["group_enrichment_FDR"] < 0.05
            batch_conf = self._batch_dominates(metrics)
            if sig and batch_conf:
                return VLRRResult("FAIL", score, {}, action="conceded", fault="upstream",
                                  upstream_target="representation",
                                  notes="Enrichment significant BUT batch-dominated: "
                                        "this is a batch false positive, not a finding.")
        if locus == "upstream":
            return VLRRResult("FAIL", score, {}, action="conceded", fault="upstream",
                              upstream_target="representation",
                              notes="Batch silhouette > group: correct batch upstream.")

        if "resolution_unstable" in triggered or "fragmented" in triggered:
            return VLRRResult(
                "WARNING", score, {}, action="retune_here",
                param_delta={"leiden_resolution": "down"},
                expected_effect={"small_cluster_fraction": "down",
                                 "ARI_across_resolutions": "up"},
                notes="Resolution instability/fragmentation -> lower leiden_resolution.")
        if "seed_unstable" in triggered:
            return VLRRResult(
                "WARNING", score, {}, action="retune_here",
                param_delta={"n_neighbors": "up"},
                expected_effect={"ARI_across_seeds": "up"},
                notes="Seed instability -> increase n_neighbors for a steadier graph.")

        # passes structure + stability; enforce multi-group effect requirements
        if self._is_multi_group(metrics):
            ok = (metrics["group_enrichment_FDR"] < 0.05
                  and metrics["effect_size"] > 0.5
                  and metrics["replicate_consistency"] > 0.70)
            if not ok:
                return VLRRResult("WARNING", score, {}, action="conceded", fault="ceiling",
                                  notes="Stable clusters but group effect fails "
                                        "FDR/effect-size/replicate thresholds.")
        return VLRRResult("PASS", score, {}, action="accept")
