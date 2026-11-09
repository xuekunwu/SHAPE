"""Conditional Harmony correction for DINOv3 CLS embeddings.

This is the product-facing boundary. Benchmark code supplies data and evaluates
the returned embeddings; it does not own the correction decision or Harmony call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import inspect
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from shape.tools.base import BaseTool


SKIPPED_NO_BATCH_EFFECT = "SKIPPED_NO_BATCH_EFFECT"
SKIPPED_INSUFFICIENT_BATCHES = "SKIPPED_INSUFFICIENT_BATCHES"
CORRECTED = "CORRECTED_HARMONY"
DETECT_MARGIN = 0.10
TOOL_VERSION = "1.4.0"

# The formal nested-LOSO policy keeps its historical, digest-bound candidate
# grid. Runtime calls use harmonypy's own adaptive defaults instead: the
# product tool must not turn a small held-out set into a 300-cluster problem.
LIBRARY_DEFAULT_PARAMS = {
    "theta": 2.0,
    "lambda_value": None,
    "sigma": 0.1,
    "nclust": None,
    "random_state": 0,
}


# Optional runtime keys (1.3.0): harmonypy solver settings (None = not passed, harmonypy's own
# default) and input processing applied before the detector and Harmony. Omitting them keeps the
# 1.2.0 behaviour exactly.
SOLVER_KEYS = ("max_iter_harmony", "max_iter_kmeans", "epsilon_cluster", "epsilon_harmony",
               "block_size", "tau")
OPTIONAL_PARAMS = {**{key: None for key in SOLVER_KEYS}, "input_scale": "none", "input_pca": 0}
INPUT_SCALES = ("none", "zscore", "l2")
# Every tunable key at its default: the configuration the budgeted loop starts from and steps (null =
# harmonypy's own value; passing it explicitly takes the same harmonypy path as the benchmark tool).
DEFAULT_CONFIG = {**LIBRARY_DEFAULT_PARAMS, **OPTIONAL_PARAMS}


def resolve_library_default_params(n_samples: int) -> dict[str, object]:
    """Resolve harmonypy's adaptive defaults for a concrete input size."""
    n = int(n_samples)
    if n < 2:
        raise ValueError("Harmony requires at least two samples")
    # This mirrors harmonypy's nclust=None rule, with a safety floor and
    # an explicit occupancy guard for small exploratory/held-out inputs.
    auto_nclust = max(1, min(int(round(n / 30.0)), 100))
    auto_nclust = min(auto_nclust, n - 1)
    params = dict(LIBRARY_DEFAULT_PARAMS)
    params["nclust"] = auto_nclust
    return params


def _resolve_runtime_params(params: Mapping[str, object] | None,
                            n_samples: int) -> dict[str, object]:
    if params is None:
        return resolve_library_default_params(n_samples)
    clean = dict(params)
    required = set(LIBRARY_DEFAULT_PARAMS)
    if not required <= set(clean) <= required | set(OPTIONAL_PARAMS):
        raise ValueError("runtime Harmony parameters have an invalid schema")
    for key, default in OPTIONAL_PARAMS.items():
        clean.setdefault(key, default)
    for key in ("max_iter_harmony", "max_iter_kmeans"):
        if clean[key] is not None:
            clean[key] = int(clean[key])
    for key in ("epsilon_cluster", "epsilon_harmony", "block_size", "tau"):
        if clean[key] is not None:
            clean[key] = float(clean[key])
    if clean["input_scale"] not in INPUT_SCALES:
        raise ValueError(f"input_scale must be one of {INPUT_SCALES}")
    clean["input_pca"] = int(clean["input_pca"])
    if clean["input_pca"] < 0:
        raise ValueError("input_pca must be >= 0")
    clean["theta"] = float(clean["theta"])
    clean["sigma"] = float(clean["sigma"])
    clean["random_state"] = int(clean["random_state"])
    if clean["lambda_value"] is not None:
        clean["lambda_value"] = float(clean["lambda_value"])
    if clean["nclust"] is None:
        # harmonypy 2.0.0 computes round(n / 30), which is zero for small
        # smoke-test inputs and then crashes while logging scalar sigma.
        clean["nclust"] = resolve_library_default_params(n_samples)["nclust"]
    else:
        clean["nclust"] = int(clean["nclust"])
        if not 1 <= clean["nclust"] < int(n_samples):
            raise ValueError("runtime Harmony nclust must be in [1, n_samples)")
    return clean


def _validate_detect_margin(value: float) -> float:
    margin = float(value)
    if not np.isfinite(margin) or margin != DETECT_MARGIN:
        raise ValueError(f"detect_margin is frozen at {DETECT_MARGIN:.2f}")
    return margin


def _validate_row_ids(row_ids: Any, n_samples: int) -> tuple[str, ...] | None:
    if row_ids is None:
        return None
    if isinstance(row_ids, (str, bytes, Mapping)):
        raise TypeError("row_ids must be an iterable of per-sample identifiers")
    values = tuple(str(value) for value in row_ids)
    if len(values) != n_samples:
        raise ValueError("row_ids and cls_output must have the same row count")
    if len(set(values)) != len(values):
        raise ValueError("row_ids must be unique")
    return values


def _row_id_digest(row_ids: tuple[str, ...] | None) -> str | None:
    if row_ids is None:
        return None
    payload = json.dumps(row_ids, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _callable_identity(value: Callable[..., Any]) -> dict[str, str]:
    return {
        "module": str(getattr(value, "__module__", "unknown")),
        "qualname": str(getattr(value, "__qualname__", type(value).__qualname__)),
        "version": TOOL_VERSION,
    }


def _package_identity(name: str) -> dict[str, str | None]:
    try:
        installed = package_version(name)
    except PackageNotFoundError:
        installed = None
    return {"name": name, "version": installed}


@dataclass(frozen=True)
class BatchEffectReport:
    batch_key: str
    n_samples: int
    n_batches: int
    chance_accuracy: float | None
    batch_predictability: float | None
    effect_margin: float | None
    detected: bool
    reason: str


def _metadata_frame(metadata: Any):
    import pandas as pd

    if isinstance(metadata, pd.DataFrame):
        return metadata.reset_index(drop=True).copy()
    if isinstance(metadata, Mapping):
        return pd.DataFrame(metadata).reset_index(drop=True)
    raise TypeError("metadata must be a pandas DataFrame or column mapping")


def detect_batch_effect(
    cls_output: np.ndarray,
    metadata: Any,
    *,
    batch_key: str,
    detect_margin: float = 0.10,
) -> BatchEffectReport:
    """Return deterministic GT-free evidence for a batch signal."""
    detect_margin = _validate_detect_margin(detect_margin)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    features = np.asarray(cls_output, dtype=float)
    meta = _metadata_frame(metadata)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("cls_output must be a finite two-dimensional numeric matrix")
    if len(meta) != len(features):
        raise ValueError("metadata and cls_output must have the same row count")
    if batch_key not in meta:
        raise KeyError("batch_key is missing from metadata")
    batch = meta[batch_key].astype(str).to_numpy()
    labels, counts = np.unique(batch, return_counts=True)
    if len(labels) < 2 or counts.min() < 2:
        return BatchEffectReport(
            batch_key=batch_key, n_samples=len(batch), n_batches=len(labels),
            chance_accuracy=None, batch_predictability=None, effect_margin=None,
            detected=False, reason="fewer_than_two_batches_or_replicates",
        )
    folds = min(5, int(counts.min()))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    predicted = cross_val_predict(model, features, batch, cv=splitter)
    score = float(balanced_accuracy_score(batch, predicted))
    chance = 1.0 / len(labels)
    margin = score - chance
    return BatchEffectReport(
        batch_key=batch_key, n_samples=len(batch), n_batches=len(labels),
        chance_accuracy=chance, batch_predictability=score, effect_margin=margin,
        detected=bool(margin >= detect_margin),
        reason="batch_predictability_above_margin" if margin >= detect_margin
        else "batch_predictability_not_above_margin",
    )


def preprocess_input(features: np.ndarray, params: Mapping[str, object]) -> tuple[np.ndarray, dict]:
    """Input processing before the tool: per-feature z-score or per-row L2, then PCA to
    `input_pca` components (0 = raw features). Same semantics as the benchmark harness port."""
    x = np.asarray(features, dtype=np.float64)
    info: dict[str, object] = {}
    scale = params.get("input_scale", "none")
    if scale == "zscore":
        sd = x.std(axis=0)
        x = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    elif scale == "l2":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    k = int(params.get("input_pca") or 0)
    if k:
        k = min(k, min(x.shape) - 1)
        u, s, _ = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
        x = u[:, :k] * s[:k]
        info["pca_components"] = k
        info["pca_explained_variance"] = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return x, info


# GT-free readouts returned with every call (1.3.0). Computed on a fixed random subsample of rows
# (seed 0, the same rows for every call on the same input), Euclidean geometry, k = 15 neighbours:
#   batch_mixing_{input,output}    fraction of a row's neighbours in other batch groups, divided by the
#                                  fraction expected if neighbours ignored the batch (1 = fully mixed)
#   centroid_ratio_{input,output}  mean distance of batch centroids to the global centroid, divided by
#                                  the mean distance of rows to their own batch centroid (0 = no offset)
#   within_batch_neighbour_overlap Jaccard overlap of each row's within-batch neighbours in the input and
#                                  in the output (1 = local structure inside a batch unchanged)
#   effective_rank_ratio           effective rank (exp entropy of the normalised spectrum), output / input
#   displacement                   ||output - input|| / ||input - mean||, only when dimensions agree
# Relative to the tool's input before input processing, so configurations with and without PCA are
# compared with the same raw features (displacement is None when PCA changes the dimension).
# Same definitions and baseline as the agentic JUMP benchmark's per-call diagnostics.
READOUT_ROWS = 3000
READOUT_NEIGHBOURS = 15
DIAGNOSTIC_KEYS = ("batch_mixing_output", "centroid_ratio_output", "within_batch_neighbour_overlap",
                   "effective_rank_ratio", "displacement")


def _readout_neighbours(x: np.ndarray, k: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    k = min(k, len(x) - 1)
    return NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]


def _batch_mixing(x: np.ndarray, batch: np.ndarray) -> float | None:
    labels, counts = np.unique(batch, return_counts=True)
    if len(labels) < 2:
        return None
    p = counts / counts.sum()
    expected = 1.0 - float((p ** 2).sum())
    ind = _readout_neighbours(x, READOUT_NEIGHBOURS)
    return float((batch[ind] != batch[:, None]).mean() / expected)


def _centroid_ratio(x: np.ndarray, batch: np.ndarray) -> float | None:
    labels = np.unique(batch)
    if len(labels) < 2:
        return None
    mu = x.mean(axis=0)
    cents = {b: x[batch == b].mean(axis=0) for b in labels}
    between = np.mean([np.linalg.norm(c - mu) for c in cents.values()])
    within = np.mean([np.linalg.norm(x[batch == b] - cents[b], axis=1).mean() for b in labels])
    return float(between / within) if within > 0 else None


def _within_batch_overlap(x0: np.ndarray, x1: np.ndarray, batch: np.ndarray) -> float | None:
    scores = []
    for b in np.unique(batch):
        idx = np.flatnonzero(batch == b)
        if len(idx) <= READOUT_NEIGHBOURS + 1:
            continue
        a, c = (_readout_neighbours(x0[idx], READOUT_NEIGHBOURS),
                _readout_neighbours(x1[idx], READOUT_NEIGHBOURS))
        scores += [len(set(r0) & set(r1)) / len(set(r0) | set(r1)) for r0, r1 in zip(a, c)]
    return float(np.mean(scores)) if scores else None


def _effective_rank(x: np.ndarray) -> float | None:
    s = np.linalg.svd(x - x.mean(axis=0), compute_uv=False) ** 2
    if s.sum() <= 0:
        return None                     # no variance: the rank is undefined
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def gt_free_readouts(before: np.ndarray, after: np.ndarray, batch: Any) -> dict[str, object]:
    """The GT-free readouts of one call (see the comment above); `before` is the tool's input BEFORE
    input processing (so every configuration is compared with the same raw features), `after` its
    output."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    n = len(before)
    rows = (np.arange(n) if n <= READOUT_ROWS
            else np.sort(np.random.default_rng(0).choice(n, READOUT_ROWS, replace=False)))
    x0, x1, b = before[rows], after[rows], batch[rows]
    rank0, rank1 = _effective_rank(x0), _effective_rank(x1)
    spread = float(np.linalg.norm(x0 - x0.mean(axis=0)))
    return {
        "n_rows": int(len(rows)), "k": READOUT_NEIGHBOURS,
        "batch_mixing_input": _batch_mixing(x0, b), "batch_mixing_output": _batch_mixing(x1, b),
        "centroid_ratio_input": _centroid_ratio(x0, b), "centroid_ratio_output": _centroid_ratio(x1, b),
        "within_batch_neighbour_overlap": _within_batch_overlap(x0, x1, b),
        "effective_rank_ratio": None if rank0 is None or rank1 is None else rank1 / rank0,
        "displacement": (float(np.linalg.norm(x1 - x0) / spread)
                         if x0.shape == x1.shape and spread > 0 else None),
    }


def _telemetry(ho) -> dict:
    """What the harmonypy object exposes about the solve: objective traces and the cluster occupancy
    (attribute names differ across versions; missing ones are left out)."""
    out = {}
    for name in ("objective_harmony", "objective_kmeans"):
        value = getattr(ho, name, None)
        if value is not None:
            out[name] = [float(v) for v in np.asarray(value, dtype=float).ravel()]
    R = getattr(ho, "R", None)
    if R is not None and np.ndim(R) == 2:
        R = np.asarray(R, dtype=float)
        mass = R.sum(axis=1) if R.shape[0] < R.shape[1] else R.sum(axis=0)
        p = mass / mass.sum()
        out["n_clusters"] = int(len(p))
        out["cluster_mass_min_max"] = [float(p.min()), float(p.max())]
        out["cluster_effective_number"] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    return out


def effective_defaults(result: Mapping[str, Any]) -> dict[str, object]:
    """The values the null / off levers act with, from a call at the defaults: n_rows; input_pca = the
    number of input features (PCA off); harmonypy's own solver defaults (its signature); lambda_value
    "dynamic" (harmonypy 2.0 estimates the ridge penalty per cluster unless a value is fixed, so there
    is no single effective value); nclust = the number of clusters Harmony used."""
    n_rows, n_features = ((result.get("provenance") or {}).get("input_shape") or [None, None])[:2]
    out: dict[str, object] = {"n_rows": n_rows, "input_pca": n_features}
    try:
        from harmonypy import run_harmony

        signature = inspect.signature(run_harmony).parameters
        out.update({k: signature[k].default for k in SOLVER_KEYS if k in signature})
        lamb = signature["lamb"].default if "lamb" in signature else None
    except ImportError:
        lamb = None
    out["lambda_value"] = float(lamb) if isinstance(lamb, (int, float)) and lamb > 0 else "dynamic"
    n_clusters = (result.get("telemetry") or {}).get("n_clusters")
    if n_clusters:
        out["nclust"] = int(n_clusters)
    return out


def _default_harmony_runner(features: np.ndarray, metadata: Any, batch_key: str,
                            params: Mapping[str, object]) -> tuple[np.ndarray, dict]:
    from harmonypy import run_harmony

    nclust = None if params.get("nclust") is None else int(params["nclust"])
    sigma: object = float(params["sigma"])
    if nclust is not None:
        sigma = np.full(nclust, float(params["sigma"]), dtype=float)
    kwargs: dict[str, object] = {
        "theta": float(params["theta"]),
        "sigma": sigma,
        "random_state": int(params["random_state"]),
    }
    if params.get("lambda_value") is not None:
        kwargs["lamb"] = float(params["lambda_value"])
    if nclust is not None:
        kwargs["nclust"] = nclust
    kwargs.update({key: params[key] for key in SOLVER_KEYS if params.get(key) is not None})
    output = run_harmony(
        np.asarray(features, dtype=float), _metadata_frame(metadata), batch_key,
        **kwargs,
    )
    corrected = np.asarray(output.Z_corr, dtype=float)
    if corrected.shape == features.shape:
        return corrected, _telemetry(output)
    if corrected.ndim == 2 and corrected.T.shape == features.shape:
        return corrected.T, _telemetry(output)
    raise ValueError("Harmony output shape does not match cls_output")


class Batch_Corrector_Tool(BaseTool):
    """Detect and conditionally correct batch effects in DINOv3 CLS output."""

    def __init__(self):
        super().__init__(
            tool_name="Batch_Corrector_Tool",
            tool_description=(
                "Detect GT-free batch effects in DINOv3 CLS embeddings and apply "
                "Harmony only when the batch signal exceeds the frozen margin."
            ),
            tool_version=TOOL_VERSION,
            input_types={
                "cls_output": "finite float matrix [n_samples, n_features]",
                "metadata": "table or column mapping with one batch column",
                "batch_key": "metadata column name",
                "row_ids": "optional unique per-sample identifiers",
            },
            output_type="corrected CLS matrix plus detector and provenance evidence",
            demo_commands=["Batch_Corrector_Tool.execute(cls_output, metadata, batch_key='batch')"],
            user_metadata={"ground_truth_free": True, "correction_backend": "harmonypy"},
        )

    def execute(
        self,
        cls_output: Any,
        metadata: Any,
        batch_key: str = "batch",
        row_ids: Any | None = None,
        detect_margin: float = DETECT_MARGIN,
        harmony_params: Mapping[str, object] | None = None,
        output_dir: str | None = None,
        run_id: str | None = None,
        detector: Callable[..., BatchEffectReport] = detect_batch_effect,
        harmony_runner: Callable[..., Any] = _default_harmony_runner,
    ) -> dict[str, Any]:
        started = time.time()
        features = np.asarray(cls_output, dtype=float)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("cls_output must be a finite two-dimensional numeric matrix")
        meta = _metadata_frame(metadata)
        if len(meta) != len(features):
            raise ValueError("metadata and cls_output must have the same row count")
        margin = _validate_detect_margin(detect_margin)
        validated_row_ids = _validate_row_ids(row_ids, len(features))
        params = _resolve_runtime_params(harmony_params, len(features))
        parameter_source = ("harmonypy_library_default" if harmony_params is None else "caller_supplied")
        preprocess: dict[str, object] = {}
        raw_input = features                     # the GT-free readouts compare the output with this
        if harmony_params is not None and (params["input_scale"] != "none" or params["input_pca"]):
            features, preprocess = preprocess_input(features, params)
        report = detector(features, meta, batch_key=batch_key, detect_margin=margin)
        corrected = features.copy()
        applied = False
        telemetry: dict[str, object] = {}
        if report.detected:
            # a runner returns the corrected matrix, or (corrected matrix, Harmony telemetry)
            output = harmony_runner(features, meta, batch_key, params)
            if isinstance(output, tuple):
                output, telemetry = output
            corrected = np.asarray(output, dtype=float)
            if corrected.shape != features.shape or not np.isfinite(corrected).all():
                raise ValueError("Harmony runner returned invalid corrected CLS output")
            applied = True
        gt_free_metrics = {"batch_effect_detected": bool(report.detected),
                           **gt_free_readouts(raw_input, corrected, meta[batch_key].to_numpy())}
        status = CORRECTED if applied else (
            SKIPPED_INSUFFICIENT_BATCHES
            if report.reason == "fewer_than_two_batches_or_replicates"
            else SKIPPED_NO_BATCH_EFFECT
        )
        result: dict[str, Any] = {
            "status": status,
            "corrected_cls_output": corrected,
            "batch_effect": (asdict(report) if hasattr(report, "__dataclass_fields__") else dict(vars(report))),
            "harmony": ({"applied": applied, "parameters": params, "parameter_source": parameter_source}
                        if applied else None),
            "preprocess": preprocess,
            "telemetry": dict(telemetry),
            "seconds": round(time.time() - started, 1),
            "gt_free_metrics": gt_free_metrics,
            "provenance": {
                "tool": "Batch_Corrector_Tool",
                "tool_version": TOOL_VERSION,
                "run_id": run_id,
                "input_shape": list(features.shape),
                "output_shape": list(corrected.shape),
                "batch_key": batch_key,
                "detect_margin": margin,
                "row_id_digest": _row_id_digest(validated_row_ids),
                "row_identity_validated": validated_row_ids is not None,
                "detector": _callable_identity(detector),
                "harmony_backend": _package_identity("harmonypy"),
                "harmony_runner": _callable_identity(harmony_runner),
            },
        }
        if output_dir is not None:
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            np.save(output / "corrected_cls_output.npy", corrected)
            serializable = {key: value for key, value in result.items()
                            if key != "corrected_cls_output"}
            serializable["corrected_cls_output_path"] = str(output / "corrected_cls_output.npy")
            (output / "BATCH_CORRECTOR_RESULT.json").write_text(
                json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
            )
        return result
