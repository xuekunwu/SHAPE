"""Batch-correction verifier: the budgeted VRR of the batch corrector (CARD.md "Budgeted VRR"), a
``base.AgenticVerifier`` run by StepVerifier's wave mode.

Rules measure and guard; the agent decides. The tool (``Batch_Corrector_Tool``) owns the batch-effect
gate, Harmony and the GT-free readouts; ``StepVerifier`` owns the loop (the budget, concurrent rounds,
the per-trial ledger against each trial's parent); the shared layer (``base.AgenticVerifier``) owns the
agent's questions (a reasoner failure raises: no deterministic fallback), the validation of its moves,
the plateau, the final choice and the prompt frame. This verifier adds what is the batch corrector's own: it measures (the
calibration readouts, the measured noise, the facts of every call), admits, guards (harmonypy's values,
the final-choice guard) and writes the prompt text; the agent (the reasoner) decides every tool call --
how many per round, from which parents, which levers -- when it is satisfied, how to read its failed
predictions, and the final call.

Trial 0 is the uncorrected input (``uncorrected_result``), a measurement and not a tool call: the gate on
the input, its calibration readouts, the noise estimate, the admission. Every later trial is a tool call
chosen by the agent. Every trial is a ``VLRRResult``: ``scalar_score`` = the calibration ``overall``;
``metrics`` = the calibration readouts (``quant.composite_all`` on the calibration rows) and the tool's
GT-free readouts; ``param_delta`` / ``expected_effect`` / ``notes`` = the move that produced it and its
diagnosis. Trial params are ``{"harmony_params": <every tunable key>}``; a move from the input changes the
tool's ``DEFAULT_CONFIG``. The levers are the CARD's "## Levers" (``card_lever_fields``), bounds
``PARAM_BOUNDS``.

Context: ``upstream_ctx["batch_task"]``, a ``BatchTask`` (per-row batch labels, the tool input, the
calibration view -- calibration reference rows and control rows, never evaluation labels -- the task's
stated noise and the cores). Without it (the planner path) separability cannot be verified: the gate
decides SKIP, otherwise ABSTAIN, and no correction is returned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json as _json
import math as _math
import statistics
import time
from typing import Any, Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from shape.tools.batch_corrector import quant
from shape.tools.batch_corrector.tool import DEFAULT_CONFIG
from shape.tools.batch_corrector.tool import effective_defaults as tool_effective_defaults
from shape.utils.schemas import QuestionFailed, VLRRResult, WaveSelection
# the shared agentic-VRR schemas, re-exported for this module's callers
from shape.utils.schemas import Expectation, Hypothesis, LeverChange, WaveMove, WavePlan  # noqa: F401

from ..base import PARAM_BOUNDS, AgenticVerifier, fmt, numeric
from ..base import concurrency_cap as _concurrency_cap
# the shared loop settings and helpers, re-exported (the loop reads base's: patch base, not this module)
from ..base import BUDGET, REASONER_RETRY_SECONDS, available_cores  # noqa: F401

# The tool's GT-free readouts (``Batch_Corrector_Tool.execute`` -> ``gt_free_metrics``). They are
# deterministic given the output and worsen with any stronger correction: context for the reasoner,
# never a calibration readout.
_READOUTS = ("batch_effect_detected", "n_rows", "k", "batch_mixing_input", "batch_mixing_output",
             "centroid_ratio_input", "centroid_ratio_output", "within_batch_neighbour_overlap",
             "effective_rank_ratio", "displacement")

NOISE_BAND = 0.002          # the stated noise when the caller states none (callers pass the task's)
NOISE_REPEATS = 3           # scorings of the input's calibration view that measure the metric's own noise
THREADS_PER_CALL = 8        # BLAS threads per tool call at most (the largest inputs); cap = cores // threads
ROWS_PER_THREAD = 25_000    # one BLAS thread per this many input rows (`call_threads`), 1 to THREADS_PER_CALL
# Measured 2026-10-10 (harmonypy 2.0.0, WSL 16 cores; one Batch_Corrector_Tool call, zscore + 50 PCs, 2 batches x
# 100 features; seconds at 1/2/4/8 BLAS threads): 500 rows 0.08/0.08/0.10/0.25; 5k 1.16/1.07/1.11/1.24;
# 50k 12.1/11.5/10.5/11.6; 200k 54/43/44/48. harmonypy 2.0.0's bundled OpenBLAS is single-threaded and its C++
# call mostly holds the GIL (k concurrent calls in one process take ~k times one), so the BLAS threads only speed
# the gate, preprocessing and readouts: small inputs gain nothing from more threads and lose up to 3x at 8.
SCORE = "overall"
OBJECTIVES = ("overall", "nonrep_mean_map")     # Pareto membership (a fact shown to the reasoner)
PRESERVE = ("bio_mean", "nonrep_mean_map")      # the final choice may not lower these below the input
ADMISSION_MIN_SHARED_GROUPS = 10
KEY_READOUTS = ("overall", "batch_mean", "bio_mean", "nonrep_mean_map", "negcon_mean_map")
UNCORRECTED = "UNCORRECTED_INPUT"
_INTEGER_LEVERS = {"nclust", "input_pca", "max_iter_harmony", "max_iter_kmeans", "random_state"}


SoftCriterion = Literal["few_reference_groups_in_all_batch_groups", "batch_group_without_negative_controls",
                        "no_batch_effect_on_negative_controls"]


class AdmissionVerdict(BaseModel):
    decision: Literal["RUN", "SKIP", "ABSTAIN"]
    reason_code: str
    reasoning: str
    concerns: list[SoftCriterion] = Field(
        default_factory=list, description="the soft criteria that hold and justify ABSTAIN over a RUN rule outcome")


# the batch corrector's final choice (``keep_input``): the call may be null, the uncorrected input is kept
class FinalChoice(BaseModel):
    call: Optional[str] = Field(None, description="the chosen call id, or null to keep the uncorrected input")
    reasoning: str
    cited_readouts: list[str] = Field(default_factory=list, description="the calibration readouts the choice rests on")


@dataclass(frozen=True, eq=False)
class CalibrationView:
    """The rows the loop may score -- calibration reference groups (e.g. compounds) and control rows of
    the tool input -- with their metadata (label, batch, and for the mAPs plate, well, plate type and
    perturbation type). Never the evaluation rows' labels."""
    rows: np.ndarray
    meta: Any                        # pandas DataFrame aligned with `rows`
    batch_key: str
    plate_types: tuple = ()
    label_key: str = quant.LABEL_KEY

    def score(self, matrix) -> dict:
        """`quant.composite_all` of `matrix` (all rows of the tool input or output) on these rows, at
        float32 precision (as the benchmark scores)."""
        feats = np.asarray(np.asarray(matrix, dtype=np.float32)[self.rows], dtype=np.float64)
        return quant.composite_all(self.meta, feats, self.label_key, self.batch_key, list(self.plate_types))


@dataclass(frozen=True, eq=False)
class BatchTask:
    """What the caller knows about the task (``upstream_ctx["batch_task"]``)."""
    batch: Any                                    # per-row batch labels of the tool input
    features: Any = None                          # the tool input (trial 0, the control-only gate)
    calibration: Optional[CalibrationView] = None
    noise_band: float = NOISE_BAND                # the task's stated noise
    brief: str = ""
    batch_column: str = "batch"                   # the batch column's name shown to the reasoner
    admission: Optional[Mapping] = None           # a precomputed or forced admission
    cores: Optional[int] = None                   # cores for the tool calls (None: the cores this process may use)
    deadline: Optional[float] = None              # epoch seconds by which the stage must end (e.g. the job's limit)


def call_threads(features: Any = None, cores: Optional[int] = None) -> int:
    """BLAS threads for one tool call on `features`: ceil(rows / ROWS_PER_THREAD) within [1, THREADS_PER_CALL]
    (THREADS_PER_CALL without features), at most `cores`."""
    threads = THREADS_PER_CALL
    if features is not None:
        threads = min(THREADS_PER_CALL, max(1, _math.ceil(int(np.shape(features)[0]) / ROWS_PER_THREAD)))
    return max(1, min(threads, int(cores))) if cores else threads


def concurrency_cap(cores: int, features: Any = None) -> int:
    """Tool calls that may run at the same time on `cores` cores, `call_threads(features, cores)` threads each."""
    return _concurrency_cap(cores, call_threads(features, cores))


def blas_limits(features: Any = None, cores: Optional[int] = None):
    """The process-wide BLAS thread limit (threadpoolctl) under which a stage runs its calls: `call_threads`, so
    cap x threads <= cores. The libraries the calls use are loaded first: a BLAS or OpenMP library loaded after
    threadpool_limits keeps its own default."""
    import importlib

    from threadpoolctl import threadpool_limits

    for module in ("harmonypy", "sklearn.neighbors", "scipy.linalg"):
        try:
            importlib.import_module(module)
        except ImportError:
            pass
    return threadpool_limits(limits=call_threads(features, cores))


def uncorrected_result(features, batch) -> dict:
    """Trial 0: the uncorrected input as a tool-result-shaped record (not a tool call) -- the tool's gate
    on all rows and the tool's GT-free readouts of the input against itself."""
    import pandas as pd

    from shape.tools.batch_corrector.tool import detect_batch_effect, gt_free_readouts

    started = time.time()
    x = np.asarray(features, dtype=float)
    labels = np.asarray(batch).astype(str)
    report = detect_batch_effect(x, pd.DataFrame({"batch": labels}), batch_key="batch")
    return {"status": UNCORRECTED, "corrected_cls_output": features, "batch_effect": asdict(report),
            "harmony": None, "preprocess": {}, "telemetry": {},
            "gt_free_metrics": {"batch_effect_detected": bool(report.detected), **gt_free_readouts(x, x, labels)},
            "provenance": {"input_shape": list(x.shape)}, "seconds": round(time.time() - started, 1)}


def admission_rule(evidence: Mapping[str, Any]) -> str:
    """SKIP without a detected batch effect; ABSTAIN when no calibration reference group occurs
    in two or more batch groups (separability cannot be verified); otherwise RUN."""
    if not (evidence.get("gate_all_rows") or {}).get("detected"):
        return "SKIP"
    if int(evidence.get("reference_groups_in_2plus_batch_groups") or 0) < 1:
        return "ABSTAIN"
    return "RUN"


def admission_min_shared_groups(evidence: Mapping[str, Any]) -> int:
    """The soft criterion's threshold, adaptive to the design size: max(2, min(ADMISSION_MIN_SHARED_GROUPS, the
    number of distinct non-control reference groups)); ADMISSION_MIN_SHARED_GROUPS without that count."""
    groups = evidence.get("reference_groups")
    return ADMISSION_MIN_SHARED_GROUPS if groups is None else max(2, min(ADMISSION_MIN_SHARED_GROUPS, int(groups)))


def admission_concerns(evidence: Mapping[str, Any]) -> dict[str, bool]:
    """The soft admission criteria (True = the concern holds). The reasoner may turn a RUN rule
    outcome into ABSTAIN only by citing one that holds. Missing evidence counts as a concern that
    holds (it cannot be verified)."""
    groups = evidence.get("n_batch_groups")
    in_all = int(evidence.get("reference_groups_in_all_batch_groups") or 0)
    control_gate = evidence.get("gate_negative_controls_only")
    return {
        # an absolute count: a design whose other reference groups are batch-specific (e.g. per-lab
        # compound plates) is still verifiable through the groups replicated in every batch group
        "few_reference_groups_in_all_batch_groups": in_all < admission_min_shared_groups(evidence),
        "batch_group_without_negative_controls":
            groups is None or len(evidence.get("negative_controls_per_batch_group") or {}) < int(groups),
        "no_batch_effect_on_negative_controls": not (control_gate or {}).get("detected", False),
    }


def admission_evidence(task: Optional[BatchTask], gate: Optional[Mapping]) -> dict:
    """Admission evidence from the natural metadata (CARD.md "Budgeted VRR", step 1): the batch groups,
    the gate on all rows (trial 0) and on the negative controls only, how many calibration reference
    groups occur in two or more / all batch groups, and the controls per batch group."""
    import pandas as pd

    from shape.tools.batch_corrector.tool import detect_batch_effect

    if task is None:
        return {"gate_all_rows": dict(gate or {}), "reference_groups": 0,
                "reference_groups_in_2plus_batch_groups": 0, "reference_groups_in_all_batch_groups": 0}
    batch = pd.Series(np.asarray(task.batch).astype(str))
    n_groups = int(batch.nunique())
    features = task.features
    evidence = {"n_rows": int(len(batch)), "n_features": int(np.shape(features)[1]) if features is not None else None,
                "batch_column": task.batch_column, "n_batch_groups": n_groups,
                "rows_per_batch_group": batch.value_counts().to_dict(), "gate_all_rows": dict(gate or {}),
                "gate_negative_controls_only": None, "reference_groups": 0,
                "reference_groups_in_2plus_batch_groups": 0, "reference_groups_in_all_batch_groups": 0,
                "negative_controls_per_batch_group": {}, "positive_controls_per_batch_group": {}}
    view = task.calibration
    if view is None:
        return evidence
    meta = view.meta.reset_index(drop=True)
    rows = np.asarray(view.rows)
    pert = (meta["Metadata_PertType"].astype(str) if "Metadata_PertType" in meta
            else pd.Series("trt", index=meta.index))
    row_batch = pd.Series(batch.to_numpy()[rows])
    reference = (pert == "trt").to_numpy()
    groups_per_ref = row_batch[reference].groupby(meta.loc[reference, view.label_key].astype(str).to_numpy()).nunique()
    negcon, poscon = (pert == "negcon").to_numpy(), (pert == "poscon").to_numpy()
    if features is not None and negcon.sum() >= 2 * n_groups:
        report = detect_batch_effect(np.asarray(np.asarray(features)[rows[negcon]], dtype=float),
                                     pd.DataFrame({"batch": row_batch[negcon].to_numpy()}), batch_key="batch")
        evidence["gate_negative_controls_only"] = {k: getattr(report, k) for k in
                                                   ("batch_predictability", "chance_accuracy", "effect_margin", "detected")}
    evidence.update({
        "reference_groups": int(len(groups_per_ref)),
        "reference_groups_in_2plus_batch_groups": int((groups_per_ref >= 2).sum()),
        "reference_groups_in_all_batch_groups": int((groups_per_ref == n_groups).sum()),
        "negative_controls_per_batch_group": row_batch[negcon].value_counts().to_dict(),
        "positive_controls_per_batch_group": row_batch[poscon].value_counts().to_dict(),
    })
    return evidence


def tool_summary(result: Mapping[str, Any]) -> dict:
    """The tool report of one call, summarised for the reasoner: Harmony clusters and their
    occupancy (smallest / largest cluster mass), rounds run, PCA components and explained variance."""
    telemetry, pre = result.get("telemetry") or {}, result.get("preprocess") or {}
    mass = telemetry.get("cluster_mass_min_max")
    return {"n_clusters": telemetry.get("n_clusters"),
            "cluster_effective_number": (round(telemetry["cluster_effective_number"], 1)
                                         if telemetry.get("cluster_effective_number") is not None else None),
            "cluster_mass_min_max": [round(m, 5) for m in mass] if mass else None,
            "harmony_rounds": len(telemetry.get("objective_harmony") or []) or None,
            "pca_components": pre.get("pca_components"),
            "pca_explained_variance": (round(pre["pca_explained_variance"], 3)
                                       if pre.get("pca_explained_variance") is not None else None),
            "seconds": result.get("seconds")}


def adapt_batch_correction(result: Mapping[str, Any], upstream=None):
    """The native tool result is the verifier input; the stage context carries the ``BatchTask``."""
    return result, dict(upstream or {})


class BatchCorrectionVerifier(AgenticVerifier):
    """The budgeted VRR of the batch corrector (``base.AgenticVerifier``); see the module docstring."""

    name = "batch_correction"
    tool_dir = "batch_corrector"
    threads_per_call = THREADS_PER_CALL
    score_key = SCORE
    keep_input = True                   # the uncorrected input is a valid output: the final choice may keep it
    admission_on_input = True
    admission_skip_reason = "no_batch_effect"
    narrowing = True                    # explore broadly, then the agent's own focus_levers, never widened
    key_readouts = KEY_READOUTS
    objectives = OBJECTIVES
    choice_schema = FinalChoice
    tool_phrase = "batch-correction tool (a batch-effect gate followed by Harmony)"
    unit = "task"
    value_examples = '400, 0.07, null, "zscore"'
    cite_text = "Cite the calibration readouts your choice rests on."

    def __init__(self, *, log: Callable[..., None] | None = None, **kwargs):
        super().__init__(log=log, **kwargs)
        self.evidence: dict = {}
        self.effective_defaults: dict = {}
        self.stated_noise = NOISE_BAND
        self.measured_noise: dict[str, float] = {}      # readout -> 2SE of a difference of two scorings
        self.noise: dict[str, float] = {}               # readout -> the noise the facts and the guard use
        self._last: Mapping[str, Any] = {}

    def begin(self, task: Optional[BatchTask] = None) -> None:
        """The threads per call from the task's tool input (`call_threads`), then the shared clock, cores and cap."""
        features = getattr(task, "features", None)
        self.threads_per_call = call_threads(features, getattr(task, "cores", None) or available_cores())
        super().begin(task)
        self.log("concurrency", text=_json.dumps({
            "rows": None if features is None else int(np.shape(features)[0]), "cores": self.cores,
            "threads_per_call": self.threads_per_call, "cap": self.cap}))

    # ---- Verify ---------------------------------------------------------------------
    def quantify(self, tool_output, context) -> dict[str, float]:
        """The calibration readouts of the output (when the task has a calibration view), then the
        tool's GT-free readouts."""
        self._last = tool_output if isinstance(tool_output, Mapping) else {}
        task = context.get("batch_task")
        metrics: dict[str, float] = {}
        if task is not None and task.calibration is not None and self._last.get("corrected_cls_output") is not None:
            metrics.update(numeric(task.calibration.score(self._last["corrected_cls_output"])))
        gt_free = self._last.get("gt_free_metrics") or {}
        metrics.update(numeric({k: gt_free[k] for k in _READOUTS if k in gt_free}))
        self._reports[int(context.get("trial_index") or 0)] = tool_summary(self._last)
        return metrics

    def localize(self, metrics, context):
        if not self._last.get("gt_free_metrics"):
            return "tool_mismatch", ["missing_gt_free_metrics"], []
        return "this_step", [], []

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        score = float(metrics.get(SCORE, float("nan")))
        decision = "PASS" if self._last.get("status") == "CORRECTED_HARMONY" else "WARNING"
        if context.get("trial_index") != 0:
            return VLRRResult(decision, score, dict(metrics), action="retune_here")
        # trial 0 (the uncorrected input): the clock, the cores and the cap (`begin`); measure, then admit
        task = context.get("batch_task")
        self.begin(task)
        self.stated_noise = float(task.noise_band) if task is not None else NOISE_BAND
        if not self.effective_defaults:
            self.effective_defaults = self._defaults_in_effect(self._last)
        self.evidence = admission_evidence(task, self._last.get("batch_effect"))
        self.set_admission(task.admission if task is not None and task.admission is not None
                           else self.admit(self.evidence))
        verdict = self.admission["decision"]
        self.measure_noise(task, metrics, repeat=verdict == "RUN")
        action, fault = self.admission_action()
        return VLRRResult(decision, score, dict(metrics), action=action, fault=fault,
                          notes=f"admission {verdict} (rule {self.admission.get('rule')})")

    @staticmethod
    def _defaults_in_effect(result: Mapping[str, Any]) -> dict:
        """The values the null / off levers act with at the defaults (``tool.effective_defaults``); without
        a Harmony run, nclust by harmonypy's own rule for the number of rows."""
        from shape.tools.batch_corrector.tool import resolve_library_default_params

        out = tool_effective_defaults(result)
        if "nclust" not in out and isinstance(out.get("n_rows"), int) and out["n_rows"] >= 2:
            out["nclust"] = resolve_library_default_params(out["n_rows"])["nclust"]
        return out

    def measure_noise(self, task: Optional[BatchTask], metrics: Mapping[str, float], repeat: bool = True) -> None:
        """The noise of every calibration readout: the larger of the task's stated noise and the measured
        one -- 2SE of the difference of two scorings of the same representation (2 * sqrt(2) * sd over
        NOISE_REPEATS scorings of the input's calibration view; the kNN graph and the Leiden clustering
        of the metric vary from run to run). Measured only when a search follows (`repeat`)."""
        calibration = {k: v for k, v in metrics.items() if k not in _READOUTS}
        self.measured_noise = {}
        if (repeat and task is not None and task.calibration is not None and task.features is not None
                and calibration):
            scorings = [calibration] + [numeric(task.calibration.score(task.features))
                                        for _ in range(NOISE_REPEATS - 1)]
            for key in calibration:
                values = [s[key] for s in scorings if key in s]
                if len(values) >= 2:
                    self.measured_noise[key] = 2 * _math.sqrt(2) * statistics.stdev(values)
            self.log("noise", text=_json.dumps({"stated": self.stated_noise, "measured": self.measured_noise},
                                               default=str))
        self.noise = {k: max(self.stated_noise, self.measured_noise.get(k, 0.0)) for k in calibration}

    def _noise(self, key: str) -> float | None:
        """The noise of a calibration readout; None for the GT-free readouts (no noise estimate)."""
        return None if key in _READOUTS else self.noise.get(key, self.stated_noise)

    def _noise_table(self) -> dict:
        return {k: {"stated": self.stated_noise, "measured": round(self.measured_noise[k], 4)
                    if k in self.measured_noise else None, "used": round(v, 4)} for k, v in self.noise.items()}

    # ---- the loop state in the loop's terms --------------------------------------------
    def trial_config(self, params: Mapping[str, Any]) -> dict:
        """A trial's configuration: its ``harmony_params``."""
        return dict(params.get("harmony_params") or {})

    def trial_params(self, config: Mapping[str, Any]) -> dict:
        return {"harmony_params": dict(config)}

    def defaults(self) -> dict:
        """The tool's ``DEFAULT_CONFIG`` (in its own key order, as the agent sees it)."""
        return dict(DEFAULT_CONFIG)

    # ---- values ---------------------------------------------------------------------
    def coerce(self, lever: str, value: Any) -> Any:
        """Parse one lever value and check it against the tool's bounds; raises ValueError."""
        from shape.tools.batch_corrector.tool import INPUT_SCALES

        if lever not in self.levers:
            raise ValueError(f"{lever!r} is not a CARD lever")
        if isinstance(value, str):
            text = value.strip()
            value = text.strip('"') if lever == "input_scale" else self._literal(lever, text)
        if lever == "input_scale":
            if value not in INPUT_SCALES:
                raise ValueError(f"input_scale must be one of {INPUT_SCALES}")
            return value
        if value is None:
            if lever in ("theta", "sigma", "input_pca"):
                lo = PARAM_BOUNDS[lever][0]                         # a value the lever accepts: its lower bound
                example = str(int(lo)) if lever in _INTEGER_LEVERS else f"{lo:g}"
                raise ValueError(f'{lever} cannot be null: write its value inside the value string, e.g. "value": '
                                 f'"{example}"')
            return None
        if isinstance(value, bool):
            raise ValueError(f"{lever}: {value!r} is not a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{lever}: {value!r} is not a number") from None
        if lever == "lambda_value" and number == -1:
            return None             # harmonypy 2.0: -1 and null are the same dynamic estimate
        if lever == "input_pca" and number == 0:
            return 0
        lo, hi, _ = PARAM_BOUNDS[lever]
        if not _math.isfinite(number) or not lo <= number <= hi:
            raise ValueError(f"{lever}={number} outside [{lo:g}, {hi:g}]")
        n_features = self.effective_defaults.get("input_pca")      # PCA off = the number of input features
        if lever == "input_pca" and n_features and number >= n_features:
            raise ValueError(f"input_pca={number:g} keeps every one of the {n_features} input features (use 0)")
        return int(round(number)) if lever in _INTEGER_LEVERS else number

    @staticmethod
    def _literal(lever: str, text: str) -> Any:
        """A lever value written by the reasoner: a JSON literal, a plain number (".07", "1e-3") or
        null / none; anything else is rejected with the text it got."""
        try:
            return _json.loads(text)
        except ValueError:
            pass
        bare = text.strip('"').strip()
        if bare.lower() in ("null", "none"):
            return None
        try:
            return float(bare)
        except ValueError:
            raise ValueError(f"{lever}: {text!r} is not a number (e.g. 0.07) or null") from None

    def _effective(self, lever: str, value: Any) -> Any:
        """The numeric value a lever acts with: null -> the effective default, input_pca 0 -> the
        number of input features; None when not comparable (a non-numeric effective default such as
        lambda_value "dynamic", categorical levers)."""
        if lever == "input_pca" and not value:
            value = self.effective_defaults.get("input_pca")
        elif value is None:
            value = self.effective_defaults.get(lever)
        if lever == "lambda_value" and value == -1:
            return None
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def check_change(self, lever: str, value: Any, base: Mapping[str, Any], parent: str) -> None:
        """A value that acts as the parent's (e.g. `nclust` 100 where null resolves to 100) is the same
        configuration; `nclust` stays below the number of rows. Raises ValueError."""
        in_effect = self._effective(lever, base.get(lever))
        if in_effect is not None and self._effective(lever, value) == in_effect:
            raise ValueError(f"{lever}={value!r} acts as {parent}'s {lever} ({in_effect!r} in effect): "
                             "the same configuration")
        n_rows = self.effective_defaults.get("n_rows")
        if lever == "nclust" and value is not None and n_rows and value >= n_rows:
            raise ValueError("nclust must be below the number of rows")

    # ---- the guard ----------------------------------------------------------------------
    def guard(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The conditions a final choice fails (empty = admissible): calibration overall above the
        uncorrected input's by more than the noise, and PRESERVE not below the input's by more than it."""
        fails = []
        base, mine, noise = self._score(inp), self._score(vres), self._noise(SCORE)
        if base is None or mine is None or mine - base <= noise:
            fails.append(f"overall {fmt(mine)} does not beat the input's {fmt(base)} by more than {noise:.4f}")
        for key in PRESERVE:
            base_k, mine_k, noise_k = self._score(inp, key), self._score(vres, key), self._noise(key)
            if base_k is not None and mine_k is not None and mine_k < base_k - noise_k:
                fails.append(f"{key} {mine_k:.4f} is below the input's {base_k:.4f} by more than {noise_k:.4f}")
        return fails

    # ---- prompts: the batch corrector's text (the frame is the shared one) ------------------
    def final_rule_text(self) -> str:
        return ("Final output: at the end you choose one finished call. It must beat the uncorrected input's\n"
                "calibration overall by more than the noise and keep bio_mean and nonrep_mean_map within the noise\n"
                "of the input or above; if you choose none, or no call qualifies, the uncorrected input is kept.")

    def evidence_text(self) -> str:
        return ("Noise per calibration readout (stated: the task's evaluation noise; measured: 2SE of a difference\n"
                f"between two scorings of the same representation, from {NOISE_REPEATS} scorings of the input; used:\n"
                f"the larger): {_json.dumps(self._noise_table(), default=str)}")

    def defaults_note(self) -> str:
        effective = {k: v for k, v in self.effective_defaults.items() if k in self.levers or k == "n_rows"}
        return ("Effective values at the defaults (null / off levers as resolved by the tool and harmonypy;\n"
                f"input_pca = the number of input features, i.e. PCA off): {effective}")

    def readouts_text(self) -> str:
        return ("Readouts: calibration readouts come from the calibration reference groups and all control rows\n"
                "(higher = better); GT-free readouts are relative to the raw input (displacement and\n"
                "centroid_ratio_output: lower = less batch offset / less change).")

    def input_text(self, inp: VLRRResult) -> str:
        return f"Uncorrected input: {_json.dumps(self._input_row(inp), default=str)}"

    def calls_legend(self) -> str:
        return ("change = relative to the parent; non_default = the full\n"
                "configuration's non-default values; vs_parent / vs_input = key readout deltas, beyond or within\n"
                "the noise -- for a move from \"input\" the parent readouts are the uncorrected input's, the tool's\n"
                "defaults have readouts only once a call runs them; prediction = each stated expectation with the\n"
                "observed change from the parent; tool = Harmony clusters and their occupancy, Harmony rounds, PCA\n"
                "components, seconds; guard = whether it is an admissible final choice; pareto = not dominated on\n"
                "(overall, nonrep_mean_map) beyond the noise")

    def _rows(self, inp, attempts) -> list[dict]:
        front, rows = self.pareto(attempts), []
        for call in attempts:                    # in run order
            parent, params, vres = attempts[call]
            parent_vres = inp if parent == "input" else attempts[parent][2]
            fails = self.guard(inp, vres)
            rows.append({
                "call": call, "parent": parent, "change": vres.param_delta or "none (the tool at its defaults)",
                "non_default": {k: v for k, v in params.items() if DEFAULT_CONFIG.get(k) != v},
                "diagnosis": vres.notes,
                "calibration": {k: round(v, 4) for k, v in (vres.metrics or {}).items() if k not in _READOUTS},
                "gt_free": {k: round(v, 4) for k, v in (vres.metrics or {}).items() if k in _READOUTS},
                "tool": self.reports.get(call), "vs_parent": self._effects(parent_vres, vres),
                "vs_input": self._effects(inp, vres), "prediction": self._prediction(parent_vres, vres),
                "guard": "admissible" if not fails else "; ".join(fails), "pareto": call in front})
        return rows

    def _input_row(self, inp) -> dict:
        return {"calibration": {k: round(v, 4) for k, v in (inp.metrics or {}).items() if k not in _READOUTS},
                "gt_free": {k: round(v, 4) for k, v in (inp.metrics or {}).items() if k in _READOUTS},
                "gate": self.evidence.get("gate_all_rows")}

    def choice_text(self) -> str:
        return ("Choose the final output of this batch-correction task: one finished call, or null to keep the\n"
                "uncorrected input.")

    def admissible_text(self) -> str:
        return ("calibration overall above the uncorrected input's by more than the noise;\n"
                "bio_mean and nonrep_mean_map not below the input's by more than the noise")

    def final_evidence_text(self, inp, attempts, admissible: list[str]) -> str:
        return f"Noise per calibration readout: {_json.dumps(self._noise_table(), default=str)}"

    # ---- admission --------------------------------------------------------------------
    def search_values(self, params: Mapping[str, Any], call: Optional[str]) -> dict:
        values = super().search_values(params, call)
        return {k: self.effective_defaults.get(k) if v is None else v for k, v in values.items()}

    def admission_prompt(self, evidence: Mapping[str, Any], rule: str, concerns: Mapping[str, bool]) -> str:
        return (
            "Decide whether a fixed batch-correction tool (gate + Harmony) should correct this dataset.\n"
            "Policy: if you cannot verify from the information provided that the batch effect is separable\n"
            "from biology, the decision is ABSTAIN (uncorrected input). SKIP means no correction is needed.\n"
            "Separability is verified by reference groups (e.g. compounds) replicated across batch groups:\n"
            "their replicates must agree across batches, so better replicate retrieval after correction is\n"
            "evidence that technical variation was removed, not that biology was distorted.\n\n"
            f"Task: {self.task_brief}\n"
            f"Evidence:\n{_json.dumps(dict(evidence), indent=1, default=str)}\n\n"
            f"Deterministic rule outcome: {rule}. Keep it, or turn RUN into ABSTAIN by citing in `concerns`\n"
            "the soft criteria below that hold (a veto citing none that holds is rejected); never turn SKIP\n"
            "or ABSTAIN into RUN. Soft criteria (true = the concern holds; missing evidence holds):\n"
            f"{_json.dumps(dict(concerns), indent=1)}\n"
            f"(few_reference_groups_in_all_batch_groups: fewer than {admission_min_shared_groups(evidence)} reference groups\n"
            "occur in every batch group.) Give a reason code and your reasoning."
        )

    def admit(self, evidence: Mapping[str, Any]) -> dict:
        """The admission rule, the soft concerns and the reasoner's veto (RUN -> ABSTAIN only by
        citing a concern that holds)."""
        rule = admission_rule(evidence)
        concerns = admission_concerns(evidence)
        try:
            verdict = self._ask(self.admission_prompt(evidence, rule, concerns), AdmissionVerdict, "admission")
        except QuestionFailed as exc:         # recorded; the tool is not searched (the uncorrected input)
            return {"decision": "NOT_SEARCHED", "rule": rule, "concerns": concerns, "veto": None,
                    "reasoner": None, "execution_error": self.question_failed(exc)}
        decision, veto = rule, None
        if rule == "RUN" and verdict.decision == "ABSTAIN":
            cited = [c for c in verdict.concerns if concerns.get(c)]
            veto = {"cited": list(verdict.concerns), "holding": cited, "accepted": bool(cited)}
            decision = "ABSTAIN" if cited else "RUN"
            if not cited:
                self.log("veto_rejected", text=f"ABSTAIN cited {verdict.concerns}, none of which holds")
        return {"decision": decision, "rule": rule, "concerns": concerns, "veto": veto,
                "reasoner": verdict.model_dump()}

    # ---- the selection ----------------------------------------------------------------
    def extra_details(self, inp, attempts) -> dict:
        return {"admission": self.admission or {}, "evidence": self.evidence, "noise": self._noise_table(),
                "threads_per_call": self.threads_per_call}

    def select_on_input(self, state, details: dict) -> Optional[WaveSelection]:
        """Selections the admission decides on trial 0: trial 0 failed or no admission -> no output; SKIP
        -> the uncorrected input (trial 0); ABSTAIN -> no output (keep the uncorrected input); RUN -> None
        (the agent's choice, ``AgenticVerifier.select``)."""
        admission = self.admission or {}
        if state.failed(0) or not admission:
            return WaveSelection(None, "conceded", "initial_trial_failed", "ceiling", details)
        if admission["decision"] == "SKIP":
            return WaveSelection(0, "satisfied", "no_batch_effect", None, details)
        if admission["decision"] == "ABSTAIN":
            return WaveSelection(None, "conceded", "separability_unverifiable", "ceiling", details)
        if admission["decision"] == "NOT_SEARCHED":   # the admission question failed: the uncorrected input
            return WaveSelection(None, "conceded", "execution_error", "ceiling", details)
        return None
