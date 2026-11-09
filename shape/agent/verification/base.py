"""VLRR verification primitives for the SHAPE agentic bioimage framework.

VLRR = Verify -> Localize -> Recommend -> Rerun.

Each verifier wraps one tool step. The four VLRR phases map onto methods:
  Verify     -> quantify()   pure deterministic metrics, no LLM
  Localize   -> localize()   decide WHERE the failure lives
                             (this step / upstream / data / tool)
  Recommend  -> adjudicate() decide WHAT to do: deterministic gates + optional
                             LLM reasoning. NOTE the recommendation is one of
                             FOUR actions, not always a parameter tweak:
                             retune_here / conceded (fault=upstream|ceiling) /
                             exclude_samples. Steps with no tunable params
                             (e.g. image loading) never emit retune_here.
  Rerun      -> run()        execute the action and, on a retune, reconcile
                             expected-vs-observed before the next pass.

Design rules:
  - metric computation and threshold gating are deterministic and testable
  - the LLM only enters at the Recommend phase, and only to weigh conflicting
    metrics or infer a root cause; it never computes a metric
  - every retune carries `expected_effect` so the next pass can check whether the
    parameter change did what we predicted. A mismatch invalidates the root-cause
    hypothesis and forces re-localization rather than more of the same tweak.

Integration note (SHAPE):
  The live rerun loop is owned by `shape.agent.verification.step_verifier`, which
  re-runs the real tool through the Executor. This module provides the per-stage
  `BaseVerifier` (quantify/localize/adjudicate + `verify`) and the shared helpers
  (reconcile / apply_bounded / judge_rank) that the loop drives.

VRR migration (task 032): the old `escalate_upstream` / `escalate_human` actions are
replaced by one `conceded` action carrying a `fault` tag. `exclude_samples` is NOT a
concede -- it is a distinct in-loop data-op (drop a few samples, keep going), unchanged.

Fault taxonomy (task 045 refinement): only two values remain, `"upstream"` and
`"ceiling"` -- the old third value `"human"` is RETIRED, folded into `"ceiling"`. A
verifier emits `fault="upstream"` when it is blaming its INPUT (a genuinely fixable,
named prior stage). It emits `fault="ceiling"` when it is blaming ITSELF/the DATA --
the tool or the data cannot do better here, whether because the tool is unfit for
this input (`tool_mismatch`) or the data has no usable structure (`data_intrinsic`);
the site's `notes` carry which. `StepVerifier` (below) ALSO assigns `fault="ceiling"`
itself, for a different reason: its OWN retune/rerun loop stalled locally
(oscillation, no improvement, an invalidated hypothesis, or max iters) without any
verifier ever calling it unfit. Both sources share the same tag because they mean
the same thing operationally to the rest of the loop -- "this stage is as good as
it is going to get; carry its best-effort output forward, do not re-invoke it" --
never a hard stop of the whole run by itself (see `shape.solver`'s D1 invariant).
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
import re
import time
from dataclasses import replace
from typing import Any, Callable, Literal, Optional

from shape.engine.errors import NoEngineError, is_fatal_llm_error
from shape.engine.structured import parse_json_answer
from shape.utils.adaptive_training import early_stopping_patience
from shape.utils.logger import logger
from shape.utils.schemas import (AdmissionAnswer, FinalChoice, QuestionFailed, VLRRResult, WaveCandidate, WavePlan, WaveSelection,
                                 card_lever_fields, lever_spec)

Decision = Literal["PASS", "WARNING", "FAIL"]
Locus = Literal["this_step", "upstream", "data_intrinsic", "tool_mismatch"]
Action = Literal[
    "accept",           # metrics good enough, proceed downstream
    "retune_here",      # adjust this tool's params and rerun
    "conceded",         # this stage's tuning is done without converging; see `fault`
    "exclude_samples",  # drop the offending images/cells, continue with the rest
]
Fault = Literal["ceiling", "upstream"]
Direction = Literal["up", "down", "toward_1", "preserve", "not_increase"]

# Directional tokens understood by `apply_bounded`. Any param_delta value that is
# NOT one of these is treated as a categorical assignment (e.g. an input-strategy
# switch like "use_masked_crop") and passed through verbatim.
_DIRECTIONS = {"up", "down", "toward_1", "preserve", "not_increase"}


class DuplicateConfiguration(ValueError):
    """An equivalent configuration was measured, failed, or is pending.

    Only a finished nonfailed configuration supplies reusable evidence. Keep the
    validation error text stable for callers and distinguish the cases by data.
    """
    def __init__(self, *, calls=(), failed=False, pending=False):
        super().__init__("this configuration was already run (or failed) or is already in this round")
        self.calls, self.failed, self.pending = tuple(calls), bool(failed), bool(pending)

    @property
    def reusable(self):
        return bool(self.calls) and not self.failed and not self.pending


class _MeasuredConfigurationRejection(str):
    """A JSON/prompt-compatible rejection with reusable measured evidence.

    Keep the existing _accept extension signature intact for tool subclasses.
    This marker exists only while validating the current plan; durable call
    references are written separately to reused_configurations.
    """



# `VLRRResult` moved to `shape.utils.schemas` (pre-merge fix, single schema home) --
# imported above. Its field types (`Decision`/`Locus`/`Action`/`Fault`/`Direction`)
# stay defined here, resolved lazily via `from __future__ import annotations`.


# ---- scalar score -----------------------------------------------------------

def normalize(value: float, good: float, bad: float) -> float:
    """Map a raw metric onto 0..1 where `good` -> 1 and `bad` -> 0.

    Works in either direction: if good < bad the metric is "lower is better".
    Clamped to [0, 1].
    """
    if good == bad:
        return 1.0
    x = (value - bad) / (good - bad)
    return max(0.0, min(1.0, x))


def weighted_score(parts: dict[str, float], weights: dict[str, float]) -> float:
    """Combine per-metric normalized scores into one comparable scalar.

    Only keys present in both dicts are used. This single scalar is what lets
    the rerun loop decide whether a parameter change was a NET improvement
    rather than robbing Peter to pay Paul on one metric.
    """
    num = sum(parts[k] * weights[k] for k in parts if k in weights)
    den = sum(weights[k] for k in parts if k in weights)
    return num / den if den else 0.0


# ---- cross-verifier gates -----------------------------------------------------

def batch_dominates_group(batch_silhouette: float, group_silhouette: float,
                          floor: float = 0.20, margin: float = 0.10) -> bool:
    """Whether batch-driven structure dominates group structure: true when
    `batch_silhouette` clears max(floor, group_silhouette + margin). Shared by
    `RepresentationVerifier` and `MorphologicalStateVerifier` (task 055: the two
    verifiers had identical `_BATCH_SILHOUETTE_FLOOR`/`_BATCH_GROUP_MARGIN`
    constants and the same comparison inlined separately; extracted here so
    there is one source of truth for the gate)."""
    return batch_silhouette > max(floor, group_silhouette + margin)


# ---- expected-vs-observed reconciliation ------------------------------------

def moved_as_expected(before: float, after: float, direction: Direction,
                      eps: float = 1e-6) -> bool:
    """Did a metric move the way a parameter change predicted?"""
    d = after - before
    if direction == "up":
        return d > eps
    if direction == "down":
        return d < -eps
    if direction == "not_increase":
        return d <= eps
    if direction == "preserve":
        return abs(d) <= max(eps, 0.2 * abs(before))
    if direction == "toward_1":
        return abs(after - 1.0) < abs(before - 1.0) - eps
    return False


def reconcile(expected: dict[str, Direction],
              before: dict[str, float],
              after: dict[str, float]) -> dict[str, bool]:
    """Check each predicted effect against what actually happened on rerun."""
    return {
        m: moved_as_expected(before[m], after[m], direction)
        for m, direction in expected.items()
        if m in before and m in after
    }


# ---- lever ledger (VRR, task 032) -------------------------------------------
#
# Reconcile-as-diagnostic (vrr.md sec 4): every reconcile pass writes a per-lever
# ledger entry, classifying it as "effective" | "no-headroom" | "inert".
#
# Pre-merge fix (QC N1): `inert` now uses the SAME "did it move" test as the
# `reconcile()` stop-gate above (`moved_as_expected`, the bare 1e-6 epsilon) instead
# of a stricter relative floor -- so `inert` consistently means "produced no
# measurable change" by the loop's own yardstick, not a different one. Because
# `classify_lever` and the stop-gate's `reconcile()` call now run the identical
# check on the identical (prev_vres, vres) pair, whenever the ledger marks a lever
# `inert` the stop-gate's own reconcile ALSO fails to confirm that metric moved --
# so (when `vres.action == "retune_here"`) `StepVerifier` concedes
# (`hypothesis_invalidated`) the SAME iteration, and there is no next iteration in
# which THIS call could re-propose the lever. (Previously the ledger's 2% relative
# floor was STRICTER than the stop-gate's bare epsilon, so a move in between the two
# could be marked `inert` while the stop-gate still called it "moved" and let the
# loop continue -- QC's demonstrated 3-call re-pick.)
#
# A move that clears the bare epsilon but is still a noise-floor wobble (< 2%
# relative -- e.g. task 021's `silhouette_by_batch` 0.503 -> 0.499, ~0.8%) is real
# per the stop-gate but too weak to CREDIT as a win: it must NOT be `inert` (it did
# produce measurable change), so it lands in `no-headroom` instead of `effective`.
#
# NOTE: the ledger remains observational-only this task (see step_verifier.py) --
# nothing reads it to skip/prefer a lever; the within-call property above is a
# consequence of this round's single-lever-per-turn control flow, not a designed
# enforcement. Recommend actively skipping inert levers (incl. across separate
# resolve_stage calls) is a follow-on, once the ledger is wired to Recommend.

_LEDGER_REL_FLOOR = 0.02   # a predicted metric must move >= 2% (relative) to be
                           # credited "effective" -- NOT the inert/no-move gate below.
_LEDGER_SCORE_EPS = 1e-6   # scalar_score must strictly improve (beyond float noise)


def _ledger_materialized(before: float, after: float, direction: Direction) -> bool:
    """A move big enough to CREDIT as "effective" -- a relative-magnitude floor,
    stricter than the bare `moved_as_expected` epsilon used for `inert` (see
    `classify_lever`). Does NOT gate inert/not-inert; only whether an already-
    non-inert move is "effective" or merely "no-headroom" (small but real)."""
    if direction not in ("up", "down"):
        return moved_as_expected(before, after, direction)
    d = after - before
    floor = max(1e-6, _LEDGER_REL_FLOOR * abs(before))
    return d > floor if direction == "up" else d < -floor


def classify_lever(prev_vres, vres, noise: Optional[dict[str, float]] = None,
                   score_noise: Optional[float] = None) -> Optional[str]:
    """Classify the lever(s) `prev_vres` retuned, given `vres`'s resulting metrics
    and score -- one of "effective" | "no-headroom" | "inert", or None when
    no effect was predicted or its before/after evidence is incomplete/nonfinite.

    inert: at least one predicted metric did NOT move per `moved_as_expected` (the
    SAME "did it move" test the loop's stop-gate uses) -- "no measurable change" by
    the loop's own yardstick, consistently (pre-merge fix, QC N1). Otherwise the
    move is real: effective if it ALSO clears the stricter 2% relative floor AND
    `scalar_score` improved; no-headroom if either falls short (a small-but-real
    move, or no net score gain).

    Adaptive floors (optional): a tool whose readouts have a known noise level passes
    `noise` (metric -> absolute noise floor) and `score_noise`. A predicted metric with a
    floor then counts as moved -- and as materialized -- only beyond its own floor, and the
    score must improve beyond `score_noise`; a `preserve` prediction on a metric with a floor
    holds only within that floor; metrics without a floor keep the rules above.
    Without these arguments the behaviour is unchanged.
    """
    if not prev_vres.param_delta or not prev_vres.expected_effect:
        return None
    noise = noise or {}
    items = [(m, direction) for m, direction in prev_vres.expected_effect.items()
             if m in prev_vres.metrics and m in vres.metrics]
    if len(items) != len(prev_vres.expected_effect):
        return None  # a missing readout is not a vacuously materialized prediction
    try:
        if not all(math.isfinite(prev_vres.metrics[m]) and math.isfinite(vres.metrics[m]) for m, _ in items):
            return None
    except (TypeError, ValueError):
        return None

    def moved_beyond_floor(m, direction):
        before, after = prev_vres.metrics[m], vres.metrics[m]
        if m in noise and direction == "preserve":      # "stayed" = within the metric's own floor
            return abs(after - before) <= noise[m]
        return moved_as_expected(before, after, direction, eps=noise.get(m, 1e-6))

    moved = all(moved_beyond_floor(m, direction) for m, direction in items)
    if not moved:
        return "inert"
    materialized = all(m in noise or _ledger_materialized(prev_vres.metrics[m], vres.metrics[m], direction)
                       for m, direction in items)
    eps = _LEDGER_SCORE_EPS if score_noise is None else score_noise
    better = vres.scalar_score > prev_vres.scalar_score + eps
    return "effective" if (materialized and better) else "no-headroom"


def classify_trial(parent_vres, vres, noise: Optional[dict[str, float]] = None,
                   score_noise: Optional[float] = None) -> dict:
    """The wave-mode ledger entry of one trial, measured against the trial it was derived from (its
    `param_delta` / `expected_effect` are the move that produced it): `hypothesis` -- the
    `classify_lever` label of the stated expected effect; `outcome` -- "effective" when the score rose
    beyond `score_noise` (the bare epsilon without it), else "inert"; the score deltas. Labels are None
    for a trial that changed nothing."""
    moved = bool(vres.param_delta)
    hypothesis = classify_lever(replace(parent_vres, param_delta=vres.param_delta,
                                        expected_effect=vres.expected_effect), vres,
                                noise=noise, score_noise=score_noise) if moved else None
    delta = vres.scalar_score - parent_vres.scalar_score
    eps = _LEDGER_SCORE_EPS if score_noise is None else score_noise
    outcome = None if not moved or not math.isfinite(delta) else ("effective" if delta > eps else "inert")
    return {"param_delta": dict(vres.param_delta or {}), "outcome": outcome, "hypothesis": hypothesis,
            "score_delta_vs_parent": delta if math.isfinite(delta) else None}


# ---- base verifier ----------------------------------------------------------

class BaseVerifier:
    """Base class for a per-stage VLRR verifier: subclasses implement
    quantify/localize/adjudicate; `verify()` composes them.

    The bounded retune/rerun loop is NOT here -- it is owned by
    `step_verifier.StepVerifier`, which re-runs the real tool through the Executor.
    (Named BaseVerifier to avoid colliding with the agent's `Verifier` pillar.)
    """

    name: str = "verifier"
    warn_score: float = 0.70
    # The tool this verifier governs. Its CARD.md (`shape/tools/<tool_dir>/CARD.md`) is the
    # declarative tuning manual -- the single source of truth for what each parameter does
    # and how to tune it. `tuning_card()` loads it for ANY verifier that sets a tool_dir;
    # where a vision-LLM judge is used (segmentation) it is injected into the prompt, and
    # for the deterministic stages it documents the recommend the code already performs.
    tool_dir: Optional[str] = None

    def tuning_card(self) -> Optional[str]:
        """Load this verifier's tool CARD.md (cached), or None if it declares no tool_dir."""
        if not getattr(self, "tool_dir", None):
            return None
        if not hasattr(self, "_card_cache"):
            from shape.tools.base import load_tuning_card
            self._card_cache = load_tuning_card(self.tool_dir)
        return self._card_cache

    # -- subclass hooks --------------------------------------------------------

    def quantify(self, tool_output, context) -> dict[str, float]:
        raise NotImplementedError

    def localize(self, metrics, context) -> tuple[Locus, list[str], list[int]]:
        raise NotImplementedError

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        raise NotImplementedError

    # -- wave mode (optional; StepVerifier._resolve_waves) ----------------------
    # A verifier that sets `uses_waves` proposes several candidates per round, each derived from any
    # earlier trial, and chooses the final trial itself; StepVerifier then runs every wave (concurrently)
    # until the budget (`max_iters - 1` rounds) is spent or a wave is empty. Its verify() on the initial result decides whether any
    # wave runs (retune_here) or not (any other action). Every other verifier keeps the
    # one-candidate-per-round loop.
    uses_waves: bool = False

    def recommend_wave(self, state) -> list:
        """The next wave of `WaveCandidate`s for a `WaveState`; [] ends the search."""
        raise NotImplementedError

    def select(self, state, stop_reason: str):
        """The final `WaveSelection` for a `WaveState`."""
        raise NotImplementedError

    def noise_floors(self, vres, context) -> tuple[Optional[dict[str, float]], Optional[float]]:
        """Per-metric and score noise floors for the wave-mode ledger (see `classify_lever`)."""
        return None, None

    # -- shared machinery ------------------------------------------------------

    def verify(self, tool_output, context) -> VLRRResult:
        metrics = self.quantify(tool_output, context)
        locus, triggered, affected = self.localize(metrics, context)
        result = self.adjudicate(metrics, locus, triggered, context)
        result.metrics = metrics
        result.locus = locus
        result.triggered = triggered
        result.affected = affected
        return result


_SEVERITY_RANK = {"none": 0, "minor": 1, "major": 2}


def judge_rank(vres) -> int:
    """The independent judge's per-attempt severity as an ordinal, LOWER = better.

    Used to keep-best and to stop: an attempt the judge found clean ("none") beats one it
    called "major" regardless of mask-shape score. Defaults to 0 when no judge ran, so
    deterministic stages keep ranking by `scalar_score` exactly as before.
    """
    return _SEVERITY_RANK.get(getattr(vres, "judge_severity", None), 0)


# Per-parameter physical bounds and step sizes. Deltas are directional ("up"/
# "down"); the actual numeric step and clamp live here so adjudicate stays
# declarative.
PARAM_BOUNDS: dict[str, tuple[float, float, float]] = {
    # name: (min, max, multiplicative_or_additive_step)
    "diameter":            (5.0, 200.0, 0.8),     # multiplicative
    "cellprob_threshold":  (-6.0, 6.0, 0.5),      # additive
    "flow_threshold":      (0.1, 1.5, 0.1),       # additive
    "min_size":            (1.0, 500.0, 0.667),   # multiplicative ("up" = x1.5)
    "niter":               (50.0, 3000.0, 0.7),   # multiplicative ("up" = x1.43, max iterations)
    "tile_overlap":        (0.0, 0.5, 0.05),       # additive (cellpose tile overlap)
    "correction_strength": (0.0, 2.0, 0.2),       # additive
    "kernel_size":         (3.0, 301.0, 0.7),     # multiplicative
    "padding":             (0.0, 128.0, 8.0),     # additive (pixels)
    "crop_size":           (24.0, 512.0, 0.8),    # multiplicative
    # single_cell_cropper (CARD.md "Levers"; the agentic crop loop names the tool's own kwargs)
    "margin":              (0.0, 128.0, 8.0),     # additive (pixels)
    "margin_frac":         (0.0, 1.0, 0.05),      # additive (fraction of the equivalent diameter)
    "tight_buffer_px":     (0.0, 32.0, 1.0),      # additive (pixels)
    "leiden_resolution":   (0.05, 3.0, 0.7),      # multiplicative
    "n_neighbors":         (5.0, 100.0, 0.667),   # multiplicative ("up" = x1.5)
    # batch_corrector (Harmony + input processing; CARD.md "Levers"). Integer levers are
    # rounded by the caller; a null value is resolved to harmonypy's effective default first.
    "theta":               (0.0, 100.0, 0.5),     # additive
    "sigma":               (1e-4, 10.0, 0.7),     # multiplicative ("up" = x1.43)
    "lambda_value":        (1e-6, 1e3, 0.5),      # multiplicative ("up" = x2)
    "nclust":              (2.0, 10000.0, 0.5),   # multiplicative ("up" = x2)
    "input_pca":           (2.0, 4096.0, 0.5),    # multiplicative ("up" = x2)
    "max_iter_harmony":    (1.0, 200.0, 0.5),     # multiplicative ("up" = x2)
    "epsilon_harmony":     (1e-10, 1.0, 0.1),     # multiplicative ("down" = x0.1)
    "max_iter_kmeans":     (1.0, 200.0, 0.5),     # multiplicative ("up" = x2)
    "epsilon_cluster":     (1e-10, 1.0, 0.1),     # multiplicative ("down" = x0.1)
    "block_size":          (1e-3, 1.0, 0.5),      # multiplicative ("up" = x2)
    "tau":                 (0.0, 1e4, 5.0),       # additive
    "random_state":        (0.0, 2147483647.0, 1.0),  # range only: a seed is never stepped
}

_ADDITIVE = {"cellprob_threshold", "flow_threshold", "correction_strength", "padding",
             "tile_overlap", "theta", "tau", "margin", "margin_frac", "tight_buffer_px"}

# The value a lever acts with when a call's params leave it out or set it null: where a directional step
# of `apply_bounded` starts. Without an entry such a step changes nothing (the next params would equal the
# current ones). The segmenter's CARD defaults:
EFFECTIVE_DEFAULTS: dict[str, float] = {
    "diameter": 30.0,        # Cellpose 4: null = no rescale, the same as 30 (image resized by 30 / diameter)
    "niter": 200.0,          # Cell_Segmenter_Tool: null = 200 dynamics iterations (CARD "niter (default 200)")
    "tile_overlap": 0.1,     # Cell_Segmenter_Tool default (CARD "tile_overlap (default 0.1)")
}


def apply_bounded(params: dict, delta: dict[str, Direction]) -> dict:
    """Apply a directional delta, clamped to physical bounds.

    Two kinds of delta entries are supported:
      - numeric/directional: value is one of `_DIRECTIONS` and the key is in
        `PARAM_BOUNDS`; the param is stepped (additive or multiplicative) and
        clamped to its physical range.
      - categorical: value is NOT a direction token (e.g. "use_masked_crop");
        it is assigned to the key verbatim so a downstream adapter can switch an
        input/strategy that has no numeric bound (see RepresentationVerifier).
        Task 038 follow-up (part B, result-driven per-cell tighten): a categorical
        value may now also be an UNHASHABLE type (e.g. the cropper's own
        `cell_overrides` dict) -- `d in _DIRECTIONS` would raise `TypeError` on an
        unhashable `d` (a real bug this round found), so membership is checked
        defensively; any unhashable value is, by construction, never a direction
        token, so it always falls through to the categorical passthrough below.

    A directional step on a lever the params leave out (or set null) starts from the lever's
    `EFFECTIVE_DEFAULTS` value -- the value the tool then acts with; a lever without one is left as it is.
    """
    out = dict(params)
    for k, d in delta.items():
        # categorical passthrough (non-directional value): set directly
        try:
            is_direction = d in _DIRECTIONS
        except TypeError:
            is_direction = False   # unhashable (e.g. a dict/list) -- never a direction token
        if not is_direction:
            out[k] = d
            continue
        if k not in PARAM_BOUNDS:
            continue
        lo, hi, step = PARAM_BOUNDS[k]
        v = out.get(k)
        if v is None:
            if k not in EFFECTIVE_DEFAULTS:
                continue
            v = EFFECTIVE_DEFAULTS[k]
        if k in _ADDITIVE:
            v = v + step if d == "up" else v - step
        else:
            v = v * (1.0 / step) if d == "up" else v * step
        out[k] = max(lo, min(hi, v))
    return out


# ---- agentic VRR (StepVerifier wave mode) -----------------------------------------------------------------
# The loop every agentically tuned tool shares (each tool's CARD.md, "Agentic VRR"). A tool's verifier
# subclasses `AgenticVerifier` in the tool's own verifier module and adds only what is the tool's own.

BUDGET = 8                   # rounds at most (a round: the calls the agent runs at once); trial 0 is not one
# the backstop stop, early stopping as SSL training's (`adaptive_training.early_stopping_patience`): PATIENCE
# rounds in a row without progress beyond the noise end the search; the agent may stop at any round
PATIENCE = early_stopping_patience(BUDGET)                   # 3 rounds without progress, for 8 rounds
REASONER_ATTEMPTS = 3        # tries of one reasoner question before it counts as failed
REASONER_RETRY_SECONDS = 30  # wait between two tries (a transient gateway error or a malformed answer)
INPUT = "input"              # trial 0 as the agent sees it; the calls are named by role (`AgenticVerifier.label`)
_EXPECTED_DIRECTIONS = ("up", "down", "preserve")


def available_cores() -> int:
    """The cores this process may use (the scheduler's CPU set where the platform exposes it)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def concurrency_cap(cores: int, threads_per_call: int = 1) -> int:
    """Tool calls that may run at the same time on `cores` cores, `threads_per_call` threads each."""
    return max(1, int(cores) // max(1, int(threads_per_call)))


def image_refs(content, label: str = "image") -> list[dict]:
    """The image parts of a model input (a prompt, or a prompt followed by label / image pairs) as
    {label, sha256, bytes}: the text part before an image names it (`label` when only the prompt does); never
    the bytes."""
    refs, name = [], None
    for i, part in enumerate(content if isinstance(content, (list, tuple)) else [content]):
        if isinstance(part, (bytes, bytearray)):
            refs.append({"label": name or label, "sha256": hashlib.sha256(part).hexdigest(), "bytes": len(part)})
            name = None
        elif isinstance(part, str):
            name = part if i else None
    return refs


def reply_record(value):
    """A model reply as returned, JSON-able (a schema object as its fields)."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): reply_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [reply_record(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


DEFAULT, REFERENCE, AGENT, FALLBACK = "default", "reference", "agent", "fallback"   # the roles a call is named by


def config_key(params: Mapping[str, Any]) -> str:
    """A configuration as a comparable key (sorted JSON)."""
    return json.dumps(dict(sorted(params.items())), sort_keys=True, default=str)


def numeric(values: Mapping[str, Any]) -> dict[str, float]:
    """The finite numbers among `values` (bools excluded), as floats."""
    return {k: float(v) for k, v in values.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}


def fmt(value) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _block(text: str) -> str:
    return f"{text}\n\n" if text else ""


def _line(text: str) -> str:
    return f"{text}\n" if text else ""


class AgenticVerifier(BaseVerifier):
    """Agentic VRR through StepVerifier's wave mode: the loop shared by every tool it tunes (each tool's
    CARD.md, "Agentic VRR"). Rules measure and guard; the agent (the reasoner) decides.

    Trial 0 is the input, a measurement and not a tool call (`adjudicate` on trial 0 calls `begin`). A
    tool whose input is measured from the stage's kwargs defines a static ``input_record(base_kwargs)``;
    ``integration.verify_managed_tool`` then runs no first call.

    Admission, before any tuning: whether the tool needs tuning, decided on what the tool delivers untuned
    (`admission`). A tool whose input is itself a valid output declares `admission_on_input`, measures
    the input and admits on trial 0 (batch correction and SSL; their `select_on_input`). A tool with
    `admission_call` runs once at its defaults as round 1 (`admit_defaults`): a measured problem -- the
    call failed, its `guard` or the tool's `admission_problems` -- starts the tuning; without one the
    agent is asked, and a major problem it names starts the tuning, else the call is delivered as it is
    (satisfied, "no_tuning_needed"). It needs trial 0 to be the input (``input_record``), not a call,
    and a budget of at least 2 rounds (the call at the defaults and one round of tuning). After a RUN the
    tool's `reference_moves` (e.g. the tool at a scale the admission measured) run as one system round
    before the agent's first; the agent reads them like any call. The first of them that finished and the
    guard admits is the reference default (`reference_default`), the call delivered unless the agent's final
    choice overrides it (`override_allowed`; the default hook allows any admissible choice).

    All four tools publish the decision with `set_admission`; `tuning_gate` permits search only on RUN.
    Their existing quality and applicability rules stay local, as do the baseline selection and delivery
    on SKIP / ABSTAIN / NOT_SEARCHED. An input preflight may decline even a default call (no objects to
    crop). `details.tuning_gate` records permission and its source alongside the unchanged raw evidence.
    SSL uses the same outer loop: a candidate is a training run, not an epoch. Its `search_stop_reason`
    accounts for training resources; checkpoint lineage, inner early stopping and validation stay local.

    Each round is one VRR cycle: the agent returns the round's calls (Recommend) -- at most the concurrency
    cap, each from any finished call or the input, changing one or several CARD levers --; StepVerifier
    runs them concurrently and verifies each result (Verify); the next prompt shows every call next to its
    parent and the input (Reconcile: the facts, each stated prediction next to the observed change --
    whether a prediction failed is the agent's judgement). The agent stops the search when it is
    satisfied (it returns no moves), at any round. The backstops: a plateau (early stopping, `_stall`:
    `patience` rounds in a row without progress beyond the noise), no valid move left, or the `budget`
    of rounds spent. The agent chooses the final call among the calls the tool's
    `guard` admits; an inadmissible choice is told why and asked once more.

    The rules only measure and guard: validation (the levers of the CARD's `lever_section` with their
    Type / Range, no repeated configuration, one repair question for rejected moves), the guard, the
    budget and the concurrency cap. Exhausted nonfatal reasoner failures are recorded as execution errors;
    no deterministic search replaces the agent. Fundamental engine/authentication/connection errors propagate.
    A failed final question produces no selected candidate; adapters may retain the stage's existing input.

    A tool's subclass, in the tool's own verifier module, provides the measurements (`quantify`,
    `localize`, `adjudicate`), `guard`, its prompt text (`tool_phrase`, `unit`, `final_rule_text`,
    `evidence_text`, `readouts_text`, `input_text`, `calls_legend`, `_rows`, `choice_text`,
    `admissible_text`, ...) and, where it has them: attachments (`round_attachments`,
    `final_attachments`), the noise of its readouts (`_noise`), a move's own part beyond its lever
    changes (`extend_move`), the admission's measured problems and text (`admission_problems`,
    `admission_checks_text`, `admission_attachments`, `record_admission`), the reference calls after a
    RUN admission (`reference_moves`) and what overrides their default (`override_allowed`, `override_text`),
    `least_bad`, `select_on_input`, `extra_details` and `deliver`.
    """

    uses_llm = True
    uses_waves = True
    llm_engine = None
    budget = BUDGET
    patience = PATIENCE
    lever_section = "## Levers"          # the CARD section that declares the levers (shown to the agent)
    threads_per_call = 1                 # threads one tool call uses: the concurrency cap = cores // this
    score_key = "score"                  # the name of `scalar_score` in the prompts and expectations
    keep_input = False                   # the final choice may keep the input (no tool output)
    admission_call = False               # round 1 = the tool at its defaults; the admission is on that call
    admission_on_input = False           # the input already contains the evidence for a required admission
    admission_skip_reason = "no_tuning_needed"
    plateau_after_plan = False           # progress is the agent's own assessment, known only after its plan
    key_readouts: tuple = ()             # readouts compared with the parent and the input, per call
    objectives: tuple = ()               # readouts of the Pareto front (a fact shown to the agent); () = none
    plan_schema = WavePlan
    choice_schema = FinalChoice
    admission_schema = AdmissionAnswer
    tool_phrase = "tool"                 # "You tune a fixed <tool_phrase> for one <unit> by choosing ..."
    unit = "input"
    value_examples = '0.5, 10, true, null, "name"'
    cite_text = "Cite the readouts your choice rests on."
    repair_note = ""                     # appended to the repair question (e.g. what may be left empty)
    narrowing = False                    # opt-in: the agent's own focus_levers, never widened (`_schedule_problem`)
    coverage_exempt = ("seed",)          # CARD Families a first focus does not need changed (`_untried_groups`)
    confirm_stop = True                  # a stop with rounds left is asked once more, with the facts (`stop_check_text`)

    def __init__(self, *, log: Callable[..., None] | None = None, search_strategy: Optional[str] = None, **_ignored):
        self.log = log or self._debug_log
        self.reasoner_failures = 0
        self.execution_errors: list[dict] = []
        self.reused_configurations: list[dict] = []  # evidence references; no new native calls
        self.fallback_rounds = 0
        self.llm_calls: list[dict] = []             # durable, per-question costs; no prompts or replies
        self.stop_checks = 0                        # stops with rounds left that were asked once more
        self.focus_history: list[list[str]] = []    # the focus_levers of each round that ran a call under one
        self._focus_round: int | None = None        # the round (len(rounds) when planned) of the last focus entry
        self._pending_focus: list[str] = []         # the focus_levers of the plan being validated
        self._attempts_seen: dict = {}              # the finished calls of the round prompt being built
        self._focus_too_early = False               # this round's first focus came before the broad search
        self.cores = 1
        self.cap = 1
        self.started: float | None = None
        self.deadline: float | None = None
        self.task_brief = ""
        self.input: dict = {}                       # trial 0's record
        self.admission: dict | None = None          # whether the tool needs tuning (the class docstring)
        self._admission_source: str | None = None
        self.reference_calls: dict[str, dict] = {}  # call -> request: the system's reference calls (`reference_moves`)
        self.default_call: str | None = None        # the reference default (`reference_default`), as last bound
        self.hypotheses: list[dict] = []
        self.plans: list[dict] = []                 # every plan the agent returned, with its round and question
        self.rounds: list[list[int]] = []           # trial indices of each finished round
        self.stop_reason: str | None = None         # set when this verifier ends the search
        self.choice: dict = {}
        self.plan_delivery: dict | None = None      # the call the stopping plan named to deliver (`plan_choice`)
        self.tokens = {"replies": 0, "reported": 0, "total": 0}   # the reported usage of this verifier's replies
        self.reports: dict[str, Any] = {}           # call label -> the tool report the subclass recorded
        self.failed: list[dict] = []
        self._reports: dict[int, dict] = {}         # trial index -> the tool report (e.g. its seconds)
        self._labels: dict[str, int] = {}
        self._names: dict[int, str] = {0: INPUT}      # trial index -> its name (`label`)
        self._proposed: dict[str, tuple] = {}         # config key -> (name, role, request) of a call not bound yet
        self._role_counts: dict[str, int] = {}
        self._counted = 1
        # the levers of the CARD's lever section, in the CARD's order, with their fields
        self.lever_fields = card_lever_fields(self.tool_dir, self.lever_section) if self.tool_dir else {}
        self.levers = list(self.lever_fields)
        from .four_stage import FourStageSearch, staged_schema, strategy_name

        self.search_strategy = strategy_name(self.name, search_strategy)
        self._search = FourStageSearch() if self.search_strategy == "four_stage" else None
        if self._search is not None:
            self.plan_schema = staged_schema(self.plan_schema)

    def _debug_log(self, kind: str, **fields) -> None:
        logger.debug("%s %s: %s", self.name, kind, str(fields.get("text", ""))[:300])

    # ---- trial 0 ----------------------------------------------------------------------
    def begin(self, task: Any = None) -> None:
        """Trial 0: the loop's clock, the cores and the concurrency cap (`concurrency`), the deadline and
        the brief, from the caller's task (any object with `cores`, `deadline`, `brief`; None: the cores
        this process may use, no deadline, no brief)."""
        self.started = time.time()
        cores = getattr(task, "cores", None)
        self.cores = int(cores) if cores else available_cores()
        self.cap = self.concurrency()
        self.deadline = getattr(task, "deadline", None)
        self.task_brief = getattr(task, "brief", "") or ""

    def concurrency(self) -> int:
        """Tool calls that may run at the same time: cores // threads_per_call, at least 1."""
        return concurrency_cap(self.cores, self.threads_per_call)

    @staticmethod
    def make_adapter(base_kwargs):
        """The native tool result is the verifier input; the stage context passes through."""
        return lambda result, upstream: (result, dict(upstream or {}))

    # ---- the CARD -----------------------------------------------------------------------
    def _card_section(self, title: Optional[str] = None) -> str:
        """The CARD's section `title` (default: `lever_section`) -- the H2 line itself, not a mention of it -- up
        to the next H2 ("" without one)."""
        body = self.tuning_card() or ""
        title = title or self.lever_section
        heading = re.search(rf"^{re.escape(title.strip())}[ \t]*$", body, re.MULTILINE)
        if heading is None:
            return ""
        end = body.find("\n## ", heading.start() + 1)
        return body[heading.start():end if end > 0 else None].strip()

    def _lever_doc(self) -> str:
        return self._card_section() or ", ".join(self.levers)

    def lever_spec(self, lever: str) -> tuple[str, Any, bool]:
        """(kind, bounds or choices, nullable) of a CARD lever (`schemas.lever_spec`)."""
        kind, bounds, nullable, _ = lever_spec(self.lever_fields.get(lever) or {})
        return kind, bounds, nullable

    def defaults(self) -> dict:
        """The tool's default configuration, the configuration of a move from "input": each CARD lever's
        Default."""
        return {lever: lever_spec(fields)[3] for lever, fields in self.lever_fields.items()}

    # ---- the reasoner -------------------------------------------------------------------
    def _ask(self, content, schema, label: str):
        """One reasoner question (a prompt, or a prompt followed by attachments), tried REASONER_ATTEMPTS
        times; exhausted nonfatal failures raise QuestionFailed. Fundamental engine, authentication,
        model and connection failures propagate immediately. No deterministic search replaces a failed question."""
        if self.llm_engine is None:
            self.reasoner_failures += 1
            error = NoEngineError(f"{self.name}: no reasoner engine is configured for the {label} question")
            error.shape_question = label
            raise error
        content = [content] if isinstance(content, str) else list(content)
        self.log("prompt", label=label, text=content[0], images=image_refs(content))
        for attempt in range(1, REASONER_ATTEMPTS + 1):
            response = None
            outcome, error, retry = "error", None, False
            started = time.perf_counter()
            elapsed = 0.0
            try:
                try:
                    response = self.llm_engine.generate(content, response_format=schema)
                finally:
                    elapsed = time.perf_counter() - started
                self._count_tokens(response)
                if isinstance(response, dict) and response.get("usage"):
                    self.log("usage", label=label, text=json.dumps(response["usage"], default=str))
                if isinstance(response, dict) and response.get("error") and not response.get("content"):
                    raise RuntimeError(f"{response['error']}: {response.get('message', '')}")
                record = response.get("content") if isinstance(response, dict) else response
                if isinstance(record, str):
                    record = parse_json_answer(schema, record)
                if not isinstance(record, schema):
                    raise TypeError(f"malformed {label} response: {type(record).__name__}")
                self.log("reasoning", label=label, text=record.model_dump_json(indent=1))
                outcome = "ok"
                return record
            except Exception as exc:   # noqa: BLE001 -- retry only nonfatal failures
                error = type(exc).__name__
                # the reply as returned (e.g. a gateway error's message, an invalid answer); None when generate raised
                self.log("reasoner_error", label=label, text=f"try {attempt}: {type(exc).__name__}: {exc}",
                         reply=reply_record(response))
                if is_fatal_llm_error(exc):
                    self.reasoner_failures += 1
                    exc.shape_question = label
                    raise
                if attempt < REASONER_ATTEMPTS:
                    retry = True
                    continue
                self.reasoner_failures += 1
                cause = f"{type(exc).__name__}: {exc}"[:1000]
                raise QuestionFailed(f"{self.name}: the {label} question failed {REASONER_ATTEMPTS} times; "
                                     f"last error: {cause}", label=label, cause=cause) from exc
            finally:
                self.record_llm_call(label, attempt, elapsed, response, content, outcome, error)
                if retry:
                    time.sleep(REASONER_RETRY_SECONDS)

    def record_llm_call(self, label, attempt, seconds, response, content, outcome, error=None):
        """Persist one returned or failed request before retrying; missing usage stays unknown."""
        parts = [content] if isinstance(content, str) else list(content or [])
        reported = response.get("usage") if isinstance(response, dict) else None
        allowed = ("total_tokens", "prompt_tokens", "completion_tokens", "input_tokens", "output_tokens",
                   "cached_tokens", "reasoning_tokens", "thinking_tokens")
        usage = {k: reported[k] for k in allowed if k in reported} if isinstance(reported, dict) else None
        event = {"stage": self.name, "phase": label, "attempt": attempt,
                 "round": len(self.rounds) + 1, "seconds": seconds, "outcome": outcome,
                 "usage": usage, "usage_reported": bool(usage), "error_type": error,
                 "text_characters": sum(len(p) for p in parts if isinstance(p, str)),
                 "attachment_count": len(image_refs(parts))}
        self.llm_calls.append(event)
        self.log("llm_call", label=label, text=json.dumps(event, default=str))

    def question_failed(self, exc: QuestionFailed) -> dict:
        entry = {"status": "execution_error", "question": exc.label, "cause": exc.cause}
        self.execution_errors.append(entry)
        self.log("execution_error", label=exc.label, text=json.dumps(entry, default=str))
        return entry

    def _ask_optional(self, content, schema, label: str):
        """Record a failed question so its caller can stop or retain already accepted work; fatal errors escape."""
        try:
            return self._ask(content, schema, label)
        except QuestionFailed as exc:
            self.question_failed(exc)
            return None

    def _count_tokens(self, response) -> None:
        """Adds a reply's reported total tokens (`tokens`); a reply without one counts as not reported, as does
        0 (an engine that lacks the usage reports 0)."""
        usage = response.get("usage") if isinstance(response, dict) else None
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        self.tokens["replies"] += 1
        if isinstance(total, (int, float)) and not isinstance(total, bool) and total > 0:
            self.tokens["reported"] += 1
            self.tokens["total"] += int(total)

    def tokens_text(self) -> str:
        """The model tokens of this verifier's own questions (plans, admission, final choice) so far, as the engine
        reported them; a tool's other model calls (e.g. vision checks) are not in it."""
        t = self.tokens
        if not t["replies"]:
            return "none yet"
        if not t["reported"]:
            return "not reported"
        missing = t["replies"] - t["reported"]
        return f"{t['total']}" + (f" ({missing} of {t['replies']} replies not reported)" if missing else "")

    # ---- the loop state in the loop's terms ----------------------------------------------
    def trial_config(self, params: Mapping[str, Any]) -> dict:
        """A trial's lever configuration from the params StepVerifier ran."""
        return dict(params)

    def trial_params(self, config: Mapping[str, Any]) -> dict:
        """The params StepVerifier runs for a configuration (the inverse of `trial_config`)."""
        return dict(config)

    def call_error(self, index: int, state) -> Optional[str]:
        """Why trial `index` gave no usable result (a failed call), or None."""
        if not state.failed(index):
            return None
        result = state.trials[index][2]
        return str(result.get("error") or "") if isinstance(result, Mapping) else ""

    # ---- the calls' names: by role ------------------------------------------------------------
    def label(self, index: int, params: Optional[Mapping[str, Any]] = None) -> str:
        """Trial `index` by its name: "input" (trial 0, the input), "default" (the tool at its defaults),
        "reference" (the system's reference call), "agent_<n>" (the agent's n-th call), "fallback_<n>" (the
        deterministic fallback's n-th call); a second default or reference call is "default_2", ... A call proposed
        and not bound yet (running, or just finished: e.g. for a StepVerifier observer or a measurement's log line)
        is named from its tool `params`, else by its place among the pending proposals (StepVerifier runs a wave's
        candidates in order); the bind (`_name_new_trials`) settles it by configuration."""
        if index in self._names:
            return self._names[index]
        proposed = self._proposed.get(self.config_key(self.trial_config(params))) if params is not None else None
        if proposed is None:
            pending, place = list(self._proposed.values()), index - len(self._names)
            proposed = pending[place] if 0 <= place < len(pending) else None
        return proposed[0] if proposed else f"call_{index}"

    def _role(self, request: Mapping[str, Any]) -> str:
        """Who proposed `request`: the tool at its defaults (from the input, no change) is the default call,
        whoever proposed it; else the request's own role (the agent's when unset)."""
        if request.get("parent") == INPUT and not request.get("param_delta"):
            return DEFAULT
        return request.get("role") or AGENT

    def _name(self, role: str) -> str:
        """The next name of `role`: "default" / "reference" for the first, "<role>_<n>" for the others (the agent's
        and the fallback's calls are always numbered)."""
        count = self._role_counts[role] = self._role_counts.get(role, 0) + 1
        return role if role in (DEFAULT, REFERENCE) and count == 1 else f"{role}_{count}"

    def _name_new_trials(self, state) -> None:
        """Binds the trials added since the last bind to the names their proposals got (`_candidate`), by their
        configuration; a reference call joins `reference_calls`."""
        for i in range(len(self._names), len(state.trials)):
            key = self.config_key(self.trial_config(state.trials[i][0]))
            name, role, request = self._proposed.pop(key, (f"call_{i}", AGENT, None))
            self._names[i] = name
            if role == REFERENCE:
                self.reference_calls[name] = request

    def _bind(self, state):
        """The wave state as (trial 0's VLRRResult, attempts): the finished calls by name (`label`), in run order
        (names do not sort in run order: iterate `attempts`, never sort it), as (parent name, configuration,
        VLRRResult); the failed calls, the tool reports, the finished rounds,
        trial 0's record and the reference default on the instance."""
        n = len(state.trials)
        self._name_new_trials(state)
        self._labels = {self.label(i): i for i in range(n)}
        if n > self._counted:                    # the trials added since the last round was planned
            self.rounds.append(list(range(self._counted, n)))
            self._counted = n
        attempts, failed = {}, []
        for i in range(1, n):
            params, vres, _ = state.trials[i]
            config = self.trial_config(params)
            error = self.call_error(i, state)
            if error is not None:
                failed.append({"call": self.label(i), "parent": self.label(state.parents[i]), "params": config,
                               "param_delta": dict(vres.param_delta or {}), "error": error})
            else:
                attempts[self.label(i)] = (self.label(state.parents[i]), config, vres)
        self.failed = failed
        self.reports = {self.label(i): report for i, report in sorted(self._reports.items()) if i < n}
        first = state.trials[0][2]
        self.input = dict(first) if isinstance(first, Mapping) else {}
        self.default_call = self.reference_default(state.trials[0][1], attempts)
        return state.trials[0][1], attempts

    def _params(self, attempts, call: str) -> dict:
        """The configuration of `call`; a move from the input starts at the tool's defaults."""
        return dict(self.defaults() if call == INPUT or call not in attempts else attempts[call][1])

    def config_key(self, params: Mapping[str, Any]) -> str:
        """What makes two configurations the same call (default: equal values)."""
        return config_key(params)

    def search_dimensions(self) -> list[str]:
        """Effective search axes; seed-only variation is not broad parameter evidence."""
        return [k for k in self.levers if k not in self.coverage_exempt
                and self.lever_fields[k].get("Family") not in self.coverage_exempt]

    def search_aliases(self) -> Mapping[str, str]:
        """Names the agent may write in focus_levers/deferred for a search dimension (e.g. the keys inside a
        compound axis); a name that is itself a dimension stays that dimension."""
        return {}

    def search_values(self, params: Mapping[str, Any], call: Optional[str]) -> dict:
        """Comparable parameter values, excluding output paths; tools resolve conditional/extra axes."""
        return {k: params.get(k) for k in self.levers}

    def search_context(self, params: Mapping[str, Any], call: str) -> Optional[dict]:
        """Matching contexts permit a local parameter contrast, never an independent quality claim."""
        return {"input": "same_verifier", "seed": params.get("seed")}

    def search_changed_dimensions(self, delta: Mapping[str, Any]) -> set[str]:
        return set(delta)

    def search_notes(self) -> str:
        return "Use the existing tool readouts and guards; parameter coverage does not establish sensitivity.\n"

    def _seen(self, attempts) -> set[str]:
        """Configurations already run, including calls that failed (they would fail again)."""
        return ({self.config_key(p) for _, p, _ in attempts.values()}
                | {self.config_key(f["params"]) for f in self.failed if f.get("params")})

    def _tried(self, attempts) -> set[str]:
        tried = {lever for _, _, v in attempts.values() for lever in (v.param_delta or {})}
        return tried | {lever for f in self.failed for lever in (f.get("param_delta") or {})}

    # ---- values -------------------------------------------------------------------------
    def coerce(self, lever: str, value: Any) -> Any:
        """One lever value written by the agent, parsed and checked against the lever's CARD Type / Range;
        raises ValueError naming what was received."""
        if lever not in self.lever_fields:
            raise ValueError(f"{lever!r} is not a CARD lever")
        return self.parse_value(lever, value, *self.lever_spec(lever))

    def parse_value(self, name: str, value: Any, kind: str, bounds: Any = None, nullable: bool = False) -> Any:
        """A value of `kind` (number, integer, bool, choice) within `bounds` ((lo, hi), or the choices);
        raises ValueError."""
        if isinstance(value, str):
            value = self.literal(name, value.strip(), kind)
        if value is None:
            if nullable:
                return None
            example = {"integer": "60", "bool": "true"}.get(kind, "1e-05")
            if kind == "choice" and bounds:
                example = str(next(iter(bounds)))
            elif kind in ("integer", "number") and bounds:          # a value the lever accepts: its lower bound
                example = str(int(bounds[0])) if kind == "integer" else f"{bounds[0]:g}"
            raise ValueError(f'{name} cannot be null: write its value inside the value string, e.g. "value": "{example}"')
        if kind == "bool":
            if isinstance(value, bool):
                return value
            raise ValueError(f"{name}: {value!r} is not true or false")
        if kind == "choice":
            if value in tuple(bounds or ()):
                return value
            raise ValueError(f"{name} must be one of {list(bounds or ())}")
        if isinstance(value, bool):
            raise ValueError(f"{name}: {value!r} is not a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {value!r} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{name}: {value!r} is not a finite number")
        if bounds is not None and not bounds[0] <= number <= bounds[1]:
            raise ValueError(f"{name}={number:g} outside [{bounds[0]:g}, {bounds[1]:g}]")
        return int(round(number)) if kind == "integer" else number

    def literal(self, name: str, text: str, kind: str = "number") -> Any:
        """A value written by the agent: a JSON literal, a plain number (".07", "1e-3"), true / false,
        null / none, or (for a choice) a bare choice; anything else is rejected with the text received."""
        try:
            return json.loads(text)
        except ValueError:
            pass
        bare = text.strip('"').strip()
        low = bare.lower()
        if low in ("null", "none"):
            return None
        if low in ("true", "false"):
            return low == "true"
        try:
            return float(bare)
        except ValueError:
            pass
        if kind == "choice":
            return bare
        raise ValueError(f"{name}: {text!r} is not a JSON value (e.g. {self.value_examples})")

    def check_change(self, lever: str, value: Any, base: Mapping[str, Any], parent: str) -> None:
        """A tool's further checks of one change of a move (raise ValueError to reject it); none here."""

    def extend_move(self, move: Mapping[str, Any], params: dict, parent: str, attempts) -> dict:
        """A tool's own part of a move beyond its lever changes (e.g. per-cell rules), after them: may update
        `params` in place (copy a nested value before changing it -- the parent's values are shared) and
        returns the entries it adds to the move's `param_delta`; raise ValueError to reject the move.
        Nothing here."""
        return {}

    def validate(self, move: Mapping[str, Any], attempts, pending: set[str]) -> dict:
        """One move as a request: a known parent, valid lever values that change the parent's
        configuration, a configuration never run, failed or already in this round; raises ValueError. A
        change that repeats the parent's value is dropped (logged), not a rejection of the move."""
        parent = str(move.get("parent") or INPUT)
        if parent != INPUT and parent not in attempts:
            raise ValueError(f"parent {parent!r} is not a finished call or \"input\"")
        base = self._params(attempts, parent)
        params, delta, unchanged = dict(base), {}, []
        for change in move.get("changes") or []:
            lever = change.get("lever")
            if lever in delta or lever in unchanged:
                raise ValueError(f"{lever} is changed twice")
            value = self.coerce(lever, change.get("value"))
            if base.get(lever) == value:
                unchanged.append(lever)
                continue
            self.check_change(lever, value, base, parent)
            params[lever] = delta[lever] = value
        if unchanged:
            self.log("ignored_change", text=f"{unchanged} already at {parent}'s value: dropped from the move")
        extra = self.extend_move(move, params, parent, attempts)
        if not delta and not extra and parent != INPUT:
            already = f" ({', '.join(unchanged)} already at {parent}'s value)" if unchanged else ""
            raise ValueError(f"no lever changed relative to {parent}{already}")
        key = self.config_key(params)
        if key in self._seen(attempts) or key in pending:
            calls = [call for call, (_, known, _) in attempts.items() if self.config_key(known) == key]
            failed = any(self.config_key(f["params"]) == key for f in self.failed if f.get("params"))
            raise DuplicateConfiguration(calls=calls, failed=failed, pending=key in pending)
        expected = {e["metric"]: e["direction"] for e in move.get("expected") or []
                    if e.get("direction") in _EXPECTED_DIRECTIONS}
        return {"parent": parent, "param_delta": {**delta, **extra}, "expected_effect": expected,
                "notes": str(move.get("diagnosis") or "").strip(), "params": params, "config_key": key}

    # ---- measurements -------------------------------------------------------------------
    def _score(self, vres: VLRRResult, key: Optional[str] = None):
        """Readout `key` of `vres` (`score_key`: its scalar_score); None when missing or not finite."""
        key = key or self.score_key
        value = vres.scalar_score if key == self.score_key else (vres.metrics or {}).get(key)
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    def _noise(self, key: str) -> Optional[float]:
        """The noise of a difference in readout `key`; None: no estimate (differences shown as they are)."""
        return None

    def noise_floors(self, vres, context):
        """StepVerifier's ledger floors: the noise of each readout that has an estimate."""
        return ({m: n for m in (vres.metrics or {}) if (n := self._noise(m)) is not None},
                self._noise(self.score_key))

    def guard(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The conditions a final choice fails (empty = admissible); the tool's own."""
        return []

    def admissible(self, inp, attempts) -> list[str]:
        return [c for c, (_, _, v) in attempts.items() if not self.guard(inp, v)]     # in run order

    def pareto(self, attempts) -> set[str]:
        """Finished calls that no other finished call dominates on `objectives` beyond the noise (no worse
        by more than the noise on every objective both report and better by more than it on one)."""
        def pairs(a, b):
            out = []
            for k in self.objectives:
                x, y = self._score(attempts[a][2], k), self._score(attempts[b][2], k)
                if x is not None and y is not None:
                    out.append((x, y, self._noise(k) or 0.0))
            return out

        def dominates(a, b):
            both = pairs(a, b)
            return bool(both) and all(x >= y - n for x, y, n in both) and any(x > y + n for x, y, n in both)

        return {c for c in attempts if not any(dominates(o, c) for o in attempts if o != c)}

    def _delta(self, before: VLRRResult, after: VLRRResult, key: str) -> str | None:
        """`after - before` for one readout, marked beyond or within its noise."""
        a, b = self._score(before, key), self._score(after, key)
        if a is None or b is None:
            return None
        noise = self._noise(key)
        if noise is None:
            return f"{b - a:+.4f}"
        return f"{b - a:+.4f} ({'beyond' if abs(b - a) > noise else 'within'} noise)"

    def _effects(self, before: VLRRResult, after: VLRRResult) -> dict:
        return {k: d for k in self.key_readouts if (d := self._delta(before, after, k)) is not None}

    def _prediction(self, parent: VLRRResult, vres: VLRRResult) -> list[dict]:
        """Each stated expectation next to the observed change from the parent (a measurement; whether a
        prediction failed is the agent's judgement)."""
        out = []
        for metric, direction in (vres.expected_effect or {}).items():
            a, b, noise = self._score(parent, metric), self._score(vres, metric), self._noise(metric)
            if a is None or b is None:
                out.append({"metric": metric, "expected": direction, "observed": "not computable"})
                continue
            d = b - a
            if noise is None:
                seen = "moved" if d else "unchanged"
            elif abs(d) <= noise:
                seen = "within noise"
            else:
                seen = "up" if d > 0 else "down"
            out.append({"metric": metric, "expected": direction, "observed": f"{d:+.4f} ({seen})"})
        return out

    def _stall(self, inp, attempts) -> int:
        """Rounds in a row without progress: the finished rounds after the last round that raised the best
        score (the input and the finished calls) beyond the noise (a round of failed calls counts)."""
        best, stall, noise = self._score(inp), 0, self._noise(self.score_key) or 0.0
        for indices in self.rounds:
            scores = [self._score(attempts[self.label(i)][2]) for i in indices if self.label(i) in attempts]
            top = max((s for s in scores if s is not None), default=None)
            if top is not None and (best is None or top > best + noise):
                best, stall = top, 0
            else:
                stall += 1
        return stall

    def _plateau(self, inp, attempts) -> bool:
        """Whether the search stops on a plateau, the backstop: `_stall` reached `patience` rounds (early
        stopping, as SSL training's, counted in rounds); records the stop."""
        if self._search is not None and not self._search.plateau_ready(self):
            return False
        stall = self._stall(inp, attempts)
        if stall < self.patience:
            return False
        self.stop_reason = "plateau"
        self.log("plateau", text=f"{stall} rounds without progress: {self.plateau_text()}")
        return True

    # ---- conservative selection; legacy move helpers are not used on reasoner failure -------
    def fallback_choice(self, inp, attempts, pool=None) -> Optional[str]:
        """The call in `pool` (default: the admissible calls) with the best score (ties: the earlier call);
        None when the pool is empty."""
        pool = self.admissible(inp, attempts) if pool is None else list(pool)
        if not pool:
            return None

        def rank(call):
            score = self._score(attempts[call][2])
            return (score if score is not None else -math.inf, -self._labels.get(call, 0))
        return max(pool, key=rank)

    def _incumbent(self, inp, attempts) -> str:
        """The admissible call with the best score, else the input (the defaults)."""
        return self.fallback_choice(inp, attempts) or INPUT

    def fallback(self, inp, attempts, n: int, pending: set[str]) -> list[dict]:
        """Deterministic moves, used only when the reasoner is unavailable or fails: the tool at its
        defaults while not run, then the tool's `fallback_moves`; at most `n`."""
        out: list[dict] = []

        def add(parent: str, changes, why: str, **move) -> None:
            if len(out) >= n or changes is None:
                return
            try:
                request = self.validate({"parent": parent, "changes": changes, "diagnosis": f"fallback: {why}",
                                         **move}, attempts, set(pending) | {r["config_key"] for r in out})
            except ValueError:
                return
            out.append({**request, "expected_effect": {self.score_key: "up"}, "role": FALLBACK})

        add(INPUT, [], "the tool at its defaults")
        self.fallback_moves(inp, attempts, add)
        return out

    def fallback_moves(self, inp, attempts, add: Callable[..., None]) -> None:
        """The tool's deterministic moves after the defaults: `add(parent, changes, why)` for each (a change:
        {"lever": ..., "value": <a JSON literal>}); `add` validates and drops what does not fit."""

    # ---- prompts: the tool's text ----------------------------------------------------------
    def task_text(self) -> str:
        return self.task_brief

    def resources_text(self) -> str:
        return (f"Resources: {self.cores} cores; at most {self.cap} calls run at the same time, "
                f"{self.threads_per_call} threads each.")

    def final_rule_text(self) -> str:
        keep = "; if you choose none, or no call qualifies, the input is kept" if self.keep_input else ""
        text = f"Final output: at the end you choose one finished call that the guard admits{keep}."
        rule = self.reference_default_text()
        return f"{text}\n{rule}" if rule else text

    def reference_default_text(self) -> str:
        """The reference default's rule as the prompts state it: the default call and what an override needs
        (`override_text`); "" without a default."""
        default = self.default_call
        if default is None:
            return ""
        need = self.override_text(default)
        if not need:
            return (f"Reference default: {default} (the system's reference call), delivered unless you choose "
                    "another call.")
        return (f"Reference default: {default} (the system's reference call) is delivered unless you choose a "
                f"call that\n{need}; a choice that does not is answered with the reason and asked once more; if "
                f"it\nstill does not, {default} is delivered.")

    def plateau_text(self) -> str:
        return f"{self.patience} rounds in a row have not raised the best {self.score_key} beyond the noise"

    def evidence_text(self) -> str:
        """What the round prompt says about reading the evidence (e.g. the noise); "" = nothing."""
        return ""

    def defaults_note(self) -> str:
        """A line after the tool defaults (e.g. the values the null levers act with); "" = none."""
        return ""

    def move_text(self) -> str:
        schedule = self.schedule_text()
        return self._move_rules() + (f"\n{schedule}" if schedule else "")

    def _move_rules(self) -> str:
        return (
            "A move names a parent (\"input\" = the tool at its defaults, or a finished call id) and the levers it\n"
            "changes relative to the parent's configuration, one or several (from \"input\", no change = the\n"
            f"tool at its defaults). Values are JSON literals (e.g. {self.value_examples}). A configuration\n"
            "already run or failed is rejected. Give each move a diagnosis naming the evidence behind it and\n"
            "the readouts you expect to move (up, down or preserve). Return your hypothesis list with every\n"
            "plan (id, statement, status open | supported | refuted | abandoned, evidence); it is shown back\n"
            "to you next round."
        )

    # ---- the narrowing: the agent's own focus, never widened (opt-in, `narrowing`) ---------------------
    def _previous_focus(self) -> Optional[list[str]]:
        """The focus of the last earlier round that ran a call under one (a focus of this round is a repair's)."""
        history = self.focus_history[:-1] if self._focus_round == len(self.rounds) else self.focus_history
        return history[-1] if history else None

    def _untried_groups(self, attempts) -> dict[str, list[str]]:
        """The lever groups (CARD Family; `coverage_exempt` left out) of which no finished call has changed a lever
        yet (a crashed or rejected call gives no evidence and does not count)."""
        tried = {lever for _, _, vres in attempts.values() for lever in (vres.param_delta or {})}
        groups: dict[str, list[str]] = {}
        for lever in self.levers:
            family = (self.lever_fields.get(lever) or {}).get("Family") or ""
            if family and family not in self.coverage_exempt:
                groups.setdefault(family, []).append(lever)
        return {family: levers for family, levers in groups.items() if not set(levers) & tried}

    def _schedule_problem(self, move: Mapping[str, Any], attempts) -> Optional[str]:
        """Why `move` breaks the narrowing (None = it fits): a first focus needs every lever group changed by a
        call first (broad, then narrow); once a round has run under a focus, every plan names focus_levers, the
        same levers or fewer; a plan that names them changes only those levers. Without a focus, any move fits."""
        previous, focus = self._previous_focus(), self._pending_focus
        if not focus:
            return f"name focus_levers: your previous focus {previous}, or fewer of them" if previous else None
        unknown = sorted(set(focus) - set(self.lever_fields))
        if unknown:
            return f"focus_levers {unknown} are not levers"
        untried = self._untried_groups(attempts) if previous is None else {}
        if untried:
            self._focus_too_early = True
            return (f"narrowing comes after the broad search: no finished call has changed a lever of "
                    f"{sorted(untried)} yet ({untried}); explore them first, with focus_levers empty, several "
                    "levers in one call if you like")
        if previous is not None and not set(focus) <= set(previous):
            return f"focus_levers {focus} widen your previous focus {previous}: keep it or narrow it"
        outside = [c.get("lever") for c in move.get("changes") or [] if c.get("lever") not in focus]
        if outside:
            return f"{outside} not in this round's focus_levers {focus}"
        return None

    def schedule_text(self) -> str:
        """The narrowing's rules as the round prompt states them, the same every round (where the search stands is
        `schedule_now_text`, in the round's part of the prompt); "" without it."""
        if self._search is not None:
            return self._search.prompt(self)
        if not self.narrowing:
            return ""
        groups = sorted({(f.get("Family") or "") for f in self.lever_fields.values()} - {"", *self.coverage_exempt})
        return (
            f"The search goes broad, then narrow; its pace is yours. First explore broadly -- several levers in one\n"
            f"call, from any finished call -- to locate roughly where this {self.unit}'s good settings lie; then narrow\n"
            "when the evidence tells you which levers matter: name them as focus_levers in a plan, and that round's\n"
            "moves change only those levers (one or several, combined as you choose). When to narrow, to which\n"
            "levers and how far is your decision. " + (
                "Two rules hold, and the rules check them: a first focus needs a\n"
                f"lever of every group ({groups}) changed by a finished call first; once a round has run under a focus, every\n"
                if groups else       # a tool without CARD Families: the one rule
                "One rule holds, and the rules check it: once a round has\n"
                "run under a focus, every ") +
            "later plan names focus_levers again -- the same levers or fewer, never a lever added.")

    def schedule_now_text(self) -> str:
        """Where the narrowing stands this round (the round's part of the prompt); "" without it."""
        if self._search is not None or not self.narrowing:
            return ""
        previous = self._previous_focus()
        now = (f"focused on {previous} (focus so far, round by round: {self.focus_history})" if previous
               else "exploring -- no focus named yet")
        untried = self._untried_groups(self._attempts_seen) if previous is None else {}
        return f"Now: {now}" + (f"; groups no call has changed yet: {untried}." if untried else ".")

    def stop_check_text(self, inp, attempts, remaining: int) -> str:
        """The question after a plan with no moves while rounds are left: the facts the stop leaves behind."""
        groups: dict[str, list[str]] = {}
        for lever in self.levers:
            if lever not in self._tried(attempts):
                family = (self.lever_fields.get(lever) or {}).get("Family") or "levers"
                groups.setdefault(family, []).append(lever)
        best = self.fallback_choice(inp, attempts)
        best_text = (f"{best} ({self.score_key} {fmt(self._score(attempts[best][2]))})" if best
                     else "none admissible yet")
        return (
            f"\n\nYou returned no moves: the search ends here ({remaining} rounds remain; the same result in fewer\n"
            f"rounds costs less). The facts before it ends -- lever groups no call has changed yet:\n"
            f"{groups or 'none'}{self._focus_note()}; the best admissible call so far: {best_text}. Return moves if\n"
            "gains are left to find, or no moves to confirm the stop."
        )

    def readouts_text(self) -> str:
        """What the readouts mean; "" = nothing."""
        return ""

    def input_text(self, inp: VLRRResult) -> str:
        """The input (trial 0) as shown to the agent."""
        return f"Input: {json.dumps({k: round(v, 4) for k, v in (inp.metrics or {}).items()}, default=str)}"

    def calls_legend(self) -> str:
        return ("change = relative to the parent; non_default = the full configuration's non-default values;\n"
                "readouts = the call's readouts; vs_parent / vs_input = key readout deltas (beyond or within the\n"
                "noise where it is measured); prediction = each stated expectation with the observed change from\n"
                "the parent; guard = whether it is an admissible final choice; pareto = not dominated on the\n"
                "objectives beyond the noise; tool = the tool's report of the call")

    def _rows(self, inp, attempts) -> list[dict]:
        """One row per finished call for the calls table."""
        front, defaults, rows = self.pareto(attempts) if self.objectives else set(), self.defaults(), []
        for call in attempts:                    # in run order (`_bind`)
            parent, params, vres = attempts[call]
            parent_vres = inp if parent == INPUT else attempts[parent][2]
            fails = self.guard(inp, vres)
            row = {"call": call, "parent": parent, "change": vres.param_delta or "none (the tool at its defaults)",
                   "non_default": {k: v for k, v in params.items() if defaults.get(k) != v},
                   "diagnosis": vres.notes,
                   "readouts": {k: round(v, 4) for k, v in (vres.metrics or {}).items()},
                   "vs_parent": self._effects(parent_vres, vres), "vs_input": self._effects(inp, vres),
                   "prediction": self._prediction(parent_vres, vres),
                   "guard": "admissible" if not fails else "; ".join(fails)}
            if self.objectives:
                row["pareto"] = call in front
            if self.reports.get(call):
                row["tool"] = self.reports[call]
            rows.append(row)
        return rows

    def _table(self, inp, attempts) -> str:
        return "\n".join(json.dumps(r, default=str) for r in self._rows(inp, attempts)) if attempts else "none yet"

    untried_note = ("A reminder to consider whether any of these untried ones warrants review: weigh that with the "
                    "evidence; leaving them untried is a valid choice.")

    def untried_text(self, attempts) -> str:
        """`untried_list` and, once after it, `untried_note`."""
        return f"{self.untried_list(attempts)}\n{self.untried_note}"

    def untried_list(self, attempts) -> str:
        """The levers no call has changed yet, with the narrowing's focus note (a tool may append its own untried
        list)."""
        return (f"Levers no call has changed yet: {[l for l in self.levers if l not in self._tried(attempts)]}"
                f"{self._focus_note()}")

    def _focus_note(self) -> str:
        """Under the narrowing's focus, which levers can still change (the others stay fixed); "" without one."""
        focus = self._previous_focus() if self.narrowing and self._search is None else None
        return f" (under your focus {focus}, only those levers can change from here on)" if focus else ""

    def round_attachments(self, inp, attempts) -> list:
        """Label / image pairs after the round prompt (e.g. the input image, the new calls' outputs)."""
        return []

    def choice_text(self) -> str:
        keep = ", or null to keep the input" if self.keep_input else ""
        return f"Choose the final output of this {self.unit}: one finished call{keep}."

    def admissible_text(self) -> str:
        return "the calls the guard admits"

    def final_evidence_text(self, inp, attempts, admissible: list[str]) -> str:
        """Lines of the final prompt before the hypotheses (e.g. the noise); "" = none."""
        return ""

    def final_attachments(self, inp, attempts, admissible: list[str]) -> list:
        return []

    # ---- shared tuning admission; tools supply the evidence and thresholds ------------------------
    def tuning_gate(self) -> dict:
        """The common search permission, separate from tool-specific admission evidence and delivery.

        Default-call tools assess their first output; input-admission tools assess trial 0. Only RUN
        permits tuning. SKIP retains the tool's baseline policy, while ABSTAIN / NOT_SEARCHED retain
        its existing conservative selection. Legacy verifiers without an admission remain ungated.
        This gate is not the final candidate guard, nor an independent confirmation of quality.
        """
        required = self.admission_call or self.admission_on_input
        source = self._admission_source or (
            "defaults" if self.admission_call else "input" if self.admission_on_input else None)
        decision = (self.admission or {}).get("decision")
        if self.admission is None:
            decision = "PENDING" if required else "NOT_REQUIRED"
        reasons = {"RUN": "tuning_allowed", "SKIP": self.admission_skip_reason,
                   "ABSTAIN": "conditions_unmet", "NOT_SEARCHED": "execution_error",
                   "PENDING": "admission_pending", "NOT_REQUIRED": "admission_not_required"}
        if decision not in reasons or (self.admission is not None and decision in {"PENDING", "NOT_REQUIRED"}):
            raise ValueError(f"invalid tuning admission decision: {decision!r}")
        return {"required": required, "source": source, "decision": decision,
                "allowed": decision in {"RUN", "NOT_REQUIRED"}, "reason": reasons[decision]}

    def set_admission(self, admission: Mapping[str, Any], *, source: Optional[str] = None) -> dict:
        """Record a tool's admission through the shared contract without changing its evidence schema."""
        record = dict(admission)
        if record.get("decision") not in {"RUN", "SKIP", "ABSTAIN", "NOT_SEARCHED"}:
            raise ValueError(f"invalid tuning admission decision: {record.get('decision')!r}")
        if source not in {None, "input", "defaults"}:
            raise ValueError(f"invalid tuning admission source: {source!r}")
        self.admission = record
        self._admission_source = source
        self.log("admission", text=json.dumps(record, default=str))
        return record

    def admission_action(self) -> tuple[str, Optional[str]]:
        """Trial-0 action for input-admission tools; final selection remains tool-specific."""
        gate = self.tuning_gate()
        if gate["decision"] == "RUN":
            return "retune_here", None
        if gate["decision"] == "SKIP":
            return "accept", None
        if gate["decision"] in {"ABSTAIN", "NOT_SEARCHED"}:
            return "conceded", "ceiling"
        raise ValueError("the input admission must be evaluated before choosing its action")

    def search_stop_reason(self) -> Optional[str]:
        """Tool-specific resource stop before a search wave, after admission (e.g. SSL training time).

        Epochs, checkpoints and per-call training caps stay inside the SSL tool/adapter; one training
        run is still one candidate in this shared loop. None leaves the existing round policy intact.
        """
        return None

    # ---- admission on the call at the defaults (`admission_call`) ---------------------------------
    def admission_problems(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The tool's own measured problems of its call at the defaults, beyond the guard ([] = none); each
        starts the tuning. None here."""
        return []

    def admission_checks_text(self) -> str:
        """What the admission measures on the call at the defaults, as its question names it."""
        return f"the guard: {self.admissible_text()}"

    def admission_attachments(self, inp, attempts, call: str) -> list:
        """Label / image pairs after the admission question (default: the round's attachments)."""
        return self.round_attachments(inp, attempts)

    def admission_call_prompt(self, inp, attempts, call: str, remaining: int) -> str:
        return (
            f"Decide whether this {self.unit} needs the tuning of a fixed {self.tool_phrase}.\n"
            f"The tool ran once at its defaults ({call}); its measurements found no problem in that output\n"
            f"({self.admission_checks_text()}). Name the main problem you see in the evidence below, or none,\n"
            f"and rate it none | minor | major: a major problem starts the tuning (a search over the tool's\n"
            f"levers, at most {remaining} more round(s)); minor or none delivers {call} as it is.\n\n"
            f"Task: {self.task_text()}\n"
            f"{_line(self.readouts_text())}{self.input_text(inp)}\n\n"
            f"The call at the defaults ({self.calls_legend()}):\n{self._table(inp, attempts)}"
        )

    def record_admission(self, answer, call: str, attempts) -> None:
        """A tool's use of the agent's admission answer on `call` (e.g. as the call's assessment); none here."""

    def admit_defaults(self, inp, attempts, remaining: int) -> dict:
        """The admission on round 1, the tool's call at its defaults: RUN on a measured problem (the call
        failed, its guard, the tool's `admission_problems`); without one the agent is asked
        (`admission_call_prompt`): a major problem it names turns SKIP into RUN; minor, none or no answer
        keeps SKIP (the call is delivered as it is)."""
        call = self.label(self.rounds[0][0])
        admission = {"call": call, "rule": "SKIP", "problems": [], "answer": None}
        if call in attempts:
            vres = attempts[call][2]
            admission["problems"] = self.guard(inp, vres) + self.admission_problems(inp, vres)
        else:
            error = next((f["error"] for f in self.failed if f["call"] == call), "")
            admission["problems"] = [f"the call at the defaults failed: {error[-300:]}"]
        if admission["problems"]:
            admission["rule"] = "RUN"
        else:
            content = [self.admission_call_prompt(inp, attempts, call, remaining),
                       *self.admission_attachments(inp, attempts, call)]
            answer = self._ask_optional(content, self.admission_schema, "admission")
            if answer is None:
                admission.update(decision="NOT_SEARCHED", execution_error=self.execution_errors[-1])
                return self.set_admission(admission)
            if answer is not None:
                admission["answer"] = answer.model_dump()
                self.record_admission(answer, call, attempts)
        major = (admission["answer"] or {}).get("severity") == "major"
        admission["decision"] = "RUN" if admission["problems"] or major else "SKIP"
        return self.set_admission(admission)

    def admission_text(self) -> str:
        """The round prompt's line on why the tuning started and the reference calls that followed ("" without
        an admission on a call)."""
        a = self.admission or {}
        if not self.admission_call or a.get("decision") != "RUN":
            return ""
        why = "; ".join(a["problems"]) or f"your admission answer, {a['answer']['problem']} (major)"
        text = f"Admission: the tool ran once at its defaults ({a['call']}); the tuning started on {why}."
        if self.reference_calls:
            text += (" Then the system ran: "
                     + "; ".join(f"{c} ({r['notes']})" for c, r in self.reference_calls.items()) + ".")
        return text

    def reference_moves(self, inp, attempts) -> list[dict]:
        """The system's reference calls after a RUN admission, run as one round before the agent's first round
        (moves as the agent writes them: parent, changes, diagnosis), e.g. the tool at a scale the admission
        measured. They are measurements for the agent, counted in the budget; at least one round is left to the
        agent. None here."""
        return []

    def reference_default(self, inp, attempts) -> Optional[str]:
        """The default delivery: the first reference call (in label order) that finished and the guard admits;
        None without one."""
        return next((c for c in self.reference_calls
                     if c in attempts and not self.guard(inp, attempts[c][2])), None)

    def override_allowed(self, inp, attempts, call: str, default: str) -> tuple[bool, str]:
        """Whether the agent's admissible final choice `call` overrides the reference default `default`, and the
        reason (answered to the agent when it does not). Here any admissible choice does."""
        return True, ""

    def override_text(self, default: str) -> str:
        """What an override of the reference default needs, as `override_allowed` checks it ("" = any call)."""
        return ""

    def _reference_round(self, inp, attempts, remaining: int) -> list[WaveCandidate]:
        """The validated `reference_moves` as one round, at most the concurrency cap; none when fewer than 2 of
        the `remaining` rounds are left (the agent keeps at least one). A move that does not validate is logged
        and left out."""
        requests: list[dict] = []
        if remaining < 2:
            return []
        for move in self.reference_moves(inp, attempts)[: self.cap]:
            try:
                requests.append(self._failed_request(move, attempts) if move.get("error") else
                                self.validate(move, attempts, {r["config_key"] for r in requests}))
            except ValueError as exc:
                self.log("reference_rejected", text=f"{exc}: {json.dumps(move, default=str)}")
        for r in requests:
            r["role"] = REFERENCE                 # named when it has run, and then in `reference_calls`
        if requests:
            self.log("reference", text=json.dumps([r["param_delta"] for r in requests], default=str))
        return [self._candidate(r) for r in requests]

    # ---- prompts: the shared frame ---------------------------------------------------------
    def call_seconds(self) -> dict:
        return {c: r.get("seconds") for c, r in self.reports.items() if c != INPUT and r}

    def _time_text(self) -> str:
        hours = (time.time() - self.started) / 3600 if self.started else 0.0
        left = (f"; time left before the stage must end (the job's time limit; calls still running then are lost): "
                f"{max(0.0, self.deadline - time.time()) / 3600:.2f} h" if self.deadline else "")
        return f"Wall time so far: {hours:.2f} h{left}"

    def round_prompt(self, inp, attempts, k_max: int, remaining: int) -> str:
        self._attempts_seen = attempts
        used = len(attempts) + len(self.failed)
        failed = [{"call": f["call"], "parent": f["parent"], "change": f["param_delta"],
                   "error": f["error"][-300:]} for f in self.failed]
        # The text the same for every question of every unit first (a prefix the API can cache), the final rule last
        # in it (a tool's may name the unit's reference default); then the unit's text; then the round's.
        return (
            f"You tune a fixed {self.tool_phrase} for one {self.unit} by\n"
            "choosing its tool calls. Each round you choose the calls to run next; they run concurrently, and\n"
            "you see every result before the next round.\n\n"
            f"{self.resources_text()}\n\n"
            f"{self._lever_doc()}\n\n"
            f"{self.move_text()}\n\n"
            f"{_block(self.readouts_text())}"
            f"{self.final_rule_text()}\n"
            "Stop: return no moves when you are satisfied, at any round; you may then name in deliver the finished\n"
            "call to deliver, which must meet the final rule (otherwise you are asked to choose separately)."
            + self.stop_confirmation_text() +
            f" As a\nbackstop, the loop also stops when the rounds are spent or when {self.plateau_text()}.\n\n"
            f"Task: {self.task_text()}\n"
            f"{_line(self.admission_text())}"
            f"Tool defaults (the configuration of a move from \"input\"): {json.dumps(self.defaults())}\n"
            f"{_line(self.defaults_note())}"
            f"{self.input_text(inp)}\n\n"
            f"{_block(self.evidence_text())}"
            f"Rounds: {len(self.rounds)} done, at most {remaining} more; calls so far: "
            f"{used}. This round: at most {k_max} call(s).\n"
            f"{_line(self.schedule_now_text())}"
            f"{self._time_text()}; seconds of each finished call: "
            f"{self.call_seconds() or 'none yet'}; tokens of your questions on this {self.unit} so far: {self.tokens_text()}.\n"
            f"Your hypotheses so far: {json.dumps(self.hypotheses, default=str) if self.hypotheses else 'none'}\n\n"
            f"Finished calls, one per line ({self.calls_legend()}):\n"
            f"{self._table(inp, attempts)}\n\n"
            f"Calls that did not finish (crashed or rejected by the tool): {failed or 'none'}\n"
            f"{self.untried_text(attempts)}"
        )

    def round_content(self, inp, attempts, k_max: int, remaining: int) -> list:
        return [self.round_prompt(inp, attempts, k_max, remaining), *self.round_attachments(inp, attempts)]

    def repair_text(self, accepted: list[dict], rejected: list[str], room: int) -> str:
        note = self.repair_note
        if self._focus_too_early:                              # the first focus came too early: none this round
            note += " (this round has no focus: return focus_levers empty)"
        elif self.narrowing and self._search is None and "focus_levers" not in note:
            note += " (an empty focus_levers keeps this round's)"
        return (
            f"\n\nSome of your moves for this round were rejected: {rejected}. These moves are "
            f"accepted and run in this round, so do not return them again: "
            f"{[{'parent': r['parent'], 'change': r['param_delta']} for r in accepted] or 'none'}. Return at "
            f"most {room} replacement move(s) for the rejected ones, or none to run only the accepted ones"
            f"{note}."
        )

    def final_prompt(self, inp, attempts, admissible: list[str]) -> str:
        return (
            f"{self.choice_text()}\n"                                  # the tool's fixed text first, as in the round
            f"{self.cite_text}\n"
            f"Task: {self.task_text()}\n"
            f"Admissible calls ({self.admissible_text()}): {admissible}\n"
            f"{_line(self.reference_default_text())}"
            f"{_line(self.final_evidence_text(inp, attempts, admissible))}"
            f"Your hypotheses: {json.dumps(self.hypotheses, default=str) if self.hypotheses else 'none'}\n"
            f"{self.input_text(inp)}\n"
            f"Finished calls, one per line:\n{self._table(inp, attempts)}"
        )

    def final_content(self, inp, attempts, admissible: list[str]) -> list:
        return [self.final_prompt(inp, attempts, admissible), *self.final_attachments(inp, attempts, admissible)]

    # ---- Recommend ----------------------------------------------------------------------
    def move_name(self, move: Mapping[str, Any]) -> str:
        return f"{[(c['lever'], c['value']) for c in move['changes']]} from {move['parent']}"

    def _accept(self, moves, attempts, target: list[dict], cap: int, rejected: list[str],
                pending: set[str] = frozenset()) -> None:
        """Validates the agent's moves into `target`; with the narrowing, each move is first checked against it
        (`_schedule_problem`, rejected with the reason as an invalid value is) and a round that runs a call under a
        focus records it (`focus_history`)."""
        for move in moves:
            data = move.model_dump()
            name = self.move_name(data)
            if len(target) >= cap:
                self.log("dropped_move", text=f"{name}: the round holds at most {cap} call(s)")
                continue
            why = self._schedule_problem(data, attempts) if self.narrowing and self._search is None else None
            if why:
                rejected.append(f"{name}: {why}")
                self.log("rejected_move", text=f"{why}: {move.model_dump_json()}")
                continue
            try:
                request = self.validate(data, attempts, set(pending) | {r["config_key"] for r in target})
                if self._search is not None:
                    self._search.validate_request(self, request, attempts)
                target.append(request)
            except DuplicateConfiguration as exc:
                message = f"{name}: {exc}"
                rejected.append(_MeasuredConfigurationRejection(message) if exc.reusable else message)
                if exc.reusable:
                    self.reused_configurations.append({"round": len(self.rounds), "move": name,
                                                      "known_calls": list(exc.calls), "native_rerun": False})
                self.log("rejected_move", text=f"{exc}: {move.model_dump_json()}")
                continue
            except (ValueError, KeyError, TypeError) as exc:
                rejected.append(f"{name}: {exc}")
                self.log("rejected_move", text=f"{exc}: {move.model_dump_json()}")
                continue
            if self.narrowing and self._search is None and self._pending_focus:
                if self._focus_round == len(self.rounds):
                    self.focus_history[-1] = list(self._pending_focus)
                else:
                    self.focus_history.append(list(self._pending_focus))
                    self._focus_round = len(self.rounds)

    def record_plan(self, plan, label: str, attempts) -> None:
        """Keeps the agent's hypothesis list (when it returns one), the plan's focus_levers for the narrowing (a
        repair's empty focus keeps the round's) and the plan itself (`plans`)."""
        focus = list(getattr(plan, "focus_levers", None) or [])
        if focus or label != "repair" or self._focus_too_early:
            self._pending_focus = focus
        if plan.hypotheses:
            self.hypotheses = [h.model_dump() for h in plan.hypotheses]
        self.plans.append({"round": len(self.rounds) + 1, "question": label, **plan.model_dump()})

    def _prepare_search_plan(self, plan, content, inp, attempts, *, repair=True, keep_pending=False):
        """One repair for an invalid stage declaration; no unvalidated fallback bypasses the policy."""
        if self._search is None:
            return plan
        try:
            self._search.prepare(self, plan, inp, attempts, keep_pending=keep_pending)
            return plan
        except ValueError as exc:
            self.log("rejected_search_plan", text=str(exc))
            if not repair:
                self.reasoner_failures += 1
                self.question_failed(QuestionFailed(str(exc), label="search_repair", cause=str(exc)))
                return None
            phase = getattr(getattr(plan, "search", None), "phase", None)
            repair_schema = self.plan_schema
            focus_rule = ""
            if phase in {"cover", "contrast", "narrow", "confirm"}:
                from .four_stage import search_repair_schema
                repair_schema = search_repair_schema(self.plan_schema)
                focus_rule = (
                    " Follow the broad-to-narrow contract: focus_levers is a narrowing lock, not "
                    "a list of parameters to emphasize. Return it empty for cover/contrast/confirm, or change the "
                    "phase to narrow and use a nonempty focus only when the finished evidence meets all narrow "
                    "prerequisites."
                )
            repair_prompt = (
                f"REPAIR REQUIRED. Invalid search directive: {exc}.{focus_rule} Return one complete corrected "
                "plan that follows the response schema exactly; do not preserve an invalid field. Values must use "
                "valid JSON syntax and types (for example true, not :true).\n\n" + content[0]
            )
            answer = self._ask_optional([repair_prompt, *content[1:]], repair_schema, "search_repair")
            if answer is None:
                return None
            self.record_plan(answer, "search_repair", attempts)
            return self._prepare_search_plan(answer, content, inp, attempts, repair=False,
                                             keep_pending=keep_pending)

    def confirm_stop_required(self, plan, inp, attempts) -> bool:
        return self.confirm_stop

    def stop_confirmation_text(self) -> str:
        return (" A stop\nwith rounds left is asked once more: the deliver of the answer that confirms it counts."
                if self.confirm_stop else "")

    def plan_round(self, inp, attempts, k_max: int, remaining: int) -> list[dict]:
        """The agent's round, validated; one repair question for rejected moves while the round has room.
        A failed question or exhausted invalid moves stops the search without rule-generated calls."""
        content = self.round_content(inp, attempts, k_max, remaining)
        plan = self._ask_optional(content, self.plan_schema, "round")
        if plan is None:
            self.stop_reason = "execution_error"
            return []
        self.record_plan(plan, "round", attempts)
        plan = self._prepare_search_plan(plan, content, inp, attempts)
        if plan is None:
            self.stop_reason = "invalid_search_plan"
            return []
        plateau_next = (self.plateau_after_plan and self._stall(inp, attempts) >= self.patience
                        and (self._search is None or self._search.plateau_ready(self)))
        if not plan.moves and remaining > 0 and not plateau_next and self.confirm_stop_required(plan, inp, attempts):
            self.stop_checks += 1                    # a stop with rounds left: asked once more, with the facts
            check = self._ask_optional([content[0] + self.stop_check_text(inp, attempts, remaining), *content[1:]],
                              self.plan_schema, "stop_check")
            if check is None:
                self.stop_reason = "execution_error"
                return []
            if check is not None:
                self.record_plan(check, "stop_check", attempts)
                plan = self._prepare_search_plan(check, content, inp, attempts)
                if plan is None:
                    self.stop_reason = "invalid_search_plan"
                    return []
        if not plan.moves:
            if self._search is not None:
                self._search.commit(self, stopping=True)
            self.stop_reason = "satisfied"
            if getattr(plan, "deliver", None):                             # `plan_choice` weighs it
                self.plan_delivery = {"call": plan.deliver}
            return []
        if self.plateau_after_plan and self._plateau(inp, attempts):
            return []
        requests: list[dict] = []
        rejected: list[str] = []
        repair = None
        terminal_reuse = False
        self._focus_too_early = False
        self._accept(plan.moves, attempts, requests, k_max, rejected)
        if self._focus_too_early:                # the repair is checked without the rejected first focus
            self._pending_focus = []
        if rejected and len(requests) < k_max:
            room = k_max - len(requests)
            from .balanced_policy import selected_policy, POLICY, repair_packet
            repair_content = (repair_packet(self, plan, requests, rejected, room)
                              if selected_policy() == POLICY else
                              [content[0] + self.repair_text(requests, rejected, room), *content[1:]])
            repair = self._ask_optional(repair_content, self.plan_schema, "repair")
            if repair is not None:
                self.record_plan(repair, "repair", attempts)
                repair = self._prepare_search_plan(repair, content, inp, attempts, repair=False,
                                                   keep_pending=bool(requests))
                if repair is not None:
                    repair_rejected = []
                    self._accept(repair.moves, attempts, requests, k_max, repair_rejected,
                                 {r["config_key"] for r in requests})
                    if repair.moves and not requests:
                        if repair_rejected and all(isinstance(d, _MeasuredConfigurationRejection) for d in repair_rejected):
                            # No model/tool execution failed: the proposed geometry
                            # already has measured evidence. Final selection still
                            # applies the normal admissibility/override guards.
                            terminal_reuse = True
                        else:
                            cause = "; ".join(repair_rejected or rejected)
                            self.reasoner_failures += 1
                            self.question_failed(QuestionFailed(cause, label="repair", cause=cause))
                    elif not repair.moves and not requests and rejected:
                        terminal_reuse = all(isinstance(d, _MeasuredConfigurationRejection) for d in rejected)
        if self._search is not None:
            if requests:
                self._search.commit(self)
            else:
                self.stop_reason = "no_new_configuration" if terminal_reuse else "no_valid_move"
            return requests
        if not requests:
            self.stop_reason = "no_new_configuration" if terminal_reuse else "no_valid_move"
        return requests

    # ---- StepVerifier wave-mode hooks ---------------------------------------------------------
    def _failed_request(self, move: Mapping[str, Any], attempts) -> dict:
        """Keep an undecidable reference as a failed call; do not run a partially decided configuration."""
        parent = str(move.get("parent") or INPUT)
        params, delta = self._params(attempts, parent), {}
        for change in move.get("changes") or []:
            params[change["lever"]] = delta[change["lever"]] = self.coerce(change["lever"], change.get("value"))
        delta.update(move.get("undecided") or {})
        return {"parent": parent, "param_delta": delta, "expected_effect": {}, "params": params,
                "notes": str(move.get("diagnosis") or "").strip(), "config_key": self.config_key(params),
                "error": str(move["error"])}

    def _candidate(self, request: dict) -> WaveCandidate:
        """The request as StepVerifier's candidate, named by its role (`_role`, `_name`) and recorded by its
        configuration: the trial it becomes is bound to the name (`_name_new_trials`)."""
        role = self._role(request)
        self._proposed[request["config_key"]] = (self._name(role), role, request)
        return WaveCandidate(parent=self._labels[request["parent"]], params=self.trial_params(request["params"]),
                             param_delta=dict(request["param_delta"]),
                             expected_effect=dict(request["expected_effect"]), notes=request["notes"],
                             error=request.get("error"))

    def recommend_wave(self, state) -> list[WaveCandidate]:
        """The next round, chosen by the agent: at most the concurrency cap. With `admission_call`, round 1
        is the tool at its defaults and the admission follows it. [] ends the search before the budget:
        no tuning needed, the agent is satisfied, the plateau, or no valid move remains."""
        inp, attempts = self._bind(state)
        gate = self.tuning_gate()
        if self.admission is not None and not gate["allowed"]:
            self.stop_reason = gate["reason"]
            return []
        if self.admission_call and not self.rounds:
            if state.remaining < 2:
                raise ValueError(f"admission_call needs a budget of at least 2 rounds, got {state.remaining}")
            return [self._candidate(self.validate(
                {"parent": INPUT, "diagnosis": "the tool at its defaults (the admission call)"}, attempts, set()))]
        newly_admitted = self.admission_call and self.admission is None
        if newly_admitted:
            self.admission = self.admit_defaults(inp, attempts, state.remaining)
        gate = self.tuning_gate()
        if gate["decision"] == "PENDING":
            raise ValueError("tuning requires an admission on the input before proposing candidates")
        if not gate["allowed"]:
            self.stop_reason = gate["reason"]
            return []
        if self._search is not None:
            self._search.observe(self, inp, attempts)
        resource_stop = self.search_stop_reason()
        if resource_stop is not None:
            self.stop_reason = resource_stop
            return []
        if newly_admitted:
            references = self._reference_round(inp, attempts, state.remaining)
            if references:
                return references
        if not self.plateau_after_plan and self._plateau(inp, attempts):
            return []
        requests = self.plan_round(inp, attempts, self.cap, state.remaining)
        return [self._candidate(r) for r in requests]

    def plan_choice(self, inp, attempts, admissible: list[str]) -> Optional[tuple[str, str]]:
        """The call the agent's stopping plan named to deliver (`plan_delivery`), as its final choice without the
        final question, when it is admissible and, with a reference default, is that default or `override_allowed`
        holds; recorded as the final answer ("question": "round"). Else None: `choose` asks the final question."""
        record = self.plan_delivery
        if record is None:
            return None
        call, default = record["call"], self.reference_default(inp, attempts)
        allowed, why = call in admissible, "" if call in admissible else "not an admissible call"
        override = None
        if allowed and default is not None and call != default:
            allowed, why = self.override_allowed(inp, attempts, call, default)
            override = {"call": call, "allowed": allowed, "reason": why}
        record.update(delivered=allowed, reason=why)
        self.log("plan_delivery", text=json.dumps(record, default=str))
        if not allowed:
            return None
        self.choice = {"answers": [{"call": call, "reasoning": self.plans[-1].get("analysis", ""),
                                    "cited_readouts": [], "question": "round"}]}
        if override:
            self.choice["override"] = override
        return call, "agent_choice"

    def choose(self, inp, attempts, admissible: list[str]) -> tuple[Optional[str], str]:
        """The agent's final choice among the admissible calls: an inadmissible choice is answered with
        the conditions it fails and asked once more; still inadmissible -> the input when `keep_input`,
        else `fallback_choice`. Failed questions produce no selected candidate. With a reference default
        (`reference_default`), a choice other than it holds only when `override_allowed` does: else the hook's
        reason is answered and the agent asked once more (the same retry). The default then replaces the input
        and `fallback_choice` above: a valid answer refused twice -> the default ("reference_default").
        A missing engine or fatal provider error propagates rather than selecting a fallback."""
        default = self.reference_default(inp, attempts)
        content = self.final_content(inp, attempts, admissible)
        self.choice = {"answers": []}
        answer = self._ask_optional(content, self.choice_schema, "final")
        if answer is None:
            return None, "execution_error"
        self.choice = {"answers": [answer.model_dump()]}
        for attempt in range(2):
            kept = answer.call is None and self.keep_input
            if kept or answer.call in admissible:
                how = "agent_kept_input" if kept else "agent_choice"
                if default is None or answer.call == default:
                    return answer.call, how
                allowed, why = self.override_allowed(inp, attempts, answer.call, default)
                self.choice["override"] = {"call": answer.call, "allowed": allowed, "reason": why}
                if allowed:
                    return answer.call, how
                retry = (f"\n\nYour choice {answer.call!r} does not override the reference default {default}: {why}. "
                         f"Choose {default}, or a call that {self.override_text(default)}.")
            else:
                why = ("not a finished call" if answer.call not in attempts
                       else "; ".join(self.guard(inp, attempts[answer.call][2])))
                retry = (f"\n\nYour choice {answer.call!r} is not admissible: {why}. Choose one of {admissible}"
                         f"{', or null' if self.keep_input else ''}.")
            if attempt == 1:
                break
            answer = self._ask_optional([content[0] + retry, *content[1:]], self.choice_schema, "final_retry")
            if answer is None:
                return None, "execution_error"
            self.choice["answers"].append(answer.model_dump())
        if default is not None:
            return default, "reference_default"
        return (None if self.keep_input else self.fallback_choice(inp, attempts, admissible)), "final_choice_inadmissible"

    def select_on_input(self, state, details: dict) -> Optional[WaveSelection]:
        """A selection decided by trial 0 alone (e.g. nothing to do), or None; trial 0 failed -> no output."""
        if state.failed(0):
            return WaveSelection(None, "conceded", "initial_trial_failed", "ceiling", details)
        return None

    def least_bad(self, inp, attempts) -> Optional[str]:
        """The output when no call is admissible (conceded), or None for no output."""
        return None

    def extra_details(self, inp, attempts) -> dict:
        """The tool's own entries of the selection details."""
        return {}

    def deliver(self, state, index: int, details: dict) -> None:
        """Hands the selected trial's output on (e.g. into the directory downstream tools read); nothing here."""

    def finalize_search(self, state, selection: WaveSelection) -> WaveSelection:
        """Summarize actual delivery checks after selection; SSL refreshes this after its frozen fallback."""
        if self._search is not None:
            inp, attempts = self._bind(state)
            self._search.finish(self, state, selection, inp, attempts)
        selection.details["llm_calls"] = list(self.llm_calls)
        from .balanced_policy import selected_policy
        selection.details['vrr_policy'] = selected_policy()
        details = selection.details
        delivered = selection.index is not None
        residual = (details.get('measured_problems') or {}).get(details.get('call'))
        details['delivery_complete'] = delivered
        details['budget_exhausted'] = selection.reason == 'max_iters_reached'
        # "satisfied" is a selection outcome, not a claim of zero QC alerts.
        summary=details.get('delivery_summary')
        details['qc_pass'] = ((summary.get('unresolved')==0) if summary is not None else
                              (delivered and not residual) if residual is not None else None)
        details['accepted_with_flags'] = bool(delivered and residual)
        return selection

    def select(self, state, stop_reason: str) -> WaveSelection:
        return self.finalize_search(state, self._select_result(state, stop_reason))

    def _select_result(self, state, stop_reason: str) -> WaveSelection:
        """The admission's SKIP of the call at the defaults (`admission_call`) -> that call, satisfied; else
        the agent's choice among the admissible calls (``guard``; the stopping plan's delivery, `plan_choice`, else
        `choose`, with the reference default): satisfied, as is the default delivered when no answer was accepted
        ("reference_default"); without a default, an agent choice still inadmissible after one retry -> the input
        (`keep_input`) or `fallback_choice`, conceded; none admissible -> `least_bad`, conceded; no finished call -> no output."""
        inp, attempts = self._bind(state)
        admissible = self.admissible(inp, attempts)
        details = {"cores": self.cores, "cap": self.cap, "rounds": [[self.label(i) for i in r] for r in self.rounds],
                   "verifier_stop": self.stop_reason, "admission": self.admission or {},
                   "tuning_gate": self.tuning_gate(),
                   "hypotheses": list(self.hypotheses), "plans": list(self.plans),
                   "failed": list(self.failed), "admissible": admissible, "reasoner_failures": self.reasoner_failures,
                   "execution_errors": self.execution_errors,
                   "reused_configurations": list(self.reused_configurations),
                   "fallback_rounds": self.fallback_rounds, "stop_checks": self.stop_checks,
                   "focus_history": list(self.focus_history), **self.extra_details(inp, attempts)}
        early = self.select_on_input(state, details)
        if early is not None:
            return early
        if self.admission_call and (self.admission or {}).get("decision") in {"SKIP", "NOT_SEARCHED"}:
            call = self.admission["call"]
            how = "execution_error" if self.admission.get("execution_error") else "no_tuning_needed"
            details.update({"call": call, "choice": how})
            self.deliver(state, self._labels[call], details)
            return WaveSelection(self._labels[call], "conceded" if how == "execution_error" else "satisfied",
                                 how, "ceiling" if how == "execution_error" else None, details)
        reason = self.stop_reason or stop_reason
        if not attempts:
            return WaveSelection(None, "conceded", "no_call_finished", "ceiling", details)
        if not admissible:
            call = self.least_bad(inp, attempts)
            details.update({"call": call, "choice": "no_admissible_call"})
            if call is None:
                return WaveSelection(None, "conceded", "no_admissible_call", "ceiling", details)
            self.deliver(state, self._labels[call], details)
            return WaveSelection(self._labels[call], "conceded", "no_admissible_call", "ceiling", details)
        call, how = self.plan_choice(inp, attempts, admissible) or self.choose(inp, attempts, admissible)
        details.update({"call": call, "choice": how, "final_answers": self.choice.get("answers"),
                        "reasoner_failures": self.reasoner_failures})     # the final question's included
        if self.plan_delivery:                   # the call the stopping plan named, whether it was delivered, why not
            details["plan_delivery"] = dict(self.plan_delivery)
        if self.reference_calls:                 # the reference default, whether the delivery overrode it, why
            default = self.reference_default(inp, attempts)
            details["reference_default"] = {"call": default, "overridden": default is not None and call != default,
                                            "reason": (self.choice.get("override") or {}).get("reason")}
        self.log("choice", text=json.dumps({"chosen": call, "how": how, "admissible": admissible}, default=str))
        if call is None:
            return WaveSelection(None, "conceded", how, "ceiling", details)
        index = self._labels[call]
        self.deliver(state, index, details)
        if how == "final_choice_inadmissible":
            return WaveSelection(index, "conceded", how, "ceiling", details)
        return WaveSelection(index, "satisfied", reason, None, details)
