"""Shared VRR data structures: the core loop's `VLRRResult`/`StageResolution` +
the M3 structured-output scaffolding (task 032). Single home for VRR schemas
(pre-merge fix -- `VLRRResult`/`StageResolution` moved here from
`verification/base.py`/`verification/step_verifier.py`, a pure relocation, so
utils is not split from the rest of `shape/utils/`).

Generic across tools (vrr.md's Verify / Reconcile / Recommend), NOT the segmenter-
specific `SegOverlayVerdict` / `RepresentationVerdict`
(`shape.agent.verification.llm_recommend`) -- those stay tool-specific pydantic
models the EXISTING recommend flow already uses; this module is the CONVENTION a
new tool wires onto (referee plan-review N2: laying the convention now that the
whole loop is in scope, not a blocker; a tool's live recommend path migrating onto
it is a separate, later change).

`lever` is deliberately a plain `str`, never a per-tool `Literal[...]`: each tool's
`CARD.md` (`### <lever_name>` headings under "Tunable parameters") is the single
source of truth for its valid levers, so adding a lever means editing the CARD, not
this schema. `validate_lever` checks a decision's `lever` against the tool's CARD
at RUNTIME instead.

Import hygiene (load-bearing): this module imports ONLY stdlib/typing/pydantic --
never `shape.agent.verification.*` -- so the dependency stays verification -> utils,
never the reverse. `VLRRResult`'s field types (`Decision`/`Locus`/`Action`/`Fault`/
`Direction`, the Literal vocabulary aliases) stay defined in `verification/base.py`
where the rest of the VRR vocabulary lives (that module's own helpers -- `reconcile`/
`apply_bounded`/`classify_lever`/`judge_rank`/`BaseVerifier` -- are NOT moved, per the
narrow scope of this relocation); `from __future__ import annotations` makes those
annotations lazy strings here, never evaluated at import time, so no import of them
is needed (verified nothing in `shape/` calls `typing.get_type_hints` on these).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---- core VRR loop data structures (moved from verification/base.py and
# verification/step_verifier.py -- pre-merge fix, single schema home) ----------

@dataclass
class VLRRResult:
    decision: Decision
    scalar_score: float                       # normalized 0..1, higher = better
    metrics: dict[str, float]
    triggered: list[str] = field(default_factory=list)
    locus: Locus = "this_step"
    affected: list[int] = field(default_factory=list)
    action: Action = "accept"
    param_delta: Optional[dict[str, Direction]] = None
    expected_effect: dict[str, Direction] = field(default_factory=dict)
    upstream_target: Optional[str] = None     # which step to escalate to
    # Set iff action == "conceded": "upstream" (see upstream_target) or "ceiling"
    # (task 045: the old third value "human" is retired, folded into "ceiling" --
    # a verifier emits it for its own "tool/data unfit" finding; StepVerifier ALSO
    # assigns "ceiling" itself when its own retune loop stalls locally).
    fault: Optional[Fault] = None
    notes: str = ""                           # LLM reasoning summary
    # The independent (vision-LLM) judge's per-attempt assessment, when it ran. This
    # closes the VLRR loop: the SAME judge that proposes the fix (Recommend) also drives
    # keep-best (Select) and the stop signal (Stop), instead of the deterministic shape
    # score / single metric. None => no judge ran (deterministic stage) -> rank by score.
    judge_severity: Optional[str] = None      # "none" | "minor" | "major"
    judge_problem: str = ""                   # the judge's named problem, for tracing
    # The vision judge's semantic image profile, carried structurally (not buried in the
    # notes string): the verification record keeps it. None on deterministic stages / when no
    # judge ran.
    judge_specimen: Optional[str] = None
    judge_cell_size: Optional[str] = None
    judge_cell_density: Optional[str] = None
    # Optional bounded-lookahead keep-best hint. Zero is the default for every
    # existing verifier; positive values mark candidates that should outrank plain
    # deterministic score when choosing an incumbent.
    selection_priority: int = 0


@dataclass
class StageResolution:
    """Outcome of verifying (and possibly retuning) one pipeline stage."""

    result: Any                       # the accepted/best tool result; None = no admissible output (keep the input)
    status: str                       # satisfied | conceded | exclude_samples
    params: dict                      # the params that produced `result`
    verdict: VLRRResult               # the VLRRResult for `result`
    trials: list = field(default_factory=list)   # [(params, VLRRResult, result), ...]
    upstream_target: Optional[str] = None
    fault: Optional[str] = None       # set iff status == "conceded": ceiling|upstream
    reason: str = ""                  # the original fine-grained reason, for tracing only
    ledger: dict = field(default_factory=dict)   # lever -> untried|effective|no-headroom|inert
    # wave mode only (a verifier with `uses_waves`; empty otherwise): parents[i] = the index of the trial
    # trials[i] was derived from (None for the initial trial); details = stop reason, selected index,
    # per-trial ledger entries and the verifier's selection record
    parents: list = field(default_factory=list)
    details: dict = field(default_factory=dict)


# ---- wave mode (StepVerifier._resolve_waves) ----------------------------------------

@dataclass
class WaveCandidate:
    """One proposal of a wave: the full `params` to run, derived from trial `parent` by `param_delta`,
    with the stated `expected_effect` and the diagnosis (`notes`)."""

    parent: int
    params: dict
    param_delta: dict
    expected_effect: dict = field(default_factory=dict)
    notes: str = ""
    # an execution error that already failed this call before it could run (e.g. the colour rule deciding
    # the system's reference call): StepVerifier records it as a failed call (EXECUTION_ERROR) without running it
    error: Optional[str] = None


@dataclass
class WaveSelection:
    """A wave-mode verifier's final choice: `index` into the trials, or None when no trial is an
    admissible output (the caller keeps the stage input)."""

    index: Optional[int]
    status: str                       # satisfied | conceded
    reason: str = ""
    fault: Optional[str] = None
    details: dict = field(default_factory=dict)


@dataclass
class WaveState:
    """What a wave-mode verifier sees when it proposes a wave or selects: the trials so far (3-tuples,
    as in StageResolution.trials), their parents, StepVerifier's per-trial ledger entries (None for
    the initial trial), the rounds left in the budget (this one included) and the stage context."""

    trials: list
    parents: list
    entries: list
    remaining: int
    context: dict

    def failed(self, index: int) -> bool:
        return is_failed_result(self.trials[index][2])


# The keys of an error result without outputs: `{"error": ...}`, or `Executor.run_managed_tool`'s
# `{"error": ..., "tool_name": ...}` when the tool raised.
_ERROR_ONLY_KEYS = frozenset({"error", "tool_name"})


class ExecutionError(RuntimeError):
    """An in-run vision call failed: a model call that processes an image during a tool call or a measurement
    (the segmenter's preflight image read, the crop mask-error check and triage, a linear verifier's LLM
    tie-breaker, ...). Not a reasoner (decision) failure: StepVerifier records the call it belongs to as failed,
    with the status EXECUTION_ERROR and the cause (as a tool call that failed), and the loop goes on; a stage
    left without a usable call raises."""


EXECUTION_ERROR = "EXECUTION_ERROR"


class QuestionFailed(ExecutionError):
    """A reasoner (decision) question that failed every try for a non-fatal reason (a malformed or unparseable
    answer, a refusal, a timeout; ``shape.engine.errors`` names the fatal class, which raises instead). The verifier
    records it (`label`, `cause`) and goes on: a failed round question ends the search, a failed final question
    delivers no call (the stage's input stays the output), a failed admission question leaves the tool unsearched."""

    def __init__(self, message: str, *, label: str, cause: str):
        super().__init__(message)
        self.label, self.cause = label, cause


def is_failed_result(result: Any) -> bool:
    """A tool call that did not produce an output: marked FAILED by StepVerifier, or an error result with no
    output key (`_ERROR_ONLY_KEYS`). A result that carries outputs next to an "error" field is not failed here;
    a verifier that reads such an error does it itself (e.g. the segmentation verifier's `call_error`)."""
    return isinstance(result, Mapping) and (
        result.get("_attempt_status") == "FAILED" or ("error" in result and set(result) <= _ERROR_ONLY_KEYS))


class VerifyReport(BaseModel):
    """Verify phase: the current output's state + fault (vrr.md sec 2, step 1)."""

    problem: str                                        # named problem, or "none"
    severity: Literal["none", "minor", "major"]
    confidence: Literal["high", "medium", "low"] = "high"
    notes: str = ""


class ReconcileVerdict(BaseModel):
    """Reconcile phase: did the last lever's predicted effect materialize (vrr.md
    sec 2, step 2)? Mirrors `shape.agent.verification.base.classify_lever`'s three
    states -- this schema is for an LLM-DRIVEN reconcile a tool may add later; the
    current architecture keeps Reconcile deterministic (see `classify_lever`)."""

    lever: str
    state: Literal["effective", "no-headroom", "inert"]
    notes: str = ""


class RecommendDecision(BaseModel):
    """Recommend phase: which lever to try next, or concede (vrr.md sec 2, step 3)."""

    lever: str                        # validated against the tool's CARD -- see validate_lever
    value: Optional[str] = None       # the categorical value / directional token for `lever`
    concede: bool = False
    fault: Optional[Literal["ceiling", "upstream"]] = None
    upstream_target: Optional[str] = None
    reasoning: str = ""


# A lever-declaring H3: `### ` + EXACTLY one space + a lower_snake_case identifier, then
# end-of-line, whitespace, or the `(` opening its annotation. Both constraints are load-bearing
# and both fail CLOSED (no match -> no lever, never a fabricated one):
#   - the single space rejects a WRAPPED heading's continuation line, which a CARD indents to
#     align under its own `###`; the old first-token parse turned such a line
#     (`###                safe range [3, 30])`) into a phantom lever named `safe`;
#   - lower_snake_case rejects prose section headings that are not levers at all
#     (`### Threshold / rule classification (...)` -> phantom lever named `Threshold`).
# A legal lever name is exactly a lower_snake_case identifier, because a lever names the tool
# execute() kwarg / verifier gate parameter it drives.
_LEVER_H3 = re.compile(r"^### ([a-z][a-z0-9_]*)(?=$|[\s(])")


def card_levers(tool_dir: str) -> set[str]:
    """The lever names a tool's CARD.md declares tunable (its `### <lever>` H3 headings) --
    the runtime source of truth a `RecommendDecision.lever` is checked against.

    Collected from the WHOLE body, not only a "Tunable parameters" section: single_cell_cropper
    declares 11 of its levers under other H2s. Matched against the RAW line (no strip), so an
    indented `###` inside a fenced code block or list item is ignored.
    """
    from shape.tools.base import load_tuning_card

    body = load_tuning_card(tool_dir) or ""
    return {m.group(1) for line in body.splitlines() if (m := _LEVER_H3.match(line))}


def card_lever_fields(tool_dir: str, section: Optional[str] = None) -> dict[str, dict[str, str]]:
    """{lever: {field: value}} from the first `Key: value | Key: value` line of each lever section
    (the lines after a lever H3 up to the next heading), in the CARD's order. Levers without such
    a line map to {}; nothing is fabricated for a heading that is not a lever. With `section` (an H2
    line, e.g. "## Levers"), only the levers declared under that H2."""
    from shape.tools.base import load_tuning_card

    declared = card_levers(tool_dir)
    out: dict[str, dict[str, str]] = {}
    current = None
    inside = section is None
    for line in (load_tuning_card(tool_dir) or "").splitlines():
        if section is not None and line.startswith("## "):
            inside = line.strip() == section.strip()
        if not inside:
            current = None
            continue
        m = _LEVER_H3.match(line)
        if m and m.group(1) in declared:
            current = m.group(1)
            out[current] = {}
            continue
        if line.startswith("#"):
            current = None
            continue
        if current is not None and not out[current] and ":" in line and "|" in line:
            out[current] = {k.strip(): v.strip() for k, v in
                            (part.split(":", 1) for part in line.split("|") if ":" in part)}
    return out


def validate_lever(lever: str, tool_dir: str) -> bool:
    """Whether `lever` is one of `tool_dir`'s CARD-declared tunable parameters."""
    return lever in card_levers(tool_dir)


def _card_value(token: str) -> Any:
    """One value written in a CARD field: a JSON literal (null, true / false, a number, a quoted
    string), else the bare text."""
    import json

    token = token.strip().strip("`").strip()
    try:
        return json.loads(token)
    except ValueError:
        return token


def lever_spec(fields: Mapping[str, str]) -> tuple[str, Any, bool, Any]:
    """(kind, bounds, nullable, default) of one lever from its CARD fields (`card_lever_fields`):
    - `Type`: number, integer, bool or choice; "... or null" allows null. Without it, a `[lo, hi]`
      Range is a number and a list of values a choice;
    - `Range`: `[lo, hi]` (bounds = (lo, hi)), or the choices separated by commas (bounds = the
      tuple of choices; null, true / false and numbers are JSON literals, anything else a string);
      none: no bounds;
    - `Default`: the tool's default, converted to the kind (None when missing); a null default, or
      null among the choices, also allows null."""
    rng = str(fields.get("Range") or "").strip()
    kind = str(fields.get("Type") or "").strip().lower()
    nullable = kind.endswith(" or null")
    kind = kind[: -len(" or null")].strip() if nullable else kind
    bounds: Any = None
    if rng.startswith("[") and rng.endswith("]"):
        lo, hi = (float(_card_value(v)) for v in rng[1:-1].split(","))
        bounds = (lo, hi)
    elif rng:
        bounds = tuple(_card_value(v) for v in rng.split(","))
    kind = kind or ("choice" if bounds is not None and not rng.startswith("[") else "number")
    default = _card_value(fields["Default"]) if "Default" in fields else None
    if default is not None:
        if kind == "number" and isinstance(default, (int, float)) and not isinstance(default, bool):
            default = float(default)
        elif kind == "integer" and isinstance(default, (int, float)) and not isinstance(default, bool):
            default = int(default)
    nullable = nullable or ("Default" in fields and default is None) or (kind == "choice" and None in (bounds or ()))
    return kind, bounds, nullable, default


# ---- agentic VRR (StepVerifier wave mode; `shape.agent.verification.base.AgenticVerifier`) ----------------
# The agent's answers in the loop every agentically tuned tool shares: the admission (whether the tool's
# call at its defaults needs tuning), a round's plan (its calls, each a parent and the levers it changes,
# with the agent's hypothesis list) and the final choice. A tool may extend them (e.g. a plan with the
# agent's assessment of each attached image) in its own verifier module.

class Expectation(BaseModel):
    metric: str
    direction: Literal["up", "down", "preserve"]


class LeverChange(BaseModel):
    """One lever of a move and its new value. The value is typed (a number, true / false, a category string or
    null) so a structured-output model writes 400 as a number; a string holding a JSON literal ("400") is still
    parsed by the verifier."""
    lever: str
    value: Optional[Union[bool, float, str]] = Field(
        None, description='the new value: a number (e.g. 400, 0.07), true / false, a category (e.g. "zscore") or null')


class WaveMove(BaseModel):
    """One tool call: a parent and the levers it changes relative to the parent's configuration."""
    parent: str = Field("input", description='"input" (the tool\'s defaults) or a finished call id')
    changes: list[LeverChange] = Field(
        default_factory=list, description='the levers changed; none from "input" = the tool at its defaults')
    diagnosis: str = ""
    expected: list[Expectation] = []


class Hypothesis(BaseModel):
    id: str
    statement: str
    status: Literal["open", "supported", "refuted", "abandoned"]
    evidence: str = ""


class WavePlan(BaseModel):
    analysis: str
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="your current hypothesis list")
    focus_levers: list[str] = Field(
        default_factory=list,
        description="the levers this round's moves may change, when you narrow (empty = no focus; in a repair, "
                    "empty keeps this round's focus)")
    moves: list[WaveMove] = Field(description="the calls of this round; none = satisfied, stop")
    deliver: Optional[str] = Field(
        None, description="with no moves: the finished call to deliver; it must meet the final rule")


class FinalChoice(BaseModel):
    """The agent's final call. A tool whose input is itself a valid output (``AgenticVerifier.keep_input``)
    declares a choice whose call may be null."""
    call: str = Field(description="the chosen call id")
    reasoning: str
    cited_readouts: list[str] = Field(default_factory=list, description="the readouts the choice rests on")


class AdmissionAnswer(BaseModel):
    """The agent's admission answer on the tool's call at its defaults (``AgenticVerifier.admission_call``):
    the evidence it weighs, then the main problem it sees in that output and its severity; a major one
    starts the tuning."""
    reasoning: str = Field(description="the evidence in the output you weigh")
    problem: str = Field(description='the main problem you see in the output, or "none"')
    severity: Literal["none", "minor", "major"]
    confidence: Literal["high", "medium", "low"] = "high"
