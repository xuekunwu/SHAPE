"""Opt-in search scheduling and evidence, without changing a tool's measurements or guards.

Switches (values: legacy, four_stage; default: legacy):
    SHAPE_SEARCH_STRATEGY
    SHAPE_SEARCH_STRATEGY_SEGMENTATION
    SHAPE_SEARCH_STRATEGY_CROP
    SHAPE_SEARCH_STRATEGY_BATCH_CORRECTION
    SHAPE_SEARCH_STRATEGY_REPRESENTATION
An explicit verifier constructor argument search_strategy overrides the tool environment setting,
which overrides the global setting. legacy disables this strategy, NOT tuning or the admission gate.
The same settings are inherited by child processes. No mode bypasses a tool's admission or guard.

The reasoner proposes stages through an extension of its existing tool-specific plan schema. Cover
records measured effective changes, not family touch or sensitivity. Contrast checks full candidate
backgrounds; its observed pairs require matching evaluation/training contexts. Narrow needs evidence,
explicit deferrals for untested active axes, a fixed background and bounded numeric ranges. A plan with
no moves (a stop, or confirm) is legal in any stage. Names a tool maps to a dimension (search_aliases) count
as that dimension; a deferred name that is no dimension is recorded, not a deferral. One
evidence-backed reopening is allowed without resetting round/deadline limits. Total budgets always
apply; early stops record stages not entered. This version does not add a held-out evaluator: confirm
records the existing guard/delivery checks and explicitly marks independent validation not performed.

Results live in StageResolution.details['search_strategy']; failed calls do not count as measured
coverage, while measurable guard failures do provide negative evidence. Actual effective values and
comparison contexts are retained; per-cell rules are one compound axis, not individual-cell causality.
"""

from __future__ import annotations

import json
import math
import os
import time
from functools import lru_cache
from itertools import combinations
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model, model_validator


def strategy_name(tool: str, explicit: Optional[str] = None) -> str:
    value = explicit if explicit is not None else os.getenv(
        f"SHAPE_SEARCH_STRATEGY_{tool.upper()}", os.getenv("SHAPE_SEARCH_STRATEGY", "legacy"))
    if value not in {"legacy", "four_stage"}:
        raise ValueError(f"unknown search strategy {value!r}: use legacy or four_stage")
    return value


class DeferredParameter(BaseModel):
    lever: str
    reason: str = Field(min_length=1)


class LocalRange(BaseModel):
    lever: str
    lower: float
    upper: float


class SearchDirective(BaseModel):
    phase: Literal["cover", "contrast", "narrow", "confirm"] = Field(
        description="search stage; focus_levers is a narrowing lock rather than a list of parameters to "
                    "emphasize: it is empty in cover, contrast and confirm and nonempty in narrow")
    reason: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list, description="finished call ids supporting this decision")
    anchor: Optional[str] = Field(None, description="fixed background call for contrast or narrow")
    deferred: list[DeferredParameter] = Field(default_factory=list, description="untested parameters and why deferred")
    ranges: list[LocalRange] = Field(default_factory=list, description="local intervals for numeric focus levers")
    contrast_waiver: str = Field("", description="why narrowing a good combination without attribution is warranted")
    reopen: bool = Field(False, description="at most one evidence-backed return from narrow to cover/contrast")


@lru_cache(maxsize=None)
def staged_schema(schema):
    # Extend the tool's own schema, preserving segmentation assessments and crop cell rules.
    return create_model(f"FourStage{schema.__name__}", __base__=schema,
                        search=(SearchDirective, Field(description="the search stage and its evidence")))


@lru_cache(maxsize=None)
def search_repair_schema(schema):
    """Expose the broad-to-narrow phase/focus contract to a semantic repair response."""
    @model_validator(mode="after")
    def phase_matches_focus(plan):
        phase, focus = plan.search.phase, plan.focus_levers
        if phase == "narrow" and not focus:
            raise ValueError("narrow requires nonempty focus_levers")
        if phase != "narrow" and focus:
            raise ValueError(f"{phase} requires empty focus_levers; focus_levers starts the narrow phase")
        return plan

    return create_model(
        f"SearchRepair{schema.__name__}", __base__=schema,
        __validators__={"phase_matches_focus": phase_matches_focus})


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def changed(a: dict, b: dict) -> set[str]:
    missing = object()
    return {k for k in a.keys() | b.keys() if a.get(k, missing) != b.get(k, missing)}


class FourStageSearch:
    """An optional policy around the existing planner, validator and selection, not another runner."""

    def __init__(self):
        self.phase = "cover"
        self.started = False
        self.phase_round = 0
        self.progress_round = 0
        self.reopens = 0
        self.directive: SearchDirective | None = None
        self.pending: SearchDirective | None = None
        self.focus: set[str] = set()
        self.pending_focus: set[str] = set()
        self.pending_deferred: dict[str, str] = {}
        self.pending_ignored: list[dict] = []
        self.history: list[dict] = []
        self.records: dict[str, dict] = {}
        self.covered: set[str] = set()
        self.controls: list[dict] = []
        self._control_axes: set[str] = set()
        self.deferred: dict[str, str] = {}

    def observe(self, verifier, inp, attempts):
        round_no = len(verifier.rounds)
        if not self.started:
            self.started = True
            self.phase_round = self.progress_round = round_no
        before = set(self.covered)
        for call, (parent, params, result) in attempts.items():
            if call in self.records:
                continue
            values = verifier.search_values(params, call)
            previous = verifier.search_values(verifier._params(attempts, parent), parent)
            axes = (changed(previous, values) & values.keys() & set(verifier.search_dimensions())
                    & verifier.search_changed_dimensions(result.param_delta or {}))
            measured = verifier._score(result) is not None
            if measured:
                self.covered.update(axes)
            self.records[call] = {
                "parent": parent, "phase": self.phase, "values": values,
                "changed": sorted(axes), "measured": measured,
                "inactive": sorted(set(verifier.search_dimensions()) - values.keys()),
                "guard_failures": verifier.guard(inp, result),
                "context": verifier.search_context(params, call),
                "seconds": verifier.reports.get(call, {}).get("seconds"),
            }
        self.controls = []
        for a, b in combinations(self.records, 2):
            left, right = self.records[a], self.records[b]
            if not left["measured"] or not right["measured"]:
                continue
            axes = changed(left["values"], right["values"])
            if (len(axes) != 1 or not axes <= set(verifier.search_dimensions())
                    or left["context"] is None or left["context"] != right["context"]):
                continue
            lever = next(iter(axes))
            self.controls.append({"calls": [a, b], "lever": lever,
                                  "difference": verifier._delta(attempts[a][2], attempts[b][2], verifier.score_key)})
        control_axes = {c["lever"] for c in self.controls}
        if self.covered != before or control_axes - self._control_axes:
            self.progress_round = round_no
        self._control_axes = control_axes

    def prepare(self, verifier, plan, inp, attempts, *, keep_pending=False):
        """Validate a requested transition; commit it only after a candidate has been accepted."""
        directive = plan.search
        if not directive.reason.strip():
            raise ValueError("search.reason must explain this stage")
        known = set(attempts) | {"input"}
        if set(directive.evidence) - known:
            raise ValueError("search.evidence must cite finished calls or input")
        dimensions = set(verifier.search_dimensions())
        aliases = verifier.search_aliases()
        focus = {lever if lever in dimensions else aliases.get(lever, lever) for lever in plan.focus_levers}
        deferrals, ignored = {}, []
        for item in directive.deferred:
            lever = item.lever if item.lever in dimensions else aliases.get(item.lever, item.lever)
            if lever in dimensions and item.reason.strip():
                deferrals.setdefault(lever, item.reason.strip())      # a duplicate keeps the first reason
            else:
                ignored.append(item.model_dump())                   # names no dimension: not a deferral
        if directive.phase == "confirm" and plan.moves:
            raise ValueError("confirm selects an existing result: return no moves")
        if not plan.moves:
            # Stopping (with or without confirm) is always legal; finish() records incomplete stages honestly.
            if directive.anchor is not None and directive.anchor not in known:
                raise ValueError("search.anchor must name a finished call")
            if getattr(plan, "deliver", None) and plan.deliver not in attempts:
                raise ValueError("deliver must name a finished call")
            if keep_pending:
                return
            self.pending, self.pending_focus = directive, set()
            self.pending_deferred, self.pending_ignored = deferrals, ignored
            return
        if keep_pending and self.pending and (directive != self.pending or focus != self.pending_focus):
            raise ValueError("repair must keep the accepted candidates' search directive")
        rank = {"cover": 0, "contrast": 1, "narrow": 2, "confirm": 3}
        if directive.reopen:
            old_evidence = set(self.directive.evidence) if self.directive else set()
            if (self.phase != "narrow" or directive.phase not in {"cover", "contrast"} or self.reopens
                    or not (set(directive.evidence) - old_evidence - {"input"})):
                raise ValueError("reopen needs new finished evidence, from narrow to cover/contrast, at most once")
        elif rank[directive.phase] < rank[self.phase]:
            raise ValueError("returning to a broader stage needs an evidence-backed reopen")
        if focus - dimensions:
            raise ValueError(f"unknown focus dimensions: {sorted(focus - dimensions)}")
        if directive.phase in {"cover", "contrast"} and focus:
            raise ValueError("leave focus_levers empty until narrow")
        if directive.phase in {"contrast", "narrow"}:
            if directive.anchor not in attempts:
                raise ValueError("contrast/narrow needs a finished search.anchor, not unmeasured tool defaults")
            if not (set(directive.evidence) & set(attempts)):
                raise ValueError("contrast/narrow needs finished search.evidence")
        if directive.phase == "narrow":
            anchor = directive.anchor
            if not focus or focus - self.covered:
                raise ValueError("narrow requires nonempty focus_levers with measured changes in earlier calls")
            if verifier.guard(inp, attempts[anchor][2]):
                raise ValueError("narrow anchor must pass the tool's existing guard")
            active = set(verifier.search_values(attempts[anchor][1], anchor)) & dimensions
            missing = active - self.covered - set(deferrals)
            if missing:
                raise ValueError(f"cover or explicitly defer untested active dimensions first: {sorted(missing)}")
            anchor_record = self.records.get(anchor, {})
            local_axes = {control["lever"] for control in self.controls
                          if anchor_record.get("context") is not None and all(
                              self.records[call]["context"] == anchor_record["context"]
                              and changed(self.records[call]["values"], anchor_record["values"]) <= {control["lever"]}
                              for call in control["calls"])}
            if not focus <= local_axes and not directive.contrast_waiver.strip():
                raise ValueError("focus lacks local controls: add contrasts or give contrast_waiver; no attribution claim")
            if self.phase == "narrow" and self.directive:
                if anchor != self.directive.anchor or not focus <= self.focus:
                    raise ValueError("keep the narrow anchor and focus subset; changing background requires reopen")
            self._check_ranges(verifier, directive, focus)
        self.pending, self.pending_focus = directive, focus
        self.pending_deferred, self.pending_ignored = deferrals, ignored

    def _check_ranges(self, verifier, directive, focus):
        ranges = {r.lever: r for r in directive.ranges}
        if len(ranges) != len(directive.ranges) or set(ranges) - focus:
            raise ValueError("local ranges must name distinct focus levers")
        old = {r.lever: r for r in self.directive.ranges} if self.phase == "narrow" and self.directive else {}
        for lever in focus:
            if lever not in verifier.lever_fields:
                continue
            kind, bounds, _ = verifier.lever_spec(lever)
            if kind not in {"number", "integer"}:
                if lever in ranges:
                    raise ValueError(f"{lever} is not numeric")
                continue
            r = ranges.get(lever)
            if r is None or not math.isfinite(r.lower) or not math.isfinite(r.upper) or r.lower > r.upper:
                raise ValueError(f"numeric focus {lever} needs a finite local range")
            if bounds and not (bounds[0] <= r.lower <= r.upper <= bounds[1]):
                raise ValueError(f"local range for {lever} is outside its CARD")
            if bounds and r.lower == bounds[0] and r.upper == bounds[1]:
                raise ValueError(f"narrow the CARD range for {lever}")
            if lever in old and not (old[lever].lower <= r.lower <= r.upper <= old[lever].upper):
                raise ValueError(f"range for {lever} widened without a reopen")

    def validate_request(self, verifier, request, attempts):
        directive = self.pending
        if directive is None:
            raise ValueError("four-stage candidates need a valid search directive")
        if directive.phase == "confirm":
            raise ValueError("confirm cannot run new candidates")
        parent_values = verifier.search_values(verifier._params(attempts, request["parent"]), None)
        candidate_values = verifier.search_values(request["params"], None)
        if ((request["parent"] != "input" or request["param_delta"])
                and not changed(parent_values, candidate_values) & set(verifier.search_dimensions())):
            raise ValueError("the proposed changes are inactive in this configuration")
        if directive.phase == "cover":
            return
        anchor = attempts[directive.anchor][1]
        # Check the complete normalized configuration, including cell rules and checkpoint lineage.
        a = verifier.search_values(anchor, None)
        b = verifier.search_values(request["params"], None)
        axes = changed(a, b)
        if directive.phase == "contrast":
            if len(axes) != 1 or not axes <= set(verifier.search_dimensions()):
                raise ValueError(f"contrast must differ from its anchor in one effective dimension, got {sorted(axes)}")
        elif axes - self.pending_focus:
            raise ValueError(f"changes outside narrow focus (including inherited background): {sorted(axes - self.pending_focus)}")
        if directive.phase == "narrow":
            for r in directive.ranges:
                value = b.get(r.lever)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not r.lower <= value <= r.upper:
                    raise ValueError(f"{r.lever} must stay within [{r.lower}, {r.upper}] during narrow")

    def commit(self, verifier, *, stopping=False):
        if self.pending is None:
            return
        d = self.pending
        if stopping and d.phase != "confirm":
            self.history.append({"round": len(verifier.rounds) + 1, **d.model_dump(), "phase": self.phase,
                                 "requested_phase": d.phase, "action": "stop", "focus": sorted(self.focus),
                                 "ignored_deferred": list(self.pending_ignored)})
            return
        if d.reopen and self.phase == "narrow" and d.phase in {"cover", "contrast"}:
            self.reopens += 1
        if d.phase != self.phase or not self.history:
            self.phase_round = len(verifier.rounds)
        self.phase, self.directive, self.focus = d.phase, d, set(self.pending_focus)
        self.deferred.update(self.pending_deferred)
        self.history.append({"round": len(verifier.rounds) + 1, **d.model_dump(),
                             "action": "stop" if stopping else "wave", "focus": sorted(self.focus),
                             "ignored_deferred": list(self.pending_ignored)})

    def plateau_ready(self, verifier) -> bool:
        start = self.phase_round
        if self.phase in {"cover", "contrast"}:
            start = max(start, self.progress_round)
        return len(verifier.rounds) - start >= verifier.patience

    def prompt(self, verifier) -> str:
        levels: dict[str, set[str]] = {}
        for record in self.records.values():
            if record["measured"]:
                for key, value in record["values"].items():
                    levels.setdefault(key, set()).add(encoded(value))
        return (
            "Four-stage strategy (search): cover -> contrast -> narrow -> confirm. Admission and all tool guards remain binding.\n"
            "focus_levers is a narrowing lock, not a list of parameters to emphasize in this wave: keep it empty in cover,\n"
            "contrast and confirm, nonempty in narrow; once set, it may only shrink.\n"
            "cover: set focus_levers=[]; try diverse effective parameters and combinations; no Cartesian exhaustive search. Track inactive parameters.\n"
            "contrast: keep focus_levers=[]; choose a finished anchor; change one effective dimension relative to its FULL configuration. Reuse prior controls.\n"
            "narrow: cite finished evidence, set a fixed anchor, NONEMPTY focus_levers and smaller ranges for numeric focus levers.\n"
            "Every active untested dimension needs a deferred entry with a reason. Focus dimensions must have measured changes.\n"
            "Without local controls for every focus dimension, give contrast_waiver: a good combination is not parameter attribution.\n"
            "Supporting controls must match the narrow anchor's background and evaluation/training context.\n"
            "A narrow episode keeps its anchor; focus/ranges may shrink, not widen. Once, new finished evidence may justify reopen=true\n"
            "and a return to cover/contrast. Reopening never resets the total budget. Repairs retain an accepted wave's search directive.\n"
            "Early plateau stopping also waits for stage-local patience; in cover/contrast, new measured axes or controls reset it.\n"
            "confirm: set focus_levers=[] and return no moves; existing selection and delivery checks run. This is NOT independent validation.\n"
            "You may stop early on any phase when gains do not justify cost; unfinished stages and untested dimensions stay recorded.\n"
            f"Current phase: {self.phase}; reopens used: {self.reopens}/1; dimensions: {verifier.search_dimensions()}.\n"
            "focus_levers, deferred and ranges name these dimensions; a deferred name that is no dimension is recorded\n"
            f"and ignored (it defers nothing). Names that stand for a dimension: {dict(verifier.search_aliases())}.\n"
            f"Measured changed dimensions: {sorted(self.covered)}; measured value counts (not range coverage): "
            f"{ {k: len(v) for k, v in levels.items()} }.\n"
            f"Local controls (not proof of causality): {json.dumps(self.controls[-12:], default=str)}.\n"
            f"Recent costs, inactive dimensions and contexts: {json.dumps({k: {'seconds': r['seconds'], 'inactive': r['inactive'], 'context': r['context']} for k, r in list(self.records.items())[-4:]}, default=str)}.\n"
            f"Current narrow directive: {self.directive.model_dump_json() if self.directive else 'none'}.\n"
            f"{verifier.search_notes()}"
        )

    def finish(self, verifier, state, selection, inp, attempts):
        self.observe(verifier, inp, attempts)
        index = selection.index
        failures = []
        if index is not None and index > 0:
            failures = verifier.guard(inp, state.trials[index][1])
        admitted = verifier.tuning_gate()["allowed"]
        confirmation = {
            "call": None if index is None else verifier.label(index),
            "status": "no_output" if index is None else "baseline_retained" if index == 0 else
                      "execution_error" if verifier.execution_errors else
                      "guard_failed" if failures else "existing_checks_passed",
            "guard_failures": failures, "independent_validation": "not_performed",
            "scope": "existing tool guard and delivery only; no held-out data was evaluated",
        }
        entered = {r["phase"] for r in self.history if r["action"] == "wave"}
        seconds = [r["seconds"] for r in self.records.values()
                   if isinstance(r["seconds"], (int, float)) and math.isfinite(r["seconds"])]
        selection.details["search_strategy"] = {
            "name": "four_stage", "status": "finished" if admitted and self.history else "not_entered",
            "stop_reason": verifier.stop_reason or selection.reason,
            "last_search_phase": self.phase, "history": list(self.history), "reopens": self.reopens,
            "coverage": {"changed": sorted(self.covered), "deferred": dict(self.deferred),
                         "untested": sorted(set(verifier.search_dimensions()) - self.covered)},
            "calls": dict(self.records), "failed": list(verifier.failed), "controls": list(self.controls),
            "stages_not_entered": [p for p in ("cover", "contrast", "narrow") if p not in entered],
            "confirmation": confirmation,
            "cost": {"rounds": len(verifier.rounds), "calls": len(self.records) + len(verifier.failed),
                     "reported_tool_seconds": sum(seconds),
                     "calls_without_timing": len(self.records) + len(verifier.failed) - len(seconds),
                     "elapsed_seconds": None if verifier.started is None else max(0.0, time.time() - verifier.started),
                     "reasoner_tokens": dict(verifier.tokens)},
        }
