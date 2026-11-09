#!/usr/bin/env python3
"""Protocol-friction (L4: P1-P4) and efficiency (L5: E1-E3, E6) readouts of the full_* agentic benchmarks, from the
records of existing run directories. No ground truth, no new runs, stdlib only; deterministic (sorted keys, no
timestamps, no absolute paths in the output).

    python agentic_process_readouts.py --run DIR --out OUT.json [--csv OUT.csv] [--md]

`DIR` is an experiment directory (arm run dirs control/ shape/ codex/ claude/ biomni/, rerun dirs shape_rerun_*/, a
failed run dir shape_failed_*/) or the curated results folder of `export_run_records.py` (same layout; the per-image
SEARCH.json trees are replaced by `<arm>/search_summary.json`). Per-image records are preferred when present. A
folder that is not an arm (compare*, posthoc*, logs, analysis) is skipped. The dataset wrappers
(`full_*/runners/process_readouts.py`) call `main()` with their marker name and defaults.

Output: {schema, source, stages, arms: {<run>: {...}}, tables: {<metric>: ...}, notes: [...]}. Each arm run has
`provenance`, `end_state`, `stages.<stage>.{state, reused_from, P1, P2, P3, P4, E1, E2, E3, unavailable}` and `totals`.
E6: a signal that is not available is JSON null and its dotted path (e.g. "E1.reasoning_tokens") maps to a reason
string in the stage's (or totals') `unavailable`; it is never 0. A stage an arm reused from a source run (RERUN.json
`reused_stages`) is marked `reused_from` and excluded from the arm's own totals (totals are reported both ways).

Sources (file, key). SHAPE = arm "shape" (also the reruns); AGENT = codex, claude, biomni; <s> = a stage.
  end state (P1)     RUN_META.json status / current_stage / error; markers CPJUMP1_COMPLETE, LIVECELL_COMPLETE,
                     RERUN_COMPLETE; <s>_COMPLETE.json status; failed run: last exception line of logs/<s>.log or the
                     sanitized FAILURE_DETAIL.txt sidecar when raw logs are not distributed.
  provenance         RUN_META.json provenance.{commit,branch,model,requested_model}, RERUN.json, CONFIG.json seed and
                     reasoning.stages.<s>.{effort,max_tokens,timeout_seconds} (recorded by the reruns only).
  P1 delivered by    AGENT: agent/<s>/SELECTED.json deliveries.<unit>.{source,session,reason} (source agent /
                     default_unnamed / default_session_unfinished). SHAPE image stages: <s>/<img>/SEARCH.json or
                     COHORT_SEARCH.json `delivered` ("agent_N" -> agent, "reference" -> cohort_reference, "default"),
                     else search_summary.json stages.<s>.delivered_kind. SHAPE embed / batch:
                     representation_SEARCH.json / batch_SEARCH.json details.call and details.choice, resolution.
  P2 AGENT           agent/<s>/ROUNDS.json action, outcome (executed, rejected, malformed, failed_round, attached,
                     finished, unfinished), message, request_json (the keys of a rejected request), refused_executions
                     (Biomni), permission_denials (Claude).
  P2 SHAPE           reasoner_failures, fallback_rounds, execution_errors {status, question, cause}: per-image records,
                     else search_summary.json (counts) and SEARCH_WARNINGS.jsonl (causes); for embed / batch the
                     stage record. Question names are the verifier's (repair, search_repair, round, ...).
  P3 AGENT           ROUNDS.json (rounds per session; allowance: 32 session rounds for group / dataset sessions, 8 for
                     a per-image session, 8 tool-call rounds per image or per dataset stage); CALLS.json call entries
                     {params, parent, status, wave, view} (rounds = distinct waves, the default call c0 of an image
                     stage is round 1); embed score = max view.val_history[].sibling_site_map, batch score =
                     calls/<c>/FEEDBACK.json calibration.overall.
  P3 SHAPE           per-image records / search_summary.json: stop_reason, resolution.reason, n_calls (calls except
                     the default); embed / batch: details.{rounds, plans, entries[].score_gain_vs_initial, call,
                     selected}; the calls of an entry are details.rounds flattened (entry i = call i - 1).
  P4 SHAPE           admission (per-image `admission`, representation_SEARCH.json / batch_SEARCH.json admission:
                     decision, concerns held, reasons, veto), details.admissible / failed vs the calls,
                     details.expert_config / expert_rejections / stop_refusals / forced_continuations, execution-error
                     causes naming a contrast, segment_COHORT.json or search_summary.json parameter_source.
  E1 tokens          SHAPE: <s>_USAGE.json (= <s>_COMPLETE.json usage) total_tokens, prompt_tokens (input),
                     completion_tokens (output), reasoning_tokens, cached_tokens; a field outside usage_observed_fields
                     is null. AGENT: the sum over agent/<s>/ROUNDS.json of total_tokens and usage (Codex input_tokens,
                     output_tokens, cached_input_tokens, reasoning_output_tokens; Claude input = input_tokens +
                     cache_creation_input_tokens + cache_read_input_tokens, cached = cache_read_input_tokens,
                     reasoning = model_usage.*.thinkingTokens; Biomni input_tokens and output_tokens only); a field is
                     null unless every round with tokens reports it. The reported stage total is SELECTED.json
                     total_tokens; `check` compares the two.
  E2 seconds         <s>_COMPLETE.json seconds (else RUN_META.json stages.<s>.seconds); share = seconds over the sum of
                     the arm's stage seconds (`share_of_own` over the arm's own, non-reused stages).
  E3 calls           SHAPE llm_calls = usage.calls; tool_calls = calls of the search records (a count that excludes the
                     default call); AGENT tool_calls = SELECTED.json n_calls (it includes the default call c0 / input),
                     agent_rounds = n_rounds, agent_turns = sum of ROUNDS num_turns (Claude) or steps (Biomni);
                     per-image median / p90 (nearest-rank) from the per-image records or CALLS.json when present.
Not applicable by design: control (no agent, no LLM) and the cluster stage (no agentic decision, no LLM call).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "agentic-process-readouts/1"
STAGES = ("segment", "crop", "embed", "batch", "cluster")
IMAGE_STAGES = ("segment", "crop")
LOOP_STAGES = ("embed", "batch")
MARKERS = ("CPJUMP1_COMPLETE", "LIVECELL_COMPLETE", "RERUN_COMPLETE", "SMOKE_COMPLETE")
NON_ARM_PREFIXES = ("compare", "comparison", "posthoc", "logs", "analysis")
AGENT_ARMS = ("codex", "claude", "biomni")
TOOL_ROUND_ALLOWANCE = 8               # shape.agent.verification.base.BUDGET = codex_arm.MAX_TOOL_ROUNDS
SESSION_ROUND_CAP = {"group": 32, "dataset": 32}     # codex_arm.MAX_ROUNDS; a per-image session has IMAGE_ROUNDS = 8
PER_IMAGE_SESSION_CAP = 8
NOT_A_TUNING_CALL = ("input", "default", "c0")
TRUNC = 160
TOP = 5
TOKEN_FIELDS = ("total", "input", "output", "reasoning", "cached")
PATH_RE = re.compile(r"(?<![\w.])(/[\w.\-]+){2,}/([\w.\-]+)")
CONTRAST_RE =re.compile(r"contrast|local controls|focus lacks", re.I)
TIMEOUT_RE = re.compile(r"deadline|timed out|timeout|budget exhausted", re.I)
CONNECTION_RE = re.compile(r"APIConnectionError|Connection error|gateway|Bad Gateway|Service Unavailable", re.I)
EXC_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Timeout)[\w]*:")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _read(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _trunc(text, n: int = TRUNC) -> str:
    """One line, absolute paths reduced to their last component (the output carries no machine paths), cut at n."""
    text = PATH_RE.sub(r".../\2", " ".join(str(text).split()))
    return text if len(text) <= n else text[: n - 3] + "..."


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _round(x, digits: int = 6):
    return None if x is None else round(x, digits)


def _median(values):
    v = sorted(values)
    if not v:
        return None
    mid = len(v) // 2
    m = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return int(m) if float(m).is_integer() else _round(m, 4)


def _p90(values):
    """Nearest-rank 90th percentile."""
    v = sorted(values)
    return v[max(0, math.ceil(0.9 * len(v)) - 1)] if v else None


def _spread(values) -> dict:
    return {"median": _median(values), "p90": _p90(values), "max": max(values) if values else None}


def _expand(hist: dict) -> list:
    """A {value: count} histogram (string keys allowed) as a list of numbers."""
    out = []
    for k, n in hist.items():
        out += [float(k) if "." in str(k) else int(k)] * int(n)
    return out


def _counts(counter) -> dict:
    return {str(k): v for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _top(counter, n: int = TOP) -> list:
    return [{"text": k, "count": v} for k, v in
            sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def _normalise_cause(text) -> str:
    text = re.sub(r"agent_\d+", "agent_N", str(text))
    text = re.sub(r"^\[.*?\]\s*", "[...] ", text)
    return _trunc(text)


class Rec:
    """A nested record whose unavailable (None) leaves carry a reason (E6)."""

    def __init__(self):
        self.data: dict = {}
        self.why: dict = {}

    def put(self, path: str, value, reason: str | None = None):
        node = self.data
        parts = path.split(".")
        for k, p in enumerate(parts[:-1]):
            if not isinstance(node.get(p), dict):      # an earlier put() left a leaf (None, with its reason) here
                node[p] = {}
                self.why.pop(".".join(parts[:k + 1]), None)
            node = node[p]
        node[parts[-1]] = value
        if value is None:
            self.why[path] = reason or "not available"
        else:
            self.why.pop(path, None)
        return value

    def putd(self, path: str, mapping: dict, reason: str):
        """put() every top-level key of `mapping`; a None leaf gets `reason`."""
        for k, v in mapping.items():
            self.put(f"{path}.{k}", v, reason)

    def na(self, prefix: str, names, reason: str):
        for n in names:
            self.put(f"{prefix}.{n}", None, reason)

    def out(self) -> dict:
        return {**self.data, "unavailable": dict(sorted(self.why.items()))}


# --------------------------------------------------------------------------- #
# arm discovery, provenance, end state
# --------------------------------------------------------------------------- #
def discover_arms(run: Path, stages=STAGES) -> list[Path]:
    arms = []
    for d in sorted((p for p in run.iterdir() if p.is_dir()), key=lambda p: p.name):
        if d.name.startswith(NON_ARM_PREFIXES):
            continue
        if any((d / f).is_file() for f in ("RESULT.json", "RUN_META.json") + MARKERS) or any(
                (d / f"{s}_COMPLETE.json").is_file() for s in stages):
            arms.append(d)
    return arms


def arm_name(d: Path) -> str:
    for f in ("RUN_META.json", "CONFIG.json", "RESULT.json"):
        a = _get(_read(d / f), "arm")
        if a:
            return str(a)
    return d.name.split("_")[0]


def arm_kind(arm: str) -> str:
    return "control" if arm == "control" else "shape" if arm == "shape" else "agent"


def reused_stages(d: Path) -> dict:
    """{stage: source run name} for the stages a rerun took from its source run."""
    rerun = _read(d / "RERUN.json")
    if isinstance(rerun, dict) and rerun.get("reused_stages"):
        source = Path(str(rerun.get("source_run") or rerun.get("recorded_source_root") or "?")
                      .replace("\\", "/")).name
        return {s: source for s in rerun["reused_stages"]}
    code = _get(_read(d / "RUN_META.json"), "stage_code")
    if isinstance(code, dict):
        return {s: "source run" for s, v in code.items() if isinstance(v, dict) and v.get("run") == "source"}
    return {}


def provenance(d: Path, stages=STAGES) -> dict:
    meta = _read(d / "RUN_META.json") or {}
    result = _read(d / "RESULT.json") or {}
    config = _read(d / "CONFIG.json") or {}
    prov = meta.get("provenance") or result.get("provenance") or {}
    commit = prov.get("commit")
    if not commit:
        for m in MARKERS:
            if (d / m).is_file():
                words = (d / m).read_text(encoding="utf-8", errors="replace").split()
                commit = words[0] if words else None
                break
    table = _get(config, "reasoning", "stages")
    reasoning = {}
    for s in stages:
        r = _get(table, s)
        reasoning[s] = ("/".join(str(r.get(k)) for k in ("effort", "max_tokens", "timeout_seconds"))
                        if isinstance(r, dict) else None)
    return {"arm": arm_name(d), "branch": prov.get("branch"), "commit": commit,
            "model": prov.get("model") or config.get("model"),
            "requested_model": prov.get("requested_model") or config.get("model"),
            "response_model_attested": prov.get("response_model_attested"),
            "seed": config.get("seed"),
            "reasoning": reasoning,
            "reasoning_reason": None if table else "not recorded in CONFIG.json (the reruns record the table)",
            "reused_stages": dict(sorted(reused_stages(d).items()))}


def _last_exception(d: Path, stage) -> str | None:
    for log in (d / "logs" / f"{stage}.log", d / "FAILURE_DETAIL.txt"):
        try:
            with open(log, "rb") as handle:
                handle.seek(0, 2)
                handle.seek(max(0, handle.tell() - 65536))
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if EXC_LINE_RE.match(line.strip()):
                return _trunc(line.strip())
    return None


def end_state(d: Path, markers, stages=STAGES) -> dict:
    meta = _read(d / "RUN_META.json") or {}
    status = meta.get("status")
    found = [m for m in MARKERS if (d / m).is_file() and (not markers or m in markers)]
    stages_done = [s for s in stages if _get(_read(d / f"{s}_COMPLETE.json"), "status") == "complete"]
    out = {"status": status, "markers": found, "stages_complete": stages_done,
           "failed_stage": None, "error": None, "error_detail": None, "failure_class": None}
    if status == "failed" or "failed" in d.name:
        out["failed_stage"] = meta.get("current_stage")
        out["error"] = _trunc(meta.get("error")) if meta.get("error") else None
        detail = _last_exception(d, out["failed_stage"]) if out["failed_stage"] else None
        out["error_detail"] = detail
        text = f"{meta.get('error') or ''} {detail or ''}"
        out["failure_class"] = ("timed_out" if TIMEOUT_RE.search(text) else
                                "gateway_connection" if CONNECTION_RE.search(text) else "other")
        out["run"] = "timed_out" if out["failure_class"] == "timed_out" else "failed"
    elif found and len(stages_done) == len(stages):
        out["run"] = "delivered"
    elif status == "complete" and len(stages_done) == len(stages):
        out["run"] = "delivered"
    else:
        out["run"] = "incomplete"
    return out


# --------------------------------------------------------------------------- #
# SHAPE records
# --------------------------------------------------------------------------- #
def _admission_of(rec: dict):
    a = rec.get("admission")
    if not isinstance(a, dict):
        return None, []
    decision = a.get("decision") or a.get("rule")
    held = []
    concerns = a.get("concerns")
    if isinstance(concerns, dict):
        held += [str(k) for k, v in sorted(concerns.items()) if v]
    elif isinstance(concerns, list):
        held += [str(c) for c in concerns]
    if isinstance(a.get("reasons"), list):
        held += [str(c) for c in a["reasons"]]
    return (str(decision) if decision is not None else None), held


def image_records(arm: Path, stage: str):
    recs, source = [], Counter()
    d = arm / stage
    if not d.is_dir():
        return recs, source
    for sub in sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: p.name):
        for name in ("SEARCH.json", "COHORT_SEARCH.json"):
            if (sub / name).is_file():
                rec = _read(sub / name)
                if isinstance(rec, dict):
                    recs.append(rec)
                    source[name] += 1
                break
    return recs, source


def summarise_records(recs: list, n_images: int | None = None) -> dict:
    """The normal form of the per-image search records of one image stage (also what search_summary.json holds)."""
    def count(rec, key):
        return rec.get(key) if isinstance(rec.get(key), (int, float)) else 0

    delivered = [str(r.get("delivered")) for r in recs]
    kind = Counter(re.sub(r"_\d+$", "", x) for x in delivered)
    errors = [e for r in recs for e in (r.get("execution_errors") or [])]
    admission, held = Counter(), Counter()
    for r in recs:
        decision, concerns = _admission_of(r)
        if decision is not None:
            admission[decision] += 1
        held.update(concerns)
    default_stop = Counter(str(r.get("stop_reason")) for r, x in zip(recs, delivered)
                           if not x.startswith("agent"))
    return {"images": n_images if n_images is not None else len(recs), "records": len(recs),
            "delivered_kind": kind,
            "resolution_reason": Counter(str((r.get("resolution") or {}).get("reason")) for r in recs),
            "stop_reason": Counter(str(r.get("stop_reason")) for r in recs),
            "calls": Counter(int(count(r, "n_calls")) for r in recs),
            "reasoner_failures": {"total": sum(count(r, "reasoner_failures") for r in recs),
                                  "images": sum(1 for r in recs if count(r, "reasoner_failures"))},
            "fallback_rounds": {"total": sum(count(r, "fallback_rounds") for r in recs),
                                "images": sum(1 for r in recs if count(r, "fallback_rounds"))},
            "execution_errors": {"total": len(errors), "images": sum(1 for r in recs if r.get("execution_errors")),
                                 "by_question": Counter(str(e.get("question")) if isinstance(e, dict)
                                                        else "unstructured" for e in errors)},
            "error_causes": Counter(_normalise_cause(e.get("cause")) for e in errors
                                    if isinstance(e, dict) and e.get("cause")),
            "seconds_total": round(sum(count(r, "seconds") for r in recs), 1),
            "admission": admission, "concerns_held": held, "default_stop_reason": default_stop,
            "parameter_source": None, "from": "per_image_records"}


def from_search_summary(s: dict) -> dict:
    ee = s.get("execution_errors") or {}
    return {"images": s.get("images"), "records": sum((s.get("record_source") or {}).values()),
            "delivered_kind": Counter({k: v for k, v in (s.get("delivered_kind") or {}).items()}),
            "resolution_reason": Counter(s.get("resolution_reason") or {}),
            "stop_reason": Counter(s.get("stop_reason") or {}),
            "calls": Counter({int(k): v for k, v in ((s.get("calls_per_image") or {}).get("distribution") or {}).items()}),
            "reasoner_failures": s.get("reasoner_failures") or {"total": None, "images": None},
            "fallback_rounds": s.get("fallback_rounds") or {"total": None, "images": None},
            "execution_errors": {"total": ee.get("total"), "images": ee.get("images"),
                                 "by_question": Counter(ee.get("by_question") or {})},
            "error_causes": None, "seconds_total": _get(s, "seconds", "total"),
            "admission": None, "concerns_held": None, "default_stop_reason": None,
            "parameter_source": Counter(s["parameter_source"]) if isinstance(s.get("parameter_source"), dict) else None,
            "from": "search_summary.json"}


def search_warnings(arm: Path) -> dict:
    """{stage: Counter of normalised execution-error causes} from SEARCH_WARNINGS.jsonl, or {} when absent."""
    path = arm / "SEARCH_WARNINGS.jsonl"
    out: dict = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        stage = row.get("stage")
        out.setdefault(stage, Counter())
        for e in row.get("execution_errors") or []:
            if isinstance(e, dict) and e.get("cause"):
                out[stage][_normalise_cause(e["cause"])] += 1
    return out


def shape_image_summary(arm: Path, stage: str, summary_file) -> dict | None:
    recs, source = image_records(arm, stage)
    if recs:
        n_images = len([p for p in (arm / stage).iterdir() if p.is_dir()])
        out = summarise_records(recs, n_images)
        out["record_source"] = dict(sorted(source.items()))
    else:
        stage_summary = _get(summary_file, "stages", stage)
        if not isinstance(stage_summary, dict):
            return None
        out = from_search_summary(stage_summary)
        out["record_source"] = stage_summary.get("record_source")
    if stage == "segment":
        cohort = _read(arm / "segment_COHORT.json")
        rows = _get(cohort, "parameter_source")
        if isinstance(rows, dict):
            out["parameter_source"] = Counter(str(v) for v in rows.values())
            out["parameter_source_from"] = "segment_COHORT.json"
        elif out["parameter_source"] is not None:
            out["parameter_source_from"] = "search_summary.json"
    return out


def _delivered_by_kind(kind: Counter) -> dict:
    mapped = Counter()
    for k, v in kind.items():
        mapped["cohort_reference" if k == "reference" else k] += v
    return mapped


def shape_loop(rec: dict) -> dict:
    """The per-call view of a SHAPE embed / batch search record: calls, scores, best, delivered, expert start."""
    det = rec.get("details") or {}
    entries = [e for e in (det.get("entries") or []) if isinstance(e, dict)]
    names = [c for r in (det.get("rounds") or []) for c in r]
    paired = len(names) == len(entries)
    calls = []
    if paired:
        calls = [{"call": n, "score": _num(e.get("score_gain_vs_initial")), "failed": bool(e.get("failed"))}
                 for n, e in zip(names, entries)]
    return {"det": det, "names": names, "calls": calls, "paired": paired, "n_entries": len(entries)}


def loop_progress(calls: list, delivered, basis: str) -> dict:
    """First-call-is-best / calls-after-best over the tuning calls of a loop; `calls` = [{call, score, failed}]."""
    tuning = [c for c in calls if c["call"] not in NOT_A_TUNING_CALL]
    scored = [c for c in tuning if c["score"] is not None and not c["failed"]]
    out = {"score_basis": basis, "tuning_calls": len(tuning), "failed_calls": sum(1 for c in tuning if c["failed"]),
           "scored_calls": len(scored)}
    if not scored:
        return {**out, "best_call": None, "best_score": None, "first_call": tuning[0]["call"] if tuning else None,
                "first_call_is_best": None, "calls_after_best": None, "delivered_call": delivered,
                "delivered_is_best": None}
    best = max(scored, key=lambda c: (c["score"], -tuning.index(c)))
    pos = tuning.index(best)
    return {**out, "best_call": best["call"], "best_score": _round(best["score"]),
            "first_call": tuning[0]["call"], "first_call_is_best": best is tuning[0],
            "calls_after_best": len(tuning) - pos - 1, "delivered_call": delivered,
            "delivered_is_best": (delivered == best["call"]) if delivered is not None else None}


# --------------------------------------------------------------------------- #
# stage readouts
# --------------------------------------------------------------------------- #
NOT_AGENTIC = "not applicable: no agentic decision in the cluster stage"
CONTROL_NA = "not applicable: control"


def _na_all(rec: Rec, reason: str, groups=("P1", "P2", "P3", "P4")):
    for g in groups:
        rec.put(g, None, reason)


def _stage_usage(arm: Path, stage: str):
    return _read(arm / f"{stage}_USAGE.json") or _get(_read(arm / f"{stage}_COMPLETE.json"), "usage")


def _seconds(arm: Path, stage: str):
    done = _read(arm / f"{stage}_COMPLETE.json")
    s = _num(_get(done, "seconds"))
    if s is None:
        s = _num(_get(_read(arm / "RUN_META.json"), "stages", stage, "seconds"))
    return s


def e1_shape(rec: Rec, usage, stage: str):
    if not isinstance(usage, dict):
        rec.na("E1", TOKEN_FIELDS, "no usage record for this stage")
        return
    obs = usage.get("usage_observed_fields")
    keys = {"total": ("total_tokens",), "input": ("prompt_tokens", "input_tokens"),
            "output": ("completion_tokens", "output_tokens"), "reasoning": ("reasoning_tokens",),
            "cached": ("cached_tokens",)}
    for field, names in keys.items():
        value = None
        for n in names:
            if _num(usage.get(n)) is not None and (obs is None or n in obs or field == "total"):
                value = usage[n]
                break
        rec.put(f"E1.{field}", value, "not reported by the provider for this stage (usage_observed_fields)")
    rec.put("E1.usage_complete", usage.get("usage_complete"), "not recorded")
    rec.put("E1.source", f"{stage}_USAGE.json", None)


def _round_split(r: dict):
    """(input, output, cached, reasoning, turns) of one agent round; None where the arm does not report it."""
    u = r.get("usage") if isinstance(r.get("usage"), dict) else {}
    if "cache_read_input_tokens" in u or "cache_creation_input_tokens" in u:           # Claude
        parts = [u.get("input_tokens"), u.get("cache_creation_input_tokens"), u.get("cache_read_input_tokens")]
        inp = sum(p for p in parts if _num(p) is not None) if all(_num(p) is not None for p in parts) else None
        think = [m.get("thinkingTokens") for m in (r.get("model_usage") or {}).values() if isinstance(m, dict)]
        reasoning = sum(think) if think and all(_num(t) is not None for t in think) else None
        return inp, _num(u.get("output_tokens")), _num(u.get("cache_read_input_tokens")), reasoning, _num(r.get("num_turns"))
    if "cached_input_tokens" in u:                                                      # Codex
        return (_num(u.get("input_tokens")), _num(u.get("output_tokens")), _num(u.get("cached_input_tokens")),
                _num(u.get("reasoning_output_tokens")), None)
    return _num(u.get("input_tokens")), _num(u.get("output_tokens")), None, None, _num(r.get("steps"))   # Biomni


def e1_agent(rec: Rec, rounds, selected, usage_json, stage: str):
    reported = _num((selected or {}).get("total_tokens")) or _num((usage_json or {}).get("total_tokens"))
    if not isinstance(rounds, list):
        rec.put("E1.total", reported, "no usage record for this stage")
        rec.na("E1", TOKEN_FIELDS[1:], "ROUNDS.json not available: only the stage total is recorded")
        return
    with_tokens = [r for r in rounds if (_num(r.get("total_tokens")) or 0) > 0]
    splits = [_round_split(r) for r in with_tokens]
    total_from_rounds = sum(_num(r.get("total_tokens")) or 0 for r in rounds)
    rec.put("E1.total", reported if reported is not None else total_from_rounds, None)
    reasons = {"input": "not reported for every round of this stage", "output": "not reported for every round of this stage",
               "cached": "not reported by this arm", "reasoning": "not reported by this arm"}
    for i, field in enumerate(("input", "output", "cached", "reasoning")):
        col = [s[i] for s in splits]
        rec.put(f"E1.{field}", sum(col) if col and all(c is not None for c in col) else None,
                reasons[field] if not (field in ("cached", "reasoning") and any(c is not None for c in col))
                else "not reported for every round of this stage")
    turns = [s[4] for s in splits]
    rec.put("E1.rounds_with_tokens", len(with_tokens), None)
    rec.put("E1.source", f"agent/{stage}/ROUNDS.json", None)
    rec.put("E1.check.total_from_rounds", total_from_rounds)
    rec.put("E1.check.total_reported", reported, "no reported stage total")
    rec.put("E1.check.consistent", reported == total_from_rounds if reported is not None else None,
            "no reported stage total")
    return turns


def agent_p2(rec: Rec, rounds: list):
    by_action = Counter(str(r.get("action")) for r in rounds)
    by_outcome = Counter(str(r.get("outcome")) for r in rounds)
    cross = Counter(f"{r.get('action')}/{r.get('outcome')}" for r in rounds)
    n = len(rounds)
    rec.put("P2.rounds", n)
    rec.put("P2.by_action", _counts(by_action))
    rec.put("P2.by_outcome", _counts(by_outcome))
    rec.put("P2.by_action_outcome", _counts(cross))
    for name in ("rejected", "malformed", "failed_round", "unfinished"):
        rec.put(f"P2.{name}", by_outcome.get(name, 0))
    failed_malformed = sum(1 for r in rounds if r.get("outcome") == "failed_round"
                           and str(r.get("error") or r.get("message")).startswith("MalformedDecision"))
    rec.put("P2.malformed_total", by_outcome.get("malformed", 0) + failed_malformed)
    rec.put("P2.rejected_rate", _round(by_outcome.get("rejected", 0) / n, 4) if n else None, "no rounds")
    rec.put("P2.malformed_rate", _round(by_outcome.get("malformed", 0) / n, 4) if n else None, "no rounds")
    rej = [r for r in rounds if r.get("outcome") == "rejected"]
    rec.put("P2.first_rejection_reasons", _top(Counter(_trunc(r.get("message")) for r in rej)))
    rec.put("P2.malformed_reasons", _top(Counter(_trunc(r.get("message")) for r in rounds
                                                 if r.get("outcome") == "malformed")))
    failed = [r for r in rounds if r.get("outcome") == "failed_round"]
    rec.put("P2.failed_round_errors", _top(Counter(_trunc(r.get("error") or r.get("message")) for r in failed)))
    rec.put("P2.timeouts", sum(1 for r in failed if TIMEOUT_RE.search(str(r.get("error") or r.get("message")))))
    keys = Counter()
    for r in rej:
        try:
            req = json.loads(r.get("request_json") or "")
        except (TypeError, ValueError):
            keys["(request not JSON)"] += 1
            continue
        found = set()
        for item in (req if isinstance(req, list) else [req]):
            if isinstance(item, dict):
                found |= set(item)
        keys.update(found or {"(no object keys)"})
    rec.put("P2.rejected_request_keys", _counts(keys))
    refused = [r.get("refused_executions") for r in rounds]
    rec.put("P2.refused_executions", sum(x for x in refused if _num(x) is not None)
            if any(_num(x) is not None for x in refused) else None, "not reported by this arm")
    denials = [r.get("permission_denials") for r in rounds]
    rec.put("P2.permission_denials", sum(len(x) for x in denials if isinstance(x, list))
            if any(isinstance(x, list) for x in denials) else None, "not reported by this arm")


def agent_p3(rec: Rec, stage: str, rounds: list, calls: dict | None):
    sessions: dict = {}
    for r in rounds:
        sessions.setdefault(str(r.get("session")), []).append(r)
    per = {s: len(rs) for s, rs in sessions.items()}
    caps = {s: SESSION_ROUND_CAP.get(s, PER_IMAGE_SESSION_CAP) for s in per}
    rec.put("P3.sessions", len(sessions))
    rec.putd("P3.session_rounds", {"total": sum(per.values()), "max": max(per.values()) if per else None,
                                   "median": _median(list(per.values())),
                                   "cap_per_session": dict(sorted(caps.items()))}, "no rounds")
    executed = [r for r in rounds if r.get("action") == "run" and r.get("outcome") == "executed"]
    rec.put("P3.rounds_with_calls", len(executed))
    rest = Counter(f"{r.get('action')}/{r.get('outcome')}" for r in rounds if r not in executed)
    rec.put("P3.rounds_without_calls", {"total": sum(rest.values()), "by_action_outcome": _counts(rest)})
    modes = Counter(str(r["table_mode"]) for r in rounds if r.get("table_mode"))
    if modes:      # group-session rounds of records written after the aggregated-table change (codex_arm.table_text)
        rec.put("P3.group_table_mode_rounds", _counts(modes))
    stops = Counter()
    for rs in sessions.values():
        stops["finished" if any(x.get("outcome") == "finished" for x in rs) else str(rs[-1].get("outcome"))] += 1
    rec.put("P3.stop_reason", _counts(stops))
    rec.put("P3.tool_call_round_allowance", TOOL_ROUND_ALLOWANCE)
    rec.put("P3.tool_call_rounds_used_by_session", {s: sum(1 for x in rs if x in executed)
                                                    for s, rs in sorted(sessions.items())})
    if stage in IMAGE_STAGES:
        waves = ([len({c.get("wave") for c in cs.values() if c.get("wave")}) for cs in calls.values()]
                 if calls else None)
        rec.put("P3.rounds_per_image", _spread(waves) if waves else None,
                "per-image CALLS.json not in this record set (over the size cap)")
    else:
        waves = {c.get("wave") for c in (calls or {}).get("dataset", {}).values() if c.get("wave")}
        rec.put("P3.tool_call_rounds_used", len(waves) if calls else None, "CALLS.json not available")


def agent_loop(rec: Rec, stage: str, arm: Path, calls: dict | None, delivered):
    """P3.loop of an agent embed / batch stage; per-call scores from the kept call records."""
    if not calls or "dataset" not in calls:
        rec.put("P3.loop", None, "CALLS.json not available")
        return
    rows = []
    for label in sorted(calls["dataset"], key=lambda x: (x != "input", int(re.sub(r"\D", "", x) or 0))):
        c = calls["dataset"][label]
        score = None
        if stage == "embed":
            history = _get(c, "view", "val_history") if isinstance(c.get("view"), dict) else None
            vals = [_num(h.get("sibling_site_map")) for h in history or [] if isinstance(h, dict)]
            vals = [v for v in vals if v is not None]
            score = max(vals) if vals else None
        else:
            feedback = _read(arm / "agent" / stage / "calls" / label / "FEEDBACK.json")
            score = _num(_get(feedback, "calibration", "overall"))
        rows.append({"call": label, "score": score, "failed": c.get("status") != "complete"})
    basis = ("best validation sibling_site_map of the call (the proxy the agent read)" if stage == "embed"
             else "calibration.overall of the call's FEEDBACK.json (the score the agent read)")
    loop = loop_progress(rows, delivered, basis)
    loop["input_score"] = _round(next((r["score"] for r in rows if r["call"] == "input"), None))
    rec.putd("P3.loop", loop, "no scored tuning call or no score recorded")
    rec.put("P3.loop.expert_start_calls", None, "not applicable: expert start is a SHAPE embed rule")


def agent_stage(rec: Rec, arm: Path, stage: str):
    base = arm / "agent" / stage
    rounds = _read(base / "ROUNDS.json")
    selected = _read(base / "SELECTED.json")
    done_usage = _get(_read(arm / f"{stage}_COMPLETE.json"), "usage")
    if selected is None and isinstance(done_usage, dict) and "deliveries" in done_usage:
        selected = done_usage
    calls = _read(base / "CALLS.json")
    deliveries = (selected or {}).get("deliveries")
    # P1
    if isinstance(deliveries, dict):
        kinds: dict = {}
        for unit, d in deliveries.items():
            k = kinds.setdefault(str(d.get("source")), {"units": 0, "sessions": set(), "reasons": Counter()})
            k["units"] += 1
            k["sessions"].add(d.get("session"))
            if d.get("reason"):
                k["reasons"][_trunc(d["reason"])] += 1
        rec.put("P1.delivered_by", {"units": len(deliveries),
                                    "by_kind": {k: v["units"] for k, v in sorted(kinds.items())},
                                    "sessions_by_kind": {k: len(v["sessions"]) for k, v in sorted(kinds.items())}})
        rec.put("P1.fallback_reasons", {k: _top(v["reasons"], 3) for k, v in sorted(kinds.items())
                                        if k != "agent"} or None, "every unit was delivered by the agent")
        rec.put("P1.agent_delivered_all", all(k == "agent" for k in kinds))
        delivered_call = (deliveries.get("dataset") or {}).get("call") if stage in LOOP_STAGES else None
    else:
        rec.put("P1.delivered_by", None, "SELECTED.json not available")
        rec.put("P1.agent_delivered_all", None, "SELECTED.json not available")
        delivered_call = None
    # P2 / P3
    if isinstance(rounds, list):
        agent_p2(rec, rounds)
        agent_p3(rec, stage, rounds, calls if isinstance(calls, dict) else None)
    else:
        rec.put("P2", None, "ROUNDS.json not available")
        rec.put("P3", None, "ROUNDS.json not available")
    if stage in LOOP_STAGES and isinstance(rounds, list):
        agent_loop(rec, stage, arm, calls if isinstance(calls, dict) else None, delivered_call)
    rec.put("P4", None, "not applicable: SHAPE admission and guard rules")
    # E
    turns = e1_agent(rec, rounds, selected, _read(arm / f"{stage}_USAGE.json"), stage)
    n_units = len(deliveries) if isinstance(deliveries, dict) else None
    n_calls = _num((selected or {}).get("n_calls"))
    rec.put("E3.llm_calls", None, "not reported by this arm (one agent invocation per round: see agent_rounds)")
    rec.put("E3.agent_rounds", _num((selected or {}).get("n_rounds")) if selected else None, "SELECTED.json not available")
    turn_vals = [t for t in (turns or []) if t is not None]
    rec.put("E3.agent_turns", sum(turn_vals) if turns and len(turn_vals) == len(turns) and turn_vals else None,
            "not reported by this arm")
    rec.put("E3.tool_calls", n_calls, "SELECTED.json not available")
    rec.put("E3.tool_calls_basis", "SELECTED.json n_calls (includes the default call c0 / input)")
    rec.put("E3.tuning_calls", n_calls - n_units if n_calls is not None and n_units is not None else None,
            "SELECTED.json not available")
    rec.put("E3.units", n_units, "SELECTED.json not available")
    if stage in IMAGE_STAGES:
        per_image = ([len([k for k in cs if k not in NOT_A_TUNING_CALL]) for cs in calls.values()]
                     if isinstance(calls, dict) else None)
        spread = _spread(per_image) if per_image else {}
        rec.put("E3.tuning_calls_per_image.mean",
                _round(n_calls / n_units - 1, 4) if n_calls is not None and n_units else None,
                "SELECTED.json not available")
        for k in ("median", "p90", "max"):
            rec.put(f"E3.tuning_calls_per_image.{k}", spread.get(k),
                    "per-image CALLS.json not in this record set (over the size cap)")


def shape_stage(rec: Rec, arm: Path, stage: str, summary_file):
    if stage in IMAGE_STAGES:
        s = shape_image_summary(arm, stage, summary_file)
        if s is None:
            for g in ("P1", "P2", "P3", "P4"):
                rec.put(g, None, "no per-image records and no search_summary.json for this stage")
            return
        rec.put("P1.record_source", s["from"])
        by = _delivered_by_kind(s["delivered_kind"])
        rec.put("P1.delivered_by", {"units": s["images"], "by_kind": _counts(by)})
        rec.put("P1.non_agent_stop_reasons", _counts(s["default_stop_reason"]) if s["default_stop_reason"] is not None else None,
                "needs per-image records (the curated copy keeps marginal counts only)")
        rf, fb, ee = s["reasoner_failures"], s["fallback_rounds"], s["execution_errors"]
        rec.putd("P2.reasoner_failures", rf, "not recorded")
        rec.putd("P2.fallback_rounds", fb, "not recorded")
        rec.put("P2.execution_errors.total", ee["total"], "not recorded")
        rec.put("P2.execution_errors.images", ee["images"], "not recorded")
        rec.put("P2.execution_errors.by_question", _counts(ee["by_question"]))
        causes = s["error_causes"]
        if causes is None:
            causes = search_warnings(arm).get(stage) if (arm / "SEARCH_WARNINGS.jsonl").is_file() else None
            if causes is not None and not causes and ee["total"]:
                causes = None
        rec.put("P2.execution_error_causes", _top(causes) if causes is not None else None,
                "no per-image records and no SEARCH_WARNINGS.jsonl")
        rec.put("P2.rejected", None, "not applicable: SHAPE has no request / reply protocol (see execution_errors)")
        rec.put("P2.malformed", None, "not applicable: SHAPE has no request / reply protocol (see execution_errors)")
        calls = _expand(s["calls"])
        rec.putd("P3.calls_per_image", {**_spread(calls), "mean": _round(sum(calls) / len(calls), 4) if calls else None,
                                        "total": sum(calls)}, "no image record")
        rec.put("P3.stop_reason", _counts(s["stop_reason"]))
        rec.put("P3.resolution_reason", _counts(s["resolution_reason"]))
        rec.put("P3.images_at_round_allowance", s["resolution_reason"].get("max_iters_reached", 0))
        rec.put("P3.tool_call_round_allowance", TOOL_ROUND_ALLOWANCE)
        rec.put("P3.rounds_used", None, "not recorded per image (calls_per_image and the stop reasons are)")
        rec.put("P3.rounds_with_calls", None, "not recorded per image")
        rec.put("P3.rounds_without_calls", None, "not recorded per image")
        causes_count = Counter()
        if causes:
            for text, n in causes.items():
                if CONTRAST_RE.search(text):
                    causes_count["contrast"] += n
        rec.put("P4.admission_decisions", _counts(s["admission"]) if s["admission"] is not None else None,
                "needs per-image records (the curated copy keeps no admission)")
        if not s["admission"] and s["admission"] is not None:
            rec.put("P4.admission_decisions", None, "per-image records carry no admission for this stage")
        rec.put("P4.concerns_held", _counts(s["concerns_held"]) if s["concerns_held"] else None,
                "no concern held, or no per-image records")
        rec.put("P4.contrast_rejections", causes_count.get("contrast", 0) if causes is not None else None,
                "execution-error causes not available")
        if stage == "segment":
            ps = s["parameter_source"]
            rec.put("P4.parameter_source", _counts(ps) if ps is not None else None,
                    "no segment_COHORT.json and no parameter_source in search_summary.json")
        else:
            rec.put("P4.parameter_source", None, "not applicable: only the segmentation cohort has parameter sources")
        rec.put("E3.tool_calls", sum(calls))
        rec.put("E3.tool_calls_basis", "search records n_calls (calls except the default call)")
        rec.put("E3.tuning_calls", sum(calls))
        rec.put("E3.units", s["images"])
        rec.putd("E3.tuning_calls_per_image", _spread(calls), "no image record")
        rec.put("E3.search_seconds", s["seconds_total"], "not recorded")
        return
    # embed / batch: the stage record
    name = "representation_SEARCH.json" if stage == "embed" else "batch_SEARCH.json"
    record = _read(arm / name)
    if not isinstance(record, dict):
        for g in ("P1", "P2", "P3", "P4"):
            rec.put(g, None, f"{name} not available")
        return
    loop = shape_loop(record)
    det = loop["det"]
    call = det.get("call")
    kind = "agent" if str(call).startswith("agent_") else "default"
    rec.put("P1.delivered_by.units", 1)
    rec.put("P1.delivered_by.by_kind", {kind: 1})
    rec.put("P1.delivered_by.call", call, "no delivered call recorded")
    rec.put("P1.delivered_by.choice", det.get("choice"), "no choice recorded")
    if kind != "agent":
        rec.putd("P1.fallback", {"stop_reason": record.get("stop_reason"),
                                 "resolution_reason": _get(record, "resolution", "reason"),
                                 "resolution_status": _get(record, "resolution", "status"),
                                 "choice": det.get("choice")}, "not recorded")
    else:
        rec.put("P1.fallback", None, "delivered by the agent")
    errors = [e for e in (record.get("execution_errors") or []) if isinstance(e, dict)]
    single = "not applicable: one search record per stage"
    rec.put("P2.reasoner_failures.total", record.get("reasoner_failures"), "not recorded")
    rec.put("P2.reasoner_failures.images", None, single)
    rec.put("P2.fallback_rounds.total", record.get("fallback_rounds"), "not recorded")
    rec.put("P2.fallback_rounds.images", None, single)
    rec.put("P2.execution_errors.total", len(errors))
    rec.put("P2.execution_errors.images", None, single)
    rec.put("P2.execution_errors.by_question", _counts(Counter(str(e.get("question")) for e in errors)))
    causes = Counter(_normalise_cause(e.get("cause")) for e in errors if e.get("cause"))
    rec.put("P2.execution_error_causes", _top(causes))
    rec.put("P2.rejected", None, "not applicable: SHAPE has no request / reply protocol (see execution_errors)")
    rec.put("P2.malformed", None, "not applicable: SHAPE has no request / reply protocol (see execution_errors)")
    rec.put("P3.stop_reason", {str(record.get("stop_reason")): 1})
    rec.put("P3.resolution_reason", {str(_get(record, "resolution", "reason")): 1})
    rec.put("P3.rounds_with_calls", len(det.get("rounds") or []) if det.get("rounds") is not None else None, "no rounds recorded")
    rec.put("P3.plans_total", len(det.get("plans") or []) if det.get("plans") is not None else None, "no plans recorded")
    rec.put("P3.tool_call_round_allowance", TOOL_ROUND_ALLOWANCE)
    rec.put("P3.rounds_without_calls", None, "not recorded (plans_total counts the reasoner's plans)")
    if loop["paired"] and loop["calls"]:
        progress = loop_progress(loop["calls"], call, "score_gain_vs_initial of the search entry (the search's own score)")
        rec.putd("P3.loop", progress, "no scored tuning call")
    else:
        rec.put("P3.loop", None, "entries and rounds of the record do not pair up")
    expert = det.get("expert_config")
    rec.put("P3.loop.expert_start_calls", len(expert) if isinstance(expert, list) else None,
            "no expert_config in this record (no expert start)")
    # P4
    decision, held = _admission_of(record)
    rec.put("P4.admission_decisions", {decision: 1} if decision else None, "no admission in the record")
    rec.put("P4.concerns_held", _counts(Counter(held)) if held else None, "no concern held")
    a = record.get("admission") or {}
    rec.put("P4.admission_veto", a.get("veto"), "no veto")
    rec.put("P4.admission_reasoner_decision", _get(a, "reasoner", "decision"), "no reasoner admission")
    admissible, failed = det.get("admissible"), det.get("failed")
    n_tuning = len([n for n in loop["names"] if n not in NOT_A_TUNING_CALL]) if loop["names"] else None
    rec.put("P4.guard.calls", n_tuning, "no rounds recorded")
    rec.put("P4.guard.admissible", len(admissible) if isinstance(admissible, list) else None, "not recorded")
    rec.put("P4.guard.failed", len(failed) if isinstance(failed, list) else None, "not recorded")
    for key in ("expert_rejections", "stop_refusals", "forced_continuations"):
        value = det.get(key)
        rec.put(f"P4.{key}", len(value) if isinstance(value, list) else None, f"no {key} in this record")
    if det.get("expert_rejections"):
        rec.put("P4.expert_rejection_reasons", _top(Counter(_trunc(x) for x in det["expert_rejections"])))
    rec.put("P4.contrast_rejections", sum(n for t, n in causes.items() if CONTRAST_RE.search(t)))
    rec.put("P4.parameter_source", None, "not applicable: only the segmentation cohort has parameter sources")
    rec.put("E3.tool_calls", len(loop["names"]) if loop["names"] else None, "no rounds recorded")
    rec.put("E3.tool_calls_basis", "calls of the search record (details.rounds; the embed input is not a call)")
    rec.put("E3.tuning_calls", n_tuning, "no rounds recorded")
    rec.put("E3.units", 1)


def stage_record(arm: Path, kind: str, stage: str, summary_file, reused: dict) -> Rec:
    rec = Rec()
    done = _read(arm / f"{stage}_COMPLETE.json")
    state = "complete" if _get(done, "status") == "complete" else None
    meta = _read(arm / "RUN_META.json") or {}
    if state is None:
        state = ("failed" if meta.get("current_stage") == stage and meta.get("status") == "failed" else "not_run")
    rec.put("state", state)
    rec.data["reused_from"] = reused.get(stage)
    if stage == "cluster":
        _na_all(rec, NOT_AGENTIC)
    elif kind == "control":
        _na_all(rec, CONTROL_NA)
    elif state != "complete":
        _na_all(rec, f"stage did not complete ({state})")
    elif kind == "shape":
        shape_stage(rec, arm, stage, summary_file)
    else:
        agent_stage(rec, arm, stage)
    # E1 for control / cluster / shape
    usage = _stage_usage(arm, stage)
    if state != "complete":
        rec.na("E1", TOKEN_FIELDS, f"stage did not complete ({state})")
        rec.put("E3.llm_calls", None, f"stage did not complete ({state})")
        rec.put("E3.tool_calls", None, f"stage did not complete ({state})")
    elif kind == "control":
        rec.na("E1", TOKEN_FIELDS, CONTROL_NA)
        rec.put("E3.llm_calls", None, CONTROL_NA)
        rec.put("E3.tool_calls", None, CONTROL_NA)
    elif stage == "cluster":
        rec.na("E1", TOKEN_FIELDS, NOT_AGENTIC)
        rec.put("E3.llm_calls", None, NOT_AGENTIC)
        rec.put("E3.tool_calls", None, NOT_AGENTIC)
    elif kind == "shape":
        e1_shape(rec, usage, stage)
        rec.put("E3.llm_calls", _num(_get(usage, "calls")), "not recorded")
    seconds = _seconds(arm, stage)
    rec.put("E2.seconds", seconds, "no completion record for this stage")
    return rec


# --------------------------------------------------------------------------- #
# arm run and experiment
# --------------------------------------------------------------------------- #
def arm_run(arm: Path, markers, stages=STAGES) -> dict:
    name = arm_name(arm)
    kind = arm_kind(name)
    reused = reused_stages(arm)
    summary_file = _read(arm / "search_summary.json")
    out_stages, recs = {}, {}
    for stage in stages:
        rec = stage_record(arm, kind, stage, summary_file, reused)
        recs[stage] = rec
    own = [s for s in stages if s not in reused]
    secs = {s: recs[s].data.get("E2", {}).get("seconds") for s in stages}
    total_all = sum(v for v in secs.values() if v is not None)
    total_own = sum(secs[s] for s in own if secs[s] is not None)
    for s in stages:
        v = secs[s]
        recs[s].put("E2.share_of_total", _round(v / total_all, 4) if v is not None and total_all else None,
                    "no seconds recorded")
        recs[s].put("E2.share_of_own", _round(v / total_own, 4) if v is not None and total_own and s in own else None,
                    "reused from a source run: not part of this run's own wall-clock" if s in reused
                    else "no seconds recorded")
        out_stages[s] = recs[s].out()
    totals = Rec()
    for field in TOKEN_FIELDS:
        for label, group in (("own", own), ("including_reused", list(stages))):
            vals = [recs[s].data.get("E1", {}).get(field) for s in group if recs[s].why.get(f"E1.{field}", "")
                    not in (NOT_AGENTIC, CONTROL_NA)]
            applicable = [v for v in vals if v is not None]
            missing = len(vals) - len(applicable)
            if kind == "control":
                totals.put(f"E1.{field}.{label}", None, CONTROL_NA)
            elif not applicable:
                totals.put(f"E1.{field}.{label}", None, "not reported by this arm")
            elif missing:
                totals.put(f"E1.{field}.{label}", None, "not reported for every stage of this arm")
            else:
                totals.put(f"E1.{field}.{label}", sum(applicable))
    totals.put("E2.seconds.own", _round(total_own, 3) if any(secs[s] is not None for s in own) else None, "no seconds recorded")
    totals.put("E2.seconds.including_reused", _round(total_all, 3) if total_all else None, "no seconds recorded")
    for field in ("tool_calls", "llm_calls"):
        for label, group in (("own", own), ("including_reused", list(stages))):
            vals = [recs[s].data.get("E3", {}).get(field) for s in group
                    if recs[s].why.get(f"E3.{field}", "") not in (NOT_AGENTIC, CONTROL_NA)]
            applicable = [v for v in vals if v is not None]
            if kind == "control":
                totals.put(f"E3.{field}.{label}", None, CONTROL_NA)
            elif not applicable:
                totals.put(f"E3.{field}.{label}", None, "not reported by this arm")
            elif len(applicable) != len(vals):
                totals.put(f"E3.{field}.{label}", None, "not reported for every stage of this arm")
            else:
                totals.put(f"E3.{field}.{label}", sum(applicable))
    prov = provenance(arm, stages)
    return {"arm": name, "kind": kind, "provenance": prov, "end_state": end_state(arm, markers, stages),
            "record_source": {s: (shape_image_summary_source(arm, s, summary_file) if kind == "shape" else
                                  ("agent/ROUNDS.json" if kind == "agent" and (arm / "agent" / s / "ROUNDS.json").is_file() else None))
                              for s in stages if s != "cluster"},
            "stages": out_stages, "totals": totals.out()}


def shape_image_summary_source(arm: Path, stage: str, summary_file):
    if stage in IMAGE_STAGES:
        recs, _ = image_records(arm, stage)
        if recs:
            return "per_image_records"
        return "search_summary.json" if isinstance(_get(summary_file, "stages", stage), dict) else None
    return {"embed": "representation_SEARCH.json", "batch": "batch_SEARCH.json"}[stage] if (
        arm / {"embed": "representation_SEARCH.json", "batch": "batch_SEARCH.json"}[stage]).is_file() else None


def _cell(table: str, run: dict, stage_or_total: str):
    """The scalar of one comparison table cell; None when unavailable."""
    stages, totals = run["stages"], run["totals"]
    st = stages.get(stage_or_total) or {}
    g = lambda *p: _get(st, *p)                         # noqa: E731
    if table == "P1_delivered_by":
        d = g("P1", "delivered_by")
        if not isinstance(d, dict):
            return None
        return ", ".join(f"{k}:{v}" for k, v in d["by_kind"].items())
    if table == "P2_rejected":
        return g("P2", "rejected")
    if table == "P2_malformed":
        return g("P2", "malformed_total")
    if table == "P2_failed_rounds":
        return g("P2", "failed_round")
    if table == "P2_reasoner_failures":
        return _get(g("P2", "reasoner_failures"), "total")
    if table == "P2_execution_errors":
        return _get(g("P2", "execution_errors"), "total")
    if table == "P3_session_rounds":
        return _get(g("P3", "session_rounds"), "total")
    if table == "P3_tool_call_rounds":
        return g("P3", "rounds_with_calls")
    if table.startswith("E1_"):
        field = table[3:]
        return _get(totals, "E1", field, "own") if stage_or_total == "total" else g("E1", field)
    if table == "E2_seconds":
        return _get(totals, "E2", "seconds", "own") if stage_or_total == "total" else g("E2", "seconds")
    if table == "E2_share":
        return None if stage_or_total == "total" else g("E2", "share_of_total")
    if table == "E3_tool_calls":
        return _get(totals, "E3", "tool_calls", "own") if stage_or_total == "total" else g("E3", "tool_calls")
    if table == "E3_llm_calls":
        return _get(totals, "E3", "llm_calls", "own") if stage_or_total == "total" else g("E3", "llm_calls")
    return None


TABLES = ("P1_delivered_by", "P2_rejected", "P2_malformed", "P2_failed_rounds", "P2_reasoner_failures",
          "P2_execution_errors", "P3_session_rounds", "P3_tool_call_rounds", "E1_total", "E1_input", "E1_output",
          "E1_reasoning", "E1_cached", "E2_seconds", "E2_share", "E3_tool_calls", "E3_llm_calls")


def comparison_tables(arms: dict, stages=STAGES) -> dict:
    out = {}
    columns = list(stages) + ["total"]
    for table in TABLES:
        cells = {name: {c: _cell(table, run, c) for c in columns} for name, run in arms.items()}
        out[table] = {"rows": list(arms), "columns": columns, "cells": cells,
                      "reused": {name: sorted(run["provenance"]["reused_stages"]) for name, run in arms.items()
                                 if run["provenance"]["reused_stages"]},
                      "total_is": "own (reused stages excluded)"}
    return out


def notes_for(arms: dict) -> list:
    notes = ["Single seed: every result in this record set is one seed; differences between arms are single-seed "
             "observations (AGENTIC_EVALUATION_FRAMEWORK.md, reporting rule 3).",
             "Process readouts are reported beside the outcome and never rank the arms; no number here is a quality score.",
             "A null is an unavailable signal and carries its reason in `unavailable`; it is never a zero."]
    complete = {n: a for n, a in arms.items() if a["end_state"]["run"] == "delivered"
                and not a["provenance"]["reused_stages"]}
    commits = {a["provenance"]["commit"] for a in complete.values() if a["provenance"]["commit"]}
    if len(commits) > 1:
        by = {c[:8]: sorted(n for n, a in complete.items() if a["provenance"]["commit"] == c) for c in sorted(commits)}
        notes.append("Not strict: the arms ran on different commits ("
                     + "; ".join(f"{c}: {', '.join(v)}" for c, v in by.items()) + ").")
    unrecorded = sorted(n for n, a in arms.items() if a["provenance"]["reasoning_reason"])
    if unrecorded:
        notes.append("Reasoning settings are not recorded in CONFIG.json for: " + ", ".join(unrecorded)
                     + ". Arms whose settings cannot be read from the records are not strictly comparable.")
    reasoning = {json.dumps(a["provenance"]["reasoning"], sort_keys=True) for a in arms.values()
                 if not a["provenance"]["reasoning_reason"]}
    if len(reasoning) > 1:
        notes.append("Arms with a recorded reasoning table differ in their settings: not strict.")
    for n, a in arms.items():
        if a["provenance"]["reused_stages"]:
            notes.append(f"{n}: stages {', '.join(sorted(a['provenance']['reused_stages']))} were reused from "
                         f"{sorted(set(a['provenance']['reused_stages'].values()))[0]}; they are excluded from its own "
                         "totals and the `total` column; their per-stage values are the source run's.")
        if a["end_state"]["run"] != "delivered":
            notes.append(f"{n}: end state {a['end_state']['run']}"
                         + (f" at stage {a['end_state']['failed_stage']}" if a["end_state"]["failed_stage"] else "") + ".")
    notes.append("Token counts are each arm's own accounting: Claude input = uncached + cache-creation + cache-read "
                 "tokens; Codex input includes cache writes; Biomni reports input and output only; SHAPE reports "
                 "the provider's usage per call. Compare totals, not fields, across arm families.")
    notes.append("tool_calls: the agent arms' count includes the default call (c0 / input); SHAPE's search records "
                 "count the calls except the default call. Use `tuning_calls` for a like-for-like count.")
    return notes


def build(run: Path, markers=(), stages=STAGES) -> dict:
    arms = {}
    for d in discover_arms(run, stages):
        arms[d.name] = arm_run(d, markers, stages)
    return {"schema": SCHEMA, "source": run.name, "stages": list(stages), "arms": arms,
            "tables": comparison_tables(arms, stages), "notes": notes_for(arms)}


# --------------------------------------------------------------------------- #
# renderings
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.1f}" if abs(v) >= 100 else f"{v:.3f}"
    return str(v)


def markdown(result: dict) -> str:
    lines = [f"# Process readouts: {result['source']}", "",
             "L4 protocol friction (P1-P4) and L5 efficiency (E1-E3, E6) from the run records; no ground truth. "
             "`n/a` = unavailable (reason in the JSON `unavailable`). `total` = the arm's own stages "
             "(reused stages excluded).", "",
             "| run | arm | commit | end state | reused stages |", "| --- | --- | --- | --- | --- |"]
    for name, a in result["arms"].items():
        p = a["provenance"]
        lines.append(f"| {name} | {a['arm']} | {(p['commit'] or 'n/a')[:8]} | {a['end_state']['run']} | "
                     f"{', '.join(sorted(p['reused_stages'])) or '-'} |")
    for table, t in result["tables"].items():
        lines += ["", f"## {table}", "", "| run | " + " | ".join(t["columns"]) + " |",
                  "| --- |" + " --- |" * len(t["columns"])]
        for name in t["rows"]:
            reused = set(t["reused"].get(name, []))
            row = [(_fmt(t["cells"][name][c]) + (" (reused)" if c in reused and t["cells"][name][c] is not None else ""))
                   for c in t["columns"]]
            lines.append(f"| {name} | " + " | ".join(row) + " |")
    lines += ["", "## Notes", ""] + [f"- {n}" for n in result["notes"]]
    return "\n".join(lines) + "\n"


def _flatten(prefix: str, value, out: list):
    if isinstance(value, dict):
        for k in sorted(value):
            _flatten(f"{prefix}.{k}" if prefix else str(k), value[k], out)
    elif isinstance(value, list):
        out.append((prefix, json.dumps(value, sort_keys=True)))
    else:
        out.append((prefix, value))


def csv_text(result: dict) -> str:
    """Long format: run, arm, stage ("total" for the arm totals), metric (dotted path), value, reason."""
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["run", "arm", "stage", "metric", "value", "reason"])
    for name, a in result["arms"].items():
        for stage in list(a["stages"]) + ["total"]:
            rec = a["totals"] if stage == "total" else a["stages"][stage]
            why = rec.get("unavailable", {})
            flat: list = []
            _flatten("", {k: v for k, v in rec.items() if k != "unavailable"}, flat)
            leaves = {m: v for m, v in flat}
            for m in sorted(set(leaves) | set(why)):
                v = leaves.get(m)
                reason = why.get(m, "")
                if v is None and not reason:
                    reason = next((r for p, r in why.items() if m.startswith(p + ".")), "")
                if v is None and m == "reused_from":
                    reason = "not reused: computed in this run"
                w.writerow([name, a["arm"], stage, m, "" if v is None else v, reason])
    return buf.getvalue()


def write_outputs(result: dict, out: Path, csv_path: Path | None = None, md: bool = False) -> list[Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    written = [out]
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        Path(csv_path).write_text(csv_text(result), encoding="utf-8", newline="\n")
        written.append(Path(csv_path))
    if md:
        md_path = out.with_suffix(".md")
        md_path.write_text(markdown(result), encoding="utf-8", newline="\n")
        written.append(md_path)
    return written


def main(argv=None, *, markers=(), stages=STAGES, default_out=None, description=None) -> int:
    ap = argparse.ArgumentParser(description=description or __doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="an experiment directory or a curated results folder")
    ap.add_argument("--out", type=Path, default=None, required=default_out is None,
                    help="output JSON" + (" (default: <run>/analysis/process_readouts.json)" if default_out else ""))
    ap.add_argument("--csv", type=Path, default=None, help="also write the long-format CSV")
    ap.add_argument("--md", action="store_true", help="also write a markdown with the same stem as --out")
    args = ap.parse_args(argv)
    if not args.run.is_dir():
        print(f"ERROR: --run {args.run}: not a directory", file=sys.stderr)
        return 2
    out = args.out or (args.run / "analysis" / default_out)
    result = build(args.run, markers, tuple(stages))
    if not result["arms"]:
        print(f"ERROR: no arm run directory found in {args.run}", file=sys.stderr)
        return 2
    for path in write_outputs(result, out, args.csv, args.md):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
