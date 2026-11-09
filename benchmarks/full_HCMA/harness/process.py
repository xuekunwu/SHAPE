"""Process metrics of a run (evaluation framework, batch 1: D1, D5, D6, E1, F3).

Read only the run's own records, never a label: the ledger (logs/tool_calls.jsonl via ledger.calls),
the call directories (per_image.csv, inputs/ copies, meta.json, cells.csv), the agent trace
(traces/round_*.jsonl with .times), AGENT_RUN.json, logs/rounds.jsonl, submission/choice.json,
workspace/NOTES.md, and RESULT.json for the audit hits when scoring has run. Writes PROCESS.json and
PROCESS.md (evaluator output, outside the workspace).

- D1 adaptivity: per tool with per-image levers (segment, crop), the share of the delivered images
  that got their own per-image setting, the dispersion of every lever over the delivered images (the
  values each image was actually processed with), and the rescues (delivered images whose masks /
  crops come from another call than the tool's main delivering call, or than its first full-set call).
- D5 substantive vs token lever coverage: a lever group changed in a call that feeds the delivered
  chain or in a full-set candidate (>= FULL_SET_SHARE of the prep's images and, for embed /
  batch_correct / cluster, of the delivered cells) is substantive; changed only in smaller probes,
  token.
- D6 traceability: the share of delivered-chain calls with a note, and with a note citing a number.
- E1 cost: wall hours, LLM turns / tokens by type / USD / API time (the trace's result events), tool
  calls and seconds per tool, GPU-hours (segment + embed seconds), the time with >= 1 tool call
  running and the maximum number of concurrent calls.
- F3 failure taxonomy: no valid choice, deadline without FINAL, FINAL rejections, audit hits, images
  of the prep without a delivered cell, failed / partial / unfinished calls.
- End state (BIABench D/N/C + timed out / refused): delivered (a valid choice: the scorer's chain verdict),
  crashed (the loop ended on an error without a valid choice), no_deliverable (otherwise).
- Claims: FINAL accepted, the closing text (the last result event's text, redacted, <= CLOSING_CHARS),
  and whether the run claimed completion (FINAL or CLAIM_RULE on the closing text).
Every E1 value names its source (E1_cost.sources); a value that is not available is null with a reason
(E1_cost.unavailable_reasons), never 0. The USD figure is Claude Code's own list-price estimate from
its result events, not the gateway's billing.
A control run (no agent) gets E1's tool part, F3 and the end state; the rest is marked not applicable. A SHAPE-arm
run (harness/shape_arm.py, `ledger.run_arm` "shape") gets D1, D5, D6 from the ledger as an agent run does, E1 with
the wall time of SHAPE_RUN.json and the reasoner usage SHAPE's engines recorded (shape/engine/usage.py, in
SHAPE_RUN.json: calls and tokens, no USD), F3 and the control's end-state rule; no claims (it has no closing text).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ledger import COMPLETED, calls, chain, run_arm

FULL_SET_SHARE = 0.5
GPU_TOOLS = ("segment", "embed")
PER_IMAGE_TOOLS = ("segment", "crop")
CALL_ID = re.compile(r"\b[SCEBK]\d{3,}\b")
NOT_AGENT = "control run: no agent decided it"
NOT_CLAUDE = "SHAPE arm: no agent rounds, FINAL flag or closing text"
SHAPE_RECORD = "SHAPE_RUN.json"
CLOSING_CHARS = 2000
CLAIM_RULE = (r"(?i)\b(done|complete[d]?|completion|finished|final(i[sz]ed)?|delivered|submitted|wrote "
              r"submission/FINAL)\b")
REFUSAL_STOP_REASONS = ("refusal",)


def _na(reason: str) -> dict:
    return {"applicable": False, "reason": reason}


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, ValueError):
        return None


def _summary(record: dict) -> dict:
    return (record.get("end") or {}).get("summary") or {}


def _tool(record: dict) -> str | None:
    return (record.get("start") or {}).get("tool")


def _completed(record: dict) -> bool:
    return (record.get("end") or {}).get("status") in COMPLETED


def _n_images(call_id: str, records: dict) -> int | None:
    """Images a call covered: its summary's n_images; a batch_correct call: its input embed's."""
    record = records.get(call_id) or {}
    n = _summary(record).get("n_images")
    if isinstance(n, int):
        return n
    if _tool(record) == "batch_correct":
        sources = ((record.get("start") or {}).get("inputs") or {}).get("from_calls") or []
        for source in sources:
            if source in records and source != call_id:
                return _n_images(source, records)
    images = ((record.get("start") or {}).get("inputs") or {}).get("images")
    return len(images) if isinstance(images, list) else None


def _n_cells(call_id: str, records: dict) -> int | None:
    summary = _summary(records.get(call_id) or {})
    for key in ("n_cells", "n_crops"):
        if isinstance(summary.get(key), int):
            return summary[key]
    return None


def _note(record: dict) -> str:
    note = (record.get("start") or {}).get("note")
    if not note and record.get("dir") is not None:
        note = (_json(Path(record["dir"]) / "meta.json") or {}).get("note")
    return str(note or "").strip()


def _per_image_values(record: dict) -> dict:
    """{image_id: {lever: value}} the call was given with --per-image (its inputs/ copy, else meta.json)."""
    directory = record.get("dir")
    if directory is None:
        return {}
    values = _json(Path(directory) / "inputs" / "per_image.json")
    if not isinstance(values, dict):
        values = ((_json(Path(directory) / "meta.json") or {}).get("per_image") or {}).get("values")
    return values if isinstance(values, dict) else {}


def _cell_overrides(record: dict) -> dict:
    directory = record.get("dir")
    if directory is None:
        return {}
    values = _json(Path(directory) / "inputs" / "cell_overrides.json")
    if not isinstance(values, dict):
        values = ((_json(Path(directory) / "meta.json") or {}).get("cell_overrides") or {}).get("values")
    return values if isinstance(values, dict) else {}


def _defaults() -> dict[str, dict]:
    """{tool: {lever: default}} from the template's registry (the CARDs of this repository); {} when
    SHAPE cannot be imported here."""
    try:
        from .coverage import REPO, _registry

        tools, tool_levers = _registry()
        return {tool: {name: info.default for name, info in tool_levers(tool, REPO).items()} for tool in tools}
    except Exception:  # noqa: BLE001 -- the registry needs SHAPE's CARD parser; the metrics do not
        return {}


def _lever_groups(records: dict) -> list[str]:
    try:
        from .coverage import lever_groups

        return lever_groups(Path("."))
    except Exception:  # noqa: BLE001 -- without the registry: the groups the ledger names
        groups = []
        for record in records.values():
            start = record.get("start") or {}
            groups += [f"{start.get('tool')}.{lever}" for lever in start.get("set_levers") or []]
        return list(dict.fromkeys(groups))


def _first_full_set(tool: str, records: dict, n_prep: int) -> str | None:
    ordered = sorted((r["start"].get("time") or 0, c) for c, r in records.items()
                     if r.get("start") and _tool(r) == tool and _completed(r))
    for _, call_id in ordered:
        n = _n_images(call_id, records)
        if n_prep and n is not None and n >= FULL_SET_SHARE * n_prep:
            return call_id
    return None


# ---------------------------------------------------------------- the delivered chain, per image
def delivery(run: Path, records: dict, choice: dict | None) -> dict:
    """The chain behind the choice and, per delivered image, the crop call and the mask call it came
    from (cells.csv of the delivered embed call, per_image.csv of its crop calls)."""
    out = {"chain": None, "embed_call": None, "cells_per_image": {}, "crop_of": {}, "mask_of": {}}
    if not isinstance(choice, dict):
        return out
    walked = chain(run, choice)
    out["chain"] = walked
    e_id = walked.get("embed_call")
    if not e_id or e_id not in records or records[e_id].get("dir") is None:
        return out
    out["embed_call"] = e_id
    cells = _csv(Path(records[e_id]["dir"]) / "cells.csv")
    if cells is None or not {"image_id", "crop_call"} <= set(cells.columns):
        return out
    out["cells_per_image"] = cells.groupby("image_id").size().astype(int).to_dict()
    for image_id, crop_calls in cells.groupby("image_id")["crop_call"]:
        out["crop_of"][image_id] = crop_calls.value_counts().index[0]
    for c_id in sorted(set(out["crop_of"].values())):
        record = records.get(c_id) or {}
        table = _csv(Path(record["dir"]) / "per_image.csv") if record.get("dir") is not None else None
        if table is None or "mask_call" not in table.columns:
            continue
        mask_calls = dict(zip(table["image_id"], table["mask_call"]))
        for image_id, crop_call in out["crop_of"].items():
            if crop_call == c_id and mask_calls.get(image_id):
                out["mask_of"][image_id] = mask_calls[image_id]
    return out


# ---------------------------------------------------------------- D1
def _dispersion(values: pd.Series) -> dict:
    present = values[values != ""]
    numbers = pd.to_numeric(present, errors="coerce")
    if len(present) and numbers.notna().all():
        q = np.quantile(numbers.to_numpy(dtype=float), [0, 0.25, 0.5, 0.75, 1])
        return {"n": int(len(present)), "n_distinct": int(numbers.nunique()), "min": float(q[0]),
                "q25": float(q[1]), "median": float(q[2]), "q75": float(q[3]), "max": float(q[4]),
                "iqr": float(q[3] - q[1])}
    return {"n": int(len(present)), "n_distinct": int(present.nunique()),
            "counts": {str(k): int(v) for k, v in present.value_counts().items()}}


def adaptivity(records: dict, deliv: dict, n_prep: int, defaults: dict) -> dict:
    out = {}
    for tool, delivered_by in (("segment", deliv["mask_of"]), ("crop", deliv["crop_of"])):
        images = sorted(delivered_by)
        if not images:
            out[tool] = _na("no delivered images resolved for this tool (no valid chain)")
            continue
        tool_defaults = defaults.get(tool) or {}
        own = non_default = 0
        rows = []
        for image_id in images:
            record = records.get(delivered_by[image_id]) or {}
            params = (record.get("start") or {}).get("params") or {}
            entry = _per_image_values(record).get(image_id) or {}
            if any(params.get(k) != v for k, v in entry.items()):
                own += 1
            if tool_defaults and any(k in tool_defaults and tool_defaults[k] != v for k, v in entry.items()):
                non_default += 1
            table = _csv(Path(record["dir"]) / "per_image.csv") if record.get("dir") is not None else None
            if table is not None:
                row = table[table["image_id"] == image_id]
                if len(row):
                    rows.append({k: row.iloc[0][k] for k in params if k in row.columns})
        resolved = pd.DataFrame(rows, dtype=str).fillna("")
        dispersion = {lever: _dispersion(resolved[lever]) for lever in resolved.columns}
        counts = pd.Series(delivered_by).value_counts()
        main = str(counts.index[0])
        first = _first_full_set(tool, records, n_prep)
        tool_out = {
            "n_delivered_images": len(images),
            "share_images_with_own_setting": own / len(images),
            "share_images_with_non_default_per_image_lever": (non_default / len(images)) if tool_defaults else None,
            "lever_dispersion": dispersion,
            "levers_varying_across_images": sorted(k for k, v in dispersion.items() if v["n_distinct"] > 1),
            "images_by_call": {str(k): int(v) for k, v in counts.items()},
            "main_call": main,
            "rescued_share": 1 - int(counts.iloc[0]) / len(images),
            "first_full_set_call": first,
            "share_not_from_first_full_set_call": (sum(1 for i in images if delivered_by[i] != first) / len(images)
                                                   if first else None)}
        if tool == "crop":
            with_override = [i for i in images if _cell_overrides(records.get(delivered_by[i]) or {}).get(i)]
            tool_out["share_images_with_cell_overrides"] = len(with_override) / len(images)
            tool_out["n_cells_overridden"] = int(sum(len(_cell_overrides(records[delivered_by[i]])[i])
                                                     for i in with_override))
        out[tool] = tool_out
    return out


# ---------------------------------------------------------------- D5, D6
def _full_set(call_id: str, records: dict, n_prep: int, reference_cells: int | None) -> bool:
    n = _n_images(call_id, records)
    if not n_prep or n is None or n < FULL_SET_SHARE * n_prep:
        return False
    if _tool(records[call_id]) in ("embed", "batch_correct", "cluster") and reference_cells:
        cells = _n_cells(call_id, records)
        return cells is not None and cells >= FULL_SET_SHARE * reference_cells
    return True


def lever_coverage(run: Path, records: dict, deliv: dict, n_prep: int, choice: dict | None) -> dict:
    from .coverage import coverage

    groups = _lever_groups(records)
    tested = coverage(records, groups)
    in_chain = set(((deliv.get("chain") or {}).get("calls") or {}))
    embed_cells = [_n_cells(c, records) for c, r in records.items() if _tool(r) == "embed" and _completed(r)]
    reference = _n_cells(deliv["embed_call"], records) if deliv.get("embed_call") else \
        max([c for c in embed_cells if c] or [0]) or None
    given = (choice or {}).get("skipped") if isinstance(choice, dict) else None
    reasons = {k for k, v in (given or {}).items() if isinstance(v, str) and v.strip()} if isinstance(given, dict) else set()
    per_group, token = {}, []
    for group, changed_in in tested.items():
        kinds = {c: ("chain" if c in in_chain else "full_set" if _full_set(c, records, n_prep, reference) else "probe")
                 for c in changed_in}
        if any(k in ("chain", "full_set") for k in kinds.values()):
            status = "substantive"
        elif kinds:
            status = "token"
            token.append(group)
        elif group in reasons or group.split(".", 1)[0] in reasons:
            status = "skipped"
        else:
            status = "untested"
        per_group[group] = {"status": status, "calls": kinds}
    statuses = [g["status"] for g in per_group.values()]
    return {"full_set_share": FULL_SET_SHARE, "reference_cells": reference, "n_groups": len(per_group),
            "n_substantive": statuses.count("substantive"), "n_token": statuses.count("token"),
            "n_skipped": statuses.count("skipped"), "n_untested": statuses.count("untested"),
            "substantive_share": statuses.count("substantive") / len(per_group) if per_group else None,
            "token_only_groups": token, "groups": per_group}


def traceability(records: dict, deliv: dict) -> dict:
    in_chain = sorted(((deliv.get("chain") or {}).get("calls") or {}))
    if not in_chain:
        return _na("no delivered chain")
    notes = {c: _note(records.get(c) or {}) for c in in_chain}
    with_note = [c for c, n in notes.items() if n]
    citing = [c for c, n in notes.items() if re.search(r"\d", CALL_ID.sub("", n))]
    return {"chain_calls": in_chain, "share_with_note": len(with_note) / len(in_chain),
            "share_note_cites_number": len(citing) / len(in_chain), "without_note": sorted(set(in_chain) - set(with_note)),
            "without_number": sorted(set(in_chain) - set(citing))}


# ---------------------------------------------------------------- E1
def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        out[prefix] = out.get(prefix, 0) + value
    elif isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)


def trace_events(run: Path) -> list[tuple[int, dict]]:
    """(round, event) of every parseable event of traces/round_*.jsonl, in order."""
    out = []
    for path in sorted((Path(run) / "traces").glob("round_*.jsonl")):
        try:
            round_no = int(path.stem.split("_", 1)[1])
        except ValueError:
            round_no = -1
        for raw in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                out.append((round_no, event))
    return out


def llm_usage(run: Path) -> dict:
    """Sums over the trace's `result` events (one per completed round). Every value is null with a
    reason in `unavailable_reasons` when no result event carries it."""
    rounds = sorted((Path(run) / "traces").glob("round_*.jsonl"))
    results = [e for _, e in trace_events(run) if e.get("type") == "result"]
    totals = {"n_rounds_traced": len(rounds), "n_result_events": len(results),
              "rounds_without_result_event": len(rounds) - len(results)}
    reasons = {}
    for key in ("num_turns", "total_cost_usd", "duration_api_ms", "duration_ms"):
        values = [e[key] for e in results if isinstance(e.get(key), (int, float)) and not isinstance(e.get(key), bool)]
        totals[key] = sum(values) if values else None
        if not values:
            reasons[key] = "no result event carries it" if results else "no result event in the trace"
    tokens, models = {}, {}
    for event in results:
        _flatten("", event.get("usage") or {}, tokens)
        _flatten("", event.get("modelUsage") or {}, models)
    totals["tokens"] = {k: v for k, v in tokens.items() if "token" in k.lower()} or None
    if totals["tokens"] is None:
        reasons["tokens"] = "no result event with usage"
    totals["thinking_tokens"] = tokens.get("output_tokens_details.thinking_tokens")
    if totals["thinking_tokens"] is None:
        reasons["thinking_tokens"] = "the result events report no thinking tokens"
    totals["by_model"] = models or None
    totals["llm_api_hours"] = totals["duration_api_ms"] / 3.6e6 if totals["duration_api_ms"] is not None else None
    if totals["llm_api_hours"] is None:
        reasons["llm_api_hours"] = reasons.get("duration_api_ms", "no duration_api_ms")
    if totals["rounds_without_result_event"] > 0 and results:
        reasons["partial"] = (f"{totals['rounds_without_result_event']} round(s) ended without a result event "
                              "(killed at the deadline or crashed): their usage is not counted, the sums are lower bounds")
    totals["sources"] = {"num_turns": "result_events", "tokens": "result_events", "thinking_tokens": "result_events",
                         "total_cost_usd": "claude_code_estimate", "duration_api_ms": "result_events",
                         "llm_api_hours": "result_events"}
    totals["usd_source"] = "claude_code_estimate"
    totals["unavailable_reasons"] = reasons
    return totals


def shape_usage(run: Path) -> dict:
    """The SHAPE arm's reasoner usage: SHAPE_RUN.json `usage.total` (shape/engine/usage.py snapshots: engine calls
    and the tokens the provider reported). Every value it does not carry is null with a reason; there is no USD."""
    record = _json(Path(run) / SHAPE_RECORD) or {}
    total = (record.get("usage") or {}).get("total")
    reasons = {"total_cost_usd": "SHAPE's usage records carry no price", "num_turns": "the SHAPE arm has no turns",
               "duration_api_ms": "SHAPE's usage records carry no API time"}
    if not isinstance(total, dict):
        total = {}
        reasons["tokens"] = f"{SHAPE_RECORD} has no usage.total"
    tokens = {k: v for k, v in total.items() if "token" in k and isinstance(v, (int, float))} or None
    if tokens is None:
        reasons.setdefault("tokens", "no token count recorded")
    return {"n_calls": total.get("calls"), "num_turns": None, "total_cost_usd": None, "duration_api_ms": None,
            "llm_api_hours": None, "tokens": tokens, "thinking_tokens": total.get("thinking_tokens"),
            "by_stage": {k: v for k, v in (record.get("usage") or {}).items() if k != "total"} or None,
            "model": record.get("model"), "usd_source": "shape_usage",
            "sources": {"tokens": "shape_usage", "n_calls": "shape_usage", "total_cost_usd": "shape_usage"},
            "unavailable_reasons": reasons}


def tool_cost(records: dict) -> dict:
    per_tool: dict[str, dict] = {}
    intervals = []
    for call_id, record in sorted(records.items()):
        tool = _tool(record) or "?"
        row = per_tool.setdefault(tool, {"calls": 0, "seconds": 0.0, "statuses": {}})
        row["calls"] += 1
        status = (record.get("end") or {}).get("status", "unfinished")
        row["statuses"][status] = row["statuses"].get(status, 0) + 1
        seconds = (record.get("end") or {}).get("seconds")
        if isinstance(seconds, (int, float)):
            row["seconds"] += seconds
        start, end = (record.get("start") or {}).get("time"), (record.get("end") or {}).get("time")
        if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
            intervals.append((start, end))
    busy, concurrent, cursor = 0.0, 0, None
    for start, end in sorted(intervals):
        if cursor is None or start > cursor:
            busy += end - start
            cursor = end
        elif end > cursor:
            busy += end - cursor
            cursor = end
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals], key=lambda x: (x[0], x[1]))
    running = 0
    for _, step in events:
        running += step
        concurrent = max(concurrent, running)
    return {"per_tool": per_tool, "n_calls": len(records),
            "gpu_hours": sum(per_tool.get(t, {}).get("seconds", 0.0) for t in GPU_TOOLS) / 3600,
            "tool_busy_hours": busy / 3600, "max_concurrent_calls": concurrent,
            "span_hours": (max(e for _, e in intervals) - min(s for s, _ in intervals)) / 3600 if intervals else None}


# ---------------------------------------------------------------- F3
def shape_llm_failures(run: Path) -> dict:
    """The SHAPE arm's LLM failures from SHAPE_RUN.json: per stage the images whose search failed and the failed
    questions by label (the verifiers' `execution_errors`: recorded, the run went on), and what stopped the run (a
    fatal LLM error), if anything."""
    record = _json(Path(run) / SHAPE_RECORD) or {}
    failed_searches, questions = {}, {}
    for stage, entry in (record.get("stages") or {}).items():
        if not isinstance(entry, dict):
            continue
        errors = list(entry.get("execution_errors") or [])
        for image, outcome in (entry.get("per_image") or {}).items():
            if not isinstance(outcome, dict):
                continue
            if outcome.get("status") == "failed":
                failed_searches.setdefault(stage, []).append(image)
            errors += outcome.get("execution_errors") or []
        for error in errors:
            label = (error or {}).get("question") or "?"
            questions.setdefault(stage, {})[label] = questions.setdefault(stage, {}).get(label, 0) + 1
    return {"failed_searches": {k: sorted(v) for k, v in failed_searches.items()},
            "failed_questions": questions,
            "n_failed_questions": sum(n for by_label in questions.values() for n in by_label.values()),
            "stopped": record.get("stopped")}


def failures(run: Path, records: dict, deliv: dict, prep_images: list[str], agent: bool,
             not_agent: str = NOT_AGENT) -> dict:
    walked = deliv.get("chain")
    result = _json(Path(run) / "RESULT.json")
    scored = result.get("chain") if isinstance(result, dict) else None
    # the scorer's verdict when it has run (it saw every file; an export lacks masks, crops, arrays)
    verdict = scored if isinstance(scored, dict) and "ok" in scored else walked
    out = {"no_valid_choice": verdict is None or not verdict.get("ok"),
           "choice_problems": ((verdict or {}).get("problems") or ["no readable submission/choice.json"]
                               if verdict is None or not verdict.get("ok") else [])[:10],
           "chain_verdict_from": "RESULT.json" if verdict is scored else "re-walked here",
           "chain_rewalked_ok": bool(walked and walked.get("ok")),
           "failed_calls": sorted(c for c, r in records.items() if (r.get("end") or {}).get("status") == "failed"),
           "partial_calls": sorted(c for c, r in records.items() if (r.get("end") or {}).get("status") == "partial"),
           "unfinished_calls": sorted(c for c, r in records.items() if r.get("end") is None)}
    delivered = deliv.get("cells_per_image") or {}
    empty = sorted(set(prep_images) - set(delivered)) if delivered else None
    out["images_without_delivered_cells"] = empty
    out["n_images_without_delivered_cells"] = len(empty) if empty is not None else None
    audit = (result or {}).get("audit") if isinstance(result, dict) else None
    out["audit_hits"] = len(audit.get("hits") or []) if isinstance(audit, dict) else None
    out["score_status"] = (result or {}).get("status") if isinstance(result, dict) else None
    if not agent:
        out.update(deadline_without_final=_na(not_agent), final_rejections=_na(not_agent), stopped_because=None)
        return out
    agent_run = _json(Path(run) / "AGENT_RUN.json") or {}
    stopped = agent_run.get("stopped_because")
    out["stopped_because"] = stopped
    out["deadline_without_final"] = stopped == "deadline"
    rejections = []
    for line in (Path(run) / "logs" / "rounds.jsonl").read_text(encoding="utf-8", errors="replace").split("\n") \
            if (Path(run) / "logs" / "rounds.jsonl").exists() else []:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("event") == "final_rejected":
            rejections.append({"round": event.get("round"), "missing": event.get("missing") or [],
                               "chain_problems": event.get("chain_problems") or []})
    out["final_rejections"] = {"n": len(rejections), "events": rejections}
    return out


# ---------------------------------------------------------------- the report
def process(run: Path) -> dict:
    run = Path(run)
    records = calls(run)
    arm = run_arm(run)
    agent, decided = arm == "agent", arm in ("agent", "shape")
    choice = _json(run / "workspace" / "submission" / "choice.json")
    images = _csv(run / "workspace" / "data" / "images.csv")
    prep_images = images["image_id"].tolist() if images is not None and "image_id" in images.columns else []
    deliv = delivery(run, records, choice if isinstance(choice, dict) else None)
    notes = run / "workspace" / "NOTES.md"
    cost = {"tools": tool_cost(records)}
    reasons = {}
    if agent:
        agent_run = _json(run / "AGENT_RUN.json") or {}
        cost.update(wall_hours=agent_run.get("wall_hours_used"), budget_hours=agent_run.get("budget_hours"),
                    llm=llm_usage(run))
        wall_source = "AGENT_RUN.json wall_hours_used"
        if cost["wall_hours"] is None:
            reasons["wall_hours"] = "AGENT_RUN.json is missing or has no wall_hours_used"
    elif arm == "shape":
        shape_run = _json(run / SHAPE_RECORD) or {}
        cost.update(wall_hours=shape_run.get("wall_hours"), budget_hours=shape_run.get("budget_hours"),
                    llm=shape_usage(run))
        wall_source = f"{SHAPE_RECORD} wall_hours"
        if cost["wall_hours"] is None:
            reasons["wall_hours"] = f"{SHAPE_RECORD} is missing or has no wall_hours"
    else:
        cost.update(wall_hours=cost["tools"]["span_hours"], budget_hours=None, llm=_na(NOT_AGENT))
        wall_source = "ledger: first tool call start to last end"
        if cost["wall_hours"] is None:
            reasons["wall_hours"] = "no tool call with a start and an end time"
    if not records:
        reasons["gpu_hours"] = reasons["tool_busy_hours"] = "no ledger"
        cost["tools"]["gpu_hours"] = cost["tools"]["tool_busy_hours"] = None
    cost["sources"] = {"wall_hours": wall_source, "budget_hours": SHAPE_RECORD if arm == "shape" else "AGENT_RUN.json",
                       "tools": "ledger (logs/tool_calls.jsonl)",
                       "gpu_hours": "ledger: summed seconds of segment + embed calls",
                       "tool_busy_hours": "ledger: union of call intervals",
                       "llm": {"agent": "trace result events", "shape": f"{SHAPE_RECORD} usage (shape_usage)"}.get(arm)}
    cost["unavailable_reasons"] = reasons
    f3 = failures(run, records, deliv, prep_images, agent, NOT_CLAUDE if arm == "shape" else NOT_AGENT)
    if arm == "shape":
        f3["shape_llm"] = shape_llm_failures(run)
    return {
        "run": run.name, "arm": arm, "agent_run": agent, "n_prep_images": len(prep_images),
        "delivered_chain": sorted(((deliv.get("chain") or {}).get("calls") or {})),
        "notes_md_chars": len(notes.read_text(encoding="utf-8", errors="replace")) if notes.exists() else None,
        "D1_adaptivity": adaptivity(records, deliv, len(prep_images), _defaults()) if decided else _na(NOT_AGENT),
        "D5_lever_coverage": lever_coverage(run, records, deliv, len(prep_images), choice) if decided else _na(NOT_AGENT),
        "D6_traceability": traceability(records, deliv) if decided else _na(NOT_AGENT),
        "E1_cost": cost,
        "F3_failures": f3,
        "end_state": end_state(run, agent, f3, "SHAPE arm" if arm == "shape" else "control"),
        "claims": claims(run, agent) if arm != "shape" else _na(NOT_CLAUDE)}


# ---------------------------------------------------------------- end state and claims (BIABench)
def _rounds_log(run: Path) -> list[dict]:
    path = Path(run) / "logs" / "rounds.jsonl"
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                out.append(event)
    return out


def end_state(run: Path, agent: bool, f3: dict, kind: str = "control") -> dict:
    """{"end_state": delivered | no_deliverable | crashed, "timed_out", "refused", signals}."""
    run = Path(run)
    delivered = not f3.get("no_valid_choice", True)
    if not agent:                   # a fixed chain (the control, the SHAPE arm): it delivered, or it broke
        return {"end_state": "delivered" if delivered else "crashed", "timed_out": False, "refused": False,
                "crash_signals": [] if delivered else [f"the {kind} chain has no valid choice"],
                "refusal_evidence": [], "rule": f"{kind}: delivered = a valid choice, else crashed"}
    agent_run = _json(run / "AGENT_RUN.json")
    stopped = (agent_run or {}).get("stopped_because")
    ends = [e for e in _rounds_log(run) if e.get("event") == "round_end"]
    last_end = ends[-1] if ends else None
    timed_out = stopped != "final_flag" and (stopped == "deadline" or bool(last_end and last_end.get("timed_out")))
    events = trace_events(run)
    last_round = max((r for r, _ in events), default=None)
    signals = []
    if agent_run is None:
        signals.append("no AGENT_RUN.json (the harness did not finish)")
    elif stopped in ("running", "repeated_failures") or str(stopped).startswith("sigterm"):
        signals.append(f"the loop stopped because: {stopped}")
    if last_end is not None and last_end.get("returncode") not in (0, None) and not last_end.get("timed_out"):
        signals.append(f"the last round exited with {last_end.get('returncode')}")
    if last_round is not None and not timed_out and not any(r == last_round and e.get("type") == "result"
                                                           for r, e in events):
        signals.append(f"no result event in the last round ({last_round})")
    refusals = []
    for round_no, event in events:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        if event.get("stop_reason") in REFUSAL_STOP_REASONS or message.get("stop_reason") in REFUSAL_STOP_REASONS:
            refusals.append(f"round {round_no}: stop_reason refusal")
        elif event.get("type") == "result" and event.get("is_error") and not str(event.get("result") or "").strip():
            refusals.append(f"round {round_no}: an error result with an empty response")
    state = "delivered" if delivered else "crashed" if signals else "no_deliverable"
    return {"end_state": state, "timed_out": timed_out, "refused": bool(refusals), "stopped_because": stopped,
            "crash_signals": signals, "refusal_evidence": refusals[:10],
            "rule": "delivered = a valid choice (the scorer's chain verdict); crashed = no valid choice and the loop "
                    "ended on an error (no AGENT_RUN.json, a stop on failures / SIGTERM / still running, a non-zero "
                    "exit, or no result event in the last round of a run that did not time out); otherwise "
                    "no_deliverable; timed_out = stopped at the deadline without an accepted FINAL"}


def claims(run: Path, agent: bool) -> dict:
    """Whether the agent claimed completion: FINAL accepted, or CLAIM_RULE on its closing text."""
    if not agent:
        return _na(NOT_AGENT)
    from .trace_md import redact, run_secrets

    stopped = (_json(Path(run) / "AGENT_RUN.json") or {}).get("stopped_because")
    results = [e for _, e in trace_events(run) if e.get("type") == "result"]
    text = str(results[-1].get("result") or "") if results else ""
    text = redact(text, run_secrets(Path(run)))[0][:CLOSING_CHARS]
    final_flag = stopped == "final_flag"
    by_text = bool(re.search(CLAIM_RULE, text))
    return {"final_flag": final_flag, "closing_text": text or None, "closing_text_claims": by_text,
            "claimed_complete": final_flag or by_text, "claim_rule": CLAIM_RULE,
            "rule": "claimed_complete = FINAL accepted or the closing text matches claim_rule"}


def _f(value, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def markdown(report: dict) -> str:
    lines = [f"# Process metrics: {report['run']}", "",
             f"Delivered chain: {', '.join(report['delivered_chain']) or 'none'}; prep images: {report['n_prep_images']}.", ""]
    d1 = report["D1_adaptivity"]
    lines += ["## D1 adaptivity", ""]
    if d1.get("applicable") is False:
        lines.append(f"Not applicable: {d1['reason']}.")
    else:
        lines += ["| tool | own per-image setting | non-default per-image lever | varying levers | rescued (not main call) "
                  "| not from first full-set call |", "| --- | --- | --- | --- | --- | --- |"]
        for tool, t in d1.items():
            if t.get("applicable") is False:
                lines.append(f"| {tool} | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(f"| {tool} | {_f(t['share_images_with_own_setting'])} | "
                         f"{_f(t['share_images_with_non_default_per_image_lever'])} | "
                         f"{', '.join(t['levers_varying_across_images']) or 'none'} | {_f(t['rescued_share'])} "
                         f"({t['main_call']} main) | {_f(t['share_not_from_first_full_set_call'])} |")
    d5 = report["D5_lever_coverage"]
    lines += ["", "## D5 lever coverage", ""]
    lines.append(f"Not applicable: {d5['reason']}." if d5.get("applicable") is False else
                 f"{d5['n_substantive']} of {d5['n_groups']} groups substantive, {d5['n_token']} token only "
                 f"({', '.join(d5['token_only_groups']) or 'none'}), {d5['n_skipped']} skipped, {d5['n_untested']} untested.")
    d6 = report["D6_traceability"]
    lines += ["", "## D6 traceability", ""]
    lines.append(f"Not applicable: {d6['reason']}." if d6.get("applicable") is False else
                 f"Delivered-chain calls with a note: {_f(d6['share_with_note'])}; citing a number: "
                 f"{_f(d6['share_note_cites_number'])}.")
    e1 = report["E1_cost"]
    tools = e1["tools"]
    lines += ["", "## E1 cost", "",
              f"Wall hours {_f(e1['wall_hours'], '.2f')}; GPU-hours {_f(tools['gpu_hours'], '.2f')}; time with a tool "
              f"call running {_f(tools['tool_busy_hours'], '.2f')} h; max concurrent calls {tools['max_concurrent_calls']}; "
              f"tool calls {tools['n_calls']} ("
              + ", ".join(f"{t} {r['calls']}" for t, r in tools["per_tool"].items()) + ")."]
    llm = e1["llm"]
    if llm.get("usd_source") == "shape_usage":
        lines.append(f"LLM (SHAPE's engines, {llm.get('model')}): {llm['n_calls'] if llm['n_calls'] is not None else 'n/a'} "
                     "calls, tokens " + (", ".join(f"{k} {v:,}" for k, v in llm["tokens"].items()) if llm["tokens"]
                                         else "n/a") + "; no USD (SHAPE's usage records carry no price).")
    elif llm.get("applicable") is not False:
        lines.append(f"LLM: {llm['num_turns'] if llm['num_turns'] is not None else 'n/a'} turns, USD "
                     f"{_f(llm['total_cost_usd'], '.2f')} (Claude Code's list-price estimate, not gateway billing), "
                     f"API time {_f(llm['llm_api_hours'], '.2f')} h, tokens "
                     + (", ".join(f"{k} {v:,}" for k, v in llm["tokens"].items()) if llm["tokens"] else "n/a") + ".")
        if llm["unavailable_reasons"]:
            lines.append("Unavailable: " + "; ".join(f"{k}: {v}" for k, v in llm["unavailable_reasons"].items()) + ".")
    f3 = report["F3_failures"]
    lines += ["", "## F3 failures", "",
              f"No valid choice: {f3['no_valid_choice']}; stopped because: {f3['stopped_because']}; audit hits: "
              f"{f3['audit_hits']}; images without delivered cells: {f3['n_images_without_delivered_cells']}; failed "
              f"calls {f3['failed_calls'] or 'none'}; partial {f3['partial_calls'] or 'none'}; unfinished "
              f"{f3['unfinished_calls'] or 'none'}."]
    shape_llm = f3.get("shape_llm")
    if isinstance(shape_llm, dict):
        lines.append(f"SHAPE LLM: failed questions {shape_llm['n_failed_questions']} "
                     f"({shape_llm['failed_questions'] or 'none'}); failed searches "
                     f"{shape_llm['failed_searches'] or 'none'}; stopped by {shape_llm['stopped'] or 'nothing'}.")
    rejections = f3.get("final_rejections")
    if isinstance(rejections, dict) and "n" in rejections:
        lines.append(f"FINAL rejections: {rejections['n']}.")
    end = report["end_state"]
    lines += ["", "## End state", "",
              f"{end['end_state']}; timed out: {end['timed_out']}; refused: {end['refused']}"
              + (f"; signals: {'; '.join(end['crash_signals'])}" if end["crash_signals"] else "") + "."]
    claim = report["claims"]
    if claim.get("applicable") is not False:
        lines.append(f"Claimed completion: {claim['claimed_complete']} (FINAL accepted {claim['final_flag']}, closing "
                     f"text {claim['closing_text_claims']}).")
    return "\n".join(lines) + "\n"


def write(run: Path, out: Path | None = None) -> dict:
    """PROCESS.json and PROCESS.md into `out` (default: the run directory)."""
    report = process(run)
    out = Path(out) if out is not None else Path(run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "PROCESS.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "PROCESS.md").write_text(markdown(report), encoding="utf-8")
    return report
