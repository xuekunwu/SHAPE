"""Convert the raw agent event streams into one human-readable trace.md (+ trace.json).

Inputs (inside a run directory):
  traces/round_XX.jsonl  verbatim Claude Code stream-json events
  traces/round_XX.times  arrival time of each line (written by agents.stream)
  logs/rounds.jsonl      harness events (round start/end, backoff, rejected FINAL)
  logs/tool_calls.jsonl  the tool ledger (one start and one end line per tool call)
  RESULT.json / .md      final harness score (optional)

Every agent event becomes a numbered step: reasoning, message, tool call (with its full input),
tool result (long outputs are shortened; the verbatim text stays in the .jsonl that is exported
next to trace.md), usage. Unknown event types are kept as raw JSON so nothing is dropped.
"""
from __future__ import annotations

import functools
import json
import re
import subprocess
import time
from pathlib import Path

RESULT_LIMIT = 4000
INPUT_LIMIT = 8000
SUMMARY_LIMIT = 600
# Credential-shaped strings are masked in trace.md and in every exported file (an agent may
# have run `env`).
SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{16,}", r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",
    r"(?i)((?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{16,}")]


@functools.lru_cache(maxsize=None)
def _helper_output(helper: str) -> str | None:
    try:
        out = subprocess.run(helper, shell=True, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and len(value) >= 8 else None


def run_secrets(run: Path) -> tuple[str, ...]:
    """The gateway key, in memory only: the output of the run's apiKeyHelper
    (<run>/claude_config/settings.json). Never printed or written; only used to mask it."""
    try:
        helper = json.loads((Path(run) / "claude_config" / "settings.json").read_text(encoding="utf-8"))["apiKeyHelper"]
    except (OSError, ValueError, KeyError, TypeError):
        return ()
    value = _helper_output(str(helper))
    return (value,) if value else ()


def redact(text: str, secrets: tuple[str, ...] = ()) -> tuple[str, int]:
    """Mask every exact secret value, then every credential-shaped string."""
    count = 0
    for secret in secrets:
        count += text.count(secret)
        text = text.replace(secret, "[REDACTED]")
    for pattern in SECRET_PATTERNS:
        text, n = pattern.subn(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
        count += n
    return text, count


def _shorten(text: str, limit: int, where: str) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-limit // 3:]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} chars omitted; verbatim in {where}] ...\n{tail}"


def _fence(text: str, lang: str = "") -> str:
    run = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, run + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or (f"[{block.get('type')} block]"))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if path.exists():
        # split on "\n" only: str.splitlines() also splits on U+2028 etc., which JSON can carry raw
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def parse_round(path: Path, round_index: int) -> list[dict]:
    """Normalize one round's events into steps: {round, t, line, kind, ...}."""
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    times_path = path.with_suffix(".times")
    times = times_path.read_text().split() if times_path.exists() else []
    steps, tool_names = [], {}

    def add(line_no, kind, **fields):
        t = float(times[line_no]) if line_no < len(times) else None
        steps.append({"round": round_index, "t": t, "line": line_no + 1, "kind": kind, **fields})

    for n, raw in enumerate(lines):
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            if raw.strip():
                add(n, "raw", text=raw)
            continue
        if not isinstance(ev, dict):
            add(n, "raw", text=raw)
            continue
        etype = ev.get("type")
        if etype == "system" and ev.get("subtype") == "init":
            add(n, "system", text=f"session {ev.get('session_id')} model={ev.get('model')} "
                                  f"cwd={ev.get('cwd')} tools={ev.get('tools')}")
        elif etype == "system":
            add(n, "system", text=raw)
        elif etype == "assistant":
            for block in (ev.get("message") or {}).get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    add(n, "thinking", text=block.get("thinking", ""))
                elif btype == "redacted_thinking":
                    add(n, "thinking", text="[redacted thinking]")
                elif btype == "text":
                    add(n, "message", text=block.get("text", ""))
                elif btype == "tool_use":
                    tool_names[block.get("id")] = block.get("name")
                    add(n, "tool_call", name=block.get("name"), input=block.get("input", {}))
                else:
                    add(n, "raw", text=json.dumps(block))
        elif etype == "user":
            content = (ev.get("message") or {}).get("content", [])
            if isinstance(content, str):
                add(n, "message", name="user", text=content)
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    add(n, "tool_result", name=tool_names.get(block.get("tool_use_id"), "?"),
                        text=_result_text(block.get("content")), error=bool(block.get("is_error")))
                elif isinstance(block, dict) and block.get("type") == "text":
                    add(n, "message", name="user", text=block.get("text", ""))
                else:
                    add(n, "raw", text=json.dumps(block)[:2000])
        elif etype == "result":
            add(n, "usage", text=(f"turns={ev.get('num_turns')} duration_ms={ev.get('duration_ms')} "
                                  f"cost_usd={ev.get('total_cost_usd')} usage={json.dumps(ev.get('usage'))} "
                                  f"subtype={ev.get('subtype')}"), usage=ev.get("usage"))
        else:
            add(n, "raw", text=raw)
    return steps


def load_steps(run: Path) -> list[dict]:
    steps = []
    for path in sorted((run / "traces").glob("round_*.jsonl")):
        steps.extend(parse_round(path, int(path.stem.split("_")[1])))
    for name, kind in (("rounds.jsonl", "harness"), ("tool_calls.jsonl", "tool")):
        for rec in _read_jsonl(run / "logs" / name):
            steps.append({"round": rec.get("round"), "t": rec.get("time"), "kind": kind, "record": rec})
    # Stable merge by time; steps without a time keep their file order at the end.
    return sorted(steps, key=lambda s: (s["t"] if isinstance(s["t"], (int, float)) else float("inf")))


def _render_input(name: str, data: dict, where: str) -> str:
    if name == "Bash" and isinstance(data, dict) and "command" in data:
        command = data["command"] if isinstance(data["command"], str) else " ".join(map(str, data["command"]))
        desc = f"*{data['description']}*\n\n" if data.get("description") else ""
        return desc + _fence(_shorten(command, INPUT_LIMIT, where), "bash")
    if name == "Write" and isinstance(data, dict):
        return f"`{data.get('file_path')}`\n\n" + _fence(_shorten(str(data.get("content", "")), INPUT_LIMIT, where))
    if name == "Edit" and isinstance(data, dict):
        return (f"`{data.get('file_path')}`\n\nold:\n" + _fence(_shorten(str(data.get("old_string", "")), INPUT_LIMIT // 2, where))
                + "\n\nnew:\n" + _fence(_shorten(str(data.get("new_string", "")), INPUT_LIMIT // 2, where)))
    return _fence(_shorten(json.dumps(data, indent=1), INPUT_LIMIT, where), "json")


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


def _call_note(run: Path, rec: dict) -> str | None:
    """The agent's --note of a call, from its meta.json."""
    if not rec.get("call_dir"):
        return None
    meta = _load_json(run / "workspace" / rec["call_dir"] / "meta.json")
    return meta.get("note") if isinstance(meta, dict) else None


def _tool_table(records: list[dict]) -> list[str]:
    """Per tool: calls started, ended by status, summed seconds."""
    table: dict[str, dict] = {}
    for rec in records:
        row = table.setdefault(rec.get("tool") or "?", {"started": 0, "done": 0, "partial": 0, "failed": 0,
                                                        "seconds": 0.0})
        if rec.get("event") == "start":
            row["started"] += 1
        elif rec.get("event") == "end":
            if rec.get("status") in row:
                row[rec["status"]] += 1
            if isinstance(rec.get("seconds"), (int, float)):
                row["seconds"] += rec["seconds"]
    lines = ["| tool | started | done | partial | failed | minutes |", "| --- | --- | --- | --- | --- | --- |"]
    for tool, row in table.items():
        lines.append(f"| {tool} | {row['started']} | {row['done']} | {row['partial']} | {row['failed']} | "
                     f"{row['seconds'] / 60:.1f} |")
    return lines


def render(run: Path) -> str:
    steps = load_steps(run)
    agent = _load_json(run / "AGENT_RUN.json") or {}
    result = _load_json(run / "RESULT.json")
    choice = _load_json(run / "workspace" / "submission" / "choice.json")
    start = agent.get("started") or next((s["t"] for s in steps if isinstance(s["t"], (int, float))), time.time())
    kinds = {}
    for s in steps:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    out = [f"# full_HCMA trace: {run.name}", ""]
    out += ["| field | value |", "| --- | --- |"]
    for key in ("agent", "model", "budget_hours", "wall_hours_used", "n_rounds", "stopped_because"):
        out.append(f"| {key} | {agent.get(key)} |")
    out.append(f"| steps | {sum(v for k, v in kinds.items() if k not in ('harness', 'tool'))} "
               f"({', '.join(f'{k}={v}' for k, v in sorted(kinds.items()))}) |")
    if isinstance(choice, dict):
        out.append(f"| choice | cluster_call={choice.get('cluster_call')} resolution={choice.get('resolution')} |")
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                out.append(f"| RESULT {key} | {value} |")
    tool_records = [s["record"] for s in steps if s["kind"] == "tool"]
    if tool_records:
        out += ["", "Tool calls (ledger `logs/tool_calls.jsonl`):", "", *_tool_table(tool_records)]
    out += ["", "Raw events: `traces/round_XX.jsonl` (line numbers below refer to them).", ""]

    number = 0
    for s in steps:
        stamp = f"+{(s['t'] - start) / 60:.1f} min" if isinstance(s["t"], (int, float)) else "time n/a"
        where = f"traces/round_{s['round']:02d}.jsonl:{s.get('line')}" if s.get("line") else ""
        kind = s["kind"]
        if kind == "harness":
            rec = s["record"]
            event = rec.get("event")
            if event == "round_start":
                out += ["", f"## Round {rec.get('round')} ({stamp})", "", "Prompt given to the agent:", "",
                        _fence(_shorten(rec.get("prompt", ""), INPUT_LIMIT, "logs/rounds.jsonl"))]
            elif event == "round_end":
                out += ["", f"> **harness** round {rec.get('round')} ended ({stamp}): rc={rec.get('returncode')} "
                        f"timed_out={rec.get('timed_out')} {rec.get('seconds')} s"
                        + (f"; killed leftovers {rec['killed_leftovers']}" if rec.get("killed_leftovers") else "")]
                if rec.get("returncode") not in (0, None) and rec.get("stderr_tail"):
                    out += ["", _fence(rec["stderr_tail"][-1500:])]
            elif event == "backoff":
                out += ["", f"> **harness** failed round {rec.get('round')}; backing off {rec.get('seconds')} s "
                            f"(consecutive failures {rec.get('failures')})"]
            elif event == "final_rejected":
                out += ["", f"> **harness** FINAL rejected before round {rec.get('round')} ({stamp}): "
                            f"missing {rec.get('missing')}; chain problems {rec.get('chain_problems')}"]
            elif event == "agent_done":
                out += ["", f"> **harness** agent finished ({stamp}): {rec.get('stopped_because')}"]
            elif event in ("sigterm", "live_trace_error"):
                out += ["", f"> **harness** {event} ({stamp}) {rec.get('error') or ''}".rstrip()]
            continue
        if kind == "tool":
            rec = s["record"]
            head = f"> **tool call {rec.get('call')} {rec.get('tool')}"
            if rec.get("event") == "start":
                note = _call_note(run, rec)
                out += ["", f"{head} `{rec.get('name')}` started** ({stamp}) levers set: "
                            f"{rec.get('set_levers') or []} values: {json.dumps(rec.get('set') or {})}"
                        + (f"; inputs: {_shorten(json.dumps(rec.get('inputs')), SUMMARY_LIMIT, 'logs/tool_calls.jsonl')}"
                           if rec.get("inputs") else "")
                        + (f"; note: {note!r}" if note else "")]
            elif rec.get("event") == "end":
                out += ["", f"{head} ended** ({stamp}) status={rec.get('status')} {rec.get('seconds')} s; summary: "
                            f"{_shorten(json.dumps(rec.get('summary')), SUMMARY_LIMIT, 'logs/tool_calls.jsonl')}"
                        + (f"; error: {rec.get('error')!r}" if rec.get("error") else "")]
            continue
        number += 1
        header = f"### Step {number} · {kind}" + (f" · {s['name']}" if s.get("name") else "") + f" · {stamp}"
        if where:
            header += f" · `{where}`"
        out += ["", header, ""]
        if kind == "tool_call":
            out.append(_render_input(s.get("name", ""), s.get("input"), where))
        elif kind == "tool_result":
            flag = " (error)" if s.get("error") else ""
            out += [f"result{flag}:", "", _fence(_shorten(s.get("text", ""), RESULT_LIMIT, where))]
        elif kind in ("thinking", "message"):
            text = s.get("text", "")
            out.append("\n".join("> " + line for line in text.splitlines()) if kind == "thinking" else text)
        else:
            out.append(_fence(_shorten(s.get("text", ""), RESULT_LIMIT, where)))

    result_md = run / "RESULT.md"
    if result_md.exists():
        out += ["", "## Final harness score", "", result_md.read_text(encoding="utf-8", errors="replace")]
    return "\n".join(out) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    # readers (a user tailing the file, a later export) never see a half-written file
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write(run: Path) -> Path:
    path = run / "trace.md"
    _atomic_write(path, redact(render(run), run_secrets(run))[0])
    return path


def write_json(run: Path) -> Path:
    """trace.json: the normalized, time-ordered steps (the same content as trace.md)."""
    agent = _load_json(run / "AGENT_RUN.json") or {}
    payload = {"generated_at": time.time(), "run": run.name, "agent_run": agent, "steps": load_steps(run)}
    path = run / "trace.json"
    _atomic_write(path, redact(json.dumps(payload, indent=1, default=str), run_secrets(run))[0])
    return path


def write_all(run: Path) -> None:
    run = Path(run)
    write(run)
    write_json(run)
