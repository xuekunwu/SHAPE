"""trace.md / trace.json of a SHAPE-arm run, and the export tarball.

Inputs: traces/round_00.jsonl (the arm's events, {"type": "shape", "kind": ...}), logs/tool_calls.jsonl
(the tool-call ledger), AGENT_RUN.json and RESULT.json. Every event becomes a numbered step (the
reasoner's prompts and answers, admission, waves, tool calls and results, the final decision); long
texts are shortened in trace.md and kept verbatim in the exported jsonl. Credential-shaped strings are
masked in trace.md and in every exported file.
"""
from __future__ import annotations

import io
import json
import re
import tarfile
import time
from pathlib import Path

from benchmarks.batch_correction_JUMP.runners.score import jsonl

TEXT_LIMIT = 4000
SECRET_PATTERNS = [re.compile(p) for p in (
    r"sk-[A-Za-z0-9_\-]{16,}", r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}",
    r"(?i)((?:api[_-]?key|token|secret)[\"']?\s*[:=]\s*[\"']?)[A-Za-z0-9._\-]{16,}")]
EXPORT_TOP = ("trace.md", "trace.json", "RESULT.md", "RESULT.json", "AGENT_RUN.json", "RUN_META.json", "NOTES.md",
              "traces", "logs", "efficiency", "submission", "modules")
EXPORT_SUFFIXES = {".md", ".json", ".jsonl", ".times", ".log", ".csv", ".txt", ".png"}
EXPORT_TOP_SUFFIXES = {"modules": {".py"}}      # also exported under these tops (RxRx1 llm arm: its module sources)
EXPORT_MAX_BYTES = 20_000_000


def redact(text: str) -> tuple[str, int]:
    count = 0
    for pattern in SECRET_PATTERNS:
        text, n = pattern.subn(lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]", text)
        count += n
    return text, count


def _shorten(text: str, limit: int = TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-limit // 3:]
    return f"{head}\n... [{len(text) - len(head) - len(tail)} chars omitted; verbatim in traces/round_00.jsonl] ...\n{tail}"


def _fence(text: str, lang: str = "") -> str:
    fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def load_steps(run: Path) -> list[dict]:
    steps = [{"t": e.get("time"), "line": i + 1, **e} for i, e in enumerate(jsonl(run / "traces" / "round_00.jsonl"))]
    steps += [{"t": r.get("time"), "kind": "ledger", "record": r} for r in jsonl(run / "logs" / "tool_calls.jsonl")]
    return sorted(steps, key=lambda s: (s["t"] if s["t"] is not None else float("inf")))


def _f(value, spec=".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def render(run: Path) -> str:
    steps = load_steps(run)
    agent = json.loads((run / "AGENT_RUN.json").read_text()) if (run / "AGENT_RUN.json").exists() else {}
    result = json.loads((run / "RESULT.json").read_text()) if (run / "RESULT.json").exists() else None
    start = agent.get("started") or next((s["t"] for s in steps if s["t"]), time.time())
    out = [f"# SHAPE batch-correction trace: {run.name}", "", "| field | value |", "| --- | --- |"]
    for key in ("agent", "model", "max_rounds", "wall_hours_used", "stopped_because", "tokens"):
        out.append(f"| {key} | {agent.get(key)} |")
    if result:
        out.append(f"| FINAL decision | {result['decision']} ({result['tool_run']['status']}); "
                   f"non-default {result['non_default_parameters']} |")
        for key in ("agent_vs_default", "agent_vs_default_pca50", "agent_vs_raw"):
            for metric in ("overall", "nonrep_mean_map"):
                b = result["comparisons_test"][key][metric]
                out.append(f"| FINAL test {metric}: {key} | delta {_f(b['delta'], '+.4f')} +- {_f(b['band_2se'])} |")
    out += ["", "Raw events: `traces/round_00.jsonl` (line numbers below refer to it).", ""]
    number = 0
    for s in steps:
        stamp = f"+{(s['t'] - start) / 60:.1f} min" if s["t"] else "time n/a"
        if s["kind"] == "ledger":
            rec = s["record"]
            if rec.get("event") == "start":
                out += ["", f"> **tool call started {rec.get('call')}** ({stamp}) parent `{rec.get('parent')}`, set "
                            f"{rec.get('set')}; expected {rec.get('expected')}\n>\n> diagnosis: {rec.get('diagnosis')}"]
            else:
                val = rec.get("val") or {}
                recon = {m: r.get("label") for m, r in (rec.get("reconciliation") or {}).items()}
                out += ["", f"> **tool call ended {rec.get('call')}** ({stamp}) status={rec.get('status')} "
                            f"tool={rec.get('tool_status')} {rec.get('seconds')} s"
                            + (f"; calibration overall {_f(val.get('overall'))} nonrep {_f(val.get('nonrep_mean_map'))}"
                               if val else "") + (f"; expectations {recon}" if recon else "")
                            + (f" error={rec.get('error')}" if rec.get("error") else "")]
            continue
        number += 1
        kind = s.get("kind")
        out += ["", f"### Step {number} · {kind}" + (f" · {s['label']}" if s.get("label") else "")
                + f" · {stamp} · `traces/round_00.jsonl:{s['line']}`", ""]
        if kind == "tool_call":
            out.append(_fence(json.dumps(s.get("input"), indent=1, default=str), "json"))
        elif kind == "thinking":
            out.append("\n".join("> " + line for line in _shorten(str(s.get("text", ""))).splitlines()))
        else:
            out.append(_fence(_shorten(str(s.get("text", "")))))
    if result:
        out += ["", "## Final score", "", result.get("markdown_table", "")]
    return "\n".join(out) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_all(run: Path) -> None:
    _atomic_write(run / "trace.md", redact(render(run))[0])
    agent = json.loads((run / "AGENT_RUN.json").read_text()) if (run / "AGENT_RUN.json").exists() else {}
    payload = {"generated_at": time.time(), "run": run.name, "agent_run": agent, "steps": load_steps(run)}
    _atomic_write(run / "trace.json", redact(json.dumps(payload, indent=1, default=str))[0])


def export(run: Path) -> Path:
    """Tarball of everything needed to audit the run; the arrays stay behind."""
    out = run / f"EXPORT_{run.name}.tar.gz"
    redactions, skipped = {}, []

    def add(tar, rel: str, data: bytes, mtime: float):
        info = tarfile.TarInfo(f"{run.name}/{rel}")
        info.size, info.mtime = len(data), int(mtime)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(out, "w:gz") as tar:
        for name in EXPORT_TOP:
            top = run / name
            for path in ([top] if top.is_file() else sorted(top.rglob("*")) if top.exists() else []):
                if not path.is_file():
                    continue
                rel, size = path.relative_to(run).as_posix(), path.stat().st_size
                suffixes = EXPORT_SUFFIXES | EXPORT_TOP_SUFFIXES.get(name, set())
                if path.suffix not in suffixes or (name not in ("traces", "logs", "trace.md", "trace.json")
                                                   and size > EXPORT_MAX_BYTES):
                    skipped.append({"path": rel, "bytes": size})
                    continue
                if path.suffix == ".png":
                    add(tar, rel, path.read_bytes(), path.stat().st_mtime)
                    continue
                text, n = redact(path.read_text(encoding="utf-8", errors="replace"))
                if n:
                    redactions[rel] = n
                add(tar, rel, text.encode("utf-8"), path.stat().st_mtime)
        add(tar, "EXPORT_MANIFEST.json", json.dumps({"redactions": redactions, "skipped_not_exported": skipped},
                                                    indent=1).encode(), time.time())
    print(f"EXPORT {out} ({out.stat().st_size / 1e6:.1f} MB); redacted {sum(redactions.values())} credential-shaped "
          f"strings in {len(redactions)} files; {len(skipped)} files not exported")
    return out
