#!/usr/bin/env python3
"""A small, curated, text-only copy of one full_CPJUMP1 experiment directory (a run dir on the VM or the cluster).

The experiment dir holds arm dirs (control/ shape/ codex/ claude/ biomni/), rerun dirs (shape_rerun_*/), comparison
dirs (comparison*/), post-hoc dirs (posthoc_rerun_*/), a failed run dir (shape_failed_*/) and driver logs. Most of it is
heavy (cells.json ~95 MB, embedding.npy, per-image segment/ and crop/ trees). This script copies only what the
reported results and their audit need, deterministically, and checks the copy against a MANIFEST.json:

    python export_run_records.py --run <experiment dir> --out <results/experiment_id> [--cap-mb 80]
                                 [--file-cap-mb 8] [--extra-log FILE]...
    python export_run_records.py --check <results/experiment_id>

Output layout (README.md and an analysis/ folder are written by hand or derived from the exported tree by analysis
scripts; the exporter never writes or checks them):

    <out>/<arm run>/             one folder per arm run (and per rerun / failed run), relative paths kept
    <out>/compare/<comparison>/  comparison* dirs
    <out>/posthoc/<posthoc>/     posthoc_rerun_* dirs
    <out>/logs/                  the run dir's own *.log / *.out / *.err / *.txt and every --extra-log FILE
    <out>/summary.csv            one row per arm run: verdict metrics, retrieval endpoints, seconds and tokens per
                                 stage, commit, reasoning table (no deltas)
    <out>/MANIFEST.json          sha256 + size per file, the `skipped` files (over a cap) and the `excluded` entries

Per arm dir the whitelist is: top-level *.json / *.jsonl / RESULT.md / completion markers / profiles.csv /
states.csv / umap.csv (the per-cell tables cells.json and development_cells.json never); evaluation/*.csv, *.json
and full_metrics_draws.npz (the one binary: verdicts are recomputed from it); logs/*.log and *.jsonl (whole files only, a line is never truncated); ssl/call_*.
params.json and call_*.result.json; batch/call_*/params.json and result.json; agent/<stage>/{CALLS,ROUNDS,SELECTED,
INFER_CONFIG}.json and, per agent/<stage>/calls/<call>/, the small ssl_result.json, FEEDBACK.json, CONFIG.json,
SSL_SELECTION.json and batch_SEARCH.json. A failed run dir keeps RUN_META.json, CONFIG.json, *.err / *.out and logs.
Never copied: cells.json, development_cells.json, *.npy, *.h5ad, *.pt, *.pth, *.tif*, *.png, the per-image
segment/<img>/ and crop/<img>/ trees (replaced by search_summary.json per arm), heavy files under agent/*/calls/*/,
anything under a `private` path.

Caps (1 MB = 2**20 bytes): a file over --file-cap-mb (default 8) is listed in MANIFEST.json `skipped` with its size
and reason; when the total would pass --cap-mb (default 80) the remaining files are skipped the same way, the
generated files first, then the sources smallest first. Nothing is dropped without a `skipped` or `excluded` entry
(`excluded` lists the top-level entries of each run dir that no rule selects; heavy files nested inside a copied
folder, e.g. agent/*/calls/*/, are not listed one by one).

Every copied text file is scanned for API keys (`sk-` + 16 alphanumerics, `Bearer `, `LLM_API_KEY=` + a value,
`api.env`); a hit aborts the export naming the file, before anything is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

SCHEMA = "full-cpjump1-run-export/1"
MIB = 1024 * 1024
DEFAULT_CAP_MB = 80.0
DEFAULT_FILE_CAP_MB = 8.0
CALL_FILE_CAP = MIB                       # the per-call JSON of agent/<stage>/calls/<call>/ is small by design
STAGES = ("segment", "crop", "embed", "batch", "cluster")
RETRIEVAL = ("uncorrected_cross_guide_cross_plate", "uncorrected_same_guide_cross_plate",
             "delivered_cross_guide_cross_plate", "delivered_same_guide_cross_plate")
VERDICT_METRICS = ("mAP", "macro_f1", "balanced_accuracy")
MARKERS = ("CPJUMP1_COMPLETE", "RERUN_COMPLETE", "SMOKE_COMPLETE")
ARM_TOP_NAMES = ("RESULT.md", "profiles.csv", "states.csv", "umap.csv") + MARKERS
AGENT_STAGE_FILES = ("CALLS.json", "ROUNDS.json", "SELECTED.json", "INFER_CONFIG.json")
AGENT_CALL_FILES = ("ssl_result.json", "FEEDBACK.json", "CONFIG.json", "SSL_SELECTION.json", "batch_SEARCH.json")
DRIVER_LOG_SUFFIXES = (".log", ".out", ".err", ".txt")
DENY_SUFFIXES = (".npy", ".h5ad", ".pt", ".pth", ".tif", ".tiff", ".png")
DENY_NAMES = ("cells.json", "development_cells.json")       # per-cell tables (95 MB and 6 MB)
BINARY_SUFFIXES = (".npz",)
SECRET_PATTERNS = {
    "sk-<16+ alphanumerics>": re.compile(rb"sk-[A-Za-z0-9]{16,}"),
    "Bearer ": re.compile(rb"Bearer "),
    "LLM_API_KEY=<value>": re.compile(rb"LLM_API_KEY=\S"),
    "api.env": re.compile(rb"api\.env"),
}
ROOT_UNCHECKED = ("MANIFEST.json", "README.md")      # not listed in the manifest, not reported by --check
ANALYSIS_DIR = "analysis"    # derived by analysis scripts from the exported tree (e.g. runners/pairwise_deltas.py): unchecked


class ExportError(RuntimeError):
    """A refused export (secret found, bad arguments); the message names the file or argument."""


class Pick(NamedTuple):
    rel: str            # path below the output folder, '/'-separated
    src: Path
    size: int
    cap: int | None     # a per-file cap that replaces the default one (small per-call JSON)
    kind: str           # record | log


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _files(d: Path) -> list[Path]:
    """Regular files of a directory (no symlinks), sorted by name; [] when `d` is not a directory."""
    if not d.is_dir():
        return []
    with os.scandir(d) as it:
        return sorted((Path(e.path) for e in it if e.is_file(follow_symlinks=False)), key=lambda p: p.name)


def _dirs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    with os.scandir(d) as it:
        return sorted((Path(e.path) for e in it if e.is_dir(follow_symlinks=False)), key=lambda p: p.name)


def denied(rel: str) -> bool:
    """The hard denylist, applied to every pick whatever the whitelist said."""
    parts = rel.split("/")
    name = parts[-1].lower()
    return ("private" in (p.lower() for p in parts) or name in DENY_NAMES or name.endswith(DENY_SUFFIXES))


def _pick(out: list, rel: str, src: Path, kind: str = "record", cap: int | None = None) -> None:
    if not denied(rel):
        out.append(Pick(rel, src, src.stat().st_size, cap, kind))


def arm_picks(arm: Path, base: str, failed: bool = False) -> list[Pick]:
    """The whitelist of one arm / rerun / failed run dir; `base` is its folder below the output."""
    out: list[Pick] = []
    for p in _files(arm):
        n = p.name
        if failed:
            ok = n in ("RUN_META.json", "CONFIG.json") or n.endswith((".err", ".out"))
        else:
            ok = n.endswith((".json", ".jsonl")) or n in ARM_TOP_NAMES
        if ok:
            _pick(out, f"{base}/{n}", p, "log" if n.endswith((".err", ".out")) else "record")
    for p in _files(arm / "logs"):
        if p.name.endswith((".log", ".jsonl")):
            _pick(out, f"{base}/logs/{p.name}", p, "log")
    if failed:
        return out
    for p in _files(arm / "evaluation"):
        if p.name.endswith((".csv", ".json")) or p.name == "full_metrics_draws.npz":
            _pick(out, f"{base}/evaluation/{p.name}", p)
    for p in _files(arm / "ssl"):
        if re.fullmatch(r"call_\d+\.(params|result)\.json", p.name):
            _pick(out, f"{base}/ssl/{p.name}", p)
    for d in _dirs(arm / "batch"):
        if d.name.startswith("call_"):
            for p in _files(d):
                if p.name in ("params.json", "result.json"):
                    _pick(out, f"{base}/batch/{d.name}/{p.name}", p)
    for stage in _dirs(arm / "agent"):
        for p in _files(stage):
            if p.name in AGENT_STAGE_FILES:
                _pick(out, f"{base}/agent/{stage.name}/{p.name}", p)
        for call in _dirs(stage / "calls"):
            for p in _files(call):
                if p.name in AGENT_CALL_FILES:
                    _pick(out, f"{base}/agent/{stage.name}/calls/{call.name}/{p.name}", p, cap=CALL_FILE_CAP)
    return out


def comparison_picks(d: Path, base: str) -> list[Pick]:
    """A comparison* or posthoc_rerun_* dir: the comparison records and each scored run's RESULT.json / RESULT.md
    (and its full_metrics_draws.npz, the recompute input)."""
    out: list[Pick] = []
    for p in _files(d):
        n = p.name
        if (re.fullmatch(r"COMPARISON.*\.json", n) or re.fullmatch(r"comparison.*\.csv", n)
                or n in ("POSTHOC_COMPARISON.json", "POSTHOC_COMPARISON.md", "RERUN.json") or n.endswith("_COMPLETE")):
            _pick(out, f"{base}/{n}", p)
    for run in _dirs(d / "scores"):
        for p in _files(run):
            if p.name in ("RESULT.json", "RESULT.md"):
                _pick(out, f"{base}/scores/{run.name}/{p.name}", p)
        for p in _files(run / "evaluation"):
            if p.name in ("full_metrics_draws.npz", "full_metrics_draws.npz.json"):
                _pick(out, f"{base}/scores/{run.name}/evaluation/{p.name}", p)
    return out


def classify(run: Path) -> dict:
    """The run dir's top-level entries by role: arms (failed ones apart), comparisons, posthoc, driver logs, other."""
    roles = {"arm": [], "failed": [], "compare": [], "posthoc": [], "logs": [], "other": []}
    with os.scandir(run) as it:
        entries = sorted(it, key=lambda e: e.name)
    for e in entries:
        p = Path(e.path)
        if e.is_symlink():
            roles["other"].append((p, "symlink"))
        elif e.is_dir():
            n = e.name
            if "failed" in n:
                roles["failed"].append(p)
            elif n.startswith("comparison"):
                roles["compare"].append(p)
            elif n.startswith("posthoc"):
                roles["posthoc"].append(p)
            elif any((p / m).is_file() for m in ("RESULT.json", "RUN_META.json") + MARKERS):
                roles["arm"].append(p)
            else:
                roles["other"].append((p, "not an arm, comparison or posthoc dir"))
        elif e.name.endswith(DRIVER_LOG_SUFFIXES):
            roles["logs"].append(p)
        else:
            roles["other"].append((p, "not a driver log"))
    return roles


def _excluded(top: Path, base: str, picked_rels: set[str]) -> list[dict]:
    """The top-level entries of a run dir that no rule selected (files with their size)."""
    out = []
    picked_top = {r[len(base) + 1:].split("/")[0] for r in picked_rels if r.startswith(base + "/")}
    with os.scandir(top) as it:
        entries = sorted(it, key=lambda e: e.name)
    for e in entries:
        if e.name in picked_top:
            continue
        rel = f"{base}/{e.name}"
        if e.is_symlink():
            out.append({"path": rel, "reason": "symlink"})
        elif e.is_dir():
            reason = ("private" if e.name == "private" else
                      "per-image tree (see search_summary.json)" if e.name in ("segment", "crop") else
                      "no whitelisted file")
            out.append({"path": rel + "/", "reason": reason})
        else:
            reason = "per-cell record" if e.name in DENY_NAMES else "not whitelisted"
            out.append({"path": rel, "bytes": e.stat().st_size, "reason": reason})
    return out


# --------------------------------------------------------------------------- #
# generated files: search_summary.json per arm, summary.csv
# --------------------------------------------------------------------------- #
def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dist(values) -> dict:
    return {str(k): v for k, v in sorted(Counter(values).items(), key=lambda kv: (str(type(kv[0])), kv[0]))}


def _stage_search(stage_dir: Path) -> dict | None:
    """The per-image search records of one stage: SEARCH.json, else COHORT_SEARCH.json (segmentation images that
    only ran the cohort mode)."""
    recs, source, unreadable = [], Counter(), 0
    n_dirs = 0
    for d in _dirs(stage_dir):
        n_dirs += 1
        for name in ("SEARCH.json", "COHORT_SEARCH.json"):
            if (d / name).is_file():
                rec = _read_json(d / name)
                if isinstance(rec, dict):
                    recs.append(rec)
                    source[name] += 1
                else:
                    unreadable += 1
                break
        else:
            source["none"] += 1
    if not source.get("SEARCH.json") and not source.get("COHORT_SEARCH.json"):
        return None

    def count(rec, key):
        v = rec.get(key)
        return v if isinstance(v, (int, float)) else 0

    delivered = [str(r.get("delivered")) for r in recs]
    errors = [e for r in recs for e in (r.get("execution_errors") or [])]
    calls = [count(r, "n_calls") for r in recs]
    seconds = [count(r, "seconds") for r in recs]
    levers = ["+".join(sorted(r.get("levers") or {})) or "(none)" for r in recs]
    return {
        "images": n_dirs, "unreadable_records": unreadable, "record_source": dict(sorted(source.items())),
        "resolution_reason": _dist(str((r.get("resolution") or {}).get("reason")) for r in recs),
        "stop_reason": _dist(str(r.get("stop_reason")) for r in recs),
        "calls_per_image": {"distribution": _dist(calls), "total": sum(calls),
                            "mean": round(sum(calls) / len(calls), 4) if calls else None},
        "delivered_call": _dist(delivered),
        "delivered_kind": _dist(re.sub(r"_\d+$", "", d) for d in delivered),
        "delivered_lever_sets": _dist(levers),
        "reasoner_failures": {"total": sum(count(r, "reasoner_failures") for r in recs),
                              "images": sum(1 for r in recs if count(r, "reasoner_failures"))},
        "fallback_rounds": {"total": sum(count(r, "fallback_rounds") for r in recs),
                            "images": sum(1 for r in recs if count(r, "fallback_rounds"))},
        "execution_errors": {"total": len(errors),
                             "images": sum(1 for r in recs if r.get("execution_errors")),
                             "by_question": _dist(str(e.get("question")) if isinstance(e, dict) else "unstructured"
                                                  for e in errors)},
        "seconds": {"total": round(sum(seconds), 1),
                    "mean": round(sum(seconds) / len(seconds), 2) if seconds else None},
    }


def search_summary(arm: Path) -> dict | None:
    """search_summary.json of an arm dir that holds per-image SEARCH.json records (the SHAPE arms), else None."""
    stages = {}
    for stage in ("segment", "crop"):
        found = _stage_search(arm / stage)
        if found:
            stages[stage] = found
    if not stages:
        return None
    if "segment" in stages:                     # segmentation: where each image's parameters came from
        cohort = _read_json(arm / "segment_COHORT.json")
        rows = None
        if isinstance(cohort, dict) and isinstance(cohort.get("parameter_source"), dict):
            rows = list(cohort["parameter_source"].values())
            stages["segment"]["parameter_source_from"] = "segment_COHORT.json"
            stages["segment"]["cohort_counts"] = cohort.get("counts")
        else:
            seg = _read_json(arm / "segmentation.json")
            if isinstance(seg, list):
                rows = [r.get("parameter_source") for r in seg if isinstance(r, dict) and "parameter_source" in r]
                stages["segment"]["parameter_source_from"] = "segmentation.json"
        if rows is not None:
            stages["segment"]["parameter_source"] = _dist(str(v) for v in rows)
    return {"schema": SCHEMA + "/search_summary", "run": arm.name, "stages": stages}


def _get(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def summary_header() -> list[str]:
    cols = ["run", "arm", "status", "commit", "model"]
    for m in VERDICT_METRICS:
        cols += [f"{m}_main", f"{m}_guard"]
    cols += [f"retrieval_{e}_mAP" for e in RETRIEVAL]
    for s in STAGES:
        cols += [f"{s}_seconds", f"{s}_total_tokens", f"{s}_reasoning_tokens"]
    cols += [f"reasoning_{s}" for s in STAGES]
    return cols


def summary_row(arm: Path, failed: bool = False) -> dict:
    result = _read_json(arm / "RESULT.json") or {}
    config = _read_json(arm / "CONFIG.json") or {}
    row = {c: "" for c in summary_header()}
    commit = _get(result, "provenance", "commit") or _get(_read_json(arm / "RUN_META.json"), "provenance", "commit")
    if not commit:
        for marker in MARKERS:
            if (arm / marker).is_file():
                words = (arm / marker).read_text(encoding="utf-8", errors="replace").split()
                commit = words[0] if words else None
                break
    row.update(run=arm.name, arm=result.get("arm") or config.get("arm") or "",
               status="failed" if failed else (result.get("status") or ""), commit=commit or "",
               model=_get(result, "provenance", "model") or config.get("model") or "")
    for m in VERDICT_METRICS:
        for split in ("main", "guard"):
            v = _get(result, "full_metrics", "metrics", m, split, "value")
            row[f"{m}_{split}"] = "" if v is None else v
    for e in RETRIEVAL:
        v = _get(result, "metrics", e, "mAP")
        row[f"retrieval_{e}_mAP"] = "" if v is None else v
    for s in STAGES:
        done = _read_json(arm / f"{s}_COMPLETE.json") or {}
        for col, v in ((f"{s}_seconds", done.get("seconds")),
                       (f"{s}_total_tokens", _get(done, "usage", "total_tokens")),
                       (f"{s}_reasoning_tokens", _get(done, "usage", "reasoning_tokens"))):
            row[col] = "" if v is None else v
        r = _get(config, "reasoning", "stages", s)
        if isinstance(r, dict):
            row[f"reasoning_{s}"] = "/".join(str(r.get(k)) for k in ("effort", "max_tokens", "timeout_seconds"))
    return row


def summary_csv(rows: list[dict]) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=summary_header(), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------- #
# the export
# --------------------------------------------------------------------------- #
def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def scan_secrets(rel: str, data: bytes) -> None:
    if rel.endswith(BINARY_SUFFIXES):
        return
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(data):
            raise ExportError(f"possible secret ({name}) in {rel}; nothing was exported")


def plan(run: Path, extra_logs=()) -> dict:
    """The selection without copying: picks (whitelist), generated files (bytes), excluded entries."""
    roles = classify(run)
    picks: list[Pick] = []
    excluded: list[dict] = []
    generated: dict[str, bytes] = {}
    rows = []
    for arm in roles["arm"] + roles["failed"]:
        failed = arm in roles["failed"]
        found = arm_picks(arm, arm.name, failed)
        picks += found
        excluded += _excluded(arm, arm.name, {p.rel for p in found})
        rows.append(summary_row(arm, failed))
        if not failed:
            summary = search_summary(arm)
            if summary:
                generated[f"{arm.name}/search_summary.json"] = _json_bytes(summary)
    for d in roles["compare"]:
        picks += comparison_picks(d, f"compare/{d.name}")
    for d in roles["posthoc"]:
        picks += comparison_picks(d, f"posthoc/{d.name}")
    for p in roles["logs"]:
        _pick(picks, f"logs/{p.name}", p, "log")
    names = {p.name for p in roles["logs"]}
    for extra in extra_logs:
        extra = Path(extra)
        if not extra.is_file():
            raise ExportError(f"--extra-log {extra}: not a file")
        if extra.name in names:
            raise ExportError(f"--extra-log {extra}: logs/{extra.name} already exists")
        names.add(extra.name)
        _pick(picks, f"logs/{extra.name}", extra, "log")
    for p, reason in roles["other"]:
        excluded.append({"path": p.name, "reason": reason})
    generated["summary.csv"] = summary_csv(rows)
    picks.sort(key=lambda p: p.rel)
    return {"picks": picks, "generated": generated, "excluded": sorted(excluded, key=lambda e: e["path"])}


def export(run, out, cap_mb: float = DEFAULT_CAP_MB, file_cap_mb: float = DEFAULT_FILE_CAP_MB, extra_logs=()) -> dict:
    run, out = Path(run), Path(out)
    if not run.is_dir():
        raise ExportError(f"--run {run}: not a directory")
    if out.exists() and any(out.iterdir()):
        raise ExportError(f"--out {out}: not empty")
    cap, file_cap = int(cap_mb * MIB), int(file_cap_mb * MIB)
    chosen = plan(run, extra_logs)
    files, skipped, total = [], [], 0
    contents: dict[str, bytes] = {}

    def keep(rel, data, kind):
        nonlocal total
        files.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "kind": kind})
        total += len(data)
        contents[rel] = data

    for rel, data in sorted(chosen["generated"].items()):
        scan_secrets(rel, data)
        if len(data) > file_cap or total + len(data) > cap:
            raise ExportError(f"generated file {rel} ({len(data)} bytes) does not fit the caps")
        keep(rel, data, "generated")
    for pick in sorted(chosen["picks"], key=lambda p: (p.size, p.rel)):
        limit = min(file_cap, pick.cap) if pick.cap else file_cap
        if pick.size > limit:
            skipped.append({"path": pick.rel, "bytes": pick.size, "reason": f"over the per-file cap ({limit} bytes)"})
        elif total + pick.size > cap:
            skipped.append({"path": pick.rel, "bytes": pick.size, "reason": f"over the total cap ({cap} bytes)"})
        else:
            data = pick.src.read_bytes()
            scan_secrets(pick.rel, data)
            keep(pick.rel, data, pick.kind)
    files.sort(key=lambda f: f["path"])
    skipped.sort(key=lambda s: s["path"])
    manifest = {"schema": SCHEMA, "source": run.name,
                "caps": {"total_bytes": cap, "per_file_bytes": file_cap, "per_call_file_bytes": CALL_FILE_CAP},
                "totals": {"files": len(files), "bytes": total, "skipped": len(skipped)},
                "files": files, "skipped": skipped, "excluded": chosen["excluded"]}
    for rel, data in contents.items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_bytes(_json_bytes(manifest))
    return manifest


def check(out) -> list[str]:
    """Problems of an exported tree against its MANIFEST.json: missing, modified (size or sha256), unlisted files."""
    out = Path(out)
    manifest = _read_json(out / "MANIFEST.json")
    if not isinstance(manifest, dict) or "files" not in manifest:
        return [f"{out / 'MANIFEST.json'}: missing or not a manifest"]
    problems, listed = [], set()
    for entry in manifest["files"]:
        rel = entry["path"]
        listed.add(rel)
        path = out / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
            continue
        data = path.read_bytes()
        if len(data) != entry["bytes"]:
            problems.append(f"modified (size {len(data)} != {entry['bytes']}): {rel}")
        elif hashlib.sha256(data).hexdigest() != entry["sha256"]:
            problems.append(f"modified (sha256): {rel}")
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out).as_posix()
        if path.is_file() and rel not in listed and rel not in ROOT_UNCHECKED and not rel.startswith(ANALYSIS_DIR + "/"):
            problems.append(f"unlisted: {rel}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Curated text-only export of a full_CPJUMP1 experiment directory.")
    ap.add_argument("--run", help="the experiment directory (arm dirs, comparison dirs, ...)")
    ap.add_argument("--out", help="the (empty or new) output folder, e.g. results/<experiment_id>")
    ap.add_argument("--cap-mb", type=float, default=DEFAULT_CAP_MB, help="total size cap (default 80)")
    ap.add_argument("--file-cap-mb", type=float, default=DEFAULT_FILE_CAP_MB, help="per-file cap (default 8)")
    ap.add_argument("--extra-log", action="append", default=[], metavar="FILE",
                    help="an experiment-level driver log outside the run dir (repeatable)")
    ap.add_argument("--check", metavar="OUT", help="verify an exported tree against its MANIFEST.json")
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.check)
        for line in problems:
            print(line)
        print(f"{'FAIL' if problems else 'OK'}: {args.check} ({len(problems)} problem(s))")
        return 1 if problems else 0
    if not (args.run and args.out):
        ap.error("--run and --out are required (or --check OUT)")
    try:
        manifest = export(args.run, args.out, args.cap_mb, args.file_cap_mb, args.extra_log)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    t = manifest["totals"]
    print(f"exported {t['files']} files, {t['bytes']} bytes to {args.out}; {t['skipped']} skipped (see MANIFEST.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
