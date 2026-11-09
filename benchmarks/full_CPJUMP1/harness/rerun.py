"""Post-hoc partial reruns of a completed full_CPJUMP1 arm (cli.rerun) and their comparison (compare_rerun).

A rerun reuses its source run's stages before --from-stage and reruns the rest with the current code. The reused
outputs come into the rerun's own directory: the JSON records as copies, the masks and crops as hard links (the same
bytes without a second copy on one filesystem; a copy where a link is not possible; never a symlink, which the
harness's `within` refuses), each checked against its hash (the source's DELIVERY.json, or cells.json's per-crop
sha256). The absolute paths the later stages read (segmentation.json `mask`, cells.json `crop_path` and
`display_crop_path`) are rewritten to those copies, and RERUN.json (cli.RERUN_RECORD) records every reused file with
its hash and how it was placed, and each rewritten file with its source's hash and its own. The rerun's directory is
then self-contained: its delivery binds every artifact inside it, scoring reads nothing of the source, and the source
can be moved or archived without breaking it. The source is only read.

The post-hoc comparison is not the strict compare (compare.py), which refuses a rerun: the compared runs differ in
code and in reasoning settings (the rerun's later stages ran the current commit under the current reasoning table,
every other arm its own commit with its own settings). It verifies every run's delivery, rescores each run into its
own directory with the current scorer (never into the runs) against the same control reading (the shared
full_metrics verdict on saved draws, as score.py), and reports the rerun next to its source SHAPE arm.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil

from .cli import AGENT_ARMS, RERUN_MARKER, RERUN_RECORD, STAGES
from .compare import MATCH_FIELDS
from .data import digest, read_json, within, write_json
from .pipeline import CPJUMP1, assert_search_clean, stage_reasoning

# The delivered files a reused stage owns besides its <stage>/ tree and its <stage>_* records.
OWNED = {"segment": ("segmentation.json",), "crop": ("cells.json", "excluded_cells.json")}
# The reused files whose absolute paths the later stages read: rewritten to the rerun's copies.
REWIRED = {"segmentation.json": ("mask",), "cells.json": ("crop_path", "display_crop_path")}
WARNINGS = "SEARCH_WARNINGS.jsonl"
RECORDS = ("DELIVERY", "RUN_META", "CONFIG", "PREFLIGHT")


def _search_record(rel: str) -> bool:
    """A SHAPE arm's search records (as compare.validate_run reads them)."""
    parts = rel.split("/")
    return rel in ("representation_SEARCH.json", "batch_SEARCH.json") or (
        len(parts) == 3 and parts[0] in ("segment", "crop") and parts[2] in ("SEARCH.json", "COHORT_SEARCH.json"))


def verify_run(run: Path, arm: str, prepare_sha256: str, spec=CPJUMP1, marker: str | None = None) -> dict:
    """A complete, delivered full-size run of `arm` on the preparation: its records agree, its marker names its
    commit, every stage completed with its usage, every delivered artifact has its hash, and a SHAPE arm's search
    records are clean. marker: RERUN_MARKER for a rerun; default the dataset's full-run marker, and a rerun is then
    refused. Returns {path, config, provenance, delivery, record, commit, config_sha256, delivery_sha256}."""
    run = Path(run)
    marker = marker or spec.marker
    try:
        delivery, record, config, provenance = [read_json(run / f"{name}.json") for name in RECORDS]
    except FileNotFoundError as missing:
        raise ValueError(f"{run} is not a complete run: {Path(missing.filename).name} is missing") from None
    rerun = (run / RERUN_MARKER).exists() or "rerun" in config
    if rerun != (marker == RERUN_MARKER):
        raise ValueError(f"{run} is a post-hoc rerun, not a complete full run" if rerun
                         else f"{run} is not a post-hoc rerun")
    if any(r.get("arm") != arm for r in (delivery, record, config)):
        raise ValueError(f"{run} is not a {arm} run")
    if delivery.get("smoke") or config.get("smoke") or (run / "SMOKE_COMPLETE").exists():
        raise ValueError(f"{run} is a smoke, not a full run")
    if any(r.get("status") != "complete" for r in (delivery, record)) or not (run / marker).is_file():
        raise ValueError(f"{run} is not complete: no {marker}")
    commit = provenance.get("commit")
    if not isinstance(commit, str) or (run / marker).read_text(encoding="ascii").strip() != commit:
        raise ValueError(f"{run}: completion marker commit mismatch")
    if (delivery.get("provenance") != provenance or record.get("provenance") != provenance
            or record.get("config") != config):
        raise ValueError(f"{run}: conflicting provenance/config records")
    if delivery.get("prepare_sha256") != prepare_sha256:
        raise ValueError(f"{run} belongs to another preparation")
    if len(config["images"]) != spec.n_images or len(set(config["images"])) != spec.n_images:
        raise ValueError(f"{run} is not a full run of {spec.n_images} views")
    artifacts = delivery["artifacts"]
    for stage in STAGES:
        done = read_json(run / f"{stage}_COMPLETE.json")
        if (done.get("status") != "complete" or done.get("stage") != stage
                or read_json(run / f"{stage}_USAGE.json") != done.get("usage")):
            raise ValueError(f"{run}: stage {stage} is not complete with its usage")
        if f"{stage}_COMPLETE.json" not in artifacts or f"{stage}_USAGE.json" not in artifacts:
            raise ValueError(f"{run}: stage {stage}'s records are not bound to the delivery")
    for rel, sha in artifacts.items():
        if digest(within(run, rel)) != sha:
            raise ValueError(f"{run}: delivered artifact changed: {rel}")
    if arm == "shape":
        for rel in artifacts:
            if _search_record(rel):
                assert_search_clean(read_json(within(run, rel)))
    return {"path": run, "config": config, "provenance": provenance, "delivery": delivery, "record": record,
            "commit": commit, "config_sha256": digest(run / "CONFIG.json"),
            "delivery_sha256": digest(run / "DELIVERY.json")}


def check_source(source: Path, arm: str, model: str, prep: Path, spec=CPJUMP1) -> dict:
    """A rerun's source: a complete full run of `arm` (verify_run) on this preparation, at --model."""
    found = verify_run(source, arm, digest(Path(prep) / "PREPARE.json"), spec)
    if found["config"].get("model") != model:
        raise ValueError(f"the source run's model {found['config'].get('model')} is not --model {model}")
    return found


def budget(source: dict, reused, hours: float | None) -> dict:
    """The rerun's budget in hours: --budget-hours, else the source's budget_hours less the seconds its reused
    stages took (the time the source arm had left for the rerun stages, its preflight aside)."""
    spent = sum(float(read_json(source["path"] / f"{stage}_COMPLETE.json")["seconds"]) for stage in reused)
    rule = "--budget-hours"
    if hours is None:
        hours = source["config"]["budget_hours"] - spent / 3600
        rule = "source budget_hours less the reused stages' seconds"
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("budget-hours must be positive and finite")
    return {"hours": float(hours), "rule": rule, "source_budget_hours": source["config"]["budget_hours"],
            "reused_stage_seconds": spent}


def resources(cores: int | None, source_cores: int) -> dict:
    """The rerun's cores: --cores, else the source run's; never more than this process may use."""
    from shape.agent.verification.base import available_cores
    available = available_cores()
    resolved = int(source_cores if cores is None else cores)
    if not 1 <= resolved <= available:
        raise ValueError(f"{resolved} cores is outside the {available} cores this process may use; pass --cores")
    return {"cores": resolved, "available_cores": available, "source": "source_run" if cores is None else "flag"}


def owner(rel: str, stages) -> str | None:
    """The stage of `stages` whose output a delivered artifact is (None: none of them)."""
    for stage in stages:
        if rel in OWNED[stage] or rel.startswith((f"{stage}/", f"{stage}_")):
            return stage
    return None


def _place(src: Path, dst: Path, sha256: str | None, link: bool) -> dict:
    """`src` at `dst` (a hard link when `link` and possible, else a copy), checked against `sha256` (None: src's).
    An existing `dst` is never written (a copy onto a hard link would write into the source): it must hold the
    expected bytes already."""
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"a reused artifact is not a regular file: {src}")
    expected = sha256 or digest(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or digest(dst) != expected:
            raise ValueError(f"a reused artifact is already placed with other bytes: {dst}")
        return {"sha256": expected, "method": "hardlink" if os.path.samefile(src, dst) else "copy"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    method = "copy"
    if link:
        try:
            os.link(src, dst)
            method = "hardlink"
        except OSError:
            pass
    if method == "copy":
        shutil.copyfile(src, dst)
    if dst.is_symlink() or digest(dst) != expected:
        raise ValueError(f"a reused artifact does not match its recorded hash: {src}")
    return {"sha256": expected, "method": method}


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).relative_to(root).as_posix()
    except ValueError:
        raise ValueError(f"a reused path lies outside the source run: {path}") from None


def reuse_upstream(source: dict, out: Path, reused) -> dict:
    """The source's outputs of the `reused` stages placed in the rerun directory `out` (module docstring): every
    delivered artifact those stages own, the crops cells.json names, the search warnings of those stages; the paths
    in REWIRED rewritten to the copies. Writes and returns RERUN_RECORD."""
    root, artifacts = source["path"], source["delivery"]["artifacts"]
    chosen = {rel: sha for rel, sha in artifacts.items() if owner(rel, reused)}
    needed = {name for stage in reused for name in (*OWNED[stage], f"{stage}_COMPLETE.json", f"{stage}_USAGE.json")}
    if needed - set(chosen):
        raise ValueError(f"the source delivery lacks {sorted(needed - set(chosen))}")
    files = {}
    for rel, sha in sorted(chosen.items()):
        if rel in REWIRED:
            if digest(within(root, rel)) != sha:
                raise ValueError(f"delivered artifact changed: {rel}")
        else:
            files[rel] = _place(within(root, rel), within(out, rel), sha, link=not rel.endswith((".json", ".jsonl")))
    # The source wrote absolute paths under its own directory at run time (it may have moved since): that directory
    # is the one above segment/<image>/ of a delivered mask.
    rows = read_json(root / "segmentation.json")
    recorded = Path(rows[0]["mask"]).parents[2]
    for row in rows:
        rel = _relative(row["mask"], recorded)
        if chosen.get(rel) != row["sha256"]:
            raise ValueError(f"a segmentation mask is not a delivered artifact of the source: {rel}")
        row["mask"] = str(within(out, rel))
    write_json(out / "segmentation.json", rows)
    placed = {field: {"n": 0, "hardlink": 0, "copy": 0} for field in REWIRED["cells.json"]}
    cells = read_json(root / "cells.json")
    for cell in cells:
        for field, sha in (("crop_path", cell["sha256"]), ("display_crop_path", None)):
            if not cell.get(field):
                continue
            rel = _relative(cell[field], recorded)
            method = _place(within(root, rel), within(out, rel), sha, link=True)["method"]
            placed[field]["n"] += 1
            placed[field][method] += 1
            cell[field] = str(within(out, rel))
    write_json(out / "cells.json", cells)
    warnings = None
    if WARNINGS in artifacts:
        if digest(root / WARNINGS) != artifacts[WARNINGS]:
            raise ValueError(f"delivered artifact changed: {WARNINGS}")
        lines = [line for line in (root / WARNINGS).read_text(encoding="utf-8").splitlines() if line.strip()]
        kept = [line for line in lines if json.loads(line).get("stage") in reused]
        if kept:
            (out / WARNINGS).write_text("".join(line + "\n" for line in kept), encoding="utf-8", newline="\n")
        warnings = {"source_sha256": artifacts[WARNINGS], "kept_stages": list(reused), "kept": len(kept),
                    "dropped": len(lines) - len(kept), "sha256": digest(out / WARNINGS) if kept else None}
    record = {"source_run": str(root), "recorded_source_root": str(recorded), "rerun_run": str(out),
              "reused_stages": list(reused),
              "placement": "JSON records copied; masks and crops hard-linked where possible, else copied; never "
                           "symlinked; each checked against its hash (the source delivery's, or cells.json's sha256)",
              "files": files,
              "rewired": {name: {"fields": list(fields), "source_sha256": artifacts[name], "sha256": digest(out / name),
                                 "from_root": str(recorded), "to_root": str(out)}
                          for name, fields in REWIRED.items() if name in chosen},
              "crops": placed, "search_warnings": warnings}
    write_json(out / RERUN_RECORD, record)
    return record


def _unwired(rows: list, fields, root: str) -> list:
    """The records with each path field relative to `root`: a rewritten file compared with its source's."""
    out = []
    for row in rows:
        row = dict(row)
        for field in fields:
            if row.get(field):
                row[field] = _relative(row[field], Path(root))
        out.append(row)
    return out


def verify_rerun(run: Path, prepare_sha256: str, spec=CPJUMP1, source: Path | None = None) -> tuple[dict, dict]:
    """A complete rerun (verify_run with RERUN_MARKER) and its source (`source`, default the path its CONFIG.json
    records; the one whose DELIVERY.json and CONFIG.json hashes it recorded, itself complete): every reused file is
    bound to the rerun's delivery with the source's hash, every rewritten file equals its source's but for its
    rewired paths, and each rerun stage recorded the settings of the rerun's reasoning protocol. Returns (rerun,
    source) as verify_run records them."""
    found = verify_run(run, "shape", prepare_sha256, spec, marker=RERUN_MARKER)
    info = found["config"].get("rerun") or {}
    missing = [k for k in ("source_run", "source_commit", "source_config_sha256", "source_delivery_sha256",
                           "reused_stages", "rerun_stages", "reason", "record_sha256") if k not in info]
    if missing or found["record"].get("rerun") != info:
        raise ValueError(f"{run}: incomplete rerun record {missing}")
    artifacts = found["delivery"]["artifacts"]
    if artifacts.get(RERUN_RECORD) != info["record_sha256"]:
        raise ValueError(f"{run}: the reuse record is not bound to the delivery")
    reuse = read_json(Path(run) / RERUN_RECORD)
    source = Path(info["source_run"] if source is None else source)
    if (not (source / "DELIVERY.json").is_file() or digest(source / "DELIVERY.json") != info["source_delivery_sha256"]
            or digest(source / "CONFIG.json") != info["source_config_sha256"]):
        raise ValueError(f"{source} is not this rerun's source run (its DELIVERY.json or CONFIG.json differs)")
    origin = verify_run(source, "shape", prepare_sha256, spec)
    if origin["commit"] != info["source_commit"]:
        raise ValueError(f"{source}: not the source commit the rerun recorded")
    for rel, entry in reuse["files"].items():
        if artifacts.get(rel) != entry["sha256"] or origin["delivery"]["artifacts"].get(rel) != entry["sha256"]:
            raise ValueError(f"{run}: reused artifact {rel} is not the source's delivered one")
    for name, entry in reuse["rewired"].items():
        if (artifacts.get(name) != entry["sha256"]
                or origin["delivery"]["artifacts"].get(name) != entry["source_sha256"]):
            raise ValueError(f"{run}: rewritten {name} is not bound to both deliveries")
        if (_unwired(read_json(source / name), entry["fields"], entry["from_root"])
                != _unwired(read_json(Path(run) / name), entry["fields"], entry["to_root"])):
            raise ValueError(f"{run}: {name} differs from the source's beyond its rewired paths")
    reasoning = found["config"]["reasoning"]
    for stage in info["rerun_stages"]:
        expected = stage_reasoning("shape", stage, reasoning["protocol"])
        usage = read_json(Path(run) / f"{stage}_USAGE.json")
        if reasoning["stages"].get(stage) != expected or (expected is not None and usage.get("reasoning") != expected):
            raise ValueError(f"{run}: rerun stage {stage} did not run with its protocol's reasoning settings")
    return found, origin


def named_runs(items) -> dict:
    """compare-rerun's --arm NAME=DIR values as {name: directory}; each name an agent arm, once."""
    runs = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not path or name not in AGENT_ARMS or name in runs:
            raise ValueError(f"--arm takes NAME=DIR, once per agent arm ({', '.join(AGENT_ARMS)}): {item!r}")
        runs[name] = Path(path).resolve()
    return runs


def stage_table(found: dict) -> dict:
    """Per stage: where it ran (this run, or a rerun's source), its commit and reasoning settings (as recorded in its
    usage, else in the run's CONFIG.json; None: none recorded), seconds, total and reasoning tokens, calls."""
    info = found["config"].get("rerun") or {}
    table = {}
    for stage in STAGES:
        done = read_json(found["path"] / f"{stage}_COMPLETE.json")
        usage = done.get("usage") or {}
        reused = stage in info.get("reused_stages", ())
        settings = ((info.get("source_reasoning") if reused else found["config"].get("reasoning")) or {}).get("stages")
        table[stage] = {"ran": "source run" if reused else "this run",
                        "commit": info["source_commit"] if reused else found["commit"],
                        "reasoning": usage.get("reasoning") or (settings or {}).get(stage),
                        "seconds": done.get("seconds"), "total_tokens": usage.get("total_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"), "calls": usage.get("calls")}
    return table


def _verdict(entry: dict | None) -> dict | None:
    return None if entry is None else {k: entry.get(k) for k in ("delta", "se", "ci95", "guard_delta", "verdict")}


def compare_rerun(prep: Path, rerun: Path, control: Path, arms: dict, out: Path, source: Path | None = None,
                  spec=CPJUMP1) -> dict:
    """The post-hoc comparison (module docstring): the rerun (shape_rerun), its source SHAPE arm (shape_source), the
    control and the given agent arms, each verified and rescored into out/scores/<run>/ against the control's reading
    there; writes POSTHOC_COMPARISON.json and .md."""
    from benchmarks._shared import full_metrics as fm
    from .score import DRAWS, score
    if out.exists():
        raise FileExistsError("comparison output must be fresh")
    prep = Path(prep)
    sha = digest(prep / "PREPARE.json")
    found, origin = verify_rerun(rerun, sha, spec, source)
    runs = {"control": verify_run(control, "control", sha, spec), "shape_source": origin, "shape_rerun": found}
    runs.update({name: verify_run(path, name, sha, spec) for name, path in arms.items()})
    for label, record in runs.items():
        if record["config"]["images"] != runs["control"]["config"]["images"]:
            raise ValueError(f"{label} ran other views than the control")
    differences = {}
    for key in [k for k in MATCH_FIELDS if k != "images"] + ["model"]:
        values = {label: r["config"].get(key) for label, r in runs.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            differences[key] = values
    for key in ("gpu", "assets"):
        values = {label: r["provenance"].get(key) for label, r in runs.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            differences[key] = values
    out.mkdir(parents=True)
    scores = out / "scores"
    # The control first: every other run's verdicts read its reading, saved under scores/control.
    results = {"control": score(prep, runs["control"]["path"], out=scores / "control")}
    for label, record in runs.items():
        if label != "control":
            results[label] = score(prep, record["path"], control=scores / "control", out=scores / label)
    readings = {label: fm.load(scores / label / "evaluation" / DRAWS) for label in ("shape_rerun", "shape_source")}
    rerun_vs_source = {m: _verdict(fm.verdict(readings["shape_rerun"], readings["shape_source"], m))
                       for m in fm.METRICS}
    report = {}
    for label, record in runs.items():
        full, stages = results[label]["full_metrics"], stage_table(record)
        reported = [s["reasoning_tokens"] for s in stages.values() if s["reasoning_tokens"] is not None]
        report[label] = {
            "arm": record["config"]["arm"], "path": str(record["path"]), "commit": record["commit"],
            "branch": record["provenance"].get("branch"), "model": record["config"].get("model"),
            "delivery_sha256": record["delivery_sha256"], "result_sha256": digest(scores / label / "RESULT.json"),
            "rerun": record["config"].get("rerun"), "ssl_expert_config": record["config"].get("ssl_expert_config"),
            "metrics": {m: {"main": (full["metrics"][m].get("main") or {}).get("value"),
                            "guard": (full["metrics"][m].get("guard") or {}).get("value"),
                            "vs_control": _verdict(full["metrics"][m]["vs_control"])} for m in fm.METRICS},
            "retrieval": results[label]["metrics"], "stages": stages,
            "total_tokens": sum(s["total_tokens"] or 0 for s in stages.values()),
            "reasoning_tokens": sum(reported) if reported else None,
            "seconds": sum(s["seconds"] or 0 for s in stages.values())}
    info = found["config"]["rerun"]
    notes = [
        "POST-HOC, NOT THE STRICT COMPARISON: the runs differ in code and in reasoning settings. shape_rerun ran "
        f"{', '.join(info['rerun_stages'])} at commit {found['commit']} under that commit's per-stage reasoning table "
        f"(CONFIG.json reasoning), on the {', '.join(info['reused_stages'])} outputs of shape_source (commit "
        f"{origin['commit']}); every other run ran all its stages at its own commit with the settings it recorded "
        "(per stage under runs.<run>.stages; a commit that recorded none ran the engine's defaults, medium effort). "
        "`compare` refuses this set.",
        "Every verdict reads the same control reading (scores/control/evaluation/" + DRAWS + "): delta = run minus "
        "control on the main split, SE = the s.d. of the paired bootstrap delta on the same draws, guard delta on the "
        "guard split (benchmarks/_shared/full_metrics.py verdict). rerun_vs_source is the same verdict with "
        "shape_source as the reference (delta = rerun minus source).",
        "Every run is rescored here with the current scorer into scores/<run>/; the runs' own directories are not "
        "written.",
        "Single seed; the SE is a bootstrap over test wells, not over seeds.",
        "Token totals are reported usage, not cost estimates; reasoning tokens appear only where the provider "
        "reported them.",
        "config_differences lists the run settings that differ between the runs (e.g. cores, budget_hours, GPU)."]
    summary = {"status": "complete", "kind": "posthoc_rerun_comparison", "strict_comparison": False,
               "manifest_sha256": results["control"]["manifest_sha256"], "prepare_sha256": sha,
               "primary": list(fm.METRICS), "verdict_rule": fm.VERDICT_RULE,
               "verdicts": {label: {m: (report[label]["metrics"][m]["vs_control"] or {}).get("verdict")
                                    for m in fm.METRICS} for label in runs if label != "control"},
               "rerun_vs_source": rerun_vs_source, "config_differences": differences,
               "commits": {label: record["commit"] for label, record in runs.items()},
               "runs": report, "notes": notes}
    write_json(out / "POSTHOC_COMPARISON.json", summary)
    (out / "POSTHOC_COMPARISON.md").write_text(markdown(summary), encoding="utf-8", newline="\n")
    print(f"POSTHOC_COMPARISON out={out}", flush=True)
    return summary


def markdown(summary: dict) -> str:
    from benchmarks._shared.full_metrics import _f
    lines = ["# full_CPJUMP1 post-hoc rerun comparison (not the strict comparison)", "", summary["notes"][0], "",
             "## Verdict metrics vs the control (the same control draws)", "",
             "| run | commit | metric | main | delta | SE | guard delta | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, run in summary["runs"].items():
        for metric, entry in run["metrics"].items():
            vs = entry["vs_control"] or {}
            lines.append(f"| {label} | {run['commit'][:8]} | {metric} | {_f(entry['main'])} | "
                         f"{_f(vs.get('delta'), '+.4f')} | {_f(vs.get('se'), '.4f')} | "
                         f"{_f(vs.get('guard_delta'), '+.4f')} | {vs.get('verdict') or 'control'} |")
    lines += ["", "## shape_rerun vs shape_source (delta = rerun minus source, the same draws)", "",
              "| metric | delta | SE | guard delta | verdict |", "| --- | --- | --- | --- | --- |"]
    for metric, vs in summary["rerun_vs_source"].items():
        lines.append(f"| {metric} | {_f(vs['delta'], '+.4f')} | {_f(vs['se'], '.4f')} | "
                     f"{_f(vs['guard_delta'], '+.4f')} | {vs['verdict']} |")
    lines += ["", "## Retrieval endpoints (secondary)", "",
              "| run | endpoint | mAP | MRR | top1 | genes | queries |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, run in summary["runs"].items():
        for endpoint, m in run["retrieval"].items():
            lines.append(f"| {label} | {endpoint} | {_f(m['mAP'])} | {_f(m['MRR'])} | {_f(m['top1'])} | "
                         f"{m['n_genes']} | {m['n_queries']} |")
    lines += ["", "## Stages: where they ran, reasoning, seconds and tokens", "",
              "| run | stage | ran | commit | reasoning | seconds | total tokens | reasoning tokens |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, run in summary["runs"].items():
        for stage, s in run["stages"].items():
            reasoning = "n/a" if not s["reasoning"] else ", ".join(f"{k} {v}" for k, v in s["reasoning"].items())
            lines.append(f"| {label} | {stage} | {s['ran']} | {s['commit'][:8]} | {reasoning} | "
                         f"{_f(s['seconds'], '.0f')} | {s['total_tokens']} | {s['reasoning_tokens']} |")
        lines.append(f"| {label} | total | | | | {_f(run['seconds'], '.0f')} | {run['total_tokens']} | "
                     f"{run['reasoning_tokens']} |")
    lines += ["", "## Config differences", ""]
    lines += [f"- {key}: " + "; ".join(f"{label} {json.dumps(v)}" for label, v in values.items())
              for key, values in summary["config_differences"].items()] or ["- none"]
    lines += ["", "## Notes", ""] + [f"- {note}" for note in summary["notes"][1:]]
    return "\n".join(lines) + "\n"
