"""Continue a failed Codex arm from its completed segment stage (cli `rerun`, `compare-rerun`).

The source is a Codex run that completed its segment stage and then failed (RUN_META status "failed", no later stage
complete, no delivery): here the crop stage hit Codex's input limit. The continuation is a fresh run directory that
reuses the source's segment stage and runs crop, embed, batch and cluster with the current code exactly as `run` does.

Reuse (`reuse_segment`): segment_COMPLETE.json, segment_USAGE.json and every file under agent/segment/ (and segment/,
if present) except the Codex auth directory are placed in the new directory, JSON as copies and the rest as hard links
(a copy where a link is not possible), each checked against its source's hash; segmentation.json is rewritten with its
mask paths in the new directory. Nothing of the source's partial later stages (agent/crop, crop/) is placed, and the
source is only read. RERUN.json records every placed file with its hash, the source's commit and why. The run is
delivered and scored as `run` does (verdicts against the control) and marked RERUN_COMPLETE, never the full-run
marker, so the strict `compare` refuses it; `compare` here is the post-hoc one (control, SHAPE and the continuation,
`strict_comparison` false).
"""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.full_CPJUMP1.harness import cli as base
from benchmarks.full_CPJUMP1.harness import rerun as posthoc
from benchmarks.full_CPJUMP1.harness.compare import MATCH_FIELDS
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, within, write_json

from .spec import LIVECELL

REUSED = ("segment",)
LATER = ("crop", "embed", "batch", "cluster")
SKIPPED_DIRS = ("agent/segment/auth/",)        # the Codex home of the failed run: never reused
REASON = ("source run failed in crop with the prompt-size limit; segment stage reused")
REQUIRED = ("source_run", "source_commit", "reused_stages", "rerun_stages", "reason", "record_sha256",
            "source_segment_complete_sha256")


def check_source(source: Path, model: str, prep: Path, spec=LIVECELL) -> dict:
    """The failed Codex run `source`: its segment stage is complete with its usage and delivered masks (each mask has its
    recorded hash, the delivered calls' artifacts are unchanged), no later stage is complete, it never delivered and is
    marked failed. Returns {path, config, provenance, commit, record}."""
    from benchmarks.full_CPJUMP1.harness.codex_arm import verify_delivery
    root = Path(source)
    try:
        config, record, provenance = [read_json(root / f"{name}.json") for name in ("CONFIG", "RUN_META", "PREFLIGHT")]
    except FileNotFoundError as missing:
        raise ValueError(f"{root} is not a Codex run: {Path(missing.filename).name} is missing") from None
    if config.get("arm") != "codex" or config.get("dataset") != spec.name or config.get("smoke"):
        raise ValueError(f"{root} is not a full {spec.name} Codex run")
    if config.get("model") != model:
        raise ValueError(f"the source run's model {config.get('model')} is not --model {model}")
    if record.get("status") != "failed" or (root / spec.marker).exists() or (root / "DELIVERY.json").exists():
        raise ValueError(f"{root} is not a failed run (status {record.get('status')!r}); a delivered run is not resumed")
    if record.get("prepare_sha256") != digest(Path(prep) / "PREPARE.json"):
        raise ValueError(f"{root} belongs to another preparation")
    later = [s for s in LATER if (root / f"{s}_COMPLETE.json").exists()]
    if later:
        raise ValueError(f"{root}: a later stage is complete ({later}); only a failure after segment is continued")
    try:
        done = read_json(root / "segment_COMPLETE.json")
        usage = read_json(root / "segment_USAGE.json")
    except FileNotFoundError as missing:
        raise ValueError(f"{root}: {Path(missing.filename).name} is missing: the segment stage did not complete") from None
    if done.get("status") != "complete" or done.get("stage") != "segment" or usage != done.get("usage"):
        raise ValueError(f"{root}: the segment stage is not complete with its usage")
    rows = read_json(root / "segmentation.json")
    if sorted(r["image_id"] for r in rows) != sorted(config["images"]):
        raise ValueError(f"{root}: segmentation.json does not cover the run's images")
    for row in rows:
        mask = Path(row["mask"])
        if not mask.is_file() or digest(mask) != row["sha256"]:
            raise ValueError(f"{root}: a delivered segmentation mask is missing or changed: {row['image_id']}")
    stage = root / "agent" / "segment"
    verify_delivery(stage, read_json(stage / "SELECTED.json"), read_json(stage / "CALLS.json"))
    return {"path": root, "config": config, "provenance": provenance, "commit": provenance.get("commit"),
            "record": record, "segment": done}


def reuse_segment(source: dict, out: Path) -> dict:
    """The source's segment stage placed in the new directory `out` (module docstring); writes and returns RERUN.json."""
    root = source["path"]
    files, skipped = {}, []

    def place(rel: str, link: bool) -> None:
        files[rel] = posthoc._place(within(root, rel), within(out, rel), None, link)

    for rel in ("segment_COMPLETE.json", "segment_USAGE.json"):
        place(rel, False)
    for top in ("agent/segment", "segment"):
        if not (root / top).is_dir():
            continue
        for path in sorted((root / top).rglob("*")):
            rel = path.relative_to(root).as_posix()
            if path.is_dir() or rel.startswith(SKIPPED_DIRS):
                continue
            if path.is_symlink() or not path.is_file():
                skipped.append(rel)
                continue
            place(rel, link=not rel.endswith((".json", ".jsonl")))
    rows = read_json(root / "segmentation.json")
    marker = "/agent/segment/"
    first = Path(rows[0]["mask"]).as_posix()
    if marker not in first:
        raise ValueError(f"a delivered mask is not under agent/segment/: {first}")
    recorded = first.split(marker, 1)[0]
    for row in rows:
        text = Path(row["mask"]).as_posix()
        if not text.startswith(recorded + "/"):
            raise ValueError(f"a delivered mask lies outside the source run: {text}")
        rel = text[len(recorded) + 1:]
        if rel not in files or files[rel]["sha256"] != row["sha256"]:
            raise ValueError(f"a delivered mask was not placed with its recorded hash: {rel}")
        row["mask"] = str(within(out, rel))
    write_json(out / "segmentation.json", rows)
    record = {"source_run": str(root), "recorded_source_root": recorded, "rerun_run": str(out),
              "reused_stages": list(REUSED), "reason": REASON, "source_commit": source["commit"],
              "source_status": source["record"].get("status"), "source_error": source["record"].get("error"),
              "source_segment_complete_sha256": files["segment_COMPLETE.json"]["sha256"],
              "placement": "JSON records copied; every other file hard-linked where possible, else copied; never "
                           "symlinked; each checked against its source's hash; the source's auth directory and its "
                           "partial later stages are not placed",
              "files": files, "not_placed_irregular": skipped,
              "rewired": {"segmentation.json": {"fields": ["mask"], "sha256": digest(out / "segmentation.json"),
                                                "from_root": recorded, "to_root": str(out)}}}
    write_json(out / base.RERUN_RECORD, record)
    return record


def rerun(args, spec=LIVECELL) -> None:
    """cli `rerun --arm codex --from-stage crop`: the continuation (module docstring)."""
    import time
    if args.out.exists():
        raise FileExistsError("use a fresh output directory; partial runs are never reused as evidence")
    sha = digest(args.prep / "PREPARE.json")
    source = check_source(args.source, args.model, args.prep, spec)
    old = source["config"]
    cores = old["cores"] if args.cores is None else args.cores
    for key, given in (("seed", args.seed), ("cores", cores), ("train_cells_per_site", args.train_cells_per_site),
                       ("inference_chunk", args.inference_chunk)):
        if old[key] != given:
            raise ValueError(f"--{key.replace('_', '-')} {given} differs from the source run's {old[key]}")
    posthoc.verify_run(args.control, "control", sha, spec)
    from .cli import resources
    resolved = resources(cores)
    spent = float(source["segment"]["seconds"])
    hours = args.budget_hours - spent / 3600
    if hours <= 0:
        raise ValueError("--budget-hours is not larger than the reused segment stage's time")
    budget = {"hours": hours, "budget_hours": args.budget_hours, "reused_stage_seconds": spent,
              "rule": "--budget-hours less the reused stage's seconds"}
    args.cores, args.smoke = resolved["cores"], False
    print(f"RERUN source={args.source} commit={source['commit']} reused={','.join(REUSED)} rerun={','.join(LATER)} "
          f"budget_hours={hours:.3f} (of {args.budget_hours}) cores={args.cores}", flush=True)
    args.out.mkdir(parents=True)
    (args.out / "logs").mkdir()
    record = {"status": "preflight", "arm": args.arm}
    try:
        provenance = base.run_preflight(args, spec)
        reuse = reuse_segment(source, args.out)
        config = {**old, "deadline": time.time() + hours * 3600, "budget_hours": hours, "cores": args.cores,
                  "smoke": False, "assets_root": str(args.assets_root), "resources": resolved,
                  "reasoning": base.reasoning_record(args.arm)}
        config["rerun"] = {
            "source_run": str(args.source), "source_commit": source["commit"],
            "source_branch": source["provenance"].get("branch"), "source_status": "failed",
            "source_config_sha256": digest(args.source / "CONFIG.json"),
            "source_segment_complete_sha256": reuse["source_segment_complete_sha256"],
            "reused_stages": list(REUSED), "rerun_stages": list(LATER), "reason": args.reason,
            "commit": provenance.get("commit"), "branch": provenance.get("branch"), "budget": budget,
            "source_cores": old["cores"], "source_reasoning": old.get("reasoning"),
            "record": base.RERUN_RECORD, "record_sha256": digest(args.out / base.RERUN_RECORD)}
        write_json(args.out / "CONFIG.json", config)
        record.update(status="running", provenance=provenance, config=config, rerun=config["rerun"],
                      prepare_sha256=sha,
                      stages={s: read_json(args.out / f"{s}_COMPLETE.json") for s in REUSED},
                      stage_code={**{s: {"run": "source", "commit": source["commit"]} for s in REUSED},
                                  **{s: {"run": "rerun", "commit": provenance.get("commit")} for s in LATER}})
        base.run_stages(args, spec, config, record, LATER)
        base.deliver(args, spec, record, provenance, base.RERUN_MARKER, control=args.control)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "RUN_META.json", record)
        raise


def verify_resumed(run: Path, source: Path | None, prepare_sha256: str, spec=LIVECELL) -> tuple[dict, dict]:
    """The continuation (a complete delivered run with RERUN_COMPLETE) and its failed source: the reuse record is bound to
    the delivery, every placed file that the delivery binds has its source's hash, and the source's segment records are
    the ones that were reused. Returns (the run as verify_run records it, its RERUN.json record)."""
    found = posthoc.verify_run(run, "codex", prepare_sha256, spec, marker=base.RERUN_MARKER)
    info = found["config"].get("rerun") or {}
    missing = [k for k in REQUIRED if k not in info]
    if missing or found["record"].get("rerun") != info:
        raise ValueError(f"{run}: incomplete rerun record {missing}")
    artifacts = found["delivery"]["artifacts"]
    if artifacts.get(base.RERUN_RECORD) != info["record_sha256"]:
        raise ValueError(f"{run}: the reuse record is not bound to the delivery")
    reuse = read_json(Path(run) / base.RERUN_RECORD)
    for rel, entry in reuse["files"].items():
        if rel in artifacts and artifacts[rel] != entry["sha256"]:
            raise ValueError(f"{run}: reused artifact {rel} differs from its source's")
    source = Path(info["source_run"] if source is None else source)
    if digest(source / "segment_COMPLETE.json") != info["source_segment_complete_sha256"]:
        raise ValueError(f"{source} is not this continuation's source run (its segment record differs)")
    if read_json(source / "PREFLIGHT.json").get("commit") != info["source_commit"]:
        raise ValueError(f"{source}: not the source commit the continuation recorded")
    return found, reuse


def compare(prep: Path, rerun_dir: Path, control: Path, shape: Path, out: Path, source: Path | None = None,
            spec=LIVECELL, agents: dict | None = None) -> dict:
    """The post-hoc comparison of the continuation with the complete control and SHAPE runs (not the strict compare:
    the runs differ in commit and the continuation in where its segment stage ran). Each run is scored as the strict
    compare scores it (the control first, the others with its verdicts); writes POSTHOC_COMPARISON.json. `agents`:
    {"claude": dir, "biomni": dir}, complete full runs of the follow-up agent arms, compared as well."""
    from benchmarks._shared import full_metrics as fm
    from .score import score
    if out.exists():
        raise FileExistsError("comparison output must be fresh")
    sha = digest(Path(prep) / "PREPARE.json")
    found, reuse = verify_resumed(rerun_dir, source, sha, spec)
    runs = {"control": posthoc.verify_run(control, "control", sha, spec),
            "shape": posthoc.verify_run(shape, "shape", sha, spec), "codex_resumed": found}
    runs.update({arm: posthoc.verify_run(path, arm, sha, spec) for arm, path in (agents or {}).items()})
    differences = {}
    for key in [k for k in MATCH_FIELDS] + ["model"]:
        values = {label: r["config"].get(key) for label, r in runs.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            differences[key] = values
    results = {"control": score(prep, runs["control"]["path"])}
    for label in [k for k in runs if k != "control"]:
        results[label] = score(prep, runs[label]["path"], control=runs["control"]["path"])
    info = found["config"]["rerun"]
    report = {}
    for label, record in runs.items():
        metrics = results[label]["full_metrics"]["metrics"]
        stages = posthoc.stage_table(record)
        report[label] = {"arm": record["config"]["arm"], "path": str(record["path"]), "commit": record["commit"],
                         "delivery_sha256": record["delivery_sha256"],
                         "metrics": {m: {"main": (metrics[m].get("main") or {}).get("value"),
                                         "vs_control": posthoc._verdict(metrics[m]["vs_control"])}
                                     for m in fm.METRICS},
                         "stages": stages, "total_tokens": sum(s["total_tokens"] or 0 for s in stages.values()),
                         "seconds": sum(s["seconds"] or 0 for s in stages.values())}
    summary = {"status": "complete", "kind": "posthoc_rerun_comparison", "strict_comparison": False,
               "prepare_sha256": sha, "primary": list(fm.METRICS),
               "reused_stages": info["reused_stages"], "rerun_stages": info["rerun_stages"],
               "source": {"run": info["source_run"], "commit": info["source_commit"], "status": info["source_status"]},
               "commits": {label: record["commit"] for label, record in runs.items()},
               "continuation_budget": info["budget"], "reason": info["reason"],
               "verdicts": {label: {m: (report[label]["metrics"][m]["vs_control"] or {}).get("verdict")
                                    for m in fm.METRICS} for label in runs if label != "control"},
               "config_differences": differences, "runs": report,
               "notes": ["POST-HOC, NOT THE STRICT COMPARISON: codex_resumed reused the segment stage of a Codex run "
                         f"at commit {info['source_commit']} (it failed in crop with the prompt-size limit) and ran "
                         f"{', '.join(info['rerun_stages'])} at commit {found['commit']}; `compare` refuses it.",
                         "The segment stage's seconds and tokens are the source's; the continuation's budget is the "
                         "full budget less those seconds (continuation_budget).",
                         "Single seed; verdict SEs are a bootstrap over test units."]}
    out.mkdir(parents=True)
    write_json(out / "POSTHOC_COMPARISON.json", summary)
    print(f"POSTHOC_COMPARISON out={out}", flush=True)
    return summary
