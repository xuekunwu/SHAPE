"""Fail-closed comparison of control, SHAPE and the agent arms, rescored from frozen delivered profiles."""
import json
from pathlib import Path
import subprocess

import pandas as pd

from .data import digest, read_json, write_json
from .cli import AGENT_ARMS, ARM_MODELS, DEFAULT_BRANCH, REPO

ARMS = ("control", "shape", "codex")                     # required
OPTIONAL_ARMS = ("claude", "biomni")
MATCH_FIELDS = ("seed", "cores", "budget_hours", "train_cells_per_site", "inference_chunk", "images")
# The code every arm's result depends on beyond its own decisions: a run at another commit than --expected-commit is
# comparable only if each of these paths has the same git object id at both commits.
# From benchmarks/_shared only full_metrics.py: it is the one file the full_* harnesses (scorers, compare, rerun) import;
# the rest of that folder (the evaluation framework, the post-hoc process readouts and their tests, the ReAct runner of
# the other benchmarks) does not change what a run delivers or how it is scored.
SHARED = ("shape", "benchmarks/_shared/full_metrics.py", "benchmarks/full_CPJUMP1/harness/pipeline.py",
          "benchmarks/full_CPJUMP1/harness/score.py", "benchmarks/full_CPJUMP1/harness/data.py",
          "benchmarks/full_CPJUMP1/harness/merged.py", "benchmarks/full_HCMA/harness",
          "benchmarks/full_HCMA/workspace_template", "benchmarks/representation_RxRx1/runners/epoch_budget.py",
          "benchmarks/full_CPJUMP1/harness/claude_arm.py")     # the native workers' registry, inputs and outputs
# the agent-arm protocol and the run driver: their diffs are reported, not refused (QC 2026-10-12)
PROTOCOL = ("benchmarks/full_CPJUMP1/harness/codex_arm.py", "benchmarks/full_CPJUMP1/harness/cli.py",
            "shape/agent/arms/protocol.py", "shape/agent/arms/codex.py", "shape/agent/arms/claude.py",
            "shape/agent/arms/biomni.py", "shape/agent/arms/biomni_driver.py", "shape/agent/arms/read_guard.py")


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def shared_ids(commit: str) -> dict:
    """The git object id of every SHARED path at `commit` (None: absent)."""
    if _git("cat-file", "-e", f"{commit}^{{commit}}").returncode:
        raise ValueError(f"commit {commit} is not in {REPO}; fetch it before the comparison")
    ids = {}
    for path in SHARED:
        found = _git("rev-parse", "--verify", "--quiet", f"{commit}:{path}")
        ids[path] = found.stdout.strip() if found.returncode == 0 else None
    return ids


def code_alignment(commit: str, expected: str) -> dict:
    """A run's code record: its commit, and when it differs from `expected` the SHARED ids (all equal, else refused,
    naming the paths) and the `git diff --stat` of the protocol module between the two commits."""
    if commit == expected:
        return {"commit": commit, "aligned_by": "same commit"}
    theirs, ours = shared_ids(commit), shared_ids(expected)
    differ = [path for path in SHARED if theirs[path] is None or theirs[path] != ours[path]]
    if differ:
        raise ValueError(f"run commit {commit} differs from {expected} in shared code: {differ}")
    stat = _git("diff", "--stat", expected, commit, "--", *PROTOCOL)
    if stat.returncode:
        raise ValueError(f"cannot diff {PROTOCOL} between {expected} and {commit}")
    return {"commit": commit, "aligned_by": "shared git object ids", "expected_commit": expected, "shared": theirs,
            "protocol_diff_stat": stat.stdout.strip()}


def segment_routed(run, images, artifacts):
    """The images whose segmentation ran its own VRR (segment/<id>/SEARCH.json). Per-image layout (no
    segment_COHORT.json): every image. Cohort layout (pipeline._segmentation_cohort, SHAPE's shared
    `segmentation_cohort.run`): segment_COHORT.json, bound to the delivery, gives every image's parameter_source
    (segmentation.json's must equal it); the images of source `cohort` and the routed ones ran the cohort mode
    (segment/<id>/COHORT_SEARCH.json, exactly those), every image not of source `cohort` its own VRR."""
    path = run / "segment_COHORT.json"
    if not path.is_file():
        return set(images)
    if "segment_COHORT.json" not in artifacts:
        raise ValueError("segmentation cohort record is not bound to the delivery")
    cohort = read_json(path)
    source = cohort.get("parameter_source") or {}
    if set(source) != set(images) or not set(source.values()) <= {"calibration", "cohort", "per_image_vrr"}:
        raise ValueError("segmentation cohort record does not cover the images")
    delivered = {r["image_id"]: r.get("parameter_source") for r in read_json(run / "segmentation.json")}
    if delivered != source:
        raise ValueError("segmentation parameter sources do not match the cohort record")
    routed = {i for group in cohort.get("groups") or [] for i in group.get("routed") or []}
    if any(source[i] != "per_image_vrr" for i in routed):
        raise ValueError("a routed image did not end with its own VRR")
    cohort_mode = {i for i, s in source.items() if s == "cohort"} | routed
    if {p.parent.name for p in (run / "segment").glob("*/COHORT_SEARCH.json")} != cohort_mode:
        raise ValueError("segmentation cohort-mode records do not match the cohort record")
    return {i for i, s in source.items() if s != "cohort"}


def validate_run(run, arm, expected, branch=DEFAULT_BRANCH):
    from .cli import ARTIFACTS, RERUN_MARKER, STAGES
    from .pipeline import assert_search_clean
    from .data import within
    delivery = read_json(run / "DELIVERY.json")
    record = read_json(run / "RUN_META.json")
    config = read_json(run / "CONFIG.json")
    provenance = read_json(run / "PREFLIGHT.json")
    if (run / RERUN_MARKER).exists() or "rerun" in config:
        raise ValueError("a post-hoc rerun is not a clean full run; the strict comparison refuses it (compare-rerun)")
    if any(r.get("arm") != arm for r in (delivery, record, config)):
        raise ValueError("arm identity mismatch")
    if delivery.get("smoke") or config.get("smoke") or any(r.get("status") != "complete" for r in (delivery, record)):
        raise ValueError("all arms must be complete full runs, not smokes")
    commit = provenance.get("commit")
    if (run / "CPJUMP1_COMPLETE").read_text(encoding="ascii").strip() != commit:
        raise ValueError("completion marker commit mismatch")
    if not isinstance(commit, str) or provenance.get("branch") != branch:
        raise ValueError("run provenance commit/branch mismatch")
    code = code_alignment(commit, expected)
    if len(config["images"]) != 1000 or len(set(config["images"])) != 1000:
        raise ValueError("full comparison requires 1000 unique views per arm")
    if arm != "control":
        if config.get("model") != ARM_MODELS[arm]:
            raise ValueError(f"the {arm} arm requires the exact model ID {ARM_MODELS[arm]}")
        if provenance.get("model") != config["model"]:
            raise ValueError("preflight model differs from requested model")
    if delivery["provenance"] != provenance or record["provenance"] != provenance or record["config"] != config:
        raise ValueError("conflicting provenance/config records")
    required = set(ARTIFACTS) | {f"{s}_COMPLETE.json" for s in STAGES} | {f"{s}_USAGE.json" for s in STAGES}
    if not required.issubset(delivery["artifacts"]):
        raise ValueError("delivery is missing required artifact hashes")
    for filename, sha in delivery["artifacts"].items():
        if digest(within(run, filename)) != sha:
            raise ValueError(f"delivered artifact changed: {filename}")
    tokens, seconds = 0, 0.0
    for stage in STAGES:
        stage_record = read_json(run / f"{stage}_COMPLETE.json")
        if stage_record.get("status") != "complete" or stage_record.get("stage") != stage:
            raise ValueError("stage is not complete")
        usage = read_json(run / f"{stage}_USAGE.json")
        if usage != stage_record["usage"]:
            raise ValueError("inconsistent stage usage")
        count = usage.get("total_tokens", 0)
        if arm != "control" and stage != "cluster" and count <= 0:
            raise ValueError("reasoning arm has missing token usage")
        tokens += count
        seconds += stage_record["seconds"]
    if arm == "shape":
        segment = list((run / "segment").glob("*/SEARCH.json"))
        crop = list((run / "crop").glob("*/SEARCH.json"))
        # Crop always searches every image; segmentation every image its layout routes to its own VRR.
        if ({p.parent.name for p in segment} != segment_routed(run, config["images"], delivery["artifacts"])
                or {p.parent.name for p in crop} != set(config["images"])):
            raise ValueError("missing SHAPE search records")
        records = [run / "representation_SEARCH.json", run / "batch_SEARCH.json", *segment, *crop,
                   *(run / "segment").glob("*/COHORT_SEARCH.json")]
        for path in records:
            if path.relative_to(run).as_posix() not in delivery["artifacts"]:
                raise ValueError("search record is not bound to the delivery")
            assert_search_clean(read_json(path))
    elif arm == "codex":
        from shape.agent.arms.codex import response, stream_usage
        from .codex_arm import verify_delivery
        for stage in STAGES[:-1]:
            root = run / "agent" / stage
            selected = read_json(root / "SELECTED.json")
            calls = read_json(root / "CALLS.json")
            # Group, dataset and every per-image session keep their own traces.
            traces = sorted(root.rglob("traces/round_*.jsonl"))
            failed = sorted(root.rglob("traces/failed_round_*.jsonl"))   # rounds whose reply was not a decision
            for path in [root / "SELECTED.json", root / "CALLS.json", *traces, *failed,
                         *[p.with_suffix(".json") for p in traces]]:
                if path.relative_to(run).as_posix() not in delivery["artifacts"]:
                    raise ValueError("Codex evidence is not bound to the delivery")
            verify_delivery(root, selected, calls)
            tokens_seen = (sum(response(p, config["model"], p.with_suffix(".json"))[1]["total_tokens"] for p in traces)
                           + sum(stream_usage(p, config["model"])["total_tokens"] for p in failed))
            if not (traces or failed) or tokens_seen != selected["total_tokens"]:
                raise ValueError("Codex trace/usage mismatch")
            if selected != read_json(run / f"{stage}_USAGE.json"):
                raise ValueError("Codex selected trial/stage usage mismatch")
    elif arm in AGENT_ARMS:
        from shape.agent.arms import backend
        from .codex_arm import verify_delivery
        agent = backend(arm)
        for stage in STAGES[:-1]:
            root = run / "agent" / stage
            selected = read_json(root / "SELECTED.json")
            calls = read_json(root / "CALLS.json")
            # Each round keeps its result JSON; a failed round's under failed_round_*.
            traces = sorted(root.rglob("traces/round_*.json")) + sorted(root.rglob("traces/failed_round_*.json"))
            for path in [root / "SELECTED.json", root / "CALLS.json", *traces]:
                if path.relative_to(run).as_posix() not in delivery["artifacts"]:
                    raise ValueError(f"{arm} evidence is not bound to the delivery")
            verify_delivery(root, selected, calls)
            if not traces or sum(agent.trace_tokens(p, config["model"]) for p in traces) != selected["total_tokens"]:
                raise ValueError(f"{arm} trace/usage mismatch")
            if selected != read_json(run / f"{stage}_USAGE.json") or selected.get("policy") != agent.POLICY:
                raise ValueError(f"{arm} selected trial/stage usage mismatch")
    return {"config": config, "provenance": provenance, "delivery": delivery,
            "seconds": seconds, "total_tokens": tokens, "code": code}


def reasoning_protocol(records) -> dict:
    """The per-stage reasoning protocol (CONFIG.json `reasoning`, pipeline.REASONING) of the compared arms: every
    arm records the same protocol and runs each stage with that protocol's settings for its arm, else refused.
    Returns {"protocol", "stages": {arm: per-stage settings}} for COMPARISON.json."""
    from .cli import STAGES
    from .pipeline import stage_reasoning
    protocols = {arm: (r["config"].get("reasoning") or {}).get("protocol") for arm, r in records.items()}
    missing = sorted(arm for arm, protocol in protocols.items() if protocol is None)
    if missing:
        raise ValueError(f"incomparable arms: no per-stage reasoning protocol recorded for {missing}")
    if len({json.dumps(protocol, sort_keys=True) for protocol in protocols.values()}) != 1:
        raise ValueError("incomparable arms: the per-stage reasoning tables differ")
    protocol = next(iter(protocols.values()))
    stages = {arm: record["config"]["reasoning"].get("stages") for arm, record in records.items()}
    for arm, recorded in stages.items():
        if recorded != {stage: stage_reasoning(arm, stage, protocol) for stage in STAGES}:
            raise ValueError(f"incomparable {arm}: its stages' reasoning settings differ from the protocol's")
    return {"protocol": protocol, "stages": stages}


def compatible(records):
    if not set(ARMS) <= set(records) <= set(ARMS) | set(OPTIONAL_ARMS):
        raise ValueError("comparison requires exactly control, shape and codex, and optionally claude and biomni")
    reference = records["control"]
    for arm, record in records.items():
        for key in MATCH_FIELDS:
            if record["config"][key] != reference["config"][key]:
                raise ValueError(f"incomparable {arm}: {key} differs")
        if record["delivery"]["prepare_sha256"] != reference["delivery"]["prepare_sha256"]:
            raise ValueError("different prepared data")
        if record["provenance"]["assets"] != reference["provenance"]["assets"]:
            raise ValueError("different model assets")
    reasoning_protocol(records)
    for arm in (*ARMS, *OPTIONAL_ARMS):
        if arm in records and arm != "control" and records[arm]["config"].get("model") != ARM_MODELS[arm]:
            raise ValueError(f"the {arm} arm must request the exact model ID {ARM_MODELS[arm]}")
    return len({r["provenance"]["gpu"] for r in records.values()}) == 1


def compare(prep: Path, runs: dict, out: Path, expected: str, branch: str = DEFAULT_BRANCH):
    from benchmarks._shared import full_metrics as fm
    from .score import DRAWS, score
    if out.exists():
        raise FileExistsError("comparison output must be fresh")
    records = {arm: validate_run(run, arm, expected, branch) for arm, run in runs.items()}
    same_gpu = compatible(records)
    arms = [arm for arm in (*ARMS, *OPTIONAL_ARMS) if arm in runs]
    agents = [arm for arm in arms if arm in AGENT_ARMS]
    # The control first: every other arm's full_metrics verdicts read its saved draws.
    results = {"control": score(prep, runs["control"])}
    results.update({arm: score(prep, run, control=runs["control"]) for arm, run in runs.items() if arm != "control"})
    verdicts = {arm: {metric: results[arm]["full_metrics"]["metrics"][metric]["vs_control"]
                      for metric in results[arm]["primary"]} for arm in arms if arm != "control"}
    # SHAPE vs each agent arm: the same shared verdict on the two arms' saved readings (delta = SHAPE minus the agent).
    shape = fm.load(runs["shape"] / "evaluation" / DRAWS)
    shape_vs = {arm: {metric: fm.verdict(shape, fm.load(runs[arm] / "evaluation" / DRAWS), metric)
                      for metric in results["shape"]["primary"]} for arm in agents}
    rows = []
    for arm in arms:
        result, record = results[arm], records[arm]
        for endpoint, metrics in result["metrics"].items():
            baseline = results["control"]["metrics"][endpoint]
            rows.append({"arm": arm, "endpoint": endpoint, **metrics,
                         "delta_mAP_vs_control": metrics["mAP"] - baseline["mAP"],
                         "delta_MRR_vs_control": metrics["MRR"] - baseline["MRR"],
                         "delta_top1_vs_control": metrics["top1"] - baseline["top1"],
                         "pipeline_seconds": record["seconds"], "reasoning_tokens": record["total_tokens"],
                         "gpu": record["provenance"]["gpu"], "commit": (record.get("code") or {}).get("commit")})
    summary = {"status": "complete", "commit": expected, "manifest_sha256": results["control"]["manifest_sha256"],
               "primary": results["control"]["primary"],
               "verdicts": {arm: {metric: entry["verdict"] for metric, entry in by.items()}
                            for arm, by in verdicts.items()},
               "full_metrics_vs_control": verdicts,
               "shape_vs_agents": {arm: {metric: entry["verdict"] for metric, entry in by.items()}
                                   for arm, by in shape_vs.items()},
               "full_metrics_shape_vs_agents": shape_vs, "reasoning": reasoning_protocol(records),
               "same_gpu_model": same_gpu,
               "timing_comparable": same_gpu,
               "code": {arm: records[arm].get("code") for arm in arms},
               "notes": ["Single seed; the verdicts' SE is a bootstrap over test wells, not over seeds.",
                         "The agent arms (" + ", ".join(agents) + ") select native-tool calls stagewise under one "
                         "protocol; the trusted worker executes them.",
                         "Each reasoning arm requests its exact model ID (" + ", ".join(
                             f"{arm} {ARM_MODELS[arm]}" for arm in arms if arm != "control")
                         + "); requested identity is not proof of the served model.",
                         "Token usage does not establish model identity when the CLI omits model telemetry.",
                         "GPU model differences confound fixed-wall-time search as well as timing.",
                         "Token totals are reported usage; they are not cost estimates."],
               "runs": {arm: {"path": str(path), "delivery_sha256": digest(path / "DELIVERY.json"),
                              "result_sha256": digest(path / "RESULT.json"),
                              "provenance": records[arm]["provenance"]} for arm, path in runs.items()}, "rows": rows}
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    write_json(out / "COMPARISON.json", summary)
    (out / "COMPARISON_COMPLETE").write_text(expected + "\n", encoding="ascii")
    print(f"COMPARISON_COMPLETE out={out}")
    return summary
