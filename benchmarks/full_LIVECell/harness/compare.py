"""Fail-closed comparison of the full_LIVECell arms (control, SHAPE, Codex; Claude Code and Biomni optional), rescored
from the frozen deliveries.

As full_CPJUMP1/harness/compare.py, with this benchmark's marker, image count and shared code; the stage GT metrics
(reported, paired bootstrap over images) sit next to the three verdicts.
"""
from pathlib import Path
import subprocess

import pandas as pd

from benchmarks.full_CPJUMP1.harness import compare as cpjump1
from benchmarks.full_CPJUMP1.harness.cli import AGENT_ARMS, ARM_MODELS, ARTIFACTS, DEFAULT_BRANCH, STAGES
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, within, write_json

from .spec import LIVECELL

REPO = Path(__file__).resolve().parents[3]
ARMS = ("control", "shape", "codex")                     # required
OPTIONAL_ARMS = cpjump1.OPTIONAL_ARMS                    # claude, biomni: follow-up arms into the same experiment
N_IMAGES, MARKER = LIVECELL.n_images, LIVECELL.marker
MATCH_FIELDS = cpjump1.MATCH_FIELDS
# Beyond full_CPJUMP1's shared code: this benchmark's scorer, preparation and dataset spec. A run at another commit than
# --expected-commit is comparable only if each path has the same git object id at both commits.
SHARED = (*cpjump1.SHARED, "benchmarks/full_LIVECell/harness/score.py", "benchmarks/full_LIVECell/harness/data.py",
          "benchmarks/full_LIVECell/harness/spec.py")
# the agent-arm protocol and the run drivers: their diffs are reported, not refused
PROTOCOL = (*cpjump1.PROTOCOL, "benchmarks/full_LIVECell/harness/cli.py")


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
    naming the paths) and the `git diff --stat` of the protocol modules between the two commits."""
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


def reasoning_tokens(usages: dict) -> int | None:
    """The reasoning tokens of an arm's reasoning stages, when every one of their usage records carries the count."""
    counts = [usages[stage].get("reasoning_tokens") for stage in STAGES if stage != "cluster"]
    return sum(counts) if all(type(c) is int for c in counts) else None


def validate_run(run, arm, expected, branch=DEFAULT_BRANCH):
    from benchmarks.full_CPJUMP1.harness.pipeline import assert_search_clean
    delivery = read_json(run / "DELIVERY.json")
    record = read_json(run / "RUN_META.json")
    config = read_json(run / "CONFIG.json")
    provenance = read_json(run / "PREFLIGHT.json")
    if arm not in (*ARMS, *OPTIONAL_ARMS) or any(r.get("arm") != arm for r in (delivery, record, config)):
        raise ValueError("arm identity mismatch")
    if delivery.get("smoke") or config.get("smoke") or any(r.get("status") != "complete" for r in (delivery, record)):
        raise ValueError("all arms must be complete full runs, not smokes")
    if config.get("dataset") != LIVECELL.name:
        raise ValueError(f"not a {LIVECELL.name} run: CONFIG.json dataset is {config.get('dataset')!r}")
    commit = provenance.get("commit")
    if (run / MARKER).read_text(encoding="ascii").strip() != commit:
        raise ValueError("completion marker commit mismatch")
    if not isinstance(commit, str) or provenance.get("branch") != branch:
        raise ValueError("run provenance commit/branch mismatch")
    code = code_alignment(commit, expected)
    if len(config["images"]) != N_IMAGES or len(set(config["images"])) != N_IMAGES:
        raise ValueError(f"full comparison requires {N_IMAGES} unique images per arm")
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
    tokens, seconds, usages = 0, 0.0, {}
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
        usages[stage] = usage
    if arm == "shape":
        segment = list((run / "segment").glob("*/SEARCH.json"))
        crop = list((run / "crop").glob("*/SEARCH.json"))
        # Crop always searches every image; segmentation every image its layout routes to its own VRR.
        if ({p.parent.name for p in segment} != cpjump1.segment_routed(run, config["images"], delivery["artifacts"])
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
        from benchmarks.full_CPJUMP1.harness.codex_arm import verify_delivery
        for stage in STAGES[:-1]:
            root = run / "agent" / stage
            selected = read_json(root / "SELECTED.json")
            calls = read_json(root / "CALLS.json")
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
            if selected != usages[stage]:
                raise ValueError("Codex selected trial/stage usage mismatch")
    elif arm in AGENT_ARMS:          # claude, biomni: the same stage protocol, their own backend's traces
        from shape.agent.arms import backend
        from benchmarks.full_CPJUMP1.harness.codex_arm import verify_delivery
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
            if selected != usages[stage] or selected.get("policy") != agent.POLICY:
                raise ValueError(f"{arm} selected trial/stage usage mismatch")
    return {"config": config, "provenance": provenance, "delivery": delivery, "seconds": seconds,
            "total_tokens": tokens, "reasoning_tokens": reasoning_tokens(usages), "code": code}


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
        if arm != "control" and record["config"].get("model") != ARM_MODELS[arm]:
            raise ValueError(f"the {arm} arm must request the exact model ID {ARM_MODELS[arm]}")
    cpjump1.reasoning_protocol(records)
    return len({r["provenance"]["gpu"] for r in records.values()}) == 1


def rows_of(arm: str, result: dict, control: dict, record: dict) -> list[dict]:
    """comparison.csv rows of one arm (long format): the three verdict metrics (main split, vs the control), the
    retrieval secondaries (point deltas) and every stage GT metric (paired bootstrap delta, SE, interval)."""
    common = {"pipeline_seconds": record["seconds"], "total_tokens": record["total_tokens"],
              "reasoning_tokens": record["reasoning_tokens"], "gpu": record["provenance"]["gpu"],
              "commit": (record.get("code") or {}).get("commit")}
    rows = []
    for metric in result["primary"]:
        entry = result["full_metrics"]["metrics"][metric]
        vs = entry["vs_control"] or {}
        rows.append({"arm": arm, "kind": "verdict", "endpoint": "full_metrics/main", "metric": metric,
                     "value": entry["main"]["value"], "delta_vs_control": vs.get("delta"),
                     "se_vs_control": vs.get("se"), "ci95_low": (vs.get("ci95") or [None, None])[0],
                     "ci95_high": (vs.get("ci95") or [None, None])[1], "guard_delta": vs.get("guard_delta"),
                     "verdict": vs.get("verdict"), "n": result["design"]["n_heldout_test_units"], **common})
    for endpoint, metrics in result["metrics"].items():
        for metric in ("mAP", "MRR", "top1"):
            rows.append({"arm": arm, "kind": "retrieval", "endpoint": endpoint, "metric": metric,
                         "value": metrics[metric], "delta_vs_control": metrics[metric] - control["metrics"][endpoint][metric],
                         "n": metrics["n_queries"], **common})
    gt = result["stage_gt"]
    for family in ("segmentation", "crop"):
        for group, entries in gt[family].items():
            for metric, entry in entries.items():
                vs = entry["vs_control"] or {}
                rows.append({"arm": arm, "kind": "stage_gt", "endpoint": f"{family}/{group}", "metric": metric,
                             "value": entry["value"], "delta_vs_control": vs.get("delta"),
                             "se_vs_control": vs.get("se"), "ci95_low": (vs.get("ci95") or [None, None])[0],
                             "ci95_high": (vs.get("ci95") or [None, None])[1], "n": gt["n_images"][group], **common})
    return rows


def compare(prep: Path, runs: dict, out: Path, expected: str, branch: str = DEFAULT_BRANCH):
    from benchmarks._shared import full_metrics as fm
    from .score import DRAWS, score
    if out.exists():
        raise FileExistsError("comparison output must be fresh")
    records = {arm: validate_run(run, arm, expected, branch) for arm, run in runs.items()}
    same_gpu = compatible(records)
    # The control first: every other arm's verdicts and stage GT deltas read its saved readings.
    arms = [arm for arm in (*ARMS, *OPTIONAL_ARMS) if arm in runs]
    agents = [arm for arm in arms if arm in AGENT_ARMS]
    results = {"control": score(prep, runs["control"])}
    results.update({arm: score(prep, runs[arm], control=runs["control"]) for arm in arms if arm != "control"})
    verdicts = {arm: {metric: results[arm]["full_metrics"]["metrics"][metric]["vs_control"]
                      for metric in results[arm]["primary"]} for arm in arms if arm != "control"}
    # SHAPE vs each agent arm: the same shared verdict on the two arms' saved readings (delta = SHAPE minus the agent).
    shape_vs = {arm: {metric: fm.verdict(fm.load(runs["shape"] / "evaluation" / DRAWS),
                                         fm.load(runs[arm] / "evaluation" / DRAWS), metric)
                      for metric in results["shape"]["primary"]} for arm in agents}
    stage_gt = {arm: {family: results[arm]["stage_gt"][family] for family in ("segmentation", "crop")}
                for arm in arms if arm != "control"}
    rows = [row for arm in arms for row in rows_of(arm, results[arm], results["control"], records[arm])]
    summary = {"status": "complete", "commit": expected, "manifest_sha256": results["control"]["manifest_sha256"],
               "design": results["control"]["design"],
               "primary": results["control"]["primary"],
               "verdicts": {arm: {metric: entry["verdict"] for metric, entry in by.items()}
                            for arm, by in verdicts.items()},
               "full_metrics_vs_control": verdicts,
               "shape_vs_agents": {arm: {metric: entry["verdict"] for metric, entry in by.items()}
                                   for arm, by in shape_vs.items()},
               "full_metrics_shape_vs_agents": shape_vs,
               "stage_gt_control": {family: results["control"]["stage_gt"][family]
                                    for family in ("segmentation", "crop")},
               "stage_gt_vs_control": stage_gt, "stage_gt_rule": results["control"]["stage_gt"]["rule"],
               "batch_decisions": {arm: results[arm]["batch_decision"] for arm in arms},
               "reasoning": cpjump1.reasoning_protocol(records),
               "same_gpu_model": same_gpu, "timing_comparable": same_gpu,
               "code": {arm: records[arm].get("code") for arm in arms},
               "notes": ["Single seed; the verdicts' SE is a bootstrap over test units, the stage GT SE a bootstrap "
                         "over images, neither over seeds.",
                         "Stage GT metrics are reported, not verdicts; Cellpose-SAM may have trained on LIVECell, "
                         "so the official_test group is the read least exposed to training leakage.",
                         "The agent arms (" + ", ".join(agents) + ") select native-tool calls stagewise under the "
                         "full_CPJUMP1 protocol; the trusted worker executes them.",
                         "Each reasoning arm requests its exact model ID (" + ", ".join(
                             f"{arm} {ARM_MODELS[arm]}" for arm in arms if arm != "control")
                         + "); requested identity is not proof of the served model.",
                         "GPU model differences confound fixed-wall-time search as well as timing.",
                         "total_tokens are reported usage, not cost estimates; reasoning_tokens is null unless every "
                         "reasoning stage's usage record carries it."],
               "runs": {arm: {"path": str(path), "delivery_sha256": digest(path / "DELIVERY.json"),
                              "result_sha256": digest(path / "RESULT.json"),
                              "provenance": records[arm]["provenance"]} for arm, path in runs.items()}, "rows": rows}
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    write_json(out / "COMPARISON.json", summary)
    (out / "COMPARISON_COMPLETE").write_text(expected + "\n", encoding="ascii")
    print(f"COMPARISON_COMPLETE out={out}")
    return summary
