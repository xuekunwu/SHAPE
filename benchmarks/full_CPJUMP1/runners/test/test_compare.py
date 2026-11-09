from copy import deepcopy

import pytest

from benchmarks.full_CPJUMP1.harness.compare import ARMS, compatible, validate_run
from benchmarks.full_CPJUMP1.harness.data import write_json
from benchmarks.full_CPJUMP1.harness.pipeline import reasoning_record


def records():
    config = dict(seed=0, cores=8, budget_hours=11, train_cells_per_site=128, inference_chunk=2048,
                  images=["i1", "i2"], model="gpt-5.6-luna")
    return {a: {"config": {**deepcopy(config), "reasoning": reasoning_record(a)}, "delivery": {"prepare_sha256": "data"},
                "provenance": {"assets": {"sha": "assets"}, "gpu": "same"}} for a in ARMS}


def test_all_three_arms_required():
    r = records()
    assert compatible(r)
    del r["codex"]
    with pytest.raises(ValueError, match="exactly"):
        compatible(r)


@pytest.mark.parametrize("key,value", [("seed", 1), ("cores", 4), ("budget_hours", 4),
                                     ("train_cells_per_site", 10), ("inference_chunk", 128),
                                     ("images", ["i1"]), ("model", "different")])
def test_unmatched_protocols_refused(key, value):
    r = records()
    r["codex"]["config"][key] = value
    with pytest.raises(ValueError):
        compatible(r)


def test_gpu_difference_flagged_and_asset_mismatch_refused():
    r = records()
    r["codex"]["provenance"]["gpu"] = "different"
    assert compatible(r) is False
    r["codex"]["provenance"]["assets"] = {"sha": "different"}
    with pytest.raises(ValueError, match="assets"):
        compatible(r)


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-luna-latest", "claude-sonnet-5", None])
def test_equal_but_unrequested_models_refused(model):
    r = records()
    for arm in ("shape", "codex"):
        r[arm]["config"]["model"] = model
    with pytest.raises(ValueError, match="gpt-5.6-luna"):
        compatible(r)


def test_smoke_not_a_comparison_result(tmp_path):
    for name in ("DELIVERY", "RUN_META", "CONFIG", "PREFLIGHT"):
        write_json(tmp_path / f"{name}.json", {"status": "complete", "arm": "shape", "smoke": True})
    with pytest.raises(ValueError, match="smokes"):
        validate_run(tmp_path, "shape", "a" * 40)


def test_comparison_recomputes_three_real_fixed_design_scores(tmp_path, monkeypatch):
    import shutil
    import importlib
    import pandas as pd
    from benchmarks.full_CPJUMP1.harness.data import read_json
    from .test_score import complete_design
    comparison = importlib.import_module("benchmarks.full_CPJUMP1.harness.compare")
    prep, control = complete_design(tmp_path)
    runs = {"control": control}
    r = records()
    for arm in ARMS:
        if arm != "control":
            runs[arm] = tmp_path / arm
            shutil.copytree(control, runs[arm])
            delivery = read_json(runs[arm] / "DELIVERY.json")
            delivery["arm"] = arm
            write_json(runs[arm] / "DELIVERY.json", delivery)
        write_json(runs[arm] / "RESULT.json", {"fake_stale_score": 999})
        r[arm].update(seconds=20, total_tokens=0 if arm == "control" else 100)
    # Provenance validation has separate tests; exercise the real numerical scorer here.
    monkeypatch.setattr(comparison, "validate_run", lambda run, arm, expected, branch: r[arm])
    summary = comparison.compare(prep, runs, tmp_path / "comparison", "a" * 40)
    assert len(summary["rows"]) == 12
    same = {m: "NO_RESOLVABLE_DIFFERENCE" for m in ("mAP", "macro_f1", "balanced_accuracy")}   # identical arms
    assert summary["verdicts"] == {"shape": same, "codex": same}
    assert summary["full_metrics_vs_control"]["shape"]["mAP"]["delta"] == 0
    assert summary["reasoning"]["protocol"]["table"]["batch"]["effort"] == "high"
    assert read_json(tmp_path / "comparison/COMPARISON.json")["reasoning"]["stages"] == {
        arm: reasoning_record(arm)["stages"] for arm in ARMS}
    assert read_json(runs["shape"] / "RESULT.json")["full_metrics"]["verdicts"] == same
    table = pd.read_csv(tmp_path / "comparison/comparison.csv")
    assert set(table.arm) == set(ARMS)
    assert set(table[table.endpoint.str.contains("cross_guide")].mAP) == {1.0}
    # A gene-perfect representation ties both guides; same-guide retrieval is intentionally lower.
    assert table[table.endpoint.str.contains("same_guide")].mAP.max() < 1
    assert set(table.delta_mAP_vs_control) == {0.0}
    assert "fake_stale_score" not in read_json(runs["codex"] / "RESULT.json")
    assert (tmp_path / "comparison/COMPARISON_COMPLETE").is_file()


def test_validate_complete_control_and_detect_tampering(tmp_path):
    from benchmarks.full_CPJUMP1.harness.cli import ARTIFACTS, STAGES
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json
    expected = "a" * 40
    config = {"arm": "control", "smoke": False, "images": [f"i{i}" for i in range(1000)]}
    provenance = {"branch": "e2e", "commit": expected}
    for filename in ARTIFACTS:
        (tmp_path / filename).write_text("synthetic")
    write_json(tmp_path / "CONFIG.json", config)
    write_json(tmp_path / "PREFLIGHT.json", provenance)
    write_json(tmp_path / "RUN_META.json", {"status": "complete", "arm": "control", "config": config, "provenance": provenance})
    for stage in STAGES:
        write_json(tmp_path / f"{stage}_USAGE.json", {"total_tokens": 0})
        write_json(tmp_path / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage,
                                                       "seconds": 5, "usage": {"total_tokens": 0}})
    artifacts = {p.name: digest(p) for p in tmp_path.iterdir() if p.is_file()}
    write_json(tmp_path / "DELIVERY.json", {"status": "complete", "arm": "control", "smoke": False,
                                           "provenance": provenance, "artifacts": artifacts})
    (tmp_path / "CPJUMP1_COMPLETE").write_text(expected)
    assert validate_run(tmp_path, "control", expected)["seconds"] == 25
    (tmp_path / "embedding.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_run(tmp_path, "control", expected)


def codex_delivery(root):
    import json
    from benchmarks.full_CPJUMP1.harness.cli import ARTIFACTS, STAGES
    from benchmarks.full_CPJUMP1.harness.data import digest
    expected = "a" * 40
    config = {"arm": "codex", "model": "gpt-5.6-luna", "smoke": False,
              "images": [f"i{i}" for i in range(1000)]}
    provenance = {"branch": "e2e", "commit": expected, "model": config["model"],
                  "requested_model": config["model"], "response_model": None,
                  "response_model_attested": False}
    for filename in ARTIFACTS:
        (root / filename).write_text("synthetic")
    write_json(root / "CONFIG.json", config)
    write_json(root / "PREFLIGHT.json", provenance)
    write_json(root / "RUN_META.json", {"status": "complete", "arm": "codex",
                                       "config": config, "provenance": provenance})
    for stage in STAGES:
        usage = {"total_tokens": 0}
        if stage != "cluster":
            agent = root / "agent" / stage
            choice = dict(action="finish", request_json='{"all": "c0"}', reason="baseline sufficient")
            usage = {"deliveries": {"i0": {"call": "c0", "source": "agent", "session": "group"}},
                     "total_tokens": 12, "requested_model": config["model"], "response_model_attested": False}
            write_json(agent / "SELECTED.json", usage)
            write_json(agent / "CALLS.json", {"i0": {"c0": {"status": "complete", "folder": "waves/w000/i0/c0",
                                                            "artifacts": {}}}})
            write_json(agent / "group" / "traces" / "round_000.json", choice)
            events = [{"type": "thread.started", "thread_id": stage}, {"type": "turn.started"},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(choice)}},
                      {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2,
                                                            "cached_input_tokens": 3}}]
            (agent / "group" / "traces" / "round_000.jsonl").write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
        write_json(root / f"{stage}_USAGE.json", usage)
        write_json(root / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": 5, "usage": usage})
    artifacts = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "DELIVERY.json", {"status": "complete", "arm": "codex", "smoke": False,
                                       "provenance": provenance, "artifacts": artifacts})
    (root / "CPJUMP1_COMPLETE").write_text(expected)
    return expected


def test_codex_comparison_reads_trace_and_final_without_attesting_model(tmp_path):
    expected = codex_delivery(tmp_path)
    result = validate_run(tmp_path, "codex", expected)
    assert result["total_tokens"] == 48
    assert result["provenance"]["response_model_attested"] is False


@pytest.mark.parametrize("filename", ["round_000.json", "round_000.jsonl"])
def test_codex_trace_and_final_must_be_bound_to_delivery(tmp_path, filename):
    from benchmarks.full_CPJUMP1.harness.data import read_json
    expected = codex_delivery(tmp_path)
    delivery = read_json(tmp_path / "DELIVERY.json")
    del delivery["artifacts"][f"agent/segment/group/traces/{filename}"]
    write_json(tmp_path / "DELIVERY.json", delivery)
    with pytest.raises(ValueError, match="bound to the delivery"):
        validate_run(tmp_path, "codex", expected)


def test_codex_final_message_tampering_refused(tmp_path):
    expected = codex_delivery(tmp_path)
    write_json(tmp_path / "agent/segment/group/traces/round_000.json", {})
    with pytest.raises(ValueError, match="artifact changed"):
        validate_run(tmp_path, "codex", expected)


CLEAN_SEARCH = {"reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [], "stop_reason": "satisfied"}


def shape_run(root, layout):
    """A complete full SHAPE run on the given segmentation layout (`per_image`, or `cohort` as
    pipeline._segmentation_cohort writes it: 28 calibration images, 3 routed, the rest cohort mode); returns a
    function that re-binds the delivery after an edit."""
    from benchmarks.full_CPJUMP1.harness.cli import ARTIFACTS, DEFAULT_MODEL, STAGES
    from benchmarks.full_CPJUMP1.harness.data import digest
    expected, images = "a" * 40, [f"i{i:04d}" for i in range(1000)]
    config = {"arm": "shape", "smoke": False, "images": images, "model": DEFAULT_MODEL}
    provenance = {"branch": "e2e", "commit": expected, "model": DEFAULT_MODEL}
    for filename in ARTIFACTS:
        (root / filename).write_text("synthetic")
    write_json(root / "CONFIG.json", config)
    write_json(root / "PREFLIGHT.json", provenance)
    write_json(root / "RUN_META.json", {"status": "complete", "arm": "shape", "config": config, "provenance": provenance})
    for stage in STAGES:
        write_json(root / f"{stage}_USAGE.json", {"total_tokens": 10})
        write_json(root / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage,
                                                   "seconds": 5, "usage": {"total_tokens": 10}})
    for name in ("representation_SEARCH.json", "batch_SEARCH.json"):
        write_json(root / name, CLEAN_SEARCH)
    for ident in images:
        write_json(root / "crop" / ident / "SEARCH.json", CLEAN_SEARCH)
    source = {i: "per_image_vrr" for i in images}
    if layout == "cohort":
        calibration, routed = images[:28], images[28:31]
        source = {i: "calibration" if i in calibration else "per_image_vrr" if i in routed else "cohort"
                  for i in images}
        write_json(root / "segment_COHORT.json", {"parameter_source": source, "groups": [
            {"images": images, "calibration": calibration, "routed": routed}]})
        for ident in images:
            if ident not in calibration:
                write_json(root / "segment" / ident / "COHORT_SEARCH.json", CLEAN_SEARCH)
    for ident in images:
        if source[ident] != "cohort":
            write_json(root / "segment" / ident / "SEARCH.json", CLEAN_SEARCH)
    write_json(root / "segmentation.json", [{"image_id": i, "parameter_source": source[i]} for i in images])
    (root / "CPJUMP1_COMPLETE").write_text(expected)

    def deliver():
        artifacts = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
                     if p.is_file() and p.name not in ("DELIVERY.json", "CPJUMP1_COMPLETE")}
        write_json(root / "DELIVERY.json", {"status": "complete", "arm": "shape", "smoke": False,
                                           "provenance": provenance, "artifacts": artifacts})

    deliver()
    return deliver


@pytest.mark.parametrize("layout", ["per_image", "cohort"])
def test_shape_search_records_follow_the_segmentation_layout(tmp_path, layout):
    shape_run(tmp_path, layout)
    assert validate_run(tmp_path, "shape", "a" * 40)["total_tokens"] == 50


def test_per_image_layout_still_needs_every_segment_search(tmp_path):
    deliver = shape_run(tmp_path, "per_image")
    (tmp_path / "segment" / "i0500" / "SEARCH.json").unlink()
    deliver()
    with pytest.raises(ValueError, match="missing SHAPE search records"):
        validate_run(tmp_path, "shape", "a" * 40)


def test_cohort_routed_image_needs_both_of_its_records(tmp_path):
    deliver = shape_run(tmp_path, "cohort")
    (tmp_path / "segment" / "i0029" / "SEARCH.json").unlink()         # a routed image's own VRR
    deliver()
    with pytest.raises(ValueError, match="missing SHAPE search records"):
        validate_run(tmp_path, "shape", "a" * 40)
    write_json(tmp_path / "segment" / "i0029" / "SEARCH.json", CLEAN_SEARCH)
    (tmp_path / "segment" / "i0029" / "COHORT_SEARCH.json").unlink()  # ... and its cohort-mode pass
    deliver()
    with pytest.raises(ValueError, match="cohort-mode records"):
        validate_run(tmp_path, "shape", "a" * 40)


def test_cohort_records_must_match_the_cohort_record_and_stay_clean(tmp_path):
    deliver = shape_run(tmp_path, "cohort")
    write_json(tmp_path / "segment" / "i0500" / "SEARCH.json", CLEAN_SEARCH)     # an unrouted extra search
    deliver()
    with pytest.raises(ValueError, match="missing SHAPE search records"):
        validate_run(tmp_path, "shape", "a" * 40)
    (tmp_path / "segment" / "i0500" / "SEARCH.json").unlink()
    write_json(tmp_path / "segment" / "i0500" / "COHORT_SEARCH.json", {**CLEAN_SEARCH, "fallback_rounds": 1})
    deliver()
    with pytest.raises(RuntimeError, match="fallback_rounds"):
        validate_run(tmp_path, "shape", "a" * 40)
    write_json(tmp_path / "segment" / "i0500" / "COHORT_SEARCH.json", CLEAN_SEARCH)
    rows = [{"image_id": f"i{i:04d}", "parameter_source": "cohort"} for i in range(1000)]
    write_json(tmp_path / "segmentation.json", rows)
    deliver()
    with pytest.raises(ValueError, match="parameter sources"):
        validate_run(tmp_path, "shape", "a" * 40)


def test_codex_failed_round_tokens_are_counted_and_bound(tmp_path):
    import json
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json
    expected = codex_delivery(tmp_path)
    agent = tmp_path / "agent" / "segment"
    events = [{"type": "thread.started", "thread_id": "failed"}, {"type": "turn.started"},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "not a decision"}},
              {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 2}}]
    failed = agent / "images" / "i0" / "traces" / "failed_round_000.jsonl"
    failed.parent.mkdir(parents=True)
    failed.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    for path in (agent / "SELECTED.json", tmp_path / "segment_USAGE.json"):
        record = read_json(path)
        record["total_tokens"] += 9
        write_json(path, record)
    complete = read_json(tmp_path / "segment_COMPLETE.json")
    complete["usage"] = read_json(tmp_path / "segment_USAGE.json")
    write_json(tmp_path / "segment_COMPLETE.json", complete)
    delivery = read_json(tmp_path / "DELIVERY.json")
    for path in (agent / "SELECTED.json", tmp_path / "segment_USAGE.json", tmp_path / "segment_COMPLETE.json", failed):
        delivery["artifacts"][path.relative_to(tmp_path).as_posix()] = digest(path)
    write_json(tmp_path / "DELIVERY.json", delivery)
    assert validate_run(tmp_path, "codex", expected)["total_tokens"] == 57
    del delivery["artifacts"]["agent/segment/images/i0/traces/failed_round_000.jsonl"]
    write_json(tmp_path / "DELIVERY.json", delivery)
    with pytest.raises(ValueError, match="bound to the delivery"):
        validate_run(tmp_path, "codex", expected)


# ---------------------------------------------------------------- the optional agent arms

def all_records():
    r = records()
    for arm in ("claude", "biomni"):
        r[arm] = deepcopy(r["codex"])
        r[arm]["config"]["reasoning"] = reasoning_record(arm)
    r["claude"]["config"]["model"] = "claude-sonnet-5"
    return r


def test_optional_agent_arms_are_checked_like_the_required_ones():
    r = all_records()
    assert compatible(r)
    del r["biomni"]
    assert compatible(r)
    r["claude"]["config"]["seed"] = 1
    with pytest.raises(ValueError, match="incomparable claude: seed"):
        compatible(r)
    r = all_records()
    r["claude"]["config"]["model"] = "gpt-5.6-luna"            # the other arms' model is not claude's
    with pytest.raises(ValueError, match="claude arm must request the exact model ID claude-sonnet-5"):
        compatible(r)
    r = all_records()
    r["biomni"]["config"]["model"] = "claude-sonnet-5"
    with pytest.raises(ValueError, match="biomni arm"):
        compatible(r)
    r = all_records()
    r["react"] = deepcopy(r["codex"])
    with pytest.raises(ValueError, match="exactly"):
        compatible(r)
    r = all_records()
    del r["shape"]
    with pytest.raises(ValueError, match="exactly"):
        compatible(r)


def test_comparison_reports_every_arm_vs_control_and_shape_vs_each_agent_arm(tmp_path, monkeypatch):
    import shutil
    import importlib
    import pandas as pd
    from benchmarks.full_CPJUMP1.harness.data import read_json
    from .test_score import complete_design
    comparison = importlib.import_module("benchmarks.full_CPJUMP1.harness.compare")
    prep, control = complete_design(tmp_path)
    runs, r = {"control": control}, all_records()
    for arm in r:
        if arm != "control":
            runs[arm] = tmp_path / arm
            shutil.copytree(control, runs[arm])
            delivery = read_json(runs[arm] / "DELIVERY.json")
            delivery["arm"] = arm
            write_json(runs[arm] / "DELIVERY.json", delivery)
        r[arm].update(seconds=20, total_tokens=0 if arm == "control" else 100,
                      code={"commit": ("b" if arm in ("claude", "biomni") else "a") * 40})
    monkeypatch.setattr(comparison, "validate_run", lambda run, arm, expected, branch: r[arm])
    summary = comparison.compare(prep, runs, tmp_path / "comparison", "a" * 40)
    same = {m: "NO_RESOLVABLE_DIFFERENCE" for m in ("mAP", "macro_f1", "balanced_accuracy")}
    assert summary["verdicts"] == {arm: same for arm in ("shape", "codex", "claude", "biomni")}
    assert summary["shape_vs_agents"] == {arm: same for arm in ("codex", "claude", "biomni")}
    assert summary["full_metrics_shape_vs_agents"]["claude"]["mAP"]["delta"] == 0
    assert summary["code"]["claude"]["commit"] == "b" * 40 and summary["code"]["codex"]["commit"] == "a" * 40
    assert len(summary["rows"]) == 20
    table = pd.read_csv(tmp_path / "comparison/comparison.csv")
    assert list(dict.fromkeys(table.arm)) == ["control", "shape", "codex", "claude", "biomni"]
    assert set(table[table.arm == "biomni"].commit) == {"b" * 40}
    assert "claude claude-sonnet-5" in " ".join(summary["notes"])


@pytest.fixture
def shared_repo(tmp_path, monkeypatch):
    """A git repository holding every SHARED path; commits: base, protocol (codex_arm.py and cli.py changed), score (a
    shared file changed). compare reads it as its REPO."""
    import subprocess
    from benchmarks.full_CPJUMP1.harness import compare
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("config", "core.autocrlf", "false")
    files = ["shape/tool.py", "benchmarks/_shared/full_metrics.py", "benchmarks/full_HCMA/harness/agents.py",
             "benchmarks/full_HCMA/workspace_template/README.md",
             "benchmarks/representation_RxRx1/runners/epoch_budget.py",
             *[f"benchmarks/full_CPJUMP1/harness/{n}.py" for n in ("pipeline", "score", "data", "merged", "codex_arm",
                                                                    "claude_arm", "cli")]]
    for name in files:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    commits = {"base": git("rev-parse", "HEAD")}
    (repo / "benchmarks/full_CPJUMP1/harness/codex_arm.py").write_text("# protocol\nBACKENDS = {}\n", encoding="utf-8")
    (repo / "benchmarks/full_CPJUMP1/harness/cli.py").write_text("# arms\nAGENT_ARMS = ()\n", encoding="utf-8")
    git("commit", "-q", "-am", "protocol")
    commits["protocol"] = git("rev-parse", "HEAD")
    (repo / "benchmarks/full_CPJUMP1/harness/score.py").write_text("# rescored\n", encoding="utf-8")
    git("commit", "-q", "-am", "score")
    commits["score"] = git("rev-parse", "HEAD")
    monkeypatch.setattr(compare, "REPO", repo)
    return commits


def test_code_alignment_accepts_equal_shared_ids_and_refuses_a_shared_difference(shared_repo):
    from benchmarks.full_CPJUMP1.harness.compare import SHARED, code_alignment
    base = shared_repo["base"]
    assert code_alignment(base, base) == {"commit": base, "aligned_by": "same commit"}
    record = code_alignment(shared_repo["protocol"], base)
    assert record["aligned_by"] == "shared git object ids" and set(record["shared"]) == set(SHARED)
    assert all(record["shared"].values())
    assert "codex_arm.py" in record["protocol_diff_stat"] and "cli.py" in record["protocol_diff_stat"]
    assert "2 files changed" in record["protocol_diff_stat"]
    with pytest.raises(ValueError, match="shared code: \\['benchmarks/full_CPJUMP1/harness/score.py'\\]"):
        code_alignment(shared_repo["score"], base)
    with pytest.raises(ValueError, match="not in"):
        code_alignment("f" * 40, base)


def agent_delivery(root, arm, commit):
    """A complete claude or biomni run at `commit`: each stage's one decision round in its own trace format."""
    import json
    from benchmarks.full_CPJUMP1.harness.cli import ARM_MODELS, ARTIFACTS, STAGES
    from benchmarks.full_CPJUMP1.harness.codex_arm import backend
    from benchmarks.full_CPJUMP1.harness.data import digest
    root.mkdir(parents=True, exist_ok=True)
    config = {"arm": arm, "model": ARM_MODELS[arm], "smoke": False, "images": [f"i{i}" for i in range(1000)]}
    provenance = {"branch": "e2e", "commit": commit, "model": config["model"]}
    for filename in ARTIFACTS:
        (root / filename).write_text("synthetic")
    write_json(root / "CONFIG.json", config)
    write_json(root / "PREFLIGHT.json", provenance)
    write_json(root / "RUN_META.json", {"status": "complete", "arm": arm, "config": config, "provenance": provenance})
    choice = dict(action="finish", request_json='{"all": "c0"}', reason="baseline sufficient")
    for stage in STAGES:
        usage = {"total_tokens": 0}
        if stage != "cluster":
            agent = root / "agent" / stage
            if arm == "claude":
                trace = {"type": "result", "subtype": "success", "is_error": False, "structured_output": choice,
                         "usage": {"input_tokens": 8, "output_tokens": 4}, "modelUsage": {config["model"]: {}}}
            else:
                trace = {"decision": choice, "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                         "steps": 1, "refused_executions": 0, "messages": []}
            (agent / "group" / "traces").mkdir(parents=True)
            (agent / "group" / "traces" / "round_000.json").write_text(json.dumps(trace), encoding="utf-8")
            usage = {"deliveries": {"i0": {"call": "c0", "source": "agent", "session": "group"}},
                     "total_tokens": 12, "requested_model": config["model"], "response_model_attested": False,
                     "policy": backend(arm).POLICY}
            write_json(agent / "SELECTED.json", usage)
            write_json(agent / "CALLS.json", {"i0": {"c0": {"status": "complete", "folder": "waves/w000/i0/c0",
                                                            "artifacts": {}}}})
        write_json(root / f"{stage}_USAGE.json", usage)
        write_json(root / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": 5, "usage": usage})
    artifacts = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file()}
    write_json(root / "DELIVERY.json", {"status": "complete", "arm": arm, "smoke": False,
                                       "provenance": provenance, "artifacts": artifacts})
    (root / "CPJUMP1_COMPLETE").write_text(commit)


@pytest.mark.parametrize("arm", ["claude", "biomni"])
def test_new_agent_arm_at_the_expected_commit_validates_its_traces(tmp_path, arm):
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json
    agent_delivery(tmp_path, arm, "a" * 40)
    record = validate_run(tmp_path, arm, "a" * 40)
    assert record["total_tokens"] == 48 and record["code"] == {"commit": "a" * 40, "aligned_by": "same commit"}
    delivery = read_json(tmp_path / "DELIVERY.json")
    trace = delivery["artifacts"].pop("agent/crop/group/traces/round_000.json")
    write_json(tmp_path / "DELIVERY.json", delivery)
    with pytest.raises(ValueError, match="bound to the delivery"):
        validate_run(tmp_path, arm, "a" * 40)
    delivery["artifacts"]["agent/crop/group/traces/round_000.json"] = trace
    # Usage the traces do not account for.
    selected = {**read_json(tmp_path / "agent/crop/SELECTED.json"), "total_tokens": 13}
    write_json(tmp_path / "agent/crop/SELECTED.json", selected)
    write_json(tmp_path / "crop_USAGE.json", selected)
    write_json(tmp_path / "crop_COMPLETE.json", {**read_json(tmp_path / "crop_COMPLETE.json"), "usage": selected})
    for path in ("agent/crop/SELECTED.json", "crop_USAGE.json", "crop_COMPLETE.json"):
        delivery["artifacts"][path] = digest(tmp_path / path)
    write_json(tmp_path / "DELIVERY.json", delivery)
    with pytest.raises(ValueError, match="trace/usage mismatch"):
        validate_run(tmp_path, arm, "a" * 40)


def test_new_agent_arm_model_is_checked_per_arm(tmp_path):
    from benchmarks.full_CPJUMP1.harness.data import read_json
    agent_delivery(tmp_path, "claude", "a" * 40)
    config = read_json(tmp_path / "CONFIG.json")
    config["model"] = "gpt-5.6-luna"
    write_json(tmp_path / "CONFIG.json", config)
    with pytest.raises(ValueError, match="claude arm requires the exact model ID claude-sonnet-5"):
        validate_run(tmp_path, "claude", "a" * 40)


def test_a_run_at_another_commit_passes_only_with_equal_shared_ids_and_records_it(tmp_path, shared_repo):
    base = shared_repo["base"]
    agent_delivery(tmp_path / "claude", "claude", shared_repo["protocol"])
    record = validate_run(tmp_path / "claude", "claude", base)
    assert record["code"]["commit"] == shared_repo["protocol"] and record["code"]["expected_commit"] == base
    assert "codex_arm.py" in record["code"]["protocol_diff_stat"]
    agent_delivery(tmp_path / "biomni", "biomni", shared_repo["score"])
    with pytest.raises(ValueError, match="differs from .* in shared code"):
        validate_run(tmp_path / "biomni", "biomni", base)
    agent_delivery(tmp_path / "stale", "claude", shared_repo["protocol"])
    (tmp_path / "stale" / "CPJUMP1_COMPLETE").write_text(base)     # the marker must name the run's own commit
    with pytest.raises(ValueError, match="completion marker"):
        validate_run(tmp_path / "stale", "claude", base)


def test_only_the_run_time_file_of_the_shared_folder_is_compared(shared_repo):
    """benchmarks/_shared holds analysis-only files too (framework text, post-hoc readouts and their tests): a commit
    that changes only those aligns; one that changes full_metrics.py is refused."""
    import subprocess
    from benchmarks.full_CPJUMP1.harness import compare
    repo = compare.REPO

    def commit(files, message):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", message], check=True, capture_output=True)
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True,
                              text=True).stdout.strip()

    base = shared_repo["score"]
    analysis = commit({"benchmarks/_shared/agentic_process_readouts.py": "# readouts\n",
                       "benchmarks/_shared/test_agentic_process_readouts.py": "# test\n",
                       "benchmarks/_shared/AGENTIC_EVALUATION_FRAMEWORK.md": "framework\n",
                       "benchmarks/_shared/FULL_METRICS_CONTRACT.md": "contract\n"}, "analysis only")
    record = compare.code_alignment(analysis, base)
    assert record["aligned_by"] == "shared git object ids" and "benchmarks/_shared" not in record["shared"]
    assert "benchmarks/_shared/full_metrics.py" in record["shared"]
    metrics = commit({"benchmarks/_shared/full_metrics.py": "# changed\n"}, "full_metrics")
    with pytest.raises(ValueError, match="shared code: \['benchmarks/_shared/full_metrics.py'\]"):
        compare.code_alignment(metrics, analysis)
