from copy import deepcopy
import json

import pytest

from benchmarks.full_CPJUMP1.harness.cli import ARTIFACTS, STAGES
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_CPJUMP1.harness.pipeline import reasoning_record
from benchmarks.full_LIVECell.harness import compare as C

EXPECTED = "a" * 40
IMAGES = [f"img_{i:04d}" for i in range(C.N_IMAGES)]


def records():
    config = dict(seed=0, cores=8, budget_hours=11, train_cells_per_site=128, inference_chunk=2048,
                  images=["i1", "i2"], model="gpt-5.6-luna")
    return {a: {"config": {**deepcopy(config), "reasoning": reasoning_record(a)}, "delivery": {"prepare_sha256": "data"},
                "provenance": {"assets": {"sha": "assets"}, "gpu": "same"}} for a in C.ARMS}


def test_control_shape_and_codex_are_required_and_claude_and_biomni_optional():
    r = records()
    assert C.compatible(r)
    del r["codex"]
    with pytest.raises(ValueError, match="exactly"):
        C.compatible(r)
    r = records()
    r["claude"] = {**deepcopy(r["codex"]), "config": {**deepcopy(r["codex"]["config"]), "model": "claude-sonnet-5",
                                                       "reasoning": reasoning_record("claude")}}
    r["biomni"] = {**deepcopy(r["codex"]), "config": {**deepcopy(r["codex"]["config"]),
                                                       "reasoning": reasoning_record("biomni")}}
    assert C.compatible(r)
    r["claude"]["config"]["model"] = "gpt-5.6-luna"
    with pytest.raises(ValueError, match="claude-sonnet-5"):
        C.compatible(r)
    r = records()
    r["other"] = deepcopy(r["codex"])
    with pytest.raises(ValueError, match="exactly"):
        C.compatible(r)


@pytest.mark.parametrize("key,value", [("seed", 1), ("cores", 4), ("budget_hours", 4), ("train_cells_per_site", 10),
                                       ("inference_chunk", 128), ("images", ["i1"]), ("model", "different")])
def test_unmatched_protocols_refused(key, value):
    r = records()
    r["codex"]["config"][key] = value
    with pytest.raises(ValueError):
        C.compatible(r)


def test_gpu_difference_flagged_and_data_or_asset_mismatch_refused():
    r = records()
    r["codex"]["provenance"]["gpu"] = "different"
    assert C.compatible(r) is False
    r["shape"]["delivery"]["prepare_sha256"] = "other"
    with pytest.raises(ValueError, match="prepared data"):
        C.compatible(r)
    r = records()
    r["codex"]["provenance"]["assets"] = {"sha": "different"}
    with pytest.raises(ValueError, match="assets"):
        C.compatible(r)


@pytest.mark.parametrize("model", ["gpt-5.6", "claude-sonnet-5", None])
def test_equal_but_unrequested_models_refused(model):
    r = records()
    for arm in ("shape", "codex"):
        r[arm]["config"]["model"] = model
    with pytest.raises(ValueError, match="gpt-5.6-luna"):
        C.compatible(r)


def test_smoke_is_not_a_comparison_result(tmp_path):
    for name in ("DELIVERY", "RUN_META", "CONFIG", "PREFLIGHT"):
        write_json(tmp_path / f"{name}.json", {"status": "complete", "arm": "shape", "smoke": True})
    with pytest.raises(ValueError, match="smokes"):
        C.validate_run(tmp_path, "shape", EXPECTED)


def complete_run(root, arm="control", *, images=IMAGES, usage=None, branch="e2e"):
    """A complete full run record of `arm` (stage usage `usage`, default zero tokens) bound to its delivery."""
    config = {"arm": arm, "smoke": False, "images": images, "dataset": "LIVECell", "reasoning": reasoning_record(arm),
              **({} if arm == "control" else {"model": "gpt-5.6-luna"})}
    provenance = {"branch": branch, "commit": EXPECTED, **({} if arm == "control" else {"model": "gpt-5.6-luna"})}
    root.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACTS:
        if not (root / filename).exists():           # keep a record the test wrote (batch_SEARCH.json)
            (root / filename).write_text("synthetic")
    write_json(root / "CONFIG.json", config)
    write_json(root / "PREFLIGHT.json", provenance)
    write_json(root / "RUN_META.json", {"status": "complete", "arm": arm, "config": config, "provenance": provenance})
    for stage in STAGES:
        spent = (usage or {}).get(stage, {"total_tokens": 0})
        write_json(root / f"{stage}_USAGE.json", spent)
        write_json(root / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": 5,
                                                   "usage": spent})
    rebind(root, provenance, arm)
    (root / C.MARKER).write_text(EXPECTED)


def rebind(root, provenance=None, arm=None):
    old = read_json(root / "DELIVERY.json") if (root / "DELIVERY.json").is_file() else {}
    artifacts = {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.name not in ("DELIVERY.json", C.MARKER)}
    write_json(root / "DELIVERY.json", {"status": "complete", "arm": arm or old["arm"], "smoke": False,
                                       "provenance": provenance or old["provenance"], "artifacts": artifacts})


def test_validate_a_complete_control_and_detect_tampering(tmp_path):
    complete_run(tmp_path)
    record = C.validate_run(tmp_path, "control", EXPECTED)
    assert record["seconds"] == 25 and record["total_tokens"] == 0 and record["reasoning_tokens"] is None
    assert record["code"] == {"commit": EXPECTED, "aligned_by": "same commit"}
    (tmp_path / "embedding.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        C.validate_run(tmp_path, "control", EXPECTED)


def test_marker_branch_and_image_count_are_checked(tmp_path):
    complete_run(tmp_path / "marker")
    (tmp_path / "marker" / C.MARKER).unlink()
    (tmp_path / "marker" / "CPJUMP1_COMPLETE").write_text(EXPECTED)            # another benchmark's marker
    with pytest.raises(FileNotFoundError):
        C.validate_run(tmp_path / "marker", "control", EXPECTED)
    complete_run(tmp_path / "stale")
    (tmp_path / "stale" / C.MARKER).write_text("b" * 40)
    with pytest.raises(ValueError, match="completion marker"):
        C.validate_run(tmp_path / "stale", "control", EXPECTED)
    complete_run(tmp_path / "few", images=IMAGES[:1000])
    with pytest.raises(ValueError, match=f"{C.N_IMAGES} unique images"):
        C.validate_run(tmp_path / "few", "control", EXPECTED)
    complete_run(tmp_path / "dup", images=IMAGES[:-1] + IMAGES[:1])
    with pytest.raises(ValueError, match="unique images"):
        C.validate_run(tmp_path / "dup", "control", EXPECTED)
    complete_run(tmp_path / "branch")
    provenance = {**read_json(tmp_path / "branch" / "PREFLIGHT.json"), "branch": "e2e-livecell"}
    write_json(tmp_path / "branch" / "PREFLIGHT.json", provenance)
    with pytest.raises(ValueError, match="branch"):
        C.validate_run(tmp_path / "branch", "control", EXPECTED)


def test_the_expected_branch_is_a_setting_and_the_dataset_must_be_livecell(tmp_path):
    complete_run(tmp_path / "main", branch="main")
    assert C.validate_run(tmp_path / "main", "control", EXPECTED, "main")["seconds"] == 25
    with pytest.raises(ValueError, match="branch"):
        C.validate_run(tmp_path / "main", "control", EXPECTED)                  # default: e2e
    complete_run(tmp_path / "other")
    with pytest.raises(ValueError, match="branch"):
        C.validate_run(tmp_path / "other", "control", EXPECTED, "main")
    config = read_json(tmp_path / "other" / "CONFIG.json")
    config.pop("dataset")                                                       # a CPJUMP1 run's CONFIG.json
    write_json(tmp_path / "other" / "CONFIG.json", config)
    with pytest.raises(ValueError, match="not a LIVECell run"):
        C.validate_run(tmp_path / "other", "control", EXPECTED)


def test_reasoning_tokens_only_when_every_reasoning_stage_carries_them():
    full = {s: {"total_tokens": 10, "reasoning_tokens": 3} for s in STAGES}
    assert C.reasoning_tokens(full) == 3 * (len(STAGES) - 1)                 # cluster does not reason
    partial = {**full, "batch": {"total_tokens": 10}}
    assert C.reasoning_tokens(partial) is None
    assert C.reasoning_tokens({s: {"total_tokens": 0} for s in STAGES}) is None


def codex_run(root):
    """A complete Codex run: each reasoning stage one decision round (exec JSONL trace + final message)."""
    choice = dict(action="finish", request_json='{"all": "c0"}', reason="baseline sufficient")
    usage = {}
    for stage in STAGES[:-1]:
        agent = root / "agent" / stage
        selected = {"deliveries": {"i0": {"call": "c0", "source": "agent", "session": "group"}}, "total_tokens": 12,
                    "requested_model": "gpt-5.6-luna", "response_model_attested": False}
        write_json(agent / "SELECTED.json", selected)
        write_json(agent / "CALLS.json", {"i0": {"c0": {"status": "complete", "folder": "waves/w000/i0/c0",
                                                        "artifacts": {}}}})
        write_json(agent / "group" / "traces" / "round_000.json", choice)
        events = [{"type": "thread.started", "thread_id": stage}, {"type": "turn.started"},
                  {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(choice)}},
                  {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}]
        (agent / "group" / "traces" / "round_000.jsonl").write_text("\n".join(map(json.dumps, events)),
                                                                    encoding="utf-8")
        usage[stage] = selected
    complete_run(root, "codex", usage=usage)


def test_codex_traces_are_verified_and_bound(tmp_path):
    codex_run(tmp_path)
    record = C.validate_run(tmp_path, "codex", EXPECTED)
    assert record["total_tokens"] == 48 and record["reasoning_tokens"] is None
    delivery = read_json(tmp_path / "DELIVERY.json")
    del delivery["artifacts"]["agent/crop/group/traces/round_000.jsonl"]
    write_json(tmp_path / "DELIVERY.json", delivery)
    with pytest.raises(ValueError, match="bound to the delivery"):
        C.validate_run(tmp_path, "codex", EXPECTED)


CLEAN_SEARCH = {"reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [], "stop_reason": "satisfied"}


def test_shape_needs_every_search_record(tmp_path, monkeypatch):
    images = IMAGES[:3]
    monkeypatch.setattr(C, "N_IMAGES", 3)
    for name in ("representation_SEARCH.json", "batch_SEARCH.json"):
        write_json(tmp_path / name, CLEAN_SEARCH)
    for ident in images:
        write_json(tmp_path / "crop" / ident / "SEARCH.json", CLEAN_SEARCH)
        write_json(tmp_path / "segment" / ident / "SEARCH.json", CLEAN_SEARCH)
    usage = {s: {"total_tokens": 10, "reasoning_tokens": 2} for s in STAGES}
    complete_run(tmp_path, "shape", images=images, usage=usage)
    record = C.validate_run(tmp_path, "shape", EXPECTED)
    assert record["total_tokens"] == 50 and record["reasoning_tokens"] == 8
    (tmp_path / "crop" / images[2] / "SEARCH.json").unlink()
    rebind(tmp_path)
    with pytest.raises(ValueError, match="missing SHAPE search records"):
        C.validate_run(tmp_path, "shape", EXPECTED)


@pytest.fixture
def shared_repo(tmp_path, monkeypatch):
    """A git repository holding every SHARED and PROTOCOL path; commits: base, protocol (the LIVECell cli changed),
    score (the LIVECell scorer changed), spec (the dataset spec changed)."""
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    git("config", "core.autocrlf", "false")
    for path in (*C.SHARED, *C.PROTOCOL):
        name = path if path.endswith(".py") else f"{path}/module.py"
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(f"# {name}\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    commits = {"base": git("rev-parse", "HEAD")}
    for key, path in (("protocol", "benchmarks/full_LIVECell/harness/cli.py"),
                      ("score", "benchmarks/full_LIVECell/harness/score.py"),
                      ("spec", "benchmarks/full_LIVECell/harness/spec.py")):
        (repo / path).write_text(f"# {key} changed\n", encoding="utf-8")
        git("commit", "-q", "-am", key)
        commits[key] = git("rev-parse", "HEAD")
    monkeypatch.setattr(C, "REPO", repo)
    return commits


def test_code_alignment_reports_protocol_diffs_and_refuses_shared_ones(shared_repo):
    base = shared_repo["base"]
    assert C.code_alignment(base, base) == {"commit": base, "aligned_by": "same commit"}
    record = C.code_alignment(shared_repo["protocol"], base)
    assert record["aligned_by"] == "shared git object ids" and set(record["shared"]) == set(C.SHARED)
    assert "full_LIVECell/harness/cli.py" in record["protocol_diff_stat"]
    with pytest.raises(ValueError, match="shared code: \\['benchmarks/full_LIVECell/harness/score.py'\\]"):
        C.code_alignment(shared_repo["score"], shared_repo["protocol"])
    with pytest.raises(ValueError, match="full_LIVECell/harness/spec.py"):
        C.code_alignment(shared_repo["spec"], shared_repo["score"])
    with pytest.raises(ValueError, match="not in"):
        C.code_alignment("f" * 40, base)


def test_comparison_rescores_the_three_arms_with_stage_gt_next_to_the_verdicts(tmp_path, monkeypatch):
    import pandas as pd
    from . import test_score as T
    T.small(monkeypatch)
    prep = T.make_prep(tmp_path / "prep")
    runs = {"control": T.make_run(prep, tmp_path / "control"),
            "shape": T.make_run(prep, tmp_path / "shape", arm="shape"),
            "codex": T.make_run(prep, tmp_path / "codex", miss_every=4, arm="codex")}
    r = records()
    for arm, run in runs.items():
        write_json(run / "RESULT.json", {"fake_stale_score": 999})
        r[arm].update(seconds=20, total_tokens=0 if arm == "control" else 100,
                      reasoning_tokens=None if arm != "shape" else 40, code={"commit": EXPECTED})
    # Provenance validation has its own tests; the real scorer runs here.
    monkeypatch.setattr(C, "validate_run", lambda run, arm, expected, branch: r[arm])
    summary = C.compare(prep, runs, tmp_path / "comparison", EXPECTED)
    same = {m: "NO_RESOLVABLE_DIFFERENCE" for m in ("mAP", "macro_f1", "balanced_accuracy")}
    assert summary["verdicts"] == {"shape": same, "codex": same}
    assert summary["shape_vs_agents"] == {"codex": same}
    gt = summary["stage_gt_vs_control"]
    assert gt["shape"]["segmentation"]["all"]["AP50"]["vs_control"]["delta"] == 0
    codex = gt["codex"]["segmentation"]["all"]["AP50"]
    assert codex["vs_control"]["delta"] == pytest.approx(-0.5 * len(range(0, 288, 4)) / 288)
    assert codex["vs_control"]["se"] > 0
    assert summary["stage_gt_control"]["crop"]["all"]["gt_recall"]["value"] == 1
    assert "fake_stale_score" not in read_json(runs["codex"] / "RESULT.json")
    assert {arm: b["decision"] for arm, b in summary["batch_decisions"].items()} == {
        "control": "RUN", "shape": "ABSTAIN", "codex": "RUN"}
    assert summary["batch_decisions"]["codex"]["call"] == "input"
    assert summary["reasoning"]["stages"] == {arm: reasoning_record(arm)["stages"] for arm in C.ARMS}
    assert read_json(tmp_path / "comparison" / "COMPARISON.json")["reasoning"] == summary["reasoning"]
    table = pd.read_csv(tmp_path / "comparison" / "comparison.csv")
    assert list(dict.fromkeys(table.arm)) == ["control", "shape", "codex"]
    assert set(table.kind) == {"verdict", "retrieval", "stage_gt"}
    assert "total_tokens" in table.columns and set(table[table.arm == "codex"].total_tokens) == {100}
    assert set(table[table.arm == "shape"].reasoning_tokens) == {40} and table[table.arm == "codex"].reasoning_tokens.isna().all()
    verdict = table[(table.kind == "verdict") & (table.arm == "codex")]
    assert set(verdict.metric) == {"mAP", "macro_f1", "balanced_accuracy"} and set(verdict.verdict) == {
        "NO_RESOLVABLE_DIFFERENCE"}
    row = table[(table.arm == "codex") & (table.endpoint == "segmentation/all") & (table.metric == "AP50")].iloc[0]
    assert row.delta_vs_control == pytest.approx(codex["vs_control"]["delta"]) and row.n == 288
    assert len(table[table.arm == "control"]) == 3 + 4 * 3 + (4 + 5) * len(summary["stage_gt_control"]["crop"])
    assert (tmp_path / "comparison" / "COMPARISON_COMPLETE").read_text(encoding="ascii") == EXPECTED + "\n"
    with pytest.raises(FileExistsError):
        C.compare(prep, runs, tmp_path / "comparison", EXPECTED)


def test_five_arm_comparison_adds_claude_and_biomni_next_to_the_three(tmp_path, monkeypatch):
    from . import test_score as T
    T.small(monkeypatch)
    prep = T.make_prep(tmp_path / "prep")
    runs = {"control": T.make_run(prep, tmp_path / "control"),
            "shape": T.make_run(prep, tmp_path / "shape", arm="shape"),
            "codex": T.make_run(prep, tmp_path / "codex", miss_every=4, arm="codex"),
            "claude": T.make_run(prep, tmp_path / "claude", miss_every=3, arm="claude"),
            "biomni": T.make_run(prep, tmp_path / "biomni", miss_every=2, arm="biomni")}
    r = records()
    for arm in ("claude", "biomni"):
        r[arm] = deepcopy(r["codex"])
        r[arm]["config"]["reasoning"] = reasoning_record(arm)
    r["claude"]["config"]["model"] = "claude-sonnet-5"
    for arm in runs:
        r[arm].update(seconds=20, total_tokens=0 if arm == "control" else 100, reasoning_tokens=None,
                      code={"commit": EXPECTED})
    monkeypatch.setattr(C, "validate_run", lambda run, arm, expected, branch: r[arm])
    summary = C.compare(prep, runs, tmp_path / "comparison", EXPECTED)
    assert list(summary["verdicts"]) == ["shape", "codex", "claude", "biomni"]
    assert list(summary["shape_vs_agents"]) == ["codex", "claude", "biomni"]
    assert list(summary["batch_decisions"]) == ["control", "shape", "codex", "claude", "biomni"]
    assert list(summary["stage_gt_vs_control"]) == ["shape", "codex", "claude", "biomni"]
    assert "codex, claude, biomni" in " ".join(summary["notes"])
    import pandas as pd
    table = pd.read_csv(tmp_path / "comparison" / "comparison.csv")
    assert list(dict.fromkeys(table.arm)) == ["control", "shape", "codex", "claude", "biomni"]
    claude = summary["stage_gt_vs_control"]["claude"]["segmentation"]["all"]["AP50"]["vs_control"]["delta"]
    assert claude == pytest.approx(-0.5 * len(range(0, 288, 3)) / 288)
