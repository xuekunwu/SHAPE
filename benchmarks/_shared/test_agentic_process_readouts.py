"""agentic_process_readouts.py on a small synthetic experiment tree (one run dir per arm kind) and on the real curated
full_CPJUMP1 results folder when it is in the repository."""
import csv
import json
import shutil
from pathlib import Path

import pytest

from benchmarks._shared import agentic_process_readouts as apr

REAL = (Path(__file__).resolve().parents[1] / "full_CPJUMP1" / "results" / "final-5arm-18f7285f-20261012T165818Z")
STAGES = apr.STAGES
IMAGES = ("img_0", "img_1", "img_2")


def put(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def shape_usage(total, calls, observed=("total_tokens", "prompt_tokens", "completion_tokens", "cached_tokens",
                                        "reasoning_tokens")):
    u = {"total_tokens": total, "prompt_tokens": total * 3 // 4, "completion_tokens": total // 4,
         "cached_tokens": 5, "reasoning_tokens": total // 8, "calls": calls,
         "usage_observed_fields": list(observed), "usage_complete": True}
    return {k: v for k, v in u.items() if k not in ("reasoning_tokens", "cached_tokens") or k in observed}


def base_arm(root: Path, name: str, arm: str, commit="c1" * 20, seconds=100.0, usage=None, marker="CPJUMP1_COMPLETE",
             branch="e2e", reasoning=None):
    d = root / name
    put(d / "RUN_META.json", {"status": "complete", "arm": arm, "provenance": {"commit": commit, "branch": branch,
                                                                                "model": "m"}})
    cfg = {"arm": arm, "seed": 0}
    if reasoning:
        cfg["reasoning"] = {"stages": {s: {"effort": "high", "max_tokens": 100, "timeout_seconds": 9} for s in STAGES}}
    put(d / "CONFIG.json", cfg)
    put(d / marker, commit + "\n")
    for i, s in enumerate(STAGES):
        u = (usage or {}).get(s) or {"total_tokens": 0, "calls": 0, "usage_observed_fields": []}
        put(d / f"{s}_COMPLETE.json", {"status": "complete", "stage": s, "seconds": seconds + i, "usage": u})
        put(d / f"{s}_USAGE.json", u)
    return d


def search_rec(delivered, n_calls, reason, errors=(), failures=0, admission=None):
    rec = {"status": "searched", "delivered": delivered, "levers": {}, "n_calls": n_calls, "seconds": 2.0,
           "stop_reason": reason, "resolution": {"status": "x", "reason": reason, "fault": None},
           "reasoner_failures": failures, "fallback_rounds": 0, "execution_errors": list(errors)}
    if admission:
        rec["admission"] = admission
    return rec


def build_shape(root: Path, name="shape", commit="c1" * 20, curated=False, reasoning=False):
    usage = {s: shape_usage(1000 * (i + 1), 10 * (i + 1)) for i, s in enumerate(STAGES[:4])}
    usage["cluster"] = {"total_tokens": 0, "calls": 0, "usage_observed_fields": []}
    d = base_arm(root, name, "shape", commit=commit, usage=usage, reasoning=reasoning)
    err = {"status": "execution_error", "question": "repair",
           "cause": "[('a', 1)] from agent_3: contrast must differ from its anchor in one dimension"}
    adm = {"decision": "RUN", "concerns": {"dense": True, "dim": False}}
    seg = [search_rec("agent_1", 2, "satisfied", admission=adm), search_rec("reference", 1, "cohort_applied"),
           search_rec("default", 3, "no_valid_move", errors=[err], failures=1)]
    crop = [search_rec("agent_2", 4, "satisfied"), search_rec("default", 2, "invalid_search_plan", errors=[err],
                                                              failures=1), search_rec("agent_1", 3, "satisfied")]
    put(d / "segment_COHORT.json", {"parameter_source": dict(zip(IMAGES, ("per_image_vrr", "cohort", "calibration")))})
    put(d / "SEARCH_WARNINGS.jsonl", "\n".join(json.dumps({"stage": st, "image_id": i, "reasoner_failures": 1,
                                                           "execution_errors": [err], "stop_reason": "x"})
                                               for st, i in (("segment", "img_2"), ("crop", "img_1"))) + "\n")
    if curated:
        summary = {"stages": {}}
        for st, recs in (("segment", seg), ("crop", crop)):
            s = apr.summarise_records(recs)
            summary["stages"][st] = {
                "images": 3, "record_source": {"SEARCH.json": 3},
                "delivered_kind": dict(s["delivered_kind"]), "resolution_reason": dict(s["resolution_reason"]),
                "stop_reason": dict(s["stop_reason"]),
                "calls_per_image": {"distribution": {str(k): v for k, v in s["calls"].items()}},
                "reasoner_failures": s["reasoner_failures"], "fallback_rounds": s["fallback_rounds"],
                "execution_errors": {"total": s["execution_errors"]["total"], "images": s["execution_errors"]["images"],
                                     "by_question": dict(s["execution_errors"]["by_question"])},
                "seconds": {"total": s["seconds_total"]}}
        put(d / "search_summary.json", summary)
    else:
        for st, recs in (("segment", seg), ("crop", crop)):
            for img, rec in zip(IMAGES, recs):
                put(d / st / img / ("COHORT_SEARCH.json" if rec["delivered"] == "reference" else "SEARCH.json"), rec)
    put(d / "representation_SEARCH.json", {
        "admission": {"decision": "RUN", "tier": "t", "reasons": ["few_wells"]}, "stop_reason": "no_valid_move",
        "reasoner_failures": 1, "fallback_rounds": 0, "execution_errors": [err],
        "resolution": {"status": "conceded", "reason": "no_admissible_call", "fault": "ceiling"},
        "details": {"entries": [None, {"score_gain_vs_initial": 0.01, "failed": False},
                                {"score_gain_vs_initial": 0.03, "failed": False},
                                {"score_gain_vs_initial": 0.02, "failed": False}],
                    "rounds": [["agent_1"], ["agent_2", "agent_3"]], "plans": [1, 2, 3], "expert_config": [{"a": 1}],
                    "expert_rejections": ["bad config"], "admissible": ["agent_2", "agent_3"], "failed": [],
                    "stop_refusals": [], "forced_continuations": [], "call": "input", "choice": "no_admissible_call"}})
    put(d / "batch_SEARCH.json", {
        "admission": {"decision": "RUN", "rule": "RUN", "concerns": {"x": True, "y": False}, "veto": None,
                      "reasoner": {"decision": "RUN"}},
        "stop_reason": "satisfied", "reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [],
        "resolution": {"status": "satisfied", "reason": "satisfied", "fault": None},
        "details": {"entries": [None, {"score_gain_vs_initial": 0.1, "failed": False},
                                {"score_gain_vs_initial": 0.2, "failed": False}],
                    "rounds": [["default", "agent_1"]], "plans": [1], "admissible": ["default", "agent_1"],
                    "failed": [], "call": "agent_1", "choice": "agent_choice"}})
    return d


def rounds_file(arm_dir: Path, stage: str, rounds: list, deliveries: dict, n_calls: int, calls=None):
    base = arm_dir / "agent" / stage
    put(base / "ROUNDS.json", rounds)
    total = sum(r.get("total_tokens") or 0 for r in rounds)
    sel = {"deliveries": deliveries, "total_tokens": total, "n_calls": n_calls, "n_rounds": len(rounds)}
    put(base / "SELECTED.json", sel)
    if calls is not None:
        put(base / "CALLS.json", calls)
    return sel


def codex_round(session, rnd, action, outcome, message="", tokens=100, request=None):
    return {"session": session, "round": rnd, "action": action, "outcome": outcome, "message": message,
            "request_json": request, "total_tokens": tokens,
            "usage": {"input_tokens": tokens - 10, "cached_input_tokens": 3, "cache_write_input_tokens": 1,
                      "output_tokens": 10, "reasoning_output_tokens": 4}}


def build_codex(root: Path, name="codex"):
    d = base_arm(root, name, "codex")
    dgroup = {i: {"call": "c1", "source": "agent", "session": "group"} for i in IMAGES}
    cell_rules = json.dumps([{"images": "all", "params": {}, "cell_rules": [{"when": [], "apply": []}]}])
    rounds_file(d, "segment", [codex_round("group", 0, "inspect", "attached"),
                               codex_round("group", 1, "run", "executed"),
                               codex_round("group", 2, "finish", "finished")], dgroup, 6,
                {i: {"c0": {"wave": "w000", "status": "complete"}, "c1": {"wave": "w001", "status": "complete"}}
                 for i in IMAGES})
    rounds_file(d, "crop", [codex_round("group", 0, "run", "rejected", "a call spec is an object with keys images",
                                        request=cell_rules),
                            codex_round("group", 1, "run", "rejected", "a call spec is an object with keys images",
                                        request=cell_rules),
                            codex_round("group", 2, "run", "executed"),
                            codex_round("group", 3, "finish", "finished")], dgroup, 6)
    ddata = {"dataset": {"call": "c2", "source": "agent", "session": "dataset"}}

    def view(vals):
        return {"val_history": [{"epoch": i + 1, "sibling_site_map": v} for i, v in enumerate(vals)]}
    calls = {"dataset": {"input": {"status": "complete", "wave": None, "view": view([0.2])},
                         "c1": {"status": "complete", "wave": "d001", "view": view([0.21, 0.25])},
                         "c2": {"status": "complete", "wave": "d002", "view": view([0.3])},
                         "c3": {"status": "complete", "wave": "d003", "view": view([0.28])}}}
    rounds_file(d, "embed", [codex_round("dataset", i, "run", "executed") for i in range(3)]
                + [codex_round("dataset", 3, "finish", "finished")], ddata, 4, calls)
    bcalls = {"dataset": {"input": {"status": "complete", "wave": None}, "c1": {"status": "complete", "wave": "d001"}}}
    for label, score in (("input", 0.4), ("c1", 0.5)):
        put(d / "agent" / "batch" / "calls" / label / "FEEDBACK.json", {"calibration": {"overall": score}})
    rounds_file(d, "batch", [codex_round("dataset", 0, "run", "executed"), codex_round("dataset", 1, "finish", "finished")],
                {"dataset": {"call": "c1", "source": "agent", "session": "dataset"}}, 2, bcalls)
    return d


def claude_round(session, rnd, action, outcome, tokens=0):
    return {"session": session, "round": rnd, "action": action, "outcome": outcome, "message": "", "request_json": None,
            "total_tokens": 2 + 30 + 4 + 10, "num_turns": 3,
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": 30, "cache_read_input_tokens": 4,
                      "output_tokens": 10},
            "model_usage": {"claude-x": {"thinkingTokens": 7}}}


def build_claude(root: Path, name="claude"):
    d = base_arm(root, name, "claude", commit="d9" * 20)
    deliveries = {i: {"call": "c1", "source": "agent", "session": "group"} for i in IMAGES}
    rounds_file(d, "segment", [claude_round("group", 0, "run", "executed"), claude_round("group", 1, "finish", "finished")],
                deliveries, 6)
    return d


def build_biomni(root: Path, name="biomni"):
    d = base_arm(root, name, "biomni", commit="d9" * 20)
    err = "MalformedDecision: Biomni reply is not a decision"
    rounds_file(d, "embed", [{"session": "dataset", "round": 0, "action": None, "outcome": "failed_round",
                              "message": err, "error": err, "total_tokens": 77}],
                {"dataset": {"call": "input", "source": "default_session_unfinished", "session": "dataset",
                             "reason": "round 1 failed: " + err}}, 1)
    return d


def build_rerun(root: Path, name="shape_rerun_x"):
    d = build_shape(root, name, commit="b2" * 20, reasoning=True)
    (d / "CPJUMP1_COMPLETE").unlink()
    put(d / "RERUN_COMPLETE", "b2" * 20 + "\n")
    put(d / "RERUN.json", {"source_run": "/vm/runs/exp/shape", "reused_stages": ["segment", "crop"]})
    return d


def build_failed(root: Path, name="shape_failed_gateway_x"):
    d = root / name
    put(d / "RUN_META.json", {"status": "failed", "arm": "shape", "current_stage": "segment",
                              "error": "RuntimeError: stage failed (exit 1); see logs/segment.log",
                              "provenance": {"commit": "c1" * 20}})
    put(d / "CONFIG.json", {"arm": "shape"})
    put(d / "logs" / "segment.log", "Traceback (most recent call last):\n  File x\nopenai.APIConnectionError: "
                                    "Connection error.\n")
    return d


@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "exp"
    base_arm(root, "control", "control")
    build_shape(root)
    build_codex(root)
    build_claude(root)
    build_biomni(root)
    build_rerun(root)
    build_failed(root)
    put(root / "compare" / "x.json", {})
    put(root / "analysis" / "old.json", {})
    put(root / "logs" / "a.log", "x")
    return root


@pytest.fixture()
def result(tree):
    return apr.build(tree)


def stage(result, run, name):
    return result["arms"][run]["stages"][name]


def null_leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("unavailable", "reused_from"):       # reused_from: null = computed in this run
                continue
            yield from null_leaves(v, f"{prefix}.{k}" if prefix else k)
    elif node is None:
        yield prefix


def test_arms_discovered_and_non_arm_dirs_skipped(result):
    assert sorted(result["arms"]) == ["biomni", "claude", "codex", "control", "shape", "shape_failed_gateway_x",
                                      "shape_rerun_x"]
    assert result["schema"] == apr.SCHEMA and result["stages"] == list(STAGES)


def test_every_null_has_a_reason_and_no_zero_is_invented(result):
    for run, arm in result["arms"].items():
        for name, st in list(arm["stages"].items()) + [("total", arm["totals"])]:
            for leaf in null_leaves(st):
                covered = [p for p in st["unavailable"] if leaf == p or leaf.startswith(p + ".")]
                assert covered, f"{run}/{name}: null {leaf} has no reason"
                assert all(st["unavailable"][p] for p in covered)


def test_control_is_not_applicable_never_zero(result):
    st = stage(result, "control", "segment")
    assert st["state"] == "complete"
    assert st["E1"]["total"] is None and st["unavailable"]["E1.total"] == "not applicable: control"
    assert st["P1"] is None and st["unavailable"]["P1"] == "not applicable: control"
    assert st["E2"]["seconds"] == 100.0
    assert result["arms"]["control"]["totals"]["E1"]["total"]["own"] is None
    assert result["arms"]["control"]["end_state"]["run"] == "delivered"


def test_shape_per_image_records(result):
    seg = stage(result, "shape", "segment")
    assert seg["P1"]["delivered_by"] == {"units": 3, "by_kind": {"agent": 1, "cohort_reference": 1, "default": 1}}
    assert seg["P1"]["record_source"] == "per_image_records"
    assert seg["P1"]["non_agent_stop_reasons"] == {"cohort_applied": 1, "no_valid_move": 1}
    assert seg["P2"]["execution_errors"] == {"total": 1, "images": 1, "by_question": {"repair": 1}}
    assert seg["P2"]["reasoner_failures"] == {"total": 1, "images": 1}
    assert seg["P2"]["execution_error_causes"] == [{"count": 1, "text": apr._normalise_cause(
        "[('a', 1)] from agent_3: contrast must differ from its anchor in one dimension")}]
    assert seg["P3"]["calls_per_image"] == {"max": 3, "mean": 2.0, "median": 2, "p90": 3, "total": 6}
    assert seg["P3"]["stop_reason"] == {"cohort_applied": 1, "no_valid_move": 1, "satisfied": 1}
    assert seg["P4"]["admission_decisions"] == {"RUN": 1}
    assert seg["P4"]["concerns_held"] == {"dense": 1}
    assert seg["P4"]["parameter_source"] == {"calibration": 1, "cohort": 1, "per_image_vrr": 1}
    assert seg["P4"]["contrast_rejections"] == 1
    assert seg["E3"]["tool_calls"] == 6 and seg["E3"]["llm_calls"] == 10
    crop = stage(result, "shape", "crop")
    assert crop["P3"]["stop_reason"]["invalid_search_plan"] == 1
    assert crop["P4"]["parameter_source"] is None
    assert "only the segmentation cohort" in crop["unavailable"]["P4.parameter_source"]


def test_shape_curated_summary_matches_per_image_records(tmp_path, result):
    root = tmp_path / "curated"
    build_shape(root, curated=True)
    cur = apr.build(root)
    for name in ("segment", "crop"):
        a, b = stage(result, "shape", name), stage(cur, "shape", name)
        assert b["P1"]["record_source"] == "search_summary.json"
        assert a["P1"]["delivered_by"] == b["P1"]["delivered_by"]
        assert a["P2"]["execution_errors"] == b["P2"]["execution_errors"]
        assert a["P2"]["execution_error_causes"] == b["P2"]["execution_error_causes"]
        assert a["P3"]["calls_per_image"] == b["P3"]["calls_per_image"]
        assert a["P3"]["stop_reason"] == b["P3"]["stop_reason"]
    seg = stage(cur, "shape", "segment")
    assert seg["P4"]["admission_decisions"] is None and "needs per-image records" in seg["unavailable"]["P4.admission_decisions"]
    assert seg["P4"]["parameter_source"] == {"calibration": 1, "cohort": 1, "per_image_vrr": 1}
    assert cur["arms"]["shape"]["record_source"]["segment"] == "search_summary.json"


def test_shape_embed_loop_and_rules(result):
    emb = stage(result, "shape", "embed")
    assert emb["P1"]["delivered_by"]["by_kind"] == {"default": 1}
    assert emb["P1"]["fallback"] == {"choice": "no_admissible_call", "resolution_reason": "no_admissible_call",
                                     "resolution_status": "conceded", "stop_reason": "no_valid_move"}
    loop = emb["P3"]["loop"]
    assert (loop["tuning_calls"], loop["best_call"], loop["first_call_is_best"], loop["calls_after_best"],
            loop["delivered_is_best"]) == (3, "agent_2", False, 1, False)
    assert loop["expert_start_calls"] == 1
    assert emb["P4"]["guard"] == {"admissible": 2, "calls": 3, "failed": 0}
    assert emb["P4"]["expert_rejections"] == 1 and emb["P4"]["stop_refusals"] == 0
    assert emb["P4"]["admission_decisions"] == {"RUN": 1} and emb["P4"]["concerns_held"] == {"few_wells": 1}
    assert emb["P4"]["contrast_rejections"] == 1
    assert emb["P2"]["rejected"] is None and "not applicable" in emb["unavailable"]["P2.rejected"]
    assert emb["E3"]["tool_calls"] == 3
    batch = stage(result, "shape", "batch")
    assert batch["P1"]["delivered_by"]["by_kind"] == {"agent": 1} and batch["P1"]["fallback"] is None
    assert batch["P3"]["loop"]["first_call"] == "agent_1" and batch["P3"]["loop"]["tuning_calls"] == 1
    assert batch["P4"]["concerns_held"] == {"x": 1} and batch["P4"]["admission_reasoner_decision"] == "RUN"


def test_shape_tokens_follow_observed_fields(tmp_path):
    root = tmp_path / "exp"
    usage = {s: shape_usage(800, 3, observed=("total_tokens", "prompt_tokens", "completion_tokens")) for s in STAGES[:4]}
    base_arm(root, "shape", "shape", usage=usage)
    res = apr.build(root)
    e1 = res["arms"]["shape"]["stages"]["segment"]["E1"]
    assert e1["total"] == 800 and e1["reasoning"] is None and e1["cached"] is None
    assert "usage_observed_fields" in res["arms"]["shape"]["stages"]["segment"]["unavailable"]["E1.reasoning"]


def test_codex_rounds_and_rejections(result):
    crop = stage(result, "codex", "crop")
    p2 = crop["P2"]
    assert p2["rejected"] == 2 and p2["malformed"] == 0 and p2["rejected_rate"] == 0.5
    assert p2["first_rejection_reasons"] == [{"count": 2, "text": "a call spec is an object with keys images"}]
    assert p2["rejected_request_keys"]["cell_rules"] == 2
    assert crop["P1"]["delivered_by"] == {"units": 3, "by_kind": {"agent": 3}, "sessions_by_kind": {"agent": 1}}
    p3 = crop["P3"]
    assert p3["rounds_with_calls"] == 1 and p3["rounds_without_calls"]["total"] == 3
    assert p3["stop_reason"] == {"finished": 1} and p3["tool_call_round_allowance"] == 8
    assert crop["P4"] is None
    assert crop["E1"]["total"] == 400 and crop["E1"]["input"] == 360 and crop["E1"]["cached"] == 12
    assert crop["E1"]["reasoning"] == 16 and crop["E1"]["check"]["consistent"] is True
    assert crop["E3"]["tool_calls"] == 6 and crop["E3"]["tuning_calls"] == 3 and crop["E3"]["llm_calls"] is None
    seg = stage(result, "codex", "segment")
    assert seg["P3"]["rounds_per_image"] == {"max": 2, "median": 2, "p90": 2}
    assert crop["E3"]["tuning_calls_per_image"]["median"] is None


def test_codex_loops_score_the_calls(result):
    emb = stage(result, "codex", "embed")["P3"]
    loop = emb["loop"]
    assert emb["tool_call_rounds_used"] == 3 and loop["tuning_calls"] == 3
    assert (loop["best_call"], loop["first_call_is_best"], loop["calls_after_best"], loop["delivered_is_best"]) == \
        ("c2", False, 1, True)
    assert loop["input_score"] == 0.2 and loop["score_basis"].startswith("best validation sibling_site_map")
    bat = stage(result, "codex", "batch")["P3"]["loop"]
    assert bat["best_call"] == "c1" and bat["first_call_is_best"] is True and bat["calls_after_best"] == 0


def test_claude_tokens_sum_cache_and_thinking(result):
    e1 = stage(result, "claude", "segment")["E1"]
    assert (e1["total"], e1["input"], e1["output"], e1["cached"], e1["reasoning"]) == (92, 72, 20, 8, 14)
    assert stage(result, "claude", "segment")["E3"]["agent_turns"] == 6


def test_biomni_failed_round_is_a_fallback_delivery(result):
    emb = stage(result, "biomni", "embed")
    assert emb["P1"]["delivered_by"]["by_kind"] == {"default_session_unfinished": 1}
    assert emb["P1"]["agent_delivered_all"] is False
    assert "MalformedDecision" in emb["P1"]["fallback_reasons"]["default_session_unfinished"][0]["text"]
    assert emb["P2"]["failed_round"] == 1 and emb["P2"]["malformed"] == 0 and emb["P2"]["malformed_total"] == 1
    assert emb["P3"]["stop_reason"] == {"failed_round": 1}
    assert emb["E1"]["total"] == 77 and emb["E1"]["input"] is None
    assert emb["unavailable"]["E1.input"]


def test_rerun_reused_stages_excluded_from_own_totals(result):
    arm = result["arms"]["shape_rerun_x"]
    assert arm["provenance"]["reused_stages"] == {"crop": "shape", "segment": "shape"}
    assert stage(result, "shape_rerun_x", "segment")["reused_from"] == "shape"
    assert stage(result, "shape_rerun_x", "embed")["reused_from"] is None
    t = arm["totals"]["E1"]["total"]
    assert t["including_reused"] - t["own"] == 1000 + 2000
    s = arm["totals"]["E2"]["seconds"]
    assert s["own"] == (102 + 103 + 104) and s["including_reused"] == 100 + 101 + 102 + 103 + 104
    assert stage(result, "shape_rerun_x", "segment")["E2"]["share_of_own"] is None
    assert result["tables"]["E1_total"]["cells"]["shape_rerun_x"]["total"] == t["own"]
    assert result["tables"]["E1_total"]["reused"]["shape_rerun_x"] == ["crop", "segment"]
    assert arm["provenance"]["reasoning"]["embed"] == "high/100/9"
    assert result["arms"]["codex"]["provenance"]["reasoning"]["embed"] is None


def test_end_states(result):
    assert result["arms"]["shape_failed_gateway_x"]["end_state"]["run"] == "failed"
    failed = result["arms"]["shape_failed_gateway_x"]["end_state"]
    assert failed["failed_stage"] == "segment" and failed["failure_class"] == "gateway_connection"
    assert failed["error_detail"] == "openai.APIConnectionError: Connection error."
    assert stage(result, "shape_failed_gateway_x", "segment")["state"] == "failed"
    assert stage(result, "shape_failed_gateway_x", "crop")["state"] == "not_run"
    assert result["arms"]["shape_rerun_x"]["end_state"]["run"] == "delivered"
    assert any("different commits" in n for n in result["notes"])


def test_timeout_is_classified(tmp_path):
    root = tmp_path / "exp"
    d = build_failed(root, "codex_failed")
    put(d / "RUN_META.json", {"status": "failed", "arm": "codex", "current_stage": "embed",
                              "error": "TimeoutError: arm budget exhausted"})
    assert apr.build(root)["arms"]["codex_failed"]["end_state"]["run"] == "timed_out"


def test_percentiles():
    assert apr._p90(list(range(1, 11))) == 9 and apr._median([1, 2, 3, 4]) == 2.5 and apr._median([1, 2, 3]) == 2
    assert apr._median([]) is None and apr._p90([]) is None


def test_deterministic_and_path_independent(tmp_path, tree):
    a = json.dumps(apr.build(tree), sort_keys=True)
    assert a == json.dumps(apr.build(tree), sort_keys=True)
    moved = tmp_path / "other_name" / "exp"
    shutil.copytree(tree, moved)
    b = json.loads(json.dumps(apr.build(moved), sort_keys=True))
    first = json.loads(a)
    assert b["arms"] == first["arms"] and b["tables"] == first["tables"]
    assert str(tmp_path) not in a


def test_cli_writes_json_csv_md_with_lf(tmp_path, tree):
    out = tmp_path / "o" / "p.json"
    assert apr.main(["--run", str(tree), "--out", str(out), "--csv", str(tmp_path / "o" / "p.csv"), "--md"]) == 0
    for path in (out, out.with_suffix(".md"), tmp_path / "o" / "p.csv"):
        assert b"\r" not in path.read_bytes()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["notes"] and "## E1_total" in out.with_suffix(".md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((tmp_path / "o" / "p.csv").open(encoding="utf-8")))
    assert set(rows[0]) == {"run", "arm", "stage", "metric", "value", "reason"}
    nulls = [r for r in rows if r["value"] == "" and r["stage"] == "segment" and r["arm"] == "control"]
    assert nulls and all(r["reason"] for r in nulls)
    assert apr.main(["--run", str(tmp_path / "missing"), "--out", str(out)]) == 2


def test_wrapper_default_out(tmp_path, tree):
    assert apr.main(["--run", str(tree)], default_out="process_readouts.json") == 0
    assert (tree / "analysis" / "process_readouts.json").is_file()


# ---------------------------------------------------------------------------------------------------------------- #
# the real curated results folder
# ---------------------------------------------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real():
    if not REAL.is_dir():
        pytest.skip("curated full_CPJUMP1 results folder not in the repository")
    return apr.build(REAL)


def test_real_arms_and_end_states(real):
    assert sorted(real["arms"]) == ["biomni", "claude", "codex", "control", "shape", "shape_failed_gateway_20261012T222009Z",
                                    "shape_rerun_e2e_21f63d2b", "shape_rerun_main_2f34db04"]
    states = {n: a["end_state"]["run"] for n, a in real["arms"].items()}
    assert states.pop("shape_failed_gateway_20261012T222009Z") == "failed" and set(states.values()) == {"delivered"}
    assert real["arms"]["shape_failed_gateway_20261012T222009Z"]["end_state"]["failure_class"] == "gateway_connection"
    assert "/home/" not in json.dumps(real)                    # machine paths are reduced to their last component


def test_sanitized_failure_detail_replaces_ignored_raw_log(tmp_path, tree):
    failed = tree / "shape_failed_gateway_x"
    shutil.rmtree(failed / "logs")
    (failed / "FAILURE_DETAIL.txt").write_text(
        "stage: segment\nopenai.APIConnectionError: Connection error.\n", encoding="utf-8"
    )
    end = apr.build(tree)["arms"]["shape_failed_gateway_x"]["end_state"]
    assert end["error_detail"] == "openai.APIConnectionError: Connection error."
    assert end["failure_class"] == "gateway_connection"


def test_real_biomni_embed_was_delivered_by_a_fallback(real):
    emb = stage(real, "biomni", "embed")
    assert emb["P1"]["delivered_by"]["by_kind"] == {"default_session_unfinished": 1}
    assert emb["P2"]["failed_round"] == 1 and emb["P2"]["malformed_total"] == 1
    for run in ("codex", "claude"):
        assert stage(real, run, "embed")["P1"]["delivered_by"]["by_kind"] == {"agent": 1}


def test_real_codex_crop_cell_rules_rejections(real):
    p2 = stage(real, "codex", "crop")["P2"]
    assert p2["rejected"] == 3 and p2["rejected_request_keys"]["cell_rules"] == 3
    assert p2["first_rejection_reasons"][0] == {"count": 2, "text": "a call spec is an object with keys images, params, per_image"}
    assert stage(real, "biomni", "crop")["P2"]["rejected"] == 10


def test_real_shape_first_run_embed_delivered_the_frozen_backbone(real):
    emb = stage(real, "shape", "embed")
    assert emb["P1"]["delivered_by"]["by_kind"] == {"default": 1}
    assert emb["P1"]["fallback"]["stop_reason"] == "no_valid_move"
    assert emb["P4"]["guard"]["admissible"] == 0 and emb["P4"]["guard"]["calls"] == 4
    assert stage(real, "shape_rerun_e2e_21f63d2b", "embed")["P3"]["loop"]["expert_start_calls"] == 2
    assert stage(real, "shape_rerun_main_2f34db04", "embed")["P3"]["loop"]["expert_start_calls"] == 1
    crop = stage(real, "shape", "crop")
    assert crop["P2"]["execution_errors"]["total"] == 167 and crop["P1"]["delivered_by"]["by_kind"]["agent"] == 960


def test_real_tokens_and_seconds_agree_with_summary_csv(real):
    rows = list(csv.DictReader((REAL / "summary.csv").open(encoding="utf-8")))
    checked = 0
    for row in rows:
        if row["status"] == "failed" or row["run"] == "control":
            continue
        for s in STAGES[:4]:
            st = stage(real, row["run"], s)
            assert st["E1"]["total"] == int(row[f"{s}_total_tokens"]), (row["run"], s)
            assert st["E2"]["seconds"] == pytest.approx(float(row[f"{s}_seconds"]))
            checked += 1
    assert checked == 6 * 4
    own = real["arms"]["shape_rerun_e2e_21f63d2b"]["totals"]["E1"]["total"]
    assert own["own"] == 87245 + 331111 and own["including_reused"] - own["own"] == 16236572 + 65521626


def test_real_agent_token_rounds_match_the_reported_totals(real):
    for run in ("codex", "claude", "biomni"):
        for s in STAGES[:4]:
            assert stage(real, run, s)["E1"]["check"]["consistent"] is True


def test_real_nulls_have_reasons(real):
    for arm in real["arms"].values():
        for st in list(arm["stages"].values()) + [arm["totals"]]:
            for leaf in null_leaves(st):
                assert any(leaf == p or leaf.startswith(p + ".") for p in st["unavailable"]), leaf


def test_rec_put_replaces_a_leaf_that_a_later_put_nests_under():
    rec = apr.Rec()
    rec.put("P3.loop", None, "entries and rounds of the record do not pair up")
    rec.put("P3.loop.expert_start_calls", 2, "none")
    assert rec.data == {"P3": {"loop": {"expert_start_calls": 2}}} and "P3.loop" not in rec.why
    rec.put("A.b", 1)
    rec.put("A.b.c", None, "why not")
    assert rec.data["A"] == {"b": {"c": None}} and rec.why == {"A.b.c": "why not"}


def test_livecell_shape_embed_record_that_does_not_pair_up_with_its_rounds(tmp_path):
    """Real LIVECell SHAPE arm: the embed search entries do not pair up with the rounds (P3.loop = None), and the
    record has an expert_config list (P3.loop.expert_start_calls nests under that None); control + shape, no codex."""
    from benchmarks.full_LIVECell.runners import process_readouts as livecell
    root = tmp_path / "exp"
    base_arm(root, "control", "control", marker="LIVECELL_COMPLETE")
    shape = build_shape(root)
    (shape / "CPJUMP1_COMPLETE").rename(shape / "LIVECELL_COMPLETE")
    path = shape / "representation_SEARCH.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["details"]["entries"] = record["details"]["entries"][:2]         # 1 entry, 3 round names: not paired
    record["details"]["expert_config"] = [{"a": 1}, {"a": 2}]
    put(path, record)
    assert livecell.main(["--run", str(root)]) == 0
    data = json.loads((root / "analysis" / "process_readouts.json").read_text(encoding="utf-8"))
    assert sorted(data["arms"]) == ["control", "shape"]
    embed = data["arms"]["shape"]["stages"]["embed"]
    assert embed["P3"]["loop"] == {"expert_start_calls": 2}
    assert "P3.loop" not in embed["unavailable"]
    assert data["arms"]["shape"]["stages"]["batch"]["P3"]["loop"]["tuning_calls"] == 1
