"""Process metrics (harness/process.py: D1, D5, D6, E1, F3) on synthetic agent and control runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import process
from benchmarks.full_HCMA.harness.cli import main

from benchmarks.full_HCMA.test._synthetic_chain import Chain, build_private, digest, informative, workspace_data

NOTES = {"S002": "per-image diameter 45 px on the dense fields", "S003": "rescue the low-count fields",
         "C001": "", "E002": "zero-shot on the crops of C001", "K001": "resolution 1.0 gives 12 states"}
TIMES = {"S001": (0, 10), "S002": (20, 120), "S003": (100, 150), "C001": (160, 200), "C002": (160, 170),
         "E001": (210, 230), "E002": (240, 600), "K001": (610, 640), "S004": (650, 660)}


def _rewrite_ledger(run: Path, summaries: dict, params: dict) -> None:
    """Times, notes, params and summaries on the synthetic ledger lines (the chain only checks files)."""
    path = run / "logs" / "tool_calls.jsonl"
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(raw)
        call, (start, end) = entry["call"], TIMES[entry["call"]]
        if entry["event"] == "start":
            entry.update(time=float(start), note=NOTES.get(call, "probe"), params=params.get(call, {}))
        else:            # the call dirs' tables were edited after the call: re-record their hashes
            directory = next((run / "workspace" / "calls").glob(f"{call}_*"))
            entry.update(time=float(end), seconds=float(end - start), summary=summaries.get(call, {}),
                         outputs={rel: digest(directory / rel) for rel in entry["outputs"]})
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def agent_run(tmp_path):
    root, run = tmp_path / "root", tmp_path / "runs" / "claude-x"
    labels = build_private(root)
    images = labels["image_id"].tolist()                     # 54 images, 2 cells each
    workspace_data(run, root)
    chain = Chain(run)
    s1 = chain.segment(images[:3], name="probe", set_levers=["diameter"])
    s2 = chain.segment(images, name="main", set_levers=["diameter", "cellprob_threshold"])
    rescue = images[-5:]
    s3 = chain.segment(rescue, name="rescue", set_levers=["augment"])
    c1 = chain.crop(s2, dict.fromkeys(images, 2), name="crops")
    c2 = chain.crop(s2, {images[0]: 2}, name="probe", set_levers=["margin_mode"])
    calls = run / "workspace" / "calls"
    # the rescued images' masks come from S003 (crop per_image mask_call), as with --masks-map
    table = pd.read_csv(calls / f"{c1}_crops" / "per_image.csv", dtype=str)
    table.loc[table["image_id"].isin(rescue), "mask_call"] = s3
    table.to_csv(calls / f"{c1}_crops" / "per_image.csv", index=False)
    e1 = chain.embed(c2, lambda cells: informative(labels, cells), name="probe", set_levers=["learning_rate"])
    e2 = chain.embed(c1, lambda cells: informative(labels, cells), name="main", set_levers=["adaptation_depth"])
    k1 = chain.cluster(e2, name="states", resolutions=(0.5,), set_levers=["resolution"])
    s4 = chain.segment(images[:1], name="broken", status="failed")
    chain.choose({"cluster_call": k1, "resolution": 0.5, "skipped": {"batch_correct": "not needed"}})
    assert (s1, s2, s3, c1, c2, e1, e2, k1, s4) == ("S001", "S002", "S003", "C001", "C002", "E001", "E002",
                                                    "K001", "S004")
    # per-image diameters for 10 main-call images (one equal to the call's global value), resolved tables
    per_image = {i: {"diameter": 45.0} for i in images[:9]} | {images[9]: {"diameter": 30.0}}
    (calls / "S002_main" / "inputs").mkdir()
    (calls / "S002_main" / "inputs" / "per_image.json").write_text(json.dumps(per_image))
    pd.DataFrame({"image_id": images, "status": "done",
                  "diameter": [per_image.get(i, {}).get("diameter", 30.0) for i in images]}) \
        .to_csv(calls / "S002_main" / "per_image.csv", index=False)
    pd.DataFrame({"image_id": rescue, "status": "done", "diameter": 60.0}).to_csv(calls / "S003_rescue" / "per_image.csv",
                                                                                     index=False)
    summaries = {"S001": {"n_images": 3}, "S002": {"n_images": 54}, "S003": {"n_images": 5}, "C001": {"n_images": 54},
                 "C002": {"n_images": 1}, "E001": {"n_images": 1, "n_cells": 2}, "E002": {"n_images": 54, "n_cells": 108},
                 "K001": {"n_images": 54, "n_cells": 108}}
    params = {"S001": {"diameter": 30.0}, "S002": {"diameter": 30.0, "cellprob_threshold": -0.5},
              "S003": {"diameter": 60.0, "augment": True}}
    _rewrite_ledger(run, summaries, params)
    with open(run / "traces" / "round_00.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "result", "num_turns": 7, "total_cost_usd": 1.5, "duration_api_ms": 1_800_000,
                                 "duration_ms": 3_600_000, "usage": {
                                     "input_tokens": 10, "output_tokens": 300, "cache_read_input_tokens": 5000,
                                     "output_tokens_details": {"thinking_tokens": 120}}}) + "\n")
    (run / "AGENT_RUN.json").write_text(json.dumps({"wall_hours_used": 0.5, "budget_hours": 1.5,
                                                    "stopped_because": "final_flag"}))
    (run / "logs" / "rounds.jsonl").write_text(json.dumps({"event": "final_rejected", "round": 1,
                                                           "missing": ["crop.recenter"], "chain_problems": []}) + "\n")
    (run / "workspace" / "NOTES.md").write_text("# NOTES\n\nran things\n")
    return run


def test_agent_run_process_metrics(agent_run):
    report = process.process(agent_run)
    assert report["agent_run"] and report["n_prep_images"] == 54
    assert report["delivered_chain"] == ["C001", "E002", "K001", "S002", "S003"]

    seg = report["D1_adaptivity"]["segment"]
    assert seg["n_delivered_images"] == 54 and seg["main_call"] == "S002" and seg["first_full_set_call"] == "S002"
    assert seg["share_images_with_own_setting"] == pytest.approx(9 / 54)       # the 30.0 entry is not its own
    assert seg["rescued_share"] == pytest.approx(5 / 54) == seg["share_not_from_first_full_set_call"]
    assert seg["images_by_call"] == {"S002": 49, "S003": 5}
    diameter = seg["lever_dispersion"]["diameter"]
    assert (diameter["min"], diameter["max"], diameter["n_distinct"]) == (30.0, 60.0, 3)
    assert "diameter" in seg["levers_varying_across_images"]
    crop = report["D1_adaptivity"]["crop"]
    assert crop["main_call"] == "C001" and crop["rescued_share"] == 0 and crop["share_images_with_cell_overrides"] == 0

    d5 = report["D5_lever_coverage"]
    groups = d5["groups"]
    assert groups["segment.diameter"]["status"] == "substantive"
    assert groups["segment.diameter"]["calls"] == {"S001": "probe", "S002": "chain"}
    assert groups["embed.adaptation_depth"]["status"] == "substantive"
    assert groups["embed.learning_rate"]["status"] == "token" and groups["crop.margin_mode"]["status"] == "token"
    assert {"embed.learning_rate", "crop.margin_mode"} <= set(d5["token_only_groups"])
    assert groups["batch_correct.theta"]["status"] == "skipped"                # a whole-tool reason
    assert groups["crop.recenter"]["status"] == "untested"

    d6 = report["D6_traceability"]
    assert d6["share_with_note"] == pytest.approx(4 / 5) and d6["without_note"] == ["C001"]
    assert d6["share_note_cites_number"] == pytest.approx(2 / 5)                # a call id is not a measurement
    assert set(d6["without_number"]) == {"C001", "E002", "S003"}

    e1 = report["E1_cost"]
    tools = e1["tools"]
    assert tools["per_tool"]["segment"]["calls"] == 4 and tools["per_tool"]["segment"]["statuses"] == {"done": 3, "failed": 1}
    assert tools["gpu_hours"] == pytest.approx((10 + 100 + 50 + 10 + 20 + 360) / 3600)
    assert tools["max_concurrent_calls"] == 2                                    # S002/S003, C001/C002
    assert tools["tool_busy_hours"] == pytest.approx((10 + 130 + 40 + 20 + 360 + 30 + 10) / 3600)
    assert e1["wall_hours"] == 0.5 and e1["budget_hours"] == 1.5
    llm = e1["llm"]
    assert (llm["num_turns"], llm["total_cost_usd"], llm["n_result_events"]) == (7, 1.5, 1)
    assert llm["llm_api_hours"] == pytest.approx(0.5) and llm["thinking_tokens"] == 120
    assert llm["tokens"]["cache_read_input_tokens"] == 5000

    f3 = report["F3_failures"]
    assert f3["no_valid_choice"] is False and f3["chain_verdict_from"] == "re-walked here"
    assert f3["failed_calls"] == ["S004"] and f3["partial_calls"] == [] and f3["unfinished_calls"] == []
    assert f3["n_images_without_delivered_cells"] == 0 and f3["audit_hits"] is None
    assert f3["deadline_without_final"] is False and f3["final_rejections"]["n"] == 1
    assert report["end_state"]["end_state"] == "delivered" and report["end_state"]["timed_out"] is False
    assert report["end_state"]["refused"] is False
    assert report["claims"]["final_flag"] is True and report["claims"]["claimed_complete"] is True
    assert report["claims"]["closing_text"] is None                    # the synthetic result event has no text
    assert llm["usd_source"] == "claude_code_estimate" and llm["sources"]["tokens"] == "result_events"
    assert e1["sources"]["gpu_hours"].startswith("ledger")
    assert report["notes_md_chars"] > 0


def test_the_scorer_verdict_and_audit_are_used_when_present(agent_run, tmp_path):
    (agent_run / "RESULT.json").write_text(json.dumps({"status": "INVALID", "chain": {"ok": False, "problems": ["x"]},
                                                       "audit": {"hits": [{"needle": "a"}, {"needle": "b"}]}}))
    f3 = process.process(agent_run)["F3_failures"]
    assert f3["no_valid_choice"] is True and f3["choice_problems"] == ["x"] and f3["chain_rewalked_ok"] is True
    assert f3["chain_verdict_from"] == "RESULT.json" and f3["audit_hits"] == 2 and f3["score_status"] == "INVALID"
    out = tmp_path / "out"
    main(["process", "--run", str(agent_run), "--out", str(out)])
    assert json.loads((out / "PROCESS.json").read_text())["run"] == "claude-x"
    text = (out / "PROCESS.md").read_text()
    assert "## D1 adaptivity" in text and "token only" in text and "No valid choice: True" in text
    assert not (agent_run / "PROCESS.json").exists()                          # --out keeps the run untouched


def test_control_run_and_empty_images(tmp_path):
    root, run = tmp_path / "root", tmp_path / "control" / "pairs" / "full" / "gpu"
    labels = build_private(root)
    images = labels["image_id"].tolist()
    workspace_data(run, root)
    chain = Chain(run, traced=False)
    s = chain.segment(images)
    c = chain.crop(s, dict.fromkeys(images[:-4], 1))                # 4 images without a cell
    e = chain.embed(c, lambda cells: informative(labels, cells))
    k = chain.cluster(e)
    chain.choose({"cluster_call": k, "resolution": 0.5, "skipped": {}})
    (run / "logs" / "control_commands.jsonl").write_text("".join(            # the control driver's own log
        json.dumps({"tool": tool, "argv": ["--name", name]}) + "\n"
        for tool, name in (("segment", "seg"), ("crop", "crop"), ("embed", "emb"), ("cluster", "k"))))
    report = process.write(run)
    assert not report["agent_run"]
    for key in ("D1_adaptivity", "D5_lever_coverage", "D6_traceability"):
        assert report[key]["applicable"] is False
    assert report["E1_cost"]["llm"]["applicable"] is False
    assert report["E1_cost"]["tools"]["n_calls"] == 4 and report["E1_cost"]["tools"]["gpu_hours"] > 0
    f3 = report["F3_failures"]
    assert f3["no_valid_choice"] is False, f3["choice_problems"]
    assert f3["final_rejections"]["applicable"] is False
    assert f3["images_without_delivered_cells"] == sorted(images[-4:])
    assert (run / "PROCESS.json").exists() and "Not applicable" in (run / "PROCESS.md").read_text()
    assert report["end_state"]["end_state"] == "delivered" and report["claims"]["applicable"] is False


def test_no_choice_at_all(tmp_path):
    run = tmp_path / "runs" / "claude-empty"
    (run / "workspace" / "submission").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "AGENT_RUN.json").write_text(json.dumps({"wall_hours_used": 1.0, "stopped_because": "deadline"}))
    report = process.process(run)
    f3 = report["F3_failures"]
    assert f3["no_valid_choice"] is True and f3["deadline_without_final"] is True
    assert report["D1_adaptivity"]["segment"]["applicable"] is False
    assert report["D6_traceability"]["applicable"] is False
    assert report["end_state"]["end_state"] == "no_deliverable" and report["end_state"]["timed_out"] is True
    llm = report["E1_cost"]["llm"]                                 # no trace: null with a reason, never 0
    assert llm["total_cost_usd"] is None and llm["tokens"] is None and llm["num_turns"] is None
    assert llm["unavailable_reasons"]["total_cost_usd"] == "no result event in the trace"
    assert report["claims"]["claimed_complete"] is False and report["claims"]["closing_text"] is None
    process.markdown(report)                                       # renders without a chain


def _trace(run: Path, events: list[dict]) -> None:
    (run / "traces").mkdir(parents=True, exist_ok=True)
    (run / "traces" / "round_00.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def test_crash_refusal_and_claims(tmp_path):
    run = tmp_path / "runs" / "claude-crash"
    (run / "workspace" / "submission").mkdir(parents=True)
    (run / "logs").mkdir()
    (run / "AGENT_RUN.json").write_text(json.dumps({"wall_hours_used": 0.2, "stopped_because": "repeated_failures"}))
    (run / "logs" / "rounds.jsonl").write_text(json.dumps({"event": "round_end", "round": 0, "returncode": 1,
                                                           "timed_out": False}) + "\n")
    _trace(run, [{"type": "system", "subtype": "init"},
                 {"type": "result", "is_error": True, "result": "", "num_turns": 1, "total_cost_usd": 0.01}])
    report = process.process(run)
    end = report["end_state"]
    assert end["end_state"] == "crashed" and end["timed_out"] is False and end["refused"] is True
    assert any("repeated_failures" in s for s in end["crash_signals"]) and any("exited with 1" in s for s in end["crash_signals"])
    assert report["claims"]["claimed_complete"] is False
    # a run that ends without FINAL but whose closing text says it is done claims completion
    key = "sk-" + "a" * 30
    (run / "AGENT_RUN.json").write_text(json.dumps({"wall_hours_used": 0.2, "stopped_because": "max_rounds"}))
    (run / "logs" / "rounds.jsonl").write_text(json.dumps({"event": "round_end", "round": 0, "returncode": 0}) + "\n")
    _trace(run, [{"type": "result", "is_error": False, "result": f"All done, the analysis is complete. {key} " + "x" * 3000}])
    report = process.process(run)
    claim = report["claims"]
    assert claim["final_flag"] is False and claim["closing_text_claims"] is True and claim["claimed_complete"] is True
    assert len(claim["closing_text"]) == process.CLOSING_CHARS and key not in claim["closing_text"]
    assert report["end_state"]["end_state"] == "no_deliverable" and report["end_state"]["refused"] is False
