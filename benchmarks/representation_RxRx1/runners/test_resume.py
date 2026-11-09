"""CPU tests of contract amendment A8's resume (resume.py) with the stand-in analyzer and scripted engines.

- the platform: the cluster's deadline is the Slurm job's (RXRX_DEADLINE_UTC ignored), the VM's RXRX_DEADLINE_UTC or none;
  without a deadline no call gets a time cap; a cluster-mode run writes no replay record and no new RUN_META key;
- shape, interrupted between two calls (at a request) and during a call, then resumed: the same final
  decision, calls, ledgers and token count as an uninterrupted run, no model request for the replayed part, the
  interrupted call's directory moved aside and rerun, the resume recorded in RUN_META.json and AGENT_RUN.json;
- a resume that is itself stopped during its replay: the next one replays the complete logs (no lost reply);
- react (react_arm.py, the shared runner, amendment A14): its replay -- interrupted at a request and inside a call,
  stopped in its replay, a replayed prompt that differs -- is tested in test_react.py;
- llm: the logged attempt stays, the interrupted one restarts (its module moved aside), every reply's tokens count;
  a logged valid attempt whose delivery was not written runs its module again without a request;
- the divergence guard (a replayed reply of another schema, a replayed prompt that is not the logged one, a replayed
  reply invalid now, a changed call output) and the refusals (a finished run, a cluster-mode run, changed settings);
- the call records: the heartbeat of a running call, the caller's seconds served again, no call without a directory;
- the llm module's audit hook through a linked root (the VM's mirror) and its denied roots in both forms.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from benchmarks.batch_correction_JUMP.runners.score import jsonl
from benchmarks.representation_RxRx1.runners import llm_arms, resume, shape_arm
from benchmarks.representation_RxRx1.runners.test_benchmark import (STUB_TOKENS, TINY_MODULE,  # noqa: F401
                                                                    ScriptedEngine, _module, llm_env, prepared,
                                                                    stub)


class Killed(BaseException):
    """A preemption: nothing after it runs."""


@pytest.fixture(autouse=True)
def _require_agentic_stays_in_the_test(monkeypatch):
    """shape_arm.run with an engine sets SHAPE_REQUIRE_AGENTIC=1 in os.environ; it must not outlive the test (a later
    test of the shared verifier would see every reasoner failure raise)."""
    monkeypatch.setenv("SHAPE_REQUIRE_AGENTIC", "0")     # records the variable's state before the test
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC")


@pytest.fixture()
def vm(monkeypatch):
    monkeypatch.setenv("RXRX_PLATFORM", "vm")
    monkeypatch.setenv("RXRX_RUN_TAG", "testvm-20261010T000000Z")
    monkeypatch.delenv("RXRX_DEADLINE_UTC", raising=False)


class PlanEngine:
    """A SHAPE reasoner stand-in: the scripted round plans in order (then an empty plan: stop), the final choice
    `final_call`; every schema asked is recorded."""

    def __init__(self, plans, final_call=None):
        self.plans, self.final_call, self.asked = list(plans), final_call, []

    def generate(self, content, response_format=None, **kwargs):
        from shape.engine import usage

        self.asked.append(response_format.__name__)
        if response_format.__name__ == "BasedPlan":
            reply = self.plans.pop(0) if self.plans else {"analysis": "stop", "moves": []}
        else:
            reply = {"call": self.final_call, "reasoning": "the best call"}
        tokens = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": STUB_TOKENS}
        usage.record(tokens)
        return {"content": response_format.model_validate(reply), "usage": tokens}


class KillAt:
    """`engine`, killed when it is asked for the `at`-th reply."""

    def __init__(self, engine, at: int):
        self.engine, self.at, self.n = engine, at, 0

    def generate(self, content, response_format=None, **kwargs):
        self.n += 1
        if self.n == self.at:
            raise Killed()
        return self.engine.generate(content, response_format=response_format, **kwargs)


def _move(parent, **changes) -> dict:
    return {"parent": parent, "changes": [{"lever": k, "value": v, "basis": "b"} for k, v in changes.items()],
            "diagnosis": "d",
            "expected": [{"metric": "sibling_site_map", "direction": "up"}]}


# A24: shape_arm's two-round expert start (round 2 resumes agent_1), then the agent's own moves
_START, _SECOND = shape_arm.EXPERT_CONFIG
PLANS = [{"analysis": "a", "moves": [_move("input", **_START)]},
         {"analysis": "b", "moves": [_move("agent_1", trainable_pct=_SECOND["trainable_pct"])]},
         {"analysis": "c", "moves": [_move("agent_2", learning_rate=1e-5)]}]


def _shape(root, run_dir, engine, **kw):
    return shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=4, engine=engine, **kw)


def _kill_in_call(stub, monkeypatch, nth: int):
    """The stand-in analyzer killed in its `nth` training call, after it wrote part of its outputs."""
    real, count = stub.execute, {"n": 0}

    def execute(self, **kw):
        if kw.get("adaptation_depth") != "zero_shot" and not kw.get("embed_only"):
            count["n"] += 1
            if count["n"] == nth:
                out = Path(kw["query_cache_dir"]) / "cell_state_analysis" / kw["output_subdir"]
                out.mkdir(parents=True, exist_ok=True)
                (out / "partial.pt").write_text("half a checkpoint")
                raise Killed()
        return real(self, **kw)

    execute.__signature__ = real.__signature__
    monkeypatch.setattr(stub, "execute", execute)
    return real


def _norm(value, run_dir: Path):
    """A record with the run directory's own paths made relative (two runs compared)."""
    text = json.dumps(value, default=str).replace(json.dumps(str(run_dir))[1:-1], "<RUN>")
    return json.loads(text)


def _trajectory(run_dir: Path) -> dict:
    """What an interrupted-then-resumed run must share with an uninterrupted one."""
    calls = jsonl(run_dir / "logs" / "tool_calls.jsonl")
    decision = json.loads((run_dir / "submission" / "decision.json").read_text())
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    out = {"decision": {k: decision.get(k) for k in ("use", "case", "status", "provisional")},
           "starts": [_norm({k: r.get(k) for k in ("call", "parent", "set", "init", "kwargs")}, run_dir)
                      for r in calls if r["event"] == "start"],
           "ends": [_norm({k: r.get(k) for k in ("call", "status", "score", "metrics", "tool")}, run_dir)
                    for r in calls if r["event"] == "end"],
           "tokens": agent["tokens"]["total_tokens"], "stopped": agent["stopped_because"],
           "rounds": agent.get("n_rounds")}
    if (run_dir / "logs" / "reasoner.jsonl").exists():
        out["requests"] = [(r["request"], r["schema"], r["usage"]) for r in jsonl(run_dir / "logs" / "reasoner.jsonl")]
    if (run_dir / "logs" / "requests.jsonl").exists():
        out["requests"] = [(r["step"], r["try"], r.get("kind"), r["error"], r.get("usage"))
                           for r in jsonl(run_dir / "logs" / "requests.jsonl")]
        out["requests_count"] = agent["requests"]
    if (run_dir / "logs" / "input_trial.json").exists():
        trial0 = json.loads((run_dir / "logs" / "input_trial.json").read_text())
        out["input"] = _norm({k: trial0.get(k) for k in ("status", "score", "tool")}, run_dir)
    return out


# ---- the platform -------------------------------------------------------------------------------------------------

def test_the_deadline_is_slurms_on_the_cluster_and_the_environments_on_the_vm(monkeypatch):
    from benchmarks.batch_correction_JUMP.runners import shape_arm as jump

    monkeypatch.setattr(jump, "slurm_deadline", lambda: 123.0)
    monkeypatch.setenv("RXRX_DEADLINE_UTC", "2026-10-11T12:00:00Z")
    monkeypatch.delenv("RXRX_PLATFORM", raising=False)
    assert resume.deadline() == 123.0 and resume.platform_keys() == {}
    monkeypatch.setenv("RXRX_PLATFORM", "vm")
    monkeypatch.setenv("RXRX_RUN_TAG", "host-20261010T000000Z")
    assert resume.deadline() == 1791720000.0          # 2026-10-11T12:00:00Z
    assert resume.platform_keys() == {"platform": "vm", "run_tag": "host-20261010T000000Z", "replay_records": True}
    monkeypatch.delenv("RXRX_DEADLINE_UTC")
    assert resume.deadline() is None
    monkeypatch.setenv("RXRX_DEADLINE_UTC", "tomorrow")
    with pytest.raises(SystemExit, match="RXRX_DEADLINE_UTC"):
        resume.deadline()


def test_a_cluster_run_writes_no_replay_record_and_no_new_key(prepared, stub, tmp_path, monkeypatch):
    monkeypatch.delenv("RXRX_PLATFORM", raising=False)
    root, _, _ = prepared
    run_dir = tmp_path / "cluster_run"
    _shape(root, run_dir, PlanEngine(PLANS, "agent_2"))
    assert not (run_dir / "logs" / "reasoner.jsonl").exists() and not list(run_dir.rglob(resume.CALL_RECORD))
    meta, agent = (json.loads((run_dir / name).read_text()) for name in ("RUN_META.json", "AGENT_RUN.json"))
    assert not {"platform", "run_tag", "replay_records", "resumes"} & set(meta) and "resumes" not in agent
    with pytest.raises(SystemExit, match="RUN_FINISHED"):
        _shape(root, run_dir, PlanEngine([]), resume=True)
    (run_dir / "AGENT_RUN.json").unlink()
    with pytest.raises(SystemExit, match="NOT_RESUMABLE"):
        _shape(root, run_dir, PlanEngine([]), resume=True)


def test_without_a_deadline_no_call_gets_a_time_cap(prepared, stub, vm, tmp_path):
    root, _, _ = prepared
    final = _shape(root, tmp_path / "run", PlanEngine(PLANS, "agent_2"))
    assert final["use"] == "agent_2"
    assert stub.calls and all(kw.get("max_minutes") is None for kw in stub.calls)
    meta = json.loads((tmp_path / "run" / "RUN_META.json").read_text())
    assert meta["deadline_utc"] is None and meta["platform"] == "vm" and meta["replay_records"] is True
    assert meta["run_tag"] == "testvm-20261010T000000Z"
    records = sorted(p.parent.name for p in (tmp_path / "run").rglob(resume.CALL_RECORD))
    assert len(records) == 4 and "input" in records                 # trial 0 and the three calls


# ---- shape -----------------------------------------------------------------------------------------------------

def test_shape_resumed_between_two_calls_replays_without_asking_the_model(prepared, stub, vm, tmp_path):
    root, _, _ = prepared
    reference_engine = PlanEngine(PLANS, "agent_2")
    _shape(root, tmp_path / "reference", reference_engine)
    reference = _trajectory(tmp_path / "reference")
    assert reference_engine.asked == ["BasedPlan"] * 5 + ["FinalChoice"]

    run_dir = tmp_path / "run"
    with pytest.raises(Killed):                                       # stopped while asking for the third plan
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=3))
    n_before = len(stub.calls)
    engine = PlanEngine(PLANS[2:], "agent_2")
    final = _shape(root, run_dir, engine, resume=True)
    assert engine.asked == reference_engine.asked[2:]                 # no model request for the two replayed ones
    assert len(stub.calls) - n_before == 1                            # agent_1, agent_2 and trial 0 served
    assert final["use"] == "agent_2" and _trajectory(run_dir) == reference
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    (entry,) = meta["resumes"]
    assert (entry["resume"], entry["logged_requests"], entry["replayed_requests"], entry["live_from"]) == (
        1, 2, 2, "request 3")
    assert len(entry["finished_calls"]) == 3 and entry["moved_aside"] == []
    assert entry["run_tag"] == "testvm-20261010T000000Z" and entry["active_seconds_before"] >= 0
    assert json.loads((run_dir / "AGENT_RUN.json").read_text())["resumes"] == meta["resumes"]
    replayed = [r for r in jsonl(run_dir / "logs" / "reasoner.jsonl") if r.get("replayed")]
    assert [r["request"] for r in replayed] == [1, 2]
    logged = {r["call"]: r["seconds"] for r in jsonl(run_dir / "logs" / "resume_1" / "tool_calls.jsonl")
              if r["event"] == "end"}
    again = {r["call"]: r["seconds"] for r in jsonl(run_dir / "logs" / "tool_calls.jsonl") if r["event"] == "end"}
    assert logged and all(again[call] == seconds for call, seconds in logged.items())   # bit-identical
    assert (run_dir / "logs" / "resume_1" / "reasoner.jsonl").exists()   # the logs of the stopped run, kept
    assert len(jsonl(run_dir / "logs" / "resume_1" / "reasoner.jsonl")) == 2


def test_shape_resumed_during_a_call_moves_the_partial_call_aside_and_reruns_it(prepared, stub, vm, tmp_path,
                                                                                monkeypatch):
    root, _, _ = prepared
    reference_engine = PlanEngine(PLANS, "agent_2")
    _shape(root, tmp_path / "reference", reference_engine)
    reference = _trajectory(tmp_path / "reference")

    run_dir = tmp_path / "run"
    real = _kill_in_call(stub, monkeypatch, nth=2)                    # killed inside agent_2
    with pytest.raises(Killed):
        _shape(root, run_dir, PlanEngine(PLANS, "agent_2"))
    monkeypatch.setattr(stub, "execute", real)
    n_before = len(stub.calls)
    engine = PlanEngine(PLANS[2:], "agent_2")
    final = _shape(root, run_dir, engine, resume=True)
    assert engine.asked == reference_engine.asked[2:]
    assert len(stub.calls) - n_before == 2                            # agent_2 again, then agent_3
    assert final["use"] == "agent_2" and _trajectory(run_dir) == reference
    (entry,) = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    (moved,) = entry["moved_aside"]
    assert entry["live_from"] == "call " + moved["from"].split("cell_state_analysis/")[1]
    assert entry["replayed_requests"] == 2 and moved["to"].endswith(".interrupted_r1")
    assert (run_dir / moved["to"] / "partial.pt").read_text() == "half a checkpoint"   # moved, never deleted
    assert (run_dir / moved["from"] / resume.CALL_RECORD).exists()


def test_a_second_resume_continues_from_the_first(prepared, stub, vm, tmp_path, monkeypatch):
    root, _, _ = prepared
    reference_engine = PlanEngine(PLANS, "agent_2")
    _shape(root, tmp_path / "reference", reference_engine)
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=2))
    with pytest.raises(Killed):                                       # the first resume is stopped too
        _shape(root, run_dir, KillAt(PlanEngine(PLANS[1:], "agent_2"), at=2), resume=True)
    engine = PlanEngine(PLANS[2:], "agent_2")
    _shape(root, run_dir, engine, resume=True)
    assert engine.asked == reference_engine.asked[2:]
    assert _trajectory(run_dir) == _trajectory(tmp_path / "reference")
    first, second = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    assert (first["replayed_requests"], second["replayed_requests"]) == (1, 2)
    assert second["active_seconds_before"] >= first["active_seconds_before"]


def test_a_replayed_reply_of_another_schema_stops_the_resume(prepared, stub, vm, tmp_path):
    root, _, _ = prepared
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=3))
    path = run_dir / "logs" / "reasoner.jsonl"
    records = jsonl(path)
    records[1]["schema"] = "FinalChoice"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(SystemExit, match="REPLAY_DIVERGED"):
        _shape(root, run_dir, PlanEngine(PLANS[2:], "agent_2"), resume=True)


def test_a_finished_call_whose_output_changed_is_refused(prepared, stub, vm, tmp_path):
    root, _, _ = prepared
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=3))
    record = next(p for p in run_dir.rglob(resume.CALL_RECORD) if p.parent.name != "input")
    adata = json.loads(record.read_text())["files"]["adata_path"]["path"]
    Path(adata).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="INTEGRITY_FAILED"):
        _shape(root, run_dir, PlanEngine(PLANS[2:], "agent_2"), resume=True)


def test_a_resume_with_other_settings_is_refused(prepared, stub, vm, tmp_path):
    root, _, _ = prepared
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=2))
    with pytest.raises(SystemExit, match="RESUME_STALE"):
        shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=6, engine=PlanEngine([]),
                      resume=True)
    # the refused resume moved the logs but recorded nothing: the next resume takes them as its source
    assert (run_dir / "logs" / "resume_1" / "reasoner.jsonl").exists()
    final = _shape(root, run_dir, PlanEngine(PLANS[1:], "agent_2"), resume=True)
    assert final["use"] == "agent_2"
    assert [r["resume"] for r in json.loads((run_dir / "RUN_META.json").read_text())["resumes"]] == [1]


# ---- llm ----------------------------------------------------------------------------------------------------------

def test_llm_resumed_keeps_its_logged_attempt_and_restarts_the_interrupted_one(prepared, stub, llm_env, vm, tmp_path,
                                                                                monkeypatch):
    root, _, _ = prepared
    run_dir = tmp_path / "llm_run"
    real = llm_arms.run_module

    def killed(*args, **kwargs):
        raise Killed()

    monkeypatch.setattr(llm_arms, "run_module", killed)
    with pytest.raises(Killed):                                      # attempt 1 fails its static check, 2 is killed
        llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub",
                     engine=ScriptedEngine([_module("import os\n"), _module(TINY_MODULE)]))
    monkeypatch.setattr(llm_arms, "run_module", real)
    engine = ScriptedEngine([_module(TINY_MODULE, reasoning="again")])
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine, resume=True)
    assert final["status"] == "delivered" and final["use"] == "attempt_2"
    assert len(engine.prompts) == 1 and "could not be used: the static check rejected" in engine.prompts[0]
    assert "import os" in engine.prompts[0]                           # the previous module, as the run had it
    requests = jsonl(run_dir / "logs" / "requests.jsonl")
    assert [(r["try"], r.get("kind")) for r in requests] == [(1, "static_check"), (2, None)]
    assert (run_dir / "modules" / "attempt_2.py.interrupted_r1").exists()
    (entry,) = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    assert [m["to"] for m in entry["moved_aside"]] == ["modules/attempt_2.py.interrupted_r1"]
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert agent["attempts"] == 2 and agent["resumes"][0]["resume"] == 1
    # three replies were received (attempt 1, the interrupted attempt 2, attempt 2 again): all count
    assert agent["tokens"]["total_tokens"] == 3 * STUB_TOKENS
    assert (run_dir / "COMPLETE").exists() is False                   # the launcher writes it, not the arm


# ---- the driver's helpers --------------------------------------------------------------------------------------------

def test_the_driver_finds_one_run_per_stage_by_its_run_tags(tmp_path):
    state, bench, commit = tmp_path / "state.json", tmp_path / "bench", "a" * 40
    resume.add_tag(state, "vm1-20261010T000000Z", commit, "HEPG2")
    with pytest.raises(SystemExit, match="STATE_MISMATCH"):
        resume.add_tag(state, "vm1-20261010T000000Z", commit, "HEPG2")
    with pytest.raises(SystemExit, match="STATE_MISMATCH"):
        resume.add_tag(state, "vm2-20261010T000000Z", "b" * 40, "HEPG2")
    assert resume.find_run(state, bench, "shape", "HEPG2", commit) is None
    run = bench / f"shape_HEPG2_{commit[:8]}_20261010T000001Z_vm1-20261010T000000Z"
    run.mkdir(parents=True)
    (bench / f"shape_HEPG2_{commit[:8]}_20261010T000001Z_other-tag").mkdir()                 # another driver's
    (bench / f"{run.name}.aborted-vm2-20261010T000000Z").mkdir()                             # kept aside
    (bench / f"{run.name.replace('000001Z', '000009Z')}.aborted-manual").mkdir()             # set aside by hand
    assert resume.find_run(state, bench, "shape", "HEPG2", commit) == run
    assert resume.run_status("shape", run) == "empty"
    (run / "RUN_META.json").write_text(json.dumps({"commit": commit}))
    assert resume.run_status("shape", run) == "incomplete"
    (run / "AGENT_RUN.json").write_text("{}")
    assert resume.run_status("shape", run) == "finished"
    # A18: a shape run of another seed (RXRX_SEED) is another run; none recorded = seed 0
    assert resume.find_run(state, bench, "shape", "HEPG2", commit, seed=1) is None
    (run / "RUN_META.json").write_text(json.dumps({"commit": commit, "seed": 1}))
    assert resume.find_run(state, bench, "shape", "HEPG2", commit, seed=1) == run
    assert resume.find_run(state, bench, "shape", "HEPG2", commit) is None
    (run / "RUN_META.json").write_text(json.dumps({"commit": commit}))
    resume.add_tag(state, "vm2-20261010T000000Z", commit, "HEPG2")
    (bench / f"shape_HEPG2_{commit[:8]}_20261010T000002Z_vm2-20261010T000000Z").mkdir()
    with pytest.raises(SystemExit, match="DUPLICATE_RUN"):
        resume.find_run(state, bench, "shape", "HEPG2", commit)
    react = bench / f"react_HEPG2_{commit[:8]}_20261010T000003Z_vm2-20261010T000000Z"
    (react / "submission").mkdir(parents=True)
    (react / "RUN_META.json").write_text(json.dumps({"commit": commit}))
    (react / "AGENT_RUN.json").write_text("{}")
    (react / "submission" / "decision.json").write_text(json.dumps({"status": "failed"}))
    assert resume.run_status("react", react) == "failed"
    (react / "submission" / "decision.json").write_text(json.dumps({"status": "delivered"}))
    assert resume.run_status("react", react) == "delivered"


# ---- a resume stopped during its replay (B1) -------------------------------------------------------------------

def _stop_in_the_replay(monkeypatch, nth: int):
    """CallCache serves its `nth` finished call: killed (a resume stopped before it went live)."""
    real, count = resume.CallCache._serve, {"n": 0}

    def serve(self, *args, **kwargs):
        count["n"] += 1
        if count["n"] == nth:
            raise Killed()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(resume.CallCache, "_serve", serve)
    return real


def test_shape_a_resume_stopped_in_its_replay_loses_no_logged_reply(prepared, stub, vm, tmp_path, monkeypatch):
    root, _, _ = prepared
    reference_engine = PlanEngine(PLANS, "agent_2")
    _shape(root, tmp_path / "reference", reference_engine)
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):                                       # two replies logged, three calls finished
        _shape(root, run_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=3))
    real = _stop_in_the_replay(monkeypatch, nth=2)
    first = PlanEngine(PLANS[2:], "agent_2")
    with pytest.raises(Killed):                                       # resume 1 stops while it replays
        _shape(root, run_dir, first, resume=True)
    monkeypatch.setattr(resume.CallCache, "_serve", real)
    assert first.asked == []
    engine = PlanEngine(PLANS[2:], "agent_2")
    _shape(root, run_dir, engine, resume=True)
    assert engine.asked == reference_engine.asked[2:]                 # no logged reply asked again
    assert _trajectory(run_dir) == _trajectory(tmp_path / "reference")   # tokens included
    first_entry, second = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    assert first_entry["live_from"] is None and first_entry["replay_source"] == "logs/resume_1"
    assert second["replay_source"] == "logs/resume_1" and second["logs_moved_to"] == "logs/resume_2"
    assert (second["logged_requests"], second["replayed_requests"]) == (2, 2)


# ---- the prompts of a replay ------------------------------------------------------------------------------------

def test_a_replayed_prompt_that_is_not_the_logged_one_stops_the_resume(prepared, stub, llm_env, vm, tmp_path):
    root, _, _ = prepared
    shape_dir = tmp_path / "shape_run"
    with pytest.raises(Killed):
        _shape(root, shape_dir, KillAt(PlanEngine(PLANS, "agent_2"), at=3))
    path = shape_dir / "logs" / "reasoner.jsonl"
    records = jsonl(path)
    records[1]["prompt_masked_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(SystemExit, match="REPLAY_DIVERGED: logged request 2: the replayed prompt"):
        _shape(root, shape_dir, PlanEngine(PLANS[2:], "agent_2"), resume=True)


def test_the_clock_text_is_masked_before_a_shape_prompt_is_compared():
    one = ["Rounds: 1 done. Wall time so far: 0.12 h; time left before the stage must end (the job's time limit; calls "
           "still running then are lost): 3.40 h; seconds of each finished call: {}"]
    two = ["Rounds: 1 done. Wall time so far: 5.01 h; time left before the stage must end (the job's time limit; calls "
           "still running then are lost): 0.20 h; seconds of each finished call: {}"]
    assert resume.masked_digest(one) == resume.masked_digest(two) != resume._prompt_digest(one)
    assert resume.masked_digest(["Wall time so far: 1.00 h; seconds"]) == resume.masked_digest(
        ["Wall time so far: 2.50 h; seconds"])
    assert resume.masked_digest(["Rounds: 1 done."]) != resume.masked_digest(["Rounds: 2 done."])


# ---- llm: a logged valid attempt whose delivery was not written ---------------------------------------------------

def test_llm_resumed_after_a_valid_attempt_runs_its_module_again_without_a_request(prepared, stub, llm_env, vm,
                                                                                    tmp_path, monkeypatch):
    root, _, _ = prepared
    run_dir = tmp_path / "llm_run"
    real = llm_arms.audit_tokens

    def killed(source):                               # right after the valid attempt is logged, before its delivery
        raise Killed()

    monkeypatch.setattr(llm_arms, "audit_tokens", killed)
    with pytest.raises(Killed):
        llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub",
                     engine=ScriptedEngine([_module(TINY_MODULE)]))
    monkeypatch.setattr(llm_arms, "audit_tokens", real)
    assert not (run_dir / "submission" / "embeddings.npy").exists()
    engine = ScriptedEngine([])
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine, resume=True)
    assert final["status"] == "delivered" and final["use"] == "attempt_1" and engine.prompts == []
    assert [(a["attempt"], a["stage"]) for a in jsonl(run_dir / "logs" / "attempts.jsonl")] == [(1, "done")]
    assert (run_dir / "logs" / "attempt_1.stderr.log.interrupted_r1").exists()
    assert (run_dir / "submission" / "embeddings.npy").exists()
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert agent["attempts"] == 1 and agent["tokens"]["total_tokens"] == STUB_TOKENS


# ---- the call records ---------------------------------------------------------------------------------------------

def test_a_running_call_touches_its_heartbeat_and_a_served_call_keeps_the_callers_seconds(tmp_path, monkeypatch):
    monkeypatch.setattr(resume, "HEARTBEAT_SECONDS", 0.05)
    cache = resume.CallCache(tmp_path / "cache")
    beat = tmp_path / "cache" / resume.HEARTBEAT
    seen = []

    def invoke():
        time.sleep(0.5)
        seen.append(beat.stat().st_mtime)
        return {"seconds": 0.5}

    started = time.time()
    params = {"output_subdir": "ssl_calls/abc", "max_minutes": 3.0}
    assert cache.run(params, invoke) == {"seconds": 0.5}
    assert seen and seen[0] >= started + 0.2                          # touched while the call ran
    cache.note_outer(params, 0.7)
    record = json.loads((tmp_path / "cache" / "cell_state_analysis" / "ssl_calls" / "abc"
                         / resume.CALL_RECORD).read_text())
    assert record["outer_seconds"] == 0.7 and resume.CallCache.seconds_of(record) == 0.7
    assert resume.CallCache.seconds_of({"wall_seconds": 0.5}) == 0.5
    with pytest.raises(SystemExit, match="CALL_WITHOUT_DIRECTORY"):
        cache.run({"max_minutes": None}, invoke)


# ---- the llm module's audit hook through the VM's mirror links (M2) -----------------------------------------------

def _link_dir(target: Path, link: Path) -> None:
    """`link` -> `target`: a symlink, else (Windows without the privilege) a junction, else skip."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError):
        pytest.skip("no directory links on this platform")


def test_the_module_process_blocks_a_denied_root_through_its_alias_and_its_real_path(tmp_path):
    import numpy as np

    real = tmp_path / "filesystem-r1" / "results"
    (real / "cache" / "HEPG2" / "private").mkdir(parents=True)
    secret = real / "cache" / "HEPG2" / "private" / "labels.csv"
    secret.write_text("cell_id,sirna\n", encoding="utf-8")
    (tmp_path / "scratch").mkdir()
    alias = tmp_path / "scratch" / "results"
    _link_dir(real, alias)
    if os.path.normcase(os.path.realpath(alias)) != os.path.normcase(os.path.realpath(real)):
        pytest.skip("the link does not resolve on this platform")
    worker = tmp_path / "worker.py"
    worker.write_bytes(llm_arms.WORKER.encode("utf-8"))
    images = np.random.default_rng(0).integers(0, 255, size=(4, 6, 64, 64), dtype=np.uint8)

    def reader(path: Path) -> str:
        return (f"import numpy as np\n\n\ndef features(crops):\n    with open({str(path)!r}) as handle:\n"
                "        handle.read()\n    return np.zeros((len(crops), 2))\n")

    for deny in ([alias], [real]):                   # the worker resolves the roots and every opened path
        for opened in (alias / "cache" / "HEPG2" / "private" / "labels.csv", secret):
            with pytest.raises(llm_arms.ModuleFailed, match="blocked by the harness: open"):
                llm_arms.run_module(reader(opened), images, np.arange(4), worker=worker,
                                    stderr_path=tmp_path / "stderr.log", deadline=time.time() + 120, deny=deny)
    roots = llm_arms.deny_roots([alias / "cache" / "HEPG2" / "private"])
    assert str(alias / "cache" / "HEPG2" / "private") in roots
    assert any(os.path.normcase(r) == os.path.normcase(os.path.realpath(secret.parent)) for r in roots)


def test_the_llm_arm_denies_a_linked_private_directory_by_its_real_path(prepared, stub, llm_env, vm, tmp_path):
    import shutil

    root, _, _ = prepared
    elsewhere = tmp_path / "filesystem-r1" / "private"
    elsewhere.parent.mkdir()
    shutil.move(str(root / "private"), str(elsewhere))
    _link_dir(elsewhere, root / "private")
    run_dir = tmp_path / "llm_run"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub",
                         engine=ScriptedEngine([_module(TINY_MODULE)]))
    assert final["status"] == "delivered"
    denied = json.loads((run_dir / "RUN_META.json").read_text())["denied_roots"]
    assert any(os.path.normcase(r) == os.path.normcase(os.path.realpath(elsewhere)) for r in denied)
    assert str((root / "private").absolute()) in denied
