"""CPU tests of the react arm (react_arm.py: the cell-state analyzer on the shared ReAct runner, contract amendment
A14) with a fake engine and the stand-in analyzer: the prompt (SHAPE's task text verbatim, SHAPE's lever names with
type and default, the PARAMETERS reference of A16 -- each lever's CARD Type, Range, Default and mechanics text, no
Family --, the PYTHON section, no SHAPE readout, search rule or expert start), the observations (the
tool's own record with val_history; trial 0 labelled as the zero-shot call it is), trial 0 (zero_shot, not a round),
the round cap, delivery (the agent's finish; else the default call, recorded), failed and refused calls (no SHAPE
value checks: only an init_from naming no finished call is refused), the shared fixed epoch cap (A6) and the A7 resume
of init_from = a call id, SHAPE's resource stop, a fatal engine reply, the python action's allow-list (each call's
h5ad and the label-free cell metadata, never a label), the records score.py reads, and the VM resume by replay
(interrupted at a request and inside a call, a resume stopped in its replay, a replayed python step, a replayed prompt
that differs)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks._shared import react_arm as shared
from benchmarks.batch_correction_JUMP.runners.score import jsonl
from benchmarks.representation_RxRx1.runners import control, prepare as prep_mod, react_arm, resume, score, shape_arm
from benchmarks.representation_RxRx1.runners.test_benchmark import (PROXY_SENTINEL, STUB_TOKENS,  # noqa: F401
                                                                    _effective, _leaks, _shape_engine, llm_env,
                                                                    prepared, stub)


class FakeEngine:
    """The reasoner stand-in: scripted ReactAction replies in order (a dict, returned parsed with its usage; a dict
    with "error": an engine's error reply, returned as it is); every prompt is recorded."""

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, content, response_format=None, **kwargs):
        from shape.engine import usage

        assert response_format is shared.ReactAction
        self.prompts.append(content if isinstance(content, str) else content[0])
        reply = self.replies.pop(0)
        if "error" in reply:
            return reply
        tokens = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": STUB_TOKENS}
        usage.record(tokens)
        return {"content": shared.ReactAction.model_validate(reply), "usage": tokens}


class Killed(BaseException):
    """A preemption: nothing after it runs."""


class KillAt(FakeEngine):
    """A FakeEngine killed when it is asked for the `at`-th reply."""

    def __init__(self, replies, at: int):
        super().__init__(replies)
        self.at, self.n = at, 0

    def generate(self, content, response_format=None, **kwargs):
        self.n += 1
        if self.n == self.at:
            raise Killed()
        return super().generate(content, response_format=response_format, **kwargs)


def _call(**params) -> dict:
    return {"thought": "t", "action": "call",
            "calls": [{"params": [{"name": k, "value": v} for k, v in params.items()]}]}


def _finish(call: str) -> dict:
    return {"thought": "done", "action": "finish", "deliver": call}


def _section(prompt: str, title: str) -> str:
    return prompt.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]


def _verifier(root):
    from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier, budget_for

    summary, cells = prep_mod.load(root)
    return FixedEpochVerifier(budget_for(summary, cells)), summary, cells


def _react(root, run_dir, engine, **kw):
    return react_arm.run(root, run_dir, cell_type="HEPG2", model="stub", engine=engine, **kw)


def test_react_prompt_parameters_observations_trial0_and_delivery(prepared, stub, llm_env, tmp_path):
    root, summary, _ = prepared
    verifier, _, cells = _verifier(root)
    engine = FakeEngine([_call(view_pairing="same_well"),
                         _call(init_from="agent_1", trainable_pct=50, view_pairing="same_well"),
                         _finish("agent_2")])
    run_dir = tmp_path / "react_run"
    final = _react(root, run_dir, engine)
    assert (final["use"], final["case"], final["status"]) == ("agent_2", "finish", "delivered")

    # the tool calls: trial 0 is SHAPE's frozen call, then exactly SHAPE's kwargs with the shared epoch cap (A6)
    cap = verifier.epoch_budget["max_epochs"]
    trial0, first, second = stub.calls
    assert trial0["adaptation_depth"] == "zero_shot" and trial0["output_subdir"] == "ssl_calls/input"
    assert first.get("init_checkpoint") is None and first["view_pairing"] == "same_well"
    calls = score.finished_calls(run_dir)
    assert second["init_checkpoint"] == calls["agent_1"]["resume_best"] and second["trainable_pct"] == 50
    assert [c["max_epochs"] for c in stub.calls] == [cap, cap, cap] and first["split_key"] == "split"
    assert final["checkpoint"] == calls["agent_2"]["resume_best"]
    starts = [r for r in jsonl(run_dir / "logs" / "tool_calls.jsonl") if r["event"] == "start"]
    assert [(r["call"], r["parent"], r["init"]) for r in starts] == [("agent_1", "input", "fresh"),
                                                                    ("agent_2", "agent_1", "resumed: agent_1")]
    assert score.moves(run_dir)["agent_2"]["set"] == {"init_from": "parent", "trainable_pct": 50}

    # the prompt: SHAPE's task text verbatim, the lever names with type and default, no CARD text, no SHAPE readout
    prompt = engine.prompts[0]
    assert re.findall(r"^## (.+)$", prompt, re.MULTILINE) == list(shared.SECTIONS)
    assert _section(prompt, "TASK").strip() == shape_arm.task_brief(summary, "HEPG2", 8)
    rows = [line[2:].split(" | ") for line in _section(prompt, "TOOL").splitlines() if line.startswith("- ")]
    rows = [(r[0], " | ".join(r[1:-1]), json.loads(r[-1])) for r in rows]          # name | type | default
    defaults = verifier.defaults()
    assert [r[0] for r in rows] == verifier.levers and len(rows) == 15 and "max_epochs" not in verifier.levers
    assert all(default == defaults[name] for name, _, default in rows)
    types = {name: kind for name, kind, _ in rows}
    assert types["view_pairing"] == "choice: same_crop | same_site | same_well"
    assert types["augmentation"] == "choice: strong | light"
    assert types["init_from"] == "choice: pretrained | <a finished call id>" and types["learning_rate"] == "number"
    assert types["trainable_pct"] == "integer or null"
    assert shared.TUNING_LINE in _section(prompt, "TOOL")
    # A16 (RxRx1 only): the INPUT section ends with the PARAMETERS reference -- each of the 15 levers with its CARD
    # Type, Range and Default and the CARD's text of what it does, read from the CARD, without the sentences on SHAPE's
    # prompt or readouts; init_from as this adapter runs it; no Family, nothing else
    reference = react_arm.parameters_text(verifier)
    assert _section(prompt, "INPUT").rstrip().endswith(reference)
    texts = react_arm.lever_texts(verifier)
    filtered = {"view_pairing", "clip_grad_norm", "koleo_weight"}
    for lever in verifier.levers:
        if lever in filtered | {"init_from"}:
            continue
        fields = verifier.lever_fields[lever]
        line = f"- {lever} | Type: {fields['Type']} | Range: {fields['Range']} | Default: {fields['Default']}\n  "
        assert len(texts[lever]) > 50 and line + texts[lever] + "\n" in reference + "\n", lever
    assert reference.count("\n- ") == 15 and "max_epochs |" not in reference and "Family" not in reference
    assert react_arm.INIT_FROM_ROW + "\n" in reference
    assert "Range: pretrained, parent" not in reference and "A move from" not in reference
    rows = dict(re.findall(r"^- (\w+) \| .*\n  (.*)$", reference, re.MULTILINE))
    assert set(rows) == set(verifier.levers)
    assert rows["view_pairing"].endswith("same_site and same_well train the cells of one site or well toward each "
                                         "other. same_site and same_well need that metadata on every crop. The pairing "
                                         "changes the level of the training loss and of the val eval loss, and the "
                                         "tool's own epoch budget (max_epochs null) and its early stop read the "
                                         "training loss.")
    assert rows["clip_grad_norm"].endswith("before each optimizer step; 0 turns clipping off.")
    assert rows["koleo_weight"].endswith("0 turns it off. Its size depends on the batch size.")
    for cut in ("val proxy measures", "readouts shown here", "alarms", "effective_rank"):
        assert cut not in reference, cut
    shape_texts = [verifier.readouts_text().splitlines()[0], verifier.defaults_note().splitlines()[0], "ALARMS",
                   verifier.goal_text().splitlines()[0], verifier.basis_text().splitlines()[0],
                   verifier.plateau_text().splitlines()[0], verifier.final_rule_text().splitlines()[0],
                   "levers no call has changed yet",
                   "comparison draws", "the stop is not accepted", "exploration round", "hypothesis_ids",
                   "Tunable parameters", "Recommend", "Switch to", "Earlier runs", "expert", "Expert", "(A16)",
                   "warm start", "within-well (biological"]
    for text in engine.prompts:
        assert "### " not in text and "## Levers" not in text and "Family" not in text
        assert not [s for s in shape_texts if s in text], [s for s in shape_texts if s in text]
        assert "Calls per round: at most 1." in text                      # SHAPE's concurrency() for this tool
        assert "outside [" not in text and "already run" not in text      # no SHAPE value check reaches react
        assert not _leaks(text)
    assert _section(prompt, "PYTHON").startswith('action "python"')
    assert "Rounds used: 0 of 8" in prompt                                # trial 0 is not a round
    assert "Rounds used: 1 of 8" in engine.prompts[1] and "Rounds used: 2 of 8" in engine.prompts[2]

    # the observation: the tool's own record, val_history and its val proxy included (A14 supersedes A4's withholding)
    history = _section(engine.prompts[1], "HISTORY")
    assert "call default params" in history
    assert f"Round 0 ({react_arm.TRIAL0_LABEL}; not a round)" in history   # the zero-shot call the system ran
    agent_1 = next(line for line in history.splitlines() if line.startswith("observation: call agent_1"))
    assert '"val_history": [{"epoch": 1, "sibling_site_map": %s' % PROXY_SENTINEL in agent_1
    assert '"loss_history": [1.23456, 1.11111]' in agent_1 and '"resume_best": "resume_best.pt"' in agent_1

    # the records
    unit = run_dir / "react" / "HEPG2"
    assert len(jsonl(unit / "TRACE.jsonl")) == 4                          # trial 0, 2 rounds, the finish
    selected = json.loads((unit / "SELECTED.json").read_text())
    assert (selected["call_id"], selected["source"], selected["rounds"]) == ("agent_2", "agent_finish", 2)
    assert [c["call_id"] for c in json.loads((unit / "CALLS.json").read_text())] == ["default", "agent_1", "agent_2"]
    assert (run_dir / "react" / "REACT_SUMMARY.json").exists()
    trial = json.loads((run_dir / "logs" / "input_trial.json").read_text())
    assert trial["status"] == "done" and trial["tool"]["adata_path"]
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    assert (meta["agent"], meta["tool_budget"], meta["round_cap"], meta["default_counts_as_round"]) == (
        "react", 8, 1, False)
    assert [p["name"] for p in meta["parameters"]] == verifier.levers
    assert meta["fixed_levers"] == {"max_epochs": cap} and meta["training_budget"] == verifier.epoch_budget
    assert meta["react_arm_sha256"] == prep_mod.sha256_file(Path(react_arm.__file__))
    assert meta["shared_react_sha256"] == prep_mod.sha256_file(Path(shared.__file__)) and "llm_arms_sha256" not in meta
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert (agent["n_rounds"], agent["tool_calls"], agent["calls_finished"], agent["requests"]) == (2, 2, 2, 3)
    assert agent["tokens"]["total_tokens"] == 3 * STUB_TOKENS
    assert score.efficiency(run_dir)["tool_calls"] == 3                   # trial 0 counted, as for SHAPE


def test_the_parameters_filter_cuts_only_its_texts_and_a_changed_card_text_raises(monkeypatch):
    """A16: the per-lever filter removes exactly its SHAPE-context texts from the CARD's lever texts; a filtered text
    the CARD no longer holds raises instead of leaking a changed sentence unreviewed."""
    from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier

    verifier = FixedEpochVerifier({"max_epochs": 12, "n_train": 100})
    texts = react_arm.lever_texts(verifier)
    reference = react_arm.parameters_text(verifier)
    for lever, (text, replacement) in react_arm.SHAPE_CONTEXT.items():
        assert text in texts[lever] and texts[lever].replace(text, replacement) in reference
    changed = {**texts, "koleo_weight": texts["koleo_weight"].replace("the alarms", "the readouts")}
    monkeypatch.setattr(react_arm, "lever_texts", lambda v: dict(changed))
    with pytest.raises(ValueError, match="koleo_weight"):
        react_arm.parameters_text(verifier)


def test_react_cap_refusals_failures_and_an_unfinished_session(prepared, stub, llm_env, tmp_path):
    root, _, _ = prepared
    two = {"thought": "t", "action": "call", "calls": [{"params": []}, {"params": []}]}
    engine = FakeEngine([two,                                             # over the cap of 1: an error, a round
                         _call(trainable_pct=500),                       # outside the CARD range: to the tool
                         _call(init_from="agent_9"),                     # not a finished call: refused
                         _call(koleo_weight=0.1),                        # the tool raises: a failed call
                         _call(drop_path=0.1),                           # no resume_best: a failed call
                         _call(),                                        # the tool at its defaults
                         _call(),                                        # the same configuration: runs again
                         _call(max_epochs=60),                           # A6: not a parameter of the arm
                         _finish("agent_4")])                            # a failed call: not deliverable
    run_dir = tmp_path / "react_unfinished"
    final = _react(root, run_dir, engine)
    assert (final["use"], final["case"], final["checkpoint"]) == ("frozen", "default_session_unfinished", None)
    history = _section(engine.prompts[-1], "HISTORY")
    assert "2 calls requested; at most 1 calls per round" in history
    assert "outside [0, 100]" not in history                              # no SHAPE value check
    assert [c["trainable_pct"] for c in stub.calls if c.get("trainable_pct") == 500] == [500]
    assert 'init_from must be \\"pretrained\\" or a finished call id' in history or \
        'init_from must be "pretrained" or a finished call id' in history
    assert "RuntimeError: stub: out of memory" in history and "the call wrote no resume_best" in history
    assert "this configuration was already run" not in history
    again = [c["output_subdir"] for c in stub.calls[-2:]]                  # the defaults twice: two dirs
    assert again[1] == f"{again[0]}_agent_6" and len(set(again)) == 2
    assert "unknown parameters ['max_epochs']" in history
    assert "Rounds used: 8 of 8" in engine.prompts[-1] and "No rounds are left" in engine.prompts[-1]
    ends = [r for r in jsonl(run_dir / "logs" / "tool_calls.jsonl") if r["event"] == "end"]
    assert [r["status"] for r in ends] == ["done", "crashed", "crashed", "done", "done"]   # refused: never reached
    assert all(c["max_epochs"] == stub.calls[0]["max_epochs"] for c in stub.calls)     # A6: every call at E
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert agent["stopped_because"] == "default_session_unfinished" and agent["n_rounds"] == 8
    selected = json.loads((run_dir / "react" / "HEPG2" / "SELECTED.json").read_text())
    assert (selected["source"], selected["call_id"]) == (shared.SOURCE_UNFINISHED, shared.DEFAULT_CALL)


def test_python_inputs_hold_the_embeddings_and_label_free_metadata_never_labels(prepared, stub, llm_env, tmp_path):
    root, _summary, _ = prepared
    code = ("import csv, json\n"
            "print(json.dumps(sorted(e['path'] for e in json.load(open('manifest.json')))))\n"
            "print(json.dumps(next(csv.reader(open('inputs/cell_metadata.csv')))))\n")
    engine = FakeEngine([_call(view_pairing="same_well"), {"thought": "t", "action": "python", "code": code},
                         _finish("agent_1")])
    run_dir = tmp_path / "react_py"
    final = _react(root, run_dir, engine)
    assert final["use"] == "agent_1"
    rows = jsonl(run_dir / "react" / "HEPG2" / "TRACE.jsonl")
    [step] = [r for r in rows if r["kind"] == "python"]
    assert step["python"]["exit"] == 0, step["python"]["output"]
    listed, header = step["python"]["output"].strip().splitlines()
    listed = json.loads(listed)
    assert "inputs/cell_metadata.csv" in listed
    assert any(p.startswith("calls/default/") and p.endswith(".h5ad") for p in listed)   # the frozen trial 0
    assert any(p.startswith("calls/agent_1/") and p.endswith(".h5ad") for p in listed)
    assert json.loads(header) == list(react_arm.PYTHON_META_KEYS)
    assert not any(word in " ".join(listed + json.loads(header)).lower() for word in ("sirna", "label", "private"))
    assert step["python"]["violations"] == []
    assert "Python actions since the last round: 1 of 2" in engine.prompts[2]   # not a round
    assert "Rounds used: 1 of 8" in engine.prompts[2]
    shared.assert_no_private([run_dir / "react_py.csv"], [root / "private"])
    with pytest.raises(AssertionError, match="private"):
        shared.assert_no_private([root / "private" / "labels.csv"], [root / "private"])


def test_react_runs_every_call_at_the_dataset_cap_as_shape_does(prepared, stub, llm_env, tmp_path):
    """A6: every react call (trial 0, a fresh call, a resumed call) runs at the dataset-derived cap SHAPE shares; a
    call at the defaults from pretrained is the analyzer at its defaults with the split and the cap E (A23: no longer
    the dinov3 control's kwargs, which train E x the rounds with no epoch floor on the train wells only); a call from a
    finished call resumes it (A7)."""
    root, summary, _ = prepared
    verifier, _, _ = _verifier(root)
    cap = verifier.epoch_budget["max_epochs"]
    ctl = control.control(root, seeds=(0,), split_key="split")
    n_control = len(stub.calls)
    engine = FakeEngine([_call(), _call(init_from="agent_1"), _finish("agent_2")])
    run_dir = tmp_path / "react_epochs"
    final = _react(root, run_dir, engine)
    assert (final["use"], final["status"]) == ("agent_2", "delivered")
    calls = stub.calls[n_control:]
    assert [kw["max_epochs"] for kw in calls] == [cap, cap, cap]
    assert calls[2]["init_checkpoint"] == score.finished_calls(run_dir)["agent_1"]["resume_best"]
    assert _effective(calls[1]) == _effective({"seed": 0, "split_key": "split", "max_epochs": cap})
    assert ctl["arms"]["dinov3"]["kwargs"] == {"seed": 0, "split_key": "split",
                                               "max_epochs": control.default_epochs(cap), "min_epochs": 1}
    assert len(calls[1]["cell_metadata"]) > ctl["arms"]["dinov3"]["input"]["n_cells"]    # react: train + val
    assert score.moves(run_dir) == {"agent_1": {"parent": "input", "set": {}},
                                    "agent_2": {"parent": "agent_1", "set": {"init_from": "parent"}}}
    from benchmarks.representation_RxRx1.runners.epoch_budget import fixed_line

    assert all(fixed_line(verifier.epoch_budget) in _section(text, "INPUT") for text in engine.prompts)
    n_before = len(stub.calls)                       # SHAPE runs from the same manifest, at the same cap
    shape_dir = tmp_path / "shape_fixed_epochs"
    shape_arm.run(root, shape_dir, cell_type="HEPG2", model="stub", max_rounds=3, engine=_shape_engine(3))
    shape_meta = json.loads((shape_dir / "RUN_META.json").read_text())
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    assert shape_meta["training_budget"] == meta["training_budget"]
    assert shape_meta["fixed_levers"] == meta["fixed_levers"]
    assert all(c["max_epochs"] == cap for c in stub.calls[n_before:])


def test_react_resource_stop_is_an_error_observation(prepared, stub, llm_env, tmp_path, monkeypatch):
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    root, _, _ = prepared
    monkeypatch.setattr(RepresentationAgenticVerifier, "max_minutes", lambda self: 0.0)
    engine = FakeEngine([_call(view_pairing="same_well"), _finish("default")])
    final = _react(root, tmp_path / "react_no_time", engine)
    assert (final["use"], final["case"]) == ("frozen", "finish_input")
    assert len(stub.calls) == 1                                           # trial 0 only
    assert "no call runs: 0.0 minute(s) are left" in engine.prompts[1]


def test_the_resource_stop_is_shapes_rule():
    """`resource_stop` is RepresentationAgenticVerifier.recommend_wave's stop: no time cap -> none; at most 0 minutes,
    or less than the shortest non-training time of a finished call (trial 0's until one finished) -> deadline."""
    from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier

    verifier = FixedEpochVerifier({"max_epochs": 12, "n_train": 100}, log=lambda *a, **k: None)
    for left, overhead, expected in ((None, {0: 5.0}, None), (0.0, {}, "deadline"), (3.0, {0: 5.0}, "deadline"),
                                     (3.0, {0: 5.0, 1: 2.0}, None), (3.0, {0: 1.0, 1: 4.0}, "deadline"),
                                     (3.0, {}, None)):
        verifier.max_minutes = lambda left=left: left
        verifier._overhead = dict(overhead)
        assert react_arm.resource_stop(verifier) == expected, (left, overhead)


def test_a_fatal_engine_reply_fails_the_arm_and_another_failed_reply_uses_its_round(prepared, stub, llm_env,
                                                                                   tmp_path):
    root, _, _ = prepared
    engine = FakeEngine([{"error": "rate_limit", "message": "slow down"}, _finish("default")])
    final = _react(root, tmp_path / "react_rate", engine)
    assert final["case"] == "finish_input" and "Rounds used: 1 of 8" in engine.prompts[1]
    assert "error: RuntimeError: rate_limit: slow down" in _section(engine.prompts[1], "HISTORY")
    run_dir = tmp_path / "react_fatal"
    engine = FakeEngine([{"error": "AuthenticationError", "message": "invalid key"}])
    with pytest.raises(SystemExit, match="REACT_LLM_FATAL: the reasoner engine returned AuthenticationError"):
        _react(root, run_dir, engine)
    assert not (run_dir / "submission" / "decision.json").exists() and not (run_dir / "AGENT_RUN.json").exists()


def test_react_without_an_engine_does_not_start(prepared, stub, tmp_path):
    root, _, _ = prepared
    with pytest.raises(SystemExit, match="NOT_READY"):
        _react(root, tmp_path / "react_none", None)
    assert not (tmp_path / "react_none" / "RUN_META.json").exists()


# ---- the VM resume by replay (A8) -----------------------------------------------------------------------------------

REACT = [_call(view_pairing="same_well"),
         {"error": "rate_limit", "message": "slow down"},                # an error reply: an observation, a round
         _call(init_from="agent_1", trainable_pct=50, view_pairing="same_well"),
         _call(init_from="agent_9"),                                     # invalid: not a finished call
         _finish("agent_2")]


@pytest.fixture()
def vm(monkeypatch):
    monkeypatch.setenv("RXRX_PLATFORM", "vm")
    monkeypatch.setenv("RXRX_RUN_TAG", "testvm-20261010T000000Z")
    monkeypatch.delenv("RXRX_DEADLINE_UTC", raising=False)


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


def _trajectory(run_dir) -> dict:
    """What an interrupted-then-resumed react run must share with an uninterrupted one."""
    def norm(value):
        return json.loads(json.dumps(value, default=str).replace(json.dumps(str(run_dir))[1:-1], "<RUN>"))

    calls = jsonl(run_dir / "logs" / "tool_calls.jsonl")
    decision = json.loads((run_dir / "submission" / "decision.json").read_text())
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    trace = jsonl(run_dir / "react" / "HEPG2" / "TRACE.jsonl")
    return {"decision": norm({k: decision.get(k) for k in ("use", "case", "status", "checkpoint")}),
            "starts": [norm({k: r.get(k) for k in ("call", "parent", "set", "init", "kwargs")})
                       for r in calls if r["event"] == "start"],
            "ends": [norm({k: r.get(k) for k in ("call", "status", "tool")}) for r in calls if r["event"] == "end"],
            "tokens": agent["tokens"]["total_tokens"], "stopped": agent["stopped_because"],
            "rounds": agent["n_rounds"],
            "requests": [(r["request"], r["schema"], r["usage"], r["prompt_masked_sha256"])
                         for r in jsonl(run_dir / "logs" / "reasoner.jsonl")],
            "trace": [(r["round"], r["kind"], r["prompt_sha256"], r["action"]) for r in trace],
            "selected": {k: v for k, v in json.loads((run_dir / "react" / "HEPG2" / "SELECTED.json").read_text())
                         .items() if k not in ("seconds",)}}


@pytest.mark.parametrize("where", ["request", "call"])
def test_react_resumed_on_the_vm_replays_its_replies_and_serves_its_calls(prepared, stub, llm_env, vm, tmp_path,
                                                                          monkeypatch, where):
    root, _, _ = prepared
    reference_engine = FakeEngine(REACT)
    _react(root, tmp_path / "reference", reference_engine)
    reference = _trajectory(tmp_path / "reference")
    assert reference["decision"]["use"] == "agent_2" and len(reference_engine.prompts) == 5

    run_dir = tmp_path / "run"
    if where == "request":                                           # stopped while asking for the invalid call
        engine, logged, served, rerun = KillAt(REACT, at=4), 3, 3, 0
    else:                                                            # stopped inside agent_2
        real = _kill_in_call(stub, monkeypatch, nth=2)
        engine, logged, served, rerun = FakeEngine(REACT), 3, 2, 1
    with pytest.raises(Killed):
        _react(root, run_dir, engine)
    if where == "call":
        monkeypatch.setattr(stub, "execute", real)
    n_before = len(stub.calls)
    engine = FakeEngine(REACT[logged:])
    final = _react(root, run_dir, engine, resume=True)
    assert len(engine.prompts) == len(REACT) - logged                 # no model request for the replayed ones
    assert len(stub.calls) - n_before == rerun                        # finished calls served, the stopped one rerun
    assert final["use"] == "agent_2" and _trajectory(run_dir) == reference
    (entry,) = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    # trial 0 is a finished call too (served)
    assert (entry["logged_requests"], entry["replayed_requests"], len(entry["finished_calls"])) == (
        logged, logged, served)
    if where == "request":
        assert entry["live_from"] == "request 4" and entry["moved_aside"] == []
    else:
        assert entry["live_from"].startswith("call ssl_calls/") and len(entry["moved_aside"]) == 1
    replayed = [bool(r.get("replayed")) for r in jsonl(run_dir / "logs" / "reasoner.jsonl")]
    assert replayed == [True] * logged + [False] * (len(REACT) - logged)
    assert (run_dir / "logs" / "resume_1" / "react" / "HEPG2" / "TRACE.jsonl").exists()   # the stopped run's records
    assert len(jsonl(run_dir / "logs" / "resume_1" / "reasoner.jsonl")) == logged


def test_react_a_resume_stopped_in_its_replay_loses_no_logged_reply(prepared, stub, llm_env, vm, tmp_path,
                                                                    monkeypatch):
    root, _, _ = prepared
    _react(root, tmp_path / "reference", FakeEngine(REACT))
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):                                       # three replies logged, three calls finished
        _react(root, run_dir, KillAt(REACT, at=4))
    real, count = resume.CallCache._serve, {"n": 0}

    def serve(self, *args, **kwargs):                                 # resume 1 stops while it serves agent_1
        count["n"] += 1
        if count["n"] == 2:
            raise Killed()
        return real(self, *args, **kwargs)

    monkeypatch.setattr(resume.CallCache, "_serve", serve)
    first = FakeEngine(REACT[3:])
    with pytest.raises(Killed):
        _react(root, run_dir, first, resume=True)
    monkeypatch.setattr(resume.CallCache, "_serve", real)
    assert first.prompts == []
    engine = FakeEngine(REACT[3:])
    _react(root, run_dir, engine, resume=True)
    assert len(engine.prompts) == 2
    assert _trajectory(run_dir) == _trajectory(tmp_path / "reference")
    _, second = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    assert second["replay_source"] == "logs/resume_1" and second["replayed_requests"] == 3


def test_react_resumed_after_a_python_step_runs_it_again_and_masks_its_observation(prepared, stub, llm_env, vm,
                                                                                    tmp_path):
    root, _, _ = prepared
    code = "import time\nprint(time.time())\n"                          # its output differs in every run
    script = [_call(view_pairing="same_well"), {"thought": "t", "action": "python", "code": code},
              _call(init_from="agent_1", trainable_pct=50), _finish("agent_2")]
    reference = _react(root, tmp_path / "reference", FakeEngine(script))
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):                                       # three replies logged, the python step among them
        _react(root, run_dir, KillAt(script, at=4))
    n_before = len(stub.calls)
    engine = FakeEngine(script[3:])
    final = _react(root, run_dir, engine, resume=True)
    assert len(engine.prompts) == 1 and len(stub.calls) == n_before    # replayed replies, served calls
    assert (final["use"], final["case"]) == (reference["use"], reference["case"]) == ("agent_2", "finish")
    (entry,) = json.loads((run_dir / "RUN_META.json").read_text())["resumes"]
    assert (entry["logged_requests"], entry["replayed_requests"]) == (3, 3)
    trace = jsonl(run_dir / "react" / "HEPG2" / "TRACE.jsonl")
    assert [r["kind"] for r in trace] == ["default", "agent", "python", "agent", "agent"]
    assert trace[2]["python"]["exit"] == 0                            # the python step ran again in the replay
    logged = jsonl(run_dir / "logs" / "resume_1" / "react" / "HEPG2" / "TRACE.jsonl")
    assert logged[2]["python"]["output"] != trace[2]["python"]["output"]
    masked = [r["prompt_masked_sha256"] for r in jsonl(run_dir / "logs" / "reasoner.jsonl")]
    assert masked == [r["prompt_masked_sha256"] for r in jsonl(tmp_path / "reference" / "logs" / "reasoner.jsonl")]


def test_a_replayed_react_prompt_that_is_not_the_logged_one_stops_the_resume(prepared, stub, llm_env, vm, tmp_path):
    root, _, _ = prepared
    run_dir = tmp_path / "run"
    with pytest.raises(Killed):
        _react(root, run_dir, KillAt(REACT, at=4))
    path = run_dir / "logs" / "reasoner.jsonl"
    records = jsonl(path)
    records[1]["prompt_masked_sha256"] = "0" * 64
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(SystemExit, match="REPLAY_DIVERGED: logged request 2: the replayed prompt"):
        _react(root, run_dir, FakeEngine(REACT[3:]), resume=True)
    # the clock line alone is masked
    one, two = "## BUDGET\nTime left: 3.0 min.\nx", "## BUDGET\nTime left: no limit.\nx"
    assert resume.react_masked_digest(one) == resume.react_masked_digest(two) != resume._prompt_digest(one)
    assert resume.react_masked_digest("Rounds used: 1 of 8") != resume.react_masked_digest("Rounds used: 2 of 8")
    # a python step's observation is masked, its code is not
    step = ("## HISTORY\nPython step 1 (after round 1; not a round)\nthought: t\naction: python\nprint(1)\n"
            "observation: python step 1: exit 0 in {s} s; files written: []\noutput:\n{out}\n\nRound 2\n"
            "action: call\n\n## RESPONSE SCHEMA\nx")
    assert (resume.react_masked_digest(step.format(s="0.3", out="1"))
            == resume.react_masked_digest(step.format(s="9.1", out="2")))
    assert (resume.react_masked_digest(step.format(s="0.3", out="1"))
            != resume.react_masked_digest(step.format(s="0.3", out="1").replace("print(1)", "print(2)")))
    assert resume.react_prompt_check(one, {"prompt_masked_sha256": resume.react_masked_digest("other")})
