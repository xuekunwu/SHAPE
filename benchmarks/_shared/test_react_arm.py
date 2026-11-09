"""CPU tests of the shared ReAct runner (react_arm.py) with a fake engine and a fake tool adapter."""
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks._shared import react_arm as ra
from shape.agent.verification.base import BUDGET
from shape.engine import usage as token_usage
from shape.engine.errors import NoEngineError

PARAMS = [ra.Param("diameter", "integer or null", None), ra.Param("threshold", "number", 0.4),
          ra.Param("mode", "choice", "fast")]


class FakeAdapter:
    tool_name = "fake_tool"
    unit_kind = "image"

    def __init__(self, units=("u1",), *, cap=4, counts=True, fail_on=None, sleep=0.0, barrier=None, images=0,
                 root=None, default_sleep=0.0):
        self._units, self._cap, self._counts = list(units), cap, counts
        self.fail_on, self.sleep, self.barrier, self.images, self.root = fail_on, sleep, barrier, images, root
        self.default_sleep = default_sleep
        self.calls, self.delivered = [], {}
        self.lock = threading.Lock()
        self.active, self.max_active = 0, 0

    def parameters(self):
        return list(PARAMS)

    def units(self):
        return list(self._units)

    def round_cap(self):
        return self._cap

    def default_counts_as_round(self):
        return self._counts

    def call(self, unit, params, call_id):
        with self.lock:
            self.calls.append((unit, call_id, dict(params)))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            time.sleep(self.default_sleep if call_id == ra.DEFAULT_CALL else self.sleep)
            if self.fail_on and self.fail_on(params):
                raise ValueError("tool failed on these params")
            images = []
            for i in range(self.images):
                path = Path(self.root) / f"{unit}_{call_id}_{i}.png"
                from PIL import Image
                Image.new("RGB", (40, 30), (i * 10 % 255, 0, 0)).save(path)
                images.append(str(path))
            output = {"unit": unit, "cells": 3, "params": params, "mask": np.zeros((4, 4)),
                      "score": np.float32(0.5), "overlay": str(Path(self.root or ".").resolve() / "o.png")}
            return {"output": output, "images": images, "error": None}
        finally:
            with self.lock:
                self.active -= 1

    def deliver(self, unit, call_id):
        self.delivered[unit] = call_id


def act(action="call", calls=None, deliver=None, thought="t"):
    return ra.ReactAction(thought=thought, action=action, deliver=deliver,
                          calls=[ra.ReactCall(params=[ra.ParamValue(name=k, value=v) for k, v in c.items()])
                                 for c in (calls or [])])


class FakeEngine:
    """Replies by `policy(unit, n, prompt, content)` (n: the unit's reply index); each reply reports `tokens`
    and records them in shape.engine.usage, as the real engines do."""
    def __init__(self, policy, tokens=100, vision=False):
        self.policy, self.tokens, self.is_multimodal = policy, tokens, vision
        self.prompts, self.contents, self.counts = {}, {}, {}
        self.lock = threading.Lock()

    def generate(self, content, response_format=None):
        assert response_format is ra.ReactAction
        prompt = content if isinstance(content, str) else content[0]
        unit = re.search(r"UNIT=(\S+)", prompt).group(1)
        with self.lock:
            n = self.counts.get(unit, 0)
            self.counts[unit] = n + 1
            self.prompts.setdefault(unit, []).append(prompt)
            self.contents.setdefault(unit, []).append(content)
        reply = self.policy(unit, n, prompt, content)
        if isinstance(reply, BaseException):
            raise reply
        usage = {"prompt_tokens": self.tokens - 10, "completion_tokens": 10, "total_tokens": self.tokens}
        token_usage.record(usage)
        return {"content": reply, "usage": usage}


def run(adapter, engine, tmp_path, **kw):
    kw.setdefault("workers", 2)
    kw.setdefault("deadline", None)
    return ra.run_react(adapter, task_text="Segment the cells.", input_text="An image.", engine=engine,
                        out_dir=tmp_path / "react", unit_input=lambda u: f"UNIT={u}", **kw)


def trace(tmp_path, unit="u1"):
    return [json.loads(line) for line in (tmp_path / "react" / unit / "TRACE.jsonl").read_text().splitlines()]


def always_call(unit, n, prompt, content):
    return act(calls=[{"threshold": 0.1 * (n + 1)}])


# ---------------------------------------------------------------- budget

def test_budget_is_shapes_constant():
    import inspect
    assert inspect.signature(ra.run_react).parameters["budget"].default is BUDGET


def test_default_counts_as_round(tmp_path):
    adapter, engine = FakeAdapter(counts=True), FakeEngine(always_call)
    out = run(adapter, engine, tmp_path, budget=3)
    sel = out["selected"]["u1"]
    assert sel["rounds"] == 3                         # the default (round 1) + 2 agent rounds
    assert [c[1] for c in adapter.calls] == ["default", "agent_1", "agent_2"]
    assert engine.counts["u1"] == 3                   # 2 rounds + the finish question after the last round
    rows = trace(tmp_path)
    assert [(r["round"], r["kind"]) for r in rows] == [(1, "default"), (2, "agent"), (3, "agent"), (3, "final")]
    assert "Rounds used: 1 of 3. Rounds left: 2." in engine.prompts["u1"][0]
    assert "Round 1 (the tool at its defaults, run by the system)" in engine.prompts["u1"][0]


def test_default_not_a_round(tmp_path):
    adapter, engine = FakeAdapter(counts=False), FakeEngine(always_call)
    out = run(adapter, engine, tmp_path, budget=3)
    assert out["selected"]["u1"]["rounds"] == 3
    assert [c[1] for c in adapter.calls] == ["default", "agent_1", "agent_2", "agent_3"]
    assert engine.counts["u1"] == 4
    assert [r["round"] for r in trace(tmp_path)] == [0, 1, 2, 3, 3]
    assert "Rounds used: 0 of 3. Rounds left: 3." in engine.prompts["u1"][0]
    assert "Round 0 (the tool at its defaults, run by the system; not a round)" in engine.prompts["u1"][0]


def test_cap_per_round(tmp_path):
    def policy(unit, n, prompt, content):
        return act(calls=[{"threshold": 0.1}, {"threshold": 0.2}, {"threshold": 0.3}]) if n == 0 else \
            act(calls=[{"threshold": 0.1}, {"threshold": 0.2}])
    adapter, engine = FakeAdapter(cap=2), FakeEngine(policy)
    out = run(adapter, engine, tmp_path, budget=3)
    rows = trace(tmp_path)
    assert rows[1]["calls"] == [] and "at most 2 calls per round" in rows[1]["observations"][0]["error"]
    assert [c[1] for c in adapter.calls] == ["default", "agent_1", "agent_2"]   # the over-cap round ran nothing
    assert out["selected"]["u1"]["rounds"] == 3
    assert "at most 2 calls per round" in engine.prompts["u1"][1]
    assert "Calls per round: at most 2." in engine.prompts["u1"][0]


def test_invalid_params_use_a_round(tmp_path):
    def policy(unit, n, prompt, content):
        return act(calls=[{"bogus": 1}, {"diameter": 30.0}]) if n == 0 else act("finish", deliver="agent_1")
    adapter, engine = FakeAdapter(), FakeEngine(policy)
    out = run(adapter, engine, tmp_path, budget=4)
    assert [c[1] for c in adapter.calls] == ["default", "agent_1"]
    assert adapter.calls[1][2] == {"diameter": 30, "threshold": 0.4, "mode": "fast"}   # integral -> int
    assert isinstance(adapter.calls[1][2]["diameter"], int)
    assert "unknown parameters ['bogus']" in engine.prompts["u1"][1]
    sel = out["selected"]["u1"]
    assert sel["rounds"] == 2 and sel["call_id"] == "agent_1" and sel["source"] == ra.SOURCE_FINISH


def test_json_string_value_decoded():
    assert ra._coerce(ra.Param("cell_overrides", "dict", {}), '{"3": {"pad": 2}}') == {"3": {"pad": 2}}
    assert ra._coerce(ra.Param("mode", "choice", "fast"), "slow") == "slow"
    assert ra._coerce(ra.Param("n", "integer", 1), 2.5) == 2.5


# ---------------------------------------------------------------- delivery

def test_finish_delivers(tmp_path):
    def policy(unit, n, prompt, content):
        return act(calls=[{"threshold": 0.2}]) if n == 0 else act("finish", deliver="agent_1")
    adapter, engine = FakeAdapter(), FakeEngine(policy)
    out = run(adapter, engine, tmp_path)
    assert adapter.delivered == {"u1": "agent_1"}
    selected = json.loads((tmp_path / "react" / "u1" / "SELECTED.json").read_text())
    assert selected["source"] == ra.SOURCE_FINISH and selected["call_id"] == "agent_1"
    assert selected["params"]["threshold"] == pytest.approx(0.2)
    calls = json.loads((tmp_path / "react" / "u1" / "CALLS.json").read_text())
    assert [c["call_id"] for c in calls] == ["default", "agent_1"]
    assert out["totals"]["unfinished"] == 0


def test_finish_on_failed_call_is_rejected(tmp_path):
    def policy(unit, n, prompt, content):
        return act(calls=[{"threshold": 0.9}]) if n == 0 else act("finish", deliver="agent_1")
    adapter = FakeAdapter(fail_on=lambda p: p["threshold"] > 0.5)
    engine = FakeEngine(policy)
    out = run(adapter, engine, tmp_path, budget=3)
    assert "error: ValueError: tool failed on these params" in engine.prompts["u1"][1]   # a failed call: observation
    assert "'agent_1' is a call that failed" in engine.prompts["u1"][2]
    assert out["selected"]["u1"]["source"] == ra.SOURCE_UNFINISHED
    assert adapter.delivered == {"u1": "default"}


def test_unfinished_delivers_default(tmp_path):
    adapter, engine = FakeAdapter(), FakeEngine(always_call)
    out = run(adapter, engine, tmp_path, budget=2)
    assert adapter.delivered == {"u1": "default"}
    selected = json.loads((tmp_path / "react" / "u1" / "SELECTED.json").read_text())
    assert selected["source"] == "default_session_unfinished" and selected["call_id"] == "default"
    assert selected["stop"] == "final_not_valid"
    summary = json.loads((tmp_path / "react" / "REACT_SUMMARY.json").read_text())
    assert summary["units"]["u1"]["source"] == "default_session_unfinished"
    assert summary["totals"]["unfinished"] == 1
    assert 'only "finish" is accepted' in trace(tmp_path)[-1]["observations"][0]["error"]


# ---------------------------------------------------------------- concurrency and the deadline

def test_sessions_run_concurrently(tmp_path):
    barrier = threading.Barrier(4, timeout=10)        # the 4 default calls must run at the same time
    adapter = FakeAdapter(units=["a", "b", "c", "d"], barrier=None)
    original = adapter.call

    def call(unit, params, call_id):
        if call_id == "default":
            barrier.wait()
        return original(unit, params, call_id)
    adapter.call = call
    engine = FakeEngine(lambda u, n, p, c: act("finish", deliver="default"))
    out = run(adapter, engine, tmp_path, workers=4)
    assert adapter.delivered == {u: "default" for u in "abcd"}
    assert out["totals"]["units"] == 4 and all(s["source"] == ra.SOURCE_FINISH for s in out["selected"].values())


def test_calls_in_a_round_run_concurrently(tmp_path):
    barrier = threading.Barrier(3, timeout=10)
    adapter = FakeAdapter(cap=3)
    original = adapter.call

    def call(unit, params, call_id):
        if call_id != "default":
            barrier.wait()
        return original(unit, params, call_id)
    adapter.call = call

    def policy(unit, n, prompt, content):
        return act(calls=[{"threshold": 0.1}, {"threshold": 0.2}, {"threshold": 0.3}]) if n == 0 else \
            act("finish", deliver="agent_3")
    out = run(adapter, FakeEngine(policy), tmp_path)
    assert out["selected"]["u1"]["call_id"] == "agent_3" and out["selected"]["u1"]["calls"] == 4


def test_deadline_passed_delivers_default_without_questions(tmp_path):
    adapter, engine = FakeAdapter(), FakeEngine(always_call)
    out = run(adapter, engine, tmp_path, deadline=time.time() - 1)
    assert engine.counts == {} and adapter.delivered == {"u1": "default"}
    assert out["selected"]["u1"]["stop"] == "deadline"
    assert out["selected"]["u1"]["source"] == ra.SOURCE_UNFINISHED


class FakeClock:
    """The runner's clock (`react_arm.time`): it moves only when `advance` is called, never by wall time."""
    def __init__(self, now=1000.0):
        self.now = now

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_deadline_stops_mid_session(tmp_path, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(ra, "time", clock)
    adapter, engine = FakeAdapter(), FakeEngine(always_call)
    original = adapter.call

    def call(unit, params, call_id):                 # each agent call takes 10 s of the clock; the default none
        if call_id != ra.DEFAULT_CALL:
            clock.advance(10)
        return original(unit, params, call_id)
    adapter.call = call
    out = run(adapter, engine, tmp_path, deadline=clock.time() + 5)
    assert engine.counts["u1"] == 1                  # the call of round 2 ran past the deadline: no more questions
    assert out["selected"]["u1"]["stop"] == "deadline" and adapter.delivered == {"u1": "default"}
    assert "Time left:" in engine.prompts["u1"][0] and "no limit" not in engine.prompts["u1"][0]


# ---------------------------------------------------------------- failures

def test_no_engine_fails_the_arm(tmp_path):
    with pytest.raises(NoEngineError):
        run(FakeAdapter(), None, tmp_path)


def test_basic_llm_failure_propagates(tmp_path):
    engine = FakeEngine(lambda u, n, p, c: NoEngineError("gone"))
    with pytest.raises(NoEngineError):
        run(FakeAdapter(units=["a", "b"]), engine, tmp_path)


def test_basic_llm_failure_in_a_chain_propagates(tmp_path):
    openai = pytest.importorskip("openai")
    import httpx
    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")

    def policy(u, n, p, c):
        wrapped = RuntimeError("gateway down")
        wrapped.__cause__ = openai.APIConnectionError(request=request)
        return wrapped
    with pytest.raises(RuntimeError, match="gateway down"):
        run(FakeAdapter(), FakeEngine(policy), tmp_path)


def test_malformed_reply_recorded_and_uses_a_round(tmp_path):
    def policy(unit, n, prompt, content):
        return "this is not json" if n == 0 else act("finish", deliver="default")
    adapter, engine = FakeAdapter(), FakeEngine(policy)
    out = run(adapter, engine, tmp_path)
    rows = trace(tmp_path)
    assert rows[1]["action"] is None and rows[1]["reply"]["content"] == "this is not json"
    assert "error" in rows[1]["observations"][0]
    assert "action: malformed reply" in engine.prompts["u1"][1]
    assert out["selected"]["u1"]["rounds"] == 2 and out["selected"]["u1"]["source"] == ra.SOURCE_FINISH


def test_nonfatal_engine_error_is_an_observation(tmp_path):
    def policy(unit, n, prompt, content):
        return TimeoutError("slow gateway") if n == 0 else act("finish", deliver="default")
    out = run(FakeAdapter(), FakeEngine(policy), tmp_path)
    assert "TimeoutError: slow gateway" in trace(tmp_path)[1]["observations"][0]["error"]
    assert out["selected"]["u1"]["source"] == ra.SOURCE_FINISH


# ---------------------------------------------------------------- tokens, records, prompt

def test_token_accounting(tmp_path):
    adapter = FakeAdapter(units=["a", "b"])
    out = run(adapter, FakeEngine(always_call, tokens=100), tmp_path, budget=3)
    for unit in ("a", "b"):
        tokens = out["selected"][unit]["tokens"]
        assert tokens["replies"] == 3 and tokens["reported"] == 3 and tokens["total_tokens"] == 300
        assert tokens["prompt_tokens"] == 270 and tokens["completion_tokens"] == 30
    assert out["totals"]["tokens"] == 600 and out["totals"]["replies"] == 6
    assert out["totals"]["process_usage_tokens"] == 600 and out["totals"]["process_usage_calls"] == 6
    assert out["totals"]["rounds"] == 6 and out["totals"]["calls"] == 6


def test_trace_records(tmp_path):
    def policy(unit, n, prompt, content):
        return act(calls=[{"threshold": 0.2}]) if n == 0 else act("finish", deliver="agent_1")
    run(FakeAdapter(root=tmp_path), FakeEngine(policy), tmp_path)
    rows = trace(tmp_path)
    assert rows[0]["prompt_sha256"] is None and rows[0]["calls"][0]["call_id"] == "default"
    assert all(re.fullmatch(r"[0-9a-f]{64}", r["prompt_sha256"]) for r in rows[1:])
    assert rows[1]["calls"] == [{"call_id": "agent_1", "params": {"diameter": None, "threshold": 0.2,
                                                                  "mode": "fast"}}]
    assert rows[1]["action"]["action"] == "call" and rows[2]["action"]["deliver"] == "agent_1"
    calls = json.loads((tmp_path / "react" / "u1" / "CALLS.json").read_text())
    output = calls[1]["output"]
    assert "mask" not in output and output["score"] == pytest.approx(0.5)     # arrays dropped, scalars kept
    assert output["overlay"] == "o.png"                                        # absolute paths made relative


def test_images_attached_for_vision_engines_at_most_16(tmp_path):
    seen = {}

    def policy(unit, n, prompt, content):
        seen[n] = content
        return act(calls=[{"threshold": 0.2}]) if n == 0 else act("finish", deliver="agent_1")
    adapter = FakeAdapter(images=20, root=tmp_path)
    run(adapter, FakeEngine(policy, vision=True), tmp_path)
    for content in seen.values():
        assert isinstance(content, list) and sum(isinstance(x, bytes) for x in content) == 16
        assert content[1].startswith("IMAGE ")
    seen.clear()
    run(FakeAdapter(images=3, root=tmp_path), FakeEngine(policy, vision=False), tmp_path / "plain")
    assert all(isinstance(c, str) for c in seen.values())


def test_prompt_holds_only_the_allowed_sections():
    history = [{"round": 1, "note": "the tool at its defaults, run by the system", "thought": None,
                "action_text": "call default", "observations": ["call default params {}: output: {}"]}]
    prompt = ra.render_prompt(task_text="TASK TEXT", input_text="INPUT TEXT", tool_name="fake_tool",
                              parameters=PARAMS, history=history, rounds_used=1, budget=BUDGET, round_cap=4,
                              deadline=None)
    assert re.findall(r"^## (.+)$", prompt, re.MULTILINE) == list(ra.SECTIONS)
    assert ra.TUNING_LINE in prompt and prompt.count(ra.TUNING_LINE) == 1
    assert "- diameter | integer or null | null" in prompt and "- threshold | number | 0.4" in prompt
    assert "- mode | choice | \"fast\"" in prompt
    for banned in ("Range", "range:", "Earlier runs", "earlier runs", "admissible", "CARD", "guard", "verdict",
                   "Description", "threshold at"):
        assert banned not in prompt, banned
    assert f"Rounds used: 1 of {BUDGET}. Rounds left: {BUDGET - 1}." in prompt and "Time left: no limit." in prompt


def test_runner_never_reads_card_files():
    source = Path(ra.__file__).read_text(encoding="utf-8")
    assert "CARD.md" not in source and "card_lever" not in source and "tuning_card" not in source


# ---------------------------------------------------------------- input images, trial-0 label, TIFF rendering

def test_unit_images_attached_to_the_first_question_only_inside_the_cap(tmp_path):
    from PIL import Image
    import io
    big = tmp_path / "raw_big.png"
    Image.new("RGB", (3000, 200), (0, 200, 0)).save(big)
    small = tmp_path / "raw_small.png"
    Image.new("L", (50, 40), 128).save(small)
    seen = {}

    def policy(unit, n, prompt, content):
        seen[n] = content
        return act(calls=[{"threshold": 0.2}]) if n == 0 else act("finish", deliver="agent_1")
    adapter = FakeAdapter(images=20, root=tmp_path)
    run(adapter, FakeEngine(policy, vision=True), tmp_path, unit_images=lambda u: [str(big), str(small)])
    first, second = seen[0], seen[1]
    labels = [x for x in first if isinstance(x, str)][1:]
    assert labels[:2] == ["INPUT IMAGE: raw_big.png", "INPUT IMAGE: raw_small.png"]
    assert all(x.startswith("IMAGE default: ") for x in labels[2:])
    assert sum(isinstance(x, bytes) for x in first) == ra.MAX_IMAGES       # 2 inputs + 14 tool images
    with Image.open(io.BytesIO(first[2])) as img:
        assert max(img.size) == ra.IMAGE_PX                               # downscaled like the others
    assert not any(isinstance(x, str) and x.startswith("INPUT IMAGE") for x in second)
    assert sum(isinstance(x, bytes) for x in second) == ra.MAX_IMAGES
    rows = trace(tmp_path)
    assert [r["label"] for r in rows[1]["images"][:2]] == labels[:2]
    assert not any(r["label"].startswith("INPUT IMAGE") for r in rows[2]["images"])


def test_unit_images_not_attached_for_text_engines(tmp_path):
    from PIL import Image
    raw = tmp_path / "raw.png"
    Image.new("L", (10, 10)).save(raw)
    engine = FakeEngine(lambda u, n, p, c: act("finish", deliver="default"), vision=False)
    run(FakeAdapter(), engine, tmp_path, unit_images=lambda u: [str(raw)])
    assert all(isinstance(c, str) for c in engine.contents["u1"])


@pytest.mark.parametrize("counts, text", [(True, "Round 1 (the uncorrected input)"),
                                          (False, "Round 0 (the uncorrected input; not a round)")])
def test_trial0_label_comes_from_the_adapter(tmp_path, counts, text):
    adapter = FakeAdapter(counts=counts)
    adapter.trial0_label = "the uncorrected input"
    engine = FakeEngine(lambda u, n, p, c: act("finish", deliver="default"))
    run(adapter, engine, tmp_path)
    assert text in engine.prompts["u1"][0] and ra.TRIAL0_LABEL not in engine.prompts["u1"][0]


def test_tiff_rendered_as_shapes_vision_reads(tmp_path):
    import io
    import tifffile
    from PIL import Image
    rng = np.random.default_rng(0)
    chw = rng.integers(0, 4000, (3, 30, 50)).astype(np.uint16)              # an ImageJ channel-first crop
    tifffile.imwrite(tmp_path / "c3.tif", chw, imagej=True)
    tifffile.imwrite(tmp_path / "c1.tif", chw[0])
    with Image.open(io.BytesIO(ra._png(str(tmp_path / "c3.tif"), 1024))) as img:
        rgb = np.asarray(img)
        assert img.mode == "RGB" and img.size == (50, 30)
    assert not (np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]))  # not one grey
    with Image.open(io.BytesIO(ra._png(str(tmp_path / "c1.tif"), 1024))) as img:
        assert img.mode == "L" and img.size == (50, 30)
        assert np.asarray(img).max() == 255 and np.asarray(img).min() == 0   # uint16 percentile-stretched
    with Image.open(io.BytesIO(ra._png(str(tmp_path / "c1.tif"), 20))) as img:
        assert max(img.size) == 20                                         # downscaled to image_px


# ---------------------------------------------------------------- python steps

class PyAdapter(FakeAdapter):
    """FakeAdapter whose calls write a mask file; `python_inputs` lists the input and each finished call's mask."""
    def __init__(self, root, **kw):
        super().__init__(root=root, **kw)
        self.inputs = Path(root) / "inputs_src"
        self.inputs.mkdir(parents=True, exist_ok=True)
        np.save(self.inputs / "image.npy", np.arange(12).reshape(3, 4))
        self.masks = {}

    def call(self, unit, params, call_id):
        out = super().call(unit, params, call_id)
        path = Path(self.root) / f"{unit}_{call_id}_mask.npy"
        np.save(path, np.ones((3, 4)) * params["threshold"])
        self.masks[call_id] = path
        return out

    def python_inputs(self, unit):
        paths = {"inputs/image.npy": self.inputs / "image.npy"}
        paths.update({f"calls/{cid}/mask.npy": p for cid, p in self.masks.items()})
        ra.assert_no_private(paths.values(), [Path(self.root) / "labels"])
        return paths


def py(code, thought="py"):
    return ra.ReactAction(thought=thought, action="python", code=code)


def test_python_step_runs_reads_inputs_and_is_not_a_round(tmp_path):
    code = ("import json, numpy as np, os\n"
            "m = json.load(open('manifest.json'))\n"
            "print(sorted(e['path'] for e in m))\n"
            "print(round(float(np.load('calls/agent_1/mask.npy').mean()), 3), np.load('inputs/image.npy').sum())\n"
            "open(os.path.join(os.environ['SCRATCH_DIR'], 'sig.txt'), 'w').write('x')\n")

    def policy(unit, n, prompt, content):
        return [act(calls=[{"threshold": 0.2}]), py(code), act("finish", deliver="agent_1")][n]
    engine = FakeEngine(policy)
    out = run(PyAdapter(tmp_path), engine, tmp_path, budget=4)
    sel = out["selected"]["u1"]
    assert sel["rounds"] == 2 and sel["python_steps"] == 1 and sel["call_id"] == "agent_1"   # python: not a round
    rows = trace(tmp_path)
    step = rows[2]
    assert step["kind"] == "python" and step["round"] == 2 and step["python"]["exit"] == 0
    assert "['calls/agent_1/mask.npy', 'calls/default/mask.npy', 'inputs/image.npy']" in step["python"]["output"]
    assert "0.2 66" in step["python"]["output"]
    assert step["python"]["files"] == ["scratch/py_1/sig.txt"] and step["python"]["violations"] == []
    assert step["python"]["code"] == code
    prompt = engine.prompts["u1"][2]
    assert "Python step 1 (after round 2; not a round)" in prompt and "exit 0" in prompt and "0.2 66" in prompt
    assert "Rounds used: 2 of 4." in prompt and "Python actions since the last round: 1 of 2." in prompt
    manifest = json.loads((tmp_path / "react" / "u1" / "workspace" / "manifest.json").read_text())
    assert {e["path"]: e["shape"] for e in manifest}["inputs/image.npy"] == [3, 4]
    assert out["totals"]["python_steps"] == 1 and out["units"]["u1"]["python_steps"] == 1


def test_python_env_is_scrubbed(tmp_path, monkeypatch):
    for name in ("LLM_API_KEY", "OPENAI_BASE_URL", "ANTHROPIC_X", "MY_SECRET_KEY", "GH_TOKEN"):
        monkeypatch.setenv(name, "s3cr3t")
    monkeypatch.setenv("HARMLESS_VAR", "kept")
    code = "import os\nprint(sorted(k for k, v in os.environ.items() if v == 's3cr3t'), os.environ.get('HARMLESS_VAR'))"
    engine = FakeEngine(lambda u, n, p, c: [py(code), act("finish", deliver="default")][n])
    run(PyAdapter(tmp_path), engine, tmp_path)
    assert "[] kept" in trace(tmp_path)[1]["python"]["output"]
    assert "LLM_API_KEY" not in ra.scrubbed_env({"LLM_API_KEY": "x", "PATH": "p"})
    assert ra.scrubbed_env({"api_token": "x", "PATH": "p"}) == {"PATH": "p"}


def test_python_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "PY_TIMEOUT", 8)
    engine = FakeEngine(lambda u, n, p, c: [py("import time\nprint('start', flush=True)\ntime.sleep(60)"),
                                            act("finish", deliver="default")][n])
    started = time.time()
    run(PyAdapter(tmp_path), engine, tmp_path)
    assert time.time() - started < 50
    step = trace(tmp_path)[1]["python"]
    assert step["timed_out"] and step["exit"] is None and "start" in step["output"]
    assert "timed out after" in engine.prompts["u1"][1]


def test_python_output_truncated(tmp_path):
    engine = FakeEngine(lambda u, n, p, c: [py("print('A' * 20000)\nprint('TAIL')"),
                                            act("finish", deliver="default")][n])
    run(PyAdapter(tmp_path), engine, tmp_path)
    output = trace(tmp_path)[1]["python"]["output"]
    assert len(output) <= ra.PY_OUTPUT_CHARS and "characters cut" in output and output.rstrip().endswith("TAIL")
    assert ra._truncate("abc", 10) == "abc"


def test_python_per_round_limit(tmp_path):
    def policy(unit, n, prompt, content):
        return [py("print(1)"), py("print(2)"), py("print(3)"), py("print(4)"), act("finish", deliver="default")][n]
    engine = FakeEngine(policy)
    out = run(PyAdapter(tmp_path), engine, tmp_path, budget=4)
    rows = trace(tmp_path)
    assert [r["kind"] for r in rows] == ["default", "python", "python", "agent", "python", "agent"]
    assert "at most 2 python actions between two rounds" in rows[3]["observations"][0]["error"]
    sel = out["selected"]["u1"]
    assert sel["python_steps"] == 3 and sel["rounds"] == 2           # the refused third used a round
    assert "Python actions since the last round: 0 of 2." in engine.prompts["u1"][3]   # reset by the round


def test_python_after_the_last_round_then_finish(tmp_path):
    def policy(unit, n, prompt, content):
        return [act(calls=[{"threshold": 0.3}]), py("print('x')"), act("finish", deliver="agent_1")][n]
    out = run(PyAdapter(tmp_path), FakeEngine(policy), tmp_path, budget=2)
    sel = out["selected"]["u1"]
    assert sel["call_id"] == "agent_1" and sel["source"] == ra.SOURCE_FINISH and sel["python_steps"] == 1


def test_python_respects_the_deadline(tmp_path):
    engine = FakeEngine(lambda u, n, p, c: py("import time\ntime.sleep(60)"))
    started = time.time()
    out = run(PyAdapter(tmp_path), engine, tmp_path, deadline=time.time() + 3)
    assert time.time() - started < 50
    sel = out["selected"]["u1"]
    assert sel["stop"] == "deadline" and sel["python_steps"] == 1 and trace(tmp_path)[1]["python"]["timed_out"]


def test_python_violation_detected(tmp_path):
    code = ("import os, numpy as np\n"
            "os.chmod('calls/default/mask.npy', 0o666)\n"
            "np.save('calls/default/mask.npy', np.zeros(2))\n"
            "open('outside.txt', 'w').write('x')\n")
    engine = FakeEngine(lambda u, n, p, c: [py(code), act("finish", deliver="default")][n])
    out = run(PyAdapter(tmp_path), engine, tmp_path)
    violations = trace(tmp_path)[1]["python"]["violations"]
    assert any(v.startswith("calls/default/mask.npy") for v in violations)
    assert "outside.txt: written outside scratch/" in violations
    assert out["selected"]["u1"]["python_violations"] == len(violations) and out["totals"]["python_violations"] > 0
    assert "violations:" in engine.prompts["u1"][1]


def test_assert_no_private(tmp_path):
    (tmp_path / "labels").mkdir()
    ra.assert_no_private([tmp_path / "images" / "a.tif"], [tmp_path / "labels"])
    with pytest.raises(AssertionError, match="private"):
        ra.assert_no_private([tmp_path / "labels" / "a_label.tiff"], [tmp_path / "labels"])
    with pytest.raises(AssertionError):
        ra.assert_no_private([tmp_path / "labels"], [tmp_path / "labels"])


def test_private_input_fails_the_arm(tmp_path):
    adapter = PyAdapter(tmp_path)
    (tmp_path / "labels").mkdir()
    np.save(tmp_path / "labels" / "gt.npy", np.zeros(2))
    adapter.masks["leak"] = tmp_path / "labels" / "gt.npy"
    with pytest.raises(AssertionError, match="private"):
        run(adapter, FakeEngine(lambda u, n, p, c: py("print(1)")), tmp_path)


def test_python_section_in_the_prompt():
    prompt = ra.render_prompt(task_text="T", input_text="I", tool_name="fake_tool", parameters=PARAMS, history=[],
                              rounds_used=0, budget=BUDGET, round_cap=4, deadline=None)
    section = prompt.split("## PYTHON\n", 1)[1].split("\n## ", 1)[0]
    for fact in ("inputs/", "calls/<call id>/", "manifest.json", "scratch/py_<n>/", "SCRATCH_DIR",
                 f"{ra.PY_TIMEOUT} s", f"{ra.PY_OUTPUT_CHARS} characters", "It is not a round", "label-free"):
        assert fact in section, fact
    assert "python" in ra.ReactAction.model_json_schema()["properties"]["action"]["enum"]
