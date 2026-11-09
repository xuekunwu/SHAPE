"""Task 032 (VRR): unit tests for the lever-ledger classification in `base.py`, and for the
shared agentic VRR (`base.AgenticVerifier`) on a toy tool run through StepVerifier.

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not in the
gitignored `tests/` dir.
"""

import json
from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic import BaseModel

from shape.agent.verification import base
from shape.agent.verification.base import (
    _ledger_materialized, classify_lever, moved_as_expected,
)
from shape.agent.verification.step_verifier import StepVerifier
from shape.utils.schemas import AdmissionAnswer, LeverChange, VLRRResult, WaveMove, WavePlan


def test_ledger_materialized_is_a_stricter_floor_used_only_for_the_effective_gate():
    # task 021's actual numbers: silhouette_by_batch 0.503 -> 0.499, predicted "down".
    # The bare stop-gate epsilon (moved_as_expected) calls this "moved" -- and (pre-merge
    # fix, QC N1) so does the ledger's `inert` gate now (same epsilon, see base.py):
    # this move is real, just too small (~0.8%) to CREDIT as "effective" -- that stricter
    # judgment is what `_ledger_materialized` is for now (see `classify_lever`).
    assert moved_as_expected(0.503, 0.499, "down") is True
    assert _ledger_materialized(0.503, 0.499, "down") is False


def test_ledger_materialized_true_for_a_real_relative_move():
    assert _ledger_materialized(0.30, 0.05, "down") is True
    assert _ledger_materialized(1.0, 5.0, "up") is True


def test_classify_lever_effective_when_materialized_and_score_improves():
    prev = VLRRResult("WARNING", 0.50, {"tiny_mask_fraction": 0.30}, action="retune_here",
                      param_delta={"min_size": "up"}, expected_effect={"tiny_mask_fraction": "down"})
    cur = VLRRResult("PASS", 0.85, {"tiny_mask_fraction": 0.05}, action="accept")
    assert classify_lever(prev, cur) == "effective"


def test_classify_lever_no_headroom_when_materialized_but_score_flat():
    prev = VLRRResult("WARNING", 0.60, {"tiny_mask_fraction": 0.30}, action="retune_here",
                      param_delta={"min_size": "up"}, expected_effect={"tiny_mask_fraction": "down"})
    cur = VLRRResult("WARNING", 0.58, {"tiny_mask_fraction": 0.05}, action="accept")
    assert classify_lever(prev, cur) == "no-headroom"


def test_classify_lever_no_headroom_when_move_is_real_but_too_small_to_credit():
    # the 021 field_aware numbers, reproduced directly against classify_lever. Pre-merge
    # fix (QC N1): `inert` now means "didn't even clear the stop-gate's own bare epsilon"
    # (it did move here, just ~0.8%, below the stricter 2% floor `_ledger_materialized`
    # uses for the effective/no-headroom split) -- so this is no longer mislabeled
    # `inert`; it's a small-but-real, uncredited move.
    prev = VLRRResult("WARNING", 0.60, {"silhouette_by_batch": 0.503}, action="retune_here",
                      param_delta={"field_aware": True},
                      expected_effect={"silhouette_by_batch": "down"})
    cur = VLRRResult("PASS", 0.61, {"silhouette_by_batch": 0.499}, action="accept")
    assert classify_lever(prev, cur) == "no-headroom"


def test_classify_lever_inert_when_move_does_not_clear_the_bare_stop_gate_epsilon():
    # aligned to the stop-gate: predicted "down" but the metric didn't move at all
    # (below even the bare 1e-6 epsilon) -- no measurable change, consistently inert.
    prev = VLRRResult("WARNING", 0.60, {"m": 0.500}, action="retune_here",
                      param_delta={"lever": "down"}, expected_effect={"m": "down"})
    cur = VLRRResult("WARNING", 0.60, {"m": 0.5000001}, action="retune_here",
                     param_delta={"lever": "down"})
    assert classify_lever(prev, cur) == "inert"


def test_classify_lever_none_when_no_lever_or_no_prediction():
    no_lever = VLRRResult("PASS", 0.9, {"m": 1.0}, action="accept")
    cur = VLRRResult("PASS", 0.9, {"m": 1.0}, action="accept")
    assert classify_lever(no_lever, cur) is None

    no_prediction = VLRRResult("WARNING", 0.5, {"m": 1.0}, action="retune_here",
                               param_delta={"diameter": "down"})   # no expected_effect
    assert classify_lever(no_prediction, cur) is None


@pytest.mark.parametrize(("before", "after"), [({}, {}), ({"count": 1}, {"coverage": 1}),
                                                ({"count": 1, "coverage": 1}, {"count": 2}),
                                                ({"count": 1, "coverage": 1}, {"count": float("nan"), "coverage": 1})])
def test_classify_lever_requires_complete_finite_prediction_evidence(before, after):
    prev = VLRRResult("PASS", .5, before, param_delta={"threshold": -2},
                      expected_effect={"count": "up", "coverage": "preserve"})
    current = VLRRResult("PASS", .8, after)
    assert classify_lever(prev, current) is None


def test_classify_lever_adaptive_floors_replace_the_relative_rule_for_noisy_readouts():
    # overall 0.55 -> 0.56: below the 2% relative floor, but five times the metric's noise floor
    prev = VLRRResult("PASS", 0.55, {"overall": 0.55}, param_delta={"nclust": "up"},
                      expected_effect={"overall": "up"})
    cur = VLRRResult("PASS", 0.56, {"overall": 0.56})
    assert classify_lever(prev, cur) == "no-headroom"
    assert classify_lever(prev, cur, noise={"overall": 0.002}, score_noise=0.002) == "effective"
    # a change inside the noise floor is not a measurable move
    within = VLRRResult("PASS", 0.551, {"overall": 0.551})
    assert classify_lever(prev, within, noise={"overall": 0.002}, score_noise=0.002) == "inert"



def test_classify_lever_preserve_with_a_floor_holds_only_within_the_floor():
    prev = VLRRResult("PASS", 0.55, {"overall": 0.55, "bio_mean": 0.60}, param_delta={"theta": "up"},
                      expected_effect={"bio_mean": "preserve"})
    dropped = VLRRResult("PASS", 0.56, {"overall": 0.56, "bio_mean": 0.51})
    kept = VLRRResult("PASS", 0.56, {"overall": 0.56, "bio_mean": 0.601})
    assert classify_lever(prev, dropped) != "inert"                     # the 20 % relative rule tolerates it
    assert classify_lever(prev, dropped, noise={"bio_mean": 0.002}, score_noise=0.002) == "inert"
    assert classify_lever(prev, kept, noise={"bio_mean": 0.002}, score_noise=0.002) == "effective"


# ---- the shared agentic VRR (`base.AgenticVerifier`) on a toy tool ------------------------------------------

TOY_CARD = """# Toy tool
## Levers
### alpha
Type: number | Range: [0, 10] | Default: 1
### mode
Type: choice | Range: a, b, null | Default: a
### flag
Type: bool | Range: true, false | Default: false
### count
Type: integer or null | Range: [1, 5] | Default: null
## Other
### ignored
Type: number | Range: [0, 1] | Default: 0
"""


@pytest.fixture()
def toy_card(monkeypatch):
    """The toy tool's CARD for the CARD readers (`card_lever_fields`, `tuning_card`); no retry wait."""
    monkeypatch.setattr("shape.tools.base.load_tuning_card", lambda tool_dir: TOY_CARD)
    monkeypatch.setattr(base.BaseVerifier, "tuning_card", lambda self: TOY_CARD)
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)


class Toy(base.AgenticVerifier):
    """quality = 0.5 + 0.02 * min(alpha, 5) (the input: 0.5); `flag` lowers `side` below the guard's 0.5;
    a null `mode` crashes the tool. Trial 0 takes the cores and the brief from the stage's `task`."""
    name = "toy"
    tool_dir = "toy_tool"
    threads_per_call = 2
    score_key = "quality"
    key_readouts = ("quality", "side")
    objectives = ("quality", "side")
    tool_phrase = "toy tool"
    unit = "task"

    def quantify(self, tool_output, context):
        return dict(tool_output.get("metrics") or {})

    def localize(self, metrics, context):
        return "this_step", [], []

    def adjudicate(self, metrics, locus, triggered, context):
        if context.get("trial_index") == 0:
            self.begin(context.get("task"))
        return VLRRResult("PASS", float(metrics.get("quality", 0.0)), dict(metrics), action="retune_here")

    def guard(self, inp, vres):
        return [] if (vres.metrics or {}).get("side", 0) >= 0.5 else ["side below 0.5"]

    def fallback_moves(self, inp, attempts, add):
        best = self._incumbent(inp, attempts)
        add(best, [{"lever": "alpha", "value": json.dumps(self._params(attempts, best)["alpha"] + 1)}], "alpha up")


class KeepChoice(BaseModel):
    call: Optional[str] = None
    reasoning: str = ""


class Keeper(Toy):
    """A tool whose input is itself a valid output: the final choice may keep it."""
    keep_input = True
    choice_schema = KeepChoice


class Delivering(Toy):
    """Hands the chosen output on (`deliver`); with no admissible call, the best-scoring one (`least_bad`)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.delivered = []

    def least_bad(self, inp, attempts):
        return self.fallback_choice(inp, attempts, pool=list(attempts))

    def deliver(self, state, index, details):
        self.delivered.append((index, details.get("choice")))


def toy_tool(calls=None, quality=None):
    def run(params):
        if calls is not None:
            calls.append(dict(params))
        if params.get("mode") is None:
            raise RuntimeError("mode null crashes")
        score = quality(params) if quality else 0.5 + 0.02 * min(params["alpha"], 5)
        return {"metrics": {"quality": score, "side": 0.4 if params["flag"] else 0.6}}
    return run


def flat(params):
    return 0.5                                          # no call beats the input


class ToyEngine:
    """Answers each round question with the next plan (then no moves: satisfied), a repair question with
    `repair`, the final question with the next of `final`, the admission question with `admission` (a
    severity); records (schema name, prompt) of every question."""

    def __init__(self, rounds=(), final=(), repair=(), admission="none"):
        self.rounds, self.final, self.repair, self.asked = list(rounds), list(final), list(repair), []
        self.admission = admission

    def generate(self, content, response_format=None):
        self.asked.append((response_format.__name__, content[0]))
        if response_format is AdmissionAnswer:
            problem = "none" if self.admission == "none" else "masks miss half the objects"
            return {"content": AdmissionAnswer(problem=problem, severity=self.admission, reasoning="r")}
        if response_format is not WavePlan:
            return {"content": response_format(call=self.final.pop(0), reasoning="r")}
        if "were rejected" in content[0]:
            return {"content": WavePlan(analysis="repair", moves=self.repair)}
        return {"content": WavePlan(analysis="a", moves=self.rounds.pop(0) if self.rounds else [])}

    def questions(self, schema="WavePlan"):
        return [text for name, text in self.asked if name == schema]


class Down:
    """A reasoner that never answers (a gateway down)."""

    def generate(self, content, response_format=None):
        raise TimeoutError("gateway down")


def mv(parent="input", **changes):
    return WaveMove(parent=parent, changes=[LeverChange(lever=k, value=json.dumps(v)) for k, v in changes.items()])


def run_toy(engine, cls=Toy, budget=8, cores=4, calls=None, quality=None, log=None):
    verifier = cls(log=log)
    verifier.llm_engine = engine
    events = []
    task = SimpleNamespace(cores=cores, deadline=None, brief="tune the toy")
    resolution = StepVerifier(max_iters=1 + budget, observer=events.append).resolve_stage(
        verifier, verifier.make_adapter({}), toy_tool(calls, quality), {}, {"metrics": {"quality": 0.5, "side": 0.6}},
        upstream_ctx={"task": task})
    return verifier, resolution, [e["size"] for e in events if e["event"] == "wave"]


def _toy_input():
    return VLRRResult("PASS", 0.5, {"quality": 0.5, "side": 0.6}, action="retune_here")


def test_the_levers_their_defaults_and_values_come_from_the_card(toy_card):
    toy = Toy()
    assert toy.levers == ["alpha", "mode", "flag", "count"]             # the lever section's, in its order
    assert toy.defaults() == {"alpha": 1.0, "mode": "a", "flag": False, "count": None}
    assert toy.lever_spec("mode") == ("choice", ("a", "b", None), True)
    assert toy.lever_spec("count") == ("integer", (1.0, 5.0), True)
    assert [toy.coerce("alpha", v) for v in ("3", ".5", "1e-1", 2)] == [3.0, 0.5, 0.1, 2.0]
    assert toy.coerce("mode", "b") == toy.coerce("mode", '"b"') == "b" and toy.coerce("mode", "null") is None
    assert toy.coerce("flag", "true") is True and toy.coerce("count", "2.6") == 3 and toy.coerce("count", None) is None
    for lever, value, message in (("alpha", "11", "outside"), ("alpha", ":", "is not a JSON value"),
                                  ("alpha", "null", "cannot be null"), ("alpha", "true", "is not a number"),
                                  ("flag", "2", "is not true or false"), ("mode", "c", "must be one of"),
                                  ("ignored", "0", "is not a CARD lever")):
        with pytest.raises(ValueError, match=message):
            toy.coerce(lever, value)
    # a null for a value that cannot be null says how to write one (a reasoner wrote null three times for 1e-05)
    # the example is a value the lever accepts: its lower bound when it has bounds
    with pytest.raises(ValueError, match='alpha cannot be null: write its value inside the value string, '
                                         'e.g. "value": "0"'):
        toy.coerce("alpha", None)
    for kind, bounds, example in (("number", None, "1e-05"), ("number", (1e-6, 1e-4), "1e-06"),
                                  ("integer", None, "60"), ("integer", (1.0, 5.0), "1"), ("bool", None, "true"),
                                  ("choice", ("same_crop", "same_site"), "same_crop")):
        with pytest.raises(ValueError, match=f'x cannot be null: .*e.g. "value": "{example}"$'):
            toy.parse_value("x", None, kind, bounds, False)
    assert "### count" in toy._lever_doc() and "### ignored" not in toy._lever_doc()


def test_the_lever_section_is_found_by_its_heading_not_a_mention(toy_card, monkeypatch):
    card = "# Toy tool\n## Loop\nThe agent sees ## Levers below.\n" + TOY_CARD.split("\n", 1)[1]
    monkeypatch.setattr("shape.tools.base.load_tuning_card", lambda tool_dir: card)
    monkeypatch.setattr(base.BaseVerifier, "tuning_card", lambda self: card)
    toy = Toy()
    assert toy._lever_doc().startswith("## Levers\n### alpha") and "The agent sees" not in toy._lever_doc()
    assert toy.levers == ["alpha", "mode", "flag", "count"]


def test_trial_0_is_the_input_and_the_budget_counts_rounds(toy_card):
    calls = []
    engine = ToyEngine([[mv(alpha=2), mv(alpha=3)], [mv("agent_2", alpha=4), mv("agent_2", alpha=5)],
                        [mv("agent_4", alpha=6)]], final=["agent_4"])
    toy, resolution, waves = run_toy(engine, budget=2, calls=calls)
    assert resolution.trials[0][0] == {} and resolution.parents == [None, 0, 0, 2, 2]
    assert len(calls) == 4 == len(resolution.trials) - 1 and waves == [2, 2]    # trial 0 is no tool call
    assert resolution.reason == "max_iters_reached" and toy.rounds == [[1, 2], [3, 4]]
    rounds = engine.questions()
    assert len(rounds) == 2                                              # 2 rounds, not 2 calls
    assert "Rounds: 0 done, at most 2 more; calls so far: 0." in rounds[0]
    assert "Rounds: 1 done, at most 1 more" in rounds[1] and "calls so far: 2." in rounds[1]
    assert 'Input: {"quality": 0.5, "side": 0.6}' in rounds[0] and "Task: tune the toy" in rounds[0]
    assert resolution.params == {**toy.defaults(), "alpha": 5.0} and resolution.details["choice"] == "agent_choice"


def test_a_round_holds_at_most_the_concurrency_cap(toy_card):
    logged = []
    engine = ToyEngine([[mv(alpha=2), mv(alpha=3), mv(alpha=4)]], final=["agent_2"])
    toy, resolution, waves = run_toy(engine, cores=4, log=lambda kind, **f: logged.append(kind))
    assert toy.cap == 2 and waves == [2] and "dropped_move" in logged      # 4 cores // 2 threads per call
    assert "Resources: 4 cores; at most 2 calls run at the same time, 2 threads each." in engine.questions()[0]
    assert "This round: at most 2 call(s)." in engine.questions()[0] and toy.stop_reason == "satisfied"
    assert [base.concurrency_cap(c, 8) for c in (1, 8, 16, 64)] == [1, 1, 2, 8] and base.concurrency_cap(3) == 3


def test_a_move_may_change_several_levers(toy_card):
    toy, resolution, _ = run_toy(ToyEngine([[mv(alpha=4, flag=True, count=2)]]))
    assert resolution.trials[1][1].param_delta == {"alpha": 4.0, "flag": True, "count": 2}
    assert resolution.trials[1][0] == {"alpha": 4.0, "mode": "a", "flag": True, "count": 2}
    assert toy.label(1) == "agent_1"
    attempts = {"agent_1": ("input", dict(resolution.trials[1][0]), resolution.trials[1][1])}
    same = [{"lever": "alpha", "value": "4"}, {"lever": "flag", "value": "true"}, {"lever": "count", "value": "2"}]
    for move, message in (({"changes": [{"lever": "alpha", "value": "2"}, {"lever": "alpha", "value": "3"}]}, "twice"),
                          ({"parent": "agent_1", "changes": []}, "no lever changed relative to agent_1"),
                          ({"parent": "agent_1", "changes": [{"lever": "alpha", "value": "4"}]},
                           "no lever changed relative to agent_1 \\(alpha already at agent_1's value\\)"),
                          ({"parent": "agent_9"}, "'agent_9' is not a finished call"),
                          ({"changes": same}, "already run")):
        with pytest.raises(ValueError, match=message):
            toy.validate(move, attempts, set())
    defaults = toy.validate({}, attempts, set())                         # no change from "input": the defaults
    assert defaults["params"] == toy.defaults() and defaults["param_delta"] == {}
    repeat = toy.validate({"changes": [{"lever": "alpha", "value": "1"}]}, attempts, set())   # 1 = the default
    assert repeat["param_delta"] == {} and repeat["params"] == toy.defaults()                # dropped, no rejection
    mixed = toy.validate({"parent": "agent_1", "changes": [{"lever": "alpha", "value": "4"},
                                                       {"lever": "count", "value": "3"}]}, attempts, set())
    assert mixed["param_delta"] == {"count": 3} and mixed["params"]["alpha"] == 4.0          # only the real change
    with pytest.raises(ValueError, match="already in this round"):
        toy.validate({}, attempts, {defaults["config_key"]})


def test_rejected_moves_get_one_repair_question(toy_card):
    engine = ToyEngine([[mv(alpha=99), mv(alpha=3)]], repair=[mv(alpha=3), mv(alpha=4)], final=["agent_2"])
    toy, resolution, waves = run_toy(engine)
    first, repair = engine.questions()[:2]
    assert repair.startswith(first) and "alpha=99 outside [0, 10]" in repair
    assert "so do not return them again: [{'parent': 'input', 'change': {'alpha': 3.0}}]" in repair
    assert "Return at most 1 replacement move(s)" in repair and repair.endswith("run only the accepted ones.")
    # the repair's repeat of an accepted move is refused, its new move fills the slot
    assert waves == [2] and [v.param_delta for _, v, _ in resolution.trials[1:]] == [{"alpha": 3.0}, {"alpha": 4.0}]
    assert [p["question"] for p in toy.plans] == ["round", "repair", "round", "stop_check"]   # a stop, rounds left
    full = ToyEngine([[mv(alpha=99), mv(alpha=3), mv(alpha=4)]])
    toy = Toy()
    toy.llm_engine = full
    assert len(toy.plan_round(_toy_input(), {}, 2, 8)) == 2 and len(full.asked) == 1     # no room: no repair
    noted = ToyEngine([[mv(alpha=99)]])
    toy = Toy()
    toy.llm_engine, toy.repair_note = noted, " (the note)"
    # the repair returns none: the agent's own stop, not a failure
    assert toy.plan_round(_toy_input(), {}, 2, 8) == [] and toy.stop_reason == "no_valid_move"
    assert (toy.reasoner_failures, toy.fallback_rounds) == (0, 0)
    assert noted.questions()[1].endswith("run only the accepted ones (the note).")


@pytest.mark.parametrize("required", ["0", "1"])
def test_moves_still_invalid_after_repair_never_run_rule_fallback(toy_card, monkeypatch, required):
    monkeypatch.setenv("SHAPE_REQUIRE_AGENTIC", required)
    toy = Toy()
    toy.llm_engine = ToyEngine([[mv(alpha=99)]], repair=[mv(alpha=98)])
    assert toy.plan_round(_toy_input(), {}, 2, 8) == []
    assert toy.stop_reason == "no_valid_move" and toy.fallback_rounds == 0
    assert toy.reasoner_failures == 1 and toy.execution_errors[0]["question"] == "repair"


def test_a_failed_repair_question_counts_one_reasoner_failure(toy_card, monkeypatch):
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)

    class RepairDown(ToyEngine):
        def generate(self, content, response_format=None):
            if "were rejected" in content[0]:
                raise TimeoutError("gateway down")
            return super().generate(content, response_format)

    toy = Toy()
    toy.llm_engine = RepairDown([[mv(alpha=99)]])
    assert toy.plan_round(_toy_input(), {}, 2, 8) == []
    assert (toy.reasoner_failures, toy.fallback_rounds) == (1, 0)
    assert toy.execution_errors[-1]["question"] == "repair"


def test_lever_values_may_be_typed_json_values():
    from shape.utils.schemas import LeverChange

    parsed = [LeverChange.model_validate_json(f'{{"lever": "x", "value": {raw}}}').value
              for raw in ("400", "0.07", "true", "null", '"zscore"', '"200"')]
    assert parsed == [400.0, 0.07, True, None, "zscore", "200"]
    assert isinstance(parsed[0], float) and isinstance(parsed[2], bool)


def test_a_crashed_call_counts_but_is_never_a_parent_rerun_or_chosen(toy_card):
    calls = []
    engine = ToyEngine([[mv(mode=None), mv(alpha=3)],
                        [mv("agent_1", alpha=4), mv(mode=None), mv("agent_2", alpha=4)]],
                       final=["agent_1", "agent_3"])
    toy, resolution, waves = run_toy(engine, calls=calls)
    assert waves == [2, 1] and len(calls) == 3                          # the crash counts in its round
    assert [f["call"] for f in toy.failed] == ["agent_1"]
    assert toy.failed[0]["error"] == "RuntimeError: mode null crashes"
    repair = [text for text in engine.questions() if "were rejected" in text][0]
    assert "from agent_1: parent" in repair and "is not a finished call" in repair
    assert "already run (or failed)" in repair
    assert "did not finish (crashed or rejected by the tool): [{'call': 'agent_1'" in engine.questions()[1]
    assert "Your choice 'agent_1' is not admissible: not a finished call." in engine.questions("FinalChoice")[1]
    assert resolution.details["admissible"] == ["agent_2", "agent_3"] and resolution.details["call"] == "agent_3"


def test_the_plateau_stops_the_search_before_the_next_plan(toy_card):
    engine = ToyEngine([[mv(alpha=2)], [mv(alpha=3)], [mv(alpha=4)], [mv(alpha=5)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, quality=flat)
    assert base.PATIENCE == toy.patience == 3                            # rounds, for 8 rounds
    assert waves == [1, 1, 1] and toy.stop_reason == "plateau" and len(engine.questions()) == 3
    assert ("Stop: return no moves when you are satisfied, at any round; you may then name in deliver the finished\n"
            "call to deliver, which must meet the final rule (otherwise you are asked to choose separately). A stop\n"
            "with rounds left is asked once more: the deliver of the answer that confirms it counts. As a\n"
            "backstop, the loop also stops when the rounds are spent or when 3 rounds in a row have not "
            "raised the best quality beyond the noise."
            in engine.questions()[0])
    assert resolution.reason == "plateau" and resolution.details["verifier_stop"] == "plateau"


def test_the_plateau_counts_rounds_not_tool_calls(toy_card):
    engine = ToyEngine([[mv(alpha=2), mv(alpha=3)], [mv(alpha=4), mv(alpha=5)], [mv(alpha=6)], [mv(alpha=7)]],
                       final=["agent_1"])
    toy, _, waves = run_toy(engine, quality=flat)
    assert waves == [2, 2, 1] and toy.stop_reason == "plateau" and len(engine.questions()) == 3   # 3 rounds, 5 calls
    rising = ToyEngine([[mv(alpha=2), mv(alpha=3)], [mv(alpha=9), mv(alpha=4)], [mv(flag=True)]], final=["agent_2"])
    toy, _, waves = run_toy(rising)                                      # rounds 1 and 2 raise the best
    assert waves == [2, 2, 1] and toy.stop_reason == "satisfied"


def test_with_plateau_after_plan_the_agent_plans_before_the_plateau_stop(toy_card):
    class Assessed(Toy):
        plateau_after_plan = True

    engine = ToyEngine([[mv(alpha=2)], [mv(alpha=3)], [mv(alpha=4)], [mv(alpha=5)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Assessed, quality=flat)
    assert waves == [1, 1, 1] and toy.stop_reason == "plateau" and len(engine.questions()) == 4
    assert [p["round"] for p in toy.plans] == [1, 2, 3, 4]               # the fourth plan is kept, not run


class Admitting(Toy):
    """Runs once at its defaults before any tuning (`admission_call`): a quality below 0.55 is a measured
    problem of that call."""
    admission_call = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorded = []

    def admission_problems(self, inp, vres):
        quality = (vres.metrics or {}).get("quality", 0.0)
        return [] if quality >= 0.55 else [f"quality {quality:.2f} below 0.55"]

    def record_admission(self, answer, call, attempts):
        self.recorded.append((call, answer.severity))


def good(params):
    return 0.6                                          # the defaults are already good; nothing beats them


def test_the_admission_runs_the_tool_once_at_its_defaults_and_skips_the_tuning_when_nothing_is_wrong(toy_card):
    calls = []
    engine = ToyEngine([[mv(alpha=3)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Admitting, calls=calls, quality=good)
    assert waves == [1] and calls == [toy.defaults()] and engine.questions() == []   # no round was planned
    assert resolution.status == "satisfied" and resolution.reason == "no_tuning_needed"
    assert resolution.params == toy.defaults() and resolution.details["call"] == "default"
    assert resolution.verdict.action == "accept" and toy.stop_reason == "no_tuning_needed"
    admission = resolution.details["admission"]
    assert admission["decision"] == admission["rule"] == "SKIP" and admission["problems"] == []
    assert admission["answer"]["severity"] == "none" and toy.recorded == [("default", "none")]
    assert resolution.details["tuning_gate"] == {
        "required": True, "source": "defaults", "decision": "SKIP", "allowed": False,
        "reason": "no_tuning_needed"}
    question = engine.questions("AdmissionAnswer")[0]
    assert question.startswith("Decide whether this task needs the tuning of a fixed toy tool.\n"
                               "The tool ran once at its defaults (default); its measurements found no problem")
    assert "(the guard: the calls the guard admits)" in question and "at most 7 more round(s)" in question
    assert '"diagnosis": "the tool at its defaults (the admission call)"' in question
    toy, minor, _ = run_toy(ToyEngine(admission="minor"), cls=Admitting, quality=good)
    assert minor.reason == "no_tuning_needed" and minor.details["admission"]["decision"] == "SKIP"
    toy, silent, waves = run_toy(Down(), cls=Admitting, quality=good)   # no answer: the rule stands
    assert waves == [1] and silent.reason == "execution_error" and silent.details["admission"]["answer"] is None
    assert silent.details["admission"]["decision"] == "NOT_SEARCHED"

    class AdmittingDelivering(Admitting, Delivering):
        pass

    toy, _, _ = run_toy(ToyEngine(), cls=AdmittingDelivering, quality=good)
    assert toy.delivered == [(1, "no_tuning_needed")]                     # the SKIP hands default on


@pytest.mark.parametrize("decision,action,fault,reason", [
    ("RUN", "retune_here", None, "tuning_allowed"),
    ("SKIP", "accept", None, "no_tuning_needed"),
    ("ABSTAIN", "conceded", "ceiling", "conditions_unmet"),
    ("NOT_SEARCHED", "conceded", "ceiling", "execution_error"),
])
def test_input_admission_blocks_search_even_when_an_adapter_requests_retuning(toy_card, decision, action, fault,
                                                                          reason):
    class InputAdmitting(Toy):
        admission_on_input = True

        def adjudicate(self, metrics, locus, triggered, context):
            vres = super().adjudicate(metrics, locus, triggered, context)
            if context.get("trial_index") == 0:
                self.set_admission({"decision": decision, "evidence": "tool-specific"})
            return vres                            # deliberately still says retune_here

    calls = []
    engine = ToyEngine([[mv(alpha=3)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=InputAdmitting, calls=calls)
    assert toy.admission_action() == (action, fault)
    assert resolution.details["tuning_gate"] == {
        "required": True, "source": "input", "decision": decision, "allowed": decision == "RUN",
        "reason": reason}
    assert toy.admission == {"decision": decision, "evidence": "tool-specific"}
    if decision == "RUN":
        assert waves == [1] and len(calls) == 1
    else:
        assert waves == calls == engine.asked == []
        assert toy.stop_reason == reason


def test_a_missing_required_input_admission_cannot_start_search(toy_card):
    class MissingAdmission(Toy):
        admission_on_input = True

    engine, calls = ToyEngine([[mv(alpha=3)]]), []
    with pytest.raises(ValueError, match="requires an admission on the input"):
        run_toy(engine, cls=MissingAdmission, calls=calls)
    assert calls == engine.asked == []


def test_input_ineligibility_prevents_even_a_default_admission_call(toy_card):
    class Ineligible(Admitting):
        def adjudicate(self, metrics, locus, triggered, context):
            vres = super().adjudicate(metrics, locus, triggered, context)
            if context.get("trial_index") == 0:
                self.set_admission({"decision": "ABSTAIN", "reasons": ["no usable input"]}, source="input")
            return vres

    engine, calls = ToyEngine(), []
    _, resolution, waves = run_toy(engine, cls=Ineligible, calls=calls)
    assert calls == waves == engine.asked == []
    assert resolution.details["tuning_gate"]["source"] == "input"
    assert resolution.details["tuning_gate"]["decision"] == "ABSTAIN"


@pytest.mark.parametrize("record", [{}, {"decision": "run"}, {"decision": "NOT_REQUIRED"},
                                    {"decision": "PENDING"}])
def test_invalid_admission_cannot_override_a_valid_gate(toy_card, record):
    toy = Toy()
    toy.set_admission({"decision": "SKIP"})
    with pytest.raises(ValueError, match="invalid tuning admission"):
        toy.set_admission(record)
    assert toy.admission == {"decision": "SKIP"}


def test_resource_stop_is_shared_and_runs_only_after_admission(toy_card):
    class Limited(Toy):
        admission_on_input = True
        checked = 0
        admission_decision = "RUN"

        def adjudicate(self, metrics, locus, triggered, context):
            vres = super().adjudicate(metrics, locus, triggered, context)
            if context.get("trial_index") == 0:
                self.set_admission({"decision": self.admission_decision})
            return vres

        def search_stop_reason(self):
            self.checked += 1
            return "deadline"

    engine, calls = ToyEngine([[mv(alpha=3)]]), []
    toy, resolution, waves = run_toy(engine, cls=Limited, calls=calls)
    assert toy.checked == 1 and toy.stop_reason == "deadline"
    assert resolution.details["tuning_gate"]["allowed"] is True
    assert calls == waves == engine.asked == []

    class Ineligible(Limited):
        admission_decision = "ABSTAIN"

    toy, _, _ = run_toy(engine, cls=Ineligible, calls=calls)
    assert toy.checked == 0 and toy.stop_reason == "conditions_unmet"


@pytest.mark.parametrize("module,name,source", [
    ("segmentation", "SegmentationAgenticVerifier", "defaults"),
    ("crop", "CropAgenticVerifier", "defaults"),
    ("batch_correction_unified", "BatchCorrectionVerifier", "input"),
    ("representation", "RepresentationAgenticVerifier", "input"),
])
def test_all_four_tools_use_the_shared_loop_and_require_admission(toy_card, module, name, source):
    from importlib import import_module

    cls = getattr(import_module(f"shape.agent.verification.verifiers.{module}"), name)
    verifier = cls()
    assert cls.recommend_wave is base.AgenticVerifier.recommend_wave
    assert verifier.tuning_gate() == {
        "required": True, "source": source, "decision": "PENDING", "allowed": False,
        "reason": "admission_pending"}


def test_a_run_admission_is_followed_by_the_tools_reference_calls_before_the_agents_first_round(toy_card):
    class Referencing(Admitting):
        """After a RUN admission, the tool at alpha 3 (e.g. a scale the admission measured) as a reference."""

        def reference_moves(self, inp, attempts):
            return [{"parent": "default", "changes": [{"lever": "alpha", "value": "3"}],
                     "diagnosis": "the tool at the measured scale"}]

    calls = []
    engine = ToyEngine([[mv("reference", alpha=4)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Referencing, calls=calls)   # the defaults: a measured problem
    assert waves == [1, 1, 1] and [c["alpha"] for c in calls] == [1.0, 3.0, 4.0]
    assert list(toy.reference_calls) == ["reference"] and toy.rounds == [[1], [2], [3]]
    assert [toy.label(i) for i in range(4)] == ["input", "default", "reference", "agent_1"]
    assert resolution.details["call"] == "agent_1"
    first = engine.questions()[0]                                           # asked only after the reference
    assert ("the tuning started on quality 0.52 below 0.55. Then the system ran: reference (the tool at the "
            "measured scale).\nTool defaults" in first and "\nRounds: 2 done, at most 6 more" in first)
    assert '"diagnosis": "the tool at the measured scale"' in first
    toy, skipped, waves = run_toy(ToyEngine(), cls=Referencing, quality=good)   # SKIP: no reference call
    assert waves == [1] and toy.reference_calls == {} and skipped.reason == "no_tuning_needed"
    # the smallest budget (2): after the admission call one call is left, and it is the agent's
    calls = []
    engine = ToyEngine([[mv("default", alpha=4)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Referencing, budget=2, calls=calls)
    assert waves == [1, 1] and [c["alpha"] for c in calls] == [1.0, 4.0] and toy.reference_calls == {}
    assert "Then the system ran" not in engine.questions()[0] and resolution.details["call"] == "agent_1"


class Defaulting(Admitting):
    """After a RUN admission, the reference call "reference" (alpha 3, quality 0.56) is the default delivery; an
    override needs a quality above the default's by more than 0.03."""

    def reference_moves(self, inp, attempts):
        return [{"parent": "default", "changes": [{"lever": "alpha", "value": "3"}], "diagnosis": "the reference"}]

    def override_allowed(self, inp, attempts, call, default):
        gain = attempts[call][2].scalar_score - attempts[default][2].scalar_score
        return gain > 0.03, f"{call} is {gain:+.2f} on {default}"

    def override_text(self, default):
        return f"beats {default} by more than 0.03"


def test_the_reference_default_is_delivered_unless_the_hook_allows_the_agents_override(toy_card):
    # agent_1 (alpha 4, quality 0.58) is +0.02 on reference: refused, asked again; still agent_1 -> the default
    engine = ToyEngine([[mv("reference", alpha=4)]], final=["agent_1", "agent_1"])
    toy, refused, _ = run_toy(engine, cls=Defaulting)
    rule = ("Reference default: reference (the system's reference call) is delivered unless you choose a call that\n"
            "beats reference by more than 0.03; a choice that does not is answered with the reason and asked once "
            "more; if it\nstill does not, reference is delivered.")
    final, retry = engine.questions("FinalChoice")
    assert rule in engine.questions()[0] and rule in final                 # the round prompt and the final prompt
    assert retry.startswith(final) and retry.endswith(
        "Your choice 'agent_1' does not override the reference default reference: agent_1 is +0.02 on reference. "
        "Choose reference, or a call that beats reference by more than 0.03.")
    assert refused.details["call"] == "reference" and refused.details["choice"] == "reference_default"
    assert refused.status == "satisfied" and refused.params["alpha"] == 3.0
    assert refused.details["reference_default"] == {"call": "reference", "overridden": False,
                                                    "reason": "agent_1 is +0.02 on reference"}
    # asked again, the agent takes the default
    _, taken, _ = run_toy(ToyEngine([[mv("reference", alpha=4)]], final=["agent_1", "reference"]), cls=Defaulting)
    assert taken.details["call"] == "reference" and taken.details["choice"] == "agent_choice"
    # an override the hook allows: agent_1 (alpha 5, quality 0.60) is +0.04 on reference
    engine = ToyEngine([[mv("reference", alpha=5)]], final=["agent_1"])
    _, allowed, _ = run_toy(engine, cls=Defaulting)
    assert allowed.details["call"] == "agent_1" and allowed.details["choice"] == "agent_choice"
    assert len(engine.questions("FinalChoice")) == 1 and allowed.details["reference_default"] == {
        "call": "reference", "overridden": True, "reason": "agent_1 is +0.04 on reference"}
    # a choice still inadmissible after the retry: the default, as for a refused override
    _, twice, _ = run_toy(ToyEngine([[mv("reference", alpha=4, flag=True)]], final=["agent_1", "agent_1"]),
                          cls=Defaulting)
    assert twice.details["call"] == "reference" and twice.details["choice"] == "reference_default"
    assert twice.status == "satisfied" and twice.details["reference_default"]["overridden"] is False


class FailingRetry(ToyEngine):
    """Every final question after the first goes unanswered (a gateway down on the retry)."""

    def generate(self, content, response_format=None):
        if response_format.__name__ == "FinalChoice" and self.questions("FinalChoice"):
            raise TimeoutError("gateway down")
        return super().generate(content, response_format)


@pytest.mark.parametrize("final", [["agent_1", "agent_2"], ["agent_2", "agent_1"]])
def test_no_accepted_answer_delivers_the_default_in_either_order(toy_card, final):
    # agent_1 (alpha 4, +0.02 on reference) is refused by the hook; agent_2 (flag) is inadmissible: either order ->
    # the default
    engine = ToyEngine([[mv("reference", alpha=4), mv("reference", alpha=5, flag=True)]], final=final)
    _, resolution, _ = run_toy(engine, cls=Defaulting)
    assert len(engine.questions("FinalChoice")) == 2 and resolution.details["call"] == "reference"
    assert resolution.details["choice"] == "reference_default" and resolution.status == "satisfied"
    assert resolution.details["reference_default"] == {"call": "reference", "overridden": False,
                                                       "reason": "agent_1 is +0.02 on reference"}


def test_a_refused_override_and_an_unanswered_retry_deliver_the_default(toy_card):
    toy, resolution, _ = run_toy(FailingRetry([[mv("reference", alpha=4)]], final=["agent_1"]), cls=Defaulting)
    assert resolution.details["call"] is None and resolution.details["choice"] == "execution_error"
    assert resolution.status == "conceded" and toy.reasoner_failures == 1


def test_a_refused_override_then_an_allowed_one_delivers_the_allowed_call(toy_card):
    # agent_1 (alpha 4) refused; agent_2 (alpha 5, quality 0.60, +0.04 on reference) allowed on the retry
    engine = ToyEngine([[mv("reference", alpha=4), mv("reference", alpha=5)]], final=["agent_1", "agent_2"])
    _, resolution, _ = run_toy(engine, cls=Defaulting)
    assert resolution.details["call"] == "agent_2" and resolution.details["choice"] == "agent_choice"
    assert resolution.status == "satisfied" and resolution.details["reference_default"] == {
        "call": "reference", "overridden": True, "reason": "agent_2 is +0.04 on reference"}


def test_failed_final_question_preserves_input_without_claiming_a_choice(toy_card):
    class NoFinal(ToyEngine):
        def generate(self, content, response_format=None):
            if response_format is not WavePlan:
                raise TimeoutError("gateway down")
            return super().generate(content, response_format)
    toy, resolution, _ = run_toy(NoFinal([[mv("reference", alpha=5)]]), cls=Defaulting)
    assert resolution.details["call"] is None and resolution.details["choice"] == "execution_error"
    assert resolution.status == "conceded" and toy.fallback_rounds == 0
    assert resolution.details["execution_errors"][-1]["question"] == "final"


def test_the_default_hook_allows_any_admissible_override_and_no_reference_call_changes_nothing(toy_card):
    class Referencing(Admitting):
        def reference_moves(self, inp, attempts):
            return [{"parent": "default", "changes": [{"lever": "alpha", "value": "3"}], "diagnosis": "the reference"}]

    engine = ToyEngine([[mv("reference", alpha=4)]], final=["agent_1"])
    _, resolution, _ = run_toy(engine, cls=Referencing)
    assert resolution.details["call"] == "agent_1" and resolution.details["choice"] == "agent_choice"
    assert resolution.details["reference_default"] == {"call": "reference", "overridden": True, "reason": ""}
    assert resolution.details["admissible"] == ["default", "reference", "agent_1"]      # in run order, not by name
    assert ("Reference default: reference (the system's reference call), delivered unless you choose another call."
            in engine.questions()[0])
    engine = ToyEngine([[mv("default", alpha=3)]], final=["agent_1"])               # no reference call
    _, resolution, _ = run_toy(engine, cls=Admitting)
    assert "reference_default" not in resolution.details and resolution.details["call"] == "agent_1"
    assert not any("Reference default" in text for _, text in engine.asked)


def test_a_step_on_a_lever_the_params_leave_out_starts_from_its_effective_default():
    stepped = base.apply_bounded({"diameter": 30.0}, {"niter": "up", "tile_overlap": "up"})
    assert stepped["niter"] == pytest.approx(200.0 / 0.7) and stepped["tile_overlap"] == pytest.approx(0.15)
    assert base.apply_bounded({"niter": None}, {"niter": "down"})["niter"] == pytest.approx(140.0)
    assert "min_size" not in base.apply_bounded({"diameter": 30.0}, {"min_size": "up"})   # no effective default
    assert base.apply_bounded({"niter": 400.0}, {"niter": "up"})["niter"] == pytest.approx(400.0 / 0.7)


def test_the_admission_needs_two_rounds_and_a_run_that_requires_the_agent_fails_without_it(toy_card, monkeypatch):
    with pytest.raises(ValueError, match="at least 2 rounds"):
        run_toy(ToyEngine(), cls=Admitting, budget=1, quality=good)
    monkeypatch.setenv("SHAPE_REQUIRE_AGENTIC", "1")                    # the admission question unanswered
    _, resolution, _ = run_toy(Down(), cls=Admitting, quality=good)
    assert resolution.reason == "execution_error"
    assert resolution.details["execution_errors"][0]["question"] == "admission"


def test_a_measured_problem_of_the_defaults_call_starts_the_tuning_without_asking(toy_card):
    engine = ToyEngine([[mv("default", alpha=3)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Admitting)             # the defaults: quality 0.52
    assert waves == [1, 1] and engine.questions("AdmissionAnswer") == []
    admission = resolution.details["admission"]
    assert admission["decision"] == admission["rule"] == "RUN" and admission["problems"] == ["quality 0.52 below 0.55"]
    first = engine.questions()[0]
    assert ("Admission: the tool ran once at its defaults (default); the tuning started on quality 0.52 below 0.55.\n"
            "Tool defaults" in first and "\nRounds: 1 done, at most 7 more" in first)
    assert resolution.details["call"] == "agent_1" and resolution.details["choice"] == "agent_choice"

    def broken_defaults(params):
        if params["alpha"] == 1:
            raise RuntimeError("the defaults crash")
        return 0.6

    engine = ToyEngine([[mv(alpha=3)]], final=["agent_1"])
    toy, resolution, waves = run_toy(engine, cls=Admitting, quality=broken_defaults)
    assert waves == [1, 1] and resolution.details["admission"]["problems"] == [
        "the call at the defaults failed: RuntimeError: the defaults crash"]


def test_a_major_problem_the_agent_names_starts_the_tuning(toy_card):
    engine = ToyEngine([[mv("default", alpha=3)]], final=["agent_1"], admission="major")
    toy, resolution, waves = run_toy(engine, cls=Admitting, quality=good)
    admission = resolution.details["admission"]
    assert admission["rule"] == "SKIP" and admission["decision"] == "RUN" and waves == [1, 1]
    assert ("the tuning started on your admission answer, masks miss half the objects (major)."
            in engine.questions()[0])
    assert resolution.reason == "satisfied" and resolution.details["call"] == "agent_1"


def test_the_final_choice_is_the_agents_among_the_admissible_calls(toy_card):
    rounds = [[mv(alpha=3), mv(alpha=4, flag=True)]]                     # agent_2 fails the guard
    engine = ToyEngine(rounds, final=["agent_1"])
    _, chosen, _ = run_toy(engine)
    assert chosen.status == "satisfied" and chosen.details["choice"] == "agent_choice" and chosen.params["alpha"] == 3.0
    final = engine.questions("FinalChoice")[0]
    assert final.startswith("Choose the final output of this task: one finished call.\n"
                            "Cite the readouts your choice rests on.\nTask: tune the toy\n")
    assert "Admissible calls (the calls the guard admits): ['agent_1']" in final
    engine = ToyEngine(rounds, final=["agent_2", "agent_1"])
    _, retried, _ = run_toy(engine)
    assert ("Your choice 'agent_2' is not admissible: side below 0.5. Choose one of ['agent_1']."
            in engine.questions("FinalChoice")[1])
    assert ([a["call"] for a in retried.details["final_answers"]] == ["agent_2", "agent_1"]
            and retried.details["call"] == "agent_1")
    _, twice, _ = run_toy(ToyEngine(rounds, final=["agent_2", "agent_2"]))   # still inadmissible: the best admissible
    assert (twice.status == "conceded" and twice.reason == "final_choice_inadmissible"
            and twice.details["call"] == "agent_1")

    class NoFinal(ToyEngine):
        def generate(self, content, response_format=None):
            if response_format is not WavePlan:
                raise TimeoutError("gateway down")
            return super().generate(content, response_format)

    toy, fallback, _ = run_toy(NoFinal(rounds))
    assert fallback.details["choice"] == "execution_error" and fallback.details["call"] is None
    assert fallback.status == "conceded" and toy.reasoner_failures == 1


def test_keep_input_lets_the_final_choice_keep_the_input(toy_card):
    rounds = [[mv(alpha=3), mv(alpha=4, flag=True)]]
    engine = ToyEngine(rounds, final=[None])
    _, kept, _ = run_toy(engine, cls=Keeper)
    assert kept.result is None and kept.status == "conceded" and kept.reason == "agent_kept_input"
    assert "if you choose none, or no call qualifies, the input is kept." in engine.questions()[0]
    assert "one finished call, or null to keep the input." in engine.questions("KeepChoice")[0]
    engine = ToyEngine(rounds, final=["agent_2", "agent_2"])
    _, twice, _ = run_toy(engine, cls=Keeper)
    assert twice.result is None and twice.reason == "final_choice_inadmissible"
    assert "Choose one of ['agent_1'], or null." in engine.questions("KeepChoice")[1]


def test_least_bad_is_the_output_when_no_call_is_admissible_and_deliver_hands_it_on(toy_card):
    all_fail = [[mv(alpha=3, flag=True), mv(alpha=4, flag=True)]]
    engine = ToyEngine(all_fail)
    toy, resolution, _ = run_toy(engine, cls=Delivering)
    assert resolution.status == "conceded" and resolution.reason == "no_admissible_call"
    assert resolution.params["alpha"] == 4.0 and toy.delivered == [(2, "no_admissible_call")]
    assert engine.questions("FinalChoice") == []                         # nothing admissible: no final question
    toy, chosen, _ = run_toy(ToyEngine([[mv(alpha=3)]], final=["agent_1"]), cls=Delivering)
    assert chosen.status == "satisfied" and toy.delivered == [(1, "agent_choice")]
    _, none, _ = run_toy(ToyEngine(all_fail))                            # no `least_bad`: no output
    assert none.result is None and none.reason == "no_admissible_call"


def test_ask_logs_the_image_refs_and_every_failed_try_with_the_reply_as_returned(monkeypatch):
    """The prompt entry names its images by label, sha256 and size (never the bytes); a failed try is logged
    with its error and the reply as returned (a gateway error, an invalid answer); the question is unchanged."""
    import hashlib

    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)
    logged, sent = [], []
    replies = iter([{"error": "rate_limit", "message": "slow down"},
                    {"content": "not json", "usage": {"total_tokens": 5}},
                    {"content": AdmissionAnswer(reasoning="r", problem="none", severity="none", confidence="high"),
                     "usage": {"total_tokens": 7, "reasoning_tokens": 3}}])
    verifier = base.AgenticVerifier(log=lambda kind, **f: logged.append((kind, f)))
    verifier.llm_engine = SimpleNamespace(generate=lambda content, **_k: sent.append(content) or next(replies))
    png = b"\x89PNG\r\n\x1a\nfake"
    content = ["the prompt", "INPUT IMAGE:", png, "CROP PANEL OF default:", png + b"2"]
    answer = verifier._ask(content, AdmissionAnswer, "admission")
    assert answer.problem == "none" and sent == [content] * 3
    prompt = [f for kind, f in logged if kind == "prompt"]
    assert len(prompt) == 1 and prompt[0]["text"] == "the prompt"
    assert prompt[0]["images"] == [
        {"label": "INPUT IMAGE:", "sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)},
        {"label": "CROP PANEL OF default:", "sha256": hashlib.sha256(png + b"2").hexdigest(), "bytes": len(png) + 1}]
    errors = [f for kind, f in logged if kind == "reasoner_error"]
    assert [e["text"].split(":")[0] for e in errors] == ["try 1", "try 2"]
    assert errors[0]["reply"] == {"error": "rate_limit", "message": "slow down"}
    assert errors[1]["reply"] == {"content": "not json", "usage": {"total_tokens": 5}}
    assert [kind for kind, _ in logged].count("usage") == 2 and "PNG" not in json.dumps(logged, default=str)
    assert base.image_refs("text only") == [] and base.image_refs(["prompt", png], "M0")[0]["label"] == "M0"


@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_missing_engine_is_fatal_and_failed_question_never_runs_fallback(toy_card, monkeypatch, strategy):
    from shape.engine.errors import NoEngineError
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY", strategy)
    with pytest.raises(NoEngineError):
        run_toy(None, budget=3)
    calls = []
    toy, resolution, waves = run_toy(Down(), budget=1, calls=calls)
    assert calls == waves == [] and toy.fallback_rounds == 0
    assert toy.reasoner_failures == 1
    assert resolution.details["execution_errors"][0]["question"] == "round"


def test_extend_move_adds_a_tools_own_part_of_a_move(toy_card):
    class Tagged(Toy):
        """A move may also carry `tags` (a tool's own part of a move, like the cropper's cell rules)."""

        def extend_move(self, move, params, parent, attempts):
            tags = list(move.get("tags") or [])
            if "bad" in tags:
                raise ValueError("tag 'bad' is not allowed")
            if not tags:
                return {}
            params["tags"] = sorted({*(params.get("tags") or []), *tags})
            return {"tags": tags}

    toy = Tagged()
    attempts = {"agent_1": ("input", {**toy.defaults(), "tags": ["x"]}, VLRRResult("PASS", 0.6, {}))}
    request = toy.validate({"parent": "agent_1", "tags": ["y"]}, attempts, set())  # no lever changed, a tag did
    assert request["param_delta"] == {"tags": ["y"]} and request["params"]["tags"] == ["x", "y"]
    assert attempts["agent_1"][1]["tags"] == ["x"]                                 # the parent's value is untouched
    both = toy.validate({"changes": [{"lever": "alpha", "value": "3"}], "tags": ["z"]}, attempts, set())
    assert both["param_delta"] == {"alpha": 3.0, "tags": ["z"]}
    with pytest.raises(ValueError, match="tag 'bad'"):
        toy.validate({"tags": ["bad"]}, attempts, set())
    with pytest.raises(ValueError, match="no lever changed relative to agent_1"):
        toy.validate({"parent": "agent_1"}, attempts, set())

# ---- the stopping plan's delivery (`WavePlan.deliver`), the untried note, the token fact -----------------------

class Stopping(ToyEngine):
    """Each round plan names `deliver` (the stopping plan's names the call to deliver); optionally reports `usage`
    on every reply."""

    def __init__(self, *args, deliver=None, usage=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deliver, self.usage = deliver, usage

    def generate(self, content, response_format=None):
        reply = super().generate(content, response_format)
        if isinstance(reply["content"], WavePlan) and reply["content"].analysis == "a":
            reply["content"] = reply["content"].model_copy(update={"deliver": self.deliver})
        return {**reply, "usage": self.usage} if self.usage is not None else reply


def test_a_stopping_plan_that_names_an_admissible_call_delivers_it_without_the_final_question(toy_card):
    rounds = [[mv(alpha=3), mv(alpha=4, flag=True)]]                     # agent_2 fails the guard
    engine = Stopping(rounds, deliver="agent_1")
    toy, resolution, _ = run_toy(engine, cls=Delivering)
    assert engine.questions("FinalChoice") == [] and toy.delivered == [(1, "agent_choice")]
    assert resolution.status == "satisfied" and resolution.reason == "satisfied" and resolution.params["alpha"] == 3.0
    assert resolution.details["choice"] == "agent_choice" and resolution.details["final_answers"] == [
        {"call": "agent_1", "reasoning": "a", "cited_readouts": [], "question": "round"}]
    assert resolution.details["plan_delivery"] == {"call": "agent_1", "delivered": True, "reason": ""}
    assert toy.plans[0]["deliver"] == "agent_1"                          # a plan with moves: its deliver is not used
    assert ("you may then name in deliver the finished\ncall to deliver, which must meet the final rule (otherwise "
            "you are asked to choose separately)." in engine.questions()[0])


class StopDelivering(ToyEngine):
    """Round plans as (moves, deliver) pairs, the stop check's included (it is a round question too)."""

    def __init__(self, plans, final=()):
        super().__init__(final=final)
        self.plans = list(plans)

    def generate(self, content, response_format=None):
        if response_format is not WavePlan or "were rejected" in content[0]:
            return super().generate(content, response_format)
        self.asked.append((response_format.__name__, content[0]))
        moves, deliver = self.plans.pop(0) if self.plans else ([], None)
        return {"content": WavePlan(analysis="a", moves=moves, deliver=deliver)}


def test_the_deliver_that_counts_is_the_confirming_answers(toy_card):
    """A stop with rounds left is asked once more: the answer that ends the search names the delivery. A deliver on
    a stop the agent withdraws is dropped; a confirmation without one asks the final question."""
    engine = StopDelivering([([mv(alpha=3)], None), ([], "agent_9"), ([], "agent_1")])   # the confirmation's counts
    toy, resolution, _ = run_toy(engine, cls=Delivering)
    assert [p["question"] for p in toy.plans] == ["round", "round", "stop_check"]
    assert engine.questions("FinalChoice") == [] and toy.delivered == [(1, "agent_choice")]
    assert resolution.details["plan_delivery"] == {"call": "agent_1", "delivered": True, "reason": ""}
    withdrawn = StopDelivering([([mv(alpha=3)], None), ([], "agent_1"), ([mv("agent_1", alpha=4)], None), ([], None),
                                ([], None)], final=["agent_2"])
    toy, resolution, waves = run_toy(withdrawn, cls=Delivering)
    assert waves == [1, 1] and toy.plan_delivery is None and "plan_delivery" not in resolution.details
    assert len(withdrawn.questions("FinalChoice")) == 1 and resolution.details["call"] == "agent_2"
    unconfirmed = StopDelivering([([mv(alpha=3)], None), ([], "agent_1"), ([], None)], final=["agent_1"])
    toy, resolution, _ = run_toy(unconfirmed, cls=Delivering)
    assert toy.plan_delivery is None and len(unconfirmed.questions("FinalChoice")) == 1


@pytest.mark.parametrize("deliver", ["agent_2", "agent_9"])
def test_an_inadmissible_plan_delivery_asks_the_final_question_with_its_retry(toy_card, deliver):
    rounds = [[mv(alpha=3), mv(alpha=4, flag=True)]]
    engine = Stopping(rounds, final=["agent_2", "agent_1"], deliver=deliver)
    _, resolution, _ = run_toy(engine)
    final = engine.questions("FinalChoice")
    assert len(final) == 2 and "Your choice 'agent_2' is not admissible: side below 0.5." in final[1]
    assert resolution.details["call"] == "agent_1" and resolution.details["choice"] == "agent_choice"
    assert [a["call"] for a in resolution.details["final_answers"]] == ["agent_2", "agent_1"]
    assert resolution.details["plan_delivery"] == {"call": deliver, "delivered": False,
                                                   "reason": "not an admissible call"}


def test_a_plan_delivery_against_the_reference_default_needs_the_override_hook(toy_card):
    # agent_1 (alpha 4) is +0.02 on the default reference: refused -> the final question, its retry, then the default
    engine = Stopping([[mv("reference", alpha=4)]], final=["agent_1", "agent_1"], deliver="agent_1")
    _, refused, _ = run_toy(engine, cls=Defaulting)
    assert len(engine.questions("FinalChoice")) == 2 and refused.details["choice"] == "reference_default"
    assert refused.details["call"] == "reference" and refused.details["plan_delivery"] == {
        "call": "agent_1", "delivered": False, "reason": "agent_1 is +0.02 on reference"}
    # agent_1 (alpha 5) is +0.04 on reference: allowed, delivered without the final question
    engine = Stopping([[mv("reference", alpha=5)]], deliver="agent_1")
    _, allowed, _ = run_toy(engine, cls=Defaulting)
    assert engine.questions("FinalChoice") == [] and allowed.details["choice"] == "agent_choice"
    assert allowed.details["reference_default"] == {"call": "reference", "overridden": True,
                                                    "reason": "agent_1 is +0.04 on reference"}
    # the default itself needs no override
    engine = Stopping([[mv("reference", alpha=4)]], deliver="reference")
    _, default, _ = run_toy(engine, cls=Defaulting)
    assert engine.questions("FinalChoice") == [] and default.details["call"] == "reference"
    assert default.details["choice"] == "agent_choice" and default.details["reference_default"]["overridden"] is False


def test_a_null_plan_delivery_asks_the_final_question_and_keep_input_is_unchanged(toy_card):
    rounds = [[mv(alpha=3), mv(alpha=4, flag=True)]]
    engine = Stopping(rounds, final=["agent_1"])
    _, asked, _ = run_toy(engine)
    assert len(engine.questions("FinalChoice")) == 1 and asked.details["choice"] == "agent_choice"
    assert "plan_delivery" not in asked.details and asked.details["final_answers"][0]["reasoning"] == "r"
    engine = Stopping(rounds, final=[None])                              # keep_input: null still asks; null keeps
    _, kept, _ = run_toy(engine, cls=Keeper)
    assert len(engine.questions("KeepChoice")) == 1 and kept.result is None and kept.reason == "agent_kept_input"
    engine = Stopping(rounds, deliver="agent_1")                             # keep_input: a named call is delivered
    _, chosen, _ = run_toy(engine, cls=Keeper)
    assert engine.questions("KeepChoice") == [] and chosen.details["choice"] == "agent_choice"
    assert chosen.status == "satisfied" and chosen.params["alpha"] == 3.0


def test_the_untried_list_ends_with_its_reminder_once(toy_card):
    engine = ToyEngine([[mv(alpha=3)]], final=["agent_1"])
    run_toy(engine)
    first, second = engine.questions()[:2]                               # then the stop check of the second
    note = base.AgenticVerifier.untried_note
    assert first.endswith(f"Levers no call has changed yet: ['alpha', 'mode', 'flag', 'count']\n{note}")
    assert second.endswith(f"Levers no call has changed yet: ['mode', 'flag', 'count']\n{note}")
    assert first.count(note) == 1 and "leaving them untried is a valid choice" in note


def test_the_cropper_untried_lists_keep_the_per_cell_keys_and_the_reminder_once():
    from shape.agent.verification.verifiers.crop import CELL_KEYS, CropAgenticVerifier

    crop = CropAgenticVerifier()
    attempts = {"agent_1": ("input", {"margin_mode": "tight", "cell_overrides": {"3": {"recenter": True}}},
                        VLRRResult("PASS", 0.5, {}, param_delta={"margin_mode": "tight"}))}
    text = crop.untried_text(attempts)
    keys = [k for k in CELL_KEYS if k != "recenter"]
    levers = [lever for lever in crop.levers if lever != "margin_mode"]
    assert text == (f"Levers no call has changed yet: {levers}; per-cell keys no rule has set yet: {keys}\n"
                    f"{crop.untried_note}")


def test_the_round_prompt_shows_the_reported_model_tokens_of_the_unit(toy_card):
    engine = Stopping([[mv(alpha=3)], [mv(alpha=4)]], final=["agent_2"], usage={"total_tokens": 120})
    toy, _, _ = run_toy(engine)
    prompts = engine.questions()
    assert "seconds of each finished call: none yet; tokens of your questions on this task so far: none yet.\n" in prompts[0]
    assert "; tokens of your questions on this task so far: 120.\n" in prompts[1] and "so far: 240.\n" in prompts[2]
    assert toy.tokens == {"replies": 5, "reported": 5, "total": 600}       # the stop check's and the final's included
    engine = ToyEngine([[mv(alpha=3)]], final=["agent_1"])                   # no usage: not reported, never 0
    run_toy(engine)
    assert "; tokens of your questions on this task so far: not reported.\n" in engine.questions()[1]
    toy = Toy()
    toy.tokens = {"replies": 3, "reported": 2, "total": 50}
    assert toy.tokens_text() == "50 (1 of 3 replies not reported)"
    toy = Toy()                                         # an engine without the usage reports 0: not reported
    toy._count_tokens({"content": "x", "usage": {"total_tokens": 0}})
    assert toy.tokens_text() == "not reported"


def test_a_moves_change_is_a_lever_and_its_value_in_every_wave_plan_schema():
    """No wave code reads a change's reasoning / concede / fault / upstream_target (2026-10-06): a change is the
    lever and its value in every tool's plan schema; the SSL verifier's own schema adds only a basis (A12)."""
    from shape.agent.verification.verifiers.batch_correction_unified import BatchCorrectionVerifier
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier, CropPlan
    from shape.agent.verification.verifiers.representation import BasedPlan, RepresentationAgenticVerifier
    from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier, SegPlan

    assert set(LeverChange.model_fields) == {"lever", "value"}
    schemas = {v.plan_schema for v in (BatchCorrectionVerifier, CropAgenticVerifier, SegmentationAgenticVerifier)}
    assert schemas == {WavePlan, CropPlan, SegPlan} and RepresentationAgenticVerifier.plan_schema is BasedPlan
    move = {"parent": "input", "changes": [{"lever": "alpha", "value": "3"}]}
    for schema in schemas:
        assert set(schema.model_json_schema()["$defs"]["LeverChange"]["properties"]) == {"lever", "value"}
        plan = schema.model_validate({"analysis": "a", "moves": [move]})
        assert plan.moves[0].changes[0].model_dump() == {"lever": "alpha", "value": "3"}
    assert set(BasedPlan.model_json_schema()["$defs"]["BasedChange"]["properties"]) == {"lever", "value", "basis"}
    assert "focus_levers" not in BasedPlan.model_fields                  # no narrowing for the SSL verifier
    plan = BasedPlan.model_validate({"analysis": "a", "moves": [move]})
    assert plan.moves[0].changes[0].model_dump() == {"lever": "alpha", "value": "3", "basis": ""}


class Described(Toy):
    """The toy with the text a tool may add: readouts (fixed) and evidence (it may vary by round)."""

    def readouts_text(self):
        return "Readouts: quality and side."

    def evidence_text(self):
        return f"Noise: none measured after {len(self.rounds)} round(s)."


def test_the_round_prompt_starts_with_the_text_the_same_for_every_question_and_unit(toy_card):
    """Every round prompt starts with the same block -- the header, the resources, the lever doc, the move text,
    the readouts, the final and stop rules --, then the unit's text (task, defaults, input), then the round's."""
    engine = ToyEngine([[mv(alpha=3)], [mv("agent_1", alpha=4)]], final=["agent_2"])
    toy, _, _ = run_toy(engine, cls=Described)
    other = Described()                                                  # another unit: its brief, its input
    other.begin(SimpleNamespace(cores=4, deadline=None, brief="tune another toy"))
    prompts = [*engine.questions(), other.round_prompt(VLRRResult("PASS", 0.7, {"quality": 0.7, "side": 0.9}), {}, 2, 8)]
    static = {p[:p.index("Task: ")] for p in prompts}
    assert len(prompts) == 5 and len(static) == 1                       # the stop check's included
    block = static.pop()
    assert block.startswith("You tune a fixed toy tool for one task by\n")
    for text in (toy.resources_text(), toy._lever_doc(), toy.move_text(), toy.readouts_text(), toy.final_rule_text(),
                 "Stop: return no moves when you are satisfied"):
        assert text in block
    for text in ("Rounds:", "Input:", "Noise:", "Tool defaults", "Wall time", "Your hypotheses", "Finished calls"):
        assert text not in block
    for prompt in prompts:
        rest = prompt[len(block):]
        order = [rest.index(t) for t in ("Task: ", "Tool defaults", "Input: ", "Noise: ", "Rounds: ", "Wall time",
                                         "Your hypotheses", "Finished calls", "Levers no call has changed yet")]
        assert order == sorted(order)
    assert "Task: tune the toy\n" in prompts[0] and "Task: tune another toy\n" in prompts[-1]



def test_calls_are_named_by_role_and_listed_in_run_order_past_nine(toy_card):
    """agent_10 and agent_11 follow agent_9 (names do not sort in run order): the rounds, the admissible calls and
    the calls table list the calls as they ran."""
    engine = ToyEngine([[mv(alpha=1 + k / 10)] for k in range(1, 12)], final=["agent_11"])   # within its range
    toy, res, _ = run_toy(engine, budget=12, quality=lambda p: 0.5 + 0.01 * p["alpha"])
    names = [f"agent_{i}" for i in range(1, 12)]
    assert [c for r in res.details["rounds"] for c in r] == names
    assert res.details["admissible"] == names and res.details["call"] == "agent_11"
    table = engine.questions("FinalChoice")[0].split("Finished calls, one per line:\n", 1)[1]
    assert [json.loads(line)["call"] for line in table.splitlines()] == names
    assert [toy.label(i) for i in range(12)] == ["input"] + names


def test_a_pending_call_is_named_by_its_params_else_by_its_place_among_the_proposals(toy_card):
    """Before its bind, a running call is named from its params, else by its place among the pending proposals
    (a measurement's log line has only the index); a later index without a proposal stays call_<i>."""
    toy = Toy()
    first, second = {"alpha": 2.0, "mode": "a", "flag": False, "count": None}, {"alpha": 3.0, "mode": "a", "flag": False,
                                                                                 "count": None}
    for config in (first, second):
        request = {"parent": "input", "param_delta": {"alpha": config["alpha"]}, "params": config,
                   "config_key": toy.config_key(config), "expected_effect": {}, "notes": ""}
        toy._labels = {"input": 0}
        toy._candidate(request)
    assert toy.label(2, toy.trial_params(second)) == "agent_2"
    assert [toy.label(i) for i in range(4)] == ["input", "agent_1", "agent_2", "call_3"]


class FocusEngine(ToyEngine):
    """A ToyEngine whose round plans carry focus_levers: each plan is (focus_levers, moves)."""

    def __init__(self, plans, final=(), repairs=()):
        super().__init__(final=final)
        self.focus_plans, self.focus_repairs = list(plans), list(repairs)

    def generate(self, content, response_format=None):
        if response_format is not WavePlan:
            return super().generate(content, response_format)
        self.asked.append((response_format.__name__, content[0]))
        if "were rejected" in content[0]:
            focus, moves = self.focus_repairs.pop(0) if self.focus_repairs else ([], [])
            return {"content": WavePlan(analysis="repair", focus_levers=focus, moves=moves)}
        focus, moves = self.focus_plans.pop(0) if self.focus_plans else ([], [])
        return {"content": WavePlan(analysis="a", focus_levers=focus, moves=moves)}


def test_the_narrowing_is_the_agents_own_focus_and_never_widens(toy_card):
    class Narrow(Toy):
        narrowing = True

    engine = FocusEngine([([], [mv(alpha=2, count=2)]),                 # explore: several levers in one call
                          (["alpha"], [mv("agent_1", count=3)]),            # narrow to alpha, but a count move
                          (["alpha"], [mv("agent_2", alpha=4)])],
                         repairs=[([], [mv("agent_1", alpha=3)])], final=["agent_3"])  # the repair keeps the focus
    toy, resolution, waves = run_toy(engine, cls=Narrow)
    assert waves == [1, 1, 1] and resolution.details["focus_history"] == [["alpha"], ["alpha"]]
    assert [v.param_delta for _, v, _ in resolution.trials[1:]] == [{"alpha": 2.0, "count": 2}, {"alpha": 3.0},
                                                                   {"alpha": 4.0}]
    rounds = engine.questions()
    assert "The search goes broad, then narrow" in rounds[0] and "exploring -- no focus named yet" in rounds[0]
    assert "not in this round's focus_levers ['alpha']" in rounds[2]                     # the repair question
    assert rounds[2].endswith("(an empty focus_levers keeps this round's).")
    assert "Now: focused on ['alpha']" in rounds[3]
    stop = next(q for q in rounds if "You returned no moves" in q)     # the facts say what can still change
    assert "under your focus ['alpha'], only those levers can change from here on" in stop
    # the one rule, move by move: once a round ran under a focus, name it again, the same levers or fewer
    toy._pending_focus = []
    assert toy._schedule_problem({"changes": [{"lever": "alpha"}]}, {}).startswith("name focus_levers")
    toy._pending_focus = ["alpha", "count"]
    assert "widen your previous focus" in toy._schedule_problem({"changes": [{"lever": "alpha"}]}, {})
    toy._pending_focus = ["alpha"]
    assert "not in this round's focus_levers" in toy._schedule_problem({"changes": [{"lever": "count"}]}, {})
    assert toy._schedule_problem({"changes": [{"lever": "alpha"}]}, {}) is None
    assert "broad, then narrow" not in Toy().move_text()                 # without the opt-in, no narrowing


def test_a_stop_with_rounds_left_is_asked_once_more_with_the_facts(toy_card):
    engine = ToyEngine([[mv(alpha=3)], [], [mv("agent_1", alpha=4)]], final=["agent_2"])
    toy, resolution, waves = run_toy(engine)
    assert waves == [1, 1] and toy.stop_checks == 2 and resolution.details["stop_checks"] == 2
    assert [p["question"] for p in toy.plans] == ["round", "round", "stop_check", "round", "stop_check"]
    checks = [q for q in engine.questions() if "You returned no moves" in q]
    assert len(checks) == 2 and "the search ends here (7 rounds remain; the same result in fewer" in checks[0]
    assert "the best admissible call so far: agent_1 (quality" in checks[0] and "'mode'" in checks[0]
    assert checks[0].endswith("Return moves if\ngains are left to find, or no moves to confirm the stop.")
    plain = Toy()
    plain.confirm_stop, plain.llm_engine = False, ToyEngine([[]])
    assert plain.plan_round(_toy_input(), {}, 2, 8) == [] and plain.stop_checks == 0
    assert len(plain.llm_engine.asked) == 1 and plain.stop_reason == "satisfied"
    stalled = Toy()                        # the plateau would end the search anyway: no extra question
    stalled.plateau_after_plan, stalled.llm_engine = True, ToyEngine([[]])
    stalled._stall = lambda inp, attempts: stalled.patience
    assert stalled.plan_round(_toy_input(), {}, 2, 8) == [] and stalled.stop_checks == 0
