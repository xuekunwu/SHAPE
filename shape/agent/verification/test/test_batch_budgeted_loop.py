"""Budgeted VRR for the batch corrector: ``BatchCorrectionVerifier`` through StepVerifier's wave mode on
a simulated tool (the calibration readouts are a function of the parameters). Rules measure and guard;
every tool call, the round sizes, the stop and the final choice are the reasoner's."""
import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.batch_correction_unified import (
    BUDGET, REASONER_RETRY_SECONDS, UNCORRECTED, AdmissionVerdict, BatchCorrectionVerifier, BatchTask, Expectation, FinalChoice, Hypothesis,
    LeverChange, WaveMove, WavePlan, _READOUTS, adapt_batch_correction, admission_concerns, admission_rule,
    call_threads, concurrency_cap,
)
from shape.tools.batch_corrector.tool import DEFAULT_CONFIG
from shape.utils.schemas import VLRRResult, card_levers

DEFAULT = dict(DEFAULT_CONFIG)
EFFECTIVE = {"nclust": 100, "max_iter_harmony": 10, "max_iter_kmeans": 4, "epsilon_cluster": 1e-3,
             "epsilon_harmony": 1e-2, "block_size": 0.05, "tau": 0.0, "n_rows": 10000, "input_pca": 1000,
             "lambda_value": "dynamic"}
INPUT = np.zeros((4, 2))                   # the simulated tool input; the readouts of the input are RAW
RAW = {"overall": 0.50, "bio_mean": 0.58, "nonrep_mean_map": 0.54, "batch_mixing_output": 0.40}


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    from shape.agent.verification import base

    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)      # the shared loop reads it at call time


def more_clusters_help(params):
    """overall grows with log2(nclust / 100); bio_mean and nonrep_mean_map stay above the input's."""
    if isinstance(params, np.ndarray):
        return dict(RAW)
    k = params.get("nclust") or 100
    return {"overall": 0.55 + 0.01 * max(0.0, math.log2(k / 100)), "bio_mean": 0.59, "nonrep_mean_map": 0.57,
            "batch_mixing_output": 0.80 + 0.01 * math.log2(k / 100 + 1)}


class View:
    """A calibration view of the simulated tool: its output is the call's parameters (the input: INPUT),
    and the calibration readouts are `readout(output)`. Two reference groups in both batch groups."""

    def __init__(self, readout, jitter=()):
        self.readout, self.rows, self.jitter, self.scored = readout, np.arange(4), list(jitter), 0
        self.meta = pd.DataFrame({"Metadata_JCP2022": ["JCP2022_1", "JCP2022_1", "JCP2022_2", "JCP2022_2"],
                                  "Metadata_PertType": ["trt"] * 4})
        self.batch_key, self.label_key, self.plate_types = "source", "Metadata_JCP2022", ()

    def score(self, matrix):
        out = {k: v for k, v in self.readout(matrix).items() if k not in _READOUTS}
        if isinstance(matrix, np.ndarray) and self.jitter:        # the metric's run-to-run variation
            out["overall"] += self.jitter[self.scored % len(self.jitter)]
            self.scored += 1
        return out


def input_result(readout, detected=True):
    gt_free = {k: v for k, v in readout(INPUT).items() if k in _READOUTS}
    return {"status": UNCORRECTED, "corrected_cls_output": INPUT, "batch_effect": {"detected": detected},
            "gt_free_metrics": {"batch_effect_detected": detected, **gt_free}, "telemetry": {},
            "provenance": {"input_shape": [10000, 1000]}, "preprocess": {}, "seconds": 0.0}


def simulated_tool(readout, crash=None, calls=None):
    def run(params):
        flat = dict(params["harmony_params"])
        if calls is not None:
            calls.append(flat)
        if crash and flat.get(crash) != DEFAULT[crash]:
            raise RuntimeError(f"{crash} crashed the tool")
        gt_free = {k: v for k, v in readout(flat).items() if k in _READOUTS}
        return {"status": "CORRECTED_HARMONY", "corrected_cls_output": flat, "batch_effect": {"detected": True},
                "gt_free_metrics": {"batch_effect_detected": True, **gt_free},
                "telemetry": {"n_clusters": 100, "cluster_mass_min_max": [0.001, 0.02]},
                "provenance": {"input_shape": [10000, 1000]}, "preprocess": {}, "seconds": 0.0}
    return run


def move(parent="input", expected=(), diagnosis="", **changes):
    return WaveMove(parent=parent, changes=[LeverChange(lever=k, value=json.dumps(v)) for k, v in changes.items()],
                    diagnosis=diagnosis, expected=[Expectation(metric=m, direction=d) for m, d in expected])


class Engine:
    """Answers each round prompt with the next plan (then no moves: satisfied), a repair prompt with
    `repair`, the final prompt with the next of `final`; no admission answer (the rule decides)."""

    def __init__(self, rounds=(), final=(), repair=None, hypotheses=()):
        self.rounds, self.final, self.repair, self.hypotheses = list(rounds), list(final), repair, list(hypotheses)
        self.prompts = []

    def generate(self, content, response_format=None):
        prompt = content[0]
        self.prompts.append(prompt)
        if response_format is AdmissionVerdict:
            rule = prompt.split("Deterministic rule outcome: ", 1)[1].split(".", 1)[0]
            return {"content": AdmissionVerdict(decision=rule, reason_code="rule", reasoning="follow the measured rule")}
        if response_format is FinalChoice:
            return {"content": FinalChoice(call=self.final.pop(0) if self.final else None, reasoning="t"), "usage": {}}
        if response_format is WavePlan:
            if "were rejected" in prompt:
                return {"content": WavePlan(analysis="repair", moves=list(self.repair or [])), "usage": {}}
            moves = self.rounds.pop(0) if self.rounds else []
            return {"content": WavePlan(analysis="t", hypotheses=self.hypotheses, moves=moves), "usage": {}}
        raise RuntimeError("no admission answer in this test")


DEFAULT_ENGINE = object()


def run(readout=more_clusters_help, engine=DEFAULT_ENGINE, budget=BUDGET, effective=None, crash=None, detected=True,
        admission=None, band=0.002, log=None, cores=16, jitter=(), calls=None):
    verifier = BatchCorrectionVerifier(log=log)
    verifier.llm_engine = Engine() if engine is DEFAULT_ENGINE else engine
    verifier.effective_defaults = dict(effective or EFFECTIVE)
    task = BatchTask(batch=np.array(["x", "y", "x", "y"]), features=INPUT, calibration=View(readout, jitter),
                     noise_band=band, brief="test", admission=admission, cores=cores)
    events = []
    resolution = StepVerifier(max_iters=1 + budget, observer=events.append).resolve_stage(
        verifier, adapt_batch_correction, simulated_tool(readout, crash, calls), {}, input_result(readout, detected),
        upstream_ctx={"batch_task": task})
    return resolution, verifier, [e["size"] for e in events if e["event"] == "wave"]


def _verifier(engine=None, effective=None):
    verifier = BatchCorrectionVerifier()
    verifier.llm_engine, verifier.effective_defaults = engine, dict(effective or EFFECTIVE)
    verifier.cap = 2
    return verifier


def _inp():
    return VLRRResult("WARNING", RAW["overall"], dict(RAW), action="retune_here")


def _call(parent, params, readout=more_clusters_help, delta=None):
    vres = VLRRResult("PASS", readout(params)["overall"], readout(params), action="retune_here")
    vres.param_delta = delta
    return parent, dict(params), vres


def test_card_declares_every_tuned_lever():
    assert {"theta", "sigma", "lambda_value", "nclust", "input_pca", "input_scale", "epsilon_harmony",
            "max_iter_harmony"} <= card_levers("batch_corrector")


def test_admission_rule():
    assert admission_rule({"gate_all_rows": {"detected": False}}) == "SKIP"
    assert admission_rule({"gate_all_rows": {"detected": True}, "reference_groups_in_2plus_batch_groups": 0}) == "ABSTAIN"
    assert admission_rule({"gate_all_rows": {"detected": True}, "reference_groups_in_2plus_batch_groups": 140}) == "RUN"


def test_no_tool_call_without_run_admission():
    calls = []
    skip, _, waves = run(detected=False, calls=calls)
    assert waves == [] and calls == [] and len(skip.trials) == 1 and skip.status == "satisfied"
    assert skip.reason == "no_batch_effect" and skip.result is skip.trials[0][2]     # the uncorrected input
    assert skip.details["tuning_gate"] == {
        "required": True, "source": "input", "decision": "SKIP", "allowed": False, "reason": "no_batch_effect"}
    abstain, _, waves = run(admission={"decision": "ABSTAIN", "rule": "ABSTAIN"}, calls=calls)
    assert waves == [] and calls == [] and abstain.result is None and abstain.status == "conceded"
    assert abstain.reason == "separability_unverifiable" and abstain.fault == "ceiling"
    assert abstain.details["tuning_gate"]["decision"] == "ABSTAIN"
    assert abstain.details["tuning_gate"]["allowed"] is False


@pytest.mark.parametrize("decision", ["RUN", "SKIP", "ABSTAIN", "NOT_SEARCHED"])
def test_precomputed_admission_uses_the_shared_gate_without_changing_delivery(decision):
    engine = Engine([[move(nclust=200)]], final=["agent_1"])
    calls = []
    resolution, verifier, waves = run(engine=engine, admission={"decision": decision, "rule": decision},
                                      calls=calls)
    gate = resolution.details["tuning_gate"]
    assert gate["decision"] == decision and gate["source"] == "input"
    assert gate["allowed"] is (decision == "RUN")
    if decision == "RUN":
        assert waves == [1] and len(calls) == 1 and resolution.details["call"] == "agent_1"
    else:
        assert waves == calls == engine.prompts == []
        expected = {"SKIP": "no_batch_effect", "ABSTAIN": "separability_unverifiable",
                    "NOT_SEARCHED": "execution_error"}[decision]
        assert resolution.reason == expected
        assert (resolution.result is resolution.trials[0][2]) is (decision == "SKIP")


def test_four_stage_batch_runs_only_after_admission(monkeypatch):
    from shape.agent.verification.test.test_four_stage import CoveringEngine

    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_BATCH_CORRECTION", "four_stage")
    engine = CoveringEngine(Engine([[move(nclust=200)]], final=["agent_1"]))
    resolution, _, waves = run(engine=engine, admission={"decision": "RUN"})
    assert waves == [1] and resolution.details["call"] == "agent_1"
    assert resolution.details["search_strategy"]["confirmation"]["status"] == "existing_checks_passed"
    blocked, _, waves = run(engine=engine, admission={"decision": "ABSTAIN"})
    assert waves == [] and blocked.details["search_strategy"]["status"] == "not_entered"


def test_trial_0_is_the_input_and_every_call_is_the_reasoners():
    engine = Engine([[move(nclust=200), move(theta=2.5)],
                     [move("agent_1", nclust=400), move("agent_1", nclust=300)],
                     [move("agent_3", nclust=800), move("agent_3", sigma=0.07)],
                     [move("agent_5", nclust=1600), move("agent_5", nclust=1200)]], final=["agent_7"])
    calls = []
    resolution, verifier, waves = run(engine=engine, calls=calls)
    assert resolution.trials[0][0] == {} and resolution.trials[0][2]["status"] == UNCORRECTED
    assert len(calls) == 8 == len(resolution.trials) - 1 and waves == [2, 2, 2, 2]
    assert DEFAULT not in calls                               # the reasoner never asked for the defaults
    assert not any(v.notes.startswith("fallback") for _, v, _ in resolution.trials[1:])
    assert resolution.params["harmony_params"]["nclust"] == 1600 and resolution.details["choice"] == "agent_choice"
    assert resolution.status == "satisfied" and resolution.reason == "satisfied"     # round 5: no moves
    assert resolution.parents == [None, 0, 0, 1, 1, 3, 3, 5, 5]


def test_the_budget_is_a_limit_on_rounds_not_calls():
    engine = Engine([[move(nclust=200), move(theta=2.5)], [move("agent_1", nclust=400), move("agent_1", nclust=300)],
                     [move("agent_3", nclust=800)]], final=["agent_3"])
    resolution, verifier, waves = run(engine=engine, budget=2)
    assert waves == [2, 2] and len(resolution.trials) == 5 and resolution.reason == "max_iters_reached"
    round_prompts = [p for p in engine.prompts if p.startswith("You tune")]
    assert "at most 2 more; calls so far" in round_prompts[0] and "at most 1 more" in round_prompts[1]


def test_round_sizes_are_the_reasoners_within_the_cap():
    assert [concurrency_cap(c) for c in (1, 8, 16, 32, 64)] == [1, 1, 2, 4, 8]
    logged = []
    engine = Engine([[move(nclust=200), move(theta=2.5), move(sigma=0.07)], [move("agent_1", nclust=400)]],
                    final=["agent_3"])
    resolution, verifier, waves = run(engine=engine, cores=2, log=lambda kind, **f: logged.append(kind))
    assert waves == [2, 1] and "dropped_move" in logged            # cap 2: the third move is dropped
    assert verifier.stop_reason == "satisfied" and resolution.reason == "satisfied"
    engine = Engine([[move(nclust=200), move(theta=2.5)], [move(sigma=0.07), move(tau=5)]])
    _, _, waves = run(engine=engine, cores=64, budget=3)
    assert waves == [2, 2]                                          # the reasoner's sizes; then no moves


def test_a_move_may_change_several_levers():
    engine = Engine([[move(nclust=400, theta=3.0)]], final=["agent_1"])
    resolution, _, _ = run(engine=engine)
    assert resolution.trials[1][1].param_delta == {"nclust": 400, "theta": 3.0}
    with pytest.raises(ValueError, match="twice"):
        _verifier().validate({"parent": "input", "changes": [{"lever": "nclust", "value": "200"},
                                                             {"lever": "nclust", "value": "400"}]}, {}, set())
    with pytest.raises(ValueError, match="no lever changed"):
        _verifier().validate({"parent": "default", "changes": []}, {"default": _call("input", DEFAULT)}, set())


def test_rejected_moves_get_one_repair_question():
    invalid = WaveMove(changes=[LeverChange(lever="input_pca", value=":")])
    engine = Engine([[invalid, move(input_scale="zscore")]], repair=[move(input_scale="zscore"), move(nclust=200)])
    requests = _verifier(engine).plan_round(_inp(), {}, 2, 8)
    assert len(engine.prompts) == 2 and "were rejected" in engine.prompts[1] and "input_pca" in engine.prompts[1]
    # the repair's repeat of an accepted move is refused, its new move fills the slot
    assert [r["param_delta"] for r in requests] == [{"input_scale": "zscore"}, {"nclust": 200}]
    full = Engine([[invalid, move(input_scale="zscore"), move(nclust=200)]])
    assert len(_verifier(full).plan_round(_inp(), {}, 2, 8)) == 2 and len(full.prompts) == 1   # no room: no repair


def test_typed_values_are_accepted_and_invalid_repairs_stop_without_fallback(monkeypatch):
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)
    typed = WaveMove(changes=[LeverChange(lever="nclust", value=400.0), LeverChange(lever="theta", value=4.0)])
    requests = _verifier(Engine([[typed]])).plan_round(_inp(), {}, 2, 8)
    assert [r["param_delta"] for r in requests] == [{"nclust": 400, "theta": 4.0}]
    with pytest.raises(ValueError, match="is not a number"):
        _verifier().coerce("theta", True)                  # a bool is not a number for a numeric lever
    # the JUMP run's failure (job 46645205): the round's values and the repair's were all empty strings
    empty = [WaveMove(parent="c04", changes=[LeverChange(lever="theta", value="")]),
             WaveMove(parent="c04", changes=[LeverChange(lever="nclust", value="")])]
    verifier = _verifier(Engine([empty], repair=empty))
    requests = verifier.plan_round(_inp(), {"c04": _call("input", DEFAULT)}, 2, 8)
    assert requests == [] and verifier.stop_reason == "no_valid_move"
    assert verifier.fallback_rounds == 0


def test_without_a_reasoner_batch_admission_is_fatal():
    from shape.engine.errors import NoEngineError
    calls = []
    with pytest.raises(NoEngineError):
        run(engine=None, calls=calls)
    assert calls == []


def test_the_hypothesis_list_is_shown_in_the_next_round():
    hypothesis = Hypothesis(id="H1", statement="the 100-cluster cap limits the correction", status="open")
    engine = Engine([[move(nclust=200)], [move("agent_1", nclust=400)]], final=["agent_2"], hypotheses=[hypothesis])
    resolution, verifier, _ = run(engine=engine)
    rounds = [p for p in engine.prompts if p.startswith("You tune")]
    assert "100-cluster cap" not in rounds[0] and "100-cluster cap" in rounds[1]
    assert resolution.details["hypotheses"][0]["id"] == "H1"


def test_an_empty_round_stops_the_search():
    resolution, verifier, waves = run(engine=Engine([[move(nclust=200)]], final=["agent_1"]))
    assert waves == [1] and len(resolution.trials) == 2 and verifier.stop_reason == "satisfied"
    assert resolution.status == "satisfied" and resolution.reason == "satisfied"


def test_a_plateau_stops_the_search():
    flat = lambda p: dict(RAW) if isinstance(p, np.ndarray) else {"overall": 0.55, "bio_mean": 0.59,  # noqa: E731
                                                                   "nonrep_mean_map": 0.57}
    engine = Engine([[move(nclust=200)], [move(theta=2.5)], [move(sigma=0.07)], [move(tau=5)]], final=["agent_1"])
    resolution, verifier, waves = run(flat, engine, cores=8)
    assert waves == [1, 1, 1, 1] and verifier.stop_reason == "plateau"  # calls 2, 3 and 4 did not raise the best


def _biology_cost(params):
    if isinstance(params, np.ndarray):
        return dict(RAW)
    if params.get("theta") == 3.0:        # best overall, but bio_mean falls below the input's
        return {"overall": 0.575, "bio_mean": 0.55, "nonrep_mean_map": 0.57}
    return {"overall": 0.56, "bio_mean": 0.59, "nonrep_mean_map": 0.57}


def test_an_inadmissible_final_choice_is_asked_once_more():
    resolution, _, _ = run(_biology_cost, Engine([[move(theta=3.0), move(nclust=200)]], final=["agent_1", "agent_2"]))
    assert resolution.details["admissible"] == ["agent_2"] and resolution.details["call"] == "agent_2"
    retry = resolution.details["final_answers"]
    assert [a["call"] for a in retry] == ["agent_1", "agent_2"]
    twice, _, _ = run(_biology_cost, Engine([[move(theta=3.0), move(nclust=200)]], final=["agent_1", "agent_1"]))
    assert twice.result is None and twice.status == "conceded" and twice.reason == "final_choice_inadmissible"
    kept, _, _ = run(_biology_cost, Engine([[move(theta=3.0), move(nclust=200)]], final=[None]))
    assert kept.result is None and kept.reason == "agent_kept_input"


def test_no_admissible_call_keeps_the_input():
    engine = Engine([[move(theta=3.0)]], final=["agent_1"])
    resolution, _, _ = run(_biology_cost, engine)
    assert resolution.result is None and resolution.reason == "no_admissible_call"
    assert not any(p.startswith("Choose the final") for p in engine.prompts)


def test_the_noise_is_measured_on_the_input():
    resolution, verifier, _ = run(engine=Engine([[move(nclust=200)]], final=["agent_1"]), jitter=(0.0, 0.003, -0.003))
    assert verifier.measured_noise["overall"] == pytest.approx(2 * math.sqrt(2) * 0.003, rel=1e-6)
    assert verifier.noise["overall"] == pytest.approx(verifier.measured_noise["overall"])
    assert verifier.noise["bio_mean"] == 0.002                           # deterministic: the stated noise
    small = lambda p: dict(RAW) if isinstance(p, np.ndarray) else {**RAW, "overall": 0.505}  # noqa: E731
    within, _, _ = run(small, Engine([[move(nclust=200)]], final=["agent_1"]), jitter=(0.0, 0.003, -0.003))
    assert within.reason == "no_admissible_call"                         # +0.005 is within the measured noise
    beyond, _, _ = run(small, Engine([[move(nclust=200)]], final=["agent_1"]))
    assert beyond.details["call"] == "agent_1"                               # but beyond the stated 0.002


def test_prompts_carry_information_only():
    engine = Engine([[move(nclust=200), move(theta=2.5)], [move("agent_1", nclust=400)]], final=["agent_3"])
    run(engine=engine)
    text = "\n".join(engine.prompts)
    for rule in ("screen", "spread the wave", "Continue along", "lever families", "at least one untried",
                 "the loop enforces", "robust", "maximin"):
        assert rule not in text, rule
    assert "JCP2022" not in text
    rounds = [p for p in engine.prompts if p.startswith("You tune")]
    assert "'nclust': 100" in rounds[0] and "'n_rows': 10000" in rounds[0] and "### nclust" in rounds[0]
    assert '"vs_parent"' in rounds[1] and '"prediction"' in rounds[1] and "beyond noise" in rounds[1]


def test_card_lists_every_tool_parameter_with_the_code_values():
    import re

    from shape.agent.verification.base import _ADDITIVE, PARAM_BOUNDS
    from shape.tools.batch_corrector.tool import INPUT_SCALES
    from shape.utils.schemas import card_lever_fields

    fields = card_lever_fields("batch_corrector")
    assert set(fields) == set(DEFAULT)                      # every accepted parameter, nothing else
    for lever, f in fields.items():
        assert f["Family"] in ("harmony", "input", "solver", "seed"), lever
        default = DEFAULT[lever]
        assert f["Default"] == ("null" if default is None else f"{default:g}" if isinstance(default, float) else str(default)), lever
        if lever == "input_scale":
            assert f["Range"] == ", ".join(INPUT_SCALES)
            continue
        lo, hi, step = PARAM_BOUNDS[lever]
        assert [float(x) for x in re.findall(r"[-+\d.e]+", f["Range"])] == [lo, hi], lever
        assert "Step" not in f and "Fallback" not in f


def test_card_loop_settings_match_the_code():
    from shape.agent.verification import base
    from shape.agent.verification.verifiers import batch_correction_unified as v
    from shape.tools.base import load_tuning_card
    from shape.tools.batch_corrector import tool

    rows, where, module = {}, {}, None
    for line in (load_tuning_card("batch_corrector") or "").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            module = module if cells[2] == "same" else cells[2].strip("`")
            rows[cells[0].strip("`")], where[cells[0].strip("`")] = cells[1], module
    expected = {"BUDGET": base.BUDGET, "PATIENCE": base.PATIENCE, "REASONER_ATTEMPTS": base.REASONER_ATTEMPTS,
                "REASONER_RETRY_SECONDS": REASONER_RETRY_SECONDS,   # as imported: the fixture patches base's
                "_LEDGER_REL_FLOOR": base._LEDGER_REL_FLOOR, "NOISE_BAND": v.NOISE_BAND,
                "NOISE_REPEATS": v.NOISE_REPEATS, "THREADS_PER_CALL": v.THREADS_PER_CALL,
                "ADMISSION_MIN_SHARED_GROUPS": v.ADMISSION_MIN_SHARED_GROUPS,
                "READOUT_ROWS": tool.READOUT_ROWS, "READOUT_NEIGHBOURS": tool.READOUT_NEIGHBOURS,
                "DETECT_MARGIN": tool.DETECT_MARGIN}
    for name, value in expected.items():
        assert float(rows[name]) == value, name
    assert rows["SCORE"] == v.SCORE and rows["PRESERVE"] == ", ".join(v.PRESERVE)
    assert rows["OBJECTIVES"] == ", ".join(v.OBJECTIVES)
    modules = {"base.py": base, "verifiers/batch_correction_unified.py": v, "tools/batch_corrector/tool.py": tool}
    assert all(hasattr(modules[where[name]], name) for name in rows), where
    assert {name for name, m in where.items() if m == "base.py"} == {       # the shared layer's settings
        "BUDGET", "PATIENCE", "REASONER_ATTEMPTS", "REASONER_RETRY_SECONDS", "_LEDGER_REL_FLOOR"}
    loop = BatchCorrectionVerifier
    assert (loop.budget, loop.patience, loop.threads_per_call, loop.score_key, loop.objectives) == (
        base.BUDGET, base.PATIENCE, v.THREADS_PER_CALL, v.SCORE, v.OBJECTIVES)
    assert not {"WAVE_SIZE", "SCREEN_WAVE_SIZE", "residual_offset_ceiling", "accept_score"} & set(rows)


S1_LIKE = {"gate_all_rows": {"detected": True}, "n_batch_groups": 3, "reference_groups": 146,
           "reference_groups_in_2plus_batch_groups": 146, "reference_groups_in_all_batch_groups": 145,
           "negative_controls_per_batch_group": {"a": 130, "b": 128, "c": 64},
           "gate_negative_controls_only": {"detected": True}}


class AdmissionEngine:
    def __init__(self, verdict):
        self.verdict, self.prompts = verdict, []

    def generate(self, content, response_format=None):
        self.prompts.append(content[0])
        return {"content": self.verdict, "usage": {}}


def test_a_veto_citing_no_holding_concern_is_rejected():
    assert not any(admission_concerns(S1_LIKE).values())
    for concerns in ([], ["no_batch_effect_on_negative_controls"]):
        engine = AdmissionEngine(AdmissionVerdict(decision="ABSTAIN", reason_code="x", reasoning="biology changed",
                                                  concerns=concerns))
        result = _verifier(engine).admit(S1_LIKE)
        assert result["decision"] == "RUN" and result["veto"]["accepted"] is False


def test_a_veto_citing_a_holding_concern_is_accepted():
    evidence = {**S1_LIKE, "reference_groups_in_all_batch_groups": 5}           # fewer than 10 shared
    engine = AdmissionEngine(AdmissionVerdict(decision="ABSTAIN", reason_code="x", reasoning="little overlap",
                                              concerns=["few_reference_groups_in_all_batch_groups"]))
    result = _verifier(engine).admit(evidence)
    assert result["decision"] == "ABSTAIN" and result["veto"]["accepted"] is True
    assert '"few_reference_groups_in_all_batch_groups": true' in engine.prompts[0]


def test_admission_counts_shared_groups_not_their_fraction():
    # a lab-specific compound-plate design: few groups are in every lab, but 140 of them are
    mixed = {**S1_LIKE, "reference_groups": 2000, "reference_groups_in_all_batch_groups": 140}
    assert admission_concerns(mixed)["few_reference_groups_in_all_batch_groups"] is False


def test_the_shared_groups_threshold_adapts_to_the_design_size():
    """max(2, min(ADMISSION_MIN_SHARED_GROUPS, the distinct non-control reference groups)): a 3-condition design
    gets 3; a JUMP-scale design keeps 10 (its prompt text unchanged); without the count, 10."""
    from shape.agent.verification.verifiers.batch_correction_unified import (
        ADMISSION_MIN_SHARED_GROUPS, admission_min_shared_groups)

    few = "few_reference_groups_in_all_batch_groups"
    three = {**S1_LIKE, "reference_groups": 3, "reference_groups_in_2plus_batch_groups": 3}
    assert admission_min_shared_groups(three) == 3
    assert admission_concerns({**three, "reference_groups_in_all_batch_groups": 3})[few] is False
    assert admission_concerns({**three, "reference_groups_in_all_batch_groups": 2})[few] is True
    assert admission_min_shared_groups(S1_LIKE) == ADMISSION_MIN_SHARED_GROUPS == 10
    assert admission_min_shared_groups({"reference_groups": 1}) == admission_min_shared_groups({"reference_groups": 0}) == 2
    assert admission_min_shared_groups({}) == 10
    for evidence, threshold in (({**three, "reference_groups_in_all_batch_groups": 2}, 3),
                                ({**S1_LIKE, "reference_groups_in_all_batch_groups": 5}, 10)):
        engine = AdmissionEngine(AdmissionVerdict(decision="ABSTAIN", reason_code="x", reasoning="r", concerns=[few]))
        assert _verifier(engine).admit(evidence)["decision"] == "ABSTAIN"
        assert f"({few}: fewer than {threshold} reference groups\n" in engine.prompts[0]


def test_agent_can_combine_two_measured_changes():
    attempts = {"agent_1": _call("input", {**DEFAULT, "nclust": 200}),
                "agent_2": _call("input", {**DEFAULT, "input_pca": 500})}
    request = _verifier().validate(move("agent_1", input_pca=500).model_dump(), attempts, set())
    assert request["params"]["nclust"] == 200 and request["params"]["input_pca"] == 500


def test_failed_admission_records_execution_error_without_a_tool_call():
    class Down(Engine):
        def generate(self, *args, **kwargs):
            raise TimeoutError("gateway timeout")
    calls = []
    resolution, verifier, waves = run(engine=Down(), calls=calls)
    assert calls == waves == []
    assert resolution.reason == "execution_error"
    assert verifier.admission["decision"] == "NOT_SEARCHED"
    assert resolution.details["execution_errors"][0]["question"] == "admission"


def test_a_crashed_configuration_is_not_proposed_again():
    resolution, _, _ = run(engine=Engine([[move(theta=2.5)], [move(theta=2.5)]], repair=[]), crash="theta")
    crashed = [i for i, (_, _, r) in enumerate(resolution.trials) if isinstance(r, dict) and r.get("_attempt_status")]
    assert len(crashed) == 1 and resolution.details["failed"][0]["param_delta"] == {"theta": 2.5}
    assert not any("theta" in (v.param_delta or {}) for _, v, _ in resolution.trials[crashed[0] + 1:])


def test_values_are_json_literals_or_plain_numbers_and_errors_say_what_was_received():
    verifier = _verifier()
    assert verifier.coerce("sigma", ".07") == 0.07 and verifier.coerce("sigma", '"0.07"') == 0.07
    assert verifier.coerce("nclust", "400") == 400 and verifier.coerce("lambda_value", "None") is None
    for text in (":4", ""):
        with pytest.raises(ValueError, match="is not a number"):
            verifier.coerce("theta", text)
    with pytest.raises(ValueError, match="is not a number"):
        verifier.coerce("theta", "[4]")
    # a null for a lever that cannot be null says how to write a value it accepts (its lower bound)
    for lever, example in (("theta", "0"), ("sigma", "0.0001"), ("input_pca", "2")):
        with pytest.raises(ValueError, match=f'{lever} cannot be null: write its value inside the value string, '
                                             f'e.g. "value": "{example}"'):
            verifier.coerce(lever, "null")


def test_a_value_that_acts_as_the_parents_is_the_same_configuration():
    verifier = _verifier()                                    # null nclust resolves to 100 here
    with pytest.raises(ValueError, match="acts as input's nclust"):
        verifier.validate({"parent": "input", "changes": [{"lever": "nclust", "value": "100"}]}, {}, set())
    with pytest.raises(ValueError, match="acts as"):
        verifier.validate({"parent": "input", "changes": [{"lever": "max_iter_harmony", "value": "10"}]}, {}, set())
    assert verifier.validate({"parent": "input", "changes": [{"lever": "nclust", "value": "200"}]},
                             {}, set())["param_delta"] == {"nclust": 200}


def test_the_repair_question_names_the_moves_that_already_run():
    invalid = WaveMove(changes=[LeverChange(lever="sigma", value="")])
    engine = Engine([[move(nclust=200), invalid]], repair=[move(sigma=0.07)])
    requests = _verifier(engine).plan_round(_inp(), {}, 2, 8)
    assert "do not return them again" in engine.prompts[1] and "{'nclust': 200}" in engine.prompts[1]
    assert "is not a number" in engine.prompts[1] and len(requests) == 2


def test_a_failed_reasoner_question_is_tried_again():
    class Flaky(Engine):
        def __init__(self):
            super().__init__([[move(nclust=200)]])
            self.failures = 1

        def generate(self, content, response_format=None):
            if response_format is WavePlan and self.failures:
                self.failures -= 1
                raise TimeoutError("gateway timeout")
            return super().generate(content, response_format)

    logged = []
    verifier = _verifier(Flaky())
    verifier.log = lambda kind, **f: logged.append(kind)
    assert [r["param_delta"] for r in verifier.plan_round(_inp(), {}, 2, 8)] == [{"nclust": 200}]
    assert logged.count("reasoner_error") == 1 and verifier.reasoner_failures == 0


def test_the_round_prompt_shows_the_time_left_when_the_caller_knows_it():
    import time

    verifier = _verifier()
    verifier.started, verifier.deadline = time.time(), time.time() + 5400
    assert "time left before the stage must end" in verifier.round_prompt(_inp(), {}, 2, 8)
    assert "1.50 h" in verifier.round_prompt(_inp(), {}, 2, 8)
    verifier.deadline = None
    assert "time left" not in verifier.round_prompt(_inp(), {}, 2, 8)


def test_input_pca_that_keeps_every_feature_is_rejected():
    with pytest.raises(ValueError, match="keeps every"):
        _verifier(effective={**EFFECTIVE, "input_pca": 1018}).coerce("input_pca", 1018)


def test_a_dynamic_lambda_default_is_not_stepped_and_minus_one_is_null():
    verifier = _verifier()
    assert verifier.coerce("lambda_value", "-1") is None                  # -1 = null = dynamic (harmonypy 2.0)
    # null is lambda's default: the change repeats the parent's value and is dropped -> the tool at its defaults
    request = verifier.validate({"parent": "input", "changes": [{"lever": "lambda_value", "value": "-1"}]}, {}, set())
    assert request["param_delta"] == {} and request["params"] == verifier._params({}, "input")


def test_the_guard_compares_with_the_uncorrected_input():
    verifier = _verifier()
    verifier.noise = {"overall": 0.002, "bio_mean": 0.002, "nonrep_mean_map": 0.002}
    _, _, good = _call("input", {**DEFAULT, "nclust": 200})
    assert verifier.guard(_inp(), good) == []
    worse = VLRRResult("PASS", 0.60, {"overall": 0.60, "bio_mean": 0.575, "nonrep_mean_map": 0.54})
    assert [f.split()[0] for f in verifier.guard(_inp(), worse)] == ["bio_mean"]
    flat = VLRRResult("PASS", 0.501, {"overall": 0.501, "bio_mean": 0.58, "nonrep_mean_map": 0.54})
    assert [f.split()[0] for f in verifier.guard(_inp(), flat)] == ["overall"]


# The prompts and the response schemas are the agent's whole information, so a change to them must be
# deliberate: the sha256 of every question of fixed scenarios (schema name and content, in order; the
# clock frozen) and of each schema (key order kept). A deliberate change updates the hashes with it.
# four_rounds, final_retry, plateau: their simulated design has 2 reference groups, so the threshold is 2
# 2026-10-05 (deliberate): the round prompt's Stop line names the plan's `deliver`, the time line shows the
# tokens of the agent's own questions on the unit and the untried list is followed by its reminder note; WavePlan gains `deliver`. With
# that text reverted, every changed hash was checked to equal its previous value.
# 2026-10-06 (deliberate): the round and final prompts put the text the same for every question first (wording
# unchanged: each question keeps its multiset of lines); a move's LeverChange has only lever and value. With the
# HEAD round_prompt / final_prompt the five changed prompt hashes came back, and with the HEAD LeverChange the
# WavePlan hash.
# 2026-10-07 (deliberate, the merge of migrate-main into cropper-5arm-fixes): the merged prompts are migrate-main's
# text (calls named by role: default, reference, agent_<n>, fallback_<n>) with this branch's additions above (the
# deliver field, the untried note, the token fact, the order). With `label` patched back to c01, c02, ... the
# merged code gives every hash of this branch before the merge; against migrate-main's prompts the only lines
# that differ are those additions. repair and deadline name no call: unchanged.
# 2026-10-07 (deliberate, the second merge of migrate-main, 18083735): migrate-main's typed LeverChange value, its
# stop check, plateau in rounds and narrowing, with this branch's additions; the Stop line adds that a stop with
# rounds left is asked once more and the confirming answer's deliver counts, and the narrowing's "Now:" line moves
# to the round's part of the prompt (the fixed prefix stays the same for every question). Against migrate-main's
# prompts, each question's lines differ only by this branch's additions (the Stop / deliver lines, the token fact,
# the untried note); with migrate-main's LeverChange and without `deliver`, WavePlan gives migrate-main's hash.
# This repair removes CARD Step/Fallback guidance and gives the fixture a valid admission answer on its first
# attempt. Snapshot the resulting role-named prompts; the legacy plan schemas and tool lever bounds stay fixed.
PROMPT_SHA256 = {
    "four_rounds": "98664f6fbefd1b50b1e589853fd404c5a82c54191098acc5d5d093bd2c7e86c5",
    "final_retry": "8f770dfdd3ee860773dcaaf290b168460486f7c036129b561934bb273a9e5a3c",
    "plateau": "90c514cae2bb860903f28c9f038e9274aeb6689973a0d5ecf36e831c3863ee23",
    "repair": "5f682d0410e73d1520e5acc82d97d61d7289e36260084d5f964e0a71ac3ce9a8",
    "deadline": "9d9da86709d3c2aade57864b9e4ccdfa1f81008e86c833903e4b39cccef2db43",
    "admission": "b5414b59c433ed15886dfef28b54c5d6a4928b3e6f3537b88bcf3a9441186973",
}
SCHEMA_SHA256 = {
    "WavePlan": "8ea8edf793b7a36b695ee669c1c2ff0e5c446e09996539e4c82a03d5f1211255",
    "FinalChoice": "d031447feead9b62c073f4afabcab494526bfd9aa6fd331b7de83176c782d5f5",
    "AdmissionVerdict": "240b6a0b8591ae9aa416e65c0c5b92a9937a3b0501b3e43a2acbdcd2949467dd",
}


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, default=str).encode("utf-8")).hexdigest()


class Recorder(Engine):
    """An Engine that records every question: the schema's name and the content."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def generate(self, content, response_format=None):
        self.log.append({"schema": getattr(response_format, "__name__", None), "content": list(content)})
        return super().generate(content, response_format)


def test_the_prompts_and_schemas_are_locked(monkeypatch):
    import time

    monkeypatch.setattr(time, "time", lambda: 1_800_000_000.0)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    flat = lambda p: dict(RAW) if isinstance(p, np.ndarray) else {"overall": 0.55, "bio_mean": 0.59,  # noqa: E731
                                                                   "nonrep_mean_map": 0.57}
    scenarios = {}
    engine = Recorder([[move(nclust=200), move(theta=2.5)], [move("agent_1", nclust=400), move("agent_1", nclust=300)],
                       [move("agent_3", nclust=800), move("agent_3", sigma=0.07)],
                       [move("agent_5", nclust=1600), move("agent_5", nclust=1200)]], final=["agent_7"])
    run(engine=engine)
    scenarios["four_rounds"] = engine.log
    engine = Recorder([[move(theta=3.0), move(nclust=200)]], final=["agent_1", "agent_2"])
    run(_biology_cost, engine)
    scenarios["final_retry"] = engine.log
    engine = Recorder([[move(nclust=200)], [move(theta=2.5)], [move(sigma=0.07)], [move(tau=5)]], final=["agent_1"])
    run(flat, engine, cores=8)
    scenarios["plateau"] = engine.log
    engine = Recorder([[WaveMove(changes=[LeverChange(lever="input_pca", value=":")]), move(input_scale="zscore")]],
                      repair=[move(input_scale="zscore"), move(nclust=200)])
    _verifier(engine).plan_round(_inp(), {}, 2, 8)
    scenarios["repair"] = engine.log
    verifier = _verifier()
    verifier.started, verifier.deadline = time.time() - 1800, time.time() + 5400
    scenarios["deadline"] = [{"schema": None, "content": [verifier.round_prompt(_inp(), {}, 2, 8)]}]
    admission = AdmissionEngine(AdmissionVerdict(decision="ABSTAIN", reason_code="x", reasoning="r",
                                                 concerns=["few_reference_groups_in_all_batch_groups"]))
    _verifier(admission).admit({**S1_LIKE, "reference_groups_in_all_batch_groups": 5})
    scenarios["admission"] = [{"schema": "AdmissionVerdict", "content": [p]} for p in admission.prompts]
    assert sum(len(log) for log in scenarios.values()) == 24         # one successful admission, not three failed tries
    assert {name: _sha256(log) for name, log in scenarios.items()} == PROMPT_SHA256
    assert {s.__name__: _sha256(s.model_json_schema()) for s in (WavePlan, FinalChoice, AdmissionVerdict)} == SCHEMA_SHA256


def test_a_first_focus_needs_every_lever_group_changed_first():
    verifier = _verifier()
    assert verifier.narrowing and verifier.coverage_exempt == ("seed",)
    verifier._pending_focus = ["nclust"]
    move_nclust = {"changes": [{"lever": "nclust", "value": 400}]}
    harmony_only = {"c01": _call("input", DEFAULT, delta={"theta": 4.0})}
    why = verifier._schedule_problem(move_nclust, harmony_only)
    assert why.startswith("narrowing comes after the broad search") and "'input'" in why and "'solver'" in why
    assert "'harmony'" not in why.split("yet (")[0]                    # theta changed the harmony group
    every_group = {**harmony_only, "c02": _call("input", DEFAULT, delta={"input_pca": 64, "max_iter_harmony": 20})}
    assert verifier._schedule_problem(move_nclust, every_group) is None  # the seed group is not needed
    verifier._attempts_seen = harmony_only
    assert "groups no call has changed yet" in verifier.schedule_now_text() and "'solver'" in verifier.schedule_now_text()



def test_a_too_early_first_focus_is_repaired_into_broad_moves(monkeypatch):
    """QC 2026-10-07: the plan narrows to nclust before the input and solver groups were changed; the repair,
    told this round has no focus, returns broad moves with focus_levers empty, and they run."""
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)

    class FocusEngine(Engine):
        def generate(self, content, response_format=None):
            prompt = content[0]
            self.prompts.append(prompt)
            if "were rejected" in prompt:
                return {"content": WavePlan(analysis="repair", moves=[move("c01", input_pca=64),
                                                                       move("c01", max_iter_harmony=20)]),
                        "usage": {}}
            return {"content": WavePlan(analysis="narrow", focus_levers=["nclust"], moves=[move("c01", nclust=400)]),
                    "usage": {}}

    engine = FocusEngine()
    verifier = _verifier(engine)
    attempts = {"c01": _call("input", DEFAULT, delta={"theta": 4.0})}
    requests = verifier.plan_round(_inp(), attempts, 2, 8)
    assert [r["param_delta"] for r in requests] == [{"input_pca": 64}, {"max_iter_harmony": 20}]
    assert "narrowing comes after the broad search" in engine.prompts[1]
    assert engine.prompts[1].endswith("(this round has no focus: return focus_levers empty).")
    assert verifier.reasoner_failures == 0 and verifier.fallback_rounds == 0 and verifier.focus_history == []
    crashed = {**attempts, "c02": _call("input", DEFAULT, delta=None)}   # a crashed call changed nothing
    assert set(verifier._untried_groups(crashed)) == {"input", "solver"}


def test_threads_per_call_follow_the_input_size():
    sizes = {500: 1, 5_000: 1, 25_000: 1, 50_000: 2, 100_000: 4, 200_000: 8, 1_000_000: 8}
    assert {n: call_threads(np.zeros((n, 1))) for n in sizes} == sizes
    assert call_threads(None) == 8 and call_threads(np.zeros((200_000, 1)), cores=4) == 4
    assert [concurrency_cap(8, np.zeros((500, 2))), concurrency_cap(8, np.zeros((200_000, 1))),
            concurrency_cap(16, np.zeros((200_000, 1))), concurrency_cap(8, np.zeros((50_000, 1)))] == [8, 1, 2, 4]


@pytest.mark.parametrize("rows,cores,threads,cap", [(500, 8, 1, 8), (200_000, 16, 8, 2), (200_000, 8, 8, 1)])
def test_the_verifier_resolves_threads_and_cap_from_the_task_input(rows, cores, threads, cap):
    logged = []
    verifier = BatchCorrectionVerifier(log=lambda kind, **f: logged.append((kind, f.get("text"))))
    verifier.begin(BatchTask(batch=np.array(["x", "y"] * (rows // 2)), features=np.zeros((rows, 1)), cores=cores))
    assert (verifier.threads_per_call, verifier.cap) == (threads, cap)
    assert f"at most {cap} calls run at the same time, {threads} threads each" in verifier.resources_text()
    record = json.loads(dict(logged)["concurrency"])
    assert record == {"rows": rows, "cores": cores, "threads_per_call": threads, "cap": cap}


def test_a_small_input_runs_several_calls_in_one_round():
    import threading
    import time

    engine = Engine([[move(nclust=200), move(theta=2.5), move(sigma=0.07), move(nclust=400)]], final=["agent_1"])
    active, peak, lock = [0], [0], threading.Lock()
    tool = simulated_tool(more_clusters_help)

    def slow(params):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        return tool(params)

    verifier = BatchCorrectionVerifier()
    verifier.llm_engine, verifier.effective_defaults = engine, dict(EFFECTIVE)
    task = BatchTask(batch=np.array(["x", "y", "x", "y"]), features=INPUT, calibration=View(more_clusters_help, ()),
                     noise_band=0.002, brief="test", cores=8)
    events = []
    resolution = StepVerifier(max_iters=1 + BUDGET, observer=events.append).resolve_stage(
        verifier, adapt_batch_correction, slow, {}, input_result(more_clusters_help, True),
        upstream_ctx={"batch_task": task})
    assert (verifier.threads_per_call, verifier.cap) == (1, 8)
    assert [e["size"] for e in events if e["event"] == "wave"][0] == 4 and peak[0] > 1    # the wave runs at once
    assert "This round: at most 8 call(s)" in next(p for p in engine.prompts if p.startswith("You tune"))
    assert resolution.details["threads_per_call"] == 1 and resolution.details["cap"] == 8
