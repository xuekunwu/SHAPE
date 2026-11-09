"""Four-stage policy contracts and a real StepVerifier loop with deterministic synthetic observations."""

import pytest

from shape.agent.verification import base
from shape.agent.verification.four_stage import (
    FourStageSearch, SearchDirective, search_repair_schema, staged_schema, strategy_name,
)
from shape.agent.verification.test.test_base_vrr import (
    Admitting, Toy, good, mv, run_toy, toy_card,  # noqa: F401 -- shared pytest fixture
)
from shape.utils.schemas import VLRRResult, WavePlan


class StagedToy(Toy):
    def __init__(self, **kwargs):
        super().__init__(search_strategy="four_stage", **kwargs)


def directive(phase="cover", **kwargs):
    return {"phase": phase, "reason": "measured evidence within the available budget", **kwargs}


def plan(moves=(), phase="cover", focus=(), **kwargs):
    return {"analysis": "synthetic plan", "moves": list(moves), "focus_levers": list(focus),
            "search": directive(phase, **kwargs)}


class Engine:
    def __init__(self, plans=(), repairs=(), choice="agent_1"):
        self.plans, self.repairs = list(plans), list(repairs)
        self.choice, self.asked = choice, []
        self.last_stop = None

    def generate(self, content, response_format=None):
        self.asked.append((response_format.__name__, content[0]))
        if issubclass(response_format, WavePlan):
            if "Invalid search directive" in content[0] or "were rejected" in content[0]:
                payload = self.repairs.pop(0) if self.repairs else plan(phase="confirm")
            elif "You returned no moves" in content[0] and self.last_stop:
                payload = self.last_stop
            else:
                payload = self.plans.pop(0) if self.plans else plan(phase="confirm")
            if not payload["moves"]:
                self.last_stop = payload
            return {"content": response_format(**payload), "usage": {"total_tokens": 10}}
        return {"content": response_format(call=self.choice, reasoning="existing guard"), "usage": {"total_tokens": 10}}


class CoveringEngine:
    """Keep each tool fixture's native evidence while adding a cover/confirm directive."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def generate(self, content, response_format=None):
        staged = response_format is not None and "search" in response_format.model_fields
        schema = response_format.__bases__[0] if staged else response_format
        result = self.inner.generate(content, response_format=schema)
        if staged and result.get("content") is not None:
            payload = result["content"].model_dump()
            payload["search"] = directive("cover" if payload["moves"] else "confirm")
            result = {**result, "content": response_format(**payload)}
        return result


def test_strategy_precedence_validation_and_legacy_default(monkeypatch):
    monkeypatch.delenv("SHAPE_SEARCH_STRATEGY", raising=False)
    monkeypatch.delenv("SHAPE_SEARCH_STRATEGY_CROP", raising=False)
    assert strategy_name("crop") == "legacy"
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY", "four_stage")
    assert strategy_name("crop") == "four_stage"
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_CROP", "legacy")
    assert strategy_name("crop") == "legacy"
    assert strategy_name("crop", "four_stage") == "four_stage"
    with pytest.raises(ValueError, match="unknown search strategy"):
        strategy_name("crop", "typo")


@pytest.mark.parametrize("module,name", [("crop", "CropAgenticVerifier"),
                                        ("segmentation", "SegmentationAgenticVerifier"),
                                        ("batch_correction_unified", "BatchCorrectionVerifier"),
                                        ("representation", "RepresentationAgenticVerifier")])
def test_all_tool_schemas_keep_their_fields_and_accept_an_explicit_switch(module, name):
    from importlib import import_module

    cls = getattr(import_module(f"shape.agent.verification.verifiers.{module}"), name)
    old = cls(search_strategy="legacy")
    new = cls(search_strategy="four_stage")
    assert old._search is None and new._search is not None
    base = old.plan_schema
    if name == "RepresentationAgenticVerifier":     # legacy BasedPlan has no focus_levers (A12); four_stage narrows
        base = import_module(f"shape.agent.verification.verifiers.{module}").BasedFocusPlan
        assert set(base.model_fields) == set(old.plan_schema.model_fields) | {"focus_levers"}
    assert new.plan_schema is staged_schema(base)
    assert set(new.plan_schema.model_fields) == set(base.model_fields) | {"search"}
    assert "search" not in old.plan_schema.model_fields
    assert new.tuning_gate()["allowed"] is False


def test_four_stages_find_a_joint_improvement_and_confirm_without_extra_tool_calls(toy_card):
    plans = [
        plan([mv(alpha=3), mv(alpha=3, count=2), mv(alpha=2, count=2)]),
        plan([mv("agent_2", alpha=4)], "contrast", anchor="agent_2", evidence=["agent_1", "agent_2"]),
        plan([mv("agent_2", alpha=4, count=3)], "narrow", focus=["alpha", "count"], anchor="agent_2",
             evidence=["agent_2", "agent_3", "agent_4"],
             deferred=[{"lever": k, "reason": "fixed background in this budget"} for k in ("mode", "flag")],
             ranges=[{"lever": "alpha", "lower": 2, "upper": 4}, {"lever": "count", "lower": 1, "upper": 3}]),
        {**plan(phase="confirm", evidence=["agent_5"]), "deliver": "agent_5"},
    ]

    def quality(params):
        if params["alpha"] == 4 and params["count"] == 3:
            return 0.95
        return 0.8 if params["alpha"] == 3 and params["count"] == 2 else 0.5

    calls = []
    verifier, resolution, waves = run_toy(Engine(plans), cls=StagedToy, cores=8, quality=quality, calls=calls)
    assert waves == [3, 1, 1] and len(calls) == 5
    assert resolution.details["call"] == "agent_5" and resolution.verdict.scalar_score == 0.95
    record = resolution.details["search_strategy"]
    assert [x["phase"] for x in record["history"]] == ["cover", "contrast", "narrow", "confirm"]
    assert record["coverage"]["changed"] == ["alpha", "count"]
    assert record["coverage"]["deferred"] == {"mode": "fixed background in this budget", "flag": "fixed background in this budget"}
    assert {c["lever"] for c in record["controls"]} == {"alpha", "count"}
    assert record["confirmation"]["status"] == "existing_checks_passed"
    assert record["confirmation"]["independent_validation"] == "not_performed"
    assert record["stages_not_entered"] == []
    assert record["cost"]["calls"] == 5 and record["cost"]["reasoner_tokens"]["reported"] > 0
    assert verifier.fallback_rounds == 0


def observations(toy):
    params = toy.defaults()
    configs = [{**params, "alpha": 2}, {**params, "alpha": 3}, {**params, "alpha": 4, "count": 2}]
    attempts = {}
    for i, config in enumerate(configs, 1):
        result = VLRRResult("PASS", 0.5 + i * 0.01, {"quality": 0.5 + i * 0.01, "side": 0.6})
        result.param_delta = {k: v for k, v in config.items() if v != params[k]}
        attempts[f"agent_{i}"] = ("input", config, result)
    toy._labels = {"input": 0, "agent_1": 1, "agent_2": 2, "agent_3": 3}
    toy.rounds = [[1, 2, 3]]
    inp = VLRRResult("PASS", 0.5, {"quality": 0.5, "side": 0.6})
    toy._search.observe(toy, inp, attempts)
    return inp, attempts


def narrow_plan(toy, **kwargs):
    spec = plan([mv("agent_2", alpha=4)], "narrow", focus=["alpha"], anchor="agent_2",
                evidence=["agent_1", "agent_2"],
                deferred=[{"lever": k, "reason": "budget"} for k in ("mode", "flag")],
                ranges=[{"lever": "alpha", "lower": 2, "upper": 5}])
    spec["search"].update(kwargs)
    return toy.plan_schema(**spec)


def test_narrow_checks_full_background_even_when_switching_parent(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    request = toy.validate(mv("agent_3", alpha=5).model_dump(), attempts, set())
    with pytest.raises(ValueError, match="inherited background.*count"):
        toy._search.validate_request(toy, request, attempts)
    good_request = toy.validate(mv("agent_2", alpha=4.5).model_dump(), attempts, set())
    toy._search.validate_request(toy, good_request, attempts)
    outside = toy.validate(mv("agent_2", alpha=6).model_dump(), attempts, set())
    with pytest.raises(ValueError, match="within"):
        toy._search.validate_request(toy, outside, attempts)


@pytest.mark.parametrize("context_mismatch", [False, True])
def test_narrow_cannot_borrow_controls_from_another_background_or_context(toy_card, context_mismatch):
    toy = StagedToy()
    inp, attempts = observations(toy)
    anchor = "agent_3" if not context_mismatch else "agent_2"
    if context_mismatch:
        toy._search.records[anchor]["context"] = {"history": "a different checkpoint"}
    assert toy._search._control_axes == {"alpha"}
    with pytest.raises(ValueError, match="lacks local controls"):
        toy._search.prepare(toy, narrow_plan(toy, anchor=anchor), inp, attempts)
    toy._search.prepare(toy, narrow_plan(toy, anchor=anchor, contrast_waiver="budget permits combination search only"),
                        inp, attempts)


def test_exhausted_invalid_stage_is_a_reported_failure_not_a_successful_delivery(toy_card):
    invalid = plan([mv(alpha=3)], "narrow", focus=["alpha"], anchor="invented")
    engine = Engine([plan([mv(alpha=2)]), invalid], repairs=[invalid])
    _, resolution, waves = run_toy(engine, cls=StagedToy, budget=3)
    assert waves == [1]
    assert resolution.details["reasoner_failures"] == 1 and resolution.details["fallback_rounds"] == 0
    assert resolution.details["execution_errors"][0]["question"] == "search_repair"
    assert resolution.details["search_strategy"]["confirmation"]["status"] == "execution_error"


def test_cover_evidence_is_not_family_touch_and_negative_results_still_count(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    bad = VLRRResult("FAIL", 0.9, {"quality": 0.9, "side": 0.2}, param_delta={"flag": True})
    attempts["agent_4"] = ("agent_2", {**attempts["agent_2"][1], "flag": True}, bad)
    toy._search.observe(toy, inp, attempts)
    assert "flag" in toy._search.covered
    assert toy._search.records["agent_4"]["guard_failures"]
    with pytest.raises(ValueError, match="existing guard"):
        toy._search.prepare(toy, narrow_plan(toy, anchor="agent_4"), inp, attempts)
    with pytest.raises(ValueError, match="untested active.*mode"):
        toy._search.prepare(toy, narrow_plan(toy, deferred=[]), inp, attempts)


def test_seed_is_fixed_background_during_narrowing(toy_card):
    toy = StagedToy()
    toy.levers.append("seed")
    toy.lever_fields["seed"] = {"Family": "seed"}
    inp, attempts = observations(toy)
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    request = {"parent": "agent_2", "params": {**attempts["agent_2"][1], "alpha": 4, "seed": 17},
               "param_delta": {"alpha": 4, "seed": 17}}
    with pytest.raises(ValueError, match="inherited background.*seed"):
        toy._search.validate_request(toy, request, attempts)


def test_unmeasurable_calls_do_not_credit_coverage(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    result = VLRRResult("FAIL", float("nan"), {}, param_delta={"mode": "b"})
    attempts["agent_4"] = ("agent_2", {**attempts["agent_2"][1], "mode": "b"}, result)
    toy._search.observe(toy, inp, attempts)
    assert not toy._search.records["agent_4"]["measured"]
    assert "mode" not in toy._search.covered
    assert all("agent_4" not in pair["calls"] for pair in toy._search.controls)


def test_early_stop_does_not_claim_the_requested_stage_was_entered(toy_card):
    engine = Engine([plan([mv(alpha=3)]), plan(phase="narrow")])
    _, resolution, waves = run_toy(engine, cls=StagedToy)
    record = resolution.details["search_strategy"]
    assert waves == [1]
    assert record["stages_not_entered"] == ["contrast", "narrow"]
    assert record["history"][-1]["requested_phase"] == "narrow"
    assert record["history"][-1]["phase"] == "cover"


def test_contrast_checks_actual_config_not_declared_changes(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    p = toy.plan_schema(**plan([mv("agent_1", alpha=4)], "contrast", anchor="agent_1", evidence=["agent_1"]))
    toy._search.prepare(toy, p, inp, attempts)
    req = toy.validate(mv("agent_3", alpha=5).model_dump(), attempts, set())
    with pytest.raises(ValueError, match="one effective dimension"):
        toy._search.validate_request(toy, req, attempts)


def test_reopen_requires_new_evidence_is_bounded_and_does_not_reset_rounds(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    toy._search.commit(toy)
    with pytest.raises(ValueError, match="new finished evidence"):
        toy._search.prepare(toy, toy.plan_schema(**plan([mv(alpha=5)], reopen=True, evidence=["agent_1"])), inp, attempts)
    reopen = toy.plan_schema(**plan([mv(alpha=5)], reopen=True, evidence=["agent_3"]))
    toy._search.prepare(toy, reopen, inp, attempts)
    toy._search.commit(toy)
    assert toy._search.reopens == 1 and toy.rounds == [[1, 2, 3]]
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    toy._search.commit(toy)
    with pytest.raises(ValueError, match="at most once"):
        toy._search.prepare(toy, reopen, inp, attempts)


def test_local_numeric_ranges_cannot_be_omitted_or_widened(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    with pytest.raises(ValueError, match="finite local range"):
        toy._search.prepare(toy, narrow_plan(toy, ranges=[]), inp, attempts)
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    toy._search.commit(toy)
    with pytest.raises(ValueError, match="widened"):
        toy._search.prepare(toy, narrow_plan(toy, ranges=[{"lever": "alpha", "lower": 1, "upper": 5}]), inp, attempts)


def test_low_budget_reports_unfinished_stages_and_keeps_existing_selection(toy_card):
    engine = Engine([plan([mv(alpha=3)])])
    _, resolution, waves = run_toy(engine, cls=StagedToy, budget=1)
    assert waves == [1]
    record = resolution.details["search_strategy"]
    assert record["stages_not_entered"] == ["contrast", "narrow"]
    assert record["confirmation"]["call"] == "agent_1"


def test_no_reasoner_is_fatal_even_with_four_stage_enabled(toy_card):
    from shape.engine.errors import NoEngineError
    with pytest.raises(NoEngineError):
        run_toy(None, cls=StagedToy)


def test_invalid_stage_gets_one_repair_not_an_unchecked_move(toy_card):
    invalid = plan([mv(alpha=3)], "narrow", focus=["alpha"], anchor="invented")
    engine = Engine([invalid], repairs=[plan([mv(alpha=3)])])
    _, resolution, waves = run_toy(engine, cls=StagedToy, budget=1)
    assert waves == [1] and resolution.details["call"] == "agent_1"
    assert sum("Invalid search directive" in text for _, text in engine.asked) == 1


def test_cover_focus_is_repaired_with_the_broad_to_narrow_phase_focus_contract(toy_card):
    invalid = plan([mv(alpha=3)], "cover", focus=["alpha"])
    engine = Engine([invalid], repairs=[plan([mv(alpha=3)])])

    _, resolution, waves = run_toy(engine, cls=StagedToy, budget=1)

    assert waves == [1] and resolution.details["call"] == "agent_1"
    assert resolution.details["reasoner_failures"] == 0
    repairs = [(name, text) for name, text in engine.asked if name.startswith("SearchRepair")]
    assert len(repairs) == 1
    assert repairs[0][1].startswith("REPAIR REQUIRED.")
    assert "Follow the broad-to-narrow contract" in repairs[0][1] and "batch" not in repairs[0][1]
    schema = search_repair_schema(StagedToy().plan_schema)
    with pytest.raises(ValueError, match="cover requires empty focus_levers"):
        schema(**invalid)
    with pytest.raises(ValueError, match="narrow requires nonempty focus_levers"):
        schema(**plan([mv(alpha=3)], "narrow"))
    assert schema(**plan([mv(alpha=3)], "narrow", focus=["alpha"])).focus_levers == ["alpha"]


def test_coverage_progress_prevents_a_premature_plateau_but_not_a_budget_stop(toy_card):
    plans = [plan([mv(alpha=2)]), plan([mv(count=2)]), plan([mv(mode="b")]), plan([mv(alpha=3)])]
    toy, resolution, waves = run_toy(Engine(plans), cls=StagedToy, budget=4, quality=lambda _: 0.5)
    assert len(waves) == 4 and toy._stall(VLRRResult("PASS", 0.5, {}), {}) >= toy.patience
    assert resolution.details["search_strategy"]["coverage"]["changed"] == ["alpha", "count", "mode"]


def test_crop_conditional_dimensions_and_cell_rules_are_preserved():
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier

    verifier = CropAgenticVerifier(search_strategy="four_stage")
    verifier.input = {"cells": {"cell": {}}}
    params = {**verifier.defaults(), "margin_mode": "mask_bbox"}
    values = verifier.search_values(params, None)
    assert not {"margin", "margin_frac", "recenter", "crop_size"} & values.keys()
    assert "cell_rules" in values and "tight_buffer_px" in values
    params["cell_overrides"] = {"cell": {"margin_mode": "fraction"}}
    assert "margin_frac" in verifier.search_values(params, None)


def test_ssl_controls_need_equal_history_steps_and_effective_batch():
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    verifier = RepresentationAgenticVerifier(search_strategy="four_stage")
    params = {**verifier.defaults(), "trainable_pct": 50, "augmentation": "ssl116", "view_pairing": "same_crop"}
    values = verifier.search_values(params, None)
    assert "patch_embed_lr_mult" not in values and "augment_color" not in values
    assert not any(k.startswith("nn_") for k in values)
    verifier._labels = {"a": 1, "b": 2}
    assert verifier.search_context(params, "a") is None
    facts = {"steps_trained": 100, "start_step": 0, "best_step": 80, "batch_size": 32}
    verifier._search_training = {1: facts, 2: {**facts}}
    assert verifier.search_context(params, "a") == verifier.search_context(params, "b")
    verifier._search_training[2]["best_step"] = 90
    assert verifier.search_context(params, "a") != verifier.search_context(params, "b")
    verifier._search_training[2] = dict(facts)
    resumed = {**params, "init_checkpoint": "different_checkpoint", "init_from": "parent"}
    assert verifier.search_context(params, "a") != verifier.search_context(resumed, "b")
    assert verifier.search_values(params, "a")["batch_size"] == 32


def test_a_stop_is_legal_in_any_phase_and_stray_or_duplicate_deferrals_are_recorded(toy_card):
    stop = {**plan(phase="confirm", evidence=["agent_1"],
                   deferred=[{"lever": "per-call notes", "reason": "not a dimension"},
                             {"lever": "alpha", "reason": "first"}, {"lever": "alpha", "reason": "second"}]),
            "deliver": "agent_1"}
    engine = Engine([plan([mv(alpha=3)]), stop])
    verifier, resolution, waves = run_toy(engine, cls=StagedToy, budget=4)
    record = resolution.details["search_strategy"]
    assert waves == [1] and resolution.details["call"] == "agent_1"
    assert verifier.stop_reason == "satisfied" and resolution.details["reasoner_failures"] == 0
    assert not any("Invalid search directive" in text for _, text in engine.asked)
    assert record["history"][-1]["phase"] == "confirm"
    assert record["history"][-1]["ignored_deferred"] == [{"lever": "per-call notes", "reason": "not a dimension"}]
    assert record["coverage"]["deferred"] == {"alpha": "first"}


def test_a_stop_skips_transition_checks_but_checks_its_call_names(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    toy._search.prepare(toy, narrow_plan(toy), inp, attempts)
    toy._search.commit(toy)
    # a broader phase, a focus outside the lock and an anchor that fails no check: a stop is still legal
    toy._search.prepare(toy, toy.plan_schema(**plan(phase="cover", focus=["mode"], anchor="agent_1")), inp, attempts)
    assert toy._search.pending_focus == set()
    for bad, match in (({"evidence": ["invented"]}, "evidence"), ({"anchor": "invented"}, "anchor")):
        with pytest.raises(ValueError, match=match):
            toy._search.prepare(toy, toy.plan_schema(**plan(phase="confirm", **bad)), inp, attempts)
    with pytest.raises(ValueError, match="deliver must name a finished call"):
        toy._search.prepare(toy, toy.plan_schema(**{**plan(phase="confirm"), "deliver": "invented"}), inp, attempts)
    with pytest.raises(ValueError, match="return no moves"):
        toy._search.prepare(toy, toy.plan_schema(**plan([mv(alpha=3)], "confirm")), inp, attempts)


def test_a_repair_without_moves_keeps_the_accepted_wave(toy_card):
    toy = StagedToy()
    inp, attempts = observations(toy)
    toy._search.prepare(toy, toy.plan_schema(**plan([mv(alpha=5)])), inp, attempts)
    accepted = toy._search.pending
    toy._search.prepare(toy, toy.plan_schema(**plan(phase="confirm")), inp, attempts, keep_pending=True)
    assert toy._search.pending is accepted


# ---- the crop search, with plans the agent returned in a 1000-image CPJUMP1 run (reasons shortened) -------------
def _crop_search():
    """A four-stage crop verifier after the run's first calls: default (mask_bbox), agent_1 (auto_tighten on every
    cell) and agent_2 (auto_tighten on c1 only); each passes the guard."""
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier

    verifier = CropAgenticVerifier(search_strategy="four_stage")
    verifier.input = {"cells": {"c1": {}, "c2": {}}, "defaults": {"margin_mode": "mask_bbox"}}
    params = verifier.defaults()
    configs = [("input", {}, None), ("default", {"c1": {"auto_tighten": True}, "c2": {"auto_tighten": True}}, "auto"),
               ("agent_1", {"c1": {"auto_tighten": True}}, "selective")]
    attempts = {}
    for i, (parent, overrides, rule) in enumerate(configs):
        metrics = {"n_crops": 2, "frac_non_square": 0.0, "min_target_retention": 1.0, "n_missing_crops": 0}
        result = VLRRResult("PASS", 0.5 + 0.1 * i, metrics, param_delta={"cell_rules": [rule]} if rule else {})
        attempts["default" if i == 0 else f"agent_{i}"] = (parent, {**params, "cell_overrides": overrides}, result)
    verifier.rounds = [[1], [2], [3]]
    inp = VLRRResult("PASS", 0.5, {})
    verifier._search.observe(verifier, inp, attempts)
    assert verifier._search.covered == {"cell_rules"}
    return verifier, inp, attempts


def _rule(*settings, when=()):
    return {"when": [{"readout": r, "op": op, "value": v} for r, op, v in when],
            "apply": [{"key": k, "value": v} for k, v in settings]}


def _move(parent, *rules, **changes):
    return {"parent": parent, "diagnosis": "from the run log", "cell_rules": list(rules),
            "changes": [{"lever": k, "value": str(v)} for k, v in changes.items()]}


def _crop_plan(verifier, focus, search, moves=(), deliver=None):
    deferred = [{"lever": lever, "reason": "deferred in the run"} for lever in search.pop("deferred")]
    return verifier.plan_schema(analysis="from the run log", focus_levers=focus, moves=list(moves), deliver=deliver,
                                search={"reason": "from the run log", "deferred": deferred, **search})


def test_crop_run_stop_plans_rejected_for_deferral_bookkeeping_now_stop():
    verifier, inp, attempts = _crop_search()
    search = verifier._search
    # img_0014 search_repair: confirm, deliver agent_1, with the per-cell key min_containment deferred
    repair = _crop_plan(verifier, [], {"phase": "confirm", "evidence": ["default", "agent_1"], "deferred": [
        "margin_mode", "margin_frac", "margin", "recenter", "tight_buffer_px", "crop_size", "min_containment"]},
        deliver="agent_1")
    search.prepare(verifier, repair, inp, attempts)
    assert search.pending_deferred.keys() == {*verifier.levers, "cell_rules"} and search.pending_ignored == []
    # img_0088 stop_check: confirm, deliver agent_2, with a free-text per-cell entry
    check = _crop_plan(verifier, [], {"phase": "confirm", "evidence": ["default", "agent_1", "agent_2"], "deferred": [
        "margin_mode", "margin_frac", "margin", "recenter", "crop_size",
        "per-cell margin_mode/recenter/min_containment"]}, deliver="agent_2")
    search.prepare(verifier, check, inp, attempts)
    search.commit(verifier, stopping=True)
    assert search.phase == "confirm"
    assert [d["lever"] for d in search.history[-1]["ignored_deferred"]] == [
        "per-cell margin_mode/recenter/min_containment"]
    assert "cell_rules" not in search.deferred           # the free text defers no dimension


def test_crop_run_per_cell_names_count_as_cell_rules():
    verifier, inp, attempts = _crop_search()
    search = verifier._search
    aliases = verifier.search_aliases()
    assert aliases["auto_tighten"] == aliases["min_containment"] == aliases["per-cell margin_mode"] == "cell_rules"
    assert "margin_mode" not in aliases and "recenter" not in aliases       # these stay the image-level levers
    # img_0024 narrow: focus_levers ["auto_tighten"] on the measured cell rules, anchor agent_2
    narrow = _crop_plan(verifier, ["auto_tighten"], {"phase": "narrow", "evidence": ["agent_2"], "anchor": "agent_2",
                        "deferred": ["margin_mode", "tight_buffer_px", "recenter", "crop_size", "margin_frac", "margin"]},
                        moves=[_move("agent_2", _rule(("reset", True)), _rule(("auto_tighten", True), when=[
                            ("neighbor_contamination", ">=", "0.30")]))])
    search.prepare(verifier, narrow, inp, attempts)
    assert search.pending_focus == {"cell_rules"}
    anchor = attempts["agent_2"][1]
    retune = {"parent": "agent_2", "params": {**anchor, "cell_overrides": {"c1": {"auto_tighten": True},
                                                                           "c2": {"min_containment": 0.95}}},
              "param_delta": {"cell_rules": ["selective, stricter"]}}
    search.validate_request(verifier, retune, attempts)
    with pytest.raises(ValueError, match="outside narrow focus.*tight_buffer_px"):
        search.validate_request(verifier, {"parent": "agent_2", "params": {**anchor, "tight_buffer_px": 0.0},
                                           "param_delta": {"tight_buffer_px": 0.0}}, attempts)
    # img_0024 search_repair: contrast of tight_buffer_px from agent_2, min_containment deferred
    fresh, inp, attempts = _crop_search()
    contrast = _crop_plan(fresh, [], {"phase": "contrast", "evidence": ["agent_1", "agent_2"], "anchor": "agent_2",
                          "deferred": ["margin_mode", "margin_frac", "margin", "recenter", "crop_size", "min_containment"]},
                          moves=[_move("agent_2", tight_buffer_px=1.0)])
    fresh._search.prepare(fresh, contrast, inp, attempts)
    assert fresh._search.pending_deferred["cell_rules"] == "deferred in the run"
    fresh._search.validate_request(fresh, {"parent": "agent_2", "params": {**anchor, "tight_buffer_px": 1.0},
                                           "param_delta": {"tight_buffer_px": 1.0}}, attempts)


def test_crop_run_invalid_narrow_is_still_rejected():
    verifier, inp, attempts = _crop_search()
    # img_0088 narrow: focus on the image-level margin_mode/recenter, which no call has changed
    narrow = _crop_plan(verifier, ["margin_mode", "recenter"], {
        "phase": "narrow", "evidence": ["agent_2"], "anchor": "agent_2",
        "deferred": ["margin_frac", "margin", "crop_size", "min_containment"]},
        moves=[_move("agent_2", _rule(("margin_mode", "tight"), ("recenter", True), when=[
            ("centroid_offset", ">", "0.2"), ("neighbor_contamination", "<=", "0.2")]))])
    with pytest.raises(ValueError, match="narrow requires nonempty focus_levers with measured changes"):
        verifier._search.prepare(verifier, narrow, inp, attempts)
    # img_0300 cover with focus_levers ["auto_tighten"]: still a cover focus, rejected (and repaired by the loop)
    cover = _crop_plan(verifier, ["auto_tighten"], {"phase": "cover", "evidence": ["default"], "deferred": []},
                       moves=[_move("default", _rule(("auto_tighten", True)))])
    with pytest.raises(ValueError, match="leave focus_levers empty until narrow"):
        verifier._search.prepare(verifier, cover, inp, attempts)
    # a narrow on a cell rule whose anchor fails the guard
    attempts["agent_2"][2].metrics["n_missing_crops"] = 1
    focused = _crop_plan(verifier, ["min_containment"], {"phase": "narrow", "evidence": ["agent_2"],
                         "anchor": "agent_2", "deferred": ["margin_mode", "tight_buffer_px"]},
                         moves=[_move("agent_2", _rule(("min_containment", 0.95)))])
    with pytest.raises(ValueError, match="existing guard"):
        verifier._search.prepare(verifier, focused, inp, attempts)


def test_the_prompt_names_the_dimensions_and_the_crop_per_cell_keys():
    verifier, _, _ = _crop_search()
    text = verifier._search.prompt(verifier)
    assert "batch" not in text.lower()
    assert "focus_levers, deferred and ranges name these dimensions" in text
    assert "'auto_tighten': 'cell_rules'" in text and "they belong to\ncell_rules" in text
