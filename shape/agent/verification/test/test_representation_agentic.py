"""The agentic SSL VRR (`verifiers/representation.py` RepresentationAgenticVerifier, contract revision 2, section 4)
on a stub analyzer run through StepVerifier's wave mode.

The stub tool writes a small h5ad per call; `quant.sibling_site_map` is replaced by a stub whose value on the
comparison draws (quant.FRESH_SEED) and on the selection draws (quant.SELECTION_SEED) come from two different
embedding columns, so every verdict can be traced to the draws it read. The other quant functions are the real ones.
Co-located with the module under test (committed -- AGENTS.md sec 6 P5).
"""

import hashlib
import inspect
import json
import re
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from shape.agent.verification import base
from shape.agent.verification.stage_map import STAGES, invoke_managed_tool, map_tool_params
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers import representation as rep
from shape.agent.verification.verifiers.representation import (BasedChange, BasedMove, RepresentationAgenticVerifier,
                                                               RepresentationTask, frozen_params)
from shape.tools.cell_state_analyzer import quant

TOOL = "Cell_State_Analyzer_Tool"

# the 16 `## Levers` field lines of contract section 3.5.1 as amended by A11 (15 levers in 4 Families, plus
# max_epochs, which the benchmark fixes; their prose is the tool part's)
LEVER_LINES = """\
view_pairing | Family: objective | Type: choice | Range: same_crop, same_site, same_well | Default: same_crop
contrastive_mask | Family: objective | Type: choice | Range: off, same_site, same_well | Default: off
batch_sampling | Family: objective | Type: choice | Range: random, per_plate, per_experiment | Default: random
trainable_pct | Family: depth | Type: integer or null | Range: [0, 100] | Default: null
init_from | Family: depth | Type: choice | Range: pretrained, parent | Default: pretrained
learning_rate | Family: optimizer | Type: number | Range: [1e-6, 1e-4] | Default: 3e-5
max_epochs | Family: optimizer | Type: integer or null | Range: [1, 60] | Default: null
head_learning_rate | Family: optimizer | Type: number or null | Range: [1e-6, 1e-2] | Default: null
weight_decay | Family: optimizer | Type: number | Range: [0, 0.5] | Default: 1e-4
clip_grad_norm | Family: optimizer | Type: number | Range: [0, 100] | Default: 0
warmup_fraction | Family: optimizer | Type: number or null | Range: [0, 0.5] | Default: null
koleo_weight | Family: objective | Type: number | Range: [0, 1] | Default: 0
intensity_norm | Family: input | Type: choice | Range: off, per_plate | Default: off
augmentation | Family: input | Type: choice | Range: strong, light | Default: strong
crop_scale_min | Family: input | Type: number | Range: [0.05, 1.0] | Default: 0.3
drop_path | Family: depth | Type: number | Range: [0, 0.5] | Default: 0
"""
LEVERS = [line.split(" | ", 1)[0] for line in LEVER_LINES.splitlines()]
STUB_CARD = ("# Cell_State_Analyzer_Tool\n## Levers\n"
             + "".join(f"### {name}\n{fields}\n" for name, fields in
                       (line.split(" | ", 1) for line in LEVER_LINES.splitlines()))
             + "## Agentic VRR (RepresentationAgenticVerifier, StepVerifier wave mode)\nFor readers.\n")
NULLABLE = {"trainable_pct", "max_epochs", "head_learning_rate", "warmup_fraction"}
# A11: the former levers, no longer in the CARD; they stay tool kwargs (at the tool's defaults) and param_map entries
FORMER_LEVERS = {"patch_embed_lr_mult", "head_warmup_epochs", "gram_weight", "plate_mean_weight",
                 "plate_adversarial_weight", "batch_size", "augment_color", "nn_k", "nn_space", "nn_mutual",
                 "nn_refresh_epochs"}
# the representation param_map of contract 4.3: 26 lever kwargs (A11: 15 levers and 11 former levers), the 2 kept
# depth kwargs, 7 non-lever kwargs and amendment A1's crop_array
PARAM_MAP_KEYS = ({lever for lever in LEVERS if lever != "init_from"} | FORMER_LEVERS
                  | {"field_aware", "adaptation_depth"}
                  | {"init_checkpoint", "output_subdir", "split_key", "val_fraction", "max_minutes",
                     "eval_every_epochs", "embed_only", "crop_array"})
# digit sequences from the task-116 RESULT.md files (results/rxrx1/*traj*/RESULT.md): none may reach a prompt
TASK116_NUMBERS = ("0.1394", "0.1409", "0.1405", "0.1403", "0.1644", "0.0949", "0.0644", "0.2182", "0.2045",
                   "0.2104", "0.2038", "0.0119", "0.0198", "0.3189", "0.2468", "0.2563", "0.2585", "0.0656",
                   "0.0237", "0.2834", "0.0293", "0.0183", "0.1360", "0.1374", "0.1393", "0.3324", "0.0819",
                   "0.0281", "0.5687", "0.5335", "0.5696", "0.024", "0.039", "0.139", "0.0012")
# stub outputs that are logged only: none may reach a prompt
LOGGED_MARKS = ("eval_loss", "mean_cosine", "uniformity", "val_history", "grad_norm_median", "clipped_fraction",
                "train_seconds", "final_loss", "loss_history", "resume_best", "resume_end", "selection_check",
                "embedding_finite", "embedding_varies", "_selection_draws", "4.321", "2.468", "1.357")


@pytest.fixture()
def card(monkeypatch):
    """The stub CARD for the CARD readers; no reasoner retry wait; no SHAPE_REQUIRE_AGENTIC."""
    monkeypatch.setattr("shape.tools.base.load_tuning_card", lambda tool_dir: STUB_CARD)
    monkeypatch.setattr(base.BaseVerifier, "tuning_card", lambda self: STUB_CARD)
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)


@pytest.fixture()
def proxy(monkeypatch):
    """quant.sibling_site_map as a stub: value = the mean of embedding column 0 on the comparison draws, of column 1
    on the selection draws; one unit per val site, jittered per (unit, value, seed); records every call."""
    calls = []

    def stub(vectors, meta, *, k=2, draws=20, seed=0, scope="plate"):
        x = np.asarray(vectors)
        value = float(x[:, 0 if seed == quant.FRESH_SEED else 1].mean())
        calls.append({"seed": seed, "k": k, "draws": draws, "rows": [int(r) for r in x[:, 7]],
                      "cells": [m["cell_id"] for m in meta]})
        units = {}
        for ids in quant.identities(meta):
            unit = f"{ids['well']}|{ids['site']}"
            r = int(hashlib.sha256(f"{unit}|{value:.6f}|{seed}".encode()).hexdigest()[:8], 16) / 16 ** 8
            units[unit] = value + 0.01 * (2 * r - 1)
        return {"value": value, "chance": 0.2, "units": units, "n_queries": len(units), "k": k, "draws": draws}

    monkeypatch.setattr(quant, "sibling_site_map", stub)
    return calls


def crops(n_plates=2, wells=("A01", "A02", "A03", "A04", "A05", "A06"), val=("A01", "A02"), sites=2, cells=3):
    """Crop metadata: plates x wells x sites x cells, plate / well / site / experiment, cell_id == crop_id and the
    benchmark's split key (the `val` wells of every plate are val)."""
    meta = []
    for p in range(n_plates):
        for w in wells:
            for s in range(sites):
                for c in range(cells):
                    i = len(meta)
                    meta.append({"cell_id": f"x{i:03d}", "crop_id": f"x{i:03d}", "plate": f"P{p}", "well": w,
                                 "site": f"{w}_s{s}", "experiment": "E0", "split": "val" if w in val else "train"})
    return meta


def frozen_quality(params):
    return 0.5, 0.5


def gain(params):
    return 0.6, 0.6


class StubTool:
    """The analyzer as the loop sees it: one h5ad per call under <cache>/cell_state_analysis/<output_subdir>/ (column
    0 = the comparison-draw value, column 1 = the selection-draw value, column 7 = the crop index; rows in reverse
    crop order with `reverse`), a resume state, the training facts and logged-only outputs. `quality(params)` ->
    (fresh value, selection value); trial 0 (zero_shot) is (0.5, 0.5); `fail(params)` raises."""

    def __init__(self, root, meta, quality=None, fail=None, reverse=False, seconds=300.0, history_offset=0.0001):
        self.root, self.meta, self.quality, self.fail, self.reverse = Path(root), meta, quality, fail, reverse
        self.seconds, self.history_offset, self.calls = seconds, history_offset, []

    def __call__(self, params):
        self.calls.append(dict(params))
        if self.fail and self.fail(params):
            raise RuntimeError("the stub tool failed")
        frozen = params.get("adaptation_depth") == "zero_shot"
        fresh, selection = (frozen_quality if frozen or self.quality is None else self.quality)(params)
        out = self.root / "cache" / "cell_state_analysis" / params["output_subdir"]
        out.mkdir(parents=True, exist_ok=True)
        n = len(self.meta)
        x = np.random.default_rng(len(self.calls)).normal(size=(n, 8))
        x[:, 0], x[:, 1], x[:, 7] = fresh, selection, np.arange(n)
        order = np.arange(n)[::-1] if self.reverse else np.arange(n)
        obs = pd.DataFrame({"cell_id": [self.meta[i]["cell_id"] for i in order]},
                           index=[self.meta[i]["cell_id"] for i in order])
        path = out / "cell_state_analyzed.h5ad"
        ad.AnnData(X=x[order], obs=obs).write_h5ad(path)
        resume = out / "resume_best.pt"
        resume.write_bytes(b"state")
        return {"adata_path": str(path), "seconds": 60.0 if frozen else self.seconds,
                "training": {"epochs_trained": 0 if frozen else 3, "best_epoch": None if frozen else 2,
                             "best_by": None if frozen else "val_proxy", "batch_size": 64, "trainable_blocks": 4,
                             "train_seconds": 0.0 if frozen else 120.0, "grad_norm_median": 4.321,
                             "clipped_fraction": 0.75},
                # the real tool: an untrained call with a split reports its val proxy as epoch 0 (no eval loss)
                "val_history": ([{"epoch": 0, "sibling_site_map": selection}]
                                if params.get("split_key") or params.get("val_fraction") is not None else [])
                if frozen else [{"epoch": 2, "sibling_site_map": selection + self.history_offset,
                                 "eval_loss": 2.468}],
                "resume_best": None if frozen else str(resume), "resume_end": None if frozen else str(resume),
                "split": {"n_train": 48, "n_val": 24, "n_val_wells": 4}, "final_loss": 1.357,
                "loss_history": [1.357]}


class Engine:
    """Answers each round question with the next plan (then none: satisfied), a repair question with `repair`, the
    final question with the next of `final` (None when spent); records (schema name, prompt) of every question."""

    def __init__(self, rounds=(), final=(), repair=()):
        self.rounds, self.final, self.repair, self.asked = list(rounds), list(final), list(repair), []

    def generate(self, content, response_format=None):
        self.asked.append((response_format.__name__, content[0]))
        if response_format.__name__ in ("WavePlan", "BasedPlan", "BasedFocusPlan"):   # BasedFocusPlan: four_stage
            if "were rejected" in content[0]:
                return {"content": response_format(analysis="repair", moves=self.repair)}
            return {"content": response_format(analysis="a", moves=self.rounds.pop(0) if self.rounds else [])}
        return {"content": response_format(call=self.final.pop(0) if self.final else None, reasoning="r")}

    def questions(self, schema="BasedPlan"):
        return [text for name, text in self.asked if name == schema]

    def prompts(self):
        return [text for _, text in self.asked]


class Down:
    """A reasoner that never answers (a gateway down)."""

    def generate(self, content, response_format=None):
        raise TimeoutError("gateway down")


def test_solve_path_starts_with_frozen_trial_and_skips_inapplicable_training(card, proxy, tmp_path, monkeypatch):
    from shape.agent.verification import integration
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    meta = crops(wells=("A01",))  # no held-out wells: training is inapplicable
    arrays = [np.zeros((8, 8), dtype=np.uint8) for _ in meta]
    monkeypatch.setattr(Cell_State_Analyzer_Tool, "_load_cell_data_from_metadata",
                        lambda self, path: (arrays, meta, {}))
    spec = next(s for s in STAGES if s.stage == "representation")
    monkeypatch.setattr(integration, "stages_for_tool", lambda name: [spec])
    tool = StubTool(tmp_path, meta)

    class Executor:
        query_cache_dir = str(tmp_path)
        enable_llm_verifier = False

        def run_managed_tool(self, name, kwargs, params):
            assert name == TOOL and kwargs["cell_crops"] is arrays and kwargs["cell_metadata"] is meta
            return tool(params)

    resolutions, result, status = integration.verify_managed_tool(
        Executor(), TOOL, {}, initial_params={"adaptation_depth": "full", "max_epochs": 60},
        request="inspect cell states")
    assert len(tool.calls) == 1 and tool.calls[0]["adaptation_depth"] == "zero_shot"
    assert result["training"]["epochs_trained"] == 0
    assert status == "satisfied" and result["adata_path"]
    assert resolutions[0].details["tuning_gate"]["decision"] == "SKIP"


def mv(parent="input", **changes):
    return BasedMove(parent=parent, changes=[BasedChange(lever=k, value=json.dumps(v), basis=f"test basis for {k}")
                                            for k, v in changes.items()])


def run(tmp_path, engine, *, quality=None, meta=None, budget=8, deadline=None, tool=None, log=None, expert=None):
    meta = meta if meta is not None else crops()
    tool = tool or StubTool(tmp_path, meta, quality)
    verifier = RepresentationAgenticVerifier(log=log, expert_config=expert)
    verifier.llm_engine = engine
    task = RepresentationTask(cores=8, gpus=2, deadline=deadline, brief="tune the SSL fine-tune of these crops")
    events = []
    initial = frozen_params(task)
    resolution = StepVerifier(max_iters=1 + budget, observer=events.append).resolve_stage(
        verifier, verifier.make_adapter({"cell_metadata": meta}), tool, initial, tool(initial),
        upstream_ctx={"representation_task": task})
    return verifier, resolution, tool, [e["size"] for e in events if e["event"] == "wave"]


def row(prompt, call):
    """One row of the calls table of a prompt, parsed."""
    return json.loads(next(line for line in prompt.splitlines() if line.startswith(f'{{"call": "{call}"')))


def same_well(params):
    """+0.1 for same_well pairing, +0.001 per trainable_pct point, +0.05 for a resumed call; the selection draws
    read the opposite."""
    q = 0.5 + 0.1 * (params.get("view_pairing") == "same_well") + 0.001 * (params.get("trainable_pct") or 0)
    q += 0.05 * bool(params.get("init_checkpoint"))
    return q, 1.0 - q


# ---- the CARD ------------------------------------------------------------------------------------------------

def test_the_real_card_levers_parse_as_the_contract_states():
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    verifier = RepresentationAgenticVerifier()
    assert verifier.levers == LEVERS                                     # the 16 levers, in the contract's order
    for lever, fields in (line.split(" | ", 1) for line in LEVER_LINES.splitlines()):
        expected = dict(p.split(": ", 1) for p in fields.split(" | "))
        assert verifier.lever_fields[lever] == expected, lever           # Family, Type, Range, Default
        assert base.lever_spec(verifier.lever_fields[lever]) == base.lever_spec(expected), lever
    assert not FORMER_LEVERS & set(verifier.lever_fields)                # A11: no former lever in the CARD
    families = {}
    for lever, fields in verifier.lever_fields.items():
        families.setdefault(fields["Family"], []).append(lever)
    assert sorted(families) == ["depth", "input", "objective", "optimizer"]
    assert not verifier.narrowing and verifier.schedule_text() == ""         # A12: no broad-then-narrow rule
    assert verifier._untried_groups({}) == families                            # the Families still label levers
    assert verifier.lever_spec("contrastive_mask") == ("choice", ("off", "same_site", "same_well"), False)
    assert verifier.lever_spec("intensity_norm") == ("choice", ("off", "per_plate"), False)
    assert verifier.lever_spec("augmentation") == ("choice", ("strong", "light"), False)
    assert {lever for lever in LEVERS if verifier.lever_spec(lever)[2]} == NULLABLE
    signature = inspect.signature(Cell_State_Analyzer_Tool.execute).parameters
    defaults = verifier.defaults()
    for lever in LEVERS:
        if lever != "init_from":                                         # a verifier lever, no tool kwarg
            assert defaults[lever] == signature[lever].default, lever
    assert defaults["init_from"] == "pretrained" and "init_from" not in signature


def test_the_round_prompt_states_the_goal_and_the_basis_rule_and_no_narrowing(card, proxy, tmp_path):
    """A12: no broad-then-narrow rule and no forced coverage -- a first focus is never rejected; every round prompt
    states the goal (the largest val proxy gain over trial 0 the budget can reach, levers the means) and that every
    lever change needs a basis."""
    engine = Engine([[mv(view_pairing="same_well", trainable_pct=33)]], final=["agent_1"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    rounds = engine.questions()
    assert rounds and verifier.schedule_text() == "" and verifier.schedule_now_text() == ""
    for q in rounds:
        assert verifier.goal_text() in q and verifier.basis_text() in q
        assert "broad, then narrow" not in q and "a first focus" not in q and "Now: exploring" not in q
    assert "Goal: the largest improvement of the val sibling_site_map" in rounds[0]
    assert "Changing levers is the means, not the goal" in rounds[0]
    verifier._pending_focus = ["trainable_pct"]          # a focus is the agent's own; nothing checks it
    accepted, rejected = [], []
    verifier._accept([mv("agent_1", trainable_pct=50)], verifier._attempts_seen, accepted, 1, rejected)
    assert len(accepted) == 1 and rejected == [] and verifier.focus_history == []


def test_a_lever_change_without_a_basis_is_rejected_and_asked_again(card, proxy, tmp_path):
    """A12: a lever change with an empty basis is rejected through the shared repair question, with the reason; the
    repaired move (with a basis) runs. The plan schema's change is the shared one plus basis."""
    bare = BasedMove(parent="input", changes=[BasedChange(lever="trainable_pct", value="33", basis="  ")])
    engine = Engine([[bare]], final=["agent_1"], repair=[mv(trainable_pct=33)])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    repair = next(q for q in engine.questions() if "were rejected" in q)
    assert "no basis for ['trainable_pct']" in repair and "testable hypothesis" in repair
    assert waves[0] == 1 and tool.calls[1]["trainable_pct"] == 33       # then forced continuations (A12)
    assert verifier.plans[1]["question"] == "repair"
    assert verifier.plans[1]["moves"][0]["changes"][0]["basis"] == "test basis for trainable_pct"
    assert set(rep.BasedPlan.model_json_schema()["$defs"]["BasedChange"]["properties"]) == {"lever", "value", "basis"}
    assert set(rep.BasedPlan.model_fields) == {"analysis", "hypotheses", "moves", "deliver"}    # no focus_levers
    assert resolution.details["trials"]["agent_1"]["param_basis"] == {"trainable_pct": "test basis for trainable_pct"}
    assert verifier._basis_problem({"changes": [{"lever": "x", "value": "1", "basis": "agent_1's proxy rose"}]}) is None
    assert verifier._basis_problem({"parent": "agent_1", "changes": []}) is None      # a pure continuation


# ---- A18/A19: the expert start, fixed expert levers per round in rounds 1-2 ------------------------------------

EXPERT = [{"view_pairing": "same_well", "contrastive_mask": "same_well", "batch_sampling": "per_plate"},
          {"intensity_norm": "per_plate", "augmentation": "light"}]
FIRST, SECOND = EXPERT
BOTH = {**FIRST, **SECOND}


def test_the_expert_start_is_off_by_default_and_its_two_rounds_run(card, proxy, tmp_path):
    """A19: off by default (no prompt block, no check); with `expert_config` the round-1 and round-2 prompts state
    it in one block, a round-1 move setting the round-1 levers and a round-2 move setting the round-2 levers from
    the round-1 call both run, and round 3 is A13's loop (no block, any lever)."""
    off = RepresentationAgenticVerifier()
    assert off.expert_config == [] and off.admission_text() == "" and not off._expert_round()
    engine = Engine([[mv(**FIRST)], [mv("agent_1", **SECOND)], [mv("agent_2", learning_rate=1e-5)]],
                    final=["agent_3"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=3, expert=EXPERT)
    rounds = engine.questions()
    block = ("Expert start: expert experience for this tool sets, in round 1, view_pairing=same_well, "
             "contrastive_mask=same_well, batch_sampling=per_plate; in round 2, building on a round-1 call, "
             "intensity_norm=per_plate, augmentation=light")
    assert block in rounds[0] and "This is round 1." in rounds[0]
    assert block in rounds[1] and "This is round 2." in rounds[1]
    assert "Expert start" not in rounds[2]
    assert verifier.goal_text() in rounds[0] and verifier.basis_text() in rounds[0]       # A13's text stays
    assert waves == [1, 1, 1] and verifier.expert_rejections == []
    assert {k: tool.calls[2][k] for k in BOTH} == BOTH and tool.calls[3]["learning_rate"] == 1e-5
    assert resolution.details["expert_config"] == EXPERT and resolution.details["expert_rejections"] == []


def test_a_round_1_move_without_per_plate_batches_is_rejected_and_the_repair_runs(card, proxy, tmp_path):
    """A19: a round-1 configuration has every round-1 lever at its expert value (the A18 run paired same_well
    without per_plate batches); the move is rejected through the repair question (with the reason, recorded), and
    the repaired move runs; a round-2 lever in round 1 is rejected too."""
    pairing = {"view_pairing": "same_well", "contrastive_mask": "same_well"}
    engine = Engine([[mv(**pairing), mv(**BOTH)]], final=["agent_1"], repair=[mv(**FIRST)])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=1, expert=EXPERT)
    assert any("were rejected" in q and "batch_sampling" in q for q in engine.questions())  # the repair question
    reasons = [r["reason"] for r in verifier.expert_rejections]
    assert "leaves ['batch_sampling'] off the expert values of round 1" in reasons[0]
    assert ("it changes ['intensity_norm', 'augmentation']: the expert start's rounds change no other lever"
            in reasons[1])
    assert waves == [1] and {k: tool.calls[1].get(k) for k in FIRST} == FIRST
    assert tool.calls[1]["intensity_norm"] == "off"
    assert [r["round"] for r in verifier.expert_rejections] == [1, 1]


def test_an_extra_lever_or_another_value_in_rounds_1_2_is_rejected(card, proxy, tmp_path):
    """A19: rounds 1-2 change no lever outside the expert start (init_from aside) and set the expert values only,
    a round-1 lever included in round 2."""
    engine = Engine([[mv(learning_rate=1e-5, **FIRST)], [mv("agent_1", view_pairing="same_site", **SECOND)]],
                    final=["agent_1"], repair=[mv(**FIRST)])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=2, expert=EXPERT)
    reasons = [r["reason"] for r in verifier.expert_rejections]
    assert ("it changes ['learning_rate']: the expert start's rounds change no other lever (init_from aside)"
            in reasons[0])
    assert 'it sets {"view_pairing": "same_site"} instead of {"view_pairing": "same_well"}' in reasons[1]
    assert all(call.get("learning_rate") != 1e-5 for call in tool.calls)
    assert all(call.get("view_pairing") != "same_site" for call in tool.calls)


def test_after_a_failed_round_1_round_2_sets_both_rounds(card, proxy, tmp_path):
    """A19: when no round-1 call finished, a round-2 move sets both rounds' levers (from input); a move with the
    round-2 levers only is rejected through the repair question."""
    def fail(params):                                     # the round-1 call (its levers only) crashes
        return params.get("contrastive_mask") == "same_well" and params.get("intensity_norm") == "off"

    tool = StubTool(tmp_path, crops(), same_well, fail=fail)
    engine = Engine([[mv(**FIRST)], [mv(**SECOND)]], final=["agent_3"], repair=[mv(**BOTH)])
    verifier, resolution, tool, waves = run(tmp_path, engine, budget=2, tool=tool, expert=EXPERT)
    assert waves == [1, 1] and [r["round"] for r in verifier.expert_rejections] == [2]
    assert ("leaves ['view_pairing', 'contrastive_mask', 'batch_sampling'] off the expert values of round 2"
            in verifier.expert_rejections[0]["reason"])
    assert {k: tool.calls[2][k] for k in BOTH} == BOTH
    assert "if no round-1 call finished, a round-2 move sets both rounds' values" in engine.questions()[1]


def test_a_one_round_expert_start_sets_its_levers_in_round_1_and_round_2_is_free(card, proxy, tmp_path):
    """A21: a one-entry `expert_config` is a one-round start: round 1 sets all its levers (a move without one of
    them is rejected through the repair question), the round-1 prompt states it, and round 2 is the agent's
    (no block, any lever)."""
    start = [BOTH]
    engine = Engine([[mv(**FIRST)], [mv("agent_1", learning_rate=1e-5)]], final=["agent_2"], repair=[mv(**BOTH)])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=2, expert=start)
    rounds = engine.questions()
    assert ("Expert start: expert experience for this tool sets, in round 1, view_pairing=same_well, "
            "contrastive_mask=same_well, batch_sampling=per_plate, intensity_norm=per_plate, augmentation=light. "
            "Round 1 changes no other lever (init_from aside)." in rounds[0])
    assert "Expert start" not in rounds[-1] and "This is round 2." not in rounds[-1]
    assert waves == [1, 1] and [r["round"] for r in verifier.expert_rejections] == [1]
    assert "leaves ['intensity_norm', 'augmentation'] off the expert values of round 1" in (
        verifier.expert_rejections[0]["reason"])
    assert {k: tool.calls[1][k] for k in BOTH} == BOTH and tool.calls[2]["learning_rate"] == 1e-5


PROGRESSIVE = [{**BOTH, "trainable_pct": 33}, {"init_from": "parent", "trainable_pct": 66}]


def test_a_round_entry_with_init_from_parent_resumes_a_previous_round_call(card, proxy, tmp_path):
    """A round entry may set init_from "parent": its move resumes a finished call of the previous round. Round 2 from
    the round-1 call with trainable_pct 66 runs as a continuation; a move from input with every expert value, or from
    the round-1 call with init_from pretrained, is rejected through the repair question; round 3 is the agent's."""
    first = {**BOTH, "trainable_pct": 33}
    fresh = BasedMove(parent="agent_1", changes=[
        BasedChange(lever="init_from", value='"pretrained"', basis="b"),
        BasedChange(lever="trainable_pct", value="66", basis="b")])
    engine = Engine([[mv(**first)], [mv(**{**first, "trainable_pct": 66}), fresh],
                     [mv("agent_2", learning_rate=1e-5)]],
                    final=["agent_3"], repair=[mv("agent_1", trainable_pct=66)])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=3, expert=PROGRESSIVE)
    rounds = engine.questions()
    assert ("in round 2, resuming a round-1 call (init_from parent), trainable_pct=66 (if no round-1 call finished, "
            "a round-2 move sets both rounds' values)") in rounds[0]
    assert "init_from=parent" not in rounds[0]
    assert waves == [1, 1, 1] and [r["round"] for r in verifier.expert_rejections] == [2, 2]
    for rejection in verifier.expert_rejections:
        assert "it does not resume a round-1 call: round 2 resumes one of ['agent_1'] (init_from parent)" in (
            rejection["reason"])
    assert tool.calls[1]["trainable_pct"] == 33 and tool.calls[1]["init_checkpoint"] is None
    assert {k: tool.calls[2][k] for k in BOTH} == BOTH and tool.calls[2]["trainable_pct"] == 66
    assert tool.calls[2]["init_checkpoint"]                           # resumes agent_1's training state
    assert resolution.details["trials"]["agent_2"]["parent"] == "agent_1"
    assert tool.calls[3]["learning_rate"] == 1e-5
    assert resolution.details["expert_config"] == PROGRESSIVE


def test_an_init_from_parent_round_after_a_failed_round_sets_every_expert_value(card, proxy, tmp_path):
    """When no call of the previous round finished, a round whose entry sets init_from "parent" takes a move from
    input with every expert value of rounds 1..k (the round-2 value of trainable_pct)."""
    tool = StubTool(tmp_path, crops(), same_well, fail=lambda params: params.get("trainable_pct") == 33)
    engine = Engine([[mv(**BOTH, trainable_pct=33)], [mv(**BOTH, trainable_pct=66)]], final=["agent_2"])
    verifier, resolution, tool, waves = run(tmp_path, engine, budget=2, tool=tool, expert=PROGRESSIVE)
    assert waves == [1, 1] and verifier.expert_rejections == []
    assert tool.calls[2]["trainable_pct"] == 66 and tool.calls[2]["init_checkpoint"] is None


def test_an_expert_entry_sets_init_from_parent_only_and_not_in_round_1():
    for bad in ([{"init_from": "parent", "trainable_pct": 33}], [{"trainable_pct": 33}, {"init_from": "pretrained"}]):
        with pytest.raises(ValueError, match='init_from "parent"'):
            RepresentationAgenticVerifier(expert_config=bad)
    assert RepresentationAgenticVerifier(expert_config=PROGRESSIVE).expert_config == PROGRESSIVE


def test_the_prompts_list_no_untried_levers_and_no_reminder_to_review_them(card, proxy, tmp_path):
    """A21: neither the round prompt nor the stop check lists the levers no call has changed or reminds the agent to
    review them; the stop check still states the rounds left and the best admissible call."""
    engine = Engine([[mv(view_pairing="same_well")], []], final=[None])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=flat, budget=3)
    questions = engine.questions()
    assert all("no call has changed" not in q and "untried" not in q for q in questions)
    stop = next(q for q in questions if "You returned no moves" in q)
    assert "rounds remain" in stop and "The best admissible call so far: none admissible yet" in stop


def test_a_stop_is_refused_while_the_last_call_improved_and_an_empty_answer_forces_a_continuation(card, proxy,
                                                                                                tmp_path):
    """A12: an empty plan while the last finished call still improved on its parent (paired delta > 0) is answered
    with that delta +/- 2 SE and a request for moves (logged, recorded); the agent's moves then run. An answer with
    no moves again does not stop: the verifier runs one continuation of that call (no lever change, init_from
    parent); the search stops at an empty plan once the last call did not improve."""
    logged = []
    engine = Engine([[mv(view_pairing="same_well")], [], [mv("agent_1", trainable_pct=50)]], final=["agent_2"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well,
                                            log=lambda kind, **f: logged.append((kind, f.get("text", ""))))
    assert waves == [1, 1, 1] and verifier.stop_reason == "satisfied" and verifier.stop_checks == 3
    refusals = [q for q in engine.questions() if "the stop is not accepted" in q]
    assert len(refusals) == 2
    assert ("the last finished call, agent_1, still improved on its parent (input): val sibling_site_map +"
            in refusals[0])
    assert "the last finished call, agent_2, still improved on its parent (agent_1)" in refusals[1]
    assert re.search(r"sibling_site_map \+0\.\d{4} \+/- \d\.\d{4}, (beyond|within) 2 SE, n \d+", refusals[0])
    assert "return moves for this round" in refusals[0] and "the system runs a continuation of agent_1" in refusals[0]
    assert "You returned no moves: the search ends here" not in refusals[0]
    assert [(r["call"], r["parent"]) for r in verifier.stop_refusals] == [("agent_1", "input"), ("agent_2", "agent_1")]
    assert all(r["delta"] > 0 for r in verifier.stop_refusals)
    # agent_2's refusal met no moves: continuation_1 resumed agent_2; it equals agent_2, so the next stop held
    [forced] = verifier.forced_continuations
    assert forced["parent"] == "agent_2"
    assert forced["basis"].startswith("forced continuation: the last call improved by +")
    assert forced["basis"].endswith("over its parent (user rule A12)")
    assert resolution.parents[-1] == 2 and resolution.trials[-1][1].param_delta == {"init_from": "parent"}
    assert tool.calls[-1]["init_checkpoint"] == resolution.trials[2][2]["call_resume_best"]    # agent_2, delivered
    assert resolution.trials[-1][1].notes == forced["basis"]
    trials = resolution.details["trials"]
    assert trials["continuation_1"]["param_basis"] == {"init_from": forced["basis"]}
    assert trials["agent_2"]["param_basis"] == {"trainable_pct": "test basis for trainable_pct"}
    assert "You returned no moves: the search ends here" in engine.questions()[-1]
    kinds = [kind for kind, _ in logged]
    assert kinds.count("stop_refused") == 2 and kinds.count("forced_continuation") == 1
    assert resolution.details["stop_refusals"] == verifier.stop_refusals
    assert resolution.details["forced_continuations"] == verifier.forced_continuations
    assert resolution.details["decision"]["use"] == "agent_2"


def test_an_agent_that_always_stops_gets_forced_continuations_until_the_budget_ends_it(card, proxy, tmp_path):
    """A12 (strict): an agent that returns no moves every round while every call improves on its parent gets one
    forced continuation per round; each counts in the budget, so the budget ends the search."""
    seen = []

    def rising(params):                       # every call 0.1 above the previous one: beyond 2 SE, no plateau
        seen.append(params)
        q = 0.5 + 0.1 * len(seen)
        return q, q

    engine = Engine([[mv(view_pairing="same_well")]], final=["continuation_3"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=rising, budget=4)
    assert waves == [1, 1, 1, 1] and len(tool.calls) == 5                          # trial 0 and the 4 rounds' calls
    names = [verifier.label(i) for i in range(1, 5)]
    assert names == ["agent_1", "continuation_1", "continuation_2", "continuation_3"]
    assert resolution.parents == [None, 0, 1, 2, 3]
    assert [r["parent"] for r in verifier.forced_continuations] == ["agent_1", "continuation_1", "continuation_2"]
    assert len(verifier.stop_refusals) == 3 and verifier.stop_reason is None       # the budget ended it, not a stop
    assert resolution.details["decision"]["use"] == "continuation_3"


def test_a_stop_is_accepted_when_the_last_call_did_not_improve(card, proxy, tmp_path):
    """A12: when the last finished call did not improve on its parent (delta <= 0), the stop gets the shared stop
    check and an empty answer ends the search; no refusal."""
    engine = Engine([[mv(view_pairing="same_well")], [mv("agent_1", view_pairing="same_crop")]], final=["agent_1"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    assert waves == [1, 1] and verifier.stop_refusals == [] and verifier.stop_checks == 1
    assert verifier._last_call(verifier._attempts_seen)[2]["delta"] < 0
    check = engine.questions()[-1]
    assert "You returned no moves: the search ends here" in check and "not accepted" not in check
    assert verifier.stop_reason == "satisfied" and resolution.details["stop_refusals"] == []


def test_the_stub_card_is_the_contract_field_lines(card):
    verifier = RepresentationAgenticVerifier()
    assert verifier.levers == LEVERS and verifier.defaults()["init_from"] == "pretrained"
    assert verifier.coerce("contrastive_mask", "off") == "off" and verifier.coerce("trainable_pct", "null") is None


# ---- the stage map -------------------------------------------------------------------------------------------

def test_the_representation_param_map_keeps_its_entries_adds_the_levers_and_never_forwards_init_from():
    spec = next(s for s in STAGES if s.stage == "representation")
    assert set(spec.param_map) == PARAM_MAP_KEYS and len(spec.param_map) == 36
    assert all(spec.param_map[k] == k for k in spec.param_map)          # identity-mapped
    assert spec.default_params == {"learning_rate": 3e-5, "max_epochs": None, "augment_color": True,
                                   "field_aware": False, "adaptation_depth": "trained"}
    params = {k: f"v_{k}" for k in PARAM_MAP_KEYS} | {"init_from": "parent"}
    mapped = map_tool_params(TOOL, params, PARAM_MAP_KEYS | {"init_from"})
    assert set(mapped) == PARAM_MAP_KEYS and "init_from" not in mapped
    task = RepresentationTask(cores=1, gpus=0, deadline=None, brief="")
    assert map_tool_params(TOOL, frozen_params(task), PARAM_MAP_KEYS) == {
        "adaptation_depth": "zero_shot", "split_key": "split", "output_subdir": "ssl_calls/input"}


def test_the_frozen_trial_reaches_the_tool_as_zero_shot_through_invoke_managed_tool():
    received = []

    class Analyzer:
        def execute(self, cell_crops=None, cell_metadata=None, adaptation_depth=None, split_key=None,
                    val_fraction=None, output_subdir=None, learning_rate=3e-5, query_cache_dir=None):
            received.append({"adaptation_depth": adaptation_depth, "split_key": split_key,
                             "output_subdir": output_subdir})
            return {"mode": "zero-shot" if adaptation_depth == "zero_shot" else "training"}

    task = RepresentationTask(cores=1, gpus=0, deadline=None, brief="", split_key=None)
    result = invoke_managed_tool(Analyzer(), TOOL, {"cell_crops": [], "cell_metadata": []}, frozen_params(task))
    assert result["mode"] == "zero-shot"
    assert received == [{"adaptation_depth": "zero_shot", "split_key": None, "output_subdir": "ssl_calls/input"}]
    assert frozen_params(task)["val_fraction"] == 0.1


# ---- admission: the gate, then the agent's round-1 plan -------------------------------------------------------

@pytest.mark.parametrize("meta", [crops(val=("A01",)), [{k: v for k, v in m.items() if k != "well"} for m in crops()]],
                         ids=["one_val_well_per_plate", "batch_tier"])
@pytest.mark.parametrize("strategy", ["legacy", "four_stage"])
def test_a_failed_gate_skips_with_zero_calls_and_delivers_frozen(card, proxy, tmp_path, meta, monkeypatch, strategy):
    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_REPRESENTATION", strategy)
    engine = Engine([[mv(trainable_pct=50)]])
    verifier, resolution, tool, waves = run(tmp_path, engine, meta=meta)
    assert len(tool.calls) == 1 and waves == [] and engine.asked == []          # trial 0 only, no question
    assert resolution.status == "satisfied" and resolution.reason == "not_measurable"
    assert resolution.details["decision"] == {"use": "frozen", "case": "not_measurable", "delta": None, "se": None,
                                              "n_units": 0}
    assert verifier.admission["decision"] == "SKIP" and verifier.admission["reasons"]
    assert resolution.details["tuning_gate"] == {
        "required": True, "source": "input", "decision": "SKIP", "allowed": False, "reason": "not_measurable"}
    assert (tmp_path / "cache" / "cell_state_analysis" / "cell_state_analyzed.h5ad").exists()


def test_four_stage_ssl_keeps_checkpoint_continuation_and_frozen_fallback(card, proxy, tmp_path, monkeypatch):
    from shape.agent.verification.test.test_four_stage import CoveringEngine

    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_REPRESENTATION", "four_stage")
    engine = CoveringEngine(Engine([[mv(trainable_pct=33)], [mv("agent_1", init_from="parent", trainable_pct=50)]],
                                   final=["agent_2"]))
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    assert waves == [1, 1, 1] and tool.calls[2]["init_checkpoint"]           # + the forced continuation (A12)
    record = resolution.details["search_strategy"]
    assert record["confirmation"]["call"] == "agent_2"
    assert "training_history" in record["coverage"]["changed"]
    assert record["controls"] == []                       # the stub has no step telemetry
    assert "search_training" in resolution.details

    _, frozen, _, _ = run(tmp_path / "frozen", CoveringEngine(Engine([[]])))
    assert frozen.details["search_strategy"]["confirmation"]["call"] == "input"
    assert frozen.details["search_strategy"]["confirmation"]["status"] == "baseline_retained"


def test_an_empty_round_1_plan_delivers_frozen_as_no_tuning_needed(card, proxy, tmp_path):
    engine = Engine([[]])
    verifier, resolution, tool, waves = run(tmp_path, engine)
    assert len(tool.calls) == 1 and waves == []
    assert [name for name, _ in engine.asked] == ["BasedPlan", "BasedPlan"]      # the round and its stop check only
    assert engine.asked[0][1].startswith("You tune a fixed self-supervised fine-tune")   # the shared round prompt
    assert "You returned no moves" in engine.asked[1][1] and verifier.stop_checks == 1
    assert verifier.admission["decision"] == "RUN" and verifier.stop_reason == "satisfied" and not verifier.rounds
    assert resolution.status == "satisfied" and resolution.reason == "no_tuning_needed" and resolution.fault is None
    assert resolution.details["decision"]["use"] == "frozen" and resolution.details["call"] == "input"
    assert resolution.result["adata_path"].endswith(str(Path("cell_state_analysis") / "cell_state_analyzed.h5ad"))


# ---- moves ---------------------------------------------------------------------------------------------------

def test_multi_lever_moves_from_input_and_from_any_finished_call(card, proxy, tmp_path):
    engine = Engine([[mv(view_pairing="same_well", contrastive_mask="same_well", trainable_pct=33)],
                     [mv("agent_1", learning_rate=1e-5, batch_sampling="per_plate")],
                     [mv("agent_1", intensity_norm="per_plate", augmentation="light")]], final=["agent_1"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    assert waves == [1, 1, 1, 1] and resolution.parents == [None, 0, 1, 1, 3]     # + a forced continuation (A12)
    deltas = [v.param_delta for _, v, _ in resolution.trials[1:]]
    assert deltas == [{"view_pairing": "same_well", "contrastive_mask": "same_well", "trainable_pct": 33},
                      {"learning_rate": 1e-5, "batch_sampling": "per_plate", "init_from": "parent"},
                      {"intensity_norm": "per_plate", "augmentation": "light", "init_from": "parent"},
                      {"init_from": "parent"}]
    third = tool.calls[3]
    assert third["view_pairing"] == "same_well" and third["trainable_pct"] == 33 and third["learning_rate"] == 3e-5
    assert "init_from" not in third and third["split_key"] == "split"
    assert third["init_checkpoint"] == str(resume_of(tmp_path, tool.calls[1]))   # a move from agent_1 resumes it (A7)


def test_a_lever_value_whose_metadata_column_is_missing_is_rejected_before_any_call(card, proxy, tmp_path):
    meta = crops()
    meta[5]["experiment"] = float("nan")
    meta[6]["experiment"] = "  "
    engine = Engine([[mv(batch_sampling="per_experiment")]])
    verifier, resolution, tool, waves = run(tmp_path, engine, meta=meta)
    repair = engine.questions()[1]
    assert "batch_sampling=per_experiment needs the metadata column" in repair and "experiment" in repair
    assert len(tool.calls) == 1 and verifier.stop_reason == "no_valid_move"
    assert resolution.reason == "no_call_ran" and resolution.details["decision"]["use"] == "frozen"
    for lever, value, column in (("view_pairing", "same_site", "site"), ("intensity_norm", "per_plate", "plate"),
                                 ("contrastive_mask", "same_well", "well")):
        verifier._ids = quant.identities([{k: v for k, v in m.items() if k != column} for m in crops()])
        with pytest.raises(ValueError, match=f"needs the metadata column '{column}'"):
            verifier.check_change(lever, value, {}, "input")
    verifier._ids = quant.identities([{**{k: v for k, v in m.items() if k != "plate"}, "plate_id": m["plate"]}
                                      for m in crops()])
    verifier.check_change("intensity_norm", "per_plate", {}, "input")    # a <name>_id key is accepted


def test_init_from_parent_lineage_continuation_and_hashed_directories(card, proxy, tmp_path):
    engine = Engine([[mv(trainable_pct=33)], [mv("agent_1", init_from="parent", trainable_pct=50)],
                     [mv("agent_2")], [mv("agent_1", init_from="pretrained")]], final=["agent_3"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    assert waves == [1, 1, 1]       # round 4's fresh agent_1 is agent_1 itself: "no lever changed relative to agent_1"
    with pytest.raises(ValueError, match="init_from=parent needs a finished call as the parent"):
        verifier.validate({"parent": "input", "changes": [{"lever": "init_from", "value": "parent"}]},
                          {}, set())
    assert "no lever changed relative to agent_1" in engine.questions()[-1]
    agent_1, agent_2, agent_3 = tool.calls[1:]
    trials = resolution.trials
    assert agent_2["init_checkpoint"] == trials[1][2]["resume_best"]
    assert agent_3["init_checkpoint"] == trials[2][2]["resume_best"] and agent_3["trainable_pct"] == 50
    assert trials[2][1].param_delta == {"init_from": "parent", "trainable_pct": 50}
    assert trials[3][1].param_delta == {"init_from": "parent"}           # a pure continuation is a move
    keys = [verifier.config_key(verifier.trial_config(p)) for p in (agent_1, agent_2, agent_3)]
    assert len(set(keys)) == 3
    assert verifier._parent_keys[agent_2["init_checkpoint"]] == keys[0]
    assert verifier._parent_keys[agent_3["init_checkpoint"]] == keys[1]
    for params, key in zip((agent_1, agent_2, agent_3), keys):
        assert params["output_subdir"] == f"ssl_calls/{hashlib.sha1(key.encode()).hexdigest()[:12]}"
        assert "init_from" not in params and params["max_minutes"] is None
    assert resolution.details["decision"]["use"] == "agent_3"


# the SHAPE moves of the basic run 47010208 (commit 9b86927e, HEPG2; logs/tool_calls.jsonl): agent_2..agent_4 named
# a finished parent and set no init_from, so each restarted from the pretrained backbone (no init_checkpoint)
RUN_47010208_AGENT_1 = {"view_pairing": "same_well", "contrastive_mask": "same_well", "batch_sampling": "per_plate",
                        "trainable_pct": 33, "learning_rate": 2e-05, "clip_grad_norm": 3.0, "warmup_fraction": 0.1,
                        "intensity_norm": "per_plate", "augmentation": "light",
                        "drop_path": 0.1}        # its head_warmup_epochs 0.1 left out: no lever since A11


def test_a_move_from_a_finished_call_resumes_it_unless_it_sets_init_from_amendment_a7(card, proxy, tmp_path):
    engine = Engine([[mv(**RUN_47010208_AGENT_1)], [mv("agent_1", trainable_pct=66)],
                     [mv("agent_2", trainable_pct=100)], [mv("agent_2", head_learning_rate=0.001)],
                     [mv("agent_2", init_from="pretrained", trainable_pct=100)], [mv(trainable_pct=50)]],
                    final=["agent_3"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well, budget=6)
    assert waves == [1, 1, 1, 1, 1, 1]
    trials = resolution.trials
    agent_1, agent_2, agent_3, agent_4, agent_5, agent_6 = tool.calls[1:]
    best = {i: trials[i][2]["resume_best"] for i in (1, 2)}
    assert agent_1["init_checkpoint"] is None                            # the first call from input: pretrained
    # the 47010208 moves: init_from = parent by default, the parent's resume_best as init_checkpoint
    assert agent_2["init_checkpoint"] == best[1] and agent_2["trainable_pct"] == 66
    assert agent_3["init_checkpoint"] == best[2] and agent_3["trainable_pct"] == 100
    assert agent_4["init_checkpoint"] == best[2] and agent_4["head_learning_rate"] == 0.001
    assert [trials[i][1].param_delta for i in (2, 3, 4)] == [
        {"trainable_pct": 66, "init_from": "parent"}, {"trainable_pct": 100, "init_from": "parent"},
        {"head_learning_rate": 0.001, "init_from": "parent"}]
    # an explicit pretrained restarts with the parent's other levers; a move from input stays pretrained
    assert agent_5["init_checkpoint"] is None and agent_5["trainable_pct"] == 100
    assert agent_5["view_pairing"] == "same_well" and agent_5["learning_rate"] == 2e-05
    assert trials[5][1].param_delta == {"init_from": "pretrained", "trainable_pct": 100}
    assert agent_6["init_checkpoint"] is None and agent_6["view_pairing"] == "same_crop"
    assert agent_5["output_subdir"] != agent_3["output_subdir"]          # fresh and resumed at 100 are two calls
    assert [verifier.start_state(c, p) for c, p in zip(("input", "agent_1", "agent_2", "agent_2", "agent_2", "input"),
                                                      tool.calls[1:])] == [
        "fresh", "resumed: agent_1", "resumed: agent_2", "resumed: agent_2", "fresh", "fresh"]
    note = verifier.defaults_note()
    assert "resumes that\ncall (init_from parent) unless the move sets init_from" in note
    assert all(note in prompt for prompt in engine.questions())


def test_explicit_pretrained_from_a_fresh_parent_restarts_and_a_bare_continuation_runs_once(card, proxy, tmp_path):
    """A7: from a fresh parent, init_from pretrained with another change restarts from the pretrained backbone with the
    parent's other levers; a move with no change resumes the parent (a continuation) and repeating it is rejected."""
    engine = Engine([[mv(view_pairing="same_well", trainable_pct=33)],
                     [mv("agent_1", init_from="pretrained", learning_rate=1e-5)],
                     [mv("agent_1")], [mv("agent_1")]], final=["agent_3"])
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=same_well)
    assert waves == [1, 1, 1]
    agent_1, agent_2, agent_3 = tool.calls[1:]
    assert agent_2["init_checkpoint"] is None and agent_2["learning_rate"] == 1e-5
    assert (agent_2["view_pairing"], agent_2["trainable_pct"]) == ("same_well", 33)        # the parent's other levers
    assert resolution.trials[2][1].param_delta == {"learning_rate": 1e-5}       # pretrained is agent_1's own value
    assert agent_3["init_checkpoint"] == str(resume_of(tmp_path, agent_1))
    assert (agent_3["view_pairing"], agent_3["trainable_pct"], agent_3["learning_rate"]) == ("same_well", 33, 3e-5)
    assert resolution.trials[3][1].param_delta == {"init_from": "parent"}
    assert [verifier.start_state(p, c) for p, c in (("input", agent_1), ("agent_1", agent_2), ("agent_1", agent_3))] \
        == ["fresh", "fresh", "resumed: agent_1"]
    assert "this configuration was already run" in engine.questions()[-1]         # the repeated continuation


def test_a_parent_without_a_resume_state_is_rejected_without_naming_its_file(card, proxy, tmp_path):
    """A7: a move from a finished call that left no resume state (and sets no init_from) is rejected with a message
    that names no logged-only output (no resume_best path or file name) and says how to restart."""
    meta = crops()
    stub = StubTool(tmp_path, meta, same_well)

    def tool(params):
        out = stub(params)
        return {**out, "resume_best": None, "resume_end": None}

    engine = Engine([[mv(view_pairing="same_well")], [mv("agent_1", trainable_pct=50)]], final=["agent_1"])
    verifier, resolution, _, waves = run(tmp_path, engine, meta=meta, tool=tool)
    assert waves == [1]
    repair = next(q for q in engine.questions() if "were rejected" in q)
    assert ("agent_1 kept no training state to resume; set init_from pretrained to restart from the pretrained "
            "backbone") in repair
    for prompt in engine.prompts():
        assert "resume_best" not in prompt and ".pt" not in prompt
    with pytest.raises(ValueError) as error:
        verifier.validate({"parent": "agent_1", "changes": [{"lever": "trainable_pct", "value": "50"}]},
                          verifier._bind(StepVerifier._wave_state(resolution.trials, resolution.parents, [], 0,
                                                                  {}))[1], set())
    assert "resume_best" not in str(error.value) and "kept no training state" in str(error.value)


# ---- the plateau, the comparison draws -----------------------------------------------------------------------

def flat(params):
    """Every call at the frozen input's value: no progress, ever."""
    return 0.5, 0.5


def kinds_of(logged):
    return [kind for kind, _ in logged]


def test_with_a_reasoner_a_plateau_neither_stops_the_search_nor_forces_a_move(card, proxy, tmp_path):
    """A20 (withdraws A13's exploration round): 3 and more rounds without progress do not stop the search and start
    no exploration round; every planned move runs (any lever, again or not) until the budget ends it, and the round
    prompt states that a plateau is the agent's to judge."""
    moves = [[mv(trainable_pct=p)] for p in (10, 20, 30, 40, 50)]
    engine = Engine(moves)
    verifier, resolution, tool, waves = run(tmp_path, engine, quality=flat, budget=5)
    inp, attempts = verifier._bind(StepVerifier._wave_state(resolution.trials, resolution.parents, [], 0, {}))
    assert base.PATIENCE == verifier.patience == 3 and verifier._stall(inp, attempts) >= 3
    assert waves == [1, 1, 1, 1, 1] and verifier.stop_reason != "plateau"
    assert [c.get("trainable_pct") for c in tool.calls[1:]] == [10, 20, 30, 40, 50]
    assert all("a plateau does not stop the search" in q for q in engine.questions()[:5])
    assert "plateau_explorations" not in resolution.details and not hasattr(verifier, "_exploration")


def test_the_guard_the_plateau_and_the_shown_deltas_read_the_comparison_draws(card, proxy, tmp_path):
    def crossed(params):                                                # fresh and selection values disagree
        return {10: (0.5, 0.8765), 20: (0.6, 0.3456)}[params["trainable_pct"]]

    engine = Engine([[mv(trainable_pct=10)], [mv(trainable_pct=20)]], final=["agent_1", "agent_2"])
    verifier, resolution, _, _ = run(tmp_path, engine, quality=crossed)
    inp, attempts = verifier._bind(StepVerifier._wave_state(resolution.trials, resolution.parents, [], 0, {}))
    assert verifier.guard(inp, attempts["agent_1"][2]) and not verifier.guard(inp, attempts["agent_2"][2])
    # agent_2 beat frozen on the comparison draws; the forced continuation of agent_2 (A12) equals it: one round
    assert verifier._stall(inp, attempts) == 1 and "continuation_1" in attempts
    final = engine.questions("FinalChoice")
    assert "Your choice 'agent_1' is not admissible: val sibling_site_map does not beat" in final[1]
    assert resolution.details["call"] == "agent_2" and resolution.details["choice"] == "agent_choice"
    shown = row(engine.questions()[-1], "agent_2")
    assert float(shown["vs_input"].split()[0]) == pytest.approx(0.1, abs=0.01) and "beyond 2 SE" in shown["vs_input"]
    assert shown["sibling_site_map"]["value"] == 0.6 and row(final[0], "agent_1")["vs_input"].startswith("+0.0000")
    assert seeds_seen(proxy) == {quant.FRESH_SEED, quant.SELECTION_SEED}
    for prompt in engine.prompts():
        assert "0.8765" not in prompt and "0.3456" not in prompt
    logged = resolution.details["trials"]["agent_1"]["logged"]
    assert logged["sibling_site_map_selection_draws"] == pytest.approx(0.8765)
    assert resolution.details["decision"]["delta"] == pytest.approx(0.1, abs=0.01)


def seeds_seen(calls):
    return {c["seed"] for c in calls}


# ---- resources -----------------------------------------------------------------------------------------------

def test_the_resource_stop_fires_at_max_minutes_0_and_below_a_calls_or_trial_0s_non_training_time(card, proxy, tmp_path):
    verifier, resolution, tool, waves = run(tmp_path, Engine([[mv(trainable_pct=10)]]), quality=same_well,
                                            deadline=time.time() + 10 * 60)
    assert waves == [] and len(tool.calls) == 1 and verifier.stop_reason == "deadline"
    assert resolution.reason == "no_call_ran" and resolution.details["decision"]["use"] == "frozen"
    # before any call finished: 0.5 minutes left, below trial 0's 1 non-training minute
    verifier, resolution, tool, waves = run(tmp_path / "a", Engine([[mv(trainable_pct=10)]]), quality=same_well,
                                            deadline=time.time() + (rep.RESERVE_MINUTES + 0.5) * 60)
    assert waves == [] and len(tool.calls) == 1 and verifier.stop_reason == "deadline"
    assert resolution.reason == "no_call_ran"
    # 1.5 minutes left for a call after round 1, below agent_1's 3 non-training minutes: the final question still
    # runs
    engine = Engine([[mv(view_pairing="same_well")], [mv(trainable_pct=20)]], final=["agent_1"])
    verifier, resolution, tool, waves = run(tmp_path / "b", engine, quality=same_well,
                                            deadline=time.time() + (rep.RESERVE_MINUTES + 1.5) * 60)
    assert waves == [1] and verifier.stop_reason == "deadline" and 0 < tool.calls[1]["max_minutes"] <= 1.5
    assert len(engine.questions("FinalChoice")) == 1 and resolution.details["call"] == "agent_1"
    assert resolution.reason == "deadline"


def test_one_call_runs_at_a_time(card, proxy, tmp_path):
    logged = []
    engine = Engine([[mv(trainable_pct=10), mv(trainable_pct=20), mv(trainable_pct=30)]], final=["agent_1"])
    verifier, _, _, waves = run(tmp_path, engine, quality=same_well, log=lambda kind, **f: logged.append(kind))
    assert verifier.cap == 1 and verifier.gpus == 2 and waves == [1, 1, 1] and "dropped_move" in logged   # A12
    assert "This round: at most 1 call(s)." in engine.questions()[0]
    assert "one call runs at a time" in engine.questions()[0]


# ---- the final choice and the frozen delivery -----------------------------------------------------------------

def test_the_agents_admissible_choice_is_delivered_with_its_resume_state(card, proxy, tmp_path):
    engine = Engine([[mv(view_pairing="same_well")]], final=["agent_1"])
    verifier, resolution, _, _ = run(tmp_path, engine, quality=same_well)
    assert resolution.status == "satisfied" and resolution.details["choice"] == "agent_choice"
    root = tmp_path / "cache" / "cell_state_analysis"
    assert resolution.result["adata_path"] == str(root / "cell_state_analyzed.h5ad")
    assert resolution.result["resume_best"] == str(root / "resume_best.pt") and (root / "resume_best.pt").exists()
    decision = resolution.details["decision"]
    assert decision["use"] == "agent_1" and decision["case"] == "agent_choice" and decision["n_units"] == 8
    assert decision["delta"] == pytest.approx(0.1, abs=0.01) and decision["se"] is not None
    assert resolution.details["tier"] == "replicate" and resolution.details["adopted_proxy"] == "sibling_site_map"
    record = resolution.details["trials"]["agent_1"]
    assert record["change"] == {"view_pairing": "same_well"} and record["parent"] == "input"


@pytest.mark.parametrize("case, final, quality, fail, status", [
    ("agent_kept_input", [None], gain, None, "conceded"),
    ("final_choice_inadmissible", ["agent_9", "agent_9"], gain, None, "conceded"),
    ("no_admissible_call", [], frozen_quality, None, "conceded"),
    ("every_call_failed", [], gain, lambda params: params.get("trainable_pct") == 10, "conceded"),
])
def test_frozen_is_delivered_in_each_select_case_with_its_name(card, proxy, tmp_path, case, final, quality, fail,
                                                              status):
    meta = crops()
    tool = StubTool(tmp_path, meta, quality, fail=fail)
    engine = Engine([[mv(trainable_pct=10)]], final=final)
    verifier, resolution, _, _ = run(tmp_path, engine, meta=meta, tool=tool)
    assert resolution.reason == case and resolution.status == status and resolution.details["choice"] == case
    assert resolution.details["call"] == "input" and resolution.details["decision"]["use"] == "frozen"
    assert resolution.details["decision"]["case"] == case and resolution.details["selected"] == 0
    assert resolution.result["adata_path"].endswith(str(Path("cell_state_analysis") / "cell_state_analyzed.h5ad"))
    assert resolution.result["resume_best"] is None
    if case == "final_choice_inadmissible":
        assert len(engine.questions("FinalChoice")) == 2                 # one retry


# ---- the engine-less fallback --------------------------------------------------------------------------------

def test_ssl_missing_engine_never_starts_rule_based_training(card, proxy, tmp_path):
    from shape.engine.errors import NoEngineError
    with pytest.raises(NoEngineError):
        run(tmp_path, None, quality=same_well)


def resume_of(tmp_path, params):
    return tmp_path / "cache" / "cell_state_analysis" / params["output_subdir"] / "resume_best.pt"


def test_failed_ssl_reasoner_keeps_frozen_input_without_training(card, proxy, tmp_path):
    verifier, resolution, tool, waves = run(tmp_path, Down(), quality=same_well)
    assert waves == [] and len(tool.calls) == 1
    assert verifier.reasoner_failures == 1 and verifier.fallback_rounds == 0
    assert resolution.details["selected"] == 0
    assert resolution.details["execution_errors"][0]["question"] == "round"

def test_no_logged_only_readout_reaches_a_prompt_and_alarms_show_trial_0s_values(card, proxy, tmp_path):
    engine = Engine([[mv(view_pairing="same_well")], [mv("agent_1", init_from="parent", trainable_pct=50)]],
                    final=["agent_2"])
    verifier, resolution, _, _ = run(tmp_path, engine, quality=same_well)
    prompts = engine.prompts()
    assert len(prompts) == 7      # 3 rounds, the refusal, a forced continuation's round and stop check, the final
    for prompt in prompts:
        for mark in LOGGED_MARKS:
            assert mark not in prompt, mark
        assert str(tmp_path) not in prompt and "cache" not in prompt     # no resume path
    last = engine.questions()[-1]
    first_input = verifier._shown(0, resolution.trials[0][1])["ALARMS"]
    alarms = row(last, "agent_1")["ALARMS"]
    assert set(alarms) == set(rep.ALARMS) and all(alarms[k]["input"] == first_input[k] for k in rep.ALARMS)
    assert '"ALARMS"' in last and "ALARMS. They flag a failure mode." in last
    assert row(last, "agent_1")["training"] == {"epochs_trained": 3, "best_epoch": 2, "best_by": "val_proxy",
                                            "batch_size": 64, "trainable_blocks": 4, "seconds": 300.0}
    assert "n_train 48 cells, n_val 24 cells, n_val_wells 4" in last and "Tier: replicate" in last
    assert "ALARMS. They flag a failure mode." in engine.questions("FinalChoice")[0]
    trials = resolution.details["trials"]
    assert trials["agent_1"]["logged"]["val_history"][0]["eval_loss"] == 2.468
    assert trials["agent_1"]["logged"]["training"]["grad_norm_median"] == 4.321


def test_the_crop_array_is_no_lever_and_never_reaches_a_prompt(proxy, tmp_path, monkeypatch):
    """Amendment A1: crop_array is a bookkeeping kwarg of every call (the benchmark's base kwargs, through
    invoke_managed_tool), never a lever of the real CARD, never in its lever doc or in any prompt."""
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC", raising=False)
    verifier = RepresentationAgenticVerifier()
    assert "crop_array" not in verifier.levers and "crop_array" not in verifier.lever_fields
    assert "crop_array" not in verifier._lever_doc() and "crop_array" not in (verifier.tuning_card() or "")
    secret = str(tmp_path / "PACKED_SECRET" / "images.npy")
    meta = [{**m, "array_row": i} for i, m in enumerate(crops())]
    base_kwargs = {"crop_array": secret, "cell_metadata": meta, "query_cache_dir": str(tmp_path / "cache")}
    stub, received = StubTool(tmp_path, meta, same_well), []

    class Analyzer:
        def execute(self, **kwargs):
            received.append(kwargs)
            return stub({k: v for k, v in kwargs.items() if k not in base_kwargs})

    Analyzer.execute.__signature__ = inspect.signature(Cell_State_Analyzer_Tool.execute)

    def tool(params):
        return invoke_managed_tool(Analyzer(), TOOL, {**base_kwargs, "cell_metadata": [dict(m) for m in meta]},
                                   params)

    engine = Engine([[mv(view_pairing="same_well")], [mv("agent_1", init_from="parent", trainable_pct=50)]],
                    final=["agent_2"])
    verifier.llm_engine = engine
    task = RepresentationTask(cores=8, gpus=2, deadline=None, brief="tune the SSL fine-tune of these crops")
    initial = frozen_params(task)
    StepVerifier(max_iters=9).resolve_stage(verifier, verifier.make_adapter(base_kwargs), tool, initial,
                                            tool(initial), upstream_ctx={"representation_task": task})
    assert len(received) == 4 and all(kwargs["crop_array"] == secret for kwargs in received)   # + a continuation
    assert len(engine.prompts()) == 7      # 3 rounds, the refusal, the continuation's round and stop check, final
    for prompt in engine.prompts():
        assert "crop_array" not in prompt and "PACKED_SECRET" not in prompt and "images.npy" not in prompt
        assert "array_row" not in prompt


def test_the_readouts_text_states_no_task_116_number_and_names_no_lever(card):
    text = RepresentationAgenticVerifier().readouts_text()
    for number in TASK116_NUMBERS:
        assert number not in text, number
    assert not re.search(r"\d\.\d{3,}", text)                             # no result-like decimal at all
    for lever in LEVERS:
        assert not re.search(rf"\b{lever}\b", text), lever
    for part in ("Val wells.", "sibling_site_map (the val proxy).", "How far it can be trusted:", "Two sets of draws.",
                 "Comparisons.", "ALARMS.", "- effective_rank:", "- dead_dim_frac:", "- image_leakage:",
                 "Training facts per call:"):
        assert part in text


# ---- the val rows and the selection check -----------------------------------------------------------------------

def test_val_rows_are_taken_in_the_tools_validation_order_by_cell_id(card, proxy, tmp_path):
    meta = crops()
    tool = StubTool(tmp_path, meta, same_well, reverse=True)
    run(tmp_path, Engine([[mv(trainable_pct=10)]], final=["agent_1"]), meta=meta, tool=tool)
    val = [i for i, m in enumerate(meta) if m["split"] == "val"]
    assert proxy and all(c["rows"] == val and c["cells"] == [meta[i]["cell_id"] for i in val] for c in proxy)
    assert all((c["k"], c["draws"]) == (2, 20) for c in proxy)


def test_rows_that_do_not_match_the_crops_are_a_call_error(card, proxy, tmp_path):
    meta = crops()

    class Renamed(StubTool):
        def __call__(self, params):
            result = super().__call__(params)
            if params.get("trainable_pct"):
                adata = ad.read_h5ad(result["adata_path"])
                adata.obs["cell_id"] = [f"other{i}" for i in range(adata.n_obs)]
                adata.write_h5ad(result["adata_path"])
            return result

    verifier, resolution, _, _ = run(tmp_path, Engine([[mv(trainable_pct=10)]]), meta=meta,
                                     tool=Renamed(tmp_path, meta, same_well))
    assert [f["error"] for f in verifier.failed] == ["the embedding's rows do not match the crops' metadata"]
    assert resolution.reason == "every_call_failed"


def test_a_selection_draw_mismatch_is_recorded_never_a_call_error_never_shown(card, proxy, tmp_path):
    meta = crops()
    tool = StubTool(tmp_path, meta, same_well, history_offset=0.05)
    engine = Engine([[mv(view_pairing="same_well")]], final=["agent_1"])
    verifier, resolution, _, _ = run(tmp_path, engine, meta=meta, tool=tool)
    check = resolution.details["trials"]["agent_1"]["logged"]["selection_check"]
    assert check["within_tol"] is False and check["abs_diff"] == pytest.approx(0.05)
    assert check["recomputed"] == pytest.approx(0.4) and check["val_history_best"] == pytest.approx(0.45)
    assert not verifier.failed and resolution.details["call"] == "agent_1"
    frozen_check = resolution.details["trials"]["input"]["logged"]["selection_check"]   # trial 0's epoch-0 entry
    assert frozen_check["within_tol"] is True and frozen_check["val_history_best"] == pytest.approx(frozen_check["recomputed"])
    for prompt in engine.prompts():
        assert "selection_check" not in prompt and "0.45" not in prompt
