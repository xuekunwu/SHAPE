"""Search-level early stopping of the representation verifier (`early_stop`, SHAPE-ES; default off = `main`):
a stop is refused only while the last call beat its parent by more than 2 SE, and the search ends after
`plateau_rounds` rounds in a row without a call that beats the best so far by more than 2 SE.
"""
import pytest

from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.test.test_representation_agentic import (  # noqa: F401  (fixtures)
    Engine, StubTool, card, crops, flat, mv, proxy)
from shape.agent.verification.verifiers.representation import (RepresentationAgenticVerifier, RepresentationTask,
                                                               frozen_params)


def run_es(tmp_path, engine, *, quality, budget=8, early_stop=None):
    meta = crops()
    tool = StubTool(tmp_path, meta, quality)
    verifier = RepresentationAgenticVerifier(early_stop=early_stop)
    verifier.llm_engine = engine
    task = RepresentationTask(cores=8, gpus=2, deadline=None, brief="tune the SSL fine-tune of these crops")
    events = []
    initial = frozen_params(task)
    resolution = StepVerifier(max_iters=1 + budget, observer=events.append).resolve_stage(
        verifier, verifier.make_adapter({"cell_metadata": meta}), tool, initial, tool(initial),
        upstream_ctx={"representation_task": task})
    return verifier, resolution, tool, [e["size"] for e in events if e["event"] == "wave"]


def tiny(params):
    """+0.1 for same_well pairing, +0.0005 for trainable_pct 1 (within the noise), no bonus for resuming."""
    q = 0.5 + 0.1 * (params.get("view_pairing") == "same_well") + 0.0005 * (params.get("trainable_pct") == 1)
    return q, 1.0 - q


def test_default_is_off():
    v = RepresentationAgenticVerifier()
    assert v.early_stop is None and v.patience == 3
    with pytest.raises(ValueError):
        RepresentationAgenticVerifier(early_stop={"plateau_rounds": 0})


def test_with_early_stop_a_plateau_ends_the_search(card, proxy, tmp_path):
    moves = [[mv(trainable_pct=p)] for p in (10, 20, 30, 40, 50)]
    verifier, _, tool, waves = run_es(tmp_path, Engine(moves), quality=flat, budget=5,
                                      early_stop={"plateau_rounds": 2})
    assert verifier.stop_reason == "plateau" and waves == [1, 1]
    assert [c.get("trainable_pct") for c in tool.calls[1:]] == [10, 20]


def test_without_early_stop_the_same_plateau_runs_to_the_budget(card, proxy, tmp_path):
    moves = [[mv(trainable_pct=p)] for p in (10, 20, 30, 40, 50)]
    verifier, _, _, waves = run_es(tmp_path, Engine(moves), quality=flat, budget=5)
    assert verifier.stop_reason != "plateau" and waves == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("early_stop,refused", [(None, True), ({"plateau_rounds": 2}, False)])
def test_a_stop_after_a_gain_within_the_noise(card, proxy, tmp_path, early_stop, refused):
    engine = Engine([[mv(view_pairing="same_well")], [mv("agent_1", trainable_pct=1)], [], []],
                    final=["agent_1"] * 3)
    verifier, _, _, _ = run_es(tmp_path, engine, quality=tiny, budget=5, early_stop=early_stop)
    last = verifier._pairs
    assert any(p.get("delta") is not None and 0 < p["delta"] and not p["beats"] for p in last.values())
    assert bool(verifier.stop_refusals) is refused
    if not refused:
        assert verifier.stop_reason == "satisfied" and not verifier.forced_continuations
