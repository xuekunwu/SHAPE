"""The analyzer's optional `min_epochs` (the early stopper's floor; representation_RxRx1 amendment A23): None keeps
today's resolved training config for every caller, an explicit value overrides only min_epochs and reaches the
stopper; it is no CARD lever and no managed param (the agents never see it), and the fingerprint of a call without
it is unchanged.
"""
import inspect
import math

import pytest

import shape.tools.cell_state_analyzer.tool as tool_mod
from shape.utils.adaptive_training import AdaptiveEarlyStopper, resolve_training_config

ad = pytest.importorskip("anndata")


def _todays_floor(max_epochs):
    return min(max_epochs, max(4, math.ceil(0.35 * max_epochs)))


@pytest.mark.parametrize("samples,max_epochs,patience", [(24, None, None), (5000, None, None), (300, 72, None),
                                                         (300, 3, None), (300, 9, 2)])
def test_min_epochs_none_keeps_todays_config_and_a_value_overrides_only_it(samples, max_epochs, patience):
    today = resolve_training_config(samples, 16, max_epochs, patience)
    assert resolve_training_config(samples, 16, max_epochs, patience, None) == today
    assert today["min_epochs"] == _todays_floor(today["max_epochs"])
    for value, expected in ((1, 1), (0, 1), (2, min(2, today["max_epochs"])), (10 ** 6, today["max_epochs"])):
        explicit = resolve_training_config(samples, 16, max_epochs, patience, min_epochs=value)
        assert explicit == {**today, "min_epochs": expected}


def test_without_a_floor_the_plateau_stops_as_soon_as_its_window_is_full():
    """min_epochs 1: a flat loss stops after `patience` + 1 epochs (the plateau window); today's floor at 72
    epochs holds it to 26."""
    config = resolve_training_config(300, 16, 72)
    stops = {}
    for floor in (None, 1):
        resolved = resolve_training_config(300, 16, 72, min_epochs=floor)
        stopper = AdaptiveEarlyStopper(min_epochs=resolved["min_epochs"], patience=resolved["patience"],
                                       relative_min_delta=resolved["relative_min_delta"])
        stops[floor] = next(epoch for epoch in range(1, 73) if stopper.observe(1.0, epoch)[1])
    assert stops == {None: config["min_epochs"], 1: config["patience"] + 1}


def test_execute_passes_min_epochs_to_the_stopper_and_records_it(run_call, train_spy, monkeypatch):
    floors = []
    real = tool_mod.AdaptiveEarlyStopper

    def recording(**kwargs):
        floors.append(kwargs["min_epochs"])
        return real(**kwargs)

    monkeypatch.setattr(tool_mod, "AdaptiveEarlyStopper", recording)
    default = run_call(subdir="default", split_key="split", max_epochs=9)
    floor = run_call(subdir="floor", split_key="split", max_epochs=9, min_epochs=1)
    assert floors == [_todays_floor(9), 1]
    assert [call["kwargs"]["min_epochs"] for call in train_spy] == [None, 1]
    assert default["training_config"]["min_epochs"] == _todays_floor(9) and floor["training_config"]["min_epochs"] == 1
    assert "min_epochs" not in ad.read_h5ad(default["adata_path"]).uns["analyzer_kwargs"]
    assert ad.read_h5ad(floor["adata_path"]).uns["analyzer_kwargs"]["min_epochs"] == 1


def test_min_epochs_is_no_lever_and_leaves_other_fingerprints_unchanged():
    from shape.agent.verification.stage_map import map_tool_params, stages_for_tool
    from shape.tools.base import load_tuning_card

    assert inspect.signature(tool_mod.Cell_State_Analyzer_Tool.execute).parameters["min_epochs"].default is None
    assert all("min_epochs" not in stage.param_map for stage in stages_for_tool("Cell_State_Analyzer_Tool"))
    assert map_tool_params("Cell_State_Analyzer_Tool", {"min_epochs": 1}, {"min_epochs"}) == {}
    assert "min_epochs" not in load_tuning_card("cell_state_analyzer")
    params = {"seed": 0, "max_epochs": 9}
    base = tool_mod.embedding_fingerprint(params)
    assert tool_mod.embedding_fingerprint({**params, "min_epochs": None}) == base
    assert tool_mod.embedding_fingerprint({**params, "min_epochs": 1}) != base
