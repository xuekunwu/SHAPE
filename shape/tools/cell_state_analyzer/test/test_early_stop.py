"""The analyzer's optional within-call early stopping (`early_stop`, SHAPE-ES): default None keeps `main`'s
training-loss stopper and every fingerprint; a rule stops a call with a measurable val split on its val proxy
(`ProxyStopper`) and records the overfitting gap of every evaluation.
"""
import inspect
import json

import pytest

import shape.tools.cell_state_analyzer.tool as tool_mod
from shape.tools.cell_state_analyzer.tool import ProxyStopper, resolve_early_stop

ad = pytest.importorskip("anndata")


def _rule(**kw):
    return resolve_early_stop(kw)


def test_default_is_off_and_fingerprint_unchanged():
    assert inspect.signature(tool_mod.Cell_State_Analyzer_Tool.execute).parameters["early_stop"].default is None
    assert resolve_early_stop(None) is None
    params = {"seed": 0, "max_epochs": 9}
    base = tool_mod.embedding_fingerprint(params)
    assert tool_mod.embedding_fingerprint({**params, "early_stop": None}) == base
    assert tool_mod.embedding_fingerprint({**params, "early_stop": _rule()}) != base
    assert (tool_mod.embedding_fingerprint({**params, "early_stop": _rule(patience=3)})
            != tool_mod.embedding_fingerprint({**params, "early_stop": _rule()}))


def test_resolve_validates():
    assert _rule() == tool_mod.EARLY_STOP_DEFAULTS
    with pytest.raises(ValueError):
        resolve_early_stop({"nope": 1})
    with pytest.raises(ValueError):
        resolve_early_stop({"stop_by": "train_loss"})
    with pytest.raises(ValueError):
        resolve_early_stop({"patience": 0})


def test_plateau_stops_after_patience_and_respects_the_floor():
    s = ProxyStopper(_rule(overfit_gap=False), min_epochs=1)
    assert [s.observe(e, p) for e, p in enumerate([0.5, 0.6, 0.59, 0.58], 1)] == [
        (False, None), (False, None), (False, None), (True, "val_proxy_plateau")]
    s = ProxyStopper(_rule(overfit_gap=False), min_epochs=5)
    out = [s.observe(e, p) for e, p in enumerate([0.6, 0.5, 0.5, 0.5, 0.5], 1)]
    assert out[:4] == [(False, None)] * 4 and out[4] == (True, "val_proxy_plateau")


def test_gap_signal_stops_one_evaluation_earlier():
    s = ProxyStopper(_rule(patience=5), min_epochs=1)
    # val best at 2, then flat while the train-val gap grows twice
    seq = [(0.50, 0.10), (0.60, 0.10), (0.59, 0.12), (0.59, 0.15)]
    out = [s.observe(e, p, g) for e, (p, g) in enumerate(seq, 1)]
    assert out[-1] == (True, "overfit_gap") and not any(stop for stop, _ in out[:-1])


def test_gap_does_not_stop_while_val_improves():
    s = ProxyStopper(_rule(patience=5), min_epochs=1)
    out = [s.observe(e, p, g) for e, (p, g) in enumerate([(0.5, 0.1), (0.6, 0.2), (0.7, 0.3), (0.8, 0.4)], 1)]
    assert not any(stop for stop, _ in out)


def test_execute_stops_on_the_proxy_and_records_the_gap(run_call, train_spy, monkeypatch):
    proxies = iter([0.30, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33] * 4)
    monkeypatch.setattr(tool_mod.Cell_State_Analyzer_Tool, "_val_proxy", staticmethod(lambda v, m: next(proxies)))
    rule = {"patience": 2, "overfit_gap": True, "train_wells": 2}
    out = run_call(subdir="es", split_key="split", max_epochs=9, min_epochs=1, early_stop=rule)
    cfg = out["training_config"]
    assert train_spy[-1]["kwargs"]["early_stop"] == resolve_early_stop(rule)
    assert cfg["stop_reason"] in ("val_proxy_plateau", "overfit_gap")
    assert cfg["epochs_trained"] < 9 and cfg["epochs_saved"] == 9 - cfg["epochs_trained"]
    assert cfg["best_epoch"] == 2
    assert all("overfit_gap" in r and "train_sibling_site_map" in r for r in cfg["val_history"])
    kwargs = ad.read_h5ad(out["adata_path"]).uns["analyzer_kwargs"]
    assert json.loads(kwargs["early_stop"]) == resolve_early_stop(rule)


def test_execute_default_has_no_proxy_stop(run_call, train_spy):
    out = run_call(subdir="default", split_key="split", max_epochs=2)
    cfg = out["training_config"]
    assert train_spy[-1]["kwargs"]["early_stop"] is None
    assert cfg["early_stop"] is None and cfg["epochs_saved"] == 0
    assert all("overfit_gap" not in r for r in cfg["val_history"])
    assert "early_stop" not in ad.read_h5ad(out["adata_path"]).uns["analyzer_kwargs"]
