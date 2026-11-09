"""The epoch limit depends on training data only and is identical for both agents."""
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier, budget_for, shared_budget
from shape.utils.adaptive_training import resolve_training_config


def data(n_train):
    cells = pd.DataFrame({"cell_id": [f"c{i}" for i in range(n_train + 3)],
                          "array_row": list(range(n_train + 3)),
                          "split": ["train"] * n_train + ["val", "unused", "heldout"]})
    summary = {"tool_input": {"train_cells": n_train}, "cells_sha256": "prep",
               "crop_array": {"sha256": "images"}}
    return summary, cells


@pytest.mark.parametrize("n_train,cap", [(1, 20), (128, 14), (512, 11), (100000, 8)])
def test_dataset_cap_and_hard_limit(n_train, cap):
    summary, cells = data(n_train)
    budget = budget_for(summary, cells)
    assert budget["n_train"] == n_train and budget["max_epochs"] == cap
    more_eval = pd.concat([cells, cells[cells.split != "train"]] * 2, ignore_index=True)
    more_eval = pd.concat([cells[cells.split == "train"], more_eval[more_eval.split != "train"]])
    assert budget_for(summary, more_eval) == budget
    for batch_size in (1, 16, 128):
        config = resolve_training_config(n_train, batch_size, max_epochs=cap)
        assert config["max_epochs"] == config["max_epochs_hard_cap"] == cap


def test_parallel_jobs_share_one_immutable_manifest(tmp_path):
    summary, cells = data(128)
    with ThreadPoolExecutor(max_workers=4) as pool:
        budgets = list(pool.map(lambda _: shared_budget(tmp_path, summary, cells), range(4)))
    assert all(b == budgets[0] for b in budgets)
    path = tmp_path / "TRAINING_BUDGET.json"
    assert json.loads(path.read_text()) == budgets[0]
    path.write_text(json.dumps({**budgets[0], "max_epochs": 60}))
    with pytest.raises(SystemExit, match="TRAINING_BUDGET_STALE"):
        shared_budget(tmp_path, summary, cells)


@pytest.mark.parametrize("value", [None, 14, 60])
def test_agents_cannot_override_even_the_same_epoch_cap(value):
    budget = budget_for(*data(128))
    verifier = FixedEpochVerifier(budget)
    assert "max_epochs" not in verifier.levers
    assert "### max_epochs" not in verifier._card_section()
    assert "max_epochs is fixed at 14" in verifier._card_section()
    with pytest.raises(ValueError, match="max_epochs is fixed"):
        verifier.validate({"parent": "input", "changes": [{"lever": "max_epochs", "value": value}]}, {}, set())
    for config in ({}, {"max_epochs": None}, {"max_epochs": 60, "init_checkpoint": "parent-state"}):
        params = verifier.trial_params(config)
        assert params["max_epochs"] == 14
        assert params["init_checkpoint"] == config.get("init_checkpoint")
        assert verifier.trial_config(params)["max_epochs"] == 14
