"""execute()-level checks of the revision-2 levers that change the views and the pairing (contract
revision 2, sections 3.1 and 3.6) on the CPU with the stub backbone of conftest.py: augmentation and
crop_scale_min reach every augmented view, drop_path, the plate penalties' eligibility, intensity_norm's
errors and view_pairing = nn_other_image (table space, refresh, inert sub-options).
"""
import numpy as np
import pytest
import torch
from torchvision import transforms

import shape.tools.cell_state_analyzer.tool as tool_mod

ad = pytest.importorskip("anndata")


@pytest.fixture
def datasets(monkeypatch):
    made = []
    real_init = tool_mod.CellCropDataset.__init__

    def record(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(tool_mod.CellCropDataset, "__init__", record)
    return made


def _is_eval(transform):
    return isinstance(transform, transforms.Compose) and [type(t) for t in transform.transforms] == [transforms.Resize]


@pytest.mark.parametrize("augmentation", ["strong", "light"])
def test_augmentation_and_crop_scale_reach_every_augmented_view(run_call, datasets, train_spy, augmentation):
    run_call(split_key="split", augmentation=augmentation, crop_scale_min=0.6, compute_alignment=True)
    train_dataset = train_spy[0]["loader"].dataset
    loss_dataset = train_spy[0]["kwargs"]["validation"]["loss_loader"].dataset
    alignment_dataset = datasets[-1]
    for dataset in (train_dataset, loss_dataset, alignment_dataset):
        if augmentation == "light":
            assert isinstance(dataset.transform, tool_mod.SSL116View) and dataset.transform.scale_min == 0.6
            v1, v2 = dataset[0][:2]
            assert v1.shape == (6, 224, 224) and 0.0 <= float(v1.min()) and float(v1.max()) <= 1.0
        else:
            assert dataset.transform.transforms[0].scale == (0.6, 1.0)
    assert _is_eval(datasets[0].transform)                                    # the embedding transform
    assert _is_eval(train_spy[0]["kwargs"]["validation"]["loader"].dataset.transform)


def test_drop_path_zero_runs_on_blocks_without_the_attribute_and_plate_weights_stay_gated(
        run_call, crops, stub_backbone, train_spy):
    """drop_path 0 sets and checks nothing (the stub's blocks lack sample_drop_ratio here); one plate
    is ineligible metadata, so the plate weights are forced to 0 and the reason recorded."""
    stub_backbone["options"]["with_ratio"] = False
    paths, meta, _ = crops
    one_plate = [i for i, m in enumerate(meta) if m["plate"] == "P1"]
    out = run_call(paths=[paths[i] for i in one_plate], meta=[meta[i] for i in one_plate], drop_path=0.0,
                   plate_mean_weight=1.0, plate_adversarial_weight=0.5)
    route = out["training_config"]["plate_nuisance_route"]
    assert route["eligible"] is False and route["rule"]
    assert route["plate_mean_weight"] == 0.0 and route["plate_adversarial_weight"] == 0.0
    assert train_spy[0]["kwargs"]["plate_mean_weight"] == 0.0
    assert train_spy[0]["kwargs"]["plate_adversarial_weight"] == 0.0
    with pytest.raises(ValueError, match="sample_drop_ratio"):
        run_call("raises", drop_path=0.1)


def test_drop_path_reaches_the_blocks(run_call, train_spy, monkeypatch):
    ratios = []
    real = tool_mod.set_drop_path
    monkeypatch.setattr(tool_mod, "set_drop_path",
                        lambda backbone, rate: real(backbone, rate) or ratios.extend(
                            b.sample_drop_ratio for b in backbone.blocks))
    run_call(drop_path=0.3)
    assert ratios == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_intensity_norm_errors(run_call, crops):
    _, meta, _ = crops
    with pytest.raises(ValueError, match="off.*per_plate"):
        run_call(intensity_norm="per_site")
    missing = [dict(m) for m in meta]
    missing[3]["plate"] = float("nan")
    with pytest.raises(ValueError, match="'plate'"):
        run_call(meta=missing, intensity_norm="per_plate")


def test_a_lever_needing_a_missing_column_names_it(run_call, crops):
    _, meta, _ = crops
    for kwargs, column in (({"view_pairing": "nn_other_image"}, "site"), ({"contrastive_mask": "same_well"}, "well"),
                           ({"batch_sampling": "per_experiment"}, "experiment")):
        missing = [{k: v for k, v in m.items() if k != column} for m in meta]
        with pytest.raises(ValueError, match=f"'{column}'"):
            run_call(meta=missing, **kwargs)


@pytest.fixture
def tables(monkeypatch):
    """Every neighbour_table build: {"train": bool (the training crops' sites W2 / W3), "table"}."""
    built = []
    real = tool_mod.neighbour_table

    def record(vectors, groups, k, **kwargs):
        table = real(vectors, groups, k, **kwargs)
        built.append({"train": any("W2" in str(g) for g in groups), "table": table, "k": k,
                      "mutual": kwargs.get("mutual")})
        return table

    monkeypatch.setattr(tool_mod, "neighbour_table", record)
    return built


@pytest.fixture
def extract_norms(cpu_tool, monkeypatch):
    """The backbone's final norm weight at every _extract_features call (which state embedded)."""
    seen = []
    real = cpu_tool._extract_features

    def record(model, loader, use_projector=False):
        seen.append(model.backbone.norm.weight.detach().clone())
        return real(model, loader, use_projector=use_projector)

    monkeypatch.setattr(cpu_tool, "_extract_features", record)
    return seen


def test_nn_current_resumed_refreshes_and_pairs_other_sites(run_call, crops, tables, train_spy, extract_norms):
    _, meta, _ = crops
    parent = run_call("parent", split_key="split", learning_rate=1e-3)
    resumed = torch.load(parent["resume_best"], weights_only=True)["model"]["backbone.norm.weight"]
    tables.clear()
    extract_norms.clear()
    out = run_call("child", split_key="split", view_pairing="nn_other_image", nn_space="current", nn_k=3,
                   nn_refresh_epochs=0.5, batch_size=4, init_checkpoint=parent["resume_best"])
    assert out["training_config"]["nn_refresh_steps"] == 2                   # 4 steps per epoch
    assert [t["train"] for t in tables] == [True, False, True, True]          # call start, then 2 refreshes
    assert torch.equal(extract_norms[0], resumed)                            # the model as loaded
    units = train_spy[-1]["kwargs"]["units"]
    sites = units.columns["site"]
    table = tables[0]["table"]
    assert table.shape == (16, 3)
    assert all(sites[j] != sites[i] for i in range(len(table)) for j in table[i] if j >= 0)


def test_nn_frozen_uses_the_pretrained_state_and_never_refreshes(run_call, tables, extract_norms, stub_backbone):
    parent = run_call("parent", split_key="split", learning_rate=1e-3)
    tables.clear()
    extract_norms.clear()
    out = run_call("child", split_key="split", view_pairing="nn_other_image", nn_space="frozen", nn_mutual=True,
                   nn_refresh_epochs=0.5, batch_size=4, init_checkpoint=parent["resume_best"])
    pretrained = torch.load(stub_backbone["weights"], weights_only=True)["norm.weight"]
    assert torch.equal(extract_norms[0], pretrained) and torch.equal(extract_norms[1], pretrained)
    assert [t["train"] for t in tables] == [True, False] and all(t["mutual"] is True for t in tables)
    assert out["training_config"]["nn_refresh_steps"] == 0


def test_nn_refresh_zero_builds_once(run_call, tables):
    out = run_call(split_key="split", view_pairing="nn_other_image", nn_space="current", nn_refresh_epochs=0.0,
                   batch_size=4)
    assert [t["train"] for t in tables] == [True, False]
    assert out["training_config"]["nn_refresh_steps"] == 0


def test_nn_options_are_inert_under_other_pairings(run_call, tables):
    default = run_call("default")
    other = run_call("other", nn_k=2, nn_space="current", nn_mutual=True, nn_refresh_epochs=1.0)
    assert not tables
    assert np.array_equal(ad.read_h5ad(default["adata_path"]).X, ad.read_h5ad(other["adata_path"]).X)
    assert default["loss_history"] == other["loss_history"]
