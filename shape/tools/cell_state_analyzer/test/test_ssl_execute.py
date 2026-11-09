"""execute()-level wiring of the analyzer's SSL levers (contract revision 2, sections 3.1, 3.3, 3.4, 3.6) on
the CPU with the stub backbone of conftest.py: the split, the levers reaching `_train_model`, the result
keys, the deadline, the resume state (round trip, an OOM retry), embed_only with the stored intensity
reference, the loss curve in a worker thread, the trainable depth and the Gram anchor.
"""
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import shape.tools.cell_state_analyzer.tool as tool_mod

ad = pytest.importorskip("anndata")

TRAINING_KEYS = {"epochs_trained", "best_epoch", "best_by", "batch_size", "trainable_blocks", "train_seconds",
                 "start_step", "end_step", "steps_trained", "best_step"}


def _train_rows(meta):
    return [i for i, m in enumerate(meta) if m.get("split") == "train"]


def test_split_levers_and_result_keys(run_call, crops, train_spy, monkeypatch):
    paths, meta, _ = crops
    fingerprinted = []
    real_fingerprint = tool_mod.embedding_fingerprint
    monkeypatch.setattr(tool_mod, "embedding_fingerprint",
                        lambda params, crop_paths=(): fingerprinted.append(dict(params)) or real_fingerprint(params, crop_paths))
    out = run_call(split_key="split", view_pairing="same_well", contrastive_mask="same_site",
                   batch_sampling="per_plate", trainable_pct=50)
    call = train_spy[0]
    dataset = call["loader"].dataset
    train = _train_rows(meta)
    assert dataset.image_paths == [paths[i] for i in train]                 # val crops never train
    assert isinstance(call["loader"].batch_sampler, tool_mod.UnitBatchSampler)
    assert dataset.view_pairing == "same_well" and call["kwargs"]["units"] is dataset.units
    assert np.array_equal(call["kwargs"]["mask_units"], dataset.units.codes("site"))
    assert call["kwargs"]["save_state"] is True and call["kwargs"]["resume"] is None
    blocks = {n.split(".")[2] for n in call["trainable"] if n.startswith("backbone.blocks.")}
    assert blocks == {"2", "3"} and not any("patch_embed" in n for n in call["trainable"])
    validation = call["kwargs"]["validation"]
    assert validation["meta"] == [meta[i] for i in range(len(meta)) if meta[i]["split"] == "val"]
    assert validation["measurable"] is True

    assert out["split"] == {"n_train": 16, "n_val": 16, "n_val_wells": 4}
    assert set(out["training"]) == TRAINING_KEYS                              # no clip facts at clip 0
    assert out["training"]["trainable_blocks"] == 2 and out["training"]["best_by"] == "val_proxy"
    assert out["training"]["epochs_trained"] == 1 and out["training"]["best_epoch"] == 1
    training = out["training"]
    assert training["start_step"] == 0 and training["steps_trained"] == training["end_step"] > 0
    assert 0 < training["best_step"] <= training["end_step"]
    assert [set(v) for v in out["val_history"]] == [{"epoch", "sibling_site_map", "eval_loss"}]
    assert out["seconds"] > 0 and out["intensity_norm_used"] == "off"
    assert os.path.exists(out["resume_best"]) and os.path.exists(out["resume_end"])
    assert os.path.dirname(out["adata_path"]).endswith(os.path.join("cell_state_analysis", "call"))
    adata = ad.read_h5ad(out["adata_path"])
    assert list(adata.obs["split"]) == [m["split"] for m in meta]
    assert list(adata.obs["trained"]) == [m["split"] == "train" for m in meta]
    assert {"weight_decay", "gram_weight", "augmentation", "nn_refresh_epochs"} <= set(fingerprinted[0])
    assert set(tool_mod._EMBEDDING_PARAMS) <= set(fingerprinted[0]) | {"backbone_name", "proj_dim", "batch_size"}


def test_a_call_without_a_split_keeps_todays_shape(run_call, crops, train_spy):
    out = run_call()
    assert len(train_spy[0]["loader"].dataset) == len(crops[0])
    assert train_spy[0]["kwargs"]["validation"] is None and train_spy[0]["kwargs"]["save_state"] is False
    assert out["split"] is None and out["val_history"] == []
    assert out["resume_best"] is None and out["resume_end"] is None
    assert out["training"]["best_by"] == "train_loss"
    assert "split" not in ad.read_h5ad(out["adata_path"]).obs


def test_a_zero_shot_call_with_a_split_reports_the_val_proxy_as_epoch_0(cpu_tool, run_call, crops, train_spy):
    """The untrained call's val_history: one epoch-0 entry, the proxy only, equal to the training-time evaluator
    (`_evaluate_val`) on the pretrained model and to the proxy of the val rows of the delivered embedding."""
    from torch.utils.data import DataLoader

    paths, meta, _ = crops
    out = run_call(split_key="split", adaptation_depth="zero_shot")
    assert not train_spy and out["mode"] == "zero-shot"
    (entry,) = out["val_history"]
    assert set(entry) == {"epoch", "sibling_site_map"} and entry["epoch"] == 0
    rows = [i for i, m in enumerate(meta) if m["split"] == "val"]
    val_meta = [meta[i] for i in rows]
    x = ad.read_h5ad(out["adata_path"]).X
    expected = tool_mod.quant.sibling_site_map(x[rows], val_meta, k=2, draws=20,
                                               seed=tool_mod.quant.SELECTION_SEED)["value"]
    assert expected is not None and entry["sibling_site_map"] == expected
    model = tool_mod.DinoV3Projector(input_channels=6)
    val = tool_mod.CellCropDataset([paths[i] for i in rows], transform=cpu_tool._get_eval_transform())
    validation = {"loader": DataLoader(val, batch_size=out["batch_config"]["batch_size"]), "meta": val_meta,
                  "use_projector": False, "loss_loader": [], "loss_units": None, "mask_units": None}
    trained_way = cpu_tool._evaluate_val(model, validation, 0.1, False)
    assert entry["sibling_site_map"] == pytest.approx(trained_way["sibling_site_map"], abs=1e-9)
    assert run_call("no_split", adaptation_depth="zero_shot")["val_history"] == []


def test_val_fraction_assigns_val_wells(run_call, crops, train_spy):
    _, meta, _ = crops
    no_split = [{k: v for k, v in m.items() if k != "split"} for m in meta]
    out = run_call(meta=no_split, val_fraction=0.1)
    assert out["split"]["n_val_wells"] == 4                                  # MIN_VAL_WELLS per plate
    assert len(train_spy[0]["loader"].dataset) == out["split"]["n_train"] == 16


def test_split_key_rejects_other_values(run_call, crops):
    meta = [dict(m, split="heldout") for m in crops[1]]
    with pytest.raises(ValueError, match="split_key"):
        run_call(meta=meta, split_key="split")


def test_max_minutes_is_a_deadline_after_one_step(run_call, monkeypatch):
    steps = []

    class CountingScaler(tool_mod.GradScaler):
        def step(self, optimizer, *args, **kwargs):
            steps.append(1)
            return super().step(optimizer, *args, **kwargs)

    monkeypatch.setattr(tool_mod, "GradScaler", CountingScaler)
    out = run_call(max_minutes=0, max_epochs=3, batch_size=4)
    assert len(steps) == 1
    assert out["training_config"]["stop_reason"] == "max_minutes_reached"
    assert out["training"]["epochs_trained"] == 1 and os.path.exists(out["adata_path"])


def test_resume_round_trip_and_an_oom_retry(cpu_tool, run_call, train_spy):
    first = run_call("first", split_key="split", trainable_pct=50, learning_rate=1e-3)
    state = torch.load(first["resume_best"], map_location="cpu", weights_only=True)
    assert set(tool_mod._RESUME_KEYS) <= set(state) and state["intensity_reference"] is None
    best = torch.load(os.path.join(os.path.dirname(first["resume_best"]), "best_model.pth"), weights_only=True)
    assert all(torch.equal(state["model"][k], v) for k, v in best.items())
    first_trainable = train_spy[0]["trainable"]
    assert sorted(state["optimizer"]) == sorted(first_trainable)              # moments by name
    assert all({"exp_avg", "exp_avg_sq"} <= set(m) for m in state["optimizer"].values())
    assert state["counters"] == {"epoch": 1, "step": 1}
    assert {"torch", "numpy", "python", "loader", "units"} <= set(state["rng"])
    assert state["config"]["trainable_pct"] == 50 and state["config"]["seed"] == 0

    def oom_once(n, model):
        if n == 2:                                    # the first attempt of the resumed call
            with torch.no_grad():
                next(model.parameters()).add_(1.0)    # a partial step, then CUDA runs out of memory
            cpu_tool.device = SimpleNamespace(type="cuda")
            raise RuntimeError("CUDA out of memory")
        cpu_tool.device = torch.device("cpu")

    train_spy.spy.before = oom_once
    second = run_call("second", split_key="split", trainable_pct=100, init_checkpoint=first["resume_best"],
                      warmup_fraction=0.5)
    retry = train_spy[2]
    assert all(torch.equal(retry["state"][k], v) for k, v in state["model"].items())   # resumed, not pretrained
    assert len(retry["loader"]) == 2                                          # the halved batch
    config = second["training_config"]
    assert config["warmup_steps"] == int(0.5 * 1 * 2)                         # from the halved loader
    assert config["resume"]["moments_kept"] == len(first_trainable)
    assert config["resume"]["moments_new"] == len(retry["trainable"]) - len(first_trainable)
    assert config["resume"]["random_streams"] == "continued"
    assert [a["status"] for a in second["batch_config"]["attempts"]] == ["cuda_oom", "success"]
    end = torch.load(second["resume_end"], map_location="cpu", weights_only=True)
    assert end["counters"] == {"epoch": 2, "step": 3}
    assert sorted(end["optimizer"]) == sorted(retry["trainable"])


def test_embed_only_uses_the_stored_state_and_reference(cpu_tool, run_call, crops, crop_factory, monkeypatch,
                                                         stub_backbone):
    paths, meta, images = crops
    datasets = []
    real_init = tool_mod.CellCropDataset.__init__

    def record(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        datasets.append(self)

    monkeypatch.setattr(tool_mod.CellCropDataset, "__init__", record)
    trained = run_call("trained", split_key="split", intensity_norm="per_plate", learning_rate=1e-3)
    train = _train_rows(meta)
    state = torch.load(trained["resume_best"], map_location="cpu", weights_only=True)
    expected = tool_mod.plate_reference([paths[i] for i in train], [meta[i]["plate"] for i in train])
    assert state["intensity_reference"] == expected
    assert torch.load(trained["resume_end"], weights_only=True)["intensity_reference"] == expected
    pretrained = torch.load(stub_backbone["weights"], weights_only=True)
    assert not torch.equal(state["model"]["backbone.norm.weight"], pretrained["norm.weight"])
    training_eval = datasets[0]
    assert len(training_eval) == len(paths) and training_eval.affine is not None

    held_images, held_meta = crop_factory(plates=("P3",), wells=2, seed=5, split=False)
    images.update(held_images)
    for row in held_meta:
        row["split"] = "heldout"                                              # no train / val label
    embed_paths = [paths[i] for i in train] + list(held_images)
    embed_meta = [meta[i] for i in train] + held_meta
    start = len(datasets)
    out = run_call("embed", paths=embed_paths, meta=embed_meta, init_checkpoint=trained["resume_best"],
                   embed_only=True, split_key="split")
    assert out["intensity_norm_used"] == "per_plate"
    assert out["training_config"]["stop_reason"] == "embed_only" and out["epochs_trained"] == 0
    assert out["split"] is None and out["resume_best"] is None
    embed_eval = datasets[start]
    for j, i in enumerate(train):                                             # the training-time tensors
        assert torch.equal(embed_eval._tensor(j), training_eval._tensor(i))
    held = list(range(len(train), len(embed_paths)))
    own_scale, own_shift = tool_mod.norm_affine(list(held_images), ["P3"] * len(held), expected)
    assert np.array_equal(embed_eval.affine[0][held], own_scale)
    assert np.array_equal(embed_eval.affine[1][held], own_shift)
    x_trained = ad.read_h5ad(trained["adata_path"]).X
    x_embed = ad.read_h5ad(out["adata_path"]).X
    assert np.allclose(x_embed[:len(train)], x_trained[train], atol=1e-5)  # the checkpoint's weights
    assert "split" not in ad.read_h5ad(out["adata_path"]).obs


def test_embed_only_needs_a_checkpoint(run_call):
    with pytest.raises(ValueError, match="init_checkpoint"):
        run_call(embed_only=True)


def test_the_loss_curve_uses_pil_in_a_worker_thread(run_call, monkeypatch):
    drawn = []
    real = tool_mod._save_loss_curve_png
    monkeypatch.setattr(tool_mod, "_save_loss_curve_png", lambda history, path: drawn.append(path) or real(history, path))
    monkeypatch.setattr(tool_mod.plt, "subplots", lambda *a, **k: pytest.fail("pyplot in a worker thread"))
    result, errors = {}, []

    def work():
        try:
            result.update(run_call())
        except BaseException as error:             # noqa: BLE001 - re-raised below in the main thread
            errors.append(error)

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert not errors, errors
    assert len(drawn) == 1 and os.path.exists(drawn[0]) and drawn[0] in result["deliverables"]


def test_trainable_pct_100_trains_what_none_trains_and_50_leaves_patch_embed_out(run_call, train_spy, monkeypatch):
    groups = []
    real_groups = tool_mod.optimizer_groups
    monkeypatch.setattr(tool_mod, "optimizer_groups",
                        lambda *a, **k: groups.append(real_groups(*a, **k)) or groups[-1])
    run_call("none")
    out = run_call("full", trainable_pct=100, clip_grad_norm=1.0)
    run_call("half", trainable_pct=50)
    assert train_spy[0]["trainable"] == train_spy[1]["trainable"]
    assert any("patch_embed" in n for n in train_spy[1]["trainable"])
    assert not any("patch_embed" in n for group in groups[2] for n in group["names"])
    assert [g["kind"] for g in groups[2]] == ["body", "head"]
    assert out["training"]["grad_norm_median"] > 0 and 0 <= out["training"]["clipped_fraction"] <= 1
    assert "grad_norm_median" not in out and "clipped_fraction" not in out


def test_gram_anchor_is_the_pretrained_backbone_outside_the_model(cpu_tool, run_call, train_spy, monkeypatch,
                                                                   stub_backbone):
    estimates = []
    real_resolve = tool_mod.resolve_batch_size
    monkeypatch.setattr(tool_mod, "resolve_batch_size",
                        lambda **kw: estimates.append(kw["model_parameter_bytes"]) or real_resolve(**kw))
    first = run_call("first", split_key="split", learning_rate=1e-3)
    assert train_spy[0]["kwargs"]["gram_anchor"] is None                      # gram 0 builds nothing
    second = run_call("second", split_key="split", gram_weight=0.5, init_checkpoint=first["resume_best"])
    anchor = train_spy[1]["kwargs"]["gram_anchor"]
    pretrained = torch.load(stub_backbone["weights"], weights_only=True)
    assert all(torch.equal(anchor.state_dict()[k], v) for k, v in pretrained.items())
    resumed = torch.load(first["resume_best"], weights_only=True)["model"]
    assert not torch.equal(train_spy[1]["state"]["backbone.norm.weight"], pretrained["norm.weight"])
    assert torch.equal(train_spy[1]["state"]["backbone.norm.weight"], resumed["backbone.norm.weight"])
    assert not anchor.training and not any(p.requires_grad for p in anchor.parameters())
    model_bytes = estimates[0]
    anchor_bytes = sum(p.numel() * p.element_size() for p in anchor.parameters())
    assert estimates[1] == model_bytes + anchor_bytes // 3
    end = torch.load(second["resume_end"], weights_only=True)
    assert set(end["model"]) == set(resumed)                                  # no anchor in the state
    assert set(end) == set(tool_mod._RESUME_KEYS) | {"intensity_reference"}
    assert second["training"]["epochs_trained"] == 1


def test_a_split_call_without_memory_for_a_batch_fails_and_a_call_without_a_split_stays_zero_shot(
        run_call, monkeypatch):
    real_resolve = tool_mod.resolve_batch_size
    monkeypatch.setattr(tool_mod, "resolve_batch_size", lambda **kw: {**real_resolve(**kw), "batch_size": 1})
    failed = run_call("split", split_key="split")
    assert set(failed) == {"error"} and "insufficient memory" in failed["error"]
    out = run_call("no_split")
    assert out["training_config"]["stop_reason"] == "insufficient_memory_for_contrastive_batch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memory_cached_by_an_earlier_call_does_not_shrink_the_next_batch(cpu_tool, run_call):
    cpu_tool.device = torch.device("cuda")
    first = run_call("first", split_key="split")
    free, _ = torch.cuda.mem_get_info()
    held = torch.empty(int(0.9 * free), dtype=torch.uint8, device="cuda")
    del held                                          # freed, but kept in the caching allocator's pool
    assert torch.cuda.memory_reserved() - torch.cuda.memory_allocated() >= int(0.9 * free)
    second = run_call("second", split_key="split")
    assert "error" not in second, second.get("error")
    assert first["training"]["batch_size"] >= 2
    assert second["training"]["batch_size"] == first["training"]["batch_size"]


def test_a_checkpoint_write_survives_a_lustre_eviction(tmp_path, monkeypatch):
    sleeps, real_replace = [], os.replace
    monkeypatch.setattr(tool_mod.time, "sleep", sleeps.append)
    failures = [OSError(108, "Cannot send after transport endpoint shutdown")]

    def flaky(src, dst):
        if failures:
            raise failures.pop()
        real_replace(src, dst)

    monkeypatch.setattr(tool_mod.os, "replace", flaky)
    path = str(tmp_path / "state.pt")
    tool_mod.save_retrying({"w": torch.arange(3)}, path)
    assert torch.equal(torch.load(path, weights_only=True)["w"], torch.arange(3)) and sleeps == [5]
    failures.append(PermissionError(13, "denied"))                          # not transient: raised at once
    with pytest.raises(PermissionError):
        tool_mod.save_retrying({"w": torch.arange(3)}, path)
    assert sleeps == [5]
