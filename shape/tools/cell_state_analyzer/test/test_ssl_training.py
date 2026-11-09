"""Unit tests of the analyzer's SSL training levers (contract revision 2, section 3.6), on the CPU with the
stub backbone of conftest.py: the optimizer step at the defaults, gradient clipping, param groups and
warmups, the Gram and KoLeo terms, intensity_norm, batches, both augmentation pipelines, drop_path, the
nn_other_image table and the fingerprint's coverage. The execute()-level wiring is in test_ssl_execute.py.
"""
import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import shape.tools.cell_state_analyzer.tool as tool_mod
from shape.tools.cell_state_analyzer import quant


class SpyScaler(tool_mod.GradScaler):
    """GradScaler that records, at every optimizer step, each group's lr and kind, whether the body
    (non-head) parameters have a gradient, and the global gradient norm the optimizer sees."""
    records = []

    def step(self, optimizer, *args, **kwargs):
        grads = [p.grad for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        SpyScaler.records.append({
            "lrs": [g["lr"] for g in optimizer.param_groups],
            "kinds": [g.get("kind") for g in optimizer.param_groups],
            "body_grads": any(p.grad is not None for g in optimizer.param_groups if g.get("kind") != "head"
                              for p in g["params"]),
            "norm": float(torch.norm(torch.stack([g.detach().norm() for g in grads]))),
        })
        return super().step(optimizer, *args, **kwargs)


@pytest.fixture
def spy_scaler(monkeypatch):
    SpyScaler.records = []
    monkeypatch.setattr(tool_mod, "GradScaler", SpyScaler)
    return SpyScaler.records


@pytest.fixture
def model(stub_backbone):
    torch.manual_seed(0)
    return tool_mod.DinoV3Projector(input_channels=6)


def _loader(tool, paths, batch, transform=None, **dataset_kwargs):
    dataset = tool_mod.CellCropDataset(paths, transform=transform or tool._get_eval_transform(),
                                       plate_ids=[""] * len(paths), **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(0))


def _train(tool, model, loader, tmp_path, max_epochs=1, **kwargs):
    return tool._train_model(model, loader, max_epochs, None, kwargs.pop("learning_rate", 3e-5),
                             str(tmp_path), [], **kwargs)


def _todays_step(model, loader, learning_rate=3e-5, weight_decay=1e-4):
    """One step of the analyzer before revision 2 (fabb1e25): one AdamW group over the trainable
    parameters, InfoNCE plus the zero-weighted plate terms, no warmup at this budget, no clip."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=weight_decay)
    model.train()
    v1, v2, _, _, identity = next(iter(loader))
    optimizer.zero_grad()
    z1, z2 = model(v1), model(v2)
    contrastive = tool_mod.contrastive_loss(z1, z2, temperature=0.1, source_ids=None)
    plate_mean = tool_mod.plate_mean_alignment_loss(z1, z2, list(identity["plate_id"]))
    loss = contrastive + 0.0 * plate_mean + 0.0 * contrastive.new_zeros(())
    loss.backward()
    optimizer.step()


# --------------------------------------------------------------------- the step at the defaults

def test_default_step_is_bitwise_todays_step(cpu_tool, model, crops, tmp_path, monkeypatch):
    """clip_grad_norm 0, the param groups at head_learning_rate None / patch_embed_lr_mult 1, gram_weight 0,
    koleo_weight 0 and the plate weights 0 leave one step bitwise equal to today's; neither regulariser
    is computed."""
    called = []
    monkeypatch.setattr(tool_mod, "koleo_loss", lambda *a: called.append("koleo"))
    monkeypatch.setattr(tool_mod, "gram_loss", lambda *a: called.append("gram"))
    paths = crops[0][:8]
    reference = copy.deepcopy(model)
    initial = copy.deepcopy(model)
    loader = _loader(cpu_tool, paths, 8)
    _train(cpu_tool, model, loader, tmp_path, head_learning_rate=None, patch_embed_lr_mult=1.0,
           clip_grad_norm=0.0, gram_weight=0.0, koleo_weight=0.0, plate_mean_weight=0.0,
           plate_adversarial_weight=0.0, warmup_fraction=None, head_warmup_epochs=0.0)
    _todays_step(reference, loader)
    assert not called
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert torch.equal(p, q), name
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), initial.parameters()))   # it stepped


def test_training_reports_actual_steps_and_selected_checkpoint_step(cpu_tool, model, crops, tmp_path):
    loader = _loader(cpu_tool, crops[0][:8], 4)
    _, _, _, config = _train(cpu_tool, model, loader, tmp_path, max_epochs=2)
    assert config["start_step"] == 0
    assert config["steps_trained"] == config["end_step"] == 2 * len(loader)
    assert 0 < config["best_step"] <= config["end_step"]


def test_clip_bounds_the_optimizer_seen_norm(cpu_tool, model, crops, tmp_path, spy_scaler):
    clip = 1e-6
    _, _, _, config = _train(cpu_tool, model, _loader(cpu_tool, crops[0][:8], 4), tmp_path, clip_grad_norm=clip)
    assert len(spy_scaler) == 2
    assert all(record["norm"] <= clip * (1 + 1e-4) for record in spy_scaler)
    assert config["clipped_fraction"] == 1.0
    assert config["grad_norm_median"] > 100 * clip


def test_no_clip_facts_at_zero(cpu_tool, model, crops, tmp_path):
    _, _, _, config = _train(cpu_tool, model, _loader(cpu_tool, crops[0][:4], 4), tmp_path)
    assert "grad_norm_median" not in config and "clipped_fraction" not in config


# --------------------------------------------------------------------- param groups and warmups

def test_optimizer_groups_order_lrs_and_names(model):
    groups = tool_mod.optimizer_groups(model, 1e-4, head_learning_rate=1e-3, patch_embed_lr_mult=0.5)
    assert [g["kind"] for g in groups] == ["body", "patch_embed", "head"]
    assert [g["lr"] for g in groups] == [1e-4, 5e-5, 1e-3]
    names = [name for g in groups for name in g["names"]]
    assert sorted(names) == sorted(n for n, p in model.named_parameters() if p.requires_grad)
    assert all(n.startswith("backbone.patch_embed.") for n in groups[1]["names"])
    assert all(n.startswith("projector.") for n in groups[2]["names"])
    assert any(n.startswith("channel_adapter.") for n in groups[0]["names"])
    defaults = tool_mod.optimizer_groups(model, 3e-5)
    assert {g["lr"] for g in defaults} == {3e-5}


def test_per_group_warmup_from_each_base(cpu_tool, model, crops, tmp_path, spy_scaler):
    loader = _loader(cpu_tool, crops[0][:16], 8)                     # 2 steps per epoch, 4 in all
    _, _, _, config = _train(cpu_tool, model, loader, tmp_path, max_epochs=2, learning_rate=1e-4,
                             head_learning_rate=1e-3, patch_embed_lr_mult=0.5, warmup_fraction=0.5)
    assert config["warmup_steps"] == 2
    assert spy_scaler[0]["kinds"] == ["body", "patch_embed", "head"]
    assert spy_scaler[0]["lrs"] == pytest.approx([5e-5, 2.5e-5, 5e-4])
    assert all(r["lrs"] == pytest.approx([1e-4, 5e-5, 1e-3]) for r in spy_scaler[1:])


def test_warmup_none_is_todays_rule(cpu_tool, model, crops, tmp_path):
    paths = crops[0]                                                  # 32 steps of one crop
    _, _, _, fresh = _train(cpu_tool, copy.deepcopy(model), _loader(cpu_tool, paths, 1), tmp_path)
    assert fresh["warmup_steps"] == int(0.05 * 1 * 32)
    resume = {"optimizer": {}, "scaler": {}, "counters": {"step": 10, "epoch": 1}, "config": {"seed": -1},
              "rng": None}
    _, _, _, resumed = _train(cpu_tool, model, _loader(cpu_tool, paths[:4], 1), tmp_path, resume=resume)
    assert resumed["warmup_steps"] == 0


def test_resumed_warmup_counts_local_steps(cpu_tool, model, crops, tmp_path, spy_scaler):
    """A resumed call (parent counter 1000) with warmup_fraction 0.1 warms up over its own first steps."""
    resume = {"optimizer": {}, "scaler": {}, "counters": {"step": 1000, "epoch": 7}, "config": {"seed": -1},
              "rng": None}
    _, _, _, config = _train(cpu_tool, model, _loader(cpu_tool, crops[0][:30], 1), tmp_path,
                             warmup_fraction=0.1, resume=resume)
    assert config["warmup_steps"] == 3 and config["resume"]["step"] == 1000
    lrs = [r["lrs"][0] for r in spy_scaler]
    assert lrs[:4] == pytest.approx([1e-5, 2e-5, 3e-5, 3e-5])
    assert all(lr == pytest.approx(3e-5) for lr in lrs[3:])


def test_head_warm_in_freezes_the_body_then_restores_it(cpu_tool, model, crops, tmp_path, spy_scaler):
    body = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("projector.")}
    head = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("projector.")}
    _train(cpu_tool, model, _loader(cpu_tool, crops[0][:8], 4), tmp_path, head_warmup_epochs=1.0)
    assert [r["body_grads"] for r in spy_scaler] == [False, False]
    params = dict(model.named_parameters())
    assert all(torch.equal(params[n], v) and params[n].grad is None and params[n].requires_grad
               for n, v in body.items())
    assert any(not torch.equal(params[n], v) for n, v in head.items())


def test_head_warm_in_delays_the_body_warmup(cpu_tool, model, crops, tmp_path, spy_scaler):
    loader = _loader(cpu_tool, crops[0][:16], 8)                     # 2 steps per epoch
    _, _, _, config = _train(cpu_tool, model, loader, tmp_path, max_epochs=2, learning_rate=1e-4,
                             head_learning_rate=1e-3, warmup_fraction=0.5, head_warmup_epochs=0.5)
    assert (config["head_warmup_steps"], config["warmup_steps"]) == (1, 2)
    assert [r["body_grads"] for r in spy_scaler] == [False, True, True, True]
    assert spy_scaler[0]["lrs"][-1] == pytest.approx(5e-4)            # the head from step 0
    assert spy_scaler[1]["lrs"] == pytest.approx([5e-5, 5e-5, 1e-3])  # the body from the warm-in's end
    assert spy_scaler[2]["lrs"] == pytest.approx([1e-4, 1e-4, 1e-3])


# --------------------------------------------------------------------- gram and koleo

def test_gram_loss_cases():
    x = torch.randn(2, 5, 8)
    assert float(tool_mod.gram_loss(x, x)) == pytest.approx(0.0, abs=1e-7)
    scale = torch.rand(2, 5, 1) + 0.5                                 # a positive rescale per token
    target = torch.randn(2, 5, 8)
    assert float(tool_mod.gram_loss(x * scale, target)) == pytest.approx(float(tool_mod.gram_loss(x, target)), rel=1e-5)
    student = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])                # Gram I
    anchor = torch.tensor([[[3.0, 0.0], [1.0, 0.0]]])                 # Gram all ones
    assert float(tool_mod.gram_loss(student, anchor)) == pytest.approx(0.5)


def test_koleo_loss_cases():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expected = -math.log(math.sqrt(2 - math.sqrt(2)) + 1e-8)          # every row's nearest is at 45 degrees
    assert float(tool_mod.koleo_loss(x)) == pytest.approx(expected, rel=1e-5)
    assert float(tool_mod.koleo_loss(x * torch.tensor([[2.0], [0.5], [3.0]]))) == pytest.approx(expected, rel=1e-5)
    assert float(tool_mod.koleo_loss(torch.randn(1, 4))) == 0.0


def test_forward_features_cls_equals_the_backbone_output(model):
    model.eval()
    x = torch.rand(3, 6, 224, 224)
    inputs = model.backbone_input(x)
    with torch.no_grad():
        assert torch.equal(model.backbone.forward_features(inputs)["x_norm_clstoken"], model.backbone(inputs))
        assert torch.equal(model.backbone(inputs), model.encode_backbone(x))


def test_batch_loss_reads_the_backbone_cls_and_keeps_z(cpu_tool, model):
    model.eval()
    v1, v2 = torch.rand(4, 6, 224, 224), torch.rand(4, 6, 224, 224)
    with torch.no_grad():
        for patches in (False, True):
            z1, z2, _, views = cpu_tool._batch_loss(model, v1, v2, list("abcd"), {}, 0.1, False, None,
                                                    patches=patches)
            assert torch.allclose(z1, model(v1), atol=1e-6) and torch.allclose(z2, model(v2), atol=1e-6)
            assert torch.allclose(views[0]["cls"], model.encode_backbone(v1), atol=1e-6)
            assert torch.allclose(views[1]["cls"], model.encode_backbone(v2), atol=1e-6)
        z1, _, _, _ = cpu_tool._batch_loss(model, v1, v2, list("abcd"), {}, 0.1, False, None)
        assert torch.equal(z1, model(v1))
    assert set(views[0]) == {"cls", "tokens", "input"}


def _validation(tool, crops):
    paths, meta, _ = crops
    rows = [i for i, m in enumerate(meta) if m["split"] == "val"]
    val = tool_mod.CellCropDataset([paths[i] for i in rows], transform=tool._get_eval_transform())
    loss = tool_mod.CellCropDataset([paths[i] for i in rows], transform=tool._get_augmentation_transform())
    return {"loader": DataLoader(val, batch_size=8), "meta": [meta[i] for i in rows], "use_projector": False,
            "loss_loader": DataLoader(loss, batch_size=8), "loss_units": None, "mask_units": None,
            "every": 1, "measurable": True}


def test_the_val_eval_loss_excludes_gram_and_koleo(cpu_tool, model, crops, tmp_path):
    """At lr 0 the weights stay put, so the logged val eval loss must not move with gram_weight or
    koleo_weight, while the training loss does."""
    loader = _loader(cpu_tool, crops[0][16:24], 8)
    plain = copy.deepcopy(model)
    anchor = copy.deepcopy(model.backbone).eval().requires_grad_(False)
    history_a, _, _, a = _train(cpu_tool, plain, loader, tmp_path, learning_rate=0.0,
                                validation=_validation(cpu_tool, crops))
    history_b, _, _, b = _train(cpu_tool, model, loader, tmp_path, learning_rate=0.0,
                                validation=_validation(cpu_tool, crops), gram_weight=1.0, gram_anchor=anchor,
                                koleo_weight=0.5)
    assert a["val_history"][0]["eval_loss"] == b["val_history"][0]["eval_loss"]
    assert a["val_history"][0]["sibling_site_map"] == b["val_history"][0]["sibling_site_map"]
    assert history_b[0] != history_a[0]


# --------------------------------------------------------------------- intensity_norm

def test_intensity_off_is_todays_tensor(crops):
    paths = crops[0][:3]
    dataset = tool_mod.CellCropDataset(paths, transform=None)
    for i, path in enumerate(paths):
        assert torch.equal(dataset[i][0], tool_mod._normalized_channel_tensor(tool_mod.io.imread(path), path))


def test_per_plate_affine_matches_the_global_statistics(crops):
    paths, meta, _ = crops
    plates = [m["plate"] for m in meta]
    reference = tool_mod.plate_reference(paths, plates)
    assert sorted(reference["plates"]) == ["P1", "P2"]
    affine = tool_mod.norm_affine(paths, plates, reference)
    dataset = tool_mod.CellCropDataset(paths, transform=None, affine=affine)
    tensors = torch.stack([dataset[i][0] for i in range(len(paths))])
    assert float(tensors.min()) >= 0.0 and float(tensors.max()) <= 1.0
    means = tensors.mean(dim=(2, 3)).double().numpy()
    raw = np.stack([tool_mod._normalized_channel_tensor(tool_mod.io.imread(p), p).mean(dim=(1, 2)).numpy()
                    for p in paths])
    for plate in ("P1", "P2"):
        rows = np.array(plates) == plate
        assert np.allclose(means[rows].mean(0), reference["global_mean"], atol=2e-3)
    assert not np.allclose(raw[np.array(plates) == "P1"].mean(0), raw[np.array(plates) == "P2"].mean(0), atol=0.05)


def test_the_partner_view_uses_its_own_plates_affine(crops):
    paths, meta, _ = crops
    plates = [m["plate"] for m in meta]
    affine = tool_mod.norm_affine(paths, plates, tool_mod.plate_reference(paths, plates))
    units = tool_mod.CropUnits(quant.identities(meta), seed=0)
    units.nn = np.full((len(paths), 1), -1, dtype=np.int64)
    units.nn[0, 0] = 20                                               # crop 0 (P1) pairs with crop 20 (P2)
    dataset = tool_mod.CellCropDataset(paths, transform=None, units=units, view_pairing="nn_other_image",
                                       affine=affine, plate_ids=[""] * len(paths))
    v1, v2, _, _, identity = dataset[0]
    assert identity["partner"] == 20 and plates[20] == "P2"
    assert torch.equal(v2, dataset._tensor(20)) and torch.equal(v1, dataset._tensor(0))
    assert not torch.equal(dataset._tensor(20), tool_mod._normalized_channel_tensor(
        tool_mod.io.imread(paths[20]), paths[20]))


# --------------------------------------------------------------------- batches

def test_resolve_batch_size_treats_the_request_as_a_ceiling(monkeypatch):
    import shape.utils.adaptive_training as adaptive

    mb = 1024 ** 2
    monkeypatch.setattr(adaptive, "_available_memory_bytes",
                        lambda device: (int(math.ceil(64 * 96 * mb / 0.84)) + mb, "test"))
    resolve = lambda requested: adaptive.resolve_batch_size(1000, mb, 0, torch.device("cuda"), requested)
    assert resolve(None)["batch_size"] == 64
    assert resolve(16)["batch_size"] == 16
    assert resolve(256)["batch_size"] == 64


def test_unit_batches_come_from_one_plate(crops):
    _, meta, _ = crops
    units = tool_mod.CropUnits(quant.identities(meta), seed=0)
    batches = list(tool_mod.UnitBatchSampler(units, 5, "per_plate"))
    assert len(batches) == len(tool_mod.UnitBatchSampler(units, 5, "per_plate"))
    assert all(2 <= len(b) <= 5 and len({meta[i]["plate"] for i in b}) == 1 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(len(meta)))


# --------------------------------------------------------------------- augmentation

def _ops(transform):
    return [(type(op).__name__, type(op.transforms[0]).__name__ if isinstance(op, transforms.RandomApply) else None)
            for op in transform.transforms]


def test_shape_augmentation_is_todays_compose(cpu_tool):
    t = cpu_tool._get_augmentation_transform()
    assert _ops(t) == [("RandomResizedCrop", None), ("RandomHorizontalFlip", None), ("RandomVerticalFlip", None),
                       ("RandomRotation", None), ("RandomApply", "ChannelIntensityJitter"),
                       ("RandomApply", "GaussianBlur"), ("RandomErasing", None)]
    crop, hflip, vflip, rotation, color, blur, erasing = t.transforms
    assert crop.size == (224, 224) and crop.scale == (0.3, 1.0)
    assert (hflip.p, vflip.p, rotation.degrees) == (0.5, 0.2, [-60.0, 60.0])
    assert color.p == 0.8 and color.transforms[0].strength == 0.2
    assert blur.p == 0.8 and blur.transforms[0].kernel_size == (23, 23) and blur.transforms[0].sigma == (0.1, 2.0)
    assert (erasing.p, erasing.scale, erasing.ratio) == (0.3, (0.02, 0.2), (0.3, 3.3))


def test_augment_color_false_drops_exactly_the_intensity_jitter(cpu_tool):
    with_color = _ops(cpu_tool._get_augmentation_transform(augment_color=True))
    without = _ops(cpu_tool._get_augmentation_transform(augment_color=False))
    assert [op for op in with_color if op != ("RandomApply", "ChannelIntensityJitter")] == without
    assert len(with_color) == len(without) + 1


def test_crop_scale_min_under_both_pipelines(cpu_tool):
    assert cpu_tool._get_augmentation_transform(crop_scale_min=0.6).transforms[0].scale == (0.6, 1.0)
    assert cpu_tool._get_augmentation_transform(augmentation="light", crop_scale_min=0.6).scale_min == 0.6
    for bad in (0.01, 1.5):
        with pytest.raises(ValueError, match="crop_scale_min"):
            cpu_tool._get_augmentation_transform(crop_scale_min=bad)


def test_the_old_augmentation_names_are_aliases(cpu_tool):
    assert isinstance(cpu_tool._get_augmentation_transform(augmentation="ssl116"), tool_mod.SSL116View)
    assert not isinstance(cpu_tool._get_augmentation_transform(augmentation="shape"), tool_mod.SSL116View)
    assert tool_mod.AUGMENTATION_ALIASES == {"shape": "strong", "ssl116": "light"}


def test_unknown_augmentation_lists_the_values(cpu_tool):
    with pytest.raises(ValueError, match="strong.*light"):
        cpu_tool._get_augmentation_transform(augmentation="dino")


def test_ssl116_view_shape_range_and_shared_geometry():
    torch.manual_seed(0)
    x = torch.rand(6, 24, 24)
    out = tool_mod.SSL116View(0.3)(x)
    assert out.shape == (6, 224, 224) and float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    same = torch.rand(1, 24, 24).repeat(6, 1, 1)
    geometry_only = tool_mod.SSL116View(0.3, jitter=0.0, blur_prob=0.0, noise_std=0.0)
    for _ in range(5):
        view = geometry_only(same)
        assert all(torch.equal(view[0], view[c]) for c in range(1, 6))


def test_ssl116_full_area_view_is_a_rotation_or_mirror_of_the_resized_crop():
    torch.manual_seed(1)
    x = torch.rand(6, 24, 24)
    resized = F.interpolate(x[None], size=(224, 224), mode="bilinear", align_corners=False)[0]
    candidates = [torch.rot90(image, k, dims=(1, 2)) for image in (resized, resized.flip(2)) for k in range(4)]
    view = tool_mod.SSL116View(1.0, ratio=(1.0, 1.0), jitter=0.0, blur_prob=0.0, noise_std=0.0)
    seen = set()
    for _ in range(40):
        out = view(x)
        matches = [i for i, c in enumerate(candidates) if torch.allclose(out, c, atol=1e-5)]
        assert matches
        seen.add(matches[0])
    assert len(seen) > 4                                              # the rotations and mirrors are drawn


def test_ssl116_blur_kernel_is_116s_kernel_scaled_by_two():
    for sigma in (0.3, 0.8, 1.5):
        r = torch.arange(-2, 3, dtype=torch.float32)
        k1 = torch.exp(-(r ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum()
        k116 = k1[:, None] * k1[None, :]
        ours = tool_mod._ssl116_blur_kernel(2 * sigma)
        assert ours.shape == (9, 9) and float(ours.sum()) == pytest.approx(1.0)
        even = ours[::2, ::2]
        assert torch.allclose(even / even.sum(), k116, atol=1e-6)


def test_ssl116_photometric_draws_differ_between_channels():
    torch.manual_seed(2)
    same = (0.3 + 0.4 * torch.rand(1, 224, 224)).repeat(6, 1, 1)
    out = tool_mod.SSL116View(0.3, blur_prob=0.0, noise_std=0.0).photometric(same)
    assert not any(torch.equal(out[0], out[c]) for c in range(1, 6))


def test_augment_color_is_inert_under_ssl116(cpu_tool):
    x = torch.rand(6, 24, 24)
    outputs = []
    for color in (True, False):
        torch.manual_seed(3)
        outputs.append(cpu_tool._get_augmentation_transform(augment_color=color, augmentation="light")(x))
    assert torch.equal(*outputs)


# --------------------------------------------------------------------- drop_path

def test_drop_path_ratios_and_checks(stub_vit):
    backbone = stub_vit(depth=12)
    tool_mod.set_drop_path(backbone, 0.1)
    assert [b.sample_drop_ratio for b in backbone.blocks] == pytest.approx([0.1 * i / 11 for i in range(12)])
    untouched = stub_vit(depth=3)
    tool_mod.set_drop_path(untouched, 0.0)
    assert all(b.sample_drop_ratio == 0.0 for b in untouched.blocks)
    bare = stub_vit(depth=3, with_ratio=False)
    tool_mod.set_drop_path(bare, 0.0)                                 # nothing set or checked at 0
    assert not any(hasattr(b, "sample_drop_ratio") for b in bare.blocks)
    with pytest.raises(ValueError, match="sample_drop_ratio"):
        tool_mod.set_drop_path(bare, 0.1)


def _dinov3_block():
    """dinov3's SelfAttentionBlock from the torch hub cache with a zero MLP branch (only the attention branch
    moves a row), or None when the dinov3 code is not cached."""
    import os
    import sys

    repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov3_main")
    if not os.path.isdir(repo):
        return None
    sys.path.insert(0, repo)
    try:
        from dinov3.layers.block import SelfAttentionBlock
    except ImportError:
        return None
    finally:
        sys.path.remove(repo)
    block = SelfAttentionBlock(dim=8, num_heads=2)
    torch.nn.init.zeros_(block.mlp.fc2.weight)
    torch.nn.init.zeros_(block.mlp.fc2.bias)
    return block


def test_drop_path_drops_a_subset_of_every_batch_as_dinov3_blocks_do(stub_vit):
    """The CARD's drop_path mechanism: each block's residual branch skips b - max(int(b (1 - p)), 1) of a batch's b
    crops, at least one at any p > 0 (block 1 of 12 at drop_path 0.1 drops 1 of 8, 16 or 32 crops), for the
    stub's blocks and, when its code is cached, dinov3's own block."""
    blocks = [("stub", stub_vit(depth=1).blocks[0])]
    real = _dinov3_block()
    if real is not None:
        blocks.append(("dinov3", real))
    torch.manual_seed(0)
    for name, block in blocks:
        block.train()
        width = 8 if name == "dinov3" else 768
        for p, b in ((0.1 / 11, 8), (0.1 / 11, 16), (0.1 / 11, 32), (0.1, 8), (0.1, 32)):
            block.sample_drop_ratio = p
            x = torch.randn(b, 5, width)
            with torch.no_grad():
                y = block(x)
            dropped = sum(torch.equal(y[i], x[i]) for i in range(b))
            assert dropped == b - max(int(b * (1 - p)), 1) >= 1, (name, p, b)


def test_drop_path_is_off_in_eval_passes(stub_vit):
    backbone = stub_vit(depth=4)
    tool_mod.set_drop_path(backbone, 0.5)
    backbone.eval()
    x = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.equal(backbone(x), backbone(x))


# --------------------------------------------------------------------- nn_other_image

def test_neighbour_table_excludes_the_own_site_and_ranks_by_cosine():
    angles = np.array([0.0, 0.1, 0.5, 0.55, 1.4, 1.5])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    sites = np.array(["a", "a", "b", "b", "c", "c"])
    table = tool_mod.neighbour_table(vectors, sites, 2)
    assert table.shape == (6, 2)
    assert all(sites[j] != sites[i] for i in range(6) for j in table[i])
    assert table[0].tolist() == [2, 3] and table[5].tolist() == [3, 2]
    assert table[2].tolist() == [1, 0]


def test_neighbour_table_mutual_and_self_pairing():
    angles = np.array([0.0, 0.3, 0.35, 1.5])
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    sites = np.array(["a", "b", "c", "d"])
    table = tool_mod.neighbour_table(vectors, sites, 1, mutual=True)
    # 1 <-> 2 are mutual; 0's nearest is 1 (not mutual), 3's is 2 (not mutual)
    assert table[:, 0].tolist() == [-1, 2, 1, -1]
    units = tool_mod.CropUnits([{"plate": None, "experiment": None, "well": None, "site": s} for s in sites], 0)
    units.nn = table
    assert units.partners(np.arange(4), "nn_other_image").tolist() == [0, 2, 1, 3]


def test_partner_draws_continue_after_a_resume(crops):
    _, meta, _ = crops
    ids = quant.identities(meta)
    table = np.tile(np.arange(5, dtype=np.int64), (len(meta), 1))
    rows = np.arange(len(meta))
    first = tool_mod.CropUnits(ids, seed=0)
    first.nn = table
    first.partners(rows, "nn_other_image")
    state = tool_mod._rng_state(None, first)
    expected = [first.partners(rows, "nn_other_image").tolist(), first.partners(rows, "same_well").tolist()]
    second = tool_mod.CropUnits(ids, seed=99)
    second.nn = table
    tool_mod._restore_rng(state, None, second)
    assert [second.partners(rows, "nn_other_image").tolist(), second.partners(rows, "same_well").tolist()] == expected


# --------------------------------------------------------------------- fingerprint

REVISION_2_KWARGS = ("view_pairing", "contrastive_mask", "batch_sampling", "trainable_pct", "init_checkpoint",
                     "split_key", "val_fraction", "output_subdir", "max_minutes", "eval_every_epochs", "embed_only",
                     "head_learning_rate", "patch_embed_lr_mult", "clip_grad_norm", "warmup_fraction",
                     "head_warmup_epochs", "gram_weight", "koleo_weight", "intensity_norm", "augmentation",
                     "crop_scale_min", "drop_path", "nn_k", "nn_space", "nn_mutual", "nn_refresh_epochs",
                     "weight_decay")


@pytest.mark.parametrize("name", REVISION_2_KWARGS)
def test_every_new_kwarg_enters_the_fingerprint(name):
    assert name in tool_mod._EMBEDDING_PARAMS
    assert tool_mod.embedding_fingerprint({name: "a"}, ["x.png"]) != tool_mod.embedding_fingerprint({name: "b"}, ["x.png"])
