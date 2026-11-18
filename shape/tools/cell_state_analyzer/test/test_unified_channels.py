import numpy as np
import pytest
import torch

from shape.tools.cell_state_analyzer.tool import (
    ChannelAdapter,
    _channel_first,
    contrastive_loss,
    infer_channel_spec,
    plate_mean_alignment_loss,
    resolve_plate_nuisance,
)


def test_channel_layout_detection_supports_hw_chw_and_hwc():
    assert _channel_first(np.zeros((16, 16), dtype=np.uint8))[0].shape == (1, 16, 16)
    assert _channel_first(np.zeros((6, 16, 16), dtype=np.uint8))[0].shape == (6, 16, 16)
    assert _channel_first(np.zeros((16, 16, 6), dtype=np.uint8))[0].shape == (6, 16, 16)


def test_channel_layout_detection_fails_closed_when_ambiguous():
    with pytest.raises(ValueError, match="Ambiguous channel layout"):
        _channel_first(np.zeros((5, 3, 8), dtype=np.uint8))
    assert _channel_first(np.zeros((3, 3, 3), dtype=np.uint8))[1] == "CHW"   # a square tie is a channel-first crop


def test_channel_inference_checks_every_crop(monkeypatch):
    images = {
        "a.tif": np.zeros((6, 24, 24), dtype=np.uint8),
        "b.tif": np.zeros((24, 24, 6), dtype=np.uint8),
    }
    monkeypatch.setattr(
        "shape.tools.cell_state_analyzer.tool.io.imread",
        lambda path: images[path],
    )
    channels, layouts = infer_channel_spec(list(images))
    assert channels == 6
    assert layouts == ["CHW", "HWC"]

    images["b.tif"] = np.zeros((24, 24), dtype=np.uint8)
    with pytest.raises(ValueError, match="Mixed channel counts"):
        infer_channel_spec(list(images))


def test_v63_six_channel_adapter_initialization_is_exact():
    adapter = ChannelAdapter(6)
    weights = adapter.proj.weight.detach()[:, :, 0, 0]
    expected = torch.zeros((3, 6))
    expected[0, [2, 4]] = 0.5
    expected[1, [1, 5]] = 0.5
    expected[2, [0, 3]] = 0.5
    assert adapter.init_name == "v63_biological_6_to_rgb"
    assert torch.equal(weights, expected)
    assert torch.count_nonzero(adapter.proj.bias) == 0


def test_info_nce_is_symmetric():
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    second = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.7]])
    assert torch.allclose(
        contrastive_loss(first, second),
        contrastive_loss(second, first),
    )


def test_plate_losses_require_complete_technical_metadata():
    incomplete = [{"plate": "p1"}, {"plate": "p2"}]
    route = resolve_plate_nuisance(incomplete, 1.0, 0.1)
    assert route["eligible"] is False
    assert route["plate_mean_weight"] == 0.0
    assert route["plate_adversarial_weight"] == 0.0

    complete = [
        {"plate": "p1", "site": "s1"},
        {"plate": "p1", "site": "s2"},
        {"plate": "p2", "site": "s1"},
        {"plate": "p2", "site": "s2"},
    ]
    route = resolve_plate_nuisance(complete, 1.0, 0.1)
    assert route["eligible"] is True
    assert route["plate_mean_weight"] == 1.0
    assert route["plate_adversarial_weight"] == 0.1


def test_plate_mean_penalty_is_zero_for_equal_plate_means():
    first = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    second = first.clone()
    loss = plate_mean_alignment_loss(first, second, ["p1", "p1", "p2", "p2"])
    assert loss.item() == pytest.approx(0.0)
