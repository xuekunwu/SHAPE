from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

import cellprofiler_native
import cropper_arms as arms
import run_benchmark


def test_formal_arm_registry_is_the_requested_five_groups():
    assert arms.ARMS == ("maskcrop", "cellprofiler", "gpt", "react", "shape")
    assert arms.LLM_ARMS == ("gpt", "react", "shape")


def test_cellprofiler_arm_invokes_native_helper_and_returns_its_crops(tmp_path, monkeypatch):
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:6, 3:8] = 4
    mask[10:15, 1:4] = 9
    image_path = tmp_path / "image.tif"
    image = np.arange(20 * 20, dtype=np.uint8).reshape(20, 20)
    Image.fromarray(image).save(image_path, compression="tiff_lzw")   # what the CellProfiler image cannot read
    seen = {}
    real_run = subprocess.run

    def fake_run(command, **kwargs):
        if not isinstance(command, list):              # e.g. the TIFF codec import's CPU probe
            return real_run(command, **kwargs)
        seen["command"] = command
        seen["kwargs"] = kwargs
        seen["image"] = np.load(command[command.index("--image-path") + 1])
        result_path = Path(command[command.index("--result-json") + 1])
        result_path.write_text(json.dumps({
            "backend": "CellProfiler",
            "module": "SaveCroppedObjects",
            "cellprofiler_version": "4.2.8",
            "cellprofiler_core_version": "4.2.8",
            "crops": [
                {"cell_label": 4, "bbox": [2, 3, 6, 8],
                 "source": "cellprofiler.SaveCroppedObjects"},
                {"cell_label": 9, "bbox": [10, 1, 15, 4],
                 "source": "cellprofiler.SaveCroppedObjects"},
            ],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="CELLPROFILER_NATIVE_OK", stderr="")

    monkeypatch.setattr(arms.subprocess, "run", fake_run)
    arm = arms.CellProfiler(min_area=1, cellprofiler_python="cp-python")
    result = arm(image_path, Path("mask.tiff"), mask, tmp_path)
    assert result.tool_calls == 1
    assert [crop["cell_label"] for crop in result.crops] == [4, 9]
    assert [crop["bbox"] for crop in result.crops] == [[0, 1, 8, 9], [8, 0, 16, 8]]
    assert all(crop["source"] == arms.CELLPROFILER_NORMALIZED_SOURCE for crop in result.crops)
    assert [crop["native_bbox"] for crop in result.crops] == [[2, 3, 6, 8], [10, 1, 15, 4]]
    assert all(Path(crop["crop_image_path"]).is_file() for crop in result.crops)
    assert all((crop["bbox"][2] - crop["bbox"][0]) ==
               (crop["bbox"][3] - crop["bbox"][1]) for crop in result.crops)
    assert seen["command"][0] == "cp-python"
    assert str(arms.CELLPROFILER_HELPER) in seen["command"]
    decoded_path = Path(seen["command"][seen["command"].index("--image-path") + 1])
    assert decoded_path.suffix == ".npy" and not decoded_path.exists()   # host-decoded, removed after the run
    np.testing.assert_array_equal(seen["image"], image)
    assert seen["kwargs"] == {"capture_output": True, "text": True, "check": False}
    assert result.trace["module"] == "SaveCroppedObjects"
    assert result.trace["cellprofiler_version"] == "4.2.8"
    assert result.trace["square_normalization"] == {
        "geometry": "maskcrop_fraction_square",
        "margin": 1,
        "margin_frac": 0.2,
        "recenter": False,
    }


def test_cellprofiler_arm_requires_an_isolated_python(monkeypatch):
    monkeypatch.delenv("CELLPROFILER_PYTHON", raising=False)
    arm = arms.CellProfiler(min_area=1)
    with np.testing.assert_raises_regex(RuntimeError, "CELLPROFILER_PYTHON"):
        arm(Path("img.tif"), Path("mask.tiff"), np.zeros((2, 2)), Path("out"))


def test_cellprofiler_helper_filters_small_labels_and_preserves_ids():
    mask = np.zeros((8, 9), dtype=np.uint16)
    mask[1:3, 2:5] = 4
    mask[6, 7] = 9
    filtered, labels = cellprofiler_native._filter_labels(mask, min_area=2)
    assert labels == [4]
    assert set(np.unique(filtered)) == {0, 4}
    assert cellprofiler_native._bbox(filtered, 4) == [1, 2, 3, 5]


def test_cellprofiler_helper_accepts_channel_first_images():
    image = np.zeros((2, 8, 9), dtype=np.uint16)
    prepared = cellprofiler_native._prepare_image(image, (8, 9))
    assert prepared.shape == (8, 9, 2)
    assert prepared.dtype == np.float32


def test_cellprofiler_helper_reads_tiff_lzw_tiff_png_bmp_and_npy(tmp_path):
    image = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9)
    tifffile.imwrite(tmp_path / "image.tiff", image)
    Image.fromarray(image).save(tmp_path / "image_lzw.tiff", compression="tiff_lzw")
    Image.fromarray(image).save(tmp_path / "image.png")
    Image.fromarray(image).save(tmp_path / "image.bmp")
    np.save(tmp_path / "image.npy", image)

    for name in ("image.tiff", "image_lzw.tiff", "image.png", "image.bmp", "image.npy"):
        loaded = cellprofiler_native._read_source_image(tmp_path / name)
        np.testing.assert_array_equal(loaded, image)


def test_shared_cellprofiler_square_geometry_matches_maskcrop_control(tmp_path):
    image = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    mask = np.zeros((32, 40), dtype=np.uint16)
    mask[3:11, 5:18] = 4
    image_path = tmp_path / "image.tiff"
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)

    result = arms.MaskCrop(min_area=1, margin=1, margin_frac=0.2)(
        image_path, mask_path, mask, tmp_path / "maskcrop")
    expected = arms.maskcrop_square_bboxes(mask, min_area=1, margin=1, margin_frac=0.2)
    assert {crop["cell_label"]: crop["bbox"] for crop in result.crops} == expected


def test_gateway_response_content_is_unwrapped_for_gpt_and_react():
    reply = {
        "content": '[{"cell_label":4,"bbox":[2,3,6,8]}]',
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    assert arms._parse_json_list(reply) == [
        {"cell_label": 4, "bbox": [2, 3, 6, 8], "source": "llm"}
    ]
    assert arms._response_content({"content": {"action": "FINISH"}}) == {"action": "FINISH"}


def _text(content):
    """The text part of an engine call's content (a prompt, or a prompt followed by images)."""
    return content if isinstance(content, str) else content[0]


def _field(tmp_path, name="img", spacing=11, ry=6, rx=4, jitter=2, seed=1, size=96):
    """A dense synthetic field of touching ellipses: its image and ground-truth-like label mask on disk."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint16)
    rr, cc = np.ogrid[:size, :size]
    label = 1
    for r in range(8, size - 6, spacing):
        for c in range(8, size - 6, spacing):
            r0 = r + rng.integers(-jitter, jitter + 1)
            c0 = c + rng.integers(-jitter, jitter + 1)
            region = ((rr - r0) ** 2 / ry ** 2 + (cc - c0) ** 2 / rx ** 2 <= 1) & (mask == 0)
            if region.sum() >= 15:
                mask[region] = label
                label += 1
    image = (mask > 0).astype(np.uint16) * 1000 + rng.integers(0, 50, mask.shape).astype(np.uint16)
    image_path, mask_path = tmp_path / f"{name}.tiff", tmp_path / f"{name}_label.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)
    return image_path, mask_path, mask


def test_metadata_to_crops_matches_labels_by_original_bbox_and_raises_on_mismatch(tmp_path):
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:6, 3:8] = 9          # label order differs from the metadata order below
    mask[10:15, 1:4] = 4
    metadata = tmp_path / "cell_crops_metadata_x.json"
    entries = [{"cell_id": "a", "original_bbox": [10, 1, 15, 4], "crop_bbox": [9, 0, 17, 8], "crop_path": "p4"},
               {"cell_id": "b", "original_bbox": [2, 3, 6, 8], "crop_bbox": [0, 1, 8, 9], "crop_path": "p9"}]
    metadata.write_text(json.dumps({"cell_metadata": entries}), encoding="utf-8")
    crops = arms._metadata_to_crops({"cell_crops_metadata_path": str(metadata)}, mask, 1)
    assert {crop["cell_label"]: crop["bbox"] for crop in crops} == {4: [9, 0, 17, 8], 9: [0, 1, 8, 9]}
    assert {crop["cell_label"]: crop["crop_image_path"] for crop in crops} == {4: "p4", 9: "p9"}

    metadata.write_text(json.dumps({"cell_metadata": entries[:1]}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"1 matched of 2 targets, missing=\[9\]"):
        arms._metadata_to_crops({"cell_crops_metadata_path": str(metadata)}, mask, 1)
    metadata.write_text(json.dumps({"cell_metadata": [*entries, {**entries[0], "original_bbox": [0, 0, 1, 1]}]}),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="unmatched_entries=1"):
        arms._metadata_to_crops({"cell_crops_metadata_path": str(metadata)}, mask, 1)


def test_gpt_batches_every_eligible_cell_and_sees_every_box(monkeypatch, tmp_path):
    mask = np.arange(1, 131, dtype=np.uint16).reshape(13, 10)
    prompts = []

    class StubEngine:
        def generate(self, content, **_kwargs):
            prompts.append(_text(content))
            targets = json.loads(_text(content).split("Targets: ", 1)[1].split("\n", 1)[0])
            boxes = {row["cell_label"]: row["bbox"] for row in
                     json.loads(_text(content).split("All objects of the mask (cell_label, bbox, area): ", 1)[1]
                                .split("\n", 1)[0])}
            return {"content": json.dumps([{"cell_label": label, "bbox": boxes[label]} for label in targets])}

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    result = arms.GPT("gpt-5.6-luna", min_area=1)(
        Path("image.tif"), Path("mask.tiff"), mask, tmp_path)
    assert [crop["cell_label"] for crop in result.crops] == list(range(1, 131))
    assert [batch["n_targets"] for batch in result.trace["batches"]] == [64, 64, 2]
    assert len(prompts) == 3
    # every request carries every object of the mask, not only its batch
    assert all('"cell_label":1,' in prompt and '"cell_label":130,' in prompt for prompt in prompts)


def test_gpt_gets_the_input_image(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path, size=48)
    seen = []

    class StubEngine:
        def generate(self, content, **_kwargs):
            seen.append(content)
            targets = json.loads(_text(content).split("Targets: ", 1)[1].split("\n", 1)[0])
            return {"content": json.dumps([{"cell_label": label, "bbox": [0, 0, 4, 4]} for label in targets])}

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    result = arms.GPT("gpt-5.6-luna", min_area=1)(image_path, mask_path, mask, tmp_path / "out")
    assert result.trace["image_attached"] is True
    assert isinstance(seen[0], list) and isinstance(seen[0][-1], bytes)
    assert seen[0][1] == "INPUT IMAGE (the mask's object boundaries in green):"


def test_gpt_retries_an_incomplete_batch(monkeypatch, tmp_path):
    mask = np.array([[1, 2]], dtype=np.uint16)
    calls = []

    class StubEngine:
        def generate(self, content, **_kwargs):
            calls.append(_text(content))
            targets = json.loads(_text(content).split("Targets: ", 1)[1].split("\n", 1)[0])
            return {"content": json.dumps([
                {"cell_label": label, "bbox": [0, label - 1, 1, label]}
                for label in (targets[:1] if len(calls) == 1 else targets)])}

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    result = arms.GPT("gpt-5.6-luna", min_area=1)(
        Path("image.tif"), Path("mask.tiff"), mask, tmp_path)
    assert len(result.crops) == 2
    assert result.trace["batches"][0]["attempts"] == 2
    assert "Your previous answer was invalid" in calls[1]


def test_gpt_waits_after_a_gateway_error(monkeypatch, tmp_path):
    mask = np.array([[1, 2]], dtype=np.uint16)
    replies = iter([{"error": "rate_limit", "message": "slow down"},
                    {"content": '[{"cell_label":1,"bbox":[0,0,1,1]},{"cell_label":2,"bbox":[0,1,1,2]}]'}])
    sleeps = []

    class StubEngine:
        def generate(self, _content, **_kwargs):
            return next(replies)

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    monkeypatch.setattr(arms.time, "sleep", sleeps.append)
    result = arms.GPT("gpt-5.6-luna", min_area=1)(Path("image.tif"), Path("mask.tiff"), mask, tmp_path)
    assert len(result.crops) == 2 and result.trace["batches"][0]["attempts"] == 2
    assert sleeps == [arms.LLM_RETRY_SECONDS]


FAKE_KEY = "sk-fake-test-key-0123456789"
CALL_KEYS = {"batch_index", "attempt", "targets", "prompt", "images", "params", "reply", "usage", "error", "seconds"}


def _no_bytes_or_key(text: str) -> bool:
    """No image bytes (repr'd or base64 PNG) and no API key in a serialized record."""
    return not any(marker in text for marker in ("\\x89PNG", "iVBORw0KGgo", "data:image", FAKE_KEY))


def test_gpt_records_every_request_with_its_prompt_image_refs_reply_and_error(monkeypatch, tmp_path):
    import hashlib

    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    image_path, mask_path, mask = _field(tmp_path, size=48)
    seen = []

    class StubEngine:
        def generate(self, content, **_kwargs):
            seen.append(content)
            targets = json.loads(_text(content).split("Targets: ", 1)[1].split("\n", 1)[0])
            boxes = [{"cell_label": label, "bbox": [0, 0, 4, 4]} for label in targets]
            return {"content": json.dumps(boxes[:1] if len(seen) == 1 else boxes),
                    "usage": {"total_tokens": 11, "reasoning_tokens": 4}}

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    result = arms.GPT("gpt-5.6-luna", min_area=1)(image_path, mask_path, mask, tmp_path / "out")
    calls = result.trace["calls"]
    assert len(calls) == 2 and all(set(call) == CALL_KEYS for call in calls)
    assert [(c["batch_index"], c["attempt"]) for c in calls] == [(0, 1), (0, 2)]
    assert calls[0]["targets"] == arms.labels_in_tool_order(mask, 1)
    assert [c["prompt"] for c in calls] == [content[0] for content in seen]       # the text as sent
    png = seen[0][2]
    assert calls[0]["images"] == [{"label": "INPUT IMAGE (the mask's object boundaries in green):",
                                   "sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}]
    assert calls[0]["params"] == {"model": "gpt-5.6-luna", "max_tokens": arms.GPT_MAX_COMPLETION_TOKENS,
                                  "response_format": None}
    assert len(json.loads(calls[0]["reply"])) == 1 and calls[0]["usage"] == {"total_tokens": 11,
                                                                            "reasoning_tokens": 4}
    assert "target-cell contract failed" in calls[0]["error"] and calls[1]["error"] is None
    assert all(isinstance(c["seconds"], float) for c in calls)
    assert result.trace["batches"] == [{"index": 0, "n_targets": len(calls[0]["targets"]), "attempts": 2}]
    assert _no_bytes_or_key(json.dumps(result.trace, default=str))

    # a failed image: the error carries every request (the runner records it)
    monkeypatch.setattr(arms.time, "sleep", lambda _s: None)
    gateway = SimpleNamespace(generate=lambda *_a, **_k: {"error": "rate_limit", "message": "slow down"})
    monkeypatch.setattr(arms, "_engine", lambda _model: gateway)
    with pytest.raises(ValueError, match="failed after 3 attempts") as failed:
        arms.GPT("gpt-5.6-luna", min_area=1)(image_path, mask_path, mask, tmp_path / "out2")
    assert [c["attempt"] for c in failed.value.trace["calls"]] == [1, 2, 3]
    assert failed.value.trace["calls"][0]["reply"] == {"error": "rate_limit", "message": "slow down"}
    assert "rate_limit: slow down" in failed.value.trace["calls"][0]["error"]


def test_shape_counts_real_tool_invocations(monkeypatch, tmp_path):
    class StubTool:
        def execute(self, **_kwargs):
            return {}

    class StubEngine:
        def generate(self, *_args, **_kwargs):
            return {"usage": {"total_tokens": 7}, "content": {}}

    def fake_agentic(tool, *_args, **_kwargs):
        for _ in range(3):
            tool.execute()
        engine = _args[-1]
        engine.generate("plan", response_format=type("RoundPlan", (), {}))
        engine.generate("montage", response_format=type("MaskCheckVerdict", (), {}))
        return SimpleNamespace(trials=[], result={}, status="satisfied", reason="r", details={})

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    monkeypatch.setattr(arms.rsc, "run_shape_agentic_arm", fake_agentic)
    monkeypatch.setattr(arms.rsc, "_organize_kept_rejected_crops", lambda *_args: None)
    monkeypatch.setattr(arms, "_metadata_to_crops", lambda *_args: [])
    shape = arms.Shape("gpt-5.6-luna", min_area=1, margin=1, margin_frac=0.2)
    shape.tool = StubTool()
    result = shape(Path("image.tif"), Path("mask.tiff"), np.zeros((2, 2)), tmp_path)
    assert result.tool_calls == 3
    assert result.trace["model_usage"] == {
        "reasoner_calls_per_image": 1, "vision_calls_per_image": 1,
        "reasoner_tokens_per_image": 7, "vision_tokens_per_image": 7,
    }


def test_shape_meters_model_usage_per_image(monkeypatch, tmp_path):
    """A call without provider usage nulls only its own image's token count."""
    replies = iter([{"content": {}}, {"usage": {"total_tokens": 5}, "content": {}},
                    {"usage": {"total_tokens": 7}, "content": {}}])

    class StubEngine:
        def generate(self, *_args, **_kwargs):
            return next(replies)

    def fake_agentic(tool, *_args, **_kwargs):
        _args[-1].generate("plan", response_format=type("RoundPlan", (), {}))
        return SimpleNamespace(trials=[], result={}, status="satisfied", reason="r", details={})

    monkeypatch.setattr(arms, "_engine", lambda _model: StubEngine())
    monkeypatch.setattr(arms.rsc, "run_shape_agentic_arm", fake_agentic)
    monkeypatch.setattr(arms.rsc, "_organize_kept_rejected_crops", lambda *_args: None)
    monkeypatch.setattr(arms, "_metadata_to_crops", lambda *_args: [])
    shape = arms.Shape("gpt-5.6-luna", min_area=1, margin=1, margin_frac=0.2)
    usage = [shape(Path("image.tif"), Path("mask.tiff"), np.zeros((2, 2)), tmp_path / str(i)).trace["model_usage"]
             for i in range(3)]
    assert [u["reasoner_calls_per_image"] for u in usage] == [1, 1, 1]
    assert [u["reasoner_tokens_per_image"] for u in usage] == [None, 5, 7]


def test_shape_scores_the_delivered_call_of_the_real_loop(monkeypatch, tmp_path):
    """The real agentic loop with a scripted engine on a dense field where the delivered call is not the last
    trial: the agent's round runs agent_1 from the call at the defaults, its final choice is the default call;
    the scored crops are the delivered call's, not `_converged_result`'s (the last trial's)."""
    from shape.agent.verification.crop_triage import MaskCheckPositionVerdict, MaskCheckVerdict
    from shape.agent.verification.verifiers.crop import CropMove, CropPlan
    from shape.utils.schemas import AdmissionAnswer, FinalChoice, LeverChange

    class Engine:
        def __init__(self):
            self.asked = []

        def generate(self, content, response_format=None, **_kwargs):
            self.asked.append(response_format)
            if response_format is MaskCheckVerdict:
                ids = [line.split("crop_id=")[1].strip().strip("'") for line in content[0].splitlines()
                       if "crop_id=" in line]
                return {"content": MaskCheckVerdict(verdicts=[MaskCheckPositionVerdict(
                    index=i, crop_id=c, verdict="cell_present", confidence="high") for i, c in enumerate(ids)])}
            if response_format is AdmissionAnswer:
                return {"content": AdmissionAnswer(problem="crowded", severity="major", reasoning="t")}
            if response_format is CropPlan:                 # one call in the first round, then satisfied
                moves = [] if CropPlan in self.asked[:-1] else [
                    CropMove(parent="default", changes=[LeverChange(lever="margin_frac", value=0.5)])]
                return {"content": CropPlan(analysis="t", moves=moves)}
            if response_format is FinalChoice:
                return {"content": FinalChoice(call="default", reasoning="t")}
            raise RuntimeError(f"unexpected question {response_format}")

    image_path, mask_path, mask = _field(tmp_path)          # spacing 11, seed 1: a dense field
    real_loop = arms.rsc.run_shape_agentic_arm
    captured = {}

    def loop(tool, *args, **kwargs):
        captured["resolution"] = real_loop(tool, *args, **kwargs)
        return captured["resolution"]

    monkeypatch.setattr(arms, "_engine", lambda _model: Engine())
    monkeypatch.setattr(arms.rsc, "run_shape_agentic_arm", loop)
    shape = arms.Shape("gpt-5.6-luna", min_area=15, margin=1, margin_frac=0.2, cores=4)
    result = shape(image_path, mask_path, mask, tmp_path / "shape")
    resolution = captured["resolution"]
    assert result.trace["delivered_call"] == "default" and len(resolution.trials) == 3   # input, default, agent_1
    delivered = resolution.details["delivered_metadata_path"]
    assert result.trace["scored_metadata_path"] == delivered == result.trace["delivered_metadata_path"]
    assert arms.rsc._converged_result(resolution)["cell_crops_metadata_path"] != delivered
    meta = json.loads(Path(delivered).read_text(encoding="utf-8"))["cell_metadata"]
    assert sorted(crop["bbox"] for crop in result.crops) == sorted(m["crop_bbox"] for m in meta)
    assert len(result.crops) == len(arms.labels_in_tool_order(mask, 15))
    assert result.trace["n_triage_rejected"] == 0


def test_shape_scores_triage_rejects_and_flags_them(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path, size=48)
    maskcrop = arms.MaskCrop(min_area=15, margin=1, margin_frac=0.2)(image_path, mask_path, mask, tmp_path / "mc")
    metadata_path = maskcrop.trace["result"]["cell_crops_metadata_path"]
    paths = [crop["crop_image_path"] for crop in maskcrop.crops]
    Path(metadata_path.replace("cell_crops_metadata_", "cell_crops_kept_manifest_")).write_text(
        json.dumps({"kept_crop_paths": paths[1:]}), encoding="utf-8")

    def fake_agentic(*_args, **_kwargs):
        return SimpleNamespace(trials=[], result={"cell_crops_metadata_path": metadata_path},
                               status="satisfied", reason="r", details={"delivered_metadata_path": metadata_path})

    monkeypatch.setattr(arms, "_engine", lambda _model: object())
    monkeypatch.setattr(arms.rsc, "run_shape_agentic_arm", fake_agentic)
    result = arms.Shape("gpt-5.6-luna", min_area=15, margin=1, margin_frac=0.2)(
        image_path, mask_path, mask, tmp_path / "shape")
    assert len(result.crops) == len(maskcrop.crops)                 # every target is still scored
    assert [crop["triage_rejected"] for crop in result.crops] == [True] + [False] * (len(paths) - 1)
    assert result.trace["n_triage_rejected"] == 1
    scored = run_benchmark._score_one(mask, result.crops, 1.0, 1, 0)
    assert scored["n_triage_rejected"] == 1 and scored["n_crops"] == len(paths)


def test_shape_flags_the_crops_the_delivered_calls_mask_check_rejected(monkeypatch, tmp_path):
    """The real agentic loop with a scripted engine: the mask check runs once, after the final choice, on the
    delivered call, and the crops it rejected are those flagged `triage_rejected` (`n_triage_rejected`)."""
    from shape.agent.verification.crop_triage import MaskCheckPositionVerdict, MaskCheckVerdict
    from shape.agent.verification.verifiers.crop import CropPlan
    from shape.utils.schemas import AdmissionAnswer, FinalChoice

    class Engine:
        def __init__(self):
            self.asked = []

        def generate(self, content, response_format=None, **_kwargs):
            self.asked.append(response_format)
            if response_format is MaskCheckVerdict:
                ids = [line.split("crop_id=")[1].strip().strip("'") for line in content[0].splitlines()
                       if "crop_id=" in line]
                first = self.asked.count(MaskCheckVerdict) == 1
                return {"content": MaskCheckVerdict(verdicts=[MaskCheckPositionVerdict(
                    index=i, crop_id=c, verdict="wrong_object" if first and i == 0 else "cell_present",
                    confidence="high") for i, c in enumerate(ids)])}
            if response_format is AdmissionAnswer:
                return {"content": AdmissionAnswer(problem="crowded", severity="major", reasoning="t")}
            if response_format is CropPlan:
                return {"content": CropPlan(analysis="t", moves=[])}
            if response_format is FinalChoice:
                return {"content": FinalChoice(call="default", reasoning="t")}
            raise RuntimeError(f"unexpected question {response_format}")

    engine = Engine()
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    image_path, mask_path, mask = _field(tmp_path, size=48)
    monkeypatch.setattr(arms, "_engine", lambda _model: engine)
    result = arms.Shape("gpt-5.6-luna", min_area=15, margin=1, margin_frac=0.2, cores=4)(
        image_path, mask_path, mask, tmp_path / "shape")
    first = engine.asked.index(MaskCheckVerdict)
    assert FinalChoice in engine.asked[:first] and set(engine.asked[first:]) == {MaskCheckVerdict}
    details = json.loads((tmp_path / "shape" / "agent_details.json").read_text(encoding="utf-8"))
    check = details["mask_check"]
    assert check["call"] == details["call"] == result.trace["delivered_call"] == "default"
    assert result.trace["n_triage_rejected"] == check["mask_check_rejected"] == len(check["rejected_cells"]) == 1
    meta = json.loads(Path(result.trace["scored_metadata_path"]).read_text(encoding="utf-8"))["cell_metadata"]
    assert {c["crop_image_path"] for c in result.crops if c["triage_rejected"]} == {
        m["crop_path"] for m in meta if m["cell_id"] in check["rejected_cells"]}
    assert result.trace["model_usage"]["vision_calls_per_image"] == len(engine.asked) - first
    # the record: every montage question and every judged crop's verdict; the prompts' image refs
    assert check["montages"] and all(m["index"] == i for i, m in enumerate(check["montages"]))
    for montage in check["montages"]:
        assert set(montage) == {"index", "prompt", "image", "montage_path", "reply", "error"}
        assert set(montage["image"]) == {"label", "sha256", "bytes"} and montage["error"] is None
        assert montage["reply"]["verdicts"] and Path(montage["montage_path"]).is_file()
    assert len(check["verdicts"]) == check["mask_check_judged"]
    assert all(set(v) == {"crop_id", "verdict", "confidence", "reject", "montage_path"} for v in check["verdicts"])
    assert [v for v in check["verdicts"] if v["reject"]] == [
        {"crop_id": check["rejected_cells"][0], "verdict": "wrong_object", "confidence": "high", "reject": True,
         "montage_path": check["montages"][0]["montage_path"]}]
    log_text = (tmp_path / "shape" / "agent_log.jsonl").read_text(encoding="utf-8")
    prompts = [entry for entry in map(json.loads, log_text.splitlines()) if entry["kind"] == "prompt"]
    assert {p["label"] for p in prompts} >= {"round", "final"}
    assert all(p["images"] and all(set(ref) == {"label", "sha256", "bytes"} for ref in p["images"])
               for p in prompts)
    assert prompts[0]["images"][0]["label"] == "INPUT IMAGE (the mask's object boundaries in green):"
    details_text = (tmp_path / "shape" / "agent_details.json").read_text(encoding="utf-8")
    assert _no_bytes_or_key(log_text) and _no_bytes_or_key(details_text)


def test_shape_raises_when_the_loop_delivers_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(arms, "_engine", lambda _model: object())
    monkeypatch.setattr(arms.rsc, "run_shape_agentic_arm", lambda *_a, **_k: SimpleNamespace(
        trials=[], result=None, status="conceded", reason="no_objects", details={}))
    with pytest.raises(RuntimeError, match="SHAPE delivered no call: conceded"):
        arms.Shape("gpt-5.6-luna", min_area=1, margin=1, margin_frac=0.2)(
            Path("image.tif"), Path("mask.tiff"), np.zeros((2, 2)), tmp_path)


def test_all_arms_must_return_the_same_expected_target_cells():
    crops = [{"cell_label": 4, "bbox": [2, 3, 7, 8]}]
    for arm in arms.ARMS:
        run_benchmark._validate_crops(arm, crops, [4], (20, 20))
        with pytest.raises(ValueError, match="target-cell contract failed"):
            run_benchmark._validate_crops(arm, crops, [4, 9], (20, 20))
        with pytest.raises(ValueError, match="non-square bbox"):
            run_benchmark._validate_crops(
                arm, [{"cell_label": 4, "bbox": [2, 3, 6, 8]}], [4], (20, 20))


def test_usage_delta_reports_only_provider_observed_fields():
    before = {"calls": 2, "total_tokens": 100, "input_tokens": 60, "output_tokens": 40,
              "usage_field_observed_calls": {"input_tokens": 2, "output_tokens": 2}}
    after = {"calls": 4, "total_tokens": 150, "input_tokens": 90, "output_tokens": 50,
             "usage_field_observed_calls": {"input_tokens": 4, "output_tokens": 3}}
    assert run_benchmark._usage_delta(before, after) == {
        "model_calls_per_image": 2, "input_tokens_per_image": 30,
        "output_tokens_per_image": None, "cached_tokens_per_image": None, "reasoning_tokens_per_image": None}


def test_usage_delta_reports_cached_and_reasoning_tokens_only_when_every_call_reports_them():
    before = {"calls": 1, "cached_tokens": 10, "reasoning_tokens": 5,
              "usage_field_observed_calls": {"cached_tokens": 1, "reasoning_tokens": 1}}
    after = {"calls": 3, "cached_tokens": 1034, "reasoning_tokens": 45,
             "usage_field_observed_calls": {"cached_tokens": 3, "reasoning_tokens": 2}}
    delta = run_benchmark._usage_delta(before, after)
    assert delta["cached_tokens_per_image"] == 1024 and delta["reasoning_tokens_per_image"] is None
    assert {"cached_tokens_per_image", "reasoning_tokens_per_image"} <= set(run_benchmark.MODEL_USAGE_FIELDS)
    assert set(run_benchmark.MODEL_USAGE_FIELDS) <= set(run_benchmark.PER_IMAGE_FIELDS)


def _run_args(run_dir, arm, **overrides):
    return SimpleNamespace(**{"run_dir": run_dir, "arm": arm, "model": "gpt-5.6-luna", "min_area": 1,
                              "margin": 1, "maskcrop_margin_frac": 0.2, "gpt_batch_size": 64, "cores": 4,
                              "deadline": None, "shape_loop": "agentic", "cellprofiler_python": None,
                              **overrides})


def test_runner_marks_incomplete_gpt_output_as_error(tmp_path, monkeypatch):
    mask = np.array([[1, 2]], dtype=np.uint16)
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(mask_path, mask)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"selected_images": [{
        "image_id": "img", "image_path": str(tmp_path / "image.tiff"),
        "mask_path": str(mask_path), "gt_path": str(mask_path),
    }]}), encoding="utf-8")

    class IncompleteGPT:
        def __call__(self, *_args):
            return arms.ArmResult([{"cell_label": 1, "bbox": [0, 0, 1, 1]}], 0, {})

    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda *_args, **_kwargs: IncompleteGPT())
    run_benchmark.run_arm(_run_args(run_dir, "gpt"))
    record = json.loads((run_dir / "gpt" / "records.jsonl").read_text())
    assert record["status"] == "error"
    assert "missing=[2]" in record["error"]


def test_runner_records_the_agentic_budget_and_passes_cores_and_deadline(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"selected_images": [], "mask_source": "ground_truth"}),
                                            encoding="utf-8")
    seen = {}
    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda name, **kwargs: seen.update(kwargs))
    run_benchmark.run_arm(_run_args(run_dir, "react", cores=8, deadline=123.0))
    meta = json.loads((run_dir / "react" / "run_meta.json").read_text())
    assert (seen["cores"], seen["deadline"]) == (8, 123.0)
    assert (meta["cores"], meta["concurrency_cap"], meta["react_tool_budget"]) == (8, 4, 1 + 7 * 4)
    assert meta["mask_source"] == "ground_truth"


def test_run_meta_records_the_llm_configuration_host_packages_and_seed_never_the_key(tmp_path, monkeypatch):
    """Every arm: host, python, packages, seed; the LLM arms: the configuration their requests carry (the engine
    built as the run builds it, offline), the gateway's host only; the key is in no record."""
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example.org/v1")
    monkeypatch.delenv("SHAPE_REASONING_EFFORT", raising=False)
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(mask_path, np.array([[1, 2]], dtype=np.uint16))
    item = {"image_id": "img", "image_path": str(tmp_path / "image.tiff"), "mask_path": str(mask_path),
            "gt_path": str(mask_path)}
    real_make_arm = arms.make_arm

    def make_arm(name, **kwargs):          # the real arm and engine; the gpt arm's requests answered offline
        arm = real_make_arm(name, **kwargs)
        if name == "gpt":
            arm.engine.generate = lambda *_a, **_k: {
                "content": '[{"cell_label":1,"bbox":[0,0,1,1]},{"cell_label":2,"bbox":[0,1,1,2]}]',
                "usage": {"total_tokens": 3}}
        return arm

    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", make_arm)
    expected = {"gpt": ({"crop_boxes": 8000}, {"crop_boxes": None}),
                "react": ({"action": 4000}, {"action": "ReactAction"}),
                "shape": ({k: 4000 for k in ("admission", "round", "repair", "final", "final_retry", "mask_check")},
                          {"admission": "AdmissionAnswer", "round": "CropPlan", "repair": "CropPlan",
                           "final": "FinalChoice", "final_retry": "FinalChoice", "mask_check": "MaskCheckVerdict"})}
    for name in ("maskcrop", "gpt", "react", "shape"):
        run_dir = tmp_path / name             # react and shape would ask the model: no image here
        run_dir.mkdir()
        (run_dir / "selection.json").write_text(json.dumps({
            "seed": 1881694274, "selected_images": [item] if name in ("maskcrop", "gpt") else []}),
            encoding="utf-8")
        run_benchmark.run_arm(_run_args(run_dir, name))
        text = (run_dir / name / "run_meta.json").read_text(encoding="utf-8")
        meta = json.loads(text)
        assert meta["seed"] == 1881694274 and meta["host"] and meta["python"].count(".") == 2
        assert {"numpy", "scipy", "scikit-image", "tifffile", "pydantic"} <= set(meta["packages"])
        assert FAKE_KEY not in text and "https://" not in text
        if name == "maskcrop":
            assert meta["llm"] is None
            continue
        assert meta["llm"] == {
            "model": "gpt-5.6-luna", "provider": "openai", "base_url_host": "gateway.example.org",
            "reasoning_effort": "medium", "temperature": None, "top_p": None,
            "max_tokens": expected[name][0], "response_schema": expected[name][1],
            "llm_attempts": {kind: 1 if kind == "mask_check" or name == "react" else 3 for kind in expected[name][0]},
            "llm_retry_seconds": None if name == "react" else 30, "response_cache": False,
            "system_prompt": "You are a helpful, creative, and smart assistant.",
            "sdk_max_retries": 2, "sdk_timeout_seconds": 120.0,
            "structured_parse_fallback": "one plain request when the structured parse fails"}
    records = (tmp_path / "gpt" / "gpt" / "records.jsonl").read_text(encoding="utf-8")
    assert json.loads(records)["status"] == "ok" and json.loads(records)["trace"]["calls"]
    assert _no_bytes_or_key(records)


def test_runner_keeps_the_trace_of_a_failed_image(tmp_path, monkeypatch):
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(mask_path, np.array([[1, 2]], dtype=np.uint16))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"selected_images": [{
        "image_id": "img", "image_path": str(tmp_path / "image.tiff"), "mask_path": str(mask_path),
        "gt_path": str(mask_path)}]}), encoding="utf-8")

    def failing(*_args):
        error = ValueError("GPT batch 1 failed after 3 attempts: x")
        error.trace = {"calls": [{"attempt": 1, "error": "x"}]}
        raise error

    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda *_a, **_k: failing)
    run_benchmark.run_arm(_run_args(run_dir, "gpt"))
    record = json.loads((run_dir / "gpt" / "records.jsonl").read_text())
    assert record["status"] == "error" and record["trace"] == {"calls": [{"attempt": 1, "error": "x"}]}


def _pixel_writing_arm(fail: bool):
    """An arm writing pixel and JSON files under the dir it is given (top level and nested), recording their paths."""
    def arm(_image_path, _mask_path, _mask, out_dir):
        nested = out_dir / "tool_cache" / "crop_calls" / "abc"
        nested.mkdir(parents=True)
        for name in ("cell_0001.tiff", "cell_0001_mask.png", "crops.zip", "cell_crops_metadata_x.json"):
            (nested / name).write_bytes(b"x")
        (out_dir / "agent_log.jsonl").write_text("{}\n", encoding="utf-8")
        paths = {"str": str(nested / "cell_0001.tiff"), "posix": (nested / "cell_0001.tiff").as_posix(),
                 "metadata": str(nested / "cell_crops_metadata_x.json"), "dir": str(out_dir)}
        if fail:
            error = RuntimeError(f"failed after writing {paths['metadata']}")
            error.trace = {"paths": paths}
            raise error
        crops = [{"cell_label": 1, "bbox": [0, 0, 1, 1], "crop_image_path": paths["str"]},
                 {"cell_label": 2, "bbox": [0, 1, 1, 2], "crop_image_path": paths["posix"]}]
        return arms.ArmResult(crops, 1, {"paths": paths})
    return arm


def _one_image_run(tmp_path):
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(mask_path, np.array([[1, 2]], dtype=np.uint16))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"selected_images": [{
        "image_id": "img", "image_path": str(tmp_path / "image.tiff"), "mask_path": str(mask_path),
        "gt_path": str(mask_path)}]}), encoding="utf-8")
    return run_dir


@pytest.mark.parametrize("fail", [False, True])
def test_work_root_keeps_only_non_pixel_files_and_repoints_the_record(tmp_path, monkeypatch, fail):
    """CROPPER_WORK_ROOT: the arm runs in <root>/<arm>/<image>; only its non-pixel files reach the run dir (same
    relative paths), every recorded path names the run dir, the work dir is removed; on success and on error."""
    run_dir = _one_image_run(tmp_path)
    root = tmp_path / "node_local"
    monkeypatch.setenv("CROPPER_WORK_ROOT", str(root))
    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda *_a, **_k: _pixel_writing_arm(fail))
    run_benchmark.run_arm(_run_args(run_dir, "maskcrop"))
    image_dir = run_dir / "maskcrop" / "images" / "img"
    kept = sorted(p.relative_to(image_dir).as_posix() for p in image_dir.rglob("*") if p.is_file())
    assert kept == ["agent_log.jsonl", "tool_cache/crop_calls/abc/cell_crops_metadata_x.json"]
    assert not (root / "maskcrop" / "img").exists()
    text = (run_dir / "maskcrop" / "records.jsonl").read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["status"] == ("error" if fail else "ok")
    assert str(root) not in text and root.as_posix() not in text
    paths = record["trace"]["paths"]
    tiff = image_dir / "tool_cache" / "crop_calls" / "abc" / "cell_0001.tiff"
    assert (paths["str"], paths["posix"], paths["dir"]) == (str(tiff), tiff.as_posix(), str(image_dir))
    if fail:
        assert record["error"] == f"failed after writing {paths['metadata']}"
    else:
        assert [c["crop_image_path"] for c in record["crops"]] == [str(tiff), tiff.as_posix()]
        assert [c["crop_image_path"] for c in record["crop_results"]] == [str(tiff), tiff.as_posix()]


def test_work_root_appends_a_resumed_images_jsonl_log(tmp_path, monkeypatch):
    """An image run again (a resume after a failed attempt) keeps the earlier attempt's log lines."""
    image_dir = tmp_path / "run" / "shape" / "images" / "img"
    work = tmp_path / "node_local" / "shape" / "img"
    (image_dir).mkdir(parents=True)
    (image_dir / "agent_log.jsonl").write_text('{"attempt": 1}\n', encoding="utf-8")
    (image_dir / "agent_details.json").write_text('{"attempt": 1}', encoding="utf-8")
    work.mkdir(parents=True)
    (work / "agent_log.jsonl").write_text('{"attempt": 2}\n', encoding="utf-8")
    (work / "agent_details.json").write_text('{"attempt": 2}', encoding="utf-8")
    run_benchmark._keep_non_pixel_files(work, image_dir)
    assert (image_dir / "agent_log.jsonl").read_text(encoding="utf-8") == '{"attempt": 1}\n{"attempt": 2}\n'
    assert (image_dir / "agent_details.json").read_text(encoding="utf-8") == '{"attempt": 2}'
    assert not work.exists()


def test_without_work_root_the_arm_writes_into_the_run_dir_as_before(tmp_path, monkeypatch):
    run_dir = _one_image_run(tmp_path)
    monkeypatch.delenv("CROPPER_WORK_ROOT", raising=False)
    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda *_a, **_k: _pixel_writing_arm(False))
    run_benchmark.run_arm(_run_args(run_dir, "maskcrop"))
    image_dir = run_dir / "maskcrop" / "images" / "img"
    assert len([p for p in image_dir.rglob("*") if p.is_file()]) == 5           # the pixel files too
    record = json.loads((run_dir / "maskcrop" / "records.jsonl").read_text(encoding="utf-8"))
    nested = image_dir / "tool_cache" / "crop_calls" / "abc"
    assert record["trace"]["paths"]["dir"] == str(image_dir)
    assert record["crops"][0]["crop_image_path"] == str(nested / "cell_0001.tiff")


def test_score_reports_the_registered_metrics():
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:6, 3:8] = 4
    crops = [{"cell_label": 4, "bbox": [2, 3, 6, 8]}]
    scored = run_benchmark._score_one(mask, crops, seconds=1.25, tool_calls=1, tokens=0)
    assert run_benchmark.METRICS == ("neighbor_contamination", "target_retention", "runtime_per_image",
                                     "tool_calls_per_image", "tokens_per_image")
    assert set(run_benchmark.METRICS) <= set(scored)
    assert scored["neighbor_contamination"] == 0.0
    assert scored["target_retention"] == 1.0
    assert scored["runtime_per_image"] == 1.25
    assert scored["tool_calls_per_image"] == 1
    assert scored["tokens_per_image"] == 0
    assert scored["n_triage_rejected"] is None
    assert "ap50" not in scored and "ap50_gain_per_tool_call" not in scored


def test_prepare_selects_random_images_with_their_ground_truth_masks(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "labels").mkdir()
    for i in range(4):
        arr = np.zeros((8, 8), dtype=np.uint16)
        tifffile.imwrite(data / "images" / f"img{i}.tif", arr)
        if i != 2:
            tifffile.imwrite(data / "labels" / f"img{i}_label.tiff", arr)
    args = SimpleNamespace(run_dir=tmp_path / "run", data_dir=data, n_images=30, seed=0,
                           label_suffix="_label", label_ext=".tiff")
    run_benchmark.prepare(args)
    payload = json.loads((tmp_path / "run" / "selection.json").read_text())
    assert payload["mask_source"] == "ground_truth"
    assert {x["image_id"] for x in payload["selected_images"]} == {"img0", "img1", "img3"}
    assert all(x["mask_path"] == x["gt_path"] for x in payload["selected_images"])


def _write_arm(run_dir, arm, mask, selection_hash, records):
    arm_dir = run_dir / arm
    arm_dir.mkdir()
    lines = []
    for image_id, crops in records:
        if crops is None:
            lines.append({"image_id": image_id, "status": "error", "arm": arm, "error": "boom",
                          "metrics": {"runtime_per_image": 9.0}})
            continue
        crop_results = run_benchmark._score_crops(mask, crops)
        metrics = run_benchmark._score_one(mask, crops, seconds=1.0, tool_calls=1, tokens=0,
                                           crop_results=crop_results)
        metrics.update({"n_expected_crops": 1, "crop_coverage": 1.0})
        lines.append({"image_id": image_id, "status": "ok", "arm": arm, "metrics": metrics, "crops": crops,
                      "crop_results": crop_results, "trace": {}})
    (arm_dir / "records.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    (arm_dir / "run_meta.json").write_text(json.dumps({
        "arm": arm, "selection_sha256": selection_hash, "min_area": 1,
        "margin": 1, "maskcrop_margin_frac": 0.2,
        "model": "gpt-5.6-luna" if arm in arms.LLM_ARMS else None,
        "cores": 4 if arm in ("react", "shape") else None,
        "concurrency_cap": 2 if arm in ("react", "shape") else None,
        "react_tool_budget": 1 + (arms.MAX_ROUNDS - 1) * 2 if arm == "react" else None,
    }), encoding="utf-8")


def test_compare_writes_per_image_per_crop_and_paired_domain_tool_results(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mask = np.zeros((10, 12), dtype=np.uint16)
    mask[2:6, 3:8] = 4
    mask_path = tmp_path / "img_label.tiff"
    tifffile.imwrite(mask_path, mask)
    selection = {
        "schema": "shape.cropper.formal.selection.v2",
        "mask_source": "ground_truth",
        "selected_images": [{
            "image_id": image_id,
            "image_path": str(tmp_path / "img.png"),
            "mask_path": str(mask_path),
            "gt_path": str(mask_path),
        } for image_id in ("img", "img2")],
    }
    selection_path = run_dir / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_hash = run_benchmark._sha256(selection_path)

    crops_by_arm = {
        "maskcrop": [{"cell_label": 4, "bbox": [1, 2, 7, 8], "source": "maskcrop_metadata"}],
        "cellprofiler": [{"cell_label": 4, "bbox": [1, 2, 7, 8],
                          "source": arms.CELLPROFILER_NORMALIZED_SOURCE,
                          "native_bbox": [2, 3, 6, 8]}],
        "gpt": [{"cell_label": 4, "bbox": [1, 2, 7, 8], "source": "llm"}],
        "react": [{"cell_label": 4, "bbox": [1, 2, 7, 8], "source": "maskcrop_metadata"}],
        "shape": [{"cell_label": 4, "bbox": [1, 2, 7, 8], "source": "maskcrop_metadata",
                   "triage_rejected": True}],
    }
    # img2: the gpt arm failed, so img2 is outside the comparable set; maskcrop's img2 crop is worse
    worse = [{"cell_label": 4, "bbox": [0, 0, 10, 10], "source": "maskcrop_metadata"}]
    for arm, crops in crops_by_arm.items():
        second = None if arm == "gpt" else (worse if arm in ("maskcrop", "cellprofiler") else crops)
        _write_arm(run_dir, arm, mask, selection_hash, [("img", crops), ("img2", second)])

    args = type("Args", (), {"run_dir": run_dir})()
    run_benchmark.compare(args)

    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["schema"] == "shape.cropper.formal.compare.v4"
    assert summary["comparability"]["domain_tools"]["comparable"] is True
    assert summary["comparability"]["all_arms"]["comparable"] is False       # gpt failed img2
    assert summary["comparable_image_ids"] == ["img"]
    assert summary["arms"]["maskcrop"]["n_images"] == 2
    assert summary["arms"]["maskcrop"]["n_comparable_images"] == 1
    assert summary["arms"]["maskcrop"]["target_retention"] == 1.0          # img only, not img2's worse crop
    assert summary["arms"]["shape"]["n_triage_rejected"] == 1.0
    assert summary["arms"]["maskcrop"]["n_triage_rejected"] is None
    with (run_dir / "compare" / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["comparable"] for row in rows] == ["False"] * 5
    assert {row["n_comparable_images"] for row in rows} == {"1"}
    assert "ap50_gain_per_tool_call" not in rows[0]

    crop_rows = [json.loads(line) for line in
                 (run_dir / "compare" / "per_crop.jsonl").read_text().splitlines()]
    cellprofiler_crop = next(row for row in crop_rows if row["arm"] == "cellprofiler")
    assert [cellprofiler_crop[key] for key in
            ("native_bbox_minr", "native_bbox_minc", "native_bbox_maxr", "native_bbox_maxc")] == [2, 3, 6, 8]
    assert next(row for row in crop_rows if row["arm"] == "shape")["triage_rejected"] is True
    pairs = [json.loads(line) for line in
             (run_dir / "compare" / "domain_tool_pairs.jsonl").read_text().splitlines()]
    assert pairs[0]["maskcrop_present"] is True
    assert pairs[0]["cellprofiler_present"] is True
    assert pairs[0]["maskcrop_is_square"] is True
    assert pairs[0]["cellprofiler_is_square"] is True
    assert pairs[0]["same_bbox"] is True

    # every arm finishes img2 -> comparable; a wrong ReAct budget breaks it again
    _rewrite = run_dir / "gpt"
    for path in _rewrite.iterdir():
        path.unlink()
    _rewrite.rmdir()
    _write_arm(run_dir, "gpt", mask, selection_hash, [("img", crops_by_arm["gpt"]), ("img2", crops_by_arm["gpt"])])
    run_benchmark.compare(args)
    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["comparability"]["all_arms"]["comparable"] is True
    assert summary["n_comparable_images"] == 2
    assert len((run_dir / "compare" / "per_image.jsonl").read_text().splitlines()) == 10
    react_meta = run_dir / "react" / "run_meta.json"
    meta = json.loads(react_meta.read_text())
    react_meta.write_text(json.dumps({**meta, "react_tool_budget": arms.MAX_ROUNDS * 2}), encoding="utf-8")
    run_benchmark.compare(args)
    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["comparability"]["all_arms"]["comparable"] is False
    assert any("SHAPE's budget" in reason for reason in summary["comparability"]["all_arms"]["reasons"])

    gpt_path = run_dir / "gpt" / "records.jsonl"
    gpt_records = [json.loads(line) for line in gpt_path.read_text().splitlines()]
    gpt_records[0]["crops"] = []
    gpt_path.write_text("".join(json.dumps(r) + "\n" for r in gpt_records), encoding="utf-8")
    react_meta.write_text(json.dumps(meta), encoding="utf-8")
    run_benchmark.compare(args)
    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["comparability"]["all_arms"]["comparable"] is False
    assert summary["comparability"]["all_arms"]["arms"]["gpt"]["wrong_target_image_ids"] == ["img"]


def test_compare_comparable_set_ignores_absent_arms_and_the_control(tmp_path):
    """An absent arm and the CellProfiler control's failures do not blank or shrink the other arms' means;
    the absent arm is reported by the comparable flag and its reasons."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mask = np.zeros((10, 12), dtype=np.uint16)
    mask[2:6, 3:8] = 4
    mask_path = tmp_path / "img_label.tiff"
    tifffile.imwrite(mask_path, mask)
    selection = {"mask_source": "ground_truth", "selected_images": [
        {"image_id": image_id, "image_path": str(tmp_path / "img.png"), "mask_path": str(mask_path),
         "gt_path": str(mask_path)} for image_id in ("img", "img2")]}
    selection_path = run_dir / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_hash = run_benchmark._sha256(selection_path)
    crops = [{"cell_label": 4, "bbox": [1, 2, 7, 8], "source": "maskcrop_metadata"}]
    for arm in ("maskcrop", "cellprofiler", "gpt", "react"):              # shape is absent
        second = None if arm == "cellprofiler" else crops                 # the control fails img2
        _write_arm(run_dir, arm, mask, selection_hash, [("img", crops), ("img2", second)])
    run_benchmark.compare(type("Args", (), {"run_dir": run_dir})())
    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["missing_arms"] == ["shape"]
    assert summary["comparable_image_ids"] == ["img", "img2"]
    assert summary["arms"]["maskcrop"]["n_comparable_images"] == 2
    assert summary["arms"]["maskcrop"]["target_retention"] == 1.0
    assert summary["arms"]["cellprofiler"]["n_comparable_images"] == 1
    all_arms = summary["comparability"]["all_arms"]
    assert all_arms["comparable"] is False and "cellprofiler" not in all_arms["gated_arms"]
    assert "shape has no records (the arm is missing)" in all_arms["reasons"]
    assert not any(reason.startswith("cellprofiler") for reason in all_arms["reasons"])
    assert summary["comparability"]["domain_tools"]["comparable"] is False      # the control's gap


def test_runner_resume_keeps_finished_images_and_reruns_the_rest(tmp_path, monkeypatch):
    mask = np.array([[1, 2]], dtype=np.uint16)
    mask_path = tmp_path / "mask.tiff"
    tifffile.imwrite(mask_path, mask)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"selected_images": [
        {"image_id": image_id, "image_path": str(tmp_path / "image.tiff"), "mask_path": str(mask_path),
         "gt_path": str(mask_path)} for image_id in ("a", "b", "c")]}), encoding="utf-8")
    seen = []

    class FlakyArm:
        def __call__(self, image_path, mask_path, mask, out_dir):
            seen.append(out_dir.name)
            if out_dir.name == "b" and len(seen) <= 3:
                raise RuntimeError("timeout-like failure")
            return arms.ArmResult([{"cell_label": 1, "bbox": [0, 0, 1, 1]},
                                   {"cell_label": 2, "bbox": [0, 1, 1, 2]}], 1, {})

    monkeypatch.setattr(run_benchmark.arm_lib, "make_arm", lambda *_args, **_kwargs: FlakyArm())
    run_benchmark.run_arm(_run_args(run_dir, "maskcrop"))
    first = [json.loads(line) for line in (run_dir / "maskcrop" / "records.jsonl").read_text().splitlines()]
    assert [r["status"] for r in first] == ["ok", "error", "ok"]
    run_benchmark.run_arm(_run_args(run_dir, "maskcrop", resume=True, deadline=99.0))
    records = [json.loads(line) for line in (run_dir / "maskcrop" / "records.jsonl").read_text().splitlines()]
    assert sorted((r["image_id"], r["status"]) for r in records) == [("a", "ok"), ("b", "ok"), ("c", "ok")]
    assert seen == ["a", "b", "c", "b"]
    assert len(json.loads((run_dir / "maskcrop" / "run_meta.json").read_text())["resumed_utc"]) == 1
    with pytest.raises(SystemExit, match="settings changed"):
        run_benchmark.run_arm(_run_args(run_dir, "maskcrop", resume=True, min_area=2))


def _fake_pairs(n):
    return [(Path(f"img{i:03d}.tif"), Path(f"img{i:03d}_label.tiff")) for i in range(n)]


def test_select_random_images_draws_n_distinct_pairs_reproducibly():
    pairs = _fake_pairs(20)
    selected = run_benchmark._select_random_images(pairs, 5, np.random.RandomState(7))
    assert len(selected) == len(set(selected)) == 5                   # without replacement
    assert all(p in pairs for p in selected)
    assert selected == run_benchmark._select_random_images(pairs, 5, np.random.RandomState(7))
    assert selected != run_benchmark._select_random_images(pairs, 5, np.random.RandomState(8))


def test_select_random_images_takes_every_pair_when_fewer_than_n():
    pairs = _fake_pairs(3)                                            # never padded
    assert set(run_benchmark._select_random_images(pairs, 10, np.random.RandomState(0))) == set(pairs)


def test_find_image_mask_pairs_matches_layout(tmp_path):
    # <dir>/images/<name>.tif + <dir>/labels/<name>_label.tiff; a label with no matching image is skipped
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for name in ("img1", "img2"):
        tifffile.imwrite(tmp_path / "images" / f"{name}.tif", np.zeros((8, 8), dtype=np.uint16))
        tifffile.imwrite(tmp_path / "labels" / f"{name}_label.tiff", np.zeros((8, 8), dtype=np.uint16))
    (tmp_path / "labels" / "orphan_label.tiff").write_bytes(b"")
    pairs = run_benchmark._find_image_mask_pairs(tmp_path)
    assert [p[0].stem for p in pairs] == ["img1", "img2"]
