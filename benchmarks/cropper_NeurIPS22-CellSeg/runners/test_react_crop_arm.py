"""The cropper's ReAct arm on the shared runner (benchmarks/_shared/REACT_ARM_CONTRACT.md): a fake engine, the real
cropper on small synthetic fields (and a stub tool where only the call matters)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import functools
import inspect

import numpy as np
import pytest
import tifffile

import cropper_arms as arms
import run_benchmark
from benchmarks._shared import react_arm
from shape.agent.verification.base import BUDGET
from shape.agent.verification.verifiers.crop import CropAgenticVerifier, CropTask, input_record

FAKE_KEY = "sk-fake-test-key-0123456789"
# SHAPE's verifier layer: none of it may reach ReAct's prompt or observations
SHAPE_ONLY = ("mean_target_dominance", "mean_neighbor_contamination", "mean_target_retention", "mean_centered",
              "admissible", "guard", "readout", "Objectives", "per_cell_quantiles", "panel", "INPUT IMAGE",
              "What the crops are for", "Type: choice", "Range", "uncroppable", "bbox_other_fraction",
              "mask-error check")


def _field(tmp_path, name="img", spacing=11, ry=6, rx=4, seed=1, size=48):
    """A dense synthetic field of touching ellipses: its image and label mask on disk."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), np.uint16)
    rr, cc = np.ogrid[:size, :size]
    label = 1
    for r in range(8, size - 6, spacing):
        for c in range(8, size - 6, spacing):
            r0, c0 = r + rng.integers(-2, 3), c + rng.integers(-2, 3)
            region = ((rr - r0) ** 2 / ry ** 2 + (cc - c0) ** 2 / rx ** 2 <= 1) & (mask == 0)
            if region.sum() >= 15:
                mask[region] = label
                label += 1
    image = (mask > 0).astype(np.uint16) * 1000 + rng.integers(0, 50, mask.shape).astype(np.uint16)
    image_path, mask_path = tmp_path / f"{name}.tiff", tmp_path / f"{name}_label.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)
    return image_path, mask_path, mask


def _action(action="call", calls=(), deliver=None, thought="t"):
    return react_arm.ReactAction(thought=thought, action=action, deliver=deliver, calls=[
        react_arm.ReactCall(params=[react_arm.ParamValue(name=k, value=v) for k, v in call.items()])
        for call in calls])


class FakeEngine:
    """Answers `replies` in order (then "finish default"); records every content it is sent."""
    is_multimodal = True

    def __init__(self, replies):
        self.replies, self.contents = list(replies), []

    def generate(self, content, **_kwargs):
        self.contents.append(content)
        reply = self.replies.pop(0) if self.replies else _action("finish", deliver="default")
        return {"content": reply, "usage": {"total_tokens": 5}}

    def prompts(self):
        return [c if isinstance(c, str) else c[0] for c in self.contents]


def _arm(monkeypatch, replies, cores=4, deadline=None):
    engine = FakeEngine(replies)
    monkeypatch.setattr(arms, "_engine", lambda _model: engine)
    return arms.ReAct("gpt-5.6-luna", min_area=15, margin=1, margin_frac=0.2, cores=cores, deadline=deadline), engine


def _table(prompt):
    body = prompt.split("Parameters (name | type | default):\n", 1)[1].split("\n\n", 1)[0]
    return [tuple(part.strip() for part in line[2:].split(" | ")) for line in body.splitlines()]


def _shape_verifier(image_path, mask_path, cores=4):
    verifier = CropAgenticVerifier()
    verifier.begin(CropTask(cores=cores))
    verifier.input = input_record({"original_image": str(image_path), "nuclei_mask": str(mask_path),
                                   "min_area": 15, "margin": 1, "margin_frac": 0.2})
    return verifier


def test_parameters_are_shapes_crop_levers_with_its_defaults_and_cell_overrides(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    arm, engine = _arm(monkeypatch, [_action("finish", deliver="default")])
    arm(image_path, mask_path, mask, tmp_path / "react")
    shape = _shape_verifier(image_path, mask_path)
    table = _table(engine.prompts()[0])
    assert [row[0] for row in table] == [*shape.levers, "cell_overrides"]
    assert all(len(row) == 3 for row in table)                       # name, type, default only
    defaults = {**shape.defaults(), "cell_overrides": {}}
    assert {row[0]: json.loads(row[2]) for row in table} == defaults
    assert defaults["margin"] == 1 and defaults["margin_frac"] == 0.2   # resolved as SHAPE resolves them
    types = dict((row[0], row[1]) for row in table)
    assert types["margin_mode"] == 'choice of "fraction", "tight", "mask_bbox"' and types["margin"] == "integer or null"
    assert types["recenter"] == "bool" and types["margin_frac"] == "number"
    assert types["cell_overrides"].startswith('object: cell id -> {margin_mode: choice of "fraction", "tight", '
                                              '"mask_bbox", recenter: bool, auto_tighten: bool, '
                                              'min_containment: number}')
    assert f'"default_{image_path.stem}_cell_<NNNN>"' in types["cell_overrides"]


def test_task_is_shapes_and_no_shape_layer_reaches_the_prompt_or_observations(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    arm, engine = _arm(monkeypatch, [_action(calls=[{"margin_frac": 0.4}, {"margin_mode": "mask_bbox"}]),
                                     _action("finish", deliver="agent_2")])
    result = arm(image_path, mask_path, mask, tmp_path / "react")
    prompts = engine.prompts()
    shape = _shape_verifier(image_path, mask_path)
    for prompt in prompts:
        assert prompt.split("## TASK\n", 1)[1].split("\n\n## INPUT", 1)[0] == shape.task_text()
        assert [line[3:] for line in prompt.splitlines() if line.startswith("## ")] == list(react_arm.SECTIONS)
        assert react_arm.TUNING_LINE in prompt
        for text in SHAPE_ONLY:
            assert text not in prompt, text
    assert "Image: img.tiff, 48 x 48 pixels" in prompts[0] and "min_area: 15 (fixed" in prompts[0]
    # images: only the crop tiles the tool wrote, from the previous round, at most MAX_IMAGES
    first = engine.contents[0]
    assert not isinstance(first, list) or all("_crop." in label for label in first[1::2] if "tool_cache" in label)
    second = engine.contents[1]
    labels = second[1::2]
    assert 0 < len(labels) <= react_arm.MAX_IMAGES
    assert all(label.startswith("IMAGE agent_") and "tool_cache/crop_calls/agent_" in label and "_crop." in label
               for label in labels)
    observations = json.dumps(result.trace["tool_calls"])
    for text in SHAPE_ONLY:
        assert text not in observations, text
    assert all("cell_crop_objects" not in c["output"] for c in result.trace["tool_calls"])
    assert result.trace["tool_calls"][0]["output"]["cell_count"] == len(result.crops)


def test_the_tool_is_called_with_shapes_kwargs_and_cell_overrides_as_a_dict(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    seen = []

    class StubTool:
        def execute(self, **kwargs):
            seen.append(kwargs)
            return {"error": "stub"}

    arm, _engine = _arm(monkeypatch, [_action(calls=[{"cell_overrides": '{"default_img_cell_0001": '
                                                                         '{"margin_mode": "mask_bbox"}}'}])])
    arm.tool = StubTool()
    try:
        arm(image_path, mask_path, mask, tmp_path / "react")
    except RuntimeError as exc:                 # the default call failed: the image fails, with its trace
        assert "the delivered call default failed: stub" in str(exc) and exc.trace["tool_calls"]
    shape = _shape_verifier(image_path, mask_path)
    config = shape.trial_config({**shape.defaults(), "cell_overrides": {}})
    expected = set(shape.trial_params(config)) | {"original_image", "nuclei_mask", "query_cache_dir"}
    assert [set(kw) for kw in seen] == [expected, expected]
    assert seen[0]["output_subdir"] == "crop_calls/default" and seen[1]["output_subdir"] == "crop_calls/agent_1"
    assert seen[0]["query_cache_dir"] == str(tmp_path / "react") and seen[0]["min_area"] == 15
    assert {k: seen[0][k] for k in shape.levers} == shape.defaults() and seen[0]["cell_overrides"] == {}
    assert seen[1]["cell_overrides"] == {"default_img_cell_0001": {"margin_mode": "mask_bbox"}}


def test_cap_is_shapes_concurrency_and_the_default_call_is_round_one(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    over = [{"margin_frac": 0.1 * (i + 1)} for i in range(4)]
    arm, engine = _arm(monkeypatch, [_action(calls=over), _action(calls=over[:3])], cores=6)
    shape = _shape_verifier(image_path, mask_path, cores=6)
    assert arm.cap == shape.concurrency() == 3
    result = arm(image_path, mask_path, mask, tmp_path / "react")
    prompts = engine.prompts()
    assert f"Rounds used: 1 of {BUDGET}." in prompts[0] and "Calls per round: at most 3." in prompts[0]
    assert "4 calls requested; at most 3 calls per round" in prompts[1] and f"Rounds used: 2 of {BUDGET}." in prompts[1]
    assert f"Rounds used: 3 of {BUDGET}." in prompts[2]
    assert [c["call_id"] for c in result.trace["tool_calls"]] == ["default", "agent_1", "agent_2", "agent_3"]
    assert (result.trace["round_cap"], result.trace["budget"], result.trace["default_counts_as_round"]) == (3, BUDGET,
                                                                                                           True)
    assert result.tool_calls == 4 and result.trace["rounds"] == 3


def test_delivery_is_the_agents_finish_else_the_default_call(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    maskcrop = arms.MaskCrop(min_area=15, margin=1, margin_frac=0.2)(image_path, mask_path, mask, tmp_path / "mc")
    boxes = {c["cell_label"]: c["bbox"] for c in maskcrop.crops}

    arm, _ = _arm(monkeypatch, [_action(calls=[{"margin_frac": 0.6}]), _action("finish", deliver="agent_1")])
    finished = arm(image_path, mask_path, mask, tmp_path / "react_finish")
    assert finished.trace["delivery"] == "agent_finish" and finished.trace["delivered_call"] == "agent_1"
    assert {c["cell_label"]: c["bbox"] for c in finished.crops} != boxes
    assert all("crop_calls/agent_1" in Path(c["crop_image_path"]).as_posix() for c in finished.crops)

    arm, engine = _arm(monkeypatch, ["not json"] * (BUDGET + 1))
    unfinished = arm(image_path, mask_path, mask, tmp_path / "react_unfinished")
    assert unfinished.trace["delivery"] == react_arm.SOURCE_UNFINISHED and unfinished.trace["delivered_call"] == "default"
    assert {c["cell_label"]: c["bbox"] for c in unfinished.crops} == boxes   # the tool at its defaults = MaskCrop
    assert len(engine.contents) == BUDGET and unfinished.trace["rounds"] == BUDGET


def test_records_runner_compare_and_export(monkeypatch, tmp_path):
    """The per-image records (TRACE.jsonl, CALLS.json, SELECTED.json, REACT_SUMMARY.json), the runner's record and
    run_meta, compare's rows and the exporter's audit record."""
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    image_path, mask_path, mask = _field(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "selection.json").write_text(json.dumps({"seed": 7, "mask_source": "ground_truth", "selected_images": [
        {"image_id": "img", "image_path": str(image_path), "mask_path": str(mask_path), "gt_path": str(mask_path)}]}),
        encoding="utf-8")
    engine = FakeEngine([_action(calls=[{"margin_frac": 0.4}], thought="less context"),
                         _action("finish", deliver="agent_1", thought="agent_1 is tighter")])
    monkeypatch.setattr(arms, "_engine", lambda _model: engine)
    run_benchmark.run_arm(SimpleNamespace(
        run_dir=run_dir, arm="react", model="gpt-5.6-luna", min_area=15, margin=1, maskcrop_margin_frac=0.2,
        gpt_batch_size=64, cores=8, deadline=None, shape_loop="agentic", cellprofiler_python=None))
    image_dir = run_dir / "react" / "images" / "img"
    assert {p.name for p in (image_dir / "img").iterdir()} == {"TRACE.jsonl", "CALLS.json", "SELECTED.json"}
    assert json.loads((image_dir / "REACT_SUMMARY.json").read_text())["totals"]["calls"] == 2
    selected = json.loads((image_dir / "img" / "SELECTED.json").read_text())
    assert (selected["call_id"], selected["source"], selected["rounds"]) == ("agent_1", "agent_finish", 2)
    meta = json.loads((run_dir / "react" / "run_meta.json").read_text())
    assert (meta["cores"], meta["concurrency_cap"], meta["max_rounds"], meta["react_tool_budget"]) == (8, 4, BUDGET, 29)
    assert meta["react_runner"] == "benchmarks/_shared/react_arm.py"
    assert meta["llm"]["response_schema"] == {"action": "ReactAction"} and meta["llm"]["llm_attempts"] == {"action": 1}
    record = json.loads((run_dir / "react" / "records.jsonl").read_text())
    assert record["status"] == "ok", record.get("error")
    m = record["metrics"]
    assert (m["delivery"], m["rounds_per_image"], m["tool_calls_per_image"], m["crop_coverage"]) == (
        "agent_finish", 2, 2, 1.0)
    assert m["tokens_per_image"] == 0          # the fake engine reports no usage to SHAPE's usage counter
    assert FAKE_KEY not in (run_dir / "react" / "records.jsonl").read_text()

    run_benchmark.compare(SimpleNamespace(run_dir=run_dir))
    rows = [json.loads(line) for line in (run_dir / "compare" / "per_image.jsonl").read_text().splitlines()]
    assert [(r["arm"], r["status"], r["delivery"], r["rounds_per_image"]) for r in rows] == [
        ("react", "ok", "agent_finish", 2)]
    summary = json.loads((run_dir / "compare" / "summary.json").read_text())
    assert summary["arms"]["react"]["rounds_per_image"] == 2.0

    exporter = run_benchmark.load_exporter()
    exporter.export(run_dir, arms=("react",))
    out = json.loads((run_dir / "audit" / "react" / "img.json").read_text())
    assert [s["step"] for s in out["steps"]] == ["tool_call", "llm_call", "tool_call", "llm_call"]
    assert out["steps"][0]["by"].startswith("the benchmark") and out["steps"][1]["rationale"] == "less context"
    assert out["steps"][2]["tool_call"]["params"]["margin_frac"] == 0.4
    assert out["steps"][3]["deliver"] == "agent_1" and out["steps"][1]["prompt_sha256"]
    assert out["final_output"]["delivered_call"] == "agent_1" and out["status"]["fallbacks"] == []


@pytest.mark.skipif("unit_images" not in inspect.signature(react_arm.run_react).parameters,
                    reason="the shared runner has no unit_images kwarg yet")
def test_the_raw_input_image_is_passed_as_the_sessions_unit_image(monkeypatch, tmp_path):
    image_path, mask_path, mask = _field(tmp_path)
    seen = {}
    real = react_arm.run_react

    @functools.wraps(real)                  # keeps the signature the arm checks
    def run_react(adapter, **kwargs):
        seen.update(kwargs)
        return real(adapter, **kwargs)

    monkeypatch.setattr(react_arm, "run_react", run_react)
    arm, _engine = _arm(monkeypatch, [_action("finish", deliver="default")])
    arm(image_path, mask_path, mask, tmp_path / "react")
    assert seen["unit_images"]("img") == [str(image_path)]


def test_python_inputs_hold_the_image_the_given_mask_and_the_call_outputs_never_the_scoring_gt(tmp_path):
    image_path, mask_path, _mask = _field(tmp_path)
    gt_dir = tmp_path / "scoring_gt"
    gt_dir.mkdir()
    tifffile.imwrite(gt_dir / "img_gt.tiff", np.zeros((4, 4), np.uint16))
    adapter = arms.CropToolAdapter(arms.Single_Cell_Cropper_Tool(), image_path, mask_path, tmp_path / "out",
                                   _shape_verifier(image_path, mask_path), private_roots=[gt_dir])
    params = {p.name: p.default for p in adapter.parameters()}
    assert adapter.call(adapter.unit, params, "default")["error"] is None
    paths = adapter.python_inputs(adapter.unit)
    assert paths["inputs/img.tiff"] == image_path and paths["inputs/img_label.tiff"] == mask_path
    call_files = [k for k in paths if k.startswith("calls/default/")]
    assert any(k.endswith(".json") for k in call_files) and len(call_files) > 1    # metadata + its crop files
    assert all(Path(p).is_file() and gt_dir not in Path(p).parents for p in paths.values())
    (gt_dir / "gt.json").write_text("{}", encoding="utf-8")
    adapter.results["leak"] = {"cell_crops_metadata_path": str(gt_dir / "gt.json")}
    with pytest.raises(AssertionError, match="private"):
        adapter.python_inputs(adapter.unit)
