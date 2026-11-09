"""Tests of the CellSeg react arm on the shared ReAct runner (cellseg_arms.SegmenterAdapter,
run_cellseg_benchmark.run_react_arm): a fake engine and a stub segmenter; no model, LLM, network or dataset.
The parameters, the round cap and the default-round rule are checked against SHAPE's own segmentation verifier."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.io import imsave

_BENCH = Path(__file__).resolve().parents[1]                # benchmarks/segmentation_NeurIPS22-CellSeg
sys.path[:0] = [str(_BENCH.parents[1]), str(_BENCH), str(_BENCH / "runners")]
import cellseg_arms as arms  # noqa: E402
import export_run_records as audit  # noqa: E402
import run_cellseg_benchmark as rb  # noqa: E402
import run_shape_neurips_cellseg as bench  # noqa: E402
from benchmarks._shared import react_arm as ra  # noqa: E402
from shape.agent.verification.base import BUDGET  # noqa: E402
from shape.agent.verification.verifiers import segmentation as seg  # noqa: E402

SHAPE_LEVERS = ["diameter", "cellprob_threshold", "flow_threshold", "min_size", "niter", "tile_overlap", "augment",
                "resample", "color_class_filter"]


def _two_disks(h=40, w=40):
    yy, xx = np.mgrid[0:h, 0:w]
    gt = np.zeros((h, w), dtype=np.uint16)
    gt[(yy - 12) ** 2 + (xx - 12) ** 2 < 36] = 1
    gt[(yy - 28) ** 2 + (xx - 28) ** 2 < 36] = 2
    return gt


class StubSegmenter:
    """Cell_Segmenter_Tool's interface: the defaults merge the two cells into one mask, any other diameter
    segments both; writes the mask (.npy) and the overlay PNG under query_cache_dir/output_visualizations, and
    records the largest number of calls running at the same time."""
    _sam_server_url = None

    def __init__(self, gt):
        self.gt, self.calls = gt, []
        self.lock = threading.Lock()
        self.active = self.max_active = 0

    def execute(self, image, diameter=30, flow_threshold=0.4, cellprob_threshold=0, min_size=15, niter=None,
                bsize=256, augment=False, resample=True, tile_overlap=0.1, model_type="cpsam", query_cache_dir=None,
                image_id=None, color_class_filter=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({"image_id": image_id, "diameter": diameter, "resample": resample,
                               "color_class_filter": color_class_filter, "niter": niter})
            k = sum(c["image_id"] == image_id for c in self.calls)
        try:
            time.sleep(0.02)
            mask = (self.gt > 0).astype(np.int32) if diameter == 30 else self.gt.astype(np.int32)
            out = Path(query_cache_dir) / "output_visualizations"
            out.mkdir(parents=True, exist_ok=True)
            mask_path, overlay = out / f"cell_mask_{image_id}_a{k}.npy", out / f"cell_overlay_{image_id}_a{k}.png"
            np.save(mask_path, mask)
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(overlay)
            return {"summary": f"{int(mask.max())} cells identified and segmented successfully.",
                    "status": "success", "cell_count": int(mask.max()), "mask_path": str(mask_path),
                    "overlay_path": str(overlay), "attempt": k, "parameters": {"diameter": diameter}}
        finally:
            with self.lock:
                self.active -= 1


def _act(action="call", calls=None, deliver=None):
    return ra.ReactAction(thought="t", action=action, deliver=deliver,
                          calls=[ra.ReactCall(params=[ra.ParamValue(name=k, value=v) for k, v in c.items()])
                                 for c in calls or []])


class FakeEngine:
    """A vision engine replying by `policy(image, n)` (n: the image's reply index); it records every content."""
    is_multimodal = True

    def __init__(self, policy):
        self.policy, self.contents, self.counts = policy, {}, {}
        self.lock = threading.Lock()

    def generate(self, content, response_format=None):
        assert response_format is ra.ReactAction
        image = re.search(r"Image: (\w+)\.png", content[0] if isinstance(content, list) else content).group(1)
        with self.lock:
            n = self.counts.get(image, 0)
            self.counts[image] = n + 1
            self.contents.setdefault(image, []).append(content)
        return {"content": self.policy(image, n),
                "usage": {"total_tokens": 100, "prompt_tokens": 90, "completion_tokens": 10}}


def _finish_a_call_b(image, n):
    """Image a: one call at diameter 12, then finish it; image b: calls every round (no finish)."""
    if image == "a":
        return _act(calls=[{"diameter": 12}]) if n == 0 else _act("finish", deliver="agent_1")
    return _act(calls=[{"diameter": 30}])


def _run_dir(tmp_path, stems=("a", "b")):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    gt = _two_disks()
    for s in stems:
        imsave(root / "images" / f"{s}.png", (gt > 0).astype(np.uint8) * 200, check_contrast=False)
        imsave(root / "labels" / f"{s}_label.tiff", gt, check_contrast=False)
    run = tmp_path / "run"
    (run / "lists").mkdir(parents=True)
    for band in rb.BANDS:
        (run / "lists" / f"{band}.txt").write_text("\n".join(stems) + "\n" if band == "hard" else "",
                                                     encoding="utf-8")
    return run, root, gt


@pytest.fixture(autouse=True)
def _in_process(monkeypatch):
    monkeypatch.delenv("SHAPE_SAM_SERVER_URL", raising=False)


# ---------------------------------------------------------------- the parameters, cap and default rule

def test_parameters_are_shapes_lever_set_with_name_type_default_only():
    verifier = seg.SegmentationAgenticVerifier()
    params = arms.SegmenterAdapter({}, {}, tool=StubSegmenter(_two_disks()), verifier=verifier).parameters()
    assert verifier.levers == SHAPE_LEVERS
    assert [p.name for p in params] == [l for l in SHAPE_LEVERS if l not in ("color_class_filter", "resample")]
    assert arms.SHAPE_ONLY_LEVERS == ("color_class_filter", "resample")
    assert {p.name: p.default for p in params} == {k: v for k, v in verifier.defaults().items()
                                                    if k not in arms.SHAPE_ONLY_LEVERS}
    assert {p.name: p.type for p in params} == {
        "diameter": "number", "cellprob_threshold": "number", "flow_threshold": "number", "min_size": "integer",
        "niter": "integer or null", "tile_overlap": "number", "augment": "bool"}
    assert all(set(vars(p)) == {"name", "type", "default"} for p in params)


def test_round_cap_is_the_verifiers_concurrency_and_the_default_is_round_1(monkeypatch):
    adapter = arms.SegmenterAdapter({}, {}, tool=StubSegmenter(_two_disks()))
    assert adapter.round_cap() == seg.sam_concurrency() == 1                # in-process: one call at a time
    monkeypatch.setattr(seg, "sam_concurrency", lambda: 3)                  # e.g. a SAM server with 3 replicas
    assert adapter.round_cap() == 3
    assert adapter.default_counts_as_round() is True and seg.SegmentationAgenticVerifier.admission_call is True


def test_the_react_arm_is_not_a_per_image_callable():
    with pytest.raises(ValueError, match="run_react_arm"):
        arms.make_arm("react")


# ---------------------------------------------------------------- the arm end to end

def test_react_arm_runs_sessions_delivers_and_records(tmp_path):
    run, data, gt = _run_dir(tmp_path)
    tool, engine = StubSegmenter(gt), FakeEngine(_finish_a_call_b)
    rb.run_react_arm(run, data, model="m", workers=2, engine=engine, tool=tool)

    recs = {r["image"]: r for r in rb.load_records(run / "react" / "hard" / "records.jsonl")}
    a, b = recs["a"], recs["b"]
    # a: the agent's finish delivers agent_1 (diameter 12: both cells)
    assert a["delivery"] == ra.SOURCE_FINISH and a["delivered_call"] == "agent_1" and a["ap50"] == 1.0
    assert a["rounds"] == 2 and a["tool_calls"] == 2 and a["tokens"] == 200 and a["delivered"]
    # b: no finish within the budget -> its default call (the merged mask), recorded
    assert b["delivery"] == ra.SOURCE_UNFINISHED and b["delivered_call"] == "default" and b["ap50"] == 0.5
    assert b["rounds"] == BUDGET and b["tool_calls"] == BUDGET and b["stop"] == "final_not_valid"
    assert engine.counts == {"a": 2, "b": BUDGET}                # BUDGET - 1 rounds + the finish question
    for r in (a, b):
        assert r["error"] is None and r["band"] == "hard" and r["wall_seconds_per_image"] > 0
        assert {"seconds", "map_50_95", "predicted_cell_count", "ground_truth_cell_count"} <= set(r)
        img = run / "react" / "hard" / r["image"]
        assert all((img / f).is_file() for f in ("TRACE.jsonl", "CALLS.json", "SELECTED.json", "mask.tif"))
        assert json.loads((img / "trace.json").read_text())["design"] == rb.REACT_DESIGN

    # the tool: every call at SHAPE's lever defaults except what the agent set; one call at a time (one GPU)
    first = next(c for c in tool.calls if c["image_id"] == "a")
    assert first == {"image_id": "a", "diameter": 30.0, "resample": True, "color_class_filter": None, "niter": None}
    assert [c["diameter"] for c in tool.calls if c["image_id"] == "a"] == [30.0, 12]
    assert tool.max_active == 1

    # the prompt: SHAPE's task text, the lever table (name | type | default), no CARD text; the overlay attached
    content = engine.contents["a"][0]
    prompt = content[0]
    assert bench._default_question().strip() in prompt and "## Target" in prompt
    assert "color_class_filter" not in prompt and "- resample |" not in prompt      # SHAPE's own levers
    assert "- diameter | number | 30.0" in prompt
    assert "Calls per round: at most 1." in prompt and f"Rounds used: 1 of {BUDGET}." in prompt
    assert "Image: a.png (40 x 40 pixels, 1 channel(s), dtype uint8)." in prompt
    card = (Path(seg.__file__).resolve().parents[3] / "tools" / "cell_segmenter" / "CARD.md").read_text(
        encoding="utf-8")
    for line in card.splitlines():
        if line.startswith(("Type:", "Expected cell diameter", "- **", "- Effect")):
            assert line.strip() not in prompt, line
    for banned in ("Range:", "Lower it", "Raise it", "Otsu", "Segments whole cells in phase-contrast"):
        assert banned not in prompt, banned
    assert content[1] == "INPUT IMAGE: a.png" and isinstance(content[2], bytes)      # the raw input, first question
    assert content[3].startswith("IMAGE default: ") and "cell_overlay_a_a1.png" in content[3]
    assert isinstance(content[4], bytes) and len(content) == 5
    later = engine.contents["a"][1]
    assert not any(isinstance(x, str) and x.startswith("INPUT IMAGE") for x in later)   # the first question only

    meta = json.loads((run / "react" / "run_meta.json").read_text())
    assert meta["design"] == rb.REACT_DESIGN and meta["budget"] == BUDGET and meta["round_cap"] == 1
    assert meta["default_counts_as_round"] is True and meta["workers"] == 2
    assert [p["name"] for p in meta["parameters"]] == [l for l in SHAPE_LEVERS if l not in arms.SHAPE_ONLY_LEVERS] and "ended_utc" in meta

    # compare and the audit records read the new records
    result = rb.compare(run, arms=("react",))
    assert result["bands"]["hard"]["react"]["n"] == 2 and result["bands"]["hard"]["react"]["ap50"] == 0.75
    rb.write_compare(run, result)
    audit.export(run, arms=("react",))
    r = json.loads((run / "audit" / "react" / "hard" / "a.json").read_text(encoding="utf-8"))
    assert [s["kind"] for s in r["steps"]] == ["default", "agent", "agent"]
    assert r["steps"][1]["tool_calls"][0]["arguments"]["diameter"] == 12
    assert r["steps"][1]["tool_calls"][0]["tool_result"]["output"]["cell_count"] == 2
    assert r["steps"][2]["action"] == "finish" and r["steps"][2]["deliver_requested"] == "agent_1"
    assert r["final_output"]["delivered_call"] == "agent_1"
    assert r["final_output"]["how"] == "the call the agent named at finish"
    assert r["final_output"]["mask_path"].endswith("cell_mask_a_a2.npy")
    assert r["config"]["budget"] == BUDGET and r["sources"]["missing"] == []
    rb_ = json.loads((run / "audit" / "react" / "hard" / "b.json").read_text(encoding="utf-8"))
    assert rb_["status"]["delivery"] == ra.SOURCE_UNFINISHED and rb_["final_output"]["delivered_call"] == "default"


def test_react_arm_resumes_and_refuses_records_of_the_earlier_design(tmp_path):
    run, data, gt = _run_dir(tmp_path)
    rb.run_react_arm(run, data, model="m", workers=2, engine=FakeEngine(_finish_a_call_b), tool=StubSegmenter(gt))
    engine = FakeEngine(_finish_a_call_b)
    rb.run_react_arm(run, data, model="m", workers=2, engine=engine, tool=StubSegmenter(gt))
    assert engine.counts == {}                                     # every image is recorded: nothing reruns
    assert len(rb.load_records(run / "react" / "hard" / "records.jsonl")) == 2

    old = tmp_path / "old"
    (old / "lists").mkdir(parents=True)
    (old / "react" / "hard").mkdir(parents=True)
    (old / "react" / "run_meta.json").write_text(json.dumps({"arm": "react", "max_steps": 10}), encoding="utf-8")
    (old / "react" / "hard" / "records.jsonl").write_text(json.dumps({"image": "a", "ap50": 1.0}) + "\n",
                                                          encoding="utf-8")
    with pytest.raises(SystemExit, match="earlier ReAct design"):
        rb.run_react_arm(old, data, model="m", engine=FakeEngine(_finish_a_call_b), tool=StubSegmenter(gt))


def test_run_arm_dispatches_react_to_the_shared_runner(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(rb, "run_react_arm", lambda run_dir, data_root, **kw: seen.update(kw))
    rb.run_arm(tmp_path, "react", tmp_path, model="m", max_steps=3, workers=5)
    assert seen["workers"] == 5 and seen["model"] == "m"


def test_a_failed_call_is_an_observation_and_unfinished_delivers_the_default(tmp_path):
    run, data, gt = _run_dir(tmp_path, stems=("a",))

    class Failing(StubSegmenter):
        def execute(self, image, diameter=30, **kw):
            if diameter != 30:
                raise RuntimeError("CUDA out of memory")
            return super().execute(image, diameter=diameter, **kw)

    def policy(image, n):
        return _act(calls=[{"diameter": 12}]) if n == 0 else _act("finish", deliver="agent_1")
    engine = FakeEngine(policy)
    rb.run_react_arm(run, data, model="m", engine=engine, tool=Failing(gt), bands=("hard",))
    [rec] = rb.load_records(run / "react" / "hard" / "records.jsonl")
    assert "error: RuntimeError: CUDA out of memory" in engine.contents["a"][1]   # a text prompt: no image written
    assert rec["delivered_call"] == "default" and rec["delivery"] == ra.SOURCE_UNFINISHED
    assert rec["ap50"] == 0.5 and rec["delivered"] and rec["error"] is None


def test_input_images_are_the_raw_file_or_shapes_rendering_of_it(tmp_path):
    png8, png16, tif = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.tif"
    Image.fromarray(np.full((20, 30), 100, dtype=np.uint8)).save(png8)
    wide = np.zeros((20, 30), dtype=np.uint16)
    wide[5:15, 5:25] = 3000
    Image.fromarray(wide).save(png16)                                        # a 16-bit PNG (mode I;16)
    import tifffile
    tifffile.imwrite(tif, np.stack([wide] * 3), imagej=True)
    dirs = {u: tmp_path / u for u in "abc"}
    for d in dirs.values():
        d.mkdir()
    adapter = arms.SegmenterAdapter({"a": png8, "b": png16, "c": tif}, dirs, tool=StubSegmenter(_two_disks()))
    assert adapter.input_images("a") == [str(png8)] and adapter.input_images("c") == [str(tif)]
    [rendered] = adapter.input_images("b")
    assert rendered == str(dirs["b"] / "input_image.png")
    with Image.open(rendered) as img:
        assert img.mode == "L" and img.size == (30, 20) and np.asarray(img).max() == 255   # stretched, not clipped


def test_python_inputs_hold_the_image_and_the_call_outputs_never_the_labels(tmp_path):
    run, data, gt = _run_dir(tmp_path, stems=("a",))
    code = "import json\nprint(sorted(e['path'] for e in json.load(open('manifest.json'))))"
    steps = [_act(calls=[{"diameter": 12}]), ra.ReactAction(thought="t", action="python", code=code),
             _act("finish", deliver="agent_1")]
    rb.run_react_arm(run, data, model="m", workers=1, engine=FakeEngine(lambda image, n: steps[n]),
                     tool=StubSegmenter(gt), bands=("hard",))
    rows = [json.loads(line) for line in (run / "react" / "hard" / "a" / "TRACE.jsonl").read_text().splitlines()]
    [step] = [r for r in rows if r["kind"] == "python"]
    listed = json.loads(step["python"]["output"].strip().replace("'", '"'))
    assert listed == ["calls/agent_1/cell_mask_a_a2.npy", "calls/agent_1/cell_overlay_a_a2.png",
                      "calls/default/cell_mask_a_a1.npy", "calls/default/cell_overlay_a_a1.png", "inputs/a.png"]
    assert not any("label" in p for p in listed) and step["python"]["violations"] == []
    adapter = arms.SegmenterAdapter({"a": data / "images" / "a.png"}, {"a": tmp_path}, tool=StubSegmenter(gt),
                                    private_roots=[data / "labels"])
    adapter.results[("a", "leak")] = {"mask_path": str(data / "labels" / "a_label.tiff")}
    with pytest.raises(AssertionError, match="private"):
        adapter.python_inputs("a")
