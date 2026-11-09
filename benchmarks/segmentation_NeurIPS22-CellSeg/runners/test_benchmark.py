"""Tests for segmentation_NeurIPS22-CellSeg: the six-arm benchmark (cellseg_arms.py,
run_cellseg_benchmark.py),
the standalone SHAPE runner (run_shape_neurips_cellseg.py) and the segmenter's GT-free quant
score that this benchmark measures (shape/tools/cell_segmenter/quant.py).

No model, LLM, network or dataset: arms, engines, the tool and solve() are fakes; scoring runs
for real on small synthetic masks.
"""

from __future__ import annotations

import inspect
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from skimage.io import imsave

_BENCH = Path(__file__).resolve().parents[1]                # benchmarks/segmentation_NeurIPS22-CellSeg
sys.path[:0] = [str(_BENCH), str(_BENCH / "runners")]
import cellseg_arms as arms  # noqa: E402
import export_run_records as audit  # noqa: E402
import replay_shape_delivery as replay  # noqa: E402
import run_cellseg_benchmark as rb  # noqa: E402
import run_shape_neurips_cellseg as bench  # noqa: E402
from shape.agent.memory import Memory  # noqa: E402
from shape.tools.cell_segmenter import quant  # noqa: E402


def _two_disks(h=40, w=40):
    yy, xx = np.mgrid[0:h, 0:w]
    gt = np.zeros((h, w), dtype=np.uint16)
    gt[(yy - 12) ** 2 + (xx - 12) ** 2 < 36] = 1
    gt[(yy - 28) ** 2 + (xx - 28) ** 2 < 36] = 2
    return gt


def _dataset(tmp_path: Path, stems, h=40, w=40) -> Path:
    """<root>/images/<stem>.png + <root>/labels/<stem>_label.tiff (two disks each)."""
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    gt = _two_disks(h, w)
    for s in stems:
        imsave(root / "images" / f"{s}.png", (gt > 0).astype(np.uint8) * 200, check_contrast=False)
        imsave(root / "labels" / f"{s}_label.tiff", gt, check_contrast=False)
    return root


# =========================================================================== #
# Bands and lists
# =========================================================================== #
def test_the_inputs_in_results_are_tracked_by_git():
    """results/ holds the benchmark's committed inputs (and its reported runs); the repo .gitignore ignores
    data/, logs/ and *.log, so a move or `git add -A` can silently drop files (commit 82327225 did). Fail if
    the inputs are untracked."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("no git")
    inside = subprocess.run(["git", "-C", str(_BENCH), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not a git checkout")
    for name in ("Prompt.txt", "NeurIPS22-CellSeg-cell_image_metrics.csv"):
        r = subprocess.run(["git", "-C", str(_BENCH), "ls-files", "--error-unmatch", f"results/{name}"],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"results/{name} is not tracked by git: {r.stderr.strip()}"


def test_the_committed_csv_gives_three_bands_over_all_1000_images():
    bands = rb.partition_bands(rb.read_band_metric())
    assert {b: len(v) for b, v in bands.items()} == {"hard": 213, "medium": 171, "easy": 616}
    assert all(a < 0.30 for _, a in bands["hard"])
    assert all(0.30 <= a <= 0.60 for _, a in bands["medium"])
    assert all(a > 0.60 for _, a in bands["easy"])
    assert all(not s.endswith((".bmp", ".png", ".tif", ".tiff")) for s, _ in bands["hard"])


def test_lists_are_drawn_once_with_the_seed(tmp_path):
    first = rb.write_lists(tmp_path, n_per_band=20, seed=0)
    assert {b: len(v) for b, v in first.items()} == {"hard": 20, "medium": 20, "easy": 20}
    assert rb.write_lists(tmp_path, n_per_band=5, seed=1) == first       # kept, not redrawn
    other = tmp_path / "other"
    assert rb.write_lists(other, n_per_band=20, seed=0) == first          # same seed, same draw
    assert len({s for v in first.values() for s in v}) == 60


def test_a_band_smaller_than_n_is_taken_whole():
    rows = [("a", 0.1), ("b", 0.2)]
    assert rb.sample_band(rows, 5, seed=0) == rows


# =========================================================================== #
# Arm helpers
# =========================================================================== #
def test_read_image_moves_channels_last_and_drops_alpha(tmp_path):
    import tifffile
    chw = np.random.default_rng(0).integers(0, 255, size=(3, 30, 50)).astype(np.uint8)
    tifffile.imwrite(str(tmp_path / "chw.tif"), chw, photometric="rgb", planarconfig="separate")
    assert arms.read_image(tmp_path / "chw.tif").shape == (30, 50, 3)
    Image.fromarray(np.zeros((30, 50, 4), dtype=np.uint8)).save(tmp_path / "rgba.png")
    assert arms.read_image(tmp_path / "rgba.png").shape == (30, 50, 3)


def test_display_png_caps_the_long_side_and_reports_the_scale():
    png, scale = arms.display_png(np.zeros((500, 2000), dtype=np.uint16), max_px=1000)
    assert Image.open(io.BytesIO(png)).size == (1000, 250)
    assert scale == pytest.approx(0.5)
    _, scale = arms.display_png(np.zeros((100, 200, 3), dtype=np.uint8), max_px=1000)
    assert scale == 1.0


def test_rasterize_polygons_maps_the_shown_frame_back_to_the_image():
    polys = [[[0, 0], [10, 0], [10, 10], [0, 10]],          # shown at scale 0.5 -> 20 x 20 px
             [[20, 20], [30, 20], [30, 30]],                 # a triangle
             [[1, 1], [2, 2]],                               # < 3 vertices: skipped
             [[1, 2, 3], [4, 5]],                            # ragged: skipped
             [[1, 1], [5, 1], [5]]]                          # a bad vertex: skipped
    mask, skipped = arms.rasterize_polygons(polys, (100, 100), scale=0.5)
    assert skipped == 3
    assert sorted(np.unique(mask)) == [0, 1, 2]
    assert mask[5, 5] == 1 and mask[19, 19] == 1 and mask[25, 25] == 0
    assert mask[45, 55] == 2 and mask[55, 45] == 0            # (x, y) -> (col, row), scaled x2


def test_every_llm_arm_is_told_the_prompt_target():
    target = arms.target_text()
    assert "blood smear" in target and "Each whole cell is one instance" in target


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        arms.make_arm("cellpose2")


def _fake_cellpose(monkeypatch, version):
    """A stand-in `cellpose` package (3.x API: models.Cellpose(...).eval -> 4 values)."""
    import types
    from importlib import metadata
    calls = {}

    class _Model:
        def __init__(self, gpu, model_type):
            calls["init"] = (gpu, model_type)

        def eval(self, img, diameter, channels):
            calls["eval"] = (img.shape, img.ndim, diameter, channels)
            return _two_disks(*img.shape[:2]), None, None, np.array([17.5])

    pkg = types.ModuleType("cellpose")
    pkg.core = types.SimpleNamespace(use_gpu=lambda: False)
    pkg.models = types.SimpleNamespace(Cellpose=_Model)
    monkeypatch.setitem(sys.modules, "cellpose", pkg)
    monkeypatch.setitem(sys.modules, "cellpose.core", pkg.core)
    monkeypatch.setitem(sys.modules, "cellpose.models", pkg.models)
    real = metadata.version
    monkeypatch.setattr(metadata, "version", lambda name: version if name == "cellpose" else real(name))
    return calls


def test_cellpose3_arm_runs_cyto3_with_an_estimated_diameter_on_one_channel(tmp_path, monkeypatch):
    calls = _fake_cellpose(monkeypatch, "3.1.1.2")
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(img)
    out = arms.make_arm("cellpose3")(img, tmp_path)
    assert calls["init"] == (False, "cyto3")
    assert calls["eval"] == ((40, 40), 2, None, [0, 0])
    assert out["tool_calls"] == 1 and out["mask"].max() == 2
    assert out["trace"]["estimated_diameter"] == pytest.approx(17.5)
    assert out["trace"]["cellpose"] == "3.1.1.2"


def test_cellpose3_arm_refuses_another_cellpose_major_version(monkeypatch):
    _fake_cellpose(monkeypatch, "4.0.6")
    with pytest.raises(RuntimeError, match="needs cellpose 3.x"):
        arms.make_arm("cellpose3")


# =========================================================================== #
# The LLM arms with fake engines / tool
# =========================================================================== #
class _Engine:
    """Replays `replies` (a schema instance, a JSON string, or a callable(schema))."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def generate(self, content, response_format=None, max_tokens=None):
        self.calls.append(content)
        reply = self.replies.pop(0)
        return {"content": reply(response_format) if callable(reply) else reply}


def test_gpt_arm_rasterizes_the_polygons_it_is_given(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((40, 80), dtype=np.uint8)).save(img)
    reply = json.dumps({"cells": [{"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                                  {"polygon": [[40, 20], [60, 20], [60, 30], [40, 30]]}]})
    arm = arms.GPTPolygons("gpt-x", engine=_Engine([reply]))
    out = arm(img, tmp_path)
    assert out["tool_calls"] == 0 and out["trace"]["n_polygons"] == 2
    assert out["mask"].shape == (40, 80) and out["mask"].max() == 2
    prompt = arm.engine.calls[0][0]
    assert "80 x 40 pixels" in prompt and "blood smear" in prompt


def test_gpt_arm_keeps_the_complete_polygons_of_a_truncated_reply(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((40, 80), dtype=np.uint8)).save(img)
    cut = ('{"cells": [{"polygon": [[0, 0], [10, 0], [10, 10]]}, '
           '{"polygon": [[40, 20], [60, 20], [60, 30]]}, {"polygon": [[1, 2], [3')
    out = arms.GPTPolygons("gpt-x", engine=_Engine([cut]))(img, tmp_path)
    assert out["trace"]["parsed"] is False and out["trace"]["n_polygons"] == 2
    assert out["mask"].max() == 2 and out["trace"]["n_skipped"] == 0


def test_gpt_arm_without_a_parsable_reply_delivers_no_mask(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(img)
    out = arms.GPTPolygons("gpt-x", engine=_Engine(["sorry, no"]))(img, tmp_path)
    assert out["mask"] is None and "reply_head" in out["trace"]


def _fake_solve(mask_path: Path):
    """A stand-in for shape.solver.solve: records its kwargs, returns the real result contract."""
    calls: list[dict] = []

    def _solve(**kwargs):
        calls.append(kwargs)
        memory = Memory()
        memory.set_query(kwargs["question"])
        image_id = Path(kwargs["images"][0]["path"]).stem
        memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment",
                          command="Cell_Segmenter_Tool.execute(...)",
                          result={"mask_path": str(mask_path), "image_id": image_id, "cell_count": 2})
        return {
            "direct_output": "Segmentation completed; cell_count=2", "memory": memory,
            "steps": memory.get_actions(llm_safe=False), "verification_summary": [],
            "stop_reason": "max_steps_reached",
        }

    return _solve, calls


def test_shape_arm_runs_solve_with_the_prompt_and_counts_the_segmenter_calls(tmp_path, monkeypatch):
    gt = _two_disks()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for k in (1, 2):
        np.save(out_dir / f"img0_a{k}.npy", gt)
    fake, calls = _fake_solve(out_dir / "img0_a2.npy")
    monkeypatch.setattr(bench, "solve", fake)
    res = arms.Shape("gpt-x", max_steps=7)(tmp_path / "img0.png", out_dir)
    assert calls[0]["max_steps"] == 7 and calls[0]["enabled_tools"] == bench._SHAPE_MAIN_TOOLS
    assert calls[0]["question"] == bench._default_question()
    assert res["tool_calls"] == 2 and res["mask"].max() == 2
    assert (out_dir / "verification_summary.json").exists()


# =========================================================================== #
# What the LLM arms write for the audit records
# =========================================================================== #
def test_gpt_arm_writes_its_prompt_and_reply(tmp_path):
    img = tmp_path / "x.png"
    Image.fromarray(np.zeros((40, 80), dtype=np.uint8)).save(img)
    reply = json.dumps({"cells": [{"polygon": [[0, 0], [10, 0], [10, 10]]}]})
    arms.GPTPolygons("gpt-x", engine=_Engine([reply]))(img, tmp_path)
    saved = json.loads((tmp_path / "gpt_reply.json").read_text(encoding="utf-8"))
    assert "80 x 40 pixels" in saved["prompt"]
    assert saved["reply"]["cells"][0]["polygon"] == [[0, 0], [10, 0], [10, 10]]
    assert saved["image_shown"]["size"] == [80, 40]


# =========================================================================== #
# export_run_records.py: one audit record per arm x image
# =========================================================================== #
def _audit_run(tmp_path) -> Path:
    run = tmp_path / "run"

    def put(rel, obj):
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")

    def rec(arm, **kw):
        base = {"image": "img0", "arm": arm, "error": None, "seconds": 1.5, "tool_calls": 1, "tokens": 0,
                "ap50": 0.5, "map_50_95": 0.3, "predicted_cell_count": 2, "ground_truth_cell_count": 2,
                "delivered": True, "band": "hard"}
        base.update(kw)
        (run / arm / "hard").mkdir(parents=True, exist_ok=True)
        (run / arm / "hard" / "records.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")

    for arm in ("cellpose_sam", "cellpose3", "microsam", "stardist"):
        rec(arm)
        put(f"{arm}/hard/img0/trace.json", {"setting": arm})
        put(f"{arm}/run_meta.json", {"commit": "abc", "model": "m", "max_steps": 10})
    rec("gpt", tool_calls=0, tokens=900)
    put("gpt/hard/img0/trace.json", {"shown_size": [80, 40], "scale": 1.0, "parsed": True, "n_polygons": 2})
    put("gpt/hard/img0/gpt_reply.json", {"prompt": "P", "reply": {"cells": []}})
    rec("react", tool_calls=1, tokens=500)
    put("react/hard/img0/trace.json", {
        "steps": [{"turn": 1, "thought": "large cells: diameter 80", "action": "segment", "diameter": 80.0,
                   "flow_threshold": 0.4, "cellprob_threshold": 0.0, "min_size": 15, "niter": 0, "augment": False,
                   "tile_overlap": 0.1, "deliver_attempt": 0, "prompt": "PROMPT 1", "images": ["the input image"],
                   "observation": "attempt 1: ok cell_count=2 parameters={}"},
                  {"turn": 2, "parsed": False, "reply_head": "oops"},
                  {"turn": 3, "thought": "good", "action": "finish", "deliver_attempt": 1, "delivered": 1}],
        "attempts": [{"mask_path": "m1.npy", "overlay": "o1.png"}], "delivered_attempt": 1})
    rec("shape", tool_calls=2, tokens=3000)
    put("shape/hard/img0/trace.json", {"stop_reason": "planner_done", "mask_path": "s2.npy"})
    put("shape/hard/img0/verification_summary.json", [{"tool": "Cell_Segmenter_Tool", "per_image": [{
        "final_params": {"diameter": 60.0},
        "stages": [{"stage": "segmentation", "decision": "PASS", "action": "accept", "status": "satisfied",
                    "reason": "plateau", "score": 0.7,
                    "trials": [{"notes": "the input image"},
                               {"params": {"diameter": 30}, "notes": "the tool at its defaults", "parent": 0,
                                "result": {"cell_count": 5, "mask_path": "s1.npy"}, "metrics": {"cell_count": 5}},
                               {"params": {"diameter": 60}, "notes": "the agent's move", "parent": 1,
                                "result": {"cell_count": 2, "mask_path": "s2.npy"}, "metrics": {"cell_count": 2}}],
                    "details": {"preflight": {"specimen_context": "blood_smear"}, "target": {"text": "T"},
                                "admission": {"call": "default", "decision": "RUN"},
                                "measured_problems": {"default": ["x"], "agent_1": []},
                                "assessments": {"agent_1": {"severity": "minor"}},
                                "rounds": [["default"], ["agent_1"]],
                                "plans": [{"round": 2, "question": "round", "analysis": "cells are larger",
                                           "focus_levers": ["diameter"], "moves": [{"changes": [{"lever": "diameter", "value": "60",
                                                                   "reasoning": "measured size"}]}]}],
                                "final_answers": [{"call": "agent_1", "reasoning": "fewer problems"}],
                                "reference_default": {"call": "default", "overridden": True},
                                "focus_history": [["diameter"]],
                                "choice": "agent_choice", "selected": 2, "stop_reason": "no_new_configuration",
                                "verifier_stop": "plateau", "reasoner_failures": 0, "fallback_rounds": 0}}]}]}])
    (run / "logs").mkdir(parents=True, exist_ok=True)
    (run / "logs" / "shape.log").write_text(
        "noise\nCell_Segmenter_Tool: Using device: cpu\n2026 - shape - WARNING - something\n"
        "[shape/hard 1/1] img0: AP50=0.500 mAP=0.300\n", encoding="utf-8")
    return run


def test_export_writes_a_full_record_for_every_arm(tmp_path):
    run = _audit_run(tmp_path)
    summary = audit.export(run)
    assert summary["n_records"] == 7
    out = run / "audit"
    for arm in audit.ARMS:
        r = json.loads((out / arm / "hard" / "img0.json").read_text(encoding="utf-8"))
        assert r["schema"] == audit.SCHEMA and r["arm"] == arm and r["input"]["image_id"] == "img0"
        assert {"provenance", "config", "steps", "final_output", "score", "cost", "status", "sources"} <= r.keys()
    tool = json.loads((out / "stardist" / "hard" / "img0.json").read_text(encoding="utf-8"))
    assert tool["steps"][0]["tool_call"]["arguments"] == {"setting": "stardist"}
    assert tool["provenance"]["commit"] == "abc"
    gpt = json.loads((out / "gpt" / "hard" / "img0.json").read_text(encoding="utf-8"))
    assert gpt["steps"][0]["prompt"] == "P" and gpt["steps"][0]["result"]["reply"] == {"cells": []}


def test_export_react_record_of_the_earlier_design_keeps_each_turn(tmp_path):
    run = _audit_run(tmp_path)
    audit.export(run, arms=("react",))
    r = json.loads((run / "audit" / "react" / "hard" / "img0.json").read_text(encoding="utf-8"))
    t1, t2, t3 = r["steps"]
    assert t1["rationale"] == "large cells: diameter 80" and t1["prompt"] == "PROMPT 1"
    assert t1["tool_call"]["arguments"]["diameter"] == 80.0 and t1["tool_call"]["arguments"]["niter"] is None
    assert t1["tool_result"]["attempt"] == 1 and t1["tool_result"]["mask_path"] == "m1.npy"
    assert t2["parsed"] is False and t2["reply_head"] == "oops"
    assert t3["action"] == "finish" and t3["deliver_attempt_accepted"] == 1
    assert r["final_output"]["delivered_attempt"] == 1 and r["final_output"]["mask_path"] == "m1.npy"
    assert r["final_output"]["how"] == "the attempt the agent named at finish"
    assert r["status"]["unparsed_turns"] == 1


def test_export_shape_record_keeps_the_vrr_steps_and_the_log_events(tmp_path):
    run = _audit_run(tmp_path)
    audit.export(run, arms=("shape",))
    r = json.loads((run / "audit" / "shape" / "hard" / "img0.json").read_text(encoding="utf-8"))
    kinds = [s["step"] for s in r["steps"]]
    assert kinds == ["preflight_read", "target", "tool_call", "admission", "tool_call", "plan", "final_choice"]
    step = r["steps"][4]
    assert step["call"] == "agent_1" and step["parent"] == "default"
    assert step["agent_assessment"] == {"severity": "minor"}
    assert step["tool_result"]["measured_problems"] == []
    plan = r["steps"][5]
    assert plan["rationale"] == "cells are larger" and plan["moves"][0]["changes"][0]["reasoning"] == "measured size"
    assert plan["focus_levers"] == ["diameter"] and r["status"]["focus_history"] == [["diameter"]]
    final = r["steps"][6]
    assert final["agent_answers"][0]["reasoning"] == "fewer problems" and final["selected"] == "agent_1"
    assert r["final_output"]["delivered_call"] == "agent_1" and r["status"]["solver_stop_reason"] == "planner_done"
    assert any("Using device: cpu" in e for e in r["status"]["log_events"])
    assert any("WARNING" in e for e in r["status"]["log_events"])


# =========================================================================== #
# run_arm -> records -> compare
# =========================================================================== #
class _FakeArm:
    """Delivers the GT (or nothing, or raises) per image, with fixed tool calls."""

    def __init__(self, gt, calls, deliver=True, raise_on=()):
        self.gt, self.calls, self.deliver, self.raise_on = gt, calls, deliver, set(raise_on)

    def __call__(self, image_path, out_dir):
        if image_path.stem in self.raise_on:
            raise RuntimeError("boom")
        mask = self.gt if self.deliver else None
        return {"mask": mask, "tool_calls": self.calls, "trace": {"ok": True}}


def _run(tmp_path, stems, arm_specs):
    data = _dataset(tmp_path, stems)
    run_dir = tmp_path / "run"
    (run_dir / "lists").mkdir(parents=True)
    for band in rb.BANDS:
        (run_dir / "lists" / f"{band}.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")
    for name, arm in arm_specs.items():
        rb.run_arm(run_dir, name, data, bands=("hard",), model="m", max_steps=3, arm=arm)
    return run_dir


def test_run_arm_scores_records_and_resumes(tmp_path):
    gt = _two_disks()
    run_dir = _run(tmp_path, ["a", "b", "c"],
                   {"stardist": _FakeArm(gt, 1, raise_on={"b"})})
    recs = rb.load_records(run_dir / "stardist" / "hard" / "records.jsonl")
    by = {r["image"]: r for r in recs}
    assert by["a"]["ap50"] == 1.0 and by["a"]["delivered"] and by["a"]["tokens"] == 0
    assert by["b"]["ap50"] == 0.0 and "boom" in by["b"]["error"] and not by["b"]["delivered"]
    assert (run_dir / "stardist" / "hard" / "a" / "mask.tif").exists()
    meta = json.loads((run_dir / "stardist" / "run_meta.json").read_text())
    assert meta["arm"] == "stardist" and "ended_utc" in meta
    summary = json.loads((run_dir / "stardist" / "hard" / "run_summary.json").read_text())
    assert summary["n_attempted"] == 3 and summary["n_errors"] == 1
    # a second run skips the images already recorded
    rb.run_arm(run_dir, "stardist", run_dir.parent / "data", bands=("hard",), model="m",
               max_steps=3, arm=_FakeArm(gt, 1, raise_on={"a", "b", "c"}))
    assert len(rb.load_records(run_dir / "stardist" / "hard" / "records.jsonl")) == 3


def test_compare_gives_the_six_readouts_and_the_paired_shape_deltas(tmp_path):
    gt = _two_disks()
    stems = [f"img{i}" for i in range(4)]
    half = gt.copy()
    half[half == 2] = 0                                     # one of two cells -> AP50 0.5
    run_dir = _run(tmp_path, stems, {
        "cellpose_sam": _FakeArm(half, 1),
        "gpt": _FakeArm(gt, 0, deliver=False),
        "shape": _FakeArm(gt, 2),
    })
    result = rb.compare(run_dir)
    hard = result["bands"]["hard"]
    assert set(hard) == {"cellpose_sam", "gpt", "shape"}
    assert hard["cellpose_sam"]["ap50"] == pytest.approx(0.5)
    assert hard["shape"]["ap50"] == pytest.approx(1.0)
    assert hard["shape"]["map_50_95"] == pytest.approx(1.0)
    assert hard["shape"]["tool_calls_per_image"] == 2
    assert hard["shape"]["ap50_gain_per_tool_call"] == pytest.approx(0.25)   # (1 - 0.5) / 2
    assert hard["cellpose_sam"]["ap50_gain_per_tool_call"] == pytest.approx(0.0)
    assert hard["gpt"]["ap50"] == 0.0 and hard["gpt"]["ap50_gain_per_tool_call"] is None
    assert hard["shape"]["tokens_per_image"] == 0
    assert hard["shape"]["seconds_per_image"] is not None
    paired = result["paired_vs_shape"]["hard"]["cellpose_sam"]
    assert paired["n"] == 4 and paired["mean_delta_ap50"] == pytest.approx(0.5)
    assert paired["sign_test_p"] == pytest.approx(0.125)
    assert result["bands"]["all"]["shape"]["n"] == 4      # only hard ran
    out = rb.write_compare(run_dir, result)
    for f in ("readouts.csv", "per_image.csv", "paired_vs_shape.csv", "summary.md",
              "compare.json", "readouts.png"):
        assert (out / f).exists(), f
    assert "AP50 gain / tool call" in (out / "summary.md").read_text()


def test_an_arm_dir_without_records_does_not_empty_the_comparison(tmp_path):
    gt = _two_disks()
    run_dir = _run(tmp_path, ["a", "b"], {"cellpose_sam": _FakeArm(gt, 1), "shape": _FakeArm(gt, 1)})
    (run_dir / "microsam").mkdir()                      # e.g. died while loading its model
    result = rb.compare(run_dir)
    assert result["arms"] == ["cellpose_sam", "shape"]
    assert result["arms_without_records"] == ["cellpose3", "microsam", "stardist", "gpt", "react"]
    assert result["bands"]["hard"]["shape"]["n"] == 2
    assert rb.compare(run_dir, ["shape"])["arms"] == ["shape"]


def test_a_rerun_image_starts_from_an_empty_dir(tmp_path):
    gt = _two_disks()
    data = _dataset(tmp_path, ["a"])
    out_dir = tmp_path / "out" / "a"
    out_dir.mkdir(parents=True)
    (out_dir / "a_a5.npy").write_bytes(b"stale")
    rb.run_image(_FakeArm(gt, 1), "shape", data / "images" / "a.png",
                 data / "labels" / "a_label.tiff", out_dir)
    assert not (out_dir / "a_a5.npy").exists() and (out_dir / "mask.tif").exists()


def test_compare_keeps_only_images_every_arm_attempted(tmp_path):
    gt = _two_disks()
    run_dir = _run(tmp_path, ["a", "b"], {"cellpose_sam": _FakeArm(gt, 1), "shape": _FakeArm(gt, 1)})
    recs = run_dir / "shape" / "hard" / "records.jsonl"
    recs.write_text(recs.read_text().splitlines()[0] + "\n")      # shape only attempted "a"
    result = rb.compare(run_dir)
    assert result["bands"]["hard"]["cellpose_sam"]["n"] == 1
    assert result["excluded"]["hard"] == ["b"]


def test_sign_test_and_bootstrap():
    assert rb.sign_test_p([0.1] * 10) == pytest.approx(2 / 2 ** 10)
    assert rb.sign_test_p([0.0, 0.0]) == 1.0
    assert rb.bootstrap_ci([0.1, 0.2, 0.3]) == rb.bootstrap_ci([0.1, 0.2, 0.3])
    lo, hi = rb.bootstrap_ci([0.1, 0.2, 0.3])
    assert 0.1 <= lo <= 0.2 <= hi <= 0.3


# =========================================================================== #
# The standalone SHAPE runner (run_shape_neurips_cellseg.run_batch) -- the summary
# schema other benchmarks consume.
# =========================================================================== #
def _write_synthetic_pair(tmp_path: Path):
    """One image 'img0' (never read: solve() is faked) + a real 2-instance GT TIFF."""
    gt = _two_disks()
    image_dir, label_dir = tmp_path / "images", tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()
    (image_dir / "img0.png").write_bytes(b"")
    imsave(label_dir / "img0_label.tiff", gt, check_contrast=False)
    return image_dir, label_dir, gt


def test_run_batch_summary_schema(tmp_path, monkeypatch):
    image_dir, label_dir, gt = _write_synthetic_pair(tmp_path)
    mask_path = tmp_path / "pred_mask.npy"
    np.save(mask_path, gt)
    fake, _ = _fake_solve(mask_path)
    monkeypatch.setattr(bench, "solve", fake)
    summary = bench.run_batch(image_dir, label_dir, tmp_path / "out", run_baseline=False,
                              model="gpt-4o", max_steps=4)
    assert {"dataset", "task", "image_count", "total_available", "sampling", "verify_refine",
            "metric_type", "predicted_mask_images", "shape_runtime_seconds_total",
            "segmentation_evaluation", "agent_activity", "agent_trace", "efficiency_metrics",
            "efficiency_records"} <= summary.keys()
    per_image = summary["segmentation_evaluation"]["per_image"][0]
    assert {"image_name", "ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50",
            "f1_at_50", "predicted_cell_count", "ground_truth_cell_count", "seconds",
            "escalated", "judge_profile"} <= per_image.keys()
    record = summary["efficiency_records"][0]
    assert {"image_id", "baseline_ap50", "call_ap50", "tokens", "seconds"} <= record.keys()
    assert record["call_ap50"] == pytest.approx([1.0])


def test_both_scorers_agree():
    """run_shape_neurips_cellseg (paths) and run_cellpose_sam_only (arrays, used by
    run_benchmark) implement the same AP definition."""
    gt = _two_disks()
    pred = gt.copy()
    pred[pred == 2] = 0
    pred[0:5, 0:5] = 3
    import run_cellpose_sam_only as cps
    a = cps.evaluate_instance_segmentation(pred, gt)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "p.npy", pred)
        imsave(Path(tmp) / "g.tiff", gt, check_contrast=False)
        b = bench.evaluate_instance_segmentation(str(Path(tmp) / "p.npy"), Path(tmp) / "g.tiff")
    for k in ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50"):
        assert a[k] == pytest.approx(b[k])


# =========================================================================== #
# shape/tools/cell_segmenter/quant.py: the GT-free quality score (tasks 032/033)
# =========================================================================== #
def _synthetic_image_and_clean_masks(seed: int = 0):
    """5 bright disks on a dark, mildly noisy background; clean masks match the disks."""
    rng = np.random.RandomState(seed)
    h, w = 120, 120
    image = np.full((h, w), 20.0)
    centers = [(20, 20), (20, 80), (80, 20), (80, 80), (60, 60)]
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in centers:
        image[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < 12] = 200.0
    image = image + rng.normal(0, 3, size=image.shape)
    masks = np.zeros((h, w), dtype=np.int32)
    for i, (cy, cx) in enumerate(centers):
        masks[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < 12] = i + 1
    return image, masks, centers


def _shatter_into_over_segmentation(masks, centers):
    """The same disks shattered into small fragments + background specks."""
    rng = np.random.RandomState(1)
    h, w = masks.shape
    over = np.zeros_like(masks)
    label = 1
    for cy, cx in centers:
        for dy in range(-12, 12, 3):
            for dx in range(-12, 12, 3):
                y0, x0 = cy + dy, cx + dx
                if 0 <= y0 < h - 2 and 0 <= x0 < w - 2 and np.sqrt(dy ** 2 + dx ** 2) < 12:
                    over[y0:y0 + 2, x0:x0 + 2] = label
                    label += 1
    for _ in range(30):
        y0, x0 = rng.randint(0, h - 2), rng.randint(0, w - 2)
        over[y0:y0 + 2, x0:x0 + 2] = label
        label += 1
    return over


def test_quant_score_orders_clean_above_over_segmented():
    image, clean_masks, centers = _synthetic_image_and_clean_masks()
    clean = quant.quant_score(image, clean_masks)
    over = quant.quant_score(image, _shatter_into_over_segmentation(clean_masks, centers))
    assert clean["quant_score"] > over["quant_score"]
    assert clean["boundary_gradient_adherence"] > over["boundary_gradient_adherence"]


def test_quant_score_orders_clean_above_merged_under_segmentation():
    image, clean_masks, _ = _synthetic_image_and_clean_masks()
    merged = np.zeros_like(clean_masks)
    merged[10:95, 10:95] = 1
    clean = quant.quant_score(image, clean_masks)
    under = quant.quant_score(image, merged)
    assert clean["quant_score"] > under["quant_score"]
    assert clean["within_mask_homogeneity"] > under["within_mask_homogeneity"]


def test_quant_score_is_deterministic_and_bounded():
    image, masks, _ = _synthetic_image_and_clean_masks()
    first = quant.quant_score(image, masks)
    assert first == quant.quant_score(image, masks)
    for k, v in first.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


def test_quant_public_api_takes_no_ground_truth_argument():
    gt_terms = ("gt", "ground_truth", "groundtruth", "label", "annotation", "truth")
    public = [f for name, f in vars(quant).items()
              if inspect.isfunction(f) and not name.startswith("_") and f.__module__ == quant.__name__]
    assert public
    for f in public:
        params = [p.lower() for p in inspect.signature(f).parameters]
        assert not any(t in p for t in gt_terms for p in params), (f.__name__, params)


# =========================================================================== #
# lists --largest (the stress test) and replay_shape_delivery.py (the B6 replay)
# =========================================================================== #
def test_largest_images_keeps_the_biggest_in_their_own_bands(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    import tifffile
    stems = [s for s, _ in rb.read_band_metric()]
    sizes = {stems[0]: (300, 400), stems[1]: (50, 60), stems[700]: (200, 200)}
    for stem, (h, w) in sizes.items():
        Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)).save(root / "images" / f"{stem}.png")
    tifffile.imwrite(str(root / "images" / f"{stems[900]}.tif"), np.zeros((3, 500, 600), dtype=np.uint8),
                     photometric="rgb", planarconfig="separate")
    big = rb.largest_images(root, 2)
    assert big == {stems[900]: 300000, stems[0]: 120000}
    lists = rb.write_lists(tmp_path / "run", 20, 0, largest=big)
    assert sorted(s for v in lists.values() for s in v) == sorted(big)
    assert "pixels=" in (tmp_path / "run" / "lists" / "easy.txt").read_text() + \
        (tmp_path / "run" / "lists" / "hard.txt").read_text() + (tmp_path / "run" / "lists" / "medium.txt").read_text()


def _replay_stage(tmp_path, gt):
    """default = defaults (half the cells), reference = reference (perfect), agent_1 = the agent's pick (empty-ish)."""
    half = gt.copy()
    half[half == 2] = 0
    wrong = np.zeros_like(gt)
    wrong[0:3, 0:3] = 1
    paths = {}
    for name, m in (("default", half), ("reference", gt), ("agent_1", wrong)):
        paths[name] = tmp_path / f"{name}.npy"
        np.save(paths[name], m)
    trial = lambda c, qc: {"score": qc, "result": {"mask_path": str(paths[c])}}  # noqa: E731
    return {"stage": "segmentation",
            "trials": [{}, trial("default", 0.5), trial("reference", 0.6), trial("agent_1", 0.9)],
            "details": {"selected": 3, "admissible": ["default", "reference", "agent_1"],
                        "rounds": [["default"], ["reference"], ["agent_1"]],
                        "reference_default": {"call": "reference", "overridden": True},
                        "final_answers": [{"call": "default"}, {"call": "agent_1"}],
                        "measured_problems": {"default": ["a", "b"], "reference": ["a"], "agent_1": []},
                        "assessments": {"default": {"severity": "major"}, "reference": {"severity": "minor"},
                                        "agent_1": {"severity": "major"}}}}


def test_replay_scores_every_call_and_each_policy_picks_from_the_record(tmp_path):
    gt = _two_disks()
    stage = _replay_stage(tmp_path, gt)
    calls = replay.score_calls(stage, gt)
    assert calls["reference"]["ap50"] == pytest.approx(1.0) and calls["default"]["ap50"] == pytest.approx(0.5)
    assert calls["agent_1"]["n_problems"] == 0 and calls["reference"]["severity"] == "minor"
    p = replay.picks(stage, calls)
    assert p == {"current": "agent_1", "reference": "reference", "agent_first": "default",
                 "agent_assessed": "reference",
                 "fewest_problems": "agent_1", "defaults": "default", "oracle": "reference"}


def test_replay_end_to_end_on_a_run_dir(tmp_path):
    gt = _two_disks()
    data = _dataset(tmp_path, ["img0"])
    run = tmp_path / "run"
    stage = _replay_stage(tmp_path, gt)
    vs_dir = run / "shape" / "hard" / "img0"
    vs_dir.mkdir(parents=True)
    (vs_dir / "verification_summary.json").write_text(
        json.dumps([{"per_image": [{"stages": [stage]}]}]), encoding="utf-8")
    (run / "cellpose_sam" / "hard").mkdir(parents=True)
    (run / "cellpose_sam" / "hard" / "records.jsonl").write_text(json.dumps({"image": "img0", "ap50": 0.5}) + "\n")
    result = replay.replay([run], data)
    s = result["summary"]["all"]
    assert result["n_images"] == 1 and s["cellpose_sam_ap50"] == pytest.approx(0.5)
    assert s["oracle"]["ap50"] == pytest.approx(1.0) and s["current"]["ap50"] < 0.5
    assert s["reference"]["vs_current"] == {"wins": 1, "ties": 0, "losses": 0}
    assert "| oracle | 1.000 |" in replay.markdown(result)


def test_the_scripts_name_the_trials_from_the_records_rounds_old_records_included():
    new = {"rounds": [["default"], ["reference"], ["agent_1", "agent_2"]]}
    old = {"rounds": [["c01"], ["c02"], ["c03"]]}                     # a record from before the role names
    assert audit.call_names(new) == ["input", "default", "reference", "agent_1", "agent_2"]
    assert replay.call_names(old) == ["input", "c01", "c02", "c03"] and replay._call(replay.call_names(old), 2) == "c02"
    assert audit._call(audit.call_names({}), 1) == "trial_1"


def test_lists_exclude_draws_only_images_no_earlier_run_used(tmp_path):
    earlier = rb.write_lists(tmp_path / "earlier", 20, 0)
    used = {s for v in earlier.values() for s in v}
    fresh = rb.write_lists(tmp_path / "fresh", 20, 0, exclude=used)
    assert all(len(v) == 20 for v in fresh.values())
    assert not used & {s for v in fresh.values() for s in v}


# =========================================================================== #
# The set arm: one solve() over a set of images (the segmentation cohort)
# =========================================================================== #
class _FakeSetSolve:
    """solve() over a set: each image's mask (the GT, `calls` attempt files) in the query cache, one segmentation
    step with per_image results and stage records (a measured problem on `flagged`), a cohort record when on;
    `fail`: no mask at all."""

    def __init__(self, gt, calls=2, flagged=(), fail=False):
        self.gt, self.calls, self.flagged, self.fail, self.seen = gt, calls, set(flagged), fail, []

    def __call__(self, images, query_cache_dir, image_workers, segmentation_cohort):
        self.seen.append(([s["image_id"] for s in images], image_workers, segmentation_cohort))
        memory = Memory()
        results, records = [], []
        for spec in images:
            i = spec["image_id"]
            path = None
            if not self.fail:
                for a in range(1, self.calls + 1):
                    path = Path(query_cache_dir) / f"cell_mask_{i}_a{a}.npy"
                    np.save(path, self.gt)
            source = {"parameter_source": "cohort"} if segmentation_cohort else {}
            results.append({"image_id": i, "mask_path": str(path) if path else None, **source})
            problems = ["tiny_mask_fraction 0.2 above 0.15"] if i in self.flagged else []
            records.append({"stages": [{"stage": "segmentation", "status": "satisfied", "no_output": path is None,
                                        "details": {"call": "reference",
                                                    "measured_problems": {"reference": problems}}}]})
        verification = {"tool": "Cell_Segmenter_Tool", "per_image": records}
        if segmentation_cohort:
            verification["cohort"] = {"counts": {"cohort": len(images)}}
        memory.add_action(step_count=1, tool_name="Cell_Segmenter_Tool", sub_goal="segment", command="c",
                          result={"per_image": results}, verification=verification)
        return {"memory": memory, "stop_reason": None, "verification_summary": [verification]}


def _set_run(tmp_path, stems):
    data = _dataset(tmp_path, stems)
    run_dir = tmp_path / "run"
    (run_dir / "lists").mkdir(parents=True)
    half = len(stems) // 2
    (run_dir / "lists" / "hard.txt").write_text("\n".join(stems[:half]) + "\n", encoding="utf-8")
    (run_dir / "lists" / "medium.txt").write_text("\n".join(stems[half:]) + "\n", encoding="utf-8")
    (run_dir / "lists" / "easy.txt").write_text("", encoding="utf-8")
    return data, run_dir


def test_the_cohort_arm_is_one_solve_over_every_band_and_records_each_image(tmp_path):
    """The set arm: one solve() with the cohort on over the lists of every band; each image recorded in its band
    with its parameter source and the admission-standard problems of its delivered call."""
    gt = _two_disks()
    data, run_dir = _set_run(tmp_path, ["a", "b", "c", "d"])
    solve = _FakeSetSolve(gt, calls=2, flagged={"c"})
    rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, workers=4, solve=solve)
    assert solve.seen == [(["a", "b", "c", "d"], 4, True)]               # one solve(), the bands together
    hard = {r["image"]: r for r in rb.load_records(run_dir / "shape_cohort" / "hard" / "records.jsonl")}
    medium = {r["image"]: r for r in rb.load_records(run_dir / "shape_cohort" / "medium" / "records.jsonl")}
    assert set(hard) == {"a", "b"} and set(medium) == {"c", "d"}
    assert hard["a"]["ap50"] == 1.0 and hard["a"]["tool_calls"] == 2 and hard["a"]["parameter_source"] == "cohort"
    assert hard["a"]["problems"] == [] and medium["c"]["problems"] == ["tiny_mask_fraction 0.2 above 0.15"]
    meta = json.loads((run_dir / "shape_cohort" / "set" / "set_meta.json").read_text())
    assert meta["images"] == ["a", "b", "c", "d"] and meta["cohort"] == {"counts": {"cohort": 4}}
    assert (run_dir / "shape_cohort" / "hard" / "a" / "mask.tif").exists()
    rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, solve=solve)    # all recorded: nothing runs
    assert len(solve.seen) == 1
    (run_dir / "shape_cohort" / "medium" / "records.jsonl").unlink()                  # partly recorded: refused
    with pytest.raises(SystemExit, match="remove"):
        rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, solve=solve)


def test_a_cohort_run_without_any_mask_records_nothing(tmp_path):
    """e.g. the gateway down for the whole solve(): no record, so a rerun starts the set again."""
    data, run_dir = _set_run(tmp_path, ["a", "b"])
    with pytest.raises(RuntimeError, match="no image got a mask"):
        rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, solve=_FakeSetSolve(_two_disks(), fail=True))
    assert not list((run_dir / "shape_cohort").glob("*/records.jsonl"))


def test_compare_takes_the_paired_deltas_from_the_reference_set_arm(tmp_path):
    gt = _two_disks()
    half = gt.copy()
    half[half == 2] = 0
    data, run_dir = _set_run(tmp_path, ["a", "b", "c", "d"])
    rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, solve=_FakeSetSolve(gt))
    for band in ("hard", "medium"):
        rb.run_arm(run_dir, "cellpose_sam", data, bands=(band,), model="m", max_steps=3, arm=_FakeArm(half, 1))
    result = rb.compare(run_dir, reference="shape_cohort")
    assert result["arms"] == ["cellpose_sam", "shape_cohort"] and result["reference"] == "shape_cohort"
    assert result["paired_vs_shape"]["all"]["cellpose_sam"]["mean_delta_ap50"] == pytest.approx(0.5)
    assert result["bands"]["all"]["shape_cohort"]["ap50_gain_per_tool_call"] == pytest.approx(0.25)   # 0.5 / 2 calls
    out = rb.write_compare(run_dir, result)
    assert "shape_cohort minus arm" in (out / "summary.md").read_text()


def test_an_earlier_runs_preflight_reads_are_reused_and_this_runs_saved(tmp_path, monkeypatch):
    from shape.agent.verification import image_context
    monkeypatch.setattr(image_context, "_PROFILE_CACHE", {})
    data, run_dir = _set_run(tmp_path, ["a", "b"])
    profile = image_context.default_unknown_profile().model_dump(mode="json")
    cache = tmp_path / "earlier_preflight_cache.json"
    cache.write_text(json.dumps([[["/x/a.png", 1.0, 10], profile]]), encoding="utf-8")
    rb.run_set(run_dir, "shape_cohort", data, model="m", max_steps=3, preflight_cache=cache,
               solve=_FakeSetSolve(_two_disks()))
    assert ("/x/a.png", 1.0, 10) in image_context._PROFILE_CACHE
    meta = json.loads((run_dir / "shape_cohort" / "run_meta.json").read_text())
    assert meta["preflight_reused"] == 1 and meta["preflight_cache"] == str(cache)
    assert json.loads((run_dir / "shape_cohort" / "set" / "set_meta.json").read_text())["preflight_reused"] == 1
    saved = json.loads((run_dir / "shape_cohort" / "set" / "preflight_cache.json").read_text())
    assert saved == [[["/x/a.png", 1.0, 10], profile]]
