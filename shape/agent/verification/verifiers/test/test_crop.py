"""Task 042 (B): `crop.py::is_crop_keep`, the lenient, KEEP-BIASED "big problem" reject
gate. Co-located with the module under test (committed -- AGENTS.md sec 6 P5).

Covers: a clearly-good crop is kept (the keep-bias test, required by BRIEF/CODING_PROMPT
-- proves the gate does NOT over-reject); each of the 5 "big problem" conditions (blur,
blank, saturated, target-swamped, fragmented) fires on its own dedicated synthetic case;
the fragmentation boundary at n_significant=2 (the exact-fraction bug this task's own
coding log documents catching); the mask-based `target_dominance` override takes
precedence over the pixel-Otsu fallback; determinism.

Second section: `CropAgenticVerifier` (the cropper's `base.AgenticVerifier`, CARD.md "Agentic VRR")
through StepVerifier's wave mode, on the REAL Single_Cell_Cropper_Tool and a fixture shaped like
NeurIPS22-CellSeg (uint16 label mask, 3-channel image whose signal is in channel 1, an object below
min_area), with a scripted fake engine; the admission on the call at the tool's defaults (round 1); the
CARD's levers and loop settings against the code; the solve() path.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np
import pytest
import tifffile

from shape.agent.verification import base
from shape.agent.verification.crop_triage import MaskCheckPositionVerdict, MaskCheckVerdict
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers import crop as cv
from shape.agent.verification.verifiers.crop import (
    CellCondition, CellRule, CellSetting, CropAgenticVerifier, CropMove, CropPlan, CropTask, CropVerifier,
    adapt_crop, input_record, is_crop_keep, tool_default_min_area_margin)
from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
from shape.utils.schemas import (AdmissionAnswer, Expectation, FinalChoice, Hypothesis, LeverChange, VLRRResult,
                                 WaveState)


def test_cell_setting_accepts_typed_values_and_rejects_malformed_json_literals():
    assert CellSetting(key="auto_tighten", value=True).value is True
    CellSetting(key="auto_tighten", value="true")
    CellSetting(key="min_containment", value=0.95)
    with pytest.raises(ValueError, match="JSON boolean"):
        CellSetting(key="auto_tighten", value=":true")
    with pytest.raises(ValueError, match="must be within"):
        CellSetting(key="min_containment", value=1.1)


def _disk_crop(size=64, cy=32, cx=32, r=18, seed=0, lo=40.0, hi=200.0):
    """A bright disk on a dark, mildly noisy background -- one clean, sharp object."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    crop = np.full((size, size), lo)
    crop[np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r] = hi
    crop += rng.normal(0, 3, size=crop.shape)
    return np.clip(crop, 0, 255).astype(np.uint8)


def _single_blob_mask(size=64, cy=32, cx=32, r=18):
    yy, xx = np.mgrid[0:size, 0:size]
    return (np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < r)


def test_clearly_good_crop_is_kept():
    """The keep-bias requirement (CODING_PROMPT rule 2): a clearly-good crop -- sharp,
    single connected target mask, high mask-based target_dominance, normal contrast --
    passes with zero reasons."""
    crop = _disk_crop()
    mask = _single_blob_mask()
    keep, reasons = is_crop_keep(crop, mask, target_dominance=0.95)
    assert keep is True
    assert reasons == []


def test_keeps_almost_all_of_a_good_batch():
    """Keep-bias at batch scale: a set of independently-seeded, clearly-good crops (same
    generator, different noise draws + mask position) should be kept ~100% of the time --
    the gate must not spuriously fire on ordinary crop-to-crop variation."""
    n_reject = 0
    n = 30
    for seed in range(n):
        crop = _disk_crop(seed=seed, cy=28 + (seed % 5), cx=28 + (seed % 7))
        mask = _single_blob_mask(cy=28 + (seed % 5), cx=28 + (seed % 7))
        keep, _ = is_crop_keep(crop, mask, target_dominance=0.9)
        n_reject += int(not keep)
    assert n_reject / n <= 0.10, f"rejected {n_reject}/{n} clearly-good crops -- not keep-biased"


def test_blank_crop_rejected_as_blank():
    blank = np.full((64, 64), 128, dtype=np.uint8)
    keep, reasons = is_crop_keep(blank, _single_blob_mask())
    assert keep is False
    assert "blank" in reasons


def test_saturated_crop_rejected_as_saturated():
    sat = np.full((64, 64), 255, dtype=np.uint8)
    sat[0, 0] = 0  # avoid a perfectly flat (all-identical) image
    keep, reasons = is_crop_keep(sat, _single_blob_mask())
    assert keep is False
    assert "saturated" in reasons


def test_blurred_crop_rejected_as_blur():
    """A heavily blurred version of the same clean disk should score below the lenient
    sharpness floor; the clean version must not."""
    from scipy.ndimage import gaussian_filter

    clean = _disk_crop()
    blurred = np.clip(gaussian_filter(clean.astype(np.float64), sigma=10), 0, 255).astype(np.uint8)
    mask = _single_blob_mask()
    keep_clean, reasons_clean = is_crop_keep(clean, mask, target_dominance=0.95)
    keep_blur, reasons_blur = is_crop_keep(blurred, mask, target_dominance=0.95)
    assert keep_clean is True
    assert keep_blur is False
    assert "blur" in reasons_blur


def test_target_swamped_uses_mask_based_dominance_override():
    """The MASK-based `target_dominance` param (tool.py's real, neighbor-aware metric)
    is authoritative when supplied -- a low value rejects regardless of the pixel-Otsu
    fallback estimate the crop pixels alone would otherwise give."""
    crop = _disk_crop()
    mask = _single_blob_mask()
    keep, reasons = is_crop_keep(crop, mask, target_dominance=0.15)
    assert keep is False
    assert reasons == ["target_swamped"]

    keep2, reasons2 = is_crop_keep(crop, mask, target_dominance=0.85)
    assert keep2 is True
    assert reasons2 == []


def test_fragmented_target_mask_rejected_at_two_significant_pieces():
    """The target's OWN mask split into two comparably-sized pieces (n_significant=2,
    single_object score == 0.5 exactly) must fire "fragmented" at the default threshold
    -- this is the exact boundary the coding log documents a `<` vs `<=` bug on."""
    crop = _disk_crop()
    frag_mask = np.zeros((64, 64), dtype=bool)
    frag_mask[4:20, 4:20] = True
    frag_mask[44:60, 44:60] = True
    keep, reasons = is_crop_keep(crop, frag_mask, target_dominance=0.9)
    assert keep is False
    assert "fragmented" in reasons


def test_single_piece_target_mask_not_fragmented():
    crop = _disk_crop()
    mask = _single_blob_mask()
    keep, reasons = is_crop_keep(crop, mask, target_dominance=0.9)
    assert "fragmented" not in reasons


def test_is_crop_keep_deterministic():
    crop = _disk_crop()
    mask = _single_blob_mask()
    first = is_crop_keep(crop, mask, target_dominance=0.9)
    second = is_crop_keep(crop, mask, target_dominance=0.9)
    assert first == second


def test_thresholds_are_tunable_params():
    """Every threshold is a clearly-named kwarg the caller (human) can override --
    CODING_PROMPT's 'expose the thresholds as clearly-named params' requirement."""
    import inspect

    params = set(inspect.signature(is_crop_keep).parameters)
    assert {"sharpness_min", "single_object_min", "target_dominance_min",
            "min_dynamic_range", "max_saturation"} <= params


# ---- CropAgenticVerifier: the agentic VRR on the real tool ---------------------------------------------------

@pytest.fixture
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)


def _label_mask() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[10:40, 10:40] = 1           # a cell with a touching neighbour (label 2)
    mask[10:40, 40:60] = 2
    mask[70:90, 10:80] = 3           # elongated, isolated
    mask[120:160, 120:160] = 4       # a near pair (2 px gap)
    mask[125:140, 162:185] = 5
    mask[60:100, 130:140] = 6        # L-shaped: its centroid is off its bbox centre
    mask[90:100, 130:170] = 6
    mask[190:193, 5:8] = 7           # below min_area: skipped, but it still takes a regionprops index
    mask[0:20, 170:200] = 8          # touches the image border
    return mask


def _clean_mask() -> np.ndarray:
    """Separated rectangles away from the image border: at the defaults no crop holds neighbour pixels or an
    off-centre target."""
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[20:44, 20:44] = 1
    mask[20:40, 120:150] = 2
    mask[110:140, 30:50] = 3
    mask[120:150, 120:150] = 4
    return mask


def _contaminated_mask() -> np.ndarray:
    """A thin object between two large ones it touches: at the defaults its crop is mostly theirs."""
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[30:80, 75:125] = 1
    mask[80:86, 85:115] = 2
    mask[86:136, 75:125] = 3
    return mask


def _near_threshold_mask() -> np.ndarray:
    """At the defaults, crops just inside the linear loop's round-1 thresholds: a touching pair (each crop holds
    a 7 px band of the other, contamination 0.189) and a square 4 px from the top border (its 44 px crop square
    shifts down inside the image, centroid_offset 0.161)."""
    mask = np.zeros((200, 200), dtype=np.uint16)
    mask[40:70, 40:70] = 1
    mask[40:70, 70:100] = 2
    mask[4:34, 130:160] = 3
    mask[120:150, 120:150] = 4
    return mask


def _corner_mask() -> np.ndarray:
    """The clean mask and an object in the image corner: its crop square sits flush in the corner, the centroid
    off its centre toward both edges (edge-limited)."""
    mask = _clean_mask()
    mask[0:20, 170:200] = 5
    return mask


def _off_centre_mask() -> np.ndarray:
    """The clean mask and an interior right triangle (bbox 40 x 40, the right angle at the bottom left): at the
    defaults its square is centred on the bbox midpoint and the triangle's centroid sits off it, away from every
    image edge."""
    mask = _clean_mask()
    for r in range(60, 100):
        mask[r, 70:71 + (r - 60)] = 5
    return mask


def _small_image_mask() -> np.ndarray:
    """The 24 x 30 image of `single_cell_cropper/test/test_tool.py`: a cell spanning the image's width (bbox
    10 x 30, no square inside the 24 px tall image keeps 0.9 of it: uncroppable), a cell below min_area 50 and a
    cell in the bottom-right corner."""
    mask = np.zeros((24, 30), dtype=np.uint16)
    mask[2:12, 0:30] = 1
    mask[15:22, 3:9] = 2
    mask[14:24, 20:30] = 3
    return mask


def _long_cell_mask() -> np.ndarray:
    """A 40 x 120 image: a 10 x 42 cell longer than the image's height whose largest square keeps 40/42 of it
    (croppable at 0.9, not exempt; in fraction mode at the default min_containment 1 it gets no crop), a 10 x 10
    cell and a 30 x 30 cell."""
    mask = np.zeros((40, 120), dtype=np.uint16)
    mask[5:15, 5:47] = 1
    mask[20:30, 10:20] = 2
    mask[5:35, 60:90] = 3
    return mask


def _make_input(root: Path, signal_channel: int = 1, name: str = "img", mask=None) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    mask = _label_mask() if mask is None else mask
    rng = np.random.RandomState(0)
    planes = []
    for ch in range(3):
        lo, hi = (40.0, 200.0) if ch == signal_channel else (10.0, 15.0)
        plane = np.full(mask.shape, lo)
        plane[mask > 0] = hi
        planes.append(np.clip(plane + rng.normal(0, 3, size=mask.shape), 0, 255).astype(np.uint8))
    image = np.stack(planes, axis=-1)
    assert image[..., 0].max() < image[..., signal_channel].max()
    image_path, mask_path = root / f"{name}.tif", root / f"{name}_pred.tiff"
    tifffile.imwrite(image_path, image)
    tifffile.imwrite(mask_path, mask)
    return image_path, mask_path


def _runner(image: Path, mask: Path, out: Path, calls=None, crash=None):
    tool = Single_Cell_Cropper_Tool()

    def run(params):
        if calls is not None:
            calls.append(dict(params))
        if crash is not None and crash(params):
            raise RuntimeError("the tool crashed")
        result = tool.execute(original_image=str(image), nuclei_mask=str(mask), query_cache_dir=str(out), **params)
        if result.get("error") or not result.get("cell_count"):
            raise RuntimeError(result.get("error") or "no crops")
        return result
    return run


def _rule(conditions, **settings):
    return CellRule(when=[CellCondition(readout=r, op=o, value=json.dumps(v)) for r, o, v in conditions],
                    apply=[CellSetting(key=k, value=json.dumps(v)) for k, v in settings.items()])


def _move(parent="input", expected=(), diagnosis="", rules=(), **levers):
    return CropMove(parent=parent, changes=[LeverChange(lever=k, value=json.dumps(v)) for k, v in levers.items()],
                    cell_rules=list(rules), diagnosis=diagnosis,
                    expected=[Expectation(metric=m, direction=d) for m, d in expected])


class _Engine:
    """Answers each round prompt with the next plan (then no moves: satisfied), a repair prompt with `repair`,
    the final prompt with the next of `final`, the admission question with `admission` (a severity; None: no
    answer, the gateway is down) and the mask check's montages with `mask_check` (default: no verdict)."""

    admission_problem = "neighbour cells fill the crop frames"

    def __init__(self, rounds=(), final=(), repair=None, hypotheses=(), mask_check=None, admission="none"):
        self.rounds, self.final, self.repair, self.hypotheses = list(rounds), list(final), repair, list(hypotheses)
        self.mask_check, self.admission = mask_check, admission
        self.prompts, self.contents, self.threads = [], [], []

    def generate(self, content, response_format=None):
        self.threads.append(threading.get_ident())
        if response_format is MaskCheckVerdict:
            return {"content": self.mask_check(content[0]) if self.mask_check else MaskCheckVerdict(verdicts=[])}
        self.prompts.append(content[0])
        self.contents.append(content)
        if response_format is AdmissionAnswer:
            if self.admission is None:
                raise TimeoutError("gateway down")
            problem = "none" if self.admission == "none" else self.admission_problem
            return {"content": AdmissionAnswer(problem=problem, severity=self.admission, reasoning="t"), "usage": {}}
        if response_format is FinalChoice:
            return {"content": FinalChoice(call=self.final.pop(0) if self.final else "default", reasoning="t"), "usage": {}}
        if response_format is CropPlan:
            if "were rejected" in content[0]:
                return {"content": CropPlan(analysis="repair", moves=list(self.repair or []))}
            moves = self.rounds.pop(0) if self.rounds else []
            return {"content": CropPlan(analysis="t", hypotheses=self.hypotheses, moves=moves), "usage": {}}
        raise RuntimeError(f"unexpected question {response_format}")


def _base_kwargs(image: Path, mask: Path, **extra) -> dict:
    return {"original_image": str(image), "nuclei_mask": str(mask), "min_area": 50, "margin": 1, **extra}


def _run(tmp_path, engine=None, budget=base.BUDGET, cores=4, calls=None, crash=None, log=None, deadline=None,
         workers=None, mask=None, cohort_levers=None, start_params=None, **extra):
    image, mask = _make_input(tmp_path / "data", mask=mask)
    kwargs = _base_kwargs(image, mask, **extra)
    verifier = CropAgenticVerifier(log=log)
    verifier.llm_engine = engine
    verifier.cohort_levers = cohort_levers    # the cropping cohort sets these through verifier_options
    verifier.start_params = start_params
    events = []
    resolution = StepVerifier(max_iters=1 + budget, wave_workers=workers, observer=events.append).resolve_stage(
        verifier, verifier.make_adapter(kwargs), _runner(image, mask, tmp_path / "out", calls, crash), {},
        input_record(kwargs), upstream_ctx={"crop_task": CropTask(cores=cores, deadline=deadline, brief="test")})
    return resolution, verifier, [e["size"] for e in events if e["event"] == "wave"]


def _round_prompts(engine):
    return [p for p in engine.prompts if p.startswith("You tune")]


def _admission_questions(engine):
    return [c for c in engine.contents if c[0].startswith("Decide whether")]


def _bound(tmp_path, rounds, final=("default",)):
    """A finished run, then its verifier re-bound to the final state (for direct validate() calls)."""
    resolution, verifier, _ = _run(tmp_path, _Engine(rounds, final=list(final)))
    state = WaveState(trials=list(resolution.trials), parents=list(resolution.parents), entries=[], remaining=1,
                      context={})
    _, attempts = verifier._bind(state)
    return verifier, attempts


def test_agentic_input_record_cell_ids_match_the_tool(tmp_path):
    image, mask = _make_input(tmp_path / "data")
    record = input_record(_base_kwargs(image, mask))
    result = Single_Cell_Cropper_Tool().execute(original_image=str(image), nuclei_mask=str(mask), min_area=50,
                                                margin=1, query_cache_dir=str(tmp_path / "q"))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert set(record["cells"]) == {m["cell_id"] for m in meta["cell_metadata"]}
    assert "default_img_cell_0006" not in record["cells"] and "default_img_cell_0007" in record["cells"]
    assert record["n_objects"] == 8 and record["n_croppable"] == 7
    assert record["cells"]["default_img_cell_0000"]["nn_gap_px"] == 0.0            # touches its neighbour
    assert 0 < record["cells"]["default_img_cell_0003"]["nn_gap_px"] < 5          # the near pair
    assert record["cells"]["default_img_cell_0007"]["touches_image_border"] is True
    assert record["channels"] == 3 and record["channel_p99"][1] > record["channel_p99"][0]
    assert record["defaults"] == {"margin": 1} and len(record["panel_cells"]) == 7


def test_agentic_the_input_counts_objects_from_the_segmenters_min_size(tmp_path):
    """Without a caller's min_area the loop's input is every object of at least the cell segmenter's default
    min_size (15 px, the tool's own rule -- 50 before 2026-10-01): a 20 px object is a croppable cell, measured,
    and the call at the defaults gives it a crop (the guard counts it); the 9 px object is below min_area."""
    mask = _label_mask()
    mask[190:194, 100:105] = 9                                              # 20 px
    image, mask_path = _make_input(tmp_path / "data", mask=mask)
    kwargs = {"original_image": str(image), "nuclei_mask": str(mask_path)}
    record = input_record(kwargs)
    assert record["min_area"] == 15 and record["defaults"] == {"margin": 25}
    assert record["n_objects"] == 9 and record["n_croppable"] == 8 and "default_img_cell_0006" not in record["cells"]
    assert record["cells"]["default_img_cell_0008"]["area"] == 20
    verifier = CropAgenticVerifier()
    verifier.input = record
    result = Single_Cell_Cropper_Tool().execute(query_cache_dir=str(tmp_path / "q"), **kwargs,
                                                **verifier.trial_params(verifier.defaults()))
    metrics = verifier.quantify(result, {"trial_index": 1})
    assert metrics["n_crops"] == 8 and metrics["n_missing_crops"] == 0
    assert "default_img_cell_0008" in verifier._cells[1]


@pytest.mark.parametrize("name", ["a_cell_mask.png", "a_organoid_mask.png", "a_nuclei_mask.png", "a_pred.tiff"])
def test_agentic_default_min_area_and_margin_follow_the_tools_rule(tmp_path, name):
    import cv2

    image, _ = _make_input(tmp_path / "data")
    mask_path = tmp_path / name
    if name.endswith(".png"):
        cv2.imwrite(str(mask_path), _label_mask().astype(np.uint8))
    else:
        tifffile.imwrite(mask_path, _label_mask())
    result = Single_Cell_Cropper_Tool().execute(original_image=str(image), nuclei_mask=str(mask_path),
                                                query_cache_dir=str(tmp_path / "q"))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    assert tool_default_min_area_margin(str(mask_path)) == (meta["parameters"]["min_area"], meta["parameters"]["margin"])


def test_agentic_trial_0_is_the_input_round_1_the_defaults_and_every_later_call_the_reasoners(tmp_path,
                                                                                            no_retry_wait):
    engine = _Engine([[_move(margin_mode="tight")],
                      [_move("agent_1", rules=[_rule([("neighbor_contamination", ">", 0.1)], margin_mode="mask_bbox",
                                                 auto_tighten=True)])]], final=["agent_2"])
    calls = []
    resolution, verifier, waves = _run(tmp_path, engine, calls=calls)
    assert resolution.trials[0][0] == {} and resolution.trials[0][2]["status"] == cv.INPUT_STATUS
    assert len(calls) == 3 == len(resolution.trials) - 1 and waves == [1, 1, 1]
    defaults = verifier.config_key(verifier.defaults())
    assert verifier.config_key(calls[0]) == defaults and resolution.parents[1] == 0      # round 1: the defaults
    assert resolution.trials[1][1].notes == "the tool at its defaults (the admission call)"
    assert resolution.details["admission"]["decision"] == "RUN"
    assert all(verifier.config_key(p) != defaults for p in calls[1:])    # the reasoner never asked for them
    assert resolution.status == "satisfied" and resolution.reason == "satisfied"
    assert resolution.details["choice"] == "agent_choice" and resolution.parents == [None, 0, 0, 2]
    overrides = resolution.params["cell_overrides"]
    assert overrides and all(v == {"margin_mode": "mask_bbox", "auto_tighten": True} for v in overrides.values())
    assert resolution.params["margin_mode"] == "tight" and resolution.params["min_area"] == 50


def test_four_stage_crop_preserves_rules_and_delivery(tmp_path, no_retry_wait, monkeypatch):
    from shape.agent.verification.test.test_four_stage import CoveringEngine

    monkeypatch.setenv("SHAPE_SEARCH_STRATEGY_CROP", "four_stage")
    engine = CoveringEngine(_Engine([[_move(margin_mode="tight")],
                                    [_move("agent_1", rules=[_rule([], margin_mode="mask_bbox", auto_tighten=True)])]],
                                   final=["agent_2"]))
    resolution, verifier, waves = _run(tmp_path, engine)
    assert waves == [1, 1, 1] and verifier.stop_reason == "satisfied"
    record = resolution.details["search_strategy"]
    assert record["name"] == "four_stage" and record["confirmation"]["call"] == "agent_2"
    assert "cell_rules" in record["coverage"]["changed"]
    assert resolution.params["cell_overrides"]


def test_agentic_levers_defaults_and_values_come_from_the_card():
    verifier = CropAgenticVerifier()
    assert verifier.levers == ["margin_mode", "margin_frac", "margin", "recenter", "tight_buffer_px", "crop_size"]
    assert verifier.defaults() == {"margin_mode": "fraction", "margin_frac": 0.2, "margin": None, "recenter": False,
                                   "tight_buffer_px": 2, "crop_size": None}
    assert verifier.coerce("margin_frac", ".1") == 0.1 and verifier.coerce("margin_frac", '"0.1"') == 0.1
    assert verifier.coerce("margin", "8.4") == 8 and verifier.coerce("crop_size", "None") is None
    assert verifier.coerce("margin_mode", "tight") == "tight" and verifier.coerce("recenter", "true") is True
    for lever, text, message in (("margin", ":4", "is not a JSON value"), ("margin_frac", "2", "outside"),
                                 ("margin_mode", '"square"', "must be one of"), ("recenter", "1", "true or false"),
                                 ("tight_buffer_px", "null", "cannot be null"), ("padding", "4", "is not a CARD lever")):
        with pytest.raises(ValueError, match=message):
            verifier.coerce(lever, text)
    verifier.input = {"defaults": {"margin": 25}}                  # the tool's rule, resolved on trial 0
    assert verifier.defaults()["margin"] == 25 and verifier.coerce("margin", "null") == 25


def test_agentic_round_sizes_are_the_reasoners_within_the_cap(tmp_path, no_retry_wait):
    assert [base.concurrency_cap(c, cv.THREADS_PER_CALL) for c in (1, 2, 4, 8, 32)] == [1, 1, 2, 4, 16]
    logged = []
    engine = _Engine([[_move(margin_mode="tight"), _move(margin_mode="mask_bbox"), _move(recenter=True)],
                      [_move("agent_1", tight_buffer_px=0)]], final=["agent_3"])
    resolution, _, waves = _run(tmp_path, engine, cores=4, log=lambda kind, **f: logged.append(kind))
    assert waves == [1, 2, 1] and "dropped_move" in logged                # cap 2: the third move is dropped
    assert resolution.reason == "satisfied"


def test_agentic_budget_is_a_limit_on_rounds(tmp_path, no_retry_wait):
    engine = _Engine([[_move("default", margin_mode="tight")], [_move("agent_1", tight_buffer_px=0)],
                      [_move("agent_2", recenter=True)]], final=["agent_1"])
    resolution, _, waves = _run(tmp_path, engine, budget=3)              # round 1 is the admission's call
    assert waves == [1, 1, 1] and resolution.reason == "max_iters_reached"
    prompts = _round_prompts(engine)
    assert len(prompts) == 2 and "Rounds: 1 done, at most 2 more; calls so far" in prompts[0]
    assert "Rounds: 2 done, at most 1 more" in prompts[1]


def test_agentic_cell_rules_compile_to_cell_overrides(tmp_path, no_retry_wait):
    # default = the tool at its defaults (the admission's call), agent_1 = recenter for every cell
    verifier, attempts = _bound(tmp_path, [[_move(rules=[_rule([("area", ">=", 0)], recenter=True)])]])
    cells = verifier._cells[1]
    dirty = sorted(c for c, v in cells.items() if v["neighbor_contamination"] > 0.1)
    assert dirty and len(dirty) < len(cells)
    request = verifier.validate({"parent": "default", "cell_rules": [
        {"when": [{"readout": "neighbor_contamination", "op": ">", "value": "0.1"}],
         "apply": [{"key": "margin_mode", "value": '"mask_bbox"'}]},
        {"when": [{"readout": "area", "op": ">=", "value": "0"}], "apply": [{"key": "recenter", "value": "true"}]},
        {"when": [{"readout": "touches_image_border", "op": "==", "value": "true"}],
         "apply": [{"key": "reset", "value": "true"}]},
    ]}, attempts, set())
    overrides = request["params"]["cell_overrides"]
    border = "default_img_cell_0007"
    assert border not in overrides                                       # reset removed it
    assert all(overrides[c]["margin_mode"] == "mask_bbox" for c in dirty if c != border)
    assert all(overrides[c]["recenter"] is True for c in overrides)      # the later rule added its key
    assert request["param_delta"]["cell_rules"][0].endswith(f"({len(dirty)} cells)")
    assert verifier.trial_params(request["params"])["output_subdir"].startswith("crop_calls/img/")
    before = copy.deepcopy(attempts["agent_1"][1]["cell_overrides"])
    assert before and all(ov == {"recenter": True} for ov in before.values())
    child = verifier.validate({"parent": "agent_1", "cell_rules": [
        {"when": [{"readout": "area", "op": ">=", "value": "0"}], "apply": [{"key": "margin_mode", "value": '"tight"'}]},
    ]}, attempts, set())
    assert attempts["agent_1"][1]["cell_overrides"] == before                # the parent's settings are copied, not changed
    assert child["params"]["cell_overrides"] == {c: {"recenter": True, "margin_mode": "tight"} for c in before}
    assert set(child["param_delta"]) == {"cell_rules"}


@pytest.mark.parametrize("rules, message", [
    ([{"when": [{"readout": "area", "op": ">", "value": "0"}], "apply": [{"key": "min_containment", "value": "0.8"}]}],
     "outside"),
    ([{"when": [{"readout": "area", "op": ">", "value": "0"}], "apply": [{"key": "auto_tighten", "value": "true"}]}],
     "auto_tighten acts only in mask_bbox"),
    ([{"when": [{"readout": "area", "op": ">", "value": "1e9"}], "apply": [{"key": "recenter", "value": "true"}]}],
     "matches no cell"),
    ([{"when": [{"readout": "neighbor_contamination", "op": ">", "value": "0.2"}],
       "apply": [{"key": "recenter", "value": "true"}]}], "is not a per-cell readout of the input"),
    ([{"when": [{"readout": "touches_image_border", "op": ">", "value": "0"}],
       "apply": [{"key": "recenter", "value": "true"}]}], "only == applies"),
    ([{"when": [{"readout": "area", "op": ">", "value": "big"}], "apply": [{"key": "recenter", "value": "true"}]}],
     "not a number"),
    ([{"when": [{"readout": "area", "op": ">", "value": "0"}],
       "apply": [{"key": "reset", "value": "true"}, {"key": "recenter", "value": "true"}]}], "alone"),
])
def test_agentic_invalid_rules_are_rejected_with_the_reason(tmp_path, rules, message):
    image, mask = _make_input(tmp_path / "data")
    verifier = CropAgenticVerifier()
    verifier.input = input_record(_base_kwargs(image, mask))
    with pytest.raises(ValueError, match=message):
        verifier.validate({"parent": "input", "cell_rules": rules}, {}, set())


def _bare_verifier(root: Path, mask=None) -> tuple[CropAgenticVerifier, Path, Path]:
    """A verifier bound to an input record only (no loop), for direct `validate` / `config_key` calls."""
    image, mask_path = _make_input(root, mask=mask)
    verifier = CropAgenticVerifier(log=lambda kind, **f: verifier.events.append((kind, f.get("text"))))
    verifier.events = []
    verifier.input = input_record(_base_kwargs(image, mask_path))
    return verifier, image, mask_path


def _tool_boxes(verifier, image: Path, mask_path: Path, config: dict, out: Path) -> dict:
    """The real tool's crop box of every croppable cell under `config` (None: no crop)."""
    result = Single_Cell_Cropper_Tool().execute(original_image=str(image), nuclei_mask=str(mask_path),
                                                query_cache_dir=str(out), **verifier.trial_params(config))
    meta = json.loads(Path(result["cell_crops_metadata_path"]).read_text())
    boxes = {m["cell_id"]: m["crop_bbox"] for m in meta["cell_metadata"]}
    return {cid: boxes.get(cid) for cid in verifier.input["cells"]}


def _every_cell(**settings) -> list[dict]:
    return [{"when": [{"readout": "area", "op": ">=", "value": "0"}],
             "apply": [{"key": k, "value": json.dumps(v)} for k, v in settings.items()]}]


def test_agentic_a_rule_moving_a_cell_out_of_mask_bbox_drops_its_inherited_auto_tighten(tmp_path):
    """The second user-approved relaxation (2026-10-01): a cell a rule moves out of mask_bbox drops the
    auto_tighten it inherited from the parent instead of rejecting the move (the pilot: 7 such rejections on one
    image, then no_valid_move) -- logged (`dropped_auto_tighten`) and stated in the move's param_delta; the other
    cells keep theirs, the parent's settings are unchanged. Still rejected: an auto_tighten the move itself sets
    outside mask_bbox, an inherited one on cells the lever change alone moves out and a rule matching no cell."""
    verifier, _, _ = _bare_verifier(tmp_path / "data")
    cells = sorted(verifier.input["cells"])
    parent = {**verifier.defaults(), "margin_mode": "mask_bbox",
              "cell_overrides": {c: {"auto_tighten": True} for c in cells}}
    attempts = {"agent_1": ("default", parent, VLRRResult("PASS", 0.0, {}))}
    small = {"readout": "area", "op": "<", "value": "400"}                  # the 15 x 23 cell alone
    to_tight = {"when": [small], "apply": [{"key": "margin_mode", "value": '"tight"'}]}
    request = verifier.validate({"parent": "agent_1", "cell_rules": [to_tight]}, attempts, set())
    overrides = request["params"]["cell_overrides"]
    assert overrides["default_img_cell_0004"] == {"margin_mode": "tight"}
    assert all(overrides[c] == {"auto_tighten": True} for c in cells if c != "default_img_cell_0004")
    assert attempts["agent_1"][1]["cell_overrides"]["default_img_cell_0004"] == {"auto_tighten": True}
    note = ("1 cell(s) a rule moved out of mask_bbox: the auto_tighten they had in agent_1 dropped (it acts only in "
            "mask_bbox)")
    assert request["param_delta"] == {"cell_rules": ['area < 400 -> margin_mode="tight" (1 cells)'],
                                      "auto_tighten_dropped": note}
    assert verifier.events == [("dropped_auto_tighten", f"{note}: ['default_img_cell_0004']")]
    own = {**to_tight, "apply": [*to_tight["apply"], {"key": "auto_tighten", "value": "true"}]}
    for move in ({"parent": "agent_1", "cell_rules": [own]},
                 {"parent": "agent_1", "changes": [{"lever": "margin_mode", "value": '"tight"'}]}):
        with pytest.raises(ValueError, match="auto_tighten acts only in mask_bbox mode"):
            verifier.validate(move, attempts, set())
    with pytest.raises(ValueError, match="matches no cell"):
        verifier.validate({"parent": "agent_1", "cell_rules": [
            {**to_tight, "when": [{**small, "op": ">", "value": "1e9"}]}]}, attempts, set())


def test_agentic_a_cell_rule_without_a_condition_selects_every_cell(tmp_path):
    """Vacuous truth: a cell rule with an empty `when` matches every cell, from the input and from a call; the
    move text and the schema's `when` description say so."""
    verifier, _, _ = _bare_verifier(tmp_path / "data")
    cells = sorted(verifier.input["cells"])
    rule = {"when": [], "apply": [{"key": "recenter", "value": "true"}]}
    request = verifier.validate({"parent": "input", "cell_rules": [rule]}, {}, set())
    assert request["params"]["cell_overrides"] == {c: {"recenter": True} for c in cells}
    assert request["param_delta"]["cell_rules"] == [f"every cell -> recenter=true ({len(cells)} cells)"]
    parent = {**verifier.defaults(), "cell_overrides": {cells[0]: {"recenter": True}}}
    attempts = {"agent_1": ("default", parent, VLRRResult("PASS", 0.0, {}))}
    child = verifier.validate({"parent": "agent_1", "cell_rules": [{"when": [], "apply": [
        {"key": "margin_mode", "value": '"tight"'}]}]}, attempts, set())
    assert child["params"]["cell_overrides"] == {c: {"recenter": True, "margin_mode": "tight"} if c == cells[0]
                                                 else {"margin_mode": "tight"} for c in cells}
    assert "conditions hold (no condition: every cell)" in verifier.move_text()
    assert "none = every cell" in json.dumps(CropPlan.model_json_schema())


def test_agentic_the_dropped_auto_tighten_is_run_and_shown_in_the_calls_table(tmp_path, no_retry_wait):
    """End to end: the move that moves a cell with an inherited auto_tighten out of mask_bbox runs (no
    no_valid_move), the cell is cropped in tight mode, and the next round's calls table states the drop in the
    call's change."""
    logged = []
    engine = _Engine([[_move("default", margin_mode="mask_bbox", rules=[_rule([("area", ">=", 0)], auto_tighten=True)])],
                      [_move("agent_1", rules=[_rule([("area", "<", 400)], margin_mode="tight")])]], final=["agent_2"])
    resolution, verifier, waves = _run(tmp_path, engine, log=lambda kind, **f: logged.append(kind))
    assert waves == [1, 1, 1] and resolution.reason == "satisfied" and "dropped_auto_tighten" in logged
    assert verifier._cells[3]["default_img_cell_0004"]["margin_mode"] == "tight"
    row = _table_rows(_round_prompts(engine)[2])["agent_2"]
    assert row["change"]["auto_tighten_dropped"].startswith("1 cell(s) a rule moved out of mask_bbox")


def test_agentic_a_configuration_that_crops_every_cell_alike_is_rejected(tmp_path, no_retry_wait):
    # default = the tool at its defaults (the admission's call), agent_1 = tight
    verifier, attempts = _bound(tmp_path, [[_move(margin_mode="tight")]])
    repeat = r"this configuration was already run \(or failed\) or is already in this round"
    for levers in ({"margin": 9}, {"margin_frac": 0.4}):                  # inert in tight mode
        with pytest.raises(ValueError, match=repeat):
            verifier.validate({"parent": "agent_1", "changes": [{"lever": k, "value": json.dumps(v)}
                                                           for k, v in levers.items()]}, attempts, set())
    with pytest.raises(ValueError, match=repeat):                          # margin 1 -> 2: margin_frac dominates
        verifier.validate({"parent": "input", "changes": [{"lever": "margin", "value": "2"}]}, attempts, set())
    with pytest.raises(ValueError, match=repeat):                          # a cap above every crop: inert
        verifier.validate({"parent": "input", "changes": [{"lever": "crop_size", "value": "100"}]}, attempts, set())
    first = verifier.validate({"parent": "input", "changes": [{"lever": "crop_size", "value": "80"}]}, attempts, set())
    with pytest.raises(ValueError, match=repeat):                          # the same half-side cap (40)
        verifier.validate({"parent": "input", "changes": [{"lever": "crop_size", "value": "81"}]}, attempts,
                          {first["config_key"]})
    with pytest.raises(ValueError, match="already"):
        verifier.validate({"parent": "agent_1", "changes": [{"lever": "margin_mode", "value": '"tight"'}]}, attempts, set())
    with pytest.raises(ValueError, match="no lever changed relative to agent_1"):
        verifier.validate({"parent": "agent_1"}, attempts, set())
    assert verifier.validate({"parent": "agent_1", "changes": [{"lever": "tight_buffer_px", "value": "0"}]},
                             attempts, set())["param_delta"] == {"tight_buffer_px": 0}


def test_agentic_the_configuration_key_is_the_crop_boxes_the_tool_cuts(tmp_path):
    """`config_key` keys a configuration by the crops it would cut, computed before the call by the tool's own
    per-cell geometry (`cell_boxes`). On the fixture (a touching pair, a near pair, an L, a 15 x 23 cell, a
    border cell) and for every kind of setting -- the defaults, tight at buffer 0, mask_bbox with auto_tighten
    everywhere, a crop_size cap that drops cells, per-cell recenter / tight with min_containment / auto_tighten
    -- the computed boxes are exactly the real tool's, dropped cells included, and each configuration has its own
    key."""
    verifier, image, mask_path = _bare_verifier(tmp_path / "data")
    cells, d = sorted(verifier.input["cells"]), verifier.defaults()
    kinds = [{"recenter": True}, {"margin_mode": "tight", "min_containment": 0.9},
             {"margin_mode": "mask_bbox", "auto_tighten": True}, {}]
    configs = {"defaults": d, "tight0": {**d, "margin_mode": "tight", "tight_buffer_px": 0},
               "auto": {**d, "margin_mode": "mask_bbox", "cell_overrides": {c: {"auto_tighten": True} for c in cells}},
               "cap": {**d, "crop_size": 30},
               "mixed": {**d, "margin_frac": 0.4, "cell_overrides": {c: kinds[i % 4] for i, c in enumerate(cells)}}}
    for name, config in configs.items():
        assert verifier.cell_boxes(config) == _tool_boxes(verifier, image, mask_path, config, tmp_path / name), name
    assert len({verifier.config_key(c) for c in configs.values()}) == len(configs)
    assert sum(box is None for box in verifier.cell_boxes(configs["cap"]).values()) == 3     # the cap drops cells


def test_agentic_one_verifier_on_two_masks_gives_each_its_own_boxes_and_key(tmp_path):
    """One verifier bound in turn to two inputs (the same image name, so the same cell ids) with the same
    configurations gives each mask its own boxes -- the real tool's -- and the keys a fresh verifier gives it;
    back on the first mask, its first keys. On the clean mask auto_tighten everywhere cuts nothing (one key with
    plain mask_bbox), on the contaminated one it cuts (two keys)."""
    inputs = {}
    for name, mask in (("clean", _clean_mask()), ("dirty", _contaminated_mask())):
        image, mask_path = _make_input(tmp_path / name, mask=mask)
        inputs[name] = (image, mask_path, input_record(_base_kwargs(image, mask_path)))
    verifier = CropAgenticVerifier()
    verifier.input = inputs["clean"][2]
    plain = {**verifier.defaults(), "margin_mode": "mask_bbox"}
    every_cell = {f"default_img_cell_{i:04d}": {"auto_tighten": True} for i in range(4)}     # either mask's cells
    configs = {"plain": plain, "tightened": {**plain, "cell_overrides": every_cell}}
    keys = {}
    for step, name in enumerate(("clean", "dirty", "clean")):
        image, mask_path, record = inputs[name]
        fresh = CropAgenticVerifier()
        verifier.input = fresh.input = record
        for kind, config in configs.items():
            assert verifier.cell_boxes(config) == _tool_boxes(verifier, image, mask_path, config,
                                                              tmp_path / f"{step}_{kind}"), (name, kind)
        now = {kind: verifier.config_key(config) for kind, config in configs.items()}
        assert now == {kind: fresh.config_key(config) for kind, config in configs.items()}, name
        assert keys.setdefault(name, now) == now, name
    assert keys["clean"]["plain"] == keys["clean"]["tightened"]
    assert keys["dirty"]["plain"] != keys["dirty"]["tightened"]
    assert keys["dirty"]["plain"] != keys["clean"]["plain"]


def test_agentic_configurations_that_crop_every_cell_alike_are_one_call_and_one_pixel_is_another(tmp_path):
    """The exact key (2026-10-01; the pilot ran 6 of 69 calls whose crops repeated an earlier call's), each verdict
    checked against the real tool's crops:
    - auto_tighten on cells it does not cut (the clean mask: no neighbour in any square) is the call without it;
      on a cell it cuts (the thin cell between two large ones) it is another call;
    - tight and plain mask_bbox cut the same square when every longer bbox side is even: one call;
    - a near-duplicate is not merged: with one 21 px (odd) longer side, mask_bbox's square for that cell (21 + 2 x 2)
      is one pixel smaller than tight's (2 x (11 + 2)) -- every other crop alike -- two calls."""
    repeat = r"this configuration was already run \(or failed\) or is already in this round"
    to_tight = {"parent": "input", "changes": [{"lever": "margin_mode", "value": '"tight"'}]}

    clean, image, mask_path = _bare_verifier(tmp_path / "clean", _clean_mask())
    plain, tight = {**clean.defaults(), "margin_mode": "mask_bbox"}, {**clean.defaults(), "margin_mode": "tight"}
    attempts = {"agent_1": ("default", plain, VLRRResult("PASS", 0.0, {}))}
    with pytest.raises(ValueError, match=repeat):
        clean.validate({"parent": "agent_1", "cell_rules": _every_cell(auto_tighten=True)}, attempts, set())
    with pytest.raises(ValueError, match=repeat):
        clean.validate(to_tight, attempts, set())
    plain_boxes = _tool_boxes(clean, image, mask_path, plain, tmp_path / "plain")
    tightened = {**plain, "cell_overrides": {c: {"auto_tighten": True} for c in clean.input["cells"]}}
    assert _tool_boxes(clean, image, mask_path, tightened, tmp_path / "tightened") == plain_boxes
    assert _tool_boxes(clean, image, mask_path, tight, tmp_path / "tight") == plain_boxes

    dirty, image, mask_path = _bare_verifier(tmp_path / "dirty", _contaminated_mask())
    plain = {**dirty.defaults(), "margin_mode": "mask_bbox"}
    cut = dirty.validate({"parent": "agent_1", "cell_rules": _every_cell(auto_tighten=True)},
                         {"agent_1": ("default", plain, VLRRResult("PASS", 0.0, {}))}, set())
    uncut_boxes = _tool_boxes(dirty, image, mask_path, plain, tmp_path / "uncut")
    cut_boxes = _tool_boxes(dirty, image, mask_path, cut["params"], tmp_path / "cut")
    assert [c for c in cut_boxes if cut_boxes[c] != uncut_boxes[c]] == ["default_img_cell_0001"]

    mask = _clean_mask()
    mask[160:181, 60:75] = 5                                                # 21 x 15
    odd, image, mask_path = _bare_verifier(tmp_path / "odd", mask)
    plain, tight = {**odd.defaults(), "margin_mode": "mask_bbox"}, {**odd.defaults(), "margin_mode": "tight"}
    near = odd.validate(to_tight, {"agent_1": ("default", plain, VLRRResult("PASS", 0.0, {}))}, set())
    assert near["config_key"] != odd.config_key(plain)
    a = _tool_boxes(odd, image, mask_path, plain, tmp_path / "odd_plain")
    b = _tool_boxes(odd, image, mask_path, tight, tmp_path / "odd_tight")
    assert [c for c in a if a[c] != b[c]] == ["default_img_cell_0004"]
    (r0, _, r1, _), (s0, _, s1, _) = a["default_img_cell_0004"], b["default_img_cell_0004"]
    assert (r1 - r0, s1 - s0) == (25, 26)


def test_agentic_rejected_moves_get_one_repair_question_naming_the_accepted_ones(tmp_path, no_retry_wait):
    invalid = CropMove(changes=[LeverChange(lever="margin_frac", value="7")])
    engine = _Engine([[invalid, _move(margin_mode="tight")]], repair=[_move(margin_mode="tight"), _move(recenter=True)],
                     final=["default"])
    resolution, _, waves = _run(tmp_path, engine, cores=4)
    repair = [p for p in engine.prompts if "were rejected" in p]
    assert len(repair) == 1 and "do not return them again" in repair[0] and "'margin_mode': 'tight'" in repair[0]
    assert "outside" in repair[0] and waves == [1, 2]   # the repair's repeat is refused, its new move fills the slot


def test_agentic_a_failed_reasoner_question_is_tried_again(tmp_path, no_retry_wait):
    class Flaky(_Engine):
        failures = 1

        def generate(self, content, response_format=None):
            if response_format is CropPlan and self.failures:
                self.failures -= 1
                raise TimeoutError("gateway timeout")
            return super().generate(content, response_format)

    logged = []
    resolution, verifier, waves = _run(tmp_path, Flaky([[_move(margin_mode="tight")]], final=["default"]),
                                       log=lambda kind, **f: logged.append(kind))
    assert waves == [1, 1] and logged.count("reasoner_error") == 1 and verifier.reasoner_failures == 0


def test_agentic_a_failed_prediction_is_a_fact_and_does_not_stop_the_search(tmp_path, no_retry_wait):
    engine = _Engine([[_move("default", margin_mode="mask_bbox", expected=[("mean_neighbor_contamination", "up")])],
                      [_move("agent_1", tight_buffer_px=0)]], final=["agent_2"])
    resolution, _, waves = _run(tmp_path, engine)
    assert waves == [1, 1, 1] and resolution.reason == "satisfied"
    third = _round_prompts(engine)[2]
    row = next(json.loads(line) for line in third.splitlines() if line.startswith('{"call": "agent_1"'))
    expected, observed = row["prediction"]["mean_neighbor_contamination"].split(" -> ")
    assert expected == "up" and "down" in observed
    assert "beyond noise" in json.dumps(row["vs_parent"]) and "vs_defaults_call" not in row   # the parent is default


def test_agentic_call_rows_drop_only_the_duplicated_fields(tmp_path, no_retry_wait):
    """A row has no crops_by_mode (n_crops x the frac_mode_* readouts) and no vs_defaults_call when its parent is
    the defaults call (vs_parent is that comparison); a call from another parent keeps vs_defaults_call, and
    every row keeps its comparison to the parent, prediction, guard, Pareto membership and diagnosis."""
    engine = _Engine([[_move("default", margin_mode="tight", diagnosis="d2")], [_move("agent_1", recenter=True)],
                      [_move(margin_frac=0.1)]], final=["agent_3"])
    resolution, verifier, _ = _run(tmp_path, engine)
    state = WaveState(trials=list(resolution.trials), parents=list(resolution.parents), entries=[], remaining=1,
                      context={})
    inp, attempts = verifier._bind(state)
    rows = {r["call"]: r for r in verifier._rows(inp, attempts)}
    assert [(c, attempts[c][0]) for c in rows] == [("default", "input"), ("agent_1", "default"), ("agent_2", "agent_1"), ("agent_3", "input")]
    assert "vs_defaults_call" not in rows["default"] and "vs_defaults_call" not in rows["agent_1"]
    assert rows["agent_2"]["vs_defaults_call"]["call"] == rows["agent_3"]["vs_defaults_call"]["call"] == "default"
    assert "noise" in json.dumps(rows["agent_1"]["vs_parent"]) and rows["agent_1"]["diagnosis"] == "d2"
    for call, row in rows.items():
        assert "crops_by_mode" not in row
        assert {"vs_parent", "prediction", "guard", "pareto", "diagnosis", "readouts"} <= set(row)
        cells, readouts = verifier._cells[verifier._labels[call]], attempts[call][2].metrics
        for mode in cv.MARGIN_MODES:              # what crops_by_mode counted, the readouts already carry
            count = sum(1 for v in cells.values() if v["margin_mode"] == mode)
            assert readouts[f"frac_mode_{mode}"] * readouts["n_crops"] == pytest.approx(count)


def _negated(fact: str) -> tuple:
    """A vs-parent fact as numbers, negated: (mean, 2SE, beyond, cells up, cells down, n) with up and down swapped."""
    m = re.fullmatch(r"([+-]\d+\.\d{4}) \+/- (\d+\.\d{4}) \((beyond|within) noise; (\d+) cells up, (\d+) down of "
                     r"(\d+)\)", fact)
    mean, se, beyond, up, down, n = m.groups()
    return -float(mean), float(se), beyond, int(down), int(up), int(n)


def _parsed(fact: str) -> tuple:
    mean, se, beyond, down, up, n = _negated(fact)
    return -mean, se, beyond, up, down, n


def test_agentic_the_compact_calls_table_keeps_every_number(tmp_path, no_retry_wait, monkeypatch):
    """The calls table names the readouts once, in its first line, and each row's readouts are the values in that
    order (every readout of every call recoverable); each prediction is one item per stated metric, "expected ->
    observed", with the observed change from the parent; the vs-parent facts leave out mean_neighbor_contamination,
    the exact complement of the kept mean_target_dominance (its fact is that one negated, cells up and down
    swapped); the final prompt carries the same table."""
    expected = [("mean_target_dominance", "up"), ("mean_target_retention", "preserve")]
    engine = _Engine([[_move("default", expected=expected, margin_mode="tight")],
                      [_move("agent_1", expected=expected, recenter=True),
                       _move("default", expected=expected, rules=[_rule([("neighbor_contamination", ">", 0.05)],
                                                                    margin_mode="mask_bbox")])]], final=["agent_2"])
    resolution, verifier, waves = _run(tmp_path, engine, cores=6)
    assert waves == [1, 1, 2]
    state = WaveState(trials=list(resolution.trials), parents=list(resolution.parents), entries=[], remaining=1,
                      context={})
    inp, attempts = verifier._bind(state)
    table = verifier._table(inp, attempts)
    head, note = table.splitlines()[:2]
    names = json.loads(head.split("Readouts, in this order (null = not measured for that call): ", 1)[1])
    assert 'prediction = {metric: "expected -> observed"}' in note and "leave out mean_neighbor_contamination" in note
    rows = _table_rows(table)
    assert list(rows) == list(attempts) == ["default", "agent_1", "agent_2", "agent_3"]   # in run order
    for call, row in rows.items():
        metrics = attempts[call][2].metrics
        assert set(names) == set(metrics) and len(row["readouts"]) == len(names)
        assert dict(zip(names, row["readouts"])) == pytest.approx({k: round(v, 4) for k, v in metrics.items()})
        assert not any(k in json.dumps(row["readouts"]) for k in names)
    monkeypatch.setattr(cv, "_FACT_COMPLEMENT", {})                 # the facts as they were, every key readout
    for call, row in rows.items():
        parent = attempts[call][0]
        if parent == "input":
            continue
        facts = verifier._facts(verifier._labels[parent], verifier._labels[call])
        assert set(facts) == set(verifier.key_readouts)
        assert row["vs_parent"] == {k: v for k, v in facts.items() if k != "mean_neighbor_contamination"}
        assert _parsed(facts["mean_neighbor_contamination"]) == _negated(facts["mean_target_dominance"])
        vres, pvres = attempts[call][2], attempts[parent][2]
        assert list(row["prediction"]) == [m for m, _ in expected]
        for metric, direction in expected:
            d = verifier._score(vres, metric) - verifier._score(pvres, metric)
            got_direction, observed = row["prediction"][metric].split(" -> ")
            assert got_direction == direction and observed.startswith(f"{d:+.4f} (")
    assert rows["agent_2"]["vs_defaults_call"]["call"] == "default" and "mean_neighbor_contamination" not in rows["agent_2"][
        "vs_defaults_call"]
    final = next(p for p in engine.prompts if p.startswith(verifier.choice_text()))
    assert f"Readouts, in this order (null = not measured for that call): {json.dumps(names)}" in final and note in final
    assert all(isinstance(r["readouts"], list) for r in _table_rows(final).values())
    assert "values in the order the first\nline below names" in _round_prompts(engine)[-1]


def test_agentic_the_hypothesis_list_is_shown_in_the_next_round(tmp_path, no_retry_wait):
    hypothesis = Hypothesis(id="H1", statement="touching neighbours dominate the contamination", status="open")
    engine = _Engine([[_move(margin_mode="tight")], [_move("default", recenter=True)]], final=["agent_1"],
                     hypotheses=[hypothesis])
    resolution, _, _ = _run(tmp_path, engine)
    prompts = _round_prompts(engine)
    assert "touching neighbours dominate" not in prompts[0] and "touching neighbours dominate" in prompts[1]
    assert resolution.details["hypotheses"][0]["id"] == "H1"


def test_agentic_every_prompt_carries_the_input_image_and_the_last_rounds_panels(tmp_path, no_retry_wait):
    engine = _Engine([[_move(margin_mode="tight"), _move(margin_mode="mask_bbox")], [_move("agent_1", recenter=True)]],
                     final=["agent_2"])
    _run(tmp_path, engine)
    rounds = [c for c in engine.contents if c[0].startswith("You tune")]
    labels = [[x for x in content if isinstance(x, str)][1:] for content in rounds]
    assert labels[0] == ["INPUT IMAGE (the mask's object boundaries in green):", "CROP PANEL OF default:"]
    assert labels[1][1:] == ["CROP PANEL OF agent_1:", "CROP PANEL OF agent_2:"]
    assert all(isinstance(x, bytes) and x[:4] == b"\x89PNG" for content in rounds for x in content[1:]
               if not isinstance(x, str))
    final = next(c for c in engine.contents if c[0].startswith("Choose the delivered"))
    assert "INPUT IMAGE (the mask's object boundaries in green):" in final
    assert any(isinstance(x, str) and x.startswith("CROP PANEL OF") for x in final)


def _assert_information_only(text: str) -> None:
    for rule_text in ("you should", "prefer", "try", "tighten when", "widen", "is better", "recommended", "avoid",
                      "must not", "never use"):
        assert not re.search(rf"\b{rule_text}\b", text.lower()), rule_text
    for gt in ("ap50", "gt_adjudication", "_label", "labels/"):
        assert gt not in text.lower(), gt
    for leak in (r"task \d+", "maskcrop", "baseline", "benchmark", "naive"):  # nothing about what it is compared with
        assert not re.search(rf"\b{leak}\b", text.lower()), leak


def test_agentic_prompts_carry_information_only(tmp_path, no_retry_wait):
    engine = _Engine([[_move(margin_mode="tight")], [_move("default", recenter=True)]], final=["agent_1"])
    _run(tmp_path, engine)
    _assert_information_only("\n".join(engine.prompts))
    prompts = _round_prompts(engine)
    assert "### margin_mode" in prompts[0] and "Per-cell keys" in prompts[0] and '"margin": 1' in prompts[0]
    assert '"vs_parent"' in prompts[1] and '"prediction"' in prompts[1] and "nn_gap_px" in prompts[0]
    assert "asked once more" in prompts[0] and "If no call qualifies, you are not asked" in prompts[0]


def test_agentic_the_round_prompt_shows_the_time_left_when_the_caller_knows_it(tmp_path, no_retry_wait):
    engine = _Engine([[_move(margin_mode="tight")]], final=["default"])
    _run(tmp_path, engine, deadline=time.time() + 5400)
    assert "time left before the stage must end" in _round_prompts(engine)[0] and "1.50 h" in _round_prompts(engine)[0]
    engine = _Engine([[_move(margin_mode="tight")]], final=["default"])
    _run(tmp_path / "b", engine)
    assert "time left" not in _round_prompts(engine)[0]


def test_agentic_an_empty_round_stops_the_search(tmp_path, no_retry_wait):
    resolution, verifier, waves = _run(tmp_path, _Engine([[_move(margin_mode="tight")]], final=["default"]))
    assert waves == [1, 1] and verifier.stop_reason == "satisfied" and resolution.status == "satisfied"


# ---- the admission on the call at the tool's defaults (round 1) ----------------------------------------------

def test_agentic_without_a_measured_problem_the_agent_is_asked_and_none_delivers_the_defaults_call(tmp_path,
                                                                                                  no_retry_wait):
    calls = []
    engine = _Engine([[_move("default", margin_mode="tight")]], final=["agent_1"], admission="none")
    resolution, verifier, waves = _run(tmp_path, engine, calls=calls, mask=_clean_mask())
    assert waves == [1] and len(calls) == 1 and verifier.config_key(calls[0]) == verifier.config_key(verifier.defaults())
    assert _round_prompts(engine) == [] and len(engine.prompts) == 1        # the admission question only
    assert resolution.status == "satisfied" and resolution.reason == "no_tuning_needed"
    assert resolution.verdict.action == "accept" and verifier.stop_reason == "no_tuning_needed"
    details = resolution.details
    assert details["call"] == "default" and details["choice"] == "no_tuning_needed" and details["rounds"] == [["default"]]
    assert details["admission"] == {"call": "default", "rule": "SKIP", "problems": [], "decision": "SKIP",
                                    "answer": {"reasoning": "t", "problem": "none", "severity": "none",
                                               "confidence": "high"}}
    # default's crops are delivered: its metadata (and sidecars) copied under a fresh name into tool_cache
    delivered = Path(details["delivered_metadata_path"])
    source = Path(resolution.result["call_metadata_path"])
    assert delivered.parent.resolve() == (tmp_path / "out" / "tool_cache").resolve()
    assert source.parent.resolve() == (tmp_path / "out" / "tool_cache" / resolution.params["output_subdir"]).resolve()
    assert resolution.result["cell_crops_metadata_path"] == str(delivered)
    assert json.loads(delivered.read_text()) == json.loads(source.read_text())
    # the question: the checks, the call's row, the input image and default's crop panel
    question, *attachments = _admission_questions(engine)[0]
    assert question.startswith(
        "Decide whether this image needs the tuning of a fixed single-cell cropping tool.\n"
        "The tool ran once at its defaults (default); its measurements found no problem in that output\n"
        "(the guard: a crop for every object at min_area that a square crop can hold at target_retention 0.9 (the "
        "uncroppable_cells are exempt), every crop square (height = width), every crop's target_retention at least "
        "0.9; per crop: neighbor_contamination at most 0.2, centroid_offset at most 0.2 (a crop whose centroid_offset "
        "the image edge forces, edge_limited, not counted)). Name the main problem you see in the evidence below, "
        "or none,\n")
    assert "levers, at most 7 more round(s)); minor or none delivers default as it is." in question
    assert '{"call": "default", "parent": "input"' in question
    assert [x for x in attachments if isinstance(x, str)] == ["INPUT IMAGE (the mask's object boundaries in green):",
                                                              "CROP PANEL OF default:"]
    assert all(isinstance(x, bytes) and x[:4] == b"\x89PNG" for x in attachments if not isinstance(x, str))
    _assert_information_only(question)


@pytest.mark.parametrize("admission", ["minor", "no answer", "no reasoner"])
def test_agentic_a_minor_problem_or_no_answer_delivers_the_defaults_call(tmp_path, no_retry_wait, admission):
    engine = None if admission == "no reasoner" else _Engine(
        [[_move("default", margin_mode="tight")]], final=["agent_1"], admission=None if admission == "no answer" else admission)
    if engine is None:
        from shape.engine.errors import NoEngineError
        with pytest.raises(NoEngineError):
            _run(tmp_path, engine, mask=_clean_mask())
        return
    resolution, _, waves = _run(tmp_path, engine, mask=_clean_mask())
    decided = resolution.details["admission"]
    if admission == "no answer":
        assert waves == [1] and resolution.reason == "execution_error"
        assert decided["decision"] == "NOT_SEARCHED"
        assert resolution.details["execution_errors"][0]["question"] == "admission"
        return
    assert waves == [1] and resolution.reason == "no_tuning_needed" and resolution.details["call"] == "default"
    assert decided["rule"] == decided["decision"] == "SKIP" and decided["problems"] == []
    assert (decided["answer"] or {}).get("severity") == ("minor" if admission == "minor" else None)
    if engine is not None:                              # no answer: every try of the question failed
        assert _round_prompts(engine) == []
        assert len(_admission_questions(engine)) == (1 if admission == "minor" else base.REASONER_ATTEMPTS)


def test_agentic_the_cohort_mode_delivers_the_group_consensus_without_an_agent_round(tmp_path, no_retry_wait):
    """The cropping cohort's no-search mode (`cohort_levers` set, `solver._cropping_cohort`): the shared loop runs
    the call at the defaults, its admission and -- after a RUN -- the ONE reference call at the group's consensus
    levers, then `plan_round` returns none ("cohort_applied") and `choose` delivers that reference unasked. No agent
    round is asked, and the delivered call carries the consensus margin_frac, not the default."""
    calls, engine = [], _Engine(admission="major")
    resolution, _, _ = _run(tmp_path, engine, calls=calls, cohort_levers={"margin_frac": 0.5})
    assert resolution.reason == "cohort_applied"
    assert _round_prompts(engine) == []                     # the agent's tuning prompt is never built...
    ran = {c.get("margin_frac") for c in calls}
    assert 0.5 in ran and 0.2 in ran                        # the defaults call (0.2) then the consensus call (0.5)
    assert resolution.params["margin_frac"] == 0.5          # delivered at the consensus, not the default


def test_agentic_a_routed_image_starts_from_the_group_consensus(tmp_path, no_retry_wait):
    """A cohort image the admission flags is routed to its own VRR, but from the group's consensus as the tool
    defaults (`start_params`): the call at the defaults (round 1) already carries the consensus margin_frac."""
    calls = []
    _run(tmp_path, _Engine(admission="none"), calls=calls, start_params={"margin_frac": 0.5})
    assert calls and calls[0].get("margin_frac") == 0.5


def test_agentic_a_major_problem_the_agent_names_starts_the_tuning(tmp_path, no_retry_wait):
    engine = _Engine([[_move("default", margin_mode="tight")]], final=["agent_1"], admission="major")
    resolution, _, waves = _run(tmp_path, engine, mask=_clean_mask())
    admission = resolution.details["admission"]
    assert admission["rule"] == "SKIP" and admission["decision"] == "RUN" and admission["problems"] == []
    assert admission["answer"]["severity"] == "major" and waves == [1, 1]
    assert ("Admission: the tool ran once at its defaults (default); the tuning started on your admission answer, "
            f"{_Engine.admission_problem} (major).\nTool defaults") in _round_prompts(engine)[0]
    assert "\nRounds: 1 done, at most 7 more" in _round_prompts(engine)[0]
    assert resolution.reason == "satisfied" and resolution.details["call"] == "agent_1"


@pytest.mark.parametrize("mask, problem", [
    (_contaminated_mask, "1 of 3 crops with neighbor_contamination above 0.2"),
    (_off_centre_mask, "1 of 5 crops with centroid_offset above 0.2"),
], ids=["contamination", "centroid_offset"])
def test_agentic_a_crop_the_linear_rule_would_rerun_starts_the_tuning_unasked(tmp_path, no_retry_wait, mask,
                                                                               problem):
    engine = _Engine([[_move("default", margin_mode="tight")]], final=["agent_1"])
    resolution, _, waves = _run(tmp_path, engine, mask=mask())
    assert resolution.details["admission"] == {"call": "default", "rule": "RUN", "problems": [problem], "answer": None,
                                               "decision": "RUN"}
    assert _admission_questions(engine) == [] and waves == [1, 1]
    assert (f"Admission: the tool ran once at its defaults (default); the tuning started on {problem}.\n"
            "Tool defaults") in _round_prompts(engine)[0]
    assert "\nRounds: 1 done, at most 7 more" in _round_prompts(engine)[0]
    # the thresholds are the linear loop's own (CropVerifier's round-1 geometry rule)
    assert CropAgenticVerifier._retune_rule() == [
        ("neighbor_contamination", CropVerifier._MASK_BBOX_CONTAMINATION_MIN),
        ("centroid_offset", CropVerifier._OFF_CENTER_MAX)]


def test_agentic_crops_just_inside_the_linear_rules_thresholds_leave_the_decision_to_the_agent(tmp_path,
                                                                                              no_retry_wait):
    engine = _Engine([[_move("default", margin_mode="tight")]], final=["agent_1"], admission="none")
    resolution, verifier, waves = _run(tmp_path, engine, mask=_near_threshold_mask())
    crops = list(verifier._crops[1].values())                           # the values the linear rule reads
    contamination = max(c["neighbor_contamination"] for c in crops)
    offset = max(c["centroid_offset"] for c in crops)
    assert 0.15 < contamination <= CropVerifier._MASK_BBOX_CONTAMINATION_MIN
    assert 0.15 < offset <= CropVerifier._OFF_CENTER_MAX
    assert resolution.details["admission"]["problems"] == [] and len(_admission_questions(engine)) == 1
    assert waves == [1] and resolution.reason == "no_tuning_needed"


def test_agentic_admission_problems_count_the_crops_strictly_above_each_threshold():
    """`admission_problems` on synthetic per-crop values: a value at a threshold is no problem, one just above
    it is; a crop above both thresholds counts in both strings."""
    contamination, offset = CropVerifier._MASK_BBOX_CONTAMINATION_MIN, CropVerifier._OFF_CENTER_MAX   # both 0.2
    verifier, vres = CropAgenticVerifier(), VLRRResult("PASS", 0.0, {})
    verifier._index_of[id(vres)] = 1

    def problems(*values):
        verifier._crops[1] = {f"cell_{i}": {"neighbor_contamination": c, "centroid_offset": o}
                              for i, (c, o) in enumerate(values)}
        return verifier.admission_problems(None, vres)

    assert problems((contamination, offset), (0.0, 0.0)) == []                        # at the thresholds
    assert problems((contamination + 1e-4, 0.0), (0.0, offset)) == [                   # 0.2001: just above
        "1 of 2 crops with neighbor_contamination above 0.2"]
    assert problems((0.0, offset + 1e-4), (contamination, 0.0)) == ["1 of 2 crops with centroid_offset above 0.2"]
    assert problems((0.5, 0.5), (contamination, offset), (0.3, 0.0)) == [               # the first counts in both
        "2 of 3 crops with neighbor_contamination above 0.2", "1 of 3 crops with centroid_offset above 0.2"]


def test_agentic_edge_limited_is_whether_the_image_edge_forces_the_offset():
    """`_edge_limited` (human decision 2026-09-30): no square of the crop's side inside the image places the
    centroid closer to the crop centre -- per axis the centroid within EDGE_TOLERANCE_PX (1 px) of the centre, or
    the box flush (within 1 px) against the image edge on the side the centroid lies toward, at least one flush."""
    edge = cv._edge_limited
    assert cv.EDGE_TOLERANCE_PX == 1
    assert edge((0, 10, 20, 30), (5.0, 20.0), (100, 100))          # flush top, the centroid above the centre
    assert edge((1, 10, 21, 30), (6.0, 20.0), (100, 100))          # 1 px from the edge counts as flush
    assert not edge((2, 10, 22, 30), (7.0, 20.0), (100, 100))      # 2 px: the square could move up
    assert edge((80, 80, 100, 100), (95.0, 95.0), (100, 100))      # flush in the bottom-right corner
    assert not edge((0, 40, 20, 60), (15.0, 50.0), (100, 100))     # flush top, but the centroid lies below the centre
    assert not edge((0, 40, 20, 60), (5.0, 45.0), (100, 100))      # flush top, off-centre sideways away from an edge
    assert not edge((40, 40, 60, 60), (45.0, 50.0), (100, 100))    # interior and off-centre
    assert not edge((40, 40, 60, 60), (50.5, 49.6), (100, 100))    # centred: no offset to force


def test_agentic_admission_does_not_count_an_edge_limited_crop_as_off_centre():
    """`admission_problems` on synthetic per-crop values: an off-centre crop that is edge-limited is left out of
    the count and named in the text; an off-centre crop that is not is counted."""
    verifier, vres = CropAgenticVerifier(), VLRRResult("PASS", 0.0, {})
    verifier._index_of[id(vres)] = 1

    def problems(*crops):           # (centroid_offset, edge_limited) per crop
        verifier._crops[1] = {f"cell_{i}": {"cell_id": f"cell_{i}", "neighbor_contamination": 0.0,
                                            "centroid_offset": o} for i, (o, _) in enumerate(crops)}
        verifier._cells[1] = {f"cell_{i}": {"edge_limited": e} for i, (_, e) in enumerate(crops)}
        return verifier.admission_problems(None, vres)

    assert problems((0.5, True), (0.0, False)) == []                   # the one off-centre crop is edge-limited
    assert problems((0.5, True), (0.4, False), (0.1, True)) == [
        "1 of 3 crops with centroid_offset above 0.2; 1 more are edge-limited"]
    assert problems((0.4, False)) == ["1 of 1 crops with centroid_offset above 0.2"]


def test_agentic_a_border_crop_flush_to_the_edge_is_not_counted_and_one_that_could_be_centred_is(tmp_path,
                                                                                                  no_retry_wait):
    """On the real tool at the defaults: the corner object's square sits flush in the corner with the centroid off
    its centre toward both edges -- edge-limited, not counted; the interior triangle is off-centre and counted; a
    top-border triangle whose square is flush at the top but whose centroid also lies off-centre sideways, where
    the square could move, is counted too. The admission's text names the left-out crop; per crop `edge_limited` is
    a readout and a cell-rule condition (== only), `n_edge_limited` the call's count."""
    mask = _off_centre_mask()                # labels 1-4 separated rectangles, 5 the interior triangle
    mask[0:20, 170:200] = 6                  # the corner object
    for r in range(0, 30):                   # a top-border triangle, the right angle at the top left
        mask[r, 60:90 - r] = 7
    engine = _Engine([[_move("default", margin_mode="tight")]], final=["agent_1"])
    resolution, verifier, _ = _run(tmp_path, engine, mask=mask)
    assert resolution.details["admission"]["problems"] == [
        "2 of 7 crops with centroid_offset above 0.2; 1 more are edge-limited"]
    cells = verifier._cells[1]
    assert {cid[-4:]: v["edge_limited"] for cid, v in cells.items() if v["centroid_offset"] > 0.2} == {
        "0004": False, "0005": True, "0006": False}
    assert resolution.details["calls"]["default"]["readouts"]["n_edge_limited"] == 1
    holds = verifier._condition({"readout": "edge_limited", "op": "==", "value": "true"}, "default")
    assert [cid for cid in sorted(verifier.input["cells"]) if holds(cid)] == ["default_img_cell_0005"]
    with pytest.raises(ValueError, match="only == applies"):
        verifier._condition({"readout": "edge_limited", "op": ">", "value": "0"}, "default")


def test_agentic_an_uncroppable_object_is_exempt_and_the_small_image_is_admissible(tmp_path, no_retry_wait):
    """The human decision of 2026-09-30: an object no square crop can hold at RETENTION_FLOOR -- the small image's
    full-width cell, its bbox 30 px long in a 24 px tall image, the largest square keeps 0.8 of it -- is flagged per
    cell on trial 0, listed in the input facts, and exempt from the guard: the call at the defaults crops every
    other object at min_area, its readouts count the exempt object apart (n_exempt_uncroppable 1, n_missing_crops
    0), and it is admissible and delivered (before, no call could be)."""
    engine = _Engine([], final=["default"])
    resolution, verifier, _ = _run(tmp_path, engine, mask=_small_image_mask())
    record = verifier.input
    assert record["n_uncroppable"] == 1 and record["n_croppable"] == 2      # label 2 is below min_area 50
    assert record["cells"]["default_img_cell_0000"]["uncroppable"] is True
    assert record["cells"]["default_img_cell_0002"]["uncroppable"] is False
    readouts = resolution.details["calls"]["default"]["readouts"]
    assert readouts["n_lost_to_geometry"] == 1 and readouts["n_exempt_uncroppable"] == 1
    assert readouts["n_missing_crops"] == 0
    assert resolution.details["admissible"] == ["default"] and resolution.details["call"] == "default"
    assert resolution.status == "satisfied"
    assert '"uncroppable_cells": ["default_img_cell_0000"]' in _round_prompts(engine)[0]


def test_agentic_a_croppable_objects_missing_crop_fails_even_beside_an_uncroppable_one(tmp_path):
    """The exemption is per cell: a croppable object without a crop fails the guard even when the image holds an
    uncroppable one, and the call's report names each. At the defaults (fraction mode, min_containment 1) the
    10 x 42 cell, longer than the 40 px image height but kept at 40/42 by the largest square (croppable), gets no
    crop beside the uncroppable 10 x 50 cell (0.8). (No call can give an uncroppable cell a crop: every crop keeps
    the 0.90 floor -- min_containment acts at the floor at least.)"""
    mask = np.zeros((40, 120), dtype=np.uint16)
    mask[5:15, 5:47] = 1                        # 10 x 42: croppable, no crop at the defaults
    mask[20:30, 10:20] = 2                      # 10 x 10: cropped
    mask[25:35, 60:110] = 3                     # 10 x 50: uncroppable
    image, mask_path = _make_input(tmp_path / "data", mask=mask)
    verifier = CropAgenticVerifier()
    verifier.input = input_record(_base_kwargs(image, mask_path))
    cells = verifier.input["cells"]
    assert [cid for cid, c in cells.items() if c["uncroppable"]] == ["default_img_cell_0002"]
    result = Single_Cell_Cropper_Tool().execute(original_image=str(image), nuclei_mask=str(mask_path), min_area=50,
                                                margin=1, query_cache_dir=str(tmp_path / "q"))
    metrics = verifier.quantify(result, {"trial_index": 1})
    assert metrics["n_crops"] == 1 and metrics["n_lost_to_geometry"] == 2
    assert metrics["n_missing_crops"] == 1 and metrics["n_exempt_uncroppable"] == 1
    assert verifier._reports[1]["missing_crop_cells"] == ["default_img_cell_0000"]
    assert verifier._reports[1]["exempt_uncroppable_cells"] == ["default_img_cell_0002"]
    assert "1 object(s) at min_area got no crop" in verifier.guard(None, VLRRResult("PASS", 0.0, metrics))


def test_agentic_uncroppable_follows_the_tools_square_rule(tmp_path):
    """`uncroppable` is what the tool can deliver: an object longer than the image's shorter side whose largest
    square still keeps at least 0.9 of it is croppable -- the tool crops it, in mask_bbox and in fraction mode with
    min_containment 0.9 -- one whose largest square keeps 0.8 is uncroppable and every such call drops it."""
    mask = np.zeros((40, 120), dtype=np.uint16)
    mask[5:15, 5:47] = 1        # 10 x 42 in a 40 px tall image: the 40 px square keeps 40 / 42
    mask[25:35, 60:110] = 2     # 10 x 50: the 40 px square keeps 0.8
    image, mask_path = _make_input(tmp_path / "data", mask=mask)
    record = input_record(_base_kwargs(image, mask_path))
    assert record["cells"]["default_img_cell_0000"]["uncroppable"] is False
    assert record["cells"]["default_img_cell_0001"]["uncroppable"] is True and record["n_uncroppable"] == 1
    for mode, overrides in (("mask_bbox", None), ("fraction", {c: {"min_containment": 0.9} for c in record["cells"]})):
        result = Single_Cell_Cropper_Tool().execute(
            original_image=str(image), nuclei_mask=str(mask_path), min_area=50, margin=1, margin_mode=mode,
            cell_overrides=overrides, query_cache_dir=str(tmp_path / mode))
        kept = {m["cell_id"]: m["qc"]["target_retention"]
                for m in json.loads(Path(result["cell_crops_metadata_path"]).read_text())["cell_metadata"]}
        assert set(kept) == {"default_img_cell_0000"} and kept["default_img_cell_0000"] >= 0.9, mode


def test_agentic_the_guard_rejects_a_call_with_a_non_square_crop_naming_the_count():
    """The human rule of 2026-09-30 (every crop square: the analyzer resizes each crop to 224 x 224 without keeping
    the aspect ratio) is part of the guard: a call whose readouts show non-square crops is inadmissible, the failure names how
    many; the count comes from the call's readouts (`frac_non_square` over `n_crops`, from each crop's
    `crop_h` / `crop_w`); a call of square crops passes (`frac_non_square` 0, `p90_crop_aspect` 1.0)."""
    verifier = CropAgenticVerifier()

    def cell(h, w):
        return {"neighbor_contamination": 0.0, "target_dominance": 1.0, "target_retention": 1.0,
                "centroid_offset": 0.0, "centered": 1.0, "occupancy": 0.5, "crop_h": h, "crop_w": w,
                "crop_aspect": max(h, w) / min(h, w), "margin_mode": "fraction", "auto_tighten_iterations": 0}

    square = verifier._readouts({f"c{i}": cell(40, 40) for i in range(4)})
    assert square["frac_non_square"] == 0.0 and square["p90_crop_aspect"] == 1.0
    assert verifier.guard(None, VLRRResult("PASS", 1.0, {**square, "n_missing_crops": 0.0})) == []
    mixed = verifier._readouts({"c0": cell(40, 40), "c1": cell(40, 26), "c2": cell(30, 30), "c3": cell(12, 36)})
    assert mixed["frac_non_square"] == 0.5
    assert verifier.guard(None, VLRRResult("PASS", 1.0, {**mixed, "n_missing_crops": 0.0})) == [
        "2 crop(s) not square"]
    one = VLRRResult("PASS", 1.0, {"n_crops": 7.0, "frac_non_square": 1 / 7, "min_target_retention": 0.95,
                                   "n_lost_to_geometry": 1.0, "n_missing_crops": 1.0})
    assert verifier.guard(None, one) == ["1 crop(s) not square", "1 object(s) at min_area got no crop"]
    assert "every crop square" in verifier.admissible_text() and "every crop square" in verifier.final_rule_text()
    assert "every crop square" in verifier.admission_checks_text()


def test_agentic_the_retention_floor_is_one_value_from_the_tool():
    """The retention floor stays 0.90 and is one number everywhere (human decision, 2026-10-01): the tool's
    `_TARGET_RETENTION_FLOOR` is the single source -- the same object is the agentic loop's `RETENTION_FLOOR`
    (the guard, the min_containment range, the uncroppable test), the linear verifier's floor and
    `is_crop_valid`'s truncation gate (0.95 before: a crop the tool and the guard keep at 0.90-0.95 read as
    truncated). A crop AT the floor keeps it -- valid and admissible; just below it is truncated and fails."""
    from shape.tools.single_cell_cropper import tool

    floor = tool._TARGET_RETENTION_FLOOR
    assert floor == 0.90
    assert cv.RETENTION_FLOOR is floor and CropVerifier._TARGET_RETENTION_FLOOR is floor and cv._CONTAINMENT_MIN is floor
    assert cv.CELL_KEYS["min_containment"] == ("number", (floor, 1.0))
    below = floor - 1e-6
    assert cv.is_crop_valid({"mask_containment": floor, "margin_mode": "fraction"}) == (True, [])
    assert cv.is_crop_valid({"mask_containment": below, "margin_mode": "fraction"}) == (False, ["truncated"])
    assert cv.crop_qc([{"mask_containment": floor}, {"mask_containment": below}])["truncation_fraction"] == 0.5
    verifier = CropAgenticVerifier()

    def guard(low):
        return verifier.guard(None, VLRRResult("PASS", 1.0, {"n_crops": 3.0, "min_target_retention": low}))

    assert guard(floor) == [] and guard(below) == ["a crop keeps 0.900 of its target, below 0.9"]


def test_agentic_every_call_on_the_real_tool_is_square_and_passes_the_square_guard(tmp_path, no_retry_wait):
    """On the real tool every call's crops are square -- the defaults (the fixture has an object on the image
    border), mask_bbox with auto_tighten and a crop_size cap -- so no call fails the square condition."""
    engine = _Engine([[_move(margin_mode="mask_bbox", rules=[_rule([("area", ">=", 0)], auto_tighten=True)]),
                       _move(crop_size=24)]], final=["default"])
    resolution, _, _ = _run(tmp_path, engine, cores=4)
    readouts = {c: v["readouts"] for c, v in resolution.details["calls"].items()}
    assert set(readouts) == {"default", "agent_1", "agent_2"}
    assert all(r["frac_non_square"] == 0.0 and r["p90_crop_aspect"] == 1.0 for r in readouts.values())
    assert not any("not square" in json.dumps(row) for p in _round_prompts(engine) for row in _table_rows(p).values())


def test_agentic_a_failed_defaults_call_starts_the_tuning_unasked(tmp_path, no_retry_wait):
    crashed = lambda params: params.get("margin_mode") == "fraction"   # noqa: E731
    engine = _Engine([[_move(margin_mode="tight")]], final=["agent_1"])
    resolution, _, waves = _run(tmp_path, engine, crash=crashed, mask=_clean_mask())
    assert resolution.details["admission"]["problems"] == [
        "the call at the defaults failed: RuntimeError: the tool crashed"]
    assert _admission_questions(engine) == [] and waves == [1, 1] and resolution.details["call"] == "agent_1"


def test_agentic_a_plateau_stops_the_search(tmp_path, monkeypatch, no_retry_wait):
    monkeypatch.setattr(cv, "NOISE_BAND", 1.0)          # nothing is beyond this noise
    engine = _Engine([[_move("default", margin_mode="tight")], [_move("agent_1", recenter=True)],
                      [_move("agent_2", tight_buffer_px=0)], [_move("agent_3", margin_frac=0.1)]], final=["default"])
    resolution, verifier, waves = _run(tmp_path, engine)
    assert base.PATIENCE == verifier.patience == 3                       # rounds
    assert waves == [1, 1, 1, 1] and verifier.stop_reason == "plateau"   # agent_1, agent_2 and agent_3 improved on nothing
    prompts = _round_prompts(engine)
    assert len(prompts) == 3 and resolution.reason == "plateau"
    assert ("Stop: return no moves when you are satisfied, at any round; you may then name in deliver the finished\n"
            "call to deliver, which must meet the final rule (otherwise you are asked to choose separately). A stop\n"
            "with rounds left is asked once more: the deliver of the answer that confirms it counts. As a\n"
            "backstop, the loop also stops when the rounds are spent or when 3 rounds in a row have not "
            "produced an admissible call better than "
            "every\nearlier admissible call beyond the noise on at least one of "
            "['mean_target_dominance', 'mean_target_retention', 'mean_centered'].\n") in prompts[0]


def test_agentic_the_plateau_counts_rounds_a_round_with_a_failed_call_included(tmp_path, monkeypatch, no_retry_wait):
    monkeypatch.setattr(cv, "NOISE_BAND", 1.0)          # nothing is beyond this noise
    crashed = lambda params: params.get("margin_mode") == "mask_bbox"   # noqa: E731
    engine = _Engine([[_move("default", margin_mode="tight"), _move("default", margin_mode="mask_bbox")],
                      [_move("agent_1", recenter=True)], [_move("agent_3", tight_buffer_px=0)]], final=["default"])
    resolution, verifier, waves = _run(tmp_path, engine, cores=4, crash=crashed)
    # rounds 2 (agent_1, and agent_2 failed), 3 (agent_3) and 4 (agent_4): 3 rounds in a row without progress
    assert [f["call"] for f in resolution.details["failed"]] == ["agent_2"]
    assert waves == [1, 2, 1, 1] and verifier.stop_reason == "plateau" and len(_round_prompts(engine)) == 3


def _table_rows(prompt: str) -> dict:
    return {r["call"]: r for r in (json.loads(line) for line in prompt.splitlines() if line.startswith('{"call": '))}


def test_agentic_the_pareto_front_holds_only_admissible_calls(tmp_path, no_retry_wait):
    # default: the tool at its defaults; agent_1: a 24 px cap drops every object but one, whose crop scores 1.0 on every
    # objective
    engine = _Engine([[_move(crop_size=24), _move(margin_mode="tight")]], final=["default"])
    resolution, verifier, waves = _run(tmp_path, engine, cores=6)
    assert waves == [1, 2] and resolution.details["admissible"] == ["default", "agent_2"]
    capped = resolution.details["calls"]["agent_1"]["readouts"]
    assert capped["n_crops"] == 1 and capped["n_lost_to_geometry"] == 6
    assert all(capped[k] == pytest.approx(1.0) for k in CropAgenticVerifier.objectives)
    rows = _table_rows(_round_prompts(engine)[1])
    assert rows["agent_1"]["pareto"] is False and (rows["default"]["pareto"] or rows["agent_2"]["pareto"])
    final = next(c for c in engine.contents if c[0].startswith(verifier.choice_text()))
    shown = re.search(r"Crop panels attached \(the admissible calls on the Pareto front\): \[(.*)\]", final[0])
    panels = re.findall(r"'([^']+)'", shown.group(1))
    assert panels and set(panels) <= {"default", "agent_2"}
    assert [x for x in final[1:] if isinstance(x, str) and x.startswith("CROP PANEL")] == \
        [f"CROP PANEL OF {c}:" for c in panels]


def test_agentic_an_inadmissible_call_is_no_progress_and_the_final_panels_fall_back(tmp_path, no_retry_wait):
    verifier, attempts = _bound(tmp_path, [[_move(margin_mode="tight")], [_move(crop_size=24)]])
    assert verifier.guard(None, attempts["agent_2"][2])                     # agent_2 lost objects to the cap
    assert all(verifier._improves("agent_2", d, attempts) for d in ("default", "agent_1"))   # ... and its one crop "improves"
    assert "agent_2" not in verifier.pareto(attempts)
    rounds = verifier.rounds
    assert rounds[-1] == [3]
    verifier.rounds = rounds[:-1]
    before = verifier._stall(None, attempts)
    verifier.rounds = rounds
    assert verifier._stall(None, attempts) == before + 1                # agent_2's round: one more call, no progress
    verifier.pareto = lambda attempts: set()                            # dominance within the noise can cycle
    calls, which = verifier._final_panels(attempts, ["default", "agent_1"])
    scores = {c: attempts[c][2].scalar_score for c in ("default", "agent_1")}
    assert calls == sorted(scores, key=lambda c: -scores[c])
    assert which == "no admissible call is on the Pareto front: the admissible calls with the highest mean_target_dominance"


def test_agentic_an_inadmissible_final_choice_is_asked_once_more(tmp_path, no_retry_wait):
    rounds = [[_move(crop_size=24)]]          # default: the tool at its defaults; agent_1: a 24 px cap drops the larger objects
    resolution, _, _ = _run(tmp_path, _Engine(rounds, final=["agent_1", "default"]))
    assert resolution.details["admissible"] == ["default"] and resolution.details["call"] == "default"
    assert [a["call"] for a in resolution.details["final_answers"]] == ["agent_1", "default"]
    twice_engine = _Engine(rounds, final=["agent_1", "agent_1"])
    twice, _, _ = _run(tmp_path / "b", twice_engine)
    assert twice.status == "conceded" and twice.reason == "final_choice_inadmissible" and twice.details["call"] == "default"
    retry = [p for p in twice_engine.prompts if "is not admissible" in p]
    assert len(retry) == 1 and "got no crop" in retry[0]


def test_agentic_without_an_admissible_call_the_least_bad_call_is_delivered(tmp_path, no_retry_wait):
    # the defaults call loses the croppable 10 x 42 cell (fraction mode, min_containment 1: default fails the
    # guard); agent_1's 24 px cap drops the 30 x 30 cell too
    engine = _Engine([[_move(crop_size=24)]], final=["default"])
    resolution, _, _ = _run(tmp_path, engine, mask=_long_cell_mask())
    admission = resolution.details["admission"]
    assert admission["decision"] == "RUN" and admission["problems"][0] == "1 object(s) at min_area got no crop"
    assert _admission_questions(engine) == []                           # the guard's problem: not asked
    assert resolution.status == "conceded" and resolution.reason == "no_admissible_call"
    assert resolution.result is not None and resolution.details["call"] == "default"   # the fewest objects lost
    assert not any(p.startswith("Choose the delivered") for p in engine.prompts)


def test_agentic_without_a_reasoner_cannot_start_tuning(tmp_path):
    from shape.engine.errors import NoEngineError
    calls = []
    with pytest.raises(NoEngineError):
        _run(tmp_path, calls=calls)
    assert len(calls) == 1


def test_agentic_failed_round_preserves_execution_error_without_rule_moves(tmp_path, no_retry_wait):
    class Down(_Engine):
        def generate(self, *args, **kwargs):
            raise TimeoutError("gateway timeout")
    calls = []
    resolution, verifier, _ = _run(tmp_path, Down(), calls=calls)
    assert len(calls) == 1 and verifier.fallback_rounds == 0
    assert resolution.details["choice"] == "execution_error"
    assert resolution.details["execution_errors"][0]["question"] == "round"


def test_agentic_a_crashed_or_empty_call_is_failed_never_a_parent_and_never_run_again(tmp_path, no_retry_wait):
    crashed = lambda params: params.get("margin_mode") == "tight"      # noqa: E731
    engine = _Engine([[_move(margin_mode="tight"), _move(margin_mode="mask_bbox")],
                      [_move("agent_1", recenter=True), _move(margin_mode="tight"), _move("agent_2", tight_buffer_px=0)]],
                     final=["agent_3"])
    calls = []
    resolution, verifier, waves = _run(tmp_path, engine, cores=8, crash=crashed, calls=calls)
    assert [f["call"] for f in resolution.details["failed"]] == ["agent_1"]
    assert sum(1 for p in calls if p.get("margin_mode") == "tight") == 1      # never re-run
    second = _round_prompts(engine)[1]
    assert "Calls that did not finish (crashed or rejected by the tool): [{'call': 'agent_1'" in second
    repair = next(p for p in engine.prompts if "were rejected" in p)
    assert "is not a finished call" in repair
    assert "this configuration was already run (or failed) or is already in this round" in repair
    assert waves == [1, 2, 1] and resolution.details["call"] == "agent_3" and resolution.parents[4] == 3


def test_agentic_each_call_has_its_own_directory_and_concurrent_calls_do_not_collide(tmp_path, no_retry_wait):
    engine = _Engine([[_move(margin_mode="tight"), _move(margin_mode="mask_bbox"), _move(recenter=True),
                       _move(margin_frac=0.1)]], final=["agent_1"])
    resolution, verifier, waves = _run(tmp_path, engine, cores=8, workers=4)
    assert waves == [1, 4]
    dirs = set()
    for params, _, result in resolution.trials[1:]:
        meta_path = Path(result.get("call_metadata_path") or result["cell_crops_metadata_path"])
        meta = json.loads(meta_path.read_text())
        call_dir = (tmp_path / "out" / "tool_cache" / params["output_subdir"]).resolve()
        assert meta_path.parent.resolve() == call_dir
        assert all(Path(m["crop_path"]).parent.resolve() == call_dir and Path(m["crop_path"]).exists()
                   for m in meta["cell_metadata"])
        assert meta["parameters"]["margin_mode"] == params["margin_mode"]
        dirs.add(call_dir)
    assert len(dirs) == 5
    delivered = Path(resolution.details["delivered_metadata_path"])
    assert delivered.parent.resolve() == (tmp_path / "out" / "tool_cache").resolve()
    assert resolution.result["cell_crops_metadata_path"] == str(delivered)


def _mask_check_verdicts(reject_first=False):
    """The mask check's answer: every montage position cell_present (the first position of the first montage
    wrong_object, with `reject_first`)."""
    pending = [reject_first]

    def verdicts(prompt):
        ids = [line.split("crop_id=")[1].strip().strip("'") for line in prompt.splitlines() if "crop_id=" in line]
        out = [MaskCheckPositionVerdict(index=i, crop_id=c, verdict="cell_present", confidence="high")
               for i, c in enumerate(ids)]
        if pending.pop() if pending else False:
            out[0] = MaskCheckPositionVerdict(index=0, crop_id=ids[0], verdict="wrong_object", confidence="high")
        return MaskCheckVerdict(verdicts=out)
    return verdicts


class _OrderedEngine(_Engine):
    """`_Engine`, recording the schema of every question in order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.asked = []

    def generate(self, content, response_format=None):
        self.asked.append(response_format)
        return super().generate(content, response_format)


def test_agentic_mask_check_runs_once_on_the_delivered_call_after_the_choice(tmp_path, no_retry_wait):
    """No vision question during the rounds and no `mask_check_*` readout of any call: the mask check runs once,
    on the loop's thread after the final choice, on the delivered call; its kept-crops manifest and decision
    trace are written in that call's directory only and copied with the delivered metadata; its counts are in
    the selection details."""
    engine = _OrderedEngine([[_move(margin_mode="tight")], [_move("default", recenter=True)]], final=["agent_2"],
                            mask_check=_mask_check_verdicts())
    resolution, verifier, _ = _run(tmp_path, engine, cores=4, workers=2)
    assert set(engine.threads) == {threading.get_ident()}                 # every model question on this thread
    first = engine.asked.index(MaskCheckVerdict)
    assert FinalChoice in engine.asked[:first] and set(engine.asked[first:]) == {MaskCheckVerdict}
    calls = resolution.details["calls"]
    assert not [k for c in calls.values() for k in c["readouts"] if k.startswith("mask_check")]
    assert not any("mask_check" in p for p in engine.prompts)
    check = resolution.details["mask_check"]
    assert resolution.details["call"] == check["call"] == "agent_2" and check["mask_check_judged"] > 0
    assert check["mask_check_reused"] == 0 and check["mask_check_rejected"] == 0 and check["rejected_cells"] == []
    call_dir = Path(resolution.trials[3][2]["call_metadata_path"]).parent
    calls_dir = tmp_path / "out" / "tool_cache" / "crop_calls"
    for prefix in ("cell_crops_kept_manifest_", "cell_crops_triage_trace_"):
        assert [p.parent for p in calls_dir.rglob(f"{prefix}*.json")] == [call_dir]
    delivered = Path(resolution.details["delivered_metadata_path"])
    assert (delivered.parent / delivered.name.replace("metadata", "kept_manifest")).exists()
    assert (delivered.parent / delivered.name.replace("metadata", "triage_trace")).exists()
    # the geometry-invariant verdicts stay cached: a second check of another call reuses them
    again = verifier._mask_check(list(verifier._crops[1].values()), verifier._reports[1]["metadata_path"], {})
    assert again["mask_check_reused"] > 0


def test_agentic_a_mask_check_reject_leaves_the_delivered_manifest(tmp_path, no_retry_wait):
    """A crop the mask check rejects on the delivered call is left out of its kept-crops manifest (the delivered
    copy included) and named in the selection details; every other crop is kept."""
    engine = _Engine([[_move(margin_mode="tight")]], final=["agent_1"], mask_check=_mask_check_verdicts(reject_first=True))
    resolution, verifier, _ = _run(tmp_path, engine)
    check = resolution.details["mask_check"]
    assert check["call"] == "agent_1" and check["mask_check_rejected"] == 1 and len(check["rejected_cells"]) == 1
    crops = verifier._crops[2]
    delivered = Path(resolution.details["delivered_metadata_path"])
    kept = set(json.loads((delivered.parent / delivered.name.replace("metadata", "kept_manifest")).read_text())
               ["kept_crop_paths"])
    assert kept == {c["crop_path"] for cid, c in crops.items() if cid not in check["rejected_cells"]}
    trace = json.loads((delivered.parent / delivered.name.replace("metadata", "triage_trace")).read_text())
    entry = next(v for v in trace["verdicts"] if v["cell_id"] == check["rejected_cells"][0])
    assert entry["mask_check"]["verdict"] == "wrong_object"
    assert entry["final"] == {"kept": False, "reason": "wrong_object"}


@pytest.mark.parametrize("fatal", [False, True])
def test_mask_check_errors_are_counted_or_propagated_on_delivery(tmp_path, no_retry_wait, fatal):
    from shape.engine.errors import NoEngineError

    def failed_check(prompt):
        raise NoEngineError("unusable engine") if fatal else TimeoutError("vision timed out")

    engine = _Engine(mask_check=failed_check)
    if fatal:
        with pytest.raises(NoEngineError) as caught:
            _run(tmp_path, engine, budget=2)
        assert caught.value.shape_question == "mask_check"
    else:
        resolution, verifier, _ = _run(tmp_path, engine, budget=2)
        assert resolution.details["reasoner_failures"] == verifier.reasoner_failures > 0
        assert resolution.details["execution_errors"][0]["question"] == "mask_check"
        assert resolution.details["fallback_rounds"] == 0


def test_agentic_every_delivery_path_runs_the_mask_check(tmp_path, no_retry_wait):
    """The admission's skip (`default` as it is), the least-bad call (no admissible call) and the fallback's choice
    (no reasoner question answered) each deliver a call with its mask check."""
    skip = _OrderedEngine(admission="none", mask_check=_mask_check_verdicts())
    resolution, _, _ = _run(tmp_path / "skip", skip, mask=_clean_mask())
    assert resolution.details["choice"] == "no_tuning_needed" and resolution.details["mask_check"]["call"] == "default"
    least_bad = _Engine([[_move(crop_size=24)]], final=["default"], mask_check=_mask_check_verdicts())
    resolution, _, _ = _run(tmp_path / "least_bad", least_bad, mask=_long_cell_mask())
    assert resolution.reason == "no_admissible_call" and resolution.details["mask_check"]["call"] == "default"

    class Silent(_Engine):                     # the gateway answers only the vision questions
        def generate(self, content, response_format=None):
            if response_format is not MaskCheckVerdict:
                raise TimeoutError("gateway down")
            return super().generate(content, response_format)

    resolution, _, _ = _run(tmp_path / "fallback", Silent(mask_check=_mask_check_verdicts()), budget=2)
    assert resolution.details["choice"] == "execution_error"
    assert resolution.details["call"] is None and "mask_check" not in resolution.details


def test_agentic_no_object_at_min_area_ends_the_stage_without_a_call(tmp_path):
    image, mask = _make_input(tmp_path / "data")
    kwargs = _base_kwargs(image, mask, min_area=10 ** 6)
    verifier = CropAgenticVerifier()
    calls = []
    resolution = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(
        verifier, verifier.make_adapter(kwargs), _runner(image, mask, tmp_path / "out", calls), {},
        input_record(kwargs), upstream_ctx={})
    assert calls == [] and resolution.result is None and resolution.reason == "no_objects"
    assert resolution.status == "conceded" and resolution.fault == "ceiling"
    assert resolution.details["tuning_gate"] == {
        "required": True, "source": "input", "decision": "ABSTAIN", "allowed": False,
        "reason": "conditions_unmet"}


def test_agentic_the_solve_path_runs_the_loop_and_delivers_the_chosen_call(tmp_path):
    import inspect

    from shape.agent.verification.integration import make_verification_record, verify_managed_tool
    from shape.agent.verification.stage_map import invoke_managed_tool, map_tool_params
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    levers = {"margin_mode": "tight", "margin": 3, "margin_frac": 0.1, "recenter": True, "tight_buffer_px": 1,
              "cell_overrides": {"x": {"recenter": True}}, "min_area": 50, "output_subdir": "a/b"}
    kwargs = set(inspect.signature(Single_Cell_Cropper_Tool.execute).parameters)
    assert map_tool_params("Single_Cell_Cropper_Tool", levers, kwargs) == levers   # the geometry reaches the tool

    image, mask = _make_input(tmp_path / "data")

    class Executor:
        enable_llm_verifier = True
        verifier_llm_engine_mm = _Engine([[_move(margin_mode="mask_bbox")]], final=["agent_1"],
                                         mask_check=_mask_check_verdicts())
        query_cache_dir = str(tmp_path / "query")

        def __init__(self):
            self.calls = []

        def run_managed_tool(self, tool_name, base_kwargs, params):
            self.calls.append(dict(params))
            return invoke_managed_tool(Single_Cell_Cropper_Tool(), tool_name, base_kwargs, params, self.query_cache_dir)

    executor = Executor()
    resolutions, result, status = verify_managed_tool(
        executor, "Single_Cell_Cropper_Tool", {"original_image": str(image), "nuclei_mask": str(mask)},
        initial_params={"padding": 25, "crop_size": 256})
    res = resolutions[0]
    assert res.trials[0][0] == {} and res.trials[0][2]["status"] == cv.INPUT_STATUS
    assert executor.calls and all(p.get("crop_size") != 256 and "padding" not in p for p in executor.calls)
    assert all(isinstance(r.get("seconds"), float) for _, _, r in res.trials[1:])   # the tool times each call
    ruled = [(p, r) for p, _, r in res.trials[1:] if p["cell_overrides"]]
    assert res.details["call"] == "agent_1" and res.details["fallback_rounds"] == 0
    for params, trial_result in ruled:
        meta = json.loads(Path(trial_result.get("call_metadata_path") or trial_result["cell_crops_metadata_path"])
                          .read_text())
        assert meta["parameters"]["cell_overrides"] == params["cell_overrides"]
    tool_cache = Path(executor.query_cache_dir) / "tool_cache"
    assert Path(result["cell_crops_metadata_path"]).parent == tool_cache
    crops, _, _ = Cell_State_Analyzer_Tool._load_cell_data_from_metadata(None, executor.query_cache_dir)
    assert crops and all(Path(p).parent == tool_cache / res.params["output_subdir"] for p in crops)
    record = make_verification_record("Single_Cell_Cropper_Tool", resolutions, status)
    assert record["stages"][0]["details"]["rounds"] and record["stages"][0]["details"]["call"] == res.details["call"]


def _card_section(heading: str) -> str:
    from shape.tools.base import load_tuning_card

    body = load_tuning_card("single_cell_cropper") or ""
    start = body.find(heading)
    return body[start:body.find("\n## ", start + 1)]


def test_agentic_card_lists_every_lever_with_the_tools_defaults_and_the_code_bounds():
    import inspect

    from shape.utils.schemas import card_lever_fields, lever_spec

    fields = card_lever_fields("single_cell_cropper", CropAgenticVerifier.lever_section)
    signature = inspect.signature(Single_Cell_Cropper_Tool.execute).parameters
    assert list(fields) == ["margin_mode", "margin_frac", "margin", "recenter", "tight_buffer_px", "crop_size"]
    for lever, f in fields.items():
        kind, bounds, nullable, default = lever_spec(f)
        assert lever in signature and default == signature[lever].default, lever   # the tool's own default
        assert "Step" not in f and "Fallback" not in f
        if kind in ("number", "integer"):
            lo, hi, _ = base.PARAM_BOUNDS[lever]
            assert bounds == (lo, hi), lever
    assert lever_spec(fields["margin_mode"])[1] == cv.MARGIN_MODES
    for key in cv.CELL_KEYS:
        assert f"| `{key}` |" in _card_section("#### Per-cell keys"), key


def test_agentic_card_loop_settings_match_the_code():
    from shape.tools.base import load_tuning_card
    from shape.tools.single_cell_cropper import tool

    rows = {}
    for line in _card_section("## Loop settings").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0].startswith("`"):
            rows[cells[0].strip("`")] = cells[1]
    expected = {"BUDGET": base.BUDGET, "PATIENCE": base.PATIENCE, "REASONER_ATTEMPTS": base.REASONER_ATTEMPTS,
                "REASONER_RETRY_SECONDS": 30, "THREADS_PER_CALL": cv.THREADS_PER_CALL, "NOISE_BAND": cv.NOISE_BAND,
                "RETENTION_FLOOR": cv.RETENTION_FLOOR, "PANEL_CELLS": cv.PANEL_CELLS,
                "NN_GAP_CAP_PX": cv.NN_GAP_CAP_PX, "FINAL_PANELS_MAX": cv.FINAL_PANELS_MAX,
                "PROMPT_IMAGE_PX": cv.PROMPT_IMAGE_PX, "FINAL_IMAGE_PX": cv.FINAL_IMAGE_PX,
                "MASK_CHECK_CAP": cv.MASK_CHECK_CAP,
                "CropVerifier._MASK_BBOX_CONTAMINATION_MIN": CropVerifier._MASK_BBOX_CONTAMINATION_MIN,
                "CropVerifier._OFF_CENTER_MAX": CropVerifier._OFF_CENTER_MAX,
                "_TARGET_RETENTION_FLOOR": tool._TARGET_RETENTION_FLOOR,
                "_MASK_BBOX_CONTAMINATION_MIN": tool._MASK_BBOX_CONTAMINATION_MIN,
                "_TIGHTEN_PLATEAU_EPS": tool._TIGHTEN_PLATEAU_EPS}
    for name, value in expected.items():
        assert float(rows[name]) == value, name
    assert base.PATIENCE == 3                                             # tool calls, for a budget of 8 rounds
    constants = {n for n, v in vars(cv).items() if n.isupper() and not n.startswith("_")
                 and isinstance(v, (int, float)) and not isinstance(v, bool)}
    assert constants <= set(rows), sorted(constants - set(rows))           # every cropper loop constant is listed
    assert rows["score_key"] == CropAgenticVerifier.score_key
    assert rows["objectives"] == ", ".join(CropAgenticVerifier.objectives)
    assert rows["admission_call"] == "true" and CropAgenticVerifier.admission_call is True
    assert cv.RETENTION_FLOOR == tool._TARGET_RETENTION_FLOOR and CropAgenticVerifier.budget == base.BUDGET
    assert "## Levers" in (load_tuning_card("single_cell_cropper") or "")
    assert math.isnan(CropAgenticVerifier().adjudicate({"n_croppable": 3.0}, "this_step", [],
                                                        {"trial_index": 0}).scalar_score)


def test_the_mask_check_runs_unless_the_crop_task_switches_it_off():
    """`CropTask.enable_mask_check` (default on) is read at the input trial; off, the delivered call is not judged."""
    class Engine:
        def generate(self, *args, **kwargs):
            raise AssertionError("the mask check asked the reasoner although it is switched off")

    verifier = CropAgenticVerifier()
    assert verifier.enable_mask_check is True and CropTask().enable_mask_check is True
    verifier.adjudicate({"n_croppable": 1.0}, "this_step", [], {"trial_index": 0, "crop_task": CropTask(enable_mask_check=False)})
    assert verifier.enable_mask_check is False
    verifier.llm_engine = Engine()
    assert verifier._mask_check([{"cell_id": "default_i_cell_0"}], "unused.json", {}) == {}
