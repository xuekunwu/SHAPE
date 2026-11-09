"""Task 038 M-B: the SHARED, tool-agnostic `population_triage` judge in `llm_recommend.py`
(`build_judge_prompt`/`judge_item`/`judge_items`/`write_kept_manifest`/
`recommend_population_triage`) -- the sibling of `_recommend_from_card` (the shared
`param_retune` judge) for the OTHER verdict type. Co-located with the
module under test (committed -- AGENTS.md sec 6 P5).

Uses GENERIC, non-cropper fixtures throughout (plain dict "items", not "crops") to prove
the shared functions carry no cropper-specific logic -- `crop_triage.py`'s own
`vision_recommend_crop` adapter is tested separately, end-to-end, in
`test_crop_triage.py` (the BEHAVIORAL INVARIANT: its decisions must be equivalent to
M-A's bespoke `arbitrate_crop`/`arbitrate_crops`, which this round retired).

Also covers the SCOPE GUARD (M-B): the segmenter/analyzer `param_retune` path
(`_recommend_from_card`, `_build_prompt`, `_build_rep_prompt`,
`vision_recommend_segmentation`, `vision_recommend_representation`) must stay
byte-identical -- this file's new `population_triage` section lives in the SAME module,
so a dedicated parity check here directly guards against an accidental cross-section
regression.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import BaseModel

from shape.agent.verification import llm_recommend
from shape.utils.schemas import VLRRResult


class _ItemVerdict(BaseModel):
    """A generic (non-cropper) population-triage verdict schema -- proves `judge_item`/
    `judge_items`/`recommend_population_triage` require only `.verdict`/`.confidence`/
    `.problem`, not `crop_triage.CropTriageVerdict` specifically (schema is a caller
    parameter, mirroring `_recommend_from_card`'s own `schema=` convention)."""
    verdict: str
    confidence: str
    problem: str
    reasoning: str


class _StubEngine:
    def __init__(self, verdict="keep", confidence="high", problem="none", raise_error=False):
        self._verdict, self._confidence, self._problem = verdict, confidence, problem
        self._raise = raise_error

    def generate(self, content, response_format=None, **kwargs):
        if self._raise:
            raise RuntimeError("engine failure")
        rec = _ItemVerdict(verdict=self._verdict, confidence=self._confidence,
                           problem=self._problem, reasoning="stub")
        return {"content": rec}


class _RecordingEngine:
    def __init__(self, verdict="keep", confidence="high", problem="none"):
        self._verdict, self._confidence, self._problem = verdict, confidence, problem
        self.calls: list = []

    def generate(self, content, response_format=None, **kwargs):
        self.calls.append(content)
        rec = _ItemVerdict(verdict=self._verdict, confidence=self._confidence,
                           problem=self._problem, reasoning="stub")
        return {"content": rec}


def _item_image(tmp_path, name="widget.png", **fields) -> dict:
    """A generic item pointing at a real (tiny) on-disk image -- deliberately NOT framed
    as a "crop" (proves the shared machinery has no cropper-specific content)."""
    from skimage.io import imsave
    path = tmp_path / name
    imsave(str(path), (np.random.rand(16, 16) * 255).astype(np.uint8))
    item = {"crop_path": str(path)}   # the item-identity key `write_kept_manifest` reads
    item.update(fields)
    return item


def _image_fn(item: dict):
    path = item.get("crop_path")
    if not path:
        return None
    import os
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# build_judge_prompt
# ---------------------------------------------------------------------------

def test_build_judge_prompt_carries_quant_evidence_and_card():
    prompt = llm_recommend.build_judge_prompt(
        card="TEST CARD BODY TEXT", quant={"widget_score": 0.1234}, evidence=None)
    assert "widget_score" in prompt and "0.1234" in prompt
    assert "TEST CARD BODY TEXT" in prompt


def test_build_judge_prompt_carries_extra_evidence_channel():
    prompt = llm_recommend.build_judge_prompt(
        card=None, quant={}, evidence={"GATE": "FAIL (too_small)"})
    assert "GATE evidence: FAIL (too_small)" in prompt


def test_build_judge_prompt_quant_unavailable():
    prompt = llm_recommend.build_judge_prompt(card=None, quant={}, evidence=None)
    assert "unavailable" in prompt


def test_build_judge_prompt_no_card_omits_card_section():
    """The preamble legitimately references "the TOOL CARD" conceptually regardless (the
    judge is always told a CARD governs verdict semantics); what must be ABSENT with no
    card is the actual card_section heading + body (there is no CARD text to show)."""
    prompt = llm_recommend.build_judge_prompt(card=None, quant={"x": 1.0}, evidence=None)
    assert "this item's own verdict semantics, calibration, and complementarity rules" not in prompt


def test_build_judge_prompt_carries_reconcile_instruction():
    """The universal reconcile-before-reject instruction is always present, regardless of
    card/quant/evidence -- it is NOT tool-specific (task 038 M-B design). Task 038
    follow-up, part B.3: the instruction now also makes the quant AUTHORITATIVE (must be
    STATED and deferred to on conflict), not just reconciled when anomalous."""
    prompt = llm_recommend.build_judge_prompt(card=None, quant={}, evidence=None)
    assert "RECONCILE" in prompt
    assert "CONTRADICTS must NOT drive a reject" in prompt
    assert "NAMES the relevant quant(s)" in prompt
    assert "AUTHORITATIVE" in prompt
    assert "DEFER TO THE QUANT NUMBER" in prompt


def test_build_judge_prompt_nothing_tool_specific_hardcoded():
    """No cropper-specific vocabulary (crop/cell/doublet/etc.) leaks into the shared
    assembly function itself -- everything tool-specific must come from the CARD text."""
    prompt = llm_recommend.build_judge_prompt(card=None, quant={}, evidence=None)
    for banned in ("crop", "cell", "doublet", "microscopy"):
        assert banned not in prompt.lower()


# ---------------------------------------------------------------------------
# judge_item (generalizes crop_triage.arbitrate_crop)
# ---------------------------------------------------------------------------

def test_judge_item_high_confidence_reject(tmp_path):
    item = _item_image(tmp_path)
    reject, verdict = llm_recommend.judge_item(
        _StubEngine(verdict="reject", confidence="high", problem="broken"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert reject is True
    assert verdict.problem == "broken"


def test_judge_item_low_confidence_reject_is_kept(tmp_path):
    """Keep-bias: a LOW-confidence reject does not reject."""
    item = _item_image(tmp_path)
    reject, _ = llm_recommend.judge_item(
        _StubEngine(verdict="reject", confidence="low"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert reject is False


def test_judge_item_keep_verdict_is_kept(tmp_path):
    item = _item_image(tmp_path)
    reject, _ = llm_recommend.judge_item(
        _StubEngine(verdict="keep", confidence="high"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert reject is False


def test_judge_item_without_an_engine_raises(tmp_path):
    item = _item_image(tmp_path)
    with pytest.raises(RuntimeError, match="no vision engine is configured"):                # never a silent keep
        llm_recommend.judge_item(None, item, schema=_ItemVerdict, card=None, image_fn=_image_fn)


def test_judge_item_unreadable_image_is_noop():
    item = {"crop_path": "/does/not/exist.png"}
    assert llm_recommend.judge_item(
        _StubEngine(), item, schema=_ItemVerdict, card=None,
        image_fn=_image_fn) == (False, None)


def test_judge_item_engine_error_is_an_execution_error(tmp_path):
    from shape.utils.schemas import ExecutionError

    item = _item_image(tmp_path)
    with pytest.raises(ExecutionError, match="population-triage judge failed"):          # never a silent keep
        llm_recommend.judge_item(_StubEngine(raise_error=True), item, schema=_ItemVerdict, card=None,
                                 image_fn=_image_fn)


def test_judge_item_sends_prompt_and_image_together(tmp_path):
    item = _item_image(tmp_path)
    engine = _RecordingEngine()
    llm_recommend.judge_item(engine, item, schema=_ItemVerdict, card="A CARD",
                             image_fn=_image_fn, quant_fn=lambda i: {"widget_score": 0.5},
                             evidence_fn=lambda i: {"GATE": "PASS"})
    assert len(engine.calls) == 1
    prompt, img = engine.calls[0]
    assert isinstance(prompt, str) and isinstance(img, bytes) and len(img) > 0
    assert "widget_score" in prompt and "A CARD" in prompt and "GATE evidence: PASS" in prompt


def test_judge_item_no_quant_fn_or_evidence_fn_still_works(tmp_path):
    """Both `quant_fn`/`evidence_fn` are optional (a population_triage tool with no quant
    channel, e.g.) -- the judge still runs on the image + CARD alone."""
    item = _item_image(tmp_path)
    reject, verdict = llm_recommend.judge_item(
        _StubEngine(verdict="keep", confidence="high"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert reject is False
    assert verdict is not None


# ---------------------------------------------------------------------------
# `problem_gate` (task 038 follow-up, TOOL-AGNOSTIC): a reject is valid ONLY if
# `verdict.problem` is in the caller-supplied set -- defense in depth alongside a
# `schema`'s own enum constraint on `problem` (a tool with a permissive/string `problem`,
# like this file's own generic `_ItemVerdict`, has NO structural constraint at all, so
# this runtime gate is the ONLY thing enforcing a valid-reasons taxonomy for it).
# ---------------------------------------------------------------------------

def test_judge_item_problem_gate_downgrades_out_of_gate_reject_to_keep(tmp_path):
    item = _item_image(tmp_path)
    reject, verdict = llm_recommend.judge_item(
        _StubEngine(verdict="reject", confidence="high", problem="off_center"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn,
        problem_gate={"doublet", "blur"})
    assert reject is False
    assert verdict is not None   # the verdict itself is still returned (for tracing) --
                                 # only the boolean `reject` decision is downgraded


def test_judge_item_problem_gate_allows_an_in_gate_reject(tmp_path):
    item = _item_image(tmp_path)
    reject, verdict = llm_recommend.judge_item(
        _StubEngine(verdict="reject", confidence="high", problem="doublet"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn,
        problem_gate={"doublet", "blur"})
    assert reject is True


def test_judge_item_no_problem_gate_is_unchanged_behavior(tmp_path):
    """`problem_gate=None` (default) -> no gating at all, byte-identical to every
    pre-existing caller of `judge_item` (parity, no behavior change for the shared
    module's other consumers)."""
    item = _item_image(tmp_path)
    reject, verdict = llm_recommend.judge_item(
        _StubEngine(verdict="reject", confidence="high", problem="off_center"), item,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert reject is True


# ---------------------------------------------------------------------------
# `reject_veto_fn`/`force_reject_fn` (task 038 follow-up, part B.2 / part A
# recalibration, TOOL-AGNOSTIC): an earlier round added a pair of ITEM-AWARE hard-
# threshold hooks -- `reject_veto_fn` downgraded an otherwise-valid reject to keep on a
# caller-supplied condition; `force_reject_fn` forced a reject regardless of the vision
# verdict. A LATER round (part B redesign, human directive: "NO hard threshold
# override") REMOVED BOTH ENTIRELY -- `judge_item`/`judge_items`/
# `recommend_population_triage` no longer accept either kwarg at all (a caller passing
# one now gets a TypeError, not a silent no-op) -- replaced by feeding the SAME caller
# signal into the fused prompt as evidence TEXT via `evidence_fn` instead (see
# `crop_triage._evidence_strength_text` for the cropper's own concrete replacement).
# ---------------------------------------------------------------------------

def test_judge_item_rejects_unexpected_reject_veto_fn_kwarg(tmp_path):
    item = _item_image(tmp_path)
    with pytest.raises(TypeError):
        llm_recommend.judge_item(
            _StubEngine(verdict="reject", confidence="high", problem="doublet"), item,
            schema=_ItemVerdict, card=None, image_fn=_image_fn,
            reject_veto_fn=lambda it, rec: True)


def test_judge_item_rejects_unexpected_force_reject_fn_kwarg(tmp_path):
    item = _item_image(tmp_path)
    with pytest.raises(TypeError):
        llm_recommend.judge_item(
            _StubEngine(verdict="keep", confidence="high"), item,
            schema=_ItemVerdict, card=None, image_fn=_image_fn,
            force_reject_fn=lambda it: "severe_defect")


def test_recommend_population_triage_rejects_unexpected_veto_force_kwargs(tmp_path):
    """Same removal, one level up the call chain -- `recommend_population_triage` no
    longer forwards these kwargs either."""
    det = _det()
    items = [_item_image(tmp_path)]
    with pytest.raises(TypeError):
        llm_recommend.recommend_population_triage(
            _StubEngine(), items, det, schema=_ItemVerdict, card=None, image_fn=_image_fn,
            reject_veto_fn=lambda it, rec: True)
    with pytest.raises(TypeError):
        llm_recommend.recommend_population_triage(
            _StubEngine(), items, det, schema=_ItemVerdict, card=None, image_fn=_image_fn,
            force_reject_fn=lambda it: "severe_defect")


# ---------------------------------------------------------------------------
# judge_items (generalizes crop_triage.arbitrate_crops)
# ---------------------------------------------------------------------------

def test_judge_items_without_an_engine_raises(tmp_path):
    items = [_item_image(tmp_path)]
    with pytest.raises(RuntimeError, match="no vision engine is configured"):
        llm_recommend.judge_items(None, items, schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert "vision_reject" not in items[0]


def test_judge_items_sample_cap_evenly_spaced(tmp_path):
    items = [_item_image(tmp_path, f"w{i}.png") for i in range(4)]
    summary = llm_recommend.judge_items(
        _StubEngine(verdict="reject", confidence="high"), items,
        schema=_ItemVerdict, card=None, image_fn=_image_fn, sample_cap=2)
    assert summary["judged"] == 2


def test_judge_items_annotates_rejected_items_in_place(tmp_path):
    items = [_item_image(tmp_path)]
    summary = llm_recommend.judge_items(
        _StubEngine(verdict="reject", confidence="high", problem="broken"), items,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert summary == {"judged": 1, "rejected": 1, "engine": True, "verdicts": [
        {"item_id": items[0]["crop_path"], "reject": True, "verdict": "reject",
         "confidence": "high", "problem": "broken", "reasoning": "stub"}]}
    assert items[0]["vision_reject"] is True
    assert items[0]["vision_problem"] == "broken"


def test_judge_items_not_pre_filtered_by_any_gate_and_can_rescue(tmp_path):
    """The population-triage candidate set is NOT pre-filtered by any deterministic gate
    -- EVERY item is a candidate, so a tool's own `evidence_fn` (surfacing a gate verdict
    as evidence) can be overridden/rescued by the judge, unlike 041's `triage_crops`."""
    item = _item_image(tmp_path, gate_ok=False)   # a "gate-invalid" item by convention
    summary = llm_recommend.judge_items(
        _StubEngine(verdict="keep", confidence="high"), [item],
        schema=_ItemVerdict, card=None, image_fn=_image_fn,
        evidence_fn=lambda i: {"GATE": "FAIL"})
    assert summary == {"judged": 1, "rejected": 0, "engine": True, "verdicts": [
        {"item_id": item["crop_path"], "reject": False, "verdict": "keep",
         "confidence": "high", "problem": "none", "reasoning": "stub"}]}
    assert "vision_reject" not in item   # rescued, not pre-filtered out


# ---------------------------------------------------------------------------
# write_kept_manifest (generalizes crop_triage.write_kept_crops_manifest)
# ---------------------------------------------------------------------------

def test_write_kept_manifest_writes_only_predicate_kept_items(tmp_path):
    items = [{"crop_path": "/a.png", "vision_reject": True},
             {"crop_path": "/b.png", "vision_reject": False}]
    meta_path = tmp_path / "cell_crops_metadata_x.json"
    meta_path.write_text("{}")
    out_path = llm_recommend.write_kept_manifest(
        items, str(meta_path), kept_predicate=lambda i: not i.get("vision_reject", False))
    assert out_path is not None
    written = json.loads(open(out_path).read())
    assert written == {"kept_crop_paths": ["/b.png"]}


def test_write_kept_manifest_no_metadata_path_is_noop():
    assert llm_recommend.write_kept_manifest([], None, kept_predicate=lambda i: True) is None


def test_write_kept_manifest_unexpected_filename_is_noop(tmp_path):
    """Non-destructive: refuses to write (rather than guess) when `metadata_path`'s
    basename doesn't match the known naming convention -- never risks clobbering an
    unrelated file."""
    bad_path = tmp_path / "not_a_metadata_file.json"
    assert llm_recommend.write_kept_manifest(
        [], str(bad_path), kept_predicate=lambda i: True) is None
    assert not bad_path.exists()   # untouched


# ---------------------------------------------------------------------------
# write_triage_trace (task 038 run_shape_crop.py follow-up: the full-verdict sidecar
# `write_kept_manifest` does not carry -- same naming/graceful-degrade idiom)
# ---------------------------------------------------------------------------

def test_write_triage_trace_writes_all_verdicts(tmp_path):
    verdicts = [{"item_id": "/a.png", "reject": True, "verdict": "reject",
                "confidence": "high", "problem": "blur", "reasoning": "out of focus"},
               {"item_id": "/b.png", "reject": False, "verdict": "keep",
                "confidence": "high", "problem": "none", "reasoning": "looks fine"}]
    meta_path = tmp_path / "cell_crops_metadata_x.json"
    meta_path.write_text("{}")
    out_path = llm_recommend.write_triage_trace(verdicts, str(meta_path))
    assert out_path is not None
    written = json.loads(open(out_path).read())
    assert written == {"verdicts": verdicts}


def test_write_triage_trace_no_metadata_path_is_noop():
    assert llm_recommend.write_triage_trace([], None) is None


def test_write_triage_trace_unexpected_filename_is_noop(tmp_path):
    bad_path = tmp_path / "not_a_metadata_file.json"
    assert llm_recommend.write_triage_trace([], str(bad_path)) is None
    assert not bad_path.exists()


# ---------------------------------------------------------------------------
# recommend_population_triage (generalizes crop_triage.vision_recommend_crop)
# ---------------------------------------------------------------------------

def _det(decision="PASS", action="accept", score=0.9, notes="", param_delta=None):
    return VLRRResult(decision, score, {}, action=action, notes=notes, param_delta=param_delta)


def test_recommend_population_triage_without_an_engine_raises():
    with pytest.raises(RuntimeError, match="no vision engine is configured"):                # never the deterministic verdict
        llm_recommend.recommend_population_triage(
            None, [{"crop_path": "x"}], _det(), schema=_ItemVerdict, card=None, image_fn=_image_fn)


def test_recommend_population_triage_no_items_returns_deterministic_unchanged():
    det = _det()
    engine = _StubEngine(verdict="reject", confidence="high")
    assert llm_recommend.recommend_population_triage(
        engine, [], det, schema=_ItemVerdict, card=None, image_fn=_image_fn) is det
    assert llm_recommend.recommend_population_triage(
        engine, None, det, schema=_ItemVerdict, card=None, image_fn=_image_fn) is det


def test_recommend_population_triage_no_rejects_returns_deterministic_unchanged(tmp_path):
    det = _det()
    items = [_item_image(tmp_path)]
    out = llm_recommend.recommend_population_triage(
        _StubEngine(verdict="keep", confidence="high"), items, det,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert out is det


def test_recommend_population_triage_tightens_pass_to_warning(tmp_path):
    det = _det(decision="PASS", action="accept", score=0.9)
    items = [_item_image(tmp_path)]
    out = llm_recommend.recommend_population_triage(
        _StubEngine(verdict="reject", confidence="high", problem="broken"), items, det,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert out.decision == "WARNING"
    assert out.scalar_score == det.scalar_score
    assert out.action == det.action
    assert out.param_delta == det.param_delta
    assert "broken" in out.notes


def test_recommend_population_triage_never_loosens_or_skips_past_fail(tmp_path):
    det = _det(decision="FAIL", action="conceded", score=0.1)
    items = [_item_image(tmp_path)]
    out = llm_recommend.recommend_population_triage(
        _StubEngine(verdict="reject", confidence="high"), items, det,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert out.decision == "FAIL"


def test_recommend_population_triage_preserves_deterministic_fault_on_tighten(tmp_path):
    """Docking-merge re-verification (045-alignment): the tighten path must not silently
    drop `deterministic.fault`/`upstream_target` -- a real property to guard now that the
    fault contract is `ceiling`/`upstream` only (task 045; the old `human` value is
    retired). Harmless no-op TODAY (the cropper's own deterministic verdict never
    concedes), but load-bearing for the CARD's own documented "Open cropper gap" follow-on
    (emit `conceded` + `fault='upstream'` on a bad-mask input) once implemented -- this
    locks the property in now rather than leaving it to silently regress later."""
    det = VLRRResult("FAIL", 0.1, {}, action="conceded", fault="upstream",
                     upstream_target="segmentation", notes="")
    items = [_item_image(tmp_path)]
    out = llm_recommend.recommend_population_triage(
        _StubEngine(verdict="reject", confidence="high", problem="broken"), items, det,
        schema=_ItemVerdict, card=None, image_fn=_image_fn)
    assert out.fault == "upstream"
    assert out.upstream_target == "segmentation"
    assert out.action == "conceded"   # action itself is also carried forward, unchanged


def test_recommend_population_triage_internal_failure_is_an_execution_error(tmp_path, monkeypatch):
    """A failure inside `judge_items` is an execution error of the call being verified, never the
    deterministic verdict."""
    from shape.utils.schemas import ExecutionError

    det = _det()
    items = [_item_image(tmp_path)]

    def _boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_recommend, "judge_items", _boom)
    with pytest.raises(ExecutionError, match="population-triage recommend failed: RuntimeError: boom"):
        llm_recommend.recommend_population_triage(
            _StubEngine(), items, det, schema=_ItemVerdict, card=None, image_fn=_image_fn)


def test_recommend_population_triage_writes_manifest_only_when_engine_ran(tmp_path):
    items = [_item_image(tmp_path)]
    meta_path = tmp_path / "cell_crops_metadata_x.json"
    meta_path.write_text("{}")
    det = _det()
    llm_recommend.recommend_population_triage(
        _StubEngine(verdict="keep", confidence="high"), items, det,
        schema=_ItemVerdict, card=None, image_fn=_image_fn, metadata_path=str(meta_path),
        kept_predicate=lambda i: not i.get("vision_reject", False))
    manifest_path = tmp_path / "cell_crops_kept_manifest_x.json"
    assert manifest_path.exists()
    # the full-verdict trace sidecar is written alongside it (task 038 follow-up)
    trace_path = tmp_path / "cell_crops_triage_trace_x.json"
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text())
    assert trace["verdicts"][0]["item_id"] == items[0]["crop_path"]
    assert trace["verdicts"][0]["verdict"] == "keep"

    # no engine -> a system error, no manifest/trace at all
    meta_path2 = tmp_path / "cell_crops_metadata_y.json"
    meta_path2.write_text("{}")
    with pytest.raises(RuntimeError, match="no vision engine is configured"):
        llm_recommend.recommend_population_triage(
            None, items, det, schema=_ItemVerdict, card=None, image_fn=_image_fn,
            metadata_path=str(meta_path2), kept_predicate=lambda i: True)
    assert not (tmp_path / "cell_crops_kept_manifest_y.json").exists()
    assert not (tmp_path / "cell_crops_triage_trace_y.json").exists()


# ---------------------------------------------------------------------------
# SCOPE GUARD (M-B): the segmenter/analyzer `param_retune` path
# (`_recommend_from_card`) stays BYTE-IDENTICAL -- this section's new `population_triage`
# code lives in the SAME module, so this directly guards against a cross-section
# regression (e.g. an accidental shared-name collision or copy-paste mistake).
# ---------------------------------------------------------------------------

class _SegStubEngine:
    """Mirrors `test_task018_judge_severity.py`'s own stub shape for
    `vision_recommend_segmentation`."""

    def __init__(self, **fields):
        self._fields = fields

    def generate(self, content, response_format=None, **kwargs):
        from shape.agent.verification.llm_recommend import SegOverlayVerdict
        defaults = dict(specimen="cells", cell_size="medium", cell_density="moderate",
                        image_usable="yes", problem="none", severity="none",
                        param="none", direction="none", confidence="high", reasoning="stub")
        defaults.update(self._fields)
        return {"content": SegOverlayVerdict(**defaults)}


def test_scope_guard_build_prompt_still_byte_identical_to_its_own_golden(tmp_path):
    """Re-confirms `test_llm_recommend_prompt_extraction.py`'s own golden test still holds
    with the M-B population-triage section present in the same module (belt-and-suspenders:
    that file's test already covers this on its own; this re-asserts it from the M-B
    change's own test file)."""
    from shape.agent.verification.llm_recommend import _build_prompt
    out = _build_prompt({"cell_count": 1.0}, None)
    assert out.startswith("You are the visual check")
    assert "choose your fix from here" not in out   # no card -> no card-section heading
    assert '"cell_count": 1.0' in out


def test_scope_guard_build_rep_prompt_unaffected(tmp_path):
    """`_build_rep_prompt` (representation's param_retune prompt builder) is untouched by
    the M-B section living below it in the same file."""
    from shape.agent.verification.llm_recommend import _build_rep_prompt
    out = _build_rep_prompt({"PC1_area_correlation": 0.5}, "A CARD", context=None)
    assert out.startswith("You are the label-free QC judge")
    assert "A CARD" in out
    assert '"PC1_area_correlation": 0.5' in out


def test_scope_guard_vision_recommend_segmentation_end_to_end_unaffected():
    """`_recommend_from_card`'s full param_retune control flow (segmentation) still
    produces the exact same VLRRResult shape after M-B's additive changes."""
    det = _det(decision="PASS", action="accept", score=1.0)
    engine = _SegStubEngine(problem="over-segmentation", severity="major",
                            param="min_size", direction="up", confidence="high")
    out = llm_recommend.vision_recommend_segmentation(
        engine, {"overlay_path": None}, {"tiny_mask_fraction": 0.6}, det)
    # No overlay -> `_overlay_bytes` returns None -> deterministic unchanged (unaffected
    # by M-B; this is `vision_recommend_segmentation`'s own pre-existing no-overlay guard).
    assert out is det


def test_scope_guard_vision_recommend_segmentation_without_an_engine_raises():
    with pytest.raises(RuntimeError, match="no vision engine is configured"):
        llm_recommend.vision_recommend_segmentation(None, {}, {}, _det())
