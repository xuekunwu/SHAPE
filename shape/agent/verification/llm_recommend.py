"""Vision-LLM Recommend phase: the LLM reasons problem -> parameter from the tool CARD.

Design (systematic, not a knob table):

  - The DETERMINISTIC metrics own fine mask GEOMETRY (size distribution, solidity,
    merged/fragmented). The vision model is worse at this from a downsampled overlay
    (it called "merged" on every touching pair), so it is NOT asked to re-judge geometry.

  - The multimodal LLM INDEPENDENTLY judges the overlay for the things metrics cannot
    see -- which objects are target cells, missed cells, false masks, wrong scale -- and,
    given the tool's CARD (its parameter manual), PROPOSES the single parameter + direction
    that fixes the problem it found. This replaces the old hard-coded symptom->knob table:
    new failure modes need no new branch, and any tool that ships a CARD becomes tunable.

  - Reliability comes from the VLRR SAFETY NET, not from trusting the LLM blindly: the
    proposal is constrained to the card's vocabulary; its `expected_effect` is derived
    from the card's documented physics so `reconcile` can check the rerun actually did
    what was predicted (a wrong proposal is invalidated and the best attempt kept);
    `apply_bounded` clamps magnitude; the deterministic `_diagnose` remains the no-LLM
    fallback + guard; and `_clamp_action` still lets vision only CONFIRM or TIGHTEN the
    deterministic verdict, never loosen it.

  - Accept TOLERANCE: only a 'major' problem warrants a retune; minor imperfections (a
    few missed/false masks) are accepted, so the loop converges instead of chasing noise.

Safety: no engine / unreadable overlay / bad output -> deterministic result unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

from pydantic import BaseModel

from shape.engine.errors import NoEngineError
from shape.utils.schemas import ExecutionError, VLRRResult

logger = logging.getLogger("shape.agent.verification")

# Severity ladder: the vision judge may CONFIRM or TIGHTEN the deterministic verdict,
# never loosen it (accept->retune allowed; retune->accept is not).
_ACTION_RANK = {"accept": 0, "retune_here": 1}

# Card physics: the deterministic metric a (param, direction) move should shift. Mirrors
# the CARD's "effect to expect" lines, so the rerun can RECONCILE the LLM's proposal. A
# proposal outside these keys is treated as invalid (-> no retune; the deterministic
# guard via the clamp still applies).
_PARAM_EFFECT: dict[tuple[str, str], dict[str, str]] = {
    ("cellprob_threshold", "down"): {"cell_count": "up"},
    ("cellprob_threshold", "up"): {"tiny_mask_fraction": "down"},
    ("diameter", "down"): {"mask_area_median": "down"},
    ("diameter", "up"): {"mask_area_median": "up"},
    ("flow_threshold", "up"): {"large_mask_fraction": "down"},
    ("flow_threshold", "down"): {"large_mask_fraction": "up"},
    ("min_size", "up"): {"tiny_mask_fraction": "down"},
    ("niter", "up"): {"cell_count": "up"},
}


def _clamp_action(llm_action: str, det_action: str) -> str:
    llm_r = min(_ACTION_RANK.get(llm_action, 1), 1)
    det_r = _ACTION_RANK.get(det_action, 0)
    return "retune_here" if max(det_r, llm_r) == 1 else "accept"


class SegOverlayVerdict(BaseModel):
    """What the vision model REPORTS + the parameter move it PROPOSES from the tool card."""
    specimen: str                # what the image shows + WHICH objects are target cells
    cell_size: Literal["tiny", "small", "medium", "large"]
    cell_density: Literal["sparse", "moderate", "crowded", "confluent"]
    image_usable: Literal["yes", "marginal", "no"]
    problem: str                 # one short phrase naming the main problem, or "none"
    severity: Literal["none", "minor", "major"]   # only "major" warrants a retune
    param: Literal["diameter", "cellprob_threshold", "flow_threshold",
                   "min_size", "niter", "none"]   # the single knob to adjust
    direction: Literal["up", "down", "none"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


def _overlay_bytes(context) -> bytes | None:
    path = (context or {}).get("overlay_path")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning("vision recommend: cannot read overlay %s (%s)", path, e)
        return None


def _render_image_context_hint(profile: dict | None) -> str | None:
    """Task 057 M1: the ONE-WAY adapter from a preflight `ImageContextProfile`
    (`image_context.py`) to prior/hint TEXT for the segmentation vision prompt.

    Never a schema merge into `SegOverlayVerdict` -- this only builds an extra text
    paragraph `_build_prompt` prepends; `SegOverlayVerdict.specimen` remains its own
    free-text field, generated fresh by the segmentation judge every time. Gated on
    `specimen_context != "unknown"` (an unknown/uncertain profile returns None, so a
    default/unchanged image's prompt stays byte-identical to before this task).
    `provisional_target_hint` is
    rendered as informational TEXT ONLY -- read by the vision judge qualitatively,
    never a deterministic branch in this (or any) function.
    """
    if not isinstance(profile, dict):
        return None
    specimen_context = profile.get("specimen_context")
    if not specimen_context or specimen_context == "unknown":
        return None
    parts = [f"specimen_context={specimen_context}"]
    classes = profile.get("visible_object_classes") or []
    if classes:
        parts.append(f"visible_object_classes={classes}")
    hint = profile.get("provisional_target_hint")
    if hint and hint != "unknown":
        parts.append(f"provisional_target_hint={hint} (advisory only -- a deferred, "
                     f"not-yet-built capability; nothing enforces it)")
    return ("PRIOR CONTEXT from an automated pre-flight read of this image, taken "
            "BEFORE segmentation (advisory only -- verify against what you actually "
            "SEE in the overlay; treat this as a hint, not a fact): " + "; ".join(parts))


def _build_prompt(metrics: dict, card: str | None = None,
                  verify_refine: bool = False, context_hint: str | None = None) -> str:
    """Assembles the vision prompt from `shape/agent/prompts/cell_segmenter/{verify,
    recommend}.md` (task 032 M3 scaffolding) -- extracted from what was previously an
    inline f-string here, byte-identical (see the co-located test).

    `verify_refine` (044 BRIEF sec B, OFF by default -- an unset flag is base-identical):
    when True, drops `flow_error` from this LOCAL prompt copy only. 043 (n=300) found
    flow_error ANTI-correlates with quality in the tunable regime (oriented rho~=-0.35:
    the model is confident precisely when it under-detects) -- feeding it to the vision
    judge as "lower = better" evidence misleads it. `flow_error` stays in `mask_qc`'s
    dict / `vres.metrics` (base.py:245 attaches the FULL dict, a separate object from
    `m` below) for every non-vision consumer: the flow gate (`localize`), the
    deterministic `_diagnose_under_detection`, `adapt_segmentation`,
    Reconcile/`classify_lever`, plus the gate-like `planner.py:839` credibility review
    and `final_review.py` (left intact -- a different, gross-unfit-regime question).

    `context_hint` (task 057 M1, OPTIONAL -- default None is BYTE-IDENTICAL to the
    pre-057 prompt): pre-rendered prior/hint text from `_render_image_context_hint`,
    prepended as its own paragraph before the segmentation verify/recommend prompt.
    """
    from shape.agent.prompts import load_prompt

    m = {k: round(float(v), 4) for k, v in metrics.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if verify_refine:
        m.pop("flow_error", None)
    card_section = (f"\nTOOL CARD -- how this segmenter's parameters work; choose your fix "
                    f"from here:\n{card.strip()}\n" if card else "")
    verify_tmpl = load_prompt("cell_segmenter", "verify")
    recommend_tmpl = load_prompt("cell_segmenter", "recommend")
    prefix = f"{context_hint}\n\n" if context_hint else ""
    return prefix + verify_tmpl + recommend_tmpl.format(card_section=card_section,
                                                         metrics_json=json.dumps(m))


def _normalize_severity(rec) -> str:
    """Coerce the judge's severity to a valid ordinal ('none'/'minor'/'major').

    `rec.severity` SHOULD already be one of these three -- the response schema
    declares it a required Literal -- but the runtime boundary (SDK/back-end drift,
    a parse fallback, a duck-typed stand-in) can still hand back None/missing/
    invalid, and a silently-dropped severity throws away the judge's own retune
    (the task-006 lesson: a reasoning decision must not be discarded downstream).
    When the value isn't one of the three, INFER it from what the judge itself
    reported: a named problem (`rec.problem` not empty/"none") plus a concrete
    param+direction (neither "none") means it found something real to fix ->
    'major'; otherwise -> 'none'. A valid severity is kept as-is.
    """
    sev = getattr(rec, "severity", None)
    if sev in ("none", "minor", "major"):
        return sev
    named_problem = getattr(rec, "problem", None) not in (None, "", "none")
    concrete_fix = (getattr(rec, "param", "none") not in (None, "none")
                    and getattr(rec, "direction", "none") not in (None, "none"))
    return "major" if named_problem and concrete_fix else "none"


def _verdict_to_recommendation(rec: SegOverlayVerdict):
    """Turn the vision PROPOSAL into (action, param_delta, expected_effect).

    Accept TOLERANCE: only a 'major' problem with a concrete knob triggers a retune;
    'none'/'minor' or no knob is accepted, so the loop does not chase trivial flaws. The
    proposal is validated against the card's documented vocabulary, and its
    expected_effect comes from the card's physics so the Rerun phase can reconcile it.
    Normalizes `rec.severity` internally (`_normalize_severity`) rather than trusting the
    raw value -- a missing/invalid severity must not silently discard a named problem+fix
    (task-006 lesson). Kept single-arg (`rec` only) so it fits the shared helper's generic
    `to_recommendation(rec)` callable contract alongside the other CARD-driven verdicts
    (`_rep_verdict_to_recommendation`, `initial_params._initial_config_to_recommendation`),
    none of which need severity normalization."""
    severity = _normalize_severity(rec)
    if severity != "major" or rec.param == "none" or rec.direction == "none":
        return "accept", None, {}
    ee = _PARAM_EFFECT.get((rec.param, rec.direction))
    if ee is None:                       # out-of-vocabulary / nonsensical direction
        return "accept", None, {}
    return "retune_here", {rec.param: rec.direction}, dict(ee)


def _stamp_vision_profile(out: VLRRResult, rec) -> VLRRResult:
    """Stamp the vision judge's image profile (specimen, cell size, cell density) onto `out`."""
    out.judge_specimen = getattr(rec, "specimen", None)
    out.judge_cell_size = getattr(rec, "cell_size", None)
    out.judge_cell_density = getattr(rec, "cell_density", None)
    return out


def _default_notes_fn(rec, sev, ptok) -> str:
    """The ORIGINAL, seg/representation-shaped notes format (extracted verbatim, task 038
    follow-up Option B) -- both `SegOverlayVerdict` and `RepresentationVerdict` share this
    exact field shape (`param`/`direction`/`image_usable`/`confidence`/`specimen`/
    `cell_size`/`cell_density`/`reasoning`, `RepresentationVerdict`'s docstring says so
    explicitly), so this is the DEFAULT `notes_fn` `_recommend_from_card` uses when a
    caller does not supply its own -- byte-identical to the pre-generalization inline
    construction, so `vision_recommend_segmentation`/`vision_recommend_representation`
    need ZERO changes to keep producing today's exact string."""
    return (f"[vision] tok={ptok} problem={getattr(rec, 'problem', '') or '?'!r} "
           f"sev={sev} fix={rec.param}/{rec.direction} "
           f"usable={rec.image_usable} ({rec.confidence}) "
           f"specimen={getattr(rec, 'specimen', '') or '?'!r} "
           f"size={rec.cell_size} density={rec.cell_density}: "
           + (rec.reasoning or "")[:130])


def _recommend_from_card(engine, *, evidence_images, prompt, schema, to_recommendation,
                         deterministic: VLRRResult, notes_fn=None) -> VLRRResult:
    """Shared Recommend-phase control flow: CARD + evidence -> structured vision verdict ->
    (action, param_delta, expected_effect) -> `_clamp_action` (vision may only CONFIRM/
    TIGHTEN the deterministic verdict). No engine: a RuntimeError; a failed or malformed call: an
    `ExecutionError` (the call being verified fails), never the deterministic verdict.

    Extracted from `vision_recommend_segmentation` (behavior-preserving) so any VLRR stage
    with a tool CARD can reuse this same mechanism -- see `vision_recommend_representation`.
    `to_recommendation(rec) -> (raw_action, param_delta, expected_effect)` turns the
    schema-specific verdict into the generic recommendation shape.

    `notes_fn(rec, sev, ptok) -> str` (task 038 follow-up, Option B -- OPTIONAL, defaults
    to `_default_notes_fn`): builds the compact `notes` string from the verdict. This is
    the ONE place `_recommend_from_card` used to assume a `SegOverlayVerdict`-shaped `rec`
    (`rec.param`/`rec.direction`/`rec.image_usable`/`rec.cell_size`/`rec.cell_density`
    accessed directly, no `getattr` fallback) -- a THIRD verdict schema with naturally
    DIFFERENT fields (no cell_size/cell_density/specimen/image_usable at all) would
    previously have raised an AttributeError here. A caller now supplies its OWN
    `notes_fn` when its schema's fields differ from seg/representation's shared shape;
    `None` (the default, used by BOTH existing callers, unchanged) reproduces today's
    EXACT string, byte-for-byte. (The cropper's own geometry-judge caller that motivated
    this generalization -- task 038 follow-up, item 3 -- was itself REMOVED by a later
    round, human directive: widening `margin_frac` to fix low dominance is backwards, see
    CARD.md SS1/SS5; this generalization stays as a harmless, genuine shared-layer
    improvement for any FUTURE third schema.)
    """
    if engine is None:                    # a system error, never the deterministic answer
        raise NoEngineError("vision recommend: no vision engine is configured")
    fmt_notes = notes_fn or _default_notes_fn
    try:
        content = [prompt, *evidence_images]
        resp = engine.generate(content, response_format=schema)
        rec = resp.get("content") if isinstance(resp, dict) else resp
        if not hasattr(rec, "severity"):
            raise ExecutionError(f"vision recommend: malformed answer ({type(rec).__name__})")
        # Normalized once here, for BOOKKEEPING (judge_severity / notes / keep-best) --
        # a missing/invalid severity must not silently discard a judge that named a real
        # problem+fix (task-006 lesson). `to_recommendation` (e.g. `_verdict_to_recommendation`)
        # normalizes independently, internally, for its own accept/retune gate.
        sev = _normalize_severity(rec)
        # token count makes it verifiable that the evidence was actually sent: a
        # high-detail image is ~1k+ prompt tokens; a near-zero count means the
        # image never reached the model.
        ptok = (resp.get("usage", {}) or {}).get("prompt_tokens") if isinstance(resp, dict) else None

        raw_action, pd, ee = to_recommendation(rec)
        # vision may only confirm/tighten the deterministic verdict, never loosen it.
        action = _clamp_action(raw_action, deterministic.action)
        if action == "retune_here" and not pd:
            # clamp upgraded det's retune but vision added no knob -> keep det's, but
            # record the judge's assessment so keep-best/stop still see this attempt.
            deterministic.judge_severity = sev
            deterministic.judge_problem = rec.problem
            deterministic.reasoning_full = rec.reasoning or ""
            return _stamp_vision_profile(deterministic, rec)

        decision = ("PASS" if action == "accept"
                    else "WARNING" if deterministic.decision == "PASS"
                    else deterministic.decision)
        out = VLRRResult(
            decision, deterministic.scalar_score, {},
            action=action,
            param_delta=pd if action == "retune_here" else None,
            expected_effect=(ee or deterministic.expected_effect) if action == "retune_here" else {},
            upstream_target=deterministic.upstream_target,
            # the judge's own per-attempt severity closes the loop (keep-best + stop)
            judge_severity=sev,
            judge_problem=rec.problem,
            notes=fmt_notes(rec, sev, ptok),
        )
        # Durable-record counterpart to the compact `notes` above: the FULL judge
        # rationale, untruncated, so vlrr_trace.json retains it. The in-context
        # digest (memory.compact_verification / get_actions(llm_safe=True)) only
        # ever reads `notes`, so this does not enlarge the prompt token budget.
        out.reasoning_full = rec.reasoning or ""
        return _stamp_vision_profile(out, rec)
    except ExecutionError:
        raise
    except Exception as e:  # an execution error of the call being verified, never the deterministic answer
        raise ExecutionError(f"vision recommend failed: {type(e).__name__}: {e}") from e


def vision_recommend_segmentation(engine_mm, context, metrics,
                                  deterministic: VLRRResult, card=None,
                                  verify_refine: bool = False) -> VLRRResult:
    """Vision proposal on the overlay (no engine: a RuntimeError; a failed call: an `ExecutionError`).

    `verify_refine` (044, OFF by default) drops flow_error from this prompt's metrics copy
    only -- see `_build_prompt`.

    Task 057 M1: `context["upstream"]["image_context_profile"]` (threaded here by
    `integration.latest_stage_metrics`, unchanged from every other `context["upstream"]`
    consumer's point of view) is rendered as a one-way prior/hint paragraph via
    `_render_image_context_hint` -- None when absent/unknown, so a run with no preflight
    profile (or an unknown one) builds a BYTE-IDENTICAL prompt to before this task.
    """
    if engine_mm is None:
        raise NoEngineError("vision recommend (segmentation): no vision engine is configured")
    overlay = _overlay_bytes(context)
    if overlay is None:
        return deterministic
    profile = (context or {}).get("upstream", {}).get("image_context_profile")
    context_hint = _render_image_context_hint(profile)
    prompt = _build_prompt(metrics, card, verify_refine=verify_refine, context_hint=context_hint)
    return _recommend_from_card(
        engine_mm, evidence_images=[overlay], prompt=prompt, schema=SegOverlayVerdict,
        to_recommendation=_verdict_to_recommendation, deterministic=deterministic)


# --------------------------------------------------------------------------- #
# Representation (DINOv3 analyzer) recommend: reuses `_recommend_from_card` with a
# CARD-driven verdict over the analyzer's label-free embedding QC (no mask overlay --
# a feature space has none -- so evidence is whatever crop-vision montage `context`
# carries, which may be empty; a text-only recommend over the metrics is still valid).
# --------------------------------------------------------------------------- #
# The mid-loop retune vocabulary is exactly {field_aware, augment_color}: the two floor levers
# `adjudicate`'s batch_driven branch chooses between (gated on `_source_reliable`), each with a
# documented, reconcilable effect on the same batch/group silhouette localize() gates on.
# Adaptation depth (formerly the bare `freeze_backbone` boolean) is a PRE-FLIGHT, one-shot
# decision -- every alternative costs a full fine-tune -- so it is NOT in this mid-loop
# vocabulary; it lives only in `initial_params` (up-front) and the CARD. A mid-loop
# recommendation of `freeze_backbone` used to validate here and then be dropped by `map_params`
# (stage_map no longer maps it), i.e. a silent no-op retune -- the exact class of dead lever
# this tool's action space was cleaned of.
_REP_PARAM_EFFECT: dict[tuple[str, str], dict[str, str]] = {
    ("field_aware", "on"): {"silhouette_by_batch": "down"},
    ("field_aware", "off"): {},
    # task 021: augment_color's brightness/contrast jitter is a per-image intensity
    # decorrelator (valid even on single-channel data, see the CARD), so -- unlike before --
    # it now has a reconcilable effect on the same batch-confound metric field_aware targets.
    ("augment_color", "on"): {"silhouette_by_batch": "down"}, ("augment_color", "off"): {},
}
# (param, direction) -> the actual param_delta VALUE (on/off -> True/False).
_REP_PARAM_VALUE = {
    ("field_aware", "on"): True, ("field_aware", "off"): False,
    ("augment_color", "on"): True, ("augment_color", "off"): False,
}
# (param, direction) -> the VLRR-internal param_delta KEY (identity-mapped; both match
# `stage_map.py`'s representation `param_map`).
_REP_PARAM_KEY = {"field_aware": "field_aware", "augment_color": "augment_color"}


class RepresentationVerdict(BaseModel):
    """What the vision/text model REPORTS about the embedding's label-free QC + the
    parameter move it PROPOSES from the tool card (mirrors SegOverlayVerdict's shape)."""
    specimen: str                 # what the crops show + WHICH objects are target cells
    cell_size: Literal["tiny", "small", "medium", "large"]
    cell_density: Literal["sparse", "moderate", "crowded", "confluent"]
    image_usable: Literal["yes", "marginal", "no"]
    problem: str                  # one short phrase naming the main problem, or "none"
    severity: Literal["none", "minor", "major"]   # only "major" warrants a retune
    # The mid-loop retune vocabulary: the two reconcilable floor levers only. Adaptation
    # depth is pre-flight (a full fine-tune per change), so it is intentionally not here.
    param: Literal["field_aware", "augment_color", "none"]   # the single knob to adjust
    direction: Literal["on", "off", "none"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


def _build_rep_prompt(metrics: dict, card: str | None = None, context: dict | None = None) -> str:
    m = {k: round(float(v), 4) for k, v in metrics.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)}
    card_section = (f"\nTOOL CARD -- how this representation's input strategy works; choose "
                    f"your fix from here:\n{card.strip()}\n" if card else "")
    # task 021: the up-front sampled-crop vision profile (`crop_vision.crop_vision_profile`),
    # when threaded into `context` (see `trial_context`), is dataset-level input evidence the
    # mid-loop recommend was previously ignoring even though it was already sitting in context.
    crop_vision = (context or {}).get("crop_vision_profile")
    crop_vision_section = (
        f"\nSampled-crop vision profile (dataset-level, judged before training): "
        f"{json.dumps(crop_vision, default=str)}\n" if crop_vision else "")
    # task 021 Stage 1b: THIS trial's already-applied config (threaded via `trial_context`),
    # so the LLM does not re-propose a lever that is already on/off -- the no-op-retune bug
    # (a "fix" identical to the current params -> StepVerifier's oscillation guard kills the
    # whole loop with zero effective retunes). Only shown when at least one of the 4 mid-loop
    # knobs is actually known for this trial.
    current_config = {k: (context or {}).get(k) for k in
                      ("field_aware", "augment_color")
                      if (context or {}).get(k) is not None}
    current_config_section = (
        f"\nTHIS trial's CURRENT config (already applied -- do NOT propose a value it "
        f"already has; pick a DIFFERENT knob or concede with param='none' if nothing else "
        f"plausibly helps): {json.dumps(current_config, default=str)}\n"
        if current_config else "")
    return (
        "You are the label-free QC judge of a self-supervised cell-representation "
        "verifier (DINOv3 embeddings of single-cell crops). An evidence montage may be "
        "attached (a sampled crop next to its nearest neighbours in embedding space); if "
        "none is attached, reason from the metrics (and the crop-vision profile, if given) "
        "alone.\n\n"
        "STEP 1 -- if a montage is attached, recognize it: specimen, typical cell size, "
        "density.\n\n"
        "STEP 2 -- find the ONE main problem, if any: the embedding is SIZE-DOMINATED "
        "(neighbours merely share cell SIZE, not cell STATE); NN-VISUALLY inconsistent "
        "(neighbours in feature space do not look alike); or a SOURCE/BATCH CONFOUND "
        "(silhouette_by_batch exceeds silhouette_by_group -- the embedding tracks which "
        "source image/field a crop came from more than its biological state). Do NOT "
        "judge mask geometry (not this stage's job).\n\n"
        "STEP 3 -- rate severity, defaulting to ACCEPT. Only a CLEARLY major problem that "
        "the card's parameter can plausibly fix warrants a retune; a merely good-enough "
        "result is severity='minor' or 'none' and param='none'.\n\n"
        "STEP 4 -- if severity is 'major', use the TOOL CARD to choose the SINGLE "
        "parameter and direction whose documented effect best fixes the problem you "
        "named; otherwise param='none', direction='none'. For a source/batch confound: "
        "field_aware needs a RELIABLE per-crop source id (real well/plate metadata AND "
        "multiple cells per source field); if the source is a filename guess with "
        "mostly-singleton fields, field_aware is near-inert -- prefer augment_color (a "
        "per-image intensity decorrelator, valid even on single-channel data) instead; "
        "if neither lever plausibly helps (e.g. a genuinely multi-channel, data-side "
        "confound), concede with param='none'. Never propose a knob/direction THIS "
        "trial's current config (below) already has -- that is a no-op, not a fix.\n"
        f"{card_section}"
        f"{crop_vision_section}"
        f"{current_config_section}\n"
        "Report: specimen; cell_size; cell_density; image_usable; problem (one short "
        "phrase or 'none'); severity; param; direction; confidence; and a one-sentence "
        "reasoning citing the evidence.\n\n"
        f"Deterministic metrics -- evidence you MAY weigh (e.g. PC1_area_correlation near "
        f"1 means the embedding just tracks cell size; nn_visual_consistency near 0 means "
        f"neighbours do not look alike; silhouette_by_batch clearly above "
        f"silhouette_by_group means a source/batch confound): {json.dumps(m)}"
    )


def _rep_verdict_to_recommendation(rec: RepresentationVerdict):
    """Turn the vision/text PROPOSAL into (action, param_delta, expected_effect).

    `param_delta` uses the VLRR-internal param name/value (`_REP_PARAM_KEY`/`_REP_PARAM_VALUE`;
    "input" for input_strategy, mapped to the tool's `input_strategy` kwarg by `stage_map`,
    identity for the other three), even though the verdict's card-facing `param` reads
    "input_strategy" -- matching the deterministic branches in `representation.py`.
    """
    if rec.severity != "major" or rec.param == "none" or rec.direction == "none":
        return "accept", None, {}
    ee = _REP_PARAM_EFFECT.get((rec.param, rec.direction))
    value = _REP_PARAM_VALUE.get((rec.param, rec.direction))
    if ee is None or value is None:      # out-of-vocabulary / nonsensical direction
        return "accept", None, {}
    return "retune_here", {_REP_PARAM_KEY[rec.param]: value}, dict(ee)


def _rep_montage_bytes(context) -> list:
    """Any crop-vision evidence montage `context` carries; [] when none (P1 has no
    producer yet -- P2's crop-vision profile attaches one), so the recommend proceeds
    text-only rather than skipping the LLM entirely."""
    path = (context or {}).get("montage_path")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "rb") as f:
            return [f.read()]
    except Exception as e:
        logger.warning("vision recommend (representation): cannot read montage %s (%s)", path, e)
        return []


def vision_recommend_representation(engine_mm, context, metrics,
                                    deterministic: VLRRResult, card=None) -> VLRRResult:
    """CARD-driven recommend over the analyzer's label-free embedding QC; falls back to
    `deterministic` on any issue. No overlay exists (a feature space, not an image) so
    evidence is whatever montage `context` carries -- may be empty (text-only recommend)."""
    prompt = _build_rep_prompt(metrics, card, context)
    return _recommend_from_card(
        engine_mm, evidence_images=_rep_montage_bytes(context), prompt=prompt,
        schema=RepresentationVerdict, to_recommendation=_rep_verdict_to_recommendation,
        deterministic=deterministic)


# =============================================================================================
# Task 038 M-B: the shared, tool-agnostic POPULATION-TRIAGE judge.
#
# `_recommend_from_card` above is the shared judge for the `param_retune` verdict type (one
# evidence -> one verdict -> a parameter move); this section is its sibling for the OTHER
# verdict type, `population_triage` (one verdict PER ITEM -> keep/reject,
# landed by excluding a reject from a downstream population rather than retuning anything).
# ADDITIVE ONLY -- nothing above this banner is touched by M-B (scope guard:
# the segmenter/analyzer `_recommend_from_card` path stays byte-identical this round).
#
# Per-tool surface for a `population_triage` tool: its own
# CARD.md (levers/verdict-declaration/calibration/complementarity, task 038 M-A) + quant.py
# (GT-free numbers) + a THIN adapter supplying `image_fn`/`quant_fn`/`evidence_fn`/
# `kept_predicate` to `recommend_population_triage` below -- see `crop_triage.
# vision_recommend_crop` for the cropper's own adapter (the only `population_triage` tool
# today).
# =============================================================================================

_TRIAGE_PREAMBLE = (
    "You are the POPULATION-TRIAGE judge for ONE item. You are given: (1) this item's own "
    "evidence image; (2) deterministic QUANT numbers computed from this same item; (3) any "
    "extra deterministic EVIDENCE below. The attached TOOL CARD declares this item's exact "
    "verdict semantics -- follow its VERDICT DECLARATION for the keep/reject criteria, its "
    "CALIBRATION for what makes a quant number anomalous, and its COMPLEMENTARITY/"
    "RELIABILITY rules for which evidence channel to trust for which kind of problem. "
    "KEEP is the OVERWHELMING default outcome -- reject is a RARE exception that needs "
    "clear, un-second-guessable evidence, not a plausible-sounding impression."
)

_TRIAGE_RECONCILE_INSTRUCTION = (
    "\n\nARBITRATE all the evidence into ONE call, following the CARD's COMPLEMENTARITY/"
    "RELIABILITY guidance for which channel leads per problem. The QUANT number is "
    "AUTHORITATIVE on any axis the CARD says it leads: if the quant value for the DEFECT "
    "you are about to name is GOOD by the CARD's own CALIBRATION range (not anomalous), "
    "you MUST NOT reject on that axis -- no matter how strong your own visual impression "
    "feels, a confidently-good quant number OVERRULES it, full stop. Before naming ANY "
    "problem, first CHECK whether that problem has a mapped quant number in the evidence "
    "above; if it does, STATE that exact number in your reasoning and explicitly say "
    "whether it is anomalous or good by the CARD's calibration BEFORE deciding. For EACH "
    "quant number that IS anomalous by the CARD's CALIBRATION ranges, you must additionally "
    "RECONCILE it against the image before it can drive your call: state what the number "
    "implies and whether the image confirms or contradicts it (e.g. a number that flags a "
    "defect the image does NOT actually show -- the defect the number implies is not "
    "visible -- means the low number has an innocent explanation, so it must NOT drive a "
    "reject). A number the image CONTRADICTS must NOT drive a reject. When your own visual "
    "read and a GOOD quant number disagree, DEFER TO THE QUANT NUMBER, not your impression "
    "-- a small, low-contrast, or low-resolution image is an easy image to misjudge by eye; "
    "the deterministic number is not. Report verdict (keep/reject), confidence (high/low), "
    "a one-phrase problem (or 'none'), and one sentence of reasoning that NAMES the "
    "relevant quant(s) (their VALUE, not just that they exist), states good-vs-anomalous "
    "for each, and HOW each was reconciled."
)


def build_judge_prompt(card: str | None, quant: dict, evidence: dict | None = None) -> str:
    """Tool-agnostic population-triage judge prompt (task 038 M-B). Assembles labelled QUANT
    numbers + any extra deterministic EVIDENCE (a `{channel_name: text}` mapping, e.g. a
    pixel-gate verdict) + the CARD text + the universal reconcile-before-reject closing
    instruction. Replaces `crop_triage._build_arbitration_prompt`. Nothing tool-specific is
    hard-coded here -- a tool's own item description, reject taxonomy, calibration ranges,
    and complementarity/reliability rules live ENTIRELY in ITS OWN CARD (task 038 M-A); this
    function only ASSEMBLES the channels, mirroring `_build_prompt`/`_build_rep_prompt`'s
    evidence+metrics+CARD shape above for the param_retune judge.
    """
    q = {k: round(float(v), 4) for k, v in (quant or {}).items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)}
    quant_section = (
        f"\n\nQUANT evidence (deterministic, 0-1, higher=better; labelled; ONE input, not "
        f"the decider): {json.dumps(q)}" if q else
        "\n\nQUANT evidence: unavailable (scoring failed) -- arbitrate on the image and any "
        "other evidence alone."
    )
    evidence_section = "".join(
        f"\n\n{name} evidence: {text}" for name, text in (evidence or {}).items())
    card_section = (f"\n\nTOOL CARD -- this item's own verdict semantics, calibration, and "
                    f"complementarity rules:\n{card.strip()}" if card else "")
    return _TRIAGE_PREAMBLE + quant_section + evidence_section + card_section + _TRIAGE_RECONCILE_INSTRUCTION


def judge_item(engine, item, *, schema, card, image_fn, quant_fn=None, evidence_fn=None,
               problem_gate: set[str] | None = None, extra_recording_fn=None):
    """Judge ONE item for population triage (task 038 M-B, generalizes
    `crop_triage.arbitrate_crop`): reads the item's evidence image (`image_fn(item)`), its
    quant numbers (`quant_fn(item)`, optional), and any extra evidence (`evidence_fn(item)`,
    optional), builds the fused prompt via `build_judge_prompt`, and asks `engine` for a
    `schema`-shaped verdict -- `schema` must expose `.verdict`/`.confidence`/`.problem`, the
    population-triage verdict CONTRACT (e.g. `crop_triage.CropTriageVerdict`; mirrors how
    `_recommend_from_card` above takes `schema` as a caller-supplied parameter). Returns
    (reject, verdict). Keep-biased + confidence-gated: `reject` is True ONLY on
    `verdict=="reject" and confidence=="high"`. Any failure (no engine, unreadable evidence
    image, quant/evidence error, engine error, malformed response) -> (False, None) so the
    caller's prior verdict stands -- same graceful-degrade contract as `_recommend_from_card`.

    `problem_gate` (task 038 follow-up, OPTIONAL, tool-agnostic -- mirrors `kept_predicate`'s
    caller-supplied-callback shape below): when given, a reject is valid ONLY if
    `rec.problem` is IN this set -- a reject naming any other problem is DOWNGRADED to keep
    (defense in depth alongside the schema's own enum constraint on `problem`, in case a
    caller's `schema` allows free text or an engine does not enforce structured output). This
    module stays tool-agnostic -- it is the CALLER's job to supply its own valid-reasons set
    (e.g. `crop_triage.vision_recommend_crop` passes the cropper CARD's exact reasons); `None`
    (default) -> no gating, unchanged behavior for any caller that has not adopted this.

    Task 038 follow-up (part B redesign, human directive): an EARLIER round added
    `reject_veto_fn`/`force_reject_fn` -- ITEM-AWARE hard-threshold hooks that silently
    downgraded/forced the judge's own verdict on a numeric quant condition. Both are
    REMOVED ("NO hard threshold override") -- a caller now feeds the SAME quant signal
    into the fused prompt as LABELED evidence TEXT (`evidence_fn`, e.g. `crop_triage`'s
    own `_evidence_strength_text`) instead, and the LLM makes the FINAL call itself,
    informed but never silently overruled. `problem_gate` is UNCHANGED (it constrains
    WHICH categories are even nameable -- a structural vocabulary limit, not a
    numeric-threshold override of the judge's own reasoning).

    `extra_recording_fn(item, rec) -> None` (task 038 follow-up, part B, OPTIONAL,
    tool-agnostic): called for EVERY judged item (keep or reject) with the raw verdict
    `rec` -- lets a caller record ADDITIONAL per-item fields onto `item` from a SCHEMA
    field this module has no vocabulary for (e.g. the cropper's own `geometry`
    tighten/leave recommendation, `crop_triage`'s own concrete use), keeping this module
    itself tool-agnostic (mirrors why `notes_fn` exists on `_recommend_from_card` above).
    Any exception is caught and logged, never raised -- same "evidence, never a silent
    break" posture as everywhere else in this module. `None` (default) -> no-op,
    unchanged behavior for any caller that has not adopted this.
    """
    if engine is None:
        raise NoEngineError("population-triage judge: no vision engine is configured")
    img = image_fn(item)
    if img is None:
        return False, None
    try:
        quant = quant_fn(item) if quant_fn else {}
        evidence = evidence_fn(item) if evidence_fn else None
        prompt = build_judge_prompt(card, quant, evidence)
        resp = engine.generate([prompt, img], response_format=schema)
        rec = resp.get("content") if isinstance(resp, dict) else resp
        if not hasattr(rec, "verdict"):
            raise ExecutionError(f"population-triage judge: malformed answer ({type(rec).__name__})")
        reject = (rec.verdict == "reject" and rec.confidence == "high"
                 and (problem_gate is None or rec.problem in problem_gate))
        if extra_recording_fn:
            try:
                extra_recording_fn(item, rec)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("population-triage judge: extra_recording_fn failed (%s)", exc)
        return reject, rec
    except ExecutionError:
        raise
    except Exception as exc:  # an execution error of the call being verified, never a silent keep
        raise ExecutionError(f"population-triage judge failed: {type(exc).__name__}: {exc}") from exc


def judge_items(engine, items: list, *, schema, card, image_fn, quant_fn=None,
                evidence_fn=None, sample_cap: int | None = None,
                problem_gate: set[str] | None = None, extra_recording_fn=None) -> dict:
    """Judge a batch of items for population triage (task 038 M-B, generalizes
    `crop_triage.arbitrate_crops`): judges up to `sample_cap` items (deterministic, evenly
    spaced; None = all), annotating each judged, REJECTED item IN PLACE with
    `vision_reject=True` / `vision_problem`. UNLIKE 041's `triage_crops`, the candidate set
    is NOT pre-filtered by any deterministic gate here -- every item is a candidate, so the
    judge can RESCUE one a gate alone would have wrongly excluded (a tool's own `evidence_fn`
    may still surface a deterministic gate's verdict as EVIDENCE the judge weighs, never a
    pre-filter -- see `crop_triage`'s cropper adapter for the concrete example). No engine ->
    no-op. `problem_gate`/`extra_recording_fn` (both optional, tool-agnostic): passed
    straight through to `judge_item` -- see its own docstring.

    The returned summary also carries `"verdicts"`: one dict per JUDGED item (kept or
    rejected) -- `{item_id, reject, verdict, confidence, problem, reasoning}` (`item_id` =
    the item's own `crop_path` when present, else its `id(item)` as a last resort) -- task
    038 run_shape_crop.py follow-up's full-process trace needs the FULL per-item verdict,
    not just the rejected ones `vision_reject`/`vision_problem` already capture.
    """
    if engine is None and items:
        raise NoEngineError("population-triage judge: no vision engine is configured")
    if not items:
        return {"judged": 0, "rejected": 0, "engine": engine is not None, "verdicts": []}
    candidates = list(items)
    if sample_cap is not None and len(candidates) > sample_cap:
        # deterministic sample: evenly spaced, so the summary is reproducible
        step = len(candidates) / float(sample_cap)
        candidates = [candidates[int(i * step)] for i in range(sample_cap)]
    judged = rejected = 0
    verdicts = []
    for item in candidates:
        reject, verdict = judge_item(engine, item, schema=schema, card=card,
                                     image_fn=image_fn, quant_fn=quant_fn,
                                     evidence_fn=evidence_fn, problem_gate=problem_gate,
                                     extra_recording_fn=extra_recording_fn)
        judged += 1
        verdicts.append({
            "item_id": item.get("crop_path") if isinstance(item, dict) else None,
            "reject": reject,
            "verdict": getattr(verdict, "verdict", None) if verdict else None,
            "confidence": getattr(verdict, "confidence", None) if verdict else None,
            "problem": getattr(verdict, "problem", None) if verdict else None,
            "reasoning": getattr(verdict, "reasoning", None) if verdict else None,
        })
        if reject:
            rejected += 1
            item["vision_reject"] = True
            item["vision_problem"] = getattr(verdict, "problem", "") if verdict else ""
    return {"judged": judged, "rejected": rejected, "engine": True, "verdicts": verdicts}


def write_kept_manifest(items: list[dict], metadata_path: str, kept_predicate) -> str | None:
    """The population-triage verdict LANDER (task 038 M-B, generalizes
    `crop_triage.write_kept_crops_manifest`): gives a `population_triage` tool's reject
    verdict downstream TEETH by persisting a sidecar JSON listing the KEPT items' own
    `"crop_path"` key (the item-identity field population-triage items are expected to
    carry), so a downstream consumer can exclude a reject from its population instead of
    ingesting every item unconditionally. `kept_predicate(item) -> bool` decides inclusion --
    supplied entirely by the TOOL adapter (e.g. the cropper's `not vision_reject`, its CARD's
    VERDICT DECLARATION), never hard-coded here, so this stays agnostic about WHAT "kept"
    means beyond "the caller decided".

    NAMING NOTE (task 038 M-B, a known genericness gap -- flagged, not silently papered over):
    the sidecar filename is derived by swapping `metadata_path`'s "cell_crops_metadata_"
    prefix for "cell_crops_kept_manifest_" -- the CROPPER's OWN naming convention (matches
    `Cell_State_Analyzer_Tool._load_kept_crops_manifest`'s read side, unchanged since M-A).
    A future `population_triage` tool with a different metadata-file naming scheme would need
    this parameterized; not done this round -- only one such tool exists today, and the agreed
    3-argument signature for this function does not name a naming-transform parameter either.

    Non-destructive: never touches an item's own file or the metadata file itself, only adds
    this new sidecar. Any failure (missing/unexpected metadata_path, write error) -> no
    manifest written, so a caller that checks for its existence falls back to an unfiltered
    read -- same graceful-degrade contract as the rest of this module. Returns the manifest
    path written, or None.
    """
    if not metadata_path:
        return None
    try:
        basename = os.path.basename(metadata_path)
        if "cell_crops_metadata_" not in basename:
            logger.warning("population-triage: unexpected metadata_path %r; "
                           "skipping kept-items manifest", metadata_path)
            return None
        manifest_path = os.path.join(
            os.path.dirname(metadata_path),
            basename.replace("cell_crops_metadata_", "cell_crops_kept_manifest_"))
        kept_paths = [i["crop_path"] for i in items if kept_predicate(i) and i.get("crop_path")]
        with open(manifest_path, "w") as fh:
            json.dump({"kept_crop_paths": kept_paths}, fh)
        return manifest_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("population-triage: failed to write kept-items manifest (%s)", exc)
        return None


def write_triage_trace(verdicts: list[dict], metadata_path: str) -> str | None:
    """Sidecar JSON with the FULL per-item triage verdict (`judge_items`'s own
    `summary["verdicts"]`: `{item_id, reject, verdict, confidence, problem, reasoning}` per
    JUDGED item, kept or rejected) -- `write_kept_manifest` only persists the KEPT paths,
    losing WHY each decision was made; this is that missing detail, for a full-process
    trace (task 038 run_shape_crop.py follow-up: "so we can see exactly what SHAPE did,
    per crop"). Same sidecar-naming idiom as `write_kept_manifest` (swap
    "cell_crops_metadata_" for "cell_crops_triage_trace_"), same graceful-degrade contract
    (missing/unexpected metadata_path or write error -> no file, never raises).
    """
    if not metadata_path:
        return None
    try:
        basename = os.path.basename(metadata_path)
        if "cell_crops_metadata_" not in basename:
            logger.warning("population-triage: unexpected metadata_path %r; "
                           "skipping triage-trace sidecar", metadata_path)
            return None
        trace_path = os.path.join(
            os.path.dirname(metadata_path),
            basename.replace("cell_crops_metadata_", "cell_crops_triage_trace_"))
        with open(trace_path, "w") as fh:
            json.dump({"verdicts": verdicts}, fh)
        return trace_path
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("population-triage: failed to write triage-trace sidecar (%s)", exc)
        return None


_TRIAGE_DECISION_RANK = {"PASS": 0, "WARNING": 1, "FAIL": 2}
_TRIAGE_RANK_DECISION = {v: k for k, v in _TRIAGE_DECISION_RANK.items()}


def recommend_population_triage(engine, items: list[dict] | None, deterministic: VLRRResult, *,
                                schema, card: str | None, image_fn, quant_fn=None,
                                evidence_fn=None, sample_cap: int | None = None,
                                metadata_path: str | None = None,
                                kept_predicate=None,
                                problem_gate: set[str] | None = None,
                                extra_recording_fn=None) -> VLRRResult:
    """The shared Recommend-phase POPULATION-TRIAGE driver (task 038 M-B, generalizes
    `crop_triage.vision_recommend_crop`'s tighten logic): judges each item (`judge_items`),
    writes the kept-items manifest when the engine actually ran (`write_kept_manifest`, gives
    the reject verdict its downstream teeth), then TIGHTENS `deterministic` by ONE severity
    notch (PASS->WARNING->FAIL) if anything was rejected -- never loosens it, never touches
    `scalar_score`/`action`/`param_delta`. No engine / no items / any failure ->
    `deterministic` UNCHANGED, mirroring `_recommend_from_card`'s own contract for the OTHER
    verdict type (`param_retune`).

    `metadata_path` + `kept_predicate` (both optional): when BOTH are given AND the engine
    actually ran, also persists the kept-items manifest AND the full-verdict triage-trace
    sidecar (`write_triage_trace`, task 038 run_shape_crop.py follow-up) -- the trace is
    written whenever judging actually happened, even if nothing was rejected (a caller
    tracing "what did the judge see" needs the KEPT verdicts too, not just rejects).
    Either `metadata_path`/`kept_predicate` omitted -> no manifest/trace, no behavior
    change vs a caller that has not adopted `population_triage` teeth.

    `problem_gate`/`extra_recording_fn` (both optional, tool-agnostic): passed straight
    through to `judge_items` -> `judge_item` -- see `judge_item`'s own docstring for the
    downgrade-to-keep / extra-field contract each implements.
    """
    if engine is None and items:
        raise NoEngineError("population-triage recommend: no vision engine is configured")
    if not items:
        return deterministic
    try:
        summary = judge_items(engine, items, schema=schema, card=card, image_fn=image_fn,
                              quant_fn=quant_fn, evidence_fn=evidence_fn, sample_cap=sample_cap,
                              problem_gate=problem_gate,
                              extra_recording_fn=extra_recording_fn)
    except ExecutionError:
        raise
    except Exception as exc:  # an execution error of the call being verified, never the deterministic answer
        raise ExecutionError(f"population-triage recommend failed: {type(exc).__name__}: {exc}") from exc
    if summary.get("engine") and metadata_path and kept_predicate:
        write_kept_manifest(items, metadata_path, kept_predicate)
        write_triage_trace(summary.get("verdicts", []), metadata_path)
    if not summary.get("engine") or not summary.get("rejected"):
        return deterministic          # no engine, nothing judged, or nothing rejected -> unchanged

    problems = sorted({i.get("vision_problem", "") for i in items if i.get("vision_reject")} - {""})
    tightened_rank = min(_TRIAGE_DECISION_RANK[deterministic.decision] + 1, _TRIAGE_DECISION_RANK["FAIL"])
    decision = _TRIAGE_RANK_DECISION[max(_TRIAGE_DECISION_RANK[deterministic.decision], tightened_rank)]
    notes = (f"{deterministic.notes} " if deterministic.notes else "") + (
        f"[population-triage] {summary['rejected']}/{summary['judged']} sampled items judged "
        f"unusable ({', '.join(problems) or 'unspecified'}).")

    return VLRRResult(
        decision, deterministic.scalar_score, {},
        action=deterministic.action,
        param_delta=deterministic.param_delta,
        expected_effect=deterministic.expected_effect,
        # docking-merge fix (045-alignment re-verification): carry the deterministic
        # verdict's fault/upstream_target forward -- a tighten must not silently drop
        # them if `deterministic.action == "conceded"`. Harmless no-op today (the
        # cropper's own deterministic verdict never concedes, so both are always None
        # here regardless), but without this the tighten path would previously have
        # SILENTLY RESET a real fault/upstream_target to None the moment the
        # cropper's "Open cropper gap" follow-on (emit `conceded` +
        # `fault="upstream"` on a bad-mask input) is ever implemented.
        fault=deterministic.fault,
        upstream_target=deterministic.upstream_target,
        judge_problem="; ".join(problems),
        notes=notes.strip(),
    )
