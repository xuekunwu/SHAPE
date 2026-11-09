"""Glue between the Executor, the per-stage adapters and the StepVerifier.

`verify_managed_tool` is what the main solve() loop calls for a managed pipeline
tool: it runs the tool once (deterministic, via Executor.run_managed_tool), then
for each verification stage the tool produces it lets the StepVerifier judge and,
on a local fault, retune+rerun the tool. Returns the per-stage resolutions and
the final (accepted/best) tool result.
"""

from __future__ import annotations

from typing import Any, Optional

from shape.utils.schemas import StageResolution

from .stage_map import stages_for_tool


def _latest_result_field(memory, *keys) -> Optional[Any]:
    """Most recent value of any of `keys` across prior tool results in memory."""
    if memory is None or not hasattr(memory, "get_actions"):
        return None
    for action in reversed(memory.get_actions(llm_safe=False)):
        result = action.get("result", {})
        if isinstance(result, dict):
            # handle per-image results by peeking at the first entry too
            candidates = [result]
            if "per_image" in result and isinstance(result["per_image"], list) and result["per_image"]:
                candidates.append(result["per_image"][0])
            for cand in candidates:
                for k in keys:
                    if isinstance(cand, dict) and cand.get(k):
                        return cand[k]
    return None


def _per_image_field(memory, image_id, *keys) -> Optional[Any]:
    """Most recent value of any of `keys` for a SPECIFIC image_id, scanning prior tool
    results' per_image entries. A multi-image run tags every per-image artifact with its
    id, so a downstream stage chains from the right upstream output (image i's segmenter
    reads image i's processed image, not whichever ran last)."""
    if memory is None or not hasattr(memory, "get_actions"):
        return None
    for action in reversed(memory.get_actions(llm_safe=False)):
        result = action.get("result", {})
        if not isinstance(result, dict):
            continue
        entries = (result["per_image"]
                   if isinstance(result.get("per_image"), list) else [result])
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if image_id is not None and entry.get("image_id") != image_id:
                continue
            for k in keys:
                if entry.get(k):
                    return entry[k]
    return None


def _mask_record(memory, image_id, mask) -> Optional[dict]:
    """The result record that delivered `mask` (a flat result, or the per_image entry for `image_id`): the
    most recent one carrying it."""
    if memory is None or not hasattr(memory, "get_actions") or not mask:
        return None
    for action in reversed(memory.get_actions(llm_safe=False)):
        result = action.get("result", {})
        if not isinstance(result, dict):
            continue
        entries = (result["per_image"]
                   if isinstance(result.get("per_image"), list) else [result])
        for entry in entries:
            if (isinstance(entry, dict) and (image_id is None or entry.get("image_id") == image_id)
                    and mask in (entry.get("mask_path"), entry.get("organoid_mask_path"))):
                return entry
    return None


def _crop_min_area(mask, record) -> Optional[int]:
    """The cropper's min_area for `mask`: what counts as a cell is the delivering segmentation's, its
    `parameters.min_size` as the segmenter reads it (int); none recorded -> None (the tool's default
    applies). An organoid mask keeps 50, as the executor's cropper command (`Executor.generate_tool_command`)."""
    path = str(mask).lower()
    if "organoid_mask" in path and "cell_mask" not in path:
        return 50
    params = record.get("parameters") if isinstance(record, dict) else None
    min_size = params.get("min_size") if isinstance(params, dict) else None
    if isinstance(min_size, (int, float)) and not isinstance(min_size, bool):
        return int(min_size)
    return None


def _latest_analyzer_action(memory) -> Optional[dict]:
    """The newest Cell_State_Analyzer_Tool action in memory whose result (flat, or its per_image[0]) delivered an
    `adata_path` and is not a batch-corrected re-cluster (`batch_corrected: True`), or None."""
    if memory is None or not hasattr(memory, "get_actions"):
        return None
    for action in reversed(memory.get_actions(llm_safe=False)):
        result = action.get("result")
        if action.get("tool_name") != "Cell_State_Analyzer_Tool" or not isinstance(result, dict):
            continue
        entry = result
        if isinstance(result.get("per_image"), list) and result["per_image"]:
            entry = result["per_image"][0]
        if isinstance(entry, dict) and not entry.get("batch_corrected") and entry.get("adata_path"):
            return action
    return None


def _latest_analyzer_h5ad(memory) -> Optional[str]:
    """The `adata_path` of `_latest_analyzer_action`'s result, or None."""
    action = _latest_analyzer_action(memory)
    if action is None:
        return None
    result = action["result"]
    entry = result["per_image"][0] if isinstance(result.get("per_image"), list) and result["per_image"] else result
    return entry["adata_path"]


def build_base_kwargs(tool_name: str, image, memory) -> dict:
    """Resolve the fixed (non-tunable) inputs a managed tool needs, from memory.

    `image` is either a path string (single-image, legacy) or an image-spec dict
    ``{"path", "image_id", "group"}`` (multi-image). Mirrors the executor's tool-chaining
    heuristics: segmenters read the source image directly; the cropper needs the
    matching mask, and its min_area is the delivering segmentation's min_size; analyzers load
    their crops from the cache directory. For a spec dict the upstream input is resolved PER
    image_id and the id/group are forwarded, so N images chain independently through
    segment -> crop.
    """
    stages = stages_for_tool(tool_name)
    stage = stages[0].stage if stages else None
    if isinstance(image, dict):
        path = image.get("path")
        image_id = image.get("image_id")
        group = image.get("group", "default")
        mask = _per_image_field(memory, image_id, "mask_path", "organoid_mask_path")
    else:
        path, image_id, group = image, None, "default"
        mask = _latest_result_field(memory, "mask_path", "organoid_mask_path")
    kwargs: dict = {}
    if stage == "segmentation":
        kwargs["image"] = path
        if image_id:
            kwargs["image_id"] = image_id
    elif stage == "crop":
        if path:
            kwargs["original_image"] = path
        if mask:
            kwargs["nuclei_mask"] = mask
            min_area = _crop_min_area(mask, _mask_record(memory, image_id, mask))
            if min_area is not None:
                kwargs["min_area"] = min_area
        if image_id:
            kwargs["source_image_id"] = image_id
        if isinstance(image, dict):
            kwargs["group"] = group
            technical_metadata = {
                key: image[key]
                for key in (
                    "plate", "plate_id", "well", "well_id", "site", "site_id",
                    "batch", "batch_id", "experiment", "experiment_id",
                    "technical_group",
                )
                if image.get(key) is not None
            }
            if technical_metadata:
                kwargs["technical_metadata"] = technical_metadata
    elif stage == "batch_correction":
        # the batch corrector's input: the newest analyzer h5ad's embedding and technical metadata
        adata_path = _latest_analyzer_h5ad(memory)
        if adata_path:
            from .batch_stage import batch_inputs_from_h5ad
            kwargs.update(batch_inputs_from_h5ad(adata_path))
    # representation / morphological_state (analyzer): inputs loaded from cache dir
    return kwargs


def latest_stage_metrics(memory) -> dict:
    """The QC metrics of the most recently verified managed step (for upstream ctx).

    Task 057 M1/M4: also carries the most recent `image_context_profile`
    (when that action's raw result has one) under the SAME key as an existing sibling
    channel, `metrics["image_context_profile"]` -- riding the channel this function
    already feeds into `adapt_segmentation`'s `context["upstream"]` (via
    `Verifier._adapter_for`/`resolve_managed_stage`'s `upstream_metrics` parameter), so
    `llm_recommend.vision_recommend_segmentation`'s one-way adapter can render it as an
    advisory hint without a new plumbing path. Harmless for every other stage/consumer:
    a stage whose raw result carries no such key leaves `metrics` unchanged (see
    CODING_LOG.md for the consumer audit -- `final_review.py`'s `_KEY_METRICS` filter and
    `_analysis_stage_ran`'s membership check are both key-gated, so a nested dict value
    here is never coerced to float).
    """
    if memory is None or not hasattr(memory, "get_actions"):
        return {}
    for action in reversed(memory.get_actions(llm_safe=False)):
        v = action.get("verification")
        if isinstance(v, dict) and v.get("metrics"):
            metrics = dict(v["metrics"])
            result = action.get("result")
            profile = result.get("image_context_profile") if isinstance(result, dict) else None
            if profile is None and isinstance(result, dict):
                per_image = result.get("per_image")
                if isinstance(per_image, list) and per_image and isinstance(per_image[0], dict):
                    profile = per_image[0].get("image_context_profile")
            if profile is not None:
                metrics["image_context_profile"] = profile
            return metrics
    return {}


_TRIAL_RESULT_KEYS = (
    "cell_count", "attempt", "param_sig", "mask_path", "overlay_path",
    "overlay_archived", "mask_archived",
    "cell_crops_metadata_path", "status", "error",
    "image_context_profile",  # task 057 M1/M4: preflight context, verification-trail persistence
    "prefilter_mask_path",    # a colour-filtered call's masks before the filter (its coverage readouts' masks)
)


def _trial_result_summary(result) -> dict:
    """Compact, JSON-friendly snapshot of one attempt's tool output, so the record
    traces each parameter change's actual segmentation result (cell_count, the mask
    and overlay it produced) -- not only its score and metrics."""
    if not isinstance(result, dict):
        return {}
    return {k: result[k] for k in _TRIAL_RESULT_KEYS
            if k in result and result[k] is not None}


# a wave-mode stage whose own calls never delivered: no output, not a kept input
_NO_CALL_REASONS = ("initial_trial_failed", "no_call_finished")


def _kept_input_reason(spec, tool_name: str, res) -> Optional[str]:
    """The reason a stage deliberately delivered its input unchanged, else None: a verifier whose final choice may
    keep the input (`keep_input`, e.g. the batch corrector) that delivered trial 0 (a SKIP) or no call (an
    ABSTAIN, the agent keeping the input, no admissible call). A stage whose calls failed delivered nothing."""
    from .stage_map import verifier_cls_for

    if not getattr(verifier_cls_for(spec, tool_name), "keep_input", False) or res.reason in _NO_CALL_REASONS:
        return None
    selected = (getattr(res, "details", None) or {}).get("selected")
    if res.result is None or selected == 0:
        return res.reason or "input_kept"
    return None


def make_verification_record(tool_name: str, resolutions, status: str, stages=None) -> dict:
    """Compact, JSON-friendly QC summary to store on the memory action. `stages` (default: every stage the tool
    produces) names the stages `resolutions` resolved, e.g. only morphological_state for a re-cluster."""
    stages = list(stages) if stages is not None else stages_for_tool(tool_name)
    per_stage = []
    for spec, res in zip(stages, resolutions):
        kept_input = _kept_input_reason(spec, tool_name, res)
        trials = []
        parents = getattr(res, "parents", None) or []     # wave mode: the tree of trials
        for index, (params, verdict, result) in enumerate(res.trials):
            trials.append({
                "params": params,
                "decision": verdict.decision,
                "action": verdict.action,
                "score": round(float(verdict.scalar_score), 4),
                "metrics": verdict.metrics,
                "triggered": verdict.triggered,
                "locus": verdict.locus,
                "param_delta": verdict.param_delta,
                "expected_effect": verdict.expected_effect,
                # the visual-reasoning verdict for THIS attempt's overlay lives here
                "notes": verdict.notes,
                # full, untruncated judge rationale for the durable record (task 014);
                # the compact `notes` above is the in-context digest and is unaffected.
                "reasoning_full": getattr(verdict, "reasoning_full", None),
                # the actual segmentation output of THIS attempt, for tracing
                "result": _trial_result_summary(result),
                # the vision judge's image profile and assessment (None when no judge ran)
                "judge_specimen": verdict.judge_specimen,
                "judge_cell_size": verdict.judge_cell_size,
                "judge_cell_density": verdict.judge_cell_density,
                "judge_problem": verdict.judge_problem,
                "judge_severity": verdict.judge_severity,
                "selection_priority": getattr(verdict, "selection_priority", 0),
            })
            if parents:
                trials[-1]["parent"] = parents[index]
        per_stage.append({
            "stage": spec.stage,
            "decision": res.verdict.decision,
            "action": res.verdict.action,
            "status": res.status,
            # set iff status == "conceded": ceiling | upstream (task 032; "human" retired -> "ceiling", task 045)
            "fault": res.fault,
            # the stage delivered no output (e.g. its only call failed: reason initial_trial_failed)
            "no_output": res.result is None and kept_input is None,
            # the stage deliberately kept its input unchanged (e.g. the batch corrector's SKIP / ABSTAIN): why
            "kept_input": kept_input,
            "reason": res.reason,
            "score": round(float(res.verdict.scalar_score), 4),
            "notes": res.verdict.notes,
            "reasoning_full": getattr(res.verdict, "reasoning_full", None),
            "n_trials": len(res.trials),
            "trials": trials,
            "ledger": dict(res.ledger),
            # wave mode: rounds, the verifier's stop, the agent's plans and final choice
            "details": dict(getattr(res, "details", None) or {}),
        })
    last = resolutions[-1] if resolutions else None
    return {
        "tool": tool_name,
        "status": status,
        "stages": per_stage,
        "metrics": (last.verdict.metrics if last else {}),
        "final_params": (last.params if last else {}),
        "upstream_target": (last.upstream_target if last else None),
    }


def _produces_segmentation(tool_name: str) -> bool:
    """True for tools whose managed stages include segmentation -- the only stage where
    the diameter-measured Recommend applies (it reads the segmenter's masks)."""
    return any(s.stage == "segmentation" for s in stages_for_tool(tool_name))


def verify_managed_tool(executor, tool_name: str, base_kwargs: dict,
                        initial_params: Optional[dict] = None,
                        upstream_metrics: Optional[dict] = None,
                        max_iters: int = 6, request: Optional[str] = None, metadata_profile=None,
                        verifier_options: Optional[dict] = None):
    """Run + verify (+ retune/rerun) a managed tool.

    Returns (resolutions, final_result, status) where:
      - resolutions: list[StageResolution], one per stage the tool produces;
      - final_result: the accepted/best tool result dict;
      - status: the status of the last stage processed (satisfied / conceded /
        exclude_samples -- see StageResolution.fault for which concede it was).
    The chain stops early at the first non-satisfied stage so a concede is
    surfaced to the caller (the main loop / planner).

    For a segmenter, the Verifier also MEASURES the true cell size from the first
    segmentation's masks and re-segments at the measured diameter when the default was
    off by >2x -- agentic reasoning over evidence (mask median area), which the
    framework's solve() path now gets too, not only the benchmark.

    `request` (the user's request: the solve() query, else the step's sub_goal) is handed to
    each stage's verifier (`Verifier.resolve_managed_stage`); the agentic segmentation verifier
    states its target from it. `metadata_profile` (the run's `metadata_profile.MetadataProfile`) is handed
    on the same way, and `verifier_options` (`Verifier.resolve_managed_stage`: e.g. the segmentation cohort's
    mode for one image). A verifier's error (e.g. a reasoner failure: `base.AgenticVerifier._ask`) propagates.
    """
    from shape.agent.verifier import Verifier

    from .stage_map import verifier_cls_for

    stages = stages_for_tool(tool_name)
    verifier = Verifier()   # the pillar owns per-stage adapt + verify + rerun

    attempt_counter = {"count": 0}

    def runner(params):
        run_params = dict(params)
        if _produces_segmentation(tool_name):
            attempt_counter["count"] += 1
            run_params.setdefault("_vlrr_attempt_index", attempt_counter["count"])
        return executor.run_managed_tool(tool_name, base_kwargs, run_params)

    # the (vision) LLM tie-breaker, if this run has it enabled
    llm_engine = (getattr(executor, "verifier_llm_engine_mm", None)
                  if getattr(executor, "enable_llm_verifier", False) else None)
    # 044: the segmenter Verify refinement (scale_mismatch + vision-prompt flow_error
    # filter), OFF by default -- an executor that never sets this attribute (every
    # pre-044 caller) is base-identical.
    verify_refine = getattr(executor, "verify_refine", False)

    _NOT_RUN = object()

    def _resolve(params: dict, initial_result=_NOT_RUN, *, upstream_ctx: Optional[dict] = None):
        """Run the tool at `params`, then VLRR-resolve each stage it produces (`upstream_ctx`: the stage
        context of every stage, e.g. a wave-mode verifier's task)."""
        params = dict(params)
        result = runner(params) if initial_result is _NOT_RUN else initial_result
        resolutions: list[StageResolution] = []
        status = "satisfied"
        for spec in stages:
            stage_runner = runner
            if (spec.stage == "morphological_state"
                    and tool_name == "Cell_State_Analyzer_Tool"):
                # A morphology retune changes graph/clustering knobs only. Lock it to
                # the representation accepted by the immediately preceding stage; the
                # analyzer fingerprint remains the fail-closed reuse gate.
                def stage_runner(stage_params):
                    return runner({**stage_params, "reuse_embedding": True})
                # after a wave-mode representation (its calls carry only that loop's kwargs): this stage's own
                # levers the delivered params lack, the planner's, else the stage defaults
                if resolutions and getattr(resolutions[-1], "parents", None):
                    params = {**{k: v for k, v in {**spec.default_params, **initial_params}.items()
                                 if k in spec.param_map}, **params}
            res = verifier.resolve_managed_stage(
                spec, stage_runner, params, result,
                base_kwargs=base_kwargs, upstream_metrics=upstream_metrics,
                llm_engine=llm_engine, max_iters=max_iters, tool_name=tool_name,
                verify_refine=verify_refine, request=request,
                upstream_ctx=upstream_ctx, metadata_profile=metadata_profile,
                verifier_options=verifier_options)
            resolutions.append(res)
            result = res.result
            params = res.params
            status = res.status
            if res.status != "satisfied":
                break
        return resolutions, result, status

    initial_params = dict(initial_params or {})

    # A wave-mode verifier that measures the input itself (trial 0 = its `input_record`, not a tool
    # call) runs its own calls: no first call at the planner's params and no diameter-correction pass
    # (a `base.AgenticVerifier` that defines `input_record`; with `admission_call` its round 1 is the
    # tool at its CARD defaults; the tool's CARD.md "Agentic VRR"). One whose trial 0 is a tool call at
    # its own configuration defines `trial0_call` (the inputs, trial 0's params and the stage context;
    # e.g. the analyzer's frozen backbone): trial 0 runs at those params, not the planner's.
    for spec in stages:
        cls = verifier_cls_for(spec, tool_name)
        if getattr(cls, "uses_waves", False) and hasattr(cls, "input_record"):
            return _resolve({}, cls.input_record(base_kwargs))
        if getattr(cls, "uses_waves", False) and hasattr(cls, "trial0_call"):
            base_kwargs, trial0, stage_ctx = cls.trial0_call(
                base_kwargs, getattr(executor, "query_cache_dir", None), request)
            return _resolve(trial0, upstream_ctx=stage_ctx)

    resolutions, result, status = _resolve(initial_params)

    # Diameter-measured Recommend (segmenters only): measure the cells from the first
    # segmentation's masks and re-segment at that size when the default was off by >2x.
    # One correction at most -- the guard reads the original `used` diameter, so it
    # cannot re-trigger on its own corrected pass.
    if _produces_segmentation(tool_name) and "diameter" in initial_params and resolutions:
        from .initial_params import diameter_correction_needed, diameter_from_mask_area
        used = float(initial_params["diameter"])
        measured = diameter_from_mask_area(
            (resolutions[-1].verdict.metrics or {}).get("mask_area_median"), used)
        if diameter_correction_needed(measured, used):
            resolutions, result, status = _resolve(dict(initial_params, diameter=measured))

    return resolutions, result, status
