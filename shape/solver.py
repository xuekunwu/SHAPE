"""
SHAPE Solver - Main entry point for the framework.

This module provides a high-level interface for using SHAPE programmatically.
"""

import os
import re
from typing import List, Optional
from shape.tools.initializer import Initializer
from shape.agent.planner import Planner
from shape.agent.executor import Executor
from shape.agent.memory import Memory
from shape.engine.factory import resolve_api_key, resolve_llm_model
from shape.utils import logger


def _validate_image_path(image_path: Optional[str]) -> None:
    """Fail early when an input image is missing or is only a Git LFS pointer."""
    if image_path is None:
        return
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    with open(image_path, "rb") as image_file:
        header = image_file.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"Image is a Git LFS pointer, not image data: {image_path}. "
            "Install Git LFS and run `git lfs install && git lfs pull` in the SHAPE repository."
        )


def _image_id_from_path(path: Optional[str]) -> Optional[str]:
    return os.path.splitext(os.path.basename(path))[0] if path else None


def _normalize_image_specs(images) -> List[dict]:
    """Normalize multi-image input while preserving allowlisted technical identities.

    Each item is a path string or a dict with ``path``/``image_path`` and optional
    ``group``/``image_id``, plate/well/site/batch identifiers and a ``control`` flag (the
    metadata profile's controls). This is what lets the agent take 1..N images (optionally grouped) through ONE loop -- per-image tools run per
    spec, the analyzer once over all crops. Unknown keys are still discarded.
    """
    specs: List[dict] = []
    for item in images:
        if isinstance(item, dict):
            path = item.get("path") or item.get("image_path")
            spec = {"path": path,
                    "image_id": item.get("image_id") or _image_id_from_path(path),
                    "group": item.get("group", "default")}
            for key in (
                "plate", "plate_id", "well", "well_id", "site", "site_id",
                "batch", "batch_id", "experiment", "experiment_id",
                "technical_group", "control",
            ):
                if item.get(key) is not None:
                    spec[key] = item[key]
            specs.append(spec)
        else:
            specs.append({"path": item, "image_id": _image_id_from_path(item),
                          "group": "default"})
    return specs


def get_available_tools(enabled_tools: Optional[List[str]] = None) -> List[str]:
    """
    Get list of available tools.
    
    Args:
        enabled_tools: Optional list of tool names to enable. If None, all tools are discovered.
    
    Returns:
        List of available tool names
    """
    initializer = Initializer(enabled_tools=enabled_tools or [])
    return initializer.available_tools


def construct_solver(
    llm_engine_name: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    query_cache_dir: str = "solver_cache",
):
    """
    Construct a SHAPE solver with all necessary components.
    
    Args:
        llm_engine_name: Optional model name; defaults to the LLM_MODEL environment variable.
        enabled_tools: Optional list of tool names to enable. If None, all tools are enabled.
        api_key: LLM gateway API key. If None, will try to get from
            LLM_API_KEY for OpenAI-compatible routes.
        query_cache_dir: Directory for caching query results
    
    Returns:
        Dictionary containing solver components:
        - planner: Planner instance
        - executor: Executor instance
        - memory: Memory instance
        - initializer: Initializer instance
        - available_tools: List of available tool names
    """
    # Resolve the API key for the model's provider (OpenAI / Anthropic / ...).
    llm_engine_name = resolve_llm_model(llm_engine_name)
    api_key = resolve_api_key(llm_engine_name, api_key)
    if api_key is None:
        raise ValueError(
            f"API key not provided for model '{llm_engine_name}'. Set the "
            "provider's API key env var (e.g. LLM_API_KEY / ANTHROPIC_API_KEY) "
            "or pass the api_key parameter."
        )
    
    # Initialize tools
    initializer = Initializer(
        enabled_tools=enabled_tools or [],
        model_string=llm_engine_name,
        api_key=api_key
    )
    
    # Initialize planner. Tool metadata is loaded lazily by the planner for only
    # the tools it selects (via the initializer), instead of reading every tool's
    # metadata up front. The system-routed tools (batch correction) are not in the
    # planner's toolbox; `available_tools` below keeps them.
    from shape.agent.verification import stage_map
    planner = Planner(
        llm_engine_name=llm_engine_name,
        toolbox_metadata={},
        available_tools=[t for t in initializer.available_tools
                         if t not in stage_map.SYSTEM_ROUTED_TOOLS],
        api_key=api_key,
        initializer=initializer
    )
    
    # Initialize executor
    executor = Executor(
        llm_engine_name=llm_engine_name,
        query_cache_dir=query_cache_dir,
        api_key=api_key,
        initializer=initializer
    )
    
    # Initialize memory
    memory = Memory()

    return {
        "planner": planner,
        "executor": executor,
        "memory": memory,
        "initializer": initializer,
        "available_tools": initializer.available_tools,
    }


def _should_stop(verification) -> bool:
    """Whether the memory verification says to terminate the loop.

    `MemoryVerification.stop_signal` is a bool; we also accept a legacy string
    form ("STOP"/"true"/...) defensively so a parser change can't silently revert
    termination to never-stopping.
    """
    stop = getattr(verification, "stop_signal", False)
    if isinstance(stop, str):
        return stop.strip().upper() in ("STOP", "TRUE", "YES", "1")
    return bool(stop)


def _vlrr_loop_decision(vstatus, fault, result, prior_escalations: int,
                        max_escalation_retries: int = 2):
    """Decide what the agent loop does after a managed-tool VLRR run.

    Returns (decision, new_escalation_count) where decision is "stop" or "continue".
    Only a hard tool error stops the agent loop immediately at this per-stage layer.
    A ``conceded`` with ``fault="upstream"`` does NOT stop -- the diagnosis is in
    memory (surfaced via the QC), so the planner re-plans (typically to re-run the
    named upstream tool); it is bounded so a stuck upstream fault can't burn the
    budget. A ``conceded`` with ``fault="ceiling"`` also continues, carrying the best
    result, instead of dropping the work -- this is the D1 invariant: a tuning
    ceiling is NOT a pipeline stop. Task 045 refinement: the old third fault value
    "human" is retired -- a verifier's own "this tool/data can't do better" finding
    (formerly "human") now also carries fault="ceiling" (was
    hypothesis_invalidated / judge_no_improvement / oscillation_stop /
    max_iters_reached / a verifier's own tool_mismatch / data_intrinsic locus), so it
    is folded into the SAME continue-and-carry-forward path as any other ceiling.
    A wave-mode stage whose calls all failed (e.g. a managed call that raised:
    `initial_trial_failed`) is such a ceiling concede with NO output (result None):
    it continues too, and the planner's QC digest says "NO OUTPUT produced".
    The task-completion planner prompt (``verificate_memory``), which sees the full
    QC digest across all steps, is the one place that decides whether the OVERALL
    task should stop here.
    """
    hard_error = isinstance(result, dict) and (
        result.get("error") or result.get("status") == "failed")
    if hard_error:
        return "stop", prior_escalations
    if vstatus == "conceded" and fault == "upstream":
        n = prior_escalations + 1
        return ("stop" if n > max_escalation_retries else "continue"), n
    return "continue", prior_escalations


def _cross_stage_recovery_loop(upstream_target, attempt, max_reruns: int = 2):
    """Bounded upstream nudge+rerun for a managed tool that conceded upstream.

    `attempt(n)` performs ONE recovery pass -- nudge the upstream stage's params,
    re-run the upstream tool, re-verify the downstream tool -- and returns its
    ``(resolutions, result, status)``, or None if the upstream rerun itself failed.
    The loop stops on convergence, on an upstream failure, or when the rerun budget is
    spent. Returns the final ``(resolutions, result, status)``, or None when the
    upstream stage has no defined remedy (caller then falls back to the LLM re-plan).

    Pure control logic (the I/O lives in ``attempt``) so it is unit-testable in
    isolation, exactly like ``_vlrr_loop_decision``.
    """
    from shape.agent.verification.feedback import can_feedback
    if not can_feedback(upstream_target):
        return None
    last = None
    for n in range(max_reruns):
        out = attempt(n)
        if out is None:               # the upstream rerun itself failed -> give up
            break
        last = out
        if out[2] == "satisfied":     # downstream re-verified clean -> done
            break
    return last


def _try_cross_stage_recovery(executor, tool_name, tool_params, image,
                              memory, step, resolutions, request=None, metadata_profile=None):
    """Deterministically CLOSE the cross-stage VLRR loop in the framework.

    When a managed tool localizes its fault to a prior stage (``conceded`` with
    ``fault="upstream"``) and that upstream stage has a defined remedy, nudge +
    re-run the upstream tool and re-verify this tool -- instead of only recording the
    diagnosis and hoping the LLM planner re-plans. Reuses the same `feedback.py`
    controller the benchmark uses, so the agent's own entry point exercises it.
    Bounded + anti-oscillation via `nudge_params`/`apply_bounded`. `image` is the
    image spec dict (multi-image) or the legacy path string; the upstream rerun is
    resolved + recorded PER image so the right image is re-processed. Returns the
    recovered ``(resolutions, result, status)`` or None to fall back to the LLM
    re-plan path. The re-verification carries the user's `request` and the run's
    `metadata_profile`, as the first one did.
    """
    from shape.agent.verification.feedback import can_feedback, nudge_params
    from shape.agent.verification.stage_map import (
        default_params_for_tool, stage_by_name)
    from shape.agent.verification.integration import (
        build_base_kwargs, latest_stage_metrics, verify_managed_tool)

    upstream_stage = resolutions[-1].upstream_target if resolutions else None
    if not can_feedback(upstream_stage):
        return None
    up_spec = stage_by_name(upstream_stage)
    if not (up_spec and up_spec.tool_names):
        return None
    up_tool = up_spec.tool_names[0]
    up_params = default_params_for_tool(up_tool)
    image_id = image.get("image_id") if isinstance(image, dict) else None

    def attempt(n):
        nonlocal up_params
        up_params = nudge_params(up_params, upstream_stage)
        up_result = executor.run_managed_tool(
            up_tool, build_base_kwargs(up_tool, image, memory), up_params)
        # tag the rerun by image_id so a multi-image re-verify chains from THIS image's
        # refreshed output (single-image keeps the flat record unchanged).
        record = ({"per_image": [dict(up_result, image_id=image_id)]}
                  if isinstance(up_result, dict) and image_id else up_result)
        memory.add_action(
            step_count=step, tool_name=up_tool,
            sub_goal=f"cross-stage feedback: re-run {upstream_stage}",
            command=f"[cross-stage] {up_tool}.execute(**{up_params})",
            result=record, params=dict(up_params),
            planner_reasoning="cross-stage feedback (no planner step)")
        if isinstance(up_result, dict) and (
                up_result.get("error") or up_result.get("status") == "failed"):
            return None
        # the upstream rerun is now in memory, so build_base_kwargs picks up its
        # refreshed output (e.g. the new processed image) for the re-verification.
        return verify_managed_tool(
            executor, tool_name, build_base_kwargs(tool_name, image, memory),
            tool_params, latest_stage_metrics(memory), request=request,
            metadata_profile=metadata_profile)

    return _cross_stage_recovery_loop(upstream_stage, attempt)


def _result_stop_reason(decision, result):
    """Map a VLRR loop decision into a Verdict ``stop_reason`` (None = keep going).

    Lets the result verdict carry WHY the loop stops, so the single decide_stop
    authority terminates on it instead of a separate stop/continue switch. Task 045:
    the "human_fault" reason is retired along with the fault value -- `_vlrr_loop_decision`
    now only stops this per-stage layer on a hard tool error or an exhausted upstream
    escalation budget, so this function no longer needs the `fault` to decide.
    """
    if decision != "stop":
        return None
    if isinstance(result, dict) and (result.get("error") or result.get("status") == "failed"):
        return "hard_error"
    return "escalation_budget_exhausted"


_ANALYSIS_TOOLS = (
    "Cell_State_Analyzer_Tool",
    "Analysis_Visualizer_Tool",
)


def _analysis_stage_ran(memory, metrics: dict) -> bool:
    """[045 scope B, Cond. 2] Structural predicate for `_finalize_with_vision_review`:
    did an analysis stage actually run -- vs a segmentation-only session, where
    `final_review`'s schema is `not_applicable` anyway (no loss in skipping it)?
    True iff an analyzer/visualizer tool appears in memory, or the latest verified
    stage's metrics carry a `_KEY_METRICS` key. NOT a figure-filename heuristic (a
    UMAP and a segmentation overlay are both PNGs)."""
    from shape.agent.verification.final_review import _KEY_METRICS
    actions = memory.get_actions(llm_safe=False) if hasattr(memory, "get_actions") else []
    if any(a.get("tool_name") in _ANALYSIS_TOOLS for a in actions):
        return True
    return any(k in (metrics or {}) for k in _KEY_METRICS)


def _finalize_with_vision_review(executor, memory, conclusion):
    """End-to-end QC the visualizer bypasses: judge the OUTPUT figures vs the
    written conclusion. Returns (possibly-caveated conclusion, review_record|None).

    When the LLM verifier is off or no figure exists, the conclusion is returned untouched (nothing to review).
    A fatal LLM error (``shape.engine.errors``: no vision engine, an invalid key, the model not found, the
    endpoint unreachable) raises. When the review call fails otherwise, the review is recorded as failed --
    status EXECUTION_ERROR with the cause, logged -- and the conclusion carries a note that its figures were not
    checked; never returned as if reviewed. The judge can only append a caveat / lower
    confidence, never bless the result.

    Gated on `_analysis_stage_ran` (task 045 vision-authority consolidation): fires
    only if an analysis stage actually ran -- a segmentation-only run never reaches
    the judge.
    """
    if not getattr(executor, "enable_llm_verifier", False):
        return conclusion, None
    try:
        from shape.agent.planner import collect_output_figures
        from shape.agent.verification.final_review import (
            caveat_from_verdict, full_reasoning, vision_review_final)
        from shape.agent.verification.integration import latest_stage_metrics

        figures = collect_output_figures(memory)
        if not figures:
            return conclusion, None
        metrics = latest_stage_metrics(memory)
        if not _analysis_stage_ran(memory, metrics):
            return conclusion, None
        engine = getattr(executor, "verifier_llm_engine_mm", None)
        verdict = vision_review_final(engine, figures, conclusion, metrics)
        if verdict is None:
            return conclusion, None
        record = {
            "tool": "final_review",
            "status": "reviewed",
            "stages": [{
                "stage": "end_to_end",
                "decision": "PASS" if verdict.action == "accept" else "WARNING",
                "action": verdict.action,
                "status": "reviewed",
                "notes": (f"[final-vision] separation={verdict.cluster_separation} "
                          f"group={verdict.group_structure} "
                          f"consistency={verdict.figure_text_consistency}: "
                          + (verdict.reasoning or "")[:200]),
                # full, untruncated judge rationale for the durable record (task 014);
                # `notes` above (the in-context/compact form) is unchanged.
                "reasoning_full": full_reasoning(verdict),
            }],
        }
        caveat = caveat_from_verdict(verdict)
        if caveat:
            record["caveat"] = caveat
            conclusion = f"{conclusion}\n\n{caveat}"
        return conclusion, record
    except Exception as e:
        from shape.engine.errors import is_fatal_llm_error
        from shape.utils.schemas import EXECUTION_ERROR
        if is_fatal_llm_error(e):            # no engine / key / permission / model / endpoint: a system error
            raise
        error = f"{type(e).__name__}: {e}"
        logger.error(f"final vision review failed ({error})")
        record = {"tool": "final_review", "status": EXECUTION_ERROR, "error": error,
                  "stages": [{"stage": "end_to_end", "decision": "FAIL", "status": EXECUTION_ERROR,
                              "notes": f"[final-vision] the review failed: {error}"[:300]}]}
        note = (f"⚠ The final vision review failed ({error[:200]}); the output figures were not checked against "
                "this conclusion.")
        return f"{conclusion}\n\n{note}", record


def decide_stop(verdict) -> bool:
    """The single stop authority (Phase 1): a Verdict carrying a stop_reason ends the loop.

    Both the task-completion check and the result verification produce a Verdict; this
    is the ONE place that reads them, so the loop has a single owner of "should we
    stop" instead of three independent switches (verificate_memory / the VLRR loop
    decision / escalation hints).
    """
    return verdict is not None and verdict.stop_reason is not None


def _verify_task_complete(planner, question, image_path, query_analysis, memory,
                          group_images=None):
    """VERIFIER (task gate): has accumulated memory already answered the query?

    Returns a plan-scope Verdict; ``stop_reason="task_complete"`` tells decide_stop to
    terminate. The first iteration always returns no-stop -- empty memory cannot answer
    the query, so a tool must run before any termination check.
    """
    from shape.agent.verifier import Verdict
    if not memory.get_actions(llm_safe=True):
        return Verdict(scope="plan", status="pass")
    verification = planner.verificate_memory(
        question=question,
        image=image_path,
        query_analysis=query_analysis,
        memory=memory,
        group_images=group_images,
    )
    stopped = _should_stop(verification)
    return Verdict(
        scope="plan",
        status="pass",
        stop_reason="task_complete" if stopped else None,
        recommendation=getattr(verification, "analysis", None),
    )


def _planner_next_step(planner, question, image_path, query_analysis, memory,
                       step, max_steps, group_images=None):
    """PLANNER: choose the next tool/action toward answering the query."""
    return planner.generate_next_step(
        question=question,
        image=image_path,
        query_analysis=query_analysis,
        memory=memory,
        step_count=step,
        max_step_count=max_steps,
        group_images=group_images,
    )


def _verify_plan(next_step, memory, task_spec=None):
    """VERIFIER (plan gate, pre-execution): is the planned tool (a) IN SCOPE for the
    query and (b) structurally sound given what has run so far? Catches an out-of-scope
    tool (045 P3 Fix 0: e.g. the cropper on a segment/count query -- the vicious-cycle
    guard) or a top-level ordering error (e.g. the analyzer before the cropper), and
    surfaces it through the single Verdict BEFORE any compute is spent. The gate hands
    an unsound plan back to the planner to re-plan (bounded) -- it does not override the
    planner's choice. Returns (verifier, plan_verdict); the verifier is reused for
    result verification on this step.
    """
    from shape.agent.verifier import Verifier
    verifier = Verifier()
    used_tools = [a.get("tool_name") for a in memory.get_actions(llm_safe=True)
                  if a.get("tool_name")]
    plan_verdict = verifier.verify_plan(next_step.tool_name, used_tools, task_spec)
    if plan_verdict.status == "fail":
        logger.warning("plan gate: %s; %s",
                       plan_verdict.fault, plan_verdict.recommendation)
    return verifier, plan_verdict


def _is_merge_all_tool(tool_name) -> bool:
    """Analyzer stages (representation/morphological_state) consume ALL crops at once, so
    the tool runs ONCE over the merged set rather than per image."""
    from shape.agent.verification.stage_map import stages_for_tool
    return any(s.stage in ("representation", "morphological_state")
               for s in stages_for_tool(tool_name))


# Keyed by (status, fault) for "conceded" (the fault tag is what carries the severity
# now, not the old finer-grained status string); plain status string for everything else.
# Task 045: the old ("conceded", "human") tier (3, the most severe) is retired along
# with the "human" fault value -- any conceded fault now shares the same severity (1).
_STATUS_SEVERITY = {("conceded", "upstream"): 1, ("conceded", "ceiling"): 1,
                    "exclude_samples": 0, "satisfied": 0}


def _severity_of(status, fault):
    if status == "conceded":
        return _STATUS_SEVERITY.get((status, fault), 1)
    return _STATUS_SEVERITY.get(status, 0)


def _worst_outcome(outcomes):
    """The most severe (status, fault, result) across per-image runs -- a hard error
    on ANY image dominates; among conceded faults, severity is uniform (task 045:
    "human" no longer outranks "ceiling"/"upstream"), so any conceded outcome simply
    outranks satisfied/exclude_samples."""
    def severity(outcome):
        status, fault, result = outcome
        hard = isinstance(result, dict) and (
            result.get("error") or result.get("status") == "failed")
        return (10 if hard else 0) + _severity_of(status, fault)
    return max(outcomes, key=severity) if outcomes else ("satisfied", None, {})


def _run_managed_image(executor, tool_name, tool_params, image, memory, step, request=None,
                       metadata_profile=None, verifier_options=None):
    """Run + VLRR-verify a managed tool for ONE image (spec dict or legacy path string),
    applying bounded cross-stage recovery. Returns (resolutions, result, status, fault);
    does not record memory (the caller records, per-image or flat). `request` (the user's
    request) and `metadata_profile` (the run's) reach the stage verifiers
    (`integration.verify_managed_tool`), with `verifier_options` (the segmentation cohort's mode for this image)."""
    from shape.agent.verification.integration import (
        build_base_kwargs, latest_stage_metrics, verify_managed_tool)
    resolutions, result, vstatus = verify_managed_tool(
        executor, tool_name, build_base_kwargs(tool_name, image, memory),
        tool_params, latest_stage_metrics(memory), request=request,
        metadata_profile=metadata_profile, verifier_options=verifier_options)
    fault = resolutions[-1].fault if resolutions else None
    # Cross-stage VLRR: a fault localized upstream gets a deterministic, bounded upstream
    # nudge+rerun + re-verify (closing the loop in the framework, not only the benchmark).
    if vstatus == "conceded" and fault == "upstream":
        recovered = _try_cross_stage_recovery(
            executor, tool_name, tool_params, image, memory, step, resolutions, request,
            metadata_profile)
        if recovered is not None:
            resolutions, result, vstatus = recovered
            fault = resolutions[-1].fault if resolutions else None
    return resolutions, result, vstatus, fault


def _record_step_error(memory, step, tool_name, sub_goal, error, planner_reasoning=""):
    """A managed step that raised: recorded as a failed action carrying the cause, and a result Verdict with
    stop_reason "hard_error" (there is no fallback path)."""
    from shape.agent.verifier import Verdict
    result = {"error": f"{tool_name} failed: {type(error).__name__}: {error}", "status": "failed"}
    memory.add_action(step_count=step, tool_name=tool_name, sub_goal=sub_goal,
                      command=f"[verified, failed] {tool_name}", result=result,
                      planner_reasoning=planner_reasoning)
    logger.error(result["error"])
    return Verdict(scope="result", status="fail", stop_reason="hard_error")


def _gpu_slots() -> dict:
    """The device slots of the per-image GPU tools (the segmentation stage's) while several images run at once:
    Cell_Segmenter_Tool runs as many calls at a time as its agentic verifier allows (`sam_concurrency`: the SAM
    server's replicas, 1 for the in-process model), every other one call at a time (one model on one device)."""
    import threading
    from shape.agent.verification.stage_map import stage_by_name
    from shape.agent.verification.verifiers.segmentation import sam_concurrency
    return {tool: threading.BoundedSemaphore(sam_concurrency() if tool == "Cell_Segmenter_Tool" else 1)
            for tool in stage_by_name("segmentation").tool_names}


def _run_images(solver, tool_name, specs, run_one) -> list:
    """`run_one(spec)` for every image spec, in order: `solver["image_workers"]` images at a time (default 1, one
    after the other). Several at a time only for a per-image tool; the executor's GPU tool calls then take turns
    on the device (`_gpu_slots`, `Executor.run_managed_tool`)."""
    workers = min(len(specs), max(1, int(solver.get("image_workers") or 1)))
    if workers < 2 or _is_merge_all_tool(tool_name):
        return [run_one(spec) for spec in specs]
    from concurrent.futures import ThreadPoolExecutor
    executor = solver["executor"]
    if getattr(executor, "gpu_slots", None) is None:
        executor.gpu_slots = _gpu_slots()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, specs))


def _uses_segmentation_cohort(solver, tool_name, specs) -> bool:
    """Whether a multi-image step runs as the segmentation cohort: the segmenter, over more than one image spec
    (dicts with an image_id), unless the run turned the cohort off (`solve(segmentation_cohort=False)`)."""
    from shape.agent.verification.segmentation_cohort import TOOLS
    return (tool_name in TOOLS and solver.get("segmentation_cohort", True) and len(specs) > 1
            and all(isinstance(s, dict) and s.get("image_id") for s in specs))


def _segmentation_cohort(solver, specs, run_one, tool_name="Cell_Segmenter_Tool"):
    """The segmenter over `specs` as a cohort (`segmentation_cohort.run`): the images' preflight reads (the
    process cache makes each image's VRR reuse its read), their groups (run at the same time), calibration,
    consensus and routing, `image_workers` images at a time. Returns (the runs in the order of `specs`, the cohort
    record)."""
    from shape.agent.verification import segmentation_cohort
    from shape.agent.verification.stage_map import stage_by_name, verifier_cls_for
    executor = solver["executor"]
    engine = (getattr(executor, "verifier_llm_engine_mm", None)
              if getattr(executor, "enable_llm_verifier", False) else None)

    def read_profile(spec):
        return segmentation_cohort.preflight_read(spec["path"], engine, spec.get("image_id"))

    card = verifier_cls_for(stage_by_name("segmentation"), tool_name)()
    cache_options = getattr(executor, "organoid_trial_cache", None)
    seed = solver.get("segmentation_cache_seed", 0)
    if tool_name == "Organoid_Segmenter_Tool":
        executor.organoid_trial_cache = None
    if tool_name == "Organoid_Segmenter_Tool" and seed is not None:
        from pathlib import Path
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("segmentation_cache_seed must be a uint32 integer or None")
        root = getattr(executor, "query_cache_dir", None)
        if root is not None:
            executor.organoid_trial_cache = {"root": str(Path(root) / "organoid_native_trials"), "seed": seed}
    try:
        return segmentation_cohort.run(
            list(specs), run_one, lambda items, fn: _run_images(solver, tool_name, items, fn),
            read_profile, card.defaults(), card.levers, workers=max(1, int(solver.get("image_workers") or 1)),
            tool_name=tool_name)
    finally:
        executor.organoid_trial_cache = cache_options
        if tool_name == "Organoid_Segmenter_Tool" and cache_options is None:
            tool = (getattr(executor, "_managed_tools", None) or {}).get(tool_name)
            if tool is not None and hasattr(tool, "disable_trial_cache"):
                tool.disable_trial_cache()



CROPPER_TOOL = "Single_Cell_Cropper_Tool"


def _uses_cropping_cohort(solver, tool_name, specs) -> bool:
    """Whether a multi-image step runs as the cropping cohort: the cropper, over more than one image spec (dicts
    with an image_id), unless the run turned the cohort off (`solve(cropping_cohort=False)`)."""
    return (tool_name == CROPPER_TOOL and solver.get("cropping_cohort", True) and len(specs) > 1
            and all(isinstance(s, dict) and s.get("image_id") for s in specs))


def _cropping_cohort(solver, specs, run_one):
    """The cropper over `specs` as a cohort, the crop analogue of `_segmentation_cohort` over the same shared
    `segmentation_cohort.run`: images group by their preflight read, a calibration sample per group runs the full
    per-image crop VRR, the consensus of the agent-tuned crop levers applies to the rest at the verifier's no-search
    cohort mode (`CropAgenticVerifier.cohort_levers`), and any image the admission flags is routed back to its own
    VRR from that consensus (`start_params`). Returns (the runs in the order of `specs`, the cohort record)."""
    from shape.agent.verification import segmentation_cohort
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier
    executor = solver["executor"]
    engine = (getattr(executor, "verifier_llm_engine_mm", None)
              if getattr(executor, "enable_llm_verifier", False) else None)

    def read_profile(spec):
        return segmentation_cohort.preflight_read(spec["path"], engine, spec.get("image_id"))

    card = CropAgenticVerifier()
    return segmentation_cohort.run(
        list(specs), run_one, lambda items, fn: _run_images(solver, CROPPER_TOOL, items, fn),
        read_profile, card.defaults(), card.levers, workers=max(1, int(solver.get("image_workers") or 1)))


def _managed_verification_record(verifier, tool_name, resolutions, vstatus, plan_verdict):
    """Per-stage QC record + single-Verdict view persisted for one managed run (carries
    observed_effect, which the within-stage loop computes but never stored, plus the
    step's plan verdict)."""
    from dataclasses import asdict
    from shape.agent.verification.integration import make_verification_record
    vrecord = make_verification_record(tool_name, resolutions, vstatus)
    vrecord["plan_verdict"] = asdict(plan_verdict)
    vrecord["verdicts"] = [
        asdict(verifier.verify_result(
            # `res.reason` (not `res.status`) so a stopped stage's stop_reason keeps
            # the ORIGINAL fine-grained reason (e.g. "hypothesis_invalidated"), not
            # just the collapsed "conceded" -- unchanged observability granularity.
            res, stop_reason=(res.reason if res.status != "satisfied" else None)))
        for res in resolutions]
    return vrecord


def _is_hard_error(result) -> bool:
    return result is None or (isinstance(result, dict) and bool(
        result.get("error") or result.get("status") == "failed"))


def _batch_followup(solver, step, memory, request=None, picked=None):
    """The system's batch step (`batch_stage.run_followup`) after a successful
    Cell_State_Analyzer_Tool step, or on a planner pick of the batch tool (`picked`: that
    step; `_system_routed_step`). Routed only when the run's metadata profile has batch
    labels; otherwise nothing is recorded (tier none). When routed but not run, it records
    why (`batch_stage.record_not_routed`): the batch tool is not among the run's enabled
    tools; the analyzer conceded upstream for another reason than batch dominance (the
    planner re-runs upstream, so this embedding is about to be replaced;
    `batch_stage.analyzer_wait_reason`); this analyzer step was already handled. An error
    propagates (the caller reports it as a hard error); otherwise the step's Verdict is not
    changed. Returns the recorded result, or None.
    Before the batch loop first runs, the batch-metrics extra is checked
    (`metadata_profile.resolve_batch_metrics`; an unusable extra downgrades the full tier to
    partial) and the run's profile is updated."""
    from shape.agent.verification import batch_stage
    from shape.agent.verification.metadata_profile import batch_routed, resolve_batch_metrics
    profile = solver.get("metadata_profile")
    if not batch_routed(profile):
        return None
    note = ({} if picked is None else
            {"sub_goal": picked.sub_goal,
             "planner_reasoning": (getattr(picked, "context", "") + " "
                                   + getattr(picked, "justification", "")).strip()})
    reason = (f"{batch_stage.TOOL} is not among this run's enabled tools"
              if batch_stage.TOOL not in (solver.get("available_tools") or ())
              else batch_stage.analyzer_wait_reason(memory))
    if reason is None:
        profile = solver["metadata_profile"] = resolve_batch_metrics(profile)
        record = batch_stage.run_followup(
            solver["executor"], memory, profile, step, request,
            cache=solver.setdefault("_batch_followups", {}))
        if record is not None:
            return record
        reason = "this analyzer step was already handled (its batch step is recorded above)"
    logger.info(f"batch step not run: {reason}")
    return batch_stage.record_not_routed(memory, step, reason, **note)


def _batch_step(solver, step, memory, request=None, picked=None):
    """`_batch_followup`; None when it ran (or was not routed), else the hard-error Verdict of its error: the
    failure is recorded once as a failed batch action (`batch_stage.run_followup` records its own failures)."""
    from shape.agent.verifier import Verdict
    from shape.agent.verification import batch_stage
    try:
        _batch_followup(solver, step, memory, request, picked=picked)
        return None
    except Exception as e:
        actions = memory.get_actions(llm_safe=False)
        last = actions[-1] if actions else {}
        recorded = (last.get("tool_name") == batch_stage.TOOL and isinstance(last.get("result"), dict)
                    and last["result"].get("status") == "failed")
        if not recorded:
            return _record_step_error(memory, step, batch_stage.TOOL, batch_stage.BATCH_SUB_GOAL, e)
        logger.error(f"{batch_stage.TOOL} failed: {type(e).__name__}: {e}")
        return Verdict(scope="result", status="fail", stop_reason="hard_error")


def _system_routed_step(solver, next_step, step, memory, request=None):
    """A planner pick of a system-routed tool (the planner's toolbox omits it): the same
    follow-up, under the same rules, when the metadata routes it and an analyzer embedding
    exists, else a 'not_routed' record. Never a tool call on the LLM path; the step passes, unless
    the follow-up raised (a hard error, `_batch_step`)."""
    from shape.agent.verifier import Verdict
    from shape.agent.verification import batch_stage
    from shape.agent.verification.integration import _latest_analyzer_h5ad
    from shape.agent.verification.metadata_profile import batch_routed
    routed = batch_routed(solver.get("metadata_profile"))
    if routed and _latest_analyzer_h5ad(memory):
        failed = _batch_step(solver, step, memory, request, picked=next_step)
        if failed is not None:
            return failed
    else:
        batch_stage.record_not_routed(
            memory, step,
            "the run's metadata profile has no batch labels" if not routed
            else "no Cell_State_Analyzer_Tool embedding yet",
            sub_goal=next_step.sub_goal,
            planner_reasoning=(getattr(next_step, "context", "") + " "
                               + getattr(next_step, "justification", "")).strip())
    return Verdict(scope="result", status="pass")


def _system_routed_name(tool_name) -> Optional[str]:
    """The system-routed tool a planner pick names, or None. The planner normalises its
    pick against its own toolbox, which omits these tools, so a pick of one comes back as
    "No matched tool given: <name>". Names compare without spaces, underscores and hyphens,
    case-folded ("Batch Corrector Tool" is Batch_Corrector_Tool)."""
    import re
    from shape.agent.verification import stage_map
    clean = str(tool_name or "")
    while "No matched tool given: " in clean:
        clean = clean.split("No matched tool given: ")[-1].strip()

    def key(name):
        return re.sub(r"[\s_-]+", "", name).casefold()

    return next((t for t in stage_map.SYSTEM_ROUTED_TOOLS if key(t) == key(clean)), None)


_NO_TOOL = re.compile(
    r"^(none|null|nil|n/?a|no( further| more)?( tools?)?( calls?| needed| required)?"
    r"|stop|done|finish(ed)?|task complete(d)?)\b", re.IGNORECASE)


def _planner_chose_no_tool(tool_name) -> bool:
    """True when the planner's pick names no tool at all (it comes back unmatched as
    "No matched tool given: <text>"): "None", "NONE - task complete", "No further tool call",
    "None (task complete)" -- the planner declaring the task done, not a missing tool."""
    clean = str(tool_name or "")
    while "No matched tool given: " in clean:
        clean = clean.split("No matched tool given: ")[-1]
    clean = clean.strip().strip("'\"`.[]()")
    return not clean or bool(_NO_TOOL.match(clean))


def _execute_and_verify(solver, next_step, step, question, image_path, memory,
                        enable_verification, escalation_counts, verifier, plan_verdict,
                        images=None) -> str:
    """EXECUTOR (+ result VERIFIER): run the chosen tool and judge its result.

    For a managed pipeline tool this is the deterministic VLRR path (run -> verify ->
    retune/rerun). With multiple images, per-image tools (preprocess/segment/crop) resolve
    once PER image (each its own VLRR resolution, recorded under per_image) while merge-all
    tools (the analyzer) run once over all crops. Single-image behaviour is unchanged. An
    error in the managed path (e.g. a reasoner failure) is a hard error with its cause
    (`_record_step_error`); there is no fallback to the LLM command path. Returns a result Verdict. After a
    successful analyzer step the system's batch follow-up runs (`_batch_followup`); a pick
    of a system-routed tool goes to `_system_routed_step`.
    """
    planner = solver["planner"]
    executor = solver["executor"]
    from shape.agent.verification import stage_map as _stage_map
    if next_step.tool_name in _stage_map.SYSTEM_ROUTED_TOOLS:
        return _system_routed_step(solver, next_step, step, memory,
                                   question or getattr(next_step, "sub_goal", None))
    metadata_profile = solver.get("metadata_profile")
    is_analyzer = next_step.tool_name == "Cell_State_Analyzer_Tool"

    # The planner stashes recommended initial params on itself (not on NextStep, to
    # keep the LLM response schema free of free-form dicts).
    tool_params = getattr(planner, "recommended_params", None)
    if enable_verification and tool_params:
        try:
            from shape.agent.verification.stage_map import is_managed_tool
            if is_managed_tool(next_step.tool_name):
                specs = images if images else [image_path]
                run_specs = ([specs[0] if specs else image_path]
                             if _is_merge_all_tool(next_step.tool_name) else specs)
                # the user's request for the stage verifiers: the query, else this step's sub_goal
                request = question or getattr(next_step, "sub_goal", None)

                # Single-image (legacy path string): one flat record, behaviour unchanged.
                if len(run_specs) == 1 and not isinstance(run_specs[0], dict):
                    resolutions, result, vstatus, fault = _run_managed_image(
                        executor, next_step.tool_name, tool_params, run_specs[0], memory, step, request,
                        metadata_profile)
                    vrecord = _managed_verification_record(
                        verifier, next_step.tool_name, resolutions, vstatus, plan_verdict)
                    final_params = resolutions[-1].params if resolutions else tool_params
                    memory.add_action(
                        step_count=step, tool_name=next_step.tool_name,
                        sub_goal=next_step.sub_goal,
                        command=f"[verified] {next_step.tool_name}.execute(**{final_params})",
                        result=result, params=final_params, verification=vrecord,
                        planner_reasoning=(getattr(next_step, "context", "") + " "
                                           + getattr(next_step, "justification", "")).strip())
                    logger.info(f"VLRR step {step}: {next_step.tool_name} -> {vstatus}"
                                + (f"/{fault}" if fault else ""))
                    if is_analyzer and not _is_hard_error(result):
                        failed = _batch_step(solver, step, memory, request)
                        if failed is not None:
                            return failed
                    decision, escalation_counts[next_step.tool_name] = _vlrr_loop_decision(
                        vstatus, fault, result, escalation_counts.get(next_step.tool_name, 0))
                    stop_reason = _result_stop_reason(decision, result)
                    if resolutions:
                        return verifier.verify_result(resolutions[-1], stop_reason=stop_reason)
                    from shape.agent.verifier import Verdict
                    return Verdict(scope="result", status="fail", stop_reason=stop_reason)

                # Multi-image: run per image (`image_workers` images at a time, `_run_images`) -- the
                # segmenter as a cohort (`_segmentation_cohort`) --, record per_image, decide on the worst outcome.
                per_results, per_vrecords, outcomes = [], [], []

                def run_one(spec, verifier_options=None):
                    return _run_managed_image(executor, next_step.tool_name, tool_params, spec, memory, step,
                                              request, metadata_profile, verifier_options)

                cohort = None
                if _uses_segmentation_cohort(solver, next_step.tool_name, run_specs):
                    runs, cohort = _segmentation_cohort(solver, run_specs, run_one, next_step.tool_name)
                elif _uses_cropping_cohort(solver, next_step.tool_name, run_specs):
                    runs, cohort = _cropping_cohort(solver, run_specs, run_one)
                else:
                    runs = _run_images(solver, next_step.tool_name, run_specs, run_one)
                for spec, (resolutions, result, vstatus, fault) in zip(run_specs, runs):
                    per_vrecords.append(_managed_verification_record(
                        verifier, next_step.tool_name, resolutions, vstatus, plan_verdict))
                    image_id = spec.get("image_id") if isinstance(spec, dict) else None
                    entry = ({"parameter_source": cohort["parameter_source"][image_id]}
                             if cohort is not None else {})
                    per_results.append(dict(result, image_id=image_id, **entry)
                                       if isinstance(result, dict)
                                       else {"result": result, "image_id": image_id, **entry})
                    outcomes.append((vstatus, fault, result))
                verification = {"tool": next_step.tool_name, "per_image": per_vrecords}
                if cohort is not None:
                    verification["cohort"] = cohort
                memory.add_action(
                    step_count=step, tool_name=next_step.tool_name,
                    sub_goal=next_step.sub_goal,
                    command=f"[verified x{len(run_specs)}] {next_step.tool_name}",
                    result={"per_image": per_results},
                    verification=verification,
                    planner_reasoning=(getattr(next_step, "context", "") + " "
                                       + getattr(next_step, "justification", "")).strip())
                logger.info(f"VLRR step {step}: {next_step.tool_name} x{len(run_specs)} -> "
                            f"{[o[0] for o in outcomes]}")
                worst_status, worst_fault, worst_result = _worst_outcome(outcomes)
                if is_analyzer and not _is_hard_error(worst_result):
                    failed = _batch_step(solver, step, memory, request)
                    if failed is not None:
                        return failed
                decision, escalation_counts[next_step.tool_name] = _vlrr_loop_decision(
                    worst_status, worst_fault, worst_result,
                    escalation_counts.get(next_step.tool_name, 0))
                stop_reason = _result_stop_reason(decision, worst_result)
                from shape.agent.verifier import Verdict
                return Verdict(scope="result",
                               status="fail" if stop_reason else "pass",
                               stop_reason=stop_reason)
        except Exception as e:
            # no fallback to an LLM-generated command: the managed step's error is a hard error
            return _record_step_error(memory, step, next_step.tool_name, next_step.sub_goal, e,
                                      (getattr(next_step, "context", "") + " "
                                       + getattr(next_step, "justification", "")).strip())

    # LLM command path (unmanaged tools, or verification disabled / no recommended params).
    # Tool metadata is loaded lazily for only this tool, not the whole toolbox.
    tool_metadata = solver["initializer"].get_tool_metadata(next_step.tool_name) or {}
    tool_command = executor.generate_tool_command(
        question=question,
        image=image_path,
        context=next_step.context,
        sub_goal=next_step.sub_goal,
        tool_name=next_step.tool_name,
        tool_metadata=tool_metadata,
        memory=memory,
    )
    result = executor.execute_tool_command(
        tool_name=next_step.tool_name,
        command=tool_command.command,
    )
    memory.add_action(
        step_count=step,
        tool_name=next_step.tool_name,
        sub_goal=next_step.sub_goal,
        command=tool_command.command,
        result=result,
        planner_reasoning=(getattr(next_step, "context", "") + " "
                           + getattr(next_step, "justification", "")).strip(),
    )
    from shape.agent.verifier import Verdict
    failed = isinstance(result, dict) and (result.get("error") or result.get("status") == "failed")
    return Verdict(scope="result", status="fail" if failed else "pass")


def _run_steps(solver, question, image_path, query_analysis, max_steps, enable_verification,
               images=None):
    """The explicit agent loop: one named step per pillar, mutating solver['memory'].

        VERIFIER (stop?) -> PLANNER (next) -> VERIFIER (plan gate) -> EXECUTOR + result VERIFIER

    Behaviour matches the prior inlined loop; the structure is made explicit so the
    three pillars (Planner, Executor, Verifier) each have one home, which the rest of
    the reorg builds on. Still unit-testable with mock planner/executor components.
    """
    planner = solver["planner"]
    memory = solver["memory"]
    group_images = images          # multi-image context for the planner (None = single)
    escalation_counts: dict = {}   # per-tool conceded(fault=upstream) count (bounds replanning)
    plan_reject_counts: dict = {}  # per-tool plan-gate rejections (bounds re-planning)

    from shape.engine import usage as _usage
    max_llm_failures = int(os.getenv("SHAPE_MAX_LLM_FAILURES", "6"))

    for step in range(1, max_steps + 1):
        # Fail-fast: if the LLM has failed back-to-back (e.g. every call 1301 content-filtered,
        # or empty responses), stop instead of thrashing tool re-runs to max_steps on garbage
        # planner/verifier output. Any single healthy response resets the streak.
        if _usage.consecutive_failures() >= max_llm_failures:
            logger.error(
                "Aborting run at step %d: %d consecutive LLM failures (content-filter/1301 or "
                "empty responses) -- the model is unavailable for this workload; stopping instead "
                "of looping to max_steps.", step, _usage.consecutive_failures())
            return "llm_unavailable"

        # VERIFIER (task gate) -> single decide_stop authority
        task_verdict = _verify_task_complete(
            planner, question, image_path, query_analysis, memory, group_images)
        if decide_stop(task_verdict):
            return getattr(task_verdict, "stop_reason", None) or "task_complete"

        # PLANNER: choose the next action
        next_step = _planner_next_step(
            planner, question, image_path, query_analysis, memory, step, max_steps,
            group_images)

        # a pick of a system-routed tool (absent from the planner's toolbox, so the planner
        # returns it unmatched) goes to `_system_routed_step`, never to the unregistered stop
        system_tool = _system_routed_name(next_step.tool_name)
        if system_tool:
            next_step.tool_name = system_tool
        elif next_step.tool_name not in solver["available_tools"] and _planner_chose_no_tool(next_step.tool_name):
            # the planner's way of saying the task is done ("None", "No further tool call",
            # "NONE - task complete"): a normal stop, recorded, not an unregistered tool
            memory.add_action(
                step_count=step,
                tool_name=next_step.tool_name,
                sub_goal=next_step.sub_goal,
                command="[not executed: the planner chose no tool]",
                result={"status": "done", "note": "the planner named no tool; the run stops"},
                planner_reasoning=(getattr(next_step, "context", "") + " "
                                   + getattr(next_step, "justification", "")).strip(),
            )
            logger.info("Planner chose no tool (%r): the run stops.", next_step.tool_name)
            return "planner_done"
        elif next_step.tool_name not in solver["available_tools"]:
            error = {
                "error": (
                    f"Planner selected unregistered tool '{next_step.tool_name}'. "
                    "No tool was executed."
                ),
                "status": "failed",
            }
            memory.add_action(
                step_count=step,
                tool_name=next_step.tool_name,
                sub_goal=next_step.sub_goal,
                command="[not executed: unregistered tool]",
                result=error,
                planner_reasoning=(getattr(next_step, "context", "") + " "
                                   + getattr(next_step, "justification", "")).strip(),
            )
            logger.error(error["error"])
            return "unregistered_tool"

        # VERIFIER (plan gate): judge the plan BEFORE spending compute. A structurally
        # unsound plan (e.g. the analyzer planned before the cropper) is blocked and handed
        # back to the planner with the diagnosis, which re-plans -- bounded so a stuck
        # planner can't loop. (J.4=B: this active gate replaces the removed `required_next`
        # override; the LLM owns forward selection, the Verifier guards plan soundness.)
        verifier, plan_verdict = _verify_plan(
            next_step, memory, getattr(planner, "task_spec", None))
        if plan_verdict.status == "fail":
            from dataclasses import asdict
            n = plan_reject_counts.get(next_step.tool_name, 0) + 1
            plan_reject_counts[next_step.tool_name] = n
            memory.add_action(
                step_count=step,
                tool_name=next_step.tool_name,
                sub_goal=next_step.sub_goal,
                command="[plan-rejected: structurally unsound]",
                result={"error": plan_verdict.fault, "status": "failed",
                        "recommendation": plan_verdict.recommendation},
                verification={"plan_verdict": asdict(plan_verdict)},
                planner_reasoning=(getattr(next_step, "context", "") + " "
                                   + getattr(next_step, "justification", "")).strip(),
            )
            if n > 2:        # planner keeps proposing an unsound plan -> give up
                return "plan_rejected_budget"
            continue         # re-plan with the diagnosis now in memory

        # EXECUTOR (+ result VERIFIER) -> single decide_stop authority
        result_verdict = _execute_and_verify(
            solver, next_step, step, question, image_path, memory,
            enable_verification, escalation_counts, verifier, plan_verdict, images)
        if decide_stop(result_verdict):
            return getattr(result_verdict, "stop_reason", None) or "result_stop"

    return "max_steps_reached"


def solve(
    question: str,
    image_path: Optional[str] = None,
    images: Optional[List] = None,
    llm_engine_name: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    max_steps: int = 10,
    query_cache_dir: str = "solver_cache",
    enable_verification: bool = True,
    enable_llm_verifier: bool = True,
    verify_refine: bool = False,
    image_workers: int = 1,
    segmentation_cohort: bool = True,
    cropping_cohort: bool = True,
    segmentation_cache_seed: Optional[int] = 0,
):
    """
    Solve a problem using SHAPE framework.

    Args:
        question: Natural language question to answer
        image_path: Optional path to image file
        llm_engine_name: Name of the LLM engine to use
        enabled_tools: Optional list of tool names to enable
        api_key: OpenAI API key
        max_steps: Maximum number of steps to execute
        query_cache_dir: Directory for caching query results
        image_workers: images whose per-image managed step (segment / crop VRR loop) runs at the same time
            (default 1: one after the other); their GPU tool calls take turns on the device.
        segmentation_cohort: segment a set of images as a cohort (default; `segmentation_cohort.run`):
            grouped by sample type, a group of >= 10 images calibrated on a sample, the consensus applied, the
            images still needing tuning by their own VRR. False: every image its own VRR.
        cropping_cohort: crop a set of images as a cohort (default; the same cohort engine over the crop VRR):
            a calibration sample per group runs the full crop VRR, the consensus crop levers apply to the rest at
            the verifier's no-search cohort mode, flagged images routed to their own VRR. False: every image its own VRR.
        segmentation_cache_seed: organoid cohort native-trial reuse with attested inputs/model/runtime
            and scoped RNG seed (default 0). None disables reuse; reasoning/QC are never cached.
    
    Returns:
        Dictionary containing:
        - direct_output: Final answer text
        - memory: Memory object with execution history
        - steps: List of executed steps
        - metadata_profile: the run's metadata profile, as a dict
          (`metadata_profile.MetadataProfile.to_dict()`)
        - cell_states: the uncorrected / corrected cell states and the batch step
          (`batch_stage.cell_states_deliverable`)
    """
    _validate_image_path(image_path)

    # The agent natively handles any number of images: a single image_path (legacy) or an
    # `images` list (paths or {path, group} dicts), optionally grouped. Per-image tools run
    # per image; the analyzer runs once over all crops.
    image_specs = _normalize_image_specs(images) if images else None
    if image_specs:
        for spec in image_specs:
            _validate_image_path(spec["path"])
        primary_image = image_specs[0]["path"]
    else:
        primary_image = image_path

    # Construct solver
    solver = construct_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=enabled_tools,
        api_key=api_key,
        query_cache_dir=query_cache_dir,
    )

    planner = solver["planner"]
    executor = solver["executor"]
    memory = solver["memory"]
    solver["image_workers"] = image_workers
    solver["segmentation_cohort"] = segmentation_cohort
    solver["cropping_cohort"] = cropping_cohort
    solver["segmentation_cache_seed"] = segmentation_cache_seed

    # The run's metadata profile (a deterministic rule over the image specs and the request),
    # built once and handed to every managed stage; it routes the system's batch step.
    from shape.agent.verification.metadata_profile import build_metadata_profile
    profile_specs = image_specs or (
        [{"path": image_path, "image_id": _image_id_from_path(image_path), "group": "default"}]
        if image_path else [])
    metadata_profile = build_metadata_profile(question, profile_specs)
    solver["metadata_profile"] = metadata_profile

    # VLRR: toggle the (vision) LLM verifier tie-breaker for this run.
    executor.enable_llm_verifier = enable_llm_verifier
    # 044: toggle the segmenter Verify refinement (scale_mismatch + vision-prompt
    # flow_error filter) for this run. OFF by default -- base-identical.
    executor.verify_refine = verify_refine

    # Set query in memory
    memory.set_query(question)

    # Analyze query
    query_analysis = planner.analyze_query(
        question=question,
        image=primary_image,
        group_images=image_specs,
    )

    # Execute steps (plan -> verify-stop -> execute), mutating memory in place
    stop_reason = _run_steps(solver, question, primary_image, query_analysis, max_steps,
                             enable_verification, images=image_specs)

    actions = memory.get_actions(llm_safe=False)
    successful_actions = []
    for action in actions:
        result = action.get("result")
        failed = (
            result is None
        ) or (
            isinstance(result, dict)
            and (result.get("error") or result.get("status") == "failed")
        ) or (
            isinstance(result, str)
            and (
                not result.strip()
                or result.lower().startswith(("error", "failed"))
            )
        )
        if not failed:
            successful_actions.append(action)

    final_review_record = None
    if not actions:
        direct_output = (
            "Analysis did not execute any tools. No segmentation, morphology "
            "measurements, embeddings, clusters, or biological conclusions were produced."
        )
    elif not successful_actions:
        failures = [
            str(action.get("result", "unknown failure"))
            for action in actions
        ]
        direct_output = (
            "Analysis failed before producing usable results. "
            f"Recorded failures: {'; '.join(failures)}"
        )
    else:
        # generate_final_output returns a plain string; be robust to either a
        # string or an object exposing .direct_output.
        # Reuse the EXACT `_analysis_stage_ran` predicate
        # `_finalize_with_vision_review` gates on below: on a segmentation-only
        # run, the answer is written text-only (no figure re-look).
        from shape.agent.verification.integration import latest_stage_metrics
        text_only_answer = not _analysis_stage_ran(
            memory, latest_stage_metrics(memory))
        final_answer = planner.generate_final_output(
            question=question,
            image=primary_image,
            memory=memory,
            text_only=text_only_answer,
        )
        direct_output = getattr(final_answer, "direct_output", None)
        if direct_output is None:
            direct_output = (
                final_answer if isinstance(final_answer, str)
                else str(final_answer)
            )

        # End-to-end finalization check: the visualizer bypasses QC, so judge the
        # output figures against the written conclusion and append a caveat if the
        # visual evidence does not support it.
        direct_output, final_review_record = _finalize_with_vision_review(
            executor, memory, direct_output)

    # VLRR: collect the per-step self-correction trail (QC verdicts, retunes,
    # escalations) for callers who want to inspect/report it.
    verification_summary = [
        a["verification"] for a in memory.get_actions(llm_safe=False)
        if a.get("verification")
    ]
    if final_review_record:
        verification_summary.append(final_review_record)

    # Persist the VLRR run trace + terminal stop_reason for offline inspection (observability;
    # best-effort so a write failure never fails the solve).
    try:
        from shape.utils.data_persistence import save_run_trace
        save_run_trace(query_cache_dir, stop_reason, verification_summary, steps=actions)
    except Exception as e:
        logger.warning(f"run-trace persistence skipped: {e}")

    # Both state sets (uncorrected, and corrected when the batch step corrected), also written
    # to CELL_STATES.json next to the analyzer's h5ad; best-effort like the run trace.
    try:
        from shape.agent.verification.batch_stage import write_cell_states
        cell_states = write_cell_states(memory, solver["metadata_profile"])
    except Exception as e:
        logger.warning(f"cell-states deliverable skipped: {e}")
        cell_states = None

    return {
        "direct_output": direct_output,
        "memory": memory,
        "steps": actions,
        "verification_summary": verification_summary,
        "stop_reason": stop_reason,
        # the end-to-end vision review of the figures: reviewed, failed (EXECUTION_ERROR) or None (nothing reviewed)
        "final_review": final_review_record,
        # the run's profile as the batch step left it (`resolve_batch_metrics`)
        "metadata_profile": solver["metadata_profile"].to_dict(),
        "cell_states": cell_states,
    }
