"""The Verifier pillar's output contract (`Verdict`).

In the agentic core (Planner | Executor | Verifier) the Verifier is the single
quality authority. It judges at two points -- the Planner's PLAN (before execution)
and the Executor's RESULT (the VLRR loop) -- and returns ONE `Verdict` to the
Planner. This module defines that contract and the mapping from the existing
within-stage VLRR result, so the new single channel is grounded in the real
producers before the loop is rewired (plan Phase 1).

Design split (see the plan file): judgment is REASONED over evidence; only
termination is deterministic. The `Verdict` therefore carries both the reasoning
(`fault`, `recommendation`, `confidence`) and the deterministic loop bookkeeping
(`expected_effect` vs `observed_effect`, `rerun_count`, `stop_reason`) -- the latter
is what lets the loop be auditable and provably bounded. `observed_effect` is newly
persisted here: the within-stage loop computes it for control flow but never stored
it (audit gap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Scope = Literal["plan", "result"]
Status = Literal["pass", "warning", "fail"]
# "conceded" replaces the old escalate_upstream/escalate_human (task 032); which of the
# two it was is carried by the StageResolution's `fault` (ceiling|upstream -- "human"
# retired into "ceiling", task 045), not this action string. `exclude_samples` stays a
# distinct, non-concede data-op (D3).
Action = Literal["accept", "retune_here", "conceded", "exclude_samples"]
Confidence = Literal["high", "medium", "low"]


@dataclass
class Verdict:
    """The single structured judgment the Verifier feeds back to the Planner.

    Used for BOTH plan verification (``scope="plan"``) and result verification
    (``scope="result"``, the VLRR loop). One channel, one owner per decision.
    """

    scope: Scope
    status: Status
    # --- reasoning (autonomous) ---
    fault: Optional[str] = None            # WHERE/what is wrong; None = nothing wrong
    recommendation: Optional[str] = None   # WHAT to do next, in words
    confidence: Confidence = "high"
    evidence: dict = field(default_factory=dict)   # metrics/observations behind the call
    # --- result-scope recovery bookkeeping (deterministic) ---
    action: Action = "accept"
    param_delta: Optional[dict] = None
    expected_effect: dict = field(default_factory=dict)   # predicted metric directions
    observed_effect: dict = field(default_factory=dict)   # what actually happened on rerun
    rerun_count: int = 0
    upstream_target: Optional[str] = None
    # --- termination (deterministic) ---
    stop_reason: Optional[str] = None      # set => the loop should terminate; carries the why

    @property
    def ok(self) -> bool:
        """Whether the Planner can proceed downstream without intervention."""
        return self.status == "pass" and self.action == "accept" and self.stop_reason is None


_DECISION_TO_STATUS = {"PASS": "pass", "WARNING": "warning", "FAIL": "fail"}


def verdict_from_vlrr(vres, *, observed_effect: Optional[dict] = None,
                      rerun_count: int = 0,
                      stop_reason: Optional[str] = None) -> Verdict:
    """Map an existing within-stage ``VLRRResult`` into the unified ``Verdict``.

    Grounds the new single channel in the real producer (the per-stage verifiers)
    WITHOUT rewriting them yet, and finally persists ``observed_effect`` -- which
    the within-stage loop computes (via ``reconcile``) for control flow but never
    stored. The caller supplies ``observed_effect`` from that reconcile result.
    """
    fault = f"{vres.locus}: {', '.join(vres.triggered)}" if vres.triggered else None
    return Verdict(
        scope="result",
        status=_DECISION_TO_STATUS.get(vres.decision, "warning"),
        fault=fault,
        recommendation=vres.notes or None,
        evidence=dict(vres.metrics or {}),
        action=vres.action,
        param_delta=vres.param_delta,
        expected_effect=dict(vres.expected_effect or {}),
        observed_effect=dict(observed_effect or {}),
        rerun_count=rerun_count,
        upstream_target=vres.upstream_target,
        stop_reason=stop_reason,
    )


class Verifier:
    """The Verifier pillar: the agent's single quality authority.

    Two entry points, one ``Verdict`` contract:
      - ``verify_plan(...)``   before execution: is the plan structurally sound?
      - ``verify_result(...)`` after execution: the VLRR judgement on the output.

    This is the additive interface the loop (plan Phase 0) will be wired onto. For
    now it COMPOSES the existing within-stage verifiers rather than re-implementing
    them; internalizing that logic is plan Phase 2.
    """

    # Hard pipeline prerequisites (stage -> the stage that MUST already have run).
    # These are structural DEPENDENCIES (the analyzer cannot run before crops exist),
    # not quality thresholds, so they stay deterministic.
    _HARD_PREREQ = {"crop": "segmentation",
                    "representation": "crop",
                    "morphological_state": "crop"}

    def verify_plan(self, next_tool: str, used_tools, task_spec=None) -> Verdict:
        """Plan-level gate: is ``next_tool`` (a) IN SCOPE for the query and (b) does it
        have its pipeline prerequisite met? -- BEFORE any compute is spent.

        Two deterministic checks, both handing an unsound plan back to the planner to
        re-plan (a bounded guard, never a competing decision -- consistent with the
        planner holding final authority):
          - [045 P3 Fix 0] SCOPE: a managed tool whose stage is out of the query's
            boundary (e.g. the cropper/analyzer on a segment/count query) is rejected
            -- the hard net against the vicious cycle. Fail-open when `task_spec` is
            None/unclassified.
          - ORDERING: a top-level dependency error (e.g. the analyzer before the
            cropper). Replaces the fragile LLM-prompt `required_next` enforcement.
        Unmanaged tools (search/visualize) are not scope- or prereq-gated here.
        """
        from shape.agent.verification.stage_map import (
            stage_by_name, stages_for_tool, tool_in_scope)

        if not tool_in_scope(next_tool, task_spec):
            terminal = getattr(task_spec, "required_terminal_stage", None)
            return Verdict(
                scope="plan", status="fail",
                fault=f"{next_tool} is out of scope for this query "
                      f"(query is answered at the '{terminal}' stage)",
                recommendation=(f"this query's task ends at the '{terminal}' stage; do "
                                f"NOT run downstream tools ({next_tool}) -- stop with a "
                                f"best-effort answer or select an in-scope tool"),
            )

        used = set(used_tools or [])
        for spec in stages_for_tool(next_tool):
            prereq = self._HARD_PREREQ.get(spec.stage)
            if not prereq:
                continue
            prereq_spec = stage_by_name(prereq)
            producers = set(prereq_spec.tool_names) if prereq_spec else set()
            if producers and used.isdisjoint(producers):
                return Verdict(
                    scope="plan", status="fail",
                    fault=f"{next_tool} needs the '{prereq}' stage first",
                    recommendation=(f"run a {prereq} tool "
                                    f"({' / '.join(sorted(producers))}) before {next_tool}"),
                )
        return Verdict(scope="plan", status="pass")

    def verify_result(self, resolution, *, stop_reason: Optional[str] = None) -> Verdict:
        """Map a within-stage ``StageResolution`` into a ``Verdict``.

        Persists ``observed_effect`` by reconciling the last applied prediction
        against the final metrics -- so the verdict records what the retune ACTUALLY
        did, not only what it predicted (the audit gap).
        """
        from shape.agent.verification.base import reconcile

        trials = getattr(resolution, "trials", None) or []
        parents = getattr(resolution, "parents", None) or []
        observed: dict = {}
        selected = (getattr(resolution, "details", None) or {}).get("selected")
        if parents and selected is not None and parents[selected] is not None:
            # wave mode: a trial carries the move that produced it, measured from its parent
            _, parent_vres, _ = trials[parents[selected]]
            _, chosen_vres, _ = trials[selected]
            if chosen_vres.expected_effect:
                observed = reconcile(chosen_vres.expected_effect, parent_vres.metrics, chosen_vres.metrics)
        elif not parents and len(trials) >= 2:
            _, prev_vres, _ = trials[-2]
            _, last_vres, _ = trials[-1]
            if prev_vres.expected_effect:
                observed = reconcile(prev_vres.expected_effect,
                                     prev_vres.metrics, last_vres.metrics)
        return verdict_from_vlrr(resolution.verdict, observed_effect=observed,
                                 rerun_count=max(0, len(trials) - 1),
                                 stop_reason=stop_reason)

    # -- managed-stage resolution (the Verifier owns adapt + verify + rerun) --------

    def _adapter_for(self, stage, base_kwargs, upstream_metrics):
        """Bind a stage to a (native_result, upstream_ctx) -> (verifier_input, ctx)
        adapter. Each stage's native-result adaptation now lives WITH its verifier
        (Phase 2b) -- there is no longer a shared adapters bridge module the
        integration layer wires up; the Verifier sources each from its verifier file.
        """
        bk = base_kwargs or {}
        if stage == "batch_correction":
            from .verification.verifiers.batch_correction_unified import adapt_batch_correction
            return lambda result, up: adapt_batch_correction(result, up)
        if stage == "segmentation":
            from .verification.verifiers.segmentation import adapt_segmentation
            seg_image = bk.get("image")
            return lambda result, up: adapt_segmentation(
                result, up or (upstream_metrics or {}), image_path=seg_image)
        if stage == "crop":
            from .verification.verifiers.crop import adapt_crop
            return lambda result, up: adapt_crop(result, seg_metrics=upstream_metrics or {})
        if stage == "representation":
            from .verification.verifiers.representation import adapt_representation, trial_context
            # task 008 P3 (additive): each trial's context is enriched from `base_kwargs` --
            # recomputed nn_visual_consistency for THIS trial's own adata + the up-front
            # crop_vision_profile. With no base_kwargs (every pre-P3 caller) trial_context
            # returns {}, so this is byte-identical to the old `adapt_representation(adata_path)`.
            # task 021 Stage 1b (additive): also thread THIS trial's own analyzer_kwargs (the
            # core-routed retrain runner's result, e.g. `_make_retrain_runner`) so `adjudicate`
            # sees whether a floor lever (augment_color/field_aware) is already applied instead
            # of re-proposing a no-op. `result` may lack "analyzer_kwargs" (e.g. the general
            # solver.py path's tool-native result) -> trial_kwargs=None, unchanged behavior.
            return lambda result, up: adapt_representation(
                result.get("adata_path"),
                context=trial_context(result.get("adata_path"), bk,
                                      trial_kwargs=result.get("analyzer_kwargs"),
                                      report=(result.get("source_reliability_report")
                                              or bk.get("report"))))
        if stage == "morphological_state":
            from .verification.verifiers.morphological_state import adapt_morphological
            return lambda result, up: adapt_morphological(
                result.get("adata_path"),
                resolution=float(result.get("cluster_resolution", 0.5) or 0.5))
        raise ValueError(f"no adapter bound for stage {stage}")

    @staticmethod
    def _tool_dir_for(tool_name):
        if not tool_name:
            return None
        stem = str(tool_name).replace("_Tool", "")
        return "_".join(part.lower() for part in stem.split("_"))

    def resolve_managed_stage(self, spec, runner, initial_params, initial_result, *,
                              base_kwargs=None, upstream_metrics=None,
                              llm_engine=None, max_iters=6, tool_name=None,
                              verify_refine=False, request=None,
                              upstream_ctx=None, metadata_profile=None, verifier_options=None):
        """Verify (+ retune/rerun) one managed stage from the NATIVE tool result.

        The Verifier owns the whole per-stage flow: it constructs the stage's
        within-stage verifier, wires the optional vision-LLM tie-breaker, adapts the
        native result internally, and runs the bounded retune/rerun loop. The
        integration layer hands over native results instead of wiring adapters
        itself. `request` (the user's request: the solve() query, else the step's
        sub_goal) reaches a segmentation verifier that declares a `request`
        attribute (the agentic one states its target from it). `upstream_ctx` is the
        stage context the loop hands the verifier (e.g. the batch corrector's
        ``{"batch_task": BatchTask}``); `metadata_profile` (the run's
        `metadata_profile.MetadataProfile`) is set on a verifier that declares a
        `metadata_profile` attribute. `verifier_options` ({attribute: value}) are set on the stage's verifier
        when it declares them (e.g. the segmentation cohort's `cohort_levers` / `start_params`); an option the
        verifier does not declare is an error. Returns a StageResolution.
        """
        from shape.agent.verification.stage_map import verifier_cls_for
        from shape.agent.verification.step_verifier import StepVerifier

        # a tool's own verifier (`StageSpec.verifier_by_tool`), else the stage's
        verifier_cls = verifier_cls_for(spec, tool_name)
        verifier = verifier_cls(**(spec.verifier_kwargs or {}))
        if tool_name:
            verifier.tool_name = tool_name
            tool_dir = self._tool_dir_for(tool_name)
            if tool_dir and getattr(verifier, "tool_dir", None):
                verifier.tool_dir = tool_dir
        if llm_engine is not None and getattr(verifier, "uses_llm", False):
            verifier.llm_engine = llm_engine
        if spec.stage == "segmentation":
            # 044 (VRR refinement of the working 040 loop): OFF by default is
            # code-path-identical to base -- see SegmentationVerifier.verify_refine.
            verifier.verify_refine = verify_refine
            if request and hasattr(verifier, "request"):
                verifier.request = request
        # no stage verifier declares `metadata_profile` yet: the batch stage reads the profile through its
        # BatchTask (`upstream_ctx`); this is the channel for a verifier that will
        if metadata_profile is not None and hasattr(verifier, "metadata_profile"):
            verifier.metadata_profile = metadata_profile
        for name, value in (verifier_options or {}).items():
            if not hasattr(verifier, name):
                raise ValueError(f"{type(verifier).__name__} has no option {name!r}")
            setattr(verifier, name, value)
        # a verifier may adapt the native result itself (`base.AgenticVerifier.make_adapter`)
        adapter = (verifier.make_adapter(base_kwargs) if hasattr(verifier, "make_adapter")
                   else self._adapter_for(spec.stage, base_kwargs, upstream_metrics))
        if getattr(verifier, "uses_waves", False):
            max_iters = 1 + int(verifier.budget)     # a wave-mode verifier's budget: rounds after trial 0
        sv = StepVerifier(
            max_iters=max_iters,
            zero_detection_rescue=(tool_name == "Organoid_Segmenter_Tool"),
        )
        return sv.resolve_stage(verifier, adapter, runner, initial_params,
                                initial_result, upstream_ctx=upstream_ctx or {})
