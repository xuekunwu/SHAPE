"""Organoid instance segmentation through the shared agentic VRR.

The search, admission, reconciliation and delivery are the segmentation loop's;
the fixed custom CO model, organoid scale and grayscale action space are local.
"""

from __future__ import annotations

import math
import time
from typing import Literal
from pydantic import BaseModel, Field
from shape.engine.errors import is_fatal_llm_error, NoEngineError
from shape.utils.schemas import QuestionFailed
from .segmentation import SegmentationAgenticVerifier, VerifyReport, PROMPT_IMAGE_PX, _image_array
from ..base import AgenticVerifier
from ..segmentation_policy import EFFICIENT


class _BudgetedReasoner:
    def __init__(self,engine,verifier,phase):
        self.engine,self.verifier,self.phase=engine,verifier,phase
    def __getattr__(self,name):return getattr(self.engine,name)
    def generate(self,*args,**kwargs):
        reason=self.verifier._budget_reason()
        if reason:
            self.verifier.organoid_budget_stop=reason
            raise QuestionFailed(reason,label="budget",cause=reason)
        self.verifier._check_search_reserve(self.phase)
        return self.engine.generate(*args,**kwargs)


class OrganoidRepresentativeReview(VerifyReport):
    """Assessment of the sole attached candidate, bound by code rather than a generated call ID."""


class EmptyOrganoidReview(BaseModel):
    target_presence: Literal["absent", "present", "uncertain"]
    interpretable: bool
    confidence: Literal["high", "medium", "low"]
    evidence: str = Field(min_length=10)
    limitations: list[str] = Field(default_factory=list)



class OrganoidSegmentationAgenticVerifier(SegmentationAgenticVerifier):
    """Tune the CO model's inference parameters, including four-stage search.

    Keep ``name = segmentation`` so the existing segmentation search-strategy setting
    applies to both tools. Read the organoid CARD at construction, before the common
    loop parses its levers. The executor serializes and caches this in-process model;
    the CPSAM server's replicas never determine its concurrency.
    """

    tool_dir = "organoid_segmenter"
    tool_phrase = "organoid-segmentation tool (fixed custom CO Cellpose model)"
    value_examples = "null, 15, 60, 0.6, -2, 5000, true"
    efficiency_defaults = EFFICIENT   # the organoid CARD's cost-saving stop rules are its defaults

    # Thresholds stop new work; the isolated cohort worker also enforces wall time.
    candidate_checkpoint = None
    organoid_selective_review = False
    organoid_representative = False
    preflight_override = None
    max_reported_tokens = 24000
    max_native_calls = 6
    # Reported-token admission threshold, not a hard provider billing cap.
    # Keep a final comparison possible instead of spending every token on planning.
    final_selection_reserve_tokens = 10000
    max_image_seconds = 300

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.empty_reviews = {}
        self.organoid_budget_stop = None
        self._best_qc = -1.0
        self.search_budget_stop = None
        self._body_image = None

    def _preflight(self, path):
        return self.preflight_override if self.preflight_override is not None else super()._preflight(path)

    def _budget_reason(self):
        if self.organoid_budget_stop:
            return self.organoid_budget_stop
        if self.tokens.get("total_tokens", self.tokens.get("total", self.tokens.get("reported_total",0))) >= self.max_reported_tokens:
            return "reported_token_budget"
        if getattr(self, "started", None) and time.time()-self.started >= self.max_image_seconds:
            return "image_wall_budget"
        return None

    def _check_search_reserve(self, phase):
        # Check on every logical retry as well as before a new round. Final
        # selection and its admissibility retry may use the reserved balance.
        if phase in ("final", "final_retry"):
            return
        spent = self.tokens.get("total_tokens", self.tokens.get("total", self.tokens.get("reported_total", 0)))
        reserve = min(self.final_selection_reserve_tokens, self.max_reported_tokens / 2)
        if spent >= max(0, self.max_reported_tokens - reserve):
            self.search_budget_stop = "final_selection_reserve"
            self.stop_reason = self.search_budget_stop
            raise QuestionFailed(self.search_budget_stop, label=phase, cause=self.search_budget_stop)

    def _ask(self, content, schema, label):
        reason = self._budget_reason()
        if reason:
            self.organoid_budget_stop = reason
            raise QuestionFailed(reason, label=label, cause=reason)
        self._check_search_reserve(label)
        engine=self.llm_engine
        if engine is None:
            return super()._ask(content,schema,label)
        self.llm_engine=_BudgetedReasoner(engine,self,label)
        try:
            return super()._ask(content,schema,label)
        finally:
            self.llm_engine=engine

    def _review_empty(self, index):
        """One raw-image review, never GT or cohort consensus, before an empty mask is accepted."""
        if index in self.empty_reviews:
            return self.empty_reviews[index]
        if self.empty_reviews:
            original_index = next(iter(self.empty_reviews))
            self.empty_reviews[index] = {**self.empty_reviews[original_index],"reused_raw_review_from":original_index}
            return self.empty_reviews[index]
        if self.llm_engine is None:
            raise NoEngineError("Organoid empty-result review requires a vision reasoner")
        image = self._input_image()
        answer = None
        if image is not None and self.llm_engine is not None and not self._budget_reason():
            prompt = ("Review this raw cardiac organoid image independently of any segmentation. "
                      "Is any whole organoid or spheroid visible, including faint or border targets? "
                      "Use absent only when the image is interpretable and no target is visible. "
                      "Use uncertain when darkness, low contrast, blur or clipping prevent determining presence. "
                      "Uncertainty about the exact boundary alone does not negate a clearly visible target. "
                      "Do not assume a target must exist. Cite visible evidence and limitations. "
                      "No annotation or ground truth is provided.")
            started = time.perf_counter()
            reply, outcome, error = None, "error", None
            try:
                reply = self.llm_engine.generate([prompt, image], response_format=EmptyOrganoidReview, max_tokens=1500)
                self._count_tokens(reply)
                value = reply.get("content") if isinstance(reply, dict) else reply
                answer = value if isinstance(value, EmptyOrganoidReview) else EmptyOrganoidReview.model_validate_json(value) if isinstance(value,str) else EmptyOrganoidReview.model_validate(value)
                outcome = "ok"
            except Exception as exc:
                error = type(exc).__name__
                if is_fatal_llm_error(exc):
                    raise
            finally:
                self.record_llm_call("negative_review",1,time.perf_counter()-started,reply,[prompt,image],outcome,error)
        entry = answer.model_dump() if answer else dict(target_presence="uncertain",interpretable=False,confidence="low",evidence="No usable independent raw-image review",limitations=["review unavailable or invalid"])
        confident = bool(answer and answer.interpretable and answer.confidence=="high")
        usable = confident and not answer.limitations
        # Boundary uncertainty does not negate a confident observation that a target exists.
        # Accepting absence remains stricter: no unresolved limitation may hide a target.
        if not confident or (answer.target_presence=="absent" and not usable):
            entry["target_presence"] = "uncertain"
        qc = self.input.get("image_qc") or {}
        if qc.get("finite_fraction",0)<1 or qc.get("dynamic_range",0)<=0:
            usable = False
            entry["target_presence"] = "uncertain"
            entry["limitations"] = list(entry["limitations"])+["raw image has no finite interpretable dynamic range"]
        entry["accepted"] = bool(usable and answer.target_presence=="absent")
        self.empty_reviews[index] = entry
        return entry

    def admit_defaults(self, inp, attempts, remaining):
        if self.empty_reviews:
            review = next(iter(self.empty_reviews.values()))
            if not review["accepted"] and review["target_presence"] != "present":
                call = self.label(self.rounds[0][0])
                return self.set_admission(dict(call=call,rule="NOT_SEARCHED",decision="NOT_SEARCHED",problems=["empty result needs independent review"],answer=review,execution_error="negative_review_uncertain"))
        admission = super().admit_defaults(inp, attempts, remaining)
        if not (self.organoid_selective_review and self.organoid_representative
                and admission["decision"] == "SKIP"):
            return admission
        call = admission["call"]
        raw, overlay = self._input_image(), self._overlay(call, PROMPT_IMAGE_PX)
        try:
            if raw is None or overlay is None:
                raise QuestionFailed("representative image/overlay unavailable", label="representative_review", cause="invalid_review")
            prompt = (f"Review representative cardiac organoid segmentation {call}. Each whole organoid or "
                      "spheroid is one instance. Compare the raw image and overlay: is the full outer target "
                      "covered, or only its internal core? Check splitting, missed targets and background. "
                      "Return a brief assessment for this exact call, with severity and confidence. "
                      "A major defect or low confidence cannot authorize cohort calibration. No GT is provided.")
            answer = self._ask([prompt, "RAW IMAGE:", raw, f"OVERLAY {call}:", overlay],
                               OrganoidRepresentativeReview, "representative_review")
            self._assess(call, answer, attempts)
        except QuestionFailed as exc:
            return self.set_admission(dict(call=call, rule="NOT_SEARCHED", decision="NOT_SEARCHED",
                problems=[str(exc)], answer=None, execution_error="representative_review_unavailable"))
        issues = self.call_problems(attempts, call)
        return self.set_admission(dict(call=call, rule="RUN" if issues else "SKIP",
            decision="RUN" if issues else "SKIP", problems=issues, answer=answer.model_dump()))

    def measured_problems(self, vres, diameter, populations, color_filter=None, prefilter=None):
        found = super().measured_problems(vres, diameter, populations, color_filter, prefilter)
        if self.organoid_selective_review:
            from shape.tools.organoid_segmenter.body_qc import problems
            found += problems(vres.metrics or {})
        return found

    def call_problems(self, attempts, call):
        found = super().call_problems(attempts, call)
        assessment = self.assessments.get(call) or {}
        if self.organoid_selective_review and (assessment.get("severity") == "major"
                                               or assessment.get("confidence") == "low"):
            found.append("whole-organoid visual assessment requires review")
        return found

    def plan_round(self, inp, attempts, k_max, remaining):
        reason = self._budget_reason()
        if reason or sum(len(r) for r in self.rounds) >= self.max_native_calls:
            self.organoid_budget_stop = reason or "native_call_budget"
            self.stop_reason = self.organoid_budget_stop
            return []
        if self.cohort_levers is None:
            try:
                self._check_search_reserve("round")
            except QuestionFailed:
                return []
        return super().plan_round(inp,attempts,k_max,remaining)

    def concurrency(self) -> int:
        return 1

    def _color_populations(self, masks):
        # The organoid tool segments channel 0; it has no RGB class-filter lever.
        return None

    def _color_rule(self, index):
        return None

    def quantify(self, tool_output, context):
        metrics = super().quantify(tool_output, context)
        index = int(context.get("trial_index") or 0)
        if self.organoid_selective_review and index and metrics.get("cell_count", 0) > 0:
            import numpy as np
            from PIL import Image
            from shape.tools.organoid_segmenter.body_qc import body_qc
            if self._body_image is None:
                self._body_image = _image_array(self._input_path)
            path = self._masks.get(index)
            if path:
                if str(path).endswith(".npy"):
                    masks = np.load(path)
                else:
                    with Image.open(path) as image: masks = np.asarray(image).copy()
                metrics.update(body_qc(self._body_image, masks))
        if index and metrics.get("cell_count") == 0 and isinstance(tool_output, dict) and tool_output.get("mask_path",tool_output.get("organoid_mask_path")):
            review = self._review_empty(index)
            metrics["negative_review_accepted"] = float(review["accepted"])
            if review["accepted"]:
                tool_output["status"] = "success"
                tool_output.pop("error",None)
            tool_output["negative_review"] = review
        if index and callable(self.candidate_checkpoint) and isinstance(tool_output,dict):
            valid = not self.guard(None, type("Readout",(),{"metrics":metrics})()) and tool_output.get("status")=="success"
            score = self.qc_score(metrics)
            if valid and score > self._best_qc:
                self._best_qc = score
                self.candidate_checkpoint(dict(result=dict(tool_output),qc_score=score,selection_basis="best completed native QC; not quality-certified"))
        area = metrics.get("mask_area_median")
        if area and math.isfinite(area) and area > 0:
            # The shared cell helper clamps at 200 px; report a large organoid's true size.
            _, bounds, _ = self.lever_spec("diameter")
            diameter = 2.0 * math.sqrt(area / math.pi)
            metrics["mask_diameter_px"] = min(bounds[1], max(bounds[0], diameter))
        return metrics

    @staticmethod
    def _scale_problem(metrics, used):
        # Cellpose resizes by 30 / diameter, and this checkpoint was trained at native scale:
        # a mask's pixel size says nothing about the right diameter, so no mismatch is a problem
        # and no reference call moves diameter to the measured size.
        return None

    def call_error(self, index, state):
        result = state.trials[index][2]
        review = self.empty_reviews.get(index,{})
        if review.get("accepted") and isinstance(result,dict) and result.get("organoid_count",result.get("cell_count"))==0:
            return None
        return super().call_error(index,state)

    def guard(self, inp, vres):
        problems = super().guard(inp, vres)
        if (vres.metrics or {}).get("negative_review_accepted") == 1:
            problems = [p for p in problems if p not in ("no cells detected","visual output invalid (black overlay or implausible coverage)")]
        return [p.replace("no cells detected", "no organoids detected") for p in problems]


    def readouts_text(self) -> str:
        selective = ("Dark-body thumbnail coverage is a GT-free triage cue, not an annotation or a mask. "
                     "A compact dark body mostly outside the prediction suggests internal-core undersegmentation; "
                     "many instances per body suggest splitting. Check the raw image and overlay before tuning. "
                     "Do not grow masks merely to satisfy a heuristic.\n") if self.organoid_selective_review else ""
        return selective + (
            "Input image facts: size, channels, dtype, image QC and the raw-image preflight.\n"
            "The fixed custom CO model segments a 2D grayscale image (channel 0). Each whole organoid or\n"
            "spheroid is one instance; its constituent cells and internal texture are not separate targets.\n"
            "QC readouts (GT-free evidence, not ground truth): cell_count is the organoid instance count;\n"
            "foreground_coverage, mask_area_median, mask_area_CV, tiny_mask_fraction (< 0.3 x median area),\n"
            "large_mask_fraction (> 3 x median area), border_mask_fraction, mean_solidity (area / bounding\n"
            "box), and qc_score. mask_diameter_px is the equivalent circular diameter of the median mask in\n"
            "pixels, a size readout only: never use it as the diameter lever. Cellpose resizes the image by\n"
            "30 / diameter; null keeps the checkpoint's native training scale, lower values enlarge the image\n"
            "and higher values shrink it.\n"
            "missed_foreground_fraction, count_miss_fraction and missed_blob_size_ratio describe uncovered\n"
            "texture. Halos, lumens and background texture can contribute: inspect the raw image and overlay\n"
            "before attributing these readouts to missed organoids. Natural size variation, irregular outlines\n"
            "and border-truncated organoids also need visual interpretation.\n"
            "Tune only the CARD inference levers. The model and input channel stay fixed; no colour-class\n"
            "filter is implemented. A zero-detection call needs an independent, confident raw-image negative review; "
            "uncertain empty results require review and must not be tuned solely to force a nonempty mask."
        )

    def calls_legend(self) -> str:
        return (
            "change = relative to the parent; non_default = the configuration's non-default values;\n"
            "vs_parent = QC readout deltas; prediction = the predicted change beside its observed change;\n"
            "measured_problems = guard failures and shape checks;\n"
            "your_assessment = your assessment of the organoid overlay; guard = admissible final choice."
        )

    def _rows(self, inp, attempts):
        rows = super()._rows(inp, attempts)
        for row in rows:
            row.pop("color_populations", None)
        return rows

    def finalize_search(self, state, selection):
        if self.organoid_budget_stop and selection.index is None:
            inp, attempts = self._bind(state)
            candidates = []
            for call in self.admissible(inp, attempts):
                index = self._labels[call]
                result = state.trials[index][2]
                score = self.qc_score(attempts[call][2].metrics)
                if (isinstance(result, dict) and result.get("status")=="success"
                        and result.get("mask_path", result.get("organoid_mask_path")) and math.isfinite(score)):
                    candidates.append((index, call, score))
            if candidates:
                index, call, score = max(candidates, key=lambda c:c[2])
                selection.index = index
                selection.status, selection.fault = "conceded", "ceiling"
                selection.details.update(call=call, choice="budget_retained_candidate",
                    retained_candidate_basis="best completed native QC; not quality-certified",
                    retained_candidate_native_qc=score)
                self.deliver(state, index, selection.details)
        selection = super().finalize_search(state, selection)
        reconciled = []
        for index, (params, vres, _) in enumerate(state.trials):
            parent = state.parents[index]
            if parent is None or not vres.param_delta:
                continue
            before = state.trials[parent][1]
            # The persisted audit includes all stated effects, even a readout the
            # segmentation prompt hides. No extra reasoning call is necessary.
            predictions = AgenticVerifier._prediction(self, before, vres)
            complete = bool(predictions) and all(p["observed"] != "not computable" for p in predictions)
            entry = state.entries[index] or {}
            error = self.call_error(index, state)
            failed = state.failed(index) or error is not None
            status = ("failed" if failed else "unpredicted" if not predictions
                      else "reconciled" if complete else "not_computable")
            reconciled.append({"call": self.label(index), "parent": self.label(parent),
                               "changes": dict(vres.param_delta), "status": status,
                               "expected_effect": dict(vres.expected_effect),
                               "before": {m: self._score(before, m) for m in vres.expected_effect},
                               "after": {m: self._score(vres, m) for m in vres.expected_effect},
                               "observed": predictions, "metric_deltas": self._effects(before, vres),
                               # An absent parent readout is not a measured success.
                               "hypothesis": entry.get("hypothesis") if complete and status != "failed" else None,
                               "score_outcome": entry.get("outcome") if not failed else None,
                               "error": error})
        selection.details["reconciliation"] = reconciled
        selection.details["negative_reviews"] = dict(self.empty_reviews)
        selection.details["organoid_budget_stop"] = self.organoid_budget_stop
        selection.details["search_budget_stop"] = self.search_budget_stop
        selection.details["organoid_budget_policy"] = dict(
            max_reported_tokens=self.max_reported_tokens,
            final_selection_reserve_tokens=self.final_selection_reserve_tokens,
            max_native_calls=self.max_native_calls, max_image_seconds=self.max_image_seconds)
        assessment = self.assessments.get(selection.details.get("call"), {})
        selection.details["selected_visual_assessment"] = dict(assessment)
        body_unreviewed = False
        if self.organoid_selective_review and selection.index is not None:
            from shape.tools.organoid_segmenter.body_qc import problems
            body_issues = problems(state.trials[selection.index][1].metrics or {})
            body_unreviewed = bool(body_issues and not (assessment.get("severity") in ("none", "minor")
                                                         and assessment.get("confidence") == "high"))
        selection.details["organoid_body_review_required"] = body_unreviewed
        visual_review = body_unreviewed or bool(assessment and (
            assessment.get("severity") == "major" or assessment.get("confidence") == "low"))
        selection.details["visual_review_required"] = visual_review
        selection.details["quality_certified"] = False
        selection.details["needs_review"] = bool(selection.status!="satisfied" or selection.details.get("accepted_with_flags") or visual_review)
        if visual_review:
            selection.details["accepted_with_flags"] = selection.index is not None
            selection.status, selection.fault = "conceded", "ceiling"
            selection.reason = "visual_review_required"
        review = self.empty_reviews.get(selection.index, {})
        selection.details["negative_confirmed"] = bool(review.get("accepted"))
        if review.get("accepted") and selection.index is not None:
            result = state.trials[selection.index][2]
            result["status"] = "success"
            result.pop("error",None)
            result["negative_confirmed"] = True
            result["summary"] = "Empty mask accepted after independent raw-image review confirmed no visible organoids; native QC readouts retained."
        if self.organoid_budget_stop:
            selection.details["accepted_with_flags"] = selection.index is not None
            selection.details["needs_review"] = True
            selection.reason = self.organoid_budget_stop
        return selection

    def final_rule_text(self) -> str:
        return super().final_rule_text().replace("cells and a valid overlay", "organoids and a valid overlay")

    def admissible_text(self) -> str:
        return "organoids, a valid overlay"

    def extra_details(self, inp, attempts):
        details = super().extra_details(inp, attempts)
        for key in ("color_evidence", "color_decision", "color_populations"):
            details.pop(key, None)
        details["segmentation_tool"] = "Organoid_Segmenter_Tool"
        details["organoid_selective_review"] = self.organoid_selective_review
        details["organoid_representative"] = self.organoid_representative
        if self.organoid_selective_review:
            details["organoid_body_triage"] = {c: {k: v for k, v in a[2].metrics.items()
                if k.startswith("organoid_body_") or k == "organoid_fragments_per_body"}
                for c, a in attempts.items()}
        return details
