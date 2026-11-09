"""Load-bearing tests for the Batch Corrector verifier outside the budgeted search itself: the managed
stage, the native path without a task, the admission evidence and the calibration view."""

import numpy as np
import pandas as pd

from shape.agent.verification.stage_map import stage_by_name
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.batch_correction_unified import (
    BatchCorrectionVerifier,
    BatchTask,
    CalibrationView,
    adapt_batch_correction,
    admission_evidence,
    uncorrected_result,
)
from shape.tools.batch_corrector.tool import DEFAULT_CONFIG
from shape.utils.schemas import VLRRResult


class RuleEngine:
    """A reasoner that answers the admission question with the deterministic rule's outcome (no veto)."""

    def generate(self, content, response_format=None):
        import re

        from shape.agent.verification.verifiers.batch_correction_unified import AdmissionVerdict

        rule = re.search(r"Deterministic rule outcome: (\w+)\.", content[0]).group(1)
        return {"content": AdmissionVerdict(decision=rule, reason_code="rule", reasoning="r")}


def _with_engine():
    verifier = BatchCorrectionVerifier()
    verifier.llm_engine = RuleEngine()
    return verifier


def _result(detected=True):
    return {"status": "CORRECTED_HARMONY" if detected else "SKIPPED_NO_BATCH_EFFECT",
            "corrected_cls_output": np.zeros((4, 2)), "batch_effect": {"detected": detected},
            "gt_free_metrics": {"batch_effect_detected": detected, "batch_mixing_input": 0.4,
                                "batch_mixing_output": 0.9, "centroid_ratio_input": 0.2, "centroid_ratio_output": 0.05,
                                "within_batch_neighbour_overlap": 0.8, "effective_rank_ratio": 1.0, "displacement": 0.1},
            "telemetry": {}, "provenance": {"input_shape": [4, 2]}, "preprocess": {}, "seconds": 0.0}


def test_batch_corrector_is_a_managed_stage():
    spec = stage_by_name("batch_correction")
    assert spec is not None
    assert spec.verifier_cls is BatchCorrectionVerifier
    assert spec.tool_names == ("Batch_Corrector_Tool",)
    assert set(spec.param_map) == {"harmony_params"}
    assert spec.default_params == {"harmony_params": DEFAULT_CONFIG}


def test_without_a_task_the_gate_skips_or_the_verifier_abstains():
    calls = []

    def runner(params):
        calls.append(params)
        return _result()

    initial = {"harmony_params": dict(DEFAULT_CONFIG)}
    abstain = StepVerifier(max_iters=9).resolve_stage(_with_engine(), adapt_batch_correction, runner,
                                                      initial, _result())
    assert calls == [] and abstain.result is None and abstain.status == "conceded"
    assert abstain.reason == "separability_unverifiable"
    skip = StepVerifier(max_iters=9).resolve_stage(_with_engine(), adapt_batch_correction, runner,
                                                   initial, _result(detected=False))
    assert calls == [] and skip.status == "satisfied" and skip.result is skip.trials[0][2]


def test_the_managed_stage_spends_the_verifier_budget():
    from shape.agent.verifier import Verifier

    seen = {}

    class Recording(StepVerifier):
        def resolve_stage(self, *args, **kwargs):
            seen["max_iters"] = self.max_iters
            return super().resolve_stage(*args, **kwargs)

    import shape.agent.verification.step_verifier as sv

    original = sv.StepVerifier
    sv.StepVerifier = Recording
    try:
        resolution = Verifier().resolve_managed_stage(stage_by_name("batch_correction"), lambda p: _result(),
                                                      {"harmony_params": dict(DEFAULT_CONFIG)}, _result(),
                                                      llm_engine=RuleEngine())
    finally:
        sv.StepVerifier = original
    assert seen["max_iters"] == 1 + BatchCorrectionVerifier.budget and resolution.result is None


def test_step_verifier_keep_best_excludes_inconclusive_trials():
    verifier = StepVerifier()

    class FailedObject:
        _attempt_status = "FAILED"

    failed = VLRRResult("PASS", 0.99, {"score": 0.99}, action="accept", judge_severity="none")
    ready = VLRRResult("WARNING", 0.40, {"score": 0.40}, action="retune_here", judge_severity="major")
    result = verifier._best(
        [({"harmony_params": {"theta": 2.0}}, failed, {"_attempt_status": "INCONCLUSIVE"}),
         ({"harmony_params": {"theta": 1.5}}, failed, {"_attempt_status": "FAILED"}),
         ({"harmony_params": {"theta": 1.0}}, failed, FailedObject()),
         ({"harmony_params": {"theta": 2.5}}, ready, {"_attempt_status": "READY"})],
        {}, "conceded")
    assert result.params["harmony_params"]["theta"] == 2.5
    mixed = verifier._best([({"harmony_params": {"theta": 1.0}}, failed, {}),
                            ({"harmony_params": {"theta": 2.5}}, ready, {"_attempt_status": "READY"})], {}, "conceded")
    assert mixed.params["harmony_params"]["theta"] == 2.5
    no_ready = verifier._best([({"harmony_params": {"theta": 1.0}}, failed, FailedObject())], {}, "satisfied")
    assert no_ready.result is None and no_ready.reason == "no_ready_trials"


def test_verifier_ignores_ground_truth_shaped_fields():
    native = _result()
    native["gt_free_metrics"].update({"label_accuracy": 0.9, "compound_map": 0.7})
    native["ground_truth_labels"] = ["bad"] * 4
    adapted, context = adapt_batch_correction(native)
    verdict = BatchCorrectionVerifier().verify(adapted, {**context, "trial_index": 1})
    assert set(verdict.metrics) <= {"n_rows", "k", "batch_mixing_input", "batch_mixing_output", "centroid_ratio_input",
                                    "centroid_ratio_output", "within_batch_neighbour_overlap", "effective_rank_ratio",
                                    "displacement"}
    assert "label_accuracy" not in verdict.metrics and "compound_map" not in verdict.metrics


def _design():
    """Two sources; compounds a, b in both, c in one; DMSO wells in both; one positive control."""
    meta = pd.DataFrame({
        "Metadata_JCP2022": ["a", "a", "b", "b", "c", "DMSO", "DMSO", "DMSO", "DMSO", "p"],
        "Metadata_PertType": ["trt"] * 5 + ["negcon"] * 4 + ["poscon"]})
    batch = np.array(["s1", "s2", "s1", "s2", "s1", "s1", "s2", "s1", "s2", "s2", "s1", "s2"])   # 12 rows
    rows = np.arange(10)                                  # rows 10, 11: evaluation compounds, not in the view
    features = np.random.default_rng(0).normal(size=(12, 3))
    return meta, batch, rows, features


def test_admission_evidence_counts_the_calibration_design():
    meta, batch, rows, features = _design()
    view = CalibrationView(rows=rows, meta=meta, batch_key="Metadata_Source")
    task = BatchTask(batch=batch, features=features, calibration=view, batch_column="source")
    evidence = admission_evidence(task, {"detected": True})
    assert evidence["n_rows"] == 12 and evidence["n_features"] == 3 and evidence["n_batch_groups"] == 2
    assert evidence["reference_groups"] == 3 and evidence["reference_groups_in_2plus_batch_groups"] == 2
    assert evidence["reference_groups_in_all_batch_groups"] == 2
    assert evidence["negative_controls_per_batch_group"] == {"s1": 2, "s2": 2}
    assert evidence["positive_controls_per_batch_group"] == {"s2": 1}
    assert evidence["gate_negative_controls_only"] is not None       # 4 DMSO wells >= 2 per batch group
    assert "a" not in str(evidence["rows_per_batch_group"])          # counts only, no reference identities


def test_the_calibration_view_scores_its_rows_at_float32(monkeypatch):
    from shape.tools.batch_corrector import quant

    seen = {}

    def composite_all(meta, feats, label_key, batch_key, plate_types, n_neighbors=25):
        seen.update(rows=len(meta), dtype=feats.dtype, first=feats[0, 0], label=label_key, batch=batch_key)
        return {"overall": 0.5}

    monkeypatch.setattr(quant, "composite_all", composite_all)
    meta, _, rows, _ = _design()
    matrix = np.full((12, 2), 1 / 3, dtype=np.float64)
    view = CalibrationView(rows=rows, meta=meta, batch_key="Metadata_Source", plate_types=("TARGET2",))
    assert view.score(matrix) == {"overall": 0.5}
    assert seen["rows"] == 10 and seen["dtype"] == np.float64 and seen["first"] == np.float64(np.float32(1 / 3))
    assert seen["label"] == "Metadata_JCP2022" and seen["batch"] == "Metadata_Source"


def test_trial_0_is_the_gate_and_the_readouts_of_the_uncorrected_input():
    rng = np.random.default_rng(0)
    batch = np.repeat(["s1", "s2"], 40)
    features = rng.normal(size=(80, 5)) + np.where(batch == "s1", 0.0, 3.0)[:, None]     # a batch offset
    result = uncorrected_result(features, batch)
    assert result["status"] == "UNCORRECTED_INPUT" and result["corrected_cls_output"] is features
    assert result["batch_effect"]["detected"] is True and result["provenance"]["input_shape"] == [80, 5]
    readouts = result["gt_free_metrics"]
    assert readouts["batch_mixing_input"] == readouts["batch_mixing_output"] and readouts["displacement"] == 0.0
    assert readouts["within_batch_neighbour_overlap"] == 1.0 and readouts["effective_rank_ratio"] == 1.0
