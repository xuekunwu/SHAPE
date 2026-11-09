"""A managed tool call that raised is a failed call (`schemas.is_failed_result`): `Executor.run_managed_tool`
returns `{"error": ..., "tool_name": ...}` on an exception, which counts as failed, while a tool result that
carries outputs next to an "error" field does not (a verifier reads that error itself). On the solve() path a
planner-chosen Batch_Corrector_Tool call without its inputs therefore ends as a failed call, not as
"satisfied / no_batch_effect"."""
from __future__ import annotations

from shape.agent.executor import Executor
from shape.agent.verification.integration import build_base_kwargs, verify_managed_tool
from shape.agent.verification.stage_map import default_params_for_tool
from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier
from shape.utils.schemas import VLRRResult, WaveState, is_failed_result


def _executor(tmp_path) -> Executor:
    executor = Executor("unused-model", query_cache_dir=str(tmp_path))
    executor.enable_llm_verifier = False
    return executor


def test_the_executors_exception_result_is_a_failed_call(tmp_path):
    result = _executor(tmp_path).run_managed_tool("Batch_Corrector_Tool", {}, {})
    assert set(result) == {"error", "tool_name"} and result["tool_name"] == "Batch_Corrector_Tool"
    assert is_failed_result(result)
    assert is_failed_result({"error": "boom"})
    assert is_failed_result({"_attempt_status": "FAILED", "error": "RuntimeError: boom"})


def test_a_result_with_outputs_next_to_an_error_is_not_failed_and_the_seg_verifier_still_reads_its_error():
    result = {"error": "overlay QC failed", "mask_path": "m.npy", "cell_count": 3, "tool_name": "Cell_Segmenter_Tool"}
    assert not is_failed_result(result)
    assert not is_failed_result({"cell_count": 3, "mask_path": "m.npy"})
    vres = VLRRResult("PASS", 0.0, {})
    state = WaveState(trials=[({}, vres, {"status": "input"}), ({}, vres, result)], parents=[None, 0],
                      entries=[None, None], remaining=1, context={})
    assert SegmentationAgenticVerifier().call_error(1, state) == "overlay QC failed"


def test_a_batch_call_without_its_inputs_ends_as_a_failed_call(tmp_path):
    """The audit's probe: the solve() path hands the batch tool no inputs (`build_base_kwargs` has no batch
    branch), the call raises in the executor, and the stage concedes on the failed call."""
    spec = {"path": str(tmp_path / "img1.png"), "image_id": "img1", "group": "default"}
    kwargs = build_base_kwargs("Batch_Corrector_Tool", spec, memory=None)
    resolutions, result, status = verify_managed_tool(
        _executor(tmp_path), "Batch_Corrector_Tool", kwargs, default_params_for_tool("Batch_Corrector_Tool"), {})
    res = resolutions[-1]
    assert (status, res.reason, res.fault) == ("conceded", "initial_trial_failed", "ceiling")
    assert result is None and res.result is None
    _, trial0, failed_result = res.trials[0]
    assert is_failed_result(failed_result) and "run_managed_tool failed" in failed_result["error"]
    assert trial0.decision == "FAIL" and trial0.triggered == ["tool_error"]
