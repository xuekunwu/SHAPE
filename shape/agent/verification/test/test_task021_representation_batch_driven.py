"""Task 032 (VRR) whole-loop regression: task 021's `batch_driven` floor-lever
behavior in `RepresentationVerifier` is preserved. (021's own archived test files
were lost -- they lived in the gitignored `tests/` of a now-removed worktree; per
CODING_LOG.md this task recovers them by reconstructing the documented behavior
against the current tree, the same "026 approach" used for 018/020 above.)

021's core fix: a batch-vs-group silhouette confound (`locus="upstream"`,
`triggered=["batch_driven"]`) is NOT a terminal escalation -- it is EVIDENCE that
falls through to the shared LLM tail, same as every other locus, so the loop reasons
input-features -> best lever (`field_aware` when the source id is reliable,
`augment_color` otherwise) instead of a hard-coded route. This is UNRELATED to task
032's emit-site migration (only the "collapse" locus in this verifier concedes now);
this file proves that untouched mechanism still works under the new tree.
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Literal

from shape.agent.verification.llm_recommend import RepresentationVerdict
from shape.agent.verification.verifiers.representation import RepresentationVerifier
from shape.utils.schemas import VLRRResult


class _StubEngine:
    """No-API stand-in for the multimodal engine: returns a fixed, schema-valid
    verdict that CONFIRMS the deterministic floor lever (severity="major", same
    param/direction) -- exercising the full `_recommend_from_card` tail, not just
    the deterministic branch."""

    def __init__(self, param: str, direction: str):
        self._param, self._direction = param, direction

    def generate(self, content, response_format=None, **kwargs):
        verdict = RepresentationVerdict(
            specimen="HeLa cells", cell_size="medium", cell_density="moderate",
            image_usable="yes", problem="source/batch confound", severity="major",
            param=self._param, direction=self._direction, confidence="high",
            reasoning="silhouette_by_batch clearly exceeds silhouette_by_group",
        )
        return {"content": verdict, "usage": {"prompt_tokens": 500}}


_METRICS = {
    "effective_rank": 50.0, "feature_variance_mean": 1.0, "PC1_area_correlation": 0.1,
    "silhouette_by_batch": 0.5, "silhouette_by_group": 0.1, "nn_visual_consistency": 0.9,
}


def test_batch_driven_reliable_source_picks_field_aware_lever():
    verifier = RepresentationVerifier()
    verifier.llm_engine = _StubEngine(param="field_aware", direction="on")
    context = {"report": {
        "scale_and_balance": {
            "metadata_availability": {"well_id": "A1"},
            "median_crops_per_source": 5,
        }
    }}
    out = verifier.adjudicate(_METRICS, "upstream", ["batch_driven"], context)
    assert out.action == "retune_here"
    assert out.param_delta == {"field_aware": True}
    assert out.expected_effect == {"silhouette_by_batch": "down"}


def test_batch_driven_unreliable_source_picks_augment_color_lever():
    verifier = RepresentationVerifier()
    verifier.llm_engine = _StubEngine(param="augment_color", direction="on")
    context = {"report": {}}   # no metadata -> _source_reliable() is conservatively False
    out = verifier.adjudicate(_METRICS, "upstream", ["batch_driven"], context)
    assert out.action == "retune_here"
    assert out.param_delta == {"augment_color": True}
    assert out.expected_effect == {"silhouette_by_batch": "down"}


def test_batch_driven_is_evidence_not_a_terminal_concede():
    """batch_driven falls through to the shared LLM tail -- it never returns a
    `conceded` action directly (unlike "collapse", which is a mandatory concede)."""
    verifier = RepresentationVerifier()   # no llm_engine wired -> deterministic only
    verifier.uses_llm = False   # the deterministic floor: no vision judge
    context = {"report": {}}
    out = verifier.adjudicate(_METRICS, "upstream", ["batch_driven"], context)
    assert out.action == "retune_here"
    assert out.fault is None


def test_collapse_locus_still_concedes_upstream_after_migration():
    """The ONE locus in this verifier that DOES concede (task 032's emit-site
    migration target, BRIEF sec B) -- confirms it still fires correctly."""
    verifier = RepresentationVerifier()
    metrics = dict(_METRICS, effective_rank=1.0, feature_variance_mean=1e-6)
    out = verifier.adjudicate(metrics, "upstream", ["collapse"], {})
    assert out.action == "conceded"
    assert out.fault == "upstream"
    assert out.upstream_target == "image_loading"
