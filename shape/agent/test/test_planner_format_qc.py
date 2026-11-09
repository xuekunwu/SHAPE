"""Task 032 (VRR): D2 non-LLM assertion -- `Planner._format_qc` renders the
`conceded`/`fault`/`upstream_target` vocabulary correctly. This is deterministic,
constructed-dict rendering (no LLM/engine needed), covering the path the LLM-gated
regression cannot: a correct REMAP of the digest, not just a grep-clean removal of
the old action-string tokens.

Co-located with the module under test (committed -- AGENTS.md sec 6 P5), not in the
gitignored `tests/` dir. `Planner._format_qc` reads no instance state, so it is
called unbound (no LLM engine construction needed for this unit test).
"""

from shape.agent.planner import Planner

_format_qc = Planner._format_qc


def test_conceded_upstream_renders_target_and_a_rerun_cue():
    verification = {
        "status": "conceded",
        "upstream_target": "image_loading",
        "stages": [
            {"stage": "segmentation", "decision": "FAIL", "action": "conceded",
            "fault": "upstream", "notes": "High flow_error traced to poor input quality.",
            "metrics": {"flow_error": 0.9}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "upstream_target=image_loading" in digest
    # a re-run cue, not just the bare fact -- deterministic, not LLM-authored.
    assert "re-run" in digest.lower()
    assert "segmentation=FAIL/conceded" in digest
    assert "overall=conceded" in digest


def test_conceded_ceiling_renders_no_upstream_target_or_rerun_cue():
    verification = {
        "status": "conceded",
        "upstream_target": None,
        "stages": [
            {"stage": "morphological_state", "decision": "WARNING", "action": "conceded",
            "fault": "ceiling", "notes": "group effect fails thresholds.", "metrics": {}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "upstream_target=" not in digest
    assert "re-run" not in digest.lower()
    assert "morphological_state=WARNING/conceded" in digest


def test_ceiling_concede_stage_action_stays_retune_here_no_false_escalation_cue():
    """A tuning-ceiling concede (D1) never sets the per-stage action to "conceded" --
    that action is reserved for a VERIFIER-emitted upstream/ceiling fault. The digest
    must not fabricate an upstream_target/re-run cue for a plain ceiling."""
    verification = {
        "status": "conceded",
        "upstream_target": None,
        "stages": [
            {"stage": "segmentation", "decision": "WARNING", "action": "retune_here",
            "fault": None, "notes": "", "metrics": {}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "upstream_target=" not in digest
    assert "re-run" not in digest.lower()
    assert "overall=conceded" in digest    # still visible -- the LLM sees SOMETHING happened
    # the per-stage segment is NOT flagged "concerning" by action alone (decision is
    # only WARNING and action stays retune_here) -- matches pre-migration behavior
    # (a plain retune_here WARNING was never treated as an escalation either).
    assert "metrics=" not in digest


def test_exclude_samples_status_is_not_treated_as_conceded():
    verification = {
        "status": "exclude_samples",
        "upstream_target": None,
        "stages": [
            {"stage": "image_loading", "decision": "WARNING", "action": "exclude_samples",
            "fault": None, "notes": "Drop the few unrecoverable images, proceed.",
            "metrics": {}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "overall=exclude_samples" in digest
    assert "image_loading=WARNING/exclude_samples" in digest
    assert "upstream_target=" not in digest


def test_satisfied_status_renders_no_overall_line():
    verification = {
        "status": "satisfied",
        "stages": [{"stage": "segmentation", "decision": "PASS", "action": "accept",
                   "fault": None, "notes": "", "metrics": {}}],
    }
    digest = _format_qc(None, verification)
    assert "overall=" not in digest


# -- 045 Phase 3 Fix 4: the reconcile lever ledger in the digest ------------------

def test_concerning_stage_surfaces_the_lever_ledger():
    """A conceded stage surfaces which retunes were inert vs effective -- the planner
    now SEES 're-tuning isn't helping' as evidence (was computed, then dropped)."""
    verification = {
        "status": "conceded", "upstream_target": None,
        "stages": [
            {"stage": "segmentation", "decision": "FAIL", "action": "conceded",
             "fault": "ceiling", "notes": "faint cells missed.",
             "ledger": {"cellprob_threshold": "inert", "augment": "inert"},
             "metrics": {"cell_count": 40}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "ledger=" in digest
    assert "inert" in digest


def test_ledger_omits_uninformative_untried_levers():
    verification = {
        "status": "conceded", "upstream_target": None,
        "stages": [
            {"stage": "segmentation", "decision": "FAIL", "action": "conceded",
             "fault": "ceiling", "notes": "",
             "ledger": {"cellprob_threshold": "untried"}, "metrics": {}},
        ],
    }
    digest = _format_qc(None, verification)
    assert "ledger=" not in digest        # only 'untried' -> nothing informative


def test_non_concerning_accept_renders_no_ledger():
    verification = {
        "status": "satisfied",
        "stages": [{"stage": "segmentation", "decision": "PASS", "action": "accept",
                    "fault": None, "notes": "",
                    "ledger": {"diameter": "effective"}, "metrics": {}}],
    }
    digest = _format_qc(None, verification)
    assert "ledger=" not in digest
