<!--
Reconcile has no LLM prompt for this tool (or any tool) today. Per vrr.md sec 3
("Reconcile is the verifier's measurement, reused -- not a new model") and this
codebase's existing design (see `base.py`'s "the LLM only enters at the Recommend
phase" rule), Reconcile stays a deterministic function: it diffs the quantitative
Verify signal's BEFORE/AFTER against the last Recommend's `expected_effect` and
classifies the lever as effective / no-headroom / inert. See
`shape.agent.verification.base.classify_lever` for the current rule.

This file is a placeholder for the convention (`{verify,reconcile,recommend}.md`
per tool) -- it exists so the convention is documented and complete, not because an
LLM call reads it. If a tool ever needs an LLM-driven Reconcile (e.g. an ambiguous
quantitative signal that benefits from vision judgment), its prompt belongs here.
-->
