"""Denominator-aware F3/tool efficiency. No fabricated utilization or grade."""
from collections import Counter
from evaluation_common import fraction


def classify_error(message):
    text = (message or "").lower()
    if "visual qc" in text:
        return "native_quality_gate"
    if "above" in text and ("attachment" in text or "json evidence" in text):
        return "evidence_size_limit"
    if any(word in text for word in ("keys must", "call spec", "non-empty json list")):
        return "request_schema"
    return "other_recorded_error"


def evaluate(events, decisions, expected=()):
    """Logical native attempts differ from model decisions and input records.

    A rejected run never admitted to a tool contributes to request rejection,
    not native execution failure. A selected uncorrected input is abstention.
    Selected-attempt share is descriptive, not causal search waste.
    """
    result = []
    keys = sorted({(e["arm"], e["stage"]) for e in events + decisions} | set(expected))
    for arm, stage in keys:
        native = [e for e in events if (e["arm"], e["stage"]) == (arm, stage) and e["is_tool_attempt"]]
        inputs = [e for e in events if (e["arm"], e["stage"]) == (arm, stage) and not e["is_tool_attempt"]]
        ds = [e for e in decisions if (e["arm"], e["stage"]) == (arm, stage)]
        runs = [d for d in ds if d["action"] == "run"]
        inspections = [d for d in ds if d["action"] == "inspect"]
        succeeded = sum(e["status"] == "complete" for e in native)
        errors = [e for e in native if e["status"] == "error"]
        selected = sum(e["selected"] for e in native)
        rejected = [d for d in ds if d["outcome"] == "rejected"]
        delivered_units = len({e["unit"] for e in native + inputs if e["selected"]})
        known = bool(native)  # Absence of a native ledger is not perfect reliability.
        result.append(dict(
            arm=arm, stage=stage,
            logical_attempts=len(native) if known else None,
            completed_attempts=succeeded if known else None,
            native_errors=len(errors) if known else None,
            native_error_classes=dict(Counter(classify_error(e["error"]) for e in errors)),
            tool_completion_rate=fraction(succeeded, len(native)),
            delivered_units=delivered_units or None,
            work_amplification=len(native) / delivered_units if known and delivered_units else None,
            selected_attempts=selected if known else None,
            selected_attempt_share=fraction(selected, len(native)),
            non_tool_input_records=len(inputs),
            decisions=len(ds) or None,
            rejected_requests=len(rejected) if ds else None,
            request_rejection_rate=fraction(len(rejected), len(ds)),
            rejection_classes=dict(Counter(classify_error(d["message"]) for d in rejected)),
            run_requests=len(runs) or None,
            run_admission_rate=fraction(sum(d["outcome"] == "executed" for d in runs), len(runs)),
            evidence_requests=len(inspections) or None,
            evidence_admission_rate=fraction(sum(d["outcome"] == "attached" for d in inspections), len(inspections)),
            decisions_with_reason=sum(bool(d["has_reason"]) for d in ds) if ds else None,
            reason_coverage=fraction(sum(bool(d["has_reason"]) for d in ds), len(ds)),
        ))
    return {
        "schema": "staining-failure-efficiency-v1",
        "unit": "logical native attempt, not GPU kernel or HTTP request",
        "rows": result,
        "assessment": {
            "execution_reliability": "measured where native ledgers exist; errors classified separately from QC state",
            "request_validity": "rejected/recorded decisions; run and evidence admission use their own denominators",
            "search_efficiency": "attempts/delivered units; selected share does not prove discarded search was useless",
            "traceability": "reason-bearing recorded controller decisions; not semantic correctness",
            "delivery": "456 image records per arm; one zero-object crop image each is reported, not automatically a failure",
            "system_utilization": "unavailable: no complete CPU/GPU busy integral, interval union, queue or memory telemetry",
            "recoverability": "external VM reboot interrupted trusted readout; observed resume 227.34 s on 16 CPU workers; no comparable recovery SLO",
            "fairness": "historical arms have different CPU allocations, decision budgets and cell populations; cost is descriptive",
            "overall_grade": None,
            "grade_reason": "no predeclared SLO, validated weights or complete common telemetry; no defensible A/B composite",
        },
    }
