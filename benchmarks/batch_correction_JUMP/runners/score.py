"""Final score of one SHAPE-arm run on one scenario.

submission/decision.json is validated. RUN: the submitted call's stored matrix; SKIP / ABSTAIN: the
uncorrected input. That representation is
scored like the control arms (raw, harmony, scanorama) on the test view (primary; plate-half
band), the all-wells view and the calibration view. The decision and claims are reported next to the
evaluator's admission evidence; no pass/fail margins.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BAND_KEYS = ("overall", "batch_mean", "bio_mean", "nonrep_mean_map", "negcon_mean_map")
DECISIONS = ("RUN", "SKIP", "ABSTAIN")
# claim type -> (comparison, metric) whose sign on the test view the claim asserts
CLAIM_CHECKS = {"batch_reduced": ("agent_vs_raw", "batch_mean"),
                "biology_preserved": ("agent_vs_raw", "bio_mean"),
                "replicate_retrieval_improved": ("agent_vs_raw", "nonrep_mean_map")}


def jsonl(path: Path) -> list[dict]:
    out = []
    for line in (path.read_text(encoding="utf-8", errors="replace").split("\n") if path.exists() else []):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def resolve_decision(run: Path) -> dict:
    path = run / "submission" / "decision.json"
    if not path.exists():
        raise SystemExit("NO_SUBMISSION: submission/decision.json is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        decision = payload.get("decision")
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
        from_call, record = payload.get("from_call"), None
        if decision == "RUN":
            ends = {r["call"]: r for r in jsonl(run / "logs" / "tool_calls.jsonl")
                    if r.get("event") == "end" and r.get("status") == "done"}
            starts = {r["call"]: r for r in jsonl(run / "logs" / "tool_calls.jsonl") if r.get("event") == "start"}
            if from_call not in ends:
                raise ValueError(f"RUN needs from_call = a finished call; finished: {sorted(ends)}")
            record = {**starts[from_call], **ends[from_call]}
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"INVALID_SUBMISSION: {error}") from None
    return {"decision": decision, "from_call": from_call, "record": record, "submitted": payload,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_claims(claims: list, bands: dict, evidence: dict) -> list[dict]:
    """Annotate each claim with the evaluator evidence it can be checked against (None = not checked)."""
    out = []
    for claim in claims:
        ctype = claim.get("type") if isinstance(claim, dict) else None
        entry = {"claim": claim, "consistent": None, "harness_evidence": None}
        if ctype in CLAIM_CHECKS:
            comparison, key = CLAIM_CHECKS[ctype]
            band = bands.get(comparison, {}).get(key, {})
            if band.get("delta") is not None:
                lower = band["delta"] - (band.get("band_2se") or 0.0)
                upper = band["delta"] + (band.get("band_2se") or 0.0)
                entry["harness_evidence"] = {"comparison": comparison, "metric": key, **band}
                entry["consistent"] = (upper >= 0) if ctype == "biology_preserved" else (lower > 0)
        elif ctype == "batch_effect_present":
            gate = evidence.get("gate_all_wells") or {}
            entry["harness_evidence"] = {"gate_all_wells": gate}
            entry["consistent"] = gate.get("detected")
        elif ctype == "separable_from_biology":
            entry["harness_evidence"] = {k: evidence.get(k) for k in
                                         ("n_groups", "compounds_total", "compounds_in_all_groups",
                                          "compounds_in_one_group_only")}
        out.append(entry)
    return out


def usage(run: Path) -> dict:
    ledger = jsonl(run / "logs" / "tool_calls.jsonl")
    ends = [e for e in ledger if e.get("event") == "end"]
    scored = [e for e in ends if (e.get("val") or {}).get("overall") is not None]
    best = max(scored, key=lambda e: e["val"]["overall"]) if scored else None
    labels = [v.get("label") for e in ends for v in (e.get("reconciliation") or {}).values()]
    meta = json.loads((run / "RUN_META.json").read_text(encoding="utf-8")) if (run / "RUN_META.json").exists() else {}
    agent = json.loads((run / "AGENT_RUN.json").read_text(encoding="utf-8")) if (run / "AGENT_RUN.json").exists() else {}
    return {"max_rounds": meta.get("max_rounds"),
            "tool_calls_started": sum(1 for e in ledger if e.get("event") == "start"),
            "tool_calls_finished": sum(1 for e in ends if e.get("status") == "done"),
            "tool_statuses": {str(s): sum(1 for e in ends if e.get("tool_status", e.get("status")) == s)
                              for s in {e.get("tool_status", e.get("status")) for e in ends}},
            "expectations": {lab: labels.count(lab) for lab in sorted({str(x) for x in labels})},
            "best_calibration_overall": best and {"value": best["val"]["overall"], "call": best["call"]},
            "tokens": agent.get("tokens")}


def _f(value, spec=".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def final(run: Path, prep: Path) -> dict:
    from shape.tools.batch_corrector import quant
    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG

    from benchmarks.batch_correction_JUMP.runners.control import CONTROL_DIR, provenance, score_views
    from benchmarks.batch_correction_JUMP.runners.prepare import load, sha256_file

    summary_prep, meta, feats = load(prep)
    batch_key, plate_types = summary_prep["scenario"]["batch_key"], summary_prep["scenario"]["plate_types"]
    control_dir = prep / CONTROL_DIR
    control = json.loads((control_dir / "CONTROL.json").read_text(encoding="utf-8"))
    tampered = [a for a, c in control["arms"].items()
                if c.get("array") and sha256_file(control_dir / c["array"]) != c["array_sha256"]]
    if tampered:
        raise SystemExit(f"INTEGRITY_FAILED: control arrays {tampered} changed since they were computed")
    stale = {k: (control["provenance"].get(k), v) for k, v in provenance().items() if control["provenance"].get(k) != v}
    if stale:
        raise SystemExit(f"CONTROL_STALE: the control and the current code differ: {stale}")

    choice = resolve_decision(run)
    record = choice["record"] or {}
    if choice["decision"] == "RUN":
        path = Path(record.get("corrected") or "")
        agent = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        report = record.get("tool_report") or {}
        run_info = {"decision": "RUN", "status": record.get("tool_status"), "seconds": record.get("seconds"),
                    **{k: report.get(k) for k in ("batch_effect", "harmony", "telemetry", "preprocess")},
                    "array": str(path), "array_sha256": sha256_file(path)}
    else:
        agent = np.asarray(feats, dtype=np.float32)
        run_info = {"decision": choice["decision"], "status": "UNCORRECTED_INPUT"}
    config = record.get("config") if choice["decision"] == "RUN" else None

    halves = control["band"]["halves"]
    arms = {name: {k: c[k] for k in ("val", "test", "all", "test_halves")} for name, c in control["arms"].items()}
    arms["agent"] = score_views(meta, agent, halves, batch_key=batch_key, plate_types=plate_types)
    comparisons = {"agent_vs_harmony": ("harmony", "agent"), "agent_vs_raw": ("raw", "agent"),
                   "harmony_vs_raw": ("raw", "harmony"),
                   "agent_vs_scanorama": ("scanorama", "agent"),
                   "scanorama_vs_harmony": ("harmony", "scanorama")}
    bands = {name: {key: quant.paired_band(arms[ref]["test"], arms[new]["test"], arms[ref]["test_halves"],
                                           arms[new]["test_halves"], key) for key in BAND_KEYS}
             for name, (ref, new) in comparisons.items()}
    all_wells = {name: {key: (None if arms[ref]["all"].get(key) is None or arms[new]["all"].get(key) is None
                              else arms[new]["all"][key] - arms[ref]["all"][key]) for key in BAND_KEYS}
                 for name, (ref, new) in comparisons.items()}
    evidence = control.get("admission_evidence", {})
    claims = check_claims(choice["submitted"].get("claims") or [], bands, evidence)
    names = ["agent", "harmony", "scanorama", "raw"]
    rows = ["| representation | view | overall | batch_mean | bio_mean | nonrep_mAP | negcon_mAP | skipped |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        for view in ("test", "all", "val"):
            s = arms[name][view]
            rows.append(f"| {name} | {view} | " + " | ".join(_f(s.get(k)) for k in BAND_KEYS)
                        + f" | {', '.join(sorted(s.get('skipped', {}))) or '-'} |")
    rows += ["", "| comparison | metric | test delta +- 2SE (plate halves) | all-wells delta |", "| --- | --- | --- | --- |"]
    for cname, per_key in bands.items():
        for key, b in per_key.items():
            rows.append(f"| {cname} | {key} | {_f(b['delta'], '+.4f')} +- {_f(b['band_2se'])} | "
                        f"{_f(all_wells[cname][key], '+.4f')} |")
    result = {
        "scenario": summary_prep["scenario"], "decision": choice["decision"],
        "submission": choice["submitted"], "submission_sha256": choice["sha256"], "chosen_config": config,
        "non_default_parameters": ({k: v for k, v in config.items() if DEFAULT_CONFIG.get(k) != v} if config else None),
        "tool_run": run_info, "provenance": control["provenance"],
        "scores": {n: {v: arms[n][v] for v in ("test", "all", "val")} for n in names},
        "comparisons_test": bands, "comparisons_all_wells": all_wells,
        "admission_evidence": evidence, "claims_checked": claims, "usage": usage(run),
        "from_call": choice["from_call"],
        "rule": "no pass/fail margins; test view = evaluation compounds + controls (primary), band = plate halves "
                "drawn within batch groups (2SE ~ 2 sd(half deltas)/sqrt(2)); all-wells view includes the "
                "calibration compounds the arm scored",
        "split_design": summary_prep["split_rule"], "markdown_table": "\n".join(rows),
    }
    (run / "RESULT.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    claim_lines = "\n".join(f"- {c['claim'].get('type') if isinstance(c['claim'], dict) else c['claim']}: "
                            f"consistent={c['consistent']}" for c in claims) or "- (none)"
    (run / "RESULT.md").write_text(
        f"# Result {summary_prep['scenario']['name']}: decision {choice['decision']} ({run_info['status']})\n\n"
        f"Submitted call: {choice['from_call']}; non-default parameters: {result['non_default_parameters']}\n\n"
        + result["markdown_table"]
        + f"\n\nClaims:\n{claim_lines}\n\nAdmission evidence (evaluator, all labels): "
          f"{json.dumps({k: v for k, v in evidence.items() if not isinstance(v, dict)}, default=str)}\n\n"
          f"Usage: {result['usage']}.\n", encoding="utf-8")
    print(result["markdown_table"])
    print(f"DECISION {choice['decision']}  status={run_info['status']}  usage={result['usage']}")
    return result
