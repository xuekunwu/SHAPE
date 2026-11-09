"""Shared benchmark runner utilities.

Extracted from the per-benchmark runner scripts (NeurIPS22-CellSeg,
MedSegBench, Inhouse_CellSeg, Inhouse_OrganoidSeg) — identical logic
that was copy-pasted across all segmentation-family benchmarks.

Provides:
  BaseRunConfig      — dataclass of common solver/CLI defaults
  build_argparser    — pre-loaded argparse.ArgumentParser with all shared flags
  aggregate_vlrr_activity — extract VLRR agentic-activity counts from verification_summary
  write_results_csv  — common CSV writing helper
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Shared configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaseRunConfig:
    """Common hyperparameters shared by all segmentation-family benchmark runners."""
    model: str = "gpt-5.6-luna"
    max_steps: int = 10
    output_root: Path = Path("solver_cache")
    data_root: Optional[Path] = None
    limit: Optional[int] = None
    seed: int = 0
    verify_refine: bool = False
    run_baseline: bool = True
    prompt_file: Optional[Path] = None


# ---------------------------------------------------------------------------
# Shared argparser
# ---------------------------------------------------------------------------

def build_argparser(description: str, default_output: str = "solver_cache") -> argparse.ArgumentParser:
    """Return an ArgumentParser pre-loaded with the flags shared by all
    segmentation-family benchmark runners.

    Callers add their own benchmark-specific flags after receiving the parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model", default="gpt-5.6-luna",
        help="LLM the agent uses (planner + vision verifier). Default: gpt-5.6-luna.")
    parser.add_argument(
        "--max-steps", type=int, default=10,
        help="Maximum agent steps per image (default 10).")
    parser.add_argument(
        "--output-root", "--out-root", dest="output_root", type=Path,
        default=Path(default_output),
        help="Directory for all outputs: masks, CSV, solver_cache.")
    parser.add_argument(
        "--data-root", type=Path,
        default=(Path(os.environ["SHAPE_BENCH_DATA_ROOT"])
                 if os.environ.get("SHAPE_BENCH_DATA_ROOT") else None),
        help="Dataset root. Defaults to $SHAPE_BENCH_DATA_ROOT/<benchmark>.")
    parser.add_argument(
        "--limit", type=int,
        help="Hard cap on the number of images to run (applied after any sampling).")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for reproducible random sampling (default 0).")
    parser.add_argument(
        "--prompt-file", type=Path,
        help="Path to a text file with the agent's query prompt. "
             "Default: each benchmark reads its own Prompt.txt at runtime.")
    parser.add_argument(
        "--baseline", action=argparse.BooleanOptionalAction, default=True,
        help="Also run the direct tool baseline (no agent) and compare. "
             "Use --no-baseline to skip.")
    parser.add_argument(
        "--verify-refine", action=argparse.BooleanOptionalAction, default=False,
        help="Enable segmenter Verify refinement (scale_mismatch evidence, "
             "vision prompt flow_error filter). Default off.")
    return parser


# ---------------------------------------------------------------------------
# VLRR activity aggregation (previously copy-pasted in each runner)
# ---------------------------------------------------------------------------

def _iter_stages(verification_summary):
    """Yield every stage dict across a flat or per_image verification_summary."""
    for record in verification_summary or []:
        if not isinstance(record, dict):
            continue
        groups = (record.get("per_image") if isinstance(record.get("per_image"), list)
                  else [record])
        for grp in groups:
            if isinstance(grp, dict):
                for stage in (grp.get("stages") or []):
                    yield stage


def aggregate_vlrr_activity(verification_summary: list) -> dict:
    """Aggregate the agent's VLRR activity from solve()'s verification_summary.

    Returns a dict with keys:
      records              — number of verification records
      segmentation_retuned — images where the segmenter ran >1 trial
      vision_judged        — stages where the vision judge fired
      escalations          — stages that escalated

    Extracted verbatim from run_shape_neurips_cellseg._summarize_agent_activity
    (identical logic was present in NeurIPS22, MedSegBench, and Inhouse runners).
    """
    activity: dict = {
        "records": len(verification_summary or []),
        "segmentation_retuned": 0,
        "vision_judged": 0,
        "escalations": 0,
    }
    for stage in _iter_stages(verification_summary):
        trials = stage.get("trials") or []
        if stage.get("stage") == "segmentation" and (stage.get("n_trials") or 0) > 1:
            activity["segmentation_retuned"] += 1
        if any("[vision]" in str(t.get("notes", "")) for t in trials):
            activity["vision_judged"] += 1
        action = stage.get("action")
        escalated = isinstance(action, str) and action.startswith("escalate")
        if escalated:
            activity["escalations"] += 1
    return activity


def agentic_record(verification_summary: list, stage: str) -> dict:
    """One image's agentic VRR record of `stage` (the wave-mode `details` of the verification record,
    `shape.agent.verification.base.AgenticVerifier`): how often the stage was verified (the planner may
    invoke the tool again), and for the last invocation the admission (its decision, its rule's outcome
    and the severity the agent answered -- None when it was not asked or gave no answer), the rounds, the
    tool calls, the verifier's stop, the final choice, the reasoner failures and the rounds the fallback
    planned."""
    stages = [s for s in _iter_stages(verification_summary) if s.get("stage") == stage]
    last = (stages[-1].get("details") or {}) if stages else {}
    rounds = last.get("rounds") or []
    admission = last.get("admission") or {}
    return {
        "invocations": len(stages),
        "admission": admission.get("decision"),
        "admission_rule": admission.get("rule"),
        "admission_severity": (admission.get("answer") or {}).get("severity"),
        "rounds": len(rounds) or None,
        "calls": sum(len(r) for r in rounds) or None,
        "stop": last.get("verifier_stop") or last.get("stop_reason"),
        "choice": last.get("choice"),
        "reasoner_failures": last.get("reasoner_failures"),
        "fallback_rounds": last.get("fallback_rounds"),
    }


# ---------------------------------------------------------------------------
# CSV writing helper
# ---------------------------------------------------------------------------

def write_results_csv(rows: list[dict], fields: list[str], output_path: Path) -> None:
    """Write a list of result dicts to a CSV file.

    Only the keys listed in `fields` are written; missing keys are written as
    empty strings. Extracted from the identical pattern in each segmentation runner.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
