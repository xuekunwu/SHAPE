"""Validate one completed six-arm scenario run before writing COMPLETE."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.batch_correction_JUMP.runners import benchmark_contract as contract


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--scenario", default=contract.PILOT_SCENARIO)
    parser.add_argument("--model", default=contract.LUNA_MODEL)
    args = parser.parse_args()

    root = Path(args.run)
    report = read(root / "BENCHMARK_REPORT.json")
    expected_roster = contract.arm_ids(include_raw=True)
    if report.get("git_commit") != args.expected_commit:
        raise RuntimeError("report commit mismatch")
    if report.get("scenario") != args.scenario or report.get("model") != args.model:
        raise RuntimeError("report scenario/model mismatch")
    if tuple(report.get("roster", ())) != expected_roster:
        raise RuntimeError("report roster mismatch")
    rows = {row["arm_id"]: row for row in report.get("rows", [])}
    if tuple(rows) != expected_roster:
        raise RuntimeError("report rows are missing or reordered")
    expected_calls = {"raw": 0, "harmony": 1, "scanorama": 1, "gpt_luna": 0}
    for arm_id, value in expected_calls.items():
        if rows[arm_id]["tool_calls"] != value:
            raise RuntimeError(f"{arm_id} tool-call count mismatch")
    react_dir = root / "react_luna_harmony"      # delivering the input without a tool call is the agent's choice
    react = read(react_dir / "ARM_OUTPUT.json")
    unit = react_dir / "react" / str(react.get("scenario"))
    for path in (react_dir / "react" / "REACT_SUMMARY.json", unit / "TRACE.jsonl", unit / "CALLS.json",
                 unit / "SELECTED.json", Path(react.get("array") or react_dir / "corrected.npy")):
        if not path.is_file():
            raise RuntimeError(f"ReAct record missing: {path}")
    selected = read(unit / "SELECTED.json")
    if selected.get("call_id") != react.get("selected_call"):
        raise RuntimeError("ReAct SELECTED.json disagrees with ARM_OUTPUT.json")
    if rows["react_luna_harmony"]["tool_calls"] != react.get("tool_calls"):
        raise RuntimeError("ReAct tool-call count disagrees with ARM_OUTPUT.json")
    for arm_id in ("gpt_luna", "react_luna_harmony", "shape_luna_harmony"):
        if rows[arm_id]["tokens"] <= 0:
            raise RuntimeError(f"{arm_id} has non-positive token usage")
    for arm_id in ("gpt_luna", "react_luna_harmony"):
        arm = read(root / arm_id / "ARM_OUTPUT.json")
        if arm.get("git_commit") != args.expected_commit or arm.get("model") != args.model:
            raise RuntimeError(f"{arm_id} provenance mismatch")
    shape_meta = read(root / "shape_luna_harmony" / "RUN_META.json")
    if shape_meta.get("commit") != args.expected_commit or shape_meta.get("model") != args.model:
        raise RuntimeError("SHAPE provenance mismatch")
    (root / "COMPLETE").write_text("VALID_RUN=OK\n", encoding="utf-8")
    print("VALID_RUN=OK")
    print(f"RUN_ROOT={root}")
    print(f"REPORT={root / 'BENCHMARK_REPORT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
