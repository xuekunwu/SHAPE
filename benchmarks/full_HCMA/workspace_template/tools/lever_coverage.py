"""Lever coverage of this run: which lever groups (see each tool's --describe) a completed tool call has
changed, or which submission/choice.json gives a reason for under "skipped".

    python tools/lever_coverage.py            # lever coverage only (fast)
    python tools/lever_coverage.py --chain    # the full FINAL check: coverage and submission/choice.json's chain

FINAL is accepted only when every group is tested or has a reason AND the chain of choice.json verifies.
The chain check re-hashes every file of the chain, every crop file of the embed call included, so on a
large run it takes minutes; the default run does not do it.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

NOTE = ("FINAL acceptance also verifies submission/choice.json's chain (every file's sha256, every crop file "
        "of the embed call included; minutes on a large run): python tools/lever_coverage.py --chain")


def lever_coverage(cov, run: Path) -> str:
    """The coverage part of `final_check` (the same groups, counting and skip reasons), without the chain."""
    groups = cov.lever_groups(run)
    tested = cov.coverage(cov.calls(run), groups)          # the ledger's first start / end line per call
    given = (cov.read_choice(run) or {}).get("skipped")
    reasons = {k: r.strip() for k, r in given.items() if isinstance(r, str) and r.strip()} \
        if isinstance(given, dict) else {}
    lines = ["lever coverage (a group counts when a done or partial call changed it, or submission/choice.json "
             "'skipped' gives a reason; the chain is not checked here):"]
    missing = []
    for group, calls in tested.items():
        reason = reasons.get(group) or reasons.get(group.split(".", 1)[0])
        if calls:
            lines.append(f"  {group}: tested in {', '.join(calls)}")
        elif reason:
            lines.append(f"  {group}: untested, skipped: {reason}")
        else:
            lines.append(f"  {group}: UNTESTED")
            missing.append(group)
    lines.append(f"untested without a reason: {', '.join(missing)}" if missing
                 else "every group is tested or has a reason")
    lines.append(NOTE)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lever_coverage.py", description="Lever coverage of this run (--chain: the full FINAL check)")
    ap.add_argument("--chain", action="store_true", help="also verify submission/choice.json's chain (slow)")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    cfg = C.load_config()
    if cfg["repo"] not in sys.path:
        sys.path.insert(0, cfg["repo"])
    cov = importlib.import_module("benchmarks.full_HCMA.harness.coverage")

    run = Path(cfg["run"])
    print(cov.describe(cov.final_check(run)) if args.chain else lever_coverage(cov, run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
