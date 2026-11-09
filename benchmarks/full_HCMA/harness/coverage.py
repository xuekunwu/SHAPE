"""Lever coverage -- a rule shared by every agentic tuning run.

A run may end early (submission/FINAL) only once every lever group has been changed by at least
one completed call (`done` / `partial`), or submission/choice.json gives a reason for it under
`"skipped"`; FINAL also needs a valid choice (ledger.chain). The deadline ends a run in any case.

The groups come from the tools' own lever registries (`tool_levers` of the repository template's
`tools/_common.py`, never the run's agent-writable copy), one per
lever: `segment.<lever>`, `crop.<lever>`, `embed.<lever>`, `batch_correct.<lever>`,
`cluster.<lever>`, plus `crop.cell_overrides` (the cropper's per-cell rules). A skip key may also
be a whole tool (`"batch_correct": "why"`), which covers all its groups. A call changes a lever
when it sets it to a value other than the tool default, globally, per image or per cell; the tool
records these names as `set_levers` in its ledger `start` line.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from pathlib import Path

from .ledger import COMPLETED, TEMPLATE_TOOLS, calls, chain

REPO = Path(__file__).resolve().parents[3]
PER_CELL_GROUP = "crop.cell_overrides"
_TEMPLATE_MODULE = "_full_hcma_template_common"
# One load at a time: the module is in sys.modules before its code has run (its dataclasses need that), so a
# second thread (a threaded crop stage) would otherwise get it half-initialised.
_TEMPLATE_LOCK = threading.Lock()


def _registry(tools_dir: Path = TEMPLATE_TOOLS):
    """(TOOLS, tool_levers) of the template's `_common.py`, loaded under its own module name (a
    workspace `_common` already imported in this process is not reused)."""
    with _TEMPLATE_LOCK:
        module = sys.modules.get(_TEMPLATE_MODULE)
        if module is None:
            spec = importlib.util.spec_from_file_location(_TEMPLATE_MODULE, Path(tools_dir) / "_common.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_TEMPLATE_MODULE] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[_TEMPLATE_MODULE]
                raise
    return module.TOOLS, module.tool_levers


def lever_groups(run: Path) -> list[str]:
    """Every lever group, in tool order, from the template's registry (CARDs of this repository)."""
    tools, tool_levers = _registry(TEMPLATE_TOOLS)
    groups = []
    for tool in tools:
        groups += [f"{tool}.{name}" for name in tool_levers(tool, REPO)]
        if tool == "crop":
            groups.append(PER_CELL_GROUP)
    return list(dict.fromkeys(groups))


def coverage(records: dict[str, dict], groups: list[str]) -> dict[str, list[str]]:
    """{group: completed calls that changed a lever of that group} (every group present), from
    ledger.calls(): the first start and the first end line of each call, as for the chain."""
    tested: dict[str, list[str]] = {g: [] for g in groups}
    for call, record in records.items():
        e = record["start"]
        if e is None or (record["end"] or {}).get("status") not in COMPLETED:
            continue
        levers = e.get("set_levers")
        if levers is None:
            levers = list(e.get("set") or {})
        for lever in levers:
            group = f"{e.get('tool')}.{lever}"
            if group in tested and call not in tested[group]:
                tested[group].append(call)
    return tested


def read_choice(run: Path) -> dict | None:
    """submission/choice.json, None when it is missing or not a JSON object."""
    path = Path(run) / "workspace" / "submission" / "choice.json"
    try:
        choice = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return choice if isinstance(choice, dict) else None


def final_check(run: Path) -> dict:
    """Whether submission/FINAL can be accepted: every lever group tested or given a reason, and
    a choice whose chain verifies."""
    run = Path(run)
    groups = lever_groups(run)
    tested = coverage(calls(run), groups)
    choice = read_choice(run)
    given = (choice or {}).get("skipped")
    reasons = {k: r.strip() for k, r in given.items() if isinstance(r, str) and r.strip()} \
        if isinstance(given, dict) else {}
    skipped = {g: reasons.get(g) or reasons[g.split(".", 1)[0]] for g in groups
               if g in reasons or g.split(".", 1)[0] in reasons}
    missing = [g for g, calls in tested.items() if not calls and g not in skipped]
    if choice is None:
        chain_problems = ["no readable submission/choice.json"]
    else:
        chain_problems = chain(run, choice)["problems"]
    return {"accepted": not missing and not chain_problems, "tested": tested, "skipped": skipped,
            "missing": missing, "chain_problems": chain_problems}


def describe(check: dict) -> str:
    lines = ["lever coverage (FINAL is accepted once every group is tested or has a reason in "
             "submission/choice.json 'skipped', and the choice's chain verifies):"]
    for group, calls in check["tested"].items():
        state = (f"tested in {', '.join(calls)}" if calls else
                 f"untested, skipped: {check['skipped'][group]}" if group in check["skipped"] else "UNTESTED")
        lines.append(f"  {group}: {state}")
    if check["chain_problems"]:
        lines.append("choice.json chain: " + "; ".join(check["chain_problems"]))
    else:
        lines.append("choice.json chain: verified")
    if check["accepted"]:
        lines.append("FINAL would be accepted.")
    else:
        reasons = []
        if check["missing"]:
            reasons.append(f"{', '.join(check['missing'])} untested without a reason")
        if check["chain_problems"]:
            reasons.append("the choice does not verify")
        lines.append(f"FINAL would NOT be accepted: {'; '.join(reasons)}.")
    return "\n".join(lines)
