"""The tool ledger of a run and the call chain behind a submission.

Every tool call appends one `start` and one `end` line to <run>/logs/tool_calls.jsonl and works in
<run>/workspace/calls/<ID>_<NAME>/. `chain` walks a submission back from its cluster call:

    K (cluster, done) -> its input (E, or B and B's input E) -> E's crop calls -> their mask calls

Each call must have ended `done` or `partial` (one start and one end line), every output it
recorded must still have the recorded sha256 (for crops also each crop file used by E against its
crops.csv sha256), and the cells must agree: the K states' cells == the embedding's cells == the
crop rows E used minus the cells E excluded. Each call must also have been run by the real tool:
its start line names the sha256 of the tool file and of _common.py, equal to the repository
template's, with the test stubs off, and a command that ran `tools/<tool>.py ... --name <name>`
exists -- a traced agent Bash command (agent runs) or the driver's log (logs/control_commands.jsonl: control,
smoke and SHAPE-arm runs, `run_arm`). An imported record (a replay workspace of
`harness attribute`: a donor run's call copied with its ledger lines, tagged `imported_from` =
{"run", "call"}) is accepted when the donor's ledger holds the same start and end lines, the donor's
own command provenance holds, and the copied outputs verify against the recorded sha256.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

COMPLETED = ("done", "partial")
TEMPLATE_TOOLS = Path(__file__).resolve().parents[1] / "workspace_template" / "tools"
CONTROL_COMMANDS = "control_commands.jsonl"
ALLOW_STUBS = False       # tests only: accept calls made with FULL_HCMA_TOOL_STUBS active
CALL_ID = re.compile(r"^[SCEBK]\d{3,}$")
IMPORTED = "imported_from"       # a call copied into a replay workspace from a donor run (harness attribute)
_EXAMPLES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ledger(run: Path) -> list[dict]:
    """Every parseable line of <run>/logs/tool_calls.jsonl, in file order."""
    path = Path(run) / "logs" / "tool_calls.jsonl"
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def calls(run: Path) -> dict[str, dict]:
    """{call_id: {"start": line | None, "end": line | None, "dir": Path | None, "extra_lines": n}};
    the first start / end line of a call id counts; `extra_lines` counts further ones (a problem
    for the chain)."""
    out: dict[str, dict] = {}
    for entry in read_ledger(run):
        call_id, event = entry.get("call"), entry.get("event")
        if not isinstance(call_id, str) or event not in ("start", "end"):
            continue
        record = out.setdefault(call_id, {"start": None, "end": None, "dir": None, "extra_lines": 0})
        if record[event] is not None:
            record["extra_lines"] += 1
            continue
        record[event] = entry
        if event == "start" and entry.get("call_dir"):
            record["dir"] = Path(run) / "workspace" / entry["call_dir"]
    return out


def traced_tool_uses(run: Path) -> list[tuple[int, int, str, dict]]:
    """(round, line, tool name, input) of every tool call of the agent in traces/round_*.jsonl
    (Claude stream-json `assistant` events)."""
    out = []
    for path in sorted((Path(run) / "traces").glob("round_*.jsonl")):
        try:
            round_no = int(path.stem.split("_", 1)[1])
        except ValueError:
            round_no = -1
        for number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    data = block.get("input")
                    out.append((round_no, number, str(block.get("name")), data if isinstance(data, dict) else {}))
    return out


def run_arm(run: Path) -> str:
    """"agent" (a Claude Code run: AGENT_RUN.json or a traced round, or a RUN_META.json of another arm than
    "shape"), "shape" (RUN_META.json arm "shape" and none of the agent's files: harness/shape_arm.py, whose tool
    commands are in logs/control_commands.jsonl), else "control" (the control and the smoke)."""
    run = Path(run)
    if (run / "AGENT_RUN.json").exists() or any((run / "traces").glob("round_*.jsonl")):
        return "agent"
    if not (run / "RUN_META.json").exists():
        return "control"
    try:
        meta = json.loads((run / "RUN_META.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    return "shape" if isinstance(meta, dict) and meta.get("arm") == "shape" else "agent"


def is_agent_run(run: Path) -> bool:
    return run_arm(run) == "agent"


def tool_commands(run: Path) -> list[str]:
    """The command lines that ran tools: the agent's traced Bash commands (agent runs), else the
    control driver's log of its subprocess calls."""
    run = Path(run)
    if is_agent_run(run):
        return [str(data.get("command", "")) for _, _, name, data in traced_tool_uses(run) if name == "Bash"]
    path = run / "logs" / CONTROL_COMMANDS
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            out.append(" ".join([f"tools/{entry['tool']}.py", *map(str, entry["argv"])]))
        except (ValueError, KeyError, TypeError):
            continue
    return out


def ran_by_command(tool: str, name: str, commands: list[str]) -> bool:
    """A command line runs tools/<tool>.py with --name <name>."""
    script = re.compile(rf"(^|[\s/'\"]){re.escape(tool)}\.py\b")
    named = re.compile(rf"--name[\s=]+['\"]?{re.escape(name)}['\"]?(?=$|[\s;&|)])")
    return any("tools/" in c and script.search(c) and named.search(c) for c in commands)


def template_hashes() -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(TEMPLATE_TOOLS.glob("*.py"))}


def input_calls(start: dict) -> list[str]:
    """The call ids named anywhere in a start line's `inputs`, in order of appearance."""
    found: list[str] = []

    def visit(value) -> None:
        if isinstance(value, str) and CALL_ID.match(value) and value not in found:
            found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(start.get("inputs"))
    return found


def state_columns(columns) -> dict[float, str]:
    """{resolution: column} of the `state_<res>` columns of a states table."""
    out = {}
    for column in columns:
        if str(column).startswith("state_"):
            try:
                out[float(str(column)[len("state_"):])] = str(column)
            except ValueError:
                continue
    return out


def _examples(items) -> str:
    items = sorted(items)
    return ", ".join(items[:_EXAMPLES]) + (f" (+{len(items) - _EXAMPLES} more)" if len(items) > _EXAMPLES else "")


def _excluded_cells(workspace: Path, start: dict) -> tuple[set[str], str | None]:
    """The cells an embed call was told to leave out (`inputs.exclude_cells`, a workspace-relative
    file with one cell_id per line, sha256 `inputs.exclude_cells_sha256` when recorded)."""
    inputs = start.get("inputs") or {}
    rel = inputs.get("exclude_cells")
    if not rel:
        return set(), None
    path = workspace / rel
    if not path.is_file():
        return set(), f"its exclude-cells file {rel} is missing"
    recorded = inputs.get("exclude_cells_sha256")
    if recorded and sha256(path) != recorded:
        return set(), f"its exclude-cells file {rel} changed since the call"
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}, None


class _Chain:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.records = calls(self.run)
        self.commands = tool_commands(self.run)
        self.template = template_hashes()
        self.donors: dict[str, tuple[dict, list[str]]] = {}
        self.problems: list[str] = []
        self.used: dict[str, dict] = {}
        self.usable: set[str] = set()
        self.out = {"ok": False, "problems": self.problems, "calls": self.used, "cluster_call": None,
                    "resolution": None, "state_column": None, "embedding_call": None, "embed_call": None,
                    "n_cells": None, "n_excluded": None, "excluded_per_image": {}, "excluded_cells": []}

    def call(self, call_id: str, tools: tuple, statuses: tuple = COMPLETED, consumed: set | None = None) -> dict | None:
        """The ledger record of a chain call, None when it cannot serve. Its recorded outputs are
        verified against their sha256 (only the `consumed` ones when given)."""
        if call_id in self.used:
            return self.records[call_id] if call_id in self.usable else None
        record = self.records.get(call_id)
        if record is None or record["start"] is None:
            self.problems.append(f"{call_id}: no such call in the ledger")
            self.used[call_id] = {"tool": None, "name": None, "status": None, "dir": None}
            return None
        start, end = record["start"], record["end"] or {}
        self.used[call_id] = {"tool": start.get("tool"), "name": start.get("name"), "status": end.get("status"),
                              "dir": start.get("call_dir")}
        if start.get("tool") not in tools:
            self.problems.append(f"{call_id}: a {start.get('tool')} call where the chain needs {' or '.join(tools)}")
            return None
        if end.get("status") not in statuses:
            self.problems.append(f"{call_id}: status {end.get('status') or 'not ended'} (needs {' or '.join(statuses)})")
            return None
        if record["dir"] is None or not record["dir"].is_dir():
            self.problems.append(f"{call_id}: call directory {start.get('call_dir')} is missing")
            return None
        self.provenance(call_id, record)
        outputs = end.get("outputs") or {}
        if not outputs:
            self.problems.append(f"{call_id}: no outputs recorded")
        for rel, digest in outputs.items():
            if consumed is not None and rel not in consumed:
                continue
            path = record["dir"] / rel
            if not path.is_file():
                self.problems.append(f"{call_id}: recorded output {rel} is missing")
            elif sha256(path) != digest:
                self.problems.append(f"{call_id}: {rel} changed since the call (sha256 differs from the recorded value)")
        self.usable.add(call_id)
        return record

    def imported(self, call_id: str, record: dict) -> list[str] | None:
        """The donor's command lines for an imported record, None (and a problem) when the import does
        not match the donor's ledger."""
        tag = record["start"].get(IMPORTED) or {}
        donor = str(tag.get("run") or "")
        if donor not in self.donors:
            self.donors[donor] = (calls(Path(donor)), tool_commands(Path(donor))) if donor and Path(donor).is_dir() \
                else ({}, [])
        records, commands = self.donors[donor]
        theirs = records.get(tag.get("call") or call_id)
        strip = lambda line: {k: v for k, v in (line or {}).items() if k != IMPORTED}      # noqa: E731
        if theirs is None or strip(theirs["start"]) != strip(record["start"]) \
                or strip(theirs["end"]) != strip(record["end"]):
            self.problems.append(f"{call_id}: imported from {donor or 'an unnamed run'} but its ledger lines differ "
                                 "from the donor's")
            return None
        return commands

    def provenance(self, call_id: str, record: dict) -> None:
        """The call was made by the repository's tool code, by a command that ran it (for an imported
        record: the donor's command, and the donor's ledger lines are the same)."""
        start, tool = record["start"], record["start"].get("tool")
        commands = self.commands
        if start.get(IMPORTED):
            commands = self.imported(call_id, record)
            if commands is None:
                return
        if record["extra_lines"]:
            self.problems.append(f"{call_id}: {record['extra_lines']} more start / end line(s) in the ledger")
        if start.get("tool_sha256") != self.template.get(f"{tool}.py"):
            self.problems.append(f"{call_id}: tool_sha256 is not the sha256 of the template's {tool}.py")
        if start.get("common_sha256") != self.template.get("_common.py"):
            self.problems.append(f"{call_id}: common_sha256 is not the sha256 of the template's _common.py")
        if start.get("stubs_active") is not False and not (ALLOW_STUBS and start.get("stubs_active") is True):
            self.problems.append(f"{call_id}: stubs_active is {start.get('stubs_active')!r} (needs false)")
        if not ran_by_command(tool, str(start.get("name")), commands):
            source = ("the donor's command" if start.get(IMPORTED) else "traced agent Bash command"
                      if is_agent_run(self.run) else f"line of logs/{CONTROL_COMMANDS}")
            self.problems.append(f"{call_id}: no {source} runs tools/{tool}.py --name {start.get('name')}")

    def table(self, call_id: str, record: dict, rel: str) -> pd.DataFrame | None:
        if rel not in (record["end"].get("outputs") or {}):
            self.problems.append(f"{call_id}: {rel} is not among the outputs the call recorded")
            return None
        if not (record["dir"] / rel).is_file():
            return None                                   # reported by call()
        return pd.read_csv(record["dir"] / rel, dtype=str, keep_default_na=False)

    def single_input(self, call_id: str, record: dict, prefixes: str) -> str | None:
        found = [c for c in input_calls(record["start"]) if c[0] in prefixes]
        if len(found) != 1:
            self.problems.append(f"{call_id}: needs exactly one input call of kind {'/'.join(prefixes)}, "
                                 f"its inputs name {found or 'none'}")
            return None
        return found[0]

    def walk(self, choice: dict) -> None:
        out, problems = self.out, self.problems
        k_id = choice.get("cluster_call") if isinstance(choice, dict) else None
        if not isinstance(k_id, str) or not k_id:
            problems.append("choice.json names no cluster_call")
            return
        out["cluster_call"] = k_id
        k = self.call(k_id, ("cluster",), ("done",))
        if k is None:
            return
        states = self.table(k_id, k, "states.csv")
        if states is not None:
            self.resolution(k_id, states, choice.get("resolution"))
            if states["cell_id"].duplicated().any():
                problems.append(f"{k_id}: states.csv lists a cell twice")

        src_id = self.single_input(k_id, k, "EB")
        if src_id is None:
            return
        out["embedding_call"] = src_id
        src = self.call(src_id, ("embed", "batch_correct"))
        if src is None:
            return
        src_cells = self.table(src_id, src, "cells.csv")
        if "embedding.npy" not in (src["end"].get("outputs") or {}):
            problems.append(f"{src_id}: embedding.npy is not among the outputs the call recorded")
        elif src_cells is not None and (src["dir"] / "embedding.npy").is_file():
            rows = np.load(src["dir"] / "embedding.npy", mmap_mode="r").shape[0]
            if rows != len(src_cells):
                problems.append(f"{src_id}: embedding.npy has {rows} rows, cells.csv {len(src_cells)}")
        if states is not None and src_cells is not None:
            mine, theirs = set(states["cell_id"]), set(src_cells["cell_id"])
            if mine != theirs or len(states) != len(src_cells):
                problems.append(f"{k_id}: its states cover other cells than {src_id}'s embedding "
                                f"({len(mine - theirs)} extra, {len(theirs - mine)} missing)")

        e_id, e, e_cells = src_id, src, src_cells
        if src["start"].get("tool") == "batch_correct":
            e_id = self.single_input(src_id, src, "E")
            if e_id is None:
                return
            e = self.call(e_id, ("embed",))
            if e is None:
                return
            e_cells = self.table(e_id, e, "cells.csv")
            if e_cells is not None and src_cells is not None \
                    and list(src_cells["cell_id"]) != list(e_cells["cell_id"]):
                problems.append(f"{src_id}: its cells differ from its input {e_id}'s cells")
        out["embed_call"] = e_id
        if e_cells is None:
            return
        out["n_cells"] = int(len(e_cells))
        if e_cells["cell_id"].duplicated().any():
            problems.append(f"{e_id}: cells.csv lists a cell twice")
            return
        self.crops(e_id, e, e_cells)

    def resolution(self, k_id: str, states: pd.DataFrame, resolution) -> None:
        columns = state_columns(states.columns)
        if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
            self.problems.append(f"resolution {resolution!r} is not a number")
            return
        match = [col for res, col in columns.items() if math.isclose(res, float(resolution), rel_tol=1e-9)]
        if not match:
            self.problems.append(f"resolution {resolution} is not one of {k_id}'s resolutions {sorted(columns)}")
            return
        self.out["resolution"], self.out["state_column"] = float(resolution), match[0]

    def crops(self, e_id: str, e: dict, e_cells: pd.DataFrame) -> None:
        """E's cells against the crop rows of its crop calls, and those crops' mask calls."""
        problems = self.problems
        excluded, problem = _excluded_cells(self.run / "workspace", e["start"])
        if problem:
            problems.append(f"{e_id}: {problem}")
        embedded = set(e_cells["cell_id"])
        if excluded & embedded:
            problems.append(f"{e_id}: embeds cells it excluded: {_examples(excluded & embedded)}")
        masks_used = {}
        for c_id, delivered in e_cells.groupby("crop_call", sort=True):
            c = self.call(c_id, ("crop",))
            if c is None:
                continue
            crops, per_image = self.table(c_id, c, "crops.csv"), self.table(c_id, c, "per_image.csv")
            if crops is None or per_image is None:
                continue
            if crops["cell_id"].duplicated().any():
                problems.append(f"{c_id}: crops.csv lists a cell twice")
                continue
            rows = crops.set_index("cell_id", drop=False)
            unknown = set(delivered["cell_id"]) - set(rows.index)
            if unknown:
                problems.append(f"{e_id}: cells not among {c_id}'s crops: {_examples(unknown)}")
            known = delivered[delivered["cell_id"].isin(rows.index)]
            other = known["image_id"].to_numpy() != rows.loc[known["cell_id"], "image_id"].to_numpy()
            if other.any():
                problems.append(f"{e_id}: cells with another image than in {c_id}: {_examples(known['cell_id'][other])}")
            changed = [cell for cell in known["cell_id"]
                       if not (c["dir"] / rows.at[cell, "crop_path"]).is_file()
                       or sha256(c["dir"] / rows.at[cell, "crop_path"]) != rows.at[cell, "sha256"]]
            if changed:
                problems.append(f"{c_id}: crop files missing or changed since the call: {_examples(changed)}")
            # every crop row of the (image, crop call) pairs E used is embedded or excluded
            images = set(delivered["image_id"])
            dropped = set(crops.loc[crops["image_id"].isin(images), "cell_id"]) - embedded
            if dropped - excluded:
                problems.append(f"{e_id}: crop rows of {c_id} neither embedded nor excluded: "
                                f"{_examples(dropped - excluded)}")
            if "mask_call" not in per_image:
                problems.append(f"{c_id}: per_image.csv has no mask_call column")
                continue
            mask_calls = dict(zip(per_image["image_id"], per_image["mask_call"]))
            for image_id in sorted(images):
                if not mask_calls.get(image_id):
                    problems.append(f"{c_id}: no mask call recorded for {image_id}")
                else:
                    masks_used.setdefault(mask_calls[image_id], set()).add(image_id)
        # a segment call: only the masks the chain used are verified (and its per_image.csv)
        for s_id, images in sorted(masks_used.items()):
            masks = {f"masks/{image_id}.tif" for image_id in images}
            s = self.call(s_id, ("segment",), consumed={"per_image.csv", *masks})
            unrecorded = masks - set(s["end"].get("outputs") or {}) if s is not None else set()
            if unrecorded:
                problems.append(f"{s_id}: no recorded mask for {_examples(unrecorded)}")
        # every cell E was told to leave out, per image (cell_id = <image_id>_<mask call>_<label>):
        # the scorer counts them as unlabeled for their image's type
        per_image: dict[str, int] = {}
        for cell in excluded:
            image_id = cell.rsplit("_", 2)[0]
            per_image[image_id] = per_image.get(image_id, 0) + 1
        self.out["n_excluded"], self.out["excluded_per_image"] = len(excluded), dict(sorted(per_image.items()))
        self.out["excluded_cells"] = sorted(excluded)


def chain(run: Path, choice: dict) -> dict:
    """Verify the chain behind `choice` ({"cluster_call", "resolution", ...}).

    {"ok", "problems", "calls": {call_id: {tool, name, status, dir}}, "cluster_call",
     "resolution", "state_column", "embedding_call" (the K call's input: E or B), "embed_call",
     "n_cells", "n_excluded", "excluded_per_image" ({image_id: cells E excluded}), "excluded_cells"}."""
    walk = _Chain(run)
    walk.walk(choice)
    walk.out["ok"] = not walk.problems
    return walk.out
