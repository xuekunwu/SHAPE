"""The CPJUMP1 agent-arm stage orchestration: StageRun drives a chosen backend through the five stages.

The backends (codex, claude, biomni) and the arm registry live in shape/agent/arms; this module is the
dataset-agnostic-at-the-round-level StageRun. Each round the agent gets SHAPE's task, inputs, tool levers and
per-image call budget and returns a decision; trusted native workers here (claude_arm) run the requested tool calls.
The agent sees only what the tools return and write, plus the raw inputs it asks to inspect.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import threading
import time

from shape.agent.verification.base import BUDGET as MAX_TOOL_ROUNDS

from . import claude_arm as native
from shape.agent.arms import ARM_NAMES, EVIDENCE_POLICY, MAX_INSPECT, MalformedDecision, RoundTimeout, backend
from .data import digest, load_prepared, read_json, within, write_json
from .pipeline import CPJUMP1, dataset, stage_reasoning

# ---------------------------------------------------------------- the stage protocol

# The agent arms share this protocol; only the backend answering each round differs. The
# backends and the arm registry live in shape/agent/arms; this module is the dataset-agnostic StageRun that drives
# a chosen backend (`backend(arm)`) through the five CPJUMP1 stages.
MAX_ROUNDS = 32           # rounds of a group or dataset session (config `agent_decision_rounds` overrides it)
IMAGE_ROUNDS = 8          # rounds of one per-image session
# SHAPE's BUDGET (shape/agent/verification/base.py) is the tool-call rounds per image per stage. Segment/crop count
# the default call as round 1 (their verifiers' `admission_call`); embed/batch do not count trial 0, the input
# (`admission_on_input`). The calls one round may hold per image: `round_cap`.
OUTPUTS = native.OUTPUTS


IMAGE_STAGES = ("segment", "crop")
TOOLS = {"segment": ("cell_segmenter", "Cell_Segmenter_Tool"), "crop": ("single_cell_cropper", "Single_Cell_Cropper_Tool"),
         "embed": ("cell_state_analyzer", "Cell_State_Analyzer_Tool"), "batch": ("batch_corrector", "Batch_Corrector_Tool")}
# The CARD sections SHAPE's agentic loop puts in its prompts (`AgenticVerifier._card_section`; segmentation also shows
# its default target convention).
CARD_SECTIONS = {"segment": ("## Levers", "## Default target convention"), "crop": ("## Levers",),
                 "embed": ("## Levers",), "batch": ("## Levers",)}
# SHAPE's task wording for each stage: the dataset's (`pipeline.Dataset.tasks`), the request / brief its verifiers
# get; these are CPJUMP1's.
TASKS = CPJUMP1.tasks
INPUT_FILE = {"segment": "segment_file", "crop": "file"}
INSPECTABLE = (".png", ".tif", ".tiff", ".json")
HEAVY = ("cell_crop_objects", "corrected_cls_output", "cell_crops_paths")
JSON_TEXT_LIMIT = 200_000
# The texts of one round's inspected JSON files share INSPECT_JSON_BUDGET characters: a file gets at most
# min(JSON_TEXT_LIMIT, INSPECT_JSON_BUDGET // number of JSON files in the request). Codex rejects an input above
# 1,048,576 characters; the parts of a group prompt at their bounds are the call table (PROMPT_TABLE_BUDGET, 400k), the
# inspected JSON (250k), the echoed previous decisions (DECISIONS_ECHO_BUDGET, 120k) and the facts with the registry and
# CARD (about 10-30k), about 800k. PROMPT_CHAR_LIMIT is the last line: a prompt above it never reaches the CLI.
INSPECT_JSON_BUDGET = 250_000
PROMPT_CHAR_LIMIT = 900_000
# One protocol, one prompt for the three agent arms (comparability), so the budgets above are the same for all; only the
# last-line guard follows the arm's own input path. Codex: the gateway's 1,048,576-character input limit (hit on the VM).
# Claude Code: one fresh headless `claude -p` per round (no session carried between rounds), the prompt on stdin; the
# window is not stated in the repository, but a single request of 448,646 cache-creation tokens (two turns) was accepted
# in the CPJUMP1 five-arm run (crop round 3), above what 900,000 characters of this JSON-heavy text come to (about 300-400k
# tokens), and an overflow is a CLI API error that fails the arm (claude.response). Biomni: the same model and
# gateway as Codex (1,048,576 characters), but A1 adds its own system prompt and every generate step's reply to the
# request within a round (recursion_limit 30; observed 1-3 steps of a few hundred output tokens), so its guard keeps 248k
# characters back.
ARM_PROMPT_LIMIT = {"biomni": 800_000}
# The previous decisions are echoed in every prompt; a decision whose JSON exceeds DECISION_ECHO_LIMIT characters (a run
# request naming 1310 images, a message listing their calls) is echoed with its request_json / message / reason cut,
# and when the echo still exceeds DECISIONS_ECHO_BUDGET the oldest decisions are cut to DECISION_ECHO_STUB characters per
# field. The decision itself stays complete in ROUNDS.json.
DECISION_ECHO_LIMIT = 20_000
DECISIONS_ECHO_BUDGET = 120_000
DECISION_ECHO_STUB = 1_000
RENDER_WIDTH = 2560
# A group session's prompt holds one table row per image per call (~170 characters each). Codex rejects an input above
# 1,048,576 characters, and the table is only part of the prompt (the facts with the registry and CARD, the session's
# decisions and the inspected JSON are the rest), so a table above PROMPT_TABLE_BUDGET characters is replaced by an
# aggregated one (`StageRun.aggregate`). Prompts whose table fits are unchanged.
PROMPT_TABLE_BUDGET = 400_000
# The aggregated table's per-image rows (images that are flagged or failed in some call) stop at this many characters, so
# the aggregated table stays far below PROMPT_TABLE_BUDGET; the images beyond it are counted, not shown.
FLAGGED_ROWS_BUDGET = 150_000
FLAGGED_IDS_PER_CALL = 20
PARAM_SETS_PER_CALL = 10
# The tool's own reported quality problems in a call's result (its scalars, `_scalars`): value that flags the call.
# Segmentation reports its visual QC as overlay_qc.*; a call failing it is also recorded failed (its `error` is set).
# The cropper reports no per-call quality field in its result (its QC is per crop, in metadata the agent does not see),
# so only its failed calls are flagged.
QC_FLAG_FIELDS = {"overlay_qc.passed": False, "overlay_qc.coverage_implausible": True, "overlay_qc.overlay_is_black": True}


# The decision format every prompt states (`facts`), and states again after a malformed reply (`prompt`).
RESPONSE_FORMAT = ("{\"action\": one of run, inspect, handoff, finish, \"request_json\": ..., \"reason\": a short "
                   "note recorded with the decision}")


class Rejected(Exception):
    """A request the harness does not execute; its message is shown to the agent in the session's next round."""


def round_cap(stage: str, config: dict) -> int:
    """Calls one image (embed/batch: the stage) may get in one round: the concurrency cap SHAPE's verifier of the
    stage plans each round with (`AgenticVerifier.concurrency`, `plan_round`'s k_max), at the cores this benchmark
    gives that verifier (crop: each image's share under SHAPE's crop policy, `pipeline.crop_workers`)."""
    cores = max(1, int(config.get("cores") or 1))
    if stage == "segment":
        from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier as Verifier
    elif stage == "crop":
        from shape.agent.verification.verifiers.crop import CropAgenticVerifier as Verifier
        from .cli import shape_policy
        from .pipeline import crop_workers
        cores = crop_workers({**config, "cores": cores, "shape_policy": shape_policy(cores)}, len(config["images"]))[1]
    elif stage == "embed":
        from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier as Verifier
    else:
        from shape.agent.verification.verifiers.batch_correction_unified import BatchCorrectionVerifier as Verifier
    verifier = Verifier()
    verifier.cores = cores
    return max(1, int(verifier.concurrency()))


def registry(stage):
    """The levers SHAPE's verifier of the stage sets: each tool CARD's `## Levers` (crop adds `cell_overrides`;
    embed leaves out max_epochs, fixed for every call by `pipeline.ssl_epoch_budget`, as SHAPE's FixedEpochVerifier
    does)."""
    if stage != "embed":
        return native.registry(stage)
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    from shape.utils.schemas import card_lever_fields, lever_spec
    from .cli import REPO
    C = tools_common()
    fields = card_lever_fields(str(REPO / "shape" / "tools" / TOOLS["embed"][0]), "## Levers")
    return {name: C.LeverInfo(name, *lever_spec(spec)) for name, spec in fields.items() if name != "max_epochs"}


def _cell_rules_text() -> str:
    from shape.agent.verification.verifiers.crop import CALL_CELL_READOUTS, CELL_KEYS, INPUT_CELL_READOUTS
    return ("- cell_rules (crop, optional; SHAPE's cell rules): a list of rules {\"when\": [{\"readout\": name, \"op\": "
            "one of >, >=, <, <=, ==, \"value\": a JSON literal}], \"apply\": [{\"key\": a per-cell key, \"value\": a "
            "JSON value}]}. A cell rule selects the cells for which all its conditions hold (no condition: every cell) "
            "and sets per-cell keys for them; rules apply in order on top of the call's own cell_overrides, a later "
            "rule overwriting an earlier one's keys, reset removes a cell's per-cell settings. Conditions read the "
            f"per-cell input measurements {list(INPUT_CELL_READOUTS)} and, with rules_parent = a finished call of the "
            f"image (e.g. \"c2\"), that call's per-cell readouts {list(CALL_CELL_READOUTS)} as the tool wrote them; "
            "== is the only op for margin_mode, touches_image_border and edge_limited. Per-cell keys: "
            f"{list(CELL_KEYS)} (auto_tighten acts only in mask_bbox mode; reset is true and alone in its rule). "
            "Each image's rules are compiled into its cell_overrides; a rule that matches no cell of an image fails "
            "that image's call, with the reason.")


def _cell_overrides(C, value):
    if not isinstance(value, dict):
        raise ValueError("cell_overrides must be an object {cell_id: {key: value}}")
    out = {}
    for cell, keys in value.items():
        if not isinstance(keys, dict) or set(keys) - set(C.CELL_KEYS):
            raise ValueError(f"cell_overrides[{cell}]: keys must be among {list(C.CELL_KEYS)}")
        out[str(cell)] = {k: C.check_value(C.CELL_KEYS[k], v) for k, v in keys.items()}
    return out


# SHAPE's cell rules for a crop call (user decision 2026-10-12: the Codex arm gets the permission SHAPE's agent has):
# `cell_rules` in SHAPE's own format (`CropMove.cell_rules`), compiled per image by SHAPE's compiler
# (`pipeline.compile_cell_rules`); `rules_parent` names the image's finished call whose per-cell readouts they may read.
CELL_RULE_PARAMS = ("cell_rules", "rules_parent")


def cell_rules(value, with_parent: bool) -> list[dict]:
    """`value` as SHAPE's cell rules (`CellRule`: when [{readout, op, value}], apply [{key, value}]); a condition reads a
    per-cell input measurement, or with a rules_parent also that call's per-cell readouts. Raises ValueError."""
    from pydantic import TypeAdapter, ValidationError
    from shape.agent.verification.verifiers.crop import CALL_CELL_READOUTS, INPUT_CELL_READOUTS, CellRule

    def literal(cond):                               # a condition's value is a JSON literal string, as SHAPE writes it
        if isinstance(cond, dict) and not isinstance(cond.get("value"), str):
            return {**cond, "value": json.dumps(cond.get("value"))}
        return cond

    if isinstance(value, list):
        value = [{**rule, "when": [literal(c) for c in rule.get("when") or []]} if isinstance(rule, dict) else rule
                 for rule in value]
    try:
        rules = [rule.model_dump() for rule in TypeAdapter(list[CellRule]).validate_python(value)]
    except ValidationError as error:
        raise ValueError(f"cell_rules: {error.errors()[0].get('loc')}: {error.errors()[0].get('msg')}") from None
    if not rules:
        raise ValueError("cell_rules must hold at least one rule")
    known = INPUT_CELL_READOUTS + (CALL_CELL_READOUTS if with_parent else ())
    for k, rule in enumerate(rules, 1):
        for cond in rule["when"]:
            if cond["readout"] not in known:
                raise ValueError(f"cell rule {k}: {cond['readout']!r} is not a per-cell readout of the input"
                                 + (" or rules_parent" if with_parent else " (call readouts need rules_parent)")
                                 + f" (known: {list(known)})")
    return rules


def parameters(stage, given):
    """Registry parameters only, each checked against its CARD type and range; crop also takes SHAPE's cell rules
    (`cell_rules`, `rules_parent`). Raises ValueError."""
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    C = tools_common()
    if not isinstance(given, dict):
        raise ValueError("parameters must be a JSON object")
    rules = {k: given[k] for k in CELL_RULE_PARAMS if k in given} if stage == "crop" else {}
    given = {k: v for k, v in given.items() if k not in rules}
    levers = registry(stage)
    unknown = sorted(set(given) - set(levers))
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}")
    try:
        out = {k: _cell_overrides(C, v) if levers[k].kind == "per_cell" else C.check_value(levers[k], v)
               for k, v in given.items()}
    except C.Invalid as error:
        raise ValueError(str(error)) from None
    if "rules_parent" in rules:
        if "cell_rules" not in rules:
            raise ValueError("rules_parent names the call whose per-cell readouts cell_rules read; it needs cell_rules")
        if not isinstance(rules["rules_parent"], str) or not rules["rules_parent"]:
            raise ValueError("rules_parent must be a call name of this image, e.g. \"c2\"")
        out["rules_parent"] = rules["rules_parent"]
    if "cell_rules" in rules:
        out["cell_rules"] = cell_rules(rules["cell_rules"], "rules_parent" in rules)
    return out


def tool_view(value, base: Path | None = None):
    """A tool's returned dict as the agent sees it: arrays and in-memory objects left out, long numeric lists
    replaced by their length, paths under the call folder `base` relative to it."""
    prefixes = [] if base is None else sorted({str(base), str(Path(base).resolve())}, key=len, reverse=True)
    if isinstance(value, dict):
        return {str(k): tool_view(v, base) for k, v in value.items() if k not in HEAVY}
    if isinstance(value, (list, tuple)):
        if len(value) > 256 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return f"<{len(value)} numbers omitted>"
        return [tool_view(v, base) for v in value]
    if isinstance(value, str):
        for prefix in prefixes:
            if value.startswith(prefix):
                return PurePosixPath(*Path(value[len(prefix):].lstrip("/\\")).parts).as_posix()
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    return value


def _scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float)) or (
        isinstance(value, str) and len(value) <= 40 and "/" not in value and "\\" not in value)


def _scalars(view: dict) -> dict:
    """The tool's own scalar outputs of one call, one level of nesting flattened (`key.subkey`)."""
    out = {}
    for key, value in view.items():
        if key in ("parameters", "summary"):
            continue
        if isinstance(value, dict):
            out.update({f"{key}.{k}": v for k, v in value.items() if _scalar(v)})
        elif _scalar(value):
            out[key] = value
    return out


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def _flag(record: dict) -> str | None:
    """Why a call is flagged in the aggregated table: "failed" (not complete, or an error recorded) or "qc" (the tool's
    own quality fields, QC_FLAG_FIELDS); None for a call without a problem."""
    if record["status"] != "complete" or record["error"]:
        return "failed"
    values = _scalars(record["view"]) if isinstance(record["view"], dict) else {}
    return "qc" if any(values.get(k) is bad for k, bad in QC_FLAG_FIELDS.items()) else None


def _distribution(values: list) -> dict | None:
    """n, mean, median, p10, p90, min, max of the numbers among `values` (booleans count as 0 / 1), else the counts of
    the (at most 5 most frequent) text values; None for no value."""
    numbers = sorted(float(v) if isinstance(v, float) else int(v) for v in values
                     if isinstance(v, (int, float)))
    if not numbers:
        texts = Counter(v for v in values if isinstance(v, str))
        return {"n": sum(texts.values()), "values": dict(sorted(texts.items(), key=lambda kv: (-kv[1], kv[0]))[:5])} \
            if texts else None

    def quantile(p: float) -> float:
        position = (len(numbers) - 1) * p
        low = int(position)
        high = min(low + 1, len(numbers) - 1)
        return round(numbers[low] + (numbers[high] - numbers[low]) * (position - low), 6)

    return {"n": len(numbers), "mean": round(sum(numbers) / len(numbers), 6), "median": quantile(0.5),
            "p10": quantile(0.1), "p90": quantile(0.9), "min": round(numbers[0], 6), "max": round(numbers[-1], 6)}


def _render(source: Path, target: Path) -> Path:
    """A PNG of a TIFF: channels side by side, each scaled from its 1st to its 99th percentile, at most RENDER_WIDTH
    wide. Scientific pixels are never modified."""
    import numpy as np
    import tifffile
    from PIL import Image
    data = np.squeeze(np.asarray(tifffile.imread(source)))
    if data.ndim == 2:
        channels = [data]
    elif data.ndim == 3 and data.shape[0] <= 8:
        channels = list(data)
    elif data.ndim == 3 and data.shape[-1] <= 8:
        channels = list(np.moveaxis(data, -1, 0))
    else:
        raise Rejected(f"{source.name}: cannot render an image of shape {list(data.shape)}")
    tiles = []
    for channel in channels:
        lo, hi = np.percentile(channel, [1, 99])
        tiles.append(np.clip((channel.astype(float) - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8))
    canvas = Image.fromarray(np.concatenate(tiles, axis=1))
    if canvas.width > RENDER_WIDTH:
        canvas = canvas.resize((RENDER_WIDTH, max(1, round(canvas.height * RENDER_WIDTH / canvas.width))))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    return target


def verify_call(root: Path, record: dict) -> None:
    folder = root / record["folder"]
    for path, sha in record["artifacts"].items():
        if digest(within(folder, path)) != sha:
            raise ValueError("Codex call artifact changed")


def verify_delivery(root: Path, selected: dict, calls: dict) -> None:
    """Every delivered call exists, completed and is unchanged."""
    for unit, choice in selected["deliveries"].items():
        record = calls[unit][choice["call"]]
        if record["status"] != "complete":
            raise ValueError("Codex delivered a failed call")
        verify_call(root, record)


class Session:
    def __init__(self, root: Path, kind: str, units: list[str]):
        self.root, self.kind, self.units = root, kind, list(units)
        self.decisions: list[dict] = []
        self.attach: list[Path] = []
        self.labels: list[str] = []
        self.texts: list[str] = []
        self.table_info: dict | None = None     # group session: table_mode / table_chars of the last prompt
        self.prompt_chars: int | None = None    # the length of the last prompt sent
        self.done = False

    @property
    def name(self) -> str:
        return self.kind if self.kind != "image" else f"image:{self.units[0]}"


class StageRun:
    """One Codex stage: the call ledger, the sessions, the request waves and the delivery."""

    def __init__(self, stage, prep, run, config):
        from .cli import REPO
        from benchmarks.full_HCMA.harness.shape_arm import tools_common
        self.arm, self.backend = config["arm"], backend(config["arm"])
        self.backend.route(config["model"])
        self.stage, self.prep, self.run, self.config = stage, Path(prep), Path(run), config
        self.spec = dataset(config)
        rounds = int((config.get("smoke_policy") or {}).get("codex_rounds", config.get("agent_decision_rounds", MAX_ROUNDS)))
        if rounds < 2:
            raise ValueError("Codex needs at least two decision rounds")
        self.root = self.run / "agent" / stage
        # A re-run of a staged stage (its previous attempt failed, or it is being redone) finds this
        # agent working dir already there; mkdir without exist_ok used to abort with FileExistsError.
        # It holds only this stage's agent scratch (auth / calls / rounds) -- the delivered stage
        # outputs live at the run root -- and every attempt rebuilds it from scratch, so clear a
        # stale one and start fresh rather than refuse.
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        # The stage's reasoning settings (pipeline.REASONING; None: the table does not apply to this arm), passed
        # to the backend's setup and every round.
        self.reasoning = stage_reasoning(self.arm, stage, config.get("reasoning", {}).get("protocol"))
        self.options = {} if self.reasoning is None else {"effort": self.reasoning["model_reasoning_effort"]}
        if self.reasoning is not None:
            print(f"REASONING stage={stage} arm={self.arm} {json.dumps(self.reasoning)}", flush=True)
        self.settings = self.backend.setup(self.root / "auth", config["model"], **self.options)
        self.images = list(config["images"])
        self.cores = max(1, int(config.get("cores") or 1))
        self.max_rounds, self.image_rounds = rounds, min(IMAGE_ROUNDS, rounds)
        self.cap = round_cap(stage, config)
        self.ssl_deadline = None
        if stage == "embed":
            from .pipeline import ssl_search_deadline
            self.ssl_deadline = ssl_search_deadline(config)
        self.calls: dict[str, dict[str, dict]] = {}
        self.rounds: list[dict] = []
        self.deliveries: dict[str, dict] = {}
        self.waves = 0
        self.batch_rule = None
        self.lock = threading.Lock()
        self._rows = None
        self._segmented = None
        directory = TOOLS[stage][0]
        self.card = "\n\n".join(tools_common().card_section(REPO, directory, s) for s in CARD_SECTIONS[stage])

    def rows(self) -> dict:
        if self._rows is None:
            wanted = set(self.images)
            self._rows = {r["image_id"]: r for r in self.spec.loader(load_prepared)(self.prep)
                          if r["image_id"] in wanted}
        return self._rows

    def dtype(self) -> str:
        """The raw images' dtype: the dataset's, else the one dtype of this stage's prepared rows."""
        if self.spec.dtype:
            return self.spec.dtype
        dtypes = {r["dtype"] for r in self.rows().values()}
        if len(dtypes) != 1:
            raise ValueError(f"the images do not share one dtype: {sorted(dtypes)}")
        return dtypes.pop()

    def save(self) -> None:
        write_json(self.root / "CALLS.json", self.calls)

    def check_time(self) -> None:
        if time.time() >= self.config["deadline"]:
            raise RuntimeError("Codex arm exceeded its deadline")

    # ---- one Codex round ------------------------------------------------------------------------------------
    def ask(self, session: Session, prompt: str, index: int):
        """(decision, usage); decision None for a malformed reply (usage["malformed"]: the parse error) or a failed
        round (the round's time cap). Basic LLM failures (route, key, configuration, CLI, failed or invalid event
        stream) and the arm deadline raise."""
        self.check_time()
        images = list(session.attach)
        try:
            decision, usage, _ = self.backend.invoke(session.root, prompt, self.config["model"], self.settings,
                                                     min(self.config["deadline"], time.time() + 600), index,
                                                     images=images, marker=self.root, **self.options)
        except (MalformedDecision, RoundTimeout) as error:
            self.check_time()
            usage = {"total_tokens": getattr(error, "tokens", 0), "n_images": len(images),
                     "error": f"{type(error).__name__}: {error}"}
            if isinstance(error, MalformedDecision):
                usage["malformed"] = str(error)
            return None, usage
        self.check_time()
        return decision, {**usage, "n_images": len(images)}

    def malformed(self, session: Session, error: str, last: bool) -> tuple[str, str]:
        """(outcome, message) of a reply that is not a valid decision, handled as a rejected request: nothing runs, it
        uses a session round and no tool-call round, and the next round states the error with the required format
        (`prompt`); on the last round the session ends unfinished, as after a rejected last round."""
        if last:
            self.unfinish(session, f"the last round's reply was not a valid decision: {error}")
        return "malformed", error

    def record(self, session: Session, index: int, decision: dict, usage: dict, outcome: str, message: str) -> None:
        decision = decision or {"action": None, "request_json": None, "reason": None}
        entry = {"round": index, "action": decision["action"], "request_json": decision["request_json"],
                 "reason": decision["reason"], "outcome": outcome, "message": message}
        session.decisions.append(entry)
        with self.lock:
            self.rounds.append({"session": session.name, **entry, "decision": decision, **(session.table_info or {}),
                                **({} if session.prompt_chars is None else {"prompt_chars": session.prompt_chars}),
                                **usage})
            write_json(self.root / "ROUNDS.json", self.rounds)

    def rounds_used(self, unit: str) -> int:
        """Tool-call rounds of an image (c0's included) or of the dataset stage (the input's excluded)."""
        return len({r["wave"] for r in self.calls.get(unit, {}).values() if r.get("wave")})

    def unfinish(self, session: Session, reason: str) -> None:
        """A session that ends without a finish: its images (or the dataset) get the default call, recorded."""
        default = "c0" if self.stage in IMAGE_STAGES else "input"
        for unit in session.units:
            self.deliveries[unit] = {"call": default, "source": "default_session_unfinished", "reason": reason,
                                     "session": session.name}
        session.done = True

    # ---- tool calls -----------------------------------------------------------------------------------------
    def native_config(self, **fields) -> dict:
        """A native worker's CONFIG.json: every agent arm takes pipeline.py's agent-arm native-call path (arm
        "codex"); agent_arm names the real arm."""
        return {**self.config, "arm": "codex", "agent_arm": self.arm, **fields}

    def wave(self, requests: list[tuple[str, dict]]) -> list[str]:
        """Run the requested image-stage calls in one native worker; returns their names (image/call)."""
        from .cli import child
        self.check_time()
        name = f"w{self.waves:03d}"
        self.waves += 1
        folder = self.root / "waves" / name
        folder.mkdir(parents=True)
        pending, planned = Counter(), []
        for image, params in requests:
            planned.append({"image_id": image, "call": f"c{len(self.calls.get(image, {})) + pending[image]}",
                            "params": params})
            if params.get("rules_parent"):                 # the call folder whose crop readouts the rules read
                planned[-1]["rules_parent_folder"] = str(self.root / self.calls[image][params["rules_parent"]]["folder"])
            pending[image] += 1
        for filename in native.INPUTS[self.stage]:
            shutil.copy2(self.run / filename, folder / filename)
        write_json(folder / "CONFIG.json", self.native_config(images=sorted(pending), calls=planned))
        child([sys.executable, "-m", self.spec.cli, "native-trial", self.stage, "--prep", str(self.prep),
               "--run", str(folder)],
              folder / "tool.log", self.config["deadline"] - time.time(), new_session=False)
        for call in planned:
            image, label = call["image_id"], call["call"]
            directory = folder / image / label
            record = read_json(directory / "CALL.json")
            if (record["image_id"], record["call"], record["params"]) != (image, label, call["params"]):
                raise ValueError("native call record does not match its request")
            self.calls.setdefault(image, {})[label] = {
                "params": call["params"], "status": record["status"], "error": record["error"],
                "seconds": record["seconds"], "wave": name, "folder": directory.relative_to(self.root).as_posix(),
                "artifacts": {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob("*") if p.is_file()},
                "view": tool_view(read_json(directory / "result.json"), directory), "delivery": record["delivery"]}
        self.save()
        return [f"{c['image_id']}/{c['call']}" for c in planned]

    def dataset_call(self, label: str, params: dict, parent: str = "", wave: str | None = None) -> None:
        """One embed/batch call (the existing native worker, its own process)."""
        from .cli import child
        self.check_time()
        folder = self.root / "calls" / label
        folder.mkdir(parents=True)
        cfg = self.native_config(tool_params={k: v for k, v in params.items() if k != "init_from"},
                                 fit_only=self.stage == "embed")
        if self.stage == "batch":
            cfg["uncorrected"] = label == "input"
        if self.ssl_deadline is not None:
            cfg["ssl_search_deadline"] = self.ssl_deadline
        if parent:
            previous = self.calls["dataset"][parent]
            verify_call(self.root, previous)
            cfg["resume_checkpoint"] = read_json(self.root / previous["folder"] / "SSL_SELECTION.json")["checkpoint"]
        for filename in native.INPUTS[self.stage]:
            shutil.copy2(self.run / filename, folder / filename)
        write_json(folder / "CONFIG.json", cfg)
        started, error = time.time(), None
        try:
            child([sys.executable, "-m", self.spec.cli, "native-trial", self.stage, "--prep", str(self.prep),
                   "--run", str(folder)],
                  folder / "tool.log", self.config["deadline"] - time.time(), new_session=False)
        except RuntimeError as failure:
            if label == "input":
                raise
            log = folder / "tool.log"
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-5:] if log.is_file() else []
            error = f"{failure}: " + " | ".join(tail)
        if self.stage == "embed":
            result = folder / "ssl_result.json"
            view = tool_view(read_json(result), folder) if result.is_file() else {}
        else:
            result = folder / "batch" / "call_000" / "result.json"
            view = tool_view(read_json(result), folder) if result.is_file() else (
                {"status": "uncorrected input (not a tool call)"} if label == "input" else {})
        self.calls.setdefault("dataset", {})[label] = {
            "params": params, "parent": parent, "status": "error" if error else "complete", "error": error,
            "seconds": round(time.time() - started, 1), "wave": wave,
            "folder": folder.relative_to(self.root).as_posix(),
            "artifacts": {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file()},
            "view": view, "delivery": None}
        self.save()

    # ---- request validation ---------------------------------------------------------------------------------
    def expand(self, specs, units: list[str]) -> list[tuple[str, dict]]:
        """Image-stage call specs -> [(image, params)]; the per-image cap applies to the whole request."""
        if not isinstance(specs, list) or not specs:
            raise Rejected("run needs a non-empty JSON list of call specs")
        requests = []
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) - {"images", "params", "per_image"}:
                raise Rejected("a call spec is an object with keys images, params, per_image")
            names = spec.get("images", "all")
            ids = list(units) if names == "all" else names
            if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                    or len(set(ids)) != len(ids) or any(i not in units for i in ids)):
                raise Rejected("a call spec's images must be \"all\" or a list of distinct image IDs of this session")
            shared, per = spec.get("params", {}), spec.get("per_image", {})
            if not isinstance(shared, dict) or not isinstance(per, dict) or set(per) - set(ids):
                raise Rejected("params must be an object and per_image an object keyed by the spec's images")
            for image in ids:
                own = per.get(image, {})
                if not isinstance(own, dict):
                    raise Rejected(f"per_image[{image}] must be an object")
                try:
                    requests.append((image, parameters(self.stage, {**shared, **own})))
                except ValueError as error:
                    raise Rejected(f"{image}: {error}") from None
                parent = requests[-1][1].get("rules_parent")
                if parent and (self.calls.get(image, {}).get(parent) or {}).get("status") != "complete":
                    raise Rejected(f"{image}: rules_parent {parent!r} is not a finished call of this image")
        counts = Counter(image for image, _ in requests)
        crowded = {i: n for i, n in counts.items() if n > self.cap}
        if crowded:
            raise Rejected(f"rejected: the request gives {len(crowded)} image(s) more than {self.cap} call(s) in one "
                           f"round (calls per image: {dict(list(crowded.items())[:20])}); nothing was run")
        over = sorted(i for i in counts if self.rounds_used(i) + 1 > MAX_TOOL_ROUNDS)
        if over:
            raise Rejected(f"rejected: {len(over)} image(s) already used all {MAX_TOOL_ROUNDS} tool-call rounds of this "
                           f"stage ({over[:20]}); nothing was run")
        return requests

    def deliverable(self, unit: str, label: str) -> str | None:
        record = self.calls.get(unit, {}).get(label)
        if record is None:
            return f"{unit} has no call {label}"
        if record["status"] != "complete":
            return f"{unit}/{label} failed"
        delivery = record["delivery"] or {}
        # A dataset that allows empty masks (pipeline.Dataset.allow_empty_masks) accepts a segmentation with no cells
        # and, for an image whose delivered segmentation has none, a crop call with no crops, as its native path does.
        if self.stage == "segment" and not delivery.get("n_cells") and not self.spec.allow_empty_masks:
            return f"{unit}/{label} has no cells"
        if self.stage == "crop" and not delivery.get("n_crops") and not (
                self.spec.allow_empty_masks and self.segmented_cells(unit) == 0):
            return f"{unit}/{label} has no crops"
        return None

    def segmented_cells(self, unit: str) -> int:
        """The cell count of an image's delivered segmentation (crop stage)."""
        if self._segmented is None:
            self._segmented = {r["image_id"]: r["n_cells"] for r in read_json(self.run / "segmentation.json")}
        return self._segmented[unit]

    def choose(self, request, units: list[str]) -> dict:
        """finish: {unit: call}, "all" for every unit not named; a unit left unnamed gets the default call."""
        if (not isinstance(request, dict) or set(request) - {*units, "all"}
                or any(not isinstance(v, str) or not v for v in request.values())):
            raise Rejected("finish needs an object {image ID or \"all\": call name} for this session's images")
        default = "c0" if self.stage in IMAGE_STAGES else "input"
        chosen = {}
        for unit in units:
            label = request.get(unit, request.get("all"))
            chosen[unit] = {"call": label or default, "source": "agent" if label else "default_unnamed"}
        problems = [p for unit, c in chosen.items() if (p := self.deliverable(unit, c["call"]))]
        if problems:
            raise Rejected(f"finish rejected, not deliverable: {problems[:20]}")
        return chosen

    def inspect(self, session: Session, items, index: int) -> str:
        if self.stage not in IMAGE_STAGES:
            raise Rejected("inspect is not available in this stage")
        if not isinstance(items, list) or not items or len(items) > MAX_INSPECT:
            raise Rejected(f"inspect needs a list of 1 to {MAX_INSPECT} items")
        folder = session.root / "inspect" / f"r{index:03d}"
        attach, labels, pending, size = [], [], [], 0
        for n, item in enumerate(items):
            if (not isinstance(item, dict) or set(item) - {"image", "call", "file"}
                    or item.get("image") not in session.units):
                raise Rejected("an inspect item is {\"image\": an image of this session, \"call\": \"input\" or a call, "
                               "\"file\": a path or path key of the call's result}")
            image, label = item["image"], str(item.get("call", ""))
            target = folder / f"{n:03d}.png"
            if label == "input":
                path = _render(within(self.prep / "public", self.rows()[image][INPUT_FILE[self.stage]]), target)
                attach.append(path)
                labels.append(f"image {len(attach)}: {image} input ({self.spec.input_channels[self.stage]})")
                size += path.stat().st_size
                continue
            record = self.calls.get(image, {}).get(label)
            if record is None:
                raise Rejected(f"{image} has no call {label}")
            relative = str(item.get("file", ""))
            keyed = record["view"].get(relative) if isinstance(record["view"], dict) else None
            if isinstance(keyed, str) and keyed in record["artifacts"]:
                relative = keyed
            relative = relative.replace("\\", "/")
            if (relative not in record["artifacts"] or not relative.startswith("tool/")
                    or not relative.lower().endswith(INSPECTABLE)):
                raise Rejected(f"{image}/{label}: {item.get('file')!r} is not a PNG, TIFF or JSON file the tool wrote")
            path = within(self.root / record["folder"], relative)
            if digest(path) != record["artifacts"][relative]:
                raise ValueError("Codex call artifact changed")
            if relative.lower().endswith(".json"):
                text = json.dumps(tool_view(read_json(path), self.root / record["folder"]), separators=(",", ":"))
                pending.append((f"{image}/{label}/{relative}", text))
                continue
            if relative.lower().endswith(".png"):
                target.parent.mkdir(parents=True, exist_ok=True)
                path = Path(shutil.copy2(path, target))
            else:
                path = _render(path, target)
            attach.append(path)
            labels.append(f"image {len(attach)}: {image}/{label}/{relative}")
            size += path.stat().st_size
        if size > EVIDENCE_POLICY["max_attachment_bytes"]:
            raise Rejected(f"inspect rejected: the attachments total {size} bytes, above "
                           f"{EVIDENCE_POLICY['max_attachment_bytes']}")
        limit = min(JSON_TEXT_LIMIT, INSPECT_JSON_BUDGET // max(1, len(pending)))
        texts = []
        for name, text in pending:
            if len(text) <= limit:
                note = ""
            elif limit < JSON_TEXT_LIMIT:
                note = f" (first {limit} of {len(text)} characters, shared inspect budget)"
            else:
                note = f" (first {limit} of {len(text)} characters)"
            texts.append(f"{name}{note}: {text[:limit]}")
        session.attach, session.labels, session.texts = attach, labels, texts
        return f"attached to the next round: {len(attach)} image(s), {len(texts)} JSON file(s)"

    def call_minutes(self) -> float:
        """A training call's time cap now: SHAPE's `RepresentationAgenticVerifier.max_minutes` at the search deadline."""
        from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier
        verifier = RepresentationAgenticVerifier()
        verifier.deadline = self.ssl_deadline
        return verifier.max_minutes()

    def run_dataset(self, specs) -> str:
        if not isinstance(specs, list) or not specs:
            raise Rejected("run needs a non-empty JSON list of call specs")
        if self.stage == "batch" and self.batch_rule != "RUN":
            raise Rejected(f"rejected: the batch calibration rule's decision is "
                           f"{self.spec.rules.get(self.batch_rule, self.batch_rule)}; no correction call can run")
        if len(specs) > self.cap:
            raise Rejected(f"rejected: {len(specs)} calls in one round, more than {self.cap}; nothing was run")
        if self.rounds_used("dataset") + 1 > MAX_TOOL_ROUNDS:
            raise Rejected(f"rejected: all {MAX_TOOL_ROUNDS} tool-call rounds of this stage are used; nothing was run")
        if self.stage == "embed" and self.call_minutes() <= 0:
            raise Rejected("rejected: the training window of this stage has ended (no minutes left for a call); "
                           "nothing was run")
        planned = []
        allowed = {"params", "parent"} if self.stage == "embed" else {"params"}
        for spec in specs:
            if not isinstance(spec, dict) or set(spec) - allowed:
                raise Rejected(f"a call spec is an object with keys {sorted(allowed)}")
            try:
                params = parameters(self.stage, spec.get("params", {}))
            except ValueError as error:
                raise Rejected(str(error)) from None
            parent = spec.get("parent", "") or ""
            if params.get("init_from") == "parent":
                record = self.calls["dataset"].get(parent) if parent != "input" else None
                if record is None or record["status"] != "complete" or not (
                        read_json(self.root / record["folder"] / "SSL_SELECTION.json").get("checkpoint")):
                    raise Rejected("init_from parent needs parent = a finished training call with a checkpoint")
            elif parent:
                raise Rejected("parent is used only with init_from parent")
            planned.append((params, parent))
        names, wave = [], f"d{self.rounds_used('dataset') + 1:03d}"
        for params, parent in planned:
            label = f"c{len(self.calls['dataset'])}"
            self.dataset_call(label, params, parent, wave)
            names.append(label)
        return f"ran {len(names)} call(s): {names}"

    # ---- sessions -------------------------------------------------------------------------------------------
    def handle(self, session: Session, decision: dict, index: int, last: bool) -> tuple[str, str, list]:
        """(outcome, message, image-stage calls to run in the next wave); every other action is applied here."""
        action = decision["action"]
        request = json.loads(decision["request_json"])
        if last and action != "finish":
            reason = f"the last round's action was {action}, not finish"
            self.unfinish(session, reason)
            return "unfinished", reason, []
        try:
            if action == "run":
                if self.stage in IMAGE_STAGES:
                    return "accepted", "", self.expand(request, session.units)
                return "executed", self.run_dataset(request), []
            if action == "inspect":
                return "attached", self.inspect(session, request, index), []
            if action == "handoff":
                if session.kind != "group":
                    raise Rejected("handoff is available only in a group session of an image stage")
                ids = list(session.units) if request == "all" else request
                if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                        or len(set(ids)) != len(ids) or any(i not in session.units for i in ids)):
                    raise Rejected("handoff needs \"all\" or a list of distinct image IDs of this group")
                self.per_image(ids)
                session.units = [u for u in session.units if u not in set(ids)]
                session.done = not session.units
                return "executed", f"{len(ids)} image(s) handed to per-image mode; their sessions finished", []
            chosen = self.choose(request, session.units)
            for unit, choice in chosen.items():
                self.deliveries[unit] = {**choice, "session": session.name}
            session.done = True
            unnamed = sorted(u for u, c in chosen.items() if c["source"] == "default_unnamed")
            return "finished", f"delivered; default call for unnamed: {unnamed[:20]}" if unnamed else "delivered", []
        except Rejected as error:
            if last:
                reason = f"the last round's finish was rejected: {error}"
                self.unfinish(session, reason)
                return "unfinished", reason, []
            return "rejected", str(error), []

    def group(self, kind: str, units: list[str]) -> None:
        session = Session(self.root / kind, kind, units)
        for index in range(self.max_rounds):
            last = index == self.max_rounds - 1
            decision, usage = self.ask(session, self.prompt(session, index, self.max_rounds), index)
            session.attach, session.labels, session.texts = [], [], []
            if decision is None and "malformed" in usage:
                self.record(session, index, None, usage, *self.malformed(session, usage["malformed"], last))
                continue
            if decision is None:
                self.unfinish(session, f"round {index + 1} failed: {usage['error']}")
                self.record(session, index, None, usage, "failed_round", usage["error"])
                return
            outcome, message, requests = self.handle(session, decision, index, last)
            if requests:
                names = self.wave(requests)
                outcome, message = "executed", f"ran {len(names)} call(s): {names[:40]}"
            self.record(session, index, decision, usage, outcome, message)
            if session.done:
                return

    def per_image(self, ids: list[str]) -> None:
        """One ephemeral session per image, advancing in waves: concurrent rounds, then one native worker for the
        wave's accepted run requests."""
        from .pipeline import _parallel
        sessions = [Session(self.root / "images" / image, "image", [image]) for image in ids]
        for index in range(self.image_rounds):
            active = [s for s in sessions if not s.done]
            if not active:
                return
            last = index == self.image_rounds - 1
            prompts = [self.prompt(s, index, self.image_rounds) for s in active]
            answers = _parallel(list(zip(active, prompts)), lambda pair: self.ask(pair[0], pair[1], index),
                                min(self.cores, len(active)))
            handled, requests = [], []
            for session, (decision, usage) in zip(active, answers):
                session.attach, session.labels, session.texts = [], [], []
                if decision is None and "malformed" in usage:
                    handled.append((session, decision, usage, *self.malformed(session, usage["malformed"], last), 0))
                    continue
                if decision is None:
                    self.unfinish(session, f"round {index + 1} failed: {usage['error']}")
                    handled.append((session, decision, usage, "failed_round", usage["error"], 0))
                    continue
                outcome, message, wanted = self.handle(session, decision, index, last)
                requests += wanted
                handled.append((session, decision, usage, outcome, message, len(wanted)))
            names = self.wave(requests) if requests else []
            for session, decision, usage, outcome, message, n in handled:
                if n:
                    own = [x for x in names if x.split("/", 1)[0] == session.units[0]]
                    outcome, message = "executed", f"ran {len(own)} call(s): {own}"
                self.record(session, index, decision, usage, outcome, message)

    # ---- prompts --------------------------------------------------------------------------------------------
    def facts(self, rounds: int) -> str:
        stage, n, spec = self.stage, len(self.images), self.spec
        lines = [f"You are the {ARM_NAMES[self.arm]} arm of the {spec.name} benchmark, stage {stage}, "
                 f"tool {TOOLS[stage][1]}.",
                 f"Task: {spec.tasks[stage]}",
                 spec.inputs[stage].format(n=n, cells=self.config.get("train_cells_per_site"), dtype=self.dtype(),
                                           rule=spec.rules.get(self.batch_rule, self.batch_rule))]
        if stage == "embed":
            from benchmarks.representation_RxRx1.runners.epoch_budget import fixed_line
            lines.append(fixed_line(self.config["ssl_epoch_budget"]))
        lines += ["Tuning the tool's parameters can improve the tool's performance.",
                  "The harness runs the tool calls you request with the tool's own code and returns what the tool "
                  "returns (the dict its execute() returns, arrays omitted, paths relative to the call's folder). "
                  "Filesystem, shell, web and MCP tools are disabled: this prompt and attached images are all you "
                  f"receive. {spec.withheld}"]
        seconds = self.config["deadline"] - time.time()
        if stage in IMAGE_STAGES:
            lines += [f"Budget: at most {MAX_TOOL_ROUNDS} tool-call rounds per image in this stage; each round may give "
                      f"an image at most {self.cap} call(s), and each call may change any number of parameters. c0, "
                      "the tool at its defaults, has run for every image and is its round 1. A run request is one "
                      "round for every image it names; a request that exceeds either limit for any image is "
                      "rejected as a whole. The arm's wall-clock deadline covers all remaining stages: "
                      f"{seconds:.0f} seconds remain for the entire arm.",
                      "One run request can process many images at once: all its calls run together in one worker, "
                      f"in parallel over {self.cores} threads (segmentation calls share one GPU model, one at a time). "
                      "Images that share parameters can be batched in one call spec.",
                      "Modes. Group mode: one session handles a set of images and sees their outputs as one compact "
                      "row per image. The group session starts with all images; its first round is the stage's "
                      "planning decision. Per-image mode: each image handed off gets its own session of at most "
                      f"{self.image_rounds} rounds that sees only that image's calls; up to {self.cores} per-image "
                      "sessions run concurrently, and their run requests of a round are executed together. Handed-off "
                      "images leave the group; their own sessions choose their delivery.",
                      "Actions (request_json is a JSON value encoded as a string):",
                      "- run: a list of call specs {\"images\": \"all\" or [image IDs], \"params\": {parameter: value}, "
                      "\"per_image\": {image ID: {parameter: value}}}. A spec adds one call for each image it names "
                      "(all = every image of this session); per_image values replace params for that image"
                      + (" (cell_overrides, {cell_id: {key: value}}, is a per-image parameter)" if stage == "crop" else "")
                      + ". Unspecified parameters use the tool's defaults. Calls are named c1, c2, ... per image.",
                      *([_cell_rules_text()] if stage == "crop" else []),
                      "- inspect: a list of at most 16 items {\"image\": image ID, \"call\": \"input\" or a call name, "
                      "\"file\": a path in the call's result, relative to its folder, or the result key holding it}. "
                      "input attaches the raw input image. PNG files are attached as written, TIFF files as a PNG "
                      "rendering (channels side by side, each scaled from its 1st to its 99th percentile), JSON files "
                      "written by the tool as text. Attachments appear in your next round only. Inspect is not a "
                      "tool call; it costs a round.",
                      "- handoff (group session only): a list of image IDs, or \"all\": they move to per-image mode "
                      "now; the group session continues with the rest.",
                      "- finish: an object {image ID: call name}; the key \"all\" names a call for every image not "
                      "named. Only completed calls can be delivered. An image left unnamed gets its c0."]
        else:
            lines += [f"Budget: at most {MAX_TOOL_ROUNDS} tool-call rounds in this stage (input is not a call); each round "
                      f"(run request) may hold at most {self.cap} call(s), and each call may change any number of "
                      "parameters. A request that exceeds either limit is rejected as a whole. The arm's wall-clock "
                      f"deadline covers all remaining stages: {seconds:.0f} seconds remain for the entire arm. Calls "
                      "run one after another."]
            if stage == "embed":
                from shape.agent.verification.verifiers.representation import RESERVE_MINUTES
                from .pipeline import SSL_SEARCH_SHARE
                lines.append("Training time: training calls must end by this stage's search deadline, "
                             f"{self.ssl_deadline - time.time():.0f} seconds from now (the share "
                             f"{SSL_SEARCH_SHARE} of the arm's remaining time when this stage started; the rest is "
                             "kept for frozen inference). Each training call's time cap is that deadline minus "
                             f"{RESERVE_MINUTES} minutes, now {self.call_minutes()} minutes; with no minutes left a "
                             "run request is rejected.")
            lines += ["Actions (request_json is a JSON value encoded as a string):",
                      "- run: a list of call specs {\"params\": {parameter: value}"
                      + (", \"parent\": a call name or \"\"}. With init_from \"parent\" the call resumes the parent "
                         "call's training state at its best validation point" if stage == "embed" else "}")
                      + ". Unspecified parameters use the tool's defaults. Calls are named c1, c2, ...",
                      "- finish: an object {\"all\": \"input\" or a call name}. Only completed calls can be delivered.",
                      "inspect and handoff are not available in this stage."]
        lines += [f"Every response: {RESPONSE_FORMAT}. A rejected request, or a reply that is not a valid decision, "
                  f"is not executed; the next round states why. This session has at most {rounds} rounds; the last "
                  "must be finish. A session that ends without a valid finish delivers the default call (c0, or "
                  "input) for its images.",
                  f"Registry: {json.dumps({k: asdict(v) for k, v in registry(stage).items()})}",
                  f"Tool CARD:\n{self.card}\n"]
        return "\n".join(lines)

    def table(self, units: list[str]) -> dict:
        """One compact row per image: [call, parameter set, status, the tool's scalar outputs...] per call."""
        params, columns, rows, paths = {}, [], {}, set()
        for image in units:
            for record in self.calls.get(image, {}).values():
                for key in _scalars(record["view"]):
                    if key not in columns:
                        columns.append(key)
                paths.update(k for k, v in record["view"].items() if isinstance(v, str) and v in record["artifacts"])
        for image in units:
            rows[image] = []
            for label, record in self.calls.get(image, {}).items():
                key = json.dumps(record["params"], sort_keys=True)
                params.setdefault(key, f"p{len(params)}")
                values = _scalars(record["view"])
                row = [label, params[key], record["status"], *[values.get(c) for c in columns]]
                if record["error"]:
                    row.append(str(record["error"])[:200])
                rows[image].append(row)
        return {"columns": ["call", "params", "status", *columns, "(error)"],
                "params": {name: json.loads(key) for key, name in params.items()},
                "path_keys": sorted(paths), "rows": rows}

    def table_text(self, units: list[str]) -> tuple[str, dict]:
        """(the group prompt's table line, {"table_mode", "table_chars"} for the round's ROUNDS.json entry): the full
        per-image table when its JSON fits PROMPT_TABLE_BUDGET characters, else the aggregated one."""
        table = self.table(units)
        full = _compact(table)
        if len(full) <= PROMPT_TABLE_BUDGET:
            return f"Calls of this group (compact): {full}", {"table_mode": "full", "table_chars": len(full)}
        aggregated = _compact(self.aggregate(table, units))
        text = (f"The per-image table of this group's calls would be {len(full)} characters, above the prompt-table "
                f"budget of {PROMPT_TABLE_BUDGET} characters, so you see an aggregated table (\"aggregated\": true) "
                "instead. \"calls\": per call (c0, c1, ...) \"images\" (how many images have it), \"ran\" (status "
                "complete), \"failed\", \"qc_flagged\" (complete, but the tool's own quality fields flag it: "
                f"{', '.join(f'{k} = {json.dumps(v)}' for k, v in QC_FLAG_FIELDS.items())}), \"param_sets\" (parameter "
                "set name: images, defined in \"params\"; the most frequent ones only, the rest counted in "
                "\"other_param_sets\"), \"stats\" (per scalar output column, over the group's images that have the "
                "call: n, mean, median, p10, p90, min, max; text columns: counts of the most frequent values) and "
                f"\"flagged_image_ids\" (flagged = failed or qc_flagged; the first {FLAGGED_IDS_PER_CALL} by image ID "
                "of \"flagged_total\"). \"flagged_rows\": the rows of the full table (same \"columns\", one row per "
                "call) of every image flagged in at least one call, by image ID, until "
                f"{FLAGGED_ROWS_BUDGET} characters; \"flagged_rows_omitted\" counts the flagged images not shown. "
                "Use inspect to see any single image's call outputs and files, and handoff to move images to "
                f"per-image mode, where a session sees every call of its image. Aggregated table: {aggregated}")
        return text, {"table_mode": "aggregated", "table_chars": len(aggregated), "table_full_chars": len(full)}

    def aggregate(self, table: dict, units: list[str]) -> dict:
        """The aggregated form of `table(units)`: per call its counts, parameter sets and column distributions, plus
        the full rows of the flagged images (up to FLAGGED_ROWS_BUDGET characters, by image ID)."""
        columns = table["columns"][3:-1]
        calls: dict[str, dict] = {}
        flagged: dict[str, dict[str, str]] = {}      # call -> {image: reason}
        for image in units:
            for row in table["rows"][image]:
                label = row[0]
                call = calls.setdefault(label, {"images": 0, "ran": 0, "failed": 0, "qc_flagged": 0,
                                                "sets": Counter(), "values": {c: [] for c in columns}})
                call["images"] += 1
                call["sets"][row[1]] += 1
                for column, value in zip(columns, row[3:3 + len(columns)]):
                    call["values"][column].append(value)
                reason = _flag(self.calls[image][label])
                call["ran"] += reason is None or reason == "qc"
                call["failed"] += reason == "failed"
                call["qc_flagged"] += reason == "qc"
                if reason:
                    flagged.setdefault(label, {})[image] = reason
        shown_sets: set[str] = set()
        out = {}
        for label, call in calls.items():
            top = sorted(call["sets"].items(), key=lambda kv: (-kv[1], kv[0]))[:PARAM_SETS_PER_CALL]
            shown_sets.update(name for name, _ in top)
            ids = sorted(flagged.get(label, {}))
            entry = {"images": call["images"], "ran": call["ran"], "failed": call["failed"],
                     "qc_flagged": call["qc_flagged"], "param_sets": dict(top)}
            if len(call["sets"]) > len(top):
                entry["other_param_sets"] = len(call["sets"]) - len(top)
            entry["stats"] = {c: d for c in columns if (d := _distribution(call["values"][c])) is not None}
            entry["flagged_total"], entry["flagged_image_ids"] = len(ids), ids[:FLAGGED_IDS_PER_CALL]
            out[label] = entry
        every = sorted({image for by_image in flagged.values() for image in by_image})
        names = {name: table["params"][name] for name in shown_sets}
        rows, used, omitted = {}, len(_compact(names)), 0
        for image in every:
            new = {r[1]: table["params"][r[1]] for r in table["rows"][image] if r[1] not in names}
            cost = len(_compact({image: table["rows"][image]})) + len(_compact(new))
            if omitted or used + cost > FLAGGED_ROWS_BUDGET:
                omitted += 1
                continue
            rows[image] = table["rows"][image]
            names.update(new)
            used += cost
        return {"aggregated": True, "columns": table["columns"], "params": dict(sorted(names.items(), key=lambda kv: int(kv[0][1:]))),
                "path_keys": table["path_keys"], "calls": out, "flagged_images": len(every),
                "flagged_rows": rows, "flagged_rows_omitted": omitted}

    @staticmethod
    def echo_decisions(decisions: list[dict]) -> list[dict]:
        """The previous decisions as echoed in a prompt: bounded by DECISION_ECHO_LIMIT per decision and
        DECISIONS_ECHO_BUDGET in all (the oldest cut first); decisions within the bounds are echoed as they are."""
        def cut(decision: dict, size: int) -> dict:
            out = dict(decision)
            for key in ("request_json", "message", "reason"):
                value = out.get(key)
                if isinstance(value, str) and len(value) > size:
                    out[key] = (f"{value[:size]}... [cut in this echo: first {size} of {len(value)} characters; "
                                "the full decision is recorded]")
            return out

        echoed = [cut(d, DECISION_ECHO_LIMIT // 4) if len(json.dumps(d)) > DECISION_ECHO_LIMIT else d
                  for d in decisions]
        for k in range(len(echoed)):
            if len(json.dumps(echoed)) <= DECISIONS_ECHO_BUDGET:
                break
            echoed[k] = cut(echoed[k], DECISION_ECHO_STUB)
        return echoed

    def prompt(self, session: Session, index: int, rounds: int) -> str:
        """The round's prompt. Its parts are bounded (see PROMPT_TABLE_BUDGET, INSPECT_JSON_BUDGET, the decision echo);
        above PROMPT_CHAR_LIMIT the inspected JSON texts are cut proportionally, and when that is not enough the
        prompt is refused here with the sizes of its parts, so the Codex CLI never rejects it."""
        named = [("facts", self.facts(rounds))]
        if session.kind == "group":
            named.append(("group", f"This group session: {len(session.units)} image(s); "
                                   f"{len(self.images) - len(session.units)} image(s) left the group for per-image mode."))
            text, session.table_info = self.table_text(session.units)
            named.append(("table", text))
        elif session.kind == "image":
            image = session.units[0]
            calls = {label: {"params": r["params"], "status": r["status"], "error": r["error"],
                             "seconds": r["seconds"], "result": r["view"],
                             "files": sorted(p for p in r["artifacts"]
                                             if p.startswith("tool/") and p.lower().endswith(INSPECTABLE))}
                     for label, r in self.calls.get(image, {}).items()}
            named.append(("calls", f"This session handles image {image} only. Its calls: {json.dumps(calls)}"))
        else:
            calls = {label: {"params": r["params"], "parent": r["parent"], "status": r["status"], "error": r["error"],
                             "seconds": r["seconds"], "result": r["view"]}
                     for label, r in self.calls.get("dataset", {}).items()}
            named.append(("calls", f"Calls of this stage: {json.dumps(calls)}"))
        named.append(("decisions",
                      f"Your previous decisions in this session: {json.dumps(self.echo_decisions(session.decisions))}"))
        if session.decisions and session.decisions[-1]["outcome"] == "malformed":
            previous = session.decisions[-1]
            named.append(("malformed", f"Your reply in round {previous['round'] + 1} was not a valid decision and was "
                                       f"not executed: {previous['message']}. Every response must be {RESPONSE_FORMAT}."))
        if session.labels:
            named.append(("attached", "Attached images: " + "; ".join(session.labels)))
        tail = [("round", f"Round {index + 1} of {rounds}. Seconds remaining for the entire arm: "
                          f"{self.config['deadline'] - time.time():.0f}.")]
        if index == rounds - 1:
            tail.append(("last", "This is the last round of this session: only finish is accepted."))

        def join(texts: list[str]) -> str:
            return "\n".join([t for _, t in named] + [f"Inspected JSON {x}" for x in texts] + [t for _, t in tail])

        limit = min(PROMPT_CHAR_LIMIT, ARM_PROMPT_LIMIT.get(getattr(self, "arm", None), PROMPT_CHAR_LIMIT))
        texts = list(session.texts)
        prompt = join(texts)
        if len(prompt) > limit and texts:
            total = sum(len(x) for x in texts)
            target = total - (len(prompt) - limit) - 200 * len(texts)      # room for the cut notes
            if target > 0:
                texts = [f"{x[:int(len(x) * target / total)]} [cut to fit the prompt limit {limit}: first "
                         f"{int(len(x) * target / total)} of {len(x)} characters]" for x in texts]
                prompt = join(texts)
        if len(prompt) > limit:
            sizes = {name: len(text) for name, text in named + tail}
            sizes["inspected_json"] = sum(len(x) for x in session.texts)
            raise RuntimeError(f"{ARM_NAMES.get(getattr(self, 'arm', None), 'Agent')} prompt of {len(prompt)} characters "
                               f"exceeds the harness limit {limit} in session {session.name}, round {index + 1}; "
                               f"sizes of its parts: {sizes}")
        session.prompt_chars = len(prompt)
        return prompt

    # ---- delivery -------------------------------------------------------------------------------------------
    def deliver(self) -> None:
        from .cli import child
        units = self.images if self.stage in IMAGE_STAGES else ["dataset"]
        if set(self.deliveries) != set(units):
            raise RuntimeError("Codex stage ended without a delivery for every image")
        for unit in units:
            problem = self.deliverable(unit, self.deliveries[unit]["call"])
            if problem:
                raise RuntimeError(f"Codex delivery failed: {problem}; recorded failure, not silently omitted")
            verify_call(self.root, self.calls[unit][self.deliveries[unit]["call"]])
        if self.stage == "segment":
            out = []
            for image in self.images:
                label = self.deliveries[image]["call"]
                d = self.calls[image][label]["delivery"]
                out.append({"image_id": image, "mask": d["mask"], "sha256": d["sha256"], "min_area": d["min_area"],
                            "n_cells": d["n_cells"], "call": label})
            write_json(self.run / "segmentation.json", out)
        elif self.stage == "crop":
            import tifffile
            from .pipeline import _parallel, raw_stack, retained_crops
            masks = {r["image_id"]: r for r in read_json(self.run / "segmentation.json")}

            def one(image):
                d = self.calls[image][self.deliveries[image]["call"]]["delivery"]
                crops = read_json(Path(d["metadata"]))["cell_metadata"]
                raw = raw_stack(within(self.prep / "public", self.rows()[image]["file"]), self.spec)
                mask = None
                if self.spec.mask_label:
                    if digest(Path(masks[image]["mask"])) != masks[image]["sha256"]:
                        raise ValueError(f"segmentation mask changed before crop delivery: {image}")
                    mask = tifffile.imread(masks[image]["mask"])
                return retained_crops(raw, self.rows()[image], crops, set(), self.run / "crop" / image,
                                      self.spec, mask)[0]

            cells = [c for kept in _parallel(self.images, one, self.cores) for c in kept]
            write_json(self.run / "cells.json", cells)
            write_json(self.run / "excluded_cells.json", [])
        else:
            chosen = self.root / self.calls["dataset"][self.deliveries["dataset"]["call"]]["folder"]
            for filename in OUTPUTS[self.stage]:
                shutil.copy2(chosen / filename, self.run / filename)
        write_json(self.root / "SELECTED.json", {
            "deliveries": self.deliveries, "total_tokens": sum(r["total_tokens"] for r in self.rounds),
            "n_calls": sum(len(c) for c in self.calls.values()), "n_rounds": len(self.rounds),
            "policy": self.backend.POLICY,
            "requested_model": self.config["model"], "response_model_attested": False,
            "evidence_policy": EVIDENCE_POLICY, **({} if self.reasoning is None else {"reasoning": self.reasoning})})
        if self.stage == "embed":
            # A trusted fresh process performs full-data inference only after selection is final.
            write_json(self.root / "INFER_CONFIG.json",
                       self.native_config(ssl_selection=read_json(self.run / "SSL_SELECTION.json")))
            child([sys.executable, "-m", self.spec.cli, "native-trial", "embed", "--prep", str(self.prep),
                   "--run", str(self.run), "--config", str(self.root / "INFER_CONFIG.json")],
                  self.run / "logs" / "frozen_inference.log",
                  self.config["deadline"] - time.time(), new_session=False)

    def execute(self) -> None:
        self.check_time()
        if self.stage in IMAGE_STAGES:
            self.wave([(image, {}) for image in self.images])
            self.group("group", self.images)
        else:
            self.dataset_call("input", {"adaptation_depth": "zero_shot"} if self.stage == "embed" else {})
            if self.stage == "batch":
                self.batch_rule = read_json(self.root / "calls" / "input" / "FEEDBACK.json")["admission"]
            self.group("dataset", ["dataset"])
        self.deliver()


def run_stage(stage, prep, run, config):
    StageRun(stage, prep, run, config).execute()
