"""The react arm: an LLM plus the cell-state analyzer on the shared ReAct runner (benchmarks/_shared/react_arm.py,
benchmarks/_shared/REACT_ARM_CONTRACT.md). This module is only the tool adapter and the run's records.

The unit is the dataset (the cell type's run prep), one ReAct session. The prompt (built by the shared runner) holds:
  TASK    the SHAPE arm's task text, verbatim (`shape_arm.task_brief` with SHAPE's BUDGET);
  INPUT   the facts SHAPE is given about its input (`input_text`): the crops and their metadata, that the default call
          is the tool at adaptation_depth zero_shot (SHAPE's trial 0, the frozen pretrained backbone), what an
          init_from value names, and the shared epoch-cap line (`epoch_budget.fixed_line`, also in SHAPE's prompt);
          then the PARAMETERS reference (`parameters_text`, user decision A16, below);
  TOOL   Cell_State_Analyzer_Tool and SHAPE's representation lever set (`FixedEpochVerifier.levers`: the CARD
          `## Levers` names without max_epochs, contract amendment A6), name, type and default only (`parameters`; a
          choice shows its values, init_from takes "pretrained" or a finished call id), and the runner's one factual
          line;
  PYTHON  the runner's python action (its own label-free signal: not a round; it reads `python_inputs`);
  BUDGET  shape.agent.verification.base.BUDGET rounds; the default call (trial 0) is not a round (as SHAPE counts
          its representation trial 0, the frozen input); at most `RepresentationAgenticVerifier.concurrency()` calls
          per round.
The default round's note is `TRIAL0_LABEL` (the zero-shot call the system ran, not the parameter values listed).
It never holds SHAPE's readouts or alarms, the guard, the lever-coverage rule, SHAPE's A12 / A13 / A15 / A16 search
rules (the stop refusal, the forced continuation, the plateau exploration, the basis and verdict rules, the expert
configuration: SHAPE's loop only) or a label.

Deviation from REACT_ARM_CONTRACT.md (RxRx1 only, user decision A16): the contract gives the agent the parameter table
with name, type and default only and no CARD text. This adapter adds the PARAMETERS reference to the INPUT section:
for each of the 15 levers of the table, its CARD Type, Range and Default and the CARD's text of what the lever does
in this tool (its `## Levers` entry after the field line), read from the CARD at run time so it cannot drift -- and
nothing else: no Family, no SHAPE readout definition, no search rule, no expert configuration, no recommendation.
Two adapter-side exceptions: the sentences that refer to SHAPE's prompt or readouts are cut by an explicit per-lever
filter (`SHAPE_CONTEXT`: view_pairing's val-proxy clause, clip_grad_norm's readouts sentence, koleo_weight's alarms
sentence; a filtered text no longer in the CARD raises), and init_from is described as this adapter runs it
(`INIT_FROM_ROW`: "pretrained" or a finished call id, whose saved best state the call resumes; its other parameters
are not inherited) in place of the CARD's SHAPE-move row. The CARD itself is unchanged. The shared runner's own
name / type / default table is unchanged.

Each observation is the tool's own returned dict (val_history, the val proxy and the training record included;
arrays dropped and paths made relative by the runner) and the image files the tool wrote (its loss_curve.png;
attached for a vision engine). A call is exactly a SHAPE call: Cell_State_Analyzer_Tool.execute through
`stage_map.invoke_managed_tool` on the SHAPE arm's tool input (the run prep's crops as crop_array, split_key "split")
with the kwargs SHAPE's verifier builds (`trial_params`: the levers, init_from as the resumed call's resume_best, the
call's own output_subdir, the split, max_minutes before the deadline) and the shared fixed epoch cap (A6: every
call, trial 0 included, at E; max_epochs is not a parameter, so a call that sets it is the runner's unknown-parameter
error). The values are not checked by SHAPE's move checks (user decision 2026-10-12, as in the other ReAct arms): they
go to the tool as given, a configuration run before runs again in its own output_subdir, and only an init_from that
names no finished call is refused. SHAPE's resource stop (`resource_stop`, the rule of
`RepresentationAgenticVerifier.recommend_wave`: no time left for a call) applies. A refused or failed call is an error
observation that uses its round. A call counts as finished when it returned no error and wrote an h5ad and a
resume_best.

Python steps (the shared runner's python action) read `python_inputs`: each finished call's embedding (the h5ad
the tool wrote; calls/default/ is the frozen trial 0, so the agent can compute its own frozen baseline) and the
label-free cell metadata (`PYTHON_META_KEYS`, written once to python_inputs/); never a file under the prep's private/
(the labels).

Basic LLM failures fail the arm (the contract): the shared runner raises on an exception of the fatal class
(`shape.engine.errors.is_fatal_llm_error`); this branch's engines return such a failure as an error reply
({"error": "<exception type>", ...}) instead, so `FatalReplies` turns a reply naming a fatal type (`FATAL_REPLIES`)
into REACT_LLM_FATAL (a SystemExit, never logged as a reply: a resume asks that request live again). Every other
failed reply is an error observation that uses its round.

Delivery: the agent's finish (default = frozen, else the call's resume_best); a session without a valid finish
delivers the default call (frozen), case `default_session_unfinished`. score.py embeds every FULL cell with the
delivered state afterwards, as for every analyzer arm.

On the VM (contract amendment A8) the reasoner requests are recorded (logs/reasoner.jsonl, resume.RecordingEngine)
and every analyzer call goes through resume.CallCache; `resume=True` resumes an unfinished run in place by replay:
the session runs again from trial 0, the logged replies are served in order (their prompts compared with the clock
line and every python step's observation masked, `resume.react_prompt_check`; a replayed python step runs again),
finished calls are served from their records, then it goes live.

A run directory (<results root>/representation_RxRx1/react_<cell type>_<commit8>_<UTC>_<job id>/) holds:
  RUN_META.json          provenance, tool_budget (= BUDGET), round_cap, card_sha256 (the lever set's source),
                         training_budget, fixed_levers, the sha256 of this module and of the shared runner
  AGENT_RUN.json         timing, the case, rounds, calls, reasoner tokens, the runner's totals, write errors
  react/<cell type>/     the runner's TRACE.jsonl, CALLS.json, SELECTED.json, python/py_<n>.py (and .out), workspace/;
                         react/REACT_SUMMARY.json
  python_inputs/         cell_metadata.csv (`PYTHON_META_KEYS`), read by the python steps
  logs/input_trial.json  trial 0 (the default call): its kwargs and the tool's facts (control.TOOL_FACTS)
  logs/tool_calls.jsonl  one start / end record per agent call that reached the tool (the SHAPE arm's format)
  logs/agent_log.jsonl   the arm's events and the verifier's log
  logs/reasoner.jsonl    VM only: every request and its reply (the replay source)
  cache/                 the tool's outputs (cell_state_analysis/ssl_calls/<call>/)
  submission/decision.json  use ("frozen" or the delivered call), its checkpoint (resume_best), the case
"""
from __future__ import annotations

import inspect
import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[3]
RECORDS_DIR = "react"              # the shared runner's records: <run>/react/<cell type>/
PYTHON_META_KEYS = ("cell_id", "plate", "experiment", "well", "site", "split")   # label-free; no siRNA, no label
# the default round's note: the call the system ran (frozen_params), not the parameter values the table lists
TRIAL0_LABEL = ('the system\'s call of the tool with adaptation_depth "zero_shot" and split_key "split": the frozen '
                "pretrained backbone, no training (the parameter values listed for this call are not used)")
# the engines' error replies of the fatal class (shape.engine.errors: authentication / permission, the model not
# found, the endpoint unreachable; a timeout is not one), by the exception type the engine names
FATAL_REPLIES = ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "APIConnectionError")


class FatalReplies:
    """The reasoner engine, with an error reply of the fatal class (`FATAL_REPLIES`) raised as REACT_LLM_FATAL."""

    def __init__(self, engine):
        self.engine = engine

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def generate(self, content, response_format=None, **kwargs):
        reply = self.engine.generate(content, response_format=response_format, **kwargs)
        if isinstance(reply, dict) and reply.get("error") in FATAL_REPLIES and not reply.get("content"):
            raise SystemExit(f"REACT_LLM_FATAL: the reasoner engine returned {reply['error']}: "
                             f"{str(reply.get('message') or '')[:300]}")
        return reply


def resource_stop(verifier) -> Optional[str]:
    """SHAPE's resource stop (`RepresentationAgenticVerifier.recommend_wave`): "deadline" when the time a call may
    take (`max_minutes`) is at most 0 or below the shortest non-training time of a finished call (its seconds minus its
    training seconds; trial 0's until a call has finished), else None."""
    left = verifier.max_minutes()
    if left is not None:
        shortest = min((m for i, m in verifier._overhead.items() if i > 0), default=verifier._overhead.get(0))
        if left <= 0 or (shortest is not None and left < shortest):
            verifier.log("deadline", text=f"{left} minute(s) left for a call; the shortest non-training time of a "
                                          f"finished call: {shortest} minute(s)")
            return "deadline"
    return None


def input_text(cell_type: str, fixed: str, parameters: str = "") -> str:
    """The INPUT section: the facts SHAPE is given about its input, nothing SHAPE computes; then `parameters`, the
    PARAMETERS reference (`parameters_text`, A16)."""
    return (f"The tool's input: the RxRx1 {cell_type} run-prep crops (train and val wells) as one packed array; every "
            "crop carries its plate, plate-qualified well, site, experiment and split (train / val); no label.\n"
            "Call default is run by the system before round 1 and is not a round: the tool with adaptation_depth "
            "zero_shot (the frozen pretrained backbone, no training). Delivering default delivers that frozen "
            "embedding.\n"
            'init_from: "pretrained" starts a call from the pretrained backbone; the id of a finished call starts it '
            "from that call's saved training state.\n"
            f"{fixed}" + (f"\n\n{parameters}" if parameters else ""))


PARAMETER_FIELDS = ("Type", "Range", "Default")    # the CARD fields the PARAMETERS reference shows (no Family)
# init_from as this adapter runs it (`AnalyzerReact._move`; the shared runner fills a parameter not given with its
# default), in place of the CARD's row, which describes SHAPE's moves
INIT_FROM_ROW = ("- init_from | Type: choice | Range: pretrained | <a finished call id> | Default: pretrained\n"
                 "  pretrained starts from the pretrained backbone; a finished call id resumes that call's saved best "
                 "state (its other parameters are not inherited: a parameter not given takes its default)")
# the CARD lever text that refers to SHAPE's prompt or readouts, cut from the reference: lever -> (text, replacement);
# a text no longer found in the CARD raises (the CARD changed: review the filter)
SHAPE_CONTEXT = {
    "view_pairing": ("; the val proxy measures that kind of within-well likeness on val wells, which never train.",
                     "."),
    "clip_grad_norm": (" The call logs its gradient norms; they are not among the readouts shown here.", ""),
    "koleo_weight": (" It acts on the embedding the alarms are computed on, so effective_rank can move with it "
                     "whatever else changes.", ""),
}


def lever_texts(verifier) -> dict[str, str]:
    """Each lever's text in the CARD's `## Levers` section as written (max_epochs included): the lines after its
    `### <lever>` heading and its field line, up to the next heading, joined into one paragraph."""
    from shape.agent.verification.base import AgenticVerifier

    section = AgenticVerifier._card_section(verifier)      # the CARD's section itself, not the benchmark's rendering
    parts = re.split(r"^### ([a-z][a-z0-9_]*)[ \t]*\n", section, flags=re.MULTILINE)
    return {name: " ".join(line.strip() for line in body.strip().splitlines()[1:] if line.strip())
            for name, body in zip(parts[1::2], parts[2::2])}


def parameters_text(verifier) -> str:
    """The PARAMETERS reference (user decision A16, this adapter only): for each parameter of the tool's table
    (`verifier.levers`, the 15 levers), its CARD Type, Range and Default and the CARD's text of what it does in this
    tool (`lever_texts`) without its sentences on SHAPE's prompt or readouts (`SHAPE_CONTEXT`); init_from as this
    adapter runs it (`INIT_FROM_ROW`); no Family and nothing else."""
    texts = lever_texts(verifier)
    for lever, (text, replacement) in SHAPE_CONTEXT.items():
        if text not in texts.get(lever, ""):
            raise ValueError(f"the CARD text of {lever} no longer holds {text!r}: review react_arm.SHAPE_CONTEXT")
        texts[lever] = texts[lever].replace(text, replacement)
    rows = []
    for lever in verifier.levers:
        if lever == "init_from":
            rows.append(INIT_FROM_ROW)
            continue
        fields = verifier.lever_fields.get(lever) or {}
        line = " | ".join(f"{k}: {fields[k]}" for k in PARAMETER_FIELDS if k in fields)
        rows.append(f"- {lever} | {line}\n  {texts.get(lever, '')}".rstrip())
    return ("PARAMETERS (the tool's documentation of each parameter of the TOOL section: its type, range and default, "
            "and what it does in this tool):\n" + "\n".join(rows))


def _type_text(verifier, lever: str) -> str:
    """A lever's type as the prompt shows it: the CARD Type; a choice with its values; init_from with a call id."""
    if lever == "init_from":
        return "choice: pretrained | <a finished call id>"
    fields = verifier.lever_fields.get(lever) or {}
    kind, bounds, nullable = verifier.lever_spec(lever)
    text = str(fields.get("Type") or kind)
    if kind == "choice" and bounds:
        text = "choice: " + " | ".join(str(b) for b in bounds) + (" | null" if nullable else "")
    return text


class AnalyzerReact:
    """The run and the shared runner's tool adapter (ToolAdapter) for Cell_State_Analyzer_Tool."""

    unit_kind = "dataset"

    def __init__(self, root: Path, run_dir: Path, *, cell_type: str, model: str, engine, resume: bool = False,
                 budget: Optional[int] = None):
        import shape
        from shape.agent.verification.base import BUDGET, available_cores
        from shape.agent.verification.verifiers.representation import RepresentationTask
        from shape.engine import usage

        from benchmarks.representation_RxRx1.runners import control, resume as resume_mod, shape_arm
        from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier, fixed_line, shared_budget
        from benchmarks.representation_RxRx1.runners.prepare import load, sha256_file, tool_input

        if engine is None:
            raise SystemExit(f"NOT_READY: no reasoner engine for {model}; the react arm does not start without one")
        if REPO not in Path(shape.__file__).resolve().parents:
            raise SystemExit(f"`import shape` resolves to {shape.__file__}, not inside {REPO}")
        self.state = resume_mod.Resume.open(run_dir, "react") if resume else None
        if self.state is None and (run_dir / "RUN_META.json").exists():
            raise SystemExit(f"{run_dir} holds a run already; use a new run directory")
        self.recording = resume_mod.on_vm()
        self.control, self.run_dir, self.cell_type, self.model = control, run_dir, cell_type, model
        self.private_roots = [Path(root) / "private"]
        self._python_meta: Optional[Path] = None
        self.tool_name = control.TOOL_NAME
        self.summary, self.cells = load(root)
        self.provenance = {**control.provenance(), "prep_cells_sha256": self.summary["cells_sha256"]}
        control_path = root / control.CONTROL_DIR / "CONTROL.json"
        if (control_path.exists()
                and json.loads(control_path.read_text(encoding="utf-8"))["provenance"] != self.provenance):
            raise SystemExit(f"CONTROL_STALE: {control_path} was computed with other code or another prep")
        for sub in ("logs", "submission", "cache"):
            control.fs_retry((run_dir / sub).mkdir, parents=True, exist_ok=True)
        self.started = time.time()
        self.tokens_before = usage.snapshot()
        self.write_errors: list = []
        self.budget = budget or BUDGET                       # SHAPE's BUDGET (a test sets a smaller SHAPE run's)
        self.epoch_budget = shared_budget(root, self.summary, self.cells)
        self.fixed = {"max_epochs": self.epoch_budget["max_epochs"]}
        self.fixed_text = fixed_line(self.epoch_budget)
        self.crop_array, self.meta = tool_input(self.summary, self.cells)
        self.base_kwargs = {"crop_array": self.crop_array, "cell_metadata": self.meta,
                            "query_cache_dir": str(run_dir / "cache")}
        self.brief = shape_arm.task_brief(self.summary, cell_type, self.budget)
        self.log = shape_arm.AgentLog(run_dir / "logs" / "agent_log.jsonl")
        self.task = RepresentationTask(cores=available_cores(), gpus=shape_arm.visible_gpus(),
                                       deadline=resume_mod.deadline(), brief=self.brief)
        # SHAPE's verifier, no engine: the lever set, the call kwargs and the resource stop (no value checks)
        self.verifier = FixedEpochVerifier(self.epoch_budget, log=self.log)
        self.verifier.begin(self.task)
        self.available = set(inspect.signature(control.analyzer().execute).parameters)
        self.attempts: dict[str, tuple] = {}                 # finished call -> (parent, configuration, None)
        self.facts: dict[str, dict] = {}                     # "default" / finished call -> the tool's facts
        self.started_calls = 0                               # agent calls that reached the tool
        self.config_keys: set[str] = set()                   # the configurations of the agent calls started
        self.delivered: Optional[str] = None
        self.call_cache = (resume_mod.CallCache(run_dir / "cache", self.state, self.write_errors)
                           if self.recording else None)
        self.engine = FatalReplies(engine)
        if self.recording:
            recorder = resume_mod.RecordingEngine(self.engine, run_dir / "logs" / "reasoner.jsonl", self.write_errors,
                                                  masked=resume_mod.react_masked_digest)
            self.engine = recorder
            if self.state is not None:
                old = run_dir / RECORDS_DIR                   # the runner's records start again (kept, moved aside)
                if old.exists():
                    control.fs_retry(os.rename, old, run_dir / self.state.entry["logs_moved_to"] / RECORDS_DIR)
                replies = resume_mod.shape_replies(self.state.source / "reasoner.jsonl")
                recorder.count = len(replies)
                self.engine = resume_mod.ReplayEngine(
                    replies, recorder, self.state, prompt_check=resume_mod.react_prompt_check,
                    echo=lambda r, content: recorder.append({
                        **{k: v for k, v in r.items() if k != "time"},
                        "prompt_sha256": resume_mod._prompt_digest(content),
                        "prompt_masked_sha256": resume_mod.react_masked_digest(content),
                        "replayed": self.state.number}))
        from benchmarks._shared import react_arm as shared

        deadline = self.task.deadline
        meta = {
            "commit": control.git_commit(), "expected_commit": os.environ.get("RXRX_EXPECTED_COMMIT"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "cell_type": cell_type,
            "agent": "react", "model": model, "max_rounds": self.budget, "tool_budget": self.budget,
            "round_cap": self.round_cap(), "default_counts_as_round": self.default_counts_as_round(),
            "fixed_levers": dict(self.fixed), "training_budget": self.epoch_budget,
            "cores": self.task.cores, "gpus": self.task.gpus, "split_key": self.task.split_key,
            "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline)) if deadline else None,
            "crop_array": self.crop_array, "card_sha256": control.card_sha256(), "provenance": self.provenance,
            "shape_file": str(Path(shape.__file__).resolve()),
            "react_arm_sha256": sha256_file(Path(__file__)), "shared_react_sha256": sha256_file(Path(shared.__file__)),
            "parameters": [p.__dict__ for p in self.parameters()],
            "reasoning_effort": os.environ.get("SHAPE_REASONING_EFFORT"),
            "max_completion_tokens": getattr(self.engine, "max_tokens", None),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            **resume_mod.platform_keys()}
        if self.state is None:
            control.write_json(run_dir / "RUN_META.json", meta)
        else:
            self.state.write_meta(meta)
        self.log("system", text=f"react arm (shared runner), model={model}, budget={self.budget} round(s), "
                                f"cap={self.round_cap()}, cores={self.task.cores}, gpus={self.task.gpus}, "
                                f"crops={len(self.meta)}, fixed={self.fixed}")

    # ---- the ToolAdapter -------------------------------------------------------------------------------------
    trial0_label = TRIAL0_LABEL

    def parameters(self) -> list:
        from benchmarks._shared.react_arm import Param

        defaults = self.verifier.defaults()
        return [Param(lever, _type_text(self.verifier, lever), defaults.get(lever)) for lever in self.verifier.levers]

    def units(self) -> list[str]:
        return [self.cell_type]

    def round_cap(self) -> int:
        return self.verifier.concurrency()

    def default_counts_as_round(self) -> bool:
        return False                                     # SHAPE's representation trial 0 (the input) is not a round

    def deliver(self, unit: str, call_id: str) -> None:
        self.delivered = call_id

    def python_inputs(self, unit: str) -> dict:
        """The files a python step may read: inputs/cell_metadata.csv (`PYTHON_META_KEYS`, one row per crop in the
        tool's input order) and calls/<call id>/<the call's h5ad> for "default" and every finished call."""
        import csv

        from benchmarks._shared.react_arm import assert_no_private

        if self._python_meta is None:
            path = self.run_dir / "python_inputs" / "cell_metadata.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh, lineterminator="\n")
                writer.writerow(PYTHON_META_KEYS)
                writer.writerows([m.get(k) for k in PYTHON_META_KEYS] for m in self.meta)
            self._python_meta = path
        paths = {"inputs/cell_metadata.csv": self._python_meta}
        for call_id, facts in list(self.facts.items()):
            adata = facts.get("adata_path")
            if adata and Path(adata).is_file():
                paths[f"calls/{call_id}/{Path(adata).name}"] = Path(adata)
        assert_no_private(paths.values(), self.private_roots)
        return paths

    def call(self, unit: str, params: dict, call_id: str) -> dict:
        from benchmarks._shared.react_arm import DEFAULT_CALL
        from shape.agent.verification.verifiers.representation import frozen_params

        if call_id == DEFAULT_CALL:                       # trial 0: SHAPE's frozen input call
            return self._run(call_id, {**frozen_params(self.task), **self.fixed}, None, {})
        stop = resource_stop(self.verifier)
        if stop:
            return {"output": None, "images": [], "error": (
                f"no call runs: {self.verifier.max_minutes()} minute(s) are left for a call before the job's limit, "
                "less than the shortest non-training time of a finished call")}
        try:
            move = self._move(params)
        except ValueError as error:
            return {"output": None, "images": [], "error": f"invalid parameters: {error}"}
        tool_params = {**self.verifier.trial_params(move["params"]), **self.fixed}
        key = self.verifier.config_key(move["params"])
        if key in self.config_keys:                       # a repeated configuration runs again, in its own dir
            tool_params["output_subdir"] = f"{tool_params['output_subdir']}_{call_id}"
        self.config_keys.add(key)
        return self._run(call_id, tool_params, move["parent"], move["param_delta"])

    def _move(self, params: dict) -> dict:
        """The call's configuration with no SHAPE check (user decision 2026-10-12: as in the other ReAct arms, the
        values go to the tool as given and the tool reports what it cannot use): init_from names a finished call,
        whose saved training state at its best val point the call loads (`init_checkpoint`, A7), or "pretrained";
        every other parameter is passed on as the runner resolved it (a left-out parameter at its default). Raises
        ValueError only when init_from names no finished call with a saved state (the call id cannot be resolved)."""
        from shape.agent.verification.base import INPUT

        verifier = self.verifier
        init = params.get("init_from")
        parent = INPUT
        config = {**{k: v for k, v in params.items() if k != "init_from"}, "init_from": "pretrained",
                  "init_checkpoint": None}
        if init not in (None, "", "pretrained"):
            resume = verifier._resume.get(verifier._labels.get(str(init), -1))
            if str(init) not in self.attempts or not resume or not os.path.exists(resume):
                raise ValueError(f'init_from must be "pretrained" or a finished call id, one of '
                                 f"{sorted(self.attempts)}")
            parent = str(init)
            config.update(init_from="parent", init_checkpoint=resume)
            verifier._parent_keys[resume] = verifier.config_key(self.attempts[parent][1])
        base = verifier._params(self.attempts, parent)
        delta = {k: v for k, v in params.items() if k != "init_from" and base.get(k) != v}
        if parent != INPUT:
            delta["init_from"] = "parent"                 # as SHAPE's ledger records a resumed call
        return {"parent": parent, "params": config, "param_delta": delta}

    def _run(self, call_id: str, tool_params: dict, parent: Optional[str], delta: dict) -> dict:
        """One analyzer call exactly as SHAPE runs it; its ledger records; the tool's own result as the output."""
        from shape.agent.verification.stage_map import invoke_managed_tool, map_tool_params
        from shape.utils.schemas import is_failed_result

        control, verifier = self.control, self.verifier
        trial0 = parent is None
        index = 0 if trial0 else int(call_id.rsplit("_", 1)[1])
        kwargs = map_tool_params(control.TOOL_NAME, tool_params, self.available)
        if not trial0:
            self.started_calls += 1
            self._append({"event": "start", "call": call_id, "parent": parent,
                          "init": verifier.start_state(parent, tool_params), "set": delta, "kwargs": kwargs,
                          "not_forwarded": {k: v for k, v in tool_params.items() if k not in kwargs}})
        started = time.time()

        def invoke():
            return invoke_managed_tool(control.analyzer(), control.TOOL_NAME,
                                       {**self.base_kwargs, "cell_metadata": [dict(m) for m in self.meta]},
                                       tool_params)

        try:
            result = invoke() if self.call_cache is None else self.call_cache.run(tool_params, invoke)
        except Exception as error:  # noqa: BLE001 -- a failed call is an observation, never delivered
            result = {"error": f"{type(error).__name__}: {error}"}
        seconds = round(time.time() - started, 1)
        if self.call_cache is not None and self.call_cache.last_served:      # a served call: its logged seconds
            seconds = self.call_cache.seconds_of(self.call_cache.last_served)
        elif self.call_cache is not None:
            self.call_cache.note_outer(tool_params, seconds)
        result = result if isinstance(result, dict) else {"error": f"the tool returned {type(result).__name__}"}
        error = None
        if is_failed_result(result) or result.get("error"):
            error = str(result.get("error") or "the call failed")
        elif not result.get("adata_path"):
            error = "the call wrote no embedding (h5ad)"
        elif not trial0 and (not result.get("resume_best") or not Path(result["resume_best"]).exists()):
            error = "the call wrote no resume_best (no training state to resume or deliver)"
        facts = {k: result.get(k) for k in control.TOOL_FACTS}
        training = result.get("training") or {}
        if error is None:
            self.facts[call_id] = facts
            if isinstance(result.get("seconds"), (int, float)) and isinstance(training.get("train_seconds"),
                                                                              (int, float)):
                verifier._overhead[index] = (result["seconds"] - training["train_seconds"]) / 60
        if trial0:
            control.write_json(self.run_dir / "logs" / "input_trial.json", {
                "call": "input", "params": tool_params, "kwargs": kwargs, "status": "done" if error is None
                else "crashed", "error": error, "tool": facts, "seconds": seconds})
        elif error is None:
            self.attempts[call_id] = (parent, verifier.trial_config(tool_params), None)
            verifier._labels[call_id], verifier._resume[index] = index, result["resume_best"]
            verifier._names[index] = call_id
            self._append({"event": "end", "call": call_id, "status": "done", "tool": facts, "seconds": seconds})
        else:
            verifier.failed.append({"call": call_id, "parent": parent, "params": verifier.trial_config(tool_params),
                                    "param_delta": dict(delta), "error": error})
            self._append({"event": "end", "call": call_id, "status": "crashed", "error": error, "tool": facts,
                          "seconds": seconds})
        self.log("call", text=json.dumps({"call": call_id, "parent": parent, "set": delta, "error": error,
                                          "seconds": seconds}, default=str))
        images = [p for p in result.get("deliverables") or []
                  if isinstance(p, str) and p.lower().endswith(".png") and Path(p).is_file()]
        return {"output": None if result.get("error") else result, "images": images, "error": error}

    def _append(self, record: dict) -> None:
        from benchmarks.representation_RxRx1.runners.resume import append_durable

        path = self.run_dir / "logs" / "tool_calls.jsonl"
        try:
            if self.recording:                                # A8 on the VM: the records survive a power cut
                append_durable(path, {"time": time.time(), **record})
            else:
                self.control.append_jsonl(path, {"time": time.time(), **record})
        except OSError as error:                              # after fs_retry: the run goes on, the failure recorded
            self.write_errors.append({"time": time.time(), "what": path.name,
                                      "error": f"{type(error).__name__}: {error}"})

    # ---- the run ---------------------------------------------------------------------------------------------
    def run(self) -> dict:
        from shape.engine import usage

        from benchmarks._shared.react_arm import DEFAULT_CALL, SOURCE_UNFINISHED, run_react
        from benchmarks.representation_RxRx1.runners import resume as resume_mod

        out = run_react(self, task_text=self.brief, input_text=input_text(self.cell_type, self.fixed_text,
                                                                         parameters_text(self.verifier)),
                        engine=self.engine, out_dir=self.run_dir / RECORDS_DIR, workers=1,
                        deadline=self.task.deadline, budget=self.budget)
        selected = out["selected"][self.cell_type]
        call = self.delivered or selected["call_id"]
        use = "frozen" if call == DEFAULT_CALL else call
        if selected["source"] == SOURCE_UNFINISHED:
            case = SOURCE_UNFINISHED
        else:
            case = "finish_input" if use == "frozen" else "finish"
        facts = self.facts.get(call, {})
        final = {"use": use, "provisional": False, "checkpoint": None if use == "frozen" else facts.get("resume_best"),
                 "adata_path": facts.get("adata_path"), "status": "delivered", "case": case,
                 "reason": f"{selected['source']} {call} after {selected['rounds']} round(s) "
                           f"(stop: {selected['stop']})"}
        self.control.write_json(self.run_dir / "submission" / "decision.json", final)
        tokens = {k: v - self.tokens_before.get(k, 0) for k, v in usage.snapshot().items() if isinstance(v, int)}
        if self.state is not None:
            self.state.close()
        ended = time.time()
        meta = self.state.meta if self.state is not None else None
        self.control.write_json(self.run_dir / "AGENT_RUN.json", {
            "agent": "react", "model": self.model, "started": self.started, "ended": ended,
            "wall_hours_used": resume_mod.active_hours(meta, self.started, ended), "stopped_because": case,
            "stop_reason": selected["stop"], "n_rounds": selected["rounds"], "max_rounds": self.budget,
            "tool_budget": self.budget, "round_cap": out["round_cap"], "tool_calls": self.started_calls,
            "calls_finished": len(self.attempts), "requests": selected["tokens"]["replies"], "tokens": tokens,
            "react": out["totals"], "write_errors": self.write_errors + self.log.write_errors,
            **({"resumes": resume_mod.resumes_record(meta)} if self.state is not None else {})})
        self.log("final", text=json.dumps(final, default=str))
        return final


def run(root: Path, run_dir: Path, *, cell_type: str, model: str, engine, resume: bool = False,
        budget: Optional[int] = None) -> dict:
    """One react run on the prepared cell type under `root` into the new directory `run_dir`; `resume`: the
    unfinished run in `run_dir`, resumed in place by replay (A8); `budget`: rounds, SHAPE's BUDGET unless given."""
    return AnalyzerReact(root, run_dir, cell_type=cell_type, model=model, engine=engine, resume=resume,
                         budget=budget).run()
