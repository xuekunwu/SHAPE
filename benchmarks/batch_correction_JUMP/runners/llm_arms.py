"""GPT-only and ReAct arms for the JUMP batch-correction benchmark.

Neither arm uses SHAPE's BatchCorrectionVerifier. GPT-only receives no domain tool and selects one generic
batch-centering transform. ReAct is the shared ReAct arm (`benchmarks/_shared/react_arm.py`,
`benchmarks/_shared/REACT_ARM_CONTRACT.md`): an LLM plus `Batch_Corrector_Tool`, called as the SHAPE arm calls
it. This module adds only its tool adapter (`BatchToolAdapter`): the SHAPE verifier's lever set (name, type,
default), SHAPE's per-round cap, the uncorrected input as trial 0 (not a round, as SHAPE's `admission_on_input`),
and as each observation the tool's own returned dict plus the calibration scores SHAPE's verifier gets.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from benchmarks.batch_correction_JUMP.runners.benchmark_contract import LUNA_MAX_TOKENS, LUNA_MODEL


class GPTPlan(BaseModel):
    reasoning: str
    action: Literal["identity", "batch_center"]
    strength: float = Field(ge=0.0, le=1.0)


def _git_commit() -> str | None:
    try:
        repo = Path(__file__).resolve().parents[3]
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def _parse_response(response, model):
    if not isinstance(response, dict) or response.get("error"):
        raise RuntimeError(f"Luna call failed: {response}")
    content = response.get("content")
    if isinstance(content, model):
        return content
    if isinstance(content, str):
        return model.model_validate_json(content)
    return model.model_validate(content)


def _ask(engine, prompt: str, model, errors: list):
    """One Luna question with the SHAPE reasoner's retry policy (REASONER_ATTEMPTS tries,
    REASONER_RETRY_SECONDS apart): the parsed answer, or None when every try fails (with
    SHAPE_REQUIRE_AGENTIC=1 the last error is raised, as in the SHAPE arm). `errors` collects the
    failed tries."""
    from shape.agent.verification import base

    for attempt in range(1, base.REASONER_ATTEMPTS + 1):
        try:
            return _parse_response(engine.generate(prompt, response_format=model,
                                                   max_tokens=LUNA_MAX_TOKENS), model)
        except Exception as error:
            errors.append(f"try {attempt}: {type(error).__name__}: {error}")
            if attempt < base.REASONER_ATTEMPTS:
                time.sleep(base.REASONER_RETRY_SECONDS)
            elif os.getenv("SHAPE_REQUIRE_AGENTIC") == "1":
                raise
    return None


def _usage_delta(before: dict, after: dict) -> dict:
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if isinstance(value, int) and isinstance(before.get(key, 0), int)
    } | {
        "usage_complete": after.get("usage_complete"),
        "usage_observed_fields": after.get("usage_observed_fields", []),
    }


def _matrix_summary(features: np.ndarray, batch) -> dict:
    values = np.asarray(features, dtype=float)
    groups = np.asarray(batch).astype(str)
    mean = values.mean(axis=0)
    total = float(((values - mean) ** 2).sum()) + 1e-12
    between = sum(
        int((groups == label).sum())
        * float(((values[groups == label].mean(axis=0) - mean) ** 2).sum())
        for label in sorted(set(groups))
    )
    return {
        "n_profiles": int(len(values)),
        "n_features": int(values.shape[1]),
        "batch_counts": {label: int((groups == label).sum()) for label in sorted(set(groups))},
        "batch_centroid_variance_ratio": between / total,
    }


def _batch_center(features: np.ndarray, batch, strength: float) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    groups = np.asarray(batch).astype(str)
    corrected = values.copy()
    global_mean = values.mean(axis=0)
    for label in sorted(set(groups)):
        rows = groups == label
        corrected[rows] -= float(strength) * (values[rows].mean(axis=0) - global_mean)
    return corrected


def run_gpt_only(prep: Path, run_dir: Path, *, engine, model: str = LUNA_MODEL,
                 require_tokens: bool = True) -> dict:
    """One Luna decision followed by a generic transform; no domain tool is available."""
    from shape.engine import usage

    from benchmarks.batch_correction_JUMP.runners.prepare import load

    if engine is None:
        raise RuntimeError("GPT-only arm requires a Luna engine")
    summary, meta, features = load(prep)
    batch_key = summary["scenario"]["batch_key"]
    batch = meta[batch_key].astype(str).to_numpy()
    started = time.time()
    matrix_summary = _matrix_summary(features, batch)
    prompt = (
        "You are the GPT-only arm of a batch-correction benchmark. You have no Harmony, "
        "Scanorama, SHAPE, metric, or other domain tool. Choose exactly one generic linear "
        "operation from the response schema. batch_center subtracts strength times each "
        "batch centroid offset from the global centroid; identity changes nothing. Use no "
        "biological labels. Matrix summary:\n"
        + json.dumps(matrix_summary, sort_keys=True)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    before = usage.snapshot()
    errors: list[str] = []
    plan = _ask(engine, prompt, GPTPlan, errors)
    if plan is None:
        raise RuntimeError(f"GPT-only Luna call failed: {errors}")
    tokens = _usage_delta(before, usage.snapshot())
    if require_tokens and tokens.get("total_tokens", 0) <= 0:
        raise RuntimeError("GPT-only Luna call recorded zero tokens")
    corrected = (
        np.asarray(features, dtype=float)
        if plan.action == "identity"
        else _batch_center(features, batch, plan.strength)
    )
    array_path = run_dir / "corrected.npy"
    np.save(array_path, np.asarray(corrected, dtype=np.float32))
    result = {
        "arm_id": "gpt_luna", "model": model, "scenario": summary["scenario"]["name"],
        "git_commit": _git_commit(), "array": str(array_path),
        "runtime_seconds": time.time() - started, "tool_calls": 0,
        "tokens": tokens, "plan": plan.model_dump(),
        "matrix_summary": matrix_summary, "prompt": prompt,
    }
    _write_json(run_dir / "ARM_OUTPUT.json", result)
    return result


REACT_ARM = "react_luna_harmony"
TOOL_NAME = "Batch_Corrector_Tool"
CAP_FROM_SHAPE = "concurrency_cap(cores, features)"     # the cap's source without a comparison run
TRIAL0_TEXT = ('Call "default" is trial 0: the uncorrected input, measured without a tool call (its output is the '
               "input's own record); it is not a round, and it is what is delivered without a valid finish.")
CALIBRATION_TEXT = ("The calibration wells (the calibration compounds and every control well) with their compound "
                    "labels, as the task gives them: python input inputs/calibration_labels.csv, column row = the row "
                    "of inputs/uncorrected.npy. The evaluation compounds are held out.")
# the label-free well metadata a python step may read (no compound, perturbation type or split)
PYTHON_META_COLUMNS = ("Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_PlateType")


def react_parameters() -> list:
    """SHAPE's lever set for the tool (`BatchCorrectionVerifier.levers`, the CARD's "## Levers", in its order): each
    lever's name, its kind as SHAPE's verifier parses it (integer for the verifier's integer levers; a choice with
    its choices, "choice of ...") and the verifier's default (`BatchCorrectionVerifier.defaults()`, the tool's
    DEFAULT_CONFIG). Nothing else of the CARD."""
    from benchmarks._shared.react_arm import Param
    from shape.agent.verification.verifiers.batch_correction_unified import _INTEGER_LEVERS, BatchCorrectionVerifier

    verifier = BatchCorrectionVerifier()
    defaults = verifier.defaults()
    out = []
    for lever in verifier.levers:
        kind, bounds, nullable = verifier.lever_spec(lever)
        kind = "integer" if lever in _INTEGER_LEVERS and kind == "number" else kind
        if kind == "choice":
            kind = "choice of " + ", ".join(json.dumps(v) for v in (bounds or ()))
        out.append(Param(lever, f"{kind} or null" if nullable else kind, defaults[lever]))
    return out


class BatchToolAdapter:
    """The shared ReAct runner's tool adapter (`benchmarks._shared.react_arm.ToolAdapter`) for this benchmark.

    One unit, the scenario's dataset. A call runs `tool` (default: `control.run_tool`, SHAPE's arm's call: every
    well, the scenario's batch key) with the call's parameters as `harmony_params`; the call "default" is trial 0,
    the uncorrected input (`uncorrected_result`, SHAPE's trial 0), not a tool call. Each observation is the tool's
    result dict only (the runner drops its arrays); SHAPE's calibration scores are not in it (user decision
    2026-10-12: as in the other ReAct arms, the agent designs its own signal with python steps). Each call's matrix
    is kept in trials/<call>.npy; `deliver` names the call whose matrix the arm writes. logs/tool_calls.jsonl holds
    one start and one end record per call, with the calibration scores for the records (`view`, never shown to the
    agent; no view, no scores). `cap`: the per-round cap of a comparison run's SHAPE arm (`comparison_cap`), else
    SHAPE's cap on these cores (`round_cap`). `python_inputs`: the uncorrected features, the label-free well
    metadata (`PYTHON_META_COLUMNS`), the calibration wells with their compound labels (the rows and columns of
    SHAPE's calibration view: the calibration compounds and every control well, no evaluation compound), both
    written once to python_inputs/ in `run_dir`, and each finished call's corrected matrix; never a file under
    `private_roots` (the prep's labelled metadata)."""

    tool_name = TOOL_NAME
    unit_kind = "dataset"
    trial0_label = "the uncorrected input (no batch correction)"   # read by the shared runner, which adds "; not a round"

    def __init__(self, features, batch, *, batch_key: str, view, cores: int, unit: str, run_dir: Path, tool=None,
                 cap: int | None = None, cap_source: str | None = None, meta=None, private_roots=()):
        from benchmarks.batch_correction_JUMP.runners.control import run_tool

        self.features, self.batch, self.view = features, batch, view
        self.meta, self.private_roots = meta, [Path(r) for r in private_roots]
        self._python_files: dict[str, Path] | None = None
        self.cores, self.unit, self.run_dir = int(cores), unit, Path(run_dir)
        self.cap = None if cap is None else int(cap)
        self.cap_source = cap_source or (CAP_FROM_SHAPE if cap is None else "given")
        self.tool = tool or (lambda params: run_tool(features, batch, params, batch_key=batch_key))
        self.paths: dict[str, Path] = {}
        self.delivered: str | None = None
        self.n_tool_calls = 0
        self._lock = threading.Lock()
        # One call computes at a time: two concurrent calls in this process deadlocked on the cluster (every thread in
        # futex_wait after both Harmony runs converged; numba / BLAS pools are not thread-safe here), and harmonypy's
        # concurrent calls serialize on the GIL anyway, so the round's calls run one after another at no cost.
        self._exec_lock = threading.Lock()
        (self.run_dir / "trials").mkdir(parents=True, exist_ok=True)
        self.ledger = self.run_dir / "logs" / "tool_calls.jsonl"

    def parameters(self) -> list:
        return react_parameters()

    def units(self) -> list[str]:
        return [self.unit]

    def round_cap(self) -> int:
        """The comparison run's SHAPE cap when given; else SHAPE's per-round cap for this tool on these cores (the
        batch verifier's size-aware concurrency_cap)."""
        from shape.agent.verification.verifiers.batch_correction_unified import concurrency_cap

        return self.cap if self.cap is not None else concurrency_cap(self.cores, self.features)

    def default_counts_as_round(self) -> bool:
        """As SHAPE counts it: the batch verifier admits on the input (trial 0), which is not a round."""
        from shape.agent.verification.verifiers.batch_correction_unified import BatchCorrectionVerifier

        return not BatchCorrectionVerifier.admission_on_input

    def _log(self, record: dict) -> None:
        with self._lock:
            _append_jsonl(self.ledger, {"time": time.time(), **record})

    def call(self, unit: str, params: dict, call_id: str) -> dict:
        from benchmarks._shared.react_arm import DEFAULT_CALL
        from shape.agent.verification.base import numeric
        from shape.agent.verification.verifiers.batch_correction_unified import uncorrected_result

        trial0 = call_id == DEFAULT_CALL
        started = time.time()
        if not trial0:
            with self._lock:
                self.n_tool_calls += 1
        self._log({"event": "start", "call": call_id, "trial0": trial0, "config": params})
        try:
            with self._exec_lock:
                result = uncorrected_result(self.features, self.batch) if trial0 else self.tool(dict(params))
                matrix = result["corrected_cls_output"]
                calibration = None if self.view is None else numeric(self.view.score(matrix))
            if not trial0:
                path = self.run_dir / "trials" / f"{call_id}.npy"
                np.save(path, np.asarray(matrix, dtype=np.float32))
                self.paths[call_id] = path
        except Exception as error:
            self._log({"event": "end", "call": call_id, "status": "crashed", "error": f"{type(error).__name__}: {error}",
                       "seconds": time.time() - started, "config": params})
            raise
        self._log({"event": "end", "call": call_id, "status": "done", "tool_status": result.get("status"),
                   "seconds": time.time() - started, "config": params, "val": calibration})
        output = {key: value for key, value in result.items() if key != "corrected_cls_output"}
        return {"output": output, "images": [], "error": None}

    def deliver(self, unit: str, call_id: str) -> None:
        self.delivered = call_id

    def python_inputs(self, unit: str) -> dict:
        """The files a python step may read: inputs/uncorrected.npy, inputs/metadata.csv (label-free columns, one
        row per well in the features' order), inputs/calibration_labels.csv (the calibration view: column row = the
        well's row in the features, then the view's columns, the compound label among them),
        calls/<call id>/corrected.npy."""
        from benchmarks._shared.react_arm import assert_no_private

        if self._python_files is None:
            folder = self.run_dir / "python_inputs"
            folder.mkdir(parents=True, exist_ok=True)
            files = {"inputs/uncorrected.npy": folder / "uncorrected.npy"}
            np.save(files["inputs/uncorrected.npy"], np.asarray(self.features, dtype=np.float32))
            if self.meta is not None:
                columns = [c for c in PYTHON_META_COLUMNS if c in self.meta.columns]
                files["inputs/metadata.csv"] = folder / "metadata.csv"
                self.meta[columns].to_csv(files["inputs/metadata.csv"], index=False)
            if self.view is not None:
                files["inputs/calibration_labels.csv"] = folder / "calibration_labels.csv"
                labels = self.view.meta.copy()
                labels.insert(0, "row", np.asarray(self.view.rows, dtype=int))
                labels.to_csv(files["inputs/calibration_labels.csv"], index=False)
            self._python_files = files
        paths = dict(self._python_files)
        paths.update({f"calls/{call_id}/corrected.npy": path for call_id, path in list(self.paths.items())})
        assert_no_private(paths.values(), self.private_roots)
        return paths

    def delivered_matrix(self) -> np.ndarray:
        """The delivered call's matrix: the uncorrected input for trial 0."""
        if self.delivered in self.paths:
            return np.load(self.paths[self.delivered])
        return np.asarray(self.features, dtype=np.float32)


def comparison_cap(run: Path) -> tuple[int, str]:
    """(the per-round cap, its source) recorded by the SHAPE arm of the run `run` (a six-arm run directory with
    suite/shape_luna_harmony/, or a record set with shape_luna_harmony/): its RUN_META.json `concurrency_cap`.
    Fails when neither file or the field is there."""
    for meta in (Path(run) / "suite" / "shape_luna_harmony" / "RUN_META.json",
                 Path(run) / "shape_luna_harmony" / "RUN_META.json"):
        if meta.is_file():
            cap = json.loads(meta.read_text(encoding="utf-8")).get("concurrency_cap")
            if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise SystemExit(f"NO_COMPARISON_CAP: {meta} records no concurrency_cap (got {cap!r})")
            return cap, f"comparison run: {meta} concurrency_cap"
    raise SystemExit(f"NO_COMPARISON_CAP: no shape_luna_harmony/RUN_META.json under {run} (or its suite/)")


def react_input_text(noise_band: float) -> str:
    """The input facts beyond the brief: the task's stated noise (SHAPE's BatchTask.noise_band), the calibration
    wells' labels SHAPE is given (`CALIBRATION_TEXT`) and what trial 0 is."""
    return f"The task's stated noise of the calibration readouts: {noise_band}.\n{CALIBRATION_TEXT}\n{TRIAL0_TEXT}"


def _react_history(unit_dir: Path) -> list[dict]:
    """One entry per call, in run order: its round, the round's thought, its parameters, its calibration scores (the
    ledger's, never shown to the agent) or its error (from the runner's TRACE.jsonl and CALLS.json and the arm's
    logs/tool_calls.jsonl)."""
    ledger = unit_dir.parent.parent / "logs" / "tool_calls.jsonl"
    scores = {}
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            record = json.loads(line) if line.strip() else {}
            if record.get("event") == "end" and record.get("status") == "done":
                scores[record["call"]] = record.get("val")
    thoughts = {}
    for line in (unit_dir / "TRACE.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("action"):
            thoughts.setdefault(record["round"], record["action"].get("thought"))
    history = []
    for call in json.loads((unit_dir / "CALLS.json").read_text(encoding="utf-8")):
        output = call.get("output") or {}
        history.append({"step": call["round"], "call": call["call_id"], "reasoning": thoughts.get(call["round"]),
                        "config": call["params"], "calibration": scores.get(call["call_id"]),
                        "tool_status": output.get("status"), "error": call.get("error")})
    return history


def run_react(prep: Path, run_dir: Path, *, engine, model: str = LUNA_MODEL, noise_band: float = 0.002,
              cores: int | None = None, tool=None, cap: tuple[int, str] | None = None,
              require_tokens: bool = True) -> dict:
    """The ReAct arm on a prepared scenario: the shared runner with `BatchToolAdapter`, one session for the dataset,
    at most BUDGET rounds; the task text is the SHAPE arm's brief. Writes ARM_OUTPUT.json and corrected.npy (the
    agent's finished call, else the uncorrected input) in `run_dir`, the runner's records in run_dir/react/. `cap`:
    (the per-round cap, its source) of a comparison run (`comparison_cap`); None: SHAPE's cap on these cores."""
    from benchmarks._shared.react_arm import BUDGET
    from benchmarks._shared.react_arm import run_react as run_shared
    from shape.agent.verification.verifiers.batch_correction_unified import available_cores, blas_limits
    from shape.engine import usage

    from benchmarks.batch_correction_JUMP.runners.prepare import load
    from benchmarks.batch_correction_JUMP.runners.shape_arm import calibration_view, slurm_deadline, task_brief

    summary, meta, features = load(prep)
    scenario = summary["scenario"]
    batch_key, plate_types = scenario["batch_key"], scenario["plate_types"]
    batch = meta[batch_key].astype(str).to_numpy()
    batch_column = "source" if batch_key == "Metadata_Source" else "batch"
    cores = int(cores or available_cores())
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    before = usage.snapshot()
    adapter = BatchToolAdapter(features, batch, batch_key=batch_key, cores=cores, unit=scenario["name"],
                               view=calibration_view(meta, batch_key=batch_key, plate_types=plate_types),
                               run_dir=run_dir, tool=tool, cap=cap[0] if cap else None,
                               cap_source=cap[1] if cap else None, meta=meta,
                               private_roots=[Path(prep) / "meta.parquet"])
    task_text, input_text = task_brief(scenario, batch_column, BUDGET), react_input_text(noise_band)
    with blas_limits(features, cores):          # the SHAPE arm's BLAS threads per call
        react = run_shared(adapter, task_text=task_text, input_text=input_text, engine=engine,
                           out_dir=run_dir / "react", workers=1, deadline=slurm_deadline(), budget=BUDGET)
    tokens = _usage_delta(before, usage.snapshot())
    if require_tokens and tokens.get("total_tokens", 0) <= 0:
        raise RuntimeError("ReAct Luna calls recorded zero tokens")
    selected = react["selected"][adapter.unit]
    array_path = run_dir / "corrected.npy"
    np.save(array_path, np.asarray(adapter.delivered_matrix(), dtype=np.float32))
    unit_dir = run_dir / "react" / adapter.unit
    result = {
        "arm_id": REACT_ARM, "model": model, "scenario": scenario["name"], "git_commit": _git_commit(),
        "array": str(array_path), "runtime_seconds": time.time() - started,
        "tool_calls": adapter.n_tool_calls, "tokens": tokens, "selected_call": selected["call_id"],
        "source": selected["source"], "stop_reason": selected["stop"], "max_steps": BUDGET, "budget": BUDGET,
        "rounds": selected["rounds"], "steps": selected["rounds"], "round_cap": react["round_cap"],
        "round_cap_source": adapter.cap_source,
        "cores": cores, "default_counts_as_round": react["default_counts_as_round"], "noise_band": noise_band,
        "tool": TOOL_NAME, "parameters": [vars(p) for p in adapter.parameters()], "task_text": task_text,
        "input_text": input_text, "totals": react["totals"], "records": "react/", "history": _react_history(unit_dir),
    }
    _write_json(run_dir / "ARM_OUTPUT.json", result)
    return result


def run_react_only(prep: Path, output: Path, *, model: str = LUNA_MODEL, noise_band: float = 0.002,
                   cores: int | None = None, compare_run: Path | None = None, engine=None) -> dict:
    """The ReAct arm alone on a prepared scenario whose control stage has run (e.g. a finished six-arm run's prep),
    scored by the suite's scorer (`suite._arm_scores`, the control's plate halves) into a report row of the suite's
    form: output/react_luna_harmony/ (the arm) and output/REACT_REPORT.json. The control is not recomputed; whether
    its tool and metric code matches the current code is recorded, not refused. With `compare_run` (the run the
    rows are compared with) the per-round cap is the one its SHAPE arm recorded (`comparison_cap`; refused when
    absent); without it, SHAPE's cap on these cores."""
    from benchmarks.batch_correction_JUMP.runners import control as control_mod
    from benchmarks.batch_correction_JUMP.runners import suite
    from benchmarks.batch_correction_JUMP.runners.prepare import load

    report_path = output / "REACT_REPORT.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists; use a fresh output directory")
    control_path = prep / control_mod.CONTROL_DIR / "CONTROL.json"
    if not control_path.exists():
        raise SystemExit(f"missing controls: {control_path} (run the control stage first)")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    cap = comparison_cap(compare_run) if compare_run is not None else None
    if engine is None:
        from shape.agent.verification.vision_engine import build_vision_engine

        engine = build_vision_engine(model)
    if engine is None:
        raise SystemExit(f"Luna engine unavailable for {model}")
    record = run_react(prep, output / REACT_ARM, engine=engine, model=model, noise_band=noise_band, cores=cores,
                       cap=cap)
    summary, meta, _ = load(prep)
    scores = suite._arm_scores(prep, Path(record["array"]), control)
    row = suite.report_row(summary["scenario"]["name"], REACT_ARM, scores["test"], control["arms"]["raw"]["test"],
                           runtime=float(record["runtime_seconds"]), n_calls=int(record["tool_calls"]),
                           tokens=int((record.get("tokens") or {}).get("total_tokens") or 0),
                           n_profiles=int(summary.get("n_wells", len(meta))))
    now = control_mod.provenance()
    then = control.get("provenance") or {}
    report = {"scenario": summary["scenario"]["name"], "git_commit": _git_commit(), "model": model, "row": row,
              "scores": scores, "control": str(control_path),
              "compare_run": None if compare_run is None else str(compare_run),
              "round_cap": record["round_cap"], "round_cap_source": record["round_cap_source"],
              "control_provenance": {"matches_current": then == now,
                                     "differs": {k: {"control": then.get(k), "current": now.get(k)}
                                                 for k in sorted(set(then) | set(now)) if then.get(k) != now.get(k)}}}
    _write_json(report_path, report)
    return report
