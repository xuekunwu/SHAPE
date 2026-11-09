"""The shared ReAct arm: an LLM plus the tool (benchmarks/_shared/REACT_ARM_CONTRACT.md).

One ReAct session per unit (an image, or the dataset). Each round the model gets one prompt: the task text, the
input text, the tool's parameter table (name, type, default), one factual line, the budget, the time left, the
response schema and the history of its earlier rounds; the images the tool itself wrote in the previous round are
attached (vision engines only, at most `MAX_IMAGES` per round), after the unit's raw input images on the session's
first question (`unit_images`). It replies with one structured action (`ReactAction`):
run up to the round cap of tool calls, run Python code in the session's workspace (`python`: not a round, at most
`PY_PER_ROUND` between two rounds), or finish by naming a finished call to deliver.

The runner never reads CARD files and never chooses among the calls by any score. Basic LLM failures
(`shape.engine.errors.is_fatal_llm_error`, or no engine) fail the arm; a malformed reply, an invalid request or a
failed tool call is an error observation that still uses its round. There is no rule fallback.

A python action runs its code in a fresh subprocess of this Python environment with a scrubbed environment (no
API keys or tokens), cwd = the unit's workspace `out_dir/<unit>/workspace/`: inputs/ and calls/<call id>/ hold the
adapter's allow-listed label-free files (`python_inputs`; symlinks, copies on Windows) listed in manifest.json, and
scratch/py_<n>/ is the only place it writes. The allow-listed files are hashed when materialised and after each step;
a change, or a file written outside scratch/, is recorded as a violation.

Records, per unit under `out_dir/<unit>/`: TRACE.jsonl (one line per round or python step), CALLS.json, SELECTED.json,
python/py_<n>.py (each python step's code); per arm: `out_dir/REACT_SUMMARY.json`.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field

from shape.agent.verification.base import BUDGET, reply_record
from shape.engine import usage as token_usage
from shape.engine.errors import NoEngineError, is_fatal_llm_error
from shape.engine.structured import parse_json_answer

MAX_IMAGES = 16              # images attached to one round's prompt, at most
IMAGE_PX = 1024              # longest side of an attached image: SHAPE's PROMPT_IMAGE_PX (segmentation, crop)
DEFAULT_CALL = "default"     # the call at the defaults; the agent's calls are agent_<n>
TUNING_LINE = "Tuning the tool's parameters can improve the tool's performance."
TRIAL0_LABEL = "the tool at its defaults, run by the system"   # the default round's note (adapter: `trial0_label`)
SECTIONS = ("TASK", "INPUT", "TOOL", "PYTHON", "BUDGET", "HISTORY", "RESPONSE SCHEMA")   # the prompt's sections
PY_TIMEOUT = 300             # seconds a python step may run (less when the deadline is nearer)
PY_OUTPUT_CHARS = 8000       # a python step's stdout + stderr, cut to this many characters
PY_PER_ROUND = 2             # python steps between two rounds, at most
WORKSPACE = "workspace"      # the unit's workspace dir, under out_dir/<unit>/
# environment variables a python step never gets (matched on the upper-cased name)
SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|^OPENAI|^ANTHROPIC|^LLM_")
SOURCE_FINISH = "agent_finish"
SOURCE_UNFINISHED = "default_session_unfinished"


# ---------------------------------------------------------------- the interface (the contract's last section)

@dataclass
class Param:
    name: str
    type: str
    default: Any


class ToolAdapter(Protocol):
    tool_name: str
    unit_kind: Literal["image", "dataset"]

    def parameters(self) -> list[Param]: ...                          # SHAPE's lever set, defaults only

    def units(self) -> list[str]: ...

    def round_cap(self) -> int: ...                                   # SHAPE's per-round cap for this tool

    def default_counts_as_round(self) -> bool: ...                    # as SHAPE counts it

    def call(self, unit: str, params: dict, call_id: str) -> dict: ...  # {"output", "images", "error"}

    def deliver(self, unit: str, call_id: str) -> None: ...

    # optional: `trial0_label: str`, the default round's note in the history (default `TRIAL0_LABEL`)
    # optional: `python_inputs(unit) -> dict[str, path]`, the label-free files a python step may read, keyed by their
    #   workspace path ("inputs/<name>" or "calls/<call id>/<name>"); it calls `assert_no_private` on its paths


# ---------------------------------------------------------------- the reply schema

class ParamValue(BaseModel):
    """One parameter of a call. The value is typed as SHAPE's `LeverChange` value is, so the gateway's structured
    output accepts the schema; a string holding a JSON value (e.g. a per-cell dict) is decoded for a parameter whose
    type is not a string or a choice."""
    name: str
    value: Optional[Union[bool, float, str]] = Field(
        None, description='a number, true / false, a string, null, or a JSON value written as a string')


class ReactCall(BaseModel):
    params: list[ParamValue] = Field(
        default_factory=list, description="the parameters this call sets; a parameter left out is at its default")


class ReactAction(BaseModel):
    thought: str
    action: Literal["call", "finish", "python"]
    calls: list[ReactCall] = Field(default_factory=list, description='with "call": the calls of this round')
    deliver: Optional[str] = Field(None, description='with "finish": the id of the finished call to deliver')
    code: Optional[str] = Field(None, description='with "python": the Python source code to run')


# ---------------------------------------------------------------- helpers

def _jsonable(value, bases: tuple[Path, ...]):
    """A tool's return value as an observation: JSON values kept, numpy scalars as numbers, absolute paths
    relative (`_rel`), arrays and other in-memory objects dropped (None)."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, os.PathLike)):
        text = os.fspath(value)
        return _rel(text, bases) if isinstance(value, os.PathLike) or _looks_like_path(text) else text
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            item = _jsonable(v, bases)
            if item is not None or v is None:
                out[str(k)] = item
        return out
    if isinstance(value, (list, tuple)):
        items = [(_jsonable(v, bases), v) for v in value]
        return [item for item, v in items if item is not None or v is None]
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", None) == 0:    # a numpy scalar
        try:
            return _jsonable(item(), bases)
        except (TypeError, ValueError):
            return None
    return None


def _looks_like_path(text: str) -> bool:
    return 0 < len(text) < 4096 and "\n" not in text and os.path.isabs(text)


def _rel(text: str, bases: tuple[Path, ...]) -> str:
    """`text` relative to the first base it lies under, else its file name."""
    path = Path(text)
    if not path.is_absolute():
        return text
    for base in bases:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return path.name


def _png(path: str, max_px: int) -> bytes:
    """The image file at `path` as PNG bytes, its longest side at most `max_px`: a TIFF as SHAPE renders it for its
    vision reads (`shape_display_image`), any other file through PIL."""
    from PIL import Image

    if str(path).lower().endswith((".tif", ".tiff")):
        img = shape_display_image(path)
    else:
        with Image.open(path) as img:
            img = img.copy()
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def shape_display_image(path: str):
    """The image file at `path` as SHAPE's vision reads render it (the preflight read's preview,
    `image_context._preview_png_bytes`): `shape.utils.image_data.ImageData` (a TIFF read with tifffile by its own axes
    tag, so a channel-first / ImageJ multi-channel stack keeps its channels), a multi-channel image as its merged RGB
    (`create_merged_rgb`: per-channel 1-99 percentile stretch, SHAPE's channel mapping), one channel as uint8
    (`to_uint8`: wider than 8 bit percentile-stretched). A PIL image (mode RGB or L)."""
    import numpy as np
    from PIL import Image

    from shape.utils.image_processor import ImageProcessor

    data = ImageProcessor.load_image(str(path))
    if data.is_multi_channel:
        return Image.fromarray((data.create_merged_rgb() * 255).astype(np.uint8), mode="RGB")
    return Image.fromarray(np.asarray(data.to_uint8(0), dtype=np.uint8), mode="L")


def _coerce(param: Param, value):
    """A wire value as the parameter's type: a JSON value written as a string decoded for a type that is not a
    string or a choice; an integral number for an integer parameter as an int. Anything else is passed as given
    (the tool reports what it cannot use)."""
    kind = str(param.type or "").strip().lower()
    kind = kind[: -len(" or null")].strip() if kind.endswith(" or null") else kind
    if isinstance(value, str) and kind not in ("str", "string", "choice", "category"):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    if kind in ("int", "integer") and isinstance(value, float) and value.is_integer():
        value = int(value)
    return value


def _unit_dir(out_dir: Path, unit: str) -> Path:
    return out_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", str(unit)) or "_")


def _usage_numbers(response) -> dict:
    usage = response.get("usage") if isinstance(response, dict) else None
    out = {}
    for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
        value = usage.get(key) if isinstance(usage, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = int(value)
    return out


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


# ---------------------------------------------------------------- python steps

def assert_no_private(paths, private_roots) -> None:
    """Fail (AssertionError) when a path of `paths` is, or lies under, one of `private_roots` (ground truth, labels):
    each adapter calls it on its `python_inputs` paths."""
    roots = [Path(r).resolve() for r in private_roots if r]
    for path in paths:
        resolved = Path(path).resolve()
        for root in roots:
            if resolved == root or root in resolved.parents:
                raise AssertionError(f"python_inputs: {path} is private (under {root})")


def scrubbed_env(env=None) -> dict:
    """`env` (default os.environ) without the variables a python step never gets (`SECRET_ENV`)."""
    env = dict(os.environ if env is None else env)
    return {k: v for k, v in env.items() if not SECRET_ENV.search(k.upper())}


def _truncate(text: str, limit: int = PY_OUTPUT_CHARS) -> str:
    """`text` cut to `limit` characters: its head and tail around a note of what was cut."""
    if len(text) <= limit:
        return text
    note = f"\n[... {len(text) - limit} characters cut ...]\n"
    keep = max(0, limit - len(note))
    head = keep // 2
    return text[:head] + note + text[len(text) - (keep - head):]


def _read_output(path: Path, limit: int) -> str:
    """The captured output file, at most its first and last `4 * limit` bytes."""
    size, chunk = path.stat().st_size, 4 * limit
    with open(path, "rb") as fh:
        if size <= 2 * chunk:
            data = fh.read()
        else:
            data = fh.read(chunk) + b"\n[...]\n"
            fh.seek(size - chunk)
            data += fh.read()
    return data.decode("utf-8", errors="replace")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _kind_and_shape(path: Path) -> tuple[str, Any]:
    suffix = path.suffix.lower()
    kind = {".npy": "array", ".tif": "image", ".tiff": "image", ".png": "image", ".jpg": "image",
            ".jpeg": "image", ".bmp": "image", ".h5ad": "anndata", ".json": "json", ".csv": "table",
            ".parquet": "table"}.get(suffix, "file")
    shape = None
    try:
        if suffix == ".npy":
            import numpy as np
            shape = list(np.load(path, mmap_mode="r", allow_pickle=False).shape)
        elif suffix in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as tif:
                shape = list(tif.series[0].shape)
        elif kind == "image":
            from PIL import Image
            with Image.open(path) as img:
                shape = [img.height, img.width, len(img.getbands())]
    except Exception:  # noqa: BLE001 -- the shape is only a convenience
        shape = None
    return kind, shape


_WS_KEY = re.compile(r"(inputs|calls/[A-Za-z0-9._-]+)/[^/\\]+")


class _Workspace:
    """A unit's python workspace: inputs/ and calls/<call id>/ (the allow-listed files: symlinks, copies on Windows),
    manifest.json, scratch/py_<n>/."""

    def __init__(self, root: Path):
        self.root = root
        self.entries: dict[str, tuple[str, str]] = {}      # workspace path -> (source path, sha256 when materialised)

    def reset(self) -> None:
        """An earlier session's workspace is removed; the workspace is made by the session's first python step."""
        if self.root.exists():
            shutil.rmtree(self.root)

    def refresh(self, allowed: dict) -> None:
        (self.root / "scratch").mkdir(parents=True, exist_ok=True)
        for key, source in allowed.items():
            key, source = str(key).replace("\\", "/"), os.fspath(source)
            if not _WS_KEY.fullmatch(key) or ".." in key.split("/"):
                raise ValueError(f"python_inputs: {key!r} is not inputs/<name> or calls/<call id>/<name>")
            if not os.path.isfile(source):
                raise ValueError(f"python_inputs: {source} is not a file")
            if self.entries.get(key, (None,))[0] == source:
                continue
            target = self.root / key
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            try:
                if os.name == "nt":
                    raise OSError("copies on Windows")
                os.symlink(os.path.abspath(source), target)
            except OSError:
                shutil.copyfile(source, target)
            self.entries[key] = (source, _sha256(source))
        manifest = []
        for key in sorted(self.entries):
            kind, shape = _kind_and_shape(self.root / key)
            manifest.append({"path": key, "kind": kind, "bytes": (self.root / key).stat().st_size, "shape": shape})
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8",
                                                 newline="\n")

    def violations(self) -> list[str]:
        """The allow-listed files changed (source or workspace entry) and the files written outside scratch/."""
        found = []
        for key, (source, digest) in self.entries.items():
            for label, path in (("source", source), ("workspace", self.root / key)):
                try:
                    changed = _sha256(path) != digest
                except OSError:
                    changed = True
                if changed:
                    found.append(f"{key}: the {label} file changed or is gone")
        for path in self.root.rglob("*"):
            rel = path.relative_to(self.root).as_posix()
            if rel.split("/")[0] == "scratch" or rel == "manifest.json" or rel in self.entries:
                continue
            if path.is_file():
                found.append(f"{rel}: written outside scratch/")
        return found


def run_python(code: str, *, workspace: Path, scratch: Path, code_path: Path, timeout: float,
               limit: int = PY_OUTPUT_CHARS) -> dict:
    """Run `code` in a fresh subprocess of this Python (`sys.executable`), cwd `workspace`, the scrubbed environment
    plus SCRATCH_DIR, at most `timeout` seconds: {"exit", "timed_out", "output" (stdout + stderr cut to `limit`),
    "seconds", "files" (the files under `scratch`, workspace-relative)}."""
    scratch.mkdir(parents=True, exist_ok=True)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(code, encoding="utf-8", newline="\n")
    out_path = code_path.with_suffix(".out")
    env = scrubbed_env()
    env.update({"SCRATCH_DIR": str(scratch.resolve()), "PYTHONIOENCODING": "utf-8"})
    started = time.time()
    timed_out = False
    with open(out_path, "wb") as fh:
        proc = subprocess.Popen([sys.executable, str(code_path.resolve())], cwd=str(workspace), env=env,
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=os.name != "nt")
        try:
            code_exit = proc.wait(timeout=max(0.0, timeout))
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name != "nt":
                import signal
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.wait()
            code_exit = None
    seconds = round(time.time() - started, 3)
    output = _truncate(_read_output(out_path, limit), limit)
    files = sorted(p.relative_to(workspace).as_posix() for p in scratch.rglob("*") if p.is_file())
    return {"exit": code_exit, "timed_out": timed_out, "output": output, "seconds": seconds, "files": files}


def _python_observation(step: int, result: dict) -> str:
    status = (f"timed out after {result['seconds']:.0f} s" if result["timed_out"]
              else f"exit {result['exit']} in {result['seconds']:.1f} s")
    files = result["files"]
    listed = json.dumps(files[:50]) + (f" (+{len(files) - 50} more)" if len(files) > 50 else "")
    text = f"python step {step}: {status}; files written: {listed}"
    if result.get("violations"):
        text += f"; violations: {json.dumps(result['violations'])}"
    return f"{text}\noutput:\n{result['output']}"


PYTHON_TEXT = (
    'action "python": runs the Python code of "code" in a fresh process of this Python environment (numpy, scipy, '
    "scikit-learn, pandas, anndata, scikit-image and torch are installed). It is not a round. At most "
    "{per_round} python actions between two rounds; one more is refused and uses a round. A run stops after "
    "{timeout} s, or when the time is up; the observation is its exit code, its seconds, the files it wrote and its "
    "stdout and stderr together, cut to {chars} characters. Its environment holds no API keys or tokens.\n"
    "The working directory is this session's workspace:\n"
    "- inputs/: the session's input files;\n"
    "- calls/<call id>/: the output files of each finished call, added after the call;\n"
    "- manifest.json: every file of inputs/ and calls/ (path, kind, bytes, shape where known);\n"
    "- scratch/py_<n>/: python step n writes here and only here (its path is the environment variable SCRATCH_DIR); "
    "earlier steps' scratch dirs stay readable.\n"
    "The files of inputs/ and calls/ are read-only: a change to them, or a file written outside scratch/, is "
    "recorded as a violation.\n"
    "The code may compute any label-free quality signal you want from these files.")


# ---------------------------------------------------------------- the prompt

def render_prompt(*, task_text: str, input_text: str, tool_name: str, parameters: list[Param], history: list[dict],
                  rounds_used: int, budget: int, round_cap: int, deadline: float | None, final: bool = False,
                  python_used: int = 0) -> str:
    """The round prompt: exactly the sections of `SECTIONS`; never CARD text, rules, verdicts or hints."""
    table = "\n".join(f"- {p.name} | {p.type} | {json.dumps(p.default, default=str)}" for p in parameters)
    left = max(0, budget - rounds_used)
    time_left = "no limit" if deadline is None else _fmt_seconds(max(0.0, deadline - time.time()))
    if final:
        rule = ('No rounds are left: reply with action "finish" and the id of the finished call to deliver, or '
                'with action "python" while python actions are left.')
    else:
        rule = (f'action "call": run 1 to {round_cap} calls of the tool with the parameters you set; it uses one '
                f'round. action "python": run Python code; it is not a round. action "finish": deliver one finished '
                f'call by its id; it ends the session. A reply that is not valid uses one round.')
    budget_text = (f"Rounds used: {rounds_used} of {budget}. Rounds left: {left}.\n"
                   f"Calls per round: at most {round_cap}.\n"
                   f"Python actions since the last round: {python_used} of {PY_PER_ROUND}.\n"
                   f"Time left: {time_left}.\n{rule}")
    schema = json.dumps(ReactAction.model_json_schema(), indent=1)
    parts = [
        ("TASK", task_text.strip()),
        ("INPUT", input_text.strip()),
        ("TOOL", f"{tool_name}\nParameters (name | type | default):\n{table}\n\n{TUNING_LINE}"),
        ("PYTHON", PYTHON_TEXT.format(per_round=PY_PER_ROUND, timeout=PY_TIMEOUT, chars=PY_OUTPUT_CHARS)),
        ("BUDGET", budget_text),
        ("HISTORY", "\n\n".join(_history_text(h) for h in history) or "No rounds yet."),
        ("RESPONSE SCHEMA", f"Reply with one JSON object of this schema:\n{schema}"),
    ]
    return "\n\n".join(f"## {title}\n{body}" for title, body in parts) + "\n"


def _history_text(entry: dict) -> str:
    lines = [entry.get("title") or f"Round {entry['round']}" + (f" ({entry['note']})" if entry.get("note") else "")]
    if entry.get("thought") is not None:
        lines.append(f"thought: {entry['thought']}")
    lines.append(f"action: {entry['action_text']}")
    for obs in entry["observations"]:
        lines.append(f"observation: {obs}")
    return "\n".join(lines)


def _call_observation(call: dict) -> str:
    head = f"call {call['call_id']} params {json.dumps(call['params'], default=str, sort_keys=True)}"
    if call.get("error"):
        return f"{head}: error: {call['error']}"
    images = f" images: {json.dumps(call['images'])}" if call.get("images") else ""
    return f"{head}: output: {json.dumps(call['output'], default=str)}{images}"


# ---------------------------------------------------------------- one session

class _Session:
    def __init__(self, adapter, unit: str, *, task_text, input_text, engine, out_dir: Path, deadline, budget,
                 round_cap: int, parameters: list[Param], image_px: int, input_images: list[str] = ()):
        self.adapter, self.unit, self.engine = adapter, unit, engine
        self.input_images = [os.fspath(p) for p in input_images]   # attached to the first question only
        self.questions = 0
        self.task_text, self.input_text = task_text, input_text
        self.deadline, self.budget, self.cap, self.image_px = deadline, budget, round_cap, image_px
        self.parameters = parameters
        self.by_name = {p.name: p for p in parameters}
        self.defaults = {p.name: p.default for p in parameters}
        self.dir = _unit_dir(out_dir, unit)
        self.bases = (self.dir, out_dir)
        self.vision = bool(getattr(engine, "is_multimodal", False))
        self.history: list[dict] = []
        self.calls: dict[str, dict] = {}
        self.rounds_used = 0
        self.n_agent = 0
        self.tokens = {"replies": 0, "reported": 0, "total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.pending_images: list[tuple[str, str]] = []     # (call id, path) written by the last round's calls
        self.workspace = _Workspace(self.dir / WORKSPACE)
        self.python_steps = 0                                # python steps run in the session
        self.python_since_round = 0                          # python steps since the last round
        self.python_violations = 0

    # -- records
    def _trace(self, record: dict) -> None:
        with open(self.dir / "TRACE.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def _write(self, name: str, data) -> None:
        (self.dir / name).write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")

    # -- tool calls
    def _run_call(self, call_id: str, params: dict, round_no: int) -> dict:
        record = {"call_id": call_id, "round": round_no, "params": params, "output": None, "images": [],
                  "error": None, "seconds": None}
        started = time.time()
        try:
            result = self.adapter.call(self.unit, dict(params), call_id) or {}
        except Exception as exc:  # noqa: BLE001 -- a failed tool call is an observation; basic LLM failures fail the arm
            if is_fatal_llm_error(exc):
                raise
            result = {"error": f"{type(exc).__name__}: {exc}"}
        record["seconds"] = round(time.time() - started, 3)
        record["error"] = str(result["error"]) if result.get("error") else None
        record["output"] = _jsonable(result.get("output"), self.bases)
        paths = [os.fspath(p) for p in (result.get("images") or [])]
        record["images"] = [_rel(p, self.bases) for p in paths]
        record["_paths"] = paths
        return record

    def _run_round(self, requests: list[tuple[str, dict]], round_no: int) -> list[dict]:
        if len(requests) == 1:
            records = [self._run_call(*requests[0], round_no)]
        else:
            with ThreadPoolExecutor(max_workers=min(len(requests), self.cap)) as pool:
                records = list(pool.map(lambda r: self._run_call(*r, round_no), requests))
        self.pending_images = []
        for rec in records:
            self.pending_images += [(rec["call_id"], p) for p in rec.pop("_paths")]
            self.calls[rec["call_id"]] = rec
        return records

    def _params(self, call: ReactCall) -> tuple[dict | None, str | None]:
        params = dict(self.defaults)
        unknown = [pv.name for pv in call.params if pv.name not in self.by_name]
        if unknown:
            return None, f"unknown parameters {unknown}; the parameters are {list(self.by_name)}"
        for pv in call.params:
            params[pv.name] = _coerce(self.by_name[pv.name], pv.value)
        return params, None

    # -- the model
    def _attachments(self) -> tuple[list, list[dict]]:
        if not self.vision:
            return [], []
        content, refs = [], []
        items = [(f"INPUT IMAGE: {Path(p).name}", p) for p in (self.input_images if not self.questions else [])]
        items += [(f"IMAGE {call_id}: {_rel(path, self.bases)}", path) for call_id, path in self.pending_images]
        for label, path in items[:MAX_IMAGES]:
            try:
                data = _png(path, self.image_px)
            except Exception as exc:  # noqa: BLE001 -- an unreadable image is left out and recorded
                refs.append({"label": label, "error": f"{type(exc).__name__}: {exc}"})
                continue
            content += [label, data]
            refs.append({"label": label, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        return content, refs

    def _ask(self, final: bool):
        """(prompt digest, image refs, reply record, the parsed action or None, the error text or None)."""
        prompt = render_prompt(task_text=self.task_text, input_text=self.input_text,
                               tool_name=self.adapter.tool_name, parameters=self.parameters, history=self.history,
                               rounds_used=self.rounds_used, budget=self.budget, round_cap=self.cap,
                               deadline=self.deadline, final=final, python_used=self.python_since_round)
        images, refs = self._attachments()
        self.questions += 1
        content = [prompt, *images] if images else prompt
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        response = None
        try:
            response = self.engine.generate(content, response_format=ReactAction)
            self._count(response)
            if isinstance(response, dict) and response.get("error") and not response.get("content"):
                raise RuntimeError(f"{response['error']}: {response.get('message', '')}")
            action = response.get("content") if isinstance(response, dict) else response
            if isinstance(action, str):
                action = parse_json_answer(ReactAction, action)
            if not isinstance(action, ReactAction):
                raise TypeError(f"malformed reply: {type(action).__name__}")
            return digest, refs, reply_record(response), action, None
        except Exception as exc:  # noqa: BLE001 -- a malformed reply is an observation; basic LLM failures fail the arm
            if is_fatal_llm_error(exc):
                raise
            return digest, refs, reply_record(response), None, f"{type(exc).__name__}: {exc}"[:2000]

    def _count(self, response) -> None:
        numbers = _usage_numbers(response)
        self.tokens["replies"] += 1
        if numbers.get("total_tokens", 0) > 0:
            self.tokens["reported"] += 1
        for key, value in numbers.items():
            self.tokens[key] += value

    def _expired(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    # -- the session
    def run(self) -> dict:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "TRACE.jsonl").unlink(missing_ok=True)
        self.workspace.reset()
        started = time.time()
        counts = bool(self.adapter.default_counts_as_round())
        round_no = 1 if counts else 0
        self.rounds_used = 1 if counts else 0
        [default] = self._run_round([(DEFAULT_CALL, dict(self.defaults))], round_no)
        note = getattr(self.adapter, "trial0_label", TRIAL0_LABEL) + ("" if counts else "; not a round")
        self._add_history(round_no, None, f"call {DEFAULT_CALL}", [_call_observation(default)], note)
        self._trace({"round": round_no, "kind": "default", "counts_as_round": counts, "prompt_sha256": None,
                     "reply": None, "action": None, "calls": [{"call_id": DEFAULT_CALL, "params": default["params"]}],
                     "observations": [self._obs_summary(default)]})

        delivered, stop = None, None
        while True:
            if self._expired():
                stop = "deadline"
                break
            final = self.rounds_used >= self.budget
            delivered, python_ran = self._turn(final)
            if delivered is not None or (final and not python_ran):
                stop = "finish" if delivered is not None else "final_not_valid"
                break

        if delivered is not None:
            source, call_id = SOURCE_FINISH, delivered
        else:
            source, call_id = SOURCE_UNFINISHED, DEFAULT_CALL
        self.adapter.deliver(self.unit, call_id)
        seconds = round(time.time() - started, 3)
        calls = list(self.calls.values())
        self._write("CALLS.json", calls)
        selected = {"unit": self.unit, "call_id": call_id, "source": source, "stop": stop,
                    "params": self.calls[call_id]["params"], "call_error": self.calls[call_id]["error"],
                    "rounds": self.rounds_used, "calls": len(calls), "python_steps": self.python_steps,
                    "python_violations": self.python_violations, "tokens": dict(self.tokens), "seconds": seconds}
        self._write("SELECTED.json", selected)
        return selected

    def _turn(self, final: bool) -> tuple[str | None, bool]:
        """One reply (a round, a python step, or the finish question after the last round): (the call a valid finish
        delivers, else None; whether a python step ran, which is not a round)."""
        round_no = self.rounds_used + (0 if final else 1)
        digest, refs, reply, action, error = self._ask(final)
        record = {"round": round_no, "kind": "final" if final else "agent", "prompt_sha256": digest, "images": refs,
                  "reply": reply, "action": action.model_dump(mode="json") if action else None, "calls": [],
                  "observations": []}
        delivered, obs, thought, action_text = None, [], None, None
        if action is None:
            action_text, obs = "malformed reply", [f"error: {error}"]
        else:
            thought = action.thought
            if action.action == "python":
                refusal = self._python_refusal(action)
                if refusal is None:
                    self._python_turn(action, record)
                    return None, True
                action_text, obs = "python", [f"error: {refusal}"]
            elif action.action == "finish":
                action_text = f"finish {action.deliver}"
                target = self.calls.get(action.deliver or "")
                if target is not None and not target["error"]:
                    delivered = action.deliver
                else:
                    why = "a call that failed" if target is not None else "not a call id"
                    obs = [f"error: {action.deliver!r} is {why}; the finished calls are "
                           f"{[c for c, r in self.calls.items() if not r['error']]}"]
            elif final:
                action_text, obs = "call", ['error: no rounds are left; only "finish" is accepted']
            else:
                action_text = f"call x{len(action.calls)}"
                obs = self._calls_turn(action.calls, round_no, record)
        record["observations"] = record["observations"] or [{"error": o} for o in obs]
        if delivered is None:
            self._add_history(round_no, thought, action_text, obs, "after the last round" if final else None)
            if not final:
                self.rounds_used += 1           # a round of calls, or a reply that is not a valid finish
                self.python_since_round = 0
        self._trace(record)
        return delivered, False

    # -- python steps
    def _python_refusal(self, action: ReactAction) -> str | None:
        if not (action.code or "").strip():
            return 'action "python" with no code'
        if self.python_since_round >= PY_PER_ROUND:
            return f"at most {PY_PER_ROUND} python actions between two rounds; this one is refused"
        if self.deadline is not None and self.deadline - time.time() <= 0:
            return "no time is left for a python action"
        return None

    def _python_inputs(self) -> dict:
        provider = getattr(self.adapter, "python_inputs", None)
        return dict(provider(self.unit) or {}) if callable(provider) else {}

    def _python_turn(self, action: ReactAction, record: dict) -> None:
        """Run one python step (not a round): its observation, history entry and TRACE line."""
        self.python_steps += 1
        self.python_since_round += 1
        step = self.python_steps
        self.workspace.refresh(self._python_inputs())
        timeout = PY_TIMEOUT if self.deadline is None else min(PY_TIMEOUT, self.deadline - time.time())
        result = run_python(action.code, workspace=self.workspace.root,
                            scratch=self.workspace.root / "scratch" / f"py_{step}",
                            code_path=self.dir / "python" / f"py_{step}.py", timeout=timeout)
        result["violations"] = self.workspace.violations()
        self.python_violations += len(result["violations"])
        observation = _python_observation(step, result)
        record.update({"round": self.rounds_used, "kind": "python",
                       "python": {"step": step, "code": action.code, **result},
                       "observations": [{"python_step": step, "exit": result["exit"],
                                         "timed_out": result["timed_out"], "seconds": result["seconds"],
                                         "files": result["files"], "violations": result["violations"]}]})
        self.history.append({"round": self.rounds_used, "title": f"Python step {step} (after round "
                             f"{self.rounds_used}; not a round)", "thought": action.thought,
                             "action_text": f"python\n{_truncate(action.code)}", "observations": [observation]})
        self._trace(record)

    def _calls_turn(self, calls: list[ReactCall], round_no: int, record: dict) -> list[str]:
        if not calls:
            return ['error: action "call" with no calls']
        if len(calls) > self.cap:
            return [f"error: {len(calls)} calls requested; at most {self.cap} calls per round"]
        obs, requests, slots = [], [], []
        for i, call in enumerate(calls):
            params, error = self._params(call)
            if error:
                slots.append(f"request {i + 1}: error: {error}")
                continue
            self.n_agent += 1
            requests.append((f"agent_{self.n_agent}", params))
            slots.append(None)
        records = iter(self._run_round(requests, round_no) if requests else [])
        for slot in slots:
            if slot is None:
                rec = next(records)
                record["calls"].append({"call_id": rec["call_id"], "params": rec["params"]})
                record["observations"].append(self._obs_summary(rec))
                obs.append(_call_observation(rec))
            else:
                record["observations"].append({"error": slot})
                obs.append(slot)
        return obs

    def _add_history(self, round_no, thought, action_text, observations, note) -> None:
        self.history.append({"round": round_no, "thought": thought, "action_text": action_text,
                             "observations": observations, "note": note})

    @staticmethod
    def _obs_summary(rec: dict) -> dict:
        return {"call_id": rec["call_id"], "error": rec["error"], "images": rec["images"], "seconds": rec["seconds"],
                "output_keys": sorted(rec["output"]) if isinstance(rec["output"], dict) else None}


# ---------------------------------------------------------------- the arm

def run_react(adapter, *, task_text, input_text, engine, out_dir, workers, deadline, budget=BUDGET,
              unit_input: Optional[Callable[[str], str]] = None,
              unit_images: Optional[Callable[[str], list[str]]] = None, image_px: int = IMAGE_PX) -> dict:
    """The ReAct arm over `adapter.units()`: one session per unit, `workers` sessions at a time. Returns
    {"selected": {unit: SELECTED}, "units": {unit: summary}, "totals": arm totals, ...}, also written to
    out_dir/REACT_SUMMARY.json (without "selected"). `deadline` is an absolute time.time() or None. `unit_input`
    (optional): the unit's own input facts, appended to `input_text` in that unit's prompts (e.g. an image's name
    and size). `unit_images` (optional): the unit's raw input image paths (the same inputs SHAPE sees), attached
    to the session's first question only, labelled "INPUT IMAGE: <name>", at `image_px`, inside the `MAX_IMAGES`
    cap (vision engines only). A basic LLM failure in any session propagates (the arm fails)."""
    if engine is None:
        raise NoEngineError("ReAct arm: no engine is configured")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters = list(adapter.parameters())
    cap = max(1, int(adapter.round_cap()))
    units = list(adapter.units())
    before = token_usage.snapshot()
    started = time.time()
    stop = threading.Event()

    def session(unit):
        if stop.is_set():
            return None
        text = input_text if unit_input is None else f"{input_text.rstrip()}\n{unit_input(unit)}"
        return _Session(adapter, unit, task_text=task_text, input_text=text, engine=engine, out_dir=out_dir,
                        deadline=deadline, budget=budget, round_cap=cap, parameters=parameters,
                        image_px=image_px, input_images=unit_images(unit) if unit_images else ()).run()

    selected: dict[str, dict] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futures = {unit: pool.submit(session, unit) for unit in units}
        for unit, future in futures.items():
            selected[unit] = future.result()
    except BaseException:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)

    after = token_usage.snapshot()
    per_unit = {u: {k: s[k] for k in ("rounds", "calls", "python_steps", "tokens", "seconds", "source", "call_id",
                                      "stop")}
                for u, s in selected.items()}
    totals = {
        "units": len(units),
        "rounds": sum(s["rounds"] for s in selected.values()),
        "calls": sum(s["calls"] for s in selected.values()),
        "python_steps": sum(s["python_steps"] for s in selected.values()),
        "python_violations": sum(s["python_violations"] for s in selected.values()),
        "tokens": sum(s["tokens"]["total_tokens"] for s in selected.values()),
        "replies": sum(s["tokens"]["replies"] for s in selected.values()),
        "replies_reported": sum(s["tokens"]["reported"] for s in selected.values()),
        "seconds": round(sum(s["seconds"] for s in selected.values()), 3),
        "wall_seconds": round(time.time() - started, 3),
        "unfinished": sum(s["source"] == SOURCE_UNFINISHED for s in selected.values()),
        # every engine call of the process in the arm's time (the agent's replies and any model call a tool makes)
        "process_usage_tokens": after["total_tokens"] - before["total_tokens"],
        "process_usage_calls": after["calls"] - before["calls"],
    }
    summary = {"tool": adapter.tool_name, "unit_kind": getattr(adapter, "unit_kind", None), "budget": budget,
               "round_cap": cap, "default_counts_as_round": bool(adapter.default_counts_as_round()),
               "units": per_unit, "totals": totals}
    (out_dir / "REACT_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str) + "\n",
                                                encoding="utf-8", newline="\n")
    return {"selected": selected, **summary}
