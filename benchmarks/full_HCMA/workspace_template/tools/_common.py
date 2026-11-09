"""Shared code of the workspace tools (segment, crop, embed, batch_correct, cluster).

Each tool is a thin CLI around one SHAPE tool class, so every arm runs the same tool code; this
module holds what they share: the lever registry (`tool_levers`, also read by the harness), value
validation, the call directory and the ledger, input verification, the GPU semaphore, the deadline
and the worker pool.

A call: `python tools/<tool>.py --name NAME [inputs] [--set k=v ...] [--config FILE.json]
[--note "why"] [--dry-run]`. Exit codes: 0 the call ended done or partial (or a dry run / describe),
1 it failed or was terminated, 2 invalid arguments (nothing written, no ledger line), 3 too little
time before the deadline, 4 an input file differs from the sha256 its producer call recorded.

Ledger (`_config.json: ledger`): one `start` and one `end` line per call. `start` records the
levers set globally with a non-default value (`set`), every lever the call changes globally, per
image or per cell (`set_levers`, what lever coverage counts), the resolved global levers (`params`),
the inputs (every producer call id it consumes appears in `inputs`), the argv and the sha256 of the
per-image / per-cell files. `end` records the status (done | partial | failed), the seconds, the
outputs with their sha256 (paths relative to the call directory), a summary and the error.

Test switch: FULL_HCMA_TOOL_STUBS=<path to a .py file> replaces the SHAPE tool modules by that
file, which defines the tool classes the CLIs use (and DEFAULT_CONFIG for the batch corrector). It
is honoured only when tools/_config.json has "allow_tool_stubs": true, which the unit tests write
and the run setup never does, so the tool code of a run cannot be swapped through the environment.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import signal
import socket
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

TOOLS = ("segment", "crop", "embed", "batch_correct", "cluster")
PREFIX = {"segment": "S", "crop": "C", "embed": "E", "batch_correct": "B", "cluster": "K"}
TOOL_OF_PREFIX = {p: t for t, p in PREFIX.items()}
# the SHAPE tool directory whose CARD declares a tool's levers ("## Levers")
CARD_DIRS = {"segment": "cell_segmenter", "crop": "single_cell_cropper", "batch_correct": "batch_corrector"}
ANALYZER_DIR = "cell_state_analyzer"
SHAPE_MODULES = {
    "Cell_Segmenter_Tool": "shape.tools.cell_segmenter.tool",
    "Single_Cell_Cropper_Tool": "shape.tools.single_cell_cropper.tool",
    "Cell_State_Analyzer_Tool": "shape.tools.cell_state_analyzer.tool",
    "Batch_Corrector_Tool": "shape.tools.batch_corrector.tool",
}
STUBS_ENV = "FULL_HCMA_TOOL_STUBS"
COMPLETED = ("done", "partial")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
CALL_ID_RE = re.compile(r"^[SCEBK]\d{3,}$")
TOOLS_DIR = Path(__file__).resolve().parent
WORKSPACE = TOOLS_DIR.parent


class Invalid(Exception):
    """Invalid arguments: exit 2, nothing written."""


class Mismatch(Exception):
    """An input file differs from the sha256 its producer call recorded: exit 4."""


class Deadline(Exception):
    """Fewer than `min_start_seconds` left before the deadline: exit 3."""


class Terminated(BaseException):
    """SIGTERM. A BaseException, so the SHAPE tools' own `except Exception` cannot swallow it."""


# ---------------------------------------------------------------- levers

@dataclass
class LeverInfo:
    name: str
    kind: str            # number | integer | bool | choice | numbers (a number or a list) | per_cell
    bounds: Any          # (lo, hi), the choices, or None
    nullable: bool
    default: Any


# The analyzer's levers: the surface SHAPE tunes -- its CARD's "## Tunable parameters" and the
# representation stage's param_map (shape/agent/verification/stage_map.py), defaults = the stage's.
EMBED_LEVERS = {
    "adaptation_depth": LeverInfo("adaptation_depth", "choice", ("zero_shot", "frozen_head", "trained"), False,
                                  "trained"),
    "learning_rate": LeverInfo("learning_rate", "number", (1e-5, 1e-4), False, 3e-5),
    "max_epochs": LeverInfo("max_epochs", "integer", (1, 100), True, None),
    "augment_color": LeverInfo("augment_color", "bool", (False, True), False, True),
    "field_aware": LeverInfo("field_aware", "bool", (False, True), False, False),
    "plate_mean_weight": LeverInfo("plate_mean_weight", "number", (0.0, 10.0), False, 0.0),
    "plate_adversarial_weight": LeverInfo("plate_adversarial_weight", "number", (0.0, 10.0), False, 0.0),
}
# settable, recorded, but not a lever (not counted by lever coverage)
EMBED_NON_LEVERS = {"seed": LeverInfo("seed", "integer", (0, 2 ** 31 - 1), False, 0)}
# The clustering levers of the morphological_state stage (stage_map: leiden_resolution ->
# cluster_resolution, n_neighbors), defaults = the stage's.
CLUSTER_LEVERS = {
    "resolution": LeverInfo("resolution", "numbers", (0.01, 10.0), False, 0.5),
    "n_neighbors": LeverInfo("n_neighbors", "integer", (2, 200), False, 15),
}
# The cropper's per-cell keys (`cell_overrides`): its CARD's "Per-cell keys" table, with the values
# shape/agent/verification/verifiers/crop.py (CELL_KEYS, RETENTION_FLOOR) accepts. A key a cell does
# not carry follows the image-level value.
RETENTION_FLOOR = 0.90
CELL_KEYS = {
    "margin_mode": LeverInfo("margin_mode", "choice", ("fraction", "tight", "mask_bbox"), False, None),
    "recenter": LeverInfo("recenter", "bool", (False, True), False, None),
    "auto_tighten": LeverInfo("auto_tighten", "bool", (False, True), False, None),
    "min_containment": LeverInfo("min_containment", "number", (RETENTION_FLOOR, 1.0), False, None),
    "reset": LeverInfo("reset", "choice", (True,), False, None),
}


def _import_path(repo) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def tool_levers(tool: str, repo: Path) -> dict[str, LeverInfo]:
    """{lever: LeverInfo} of a tool, in the order its documentation lists them: the CARD's "## Levers"
    (`shape.utils.schemas.card_lever_fields` + `lever_spec`) for segment / crop / batch_correct, the
    registries above for embed / cluster. The cropper's per-cell overrides are one more entry,
    `cell_overrides` (kind "per_cell"). Every default is the CARD's: the control, the Claude arm and SHAPE
    start from the tools' own defaults."""
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}; the tools are {TOOLS}")
    if tool == "embed":
        return dict(EMBED_LEVERS)
    if tool == "cluster":
        return dict(CLUSTER_LEVERS)
    _import_path(repo)
    from shape.utils.schemas import card_lever_fields, lever_spec

    tool_dir = str(Path(repo) / "shape" / "tools" / CARD_DIRS[tool])
    fields = card_lever_fields(tool_dir, "## Levers")
    if not fields:
        raise RuntimeError(f"no levers found under '## Levers' in {tool_dir}/CARD.md")
    levers = {name: LeverInfo(name, *lever_spec(spec)) for name, spec in fields.items()}
    if tool == "crop":
        levers["cell_overrides"] = LeverInfo("cell_overrides", "per_cell", tuple(CELL_KEYS), True, None)
    return levers


def _same(a: Any, b: Any) -> bool:
    """Equal values of the same kind (True is not 1, None is not 0)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is None and b is None
    return a == b


def _number(info: LeverInfo, value: Any, kind: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Invalid(f"{info.name}: {value!r} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise Invalid(f"{info.name}: {value!r} is not a finite number")
    if kind == "integer" and number != int(number):
        raise Invalid(f"{info.name}: {value!r} is not an integer")
    if info.bounds is not None and not info.bounds[0] <= number <= info.bounds[1]:
        raise Invalid(f"{info.name}={value!r} outside [{info.bounds[0]:g}, {info.bounds[1]:g}]")
    return int(number) if kind == "integer" else number


def normal_default(info: LeverInfo) -> Any:
    """A lever's default in the form `check_value` returns."""
    if info.default is None:
        return None
    if info.kind == "numbers":
        return [float(info.default)]
    if info.kind == "number":
        return float(info.default)
    if info.kind == "integer":
        return int(info.default)
    return info.default


def check_value(info: LeverInfo, value: Any) -> Any:
    """`value` checked against the lever's kind / bounds / choices / nullability (the CARD's Type and
    Range as `shape.utils.schemas.lever_spec` reads them); returns it normalised (numbers as float,
    integers as int, `numbers` as a list). The lever's default is always valid (e.g. a batch
    corrector `input_pca` of 0 = off, outside its Range). Raises Invalid."""
    if info.default is not None and _same(value, info.default):
        return normal_default(info)
    if value is None:
        if info.nullable:
            return None
        raise Invalid(f"{info.name} cannot be null")
    if info.kind == "bool":
        if isinstance(value, bool):
            return value
        raise Invalid(f"{info.name}: {value!r} is not true or false")
    if info.kind == "choice":
        for choice in info.bounds or ():
            if _same(value, choice):
                return choice
        raise Invalid(f"{info.name}: {value!r} is not one of {list(info.bounds or ())}")
    if info.kind == "numbers":
        values = value if isinstance(value, list) else [value]
        if not values:
            raise Invalid(f"{info.name}: an empty list")
        out = [_number(info, v, "number") for v in values]
        if len(set(out)) != len(out):
            raise Invalid(f"{info.name}: a value is listed twice in {value!r}")
        return out
    if info.kind in ("number", "integer"):
        return _number(info, value, info.kind)
    raise Invalid(f"{info.name} is not set with --set (see --describe)")


def parse_set(items: list[str]) -> dict:
    """`k=v` items; v is a JSON literal when it parses (null, true, 3, "x", [0.5, 1]), else a string."""
    out = {}
    for item in items or []:
        key, eq, raw = item.partition("=")
        if not eq or not key.strip():
            raise Invalid(f"--set {item!r}: expected key=value")
        key = key.strip()
        if key in out:
            raise Invalid(f"--set {key} is given twice")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def resolve_levers(levers: dict[str, LeverInfo], given: dict, non_levers: dict | None = None
                   ) -> tuple[dict, dict]:
    """(params, set): every lever (and non-lever) at its default, overridden by the validated `given`
    values; `set` = the given levers whose value differs from the default. Raises Invalid."""
    non_levers = non_levers or {}
    known = {**levers, **non_levers}
    unknown = sorted(set(given) - set(known))
    if unknown:
        raise Invalid(f"unknown lever(s) {unknown}; the levers are {[k for k, v in levers.items() if v.kind != 'per_cell']}"
                      + (f", other settings {list(non_levers)}" if non_levers else ""))
    params = {k: normal_default(info) for k, info in known.items() if info.kind != "per_cell"}
    for key, value in given.items():
        if known[key].kind == "per_cell":
            raise Invalid(f"{key} is given with --cell-overrides FILE.json, not with --set")
        params[key] = check_value(known[key], value)
    changed = {k: params[k] for k in given if k in levers and not _same(params[k], normal_default(levers[k]))}
    return params, changed


def resolve_per_image(levers: dict[str, LeverInfo], mapping: Any, known_images: set[str]
                      ) -> tuple[dict, list[str]]:
    """{image_id: {lever: value}} validated like --set, and the levers it sets to a non-default value
    (lever coverage). Raises Invalid."""
    if not isinstance(mapping, dict):
        raise Invalid("per_image: expected a JSON object {image_id: {lever: value}}")
    out, changed = {}, set()
    for image_id, values in mapping.items():
        if image_id not in known_images:
            raise Invalid(f"per_image: unknown image {image_id!r}")
        if not isinstance(values, dict):
            raise Invalid(f"per_image[{image_id}]: expected an object {{lever: value}}")
        unknown = sorted(set(values) - {k for k, v in levers.items() if v.kind != "per_cell"})
        if unknown:
            raise Invalid(f"per_image[{image_id}]: unknown lever(s) {unknown}")
        out[image_id] = {k: check_value(levers[k], v) for k, v in values.items()}
        changed |= {k for k, v in out[image_id].items() if not _same(v, normal_default(levers[k]))}
    return out, sorted(changed)


def registry_table(levers: dict[str, LeverInfo]) -> str:
    lines = ["| lever | type | range | null allowed | default |", "| --- | --- | --- | --- | --- |"]
    for info in levers.values():
        if info.kind == "per_cell":
            rng = ", ".join(info.bounds)
        elif isinstance(info.bounds, tuple) and len(info.bounds) == 2 and info.kind == "integer":
            rng = f"[{int(info.bounds[0])}, {int(info.bounds[1])}]"
        elif isinstance(info.bounds, tuple) and len(info.bounds) == 2 and info.kind in ("number", "numbers"):
            rng = f"[{info.bounds[0]:g}, {info.bounds[1]:g}]"
        else:
            rng = ", ".join(json.dumps(b) for b in (info.bounds or ()))
        kind = "number or list of numbers" if info.kind == "numbers" else info.kind
        lines.append(f"| {info.name} | {kind} | {rng} | {'yes' if info.nullable else 'no'} | "
                     f"{json.dumps(info.default)} |")
    return "\n".join(lines)


def card_section(repo: Path, tool_dir: str, heading: str) -> str:
    """The CARD's `heading` section, the H2 line up to the next H2, verbatim -- the text SHAPE's
    agentic loop puts in its prompts (`AgenticVerifier._card_section`)."""
    _import_path(repo)
    from shape.tools.base import load_tuning_card

    body = load_tuning_card(str(Path(repo) / "shape" / "tools" / tool_dir)) or ""
    found = re.search(rf"^{re.escape(heading.strip())}[ \t]*$", body, re.MULTILINE)
    if found is None:
        raise RuntimeError(f"no section {heading!r} in {tool_dir}/CARD.md")
    end = body.find("\n## ", found.start() + 1)
    return body[found.start():end if end > 0 else None].strip()


# ---------------------------------------------------------------- configuration, environment, tool classes

def load_config(optional: bool = False) -> dict | None:
    path = TOOLS_DIR / "_config.json"
    if not path.exists():
        if optional:
            return None
        raise SystemExit(f"{path} is missing: these tools run inside a run workspace")
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root(cfg: dict | None = None) -> Path:
    """The repository the tools import SHAPE from: `_config.json: repo`; for the template itself
    (no _config.json), the repository that contains it."""
    cfg = cfg if cfg is not None else load_config(optional=True)
    if cfg and cfg.get("repo"):
        return Path(cfg["repo"])
    return TOOLS_DIR.parents[3]


def setup(cfg: dict, *, gpu: bool) -> None:
    """Before SHAPE is imported: `_config.json: env` into os.environ (the pinned model caches, offline
    hub, fast figures), the repository on sys.path. A CPU tool hides the GPU, so it never opens a CUDA
    context on an exclusive-process GPU a GPU tool holds."""
    for key, value in (cfg.get("env") or {}).items():
        os.environ[str(key)] = str(value)
    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _import_path(cfg["repo"])


def tool_module(class_name: str, cfg: dict):
    """The module that defines a SHAPE tool class (the test stub file instead, see the module doc)."""
    if cfg.get("repo"):
        _import_path(cfg["repo"])            # the agent's environment carries no PYTHONPATH
    stub = os.environ.get(STUBS_ENV)
    if stub and cfg.get("allow_tool_stubs"):
        module = sys.modules.get("_full_hcma_tool_stubs")
        if module is None:
            spec = importlib.util.spec_from_file_location("_full_hcma_tool_stubs", stub)
            module = importlib.util.module_from_spec(spec)
            sys.modules["_full_hcma_tool_stubs"] = module
            spec.loader.exec_module(module)
        return module
    return importlib.import_module(SHAPE_MODULES[class_name])


def tool_class(class_name: str, cfg: dict):
    return getattr(tool_module(class_name, cfg), class_name)


# ---------------------------------------------------------------- small helpers

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_safe(obj: Any) -> Any:
    """A JSON-serialisable copy (numpy values as Python values, NaN / inf as null, other objects as text)."""
    try:
        import numpy as np
    except ImportError:          # pragma: no cover
        np = None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if np is not None and isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if np is not None and isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    write_atomic(path, json.dumps(json_safe(obj), indent=1) + "\n")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_safe(value))
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(json_safe(value))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _cell(row.get(k)) for k in columns})
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_stats(values: list) -> dict:
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def read_input_file(path: str, what: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise Invalid(f"{what} {path}: {error.strerror or error}") from None


def parse_json(data: bytes, what: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise Invalid(f"{what}: not valid JSON ({error})") from None


def read_config_file(path: str | None, allowed: tuple[str, ...]) -> tuple[dict, bytes | None]:
    """`--config FILE.json`: {"set": {...}, ...} with the keys `allowed`."""
    if not path:
        return {}, None
    data = read_input_file(path, "--config")
    config = parse_json(data, f"--config {path}")
    if not isinstance(config, dict):
        raise Invalid(f"--config {path}: expected a JSON object")
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise Invalid(f"--config {path}: unknown key(s) {unknown}; allowed: {list(allowed)}")
    if "set" in config and not isinstance(config["set"], dict):
        raise Invalid(f"--config {path}: 'set' must be an object {{lever: value}}")
    return config, data


def given_levers(config: dict, set_items: list[str]) -> dict:
    """The lever values of --config's "set", overridden by --set."""
    return {**dict(config.get("set") or {}), **parse_set(set_items)}


def validate_name(name: str | None) -> str:
    if not name or not NAME_RE.match(name):
        raise Invalid(f"--name {name!r}: 1-40 characters of A-Z a-z 0-9 _ -")
    return name


# ---------------------------------------------------------------- images

def read_images(cfg: dict) -> list[dict]:
    return read_csv(WORKSPACE / cfg["images_csv"])


def image_file(cfg: dict, row: dict) -> Path:
    return WORKSPACE / cfg["images_dir"] / Path(row["file"]).name


def select_images(known: list[str], spec: str | None, file: str | None) -> tuple[list[str], dict]:
    """`--images all|ID,ID,...` or `--images-file FILE` (one id per line): the ids, in the order of
    `known`, and the input files to keep with the call."""
    if spec and file:
        raise Invalid("give --images or --images-file, not both")
    files = {}
    if file:
        data = read_input_file(file, "--images-file")
        ids = [line.strip() for line in data.decode("utf-8").splitlines() if line.strip()]
        files["inputs/images.txt"] = data
    elif spec and spec != "all":
        ids = [part.strip() for part in spec.split(",") if part.strip()]
    else:
        return list(known), files
    unknown = [i for i in ids if i not in set(known)]
    if unknown:
        raise Invalid(f"unknown image id(s) {unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
    if not ids:
        raise Invalid("no images selected")
    wanted = set(ids)
    return [i for i in known if i in wanted], files


# ---------------------------------------------------------------- locks, deadline, GPU semaphore

class FileLock:
    """An exclusive lock on `path`: fcntl.flock; a pid file where the filesystem has no flock; a no-op
    where there is no fcntl (Windows, unit tests)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None
        self.pidfile = None

    def acquire(self, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:
            return True
        handle = open(self.path, "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            self.handle = handle
            return True
        except OSError as error:
            handle.close()
            if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                return False
        return self._pid_lock(blocking)       # e.g. ENOSYS / EOPNOTSUPP on a network filesystem

    def _pid_lock(self, blocking: bool) -> bool:
        pidfile = self.path.with_name(self.path.name + ".pid")
        while True:
            try:
                fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.pidfile = pidfile
                return True
            except FileExistsError:
                try:
                    os.kill(int(pidfile.read_text().strip()), 0)
                except (ValueError, ProcessLookupError, FileNotFoundError):
                    pidfile.unlink(missing_ok=True)      # stale: its process is gone
                    continue
                except PermissionError:
                    pass
            if not blocking:
                return False
            time.sleep(0.2)

    def release(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.pidfile is not None:
            self.pidfile.unlink(missing_ok=True)
            self.pidfile = None

    def __enter__(self):
        self.acquire(blocking=True)
        return self

    def __exit__(self, *exc):
        self.release()


def deadline_path(cfg: dict) -> Path:
    """`_config.json: deadline_file` (absolute, outside the workspace; a relative path is read from the
    workspace)."""
    return WORKSPACE / cfg.get("deadline_file", ".deadline")


def seconds_left(cfg: dict) -> float | None:
    """Seconds before the deadline; None without one. Raises Deadline when the run has a deadline
    (`_config.json: has_deadline`) and its file is missing."""
    path = deadline_path(cfg)
    if not path.exists():
        if cfg.get("has_deadline"):
            raise Deadline(f"this run has a deadline but its file {path} is missing; calls are refused")
        return None
    return float(path.read_text(encoding="utf-8").strip()) - time.time()


def check_deadline(cfg: dict) -> None:
    left = seconds_left(cfg)
    need = float(cfg.get("min_start_seconds", 600))
    if left is not None and left < need:
        raise Deadline(f"{max(left, 0) / 60:.1f} min left before the deadline; a call needs at least "
                       f"{need / 60:.0f} min to start")


@contextmanager
def gpu_slot(cfg: dict, on_wait: Callable[[], None]):
    """Hold one of the `gpu_slots` GPU slots (lock files <run>/logs/gpu_slot_<i>.lock) for the call;
    waits while all are taken."""
    slots = max(1, int(cfg.get("gpu_slots", 1)))
    logs = Path(cfg["run"]) / "logs"
    waited = False
    while True:
        for slot in range(slots):
            lock = FileLock(logs / f"gpu_slot_{slot}.lock")
            if lock.acquire(blocking=False):
                try:
                    yield slot
                finally:
                    lock.release()
                return
        if not waited:
            on_wait()
            waited = True
        time.sleep(5)


# ---------------------------------------------------------------- the ledger and producer calls

def append_ledger(cfg: dict, record: dict) -> None:
    path = Path(cfg["ledger"])
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(record)) + "\n"
    with FileLock(path.with_name(path.name + ".lock")):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(line)


def read_ledger(cfg: dict) -> list[dict]:
    path = Path(cfg["ledger"])
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


@dataclass
class Producer:
    """An earlier call whose outputs a call consumes."""
    call: str
    tool: str
    start: dict
    end: dict
    dir: Path

    def verify(self, rel: str) -> Path:
        """The output `rel` (relative to the call directory), checked against its recorded sha256."""
        recorded = (self.end.get("outputs") or {}).get(rel)
        path = self.dir / rel
        if recorded is None:
            raise Mismatch(f"{self.call}: {rel} is not among the outputs the call recorded")
        if not path.is_file():
            raise Mismatch(f"{self.call}: {rel} is missing")
        if sha256_file(path) != recorded:
            raise Mismatch(f"{self.call}: {rel} changed since the call (sha256 differs from the recorded value)")
        return path


def producer(cfg: dict, call_id: str, tools: tuple[str, ...]) -> Producer:
    """The ledger record of `call_id`, which must be a call of one of `tools` that ended done or
    partial. Raises Invalid."""
    if not isinstance(call_id, str) or not CALL_ID_RE.match(call_id):
        raise Invalid(f"{call_id!r} is not a call id (e.g. S001)")
    start = end = None
    for entry in read_ledger(cfg):
        if entry.get("call") == call_id:
            if entry.get("event") == "start" and start is None:
                start = entry
            elif entry.get("event") == "end" and end is None:
                end = entry
    if start is None:
        raise Invalid(f"{call_id}: no such call in the ledger")
    if start.get("tool") not in tools:
        raise Invalid(f"{call_id} is a {start.get('tool')} call; expected {' or '.join(tools)}")
    status = (end or {}).get("status")
    if status not in COMPLETED:
        raise Invalid(f"{call_id}: status {status or 'not ended (running or killed)'}; only done or partial "
                      "calls can feed later calls")
    return Producer(call_id, start["tool"], start, end, WORKSPACE / start["call_dir"])


def chain_calls(cfg: dict, call_id: str) -> list[str]:
    """`call_id` and every call it consumed, transitively (from the ledger `start.inputs`)."""
    starts = {e.get("call"): e for e in read_ledger(cfg) if e.get("event") == "start"}
    order, todo = [], [call_id]
    while todo:
        cid = todo.pop(0)
        if cid in order:
            continue
        order.append(cid)
        todo += list((starts.get(cid) or {}).get("inputs", {}).get("from_calls") or [])
    return order


# ---------------------------------------------------------------- a call

def _relpath(path: Path, base: Path) -> str:
    return Path(path).relative_to(base).as_posix()


class Call:
    """One tool call: its id and directory, meta.json, result.json and the ledger lines."""

    def __init__(self, cfg: dict, tool: str, name: str, *, note: str, argv: list[str], set_: dict,
                 set_levers: list[str], params: dict, inputs: dict, files: dict | None = None,
                 input_paths: dict | None = None, meta: dict | None = None, ledger_extra: dict | None = None):
        """`files`: {path relative to the call directory: bytes} written at the start (the input files
        given, kept with the call); `input_paths`: {inputs key: one of those paths} -- recorded in
        `inputs` as the workspace-relative path, with `<key>_sha256`."""
        self.cfg, self.tool, self.name, self.note, self.argv = cfg, tool, name, note or "", list(argv)
        self.set, self.set_levers, self.params, self.inputs = set_, sorted(set_levers), params, dict(inputs)
        self.files, self.meta, self.ledger_extra = dict(files or {}), dict(meta or {}), dict(ledger_extra or {})
        self.input_paths = dict(input_paths or {})
        self.id: str | None = None
        self.dir: Path | None = None
        self.started: float | None = None
        self.console = None          # fd of the console's stderr while the tool output goes to tool.log

    def allocate(self) -> None:
        """The next id of this tool (<P><NNN>) and its directory, under calls/.counter.lock."""
        calls = WORKSPACE / "calls"
        calls.mkdir(parents=True, exist_ok=True)
        prefix = PREFIX[self.tool]
        with FileLock(calls / ".counter.lock"):
            counter_path = calls / ".counter.json"
            counter = json.loads(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else {}
            used = [int(m.group(1)) for d in calls.iterdir()
                    if (m := re.match(rf"^{prefix}(\d{{3,}})_", d.name))]
            number = max([int(counter.get(prefix, 0)), *used]) + 1
            counter[prefix] = number
            write_atomic(counter_path, json.dumps(counter))
            self.id = f"{prefix}{number:03d}"
            self.dir = calls / f"{self.id}_{self.name}"
            self.dir.mkdir()

    def start(self) -> None:
        self.allocate()
        self.started = time.time()
        for rel, data in self.files.items():
            path = self.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        input_files = {rel: sha256_bytes(data) for rel, data in self.files.items()}
        for key, rel in self.input_paths.items():
            self.inputs[key] = _relpath(self.dir / rel, WORKSPACE)
            self.inputs[f"{key}_sha256"] = input_files[rel]
        write_json(self.dir / "meta.json", {
            "id": self.id, "tool": self.tool, "name": self.name, "note": self.note, "argv": self.argv,
            "set": self.set, "set_levers": self.set_levers, "params": self.params, "inputs": self.inputs,
            "input_files": input_files, **self.meta, "started": self.started, "pid": os.getpid(),
            "host": socket.gethostname()})
        # replaced at the end; a call killed before it ends keeps this
        write_json(self.dir / "result.json", {
            "call": self.id, "tool": self.tool, "name": self.name, "status": "failed", "seconds": None,
            "summary": {}, "outputs": {}, "error": "not finished: the call is running or was killed"})
        append_ledger(self.cfg, {
            "event": "start", "time": self.started, "call": self.id, "tool": self.tool, "name": self.name,
            "call_dir": _relpath(self.dir, WORKSPACE), "note": self.note, "argv": self.argv, "set": self.set,
            "set_levers": self.set_levers, "params": self.params, "inputs": self.inputs,
            "per_image_sha256": None, **self.ledger_extra, **code_identity(self.cfg, self.tool),
            "pid": os.getpid()})

    def outputs(self, paths: list[Path]) -> dict[str, str]:
        return {_relpath(p, self.dir): sha256_file(p) for p in paths}

    def end(self, status: str, summary: dict, outputs: dict, error: str | None) -> dict:
        seconds = round(time.time() - self.started, 1)
        record = {"call": self.id, "tool": self.tool, "name": self.name, "status": status, "seconds": seconds,
                  "summary": summary, "outputs": outputs, "error": error}
        write_json(self.dir / "result.json", record)
        append_ledger(self.cfg, {"event": "end", "time": time.time(), "call": self.id, "tool": self.tool,
                                 "status": status, "seconds": seconds, "outputs": outputs, "summary": summary,
                                 "error": error})
        return record

    def progress(self, message: str) -> None:
        line = f"{self.tool} {self.id}: {message}\n"
        if self.console is not None:
            os.write(self.console, line.encode())
        else:
            sys.stderr.write(line)
            sys.stderr.flush()


def code_identity(cfg: dict, tool: str) -> dict:
    """What ran: the sha256 of the tool's CLI file, of this file and of _config.json, and whether the
    test stubs replaced the SHAPE tools."""
    config = TOOLS_DIR / "_config.json"
    return {"tool_sha256": sha256_file(TOOLS_DIR / f"{tool}.py"), "common_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(config) if config.exists() else None,
            "stubs_active": bool(os.environ.get(STUBS_ENV) and cfg.get("allow_tool_stubs"))}


def _terminate(signum, frame):
    raise Terminated(f"signal {signum}")


@contextmanager
def tool_output(log: Path):
    """The process's stdout / stderr (file descriptors 1 and 2) into `log`; yields the console's stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    try:
        yield saved[1]
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (OSError, ValueError):
            pass
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])


def redirect_process_output(log: Path) -> None:
    """A worker process: stdout / stderr into the call's tool.log for the life of the process."""
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)


def run_call(call: Call, body: Callable[[Call], tuple[str, dict, dict]], *, gpu: bool) -> int:
    """Start the call (deadline check, GPU slot, id, meta, ledger start), run `body(call)` -> (status,
    summary, outputs) with the SHAPE tool's output in <call>/tool.log, end it (result.json, ledger end).
    Ctrl-C / SIGTERM end it as failed, "terminated". Prints the result; returns the exit code."""
    cfg = call.cfg
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        try:
            check_deadline(cfg)
        except Deadline as error:
            print(f"refused: {error}", file=sys.stderr)
            return 3
        with gpu_slot(cfg, lambda: print("waiting for a free GPU slot ...", file=sys.stderr, flush=True)) \
                if gpu else _no_slot():
            if gpu:
                try:
                    check_deadline(cfg)
                except Deadline as error:
                    print(f"refused: {error}", file=sys.stderr)
                    return 3
            call.start()
            status, summary, outputs, error = "failed", {}, {}, None
            try:
                with tool_output(call.dir / "tool.log") as console:
                    call.console = console
                    try:
                        status, summary, outputs = body(call)
                    finally:
                        call.console = None
            except (Terminated, KeyboardInterrupt):
                status, error = "failed", "terminated"
            except Exception as exc:          # noqa: BLE001 -- the call fails, the error is recorded
                status, error = "failed", f"{type(exc).__name__}: {exc}"
                (call.dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            record = call.end(status, summary, outputs, error)
    except (Terminated, KeyboardInterrupt):
        print("terminated before the call started", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(json_safe({**record, "call_dir": _relpath(call.dir, WORKSPACE)}), indent=1))
    return 0 if record["status"] in COMPLETED else 1


@contextmanager
def _no_slot():
    yield None


def finish(code_or_error) -> int:
    """The exit code of a validation error (printed)."""
    if isinstance(code_or_error, Invalid):
        print(f"invalid: {code_or_error}", file=sys.stderr)
        return 2
    if isinstance(code_or_error, Mismatch):
        print(f"input check failed: {code_or_error}", file=sys.stderr)
        return 4
    raise code_or_error


def call_status(done: int, total: int) -> str:
    return "done" if done == total and total > 0 else "partial" if done > 0 else "failed"


def dry_run(tool: str, name: str, **fields) -> int:
    print(json.dumps(json_safe({"dry_run": True, "valid": True, "tool": tool, "name": name, **fields}), indent=1))
    return 0


# ---------------------------------------------------------------- per-item work in worker processes

def run_items(call: Call, init: Callable, init_args: tuple, work: Callable, tasks: list, workers: int,
              what: str, fail: Callable[[Any, str], Any]) -> list:
    """`work((index, task))` -> (index, result) for every task: in this process (workers 1), else in
    `workers` spawned processes, each initialised once by `init(*init_args, log)` (its own SHAPE tool
    instance). A task whose worker process died gets `fail(task, error)`. Returns the results in task
    order; reports progress on the console."""
    results: list = [None] * len(tasks)
    step = max(1, len(tasks) // 20)
    workers = max(1, min(int(workers), len(tasks)))

    def report(n: int) -> None:
        if n % step == 0 or n == len(tasks):
            call.progress(f"{n}/{len(tasks)} {what}")

    if workers == 1:
        init(*init_args, None)
        for index, task in enumerate(tasks):
            _, results[index] = work((index, task))
            report(index + 1)
        return results
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # not multiprocessing.Pool: it waits forever for the task of a worker that died (a CUDA crash, the
    # OOM killer); the executor fails that task (BrokenProcessPool)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=init, initargs=(*init_args, str(call.dir / "tool.log")))
    finished = False
    try:
        futures = {pool.submit(work, (index, task)): index for index, task in enumerate(tasks)}
        for n, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            try:
                _, results[index] = future.result()
            except Exception as exc:          # noqa: BLE001 -- the worker process died (see tool.log)
                results[index] = fail(tasks[index], f"{type(exc).__name__}: {exc} (see tool.log)")
            report(n)
        finished = True
    finally:
        if not finished:                      # terminated: stop the workers now
            for process in list((getattr(pool, "_processes", None) or {}).values()):
                process.terminate()
        pool.shutdown(wait=finished, cancel_futures=True)
    return results


def remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _drop_tools_dir_from_path() -> None:
    """A tool imports only this module from tools/; afterwards tools/ leaves sys.path, so a tool file
    can never hide an installed module of the same name (tools/coverage.py once hid the `coverage`
    package from torch). Spawned workers re-run their tool script, which imports this module again."""
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != TOOLS_DIR]


_drop_tools_dir_from_path()
