"""The llm comparison arm (contract amendment A5): the SHAPE arm's model without SHAPE's loop or the analyzer.

  llm    (A5) LLM-written hand-crafted features: no analyzer, no training, no pretrained weights. One request returns a
         Python module defining `features(crops)` (`FeatureModule`). The task text (`feature_brief`) states facts only:
         the data (the cell count, the 6 channels named as external.CHANNELS, 64 x 64 px uint8 crops centred on a
         nucleus, the per-site percentile stretch), the goal, the interface, the packages and the compute budget; it
         never shows the CARD, the analyzer, the CellProfiler-arm code, SHAPE's readouts, a label, the metadata or a
         score. The module passes a static check (`static_check`: the imports and names it may use) and then runs in a
         separate Python process (`run_module`, WORKER: its own session, a fresh temporary cwd outside the run tree,
         no credential in its environment, an audit hook that blocks network, subprocesses, native code it wrote and
         an open under the benchmark's roots) that receives only the crops: the harness reads the rows of images.npy
         in prep/cells.csv order and sends them in chunks of FEATURE_CHUNK crops, first the SMOKE_CROPS first crops
         (the smoke run), then every crop (the full run), with FEATURE_CPUS cores and a GPU (else NOT_READY). The
         output of every chunk must be a float array (n, d), 1 <= d <= MAX_DIM, the same d for every chunk (NaN
         allowed). A failed static check, a crash or an invalid output goes back to the model as its error text (the
         traceback tail, at most about ERROR_CHARS characters) with its module, and the model writes a corrected
         module; no result or metric is ever fed back. After the last failed attempt the run is failed (no fallback).
         Budget (A5 as amended 2026-10-13, user decision: the SHAPE and ReAct arms' budget in kind and size; before,
         3 attempts within a fixed 5 h from the start of the run): at most `max_attempts` attempts = the SHAPE arm's
         max_rounds (config.py, 8; one request and its module per attempt, as one SHAPE round holds one call), and the
         SHAPE arms' wall-clock rule (TIME_RULE): the job's deadline (`resume.deadline`: the Slurm limit, on the VM
         RXRX_DEADLINE_UTC, else none), with SHAPE's per-call time cap -- a module runs until RESERVE_MINUTES before
         the deadline (the representation verifier's `max_minutes`), then it is stopped, which ends the run -- and
         SHAPE's resource stop: no attempt starts within RESERVE_MINUTES of the deadline. Without a deadline nothing
         caps a module's time (as nothing caps a SHAPE call's). The task text states the attempts and the time left
         (`time_text`; masked when a run's design is compared, `llm_design_problems`). The delivered embedding
         is scored as an external feature arm (score.py: native = one global centre-scale, as cellprofiler; matched =
         the per-plate centre-scale); AGENT_RUN.json carries a post-hoc audit of its text (`audit_tokens`).
The react arm is react_arm.py (the shared ReAct runner, benchmarks/_shared/REACT_ARM_CONTRACT.md).

On the VM (contract amendment A8, resume.py) `resume=True` resumes an unfinished run in place: llm keeps its logged
attempts and restarts the attempt that was running.

A run directory (<results root>/representation_RxRx1/llm_<cell type>_<commit8>_<UTC>_<job id>/) holds:
  RUN_META.json             provenance, the SHAPE arm's keys (score.py's stale check reads `provenance`), the sha256
                            of this module; its time budget, chunk, smoke-run size and attempts; the reasoner's
                            reasoning_effort and max_completion_tokens (run_rxrx1.reasoner: the SHAPE arm's)
  AGENT_RUN.json            timing, the case, requests, reasoner tokens, write errors
  logs/requests.jsonl       every request: step, try, prompt, reply (with the module source), usage, error,
                            kind (gateway / invalid / static_check / crash / timeout / invalid_output), seconds
  logs/agent_log.jsonl      the arm's events
  modules/attempt_<k>.py    the module source of attempt k, as run (its sha256 = code_sha256)
  modules/worker.py         the WORKER program that ran it
  logs/attempts.jsonl       per attempt: code_sha256, the stage it reached (static_check / crash / timeout /
                            invalid_output / done), its error, the smoke-run and full-run seconds
  logs/attempt_<k>.stderr.log  the module process's stderr (its prints and warnings)
  submission/embeddings.npy float32, one row per FULL cell in prep/cells.csv order (a delivered run only)
  submission/EMBED.json     its record: code_sha256, d, seconds, the non-finite count, the prep and images.npy sha256
  submission/decision.json  use (attempt_<k>, or null for a failed run), the case
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[3]
ARMS = ("llm",)                    # the react arm is react_arm.py (the shared ReAct runner)
GATEWAY_RETRY_SECONDS = 30         # wait after a gateway error before the next try (SHAPE's retry wait)
_sleep = time.sleep                # tests replace it

# ---- llm (A5): LLM-written hand-crafted features ----------------------------------------------------------------
FEATURE_CPUS, FEATURE_GPUS = 16, 1
# A5 as amended (module docstring): SHAPE's per-call reserve before the deadline, and the wall-clock rule in words
RESERVE_MINUTES = 15               # = shape.agent.verification.verifiers.representation.RESERVE_MINUTES (tested)
TIME_RULE = (f"the SHAPE arms' rule: the job's deadline (Slurm limit, RXRX_DEADLINE_UTC on the VM, else none); a "
             f"module stops and no attempt starts within {RESERVE_MINUTES} min of it")
TIME_LINE = re.compile(r"^Time: .*$", re.MULTILINE)   # the task text's clock line (`time_text`)
FEATURE_CHUNK = 1024               # crops per features() call
SMOKE_CROPS = 256                  # the smoke run: the first crops of prep/cells.csv
MAX_DIM = 4096
ERROR_CHARS = 2000                 # the tail of a traceback / stderr sent back to the model
STRETCH_PERCENTILES = (0.1, 99.9)  # the task-116 prep (harness/data.py stretch_site): per site and channel
# the static check (`static_check`): the modules a module may import (numpy, scipy, scikit-image, pandas, torch, and
# standard modules without file, network or process access), the submodules of those it may not, and the names it may
# not use (as a name or an attribute, or imported by name), each with what it would reach
ALLOWED_PACKAGES = ("numpy", "scipy", "skimage", "pandas", "torch")
ALLOWED_STANDARD = ("__future__", "abc", "bisect", "cmath", "collections", "contextlib", "copy", "dataclasses", "enum",
                    "functools", "heapq", "itertools", "math", "numbers", "operator", "random", "re", "statistics",
                    "string", "threading", "time", "typing", "warnings")
FORBIDDEN_SUBMODULES = {"torch.hub": "pretrained weights", "torch.utils.model_zoo": "pretrained weights",
                        "torch.multiprocessing": "a subprocess", "skimage.io": "file access",
                        "skimage.data": "file access", "scipy.io": "file access", "scipy.datasets": "file access",
                        "pandas.io": "file access"}
FORBIDDEN_NAMES = {
    **dict.fromkeys(("open", "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt",
                     "fromfile", "tofile", "memmap", "open_memmap", "imread", "imsave", "imwrite", "io", "datasets",
                     "to_csv", "to_pickle", "to_parquet", "to_hdf", "to_feather", "to_excel", "to_json", "to_sql",
                     "fromregex", "from_file", "DataSource", "load_npz", "ExcelFile", "HDFStore"), "file access"),
    **dict.fromkeys(("system", "popen", "ProcessPoolExecutor"), "a subprocess"),
    **dict.fromkeys(("distributed",), "a subprocess or network"),
    **dict.fromkeys(("cpp_extension",), "native code"),
    **dict.fromkeys(("hub", "model_zoo", "from_pretrained", "load_state_dict_from_url"), "pretrained weights"),
    **dict.fromkeys(("builtins", "os", "sys", "importlib"), "interpreter or file access"),
    **dict.fromkeys(("__builtins__", "__loader__", "__spec__", "__file__", "__globals__", "__subclasses__",
                     "__code__", "__dict__", "__class__", "__base__", "__bases__", "__mro__", "getattr", "setattr",
                     "delattr", "globals", "locals", "vars"), "interpreter access"),
}
# the post-hoc audit of a delivered module (`audit_tokens`): banned-like words anywhere in its text, comments included
AUDIT_MODULES = ("socket", "ssl", "urllib", "http", "requests", "subprocess", "multiprocessing", "ctypes", "pathlib",
                 "shutil", "pickle", "tempfile", "glob")
# the module's process: an audit hook (WORKER) blocks these events once the allowed packages are imported, and an open
# of a path under the roots the harness denies (the results root, the run's benchmark root, the preps) or of
# /proc/*/environ; a ctypes.dlopen of a library in the module's own directory (native code it wrote)
BLOCKED_EVENTS = ("socket.connect", "socket.bind", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn",
                  "os.spawn", "os.fork", "os.forkpty")
PREIMPORTS = ("scipy", "skimage", "pandas", "torch")     # imported (when installed) before the hook enforces
FORBIDDEN_BUILTINS = {"__import__": "a dynamic import", "eval": "dynamic code", "exec": "dynamic code",
                      "compile": "dynamic code", "breakpoint": "the debugger"}                # as names only
READER_PREFIX = "read_"            # pandas readers (read_csv, ...): file access
# the environment the module's process gets: no credential (LLM_API_KEY stays in the harness), no PYTHON* variable
CHILD_ENV = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# set to the module's own fresh temporary directory (its cwd), outside the run tree
CHILD_TEMP_ENV = ("TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME", "MPLCONFIGDIR", "NUMBA_CACHE_DIR", "TRITON_CACHE_DIR",
                  "CUDA_CACHE_PATH")

# The module's process (`python -I modules/worker.py <config JSON>`, cwd its own fresh temporary directory): frames of
# an 8-byte little-endian length and a payload over its stdin / stdout (its own prints go to stderr). In: the module
# source, then one chunk of crops per frame (np.save bytes), then an empty frame. Out: R (the module loaded), O +
# np.save bytes (one chunk's result), E + a traceback, V + why the result is not a numeric array. The config: the
# allowed directory (its cwd), the denied roots, the blocked audit events, the packages imported before the hook
# enforces (`guard`, installed first, enforces after those imports).
WORKER = '''import io, json, linecache, os, re, struct, sys, traceback

config = json.loads(sys.argv[1])


def forms(name):
    """A path as given (absolute) and resolved (links followed): a root reached through an alias is the same root."""
    return {os.path.normcase(os.path.abspath(name)), os.path.normcase(os.path.realpath(name))}


ALLOW = sorted(forms(config["allow"]))
DENY = sorted({form for p in config["deny"] for form in forms(p)})
BLOCKED = set(config["blocked"])
ENVIRON = re.compile("^/proc/[^/]+/environ$")
state = {"enforce": False}


def under(path, root):
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def allowed(path):
    return any(under(path, root) for root in ALLOW)


def guard(event, args):
    if event == "ctypes.dlopen":
        name = os.fsdecode(args[0]) if args and isinstance(args[0], (str, bytes, os.PathLike)) else ""
        if state["enforce"] and os.sep in name and any(allowed(path) for path in forms(name)):
            raise PermissionError(f"blocked by the harness: ctypes.dlopen({name!r}) (a library written by the module)")
    elif not state["enforce"]:
        return
    elif event in BLOCKED:
        raise PermissionError(f"blocked by the harness: {event} (no network or subprocess)")
    elif event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        given = os.fsdecode(args[0])
        paths = forms(given)
        if any(ENVIRON.match(path.replace(os.sep, "/")) for path in paths) or any(
                not allowed(path) and any(under(path, root) for root in DENY) for path in paths):
            raise PermissionError(f"blocked by the harness: open({given!r}) (no access to the benchmark's files)")


sys.addaudithook(guard)
in_fd, out_fd = os.dup(0), os.dup(1)
null = os.open(os.devnull, os.O_RDWR)
os.dup2(null, 0)
os.dup2(2, 1)
sys.stdin, sys.stdout = open(os.devnull), sys.stderr
frames_in, frames_out = os.fdopen(in_fd, "rb"), os.fdopen(out_fd, "wb")


def receive():
    head = frames_in.read(8)
    return frames_in.read(struct.unpack("<Q", head)[0]) if len(head) == 8 else b""


def send(tag, body=b""):
    frames_out.write(struct.pack("<Q", len(body) + 1))
    frames_out.write(tag)
    frames_out.write(body)
    frames_out.flush()


source = receive().decode("utf-8")              # first: the harness's write of it never waits on an import
import numpy as np

for package in config["preimports"]:
    try:
        __import__(package)
    except Exception:  # noqa: BLE001 -- a package that is not installed stays unavailable to the module
        pass
state["enforce"] = True
linecache.cache["features_module.py"] = (len(source), None, source.splitlines(True), "features_module.py")
namespace = {"__name__": "features_module"}
try:
    exec(compile(source, "features_module.py", "exec"), namespace)
    if not callable(namespace.get("features")):
        raise TypeError("the module defines no callable features(crops)")
except BaseException:
    send(b"E", traceback.format_exc().encode("utf-8", "replace"))
    sys.exit(0)
send(b"R")
while True:
    body = receive()
    if not body:
        break
    try:
        result = namespace["features"](np.load(io.BytesIO(body), allow_pickle=False))
        array = np.asarray(result)
    except BaseException:
        send(b"E", traceback.format_exc().encode("utf-8", "replace"))
        sys.exit(0)
    if array.dtype.hasobject:
        send(b"V", f"the result is not a numeric array ({type(result).__name__} of dtype {array.dtype})".encode())
        sys.exit(0)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    send(b"O", buffer.getvalue())
'''


class FeatureModule(BaseModel):
    """The llm reply (A5): the Python module."""
    reasoning: str = Field(description="your reasoning")
    module: str = Field(description="the complete Python source of the module that defines features(crops)")


class GatewayError(RuntimeError):
    """The engine returned no answer (an error reply or an exception)."""


class ModuleFailed(ValueError):
    """An llm module that cannot deliver; its message is the error text the model gets back. kind: static_check,
    crash, timeout or invalid_output."""

    def __init__(self, kind: str, text: str):
        super().__init__(text)
        self.kind = kind


def deny_roots(paths) -> list[str]:
    """The llm module's denied roots: each path as given (absolute) and resolved (A8: on the VM the benchmark's
    directories are reached through mirror links), sorted and unique. The worker checks an opened path in both forms
    too."""
    return sorted({str(form) for path in paths for form in (Path(path).absolute(), Path(path).resolve())})


def _tail(text: str, limit: int = ERROR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _content(reply: Any, schema):
    """The reply as `schema`; GatewayError for an engine error, ValueError for a reply that is not the schema."""
    if isinstance(reply, dict) and "content" not in reply:
        raise GatewayError(f"{reply.get('error') or 'no content'}: {str(reply.get('message') or '')[:300]}".strip())
    content = reply.get("content") if isinstance(reply, dict) else reply
    if isinstance(content, schema):
        return content
    try:
        if isinstance(content, str):
            return schema.model_validate_json(content)
        if isinstance(content, dict):
            return schema.model_validate(content)
        if hasattr(content, "model_dump"):
            return schema.model_validate(content.model_dump())
    except Exception as error:  # noqa: BLE001 -- pydantic's ValidationError: the answer is invalid
        raise ValueError(f"the reply does not fit the schema: {error}") from None
    raise ValueError(f"the reply is not a {schema.__name__} ({type(content).__name__})")


def _reply_record(reply: Any) -> Any:
    content = reply.get("content", reply) if isinstance(reply, dict) else reply
    return content.model_dump() if hasattr(content, "model_dump") else content


class _Run:
    """The shared part of both arms: the run directory, the requests with their tries, the records."""

    def __init__(self, root: Path, run_dir: Path, *, arm: str, cell_type: str, model: str, engine,
                 resume: bool = False):
        import shape
        from shape.engine import usage

        from benchmarks.representation_RxRx1.runners import control, resume as resume_mod
        from benchmarks.representation_RxRx1.runners.prepare import load

        if engine is None:
            raise SystemExit(f"NOT_READY: no reasoner engine for {model}; the {arm} arm does not start without one")
        if REPO not in Path(shape.__file__).resolve().parents:
            raise SystemExit(f"`import shape` resolves to {shape.__file__}, not inside {REPO}")
        # A8: an unfinished run resumed in place (resume.Resume), else a new run directory
        self.state = resume_mod.Resume.open(run_dir, arm) if resume else None
        if self.state is None and (run_dir / "RUN_META.json").exists():
            raise SystemExit(f"{run_dir} holds a run already; use a new run directory")
        self.recording = resume_mod.on_vm()                  # the replay records (A8), VM only
        self.summary, self.cells = load(root)
        self.provenance = {**control.provenance(), "prep_cells_sha256": self.summary["cells_sha256"]}
        self.control, self.run_dir, self.arm, self.model, self.engine = control, run_dir, arm, model, engine
        self.cell_type = cell_type
        self.requests = 0
        self.last_error: Optional[str] = None                # why the last failed step's last try was unusable
        self.write_errors: list = []
        self.started = time.time()
        self.tokens_before = usage.snapshot()
        self._shape_file = str(Path(shape.__file__).resolve())

    def write_meta(self, extra: dict) -> None:
        """RUN_META.json: the keys both arms record, then the arm's own (a resume adds its entry, `Resume.write_meta`,
        after checking the run's code and settings)."""
        from benchmarks.representation_RxRx1.runners import resume as resume_mod
        from benchmarks.representation_RxRx1.runners.prepare import sha256_file

        meta = {
            "commit": self.control.git_commit(), "expected_commit": os.environ.get("RXRX_EXPECTED_COMMIT"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(), "cell_type": self.cell_type,
            "agent": self.arm, "model": self.model, **extra, "shape_file": self._shape_file,
            "provenance": self.provenance, "llm_arms_sha256": sha256_file(Path(__file__)),
            "require_agentic": os.environ.get("SHAPE_REQUIRE_AGENTIC"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            **resume_mod.platform_keys()}
        if self.state is None:
            self.control.write_json(self.run_dir / "RUN_META.json", meta)
        else:
            self.state.write_meta(meta)

    # ---- records ---------------------------------------------------------------------------------------------
    def _append(self, name: str, record: dict) -> None:
        from benchmarks.representation_RxRx1.runners.resume import append_durable

        try:
            if self.recording:                                # A8 on the VM: the records survive a power cut
                append_durable(self.run_dir / "logs" / name, {"time": time.time(), **record})
            else:
                self.control.append_jsonl(self.run_dir / "logs" / name, {"time": time.time(), **record})
        except OSError as error:                              # after fs_retry: the run goes on, the failure recorded
            self.write_errors.append({"time": time.time(), "what": name, "error": f"{type(error).__name__}: {error}"})

    def _log_request(self, record: dict) -> bool:
        """Appends a request's record to logs/requests.jsonl; True for a replayed request (a resume's ReplayEngine
        served its logged reply; its prompt was compared before): the new record, marked `replayed`, after the
        divergence guard -- the same step and try, and a reply that is valid now iff it was valid then."""
        from benchmarks.representation_RxRx1.runners.resume import ReplayDiverged

        served = getattr(self.engine, "served", None) if self.state is not None else None
        if served is not None:
            if ((served.get("step"), served.get("try")) != (record["step"], record["try"])
                    or (served.get("error") is None) != (record.get("error") is None)):
                raise ReplayDiverged(f"{self.run_dir}: logged request step {served.get('step')} try {served.get('try')}"
                                     f" (error: {served.get('error')}) replays as step {record['step']} try "
                                     f"{record['try']} (error: {record.get('error')})")
            record = {**record, "replayed": self.state.number}
        self._append("requests.jsonl", record)
        return served is not None

    # ---- one request, `tries` tries ---------------------------------------------------------------------------
    def ask(self, step: int, prompt_of, schema, check, until: Optional[float] = None, *, tries: int, first: int = 1,
            error: Optional[str] = None):
        """`check(answer)` of the first valid reply to `prompt_of(error)` (error: why the previous reply could not be
        used, else None), at most `tries` tries, none started at or after `until` (a time.time() deadline); None
        when every try fails. A ValueError's `kind` attribute (ModuleFailed) names the kind of invalid answer.
        `first` / `error`: a resumed llm run's next try and the error of its last logged one."""
        for attempt in range(first, tries + 1):
            if until is not None and time.time() >= until:
                error = f"{error}; then the time budget was spent" if error else "the time budget was spent"
                break
            prompt = prompt_of(error)
            started, reply, record = time.time(), None, {"step": step, "try": attempt, "prompt": prompt}
            self.requests += 1
            try:
                try:
                    reply = self.engine.generate([prompt], response_format=schema)    # as SHAPE's `_ask` sends it
                except Exception as exc:  # noqa: BLE001 -- the engine's own failure is a gateway error
                    raise GatewayError(f"{type(exc).__name__}: {exc}") from None
                if self.recording and self.arm == "llm" and isinstance(reply, dict) and reply.get("usage"):
                    # A8: the usage of every reply, before its module runs (minutes): a resume counts it again
                    self._append("usage.jsonl", {"step": step, "try": attempt, "usage": reply["usage"]})
                answer = _content(reply, schema)
                out = check(answer)
            except GatewayError as exc:
                error = f"gateway error: {exc}"
                replayed = self._log_request({**record, "reply": _reply_record(reply), "error": error,
                                              "kind": "gateway", "seconds": round(time.time() - started, 2)})
                if attempt < tries and not replayed:
                    _sleep(GATEWAY_RETRY_SECONDS)
                continue
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                self._log_request({**record, "reply": _reply_record(reply), "error": error,
                                   "kind": getattr(exc, "kind", "invalid"),
                                   "usage": reply.get("usage") if isinstance(reply, dict) else None,
                                   "seconds": round(time.time() - started, 2)})
                continue
            self._log_request({**record, "reply": _reply_record(reply), "error": None,
                               "usage": reply.get("usage") if isinstance(reply, dict) else None,
                               "seconds": round(time.time() - started, 2)})
            return out
        self.log("request_failed", text=f"step {step}: no usable reply in {tries} tries: {error}")
        self.last_error = error
        return None

    # ---- the end -------------------------------------------------------------------------------------------------
    def write_final(self, final: dict, agent: dict) -> dict:
        """decision.json and AGENT_RUN.json (the keys both arms record, then the arm's own `agent` keys; a resumed
        run: its active hours and its resumes)."""
        from shape.engine import usage

        from benchmarks.representation_RxRx1.runners import resume as resume_mod

        self.control.write_json(self.run_dir / "submission" / "decision.json", final)
        tokens = {k: v - self.tokens_before.get(k, 0) for k, v in usage.snapshot().items() if isinstance(v, int)}
        if self.state is not None:
            self.state.close()
        ended = time.time()
        meta = self.state.meta if self.state is not None else None
        self.control.write_json(self.run_dir / "AGENT_RUN.json", {
            "agent": self.arm, "model": self.model, "started": self.started, "ended": ended,
            "wall_hours_used": resume_mod.active_hours(meta, self.started, ended), "stopped_because": final["case"],
            **agent, "requests": self.requests, "tokens": tokens,
            "write_errors": self.write_errors + self.log.write_errors,
            **({"resumes": resume_mod.resumes_record(meta)} if self.state is not None else {})})
        self.log("final", text=json.dumps(final, default=str))
        return final


# ---- llm (A5): LLM-written hand-crafted features ----------------------------------------------------------------

def static_check(source: str) -> list[str]:
    """What the static check rejects in an llm module, one line each ([] = it may run): a syntax error; an import of
    a module outside ALLOWED_PACKAGES / ALLOWED_STANDARD, of a FORBIDDEN_SUBMODULES one, or a relative import; a
    FORBIDDEN_NAMES name or attribute (also imported by name), a pandas reader (read_*), a FORBIDDEN_BUILTINS name.
    A simple check of the source text, not a sandbox."""
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"line {error.lineno}: SyntaxError: {error.msg}"]
    allowed = (*ALLOWED_PACKAGES, *ALLOWED_STANDARD)

    def submodule(name: str) -> Optional[str]:
        return next((r for m, r in FORBIDDEN_SUBMODULES.items() if name == m or name.startswith(m + ".")), None)

    def named(name: str) -> Optional[str]:
        return FORBIDDEN_NAMES.get(name) or ("file access" if name.startswith(READER_PREFIX) else None)

    out: list[str] = []
    for node in ast.walk(tree):
        line = f"line {getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.ImportFrom) and node.level:
            out.append(f"{line}: from {'.' * node.level}{node.module or ''} import ... (a relative import)")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] + [f"{node.module}.{a.name}" for a in node.names])
            for name in names:
                if name.split(".")[0] not in allowed:
                    out.append(f"{line}: import {name} (not an allowed module)")
                elif submodule(name):
                    out.append(f"{line}: import {name} ({submodule(name)})")
            if isinstance(node, ast.ImportFrom):
                out += [f"{line}: from {node.module} import {a.name} ({named(a.name) or FORBIDDEN_BUILTINS[a.name]})"
                        for a in node.names if named(a.name) or a.name in FORBIDDEN_BUILTINS]
        elif isinstance(node, ast.Name) and (named(node.id) or node.id in FORBIDDEN_BUILTINS):
            out.append(f"{line}: {node.id} ({named(node.id) or FORBIDDEN_BUILTINS[node.id]})")
        elif isinstance(node, ast.Attribute) and named(node.attr):
            out.append(f"{line}: .{node.attr} ({named(node.attr)})")
    return list(dict.fromkeys(out))


def static_rules() -> str:
    """The static check in the task text's words."""
    return (f"an import of any module other than {', '.join(ALLOWED_PACKAGES)} and the standard modules "
            f"{', '.join(m for m in ALLOWED_STANDARD if m != '__future__')}; the submodules "
            f"{', '.join(FORBIDDEN_SUBMODULES)}; the names {', '.join(FORBIDDEN_BUILTINS)}; and, as a name or an "
            f"attribute, {', '.join(FORBIDDEN_NAMES)} and every {READER_PREFIX}* name")


def audit_tokens(source: str) -> list[str]:
    """The banned-like words anywhere in a module's text (comments and strings included; the static check reads the
    syntax only): the static check's names, builtins, submodules and AUDIT_MODULES, sorted; normally none."""
    words = {*FORBIDDEN_NAMES, *FORBIDDEN_BUILTINS, *FORBIDDEN_SUBMODULES, *AUDIT_MODULES}
    return sorted(w for w in words if re.search(rf"(?<![\w.]){re.escape(w)}(?!\w)", source))


def feature_resources() -> dict:
    """What the llm arm's job has: the Slurm deadline (epoch seconds, None outside Slurm or without a limit; on the VM
    RXRX_DEADLINE_UTC or None, `resume.deadline`), the CPU cores this process may use, the CUDA GPUs it sees."""
    from shape.agent.verification.base import available_cores

    from benchmarks.representation_RxRx1.runners.resume import deadline

    try:
        import torch

        gpus = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001 -- no torch or no CUDA: no GPU
        gpus = 0
    return {"deadline": deadline(), "cores": int(available_cores()), "gpus": gpus}


def _temp_base() -> Optional[str]:
    """Where a module's fresh temporary directory goes: the job's SLURM_TMPDIR, else TMPDIR (else the system's)."""
    return os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or None


def time_text(deadline: Optional[float]) -> str:
    """The task text's clock line (TIME_LINE): the time left before the job's deadline, or no limit."""
    if deadline is None:
        return "Time: no time limit (the job has no deadline)."
    return (f"Time: the job ends in {max(0.0, deadline - time.time()) / 3600:.2f} h; a module still running "
            f"{RESERVE_MINUTES} min before then is stopped.")


def feature_brief(n_cells: int, cell_type: str, max_attempts: int, deadline: Optional[float] = None) -> str:
    """The llm arm's task as facts (A5): the data, the goal, the interface, the packages, the compute, the time left
    (`time_text` of the job's `deadline`) and the attempts; nothing about which features to compute."""
    from benchmarks.representation_RxRx1.runners.external import CHANNELS

    low, high = STRETCH_PERCENTILES
    return (
        f"Data: {n_cells} single-cell crops of RxRx1 {cell_type} cells, uint8, 64 x 64 px, centred on a nucleus, with\n"
        f"6 fluorescence channels in this order: {', '.join(CHANNELS)}. Before the crops were cut, each\n"
        f"channel of each site image was scaled linearly from its {low:g}th to its {high:g}th percentile onto 0-255\n"
        "(clipped).\n\n"
        "Goal: one label-free feature vector per cell that describes its morphology and state, so that cells and\n"
        "wells can later be compared.\n\n"
        "Interface: a Python module that defines features(crops) -> array. crops is a numpy uint8 array of shape\n"
        f"(n, 6, 64, 64); the result is a float numpy array of shape (n, d) with 1 <= d <= {MAX_DIM}, row i for\n"
        f"crop i, the same d on every call. features is called on chunks of at most {FEATURE_CHUNK} crops; NaN\n"
        f"values are allowed. The module runs first on the first {min(SMOKE_CROPS, n_cells)} crops (a smoke run),\n"
        f"then on all {n_cells} crops.\n\n"
        "Packages: numpy, scipy, scikit-image, pandas, and torch without pretrained weights. The module runs in one\n"
        "separate Python process that receives only the crops (no further process; threads are available through\n"
        "the threading module). Before it runs, a static check rejects (no network, subprocess or file access):\n"
        f"{static_rules()}.\n\n"
        f"Compute: {FEATURE_CPUS} CPUs + {FEATURE_GPUS} GPU for the whole task (every attempt: its module, its smoke\n"
        "run and its full run).\n\n"
        f"{time_text(deadline)}\n\n"
        f"Attempts: at most {max_attempts}. A module still running at the time limit is stopped, and that ends the\n"
        "run. A module that fails the static check, crashes or returns an invalid output before then gets its error\n"
        "text back, and you write a corrected module. No result or score is shown.")


class _Child:
    """One module run: `python -I modules/worker.py <config>` (WORKER) in its own session, with CHILD_ENV only and its
    temporary directories (CHILD_TEMP_ENV) set to `workdir`, its cwd; frames over its stdin / stdout, read by a thread
    so that every wait ends at `deadline`. `deny`: the roots its audit hook blocks an open under (`workdir`
    excepted)."""

    def __init__(self, source: str, *, worker: Path, workdir: Path, stderr_path: Path, deadline: Optional[float],
                 deny=()):
        self.deadline, self.stderr_path = deadline, stderr_path
        env = {k: v for k, v in os.environ.items() if k in CHILD_ENV}
        env.update(dict.fromkeys(CHILD_TEMP_ENV, str(workdir)))
        config = json.dumps({"allow": str(workdir), "deny": [str(p) for p in deny], "blocked": list(BLOCKED_EVENTS),
                             "preimports": list(PREIMPORTS)})
        with stderr_path.open("wb") as stderr:
            self.proc = subprocess.Popen([sys.executable, "-I", str(worker), config], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=stderr, cwd=str(workdir), env=env,
                                         start_new_session=True)
        self.frames: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.send(source.encode("utf-8"), "module import")

    def _read(self) -> None:
        stream = self.proc.stdout
        try:
            while True:
                head = stream.read(8)
                if len(head) < 8:
                    break
                size = struct.unpack("<Q", head)[0]
                body = stream.read(size)
                if len(body) < size:
                    break
                self.frames.put(body)
        except (OSError, ValueError):                         # the pipe closed under the reader
            pass
        finally:
            self.frames.put(None)

    def exited(self, what: str) -> ModuleFailed:
        try:
            code = self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.close()
            code = self.proc.returncode
        try:
            stderr = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stderr = ""
        return ModuleFailed("crash", f"the module's process exited (code {code}) during the {what}"
                                     + (f"; its stderr ends:\n{_tail(stderr)}" if stderr.strip() else ""))

    def send(self, payload: bytes, what: str) -> None:
        try:
            self.proc.stdin.write(struct.pack("<Q", len(payload)))
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
        except OSError:                                       # BrokenPipeError: the process is gone
            raise self.exited(what) from None

    def receive(self, what: str) -> tuple[bytes, bytes]:
        try:
            frame = self.frames.get(timeout=None if self.deadline is None else max(0.0, self.deadline - time.time()))
        except queue.Empty:
            self.close()
            raise ModuleFailed("timeout", f"the {what} was still running {RESERVE_MINUTES} min before the job's time "
                                          "limit and was stopped") from None
        if frame is None:
            raise self.exited(what)
        tag, body = frame[:1], frame[1:]
        if tag == b"E":
            raise ModuleFailed("crash", f"the {what} raised an error:\n{_tail(body.decode('utf-8', 'replace'))}")
        if tag == b"V":
            raise ModuleFailed("invalid_output", f"the {what} returned an invalid output: {body.decode('utf-8')}")
        return tag, body

    def close(self) -> None:
        """Ends the process and, on POSIX, its whole process group (its own session: anything it started too)."""
        if os.name == "posix":
            import signal

            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):     # the group is gone already
                pass
        elif self.proc.poll() is None:                        # Windows: no process groups here, the process itself
            self.proc.kill()
        self.proc.wait()
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except OSError:
                pass


def _output(body: bytes, n_rows: int, dim: Optional[int], what: str):
    """One chunk's result as float32 after the checks of the interface: a float array (n_rows, d), 1 <= d <= MAX_DIM,
    d = `dim` of the earlier chunks."""
    import numpy as np

    array = np.load(io.BytesIO(body), allow_pickle=False)
    problem = None
    if array.ndim != 2 or array.shape[0] != n_rows:
        problem = f"shape {array.shape}, expected (n, d) = ({n_rows}, d) for a chunk of {n_rows} crops"
    elif not 1 <= array.shape[1] <= MAX_DIM:
        problem = f"d = {array.shape[1]}, expected 1 <= d <= {MAX_DIM}"
    elif array.dtype.kind != "f":
        problem = f"dtype {array.dtype}, expected a float array"
    elif dim is not None and array.shape[1] != dim:
        problem = f"d = {array.shape[1]}, while the earlier chunks returned d = {dim}"
    if problem:
        raise ModuleFailed("invalid_output", f"the {what} returned an invalid output: {problem}")
    return array.astype(np.float32)


def run_module(source: str, images, rows, *, worker: Path, stderr_path: Path, deadline: Optional[float], deny=(),
               chunk: Optional[int] = None, smoke: Optional[int] = None):
    """(vectors float32 (len(rows), d) with row i = features of images[rows[i]], timing) of one module, in its own
    process (`_Child`, its cwd a fresh temporary directory under `_temp_base`, removed afterwards; `deny`: the roots its
    audit hook blocks): the smoke run (the first `smoke` rows, default SMOKE_CROPS), then the full run (every row), in
    chunks of `chunk` crops (default FEATURE_CHUNK); a crash, a timeout at `deadline` or an invalid output raises
    ModuleFailed with the error text for the model."""
    import shutil
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix="llm_module_", dir=_temp_base()))
    try:
        return _run_module(source, images, rows, worker=worker, workdir=workdir, stderr_path=stderr_path,
                           deadline=deadline, deny=deny, chunk=chunk or FEATURE_CHUNK, smoke=smoke or SMOKE_CROPS)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_module(source: str, images, rows, *, worker: Path, workdir: Path, stderr_path: Path, deadline: float, deny,
                chunk: int, smoke: int):
    import numpy as np

    def payload(block) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(images[block]), allow_pickle=False)
        return buffer.getvalue()

    n = len(rows)
    started = time.time()
    child = _Child(source, worker=worker, workdir=workdir, stderr_path=stderr_path, deadline=deadline, deny=deny)
    try:
        child.receive("module import")
        smoke_rows = rows[:min(smoke, n)]
        what = f"smoke run (crops 1-{len(smoke_rows)})"
        child.send(payload(smoke_rows), what)
        dim = _output(child.receive(what)[1], len(smoke_rows), None, what).shape[1]
        smoke_seconds = round(time.time() - started, 1)
        vectors = np.empty((n, dim), dtype=np.float32)
        full = time.time()
        for start in range(0, n, chunk):
            block = rows[start:start + chunk]
            what = f"full run (crops {start + 1}-{start + len(block)} of {n})"
            child.send(payload(block), what)
            vectors[start:start + len(block)] = _output(child.receive(what)[1], len(block), dim, what)
        child.send(b"", "end of the run")
        return vectors, {"smoke_seconds": smoke_seconds, "seconds": round(time.time() - full, 1), "dim": int(dim),
                         "nonfinite_values": int((~np.isfinite(vectors)).sum())}
    finally:
        child.close()


class Features(_Run):
    """The llm arm (A5): one request returns a module, checked and run by the harness; at most `max_attempts`
    attempts (SHAPE's max_rounds), each error fed back as text, never a result; SHAPE's wall-clock rule (TIME_RULE)."""

    def __init__(self, root: Path, run_dir: Path, *, arm: str, cell_type: str, model: str, engine,
                 resume: bool = False, max_attempts: int):
        from benchmarks.batch_correction_JUMP.runners.score import jsonl
        from benchmarks.representation_RxRx1.runners import shape_arm

        super().__init__(root, run_dir, arm=arm, cell_type=cell_type, model=model, engine=engine, resume=resume)
        # a resume (A8): the attempts logged so far stay and count against max_attempts (the deadline is the resumed
        # job's, as SHAPE's); the usage of every reply the run received (logs/usage.jsonl, the interrupted attempt's
        # included) counts in its tokens
        self.logged = jsonl(run_dir / "logs" / "requests.jsonl") if (
            self.state is not None and (run_dir / "logs" / "requests.jsonl").exists()) else []
        self._rerun = False                                  # a logged valid attempt's module running again
        if self.state is not None:
            from shape.engine import usage

            used_tokens = run_dir / "logs" / "usage.jsonl"
            for r in jsonl(used_tokens) if used_tokens.exists() else self.logged:
                if isinstance(r.get("usage"), dict):
                    usage.record(r["usage"])
        if max_attempts < 1:
            raise SystemExit(f"NOT_READY: max_attempts must be >= 1, not {max_attempts}")
        self.max_attempts = int(max_attempts)
        resources = feature_resources()                      # the job's deadline, as SHAPE's (TIME_RULE)
        job_end = resources["deadline"]
        self.job_end = job_end
        # no attempt starts and no module runs within RESERVE_MINUTES of the deadline (SHAPE's max_minutes)
        self.deadline = None if job_end is None else job_end - RESERVE_MINUTES * 60
        if self.deadline is not None and self.deadline <= self.started:
            raise SystemExit(f"NOT_READY: the job ends {(job_end - self.started) / 3600:.2f} h from now; the llm arm "
                             f"needs more than {RESERVE_MINUTES} min (the reserve before the deadline)")
        if resources["cores"] < FEATURE_CPUS:
            raise SystemExit(f"NOT_READY: {resources['cores']} CPU cores visible; the llm arm's budget is "
                             f"{FEATURE_CPUS} CPUs")
        if resources["gpus"] < FEATURE_GPUS:
            raise SystemExit(f"NOT_READY: {resources['gpus']} GPUs visible; the llm arm's budget is {FEATURE_GPUS} GPU")
        for sub in ("logs", "submission", "modules"):
            self.control.fs_retry((run_dir / sub).mkdir, parents=True, exist_ok=True)
        self.log = shape_arm.AgentLog(run_dir / "logs" / "agent_log.jsonl")
        self.n = len(self.cells)
        self.brief = feature_brief(self.n, cell_type, self.max_attempts, job_end)
        self.worker = run_dir / "modules" / "worker.py"
        self.control.fs_retry(self.worker.write_bytes, WORKER.encode("utf-8"))
        # what the module's process may not open: the results root, this benchmark's root (runs, caches, the prep and
        # its private labels), the prep and the task-116 preps the crops come from (their cells.csv carry labels)
        # -- each as given and resolved (`deny_roots`), and the prep's parts each, which may be links of their own
        crop_array, root = Path(self.summary["crop_array"]["path"]), Path(root)
        self.deny = deny_roots((run_dir.parent, root, root / "prep", root / "private", root / "external",
                                crop_array.parent, crop_array.parent.parent, crop_array.resolve().parent,
                                crop_array.resolve().parent.parent,
                                *([Path(os.environ["SHAPE_RESULTS_ROOT"])] if os.environ.get("SHAPE_RESULTS_ROOT")
                                  else [])))
        # images.npy read whole into memory (about 1.5 GB for HEPG2) at the first module that runs: plain reads with
        # fs_retry, never a memory map on Lustre (contract amendment A1)
        self.images = None
        self.last_source: Optional[str] = None               # the previous attempt's module, shown with its error
        self.write_meta({
            "time_rule": TIME_RULE, "compute": {"cpus": FEATURE_CPUS, "gpus": FEATURE_GPUS},
            "resources": {**resources, "deadline_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(job_end))
                          if job_end else None},
            "chunk": FEATURE_CHUNK, "smoke_crops": SMOKE_CROPS, "max_dim": MAX_DIM, "max_attempts": self.max_attempts,
            "n_cells": self.n, "crop_array": self.summary["crop_array"]["path"], "python": sys.executable,
            "worker_sha256": hashlib.sha256(WORKER.encode("utf-8")).hexdigest(), "denied_roots": self.deny,
            # the reasoner's settings, as shape_arm / react_arm record them (resume.STABLE_META holds them fixed)
            "reasoning_effort": os.environ.get("SHAPE_REASONING_EFFORT"),
            "max_completion_tokens": getattr(engine, "max_tokens", None)})
        self.log("system", text=f"llm arm (features), model={model}, {self.n} crops, at most {self.max_attempts} "
                                f"attempts, {time_text(job_end)}")

    def prompt(self, error: Optional[str]) -> str:
        brief = self.brief
        if getattr(self, "job_end", None) is not None:      # the clock line as of this request
            brief = TIME_LINE.sub(time_text(self.job_end), brief)
        return ("You write a Python module that computes a hand-crafted feature vector for every single-cell crop of\n"
                "a dataset.\n\n"
                f"{brief}\n\n"
                "Reply in the given schema: reasoning, then module = the complete Python source of the module.\n"
                + (f"\nYour previous module could not be used: {error}\n" if error else "")
                + (f"Your previous module:\n```python\n{self.last_source}\n```\n" if error and self.last_source else "")
                + ("Write a corrected module.\n" if error else ""))

    def check(self, answer: FeatureModule) -> dict:
        """The module of this attempt: written, statically checked, run; ModuleFailed (a ValueError) when it cannot
        deliver."""
        import numpy as np

        k = self.requests
        source = self.last_source = answer.module
        code = hashlib.sha256(source.encode("utf-8")).hexdigest()
        path = self.run_dir / "modules" / f"attempt_{k}.py"
        self.control.fs_retry(path.write_bytes, source.encode("utf-8"))
        record = {"attempt": k, "code_sha256": code, "module": path.relative_to(self.run_dir).as_posix()}
        try:
            rejected = static_check(source)
            if rejected:
                raise ModuleFailed("static_check", "the static check rejected the module: " + "; ".join(rejected))
            if self.images is None:
                self.images = self.control.fs_retry(np.load, self.summary["crop_array"]["path"])
            rows = self.cells.array_row.astype(int).to_numpy()
            vectors, timing = run_module(source, self.images, rows, worker=self.worker,
                                         stderr_path=self.run_dir / "logs" / f"attempt_{k}.stderr.log",
                                         deadline=self.deadline, deny=self.deny)
        except ModuleFailed as error:
            self._append("attempts.jsonl", {**record, "stage": error.kind, "error": str(error)})
            self.log("attempt", text=f"attempt {k}: {error.kind}: {str(error)[:300]}")
            raise
        if not self._rerun:                                  # a rerun's attempt has its done record already
            self._append("attempts.jsonl", {**record, "stage": "done", "error": None, **timing})
        self.log("attempt", text=f"attempt {k}: done, {timing}")
        return {**record, **timing, "vectors": vectors}

    def restart(self) -> tuple[int, Optional[str], Optional[dict]]:
        """A resumed run's state (A8): the logged attempts stay (`requests`, the last module and error, as the run had
        them); the attempt that was running restarts -- its module and stderr files are moved aside and it asks
        again -- except a logged valid attempt whose delivery was not written, whose module runs again from its
        logged reply (no new request). Returns (the next try, the last error, that attempt's done record or None)."""
        from benchmarks.representation_RxRx1.runners.resume import ReplayDiverged, move_aside

        ran = ("static_check", "crash", "timeout", "invalid_output")    # the kinds after which check() ran
        self.requests = len(self.logged)
        modules = [r["reply"]["module"] for r in self.logged
                   if (r.get("error") is None or r.get("kind") in ran) and isinstance(r.get("reply"), dict)
                   and isinstance(r["reply"].get("module"), str)]
        self.last_source = modules[-1] if modules else None
        if self.logged and self.logged[-1].get("error") is None:
            k = self.requests
            stderr = self.run_dir / "logs" / f"attempt_{k}.stderr.log"
            if stderr.exists():
                self.state.moved(stderr, move_aside(stderr, self.state.number))
            self.log("restart", text=f"attempt {k} was valid but not delivered: its module runs again")
            self._rerun = True
            try:
                return k + 1, None, self.check(FeatureModule.model_validate(self.logged[-1]["reply"]))
            except ModuleFailed as error:
                raise ReplayDiverged(f"{self.run_dir}: the logged valid module of attempt {k} failed when run again: "
                                     f"{error}") from None
            finally:
                self._rerun = False
        k = self.requests + 1
        for path in (self.run_dir / "modules" / f"attempt_{k}.py", self.run_dir / "logs" / f"attempt_{k}.stderr.log"):
            if path.exists():
                self.state.moved(path, move_aside(path, self.state.number))
        self.log("restart", text=f"{self.requests} attempt(s) logged; attempt {k} restarts")
        return k, self.logged[-1].get("error") if self.logged else None, None

    def run(self) -> dict:
        import numpy as np

        from benchmarks.representation_RxRx1.runners.prepare import sha256_file

        if self.state is None:
            done = self.ask(1, self.prompt, FeatureModule, self.check, until=self.deadline, tries=self.max_attempts)
        else:
            first, error, done = self.restart()
            if done is None:
                done = self.ask(1, self.prompt, FeatureModule, self.check, until=self.deadline, first=first,
                                error=error, tries=self.max_attempts)
        attempts = {"attempts": self.requests, "max_attempts": self.max_attempts, "time_rule": TIME_RULE,
                    "tool_calls": 0}
        if done is None:
            final = {"use": None, "provisional": False, "status": "failed", "case": "no_valid_module",
                     "reason": f"no usable module in {self.requests} attempt(s): {self.last_error}"}
            return self.write_final(final, attempts)
        attempts["audit"] = {"banned_like_tokens": audit_tokens(self.last_source),
                             "rule": "words of the static check's lists anywhere in the delivered module's text "
                                     "(comments and strings included), post hoc; normally none"}
        out = self.run_dir / "submission" / "embeddings.npy"
        self.control.fs_retry(np.save, out, done["vectors"])
        self.control.write_json(self.run_dir / "submission" / "EMBED.json", {
            "arm": "llm", "label_free": True, "attempt": done["attempt"], "module": done["module"],
            "code_sha256": done["code_sha256"], "dim": done["dim"], "n_cells": self.n, "dtype": "float32",
            "rows": "prep/cells.csv order", "cells_sha256": self.summary["cells_sha256"],
            "crop_array": self.summary["crop_array"]["path"], "crop_array_sha256": self.summary["crop_array"]["sha256"],
            "chunk": FEATURE_CHUNK, "smoke_seconds": done["smoke_seconds"], "seconds": done["seconds"],
            "nonfinite_values": done["nonfinite_values"], "embeddings_sha256": sha256_file(out),
            "llm_arms_sha256": sha256_file(Path(__file__)), "commit": self.control.git_commit(),
            "stored": "the module's output as float32; score.post_process scales it"})
        final = {"use": f"attempt_{done['attempt']}", "provisional": False, "status": "delivered", "case": "features",
                 "reason": f"the module of attempt {done['attempt']}, as it is", "code_sha256": done["code_sha256"],
                 "embeddings": out.relative_to(self.run_dir).as_posix(), "dim": done["dim"]}
        return self.write_final(final, attempts)


def llm_design_problems(run: Path, meta: dict) -> dict:
    """{} when the llm run `run` (its RUN_META.json `meta`) was made with the current llm design: its first request's
    prompt is the one this code writes for its cell count, cell type and attempts (the clock line masked: TIME_LINE),
    and its worker is this WORKER, with the settings the run records (time rule, compute, chunk, smoke crops, max dim)
    equal to this module's -- not the file's sha256: a run of this file from before the react arm moved to
    react_arm.py (and before A11) is current when its llm design is (A11). Its attempts must be the SHAPE run's
    max_rounds (score.llm_arm checks that). A run of the fixed 5 h budget (before 2026-10-13) is not current."""
    from types import SimpleNamespace

    from benchmarks.batch_correction_JUMP.runners.score import jsonl

    path = run / "logs" / "requests.jsonl"
    first = (jsonl(path) or [{}])[0] if path.exists() else {}
    brief = feature_brief(int(meta.get("n_cells") or 0), str(meta.get("cell_type")), int(meta.get("max_attempts") or 0))
    expected = Features.prompt(SimpleNamespace(brief=brief, last_source=None), None)
    now = {"time_rule": TIME_RULE, "compute": {"cpus": FEATURE_CPUS, "gpus": FEATURE_GPUS},
           "chunk": FEATURE_CHUNK, "smoke_crops": SMOKE_CROPS, "max_dim": MAX_DIM,
           "worker_sha256": hashlib.sha256(WORKER.encode("utf-8")).hexdigest()}
    out = {key: {"run": meta.get(key), "now": value} for key, value in now.items() if meta.get(key) != value}
    masked = TIME_LINE.sub("Time: <masked>", str(first.get("prompt")))
    if masked != TIME_LINE.sub("Time: <masked>", expected):
        out["prompt"] = {"run_sha256": hashlib.sha256(str(first.get("prompt")).encode("utf-8")).hexdigest(),
                         "now_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest()}
    return out


def load_features(run: Path, summary: dict, n_cells: int):
    """(vectors, EMBED.json) of a delivered llm run for score.py; refused when made for another prep table, another
    images.npy or another cell count, or when embeddings.npy changed since the run."""
    import numpy as np

    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    path = run / "submission" / "EMBED.json"
    if not path.exists():
        raise SystemExit(f"NO_EMBEDDING: the llm run {run} has no submission/EMBED.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("cells_sha256") != summary["cells_sha256"] or record.get("n_cells") != n_cells:
        raise SystemExit(f"ARM_STALE: the llm run {run} was made for another prep")
    if record.get("crop_array_sha256") != summary["crop_array"]["sha256"]:
        raise SystemExit(f"ARM_STALE: the llm run {run} was made from another images.npy")
    embeddings = run / "submission" / "embeddings.npy"
    if sha256_file(embeddings) != record["embeddings_sha256"]:
        raise SystemExit(f"INTEGRITY_FAILED: {embeddings} changed since the llm run")
    return np.load(embeddings), record


def run(root: Path, run_dir: Path, *, arm: str, cell_type: str, model: str, engine, resume: bool = False,
        max_attempts: Optional[int] = None) -> dict:
    """One `llm` run on the prepared cell type under `root` into the new directory `run_dir`; `resume`: the
    unfinished run in `run_dir`, resumed in place (A8); `max_attempts`: SHAPE's max_rounds (config.py) unless given.
    The react arm is react_arm.run (the shared ReAct runner)."""
    from benchmarks.representation_RxRx1.runners.config import DEFAULT

    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; one of {ARMS} (react: react_arm.py)")
    return Features(root, run_dir, arm=arm, cell_type=cell_type, model=model, engine=engine, resume=resume,
                    max_attempts=DEFAULT.max_rounds if max_attempts is None else max_attempts).run()
