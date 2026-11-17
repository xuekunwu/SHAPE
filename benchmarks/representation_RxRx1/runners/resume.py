"""The VM platform (no Slurm) and the resume of an interrupted run by replay (contract amendment A8), plus the VM
driver's helpers.

Platform. `RXRX_PLATFORM=vm` selects the VM: the deadline of a stage is
`RXRX_DEADLINE_UTC` (YYYY-MM-DDTHH:MM:SSZ) or none (`deadline`), and a new run records `platform` and its
`run_tag` (`RXRX_RUN_TAG`, a host label plus a UTC start stamp) in RUN_META.json (`platform_keys`). Without it
everything is the cluster's: the deadline is the Slurm job's (`squeue %L`) and RUN_META.json gains no key.

Resume (`Resume.open`, the shape, react and llm stages' `--resume`). A run directory whose RUN_META.json exists and
whose AGENT_RUN.json does not (the arm never finished) is resumed in place, never duplicated:
  - its logs are moved (never deleted) into logs/resume_<n>/; shape and react write them again from the start. The
    replay source is the logs of the last process that went live (or of the original run): a resume that stopped
    during its own replay (`live_from` None) wrote only the echoes it got through, so the next one replays its
    predecessor's source (`replay_source`) and no logged reply is lost;
  - shape and react replay: the arm runs again from trial 0 with a `ReplayEngine` in front of the reasoner engine.
    It serves the logged replies in order (logs/reasoner.jsonl, written by `RecordingEngine`; react also moves its
    runner records, react/, into logs/resume_<n>/) without calling the model, each with its logged usage counted,
    and switches to live requests after the last one. Every analyzer call goes through `CallCache`: a finished call
    (its CALL_RECORD.json, written when the tool returned) is served from its own directory (ssl_calls/<config digest>/) after its outputs' sha256
    are checked -- the same result object, resume_best / adata / training facts included -- and the tool is not
    called; a call that was running when the run stopped (its directory without CALL_RECORD.json) is moved aside
    (<dir>.interrupted_r<n>) and runs again from its start. The tool's own reuse path
    (`reuse_embedding`, "reused_cached_embedding") is not used: it re-clusters, rewrites the h5ad and returns no
    resume_best, training or val_history;
  - the divergence guard: a replayed reply of another schema, a replayed prompt that is not the logged one (shape:
    sha256 with the clock text masked, shape `TIME_TEXT`, react `REACT_TIME_TEXT` and its python steps'
    observations, `REACT_PYTHON_TEXT`: a replayed python step runs its logged code again), a served call out of
    order or of another configuration, or a live request or call while logged replies or finished calls remain
    unreplayed, stops the run with REPLAY_DIVERGED (a SystemExit: no fallback, no improvisation);
  - the replay records are fsynced (`write_durable`, `append_durable`); a served call reports the caller's seconds
    of the logged call (`outer_seconds`); a running call touches <query_cache_dir>/HEARTBEAT every
    HEARTBEAT_SECONDS, so the active time measured after a kill covers its tail;
  - llm restarts its current attempt (llm_arms.Features): the attempts already logged stay; the interrupted attempt's
    module and stderr files are moved aside;
  - RUN_META.json records every resume (`resumes`: its number, start, run tag, host, deadline, the active seconds of
    the run before it, the logged requests and finished calls it found, where it went live -- `replayed_requests`, the
    requests it served before the first live action -- and what it moved aside); AGENT_RUN.json repeats the list.

`python -m benchmarks.representation_RxRx1.runners.resume find-run|status|add-tag ...` are the VM driver's helpers
(the VM driver).
"""
from __future__ import annotations

import argparse
import calendar
import copy
import hashlib
import json
import os
import pickle
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

CALL_RECORD = "CALL_RECORD.json"   # a finished call's record, in its own directory, written last
CALL_RESULT = "CALL_RESULT.pkl"    # the tool's result object as returned (pickle: numpy scalars and tuples survive)
DEADLINE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
HEARTBEAT = "HEARTBEAT"            # <query_cache_dir>/HEARTBEAT, touched while an analyzer call runs
HEARTBEAT_SECONDS = 60             # how often (the active time of a process that stopped inside a call)
# The prompt text that depends on the clock (base.AgenticVerifier._time_text: the wall time so far and, with a
# deadline, the time left); it is masked before a replayed shape prompt is compared with the logged one.
TIME_TEXT = re.compile(r"Wall time so far: [0-9.]+ h(?:; time left before the stage must end [^:]*: [0-9.]+ h)?")
TIME_MASK = "Wall time so far: <masked>"
# the react prompt's clock line (the shared runner's BUDGET section), masked the same way
REACT_TIME_TEXT = re.compile(r"^Time left: .*$", re.MULTILINE)
REACT_TIME_MASK = "Time left: <masked>"
# a python step's observation in the react prompt's history (its seconds and its output are not reproducible when the
# step runs again in a replay), up to the next history entry or the next section; masked the same way
REACT_PYTHON_TEXT = re.compile(r"^observation: python step (\d+): .*?(?=\n\n(?:Round \d+|Python step \d+ \(after "
                               r"round \d+; not a round\))|\n\n## RESPONSE SCHEMA\n)", re.MULTILINE | re.DOTALL)
REACT_PYTHON_MASK = r"observation: python step \1: <masked>"
# the logs an arm writes again from the start when it is resumed (moved into logs/resume_<n>/ first)
ROTATED = {"shape": ("tool_calls.jsonl", "agent_log.jsonl", "input_trial.json", "reasoner.jsonl"),
           "react": ("tool_calls.jsonl", "agent_log.jsonl", "input_trial.json", "reasoner.jsonl"),
           "llm": ()}
# RUN_META.json keys a resume must find unchanged (the code, the prep, the arm's settings); the others (host, slurm
# job, run tag, cores, gpus, deadline, resources) are recorded per resume
STABLE_META = ("commit", "expected_commit", "cell_type", "agent", "model", "max_rounds", "tool_budget",
               "training_budget", "fixed_levers", "split_key", "crop_array", "provenance", "card_sha256",
               "llm_arms_sha256", "react_arm_sha256", "shared_react_sha256", "round_cap", "shape_file",
               "require_agentic", "platform", "replay_records", "expert_config", "seed", "reasoning_effort",
               "max_completion_tokens",
               "time_budget_seconds", "time_rule", "compute",
               "chunk", "smoke_crops", "max_dim", "max_attempts", "n_cells", "worker_sha256")
STAGES = ("shape", "llm", "react")


class ReplayDiverged(SystemExit):
    """The replayed run left the logged trajectory: it stops (never improvises)."""

    def __init__(self, text: str):
        super().__init__(f"REPLAY_DIVERGED: {text}")


# ---- the platform ------------------------------------------------------------------------------------------------

def on_vm() -> bool:
    return os.environ.get("RXRX_PLATFORM") == "vm"


def deadline() -> Optional[float]:
    """Epoch seconds by which the stage must end: the cluster -- the Slurm job's limit (`slurm_deadline`, None outside
    Slurm or without a limit); VM -- RXRX_DEADLINE_UTC, else None (no deadline: no call gets a time cap and the
    resource stop never fires)."""
    if not on_vm():
        from benchmarks.batch_correction_JUMP.runners import shape_arm as jump

        return jump.slurm_deadline()
    text = os.environ.get("RXRX_DEADLINE_UTC")
    if not text:
        return None
    try:
        return float(calendar.timegm(time.strptime(text, DEADLINE_FORMAT)))
    except ValueError:
        raise SystemExit(f"NOT_READY: RXRX_DEADLINE_UTC={text!r} is not {DEADLINE_FORMAT}") from None


def platform_keys() -> dict:
    """The RUN_META.json keys of the platform: none on the cluster (its records are unchanged); on the VM the platform,
    the run tag and `replay_records` (the arm writes its replay records: logs/reasoner.jsonl and the calls'
    CALL_RECORD.json; only such a run can be resumed)."""
    return {"platform": "vm", "run_tag": os.environ.get("RXRX_RUN_TAG"), "replay_records": True} if on_vm() else {}


def utc(epoch: Optional[float] = None) -> str:
    return time.strftime(DEADLINE_FORMAT, time.gmtime(time.time() if epoch is None else epoch))


def _epoch(text: str) -> float:
    return float(calendar.timegm(time.strptime(text, DEADLINE_FORMAT)))


def last_activity(run_dir: Path) -> float:
    """The latest mtime of any file under the run directory (the last write of the process that stopped)."""
    latest = 0.0
    for path in Path(run_dir).rglob("*"):
        try:
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
        except OSError:
            continue
    return latest


def move_aside(path: Path, number: int) -> Path:
    """`path` renamed to <name>.interrupted_r<number> (with _<k> when that exists); never deleted."""
    target = path.with_name(f"{path.name}.interrupted_r{number}")
    k = 1
    while target.exists():
        k += 1
        target = path.with_name(f"{path.name}.interrupted_r{number}_{k}")
    from benchmarks.representation_RxRx1.runners.control import fs_retry

    fs_retry(os.rename, path, target)
    return target


# ---- durable writes (the replay records: they must survive a power cut) -------------------------------------------

def _fsync_dir(directory: Path) -> None:
    """fsync of a directory (its new entries); a platform that cannot open a directory (Windows) skips it."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_durable(path: Path, data: bytes) -> None:
    """`data` into `path` through a temporary file -- flushed, fsynced, renamed, its directory fsynced (fs_retry)."""
    from benchmarks.representation_RxRx1.runners.control import fs_retry

    path = Path(path)

    def write():
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        _fsync_dir(path.parent)

    fs_retry(write)


def write_json_durable(path: Path, payload: dict) -> None:
    write_durable(path, json.dumps(payload, indent=1, default=str).encode("utf-8"))


def append_durable(path: Path, record: dict) -> None:
    """One JSON line appended to `path`, flushed and fsynced (fs_retry)."""
    from benchmarks.representation_RxRx1.runners.control import fs_retry

    line = (json.dumps(record, default=str) + "\n").encode("utf-8")

    def write():
        with Path(path).open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    fs_retry(write)


# ---- one resume of a run directory -------------------------------------------------------------------------------

class Resume:
    """One resume of a run directory (`open`): its number, the RUN_META.json entry it adds, the replay source and the
    divergence guard (`live`)."""

    def __init__(self, run_dir: Path, arm: str, meta: dict, entry: dict, source: Path):
        self.run_dir, self.arm, self.meta, self.entry, self.source = run_dir, arm, meta, entry, source
        self.number = entry["resume"]
        self.engine: Optional[ReplayEngine] = None   # set by ReplayEngine
        self.calls_served = 0
        self.is_live = False
        self.lock = threading.Lock()

    @classmethod
    def open(cls, run_dir: Path, arm: str) -> "Resume":
        """The resume of an unfinished run of `arm`: refuses a directory that is not a run of that arm or that
        finished; moves the arm's rotated logs into logs/resume_<n>/ (the replay's source)."""
        run_dir = Path(run_dir)
        meta_path = run_dir / "RUN_META.json"
        if not meta_path.exists():
            raise SystemExit(f"NOT_A_RUN: {run_dir} has no RUN_META.json; nothing to resume")
        if (run_dir / "AGENT_RUN.json").exists():
            raise SystemExit(f"RUN_FINISHED: {run_dir} has its AGENT_RUN.json; a finished run is never resumed")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("agent") != arm:
            raise SystemExit(f"WRONG_ARM: {run_dir} is a {meta.get('agent')} run, not a {arm} run")
        if not meta.get("replay_records"):
            raise SystemExit(f"NOT_RESUMABLE: {run_dir} was not started in the VM mode (no replay records)")
        resumes = list(meta.get("resumes") or [])
        number = len(resumes) + 1
        previous_start = resumes[-1]["started"] if resumes else _epoch(meta["started_utc"])
        before = (resumes[-1]["active_seconds_before"] if resumes else 0.0) + max(
            0.0, last_activity(run_dir) - previous_start)
        archive = run_dir / "logs" / f"resume_{number}"
        rotated = ROTATED[arm]
        from benchmarks.representation_RxRx1.runners.control import fs_retry

        if archive.exists():
            # an earlier resume stopped after moving the logs and before its RUN_META.json entry: nothing new since
            if any((run_dir / "logs" / name).exists() for name in rotated):
                raise SystemExit(f"RESUME_STALE: {archive} exists next to new logs while RUN_META.json records "
                                 f"{len(resumes)} resume(s)")
        elif rotated:
            fs_retry(archive.mkdir, parents=True)
            for name in rotated:
                if (run_dir / "logs" / name).exists():
                    fs_retry(os.rename, run_dir / "logs" / name, archive / name)
        # The replay source: the logs of the last process that went live (or of the original run). A resume that
        # stopped during its replay (live_from None) wrote only the echoes it got through: its own source stays the
        # source, and its partial logs stay archived.
        previous = resumes[-1] if resumes else None
        if rotated and previous is not None and previous.get("live_from") is None:
            source = run_dir / (previous.get("replay_source") or previous["logs_moved_to"])
        else:
            source = archive
        finished = sorted(str(p.parent.relative_to(run_dir)).replace(os.sep, "/")
                          for p in (run_dir / "cache").rglob(CALL_RECORD)) if (run_dir / "cache").exists() else []
        started = time.time()
        entry = {"resume": number, "started": started, "started_utc": utc(started),
                 "run_tag": os.environ.get("RXRX_RUN_TAG"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                 "host": socket.gethostname(), "active_seconds_before": round(before, 1),
                 "logs_moved_to": archive.relative_to(run_dir).as_posix() if rotated else None,
                 "replay_source": source.relative_to(run_dir).as_posix() if rotated else None,
                 "logged_requests": None,
                 "finished_calls": finished, "replayed_requests": None, "live_from": None, "live_utc": None,
                 "moved_aside": []}
        print(f"RESUME {number} of {run_dir}: {len(finished)} finished call(s), "
              f"{round(before / 3600, 3)} h active before", flush=True)
        return cls(run_dir, arm, meta, entry, source)

    def write_meta(self, current: dict) -> dict:
        """RUN_META.json with this resume added, after checking that the code, the prep and the arm's settings are
        the ones the run started with (`STABLE_META`); returns it. The per-resume facts of `current` (deadline,
        cores, GPUs, resources) go into the entry."""
        stored = self.meta
        changed = {k: {"run": stored.get(k), "now": current.get(k)} for k in STABLE_META
                   if (k in stored or k in current) and stored.get(k) != current.get(k)}
        if changed:
            raise SystemExit(f"RESUME_STALE: {self.run_dir} was started with other code, prep or settings: {changed}")
        self.entry.update({k: current.get(k) for k in ("deadline_utc", "cores", "gpus", "resources")
                           if k in current})
        self.meta = {**stored, "resumes": [*(stored.get("resumes") or []), self.entry]}
        write_json_durable(self.run_dir / "RUN_META.json", self.meta)
        return self.meta

    def _save(self) -> None:
        resumes = list(self.meta.get("resumes") or [])
        resumes[-1] = self.entry
        self.meta = {**self.meta, "resumes": resumes}
        write_json_durable(self.run_dir / "RUN_META.json", self.meta)

    def moved(self, path: Path, target: Path) -> None:
        self.entry["moved_aside"].append({"from": str(Path(path).relative_to(self.run_dir)).replace(os.sep, "/"),
                                          "to": str(Path(target).relative_to(self.run_dir)).replace(os.sep, "/")})
        self._save()

    def live(self, what: str) -> None:
        """The first live action of this resume (a request to the model or a call of the tool): every logged reply
        and every finished call must have been replayed by then (else REPLAY_DIVERGED); records where it went
        live."""
        with self.lock:
            if self.is_live:
                return
            engine = self.engine
            pending = len(engine.replies) - engine.position if engine is not None else 0
            unserved = len(self.entry["finished_calls"]) - self.calls_served
            if pending or unserved:
                raise ReplayDiverged(f"{self.run_dir}: a live {what} while {pending} logged reply(ies) and "
                                     f"{unserved} finished call(s) are not replayed")
            self.is_live = True
            self.entry.update(replayed_requests=engine.position if engine is not None else 0, live_from=what,
                              live_utc=utc())
            self._save()
            print(f"RESUME {self.number}: live from {what} after {self.entry['replayed_requests']} replayed "
                  f"request(s) and {self.calls_served} served call(s)", flush=True)

    def close(self) -> None:
        """The end of the resumed run, before its AGENT_RUN.json: a logged reply or a finished call it never replayed
        means the replay ended before the logged run did (REPLAY_DIVERGED); a run that never went live records the
        requests it replayed."""
        engine = self.engine
        pending = len(engine.replies) - engine.position if engine is not None else 0
        unserved = len(self.entry["finished_calls"]) - self.calls_served
        if not self.is_live and (pending or unserved):
            raise ReplayDiverged(f"{self.run_dir}: the replayed run ended with {pending} logged reply(ies) and "
                                 f"{unserved} finished call(s) not replayed")
        if not self.is_live:
            self.entry["replayed_requests"] = engine.position if engine is not None else 0
        self._save()

    def replaying(self) -> bool:
        return not self.is_live


def resumes_record(meta: Optional[dict]) -> list:
    """RUN_META.json's resumes (for AGENT_RUN.json), [] for a run never resumed."""
    return list((meta or {}).get("resumes") or [])


def active_hours(meta: Optional[dict], started: float, ended: float) -> float:
    """The run's active wall hours: this process's plus the active time before its resume (no downtime)."""
    resumes = resumes_record(meta)
    before = resumes[-1]["active_seconds_before"] if resumes else 0.0
    return round((before + ended - started) / 3600, 3)


# ---- the reasoner: the record of every reply, and its replay -------------------------------------------------------

def _serialise(value: Any) -> Any:
    """A reply as JSON, a pydantic object as {"__model__": its class name, "json": model_dump_json()}."""
    if hasattr(value, "model_dump_json"):
        return {"__model__": type(value).__name__, "json": value.model_dump_json()}
    if isinstance(value, dict):
        return {str(k): _serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _restore(value: Any, schema) -> Any:
    if isinstance(value, dict) and set(value) == {"__model__", "json"}:
        if schema is None or schema.__name__ != value["__model__"]:
            raise ReplayDiverged(f"a logged {value['__model__']} reply answers a {getattr(schema, '__name__', None)} "
                                 "request")
        return schema.model_validate_json(value["json"])
    if isinstance(value, dict):
        return {k: _restore(v, schema) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v, schema) for v in value]
    return value


def _named_error(name: str, message: str) -> Exception:
    """An exception whose `type(e).__name__` and str() are the logged ones (the arms' error text is built from
    both)."""
    return type(str(name), (RuntimeError,), {})(message)


def _prompt_text(content) -> Optional[str]:
    text = content[0] if isinstance(content, (list, tuple)) and content else content
    return text if isinstance(text, str) else None


def _prompt_digest(content) -> Optional[str]:
    text = _prompt_text(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text is not None else None


def masked_digest(content) -> Optional[str]:
    """sha256 of a prompt with its clock text masked (`TIME_TEXT`)."""
    text = _prompt_text(content)
    return hashlib.sha256(TIME_TEXT.sub(TIME_MASK, text).encode("utf-8")).hexdigest() if text is not None else None


def shape_prompt_check(content, source: dict) -> Optional[str]:
    """Why a replayed shape prompt is not the logged one (its clock text masked), else None."""
    logged = source.get("prompt_masked_sha256")
    if logged is not None and logged != masked_digest(content):
        return "the replayed prompt (its clock text masked) differs from the logged one"
    return None


def react_masked_digest(content) -> Optional[str]:
    """sha256 of a react prompt with its clock line (`REACT_TIME_TEXT`) and its python steps' observations
    (`REACT_PYTHON_TEXT`) masked."""
    text = _prompt_text(content)
    if text is None:
        return None
    text = REACT_PYTHON_TEXT.sub(REACT_PYTHON_MASK, REACT_TIME_TEXT.sub(REACT_TIME_MASK, text))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def react_prompt_check(content, source: dict) -> Optional[str]:
    """Why a replayed react prompt is not the logged one (its clock line and python observations masked), else
    None."""
    logged = source.get("prompt_masked_sha256")
    if logged is not None and logged != react_masked_digest(content):
        return "the replayed prompt (its clock line and python observations masked) differs from the logged one"
    return None


class RecordingEngine:
    """The shape and react arms' engine: the reasoner engine itself, each request and its complete reply (or the
    exception it raised) appended to logs/reasoner.jsonl as soon as it returns -- the replay source of a resume.
    `masked`: the prompt digest with the arm's clock text masked (shape: `masked_digest`, react:
    `react_masked_digest`)."""

    def __init__(self, engine, path: Path, write_errors: list, masked: Callable[[Any], Optional[str]] = None):
        self.engine, self.path, self.write_errors = engine, Path(path), write_errors
        self.masked = masked or masked_digest
        self.count = 0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def generate(self, content, response_format=None, **kwargs):
        self.count += 1
        record = {"request": self.count, "schema": getattr(response_format, "__name__", None),
                  "prompt_sha256": _prompt_digest(content), "prompt_masked_sha256": self.masked(content),
                  "returned": None, "raised": None, "usage": None}
        started = time.time()
        try:
            reply = self.engine.generate(content, response_format=response_format, **kwargs)
        except Exception as error:  # noqa: BLE001 -- logged as raised, then raised again
            self.append({**record, "raised": {"type": type(error).__name__, "message": str(error)},
                         "seconds": round(time.time() - started, 2)})
            raise
        usage = reply.get("usage") if isinstance(reply, dict) else None
        self.append({**record, "returned": _serialise(reply), "usage": usage if isinstance(usage, dict) else None,
                     "seconds": round(time.time() - started, 2)})
        return reply

    def append(self, record: dict) -> None:
        try:
            append_durable(self.path, {"time": time.time(), **record})
        except OSError as error:                     # after fs_retry: recorded, the run goes on (the arm's policy)
            self.write_errors.append({"time": time.time(), "what": self.path.name,
                                      "error": f"{type(error).__name__}: {error}"})


def shape_replies(path: Path) -> list[dict]:
    """The replay of a shape run: logs/reasoner.jsonl's records in order."""
    from benchmarks.batch_correction_JUMP.runners.score import jsonl

    return [{"schema": r.get("schema"), "returned": r.get("returned"), "raised": r.get("raised"),
             "usage": r.get("usage"), "source": r} for r in (jsonl(path) if Path(path).exists() else [])]


class ReplayEngine:
    """The engine of a resumed shape or react run: the logged replies in order (`replies`, each with its usage
    counted in SHAPE's process-level token count, as the engine counted it), then the live engine. `served` is the
    source record of the reply served last (its logs/reasoner.jsonl record), None for a live one.
    `echo(source, content)` writes a served reply's record again (shape: logs/reasoner.jsonl, with the new prompt's
    digests). `prompt_check(content, source)` names why the replayed request's prompt is not the logged one
    (REPLAY_DIVERGED), before its reply is served."""

    def __init__(self, replies: list[dict], live, resume: Resume, echo: Optional[Callable[..., None]] = None,
                 prompt_check: Optional[Callable[[Any, dict], Optional[str]]] = None):
        self.replies, self.live, self.resume, self.echo = list(replies), live, resume, echo
        self.prompt_check = prompt_check
        self.position = 0
        self.served: Optional[dict] = None
        resume.engine = self
        resume.entry["logged_requests"] = len(self.replies)

    def __getattr__(self, name):
        return getattr(self.live, name)

    @property
    def replaying(self) -> bool:
        return self.position < len(self.replies)

    def generate(self, content, response_format=None, **kwargs):
        from shape.engine import usage

        if self.position >= len(self.replies):
            self.served = None
            self.resume.live(f"request {self.position + 1}")
            return self.live.generate(content, response_format=response_format, **kwargs)
        reply = self.replies[self.position]
        self.position += 1
        self.served = reply["source"]
        expected = reply.get("schema")
        if expected and response_format is not None and expected != response_format.__name__:
            raise ReplayDiverged(f"logged request {self.position} asked for {expected}, the replay for "
                                 f"{response_format.__name__}")
        problem = self.prompt_check(content, reply["source"]) if self.prompt_check is not None else None
        if problem:
            raise ReplayDiverged(f"logged request {self.position}: {problem}")
        if self.echo is not None:
            self.echo(reply["source"], content)
        if reply.get("raised"):
            raise _named_error(reply["raised"]["type"], reply["raised"]["message"])
        if reply.get("usage"):
            usage.record(reply["usage"])
        return _restore(copy.deepcopy(reply["returned"]), response_format)


# ---- the analyzer: a finished call served from its own directory --------------------------------------------------

def _key(params: dict) -> str:
    """What identifies a call for its record: its tool params without the time cap (`max_minutes`, which depends on
    when the call starts)."""
    return json.dumps({k: v for k, v in params.items() if k != "max_minutes"}, sort_keys=True, default=str)


def _files(result: Any) -> dict:
    """sha256 of the output files a result names (adata_path, resume_best, resume_end)."""
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    out = {}
    if isinstance(result, dict):
        for name in ("adata_path", "resume_best", "resume_end"):
            path = result.get(name)
            if isinstance(path, str) and path and Path(path).is_file():
                out[name] = {"path": path, "sha256": sha256_file(Path(path))}
    return out


class CallCache:
    """Every analyzer call of a shape or react run, through `run(params, invoke)`. Each call that returns (a result,
    or an exception: a failed call is an observation too) writes CALL_RESULT.pkl and then CALL_RECORD.json (its
    ordinal in the run, its params' key, the sha256 of its result and output files, its wall seconds; later the
    caller's own `outer_seconds`, `note_outer`) into its own directory (<query_cache_dir>/cell_state_analysis/
    <output_subdir>), both fsynced. While a call runs, <query_cache_dir>/HEARTBEAT is touched every HEARTBEAT_SECONDS.
    On a resume (`resume`) a call with a record is served from it after the checks -- the same position in the run,
    the same configuration, unchanged files -- and a call without one has its leftover directory moved aside before
    it runs live. A call without output_subdir cannot be recorded: SystemExit."""

    def __init__(self, query_cache_dir: Path, resume: Optional[Resume] = None, write_errors: Optional[list] = None):
        self.root = Path(query_cache_dir) / "cell_state_analysis"
        self.resume = resume
        self.write_errors = write_errors if write_errors is not None else []
        self.ordinal = 0
        self.served: dict[str, dict] = {}            # output_subdir -> the record of a served call
        self.last_served: Optional[dict] = None
        self.lock = threading.Lock()

    def run(self, params: dict, invoke: Callable[[], Any]):
        with self.lock:
            ordinal = self.ordinal
            self.ordinal += 1
        sub = params.get("output_subdir")
        if not sub:
            raise SystemExit(f"CALL_WITHOUT_DIRECTORY: an analyzer call without output_subdir cannot be recorded "
                             f"(params {sorted(params)})")
        directory = self.root / sub
        self.last_served = None
        if self.resume is not None:
            record_path = directory / CALL_RECORD
            if record_path.exists():
                return self._serve(directory, json.loads(record_path.read_text(encoding="utf-8")), ordinal, params)
            self.resume.live(f"call {sub}")
            if directory.exists():
                self.resume.moved(directory, move_aside(directory, self.resume.number))
        started = time.time()
        stop = self._heartbeat()
        try:
            result = invoke()
        except Exception as error:  # noqa: BLE001 -- recorded, then raised again (the caller's failed call)
            stop.set()
            self._record(directory, ordinal, params, None, {"type": type(error).__name__, "message": str(error)},
                         time.time() - started)
            raise
        finally:
            stop.set()
        self._record(directory, ordinal, params, result, None, time.time() - started)
        return result

    def _heartbeat(self) -> threading.Event:
        """Touches <query_cache_dir>/HEARTBEAT now and every HEARTBEAT_SECONDS until the returned event is set (the
        last activity a resume measures covers a call that was running when the run stopped)."""
        stop = threading.Event()
        beat = self.root.parent / HEARTBEAT

        def pulse():
            while True:
                try:
                    beat.parent.mkdir(parents=True, exist_ok=True)
                    beat.touch()
                except OSError:
                    pass
                if stop.wait(HEARTBEAT_SECONDS):
                    return

        threading.Thread(target=pulse, daemon=True, name="call-heartbeat").start()
        return stop

    def note_outer(self, params: dict, seconds) -> None:
        """The caller's own wall seconds of a live call (the ledgers' `seconds`), added to its record, so that the
        call served on a resume reports the same value (`seconds_of`)."""
        sub = params.get("output_subdir")
        path = self.root / str(sub) / CALL_RECORD
        if not sub or sub in self.served or seconds is None or not path.exists():
            return
        try:
            write_json_durable(path, {**json.loads(path.read_text(encoding="utf-8")), "outer_seconds": seconds})
        except Exception as error:  # noqa: BLE001 -- recorded; a served call then reports its inner seconds
            self.write_errors.append({"time": time.time(), "what": f"{sub}/{CALL_RECORD}",
                                      "error": f"{type(error).__name__}: {error}"})

    @staticmethod
    def seconds_of(record: dict):
        """A served call's seconds: the caller's own (`outer_seconds`), else the call's (a run stopped before they
        were added)."""
        return record.get("outer_seconds", record.get("wall_seconds"))

    def _record(self, directory: Path, ordinal: int, params: dict, result, raised, seconds: float) -> None:
        """Writes the call's record; a write that still fails after fs_retry is recorded (`write_errors`) and the run
        goes on (that call would run again on a resume)."""
        from benchmarks.representation_RxRx1.runners.control import fs_retry
        from benchmarks.representation_RxRx1.runners.prepare import sha256_file

        try:
            fs_retry(directory.mkdir, parents=True, exist_ok=True)
            blob = directory / CALL_RESULT
            write_durable(blob, pickle.dumps(result, protocol=4))
            write_json_durable(directory / CALL_RECORD, {
                "ordinal": ordinal, "key": _key(params), "raised": raised, "result_sha256": sha256_file(blob),
                "files": _files(result), "wall_seconds": round(seconds, 1), "time_utc": utc()})
        except Exception as error:  # noqa: BLE001 -- a write or pickling failure never turns the call into a failure
            self.write_errors.append({"time": time.time(), "what": f"{directory.name}/{CALL_RECORD}",
                                      "error": f"{type(error).__name__}: {error}"})
            print(f"WRITE_FAILED {directory / CALL_RECORD}: {type(error).__name__}: {error}", flush=True)

    def _serve(self, directory: Path, record: dict, ordinal: int, params: dict):
        from benchmarks.representation_RxRx1.runners.prepare import sha256_file

        where = directory.relative_to(self.root).as_posix()
        if record.get("ordinal") != ordinal:
            raise ReplayDiverged(f"the replay's call {ordinal} is {where}, logged as call {record.get('ordinal')}")
        if record.get("key") != _key(params):
            raise ReplayDiverged(f"call {where}: the replay's params differ from the logged ones")
        blob = directory / CALL_RESULT
        if not blob.is_file() or sha256_file(blob) != record.get("result_sha256"):
            raise SystemExit(f"INTEGRITY_FAILED: {blob} is missing or changed since the call finished")
        for name, entry in (record.get("files") or {}).items():
            if not Path(entry["path"]).is_file() or sha256_file(Path(entry["path"])) != entry["sha256"]:
                raise SystemExit(f"INTEGRITY_FAILED: the {name} of {where} ({entry['path']}) changed since the call "
                                 "finished")
        with self.resume.lock:
            self.resume.calls_served += 1
        self.served[params.get("output_subdir")] = record
        self.last_served = record
        print(f"RESUME {self.resume.number}: call {ordinal} ({where}) served from its record", flush=True)
        if record.get("raised"):
            raise _named_error(record["raised"]["type"], record["raised"]["message"])
        with blob.open("rb") as handle:
            return pickle.load(handle)                # the benchmark's own file, checked against its sha256 above


# ---- the VM driver's helpers ---------------------------------------------------------------------------------------

def run_status(stage: str, run: Path) -> str:
    """A stage's run directory as the driver sees it: "finished" (shape: AGENT_RUN.json written), "delivered" /
    "failed" (llm, react: AGENT_RUN.json written, the decision delivered or not), "incomplete" (RUN_META.json, no
    AGENT_RUN.json: resume it), "empty" (no RUN_META.json: the arm stopped before it started)."""
    run = Path(run)
    if not (run / "RUN_META.json").exists():
        return "empty"
    if not (run / "AGENT_RUN.json").exists():
        return "incomplete"
    if stage == "shape":
        return "finished"
    decision = run / "submission" / "decision.json"
    status = json.loads(decision.read_text(encoding="utf-8")).get("status") if decision.exists() else None
    return "delivered" if status == "delivered" else "failed"


def find_run(state: Path, bench: Path, stage: str, cell_type: str, commit: str, seed: int = 0) -> Optional[Path]:
    """The run directory of `stage` made at `commit` by one of the driver's invocations (the run tags in the state
    file): <bench>/<stage>_<cell type>_<commit8>_<UTC>_<run tag>; None when there is none; SystemExit when there are
    several (a duplicate) or its RUN_META.json names another commit. A shape run whose RUN_META.json records another
    `seed` (none recorded = 0; A18, RXRX_SEED) is another run: not found here."""
    import re

    tags = json.loads(Path(state).read_text(encoding="utf-8")).get("run_tags", []) if Path(state).exists() else []
    prefix = f"{stage}_{cell_type}_{commit[:8]}_"
    names = [re.escape(prefix) + r"\d{8}T\d{6}Z_" + re.escape(tag) for tag in tags]
    found = sorted(p for p in Path(bench).glob(prefix + "*")
                   if p.is_dir() and any(re.fullmatch(name, p.name) for name in names)
                   and (stage != "shape" or not (p / "RUN_META.json").exists()
                        or json.loads((p / "RUN_META.json").read_text(encoding="utf-8")).get("seed", 0) == seed))
    if len(found) > 1:
        raise SystemExit(f"DUPLICATE_RUN: {len(found)} {stage} runs of {commit[:8]}: {[str(p) for p in found]}")
    if not found:
        return None
    meta = found[0] / "RUN_META.json"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")).get("commit") != commit:
        raise SystemExit(f"WRONG_COMMIT: {found[0]} was made at another commit")
    return found[0]


def add_tag(state: Path, tag: str, commit: str, cell_type: str) -> None:
    """Records one driver invocation's run tag in the state file (created at the first one)."""
    from benchmarks.representation_RxRx1.runners.control import write_json

    state = Path(state)
    record = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {
        "commit": commit, "cell_type": cell_type, "run_tags": [], "invocations": []}
    if record.get("commit") != commit or record.get("cell_type") != cell_type:
        raise SystemExit(f"STATE_MISMATCH: {state} belongs to {record.get('commit')} / {record.get('cell_type')}")
    if tag in record["run_tags"]:
        raise SystemExit(f"STATE_MISMATCH: run tag {tag} is already recorded in {state}")
    record["run_tags"].append(tag)
    record["invocations"].append({"run_tag": tag, "started_utc": utc(), "host": socket.gethostname()})
    state.parent.mkdir(parents=True, exist_ok=True)
    write_json(state, record)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="resume")
    sub = parser.add_subparsers(dest="cmd", required=True)
    find = sub.add_parser("find-run")
    status = sub.add_parser("status")
    tag = sub.add_parser("add-tag")
    for sp in (find, tag):
        sp.add_argument("--state", type=Path, required=True)
        sp.add_argument("--cell-type", required=True)
        sp.add_argument("--commit", required=True)
    find.add_argument("--bench", type=Path, required=True)
    find.add_argument("--stage", choices=STAGES, required=True)
    find.add_argument("--seed", type=int, default=0, help="shape: the run seed (A18, RXRX_SEED)")
    status.add_argument("--stage", choices=STAGES, required=True)
    status.add_argument("run", type=Path)
    tag.add_argument("--tag", required=True)
    a = parser.parse_args(argv)
    if a.cmd == "find-run":
        run = find_run(a.state, a.bench, a.stage, a.cell_type, a.commit, a.seed)
        print(run.as_posix() if run else "")
    elif a.cmd == "status":
        print(run_status(a.stage, a.run))
    else:
        add_tag(a.state, a.tag, a.commit, a.cell_type)


if __name__ == "__main__":
    main()
