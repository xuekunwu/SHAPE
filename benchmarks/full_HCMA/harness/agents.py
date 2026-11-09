"""Headless Claude Code runner with a hard wall-clock deadline and a multi-round loop.

Ported from the SSL agentic harness (agentic_rxrx1/harness/agents.py), Claude only:
- the deadline is enforced by a watchdog that kills the agent's whole process group;
- Claude runs with an explicit tool allow-list instead of --dangerously-skip-permissions;
  web tools are disallowed;
- credentials are never passed as arguments or written by this module: the CLI reads them
  through the apiKeyHelper settings file the launcher prepared.

Every event line the CLI prints is appended verbatim to traces/round_XX.jsonl; harness
events (round start/end, exit code, stderr tail, rejected FINAL) go to logs/rounds.jsonl.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

CLAUDE_ALLOWED = "Bash Read Write Edit Glob Grep TodoWrite"
CLAUDE_DISALLOWED = "WebFetch WebSearch"
# Stop leaking harness-private variables into the agent's shell.
_SCRUB = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "STANFORD_AI_API_KEY", "ZHIPUAI_API_KEY",
          "SHAPE_SAM_SERVER_URL", "PYTHONPATH")
# Every process the agent starts inherits this marker, so leftovers can be found even when
# they changed directory or started a new session.
MARKER_ENV = "FULL_HCMA_RUN"


_WRITE_LOCK = threading.Lock()


def _write_json(path: Path, record: dict) -> None:
    """Atomic: live viewers read AGENT_RUN.json while it is refreshed. The agent loop and the live-trace
    thread both write it: each writer uses its own temporary file, and the replace is serialized."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(record, indent=1), encoding="utf-8")
    with _WRITE_LOCK:
        for attempt in range(5):     # Windows refuses to replace a file another thread is reading
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05)


def _env(run: Path) -> dict:
    """The agent's environment: no harness secrets, no PYTHONPATH (the repo; the tools add it from
    _config.json themselves), a run-private Claude config dir (no user CLAUDE.md, hooks or settings)."""
    env = dict(os.environ)
    for key in _SCRUB:
        env.pop(key, None)
    env[MARKER_ENV] = str(run.resolve())
    env["CLAUDE_CONFIG_DIR"] = str((run / "claude_config").resolve())
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    if not hasattr(os, "killpg"):            # not POSIX (tests): no process groups
        proc.kill()
        proc.wait()
        return
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_pids(run: Path) -> list[int]:
    """Processes of this run: same uid AND (marker in environ OR cwd inside the workspace).
    Only the workspace, not the whole run dir: a user shell that cd'ed into the run directory
    to watch logs must survive."""
    found = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return found
    marker = f"{MARKER_ENV}={run.resolve()}".encode()
    root = str((run / "workspace").resolve())
    me = os.getpid()
    for entry in proc_root.glob("[0-9]*"):
        try:
            pid = int(entry.name)
            if pid == me or entry.stat().st_uid != os.getuid():
                continue
            cwd = os.readlink(entry / "cwd")
            environ = (entry / "environ").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        if marker in environ or cwd == root or cwd.startswith(root + os.sep):
            found.append(pid)
    return found


def kill_run_processes(run: Path, grace: float = 30.0) -> list[int]:
    """SIGTERM every leftover process of this run (a tool records its call as failed on SIGTERM),
    then SIGKILL whatever is still alive after `grace` seconds."""
    pids = _run_pids(run)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + grace
    while time.time() < end and set(_run_pids(run)) & set(pids):
        time.sleep(1)
    for pid in set(_run_pids(run)) & set(pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return pids


def stream(cmd: list[str], cwd: Path, trace_path: Path, deadline: float, run: Path | None = None) -> dict:
    """Run cmd, append stdout lines to trace_path, kill the group at `deadline` (epoch s)."""
    started = time.time()
    state = {"session_id": None}
    stderr_tail: list[str] = []
    timed_out = False
    # Sidecar: arrival time of every trace line, so trace.md can place steps in time.
    with open(trace_path, "a", encoding="utf-8") as tf, \
            open(trace_path.with_suffix(".times"), "a", encoding="utf-8") as times:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=_env(run or cwd), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", start_new_session=True)

        def drain_stderr():
            for line in proc.stderr:
                stderr_tail.append(line)
                del stderr_tail[:-50]

        def read_stdout():
            for line in proc.stdout:
                tf.write(line)
                tf.flush()
                times.write(f"{time.time():.3f}\n")
                times.flush()
                try:
                    event = json.loads(line)
                    state["session_id"] = event.get("session_id") or state["session_id"]
                except (json.JSONDecodeError, AttributeError):
                    pass

        threads = [threading.Thread(target=f, daemon=True) for f in (drain_stderr, read_stdout)]
        for thread in threads:
            thread.start()
        try:
            while proc.poll() is None:
                if time.time() >= deadline:
                    timed_out = True
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None:
                _kill_group(proc)
            # A descendant that inherited the pipe must not block the harness forever.
            for thread in threads:
                thread.join(timeout=60)
        rc = proc.wait()
    return {"returncode": rc, "timed_out": timed_out, "session_id": state["session_id"],
            "seconds": round(time.time() - started, 1), "stderr_tail": "".join(stderr_tail)[-4000:]}


def claude_cmd(prompt: str, model: str | None, session_id: str | None, add_dirs: list[str],
               settings: str | None = None) -> list[str]:
    cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
           "--allowedTools", CLAUDE_ALLOWED, "--disallowedTools", CLAUDE_DISALLOWED]
    if settings:  # {"apiKeyHelper": ...}: the key stays out of the agent shell's env
        cmd += ["--settings", settings]
    for directory in add_dirs:
        cmd += ["--add-dir", directory]
    if session_id:
        cmd += ["--resume", session_id]
    if model:
        cmd += ["--model", model]
    return cmd


def continuation_prompt(deadline: float, round_index: int, rejected: dict | None = None) -> str:
    left = max(0, deadline - time.time()) / 3600
    text = (f"Round {round_index}: the harness resumed you. {left:.2f} h of the wall-clock budget remain "
            "(python tools/time_left.py). Re-read TASK.md and NOTES.md, then continue the task. Keep "
            "submission/choice.json valid at all times. When you are done and the choice is final, create "
            "the empty file submission/FINAL and stop.")
    if rejected:
        text += " Your submission/FINAL was not accepted, so the harness removed it."
        if rejected.get("missing"):
            text += (f" Lever coverage (TASK.md): {', '.join(rejected['missing'])} not changed by any "
                     "completed call and no reason under 'skipped' in submission/choice.json; "
                     "python tools/lever_coverage.py shows the state.")
        if rejected.get("chain_problems"):
            text += (" submission/choice.json does not verify: "
                     + "; ".join(str(p) for p in rejected["chain_problems"]) + ".")
    return text


def open_tool_calls(run: Path) -> list[str]:
    """Calls with a ledger 'start' but no 'end' yet."""
    path = run / "logs" / "tool_calls.jsonl"
    started, ended = set(), set()
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("event") == "start":
                started.add(rec.get("call"))
            elif rec.get("event") == "end":
                ended.add(rec.get("call"))
    return sorted(started - ended)


class _Terminated(Exception):
    pass


def _render_live(run: Path) -> None:
    from .trace_md import write_all

    try:
        write_all(run)
    except Exception as error:  # noqa: BLE001 -- a render bug must never stop the agent
        _log(run / "logs", {"event": "live_trace_error", "error": f"{type(error).__name__}: {error}"})


class _LiveTrace(threading.Thread):
    """Re-render trace.md / trace.json every `interval` seconds while the agent runs."""

    def __init__(self, run: Path, interval: float, before=None):
        super().__init__(daemon=True)
        self.run_dir, self.interval, self._stop_event = run, interval, threading.Event()
        self.before = before

    def run(self) -> None:
        while not self._stop_event.wait(self.interval):
            if self.before is not None:
                try:
                    self.before()         # refresh AGENT_RUN.json (a round can last hours)
                except Exception:        # noqa: BLE001
                    pass
            _render_live(self.run_dir)

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=120)


def run_loop(run: Path, *, budget_hours: float, model: str | None, add_dirs: list[str],
             claude_settings: str | None = None, max_rounds: int = 1000, min_round_gap: float = 300.0,
             max_failure_seconds: float = 2700.0, live_interval: float = 60.0) -> dict:
    """Rounds until the deadline. A round that ends quickly (e.g. the agent started a tool call
    in the background and ended its turn) is followed by a wait so the next round starts
    >= min_round_gap after the previous one. Failed rounds back off exponentially (60 s .. 600 s);
    the loop gives up only after `max_failure_seconds` of consecutive failure.

    trace.md / trace.json are re-rendered every `live_interval` seconds while the agent runs, so a
    run that is cut off at any moment keeps an up-to-date trace; the raw traces/round_XX.jsonl are
    written line by line anyway. SIGTERM (Slurm's warning before a kill) ends the loop cleanly and
    writes the final trace.

    submission/FINAL ends the run only when harness/coverage.py's final_check accepts it (lever
    coverage and a choice whose call chain verifies); otherwise FINAL is removed and the agent is
    resumed with the missing groups and the chain problems."""
    from .coverage import final_check

    run = Path(run)
    workspace = run / "workspace"
    traces = run / "traces"
    logs = run / "logs"
    traces.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    deadline = start + budget_hours * 3600
    for path in (workspace / ".deadline", logs / ".deadline"):    # the tools read logs/.deadline
        path.write_text(f"{deadline:.0f}\n", encoding="utf-8")
    prompt = (workspace / "TASK.md").read_text(encoding="utf-8")
    session_id, failures, failing_since, rounds, stopped = None, 0, None, [], "max_rounds"
    rejected = None

    def agent_run(status: str) -> dict:
        record = {"agent": "claude", "model": model, "budget_hours": budget_hours, "started": start,
                  "deadline": deadline, "updated": time.time(),
                  "wall_hours_used": round((time.time() - start) / 3600, 3), "n_rounds": len(rounds),
                  "final_flag": (workspace / "submission" / "FINAL").exists(), "stopped_because": status}
        _write_json(run / "AGENT_RUN.json", record)
        return record

    agent_run("running")
    live = _LiveTrace(run, live_interval, before=lambda: agent_run("running"))
    live.start()
    previous_term = signal.getsignal(signal.SIGTERM)

    def on_term(signum, frame):
        raise _Terminated()

    signal.signal(signal.SIGTERM, on_term)

    def wait_until(t: float) -> None:
        while time.time() < min(t, deadline - 600):
            time.sleep(min(10.0, max(0.0, min(t, deadline - 600) - time.time())))

    try:
        for index in range(max_rounds):
            final = workspace / "submission" / "FINAL"
            if final.exists():
                try:
                    check = final_check(run)
                except Exception as error:  # noqa: BLE001 -- an odd choice.json must not end the run
                    check = {"accepted": False, "missing": [],
                             "chain_problems": [f"the FINAL check failed: {type(error).__name__}: {error}"]}
                if check["accepted"]:
                    stopped = "final_flag"
                    break
                final.unlink()
                rejected = check
                _log(logs, {"event": "final_rejected", "round": index, "missing": check.get("missing"),
                            "chain_problems": check.get("chain_problems")})
            if deadline - time.time() < 600:
                stopped = "deadline"
                break
            if index:
                prompt = continuation_prompt(deadline, index, rejected)
                rejected = None
            cmd = claude_cmd(prompt, model, session_id, add_dirs, claude_settings)
            round_start = time.time()
            _log(logs, {"event": "round_start", "round": index, "agent": "claude", "model": model,
                        "prompt": prompt, "argv_flags": [a for a in cmd if a.startswith("--")]})
            result = stream(cmd, workspace, traces / f"round_{index:02d}.jsonl", deadline, run)
            killed = kill_run_processes(run) if result["timed_out"] else []
            record = {"event": "round_end", "round": index, **result, "killed_leftovers": killed}
            _log(logs, record)
            rounds.append(record)
            agent_run("running")
            if result["timed_out"]:
                stopped = "deadline"
                break
            if result["returncode"] != 0 and result["seconds"] < 120:
                failures += 1
                failing_since = failing_since or round_start
                if time.time() - failing_since > max_failure_seconds:
                    stopped = "repeated_failures"
                    break
                if failures >= 2:
                    session_id = None   # a broken session may be what fails; restart from NOTES.md
                backoff = min(60.0 * 2 ** (failures - 1), 600.0)
                _log(logs, {"event": "backoff", "round": index, "failures": failures, "seconds": backoff})
                wait_until(time.time() + backoff)
                continue
            failures, failing_since = 0, None
            session_id = result["session_id"] or session_id
            wait_until(round_start + min_round_gap)
    except _Terminated:
        stopped = "sigterm (allocation ending?)"
        _log(logs, {"event": "sigterm"})
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    # A tool call started in the background (nohup) may still be running: let it finish, up to
    # the deadline itself (the tools refuse to START a call within 10 min of the deadline).
    while (open_tool_calls(run) and time.time() < deadline
           and not stopped.startswith("sigterm")):
        time.sleep(15)
    live.stop()
    leftovers = kill_run_processes(run)
    summary = {**agent_run(stopped), "ended": time.time(), "killed_leftovers_at_end": leftovers,
               "open_tool_calls_at_end": open_tool_calls(run)}
    _write_json(run / "AGENT_RUN.json", summary)
    _log(logs, {"event": "agent_done", **summary})
    _render_live(run)
    return summary


def _log(logs: Path, record: dict) -> None:
    record = {"time": time.time(), **record}
    with open(logs / "rounds.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
