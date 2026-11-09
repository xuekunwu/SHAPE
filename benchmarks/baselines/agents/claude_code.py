"""Claude Code headless runner. Invokes the `claude` CLI (NOT interactively) in a sandboxed
workdir with `-p/--print --output-format stream-json`, so every event -- assistant thinking,
text, tool_use, and tool_result -- streams out as JSONL and is saved verbatim as the trace.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .base import AgentRunner, RunResult

# Tools the baseline agent is allowed: shell (install + run cellpose), file I/O, web discovery.
_ALLOWED = "Bash Read Write Edit Glob Grep WebSearch WebFetch"


class ClaudeCodeRunner(AgentRunner):
    name = "claude-code"

    def _invoke(self, args: list[str], prompt: str, workdir: Path,
                trace_path: Path) -> RunResult:
        env = self._env()
        env["ANTHROPIC_API_KEY"] = self.api_key
        cmd = ["claude", "-p", prompt, "--output-format", "stream-json", "--verbose",
               "--dangerously-skip-permissions", "--allowedTools", _ALLOWED, *args]
        if self.model:
            cmd += ["--model", self.model]
        session_id, rc, err = None, None, ""
        try:
            with open(trace_path, "a", encoding="utf-8") as tf:
                proc = subprocess.Popen(cmd, cwd=str(workdir), env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout:              # stream-json: one event per line
                    tf.write(line)
                    tf.flush()
                    try:
                        ev = json.loads(line)
                        session_id = ev.get("session_id", session_id)
                    except Exception:
                        pass
                rc = proc.wait(timeout=self.timeout)
                err = (proc.stderr.read() or "")[-2000:]
        except subprocess.TimeoutExpired:
            proc.kill()
            err = "timeout"
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        return RunResult(ok=(rc == 0), agent=self.name, workdir=str(workdir),
                         trace_path=str(trace_path), session_id=session_id, returncode=rc,
                         error=err)

    def run(self, prompt: str, workdir: Path, trace_path: Path) -> RunResult:
        return self._invoke([], prompt, workdir, trace_path)

    def resume(self, prompt: str, workdir: Path, trace_path: Path,
               session_id: str | None) -> RunResult:
        args = ["--resume", session_id] if session_id else ["--continue"]
        return self._invoke(args, prompt, workdir, trace_path)
