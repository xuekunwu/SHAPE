"""Codex CLI headless runner. Invokes `codex exec --json` (non-interactive) in a sandboxed workdir,
saving the JSONL event stream (tool calls + reasoning) verbatim as the trace. The outer feedback
loop resumes by re-running in the SAME workdir -- Codex reads its prior code/masks from disk.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from .base import AgentRunner, RunResult


def _codex_bin() -> str:
    return shutil.which("codex") or str(Path.home() / ".local" / "bin" / "codex")


class CodexRunner(AgentRunner):
    name = "codex"

    def _login(self, env: dict) -> None:
        """Codex needs an explicit API-key login (writes $CODEX_HOME/auth.json); env var alone
        yields 401. Idempotent -- safe to run before each invocation."""
        try:
            subprocess.run([_codex_bin(), "login", "--with-api-key"], input=self.api_key,
                           env=env, text=True, capture_output=True, timeout=60)
        except Exception:
            pass

    def _invoke(self, prompt: str, workdir: Path, trace_path: Path) -> RunResult:
        env = self._env()
        env["OPENAI_API_KEY"] = self.api_key
        self._login(env)
        cmd = [_codex_bin(), "exec", prompt, "--json", "--cd", str(workdir),
               "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"]
        if self.model:
            cmd += ["--model", self.model]
        session_id, rc, err = None, None, ""
        try:
            with open(trace_path, "a", encoding="utf-8") as tf:
                proc = subprocess.Popen(cmd, cwd=str(workdir), env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in proc.stdout:
                    tf.write(line)
                    tf.flush()
                    try:
                        ev = json.loads(line)
                        # codex event ids carry the thread/session; grab whatever identifies it
                        session_id = (ev.get("thread_id") or ev.get("session_id")
                                      or ev.get("conversation_id") or session_id)
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
        return self._invoke(prompt, workdir, trace_path)

    def resume(self, prompt: str, workdir: Path, trace_path: Path,
               session_id: str | None) -> RunResult:
        # Filesystem continuity: re-run in the same workdir; Codex sees its prior deliverables.
        return self._invoke(prompt, workdir, trace_path)
