"""Print the time left before the run's deadline (the harness stops the run at the deadline).

    python tools/time_left.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    cfg = C.load_config()
    path = C.deadline_path(cfg)
    if not path.exists():
        print(f"the deadline file {path} is missing: tool calls are refused" if cfg.get("has_deadline")
              else "no deadline set for this run")
        return 0
    deadline = float(path.read_text(encoding="utf-8").strip())
    left = deadline - time.time()
    print(f"{max(left, 0) / 3600:.2f} h left (deadline {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))}); "
          f"a tool call starts only with at least {float(cfg.get('min_start_seconds', 600)) / 60:.0f} min left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
