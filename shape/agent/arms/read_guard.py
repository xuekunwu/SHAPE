"""Claude Code PreToolUse hook: read only the explicitly published evidence roots."""
import json
from pathlib import Path
import sys


def allowed(event, roots):
    if event.get("tool_name") == "StructuredOutput":
        return True
    if event.get("tool_name") != "Read":
        return False
    path = Path(event.get("tool_input", {}).get("file_path", "")).resolve()
    return path.is_file() and any(path.is_relative_to(Path(root).resolve()) for root in roots)


if __name__ == "__main__":
    try:
        roots = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        ok = allowed(json.load(sys.stdin), roots)
    except Exception:
        ok = False
    if not ok:
        print("Read denied: only published benchmark evidence is available.", file=sys.stderr)
        sys.exit(2)
