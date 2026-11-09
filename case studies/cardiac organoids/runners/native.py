"""Delegate inference to the complete, clean historical checkout, including SHAPE."""
from pathlib import Path
import subprocess

SOURCE_COMMITS = ("00fad58934b50008bfe4186a2e5e4cfd616662cd",
                  "8f59e6b53b84229236bd091b9bb252ba83b266d9")
ENTRY = Path("case studies/cardiac organoids/runners/segmentation/run_organoid_segmentation.py")


def command(source_root, source_commit, python, arguments):
    root = Path(source_root).resolve()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args],
                                       text=True, timeout=30).strip()

    if source_commit not in SOURCE_COMMITS:
        raise ValueError("Choose one of the two recorded source commits")
    if git("rev-parse", "HEAD") != source_commit:
        raise ValueError("Historical source checkout has the wrong commit")
    if git("status", "--porcelain"):
        raise ValueError("Historical source checkout is dirty")
    script = root / ENTRY
    if not script.is_file():
        raise ValueError("Historical native entrypoint is absent")
    if not arguments:
        raise ValueError("Native runner arguments are required after --")
    return [python, str(script), *arguments]


def launch(args):
    arguments = args.arguments
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    cmd = command(args.source_root, args.source_commit, args.python, arguments)
    if args.isolate_stardist:
        cmd = [args.python, str(Path(__file__).with_name("stardist_isolation.py")),
               "--source-root", str(Path(args.source_root).resolve()), "--", *arguments]
    # A real separate interpreter releases backend state between arms.
    return subprocess.call(cmd, cwd=Path(args.source_root).resolve())
