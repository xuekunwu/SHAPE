#!/usr/bin/env python3
"""Cardiac organoid comparison replay and pinned native inference."""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results/segmentation_20261020"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "verify"):
        cmd = sub.add_parser(name, help=f"{name.capitalize()} the published aggregate comparison")
        source = cmd.add_mutually_exclusive_group(required=True)
        source.add_argument("--input-dir", type=Path, help="Local directory containing eight CSVs and manifest.json")
        source.add_argument("--download", action="store_true", help="Use authenticated HF client at the pinned revision")
        if name == "compare":
            cmd.add_argument("--output-dir", type=Path, required=True)
        else:
            cmd.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    native = sub.add_parser("native", help="Run the original inference code in a clean, pinned external checkout")
    from native import SOURCE_COMMITS
    native.add_argument("--source-root", required=True, type=Path)
    native.add_argument("--source-commit", required=True, choices=SOURCE_COMMITS)
    native.add_argument("--python", default=sys.executable, help="Interpreter with the selected backend dependencies")
    native.add_argument("--isolate-stardist", action="store_true", help="Isolate unchanged StarDist in a spawned worker; continue after native aborts")
    native.add_argument("arguments", nargs=argparse.REMAINDER, help="Pass original prepare/run/compare arguments after --")
    args = parser.parse_args()
    if args.command == "native":
        from native import launch
        return launch(args)
    import evaluation
    folder = evaluation.download() if args.download else args.input_dir
    if args.command == "compare":
        count = len(evaluation.write(folder, args.output_dir))
        print(f"Generated {count} files in {args.output_dir}")
    else:
        count = evaluation.verify(folder, args.result_dir)
        print(f"Verified {count} files: byte-identical replay from pinned HF inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
