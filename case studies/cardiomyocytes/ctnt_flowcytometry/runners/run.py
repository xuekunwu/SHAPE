"""Public CLI dispatch for data, scientific workflows, runtime inspection and evaluation."""
from pathlib import Path
import sys

import importlib
PACKAGE = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners'

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def main():
    command = sys.argv[1] if len(sys.argv)>1 else None
    if command in ("-h", "--help"):
        print("ctnt_flowcytometry runners: metadata commands inspect, fetch-metadata, audit-metadata; experiment runner below.")
    if len(sys.argv) > 1 and sys.argv[1] in ("inspect", "fetch-metadata", "audit-metadata"):
        metadata = importlib.import_module("case studies._shared.metadata")
        metadata.main(Path(__file__).resolve().parents[1] / "harness" / "dataset.json")
    elif command=='suite':
        importlib.import_module(PACKAGE+'.workflows.suite_runner').main(sys.argv[2:])
    elif command=='materialize':
        import argparse
        parser=argparse.ArgumentParser(description='Expand pinned images from a read-only HF mount into a fresh raw tree.')
        parser.add_argument('mount',type=Path);parser.add_argument('output',type=Path)
        args=parser.parse_args(sys.argv[2:])
        importlib.import_module(PACKAGE+'.dataset.import_archive').main(args.mount.resolve(),args.output.resolve())
    elif command in ('hybrid','crop-from-masks'):
        target={'hybrid':'.workflows.hybrid_regression','crop-from-masks':'.workflows.crop_from_fixed_masks'}[command]
        sys.argv=[sys.argv[0],*sys.argv[2:]]
        importlib.import_module(PACKAGE+target).main()
    else:
        importlib.import_module(PACKAGE+'.workflows.experiment_runner').main()


if __name__ == "__main__":
    main()
