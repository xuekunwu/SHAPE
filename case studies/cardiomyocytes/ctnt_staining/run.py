"""Run from any working directory; quote the path containing 'case studies'."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import importlib


def main():
    if sys.argv[1:2] in (['-h'],['--help']):
        print('ctnt_staining runners: metadata commands inspect, fetch-metadata, audit-metadata; E2E commands below.')
    if len(sys.argv)>1 and sys.argv[1] in ('inspect','fetch-metadata','audit-metadata'):
        entry=importlib.import_module('case studies.cardiomyocytes.ctnt_staining.cli')
    else:
        entry=importlib.import_module('case studies.cardiomyocytes.ctnt_staining.e2e')
    entry.main()


if __name__ == "__main__":
    main()
