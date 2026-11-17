"""Study entry point; fluorescence label derivation is a separate preparation step."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _shared.metadata import main as metadata_main


def main():
    metadata_main(Path(__file__).with_name("dataset.json"))


if __name__ == "__main__":
    main()
