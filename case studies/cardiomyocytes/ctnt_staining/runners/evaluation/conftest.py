"""Import this evaluation's own modules by name from this directory: the path above it contains a
space, so pytest cannot resolve it as a package and a bare `import process` would otherwise pick up a
same-named module from another case study."""

from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE)]
