"""Repository-wide pytest setup: the cell-state analyzer's DataLoaders load in the test process unless a test opts
in (`SHAPE_DATALOADER_WORKERS`, `shape.tools.cell_state_analyzer.tool.LOADER_WORKERS_ENV`): spawned loader workers
would not see a test's monkeypatched readers, and one pool per call per xdist worker oversubscribes the host."""
import os

os.environ.setdefault("SHAPE_DATALOADER_WORKERS", "0")
