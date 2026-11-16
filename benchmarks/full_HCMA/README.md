# full_HCMA -- shared end-to-end harness

The HCMA benchmark itself (its dataset, run scripts and tests) is not part of this release. This
folder keeps the harness code other benchmarks import:

- `harness/` -- the segment, crop, embed, batch-correct and cluster workers, the pinned DINOv3 and
  Cellpose-SAM assets (`assets.py`), the agent-arm drivers (`agents.py`) and the scoring helpers.
  `full_CPJUMP1` and `representation_RxRx1` call into it.
- `workspace_template/` -- the tool CLIs and `TASK.md` an external coding-agent arm is given.
