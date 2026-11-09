"""The control stage, once per prep, one GPU job: Cell_State_Analyzer_Tool on rows of the FULL images.npy (the
packed-array input crop_array), called directly with explicit kwargs.

  frozen  adaptation_depth="zero_shot" on the run prep (the train and val crops): the pretrained backbone (the same
          call as the SHAPE arm's trial 0)
  dinov3  the plain DINOv3 fine-tune (amendment A23; seed 0, `dinov3_s<k>` for an extra seed k > 0): the analyzer at
          its defaults on the train wells' crops ONLY (the cells the SHAPE / ReAct calls train on), so there is no
          val split and no proxy: every crop the tool sees is "train" (split_key="split"), it builds no validation
          and keeps the epoch with the lowest training loss (best_by "train_loss") as its resume_best.
          max_epochs = `default_epochs(E)` = E x the round budget, E being the shared fixed per-call epoch cap derived
          from the training-set size (TRAINING_BUDGET.json, `epoch_budget.py`); the tool's early stopping (the
          training-loss plateau) with no epoch floor (min_epochs=MIN_EPOCHS)
It replaces A22's default_s<k> (the same defaults on the train and val crops, best point by the val wells' proxy); a
ledger holding a default_s<k> arm is stale. Outputs go to <root>/control/: CONTROL.json (each arm's kwargs, its tool
input -- the splits and the cell count -- and the tool's facts, the provenance) and the calls under
cache/cell_state_analysis/<arm>/. CONTROL.json is rewritten after every arm: a rerun keeps the finished arms (same
code, same prep, same kwargs and input) and runs the rest. Scoring refuses a control computed with other code; it
embeds every FULL cell with a dinov3 arm's resume_best (embed_only), whatever the arm trained on.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL_NAME = "Cell_State_Analyzer_Tool"
CONTROL_DIR = "control"
# A transient filesystem failure: the node's Lustre client was evicted (errno 108, ESHUTDOWN on Linux) or an I/O
# error on Lustre (errno 5, EIO); the mount works again after seconds to minutes. The benchmark's own filesystem
# steps retry it (`fs_retry`; README `## Outputs` lists them), not the analyzer's writes.
TRANSIENT_ERRNOS = frozenset({errno.EIO, 108, getattr(errno, "ESHUTDOWN", 108)})
FS_RETRY_DELAYS = (5, 10, 30, 75)        # seconds between the 5 tries of a write
_sleep = time.sleep                      # tests replace it
# the code that makes an embedding and its measurements; equal hashes = comparable arms
CODE_FILES = ("shape/tools/cell_state_analyzer/tool.py", "shape/tools/cell_state_analyzer/quant.py",
              "shape/utils/adaptive_training.py")
CARD = "shape/tools/cell_state_analyzer/CARD.md"     # the agentic arms' lever text (`card_sha256`)
# the result keys of a call the ledgers keep: its outputs, its cost, its split, its training and val history
TOOL_FACTS = ("adata_path", "resume_best", "resume_end", "seconds", "split", "training", "val_history",
              "training_config")
DINOV3 = "dinov3"            # the plain DINOv3 fine-tune control (A23), seed 0; an extra seed k > 0 is dinov3_s<k>
RETIRED = "default_s"        # A22's default_s<k> (train + val crops, val-proxy selection): a ledger with one is stale
TRAIN_SPLITS = ("train",)    # the dinov3 arms' tool input: the train wells' crops only (no val split, no proxy)
MIN_EPOCHS = 1               # the dinov3 arms' epoch floor: none (epochs count from 1); only the plateau stops them


def analyzer():
    """A fresh Cell_State_Analyzer_Tool (one per call, as SHAPE's executor builds it)."""
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    return Cell_State_Analyzer_Tool()


def provenance() -> dict:
    """sha256 of CODE_FILES (None for a missing file) and the installed torch."""
    from importlib.metadata import PackageNotFoundError, version

    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    out = {f"{Path(f).stem}_sha256": sha256_file(REPO / f) if (REPO / f).exists() else None for f in CODE_FILES}
    try:
        out["torch"] = version("torch")
    except PackageNotFoundError:
        out["torch"] = None
    return out


def card_sha256() -> str | None:
    """sha256 of the analyzer's CARD.md (CARD; None when missing): the source of every agentic arm's `## Levers`
    prompt text, recorded in the shape / llm / react RUN_META.json (not part of `provenance`, so CONTROL.json stays
    valid)."""
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    return sha256_file(REPO / CARD) if (REPO / CARD).exists() else None


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fs_retry(call, *args, **kwargs):
    """call(*args, **kwargs), tried again (FS_RETRY_DELAYS) while it raises an OSError with a TRANSIENT_ERRNOS
    errno and re-raised after the last try; any other error is raised at once (port of the task-116 harness,."""
    for delay in (*FS_RETRY_DELAYS, None):
        try:
            return call(*args, **kwargs)
        except OSError as error:
            if error.errno not in TRANSIENT_ERRNOS or delay is None:
                raise
            print(f"filesystem: {type(error).__name__}: {error}; trying again in {delay} s", flush=True)
            _sleep(delay)


def write_json(path: Path, payload: dict) -> None:
    def write():
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        tmp.replace(path)

    fs_retry(write)


def append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, default=str) + "\n"

    def write():
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    fs_retry(write)


def default_epochs(epoch_cap: int) -> int:
    """The dinov3 arms' epoch budget (amendments A22, A23): the agentic arms' total, E (the per-call cap, derived from
    the training-set size, `epoch_budget.py`) x the round budget (config.max_rounds, one call per round), so the
    control may train as long as SHAPE may in all."""
    from benchmarks.representation_RxRx1.runners.config import DEFAULT

    return int(epoch_cap) * int(DEFAULT.max_rounds)


def dinov3_arm(seed: int) -> str:
    """The control arm of seed `seed`: dinov3 (seed 0), dinov3_s<k> (an extra seed k > 0)."""
    return DINOV3 if int(seed) == 0 else f"{DINOV3}_s{int(seed)}"


def is_dinov3(name: str) -> bool:
    """Whether `name` is a dinov3 control arm (dinov3, dinov3_s<k>)."""
    suffix = name[len(DINOV3) + 2:] if name.startswith(f"{DINOV3}_s") else None
    return name == DINOV3 or (suffix is not None and suffix.isdigit())


def input_splits(arm: str) -> tuple:
    """The splits of an arm's tool input: frozen the run prep (prepare.TOOL_SPLITS, train and val), a dinov3 arm the
    train wells only (TRAIN_SPLITS)."""
    from benchmarks.representation_RxRx1.runners.prepare import TOOL_SPLITS

    return tuple(TOOL_SPLITS) if arm == "frozen" else TRAIN_SPLITS


def input_record(arm: str, meta) -> dict:
    """An arm's tool input as CONTROL.json records it (`input`): its splits and its cell count; a kept arm with
    another one is stale."""
    return {"splits": list(input_splits(arm)), "n_cells": len(meta)}


def arms(seeds, split_key: str, max_epochs: int) -> dict[str, dict]:
    """Each control arm's kwargs; everything else stays at the tool's defaults. A dinov3 arm trains at most
    `default_epochs(max_epochs)` (E x the round budget, E being the dataset-derived fixed epoch cap of every SHAPE and
    ReAct call, `epoch_budget.py`, amendment A6), an explicit cap, so the tool's adaptive budget extension never
    applies; its own early stopping (the training-loss plateau) does, with no epoch floor (min_epochs=MIN_EPOCHS).
    Its input holds train crops only (`input_splits`), so split_key gives every crop "train": no val crops, no proxy,
    the best point is the lowest training loss; the split is still passed because the tool writes its resume state
    (resume_best, what scoring embeds with) only for a call with a split."""
    out = {"frozen": {"adaptation_depth": "zero_shot", "split_key": split_key}}
    out.update({dinov3_arm(s): {"seed": int(s), "split_key": split_key, "max_epochs": default_epochs(max_epochs),
                                "min_epochs": MIN_EPOCHS} for s in seeds})
    return out


def control(root: Path, *, seeds=(0, 1, 2), split_key: str = "split") -> dict:
    import shape
    from shape.utils.schemas import is_failed_result

    from benchmarks.representation_RxRx1.runners.epoch_budget import shared_budget
    from benchmarks.representation_RxRx1.runners.prepare import load, tool_input

    out = root / CONTROL_DIR
    path = out / "CONTROL.json"
    summary_prep, cells = load(root)
    epoch_cap = shared_budget(root, summary_prep, cells)["max_epochs"]
    now = {**provenance(), "prep_cells_sha256": summary_prep["cells_sha256"]}
    summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"provenance": now, "arms": {}}
    if summary.get("complete"):
        raise SystemExit(f"{path} is complete; move {out} aside to recompute")
    if summary["provenance"] != now:
        raise SystemExit(f"CONTROL_STALE: {path} holds arms of other code or another prep; move {out} aside")
    retired = sorted(a for a in summary["arms"] if a.startswith(RETIRED))
    if retired:
        raise SystemExit(f"CONTROL_STALE: {path} holds the retired {retired} (A22, val-proxy selection); move {out} "
                         "aside")
    fs_retry(out.mkdir, parents=True, exist_ok=True)
    for arm, kwargs in arms(seeds, split_key, epoch_cap).items():
        crop_array, meta = tool_input(summary_prep, cells, input_splits(arm))
        if arm in summary["arms"]:
            kept = summary["arms"][arm]
            if kept["kwargs"] != kwargs or kept.get("input") != input_record(arm, meta):
                raise SystemExit(f"CONTROL_STALE: {path} holds {arm} with other kwargs or another input; move {out} "
                                 "aside")
            print(f"control {arm}: kept ({kept['adata_path']})", flush=True)
            continue
        started = time.time()
        result = analyzer().execute(crop_array=crop_array, cell_metadata=[dict(m) for m in meta],
                                    query_cache_dir=str(out / "cache"), output_subdir=arm, **kwargs)
        if is_failed_result(result) or not result.get("adata_path"):
            raise SystemExit(f"CONTROL_FAILED: {arm}: {result.get('error') or 'no h5ad'} (finished arms are kept)")
        if arm != "frozen" and (not result.get("resume_best")
                                or not (result.get("training") or {}).get("epochs_trained")):
            raise SystemExit(f"CONTROL_FAILED: {arm}: trained no epoch or wrote no resume_best (finished arms are kept)")
        summary["arms"][arm] = {"kwargs": kwargs, "input": input_record(arm, meta),
                                **{k: result.get(k) for k in TOOL_FACTS},
                                "wall_seconds": round(time.time() - started, 1), "commit": git_commit(),
                                "shape_file": shape.__file__, "crop_array": crop_array, "host": socket.gethostname(),
                                "slurm_job": os.environ.get("SLURM_JOB_ID")}
        write_json(path, summary)
        training = result.get("training") or {}
        print(f"control {arm}: {training.get('epochs_trained')} epochs, best {training.get('best_epoch')} "
              f"by {training.get('best_by')}, {result.get('seconds')} s", flush=True)
    summary["complete"] = True
    write_json(path, summary)
    return summary
