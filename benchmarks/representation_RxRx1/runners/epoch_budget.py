"""A dataset-derived, immutable per-call epoch cap shared by the two analyzer agents."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier
from shape.utils.adaptive_training import resolve_training_config


def budget_for(summary: dict, cells) -> dict:
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    train = cells[cells.split == "train"]
    n_train = len(train)
    if not n_train or n_train != summary["tool_input"]["train_cells"]:
        raise SystemExit("TRAINING_BUDGET_INVALID: training split is empty or differs from PREPARE.json")
    rows = train[["cell_id", "array_row"]].to_dict("records")
    split_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"policy": "dataset-fixed-v1", "rule": "clamp(round(8 + 6 * sqrt(128 / N_train)), 8, 20)",
            "policy_sha256": sha256_file(Path(__file__)),
            "n_train": n_train, "training_split_sha256": split_hash,
            "prep_cells_sha256": summary["cells_sha256"],
            "crop_array_sha256": summary["crop_array"]["sha256"],
            "max_epochs": resolve_training_config(n_train, 1)["max_epochs"]}


def shared_budget(root: Path, summary: dict, cells) -> dict:
    """Publish a complete manifest atomically; concurrent jobs may only agree with it."""
    from benchmarks.representation_RxRx1.runners.control import fs_retry

    expected = budget_for(summary, cells)
    path = root / "TRAINING_BUDGET.json"
    if not path.exists():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as handle:
            json.dump(expected, handle, indent=1)
            temporary = Path(handle.name)
        try:
            try:
                fs_retry(os.link, temporary, path)
            except FileExistsError:
                pass
        finally:
            fs_retry(temporary.unlink)
    recorded = json.loads(fs_retry(path.read_text, encoding="utf-8"))
    if recorded != expected:
        raise SystemExit(f"TRAINING_BUDGET_STALE: {path} does not match this prep or epoch policy")
    return recorded


def fixed_line(budget: dict) -> str:
    return (f"max_epochs is fixed at {budget['max_epochs']} for every call "
            f"(computed once from {budget['n_train']} training cells); it is not a lever. "
            "A resumed call may train at most this many additional epochs. "
            "The 8-call limit counts tuning calls, not total epochs.")


class FixedEpochVerifier(RepresentationAgenticVerifier):
    """Benchmark-only constraints; the general SHAPE tool retains its epoch lever."""

    def __init__(self, budget: dict, **kwargs):
        super().__init__(**kwargs)
        self.epoch_budget = dict(budget)
        self.lever_fields.pop("max_epochs")
        self.levers.remove("max_epochs")

    def defaults(self) -> dict:
        return {**super().defaults(), "max_epochs": self.epoch_budget["max_epochs"]}

    def _card_section(self, title=None) -> str:
        text = super()._card_section(title)
        if title is None or title == self.lever_section:
            text = re.sub(r"^### max_epochs[ \t]*\n.*?(?=^### |\Z)", "", text,
                          flags=re.MULTILINE | re.DOTALL).rstrip()
            text += "\n\n" + fixed_line(self.epoch_budget)
        return text

    def validate(self, move, attempts, pending):
        if any(c.get("lever") == "max_epochs" for c in move.get("changes") or []):
            raise ValueError(f"max_epochs is fixed at {self.epoch_budget['max_epochs']} for every call; "
                             "remove it from changes")
        return super().validate(move, attempts, pending)

    def trial_config(self, params):
        return {**super().trial_config(params), "max_epochs": self.epoch_budget["max_epochs"]}

    def trial_params(self, config):
        return {**super().trial_params(config), "max_epochs": self.epoch_budget["max_epochs"]}
