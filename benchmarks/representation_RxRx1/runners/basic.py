"""Non-mutating gates for the single-run, seven-arm RxRx1 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("frozen", "dinov3", "shape", "react", "llm", "openphenom", "cellprofiler")


def check_control(root: Path) -> bool:
    from benchmarks.representation_RxRx1.runners import control, prepare
    from benchmarks.representation_RxRx1.runners.epoch_budget import budget_for

    summary, cells = prepare.load(root)
    path = root / control.CONTROL_DIR / "CONTROL.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    retired = sorted(a for a in record.get("arms", {}) if a.startswith(control.RETIRED))
    if retired:                                  # A22's val-proxy control, replaced by dinov3 (A23)
        raise SystemExit(f"CONTROL_STALE: cached control holds the retired {retired}; preserved")
    expected = {**control.provenance(), "prep_cells_sha256": summary["cells_sha256"]}
    if record.get("provenance") != expected:
        raise SystemExit("CONTROL_STALE: cached control differs from this code or prep; preserved")
    planned = control.arms((0,), "split", budget_for(summary, cells)["max_epochs"])
    extra = set(record.get("arms", {})) - set(planned)
    if extra:
        raise SystemExit(f"NOT_BASIC: control contains extra arms {sorted(extra)}; preserved")
    for name, entry in record.get("arms", {}).items():
        if entry.get("kwargs") != planned[name]:
            raise SystemExit(f"CONTROL_STALE: {name} kwargs differ; preserved")
        meta = prepare.tool_input(summary, cells, control.input_splits(name))[1]
        if entry.get("input") != control.input_record(name, meta):
            raise SystemExit(f"CONTROL_STALE: {name} trained on another input; preserved")
        if not Path(entry.get("adata_path") or "").is_file():
            raise SystemExit(f"CONTROL_MISSING: {name} h5ad is missing; preserved")
        if name != "frozen" and (not Path(entry.get("resume_best") or "").is_file()
                                 or not (entry.get("training") or {}).get("epochs_trained")):
            raise SystemExit(f"CONTROL_MISSING: {name} trained checkpoint is missing; preserved")
    if record.get("complete") and set(record.get("arms", {})) != set(planned):
        raise SystemExit("CONTROL_MISSING: completed ledger lacks the basic arms; preserved")
    return bool(record.get("complete"))


def check_external(root: Path) -> bool:
    from benchmarks.representation_RxRx1.runners import external, prepare

    summary, cells = prepare.load(root)
    _, records = external.load_embeddings(root, summary, len(cells))
    op = records.get("openphenom", {})
    if op.get("status") == "scored" and op["settings"].get("fit") != "pad":
        raise SystemExit("EXTERNAL_STALE: baseline expects existing OpenPhenom pad, not resize; preserved")
    return all(records.get(name, {}).get("status") == "scored" for name in external.EXTERNAL_ARMS)


def check_result(run: Path) -> None:
    record = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    if record.get("default_arms") != ["dinov3"]:
        raise SystemExit("NOT_BASIC: result must contain only dinov3")
    if record.get("decision", {}).get("kind") != "final":
        raise SystemExit("INCOMPLETE: SHAPE decision is provisional")
    for label, tables in (("native", record.get("tables", {})),
                          ("matched", record.get("matched", {}).get("tables", {}))):
        missing = set(ARMS) - set(tables)
        if missing:
            raise SystemExit(f"INCOMPLETE: {label} lacks {sorted(missing)}")
        for name in ARMS:
            for split in ("seen", "heldout"):
                value = (tables[name].get(split) or {}).get("value")
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise SystemExit(f"INCOMPLETE: {label}/{name}/{split} has no valid well mAP")
    for name in ("llm", "react"):
        if record.get("llm_arms", {}).get(name, {}).get("status") != "delivered":
            raise SystemExit(f"INCOMPLETE: {name} did not deliver")
    for name in ("openphenom", "cellprofiler"):
        if record.get("external_arms", {}).get(name, {}).get("status") != "scored":
            raise SystemExit(f"INCOMPLETE: {name} was skipped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("control", "external", "result"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.stage == "result":
        check_result(args.path)
        print("BASIC_RESULT_OK arms=" + ",".join(ARMS), flush=True)
    else:
        ready = check_control(args.path) if args.stage == "control" else check_external(args.path)
        print(f"BASIC_{args.stage.upper()}_" + ("OK" if ready else "NEEDS_RUN"), flush=True)
        if not ready:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
