"""Batch correction with Harmony: SHAPE's Batch_Corrector_Tool on the embedding of an embed call, under
SHAPE's calibration rule. A correction is calibrated on label-free reference groups replicated across
batches: SHAPE's batch-stage decision (shape/agent/verification/metadata_profile.py,
batch_stage.py) needs at least two groups that occur in two or more batches (its "full" tier),
taken only from the workspace metadata (images.csv columns `group` or `treatment`, and `control`).
Without them the call returns its input unchanged and never runs the tool; with them it makes one tool
call at the tool's defaults, or with levers set at those levers (the tool's own batch-effect gate decides).
SHAPE's agentic batch loop needs a reasoner and has no deterministic fallback, so this CLI (no LLM) does not
run it.

    python tools/batch_correct.py --describe
    python tools/batch_correct.py --name NAME --embedding E<NNN> (--batch-key COLUMN | --batch-file FILE.csv)
                                  [--set lever=value ...] [--config FILE.json] [--note "why"] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TOOL = "batch_correct"
GROUP_COLUMNS = ("group", "treatment")     # images.csv: an image's label-free reference group (the first present)
CONTROL_COLUMN = "control"                 # images.csv: "true" marks a control image
NO_CALIBRATION = "no calibration reference: no label-free groups replicated across batches"

USAGE = """\
## Usage

    python tools/batch_correct.py --name NAME --embedding E<NNN> (--batch-key COLUMN | --batch-file FILE.csv)
                                  [--set lever=value ...] [--config FILE.json] [--note "why"] [--dry-run]

The calibration rule decides first (CPU, no reasoner): SHAPE's metadata profile of the embedding's
images -- the batch of each image, its reference group (images.csv `group`, else `treatment`) and
whether it is a control (`control` = true) -- needs at least two non-control groups that occur in two
or more batches (the profile's "full" tier; SHAPE's batch-metrics extra, scib-metrics, must import).
- No calibration view (no such groups, one batch, or the extra missing): the input is returned
  unchanged, no tool runs, status done, summary `corrected: false` with the reason and the tier.
- A calibration view: one `Batch_Corrector_Tool.execute(cls_output, metadata, batch_key, row_ids,
  harmony_params=<the tool's DEFAULT_CONFIG with the set levers, if any>)` call (the tool's own batch-effect
  gate decides). SHAPE's agentic batch loop needs a reasoner (no deterministic fallback); this CLI has none.
- `--batch-key COLUMN`: an images.csv column (or `image_id`), mapped onto the cells by image_id;
  `--batch-file FILE.csv`: columns cell_id, batch -- one row for every cell of the embedding.
- `--set lever=value` (JSON literal values); `--config FILE.json`: `{"set": {...}}` instead of --set.
- `--dry-run`: validate and print the resolved call; nothing is run or written.

Writes calls/B<NNN>_<NAME>/: `embedding.npy` (float32; the corrected embedding, or the input unchanged),
`cells.csv` (the input's, same order), `tool_result.json` (status, corrected, reason, tier, the metadata
profile, the tool's returned values without the matrix), meta.json,
result.json, tool.log. cluster.py takes it with --embedding B<NNN>."""


def describe(repo: Path) -> str:
    levers = C.tool_levers(TOOL, repo)
    return "\n\n".join([C.card_section(repo, C.CARD_DIRS[TOOL], "## Levers"),
                        "## Lever registry (validated by this CLI)\n\n" + C.registry_table(levers), USAGE])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="batch_correct.py", description="Harmony batch correction (SHAPE Batch_Corrector_Tool)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--embedding")
    ap.add_argument("--batch-key")
    ap.add_argument("--batch-file")
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--config")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def plan(args, cfg: dict) -> dict:
    """Validate everything and verify the embedding; raises C.Invalid or C.Mismatch."""
    name = C.validate_name(args.name)
    config, config_bytes = C.read_config_file(args.config, ("set",))
    levers = C.tool_levers(TOOL, cfg["repo"])
    params, changed = C.resolve_levers(levers, C.given_levers(config, args.set))
    files = {"inputs/config.json": config_bytes} if config_bytes is not None else {}
    if not args.embedding:
        raise C.Invalid("give --embedding E<NNN>")
    source = C.producer(cfg, args.embedding, ("embed",))
    if bool(args.batch_key) == bool(args.batch_file):
        raise C.Invalid("give --batch-key COLUMN or --batch-file FILE.csv")
    embedding, cells_csv = source.verify("embedding.npy"), source.verify("cells.csv")
    cells = C.read_csv(cells_csv)
    input_paths, batch_file = {}, None
    rows = C.read_images(cfg)
    if args.batch_key:
        columns = list(rows[0]) if rows else []
        if args.batch_key != "image_id" and args.batch_key not in columns:
            raise C.Invalid(f"--batch-key {args.batch_key!r} is not an images.csv column ({columns}) or image_id")
        by_id = {r["image_id"]: r for r in rows}
        key = args.batch_key
        batches = [c["image_id"] if key == "image_id" else by_id[c["image_id"]][key] for c in cells]
    else:
        data = C.read_input_file(args.batch_file, "--batch-file")
        table = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
        if not table or not {"cell_id", "batch"} <= set(table[0]):
            raise C.Invalid("--batch-file: expected a CSV with columns cell_id, batch")
        given = {}
        for row in table:
            if row["cell_id"] in given:
                raise C.Invalid(f"--batch-file: {row['cell_id']} is listed twice")
            given[row["cell_id"]] = row["batch"]
        ids = {c["cell_id"] for c in cells}
        if set(given) != ids:
            raise C.Invalid(f"--batch-file: {len(ids - set(given))} cell(s) of {args.embedding} without a batch, "
                            f"{len(set(given) - ids)} cell(s) not in {args.embedding}")
        key = "batch"
        batches = [given[c["cell_id"]] for c in cells]
        files["inputs/batch_file.csv"] = data
        input_paths["batch_file"] = "inputs/batch_file.csv"
        batch_file = args.batch_file
    units, groups, specs = calibration_design(cells, batches, rows)
    return {"name": name, "params": params, "set": changed, "set_levers": sorted(changed), "source": source,
            "embedding": embedding, "cells_csv": cells_csv, "cells": cells, "key": key, "batches": batches,
            "batch_key": args.batch_key, "batch_file": batch_file, "files": files, "input_paths": input_paths,
            "units": units, "groups": groups, "specs": specs}


def calibration_design(cells: list[dict], batches: list, rows: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    """What SHAPE's metadata profile reads, from the workspace metadata only: (each cell's unit, each cell's
    reference group, one image spec per unit). A unit is an image (`<image_id>|<batch>` for an image whose cells
    carry several batches); its spec = {image_id: unit, batch, group: the image's GROUP_COLUMNS value, control:
    its CONTROL_COLUMN value}; a cell's group is its image's ("default" without one)."""
    by_id = {r["image_id"]: r for r in rows}
    group_column = next((c for c in GROUP_COLUMNS if rows and c in rows[0]), None)
    per_image: dict[str, set] = {}
    for cell, batch in zip(cells, batches):
        per_image.setdefault(cell["image_id"], set()).add(str(batch))
    units, groups, specs = [], [], {}
    for cell, batch in zip(cells, batches):
        image_id = cell["image_id"]
        unit = image_id if len(per_image[image_id]) == 1 else f"{image_id}|{batch}"
        row = by_id.get(image_id) or {}
        group = str(row.get(group_column) or "").strip() if group_column else ""
        units.append(unit)
        groups.append(group or "default")
        specs.setdefault(unit, {"image_id": unit, "batch": str(batch), "group": group or "default",
                                "control": str(row.get(CONTROL_COLUMN) or "").strip()})
    return units, groups, list(specs.values())


def calibration(specs: list[dict], inputs: dict):
    """SHAPE's batch-stage decision on the cells (`batch_stage.run_followup`'s checks, in its order): the metadata
    profile of the image specs (`build_metadata_profile`, `resolve_batch_metrics`), the batch labels against it
    (`check_batch_labels`), the design (`design_problem`) and the cells' tier (`cells_profile`). Returns (the
    profile the stage would run under, None when it has a calibration view, else the reason it has none)."""
    from shape.agent.verification import batch_stage
    from shape.agent.verification.metadata_profile import build_metadata_profile, resolve_batch_metrics

    profile = build_metadata_profile(None, specs)
    if profile.tier != "full":
        return profile, NO_CALIBRATION
    profile = resolve_batch_metrics(profile)
    if profile.tier != "full":
        return profile, profile.reasons[-1]
    batch_stage.check_batch_labels(inputs, profile)
    problem = batch_stage.design_problem(inputs)
    if problem:
        return profile, problem
    profile = batch_stage.cells_profile(profile, inputs)
    return profile, (None if profile.tier == "full" else NO_CALIBRATION)


def run(call: C.Call, p: dict) -> tuple[str, dict, dict]:
    import numpy as np
    import pandas as pd

    from shape.agent.verification import batch_stage

    module = C.tool_module("Batch_Corrector_Tool", call.cfg)
    features = np.load(p["embedding"])
    row_ids = [c["cell_id"] for c in p["cells"]]
    # the batch stage's input, as `batch_stage.batch_inputs_from_h5ad` builds it from the analyzer's h5ad
    inputs = {"cls_output": np.asarray(features, dtype=float), "batch_key": "batch", "row_ids": row_ids,
              "metadata": pd.DataFrame({"batch": [str(b) for b in p["batches"]], "group": p["groups"],
                                        "image_name": p["units"]})}
    profile, reason = calibration(p["specs"], inputs)
    record = {"tier": profile.tier, "tier_reasons": list(profile.reasons),
              "metadata_profile": profile.to_dict(), "levers_applied": False, "harmony_params_passed": None}
    output, status, decision = features, batch_stage.UNCORRECTED, None
    if reason is None:                           # a calibration view: one tool call at the defaults + the set levers
        harmony_params = {**module.DEFAULT_CONFIG, **p["set"]}
        result = module.Batch_Corrector_Tool().execute(
            cls_output=features, metadata=pd.DataFrame({p["key"]: p["batches"]}), batch_key=p["key"],
            row_ids=row_ids, harmony_params=harmony_params, output_dir=str(call.dir / "tool"))
        output, status = result["corrected_cls_output"], result.get("status")
        record.update(tool={k: v for k, v in result.items() if k != "corrected_cls_output"},
                      levers_applied=bool(p["set"]), harmony_params_passed=harmony_params)
    corrected = status == batch_stage.CORRECTED or bool(((record.get("tool") or {}).get("harmony") or {}).get("applied"))
    np.save(call.dir / "embedding.npy", np.asarray(output, dtype=np.float32))
    shutil.copyfile(p["cells_csv"], call.dir / "cells.csv")
    C.write_json(call.dir / "tool_result.json", {"status": status, "corrected": corrected, "reason": reason,
                                                 "decision": decision, **record})
    C.remove_tree(call.dir / "tool")         # the tool's own copies of the matrix (the stage's trials)
    summary = {"n_cells": len(p["cells"]), "n_batches": len(set(map(str, p["batches"]))), "corrected": corrected,
               "reason": reason, "tier": profile.tier, "decision": decision, "status": status,
               "levers_applied": record["levers_applied"],
               "n_features_out": int(np.asarray(output).shape[1])}
    outputs = call.outputs([call.dir / "embedding.npy", call.dir / "cells.csv"])
    return "done", summary, outputs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.describe:
        cfg = C.load_config(optional=True)
        if cfg:
            C.setup(cfg, gpu=False)
        print(describe(C.repo_root(cfg)))
        return 0
    cfg = C.load_config()
    C.setup(cfg, gpu=False)
    try:
        p = plan(args, cfg)
    except (C.Invalid, C.Mismatch) as error:
        return C.finish(error)
    if args.dry_run:
        from shape.agent.verification.metadata_profile import build_metadata_profile

        design = build_metadata_profile(None, p["specs"])
        return C.dry_run(TOOL, p["name"], set=p["set"], set_levers=p["set_levers"], params=p["params"],
                         embedding=args.embedding, batch_key=p["batch_key"], batch_file=p["batch_file"],
                         n_cells=len(p["cells"]), n_batches=len(set(p["batches"])),
                         calibration_tier=design.tier, calibration_reasons=list(design.reasons),
                         action=(f"input returned unchanged ({NO_CALIBRATION})" if design.tier != "full" else
                                 "one tool call, harmony_params = DEFAULT_CONFIG + " + json.dumps(p["set"])))
    inputs = {"from_calls": [args.embedding], "embedding": args.embedding, "batch_key": p["batch_key"],
              "n_cells": len(p["cells"])}
    call = C.Call(cfg, TOOL, p["name"], note=args.note, argv=argv, set_=p["set"], set_levers=p["set_levers"],
                  params=p["params"], inputs=inputs, files=p["files"], input_paths=p["input_paths"])
    return C.run_call(call, lambda c: run(c, p), gpu=False)


if __name__ == "__main__":
    sys.exit(main())
