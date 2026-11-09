"""`harness attribute`: a 2 x 2 factorial attribution of a run's difference to the control, cut at the crops.

Factors: objects O (segment + crop + the embed-time cell exclusions: which cells are delivered and how
they are cropped) and representation R (the embed + batch_correct + cluster recipe).

    Y00 = the control's result (O0, R0)          Y01 = the control's crops + the run's recipe (O0, R1)
    Y10 = the run's crops + the control's recipe  Y11 = the run's delivered result (O1, R1)

Y10 and Y01 are built in replay workspaces (workspace.setup_run on the same prep and metadata level):
the donor's segment / crop calls the chain used are imported with their ledger lines (tagged
`imported_from`, verified against their recorded sha256 here and by the chain walker), the new calls
are issued through control._call (logged in logs/control_commands.jsonl) and the chains are scored
with score_choice on the same expected image set.

Transfers. Y10: the run's crop selection (its crop call or crop map) and its excluded cells, then the
control's embed levers (seed, --meta), its batch decision (batch key, levers) and its cluster levers
and chosen resolution. Y01: the control's crops (and its exclusions, none for a control), then the
run's delivered embed call's explicit levers (global --set and any per-call config), seed and --meta;
NOT its --exclude-cells (keyed to the run's crops: dropped and reported); the run's batch decision (no
batch_correct when the delivered embedding was not corrected; else the B call's levers and batch key;
a batch file is keyed to the run's cells and cannot transfer: replaced by the images.csv column whose
partition of the run's images it equals, else Y01 is incomplete); and the delivered cluster call's
levers and chosen resolution.

Decomposition, for the primary and for the cross-source guard:
    phi_O = 0.5 [(Y10 - Y00) + (Y11 - Y01)],  phi_R = 0.5 [(Y01 - Y00) + (Y11 - Y10)],
    I = Y11 - Y10 - Y01 + Y00,  total = Y11 - Y00 = phi_O + phi_R.
95 % CIs by a paired cluster bootstrap over the fine cell types (score.type_draws: 2000, seed 0) on
the four results' knn_type_counts tables: each resample draws the types once and recomputes all four
macro F1 values.

Refused (SystemExit with the reason) unless the four results share the expected image set sha256,
the class map sha256, the prep, the metadata level and the GPU model (the run's RUN_META GPU, the
control directory's slug and this node's), and the control is the zero-shot control.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import score
from .coverage import read_choice
from .ledger import COMPLETED, IMPORTED, calls, chain, sha256

CELLS = ("Y00", "Y10", "Y01", "Y11")
METRICS = {"primary": (score.PRIMARY, "knn_type_counts.csv"),
           "cross_source": (score.CROSS, "knn_type_counts_cross_source.csv")}
RULE = ("2 x 2 factorial, cut at the crops: O = segment + crop + embed-time cell exclusions, R = embed + "
        "batch_correct + cluster recipe; phi_O = 0.5[(Y10-Y00)+(Y11-Y01)], phi_R = 0.5[(Y01-Y00)+(Y11-Y10)], "
        "I = Y11-Y10-Y01+Y00, total = Y11-Y00 = phi_O+phi_R; 95 % CI: paired cluster bootstrap over fine cell "
        f"types ({score.N_BOOT}, seed {score.BOOT_SEED}), all four macro F1 values recomputed per resample")


# ---------------------------------------------------------------- inputs

def _settings(run: Path) -> dict:
    return json.loads((Path(run) / "workspace" / "tools" / "_config.json").read_text(encoding="utf-8"))


def _endpoint(result: dict) -> tuple:
    p = result.get("primary") or {}
    return (result.get("endpoint"), p.get("k"), p.get("max_cells_per_image"), p.get("vote"), p.get("temperature"))


CURRENT = (score.PRIMARY, score.KNN_K, score.MAX_CELLS, score.VOTE, score.KNN_TEMPERATURE)


def _scored(run: Path, root: Path, out: Path, name: str, prep_name: str, metadata: str) -> dict:
    """{result, counts: {metric: table}, dir, rescored}: the run's own RESULT.json and type counts,
    or a rescore into out/rescored_<name>/ when they are missing or from another scorer version."""
    run = Path(run)
    tables = {m: run / f for m, (_, f) in METRICS.items()}
    try:
        result = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    if result.get("status") == "OK" and _endpoint(result) == CURRENT and all(p.is_file() for p in tables.values()):
        return {"result": result, "counts": {m: _counts(p) for m, p in tables.items()}, "dir": str(run), "rescored": False}
    choice = read_choice(run) or {}
    fresh, scored_tables, _ = score.score_choice(run, root, choice, forbidden=[], control=None, lever_coverage=False,
                                                 prep_name=prep_name, metadata=metadata, figure=False)
    if fresh.get("status") != "OK":
        raise SystemExit(f"ATTRIBUTION_REFUSED: {name} ({run}) scores {fresh.get('status')}")
    target = Path(out) / f"rescored_{name}"
    score.write_outputs(target, {**fresh, **{k: result[k] for k in ("control_chain",) if k in result}}, scored_tables,
                        None)
    fresh = {**fresh, **{k: result[k] for k in ("control_chain",) if k in result}}
    return {"result": fresh, "counts": {"primary": scored_tables["knn_type_counts"],
                                        "cross_source": scored_tables["knn_type_counts_cross_source"]},
            "dir": str(target), "rescored": True}


def _counts(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"cell_state": str, "morph_class": str, "predicted": str}, keep_default_na=False)


def _gpu_of(run: Path, control: Path) -> dict:
    from .hardware import gpu_slug

    return {"run": score.control_slug(run)["slug"], "control": Path(control).name, "replay_node": gpu_slug()}


def _same(results: dict) -> list[str]:
    """Problems: the results do not share the image set, class map, prep and metadata level."""
    problems = []
    for key, get in (("expected image set sha256", lambda r: r.get("expected_images_sha256")),
                     ("class map sha256", lambda r: (r.get("class_map") or {}).get("sha256")),
                     ("prep", lambda r: r.get("prep_name")), ("metadata level", lambda r: r.get("metadata"))):
        values = {name: get(r) for name, r in results.items()}
        if len(set(map(str, values.values()))) != 1:
            problems.append(f"the results do not share the {key}: {values}")
    return problems


# ---------------------------------------------------------------- the recipes and the objects

def _literal(value) -> str:
    """A --set value as the tools parse it (a JSON literal, a bare string when it is not one)."""
    if isinstance(value, str):
        try:
            json.loads(value)
        except ValueError:
            return value
    return json.dumps(value, separators=(",", ":"))


def _set_args(values: dict) -> list[str]:
    return [f"{k}={_literal(v)}" for k, v in sorted(values.items())]


def objects(run: Path, verified: dict) -> dict:
    """The O factor of a verified chain: the crop selection of its embed call (a crop call or a crop
    map), its excluded cells, and the segment / crop calls to import."""
    records = calls(run)
    e = records[verified["embed_call"]]
    argv = list(e["start"].get("argv") or [])
    map_file = e["dir"] / "inputs" / "crops_map.json"
    if map_file.is_file():
        selection = {"crops_map": json.loads(map_file.read_text(encoding="utf-8"))}
        crop_calls = sorted(set(selection["crops_map"].values()))
    else:
        crop_call = argv[argv.index("--crops") + 1] if "--crops" in argv else (e["start"].get("inputs") or {}).get("crops")
        selection, crop_calls = {"crops": crop_call}, [crop_call]
    masks = set()
    for c_id in crop_calls:
        per_image = pd.read_csv(records[c_id]["dir"] / "per_image.csv", dtype=str, keep_default_na=False)
        masks |= set(per_image.get("mask_call", pd.Series(dtype=str))) - {""}
    return {**selection, "excluded_cells": list(verified.get("excluded_cells") or []),
            "imports": sorted(masks) + crop_calls}


def recipe(run: Path, verified: dict) -> dict:
    """The R factor of a verified chain: the embed call's explicit levers, seed and --meta; the batch
    decision (None, or the B call's levers and batch key / batch file); the cluster call's levers and
    the chosen resolution."""
    records = calls(run)
    e = records[verified["embed_call"]]["start"]
    out = {"embed_set": dict(e.get("set") or {}), "seed": int((e.get("params") or {}).get("seed", 0) or 0),
           "meta": dict((e.get("inputs") or {}).get("meta") or {}), "batch": None,
           "cluster_set": dict(records[verified["cluster_call"]]["start"].get("set") or {}),
           "resolution": verified["resolution"]}
    if verified["embedding_call"] != verified["embed_call"]:
        b = records[verified["embedding_call"]]
        inputs = b["start"].get("inputs") or {}
        batch_file = b["dir"] / "inputs" / "batch_file.csv"
        out["batch"] = {"set": dict(b["start"].get("set") or {}), "batch_key": inputs.get("batch_key"),
                        "batch_file": str(batch_file) if not inputs.get("batch_key") and batch_file.is_file() else None,
                        "call": verified["embedding_call"]}
    return out


def batch_key_of_file(run: Path, batch: dict, embedding_call: str) -> str | None:
    """The images.csv column (or image_id) whose partition of the run's images equals the batch
    file's, None when the file splits an image or matches no column."""
    records = calls(run)
    cells = pd.read_csv(records[embedding_call]["dir"] / "cells.csv", dtype=str, keep_default_na=False)
    given = pd.read_csv(batch["batch_file"], dtype=str, keep_default_na=False)
    per_image = cells.merge(given, on="cell_id")[["image_id", "batch"]].drop_duplicates()
    if per_image["image_id"].duplicated().any():
        return None
    images = pd.read_csv(Path(run) / "workspace" / "data" / "images.csv", dtype=str, keep_default_na=False)
    table = per_image.merge(images, on="image_id", how="left")
    for column in [c for c in images.columns if c != "image_id"] + ["image_id"]:
        pairs = table[["batch", column]].drop_duplicates()
        if not pairs["batch"].duplicated().any() and not pairs[column].duplicated().any():
            return column
    return None


# ---------------------------------------------------------------- the replay chains

def _link_or_copy(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def import_calls(donor: Path, replay: Path, call_ids: list[str]) -> None:
    """Copy the donor's call dirs into the replay workspace and append their ledger lines tagged
    `imported_from`; every recorded output must verify against its sha256 (else SystemExit)."""
    records = calls(donor)
    ledger = Path(replay) / "logs" / "tool_calls.jsonl"
    lines = []
    for call_id in call_ids:
        record = records.get(call_id)
        if record is None or record["start"] is None or record["end"] is None \
                or record["end"].get("status") not in COMPLETED or record["dir"] is None:
            raise SystemExit(f"ATTRIBUTION_REFUSED: {donor}: {call_id} is not a completed call to import")
        target = Path(replay) / "workspace" / record["start"]["call_dir"]
        shutil.copytree(record["dir"], target, copy_function=_link_or_copy)
        for rel, digest in (record["end"].get("outputs") or {}).items():
            if not (target / rel).is_file() or sha256(target / rel) != digest:
                raise SystemExit(f"ATTRIBUTION_REFUSED: {donor}: {call_id}'s {rel} does not match its recorded sha256")
        tag = {IMPORTED: {"run": str(Path(donor).absolute()), "call": call_id}}
        lines += [json.dumps({**record["start"], **tag}), json.dumps({**record["end"], **tag})]
    with open(ledger, "a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


def build_chain(replay: Path, runner, name: str, objects_of: dict, recipe_of: dict) -> dict:
    """The embed -> (batch_correct) -> cluster calls of one transfer; returns {choice, calls, seconds}."""
    from .control import _call

    workspace = Path(replay) / "workspace"
    inputs = workspace / "attribution_inputs"
    inputs.mkdir(exist_ok=True)
    started = time.time()
    if "crops_map" in objects_of:
        (inputs / "crops_map.json").write_text(json.dumps(objects_of["crops_map"], indent=1), encoding="utf-8")
        args = ["--crops-map", "attribution_inputs/crops_map.json"]
    else:
        args = ["--crops", objects_of["crops"]]
    if objects_of.get("excluded_cells"):
        (inputs / "exclude_cells.txt").write_text("\n".join(objects_of["excluded_cells"]) + "\n", encoding="utf-8")
        args += ["--exclude-cells", "attribution_inputs/exclude_cells.txt"]
    for key, column in sorted(recipe_of["meta"].items()):
        args += ["--meta", f"{key}={column}"]
    args += ["--set", f"seed={recipe_of['seed']}", *_set_args(recipe_of["embed_set"])]
    made = {"embed": _call(Path(replay), runner, "embed", name, args)}
    source = made["embed"]
    batch = recipe_of["batch"]
    if batch is not None:
        args = ["--embedding", source, "--batch-key", batch["batch_key"]]
        if batch["set"]:
            args += ["--set", *_set_args(batch["set"])]
        made["batch_correct"] = source = _call(Path(replay), runner, "batch_correct", name, args)
    args = ["--embedding", source] + (["--set", *_set_args(recipe_of["cluster_set"])] if recipe_of["cluster_set"] else [])
    made["cluster"] = _call(Path(replay), runner, "cluster", name, args, ("done",))
    choice = {"cluster_call": made["cluster"], "resolution": recipe_of["resolution"], "skipped": {},
              "reason": f"harness attribute: {name}"}
    (workspace / "submission" / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    records = calls(replay)
    return {"choice": choice, "calls": made, "seconds": round(time.time() - started, 1),
            "call_seconds": {cid: (records[cid]["end"] or {}).get("seconds") for cid in made.values()}}


# ---------------------------------------------------------------- the decomposition

def decompose(values: dict) -> dict:
    """phi_O, phi_R, I and total from {Y00, Y10, Y01, Y11} (numbers or arrays)."""
    y00, y10, y01, y11 = (values[c] for c in CELLS)
    return {"phi_O": 0.5 * ((y10 - y00) + (y11 - y01)), "phi_R": 0.5 * ((y01 - y00) + (y11 - y10)),
            "interaction": y11 - y10 - y01 + y00, "total": y11 - y00}


def bootstrap(counts: dict[str, pd.DataFrame]) -> dict:
    """The four macro F1 values on the full sample and per resample (score.type_draws), from the
    cells' knn_type_counts tables ({cell: table})."""
    types = sorted(set().union(*(set(t["cell_state"]) for t in counts.values())))
    classes = sorted(set().union(*(set(t["morph_class"]) for t in counts.values())))
    weights = score.type_draws(len(types))
    full, draws = {}, {}
    for cell, table in counts.items():
        matrices = score._type_matrices(table, types, classes)
        full[cell] = float(score._macro_f1(np.ones((1, len(types))), *matrices)[0])
        draws[cell] = score._macro_f1(weights, *matrices)
    return {"full": full, "draws": draws, "n_types": len(types), "n_classes": len(classes)}


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


# ---------------------------------------------------------------- outputs

def _markdown(record: dict) -> str:
    lines = ["# full_HCMA attribution (2 x 2, cut at the crops)", "", record["rule"], ""]
    for metric, block in record["metrics"].items():
        y = block["cells"]
        f = lambda v: "n/a" if v is None else f"{v:.4f}"            # noqa: E731
        lines += [f"## {metric} (`{block['key']}`)", "",
                  "| | R0: control representation | R1: run representation |", "|---|---|---|",
                  f"| O0: control objects | Y00 {f(y['Y00'])} | Y01 {f(y['Y01'])} |",
                  f"| O1: run objects | Y10 {f(y['Y10'])} | Y11 {f(y['Y11'])} |", ""]
        for key in ("phi_O", "phi_R", "interaction", "total"):
            part = (block.get("decomposition") or {}).get(key) or {}
            ci = part.get("ci95")
            lines.append(f"- {key} = {f(part.get('value'))}" + (f" (95 % CI {ci[0]:.4f} .. {ci[1]:.4f})" if ci else ""))
        lines.append("")
    for cell, transfer in record["transfers"].items():
        dropped = "; ".join(f"{k}: {v}" for k, v in transfer["dropped"].items()) or "nothing"
        lines.append(f"- {cell}: calls {transfer.get('calls')}, {transfer.get('seconds')} s; dropped: {dropped}"
                     + (f"; INCOMPLETE: {transfer['incomplete']}" if transfer.get("incomplete") else ""))
    return "\n".join(lines) + "\n"


def _figure(path: Path, record: dict) -> None:
    """Waterfall Y00 -> +phi_O -> +phi_R -> Y11 with CIs, and I as a separate bar, per metric."""
    from .figures import _apply_nature_style

    _apply_nature_style()
    import matplotlib.pyplot as plt

    metrics = [(m, b) for m, b in record["metrics"].items() if b.get("decomposition")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(3.5, 2.0 * len(metrics)), squeeze=False)
    colours = {"Y": "#4d4d4d", "phi_O": "#1b9e77", "phi_R": "#7570b3", "interaction": "#d95f02"}
    for ax, (metric, block) in zip(axes[:, 0], metrics):
        y, d = block["cells"], block["decomposition"]
        start, after_o = y["Y00"], y["Y00"] + d["phi_O"]["value"]
        bars = [("Y00", 0.0, start, None, colours["Y"]),
                ("+phi_O", start, d["phi_O"]["value"], d["phi_O"]["ci95"], colours["phi_O"]),
                ("+phi_R", after_o, d["phi_R"]["value"], d["phi_R"]["ci95"], colours["phi_R"]),
                ("Y11", 0.0, y["Y11"], None, colours["Y"]),
                ("I", 0.0, d["interaction"]["value"], d["interaction"]["ci95"], colours["interaction"])]
        positions = [0, 1, 2, 3, 4.6]
        for x, (label, bottom, height, ci, colour) in zip(positions, bars):
            ax.bar(x, height, bottom=bottom, width=0.7, color=colour, linewidth=0)
            if ci is not None:
                top = bottom + height
                ax.errorbar(x, top, yerr=[[max(0.0, height - ci[0])], [max(0.0, ci[1] - height)]], fmt="none", ecolor="black",
                            elinewidth=0.5, capsize=1.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(positions)
        ax.set_xticklabels([b[0] for b in bars])
        ax.set_ylabel(f"{metric} macro F1")
        ax.set_title(block["key"], fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- the entry point

def attribute(run: Path, control: Path, out: Path, *, root: Path | None = None, repo: Path | None = None,
              runner=None) -> dict:
    """Build Y10 and Y01, score them, decompose; writes out/ATTRIBUTION.json, ATTRIBUTION.md,
    attribution.png. Raises SystemExit (ATTRIBUTION_REFUSED: why) when the cells are not comparable."""
    from .control import subprocess_runner
    from .hardware import gpu_slots
    from .subset import prep_dir
    from .workspace import setup_run

    run, control, out = Path(run).absolute(), Path(control).absolute(), Path(out).absolute()
    config = _settings(run)
    prep_name, metadata = config.get("prep_name", "full"), config.get("metadata", "full")
    root = Path(root) if root is not None else Path(config["prep"]).parent
    repo = Path(repo) if repo is not None else Path(config["repo"])
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    y00 = _scored(control, root, out, "Y00", prep_name, metadata)
    y11 = _scored(run, root, out, "Y11", prep_name, metadata)
    problems = _same({"Y00": y00["result"], "Y11": y11["result"]})
    if (y00["result"].get("control_chain") or {}).get("embed") != "zero_shot":
        problems.append(f"the control at {control} is not the zero-shot control (control_chain.embed "
                        f"{(y00['result'].get('control_chain') or {}).get('embed')!r})")
    gpu = _gpu_of(run, control)
    if len(set(gpu.values())) != 1:
        problems.append(f"the GPU models differ: {gpu}")
    if problems:
        raise SystemExit("ATTRIBUTION_REFUSED: " + "; ".join(problems))
    run_chain = chain(run, read_choice(run) or {})
    control_chain = chain(control, read_choice(control) or {})
    for name, verified in (("run", run_chain), ("control", control_chain)):
        if not verified["ok"]:
            raise SystemExit(f"ATTRIBUTION_REFUSED: the {name}'s chain does not verify: {verified['problems'][:3]}")

    o_run, o_control = objects(run, run_chain), objects(control, control_chain)
    r_run, r_control = recipe(run, run_chain), recipe(control, control_chain)
    transfers = {"Y10": {"objects_from": str(run), "recipe_from": str(control), "dropped": {}},
                 "Y01": {"objects_from": str(control), "recipe_from": str(run), "dropped": {}}}
    if o_run["excluded_cells"]:
        transfers["Y01"]["dropped"]["run_exclude_cells"] = (f"{len(o_run['excluded_cells'])} cells of the run's "
                                                            "--exclude-cells (keyed to the run's crops; they belong to O)")
    if o_control["excluded_cells"]:
        transfers["Y10"]["dropped"]["control_exclude_cells"] = f"{len(o_control['excluded_cells'])} cells (belong to O0)"
    if r_run["batch"] is not None and r_run["batch"]["batch_file"]:
        key = batch_key_of_file(run, r_run["batch"], r_run["batch"]["call"])
        transfers["Y01"]["dropped"]["batch_file"] = (f"{r_run['batch']['call']}'s --batch-file is keyed to the run's cells; "
                                                     + (f"replaced by --batch-key {key} (the same partition of the "
                                                        "run's images)" if key else "no images.csv column gives its "
                                                        "partition"))
        if key:
            r_run["batch"] = {**r_run["batch"], "batch_key": key}
        else:
            transfers["Y01"]["incomplete"] = "the run's batch file has no batch-key equivalent"
    runner = runner or subprocess_runner(repo)
    results = {"Y00": y00, "Y11": y11}
    for cell, objects_of, recipe_of, donor in (("Y10", o_run, r_control, run), ("Y01", o_control, r_run, control)):
        if transfers[cell].get("incomplete"):
            continue
        replay = out / f"replay_{cell}"
        setup_run(replay, root, repo, budget_hours=None, gpu_slots=gpu_slots(), prep=prep_dir(root, prep_name),
                  metadata=metadata)
        import_calls(donor, replay, objects_of["imports"])
        built = build_chain(replay, runner, f"attr_{cell.lower()}", objects_of, recipe_of)
        result, tables, _ = score.score_choice(replay, root, built["choice"], forbidden=[], control=None,
                                               lever_coverage=False, prep_name=prep_name, metadata=metadata,
                                               figure=False)
        score.write_outputs(replay, result, tables, None)
        if result.get("status") != "OK":
            raise SystemExit(f"ATTRIBUTION_REFUSED: {cell} scores {result.get('status')}: "
                             f"{(result.get('chain') or {}).get('problems', [])[:3]}")
        transfers[cell].update(replay=str(replay), calls=built["calls"], seconds=built["seconds"],
                               call_seconds=built["call_seconds"], imported=objects_of["imports"])
        results[cell] = {"result": result, "counts": {"primary": tables["knn_type_counts"],
                                                      "cross_source": tables["knn_type_counts_cross_source"]},
                         "dir": str(replay), "rescored": False}
    problems = _same({c: r["result"] for c, r in results.items()})
    if problems:
        raise SystemExit("ATTRIBUTION_REFUSED: " + "; ".join(problems))

    complete = all(c in results for c in CELLS)
    record = {"rule": RULE, "complete": complete, "run": str(run), "control": str(control), "prep_name": prep_name,
              "metadata": metadata, "gpu": gpu, "expected_images_sha256": y11["result"].get("expected_images_sha256"),
              "cells": {c: {"dir": r["dir"], "rescored": r["rescored"]} for c, r in results.items()},
              "transfers": transfers, "metrics": {}}
    for metric, (key, _) in METRICS.items():
        values = {c: results[c]["result"].get(key) if c in results else None for c in CELLS}
        block = {"key": key, "cells": values, "decomposition": None}
        if complete and all(v is not None for v in values.values()):
            point = decompose({c: float(v) for c, v in values.items()})
            assert np.isclose(point["total"], point["phi_O"] + point["phi_R"])
            boot = bootstrap({c: results[c]["counts"][metric] for c in CELLS})
            parts = decompose({c: boot["draws"][c] for c in CELLS})
            block["decomposition"] = {k: {"value": float(point[k]), "ci95": _ci(parts[k])} for k in point}
            block["bootstrap"] = {"full_sample": boot["full"], "n_types": boot["n_types"], "n_classes": boot["n_classes"],
                                  "n_boot": score.N_BOOT, "seed": score.BOOT_SEED}
        record["metrics"][metric] = block
    record["seconds"] = round(time.time() - started, 1)
    (out / "ATTRIBUTION.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(_markdown(record), encoding="utf-8")
    if any(b.get("decomposition") for b in record["metrics"].values()):
        _figure(out / "attribution.png", record)
    return record
