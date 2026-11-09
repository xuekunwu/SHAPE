"""Cell embeddings with DINOv3 ViT-B/16: SHAPE's Cell_State_Analyzer_Tool on the crops of crop calls.

    python tools/embed.py --describe
    python tools/embed.py --name NAME (--crops C<NNN> | --crops-map FILE.json) [--exclude-cells FILE]
                          [--meta KEY=COLUMN ...] [--set lever=value ...] [--config FILE.json]
                          [--note "why"] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TOOL = "embed"
GROUP = "default"
# the tool's own clustering is a byproduct of the call (states come from cluster.py); fixed here
FIXED = {"cluster_resolution": 0.5, "n_neighbors": 15, "n_pcs": None}
# the analyzer's technical metadata keys (resolve_technical_metadata, resolve_plate_nuisance)
TECH_KEYS = ("plate", "experiment", "batch", "well", "site", "technical_group")

USAGE = """\
## Usage

    python tools/embed.py --name NAME (--crops C<NNN> | --crops-map FILE.json) [--exclude-cells FILE]
                          [--meta KEY=COLUMN ...] [--set lever=value ...] [--config FILE.json]
                          [--note "why"] [--dry-run]

One call runs `Cell_State_Analyzer_Tool.execute(cell_crops=[...], cell_metadata=[...], <levers>, seed=...,
cluster_resolution=0.5, n_neighbors=15, n_pcs=None)` on every selected crop (one GPU slot).
- `--crops C<NNN>`: every crop of that crop call; `--crops-map FILE.json`: `{"img_0001": "C002",
  "img_0002": "C005", ...}` (all crops of each listed image from the crop call named for it).
- `--exclude-cells FILE`: one cell_id per line, left out of the call (each must be among the selected crops).
- `--meta KEY=COLUMN`: attach an images.csv column (or `image_id`) to every crop's metadata under the
  analyzer's technical key KEY (plate, experiment, batch, well, site, technical_group). Every crop's
  metadata has cell_id, crop_id (= cell_id), image_name (= image_id), group ("default"), and the
  cropper's area and centroid.
- `--set lever=value` (JSON literal values), also `seed=N` (not a lever; default 0).
- `--config FILE.json`: `{"set": {...}}` instead of --set.
- `--dry-run`: validate and print the resolved call; nothing is run or written.

Writes calls/E<NNN>_<NAME>/: `embedding.npy` (float32, one row per cell in cells.csv order; the
h5ad's embedding, adata.X), `cells.csv` (cell_id, image_id, crop_call), `cell_state_analyzed.h5ad` (the
tool's AnnData), `loss_history.json`, `tool_result.json` (the tool's returned values), meta.json,
result.json, tool.log. batch_correct.py and cluster.py take it with --embedding E<NNN>."""


NOTE = """\
Readouts named in the text above (`baseline_embedding`, `separability_baseline`, `silhouette_by_batch`,
`silhouette_by_group`, `effective_rank`, augmentation / NN consistency) are computed by SHAPE's verifier
and are not returned by this tool.
`max_epochs` is settable here (SHAPE's representation stage param_map carries it to the tool); its
default null = the tool's own training-length rule."""


def describe(repo: Path) -> str:
    return "\n\n".join([C.card_section(repo, C.ANALYZER_DIR, "## Tunable parameters"), NOTE,
                        "## Lever registry (validated by this CLI)\n\n" + C.registry_table(C.EMBED_LEVERS)
                        + "\n\nNot a lever (recorded, not counted by lever coverage):\n\n"
                        + C.registry_table(C.EMBED_NON_LEVERS), USAGE])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="embed.py", description="DINOv3 cell embeddings (SHAPE Cell_State_Analyzer_Tool)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--crops")
    ap.add_argument("--crops-map")
    ap.add_argument("--exclude-cells")
    ap.add_argument("--meta", nargs="+", action="extend", default=[])
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--config")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def _meta(items: list[str], columns: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, eq, column = item.partition("=")
        key, column = key.strip(), column.strip()
        if not eq or key not in TECH_KEYS:
            raise C.Invalid(f"--meta {item!r}: expected KEY=COLUMN with KEY one of {list(TECH_KEYS)}")
        if column != "image_id" and column not in columns:
            raise C.Invalid(f"--meta {item!r}: {column!r} is not an images.csv column ({columns}) or image_id")
        if key in out:
            raise C.Invalid(f"--meta {key} is given twice")
        out[key] = column
    return out


def _number(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def plan(args, cfg: dict) -> dict:
    """Validate everything and verify the crops; raises C.Invalid or C.Mismatch."""
    name = C.validate_name(args.name)
    config, config_bytes = C.read_config_file(args.config, ("set",))
    params, changed = C.resolve_levers(C.EMBED_LEVERS, C.given_levers(config, args.set), C.EMBED_NON_LEVERS)
    rows = C.read_images(cfg)
    by_id = {r["image_id"]: r for r in rows}
    meta = _meta(args.meta, list(rows[0]) if rows else [])
    files = {}
    if config_bytes is not None:
        files["inputs/config.json"] = config_bytes

    if bool(args.crops) == bool(args.crops_map):
        raise C.Invalid("give --crops C<NNN> or --crops-map FILE.json")
    if args.crops:
        wanted = None
        call_ids = [args.crops]
    else:
        data = C.read_input_file(args.crops_map, "--crops-map")
        wanted = C.parse_json(data, f"--crops-map {args.crops_map}")
        if not isinstance(wanted, dict) or not wanted or not all(isinstance(v, str) for v in wanted.values()):
            raise C.Invalid("--crops-map: expected a JSON object {image_id: \"C<NNN>\"}")
        unknown = sorted(set(wanted) - set(by_id))
        if unknown:
            raise C.Invalid(f"--crops-map: unknown image(s) {unknown[:5]}")
        files["inputs/crops_map.json"] = data
        call_ids = sorted(set(wanted.values()))
    producers = {cid: C.producer(cfg, cid, ("crop",)) for cid in call_ids}
    tables = {cid: C.read_csv(prod.verify("crops.csv")) for cid, prod in producers.items()}
    if wanted is None:
        selected = [(row, args.crops) for row in tables[args.crops]]
    else:
        status = {cid: {r["image_id"]: r["status"] for r in C.read_csv(prod.verify("per_image.csv"))}
                  for cid, prod in producers.items()}
        not_done = sorted(i for i, cid in wanted.items() if status[cid].get(i) != "done")
        if not_done:
            raise C.Invalid(f"--crops-map: {not_done[:5]} not cropped (status done) by the call named for it")
        order = {r["image_id"]: n for n, r in enumerate(rows)}
        selected = [(row, cid) for image_id, cid in sorted(wanted.items(), key=lambda kv: order[kv[0]])
                    for row in tables[cid] if row["image_id"] == image_id]
    ids = [row["cell_id"] for row, _ in selected]
    if len(set(ids)) != len(ids):
        raise C.Invalid("the selected crops list a cell_id twice")
    excluded: set[str] = set()
    input_paths = {}
    if args.exclude_cells:
        data = C.read_input_file(args.exclude_cells, "--exclude-cells")
        excluded = {line.strip() for line in data.decode("utf-8").splitlines() if line.strip()}
        unknown = sorted(excluded - set(ids))
        if unknown:
            raise C.Invalid(f"--exclude-cells: {len(unknown)} cell(s) are not among the selected crops, e.g. {unknown[:5]}")
        files["inputs/exclude_cells.txt"] = data
        input_paths["exclude_cells"] = "inputs/exclude_cells.txt"
    selected = [(row, cid) for row, cid in selected if row["cell_id"] not in excluded]
    if not selected:
        raise C.Invalid("no cells left to embed")

    paths, metadata = [], []
    for row, cid in selected:
        path = producers[cid].dir / row["crop_path"]
        if not path.is_file():
            raise C.Mismatch(f"{cid}: crop {row['crop_path']} is missing")
        if C.sha256_file(path) != row["sha256"]:
            raise C.Mismatch(f"{cid}: crop {row['crop_path']} changed since the call (sha256 differs from crops.csv)")
        paths.append(str(path))
        image_id = row["image_id"]
        # The identity keys (cell_id == crop_id: the analyzer's stable-identity contract) and, as in SHAPE's
        # chain where the analyzer reads the cropper's per-crop metadata, the cropper's area and centroid.
        item = {"cell_id": row["cell_id"], "crop_id": row["cell_id"], "image_name": image_id, "group": GROUP}
        if _number(row.get("area")) is not None:
            item["area"] = _number(row["area"])
        if row.get("centroid"):
            item["centroid"] = json.loads(row["centroid"])
        for key, column in meta.items():
            item[key] = image_id if column == "image_id" else by_id[image_id][column]
        metadata.append(item)
    used_calls = sorted({cid for _, cid in selected})
    crops_of = {}
    for row, cid in selected:
        crops_of.setdefault(row["image_id"], cid)
    return {"name": name, "params": params, "set": changed, "set_levers": sorted(changed), "paths": paths,
            "metadata": metadata, "cells": [(row["cell_id"], row["image_id"], cid) for row, cid in selected],
            "from_calls": used_calls, "crops_of": crops_of, "meta": meta, "n_excluded": len(excluded),
            "files": files, "input_paths": input_paths}


def run(call: C.Call, p: dict) -> tuple[str, dict, dict]:
    import anndata as ad
    import numpy as np

    cache = call.dir / "cache"
    cache.mkdir()
    tool = C.tool_class("Cell_State_Analyzer_Tool", call.cfg)()
    levers = {k: p["params"][k] for k in C.EMBED_LEVERS}
    call.progress(f"{len(p['paths'])} crops")
    result = tool.execute(cell_crops=p["paths"], cell_metadata=p["metadata"], query_cache_dir=str(cache),
                          seed=p["params"]["seed"], **FIXED, **levers)
    if not isinstance(result, dict) or result.get("error") or not result.get("adata_path"):
        raise RuntimeError(f"the tool returned no embedding: {(result or {}).get('error') if isinstance(result, dict) else result!r}")
    h5ad = call.dir / "cell_state_analyzed.h5ad"
    shutil.move(str(result["adata_path"]), str(h5ad))
    adata = ad.read_h5ad(h5ad)
    ids = [str(v) for v in (adata.obs["cell_id"] if "cell_id" in adata.obs else adata.obs_names)]
    position = {cid: n for n, cid in enumerate(ids)}
    missing = [cid for cid, _, _ in p["cells"] if cid not in position]
    if missing or len(ids) != len(p["cells"]):
        raise RuntimeError(f"the tool's AnnData has {len(ids)} cells for {len(p['cells'])} crops "
                           f"({len(missing)} cell_ids missing)")
    matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    embedding = matrix[[position[cid] for cid, _, _ in p["cells"]]].astype(np.float32)
    np.save(call.dir / "embedding.npy", embedding)
    C.write_csv(call.dir / "cells.csv", [{"cell_id": c, "image_id": i, "crop_call": k} for c, i, k in p["cells"]],
                ["cell_id", "image_id", "crop_call"])
    C.write_json(call.dir / "loss_history.json", list(result.get("loss_history") or []))
    C.write_json(call.dir / "tool_result.json", result)
    C.remove_tree(cache)
    training = result.get("training_config") or {}
    summary = {"n_cells": len(p["cells"]), "n_images": len({i for _, i, _ in p["cells"]}),
               "n_excluded": p["n_excluded"], "embedding_dim": int(embedding.shape[1]), "mode": result.get("mode"),
               "epochs_trained": result.get("epochs_trained"), "final_loss": result.get("final_loss"),
               "stop_reason": training.get("stop_reason"), "batch_size": training.get("batch_size")}
    outputs = call.outputs([call.dir / "embedding.npy", call.dir / "cells.csv", h5ad])
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
    C.setup(cfg, gpu=True)
    try:
        p = plan(args, cfg)
    except (C.Invalid, C.Mismatch) as error:
        return C.finish(error)
    if args.dry_run:
        return C.dry_run(TOOL, p["name"], set=p["set"], set_levers=p["set_levers"], params=p["params"], fixed=FIXED,
                         crops=p["crops_of"], n_cells=len(p["cells"]), n_excluded=p["n_excluded"], meta=p["meta"])
    inputs = {"from_calls": p["from_calls"], "crops": p["crops_of"], "n_cells": len(p["cells"]),
              "n_excluded": p["n_excluded"], "meta": p["meta"]}
    call = C.Call(cfg, TOOL, p["name"], note=args.note, argv=argv, set_=p["set"], set_levers=p["set_levers"],
                  params=p["params"], inputs=inputs, files=p["files"], input_paths=p["input_paths"],
                  meta={"fixed": FIXED})
    return C.run_call(call, lambda c: run(c, p), gpu=True)


if __name__ == "__main__":
    sys.exit(main())
