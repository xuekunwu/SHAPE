"""SHAPE reference runner for the cell_state_HCMA end-to-end benchmark.

Produces the agent-side deliverable (a label-free AnnData with embedding + cluster) and
then hands it to the FIXED evaluator. The runner NEVER passes cell-type labels into the
embedding tool -- it only carries `cell_id` + `image_name` so the evaluator can rejoin the
held-out labels afterwards.

Front half (segment -> crop) is heavy and GPU/SAM-bound; this runner supports two entry
points so it is testable without re-segmenting every time:

  --crops-dir <dir>       a directory of pre-extracted single-cell crops named
                          "<image_id>__<cell_id>.png" (image id before the "__").
  --crop-manifest <pq>    a parquet with columns [path, cell_id, image_name]; most flexible.

It then calls Cell_State_Analyzer_Tool.execute() (label-free contrastive embedding +
Leiden), writes cell_state_analyzed.h5ad, and runs evaluate.evaluate() with --labels.

  python -m benchmarks.cell_state_HCMA.run_shape_hcma \
      --crop-manifest solver_cache/HCMA/crops.parquet \
      --labels data/HCMA/labels.parquet --join-key cell_id \
      --out-dir solver_cache/HCMA/run1 [--fast] [--zero-shot]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from benchmarks.cell_state_HCMA.config import DEFAULT as CFG
from benchmarks.cell_state_HCMA.evaluate import evaluate


def _load_crop_manifest(args) -> pd.DataFrame:
    if args.crop_manifest:
        df = pd.read_parquet(args.crop_manifest)
        for c in ("path", "cell_id", "image_name"):
            if c not in df.columns:
                raise KeyError(f"--crop-manifest missing column {c!r}")
        return df
    if args.crops_dir:
        rows = []
        for p in sorted(glob.glob(os.path.join(args.crops_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            if "__" not in stem:
                continue                       # need "<image_id>__<cell_id>"
            image_name, cell_id = stem.split("__", 1)
            rows.append({"path": p, "cell_id": stem, "image_name": image_name})
        if not rows:
            raise SystemExit(f"no '<image>__<cell>.png' crops under {args.crops_dir}")
        return pd.DataFrame(rows)
    raise SystemExit("provide --crop-manifest or --crops-dir")


def _embed_and_cluster(manifest: pd.DataFrame, out_dir: str, args) -> str:
    """Run the label-free Cell_State_Analyzer_Tool; return the AnnData path.
    cell_metadata carries ONLY id + source image -- never the cell-type label."""
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    crops = [np.asarray(Image.open(p).convert("RGB")) for p in manifest["path"]]
    # obs['group'] is the tool's "label" slot; we deliberately fill it with the source
    # image (label-free) so nothing supervised leaks. obs['image_name'] is the CV group.
    cell_metadata = [
        {"cell_id": str(r.cell_id), "crop_id": str(r.cell_id),
         "group": str(r.image_name), "image_name": str(r.image_name)}
        for r in manifest.itertuples()
    ]
    analysis = Cell_State_Analyzer_Tool().execute(
        cell_crops=crops, cell_metadata=cell_metadata,
        zero_shot=args.zero_shot, augment_color=CFG.augment_color,
        freeze_backbone=CFG.freeze_backbone, field_aware=CFG.field_aware,
        learning_rate=CFG.learning_rate, weight_decay=CFG.weight_decay,
        temperature=CFG.temperature, batch_size=CFG.batch_size,
        max_epochs=(CFG.fast_max_epochs if args.fast else CFG.max_epochs),
        cluster_resolution=CFG.cluster_resolution, n_neighbors=CFG.n_neighbors,
        n_pcs=CFG.n_pcs, input_strategy=CFG.input_strategy,
        query_cache_dir=out_dir, seed=CFG.seed)
    if analysis.get("status") == "failed" or "adata_path" not in analysis:
        raise SystemExit(f"embedding failed: {analysis.get('error', analysis)}")
    return analysis["adata_path"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop-manifest", default=None)
    ap.add_argument("--crops-dir", default=None)
    ap.add_argument("--labels", required=True, help="held-out GT [join-key, cell_type]")
    ap.add_argument("--join-key", default=CFG.join_key)
    ap.add_argument("--out-dir", default=os.path.join(CFG.output_root, "run"))
    ap.add_argument("--min-sources", type=int, default=CFG.min_sources)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    manifest = _load_crop_manifest(args)
    print(f"[hcma] {len(manifest)} crops | "
          f"{manifest['image_name'].nunique()} source images", flush=True)

    adata_path = _embed_and_cluster(manifest, args.out_dir, args)
    print(f"[hcma] AnnData -> {adata_path}", flush=True)

    scores = evaluate(
        adata_path, cell_type_key=CFG.cell_type_key, image_key=CFG.image_key,
        cluster_key=CFG.cluster_key, embedding_key=CFG.embedding_key,
        min_sources=args.min_sources, labels_path=args.labels, join_key=args.join_key)
    out = os.path.join(args.out_dir, "scores.json")
    json.dump(scores, open(out, "w"), indent=2)
    print(json.dumps(scores, indent=2))
    print(f"[hcma] scores -> {out}", flush=True)


if __name__ == "__main__":
    main()
