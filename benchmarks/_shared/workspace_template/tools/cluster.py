"""Morphological cell states: Leiden clustering of an embedding with the analyzer's own clustering code
(Cell_State_Analyzer_Tool._compute_umap_and_clustering: cosine neighbours, its UMAP and Leiden calls).

    python tools/cluster.py --describe
    python tools/cluster.py --name NAME --embedding E<NNN>|B<NNN> [--set resolution=[0.5,1.0] n_neighbors=15]
                            [--config FILE.json] [--note "why"] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TOOL = "cluster"

USAGE = """\
## Usage

    python tools/cluster.py --name NAME --embedding E<NNN>|B<NNN> [--set resolution=[0.5,1.0] n_neighbors=15]
                            [--config FILE.json] [--note "why"] [--dry-run]

Clusters the embedding of an embed (E) or batch_correct (B) call with the analyzer's own code path:
the neighbour graph (cosine, `n_neighbors`, at most the number of cells - 1) and the UMAP are computed
once per call by `Cell_State_Analyzer_Tool._compute_umap_and_clustering` with the first resolution, and
Leiden runs once per further resolution with the analyzer's Leiden call on the same graph (CPU only).
- `--set resolution=...`: one number or a list (`[0.5,1.0,2.0]`), each in [0.01, 10];
  `--set n_neighbors=N`; values are JSON literals. `--config FILE.json`: `{"set": {...}}` instead.
- `--dry-run`: validate and print the resolved call; nothing is run or written.

Writes calls/K<NNN>_<NAME>/: `states.csv` (cell_id, image_id, state_<resolution> per resolution),
`states.h5ad` (obs: cell_id, image_id, state_<resolution> as strings; obsm: X_embedding (the input
embedding), X_umap; uns: the call chain, the resolutions, n_neighbors, the clustering representation),
`umap.png` (the UMAP coloured by the states of the first resolution and by source), meta.json,
result.json, tool.log. The summary lists per resolution the number of states and their sizes."""


def describe() -> str:
    return "\n\n".join(["## Lever registry (validated by this CLI)\n\n" + C.registry_table(C.CLUSTER_LEVERS), USAGE])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cluster.py", description="Leiden states (the analyzer's clustering code)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--embedding")
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--config")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def state_key(resolution: float) -> str:
    return f"state_{float(resolution)}"


def compute_states(analyzer_cls, embedding, cell_ids: list[str], resolutions: list[float], n_neighbors: int):
    """(adata with X_umap and one leiden_<res> column per resolution, {resolution: that column}).
    Constructing the analyzer only sets its device; DINOv3 is loaded in execute(), never here."""
    import anndata as ad
    import numpy as np
    import scanpy as sc

    adata = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
    adata.obs_names = list(cell_ids)
    tool = analyzer_cls()
    first = tool._compute_umap_and_clustering(adata, resolutions[0], None, n_neighbors, n_pcs=None)
    if first is None:
        raise RuntimeError("the analyzer computed no clustering (anndata / scanpy unavailable)")
    keys = {resolutions[0]: first}
    for resolution in resolutions[1:]:
        key = f"leiden_{resolution}"
        sc.tl.leiden(adata, resolution=resolution, key_added=key)    # the analyzer's call, on the same graph
        keys[resolution] = key
    return adata, keys


def _apply_nature_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.0, "savefig.dpi": 300})


def umap_figure(path: Path, umap, states: list[str], sources: list[str], resolution: float) -> None:
    """Two panels: the states of the first resolution, the source of each cell (fixed palettes)."""
    _apply_nature_style()
    import matplotlib.pyplot as plt
    import numpy as np

    palette = plt.get_cmap("tab20").colors

    def colours(values: list[str]):
        order = sorted(set(values), key=lambda v: (not v.isdigit(), int(v) if v.isdigit() else 0, v))
        index = {v: n for n, v in enumerate(order)}
        return order, [palette[index[v] % len(palette)] for v in values]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    umap = np.asarray(umap)
    for ax, values, title in ((axes[0], states, f"states, resolution {resolution:g}"), (axes[1], sources, "source")):
        order, cols = colours(values)
        ax.scatter(umap[:, 0], umap[:, 1], c=cols, s=1, linewidths=0, rasterized=True)
        ax.set_title(f"{title} ({len(order)})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        if len(order) <= 20:
            for n, value in enumerate(order):
                ax.scatter([], [], color=palette[n % len(palette)], s=6, label=value or "(none)")
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plan(args, cfg: dict) -> dict:
    """Validate everything and verify the embedding; raises C.Invalid or C.Mismatch."""
    name = C.validate_name(args.name)
    config, config_bytes = C.read_config_file(args.config, ("set",))
    params, changed = C.resolve_levers(C.CLUSTER_LEVERS, C.given_levers(config, args.set))
    if not args.embedding:
        raise C.Invalid("give --embedding E<NNN> or B<NNN>")
    source = C.producer(cfg, args.embedding, ("embed", "batch_correct"))
    embedding, cells_csv = source.verify("embedding.npy"), source.verify("cells.csv")
    return {"name": name, "params": params, "set": changed, "set_levers": sorted(changed), "embedding": embedding,
            "cells": C.read_csv(cells_csv), "files": {"inputs/config.json": config_bytes} if config_bytes else {}}


def run(call: C.Call, p: dict, input_call: str) -> tuple[str, dict, dict]:
    import anndata as ad
    import numpy as np
    import pandas as pd

    embedding = np.load(p["embedding"]).astype(np.float32)
    ids = [c["cell_id"] for c in p["cells"]]
    image_ids = [c["image_id"] for c in p["cells"]]
    if embedding.shape[0] != len(ids):
        raise RuntimeError(f"{input_call}: embedding.npy has {embedding.shape[0]} rows, cells.csv {len(ids)}")
    resolutions, n_neighbors = p["params"]["resolution"], p["params"]["n_neighbors"]
    call.progress(f"{len(ids)} cells, resolutions {resolutions}")
    adata, keys = compute_states(C.tool_class("Cell_State_Analyzer_Tool", call.cfg), embedding, ids, resolutions,
                                 n_neighbors)
    states = {state_key(r): [str(v) for v in adata.obs[key]] for r, key in keys.items()}
    table = pd.DataFrame({"cell_id": ids, "image_id": image_ids, **states})
    C.write_csv(call.dir / "states.csv", table.to_dict("records"), list(table.columns))
    chain = C.chain_calls(call.cfg, input_call)
    obs = table.set_index(pd.Index(ids, name=None))
    out = ad.AnnData(obs=obs, obsm={"X_embedding": embedding, "X_umap": np.asarray(adata.obsm["X_umap"])},
                     uns={"input_call": input_call, "chain": chain, "resolutions": [float(r) for r in resolutions],
                          "n_neighbors": int(n_neighbors),
                          "clustering_representation": C.json_safe(adata.uns.get("clustering_representation") or {})})
    out.write_h5ad(call.dir / "states.h5ad")
    source_of = {r["image_id"]: r.get("source", "") for r in C.read_images(call.cfg)}
    umap_figure(call.dir / "umap.png", adata.obsm["X_umap"], states[state_key(resolutions[0])],
                [source_of.get(i, "") for i in image_ids], resolutions[0])
    per_resolution = {}
    for r in resolutions:
        sizes = sorted(pd.Series(states[state_key(r)]).value_counts().tolist(), reverse=True)
        per_resolution[str(float(r))] = {"n_states": len(sizes), "sizes": sizes}
    summary = {"n_cells": len(ids), "n_images": len(set(image_ids)), "resolutions": per_resolution,
               "n_neighbors_used": min(max(2, int(n_neighbors)), max(2, len(ids) - 1))}
    outputs = call.outputs([call.dir / "states.h5ad", call.dir / "states.csv"])
    return "done", summary, outputs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.describe:
        print(describe())
        return 0
    cfg = C.load_config()
    C.setup(cfg, gpu=False)
    try:
        p = plan(args, cfg)
    except (C.Invalid, C.Mismatch) as error:
        return C.finish(error)
    if args.dry_run:
        return C.dry_run(TOOL, p["name"], set=p["set"], set_levers=p["set_levers"], params=p["params"],
                         embedding=args.embedding, n_cells=len(p["cells"]))
    call = C.Call(cfg, TOOL, p["name"], note=args.note, argv=argv, set_=p["set"], set_levers=p["set_levers"],
                  params=p["params"], inputs={"from_calls": [args.embedding], "embedding": args.embedding},
                  files=p["files"])
    return C.run_call(call, lambda c: run(c, p, args.embedding), gpu=False)


if __name__ == "__main__":
    sys.exit(main())
