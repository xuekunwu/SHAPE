"""Synthetic $ROOT/private + $ROOT/prep and synthetic tool calls (ledger lines + call directories
with the files of contract 7.3, the template's code hashes, and the agent's traced Bash command of
each call) for the ledger / coverage / score / control tests. Not a test module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOOL_PREFIX = {"segment": "S", "crop": "C", "embed": "E", "batch_correct": "B", "cluster": "K"}
# the synthetic coarse classes of build_private's types (the real map is benchmarks/full_HCMA/morph_classes.csv)
SYNTHETIC_CLASSES = {"c0": "m0", "c1": "m0", "c2": "m1", "c3": "m1", "lonely": "m2", "single": "m2"}
TEMPLATE_TOOLS = Path(__file__).resolve().parents[1] / "workspace_template" / "tools"


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_private(root: Path, *, classes: int = 4, parents_per_class: int = 6, images_per_parent: int = 2) -> pd.DataFrame:
    """labels.csv, folds.csv, scorable.csv under root/private. Classes `c0..` span the five folds
    (parents round-robin); `lonely` has two parents in fold 0 (scorable, but unscorable at scoring);
    `single` has one parent (not scorable). Returns the labels."""
    rows, folds = [], {}
    parent_no, image_no = 0, 0

    def add(state: str, fold: int, organ: str, cell_type: str) -> None:
        nonlocal parent_no, image_no
        parent_no += 1
        parent = f"par_{parent_no:04d}"
        folds[parent] = fold
        for _ in range(images_per_parent):
            image_no += 1
            rows.append({"image_id": f"img_{image_no:04d}", "orig_image_id": f"orig{image_no}",
                         "image_filename": f"orig{image_no}.png", "parent_id": parent,
                         "orig_parent_image_id": f"origpar{parent_no}", "source": f"src{image_no % 3}",
                         "cell_type": cell_type, "cell_state": state, "organ_category": organ})

    for c in range(classes):
        for p in range(parents_per_class):
            add(f"c{c}", p % 5, f"organ{(c + p) % 3}", f"type{c}_{p % 2}")
    add("lonely", 0, "organ0", "lonely type")
    add("lonely", 0, "organ1", "lonely type")
    add("single", 1, "organ2", "single type")
    labels = pd.DataFrame(rows)
    private = Path(root) / "private"
    private.mkdir(parents=True, exist_ok=True)
    labels.to_csv(private / "labels.csv", index=False)
    fold_table = pd.DataFrame({"parent_id": sorted(folds), "fold": [folds[p] for p in sorted(folds)]})
    fold_table.to_csv(private / "folds.csv", index=False)
    n_parents = labels.groupby("cell_state")["parent_id"].nunique()
    keep = labels[labels["cell_state"].map(n_parents) >= 2]
    keep[["image_id", "cell_state", "parent_id"]].merge(fold_table, on="parent_id").to_csv(
        private / "scorable.csv", index=False)
    prep = Path(root) / "prep"
    (prep / "images").mkdir(parents=True, exist_ok=True)
    labels[["image_id", "parent_id", "source"]].assign(file=[f"images/{i}.png" for i in labels["image_id"]]) \
        .to_csv(prep / "images.csv", index=False)
    (prep / "PREPARE.json").write_text(json.dumps({"n_images": len(labels)}), encoding="utf-8")
    for image_id in labels["image_id"]:
        (prep / "images" / f"{image_id}.png").write_bytes(f"pixels of {image_id}".encode())
    (private / "PREPARE_PRIVATE.json").write_text(json.dumps({"prep_sha256": {
        f.relative_to(prep).as_posix(): digest(f) for f in sorted(prep.rglob("*")) if f.is_file()}}), encoding="utf-8")
    return labels


def build_subset(root: Path, name: str, image_ids: list[str]) -> Path:
    """$ROOT/prep_<name> with those images of the full prep, and private/SUBSET_<name>.json (prep_sha256)."""
    full, prep = Path(root) / "prep", Path(root) / f"prep_{name}"
    (prep / "images").mkdir(parents=True)
    images = pd.read_csv(full / "images.csv", dtype=str)
    images[images["image_id"].isin(image_ids)].to_csv(prep / "images.csv", index=False)
    (prep / "PREPARE.json").write_text(json.dumps({"n_images": len(image_ids)}), encoding="utf-8")
    for image_id in image_ids:
        (prep / "images" / f"{image_id}.png").write_bytes((full / "images" / f"{image_id}.png").read_bytes())
    labels = pd.read_csv(Path(root) / "private" / "labels.csv", dtype=str)
    chosen = labels.set_index("image_id").loc[image_ids].reset_index().sort_values("cell_state", kind="stable")
    selection = chosen[["image_id", "cell_state", "parent_id", "source"]].to_dict(orient="records")
    (Path(root) / "private" / f"SUBSET_{name}.json").write_text(json.dumps({"selection": selection, "prep_sha256": {
        f.relative_to(prep).as_posix(): digest(f) for f in sorted(prep.rglob("*")) if f.is_file()}}), encoding="utf-8")
    return prep


def write_class_map(path: Path, mapping: dict[str, str] | None = None) -> Path:
    """A synthetic cell_state -> morph_class map with the real file's columns (cell_state, morph_class, note)."""
    mapping = SYNTHETIC_CLASSES if mapping is None else mapping
    pd.DataFrame({"cell_state": list(mapping), "morph_class": list(mapping.values()),
                  "note": "synthetic"}).to_csv(path, index=False)
    return Path(path)


def workspace_data(run: Path, root: Path, prep: str = "prep", metadata: str = "full") -> None:
    """data/images.csv (blanked at the metadata level, workspace.blank_metadata) + data/PREPARE.json
    copied from the prep and <run>/DATA_SHA256.json, as setup does."""
    from benchmarks.full_HCMA.harness.workspace import blank_metadata

    data = Path(run) / "workspace" / "data"
    data.mkdir(parents=True, exist_ok=True)
    images = pd.read_csv(Path(root) / prep / "images.csv", dtype=str, keep_default_na=False)
    blank_metadata(images, metadata).to_csv(data / "images.csv", index=False)
    (data / "PREPARE.json").write_bytes((Path(root) / prep / "PREPARE.json").read_bytes())
    (Path(run) / "DATA_SHA256.json").write_text(json.dumps({n: digest(data / n) for n in ("images.csv", "PREPARE.json")}))


def informative(labels: pd.DataFrame, cells: pd.DataFrame, *, dim: int = 16, signal: float = 3.0, seed: int = 0) -> np.ndarray:
    """An embedding that carries the cell_state of each cell's image (plus noise)."""
    rng = np.random.default_rng(seed)
    states = sorted(labels["cell_state"].unique())
    state_of = labels.set_index("image_id")["cell_state"]
    x = rng.normal(size=(len(cells), dim))
    for i, image_id in enumerate(cells["image_id"]):
        x[i, states.index(state_of[image_id]) % dim] += signal
    return x.astype(np.float32)


def random_embedding(cells: pd.DataFrame, *, dim: int = 16, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(len(cells), dim)).astype(np.float32)


class Chain:
    """Writes synthetic tool calls into <run>: workspace/calls/<ID>_<NAME>/, logs/tool_calls.jsonl and
    (`traced`) the agent's Bash command of each call in traces/round_00.jsonl."""

    def __init__(self, run: Path, *, traced: bool = True):
        self.run, self.traced = Path(run), traced
        self.workspace = self.run / "workspace"
        for sub in ("calls", "submission", "tools", "data"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        (self.run / "logs").mkdir(parents=True, exist_ok=True)
        self.counter: dict[str, int] = {}

    def _new(self, tool: str, name: str) -> tuple[str, Path]:
        prefix = TOOL_PREFIX[tool]
        self.counter[prefix] = self.counter.get(prefix, 0) + 1
        call_id = f"{prefix}{self.counter[prefix]:03d}"
        directory = self.workspace / "calls" / f"{call_id}_{name}"
        directory.mkdir(parents=True)
        return call_id, directory

    def _log(self, line: dict) -> None:
        with open(self.run / "logs" / "tool_calls.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def _record(self, call_id: str, tool: str, name: str, directory: Path, *, inputs: dict, outputs: list[str],
                set_levers: list[str], status: str, seconds: float, argv: list[str] | None = None,
                set_values: dict | None = None, params: dict | None = None) -> None:
        """`set_values`: the start line's `set` (default {lever: 1}); `params`: its `params`."""
        argv = argv or ["--name", name]
        given = dict(set_values) if set_values is not None else {k: 1 for k in set_levers}
        set_levers = sorted(given) if set_values is not None else set_levers
        self._log({"event": "start", "time": 0.0, "call": call_id, "tool": tool, "name": name,
                   "call_dir": f"calls/{directory.name}", "set": given, "params": dict(params or {}),
                   "inputs": inputs, "per_image_sha256": None, "set_levers": set_levers, "argv": argv,
                   "tool_sha256": digest(TEMPLATE_TOOLS / f"{tool}.py"),
                   "common_sha256": digest(TEMPLATE_TOOLS / "_common.py"), "stubs_active": False})
        if self.traced:
            (self.run / "traces").mkdir(exist_ok=True)
            with open(self.run / "traces" / "round_00.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "assistant", "message": {"content": [
                    {"type": "tool_use", "id": call_id, "name": "Bash",
                     "input": {"command": " ".join(["python", f"tools/{tool}.py", *argv])}}]}}) + "\n")
        self._log({"event": "end", "time": seconds, "call": call_id, "tool": tool, "status": status,
                   "seconds": seconds, "outputs": {rel: digest(directory / rel) for rel in outputs},
                   "summary": {}, "error": None if status != "failed" else "synthetic failure"})

    def segment(self, images: list[str], *, name: str = "seg", set_levers=(), status: str = "done") -> str:
        call_id, d = self._new("segment", name)
        (d / "masks").mkdir()
        for image_id in images:
            (d / "masks" / f"{image_id}.tif").write_bytes(f"mask {call_id} {image_id}".encode())
        pd.DataFrame({"image_id": images, "status": "done", "cell_count": 1}).to_csv(d / "per_image.csv", index=False)
        self._record(call_id, "segment", name, d, inputs={"images": "all"}, set_levers=list(set_levers),
                     outputs=["per_image.csv", *[f"masks/{i}.tif" for i in images]], status=status, seconds=10.0)
        return call_id

    def crop(self, masks: str, cells_per_image: dict[str, int], *, name: str = "crop", set_levers=(),
             status: str = "done") -> str:
        call_id, d = self._new("crop", name)
        rows = []
        for image_id, n in cells_per_image.items():
            (d / "crops" / image_id).mkdir(parents=True)
            for label in range(1, n + 1):
                cell_id = f"{image_id}_{masks}_{label:05d}"
                rel = f"crops/{image_id}/{cell_id}.png"
                (d / rel).write_bytes(f"crop {call_id} {cell_id}".encode())
                rows.append({"cell_id": cell_id, "image_id": image_id, "tool_cell_id": label, "mask_label": label,
                             "crop_path": rel, "sha256": digest(d / rel)})
        pd.DataFrame(rows, columns=["cell_id", "image_id", "tool_cell_id", "mask_label", "crop_path", "sha256"]) \
            .to_csv(d / "crops.csv", index=False)
        pd.DataFrame({"image_id": list(cells_per_image), "mask_call": masks, "status": "done",
                      "n_crops": list(cells_per_image.values())}).to_csv(d / "per_image.csv", index=False)
        self._record(call_id, "crop", name, d, inputs={"masks": masks}, set_levers=list(set_levers),
                     outputs=["crops.csv", "per_image.csv"], status=status, seconds=5.0)
        return call_id

    def crop_cells(self, crop_call: str) -> pd.DataFrame:
        d = next((self.workspace / "calls").glob(f"{crop_call}_*"))
        return pd.read_csv(d / "crops.csv", dtype=str)[["cell_id", "image_id"]]

    def embed(self, crops: str | None, x_fn, *, name: str = "emb", exclude: list[str] | None = None,
              drop: list[str] | None = None, set_levers=(), status: str = "done", crops_map: dict | None = None,
              set_values: dict | None = None, seed: int = 0, meta: dict | None = None,
              exclude_file: str | None = None, argv: list[str] | None = None) -> str:
        """`x_fn(cells) -> embedding`; `exclude`: cells listed in an exclude file (`exclude_file`: its
        workspace-relative path); `drop`: cells left out without being listed (a broken delivery);
        `crops_map` ({image_id: crop call}) instead of one crop call `crops`; `argv`: the recorded
        command arguments (default: built from the call)."""
        if crops_map:
            frames = [self.crop_cells(c).assign(crop_call=c) for c in sorted(set(crops_map.values()))]
            cells = pd.concat(frames, ignore_index=True)
            cells = cells[[crops_map.get(i) == c for i, c in zip(cells["image_id"], cells["crop_call"])]]
        else:
            cells = self.crop_cells(crops).assign(crop_call=crops)
        inputs: dict = {"crops": crops_map or crops, "meta": dict(meta or {})}
        if exclude:
            rel = exclude_file or f"exclude_{name}.txt"
            path = self.workspace / rel
            path.write_text("\n".join(exclude) + "\n", encoding="utf-8")
            inputs.update(exclude_cells=rel, exclude_cells_sha256=digest(path))
        cells = cells[~cells["cell_id"].isin([*(exclude or []), *(drop or [])])].reset_index(drop=True)
        call_id, d = self._new("embed", name)
        given_argv, argv = argv, ["--name", name]
        if crops_map:
            (d / "inputs").mkdir()
            (d / "inputs" / "crops_map.json").write_text(json.dumps(crops_map), encoding="utf-8")
            argv += ["--crops-map", "crops_map.json"]
        else:
            argv += ["--crops", str(crops)]
        argv = list(given_argv) if given_argv else argv
        cells[["cell_id", "image_id", "crop_call"]].to_csv(d / "cells.csv", index=False)
        np.save(d / "embedding.npy", x_fn(cells))
        (d / "cell_state_analyzed.h5ad").write_bytes(b"synthetic h5ad")
        self._record(call_id, "embed", name, d, inputs=inputs, set_levers=list(set_levers), argv=argv,
                     outputs=["embedding.npy", "cells.csv", "cell_state_analyzed.h5ad"], status=status, seconds=60.0,
                     set_values=set_values, params={"seed": seed})
        return call_id

    def _dir(self, call_id: str) -> Path:
        return next((self.workspace / "calls").glob(f"{call_id}_*"))

    def batch_correct(self, embedding: str, *, name: str = "bc", set_levers=(), status: str = "done",
                      batch_key: str | None = "source", batch_file: pd.DataFrame | None = None,
                      set_values: dict | None = None, argv: list[str] | None = None) -> str:
        """`batch_file` (cell_id, batch) instead of `batch_key`: kept as inputs/batch_file.csv."""
        src = self._dir(embedding)
        call_id, d = self._new("batch_correct", name)
        (d / "cells.csv").write_bytes((src / "cells.csv").read_bytes())
        np.save(d / "embedding.npy", np.load(src / "embedding.npy"))
        (d / "tool_result.json").write_text('{"status": "skipped"}', encoding="utf-8")
        given_argv, argv = argv, ["--name", name, "--embedding", embedding]
        if batch_file is not None:
            (d / "inputs").mkdir()
            batch_file.to_csv(d / "inputs" / "batch_file.csv", index=False)
            batch_key = None
            argv += ["--batch-file", "batch.csv"]
        else:
            argv += ["--batch-key", str(batch_key)]
        argv = list(given_argv) if given_argv else argv
        self._record(call_id, "batch_correct", name, d, inputs={"embedding": embedding, "batch_key": batch_key},
                     set_levers=list(set_levers), outputs=["embedding.npy", "cells.csv"], status=status, seconds=3.0,
                     argv=argv, set_values=set_values)
        return call_id

    def cluster(self, embedding: str, *, name: str = "k", resolutions=(0.5,), set_levers=(), status: str = "done",
                set_values: dict | None = None, argv: list[str] | None = None) -> str:
        src = self._dir(embedding)
        cells = pd.read_csv(src / "cells.csv", dtype=str)
        x = np.load(src / "embedding.npy")
        call_id, d = self._new("cluster", name)
        states = cells[["cell_id", "image_id"]].copy()
        for res in resolutions:          # the embedding's largest dimension (pairs of them below 0.5)
            states[f"state_{float(res)}"] = (np.argmax(x, axis=1) // (1 if res >= 0.5 else 2)).astype(str)
        states.to_csv(d / "states.csv", index=False)
        import anndata as ad

        adata = ad.AnnData(X=x.astype(np.float32), obs=states.set_index("cell_id", drop=False).astype(str))
        adata.obsm["X_umap"] = np.random.default_rng(0).normal(size=(len(states), 2)) + x[:, :2]
        adata.write_h5ad(d / "states.h5ad")
        self._record(call_id, "cluster", name, d, inputs={"embedding": embedding}, set_levers=list(set_levers),
                     outputs=["states.h5ad", "states.csv"], status=status, seconds=2.0, set_values=set_values,
                     argv=argv)
        return call_id

    def choose(self, choice: dict) -> None:
        (self.workspace / "submission" / "choice.json").write_text(json.dumps(choice), encoding="utf-8")


def full_chain(run: Path, labels: pd.DataFrame, cells_per_image: dict[str, int], x_fn, *,
               batch: bool = True, root: Path | None = None, prep: str = "prep") -> tuple[Chain, dict]:
    """S001 -> C001 -> E001 (-> B001) -> K001 over the images of `cells_per_image`; with `root`
    also the workspace data of setup (from the prep directory `prep`)."""
    chain = Chain(run)
    if root is not None:
        workspace_data(run, root, prep)
    s = chain.segment(list(cells_per_image))
    c = chain.crop(s, cells_per_image)
    e = chain.embed(c, x_fn)
    src = chain.batch_correct(e) if batch else e
    k = chain.cluster(src)
    ids = {"S": s, "C": c, "E": e, "B": src if batch else None, "K": k}
    chain.choose({"cluster_call": k, "resolution": 0.5, "reason": "test", "skipped": {}})
    return chain, ids
