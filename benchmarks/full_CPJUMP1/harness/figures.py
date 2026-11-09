"""Descriptive UMAPs of a delivered full_CPJUMP1 run, for the case-study figure (cs-cm staining
format): small rasterised scatter panels, one per colouring, PNG + PDF at 180 dpi. These are VISUAL
ONLY -- not a verdict; the scored endpoint is the gene-retrieval mAP of the corrected well profiles.

Two levels:
  well  -- the delivered corrected well profiles' UMAP (the cluster stage's `umap.csv`), coloured by
           cluster state and by gene role (calibration / evaluation / negative control). 500 points.
  cell  -- the SSL cell embedding (`embedding.npy`) UMAP, fit on the development_train cells and
           transformed for all, coloured by cluster state, by split and by plate. Cached to
           `umap_cells.csv` so re-rendering is free.

    python -m benchmarks.full_CPJUMP1.harness.figures --run RESULTS/<arm> [--out RESULTS/<arm>/evaluation/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# cs-cm palette (colour-blind safe): splits and a qualitative state map.
SPLITS = ("development_train", "development_val", "evaluation_only")
SPLIT_LABELS = ("Training", "Validation", "Held-out")
SPLIT_COLORS = ("#0072B2", "#E69F00", "#009E73")
ROLE_ORDER = ("calibration", "evaluation", "negcon")
ROLE_LABELS = ("Calibration gene", "Evaluation gene", "Negative control")
ROLE_COLORS = ("#E69F00", "#0072B2", "#A9A9A9")
POINT = dict(s=.65, alpha=.35, linewidths=0, rasterized=True)      # cell level: 137k dense points
WELL_POINT = dict(s=26, alpha=.85, linewidths=.3, edgecolors="white")  # well level: 500 points


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=180, facecolor="white", bbox_inches="tight",
                    metadata={"Creator": "SHAPE full_CPJUMP1 figures"})
    plt.close(fig)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _limits(xy: np.ndarray, pad: float = 0.04):
    lo, hi = xy.min(0), xy.max(0)
    m = (hi - lo) * pad
    return lo - m, hi + m


def _scatter(ax, xy, colors, title, lim, pt=POINT):
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, **pt)
    ax.set(xlim=(lim[0][0], lim[1][0]), ylim=(lim[0][1], lim[1][1]), xticks=[], yticks=[])
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=11)


def _state_palette(states: np.ndarray):
    uniq = sorted(set(int(s) for s in states))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    lut = {s: cmap(i % cmap.N) for i, s in enumerate(uniq)}
    return lut, uniq


def well_figure(run: Path, out: Path) -> None:
    """The corrected well profiles' UMAP (umap.csv), coloured by cluster state and by gene role."""
    coords = _read_csv(run / "umap.csv")
    state_of = {r["well"]: int(r["state"]) for r in _read_csv(run / "states.csv")}
    prof = {r["well"]: r for r in _read_csv(run / "profiles.csv")}
    wells = [r["well"] for r in coords]
    xy = np.array([[float(r["UMAP1"]), float(r["UMAP2"])] for r in coords])
    lim = _limits(xy)
    states = np.array([state_of.get(w, -1) for w in wells])
    lut, uniq = _state_palette(states)

    def role(w):
        r = prof.get(w, {})
        if str(r.get("negative_control", "")).lower() in ("true", "1"):
            return "negcon"
        return "calibration" if r.get("calibration_group") else "evaluation"
    roles = [role(w) for w in wells]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    _scatter(a1, xy, [lut[s] for s in states], f"well profiles | cluster state ({len(uniq)})", lim, pt=WELL_POINT)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", color=lut[s], label=f"state {s}") for s in uniq],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, title="cluster")
    rc = dict(zip(ROLE_ORDER, ROLE_COLORS))
    _scatter(a2, xy, [rc[r] for r in roles], f"well profiles | gene role ({len(wells)} wells)", lim, pt=WELL_POINT)
    a2.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l)
                       for c, l in zip(ROLE_COLORS, ROLE_LABELS)], loc="upper left",
              bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8)
    fig.suptitle(f"{run.name} — delivered well-profile clustering (descriptive)", fontsize=12)
    _save(fig, out, "umap_well")


def cell_figure(run: Path, out: Path, seed: int = 20261017) -> None:
    """The SSL cell embedding UMAP (fit on development_train, transform all), coloured by cluster
    state, split and plate. Coordinates cached to umap_cells.csv."""
    import umap

    cells = json.load(open(run / "cells.json"))
    state_of = {r["well"]: int(r["state"]) for r in _read_csv(run / "states.csv")}
    usage = np.array([c.get("usage", "evaluation_only") for c in cells])
    plate = np.array([c.get("plate", "?") for c in cells])
    states = np.array([state_of.get(c.get("well"), -1) for c in cells])

    cache = run / "umap_cells.csv"
    if cache.exists():
        xy = np.array([[float(r["UMAP1"]), float(r["UMAP2"])] for r in _read_csv(cache)])
    else:
        emb = np.load(run / "embedding.npy")
        train = usage == "development_train"
        reducer = umap.UMAP(n_neighbors=30, metric="cosine", min_dist=0.3, n_epochs=300, random_state=seed)
        reducer.fit(emb[train] if train.sum() >= 50 else emb)
        xy = reducer.transform(emb)
        with open(cache, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["cell_id", "UMAP1", "UMAP2"])
            for c, p in zip(cells, xy):
                w.writerow([c["cell_id"], f"{p[0]:.5f}", f"{p[1]:.5f}"])
    lim = _limits(xy)
    lut, uniq = _state_palette(states)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    _scatter(axes[0], xy, [lut[s] for s in states], f"cluster state ({len(uniq)})", lim)
    sc = dict(zip(SPLITS, SPLIT_COLORS))
    _scatter(axes[1], xy, [sc.get(u, "#A9A9A9") for u in usage], "split", lim)
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l)
                            for c, l in zip(SPLIT_COLORS, SPLIT_LABELS)], loc="lower center",
                   bbox_to_anchor=(.5, -.22), ncol=3, frameon=False, fontsize=9)
    puniq = sorted(set(plate)); pcmap = plt.get_cmap("Set2")
    plut = {p: pcmap(i) for i, p in enumerate(puniq)}
    _scatter(axes[2], xy, [plut[p] for p in plate], "plate (batch)", lim)
    axes[2].legend(handles=[Line2D([], [], marker="o", ls="", color=plut[p], label=p) for p in puniq],
                   loc="lower center", bbox_to_anchor=(.5, -.22), ncol=len(puniq), frameon=False, fontsize=9)
    fig.suptitle(f"{run.name} — SSL cell embedding UMAP ({len(cells):,} cells, fit on train) [descriptive]",
                 fontsize=12)
    _save(fig, out, "umap_cell")


def compare_arms(runs: list[Path], out: Path, name: str = "umap_arms_compare") -> None:
    """One well-profile UMAP panel per arm (each its own coordinates + its own cluster state), titled with the
    arm's scored mAP -- the cs-cm multi-arm format. NOTE: the cluster-state colouring is per-arm and circular
    (each arm is coloured by the clustering computed on its own profiles), so similar-looking panels do NOT mean
    similar quality; the discriminator is the mAP in each title (gene retrieval is a rank metric, not gross
    structure). Panels are ordered as given."""
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 4.3))
    if len(runs) == 1:
        axes = [axes]
    maps = []
    for ax, run in zip(axes, runs):
        arm = run.name.split("_seed")[0]                       # clean arm name: control / codex / shape
        coords = _read_csv(run / "umap.csv")
        state_of = {r["well"]: int(r["state"]) for r in _read_csv(run / "states.csv")}
        xy = np.array([[float(r["UMAP1"]), float(r["UMAP2"])] for r in coords])
        states = np.array([state_of.get(r["well"], -1) for r in coords])
        lut, _ = _state_palette(states)
        try:
            mAP = json.load(open(run / "RESULT.json"))["full_metrics"]["metrics"]["mAP"]["main"]["value"]
            maps.append(f"{arm} {mAP:.3f}")
        except Exception:
            pass
        _scatter(ax, xy, [lut[s] for s in states], arm, _limits(xy),
                 pt={**WELL_POINT, "s": 14, "linewidths": .15})
        ax.title.set_fontsize(13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    # the arm names are the panel titles; the mAP (the real discriminator) sits once in the suptitle.
    sub = ("mAP:  " + "   ·   ".join(maps)) if maps else ""
    fig.suptitle(f"Delivered well-profile clustering across arms\n{sub}", fontsize=11.5, y=0.99)
    _save(fig, out, name)


def _norm(img, lo=1, hi=99):
    """Percentile-normalise one channel to [0,1] for display (Cell Painting uint16 is low-range)."""
    img = img.astype(np.float32)
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    return np.clip((img - a) / (b - a + 1e-6), 0, 1)


def _channel(stack, idx):
    """Channel `idx` of a (C,H,W) or (H,W,C) stack (CP has 5 channels: Mito/AGP/RNA/ER/DNA)."""
    return stack[idx] if stack.shape[0] <= 8 else stack[..., idx]


def examples(run: Path, prep: Path, out: Path, n_images: int = 3, crops_per: int = 6,
             name: str = "examples", seed: int = 0) -> None:
    """A qualitative gallery: a few delivered images with their segmentation outlines, each next to a
    strip of its single-cell crops (what the SSL embeds). DNA (nuclei) greyscale + cyan mask outlines;
    crops as a DNA/AGP composite. Descriptive, for the case-study README."""
    import tifffile
    from skimage.segmentation import find_boundaries

    seg = {r["image_id"]: r for r in (json.load(open(run / "segmentation.json")))}
    img_file = {r["image_id"]: r["file"] for r in (json.load(open(prep / "public" / "images.json")))}
    by_img: dict = {}
    for c in json.load(open(run / "cells.json")):
        by_img.setdefault(c["image_id"], []).append(c)
    rng = np.random.default_rng(seed)
    cand = sorted(i for i in by_img if i in seg and i in img_file and len(by_img[i]) >= crops_per)
    pick = sorted(rng.choice(cand, size=min(n_images, len(cand)), replace=False).tolist()) if cand else []

    cols = 1 + crops_per
    fig, axes = plt.subplots(len(pick), cols, figsize=(1.9 * cols, 2.2 * len(pick)), squeeze=False)
    for row, img_id in enumerate(pick):
        raw = tifffile.imread(prep / "public" / img_file[img_id])
        mask = tifffile.imread(seg[img_id]["mask"])
        dna = _norm(_channel(raw, -1))
        rgb = np.stack([dna, dna, dna], -1)
        bound = find_boundaries(mask, mode="outer")
        rgb[bound] = (0.1, 0.9, 0.9)                           # cyan cell outlines
        ax = axes[row][0]
        ax.imshow(rgb); ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(img_id, fontsize=8)
        if row == 0:
            ax.set_title(f"{seg[img_id]['n_cells']} cells (DNA + outlines)", fontsize=9)
        for j, c in enumerate(by_img[img_id][:crops_per]):
            crop = tifffile.imread(c["crop_path"])
            comp = np.stack([_norm(_channel(crop, 4)), _norm(_channel(crop, 1)),
                             _norm(_channel(crop, 4))], -1)   # DNA magenta-ish + AGP green
            a = axes[row][1 + j]; a.imshow(comp); a.set_xticks([]); a.set_yticks([])
            if row == 0 and j == 0:
                a.set_title("single-cell crops (DNA+AGP)", fontsize=9, loc="left")
    fig.suptitle(f"{run.name} — segmentation + single-cell crops (qualitative examples)", fontsize=11, y=1.0)
    fig.tight_layout()
    _save(fig, out, name)


def state_examples(run: Path, out: Path, per_state: int = 8, name: str = "state_examples", seed: int = 0) -> None:
    """A montage of the delivered CLUSTER STATES: one row per state, a strip of example single-cell crops
    (DNA+AGP) drawn from the wells assigned to that state -- shows what each morphological cluster looks
    like. States are well-level (on the corrected well profiles); cells inherit their well's state."""
    import tifffile

    states = {r["well"]: int(r["state"]) for r in _read_csv(run / "states.csv")}
    by_state: dict = {}
    for c in json.load(open(run / "cells.json")):
        st = states.get(c.get("well"))
        if st is not None and c.get("crop_path"):
            by_state.setdefault(st, []).append(c)
    order = sorted(by_state)
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(order), per_state, figsize=(1.55 * per_state, 1.7 * len(order)), squeeze=False)
    for row, st in enumerate(order):
        cells = by_state[st]
        n_wells = len({c["well"] for c in cells})
        idx = rng.choice(len(cells), size=min(per_state, len(cells)), replace=False)
        for j in range(per_state):
            a = axes[row][j]; a.set_xticks([]); a.set_yticks([])
            if j < len(idx):
                crop = tifffile.imread(cells[int(idx[j])]["crop_path"])
                comp = np.stack([_norm(_channel(crop, 4)), _norm(_channel(crop, 1)), _norm(_channel(crop, 4))], -1)
                a.imshow(comp)
            else:
                a.axis("off")
        axes[row][0].set_ylabel(f"state {st}\n{n_wells} wells\n{len(cells):,} cells", fontsize=8)
    fig.suptitle(f"{run.name} — example cells per delivered cluster state (DNA magenta + AGP green)", fontsize=11, y=1.0)
    fig.tight_layout()
    _save(fig, out, name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--level", choices=("well", "cell", "both"), default="both")
    a = ap.parse_args(argv)
    out = a.out or (a.run / "evaluation" / "figures")
    if a.level in ("well", "both"):
        well_figure(a.run, out); print(f"[figures] well -> {out}/umap_well.png")
    if a.level in ("cell", "both"):
        cell_figure(a.run, out); print(f"[figures] cell -> {out}/umap_cell.png")


if __name__ == "__main__":
    main()
