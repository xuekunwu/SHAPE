"""Data-native figures; private cell inputs are hash checked, never copied."""
from pathlib import Path
import importlib.metadata
import platform

from common import ARMS, NAMES, SPLITS, csv_rows, read, sha, write

COLORS = ("#757575", "#009E73", "#E69F00")
SPLIT_COLORS = ("#0072B2", "#E69F00", "#009E73")
CTNT_COLORS = ("#0072B2", "#D55E00", "#A9A9A9")


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def receipt(out, script, inputs, outputs, parameters):
    write(out / "FIGURE_EVIDENCE.json", dict(
        inputs=inputs, script=dict(path=script.name, sha256=sha(script), git_commit="see publication commit"),
        parameters=parameters,
        software=dict(python=platform.python_version(), **{p: importlib.metadata.version(p) for p in ("matplotlib", "numpy", "pandas")}),
        outputs=[dict(path=p.name, sha256=sha(p)) for p in outputs],
        render_qa=dict(inspected=False, checks=["labels", "clipping", "overlap", "colors", "panels", "values"], findings=[], disposition="pending")))


def save(fig, out, name):
    paths = []
    for ext in ("png", "pdf"):
        path = out / (name + "." + ext)
        fig.savefig(path, dpi=180, facecolor="white", metadata={"Creator": "SHAPE staining evaluation"})
        paths.append(path)
    return paths


def summary(result, output):
    import numpy as np
    plt = setup()
    root, out = Path(result), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    source = root / "main_metrics.csv"
    rows = {r["arm"]: r for r in csv_rows(source)}
    metrics = [
        ("seg_bf_proxy_Q", "Segmentation BF proxy Q", "Higher; not accuracy"),
        ("crop_R_minus_C", "Crop retention minus contamination", "Higher; own masks"),
        ("crop_DAPI_target_nuclear_retention", "DAPI target nuclear retention", "Higher; assigned targets only"),
        ("crop_DAPI_other_nuclear_pixel_fraction", "Other nuclear pixel fraction", "Lower; nuclear pixels in box"),
        ("heldout_accuracy", "Held-out single-cell accuracy", "Higher; separate supervised probe"),
        ("heldout_donor_equal_balanced_accuracy", "Held-out donor-equal balanced accuracy", "Higher; donors equally weighted"),
    ]
    outputs = []
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8))
    for ax, (key, title, subtitle) in zip(axes.flat, metrics):
        values = [float(rows[a][key]) for a in ARMS]
        ax.bar(range(3), values, color=COLORS, width=.63)
        ax.set(xticks=range(3), xticklabels=[NAMES[a] for a in ARMS], ylim=(0, 1.12))
        ax.set_title(title + "\n" + subtitle, fontsize=10, pad=12)
        for i, value in enumerate(values):
            ax.text(i, value + .025, f"{value:.4f}", ha="center", fontsize=10)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Fixed 20% staining pilot | 456 images, 114 wells", fontsize=15)
    fig.text(.5, .018, "Quality: object -> image -> equal well mean. Prediction: each arm's own cells; exploratory cTnT labels.\nDifferent cell populations; these are pipeline readouts, not matched-cell or validated segmentation accuracy.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, .94), h_pad=2)
    outputs += save(fig, out, "primary_quality_prediction")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    for ax, key, title, scale in zip(axes, ("native_stage_elapsed_sum_minutes", "total_tokens"),
                                     ("Native stage elapsed sum (minutes)", "Reported model tokens (millions)"), (1, 1e6)):
        values = [float(rows[a][key]) / scale for a in ARMS]
        ax.bar(range(3), values, color=COLORS, width=.6)
        ax.set(xticks=range(3), xticklabels=[NAMES[a] for a in ARMS], title=title)
        ax.set_ylim(0, max(values) * 1.2)
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * .035, f"{v:.2f}", ha="center")
    fig.text(.5, .02, "Historical resources, overlap and cell counts differ. Stage sums exclude independent readouts and post-hoc analysis.\nElapsed sum is neither end-to-end wall time nor GPU busy time; no controlled speedup claim.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, 1))
    outputs += save(fig, out, "costs_descriptive")
    plt.close(fig)

    donor_path = root / "donor_metrics.csv"
    donors = csv_rows(donor_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rng = np.random.default_rng(20261017)
    for ax, key, title in zip(axes, ("accuracy", "balanced_accuracy"), ("Held-out donor accuracy", "Held-out donor balanced accuracy")):
        values = [[float(r[key]) for r in donors if r["arm"] == a and r["split"] == "evaluation_only" and r[key] != ""] for a in ARMS]
        box = ax.boxplot(values, widths=.5, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], COLORS):
            patch.set(facecolor=color, alpha=.3)
        for i, (v, color) in enumerate(zip(values, COLORS), 1):
            ax.scatter(i + rng.uniform(-.12, .12, len(v)), v, c=color, s=20, alpha=.8)
        ax.set(xticks=[1, 2, 3], xticklabels=[NAMES[a] for a in ARMS], ylim=(0, 1.05), title=title)
    fig.text(.5, .02, "Each point is one donor (21 per arm). Median / IQR box; whiskers 1.5 IQR; every point shown.\nDifferent own-arm cell populations. Equal-donor averages and pooled-cell accuracy answer different questions.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, 1))
    outputs += save(fig, out, "heldout_donor_boxplots")
    plt.close(fig)
    receipt(out, Path(__file__), [dict(path=p.name, sha256=sha(p)) for p in (source, donor_path)], outputs,
            dict(quality_unit="well macro", donor_point_unit="donor", seed=20261017, error_bars="not estimated"))


def umap(result, coordinates, output):
    """Render all rows, with identical coordinates and draw order per coloring."""
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    plt = setup()
    root, private, out = Path(result), Path(coordinates), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    population = read(root / "evidence/population.json")
    tables, inputs, limits, counts = {}, [], {}, {}
    for arm in ARMS:
        path = private / arm / "all_umap_coordinates.csv"
        digest = sha(path)
        if digest != population[arm]["coordinate_csv_sha256"]:
            raise ValueError(f"Coordinate input hash mismatch: {arm}")
        d = pd.read_csv(path)
        if len(d) != population[arm]["all_cells"] or not d.cell_id.is_unique:
            raise ValueError(f"Incorrect cell population: {arm}")
        if set(d.usage.unique()) != set(SPLITS) or not np.isfinite(d[["umap_1", "umap_2"]]).all().all():
            raise ValueError(f"Invalid split or coordinates: {arm}")
        if not set(d.ctnt_label.dropna().unique()).issubset({0, 1}):
            raise ValueError(f"Invalid cTnT label: {arm}")
        split_counts = d.usage.value_counts().to_dict()
        if split_counts != population[arm]["counts_by_split"]:
            raise ValueError(f"Split counts differ from recorded population: {arm}")
        tables[arm] = d.sample(frac=1, random_state=20261017)
        xy = d[["umap_1", "umap_2"]]
        lo, hi = xy.min().to_numpy(), xy.max().to_numpy()
        pad = np.maximum(hi - lo, 1e-6) * .04
        limits[arm] = (lo - pad, hi + pad)
        inputs.append(dict(path=f"private/{arm}/all_umap_coordinates.csv", sha256=digest))
        counts[arm] = dict(all=len(d), by_split=split_counts,
                           ctnt_negative=int((d.ctnt_label == 0).sum()),
                           ctnt_positive=int((d.ctnt_label == 1).sum()),
                           ambiguous_or_excluded=int(d.ctnt_label.isna().sum()))

    def panel(ax, arm, mode):
        d = tables[arm]
        if mode == "split":
            colors = d.usage.map(dict(zip(SPLITS, SPLIT_COLORS))).to_numpy()
        else:
            colors = np.full(len(d), CTNT_COLORS[2], dtype=object)
            colors[d.ctnt_label.to_numpy() == 0] = CTNT_COLORS[0]
            colors[d.ctnt_label.to_numpy() == 1] = CTNT_COLORS[1]
        ax.scatter(d.umap_1, d.umap_2, s=.65, c=colors, alpha=.35, linewidths=0, rasterized=True)
        lo, hi = limits[arm]
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xticks=[], yticks=[])
        ax.set_title(f"{NAMES[arm]} | all {len(d):,} cells", fontsize=11)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    def legend(mode):
        labels = ("Training", "Validation", "Held-out") if mode == "split" else ("cTnT negative", "cTnT positive", "Ambiguous / excluded")
        colors = SPLIT_COLORS if mode == "split" else CTNT_COLORS
        return [Line2D([], [], marker="o", linestyle="", color=c, label=l) for c, l in zip(colors, labels)]

    outputs = []
    footer = "Every delivered embedding shown; no downsampling or shared-cell intersection. Train fits PCA/UMAP; validation and held-out transform only.\nIndependent maps per arm; cTnT labels used only for coloring. Distances and apparent separation are not prediction accuracy."
    for mode in ("split", "ctnt"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
        for ax, arm in zip(axes, ARMS):
            panel(ax, arm, mode)
        fig.suptitle("All own-arm cells | " + ("Training / validation / held-out" if mode == "split" else "cTnT evaluation labels"), fontsize=15, y=.98)
        fig.legend(handles=legend(mode), loc="upper center", bbox_to_anchor=(.5, .93), ncol=3, frameon=False)
        fig.text(.5, .015, footer, ha="center", fontsize=9)
        fig.subplots_adjust(left=.035, right=.99, bottom=.17, top=.81, wspace=.15)
        outputs += save(fig, out, "umap_all_" + mode)
        plt.close(fig)
    for arm in ARMS:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6))
        for ax, mode in zip(axes, ("split", "ctnt")):
            panel(ax, arm, mode)
            ax.legend(handles=legend(mode), loc="upper center", bbox_to_anchor=(.5, -.08), ncol=1, frameon=False, fontsize=9)
        fig.suptitle(f"{NAMES[arm]} | identical coordinates, all {len(tables[arm]):,} delivered cells", fontsize=14)
        fig.text(.5, .012, "PCA/UMAP fit on training cells only; no visual sampling. Ambiguous labels are retained in both panels.", ha="center", fontsize=9)
        fig.subplots_adjust(left=.055, right=.99, top=.86, bottom=.28, wspace=.18)
        outputs += save(fig, out, "umap_" + arm + "_split_ctnt")
        plt.close(fig)
    write(out / "COUNTS.json", counts)
    receipt(out, Path(__file__), inputs, outputs, dict(all_cells=sum(len(t) for t in tables.values()),
            visual_sampling=False, draw_order_seed=20261017, fit_split="development_train",
            val_heldout="transform only", labels="colors only", coordinate_limits="same within each arm across colorings"))
