"""The scored UMAP figure (evaluator output, written into the run directory, never the workspace).

Four panels on the chosen cluster call's own UMAP coordinates: (a) the true class (the primary's
morph_class), (b) the delivered states, (c) the kNN transfer outcome of the sampled cells (correct /
wrong / no prediction; unsampled cells pale grey, unscored grey, cells of a class unscorable at
the coarse level (fewer than 2 micrographs in the expected set) pale blue), (d) the source. Colours are fixed: class colours from the sorted list of every
class of the class map, the same class the same colour in every figure and arm (cells of images
left out of the coarse level: grey, "unscored"); states coloured by
size; both from one list of distinct colours (matplotlib's qualitative maps de-duplicated, in a fixed
order); fixed colours for the prediction status and the sources. A legend sits beside its panel when
it fits there (at most 24 entries, labels narrow enough), else in a full-width row under the panels
with as many columns as fit. The fine cell_state colouring is not drawn: with its ~50-90 types it
would need a row of its own that the 9 in height does not leave.
Nature style: Arial / Helvetica, 7 pt labels, 8 pt titles, 6 pt ticks, no top / right spines,
0.5 pt axes, 7.2 in wide, at most 9 in high, 300 dpi.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

UNSCORED, UNSCORED_COLOUR = "unscored", "#c7c7c7"      # images left out of the coarse level: grey in panel (a)
STATUS_COLOURS = {"correct": "#252525", "wrong": "#d62728", "no prediction": "#969696", "unlabeled": "#969696",
                  "not sampled": "#ededed", UNSCORED: UNSCORED_COLOUR, "unscorable": "#c6dbef"}
SOURCE_PALETTE = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666")
# tab10 first: the few coarse classes (and the largest states) get ten clearly different hues
QUALITATIVE = ("tab10", "tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2", "Accent", "Set1", "Set2", "Pastel1",
               "Pastel2")
WIDTH_IN, MAX_HEIGHT_IN = 7.2, 9.0
SIDE_LEGEND_MAX = 24            # more entries go to a full-width row under the panels
LEGEND_PT = 5


def _apply_nature_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.0, "axes.grid": False,
        "savefig.dpi": 300, "pdf.fonttype": 42,
        # the full canvas: the layout is computed for WIDTH_IN, and shape's
        # VisualizationConfig.setup_professional_style leaves savefig.bbox = "tight" set globally
        "savefig.bbox": "standard"})


def distinct_colours() -> list[str]:
    """Matplotlib's qualitative maps in a fixed order, de-duplicated (> 100 distinct colours)."""
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    return list(dict.fromkeys(to_hex(c) for name in QUALITATIVE for c in colormaps[name].colors))


def type_palette(all_types: list[str]) -> dict[str, str]:
    """A fixed colour per cell_state: the sorted list of every cell_state, one distinct colour each
    (cycled only beyond the list's length)."""
    colours = distinct_colours()
    return {t: colours[n % len(colours)] for n, t in enumerate(sorted(set(all_types)))}


def state_palette(states) -> dict[str, str]:
    """A colour per delivered state, the largest state first (ties: the state label's order)."""
    values, counts = np.unique(np.asarray(list(states), dtype=str), return_counts=True)
    order = sorted(zip(values, counts), key=lambda vc: (-vc[1], not vc[0].isdigit(),
                                                         int(vc[0]) if vc[0].isdigit() else 0, vc[0]))
    colours = distinct_colours()
    return {v: colours[n % len(colours)] for n, (v, _) in enumerate(order)}


def source_palette(all_sources: list[str]) -> dict[str, str]:
    return {s: SOURCE_PALETTE[n % len(SOURCE_PALETTE)] for n, s in enumerate(sorted(set(all_sources)))}


def _text_width_pt(labels, size: float) -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family="sans-serif", size=size)         # resolved through rcParams font.sans-serif
    return max((TextPath((0, 0), str(label), prop=prop).get_extents().width for label in labels), default=0.0)


def _grid_shape(labels, width_in: float, size: float) -> tuple[int, int]:
    """(columns, rows) of a legend of `labels` in a row `width_in` wide, entries at `size` pt."""
    entry = _text_width_pt(labels, size) + 2.4 * size          # marker, its pad and the column gap
    ncol = max(1, min(len(labels), int(width_in * 72 // entry)))
    return ncol, math.ceil(len(labels) / ncol) if labels else 0


def _legend(ax, entries: dict, *, ncol: int = 1, size: float = 6, title: str = "") -> None:
    from matplotlib.lines import Line2D

    ax.axis("off")
    if not entries:
        return
    handles = [Line2D([], [], marker="o", linestyle="", markersize=3, markerfacecolor=c, markeredgewidth=0,
                      label=str(label)) for label, c in entries.items()]
    ax.legend(handles=handles, title=title or None, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              fontsize=size, title_fontsize=6, ncol=ncol, handlelength=0.8, handletextpad=0.4, columnspacing=1.0,
              labelspacing=0.3, borderaxespad=0.0, borderpad=0.0, alignment="left")


SIDE_LEGEND_IN = 1.05


def _fits_beside(labels) -> bool:
    return (len(labels) <= SIDE_LEGEND_MAX
            and _text_width_pt(labels, LEGEND_PT) + 2.4 * LEGEND_PT <= SIDE_LEGEND_IN * 72)


def umap_scored(path: Path, umap: np.ndarray, classes, states, status, source, *, all_classes: list[str],
                all_sources: list[str], class_title: str = "morph_class", title: str = "",
                status_title: str = "kNN transfer (k = 20, DINO-weighted)") -> dict:
    """Write the four-panel figure; returns the colours used and the layout."""
    _apply_nature_style()
    import matplotlib.pyplot as plt

    umap = np.asarray(umap, dtype=float)
    palette, sources, state_colours = type_palette(all_classes), source_palette(all_sources), state_palette(states)
    palette[UNSCORED] = UNSCORED_COLOUR
    present = {c: palette[c] for c in sorted(set(classes) - {UNSCORED})}
    if UNSCORED in set(classes):
        present[UNSCORED] = UNSCORED_COLOUR
    classes_below, states_below = not _fits_beside(list(present)), not _fits_beside(list(state_colours))
    row_pt = LEGEND_PT * 1.25                                 # one legend row (measured at 5 pt: ~6 pt)
    class_cols, class_rows = _grid_shape(list(present), WIDTH_IN - 0.3, LEGEND_PT) if classes_below else (0, 0)
    state_cols, state_rows = _grid_shape(list(state_colours), WIDTH_IN - 0.3, LEGEND_PT) if states_below else (0, 0)
    legend_h = [(rows * row_pt + 12) / 72 for rows in (class_rows, state_rows) if rows]   # + the legend title
    title_h, gap_h = (0.35 if title else 0.1), 0.62          # the title; a panel's x label + the next title
    bottom_h = 0.4 if legend_h else 0.3                       # the last x label (and the space above a legend)
    panel_h = min(2.9, (MAX_HEIGHT_IN - title_h - 0.3 - gap_h - bottom_h - sum(legend_h)) / 2)
    height = title_h + 0.3 + 2 * panel_h + gap_h + bottom_h + sum(legend_h)
    fig = plt.figure(figsize=(WIDTH_IN, height))

    def box(x, top, w, h):           # inches from the left and from the top -> figure fractions
        return [x / WIDTH_IN, (height - top - h) / height, w / WIDTH_IN, h / height]

    panels = [("a", f"true {class_title} ({len(present)})", list(classes), palette,
               None if classes_below else (present, LEGEND_PT)),
              ("b", f"delivered states ({len(state_colours)})", list(states), state_colours,
               None if states_below else ({s: state_colours[s] for s in state_colours}, LEGEND_PT)),
              ("c", status_title, list(status), STATUS_COLOURS,
               ({s: STATUS_COLOURS[s] for s in STATUS_COLOURS if s in set(status)}, 6)),
              ("d", "source", list(source), sources, ({s: sources[s] for s in sorted(set(source))}, 6))]
    for n, (letter, label, values, colours, side) in enumerate(panels):
        row, col = divmod(n, 2)
        x, top = 0.42 + col * 3.6, title_h + 0.3 + row * (panel_h + gap_h)
        ax = fig.add_axes(box(x, top, 2.35, panel_h))
        if len(umap):
            ax.scatter(umap[:, 0], umap[:, 1], c=[colours[v] for v in values], s=1.5, linewidths=0, rasterized=True)
        ax.set_title(label, loc="left")
        ax.text(-0.04, 1.06, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        if side:
            _legend(fig.add_axes(box(x + 2.45, top, SIDE_LEGEND_IN, panel_h)), side[0], size=side[1])
    top = title_h + 0.3 + 2 * panel_h + gap_h + bottom_h
    below = ([(present, class_cols, class_title)] if classes_below else []) \
        + ([(state_colours, state_cols, "delivered states")] if states_below else [])
    for (entries, ncol, name), h in zip(below, legend_h):
        _legend(fig.add_axes(box(0.15, top, WIDTH_IN - 0.3, h)), entries, ncol=ncol, size=LEGEND_PT, title=name)
        top += h
    if title:
        fig.text(0.02, 1 - 0.08 / height, title, fontsize=8, ha="left", va="top")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {"classes": present, "states": state_colours, "status": dict(STATUS_COLOURS),
            "source": {s: sources[s] for s in sorted(set(source))},
            "layout": {"height_in": round(height, 2), "panel_in": round(panel_h, 2), "classes_legend_below": classes_below,
                       "class_legend_columns": class_cols, "states_legend_below": states_below}}
