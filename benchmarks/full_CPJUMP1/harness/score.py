"""Post-freeze biological scoring; no gene labels are used by the runtime stages.

The verdict metrics are the shared full_* metrics (benchmarks/_shared/FULL_METRICS_CONTRACT.md, `full_metrics`):
label = gene over the evaluation genes (development / calibration genes and controls excluded), unit = well; main
split = cross-plate (train on one plate, test on the other, and the reverse); guard split = cross-plate and
cross-guide (genes with >= 2 guides; a test well's guide is never on the train side for its gene). Batch correction
runs on well profiles only, so the cell-level probe reads the uncorrected cell embedding (embedding.npy, cells.json)
and the unit level (mAP, unit-level probes) the delivered corrected well profiles (corrected.npy). The cross-plate
retrieval endpoints below stay as reported secondaries.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from benchmarks._shared import full_metrics as fm

from .data import digest, load_prepared, read_json, within, write_json

DRAWS = "full_metrics_draws.npz"
UNIT_RULE = ("cell-level probe: the uncorrected cell embedding (embedding.npy rows of cells.json; batch correction "
             "runs on well profiles only); unit level (mAP, unit-level probes): the delivered corrected well profile "
             "(corrected.npy, L2-normalised; the pipeline's well profile is the mean over the two sites of the "
             "per-site median cell, not the contract's mean of L2-normalised cells)")


def cosine(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("non-finite profiles")
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm profiles cannot be scored")
    unit = x / norms[:, None]
    return unit @ unit.T


def retrieval(meta: pd.DataFrame, matrix: np.ndarray, *, cross_guide: bool) -> tuple[dict, pd.DataFrame]:
    sim = cosine(matrix)
    records = []
    for i, row in meta.iterrows():
        if row.role != "evaluation" or row.pert_type != "trt":
            continue
        candidates = ((meta.plate != row.plate) & (meta.role == "evaluation") & (meta.pert_type == "trt"))
        if cross_guide:
            candidates &= meta.guide != row.guide
        relevant = candidates & (meta.gene == row.gene)
        if not cross_guide:
            relevant &= meta.guide == row.guide
        if not relevant.any():
            continue
        ids = np.flatnonzero(candidates.to_numpy())
        values = sim[i, ids]
        positives = relevant.to_numpy()[ids]
        # Pessimistic tie handling: collapsed/equal scores never get perfect retrieval by row order.
        ranked = sorted(zip(values[positives], ids[positives]), reverse=True)
        precisions = []
        for value, _ in ranked:
            rank = int(np.count_nonzero(values >= value))
            hit = int(np.count_nonzero(positives & (values >= value)))
            precisions.append(hit / rank)
        best_positive = float(values[positives].max())
        rank = int(np.count_nonzero(values >= best_positive))
        records.append({"well": row.well, "gene": row.gene, "plate": row.plate,
                        "average_precision": float(np.mean(precisions)), "reciprocal_rank": 1 / rank,
                        "top1": float(rank == 1), "candidates": len(ids)})
    table = pd.DataFrame(records)
    if table.empty:
        return {"n_queries": 0, "n_genes": 0, "mAP": None, "MRR": None, "top1": None}, table
    macro = table.groupby("gene")[["average_precision", "reciprocal_rank", "top1"]].mean().mean()
    return {"n_queries": len(table), "n_genes": int(table.gene.nunique()),
            "mAP": float(macro.average_precision), "MRR": float(macro.reciprocal_rank),
            "top1": float(macro.top1)}, table


def full_layout(meta: pd.DataFrame) -> tuple[fm.Layout, pd.DataFrame]:
    """The full_metrics layout over the evaluation treatment wells (sorted by well): main = cross-plate folds, guard
    = cross-plate and cross-guide folds (genes with >= 2 guides; per test plate one part per guide index k, the
    test wells of guide k against the other plate's wells of the other guides)."""
    units = meta[(meta.role == "evaluation") & (meta.pert_type == "trt")].sort_values("well").reset_index()
    plates = sorted(units.plate.unique())
    if len(plates) != 2:
        raise ValueError("the cross-plate split needs exactly two plates")
    guides = units.groupby("gene")["guide"].agg(lambda g: sorted(set(g)))
    index = np.array([guides[g].index(v) for g, v in zip(units.gene, units.guide)])
    multi = units.gene.map(guides.map(len)).to_numpy() >= 2
    plate = units.plate.to_numpy()
    main, guard = [], []
    for test_plate in plates:
        other, here = plate != test_plate, plate == test_plate
        main.append([(np.flatnonzero(other), np.flatnonzero(here))])
        parts = [(np.flatnonzero(other & multi & (index != k)), np.flatnonzero(here & multi & (index == k)))
                 for k in sorted(set(index[multi]))]
        guard.append([part for part in parts if len(part[1])])
    describe = {"main": {"rule": "cross-plate: train on one plate, test on the other, and the reverse",
                         "n_genes": int(units.gene.nunique())},
                "guard": {"rule": "cross-plate and cross-guide: genes with >= 2 guides; a test well's guide never on "
                                  "the train side for its gene", "n_genes": int(units.gene[multi].nunique())}}
    return fm.Layout(units.well.tolist(), units.gene.tolist(), {"main": main, "guard": guard}, describe), units


def excluded_image(cell_id: str) -> str:
    """The image of a cropper cell id (`your-group_<image_id>_cell_<index>`, Single_Cell_Cropper_Tool)."""
    head, sep, _ = str(cell_id).rpartition("_cell_")
    group, _, image = head.partition("_")
    if not sep or not group or not image:
        raise ValueError(f"not a cropper cell id: {cell_id}")
    return image


def full_metrics(run: Path, meta: pd.DataFrame, corrected: np.ndarray, delivery: dict,
                 control: Path | None, images: list[dict] | None = None, output: Path | None = None) -> dict:
    """RESULT.json `full_metrics` of a run (its reading saved as <output>/DRAWS, default the run's evaluation/); with
    `control` (a scored control run, or the directory its scoring went to), the deltas, SEs and verdicts against the
    control's saved reading."""
    for name in ("cells.json", "excluded_cells.json", "embedding.npy"):
        if name not in delivery["artifacts"]:
            raise ValueError(f"full metrics need the delivered {name}")
    layout, units = full_layout(meta)
    cells = read_json(run / "cells.json")
    embedding = np.load(run / "embedding.npy", mmap_mode="r")
    if len(cells) != len(embedding):
        raise ValueError("cell embedding rows disagree with cells.json")
    position = {w: i for i, w in enumerate(units.well)}
    cell_unit = np.array([position.get(c["well"], -1) for c in cells], dtype=np.int64)
    excluded = [str(c) for c in read_json(run / "excluded_cells.json")]
    well_of = {r["image_id"]: r["well"] for r in images or []}
    if any(excluded_image(c) not in well_of for c in excluded):
        raise ValueError("an excluded cell names an image outside the preparation")
    reading = fm.evaluate(layout, embedding, cell_unit, cell_ids=[c["cell_id"] for c in cells],
                          profiles=np.asarray(corrected)[units["index"].to_numpy()],
                          excluded_unit=[position.get(well_of[excluded_image(c)], -1) for c in excluded],
                          excluded_ids=excluded)
    output = run / "evaluation" if output is None else output
    fm.save(output / DRAWS, reading)
    theirs = None
    if control is not None and Path(control).resolve() != output.parent.resolve():
        theirs = fm.load(Path(control) / "evaluation" / DRAWS)
    full = fm.record(layout, reading, theirs, settings={"label": "gene (evaluation genes; scorer only)",
                                                        "unit": "well", "cells_and_profiles": UNIT_RULE})
    has = np.bincount(cell_unit[cell_unit >= 0], minlength=len(units)) > 0
    full["reported"] = {"uncorrected_contract_profile_mAP": {
        **fm.map_reading(layout, fm.unit_profiles(embedding, cell_unit, len(units)), has),
        "rule": "reported, not a verdict: mAP of the contract's unit profile (the mean of the L2-normalised "
                "uncorrected cells, L2-normalised) on the same splits and draws"}}
    return full


def markdown(result: dict) -> str:
    full = result["full_metrics"]
    lines = [f"# full_CPJUMP1 result ({result['arm']})", "",
             "## Verdict metrics (benchmarks/_shared/FULL_METRICS_CONTRACT.md)", "",
             f"Label = gene ({full['layout']['splits']['main']['n_genes']} evaluation genes), unit = well. "
             f"{UNIT_RULE}. {full['n_units_without_cells']} wells without delivered cells (misses); <= "
             f"{full['cell_cap']} cells per well in the cell-level probes, sampled from the delivered and the "
             f"excluded cells ({full['n_excluded_sampled']} of {full['n_excluded_cells']} excluded cells sampled, "
             "each a miss).", "", fm.markdown(full), "",
             "Reported, not a verdict: mAP of the contract's profile (mean of the L2-normalised uncorrected cells): "
             + "; ".join(f"{split} {fm._f(v['value'])} [{fm._f(v['ci95'][0])}, {fm._f(v['ci95'][1])}]"
                         for split, v in full["reported"]["uncorrected_contract_profile_mAP"].items()
                         if isinstance(v, dict)) + ".", "",
             "## Reported retrieval endpoints (secondary)", "",
             "| endpoint | mAP | MRR | top1 | genes | queries |", "| --- | --- | --- | --- | --- | --- |"]
    for name, m in result["metrics"].items():
        lines.append(f"| {name} | {fm._f(m['mAP'])} | {fm._f(m['MRR'])} | {fm._f(m['top1'])} | {m['n_genes']} | "
                     f"{m['n_queries']} |")
    return "\n".join(lines) + "\n"


def score(prep: Path, run: Path, control: Path | None = None, out: Path | None = None) -> dict:
    """Score a delivered run; `control`: the scored control run of the same preparation (for the verdicts), or the
    `out` its scoring went to; `out`: where RESULT.json / .md and evaluation/ go (default: the run itself; the post-hoc
    comparison scores into its own directory and never writes into the runs it compares)."""
    delivery = read_json(run / "DELIVERY.json")
    if delivery.get("status") != "complete" or delivery.get("smoke"):
        raise ValueError("run has not delivered a complete pipeline")
    for path, sha in delivery["artifacts"].items():
        if digest(within(run, path)) != sha:
            raise ValueError(f"delivered artifact changed: {path}")
    images = load_prepared(prep)
    if delivery["prepare_sha256"] != digest(prep / "PREPARE.json"):
        raise ValueError("delivery belongs to another preparation")
    prepared = read_json(prep / "PREPARE.json")
    if digest(prep / "private" / "labels.json") != prepared["labels_sha256"]:
        raise ValueError("private evaluation labels changed")
    labels = pd.DataFrame(read_json(prep / "private" / "labels.json"))
    joined = pd.DataFrame(images)[["image_id", "well"]].merge(labels, on="image_id", validate="one_to_one")
    keys = ["well", "gene", "guide", "role", "pert_type"]
    identities = joined[keys].drop_duplicates()
    if identities.well.duplicated().any():
        raise ValueError("site labels disagree within well")
    meta = pd.read_csv(run / "profiles.csv", keep_default_na=False).merge(identities, on="well", validate="one_to_one")
    if len(meta) != 500 or set(meta.well) != set(identities.well):
        raise ValueError("evaluation needs all 500 wells")
    out = run if out is None else out
    output = out / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for variant, name in (("uncorrected", "profiles.npy"), ("delivered", "corrected.npy")):
        x = np.load(run / name)
        if variant == "delivered":
            corrected = x
        if len(x) != len(meta):
            raise ValueError("profile rows disagree with metadata")
        for cross in (True, False):
            key = f"{variant}_{'cross_guide' if cross else 'same_guide'}_cross_plate"
            metrics[key], table = retrieval(meta, x, cross_guide=cross)
            expected = (92, 368) if cross else (110, 404)
            if (metrics[key]["n_genes"], metrics[key]["n_queries"]) != expected:
                raise ValueError(f"incomplete fixed evaluation design: {key}")
            table.to_csv(output / f"{key}.csv", index=False)
        if variant == "delivered":
            treatment = meta.pert_type == "trt"
            rows, profiles = [], []
            for gene, group in meta[treatment].groupby("gene", sort=True):
                # Balance plates within guide, then guides within gene.
                per_guide = [x[g.index].mean(axis=0) for _, g in group.groupby("guide")]
                rows.append({"gene": gene, "role": group.iloc[0].role, "guides": len(per_guide)})
                profiles.append(np.mean(per_guide, axis=0))
            genes = pd.DataFrame(rows)
            genes.to_csv(output / "genes.csv", index=False)
            pd.DataFrame(cosine(np.stack(profiles)), index=genes.gene, columns=genes.gene).to_csv(
                output / "gene_similarity_cosine.csv")
    result = {"status": "complete", "arm": delivery["arm"], "provenance": delivery["provenance"],
              "manifest_sha256": prepared["manifest_sha256"], "n_views": len(images), "n_wells": len(meta),
              "metrics": metrics, "primary": list(fm.METRICS),
              "retrieval_reading": "delivered_cross_guide_cross_plate",
              "full_metrics": full_metrics(run, meta, corrected, delivery, control, images, output),
              "batch_mode": "transductive: all unlabeled well profiles; calibration-only objective",
              "gene_similarity": "descriptive signed cosine; 20 calibration genes are not held-out evidence"}
    write_json(out / "RESULT.json", result)
    (out / "RESULT.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result
