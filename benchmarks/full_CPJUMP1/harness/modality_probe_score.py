"""Scoring for the ModalityProbe dataset (pipeline.py's MODALITY_PROBE): classifies the 6-way (cell_type,
perturbation) label instead of CPJUMP1's gene-knockout label -- this probe has no shared label across all
6 groups (CRISPR wells have a gene/guide, compound wells a compound identity, ORF wells a construct; none
of those exist for the other two modalities), so `benchmarks/full_CPJUMP1/harness/score.py`'s full_layout()
(gene, cross-plate/cross-guide splits) does not apply. Everything downstream of the Layout -- fm.evaluate(),
fm.record(), the cell-level kNN/logreg probes -- is the SAME generic machinery score.py uses; only the
label and the split are different (user decision 2026-11-01: "混在一起" -- one flat classification task
instead of six separately-scored sub-benchmarks).

The verdict metrics reduce to macro_f1 / balanced_accuracy of the 6-way classification, both the cell-level
probe (uncorrected embedding) and the unit-level probe (delivered, batch-"corrected" profile -- which per
MODALITY_PROBE's batch brief should equal the uncorrected one, since the batch stage is expected to ABSTAIN:
plate is confounded with the label here, so "correcting" across plates would erase the very thing being
classified).

User decision 2026-11-02: ALSO report a per-modality gene/compound retrieval mAP, real identity now fetched
for every group (`fetch_modality_probe_identities.py` -> `private/labels.json`'s `gene` (CRISPR/ORF) and
`compound`/`compound_target` (compound; exact compound barely replicates on a single plate -- 284/302
singleton by the JUMP-Target-1_compound plate's own design, so compound uses the coarser `compound_target`,
its primary mechanism) fields) -- see `retrieval_layout`/`retrieval_metrics` below. This is additive: the
6-way classification above is unchanged and still the primary verdict; retrieval is a second, independent
block in the same RESULT.json.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from benchmarks._shared import full_metrics as fm

from .data import digest, read_json, within, write_json
from .modality_probe_data import load_prepared

DRAWS = "full_metrics_draws.npz"
UNIT_RULE = ("cell-level probe: the uncorrected cell embedding (embedding.npy rows of cells.json); unit "
             "level: the delivered well profile (corrected.npy, L2-normalised) -- expected to equal the "
             "uncorrected profile since the batch stage should ABSTAIN on this dataset (see module "
             "docstring).")


def multi_modality_layout(meta: pd.DataFrame, seed: int = 0) -> tuple[fm.Layout, pd.DataFrame]:
    """The ModalityProbe layout: one unit per well (`meta` is already one row per well by the time it gets
    here -- profiles.csv pools any multi-site wells upstream in embed/batch/cluster; ModalityProbe/2 have 1
    site per well so wells==images there, ModalityProbe3 has 2), label = "{cell_type}_{perturbation}" (6
    classes). Split: a plain 2-fold stratified CV (sklearn StratifiedKFold, no grouping) -- CPJUMP1's
    grouped_folds() needs >= 2 GROUPS per class on a column distinct from the label (there it's guide,
    shared across both plates); here every (cell_type, perturbation) class lives on exactly one plate, so
    there is no such grouping column available, and none is needed -- every well is an independent sample
    (no replicate/guide structure to guard against leaking)."""
    # reset_index() (NOT drop=True): keeps meta's original row position as an "index" column, needed by
    # full_metrics()'s `corrected[units["index"]]` to realign the well-sorted units back to corrected.npy's
    # row order (profiles.csv/corrected.npy are written in meta's original, pre-sort order).
    units = meta.sort_values("well").reset_index()
    labels = (units.cell_type + "_" + units.perturbation).to_numpy()
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    main = [[(train, test)] for train, test in skf.split(np.zeros(len(labels)), labels)]
    describe = {"main": {"rule": "2-fold stratified CV (sklearn StratifiedKFold, random_state=0), no "
                                 "grouping -- every well is an independent sample",
                         "n_genes": int(len(set(labels)))}}
    return fm.Layout(units.well.tolist(), labels.tolist(), {"main": main}, describe), units


def retrieval_layout(meta: pd.DataFrame, seed: int = 0) -> tuple[fm.Layout | None, pd.DataFrame]:
    """Per-modality gene/compound retrieval: CRISPR/ORF retrieve by `gene`, compound by `compound_target`
    (the coarser primary-mechanism grouping -- exact `compound` is 284/302 singleton on a single plate, by
    the JUMP-Target-1_compound plate's own design, not a sampling gap). Label = f"{combo}:{identity}" --
    scoped within combo, so A549_crispr's HIF1A and U2OS_crispr's HIF1A (same JUMP-Target-1_crispr_platemap
    gene catalog, different cell type) are kept as distinct classes, never matched against each other: cell
    type is a real biological difference this task is not designed to erase.

    Wells without an identity (non-targeting guide, DMSO, or a compound JUMP-Target-1's metadata had no
    target_list for) and identities with only 1 well in this run's 600 (this probe's 100-well-per-group
    sample was drawn before identity was ever looked up -- user decision 2026-11-01 was not to resample it,
    see METADATA_SUMMARY.md) are dropped entirely, not scored as 0: fm._fold_map requires every test-side
    label to have >= 1 train-side unit, which a true singleton can never satisfy. Returns (None, empty
    DataFrame) if nothing survives the filter."""
    identity = np.where(meta.perturbation == "compound", meta.compound_target, meta.gene).astype(str)
    identity = np.where(identity == "nan", "", identity)
    label_all = meta.combo + ":" + identity
    eligible = (identity != "") & label_all.isin(label_all[identity != ""].value_counts()[lambda c: c >= 2].index)
    units = meta[eligible].sort_values("well").reset_index()
    if len(units) == 0:
        return None, units
    labels = (units.combo + ":" + np.where(units.perturbation == "compound", units.compound_target,
                                           units.gene).astype(str)).to_numpy()
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    main = [[(train, test)] for train, test in skf.split(np.zeros(len(labels)), labels)]
    describe = {"main": {"rule": "2-fold stratified CV (sklearn StratifiedKFold, random_state=0), label = "
                                 "combo:gene|compound_target; wells without an identity or whose identity "
                                 "is a singleton in this sample are excluded (see docstring)",
                         "n_genes": int(len(set(labels)))}}
    return fm.Layout(units.well.tolist(), labels.tolist(), {"main": main}, describe), units


def _evaluate_layout(run: Path, layout: fm.Layout, units: pd.DataFrame, corrected: np.ndarray, delivery: dict,
                     images: list[dict] | None) -> dict:
    """The shared machinery both full_metrics (6-way) and retrieval_metrics (gene/compound) run on top of
    their own Layout/units: delivered cells + embedding + corrected profiles, realigned to `units`' order."""
    for name in ("cells.json", "excluded_cells.json", "embedding.npy"):
        if name not in delivery["artifacts"]:
            raise ValueError(f"full metrics need the delivered {name}")
    cells = read_json(run / "cells.json")
    embedding = np.load(run / "embedding.npy", mmap_mode="r")
    if len(cells) != len(embedding):
        raise ValueError("cell embedding rows disagree with cells.json")
    position = {w: i for i, w in enumerate(units.well)}
    cell_unit = np.array([position.get(c["well"], -1) for c in cells], dtype=np.int64)
    excluded = [str(c) for c in read_json(run / "excluded_cells.json")]
    from .score import excluded_image
    well_of = {r["image_id"]: r["well"] for r in images or []}
    if any(excluded_image(c) not in well_of for c in excluded):
        raise ValueError("an excluded cell names an image outside the preparation")
    # excluded cells of a well this layout dropped (e.g. a singleton identity) have no `position` entry;
    # retrieval_metrics' units are a strict subset of full_metrics', so this filters them out rather than
    # erroring the way the full 600-well layout's own check above already guards against.
    kept_excluded = [c for c in excluded if well_of[excluded_image(c)] in position]
    return fm.evaluate(layout, embedding, cell_unit, cell_ids=[c["cell_id"] for c in cells],
                       profiles=np.asarray(corrected)[units["index"].to_numpy()],
                       excluded_unit=[position[well_of[excluded_image(c)]] for c in kept_excluded],
                       excluded_ids=kept_excluded)


def full_metrics(run: Path, meta: pd.DataFrame, corrected: np.ndarray, delivery: dict,
                 control: Path | None, images: list[dict] | None = None, output: Path | None = None) -> dict:
    layout, units = multi_modality_layout(meta)
    reading = _evaluate_layout(run, layout, units, corrected, delivery, images)
    output = run / "evaluation" if output is None else output
    fm.save(output / DRAWS, reading)
    theirs = None
    if control is not None and Path(control).resolve() != output.parent.resolve():
        theirs = fm.load(Path(control) / "evaluation" / DRAWS)
    full = fm.record(layout, reading, theirs, settings={
        "label": "cell_type_perturbation (6-way; scorer only)", "unit": "well",
        "cells_and_profiles": UNIT_RULE})
    return full


RETRIEVAL_DRAWS = "retrieval_metrics_draws.npz"


def retrieval_metrics(run: Path, meta: pd.DataFrame, corrected: np.ndarray, delivery: dict,
                      images: list[dict] | None = None, output: Path | None = None) -> dict | None:
    """Per-modality gene/compound retrieval (see retrieval_layout); None if this run's sample has no
    eligible (non-singleton, identified) well at all. No cross-arm `control` comparison -- unlike the 6-way
    classification, there's no single shared null/reference reading to diff against here (each arm's own
    eligible-well subset can differ slightly if an arm excludes a different handful of cells)."""
    layout, units = retrieval_layout(meta)
    if layout is None:
        return None
    reading = _evaluate_layout(run, layout, units, corrected, delivery, images)
    output = run / "evaluation" if output is None else output
    fm.save(output / RETRIEVAL_DRAWS, reading)
    return fm.record(layout, reading, None, settings={
        "label": "combo:gene|compound_target (retrieval; scorer only)", "unit": "well",
        "cells_and_profiles": UNIT_RULE,
        "n_wells_eligible": int(len(units)), "n_wells_total": int(len(meta))})


def markdown(result: dict) -> str:
    full = result["full_metrics"]
    lines = [f"# ModalityProbe result ({result['arm']})", "",
             "## Verdict metrics (6-way cell_type x perturbation classification)", "",
             f"Label = cell_type_perturbation ({full['layout']['splits']['main']['n_genes']} classes), unit = "
             f"well. {UNIT_RULE}. {full['n_units_without_cells']} wells without delivered cells (misses); <= "
             f"{full['cell_cap']} cells per well in the cell-level probes.", "", fm.markdown(full)]
    retrieval = result.get("retrieval_metrics")
    lines += ["", "## Per-modality gene/compound retrieval", ""]
    if retrieval is None:
        lines.append("No well in this run had both an identified, non-singleton gene/compound_target -- "
                     "nothing eligible to retrieve.")
    else:
        n_eligible = retrieval["settings"]["n_wells_eligible"]
        n_total = retrieval["settings"]["n_wells_total"]
        lines += [f"Label = combo:gene|compound_target ({retrieval['layout']['splits']['main']['n_genes']} "
                  f"classes), unit = well, {n_eligible} eligible of {n_total} total wells (the rest have no "
                  f"identity, or an identity that is a singleton in this sample). {UNIT_RULE}.",
                  "", fm.markdown(retrieval)]
    return "\n".join(lines) + "\n"


def score(prep: Path, run: Path, control: Path | None = None, out: Path | None = None) -> dict:
    """Score a delivered ModalityProbe run; mirrors full_CPJUMP1/harness/score.py's score() but with the
    6-way (cell_type, perturbation) label instead of gene, plus a second, additive per-modality
    gene/compound retrieval block (see module docstring)."""
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
        raise ValueError("private labels changed")
    labels = pd.DataFrame(read_json(prep / "private" / "labels.json")).fillna("")
    for col in ("gene", "compound", "compound_target", "guide", "broad_sample"):
        if col not in labels.columns:
            labels[col] = ""
    # labels.json carries its OWN "well" (the raw platemap well_position, or an anonymised id for the
    # reused A549_crispr rows) -- drop it before merging so images.json's "well" (the one profiles.csv
    # actually keys on) survives the merge unsuffixed.
    label_cols = ["image_id", "cell_type", "perturbation", "combo", "gene", "compound_target"]
    joined = pd.DataFrame(images)[["image_id", "well"]].merge(labels[label_cols], on="image_id",
                                                               validate="one_to_one")
    keys = ["well", "cell_type", "perturbation", "combo", "gene", "compound_target"]
    identities = joined[keys].drop_duplicates()
    if identities.well.duplicated().any():
        raise ValueError("site labels disagree within well")
    meta = pd.read_csv(run / "profiles.csv", keep_default_na=False).merge(identities, on="well", validate="one_to_one")
    if len(meta) != len(identities) or set(meta.well) != set(identities.well):
        raise ValueError(f"evaluation needs all {len(identities)} wells")
    out = run if out is None else out
    output = out / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    corrected = np.load(run / "corrected.npy")
    if len(corrected) != len(meta):
        raise ValueError("profile rows disagree with metadata")
    result = {"status": "complete", "arm": delivery["arm"], "provenance": delivery["provenance"],
              "n_views": len(images), "n_wells": len(meta), "primary": list(fm.METRICS),
              "full_metrics": full_metrics(run, meta, corrected, delivery, control, images, output),
              "retrieval_metrics": retrieval_metrics(run, meta, corrected, delivery, images, output),
              "batch_mode": "expected ABSTAIN: plate is confounded with the 6-way label on this dataset"}
    write_json(out / "RESULT.json", result)
    (out / "RESULT.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result


def score_no_compound(prep: Path, run: Path, control: Path | None = None, out: Path | None = None) -> dict:
    """Rescore an already-delivered ModalityProbe3 run on just the 4 non-compound groups (A549_crispr,
    A549_orf, U2OS_crispr, U2OS_orf) -- drops both compound groups entirely, not just their identity
    labels, so the 6-way classification becomes a 4-way one and retrieval is pure gene identity (no
    compound_target branch left to take). Reuses embed/batch/cluster's EXISTING deliverables unchanged --
    no retraining, no re-embedding: excluding compound is a pure scoring-time filter on which wells count,
    same embedding space, same corrected.npy. `meta`'s row labels (pandas index) are preserved through the
    boolean filter below (no reset_index), so multi_modality_layout/retrieval_layout's own
    `units.reset_index()` still yields the ORIGINAL meta position, keeping `corrected[units["index"]]`
    correctly aligned to corrected.npy's row order without any change to those functions.

    `control`, if given, must point at CONTROL_RUN/no_compound (this function's own `out` for the control
    arm), not the control arm's root -- the original 6-way run's bootstrap draws are not comparable to a
    4-way rescore."""
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
        raise ValueError("private labels changed")
    labels = pd.DataFrame(read_json(prep / "private" / "labels.json")).fillna("")
    for col in ("gene", "compound", "compound_target", "guide", "broad_sample"):
        if col not in labels.columns:
            labels[col] = ""
    label_cols = ["image_id", "cell_type", "perturbation", "combo", "gene", "compound_target"]
    joined = pd.DataFrame(images)[["image_id", "well"]].merge(labels[label_cols], on="image_id",
                                                               validate="one_to_one")
    keys = ["well", "cell_type", "perturbation", "combo", "gene", "compound_target"]
    identities = joined[keys].drop_duplicates()
    if identities.well.duplicated().any():
        raise ValueError("site labels disagree within well")
    meta = pd.read_csv(run / "profiles.csv", keep_default_na=False).merge(identities, on="well", validate="one_to_one")
    if len(meta) != len(identities) or set(meta.well) != set(identities.well):
        raise ValueError(f"evaluation needs all {len(identities)} wells")
    corrected = np.load(run / "corrected.npy")
    if len(corrected) != len(meta):
        raise ValueError("profile rows disagree with metadata")
    meta = meta[meta.perturbation != "compound"]  # keep original index -- see docstring
    if meta.combo.str.contains("compound").any():
        raise ValueError("compound rows survived the filter")
    out = (run / "no_compound") if out is None else out
    output = out / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "complete", "arm": delivery["arm"], "provenance": delivery["provenance"],
              "n_views": len(images), "n_wells": len(meta), "primary": list(fm.METRICS),
              "excluded_groups": ["A549_compound", "U2OS_compound"],
              "full_metrics": full_metrics(run, meta, corrected, delivery, control, images, output),
              "retrieval_metrics": retrieval_metrics(run, meta, corrected, delivery, images, output),
              "batch_mode": "expected ABSTAIN: plate is confounded with the 4-way label on this dataset"}
    write_json(out / "RESULT.json", result)
    (out / "RESULT.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    return result
