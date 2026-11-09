"""Score stage (GPU, then CPU): well mAP of every arm on every FULL cell -- the only stage that reads labels.

Arms, each an embedding of every FULL cell made by the analyzer (`embed`, its UMAP / Leiden step skipped:
`embedding_tool`; <root>/score_cache/embed/<arm>-<digest>/, reused when complete):
  frozen          adaptation_depth="zero_shot"
  dinov3          the control's plain DINOv3 fine-tune (A23; dinov3_s<k>: an extra seed k): embed_only with its
                  resume_best (its lowest-training-loss epoch on the train wells; it is embedded on every cell)
  shape           the SHAPE arm's delivery (submission/decision.json, final or provisional -- RESULT says which):
                  the call's resume_best, else frozen
  shape_top       analysis only: the finished call with the highest val proxy, the guard ignored
  react           the ReAct arm (react_arm.py on the shared runner, benchmarks/_shared/REACT_ARM_CONTRACT.md,
                  amendment A14), only when score is given its run directory (`react_run`): its delivered call's
                  resume_best, or frozen (the default call, trial 0)
  llm             the LLM-written hand-crafted features (contract amendment A5, llm_arms.py), only when score is given
                  its run directory (`llm_run`): the run's own embeddings.npy (no analyzer), scored as an external
                  feature arm
                  For both: a failed run is reported, not scored; a run made with other code or another prep is refused
                  (ARM_STALE; llm: its own design, `llm_arms.llm_design_problems`; react: react_arm.py and the
                  shared runner, benchmarks/_shared/react_arm.py)
embed_only applies the stored intensity reference of a per_plate state and reads no split value, so the held-out
and unused cells need no split label. The external arms (external.py, contract amendment A2: openphenom,
cellprofiler) are read raw from <root>/external/<arm>/ when the external stage made them, else skipped with a note.
The empirical random arm of A7 is no longer scored (user decision 2026-10-11, A10 addendum): its well mAP matched
`chance` (0.038 / 0.046 vs 0.038 / 0.050 at 074711bf), so `chance` alone is reported; `random_vectors` stays for the
test that checks `chance` against seeded standard-normal vectors.
Each table also carries a `chance` row (A7, correction): per group the exact expected AP of a random ranking per query
(`expected_random_ap`), averaged like the well mAP (the audit groups' pooled like their AP); RESULT.json also keeps
`base_rate`, n_pos / n_cand per query averaged the same way (the well mAP's own `chance`, which a random ranking's AP
exceeds when a query has few positives). Both depend only on the labels and the group layout (`well_chance`), so they
are taken from the scored arms, which must all agree (`compare`, CHANCE_MISMATCH otherwise).

Post-processing (contract amendment A3, `post_process`, label-free, the same code for every arm): every arm is scored
twice. native: the analyzer arms and openphenom as embedded (raw); cellprofiler and llm (hand-crafted features) with
one global `centre_scale` over all cells (raw hand-crafted features are not commensurable without some scaling).
matched: the same per-plate `centre_scale` (fitted on all cells of each plate, transductive) on EVERY arm, the
analyzer arms included. No PCA in either table (the OpenPhenom card's PCA-CenterScale is a fitted rotation; it is
dropped for every arm rather than applied to one). The pre-registered reading is read on the native table; the
external comparisons on the matched one. An llm output that leaves a well profile of zero norm after either
post-processing is reported as unscorable (`zero_norm_wells`) and not scored.

Well mAP (a port of the task-116 harness): a well's profile is the mean of its L2-normalised
cells, L2-normalised; each well ranks the other wells of its group by cosine, positives = the wells of its siRNA; AP
per query, mean per siRNA, mean over siRNA. Groups: seen (the train, val and unused wells of the seen experiments;
the primary endpoint), heldout (the held-out experiments), run_prep (train + val: the cells the tool saw); the audit
groups trained_wells / unused_wells / val_wells hold the per-well AP of `seen` on those wells, pooled the same way.
Comparisons, paired over siRNA (cluster bootstrap, +-2 SE): shape vs the per-well mean of the default seeds (the
dinov3 control arms, A23; the RESULT keys keep the names default_arms / default_mean / best_default_seed), vs the
best default seed (per group), vs frozen, vs each external arm scored; the default mean vs frozen; frozen vs each
external arm scored; shape_top the same (analysis only). With the LLM arms: llm (A5) shape vs llm, llm vs frozen, llm
vs the default mean, llm vs cellprofiler (read on the matched table, as the external comparisons); react (A4) shape vs
react, react vs the default mean, react vs frozen (read on the native table, like the reading; react is an analyzer
arm, raw in the native table); and the efficiency of shape, llm and react (tool calls, wall hours, reasoner tokens).
Diagnostic (analysis only, never fed back): per trial of the SHAPE arm, from the call's own embedding of the run-prep
cells, its parent and lever changes, its val proxy on the comparison draws (quant.sibling_site_map at
quant.FRESH_SEED, the verifier's draws) with the paired delta vs frozen (trial 0) and vs its parent, its run-prep well
mAP with the label band vs frozen, and whether quant.beats(call, frozen) agreed with that band; Spearman of the val
proxy vs the run-prep well mAP over the calls (descriptive).
Pre-registered reading: shape - default mean > 2 SE on `seen` = SHAPE's loop beats its defaults; `heldout` is read
the same way; a frozen delivery is reported with its case (FROZEN_CASES).
Negative controls (`NEGATIVE_CONTROL`, `scored_cells`): the wells whose siRNA is "EMPTY" (RxRx1's negative_control
wells, one per plate, in a prep of every well type: the full-set v4 preps) leave the well mAP entirely -- neither
queries nor positives nor negative candidates -- and the A9 probe (secondary.py). EMPTY is no perturbation: as a
query its positives would be the other untreated wells (one per plate), a retrieval of the untreated baseline rather
than of an siRNA's phenotype; kept only as negative candidates, a condition with no phenotype would still enter every
query's ranks. The standard replicate-retrieval mAP scores the perturbations only (the controls are the reference of
a separate activity mAP), so they are dropped from both sides.
They stay in every label-free step (the per-plate centre_scale of the matched table, every embedding, the tool's
inputs): a label decides only what is scored. A prep without EMPTY wells (the v3 preps, positive controls only) is
scored exactly as before; with them RESULT.json gains `negative_controls` (the excluded wells and cells).
The A9 probe's cell unit is skipped for the full-set preps (`PROBE_SKIPPED`, `probe_units`: the well unit only).
Secondary metrics (contract amendment A9, secondary.py, scoring-only, never the reading): macro F1 and balanced
accuracy of a siRNA probe across experiments for the A9 arms in both tables, computed after every score above from
the same post-processed vectors and the labels read here (RESULT.json `secondary`, RESULT.md "Secondary metrics").
Per-arm score cache (`ArmScores`): everything `final` needs from one arm alone -- per table its post-processed well-mAP
evaluation (`evaluate`, the per-well APs the paired comparisons use included), for the A9 arms the probe's runs
(`secondary.Probe.run`: point scores, the scores under the fixed bootstrap draws, the logreg fit), for llm its
zero-norm well count -- is computed once and kept in <root>/score_cache/<sha256 of its key>.json, shared across runs.
The key: the arm's content (an analyzer arm: its kwargs, the sha256 of its checkpoint file and the analyzer's code,
`control.provenance`; an external or llm arm: its embeddings' sha256), the prep (cells, labels and images.npy sha256),
the scoring code (`scoring_code`: score.py, secondary.py, numpy, scikit-learn), the table and its post-processing. A
file under another key is recomputed, never reused. `score_arm` (stage score-arm) scores one arm into the cache as
soon as it exists; `final` reads the cache, computes a missing arm, and assembles the comparisons and the RESULT.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

QUESTION = ("Does SHAPE's agentic SSL VRR (delivering the agent's call or frozen) beat the same analyzer called once "
            "at its defaults on RxRx1 well mAP, with no labels in any decision?")
GROUPS = {"seen": ("train", "val", "unused"), "heldout": ("heldout",), "run_prep": ("train", "val")}
AUDIT_GROUPS = {"trained_wells": "train", "unused_wells": "unused", "val_wells": "val"}
READ = ("seen", "heldout")              # the groups of the pre-registered reading
EMBED_KEYS = ("cell_id", "crop_id", "image_name", "group", "plate", "experiment", "well", "site")
# why frozen was delivered: the verifier's decision case (or the resolution's reason) in the reading's words
FROZEN_CASES = {
    "not_measurable": "gate failed", "no_tuning_needed": "agent declined in round 1",
    "agent_kept_input": "agent kept frozen at the final choice",
    "final_choice_inadmissible": "agent's final choice inadmissible after its retry",
    "no_admissible_call": "guard declined every call", "every_call_failed": "no call finished",
    "no_call_ran": "no call finished", "no_call_finished": "no call finished",
    "initial_trial_failed": "trial 0 failed", "provisional": "provisional record only",
}
NEGATIVE_CONTROL = "EMPTY"              # the siRNA of RxRx1's negative_control wells: never scored (module docstring)
LLM_ARMS = ("llm", "react")             # the LLM comparison arms (amendments A4, A5)
LLM_READ = {"llm": "matched", "react": "native"}   # the table each one's comparisons are read on (A5: llm as external)
TABLES = ("native", "matched")          # the two post-processings every arm is scored with (`post_process`)
NATIVE_GLOBAL = ("cellprofiler", "llm")  # the arms whose native table is one global centre_scale (else raw)
MAD_TO_SD, SCALE_FLOOR, CLIP = 1.4826, 1e-6, 10.0
POSTPROCESS = {
    "native": "the analyzer arms (react among them) and openphenom raw; cellprofiler and llm one global centre_scale "
              "over all cells",
    "matched": "every arm: per plate (all its cells, label-free), each dimension minus its median, divided by "
               f"{MAD_TO_SD} x its MAD (its SD when that is below {SCALE_FLOOR}, 1 when both are), a non-finite value "
               f"set to 0, clipped at +-{CLIP}",
    "pca": "none in either table (the OpenPhenom card's PCA-CenterScale is dropped for every arm)",
    "read": {"reading": "native", "reading_vs_frozen": "both", "external_comparisons": "matched",
             "llm_comparisons": LLM_READ["llm"], "react_comparisons": LLM_READ["react"]},
    "gap": "an arm's native - matched gap is what the label-free per-plate centre-scale alone does to it (it removes "
           "each plate's location and per-dimension spread before the well profiles); a comparison whose sign or "
           "significance differs between the tables rests on that post-processing, not on the representation",
}


# ---- well mAP: a port of the task-116 harness, agentic_rxrx1/harness/metric.py @ 84ec172 ----------------------

QUERY_CHUNK = 256
RANDOM_DIM, RANDOM_SEED = 384, 0   # seeded standard-normal vectors: the test of `chance` (not a scored arm)


def _codes(values: Sequence[Any], allow_none: bool) -> np.ndarray:
    lookup: dict[Any, int] = {}
    codes = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if value is None:
            if not allow_none:
                raise ValueError("None label")
            codes[index] = -1
        else:
            codes[index] = lookup.setdefault(value, len(lookup))
    return codes


def l2(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class ConsistencyResult:
    value: float
    chance: float
    per_unit: dict[str, float]
    n_queries: int
    n_groups_scored: int


def consistency_map(embeddings, groups, units, sources, wells) -> ConsistencyResult | None:
    """Group-retrieval mAP excluding same-source (and same-well) candidates."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("embeddings must be a finite 2-D array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("zero-norm embedding rows")
    x = x / norms
    n = x.shape[0]
    group_codes = _codes(list(groups), False)
    source_codes = _codes(list(sources), False)
    well_codes = _codes(list(wells), True)
    units = list(units)
    ap = np.full(n, np.nan)
    chance = np.full(n, np.nan)
    for start in range(0, n, QUERY_CHUNK):
        rows = np.arange(start, min(start + QUERY_CHUNK, n))
        sim = x[rows] @ x.T
        candidate = source_codes[None, :] != source_codes[rows, None]
        both_wells = (well_codes[None, :] >= 0) & (well_codes[rows, None] >= 0)
        candidate &= ~(both_wells & (well_codes[None, :] == well_codes[rows, None]))
        candidate[np.arange(rows.size), rows] = False
        positive = candidate & (group_codes[None, :] == group_codes[rows, None])
        n_cand = candidate.sum(axis=1)
        n_pos = positive.sum(axis=1)
        scored = (n_pos > 0) & (n_pos < n_cand)
        if not scored.any():
            continue
        order = np.argsort(-sim[scored], axis=1, kind="stable")
        cand_sorted = np.take_along_axis(candidate[scored], order, axis=1)
        pos_sorted = np.take_along_axis(positive[scored], order, axis=1)
        rank = np.cumsum(cand_sorted, axis=1)
        hits = np.cumsum(pos_sorted, axis=1)
        precision = np.where(pos_sorted, hits / np.maximum(rank, 1), 0.0)
        ap[rows[scored]] = precision.sum(axis=1) / n_pos[scored]
        chance[rows[scored]] = n_pos[scored] / n_cand[scored]

    scored_rows = np.flatnonzero(~np.isnan(ap))
    cells: dict[tuple[Any, str], list[int]] = {}
    unit_rows: dict[str, list[int]] = {}
    for row in scored_rows:
        unit = str(units[row])
        cells.setdefault((group_codes[row], unit), []).append(row)
        unit_rows.setdefault(unit, []).append(row)
    group_ap: dict[Any, list[float]] = {}
    group_chance: dict[Any, list[float]] = {}
    for (group, _unit), members in cells.items():
        group_ap.setdefault(group, []).append(float(ap[members].mean()))
        group_chance.setdefault(group, []).append(float(chance[members].mean()))
    if len(group_ap) < 2:
        return None
    return ConsistencyResult(
        value=float(np.mean([np.mean(v) for v in group_ap.values()])),
        chance=float(np.mean([np.mean(v) for v in group_chance.values()])),
        per_unit={unit: float(ap[r].mean()) for unit, r in unit_rows.items()},
        n_queries=int(scored_rows.size),
        n_groups_scored=len(group_ap),
    )


def _mean_by(key: np.ndarray, vectors: np.ndarray, *columns: np.ndarray):
    """(the distinct keys, sorted; per key the L2-normalised mean of its rows of `vectors`; per key the value of each
    of `columns` at its first row). The rows of every key come from one stable sort (ascending, as a boolean mask
    takes them), so each mean is `vectors[key == value].mean(axis=0)` of the same rows in the same layout (the same
    floats), without one pass over every row per key (wells x cells: hours at the full-set sizes)."""
    vectors = np.asarray(vectors)
    order, inverse = np.unique(np.asarray(key), return_inverse=True)
    inverse = inverse.ravel()
    rows = np.argsort(inverse, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(order)))]
    means = np.empty((len(order), vectors.shape[1]), dtype=np.float64)
    for index in range(len(order)):
        means[index] = vectors[rows[bounds[index]:bounds[index + 1]]].mean(axis=0)
    first = rows[bounds[:-1]]
    return order, l2(means), [np.array(np.asarray(column)[first].tolist()) for column in columns]


def zero_norm_wells(vectors: np.ndarray, well: np.ndarray) -> int:
    """The wells whose profile (the mean of the L2-normalised cells, as `well_map` builds it) has zero norm: the well
    mAP cannot rank them (`consistency_map` refuses zero-norm rows)."""
    _, means, _ = _mean_by(np.asarray(well), l2(vectors))
    return int((np.linalg.norm(means, axis=1) <= 0).sum())


def well_map(vectors: np.ndarray, group: np.ndarray, well: np.ndarray) -> dict[str, Any]:
    """metric.score_level at level "well": the one-step mean of L2-normalised cells per plate-qualified well, then L2;
    each well retrieves the wells of its group (siRNA) among the other wells."""
    vectors = l2(vectors)
    keys, means, (groups,) = _mean_by(well, vectors, group)
    result = consistency_map(means, list(groups), [str(u) for u in keys], list(keys), list(keys))
    unit_group = {str(u): groups[i] for i, u in enumerate(keys)}
    per_unit = dict(result.per_unit) if result else {}
    return {
        "value": None if result is None else result.value,
        "chance": None if result is None else result.chance,
        "n_rows": int(len(means)),
        "n_groups": int(len(set(groups.tolist()))),
        "n_groups_scored": 0 if result is None else result.n_groups_scored,
        "per_unit": per_unit,
        "unit_group": {u: str(unit_group[u]) for u in per_unit},
    }


def expected_random_ap(n_pos: int, n_cand: int) -> float:
    """The exact expected AP of a uniformly random ranking of `n_cand` candidates of which `n_pos` are positive
    (amendment A7, correction): with R = n_pos, N = n_cand and H_N the N-th harmonic number,
    E[AP] = (1/N) [(R-1)/(N-1) (N - H_N) + H_N]; N == 1 (hence R == 1): 1.0. ValueError unless 1 <= R <= N."""
    if not 1 <= n_pos <= n_cand:
        raise ValueError(f"n_pos {n_pos} must lie in [1, n_cand {n_cand}]")
    if n_cand == 1:
        return 1.0
    harmonic = float(np.sum(1.0 / np.arange(1, n_cand + 1)))
    return ((n_pos - 1) / (n_cand - 1) * (n_cand - harmonic) + harmonic) / n_cand


def well_chance(group: np.ndarray, well: np.ndarray) -> dict[str, dict[str, float]]:
    """Per well, its query's random baselines in `well_map` (A7): `chance` = the exact expected AP of a random ranking
    (`expected_random_ap`) and `base_rate` = n_pos / n_cand, with every other well of the group a candidate and the
    wells of its siRNA the positives, for the wells `consistency_map` scores (0 < n_pos < n_cand). From the labels
    and the layout only; `well_map` is left as ported."""
    keys, index = np.unique(np.asarray(well), return_index=True)
    labels = np.asarray(group)[index]
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n_pos, n_cand = counts[inverse] - 1, len(keys) - 1
    scored = [(str(u), int(p)) for u, p in zip(keys, n_pos) if 0 < p < n_cand]
    exact = {p: expected_random_ap(p, n_cand) for p in {p for _, p in scored}}
    return {"chance": {u: exact[p] for u, p in scored}, "base_rate": {u: float(p / n_cand) for u, p in scored}}


def paired_band(baseline: dict[str, float], current: dict[str, float],
                unit_group: dict[str, str], *, resamples: int = 2000, seed: int = 0) -> dict:
    """Paired cluster bootstrap over siRNA groups; the band is +-2 SE of the mean delta."""
    units = sorted(set(baseline) & set(current))
    if not units:
        return {"delta": None, "band_2se": None, "n_units_paired": 0, "significant": None}
    delta = np.array([current[u] - baseline[u] for u in units])
    groups = np.array([unit_group[u] for u in units])
    unique = np.unique(groups)
    index = {name: np.flatnonzero(groups == name) for name in unique}
    rng = np.random.default_rng(seed)
    draws = [delta[np.concatenate([index[g] for g in rng.choice(unique, len(unique), True)])].mean()
             for _ in range(resamples)]
    band = float(2 * np.std(draws, ddof=1))
    mean = float(delta.mean())
    return {"delta": mean, "band_2se": band, "n_units_paired": len(delta),
            "significant": bool(abs(mean) > band)}


def pooled(per_unit: dict[str, float], unit_group: dict[str, str]) -> float | None:
    """Nested mean as in metric.consistency_map: wells -> siRNA -> mean over siRNA (task 116 states.py)."""
    by_group: dict[str, list[float]] = {}
    for unit, ap in per_unit.items():
        by_group.setdefault(unit_group[unit], []).append(ap)
    return float(np.mean([np.mean(v) for v in by_group.values()])) if len(by_group) >= 2 else None


def verdict(band: dict | None) -> str:
    """The pre-registered reading of shape vs the default mean (task 116 score.py's verdict rule)."""
    if not band or band.get("delta") is None:
        return "UNSCOREABLE"
    if band["significant"]:
        return "SHAPE_BEATS_DEFAULT" if band["delta"] > 0 else "SHAPE_WORSE_THAN_DEFAULT"
    return "NO_RESOLVABLE_DIFFERENCE"


def verdict_vs_frozen(band: dict | None) -> str:
    """The same rule for shape vs the frozen backbone (the only reading of a control without default arms)."""
    if not band or band.get("delta") is None:
        return "UNSCOREABLE"
    if band["significant"]:
        return "SHAPE_BEATS_FROZEN" if band["delta"] > 0 else "SHAPE_WORSE_THAN_FROZEN"
    return "NO_RESOLVABLE_DIFFERENCE"


def _band(groups: dict, group: str, name: str) -> dict | None:
    """The comparison `name` of `group` in a compare() result, else None."""
    return ((groups.get(group) or {}).get("comparisons") or {}).get(name)


# ---- post-processing (contract amendment A3): label-free, the same code for every arm ---------------------------

def centre_scale(vectors: np.ndarray, plates, *, clip: float = CLIP, floor: float = SCALE_FLOOR) -> np.ndarray:
    """Robust centre-scale per plate (all cells of each plate, no label; one value for every row = one global
    scaling): each column minus its median, divided by MAD_TO_SD x its MAD (its SD when that is below `floor`, 1 when
    both are), a non-finite value set to 0 (the median), clipped at +-clip; float32."""
    import warnings

    x = np.asarray(vectors, dtype=np.float64)
    plates = np.asarray(plates)
    out = np.empty(x.shape, dtype=np.float32)
    for plate in np.unique(plates):
        member = plates == plate
        block = x[member]
        with np.errstate(all="ignore"), warnings.catch_warnings():     # a column all NaN on a plate: set to 0
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(block, axis=0)
            scale = MAD_TO_SD * np.nanmedian(np.abs(block - median), axis=0)
            scale = np.where(scale >= floor, scale, np.nanstd(block, axis=0))
            scale = np.where(np.isfinite(scale) & (scale >= floor), scale, 1.0)
            z = (block - median) / scale
        out[member] = np.clip(np.where(np.isfinite(z), z, 0.0), -clip, clip)
    return out


def post_process(arm: str, vectors: np.ndarray, plates, table: str) -> np.ndarray:
    """An arm's vectors as scored in `table`: native = as embedded (one global centre_scale for NATIVE_GLOBAL);
    matched = the per-plate centre_scale, identical for every arm."""
    if table == "matched":
        return centre_scale(vectors, plates)
    if table != "native":
        raise ValueError(f"table must be one of {TABLES}, not {table!r}")
    if arm in NATIVE_GLOBAL:
        return centre_scale(vectors, np.zeros(len(vectors), dtype=np.int64))
    return np.asarray(vectors)


def scored_cells(sirna: np.ndarray) -> np.ndarray:
    """The cells the well mAP and the probe score: every cell but the negative controls (NEGATIVE_CONTROL)."""
    return np.asarray(sirna) != NEGATIVE_CONTROL


def negative_controls(cells, sirna: np.ndarray) -> dict | None:
    """RESULT.json `negative_controls`: the cells and wells left out of the scoring (`scored_cells`), per group; None
    when the prep has none (the v3 preps: RESULT.json unchanged)."""
    left_out = ~scored_cells(sirna)
    if not left_out.any():
        return None
    split, well = cells.split.to_numpy(), cells.well.to_numpy()
    return {"sirna": NEGATIVE_CONTROL, "cells": int(left_out.sum()), "wells": int(len(set(well[left_out]))),
            "wells_per_group": {name: int(len(set(well[left_out & np.isin(split, splits)])))
                                for name, splits in GROUPS.items()},
            "rule": "neither query, positive nor candidate in the well mAP; not in the A9 probe; in every label-free "
                    "step (embedding, post-processing)"}


# the A9 probe's skipped units per prep set (`probe_units`; user decision 2026-10-13): the full-set preps (v4 and its
# well-level resplit v5) probe the well unit only
PROBE_SKIPPED = {"full_v4": {"cell": (
    "the full-set preps hold up to 1.42 M cells (HUVEC: ~1.1 M held-out cells against ~0.3 M seen cells, ~1,138 "
    "classes); a cell-level kNN and logistic regression on every cell would take days per arm and table (a kNN chunk "
    "of 1,024 cells against 100,000 took ~12 s on 16 cores), so only the well unit is probed")},
                 "full_v5": {"cell": (
    "the full-set preps hold up to 1.42 M cells (full_v5, the cells of full_v4: 90 % of every plate's wells held "
    "out, ~1,138 classes); a cell-level kNN and logistic regression on every cell would take days per arm and table "
    "(a kNN chunk of 1,024 cells against 100,000 took ~12 s on 16 cores), so only the well unit is probed")}}


def prep_set(summary: dict) -> str:
    """The prep set (config.py PREP_SETS) of the prep whose PREPARE.json is `summary`: the set naming its FULL prep;
    v3 for any other prep (the task-116 preps, a test fixture)."""
    from benchmarks.representation_RxRx1.runners.config import PREP_SETS

    name = Path(((summary.get("sources") or {}).get("full") or {}).get("path") or "").name
    return next((key for key, preps in PREP_SETS.items() if any(name == full for full, _ in preps.values())), "v3")


def probe_units(summary: dict) -> tuple[tuple[str, ...], dict[str, str]]:
    """(the A9 probe's units, {skipped unit: why}) for the prep of `summary` (PROBE_SKIPPED)."""
    from benchmarks.representation_RxRx1.runners import secondary

    skipped = PROBE_SKIPPED.get(prep_set(summary), {})
    return tuple(u for u in secondary.UNITS if u not in skipped), dict(skipped)


# ---- the arms ------------------------------------------------------------------------------------------------

def read_labels(root: Path, cells, summary: dict) -> np.ndarray:
    """The siRNA of every FULL cell, in `cells` order, from the private labels file."""
    import pandas as pd

    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    path = root / "private" / "labels.csv"
    if sha256_file(path) != summary["labels_sha256"]:
        raise SystemExit(f"INTEGRITY_FAILED: {path} changed since the prepare stage")
    sirna = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("cell_id").sirna.reindex(cells.cell_id)
    if sirna.isna().any():
        raise SystemExit(f"LABELS_INCOMPLETE: {int(sirna.isna().sum())} cells have no label in {path}")
    return sirna.to_numpy().astype(str)


def aligned(adata, cell_ids) -> np.ndarray:
    """The rows of adata.X in the order of `cell_ids` (the analyzer names its obs by cell id)."""
    import pandas as pd

    index = pd.Index(adata.obs_names.astype(str)).get_indexer(list(cell_ids))
    if (index < 0).any():
        raise SystemExit(f"EMBED_INCOMPLETE: {int((index < 0).sum())} cells are missing from {adata.filename}")
    return np.asarray(adata.X)[index]


def _no_clustering(*args, **kwargs) -> None:
    return None


def embedding_tool():
    """The analyzer as scoring calls it: `control.analyzer()` with its UMAP / Leiden step
    (`_compute_umap_and_clustering`, run after the embedding is in adata.X and never changing it) skipped on this
    instance, as the analyzer's own tests do; scoring reads adata.X only, and tool.py is unchanged."""
    from benchmarks.representation_RxRx1.runners import control

    tool = control.analyzer()
    tool._compute_umap_and_clustering = _no_clustering
    return tool


def embed(summary: dict, cells, arm_dir: Path, kwargs: dict) -> np.ndarray:
    """Every FULL cell's vector (rows in `cells` order) from one analyzer call with `kwargs` (zero_shot, or embed_only
    with a checkpoint) on the prep's packed array (PREPARE.json `summary`), without the call's clustering
    (`embedding_tool`); an earlier call with the same kwargs in `arm_dir` (EMBED.json) is reused."""
    import anndata
    from shape.utils.schemas import is_failed_result

    from benchmarks.representation_RxRx1.runners import control
    from benchmarks.representation_RxRx1.runners.prepare import SPLITS, tool_input

    record_path = arm_dir / "EMBED.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else None
    if record is not None and record["kwargs"] != kwargs:
        raise SystemExit(f"{arm_dir} holds another embedding ({record['kwargs']}); move it aside")
    if record is None:
        crop_array, meta = tool_input(summary, cells, splits=SPLITS, keys=EMBED_KEYS)
        control.fs_retry(arm_dir.mkdir, parents=True, exist_ok=True)
        started = time.time()
        result = embedding_tool().execute(crop_array=crop_array, cell_metadata=meta,
                                          query_cache_dir=str(arm_dir), **kwargs)
        if is_failed_result(result) or not result.get("adata_path"):
            raise SystemExit(f"EMBED_FAILED: {arm_dir.name}: {result.get('error') or 'no h5ad'}")
        record = {"kwargs": kwargs, "adata_path": result["adata_path"], "n_cells": len(meta),
                  "seconds": round(time.time() - started, 1)}
        control.write_json(record_path, record)
        print(f"embed {arm_dir.name}: {len(meta)} cells, {record['seconds']} s", flush=True)
    return aligned(anndata.read_h5ad(record["adata_path"]), cells.cell_id)


def evaluate(vectors: np.ndarray, cells, sirna: np.ndarray) -> dict:
    """Per group (GROUPS): the well mAP of its cells; per audit group: the `seen` per-well AP on its wells, pooled.
    Each also carries (A7) `chance`, the exact expected AP of a random ranking, and `base_rate`, n_pos / n_cand (the
    well mAP's own `chance`, unchanged), per well from `well_chance`, pooled like the AP. The negative-control cells
    are not scored (`scored_cells`)."""
    split, well = cells.split.to_numpy(), cells.well.to_numpy()
    keep = scored_cells(sirna)
    out: dict[str, Any] = {}
    per_well: dict[str, dict] = {}
    for name, splits in GROUPS.items():
        mask = np.isin(split, splits) & keep
        entry = well_map(vectors[mask], sirna[mask], well[mask]) if mask.any() else None
        if entry is not None:
            entry["base_rate"] = entry["chance"]
            if entry["base_rate"] is not None:
                per_well[name] = well_chance(sirna[mask], well[mask])
                base = per_well[name]["base_rate"]
                if set(base) != set(entry["per_unit"]) or not math.isclose(pooled(base, entry["unit_group"]),
                                                                           entry["base_rate"], rel_tol=1e-9):
                    raise SystemExit(f"CHANCE_MISMATCH: the per-well base rate does not reproduce the {name} group's")
                entry["chance"] = pooled(per_well[name]["chance"], entry["unit_group"])
        out[name] = entry
    seen, chance = out["seen"], per_well.get("seen") or {"chance": {}, "base_rate": {}}
    for name, s in AUDIT_GROUPS.items():
        keep = set(well[split == s])
        units = {u: v for u, v in seen["per_unit"].items() if u in keep} if seen else {}
        out[name] = {"value": pooled(units, seen["unit_group"]) if units else None,
                     **{key: pooled({u: chance[key][u] for u in units}, seen["unit_group"]) if units else None
                        for key in ("chance", "base_rate")},
                     "per_unit": units, "unit_group": {u: seen["unit_group"][u] for u in units}} if seen else None
    return out


def random_vectors(n_cells: int) -> np.ndarray:
    """Seeded standard-normal vectors, RANDOM_DIM per cell, one row per FULL cell in prep/cells.csv order; label-free.
    Not scored: the test of the `chance` row uses them."""
    return np.random.default_rng(RANDOM_SEED).standard_normal((n_cells, RANDOM_DIM))


# ---- the per-arm score cache ---------------------------------------------------------------------------------

SCORE_CACHE = "score_cache"                  # <root>/score_cache/: every arm's scores, shared across runs
FROZEN = {"adaptation_depth": "zero_shot"}


def scoring_code() -> dict:
    """What an arm's cached scores depend on besides the arm, the prep and the table: score.py, secondary.py and the
    libraries that compute them."""
    from importlib.metadata import version

    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    here = Path(__file__).resolve().parent
    return {"score_sha256": sha256_file(here / "score.py"), "secondary_sha256": sha256_file(here / "secondary.py"),
            "numpy": np.__version__, "scikit_learn": version("scikit-learn")}


def embedding_stack(root: Path) -> dict:
    """What an analyzer arm's embedding depends on beyond `control.provenance()` (QC of A13): every module of the
    analyzer package (tests excluded), prepare.py (the tool input) and the pinned DINOv3 assets' manifest
    (<cache>/assets/ASSETS.json, which `prepare.check_assets` holds to the files), so the cross-run cache never
    reuses scores across a change of any of them."""
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    package = Path(__file__).resolve().parents[3] / "shape" / "tools" / "cell_state_analyzer"
    modules = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    manifest = Path(root).parent / "assets" / "ASSETS.json"
    return {"analyzer_modules": _digest(modules), "prepare_sha256": sha256_file(Path(__file__).with_name("prepare.py")),
            "assets_manifest_sha256": sha256_file(manifest) if manifest.exists() else None}


def delivered_state(choice: dict) -> dict:
    """The analyzer state of a delivery (`resolve_decision`): frozen, else embed_only with the call's resume_best."""
    return dict(FROZEN) if choice["use"] == "frozen" else {"init_checkpoint": choice["checkpoint"], "embed_only": True}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _encode(entry: dict) -> dict:
    """A cache entry as stored: the probe's runs keyed "<unit>/<classifier>", their bootstrap scores as lists."""
    out = dict(entry)
    if "probe" in out:
        out["probe"] = {f"{unit}/{name}": {"point": r["point"], "draws": {m: np.asarray(v).tolist()
                                                                           for m, v in r["draws"].items()},
                                           "fit": r["fit"]} for (unit, name), r in entry["probe"].items()}
    return out


def _decode(stored: dict) -> dict:
    out = dict(stored)
    if "probe" in out:
        out["probe"] = {tuple(key.split("/")): {"point": r["point"], "draws": {m: np.asarray(v, dtype=np.float64)
                                                                                for m, v in r["draws"].items()},
                                                "fit": r["fit"]} for key, r in stored["probe"].items()}
    return out


class ArmScores:
    """The per-arm score cache of the prep under `root` (module docstring): per arm and table one entry, kept in
    <root>/score_cache/<sha256 of its key>.json (written atomically) and in memory for this process. Its parts:
    `evaluation` (`evaluate` of the post-processed vectors), `probe` (A9 arms: `secondary.Probe.run` of them, the
    bootstrap scores under its fixed draws included, so the paired deltas need no refit) and `zero_norm_wells` (llm).
    A missing part is computed from the arm's vectors, an analyzer arm embedded only then; a stored entry under another
    key is recomputed."""

    def __init__(self, root: Path, summary: dict, cells, sirna: np.ndarray):
        from benchmarks.representation_RxRx1.runners import control

        self.summary, self.cells, self.sirna = summary, cells, sirna
        self.dir = root / SCORE_CACHE
        self.code = scoring_code()
        self.prep = {"cells_sha256": summary["cells_sha256"], "labels_sha256": summary["labels_sha256"],
                     "crop_array_sha256": summary["crop_array"]["sha256"]}
        self.analyzer = {**control.provenance(), **embedding_stack(root)}
        self._probe = None
        self._entries: dict[str, dict] = {}
        self._embedded: dict[str, np.ndarray] = {}
        self._sha: dict[str, str] = {}

    def probe(self):
        """The A9 probe's layout (labels and split only; no fit)."""
        from benchmarks.representation_RxRx1.runners import secondary

        if self._probe is None:
            units, skipped = probe_units(self.summary)
            self._probe = secondary.Probe(self.cells, self.sirna, units=units, skipped=skipped)
        return self._probe

    def analyzer_arm(self, arm: str, state: dict, *, probe: bool) -> dict[str, dict]:
        """{table: entry} of an analyzer arm with `state` (its embedding kwargs); embedded (`embed`) into
        <root>/score_cache/embed/<arm>-<digest>/ only when a part is missing."""
        from benchmarks.representation_RxRx1.runners.prepare import sha256_file

        kwargs = dict(state)
        checkpoint = kwargs.pop("init_checkpoint", None)
        if checkpoint and checkpoint not in self._sha:
            self._sha[checkpoint] = sha256_file(Path(checkpoint))
        source = {"kind": "analyzer", "state": kwargs, "checkpoint_sha256": self._sha.get(checkpoint),
                  "analyzer": self.analyzer}

        def vectors() -> np.ndarray:
            where = _digest({"kwargs": state, "source": source, "prep": self.prep})
            if where not in self._embedded:
                self._embedded[where] = embed(self.summary, self.cells, self.dir / "embed" / f"{arm}-{where[:16]}",
                                              state)
            return self._embedded[where]

        return self.tables(arm, source, vectors, ("evaluation", "probe") if probe else ("evaluation",))

    def feature_arm(self, arm: str, raw: np.ndarray, embeddings_sha256: str, *, probe: bool = True) -> dict[str, dict]:
        """{table: entry} of an external or llm arm from its raw vectors (their file's sha256 is its content)."""
        source = {"kind": "features", "arm": arm, "embeddings_sha256": embeddings_sha256}
        return self.tables(arm, source, lambda: raw, ("evaluation", "probe") if probe else ("evaluation",))

    def zero_norm(self, arm: str, raw: np.ndarray, embeddings_sha256: str) -> dict[str, int]:
        """{table: the wells of zero-norm profile} of a feature arm (`zero_norm_wells`)."""
        source = {"kind": "features", "arm": arm, "embeddings_sha256": embeddings_sha256}
        return {t: e["zero_norm_wells"] for t, e in self.tables(arm, source, lambda: raw, ("zero_norm_wells",)).items()}

    def tables(self, arm: str, source: dict, vectors: Callable[[], np.ndarray], parts: Sequence[str]) -> dict:
        from benchmarks.representation_RxRx1.runners import control

        out = {}
        for table in TABLES:
            key = {"source": source, "prep": self.prep, "code": self.code, "table": table,
                   "postprocess": "per_plate" if table == "matched" else ("global" if arm in NATIVE_GLOBAL else "raw")}
            digest = _digest(key)
            path = self.dir / f"{digest}.json"
            entry = self._entries.get(digest)
            if entry is None:
                try:
                    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
                except ValueError:                   # an unreadable entry (e.g. a crashed write) is recomputed
                    stored = {}
                entry = _decode(stored) if stored.get("key") == key else {"key": key, "arm": arm}
            missing = [part for part in parts if part not in entry]
            if missing:
                x = post_process(arm, vectors(), self.cells.plate.to_numpy(), table)
                for part in missing:
                    if part == "evaluation":
                        entry[part] = evaluate(x, self.cells, self.sirna)
                    elif part == "probe":
                        entry[part] = self.probe().run(np.asarray(x))
                    else:
                        entry[part] = zero_norm_wells(x, self.cells.well.to_numpy())
                del x
                text = json.dumps(_encode(entry))           # no default=: a non-JSON value is an error, not a str

                def write():
                    self.dir.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
                    tmp.write_text(text, encoding="utf-8")
                    tmp.replace(path)

                control.fs_retry(write)
            self._entries[digest] = entry
            out[table] = entry
        return out


def headline(arm: str, tables: dict) -> str:
    """One arm's line in the driver log: its native well mAP (seen, heldout) and the probe's native well-kNN balanced
    accuracy (n/a for an arm without the probe)."""
    evaluation = tables["native"].get("evaluation") or {}
    probe = tables["native"].get("probe")
    accuracy = probe[("well", "knn")]["point"]["balanced_accuracy"] if probe else None
    return (f"SCORE_ARM {arm}: well mAP (native) seen {_f((evaluation.get('seen') or {}).get('value'))} heldout "
            f"{_f((evaluation.get('heldout') or {}).get('value'))}; siRNA probe (native, well kNN) balanced accuracy "
            f"{_f(accuracy)}")


def group_chance(entries: dict, group: str, key: str = "chance") -> float | None:
    """The group's `chance` or `base_rate` (A7): the same for every scored arm (it depends only on the labels and the
    group layout); SystemExit CHANCE_MISMATCH when two arms differ."""
    values = {arm: e.get(key) for arm, e in entries.items()}
    if len(set(values.values())) > 1:
        raise SystemExit(f"CHANCE_MISMATCH: {group}: the arms' {key} differs: {values}")
    return next(iter(values.values()), None)


def compare(tables: dict, family: list[str]) -> dict:
    """Per group: every arm's value, the chance and base rate (`group_chance`), the default mean and best seed, and the
    paired bands over siRNA."""
    from benchmarks.representation_RxRx1.runners.external import EXTERNAL_ARMS

    out = {}
    for group in (*GROUPS, *AUDIT_GROUPS):
        entries = {arm: t[group] for arm, t in tables.items() if t.get(group)}
        if "frozen" not in entries or any(a not in entries for a in family):
            out[group] = None
            continue
        units = {arm: e["per_unit"] for arm, e in entries.items()}
        unit_group = {u: g for e in entries.values() for u, g in e["unit_group"].items()}
        values = {arm: e["value"] for arm, e in entries.items()}
        bands, best = {}, None
        if family:                               # a control without default arms (frozen only) skips these
            common = set.intersection(*(set(units[a]) for a in family))
            mean_units = {u: float(np.mean([units[a][u] for a in family])) for u in common}
            best = family[int(np.argmax([values[a] if values[a] is not None else -1 for a in family]))]
            bands["default_mean_vs_frozen"] = paired_band(units["frozen"], mean_units, unit_group)
        for arm in ("shape", "shape_top"):
            if arm in units:
                if family:
                    bands[f"{arm}_vs_default_mean"] = paired_band(mean_units, units[arm], unit_group)
                    bands[f"{arm}_vs_best_default_seed"] = paired_band(units[best], units[arm], unit_group)
                bands[f"{arm}_vs_frozen"] = paired_band(units["frozen"], units[arm], unit_group)
                for other in EXTERNAL_ARMS:
                    if other in units:
                        bands[f"{arm}_vs_{other}"] = paired_band(units[other], units[arm], unit_group)
        for other in EXTERNAL_ARMS:
            if other in units:
                bands[f"frozen_vs_{other}"] = paired_band(units[other], units["frozen"], unit_group)
        for name in llm_comparisons([a for a in LLM_ARMS if a in units], bool(family),     # amendments A4, A5
                                    [a for a in EXTERNAL_ARMS if a in units]):
            first, _, second = name.partition("_vs_")
            reference = mean_units if second == "default_mean" else units[second]
            bands[name] = paired_band(reference, units[first], unit_group)
        seeds = [values[a] for a in family if values[a] is not None]
        out[group] = {"values": values, "chance": group_chance(entries, group),
                      "base_rate": group_chance(entries, group, "base_rate"),
                      "default_mean": float(np.mean(seeds)) if seeds else None,
                      "best_default_seed": best, "comparisons": bands}
    return out


def llm_comparisons(arms: Sequence[str], with_defaults: bool = True, externals: Sequence[str] = ()) -> list[str]:
    """The comparisons of the LLM arms present, in the amendments' order (first minus second): llm (A5) shape vs llm,
    llm vs frozen, llm vs the default mean, llm vs cellprofiler (when scored); react (A4) shape vs react, react vs the
    default mean, react vs frozen."""
    out = []
    if "llm" in arms:
        out += (["shape_vs_llm", "llm_vs_frozen"] + (["llm_vs_default_mean"] if with_defaults else [])
                + (["llm_vs_cellprofiler"] if "cellprofiler" in externals else []))
    if "react" in arms:
        out += ["shape_vs_react"] + (["react_vs_default_mean"] if with_defaults else []) + ["react_vs_frozen"]
    return out


def llm_arm(run: Path, name: str, now: dict, shape_meta: dict) -> dict:
    """The `llm` / `react` run `run` as score reads it: react (A4) its delivered state (embed_only with the delivered
    call's resume_best, or frozen); llm (A5) its own embedding (`llm_arms.load_features`, read by `final`); or status
    failed (no delivery). Refused when it is not a finished run of that arm and of the SHAPE run's cell type, when its
    provenance (the code's sha256 and the prep) differs from the current one, when its arm design differs (llm, A11:
    `llm_arms.llm_design_problems`, its first prompt, worker and settings; react, A14: another react_arm.py or shared
    runner, benchmarks/_shared/react_arm.py, than the current one -- a run of the A4 / A11 react of llm_arms.py has
    neither and is never scored), when its model differs from the SHAPE run's, and for react when its round budget is
    not the SHAPE run's max_rounds or the SHAPE run records a CARD.md sha256 other than this run's (ARM_STALE)."""
    from benchmarks._shared import react_arm as shared_react
    from benchmarks.representation_RxRx1.runners import llm_arms, react_arm
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file

    meta_path = run / "RUN_META.json"
    if not meta_path.exists():
        raise SystemExit(f"NOT_A_RUN: {run} has no RUN_META.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cell_type = shape_meta.get("cell_type")
    if meta.get("agent") != name or meta.get("cell_type") != cell_type:
        raise SystemExit(f"WRONG_ARM: {run} is a {meta.get('agent')} run on {meta.get('cell_type')}, not a {name} run "
                         f"on {cell_type}")
    if name == "llm":                         # A12: the llm arm runs no analyzer, so its code hashes do not bind it
        now = {k: v for k, v in now.items() if k in ("torch", "prep_cells_sha256")}
    stale = {k: {"run": (meta.get("provenance") or {}).get(k), "now": v} for k, v in now.items()
             if (meta.get("provenance") or {}).get(k) != v}
    if name == "llm":                         # A11: the llm design itself, not the file hash
        stale.update(llm_arms.llm_design_problems(run, meta))
    else:                                     # A14: the react adapter and the shared runner it ran on
        for key, module in (("react_arm_sha256", react_arm), ("shared_react_sha256", shared_react)):
            current = sha256_file(Path(module.__file__))
            if meta.get(key) != current:
                stale[key] = {"run": meta.get(key), "now": current}
    if meta.get("model") != shape_meta.get("model"):
        stale["model"] = {"run": meta.get("model"), "shape": shape_meta.get("model")}
    if name == "llm" and meta.get("max_attempts") != shape_meta.get("max_rounds"):   # A5 as amended 2026-10-13
        stale["max_attempts"] = {"run": meta.get("max_attempts"), "shape max_rounds": shape_meta.get("max_rounds")}
    if name == "react" and meta.get("tool_budget") != shape_meta.get("max_rounds"):
        stale["tool_budget"] = {"run": meta.get("tool_budget"), "shape max_rounds": shape_meta.get("max_rounds")}
    if name == "react" and shape_meta.get("card_sha256") and meta.get("card_sha256") != shape_meta["card_sha256"]:
        stale["card_sha256"] = {"run": meta.get("card_sha256"), "shape": shape_meta["card_sha256"]}
    if name == "react":
        from benchmarks.representation_RxRx1.runners import epoch_budget

        budget = meta.get("training_budget") or {}
        if (not budget or budget != shape_meta.get("training_budget")
                or budget.get("policy_sha256") != sha256_file(Path(epoch_budget.__file__))):
            stale["training_budget"] = {"react": budget, "shape": shape_meta.get("training_budget")}
        fixed = {"max_epochs": budget.get("max_epochs")}
        if meta.get("fixed_levers") != fixed or shape_meta.get("fixed_levers") != fixed:
            stale["fixed_levers"] = {"react": meta.get("fixed_levers"), "shape": shape_meta.get("fixed_levers")}
    if stale:
        raise SystemExit(f"ARM_STALE: the {name} run {run} does not match the current code, the prep or the SHAPE run: "
                         f"{stale}")
    path = run / "submission" / "decision.json"
    if not path.exists():
        raise SystemExit(f"NO_SUBMISSION: the {name} run {run} has no submission/decision.json (it did not finish)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {"name": name, "run": run, "model": meta.get("model"), "card_sha256": meta.get("card_sha256"),
           "llm_arms_sha256": meta.get("llm_arms_sha256"), "react_arm_sha256": meta.get("react_arm_sha256"),
           "status": payload.get("status"),
           "case": payload.get("case"), "reason": payload.get("reason"), "use": payload.get("use"), "state": None}
    if payload.get("use") is None:                  # a failed run: no delivery, nothing to embed
        return {**out, "status": "failed"}
    if name == "llm":                               # its own embedding, no analyzer state
        agent_path = run / "AGENT_RUN.json"
        agent = json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.exists() else {}
        return {**out, "code_sha256": payload.get("code_sha256"), "dim": payload.get("dim"),
                "audit": agent.get("audit")}
    out["fixed_levers"] = meta.get("fixed_levers")
    out["training_budget"] = meta.get("training_budget")
    choice = resolve_decision(run, finished_calls(run))
    out["state"] = delivered_state(choice)
    return {**out, "checkpoint": choice["checkpoint"]}


def efficiency(run: Path) -> dict:
    """An agentic arm's cost (amendment A4): analyzer calls (the SHAPE arm's trial 0 included), training calls, wall
    hours, reasoner tokens (the engines' process-level count)."""
    u = usage(run)
    return {"tool_calls": u["calls_started"] + int((run / "logs" / "input_trial.json").exists()),
            "training_calls": u["calls_started"], "calls_finished": u["calls_finished"],
            "wall_hours": u.get("wall_hours_used"), "reasoner_tokens": (u.get("tokens") or {}).get("total_tokens")}


def finished_calls(run: Path) -> dict[str, dict]:
    """The tool facts of every finished trial of the SHAPE arm: "input" (trial 0), then each call by its name
    ("agent_1", ...; c01, c02, ... in runs before the role names)."""
    from benchmarks.batch_correction_JUMP.runners.score import jsonl

    out = {}
    trial0 = run / "logs" / "input_trial.json"
    if trial0.exists():
        record = json.loads(trial0.read_text(encoding="utf-8"))
        if record.get("status") == "done":
            out["input"] = record["tool"]
    for record in jsonl(run / "logs" / "tool_calls.jsonl"):
        if record.get("event") == "end" and record.get("status") == "done":
            out[record["call"]] = record["tool"]
    return out


def moves(run: Path) -> dict[str, dict]:
    """Per call of the SHAPE arm: its parent and the levers it set (the ledger's start records)."""
    from benchmarks.batch_correction_JUMP.runners.score import jsonl

    return {r["call"]: {"parent": r.get("parent"), "set": r.get("set")}
            for r in jsonl(run / "logs" / "tool_calls.jsonl") if r.get("event") == "start"}


def resolve_decision(run: Path, calls: dict[str, dict]) -> dict:
    """submission/decision.json, validated: `use` is "frozen" or a finished call with a resume_best that exists."""
    path = run / "submission" / "decision.json"
    if not path.exists():
        raise SystemExit("NO_SUBMISSION: submission/decision.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    use = payload.get("use")
    checkpoint = None
    if use != "frozen":
        checkpoint = (calls.get(use) or {}).get("resume_best") if use != "input" else None
        if not checkpoint or not Path(checkpoint).exists():
            raise SystemExit(f"INVALID_SUBMISSION: use {use!r} is not 'frozen' or a finished call with a resume_best "
                             f"(finished: {sorted(c for c in calls if c != 'input')})")
    return {"use": use, "checkpoint": checkpoint, "case": payload.get("case"), "status": payload.get("status"),
            "reason": payload.get("reason"), "provisional": payload.get("provisional", True)}


def diagnostic(cells, sirna: np.ndarray, calls: dict[str, dict], lineage: dict[str, dict]) -> dict:
    """Per finished trial of the SHAPE arm, from its own h5ad (the run-prep cells it embedded, val rows in the tool's
    validation order): its parent and lever changes (`lineage`), the val proxy on the comparison draws
    (quant.FRESH_SEED) with quant.beats vs frozen (trial 0) and quant.paired_delta vs its parent, the run-prep well
    mAP with its label band vs frozen (the negative controls not scored, `scored_cells`; the proxy, label-free, on
    every val cell), and whether the proxy's verdict agreed with that band (beats iff the band is significant and
    positive); Spearman over the calls (trial 0 is listed, not counted). Analysis only."""
    import anndata
    from scipy.stats import spearmanr

    from shape.tools.cell_state_analyzer import quant

    run_prep = np.isin(cells.split.to_numpy(), GROUPS["run_prep"])
    sub, labels = cells[run_prep], sirna[run_prep]
    val = sub.split.to_numpy() == "val"
    keep, wells = scored_cells(labels), sub.well.to_numpy()
    meta_val = sub[val][list(EMBED_KEYS)].to_dict("records")
    proxies, maps = {}, {}
    for call, facts in calls.items():
        vectors = aligned(anndata.read_h5ad(facts["adata_path"]), sub.cell_id)
        proxies[call] = quant.sibling_site_map(vectors[val], meta_val, seed=quant.FRESH_SEED)
        maps[call] = well_map(vectors[keep], labels[keep], wells[keep])
    frozen_proxy, frozen_map = proxies.get("input"), maps.get("input")
    points = []
    for call in calls:
        move = lineage.get(call) or {}
        units = proxies[call].get("units") or {}
        point = {"call": call, "parent": move.get("parent"), "set": move.get("set"),
                 "val_proxy": proxies[call].get("value"), "val_proxy_chance": proxies[call].get("chance"),
                 "run_prep_well_map": maps[call]["value"]}
        if call != "input" and frozen_proxy is not None:
            point["proxy_vs_frozen"] = quant.beats(units, frozen_proxy.get("units") or {})
            if move.get("parent") in proxies:
                point["proxy_vs_parent"] = quant.paired_delta(proxies[move["parent"]].get("units") or {}, units)
            band = paired_band(frozen_map["per_unit"], maps[call]["per_unit"],
                               {**frozen_map["unit_group"], **maps[call]["unit_group"]})
            point["label_vs_frozen"] = band
            point["proxy_agrees_with_label"] = (None if band["significant"] is None else
                                                point["proxy_vs_frozen"]["beats"] == (band["significant"]
                                                                                      and band["delta"] > 0))
        points.append(point)
    scored = [p for p in points if p["call"] != "input" and p["val_proxy"] is not None
              and p["run_prep_well_map"] is not None]
    rho = None
    if len(scored) >= 3:
        r = spearmanr([p["val_proxy"] for p in scored], [p["run_prep_well_map"] for p in scored])[0]
        rho = float(r) if math.isfinite(r) else None
    return {"points": points, "n_calls": len(scored), "spearman_rho": rho,
            "rule": "per finished call, from its own embedding of the run-prep cells: val proxy on the comparison "
                    "draws vs frozen and parent, run-prep well mAP vs frozen; analysis only, never fed back"}


def usage(run: Path) -> dict:
    from benchmarks.batch_correction_JUMP.runners.score import jsonl

    ledger = jsonl(run / "logs" / "tool_calls.jsonl")
    ends = [e for e in ledger if e.get("event") == "end"]
    path = run / "AGENT_RUN.json"
    agent = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return {"calls_started": sum(e.get("event") == "start" for e in ledger),
            "calls_finished": sum(e.get("status") == "done" for e in ends),
            "calls_failed": sum(e.get("status") == "crashed" for e in ends),
            "call_seconds": round(sum(e.get("seconds") or 0 for e in ends), 1),
            **{k: agent.get(k) for k in ("n_rounds", "stopped_because", "wall_hours_used", "tokens", "agentic")}}


def _f(value, spec=".4f") -> str:
    return "n/a" if value is None else format(value, spec)


def top_call(diag: dict) -> dict | None:
    """shape_top: the diagnostic point of the finished call with the highest val proxy (ties: the earlier call)."""
    return max((p for p in diag["points"] if p["call"] != "input" and p["val_proxy"] is not None),
               key=lambda p: p["val_proxy"], default=None)


def control_state(ctl: dict, arm: str) -> dict:
    """A control arm's scoring state from CONTROL.json `ctl`: frozen, or a dinov3 arm embed_only with its resume_best
    (every FULL cell is embedded with it; what the arm trained on does not enter)."""
    return dict(FROZEN) if arm == "frozen" else {"init_checkpoint": ctl["arms"][arm]["resume_best"],
                                                 "embed_only": True}


def score_arm(root: Path, arm: str, run: Path | None = None) -> dict[str, dict]:
    """Stage score-arm: one arm scored into the per-arm score cache (`ArmScores`) exactly as `final` scores it, as soon
    as it exists, and its headline printed (`headline`). arm: frozen or dinov3 / dinov3_s<k> (the complete, current
    control), openphenom or cellprofiler (the external stage's), llm, shape or react (`run`: that arm's run directory;
    shape also scores shape_top, which needs the run's diagnostic). The llm / react run checks against the SHAPE run
    (`llm_arm`) stay with `final`; a run that did not deliver, or an llm output with zero-norm well profiles, is
    reported and not scored. Returns {arm: {table: entry}} of what it scored."""
    from benchmarks.representation_RxRx1.runners import control, external, llm_arms
    from benchmarks.representation_RxRx1.runners.prepare import load

    if not (arm in ("frozen", *external.EXTERNAL_ARMS, "llm", "shape", "react") or control.is_dinov3(arm)):
        raise SystemExit(f"score-arm: unknown arm {arm!r} (frozen, dinov3, dinov3_s<k>, openphenom, cellprofiler, "
                         "llm, shape or react)")
    if arm in ("llm", "shape", "react") and run is None:
        raise SystemExit(f"score-arm {arm} needs its run directory (--run)")
    summary, cells = load(root)
    sirna = read_labels(root, cells, summary)
    scorer = ArmScores(root, summary, cells, sirna)
    scored: dict[str, dict] = {}
    if arm == "frozen" or control.is_dinov3(arm):
        path = root / control.CONTROL_DIR / "CONTROL.json"
        ctl = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not ctl.get("complete") or arm not in ctl["arms"]:
            raise SystemExit(f"NO_CONTROL: {path} is missing, incomplete or has no {arm} (run the control stage)")
        now = {**control.provenance(), "prep_cells_sha256": summary["cells_sha256"]}
        if any(ctl["provenance"].get(k) != v for k, v in now.items()):
            raise SystemExit(f"CONTROL_STALE: {path} differs from the current code or prep")
        if arm != "frozen" and not ctl["arms"][arm].get("resume_best"):
            raise SystemExit(f"NO_CHECKPOINT: the control arm {arm} wrote no resume_best")
        scored[arm] = scorer.analyzer_arm(arm, control_state(ctl, arm), probe=True)
    elif arm in external.EXTERNAL_ARMS:
        vectors, status = external.load_embeddings(root, summary, len(cells))
        if arm not in vectors:
            raise SystemExit(f"NO_EXTERNAL: {arm}: {status[arm]['note']}")
        scored[arm] = scorer.feature_arm(arm, vectors[arm], status[arm]["embeddings_sha256"])
    else:
        path = run / "submission" / "decision.json"
        if not path.exists():
            raise SystemExit(f"NO_SUBMISSION: the {arm} run {run} has no submission/decision.json")
        if json.loads(path.read_text(encoding="utf-8")).get("use") is None:
            print(f"SCORE_ARM {arm}: the run {run} did not deliver; not scored", flush=True)
            return scored
        if arm == "llm":
            raw, record = llm_arms.load_features(run, summary, len(cells))
            zero = scorer.zero_norm("llm", raw, record["embeddings_sha256"])
            if any(zero.values()):
                print(f"SCORE_ARM llm: unscorable (well profiles of zero norm after the post-processing: {zero}); "
                      "not scored", flush=True)
                return scored
            scored[arm] = scorer.feature_arm("llm", raw, record["embeddings_sha256"])
        else:
            calls = finished_calls(run)
            scored[arm] = scorer.analyzer_arm(arm, delivered_state(resolve_decision(run, calls)), probe=True)
            top = top_call(diagnostic(cells, sirna, calls, moves(run))) if arm == "shape" else None
            if top is not None and calls[top["call"]].get("resume_best"):
                scored["shape_top"] = scorer.analyzer_arm(
                    "shape_top", {"init_checkpoint": calls[top["call"]]["resume_best"], "embed_only": True},
                    probe=False)                      # analysis only: not an A9 arm
    for name, tables in scored.items():
        print(headline(name, tables), flush=True)
    return scored


def final(root: Path, run: Path, llm_run: Path | None = None, react_run: Path | None = None) -> dict:
    from benchmarks.representation_RxRx1.runners import control, external, llm_arms
    from benchmarks.representation_RxRx1.runners.prepare import load

    summary, cells = load(root)
    if react_run is not None:
        from benchmarks.representation_RxRx1.runners.epoch_budget import budget_for

        expected_budget = budget_for(summary, cells)
        shape_meta = json.loads((run / "RUN_META.json").read_text(encoding="utf-8"))
        if shape_meta.get("training_budget") != expected_budget:
            raise SystemExit("ARM_STALE: SHAPE training_budget differs from this prep or is missing; rerun SHAPE")
    control_path = root / control.CONTROL_DIR / "CONTROL.json"
    ctl = json.loads(control_path.read_text(encoding="utf-8")) if control_path.exists() else {}
    if not ctl.get("complete"):
        raise SystemExit(f"NO_CONTROL: {control_path} is missing or incomplete (run the control stage)")
    meta = json.loads((run / "RUN_META.json").read_text(encoding="utf-8"))
    now = {**control.provenance(), "prep_cells_sha256": summary["cells_sha256"]}
    stale = {k: {"control": ctl["provenance"].get(k), "run": meta["provenance"].get(k), "now": v}
             for k, v in now.items() if ctl["provenance"].get(k) != v or meta["provenance"].get(k) != v}
    if stale:
        raise SystemExit(f"CONTROL_STALE: the control, the run and the current code differ: {stale}")
    if (llm_run is not None or react_run is not None) and meta.get("card_sha256") != control.card_sha256():
        raise SystemExit(f"ARM_STALE: the SHAPE run {run} records CARD.md sha256 {meta.get('card_sha256')}, not the "
                         f"current {control.card_sha256()} (amendment A7: the LLM arms pair only with a SHAPE run of "
                         "the current CARD); rerun SHAPE")
    arm_runs = {name: llm_arm(Path(path), name, now, meta)
                for name, path in zip(LLM_ARMS, (llm_run, react_run)) if path is not None}
    sirna = read_labels(root, cells, summary)
    calls = finished_calls(run)
    choice = resolve_decision(run, calls)
    diag = diagnostic(cells, sirna, calls, moves(run))
    top = top_call(diag)
    family = sorted(a for a in ctl["arms"] if control.is_dinov3(a))
    missing = [a for a in family if not ctl["arms"][a].get("resume_best")]
    if missing:
        raise SystemExit(f"NO_CHECKPOINT: the control arms {missing} wrote no resume_best")
    states = {"frozen": dict(FROZEN), **{a: control_state(ctl, a) for a in family}, "shape": delivered_state(choice)}
    if top is not None and calls[top["call"]].get("resume_best"):
        states["shape_top"] = {"init_checkpoint": calls[top["call"]]["resume_best"], "embed_only": True}
    states.update({name: a["state"] for name, a in arm_runs.items() if a["state"] is not None})
    # every arm in both tables from the per-arm score cache (an arm with the content of an earlier one shares its
    # entries; an analyzer arm is embedded only for a missing entry); the A9 arms with their probe runs
    from benchmarks.representation_RxRx1.runners import secondary

    scorer = ArmScores(root, summary, cells, sirna)
    entries = {arm: scorer.analyzer_arm(arm, kwargs, probe=bool(secondary.a9_arms([arm])))
               for arm, kwargs in states.items()}
    external_vectors, external_arms = external.load_embeddings(root, summary, len(cells))
    for arm, raw in external_vectors.items():
        entries[arm] = scorer.feature_arm(arm, raw, external_arms[arm]["embeddings_sha256"])
    if (arm_runs.get("llm") or {}).get("status") not in (None, "failed"):     # A5: the llm run's own features
        raw, record = llm_arms.load_features(arm_runs["llm"]["run"], summary, len(cells))
        arm_runs["llm"]["embedding"] = {k: record.get(k) for k in ("attempt", "code_sha256", "dim", "seconds",
                                                                   "nonfinite_values", "embeddings_sha256")}
        # a degenerate output (e.g. a well whose cells are all NaN, hence 0 after the post-processing) leaves a well
        # profile of zero norm that the well mAP cannot rank: llm is then reported as unscorable, not scored
        zero = scorer.zero_norm("llm", raw, record["embeddings_sha256"])
        if any(zero.values()):
            arm_runs["llm"].update(status="unscorable", zero_norm_wells=zero)
        else:
            entries["llm"] = scorer.feature_arm("llm", raw, record["embeddings_sha256"])
        del raw
    by_table = {table: {arm: e[table]["evaluation"] for arm, e in entries.items()} for table in TABLES}
    scored_llm = [name for name in LLM_ARMS if name in entries]
    names = ["frozen", *family, *external_vectors, "shape", *scored_llm,
             *(["shape_top"] if "shape_top" in states else [])]
    columns = [*GROUPS, *AUDIT_GROUPS]

    def summarise(tables: dict) -> dict:
        groups = compare(tables, family)
        compact = {arm: {g: ({k: e.get(k) for k in ("value", "chance", "base_rate", "n_rows", "n_groups_scored")}
                             | {"n_units": len(e["per_unit"])}) if e else None for g, e in t.items()}
                   for arm, t in tables.items()}
        rows = ["| arm | " + " | ".join(columns) + " |", "| --- |" + " --- |" * len(columns)]
        for name in names:
            rows.append(f"| {name} | " + " | ".join(_f((compact[name].get(g) or {}).get("value")) for g in columns)
                        + " |")
        rows.append("| default mean | " + " | ".join(_f((groups.get(g) or {}).get("default_mean")) for g in columns)
                    + " |")
        rows.append("| chance | " + " | ".join(_f((groups.get(g) or {}).get("chance")) for g in columns) + " |")
        rows += ["", "| group | comparison | delta +- 2SE (paired over siRNA) | significant |",
                 "| --- | --- | --- | --- |"]
        for g in columns:
            for cname, b in ((groups.get(g) or {}).get("comparisons") or {}).items():
                rows.append(f"| {g} | {cname} | {_f(b['delta'], '+.4f')} +- {_f(b['band_2se'])} | "
                            f"{b['significant']} |")
        return {"groups": groups, "tables": compact, "markdown_table": "\n".join(rows),
                "reading_vs_frozen": {g: verdict_vs_frozen(_band(groups, g, "shape_vs_frozen")) for g in READ}}

    native, matched = (summarise(by_table[table]) for table in TABLES)
    reading = {g: verdict(_band(native["groups"], g, "shape_vs_default_mean")) for g in READ}
    kind = "provisional" if choice["provisional"] else "final"
    frozen_case = (FROZEN_CASES.get("provisional" if choice["provisional"] else choice["case"], choice["case"])
                   if choice["use"] == "frozen" else None)
    result = {
        "question": QUESTION, "cell_type": meta.get("cell_type"), "run": str(run),
        "crop_array": summary["crop_array"]["path"],
        "training_budget": meta.get("training_budget"),
        "decision": {**choice, "kind": kind, "frozen_case": frozen_case}, "reading": reading,
        "reading_vs_frozen": native["reading_vs_frozen"],
        "default_arms": family, "external_arms": external_arms, "postprocess": POSTPROCESS,
        "groups": native["groups"], "tables": native["tables"],        # the native table (the reading's)
        "matched": {k: matched[k] for k in ("groups", "tables", "reading_vs_frozen", "markdown_table")},
        "shape_top": top, "diagnostic": diag, "usage": usage(run), "provenance": ctl["provenance"],
        "prep": {"splits": summary["splits"], "tool_input": summary["tool_input"]},
        "rule": "well mAP of every FULL cell's embedding per arm; seen (train + val + unused wells) is primary, "
                "heldout secondary; paired cluster bootstrap over siRNA, +-2 SE; shape_top is analysis only; "
                "groups / tables / reading / reading_vs_frozen / markdown_table: the native table; matched: the "
                "matched table, the one to read for the external comparisons (postprocess)",
        "markdown_table": native["markdown_table"],
    }
    excluded = negative_controls(cells, sirna)
    if excluded is not None:                     # absent for a prep without negative controls (unchanged output)
        result["negative_controls"] = excluded
    if arm_runs:                                 # amendments A4, A5; absent without the LLM arms (unchanged output)
        compared = llm_comparisons(scored_llm, bool(family), list(external_vectors))
        result["llm_arms"] = {name: {k: (str(v) if k == "run" else v) for k, v in a.items()}
                              for name, a in arm_runs.items()}
        result["llm_comparisons"] = {
            "read": {c: LLM_READ[next(a for a in LLM_ARMS if a in c.split("_vs_"))] for c in compared},
            "names": compared,
            **{table: {g: {c: _band(t["groups"], g, c) for c in compared} for g in READ}
               for table, t in (("native", native), ("matched", matched))}}
        result["efficiency"] = {"shape": efficiency(run),
                                **{name: efficiency(a["run"]) for name, a in arm_runs.items()}}
    # amendment A9: the secondary metrics, after every existing score, from the probe runs of the same post-processed
    # vectors (the score cache's)
    started = time.time()
    a9 = secondary.a9_arms(names)
    result["secondary"] = secondary.record(
        scorer.probe(), {table: {arm: entries[arm][table]["probe"] for arm in a9} for table in TABLES}, family,
        {"native": native["groups"], "matched": matched["groups"]}, started)
    control.write_json(run / "RESULT.json", result)
    delivered = (f"frozen (case: {choice['case']}: {frozen_case})" if choice["use"] == "frozen"
                 else f"{choice['use']} (case: {choice['case']})")

    def band(b, delta="delta", se=None):
        if not b or b.get(delta) is None:
            return "n/a"
        width = b["band_2se"] if se is None else (2 * b[se] if b.get(se) is not None else None)
        return f"{_f(b[delta], '+.4f')} +- {_f(width)}"

    points = "\n".join(
        f"| {p['call']} | {p['parent'] or ''} | {json.dumps(p['set'], default=str) if p['set'] else ''} | "
        f"{_f(p['val_proxy'])} | {_f(p['val_proxy_chance'])} | {band(p.get('proxy_vs_frozen'), se='se')} | "
        f"{band(p.get('proxy_vs_parent'), se='se')} | {_f(p['run_prep_well_map'])} | "
        f"{band(p.get('label_vs_frozen'))} | {p.get('proxy_agrees_with_label', '')} |" for p in diag["points"])
    headline = reading if family else native["reading_vs_frozen"]     # without default arms: shape vs frozen only
    external_rows = "\n".join(
        f"| {g} | {name} | {band(_band(matched['groups'], g, name))} | "
        f"{(_band(matched['groups'], g, name) or {}).get('significant')} | {band(_band(native['groups'], g, name))} | "
        f"{(_band(native['groups'], g, name) or {}).get('significant')} |"
        for g in READ for name in (f"{a}_vs_{o}" for a in ("shape", "frozen") for o in external_vectors))
    llm_text = ""
    if arm_runs:
        def delivered_arm(name: str, a: dict) -> str:
            if a["status"] == "failed":
                return f"{name}: failed ({a['case']}: {a['reason']})"
            if a["status"] == "unscorable":
                return (f"{name}: unscorable ({a['use']}: well profiles of zero norm after the post-processing, "
                        + ", ".join(f"{t} {n}" for t, n in a["zero_norm_wells"].items()) + "; not scored)")
            return f"{name}: {a['use']} (case: {a['case']}" + (f", d = {a['dim']}" if name == "llm" else "") + ")"

        delivered_llm = "; ".join(delivered_arm(name, a) for name, a in arm_runs.items())
        audit = (arm_runs.get("llm") or {}).get("audit")
        if audit is not None:
            delivered_llm += (". llm audit (post hoc, the delivered module's text): banned-like tokens "
                              + (", ".join(audit.get("banned_like_tokens") or []) or "none"))
        llm_rows = "\n".join(
            f"| {g} | {name} | {result['llm_comparisons']['read'][name]} | {band(_band(native['groups'], g, name))} | "
            f"{(_band(native['groups'], g, name) or {}).get('significant')} | "
            f"{band(_band(matched['groups'], g, name))} | "
            f"{(_band(matched['groups'], g, name) or {}).get('significant')} |"
            for g in READ for name in result["llm_comparisons"]["names"])
        eff_rows = "\n".join(f"| {name} | {e['tool_calls']} | {e['training_calls']} | {_f(e['wall_hours'], '.3f')} | "
                              f"{e['reasoner_tokens'] if e['reasoner_tokens'] is not None else 'n/a'} |"
                              for name, e in result["efficiency"].items())
        llm_text = (
            "\n\n## LLM arms (contract amendments A4, A5; llm read on the matched table, react on the native table)\n\n"
            "The SHAPE arm's model without SHAPE's loop. llm: LLM-written hand-crafted features (no analyzer, no "
            "training, no pretrained weights), scored as an external feature arm (native: one global centre-scale; "
            "matched: per plate). react: the shared ReAct runner -- SHAPE's task text, the tool's parameters (name, "
            "type, default) and the tool's own outputs, no CARD text. SHAPE and react share a "
            f"dataset-derived fixed per-call epoch cap ({(meta.get('training_budget') or {}).get('max_epochs', 'n/a')}); "
            f"8 tuning calls are not an 8-epoch total budget. Delivered: {delivered_llm}.\n\n"
            "| group | comparison | read on | native delta +- 2SE | significant | matched delta +- 2SE | "
            "significant |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n" + llm_rows
            + "\n\n## Efficiency\n\n| arm | tool calls | training calls | wall hours | reasoner tokens |\n"
              "| --- | --- | --- | --- | --- |\n" + eff_rows)
    text = (
        f"# Result {meta.get('cell_type')}: seen {headline['seen']}, heldout {headline['heldout']}"
        f"{'' if family else ' (no default arm: shape vs frozen only)'}\n\n"
        f"{QUESTION}\n\nDelivered ({kind} decision.json): {delivered}"
        f"{' -- PROVISIONAL (the search did not finish)' if choice['provisional'] else ''}"
        f"; status {choice['status']}, reason {choice['reason']}. shape_top (analysis only): "
        f"{top['call'] if top else 'n/a'}.\n\nExternal arms (label-free, no training): "
        + "; ".join(f"{arm} {s['status']}" + (f" ({s['note']})" if s.get("note") else "")
                    for arm, s in external_arms.items())
        + ". cellprofiler = CellProfiler-style features per crop (cp_measure), not the full CellProfiler "
          "pipeline.\n\n"
        + (f"Negative controls: the {excluded['wells']} {NEGATIVE_CONTROL} wells ({excluded['cells']} cells) are "
           "not scored (neither query, positive nor candidate in the well mAP; not in the siRNA probe); they stay in "
           "every label-free step.\n\n" if excluded is not None else "")
        + "Random baseline (contract amendment A7): `chance` = the exact expected AP of a random ranking per query, "
          "averaged like the well mAP (RESULT.json also keeps base_rate = n_pos / n_cand); not a method under test, "
          "no comparison is read against it.\n\n"
        + (f"OpenPhenom attribution (its EULA, section 4): {external.OPENPHENOM_ATTRIBUTION}\n\n"
           if "openphenom" in external_vectors else "")
        + f"Post-processing (contract amendment A3): native = {POSTPROCESS['native']}; matched = "
          f"{POSTPROCESS['matched']}; PCA: {POSTPROCESS['pca']}. The pre-registered reading is read on the native "
          f"table; shape vs frozen: native {native['reading_vs_frozen']}, matched {matched['reading_vs_frozen']}; "
          f"read the external comparisons on the matched table. Native vs matched: {POSTPROCESS['gap']}.\n\n"
        + "## Native table (the pre-registered reading)\n\n" + native["markdown_table"]
        + "\n\n## Matched table (read the external comparisons here)\n\n" + matched["markdown_table"]
        + ("\n\n## External comparisons (matched; native for reference)\n\n"
           "| group | comparison | matched delta +- 2SE | significant | native delta +- 2SE | significant |\n"
           "| --- | --- | --- | --- | --- | --- |\n" + external_rows if external_rows else "")
        + llm_text
        + secondary.markdown(result["secondary"])
        + f"\n\n## Diagnostic (analysis only)\n\nSpearman rho of the val proxy vs the run-prep well mAP over "
          f"{diag['n_calls']} calls = {_f(diag['spearman_rho'], '+.3f')}. Proxy deltas on the comparison draws "
          "(+- 2 SE over val wells); the label band vs frozen over siRNA (+- 2 SE).\n\n"
          "| call | parent | set | val proxy | chance | proxy vs frozen | proxy vs parent | run-prep well mAP | "
          "well mAP vs frozen | proxy verdict agrees |\n" + "| --- " * 10 + "|\n" + points
        + f"\n\nUsage: {result['usage']}.\n")
    control.fs_retry((run / "RESULT.md").write_text, text, encoding="utf-8")
    print("native\n" + native["markdown_table"] + "\n\nmatched\n" + matched["markdown_table"])
    print(f"READING {reading}  delivered={delivered}")
    return result
