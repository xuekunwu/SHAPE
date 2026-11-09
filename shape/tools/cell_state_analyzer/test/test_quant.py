"""The analyzer's label-free measurement module (`quant.py`) and the CARD's `## Levers` section.

quant: identities and tiers, the val-well split, the sibling-site proxy with its k-cell profiles and seeded
draws (SELECTION_SEED / FRESH_SEED, the row-order rule), the paired delta and the 2-SE rule, the health
statistics and the image leakage, on hand-computed cases and against a copy of the task-116 all-cell
reference. CARD: the 16 levers (A11: 15 in 4 Families, plus max_epochs) parse with the shared parser
(`schemas.card_lever_fields` / `lever_spec`) to the contract's Family / Type / Range / Default, every Default equals
the execute() signature default (init_from is a verifier lever without a kwarg), the lead paragraph defines the
Families and names no key levers, and the sections the linear verifier and full_HCMA read stay. CPU only; no tool call.
"""
import hashlib
import inspect
import itertools
import math
import re

import numpy as np
import pytest

from shape.tools.cell_state_analyzer import quant


def _meta(plates=("P1",), wells=("A", "B", "C", "D"), sites=2, cells=1):
    """One row per cell: every well of every plate has `sites` sites (names unique per well) of `cells` cells."""
    return [{"plate": plate, "well": well, "site": f"{well}_s{s}"}
            for plate in plates for well in wells for s in range(sites) for _ in range(cells)]


def _reference_sibling(vectors, meta, scope="plate"):
    """The task-116 all-cell sibling retrieval (H/workspace_template/tools/quality.py `_sibling` @ f667f94),
    copied for the parity check: every site's profile is the mean of all its L2-normalised cells."""
    ids = quant.identities(meta)
    sites = np.array([r["site"] for r in ids])
    keys, first, inverse = np.unique(sites, return_index=True, return_inverse=True)
    profiles = np.zeros((len(keys), vectors.shape[1]))
    np.add.at(profiles, inverse, quant._l2(vectors))
    profiles = quant._l2(profiles)
    wells = np.array([r["well"] for r in ids])[first]
    scopes = np.array([r[scope] for r in ids])[first]
    ap, chance = {}, {}
    for value in np.unique(scopes):
        rows = np.flatnonzero(scopes == value)
        sim = profiles[rows] @ profiles[rows].T
        for i, query in enumerate(rows):
            others = np.delete(np.arange(len(rows)), i)
            positive = wells[rows[others]] == wells[query]
            n_pos, n_cand = int(positive.sum()), len(others)
            if n_pos == 0 or n_pos == n_cand:
                continue
            hit = positive[np.argsort(-sim[i, others], kind="stable")]
            unit = f"{wells[query]}|{keys[query]}"
            ap[unit] = float((np.cumsum(hit) / np.arange(1, n_cand + 1))[hit].mean())
            chance[unit] = quant._random_ap(n_pos, n_cand)
    return ap, chance


# ---------------------------------------------------------------- identities, tiers

def test_identities_plate_qualify_and_read_id_keys():
    rows = [{"plate": "P1", "well": "A01", "site": "A01_1", "experiment": "E1"},
            {"plate_id": "P2", "well_id": "A01", "site_id": "A01_1"},
            {"plate": "P1", "well": "P1/A02", "site": 7},
            {"plate": float("nan"), "well": "A03", "site": "  "},
            {"well": "A04", "experiment": ""}]
    ids = quant.identities(rows)
    assert ids[0] == {"plate": "P1", "experiment": "E1", "well": "P1/A01", "site": "P1/A01_1"}
    assert ids[1] == {"plate": "P2", "experiment": None, "well": "P2/A01", "site": "P2/A01_1"}
    assert ids[2]["well"] == "P1/A02" and ids[2]["site"] == "P1/7"          # never prefixed twice; str values
    assert ids[3] == {"plate": None, "experiment": None, "well": "A03", "site": None}   # NaN, whitespace: missing
    assert ids[4] == {"plate": None, "experiment": None, "well": "A04", "site": None}


def test_identities_accept_a_dataframe():
    pd = pytest.importorskip("pandas")
    rows = _meta(wells=("A", "B"))
    assert quant.identities(pd.DataFrame(rows)) == quant.identities(rows)


def test_tier_of():
    assert quant.tier_of(_meta()) == "replicate"
    assert quant.tier_of([{**r, "well": None} for r in _meta()]) == "batch"            # site + plate, no well
    assert quant.tier_of(_meta(sites=1)) == "batch"                                   # wells without 2 sites
    assert quant.tier_of([{"site": r["site"], "experiment": "E1"} for r in _meta()]) == "batch"
    assert quant.tier_of([{"site": r["site"]} for r in _meta()]) == "image"
    assert quant.tier_of([]) == "image"
    assert set(quant.ADOPTED_PROXY) == set(quant.TIERS)
    assert quant.ADOPTED_PROXY == {"replicate": "sibling_site_map", "batch": None, "image": None}


# ---------------------------------------------------------------- val wells

def test_assign_val_wells_per_plate_in_seeded_sha256_order():
    wells = tuple(f"W{i:02d}" for i in range(30))
    rows = _meta(plates=("P1", "P2"), wells=wells, cells=2)
    split = quant.assign_val_wells(rows, val_fraction=0.1)
    ids = quant.identities(rows)
    for plate in ("P1", "P2"):
        units = sorted({r["well"] for r in ids if r["plate"] == plate},
                       key=lambda u: hashlib.sha256(chr(0).join(("0", "val", plate, u)).encode("utf-8")).hexdigest())
        val = {r["well"] for r, s in zip(ids, split) if r["plate"] == plate and s == "val"}
        assert val == set(units[:3])                          # round(0.1 x 30) = 3
    by_well = {}
    for r, s in zip(ids, split):
        by_well.setdefault(r["well"], set()).add(s)
    assert all(len(s) == 1 for s in by_well.values())         # a well is never split
    assert split == quant.assign_val_wells(rows, val_fraction=0.1)
    assert split != quant.assign_val_wells(rows, val_fraction=0.1, seed=1)
    small = quant.assign_val_wells(_meta(wells=tuple(f"W{i}" for i in range(10))), val_fraction=0.1)
    assert small.count("val") == quant.MIN_VAL_WELLS * 2      # max(2, round(1)) wells of 2 sites


def test_assign_val_wells_reads_no_label_and_rejects_bad_input():
    rows = _meta(wells=tuple(f"W{i}" for i in range(6)))
    labelled = [{**r, "sirna": f"s{i}", "group": f"g{i}"} for i, r in enumerate(rows)]
    assert quant.assign_val_wells(labelled) == quant.assign_val_wells(rows)
    with pytest.raises(ValueError, match="well"):
        quant.assign_val_wells([{k: v for k, v in r.items() if k != "well"} for r in rows])
    with pytest.raises(ValueError, match="too few wells"):
        quant.assign_val_wells(_meta(wells=("A", "B")))
    with pytest.raises(ValueError, match="val_fraction"):
        quant.assign_val_wells(rows, val_fraction=1.0)


# ---------------------------------------------------------------- sibling_site_map

def test_seed_constants():
    assert quant.SELECTION_SEED == 0 and quant.FRESH_SEED == 1
    signature = inspect.signature(quant.sibling_site_map).parameters
    assert signature["seed"].default == quant.SELECTION_SEED
    assert (signature["k"].default, signature["draws"].default, signature["scope"].default) == (2, 20, "plate")


def test_sibling_hand_case_and_chance():
    """One plate, wells A and B with two one-cell sites each, at angles A_s0 0, B_s0 10, A_s1 30, B_s1 90
    degrees: the sibling ranks 2nd for A_s0, A_s1 and B_s1 and 3rd for B_s0; chance = H(3) / 3 = 11/18."""
    angles = np.radians([0, 30, 10, 90])                     # rows: A_s0, A_s1, B_s0, B_s1
    x = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    out = quant.sibling_site_map(x, _meta(wells=("A", "B")))
    assert out["units"] == pytest.approx({"P1/A|P1/A_s0": 0.5, "P1/A|P1/A_s1": 0.5,
                                          "P1/B|P1/B_s0": 1 / 3, "P1/B|P1/B_s1": 0.5})
    assert out["value"] == pytest.approx((0.5 + 0.5 + 1 / 3 + 0.5) / 4)
    assert out["chance"] == pytest.approx(11 / 18)
    assert (out["n_queries"], out["k"], out["draws"]) == (4, 2, 20)


def test_sibling_matches_the_116_all_cell_reference_when_k_covers_every_site():
    rng = np.random.default_rng(3)
    rows = [r for r in _meta(plates=("P1", "P2"), wells=("A", "B", "C", "D", "E"), sites=3, cells=4)
            if rng.random() > 0.2]                           # uneven cells per site
    x = rng.normal(size=(len(rows), 16))
    ref_ap, ref_chance = _reference_sibling(x, rows)
    out = quant.sibling_site_map(x, rows, k=100, draws=3)
    assert out["units"] == pytest.approx(ref_ap)
    assert out["value"] == pytest.approx(np.mean(list(ref_ap.values())))
    assert out["chance"] == pytest.approx(np.mean(list(ref_chance.values())))


def test_sibling_profiles_are_k_cells_drawn_without_replacement():
    """k = 2 of 3 cells per site, one draw: the value is one of the C(3, 2)^4 reference values on the
    chosen cells, and different seeds choose different cells."""
    rng = np.random.default_rng(0)
    rows = _meta(wells=("A", "B"), cells=3)
    x = rng.normal(size=(len(rows), 8))
    possible = set()
    for picks in itertools.product(itertools.combinations(range(3), 2), repeat=4):
        keep = [3 * site + c for site, pair in enumerate(picks) for c in pair]
        ap, _ = _reference_sibling(x[keep], [rows[i] for i in keep])
        possible.add(round(float(np.mean(list(ap.values()))), 12))
    seen = {round(quant.sibling_site_map(x, rows, k=2, draws=1, seed=s)["value"], 12) for s in range(20)}
    assert seen <= possible and len(seen) > 1


def test_sibling_ranks_within_the_plate_only():
    """Equal well names on two plates with the same vectors: plate-qualified, scoped to the plate, every
    site retrieves its own well's sibling first."""
    rows = _meta(plates=("P1", "P2"), wells=("A", "B", "C"))
    well_vec = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}
    x = np.array([well_vec[r["well"]] for r in rows], dtype=float)
    out = quant.sibling_site_map(x, rows)
    assert out["value"] == 1.0 and out["n_queries"] == 12
    assert "P2/A|P2/A_s1" in out["units"]


def test_sibling_skips_rows_without_ids_and_returns_none_without_a_query():
    rows = _meta(wells=("A", "B"), cells=2)
    x = np.random.default_rng(1).normal(size=(len(rows) + 1, 4))
    full = quant.sibling_site_map(x[:-1], rows)
    assert quant.sibling_site_map(x, rows + [{"plate": "P1", "well": "A"}]) == full
    empty = quant.sibling_site_map(x[:2], _meta(wells=("A",)))        # one well: no query has a negative
    assert empty == {"value": None, "chance": None, "units": {}, "n_queries": 0, "k": 2, "draws": 20}
    assert quant.sibling_site_map(x[:0], [])["value"] is None


def test_sibling_seeds_and_row_order():
    """The draws are seeded and follow the row order: the same rows in the same order give the same units at a
    seed; FRESH_SEED draws other cells than SELECTION_SEED on the same val sites; a permuted row order draws
    other cells, and restoring the order restores the units."""
    rng = np.random.default_rng(2)
    rows = _meta(plates=("P1", "P2"), wells=("A", "B", "C", "D"), sites=2, cells=6)
    x = rng.normal(size=(len(rows), 8))
    selection = quant.sibling_site_map(x, rows, seed=quant.SELECTION_SEED)
    assert quant.sibling_site_map(x, rows) == selection
    assert quant.sibling_site_map(x.copy(), [dict(r) for r in rows], seed=quant.SELECTION_SEED) == selection
    fresh = quant.sibling_site_map(x, rows, seed=quant.FRESH_SEED)
    assert set(fresh["units"]) == set(selection["units"]) and fresh["chance"] == selection["chance"]
    assert fresh["units"] != selection["units"]
    perm = rng.permutation(len(rows))
    permuted = quant.sibling_site_map(x[perm], [rows[i] for i in perm], seed=quant.SELECTION_SEED)
    assert set(permuted["units"]) == set(selection["units"]) and permuted["units"] != selection["units"]
    back = np.argsort(perm)
    restored = quant.sibling_site_map(x[perm][back], [rows[i] for i in perm[back]], seed=quant.SELECTION_SEED)
    assert restored == selection


# ---------------------------------------------------------------- paired delta, beats

def test_paired_delta_hand_case():
    a = {"P1/w1|s1": 0.5, "P1/w1|s2": 0.5, "P1/w2|s3": 0.2}
    b = {"P1/w1|s1": 0.7, "P1/w1|s2": 0.6, "P1/w2|s3": 0.2, "P1/w3|s4": 0.9}
    out = quant.paired_delta(a, b)
    assert out["delta"] == pytest.approx(0.1) and (out["n"], out["n_wells"]) == (3, 2)
    # bootstrap over the wells w1 (sum 0.3, 2 units) and w2 (0.0, 1 unit): the means 0.15, 0.1 and 0
    assert 0 < out["se"] < 0.15 and out == quant.paired_delta(a, b)
    assert quant.paired_delta(a, b, seed=1)["se"] != out["se"]
    one_well = quant.paired_delta({"w|s1": 0.1, "w|s2": 0.2}, {"w|s1": 0.3, "w|s2": 0.4})
    assert one_well["delta"] == pytest.approx(0.2) and one_well["se"] is None and one_well["n_wells"] == 1
    assert quant.paired_delta({"w|a": 1.0}, {"v|b": 1.0}) == {"delta": None, "se": None, "n": 0, "n_wells": 0}


def test_beats_is_the_2_se_rule_new_minus_ref():
    rng = np.random.default_rng(4)
    ref = {f"P1/w{i}|s{j}": float(rng.random()) for i in range(12) for j in range(2)}
    better = {u: v + 0.1 + 0.01 * float(rng.normal()) for u, v in ref.items()}
    up = quant.beats(better, ref)
    assert up["beats"] and up["delta"] > 2 * up["se"] and up["n"] == 24 and up["n_wells"] == 12
    assert up == {"beats": True, **quant.paired_delta(ref, better)}
    down = quant.beats(ref, better)
    assert not down["beats"] and down["delta"] < 0
    same = quant.beats(ref, ref)
    assert not same["beats"] and same["delta"] == 0
    noisy = {u: v + 0.3 * float(rng.normal()) for u, v in ref.items()}
    assert not quant.beats(noisy, ref)["beats"]
    single = quant.beats({"w|s1": 0.9, "w|s2": 0.9}, {"w|s1": 0.1, "w|s2": 0.1})
    assert single["se"] is None and single["beats"] is False             # never without an SE


# ---------------------------------------------------------------- health, image leakage

def test_health_hand_cases():
    eye = quant.health(np.eye(8))
    assert eye["effective_rank"] == pytest.approx(7.0)    # centred: 7 equal singular values
    assert eye["dead_dim_frac"] == 0.0
    assert eye["mean_cosine"] == pytest.approx(1 / math.sqrt(8))
    assert eye["uniformity"] == pytest.approx(-4.0)       # every pair at squared distance 2
    collapsed = quant.health(np.ones((10, 4)))
    assert collapsed["effective_rank"] == pytest.approx(1.0) and collapsed["dead_dim_frac"] == 1.0
    x = np.random.default_rng(5).normal(size=(50, 10))
    x[:, 3] = 2.0
    assert quant.health(x)["dead_dim_frac"] == pytest.approx(0.1)
    assert quant.health(x[:1])["uniformity"] is None
    assert set(eye) == {"effective_rank", "dead_dim_frac", "mean_cosine", "uniformity"}


def test_health_uses_a_seeded_subset_of_large_embeddings():
    x = np.random.default_rng(6).normal(size=(quant.HEALTH_ROWS + 50, 6))
    assert quant.health(x) == quant.health(x)
    assert quant.health(x) != quant.health(x[:quant.HEALTH_ROWS])


def test_image_leakage_hand_cases():
    """Each site's cells share one direction: all 5 neighbours are same-site cells (observed 1); expected =
    5 other same-site cells over the group's other cells."""
    rows = _meta(plates=("P1", "P2"), wells=("A",), sites=2, cells=6)  # 2 plates x 2 sites x 6 cells
    sites = sorted({r["plate"] + r["site"] for r in rows})
    x = np.array([np.eye(4)[sites.index(r["plate"] + r["site"])] for r in rows])
    x = x + 1e-3 * np.random.default_rng(7).normal(size=x.shape)
    assert quant.image_leakage(x, rows) == pytest.approx(11 / 5)        # per plate: 12 cells
    no_plate = [{"site": r["plate"] + "/" + r["site"]} for r in rows]
    assert quant.image_leakage(x, no_plate) == pytest.approx(23 / 5)   # all 24 cells at once
    assert quant.image_leakage(x, [{"plate": r["plate"]} for r in rows]) is None
    assert quant.image_leakage(x[:6], rows[:6]) is None                # one site, too few cells
    noise = np.random.default_rng(8).normal(size=(400, 16))
    random_rows = _meta(wells=tuple(f"W{i}" for i in range(10)), sites=2, cells=20)
    assert 0.5 < quant.image_leakage(noise, random_rows) < 1.5


# ---------------------------------------------------------------- the CARD's ## Levers

LEVERS = {   # name: (kind, bounds, nullable, default) -- contract section 3.5.1 as amended by A11
    "view_pairing": ("choice", ("same_crop", "same_site", "same_well"), False, "same_crop"),
    "contrastive_mask": ("choice", ("off", "same_site", "same_well"), False, "off"),
    "batch_sampling": ("choice", ("random", "per_plate", "per_experiment"), False, "random"),
    "trainable_pct": ("integer", (0.0, 100.0), True, None),
    "init_from": ("choice", ("pretrained", "parent"), False, "pretrained"),
    "learning_rate": ("number", (1e-6, 1e-4), False, 3e-5),
    "max_epochs": ("integer", (1.0, 60.0), True, None),
    "head_learning_rate": ("number", (1e-6, 1e-2), True, None),
    "weight_decay": ("number", (0.0, 0.5), False, 1e-4),
    "clip_grad_norm": ("number", (0.0, 100.0), False, 0.0),
    "warmup_fraction": ("number", (0.0, 0.5), True, None),
    "koleo_weight": ("number", (0.0, 1.0), False, 0.0),
    "intensity_norm": ("choice", ("off", "per_plate"), False, "off"),
    "augmentation": ("choice", ("strong", "light"), False, "strong"),
    "crop_scale_min": ("number", (0.05, 1.0), False, 0.3),
    "drop_path": ("number", (0.0, 0.5), False, 0.0),
}
FAMILIES = {   # A11: 15 levers in 4 Families, plus max_epochs (the benchmark fixes it)
    "objective": ("view_pairing", "contrastive_mask", "batch_sampling", "koleo_weight"),
    "depth": ("trainable_pct", "init_from", "drop_path"),
    "optimizer": ("learning_rate", "max_epochs", "head_learning_rate", "weight_decay", "clip_grad_norm",
                  "warmup_fraction"),
    "input": ("intensity_norm", "augmentation", "crop_scale_min"),
}
# A11: the former levers, out of the CARD's ## Levers (they stay execute() kwargs at their defaults)
FORMER_LEVERS = ("gram_weight", "augment_color", "head_warmup_epochs", "patch_embed_lr_mult", "plate_mean_weight",
                 "plate_adversarial_weight", "batch_size", "nn_k", "nn_space", "nn_mutual", "nn_refresh_epochs")
EXCLUDED = ("temperature", "layer_decay", "min_lr", "weight_decay_end", "adaptation_depth", "field_aware",
            "zero_shot", "freeze_backbone", "input_strategy", "objective", "early_stop_loss", "patience",
            "max_minutes", "eval_every_epochs", "seed", "select_on", "contrastive_unit", "global_size", "proj_dim",
            "intensity_jitter", "noise_std", "blur_prob", "flip_rotate", "ema_momentum", "cluster_resolution",
            "n_neighbors", "n_pcs", "reuse_embedding", "compute_alignment")


def _body() -> str:
    from shape.tools.base import load_tuning_card

    return load_tuning_card("cell_state_analyzer")


def _section(title: str) -> str:
    body = _body()
    start = re.search(rf"^{re.escape(title)}.*$", body, re.MULTILINE).start()
    end = body.find("\n## ", start + 1)
    return body[start:end if end > 0 else None]


def test_card_levers_parse_to_the_contract():
    from shape.utils.schemas import card_lever_fields, lever_spec

    fields = card_lever_fields("cell_state_analyzer", "## Levers")
    assert list(fields) == list(LEVERS)
    for name, expected in LEVERS.items():
        kind, bounds, nullable, default = lever_spec(fields[name])
        assert (kind, bounds, nullable) == expected[:3], name
        assert default == expected[3] and type(default) is type(expected[3]), name
    assert not set(EXCLUDED) & set(fields) and not set(FORMER_LEVERS) & set(fields)
    assert {name: fields[name]["Family"] for name in LEVERS} == {
        name: family for family, names in FAMILIES.items() for name in names}
    section = _section("## Levers").splitlines()
    for i, line in enumerate(section):
        if line.startswith("### "):
            assert section[i + 1].startswith("Family: ") and section[i + 1].count("|") == 3, line
    assert "nn_other_image" not in _section("## Levers")


def test_card_defaults_equal_the_execute_signature():
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    signature = inspect.signature(Cell_State_Analyzer_Tool.execute).parameters
    missing = [name for name in LEVERS if name != "init_from" and name not in signature]
    assert not missing, f"execute() lacks the lever kwargs {missing}"
    assert "init_from" not in signature                     # a verifier lever, never a tool kwarg
    for name, (_, _, _, default) in LEVERS.items():
        if name != "init_from":
            value = signature[name].default
            assert value == default and isinstance(value, bool) == isinstance(default, bool), (name, value)
            assert (value is None) == (default is None), (name, value)


def test_card_lead_paragraph_names_the_families_and_no_key_levers():
    """A11: the lead paragraph defines the four Families; no recipe of key levers anywhere in ## Levers."""
    section = _section("## Levers")
    lead = section.split("\n\n")[1]
    assert lead.startswith("Each lever below has one line `Family | Type | Range | Default`.")
    for family in FAMILIES:
        assert f"`{family}`" in lead, family
    assert not re.search(r"[0-9]", lead)
    assert "Key levers" not in section and "marking above" not in section and "improved" not in section


def test_card_keeps_the_linear_sections_and_the_front_matter():
    from shape.tools.base import load_card

    body = _body()
    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings == ["## Tunable parameters", "## Levers",
                        "## Agentic VRR (RepresentationAgenticVerifier, StepVerifier wave mode)", "## Notes"]
    assert re.search(r"^### adaptation_depth ", _section("## Tunable parameters"), re.MULTILINE)
    assert not re.search(r"^### [a-z]", _section("## Agentic VRR"), re.MULTILINE)
    assert "PRE-FLIGHT, one-shot" not in body and "deliberately not mid-loop VRR levers" not in body
    assert load_card("cell_state_analyzer")[0] == {
        "tool": "Cell_State_Analyzer_Tool", "category": "bioimage_core", "stage": "cell_level",
        "depends_on": ["Single_Cell_Cropper_Tool"], "tunable": True,
        "metadata_consumes": ["group", "plate", "experiment", "batch", "well", "site", "technical_group"],
        "metadata_none": "batch = the image name; the augment_color route; uncorrected states only",
        "metadata_partial": "batch = the plate; the field_aware floor eligible; uncorrected states only",
        "metadata_full": ("as partial (2 or more replicated non-control groups), plus the group silhouette; "
                          "corrected states re-clustered after a Harmony correction"),
    }


def test_card_states_no_prior_run_evidence():
    """A12: the CARD carries no "Earlier runs" notes and no sentence on what earlier runs did or saw; each lever keeps
    its field line and its mechanics; the augmentation values are named by what they do (light / strong)."""
    body = _body()
    assert "Earlier runs" not in body and "earlier runs" not in body and "those runs" not in body
    section = _section("## Levers")
    assert "that tool" not in section and "task-116" not in section and "ssl116" not in section
    for name in LEVERS:
        block = section.split(f"### {name}\n", 1)[1].split("\n### ", 1)[0]
        assert block.startswith("Family: ") and len(block.splitlines()) >= 2, name   # the field line and mechanics
    augmentation = section.split("### augmentation\n", 1)[1].split("\n### ", 1)[0]
    assert "strong, this tool's pipeline" in augmentation and "light, a lighter pipeline" in augmentation
