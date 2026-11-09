"""harness/score.py + harness/figures.py: the transfer endpoint at both label levels (the coarse
morph_class primary, the fine cell_state secondary), the class map, the secondaries, the control
comparison, run and h5ad modes, the metadata level, the audit and the input checks, the figure."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import coverage, figures, hardware, score
from benchmarks.full_HCMA.test._synthetic_chain import (SYNTHETIC_CLASSES, TEMPLATE_TOOLS, build_private,
                                                        build_subset, digest, full_chain, informative,
                                                        random_embedding, write_class_map)


def _cells(labels: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return pd.DataFrame([(f"{i}_S001_{k:05d}", i) for i in labels["image_id"] for k in range(1, n + 1)],
                        columns=["cell_id", "image_id"])


@pytest.fixture()
def root(tmp_path):
    path = tmp_path / "root"
    build_private(path)
    return path


@pytest.fixture(autouse=True)
def class_map(tmp_path, monkeypatch):
    """A synthetic morph_classes.csv (the real one is the master's, approved by the user)."""
    path = write_class_map(tmp_path / "morph_classes.csv")
    monkeypatch.setattr(score, "CLASS_MAP", path)
    return path


@pytest.fixture()
def fast_scib(monkeypatch):
    """The scIB re-clustering without the analyzer (its real path is tested once below)."""
    monkeypatch.setattr(score, "recluster", lambda x, ids, resolutions=score.SCIB_RESOLUTIONS, n_neighbors=15, timings=None: {
        r: np.argmax(np.asarray(x), axis=1).astype(str) for r in resolutions})


def _labels(root):
    return pd.read_csv(root / "private" / "labels.csv", dtype=str, keep_default_na=False)


def _by_type(labels, cells):
    return cells["image_id"].map(labels.set_index("image_id")["cell_state"]).to_numpy(dtype=object)


def _evaluate(cells, x, states, root, images, excluded=None):
    return score.evaluate(cells, x, states, root, images, excluded, class_map=SYNTHETIC_CLASSES)


def test_cluster_transfer_beats_clusters_by_image(root, fast_scib):
    """States that follow the fine type transfer across micrographs; image-specific states stay
    unlabeled. `single` has one micrograph (two panels): unscorable at the fine level; at the coarse
    level (m2 with lonely) even type-true states leave its cells unlabeled."""
    labels = _labels(root)
    cells = _cells(labels)
    x = informative(labels, cells)
    images = list(labels["image_id"])
    good, tables = _evaluate(cells, x, _by_type(labels, cells), root, images)
    by_image, image_tables = _evaluate(cells, x, cells["image_id"].to_numpy(dtype=object), root, images)
    fine = tables["per_type_cluster"].set_index("cell_state")["f1"]
    assert "single" not in fine.index and (fine == 1.0).all() and len(fine) == 5
    assert good["unscorable_classes"]["cell_state"] == [{"cell_state": "single", "n_images": 2, "n_micrographs": 1}]
    assert good["unscorable_classes"]["morph_class"] == [] and good["coverage"]["n_images_unscorable"]["cell_state"] == 2
    assert "single" not in set(tables["per_type"]["cell_state"]) and good["fine"]["n_types"] == 5
    assert "Unscorable types at the cell_state level (fewer than 2 micrographs in the expected set; not scored " \
           "there): single (2 images)." in score.markdown({**good, "mode": "h5ad", "class_map": {}})
    coarse = tables["per_class_cluster"].set_index("morph_class")["f1"]
    assert coarse["m0"] == coarse["m1"] == 1.0 and coarse["m2"] == pytest.approx(0.8)     # m2: P 1, R 32/48
    assert good["endpoint"] == score.PRIMARY == "knn_transfer_macro_f1" and good["endpoint_level"] == "morph_class"
    assert good[score.PRIMARY] > 0.6 and good[score.CLUSTER] == good["cluster_transfer"]["morph_class"]["value"]
    assert good[score.CLUSTER] == pytest.approx(2.8 / 3) == good["cluster_transfer"]["morph_class"]["value"]
    assert good[f"{score.CLUSTER}_fine"] == 1.0 == good["cluster_transfer"]["cell_state"]["value"]
    assert good["cluster_transfer"]["morph_class"]["n_classes"] == 3 and good["cluster_transfer"]["cell_state"]["n_types"] == 5 and good["cluster_transfer"]["morph_class"]["se"] > 0
    assert by_image[score.CLUSTER] == 0.0 == by_image[f"{score.CLUSTER}_fine"]
    assert by_image["secondary"]["unlabeled_fraction"] == 1.0
    assert (image_tables["cells"]["status"] == "unlabeled").all() and (image_tables["per_class_cluster"]["f1"] == 0).all()
    # plain purity cannot tell them apart; the cross-micrograph transfer can
    sec = good["secondary"]
    assert sec["morph_class"]["majority_accuracy"] == 1.0 == by_image["secondary"]["morph_class"]["majority_accuracy"]
    n_single = 2 * 8
    assert sec["morph_class"]["transfer_accuracy"] == pytest.approx(1 - n_single / len(cells))
    assert sec["cell_state"]["transfer_accuracy"] == 1.0 and sec["cell_state"]["n_cells"] == len(cells) - n_single
    assert sec["unlabeled_fraction"] == pytest.approx(n_single / len(cells)) and sec["n_clusters"] == 6
    assert sec["cell_state"]["states_vs_label"]["NMI"] == pytest.approx(1.0)
    assert sec["morph_class"]["states_vs_label"]["NMI"] < 1.0                 # 6 states over 3 classes
    assert sec["cell_state"]["scib"]["nmi_max"] > 0.5 and sec["cell_state"]["scib"]["resolution"] == 0.1
    assert sec["morph_class"]["linear_probe"]["value"] > 0.9 and sec["cell_state"]["linear_probe"]["value"] > 0.9
    assert 0 <= sec["source_residue"]["balanced_accuracy"] <= 1
    assert list(tables["per_class_cluster"].columns)[0] == "morph_class" and list(tables["per_type_cluster"].columns)[0] == "cell_state"
    per_image = tables["per_image"].set_index("image_id")
    assert (per_image.loc[per_image["cell_state"] != "single", "transfer_accuracy"] == 1.0).all()
    assert {"morph_class", "n_correct_fine", "transfer_accuracy_fine", "probe_accuracy_fine"} <= set(per_image.columns)
    for level, n_images in (("morph_class", len(images)), ("cell_state", len(images) - 2)):     # single: unscorable
        nsc = sec[level]["image_nsc"]
        assert nsc["n_images"] == n_images and nsc["nsc"]["n_evaluated"] + nsc["nsc"]["n_removed"] == n_images
        assert sec[level]["image_nsc_accuracy"] == nsc["nsc"]["accuracy"] > 0.9
        assert sec[level]["image_nscb_accuracy"] == nsc["nscb"]["accuracy"]
    assert good["expected_images_sha256"] == score.images_sha256(images) and good["n_expected_images"] == len(images)


def test_states_by_coarse_class_score_at_the_primary_level_only(root, fast_scib):
    """States that merge the fine types of one morph class: the primary is perfect, the fine level
    pays (a c0 cell's other-micrograph votes are mostly c1, and so on)."""
    labels = _labels(root)
    cells = _cells(labels)
    states = pd.Series(_by_type(labels, cells)).map(SYNTHETIC_CLASSES).to_numpy(dtype=object)
    result, tables = _evaluate(cells, informative(labels, cells), states, root, list(labels["image_id"]))
    assert result[score.CLUSTER] == 1.0
    fine = tables["per_type_cluster"].set_index("cell_state")["f1"]
    assert (fine[["c0", "c1", "c2", "c3"]] == 0).all() and "single" not in fine.index
    assert fine["lonely"] == 1.0          # single (unscorable at the fine level) neither votes nor is scored
    assert result[f"{score.CLUSTER}_fine"] == pytest.approx(1 / 5)


def test_unscored_images_leave_the_coarse_level_only(root, fast_scib, tmp_path):
    """cell_states mapped to `unscored` (mixed lineage): their images neither vote nor are scored at
    the coarse level and are no class there; the fine level keeps them unchanged."""
    labels = _labels(root)
    cells = _cells(labels)
    x, images = informative(labels, cells), list(labels["image_id"])
    mapping = {**SYNTHETIC_CLASSES, "lonely": score.UNSCORED}
    fine_states = _by_type(labels, cells)
    result, tables = score.evaluate(cells, x, fine_states, root, images, class_map=mapping)
    # m2 = single only: one micrograph (unscored images do not count), unscorable at the coarse level
    assert list(tables["per_class_cluster"]["morph_class"]) == ["m0", "m1"] == list(tables["per_class"]["morph_class"])
    assert result["unscorable_classes"]["morph_class"] == [{"morph_class": "m2", "n_images": 2, "n_micrographs": 1}]
    assert result[score.CLUSTER] == 1.0
    assert result[f"{score.CLUSTER}_fine"] == 1.0                                      # lonely still scored, F1 1
    assert tables["per_type_cluster"].set_index("cell_state").loc["lonely", "f1"] == 1.0
    cov = result["coverage"]
    assert cov["n_images_unscored_coarse"] == 4 and cov["n_cells_unscored_coarse"] == 32
    assert cov["unscored_cell_states_in_prep"] == ["lonely"]
    lonely = tables["cells"]["cell_state"] == "lonely"
    assert (tables["cells"].loc[lonely, "status"] == score.UNSCORED).all()
    assert (tables["cells"].loc[lonely, "morph_class"] == score.UNSCORED).all()
    per_image = tables["per_image"].set_index("image_id")
    rows = per_image[per_image["cell_state"] == "lonely"]
    assert rows["transfer_accuracy"].isna().all() and (rows["transfer_accuracy_fine"] == 1.0).all()
    assert not rows["probe_scorable"].any()
    single = tables["cells"]["cell_state"] == "single"
    assert (tables["cells"].loc[single, "status"] == score.UNSCORABLE).all()
    assert (tables["cells"].loc[single, "knn_status"] == score.UNSCORABLE).all()
    assert (tables["cells"].loc[single, "morph_class"] == "m2").all()
    assert result["secondary"]["morph_class"]["n_cells"] == len(cells) - 32 - 16
    assert result["secondary"]["cell_state"]["n_cells"] == len(cells) - 16          # single: unscorable at the fine level
    # neither voters: single (mapped to m1 here, with c2 and c3) shares its state with lonely; with lonely
    # scored (m2) single's cells are predicted m2, with lonely unscored they have no other micrograph to learn from
    merged = np.where(np.isin(fine_states, ["lonely", "single"]), "ls", fine_states).astype(object)
    voted, voted_tables = score.evaluate(cells, x, merged, root, images, class_map={**SYNTHETIC_CLASSES, "single": "m1"})
    alone, alone_tables = score.evaluate(cells, x, merged, root, images, class_map={**mapping, "single": "m1"})
    assert (voted_tables["cells"].loc[voted_tables["cells"]["cell_state"] == "single", "predicted"] == "m2").all()
    assert (alone_tables["cells"].loc[alone_tables["cells"]["cell_state"] == "single", "status"] == "unlabeled").all()
    # the class map record and the comparison
    path = write_class_map(tmp_path / "with_unscored.csv", mapping)
    check = score.class_map_check(root, "full", path)
    assert check["ok"] and check["classes"] == ["m0", "m1", "m2"] and check["unscored_cell_states"] == ["lonely"]
    assert score._public(check)["unscored_cell_states"] == ["lonely"]
    same = score.compare(tables, tables)[0]
    assert same["n_classes"] == 2 and same["knn_transfer_accuracy"]["n_images"] == len(images) - 4 - 2
    used = figures.umap_scored(tmp_path / "unscored.png", np.random.default_rng(0).normal(size=(len(cells), 2)),
                               tables["cells"]["morph_class"], tables["cells"]["state"], tables["cells"]["status"],
                               tables["cells"]["source"], all_classes=check["classes"], all_sources=["src0", "src1", "src2"])
    assert used["classes"][score.UNSCORED] == figures.UNSCORED_COLOUR and list(used["classes"])[-1] == score.UNSCORED


def test_partial_precision_and_recall_by_hand():
    """A(x) 3 + B(x) 2 + C(y) 1 in state 0; C(y) 2 + D(y) 2 in state 1; D(y) 1 alone in state 2.
    x: P 5/6, R 1, F1 10/11; y: P 1, R 4/6, F1 0.8; a type z without cells: F1 0."""
    images = np.array(list("AAABBC") + list("CCDD") + ["D"], dtype=object)
    types = np.array(list("xxxxxy") + list("yyyy") + ["y"], dtype=object)
    states = np.array(["0"] * 6 + ["1"] * 4 + ["2"], dtype=object)
    pred = score.transfer_predictions(images, types, states)
    assert list(pred) == ["x"] * 6 + ["y"] * 4 + [score.UNLABELED]
    image_types = pd.Series({"A": "x", "B": "x", "C": "y", "D": "y", "Z": "z"})
    n_cells = pd.Series(images).value_counts()
    table = score.per_type_table(["x", "y", "z"], types, pred, image_types, n_cells).set_index("cell_state")
    assert table.loc["x", "precision"] == pytest.approx(5 / 6) and table.loc["x", "recall"] == 1.0
    assert table.loc["x", "f1"] == pytest.approx(10 / 11)
    assert table.loc["y", "precision"] == 1.0 and table.loc["y", "recall"] == pytest.approx(4 / 6)
    assert table.loc["y", "f1"] == pytest.approx(0.8) and table.loc["y", "n_unlabeled"] == 1
    assert table.loc["z", "f1"] == 0.0 and table.loc["z", "n_images_with_cells"] == 0
    assert table.loc[["x", "y"], "f1"].mean() == pytest.approx(0.854545, abs=1e-6)
    assert table["f1"].mean() == pytest.approx(0.569697, abs=1e-6)
    keyed = score.per_type_table(["x"], types, pred, image_types, n_cells, key="morph_class")
    assert list(keyed.columns)[0] == "morph_class"


def test_panels_of_one_micrograph_do_not_vote_for_each_other(root, fast_scib):
    """build_private's micrographs have two panels each: states made of one micrograph's panels
    transfer nothing (a micrograph signal is not a cell-type signal), at either level."""
    labels = _labels(root)
    cells = _cells(labels)
    parent = cells["image_id"].map(labels.set_index("image_id")["parent_id"]).to_numpy(dtype=object)
    result, tables = _evaluate(cells, informative(labels, cells), parent, root, list(labels["image_id"]))
    assert result[score.CLUSTER] == 0.0 == result[f"{score.CLUSTER}_fine"]
    assert (tables["cells"]["status"] == "unlabeled").all()
    by_image = score.transfer_predictions(cells["image_id"].to_numpy(dtype=object), _by_type(labels, cells), parent)
    assert (by_image == _by_type(labels, cells)).all()     # judged image by image, the same states score


def test_unknown_type_cells_are_left_out(root, fast_scib):
    labels = _labels(root)
    extra = {"image_id": "img_0999", "parent_id": "par_0999", "source": "src0", "cell_type": "",
             "cell_state": "unknown", "organ_category": ""}
    pd.concat([labels, pd.DataFrame([extra])]).to_csv(root / "private" / "labels.csv", index=False)
    cells = pd.concat([_cells(labels), pd.DataFrame({"cell_id": [f"img_0999_S001_{k:05d}" for k in range(3)],
                                                     "image_id": "img_0999"})], ignore_index=True)
    states = np.array(_by_type(labels, cells.iloc[:-3]).tolist() + ["u"] * 3, dtype=object)
    states[cells.index[cells["image_id"] == "img_0001"]] = "u"     # img_0001 shares its state only with it
    images = [*labels["image_id"], "img_0999"]
    result, tables = _evaluate(cells, informative(pd.concat([labels, pd.DataFrame([extra])]), cells), states,
                               root, images)
    assert result["coverage"]["n_cells_unknown_type_left_out"] == 3
    assert "unknown" not in set(tables["per_type"]["cell_state"]) and "img_0999" not in set(tables["cells"]["image_id"])
    assert (tables["cells"].loc[tables["cells"]["image_id"] == "img_0001", "status"] == "unlabeled").all()
    assert result["coverage"]["n_images"] == len(images) and result["coverage"]["n_images_known_type"] == len(labels)


def test_excluded_cells_cannot_raise_recall(root, fast_scib):
    """The QC's probe: two images per type share a state, each image keeps 4 cells in its own state.
    Excluding those single-image cells cannot raise the cluster-level transfer (they count as
    unlabeled) nor any class's kNN recall (they are misses); the kNN value can rise through precision
    (test_an_image_left_out_costs_the_sample_cap)."""
    labels = _labels(root)
    pick = {"c0": labels.loc[labels["cell_state"] == "c0", "image_id"].iloc[[0, 2]].tolist(),
            "c2": labels.loc[labels["cell_state"] == "c2", "image_id"].iloc[[0, 2]].tolist()}
    rows, states = [], []
    for t, (a, b) in pick.items():
        for image in (a, b):
            rows += [(f"{image}_S001_{k:05d}", image) for k in range(10)]
            states += [f"shared_{t}"] * 6 + [f"own_{image}"] * 4
    cells = pd.DataFrame(rows, columns=["cell_id", "image_id"])
    images = [i for v in pick.values() for i in v]
    x = informative(labels, cells)
    before, _ = _evaluate(cells, x, np.array(states, dtype=object), root, images)
    keep = ~pd.Series(states).str.startswith("own_").to_numpy()
    after, tables = _evaluate(cells[keep].reset_index(drop=True), x[keep], np.array(states, dtype=object)[keep],
                              root, images, excluded=list(cells.loc[~keep, "cell_id"]))
    for key, level in ((score.CLUSTER, "morph_class"), (f"{score.CLUSTER}_fine", "cell_state")):
        assert before[key] == pytest.approx(0.75) and after[key] == pytest.approx(before[key])
        assert after["secondary"][level][f"{score.CLUSTER}_delivered_only"] == pytest.approx(1.0)
    assert before["secondary"]["unlabeled_fraction"] == pytest.approx(0.4) == after["secondary"]["unlabeled_fraction"]
    assert after["cluster_transfer"]["morph_class"]["n_excluded_cells_counted_unlabeled"] == 16
    # the kNN primary samples the excluded cells too: they are misses, the delivered-only value leaves them out
    assert after["sampling"]["knn"]["n_excluded_cells_sampled"] == 16
    assert after[score.PRIMARY] <= after[f"{score.PRIMARY}_delivered_only"]
    _, before_tables = _evaluate(cells, x, np.array(states, dtype=object), root, images)
    for name, key in (("per_class", "morph_class"), ("per_type", "cell_state")):
        recall = tables[name].set_index(key)["recall"]
        assert (recall <= before_tables[name].set_index(key)["recall"] + 1e-12).all()
    assert tables["per_class_cluster"]["n_excluded"].tolist() == [8, 8] and tables["per_type_cluster"]["n_excluded"].tolist() == [8, 8]
    assert tables["per_image"].set_index("image_id").loc[images[0], "n_excluded"] == 4



def test_an_image_left_out_costs_the_sample_cap(root, fast_scib):
    """The QC's probe: one c0 image (class m0) looks like c2 (class m1). Excluding its cells keeps
    m0's recall and raises m1's precision (the accepted precision gain); leaving the image out of
    the delivery counts it as MAX_CELLS misses of m0 at both levels, so it cannot raise the primary
    above the all-delivered value of the same embedding."""
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    x = informative(labels, cells, signal=6.0)
    odd = labels.loc[labels["cell_state"] == "c0", "image_id"].iloc[0]
    like = labels.loc[labels["cell_state"] == "c2", "image_id"].iloc[0]
    is_odd = (cells["image_id"] == odd).to_numpy()
    x[is_odd] = x[(cells["image_id"] == like).to_numpy()] + 0.01
    states = _by_type(labels, cells)
    full, full_tables = _evaluate(cells, x, states, root, images)
    excluded, excluded_tables = _evaluate(cells[~is_odd].reset_index(drop=True), x[~is_odd], states[~is_odd], root,
                                          images, excluded=list(cells.loc[is_odd, "cell_id"]))
    left_out, left_tables = _evaluate(cells[~is_odd].reset_index(drop=True), x[~is_odd], states[~is_odd], root, images)
    m0 = lambda t: t["per_class"].set_index("morph_class").loc["m0"]       # noqa: E731
    m1 = lambda t: t["per_class"].set_index("morph_class").loc["m1"]       # noqa: E731
    assert m0(excluded_tables)["recall"] <= m0(full_tables)["recall"] + 1e-12
    assert m1(excluded_tables)["precision"] > m1(full_tables)["precision"]
    assert excluded[score.PRIMARY] > full[score.PRIMARY]                     # exclusion can raise precision
    assert left_out[score.PRIMARY] <= full[score.PRIMARY]
    assert left_out[f"{score.PRIMARY}_fine"] <= full[f"{score.PRIMARY}_fine"]
    for block in ("primary", "fine"):
        assert left_out[block]["n_images_missing"] == 1 and left_out[block]["n_missing_queries"] == score.MAX_CELLS
        assert full[block]["n_images_missing"] == 0 == excluded[block]["n_images_missing"]
    assert m0(left_tables)["n_missing_queries"] == score.MAX_CELLS and m0(left_tables)["n_unlabeled"] >= score.MAX_CELLS
    assert left_tables["per_type"].set_index("cell_state").loc["c0", "n_missing_queries"] == score.MAX_CELLS
    assert "n_missing_queries" not in left_tables["per_class_cluster"]
    # the delivered-only value leaves the missing image out; the per-image counts carry the cap
    assert left_out[f"{score.PRIMARY}_delivered_only"] > left_out[score.PRIMARY]
    row = left_tables["per_image"].set_index("image_id").loc[odd]
    assert row["n_knn_scored"] == score.MAX_CELLS == row["n_knn_scored_fine"] and row["n_knn_correct"] == 0
    assert left_out["knn_transfer_accuracy"] < full["knn_transfer_accuracy"]


def test_knn_predictions_by_hand():
    """Other-micrograph restriction, the tie rule, no candidates, fewer than k candidates."""
    x = np.array([[1, 0], [1, 0.05], [0, 1], [0.7, 0.7]], dtype=float)
    groups = np.array(["g1", "g1", "g2", "g2"], dtype=object)
    labels = np.array(["x", "y", "y", "x"], dtype=object)
    pred = score.knn_predict(x, groups, labels, k=1)
    assert pred[0] == "x"            # its own micrograph's (y, nearest) never votes: c3 (x) is the nearest other
    assert list(pred[1:]) == ["x", "y", "y"]   # c2 and c3: their nearest other-micrograph cell is c1 (y)
    # one vote each: b (nearer, the larger weight) wins, not the lexicographically smaller a
    tie = score.knn_predict(np.array([[1, 0], [1, 0.1], [1, 0.3]]), np.array(["q", "n1", "n2"], dtype=object),
                            np.array(["c", "b", "a"], dtype=object), k=2)
    assert tie[0] == "b"
    alone = score.knn_predict(np.array([[1, 0], [0, 1]]), np.array(["g", "g"], dtype=object),
                              np.array(["x", "y"], dtype=object))
    assert list(alone) == [score.UNLABELED] * 2
    few = score.knn_predict(np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]]), np.array(["q", "a", "b", "c"], dtype=object),
                            np.array(["z", "x", "x", "y"], dtype=object), k=15)
    assert few[0] == "x"             # all 3 candidates vote: x twice and nearer, y once
    # equal similarities: the lower input row is the nearer neighbour, whatever torch.topk returns
    twins = np.array([[1, 0], [1, 0], [1, 0]], dtype=float)
    pair = np.array(["q", "r", "r"], dtype=object)
    assert score.knn_predict(twins, pair, np.array(["z", "a", "b"], dtype=object), k=1)[0] == "a"
    assert score.knn_predict(twins, pair, np.array(["z", "b", "a"], dtype=object), k=1)[0] == "b"
    flat = np.ones((40, 4))       # every similarity equal: the k lowest rows of other groups vote, the lowest is nearest
    groups = np.array([f"g{i // 2}" for i in range(40)], dtype=object)
    labels = np.array([f"l{i:02d}" for i in range(40)], dtype=object)       # equal weights: every vote ties
    pred = score.knn_predict(flat, groups, labels, k=5)
    assert pred[0] == "l02" and pred[2] == "l00" and pred[39] == "l00"


def _ray(cosine: float) -> list[float]:
    return [cosine, float(np.sqrt(1 - cosine ** 2))]


def test_the_dino_weighted_vote_by_hand():
    """Each neighbour votes exp(cos / 0.07) for its label (DINO eval_knn.py), summed per label."""
    x = np.array([[1.0, 0.0], _ray(0.99), _ray(0.99), _ray(0.80), _ray(0.80), _ray(0.80)])
    groups = np.array(["q", "g1", "g2", "g3", "g4", "g5"], dtype=object)
    labels = np.array(["z", "a", "a", "b", "b", "b"], dtype=object)
    # a nearer minority beats a farther majority: 2 exp(0.99 / 0.07) = 2.8e6 against 3 exp(0.80 / 0.07) = 2.8e5
    assert score.knn_predict(x, groups, labels, k=5)[0] == "a" == _brute_knn(x, groups, labels, 5)[0]
    assert _brute_knn(x, groups, labels, 5, temperature=None)[0] == "b"                  # the plain count
    assert score.knn_predict(x, groups, labels, k=5, temperature=100.0)[0] == "b"        # flat weights: the count
    # ... and a majority that is nearly as near wins: 3 exp(0.98 / 0.07) = 3.6e6 against 2.8e6
    x[3:] = _ray(0.98)
    assert score.knn_predict(x, groups, labels, k=5)[0] == "b" == _brute_knn(x, groups, labels, 5)[0]
    # equal summed weights (one vote each at the same similarity): the nearest, i.e. the lower row
    twins = np.array([[1.0, 0.0], _ray(0.9), _ray(0.9)])
    two = np.array(["q", "g1", "g2"], dtype=object)
    assert score.knn_predict(twins, two, np.array(["z", "a", "b"], dtype=object), k=2)[0] == "a"
    assert score.knn_predict(twins, two, np.array(["z", "b", "a"], dtype=object), k=2)[0] == "b"
    # a cell outside a level is neither a query nor a reference there
    out = score.knn_predict_levels(np.array([[1, 0], [1, 0], [0, 1]]), np.array(["g0", "g1", "g2"], dtype=object),
                                   {"level": np.array(["a", None, "b"], dtype=object)}, k=1)["level"]
    assert out[0] == "b" and out[1] is None


def _brute_knn(x, groups, labels, k, temperature=score.KNN_TEMPERATURE):
    """The rule written out cell by cell: (-similarity, index) order, other groups, votes
    exp(similarity / temperature) summed per label in that order (`temperature=None`: the plain
    count), nearest tied label."""
    v = np.asarray(x, dtype=np.float32)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    out = []
    for i in range(len(x)):
        sim = v @ v[i]
        others = [j for j in np.lexsort((np.arange(len(x)), -sim)) if groups[j] != groups[i]][:k]
        if not others:
            out.append(score.UNLABELED)
            continue
        votes: dict = {}
        for j in others:
            weight = 1.0 if temperature is None else float(np.exp(np.float64(sim[j]) / temperature))
            votes[labels[j]] = votes.get(labels[j], 0.0) + weight
        best = max(votes.values())
        out.append(next(labels[j] for j in others if votes[labels[j]] == best))
    return np.array(out, dtype=object)


def test_knn_matches_the_rule_across_blocks_and_levels(monkeypatch):
    """Several query blocks, two levels in one pass (a level that holds a subset of the cells), the
    cell-by-cell rule."""
    monkeypatch.setattr(score, "KNN_ROWS_BUDGET", 64 * 300)        # 300 cells: blocks of 64 rows
    rng = np.random.default_rng(5)
    x = rng.normal(size=(300, 8)).astype(np.float32)
    groups = np.array([f"g{i % 37}" for i in range(300)], dtype=object)
    fine = np.array([f"t{i % 5}" for i in rng.permutation(300)], dtype=object)
    coarse = np.array([None if t == "t4" else ("A" if t in ("t0", "t1") else "B") for t in fine], dtype=object)
    both = score.knn_predict_levels(x, groups, {"fine": fine, "coarse": coarse}, k=7)
    assert list(both["fine"]) == list(_brute_knn(x, groups, fine, 7)) == list(score.knn_predict(x, groups, fine, k=7))
    member = np.array([c is not None for c in coarse])
    assert all(p is None for p in both["coarse"][~member])
    assert list(both["coarse"][member]) == list(_brute_knn(x[member], groups[member], coarse[member], 7))


def test_knn_sees_classes_the_cluster_transfer_misses(root, fast_scib):
    """Image-split clusters score 0 at the cluster level; an embedding that puts same-class cells of
    different micrographs together still transfers at the cell level. An embedding that separates
    only micrographs transfers near chance."""
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    split = cells["image_id"].to_numpy(dtype=object)
    good, tables = _evaluate(cells, informative(labels, cells, signal=6.0), split, root, images)
    assert good[score.CLUSTER] == 0.0 and good[score.PRIMARY] > 0.9 and good[f"{score.PRIMARY}_fine"] > 0.6
    assert good["primary"]["k"] == score.KNN_K == 20 and good["sampling"]["knn"]["n_cells"] == len(cells)
    assert good["primary"]["vote"] == "dino_weighted" and good["primary"]["temperature"] == 0.07
    assert good["knn_transfer_median_f1"] == good["primary"]["median_f1"] == float(np.median(tables["per_class"]["f1"]))
    assert good["knn_transfer_median_f1_fine"] == float(np.median(tables["per_type"]["f1"]))
    parents = cells["image_id"].map(labels.set_index("image_id")["parent_id"])
    centre = {p: np.random.default_rng(n).normal(size=16) * 6 for n, p in enumerate(sorted(set(parents)))}
    by_micrograph = np.stack([centre[p] for p in parents]) + np.random.default_rng(9).normal(size=(len(cells), 16))
    chance, _ = _evaluate(cells, by_micrograph.astype(np.float32), split, root, images)
    assert chance[score.PRIMARY] < 0.6 < good[score.PRIMARY]
    status = tables["cells"]["knn_status"]
    assert set(status) <= {"correct", "wrong", score.NO_PREDICTION} and (status == "correct").mean() > 0.9


def test_knn_excluded_cells_unscored_references_and_empty_classes(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    x = informative(labels, cells, signal=6.0)
    states = _by_type(labels, cells)
    # excluded cells are sampled with the image's delivered cells and are misses
    c0 = cells["image_id"].isin(labels.loc[labels["cell_state"] == "c0", "image_id"]) & (cells["cell_id"].str[-1] > "4")
    kept, xs = cells[~c0].reset_index(drop=True), x[~c0.to_numpy()]
    result, tables = _evaluate(kept, xs, states[~c0.to_numpy()], root, images, excluded=list(cells.loc[c0, "cell_id"]))
    row = tables["per_type"].set_index("cell_state").loc["c0"]
    assert row["n_excluded"] == int(c0.sum()) and row["recall"] <= 1 - c0.sum() / (row["n_cells"] + row["n_excluded"]) + 1e-9
    assert result[f"{score.PRIMARY}_fine"] < result[f"{score.PRIMARY}_fine_delivered_only"]
    # unscored images are no references: lonely sits on top of single (mapped to m1 here) in the embedding
    near = x.copy()
    lonely = cells["image_id"].isin(labels.loc[labels["cell_state"] == "lonely", "image_id"]).to_numpy()
    single = cells["image_id"].isin(labels.loc[labels["cell_state"] == "single", "image_id"]).to_numpy()
    near[lonely] = x[single][0]
    _, voted = score.evaluate(cells, near, states, root, images, class_map={**SYNTHETIC_CLASSES, "single": "m1"})
    mapping = {**SYNTHETIC_CLASSES, "lonely": score.UNSCORED, "single": "m1"}
    alone, alone_tables = score.evaluate(cells, near, states, root, images, class_map=mapping)
    assert (voted["cells"].loc[single, "knn_status"] == "wrong").mean() > 0.5               # pulled to lonely's m2
    assert voted["per_class"].set_index("morph_class").loc["m2", "n_predicted"] > 32
    assert (alone_tables["cells"].loc[lonely, "knn_status"] == score.UNSCORED).all()
    assert list(alone_tables["per_class"]["morph_class"]) == ["m0", "m1"]               # m2 is not a class there


def _smoke_like(labels):
    """lonely's micrograph A (2 images, 8 cells each) and its partner micrograph B (2 images, 2 cells
    each): lonely sits near c0 in the embedding, its own cells nearest each other."""
    lonely = labels[labels["cell_state"] == "lonely"]
    a, b = sorted(lonely["parent_id"].unique())
    sizes = {i: (8 if p == a else 2) for i, p in zip(lonely["image_id"], lonely["parent_id"])}
    rows = [(f"{i}_S001_{k:05d}", i) for i in labels["image_id"] for k in range(sizes.get(i, 8))]
    cells = pd.DataFrame(rows, columns=["cell_id", "image_id"])
    state = cells["image_id"].map(labels.set_index("image_id")["cell_state"])
    centre = {f"c{c}": 6.0 * np.eye(16)[c] for c in range(4)}
    centre.update(lonely=6.0 * np.eye(16)[0] + 5.0 * np.eye(16)[4], single=6.0 * np.eye(16)[5])
    x = np.stack([centre[t] for t in state]) + 0.3 * np.random.default_rng(4).normal(size=(len(cells), 16))
    in_a = cells["image_id"].map(labels.set_index("image_id")["parent_id"]).eq(a).to_numpy() & (state == "lonely").to_numpy()
    return cells, x.astype(np.float32), state.to_numpy(dtype=object), in_a


def test_a_small_near_partner_micrograph_wins_the_dino_weighted_vote(root, fast_scib):
    """The smoke's case: a class whose partner micrograph has only 4 cells. The cells of the other
    micrograph find those 4 cells nearest (cos ~0.97), then 16 cells of the neighbouring class c0
    (cos ~0.77): the plain count says c0, the exp(cos / 0.07) weights say lonely."""
    labels = _labels(root)
    cells, x, state, in_a = _smoke_like(labels)
    result, tables = _evaluate(cells, x, state, root, list(labels["image_id"]))
    assert result["fine"]["vote"] == score.VOTE == result["primary"]["vote"] == "dino_weighted"
    assert "DINO's weighted kNN vote" in result["primary"]["rule"] and "exp(cosine similarity / 0.07)" in result["primary"]["rule"]
    fine = tables["per_type"].set_index("cell_state")
    assert fine.loc["lonely", "recall"] == 1.0 and fine.loc["lonely", "precision"] == 1.0
    assert (tables["cells"].loc[in_a, "knn_status"] == "correct").all()              # m2 at the coarse level too
    scored = state != "single"                                                         # single: unscorable (fine)
    groups = cells["image_id"].map(labels.set_index("image_id")["parent_id"]).to_numpy(dtype=object)
    plain = _brute_knn(x[scored], groups[scored], state[scored], score.KNN_K, temperature=None)
    assert (plain[in_a[scored]] == "c0").all()                                        # the plain vote loses them


def test_scorability_comes_from_the_expected_images_not_the_delivery(root, fast_scib):
    """single (one micrograph) is unscorable at the fine level; delivering nothing of one of lonely's
    two micrographs leaves lonely scorable (its images are missing-image misses)."""
    labels = _labels(root)
    cells, x, state, in_a = _smoke_like(labels)
    lonely = labels[labels["cell_state"] == "lonely"]
    b = sorted(lonely["parent_id"].unique())[1]
    dropped = cells["image_id"].isin(lonely.loc[lonely["parent_id"] == b, "image_id"]).to_numpy()
    full, _ = _evaluate(cells, x, state, root, list(labels["image_id"]))
    part, part_tables = _evaluate(cells[~dropped].reset_index(drop=True), x[~dropped], state[~dropped], root,
                                  list(labels["image_id"]))
    assert part["unscorable_classes"] == full["unscorable_classes"]
    assert [c["cell_state"] for c in part["unscorable_classes"]["cell_state"]] == ["single"]
    row = part_tables["per_type"].set_index("cell_state").loc["lonely"]
    assert row["n_missing_queries"] == 2 * score.MAX_CELLS and row["recall"] == 0.0     # no lonely reference left
    assert part["fine"]["n_types"] == full["fine"]["n_types"] == 5
    assert part[f"{score.PRIMARY}_fine"] < full[f"{score.PRIMARY}_fine"]
    assert "lonely" in set(part_tables["per_type_cluster"]["cell_state"])               # the cluster level likewise

def test_ties_take_the_smallest_type_and_lone_images_are_unlabeled():
    images = np.array(["i1", "i1", "i2", "i3", "i4"], dtype=object)
    types = np.array(["x", "x", "b", "a", "z"], dtype=object)
    states = np.array(["0", "0", "0", "0", "1"], dtype=object)
    pred = score.transfer_predictions(images, types, states)
    # i1: other images give b 1, a 1 -> a; i2: x 2 -> x; i3: x 2 -> x; i4 alone in cluster 1
    assert list(pred) == ["a", "a", "x", "x", score.UNLABELED]


def test_a_type_without_delivered_cells_has_f1_zero(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    lonely = set(labels.loc[labels["cell_state"] == "lonely", "image_id"])
    cells = cells[~cells["image_id"].isin(lonely)].reset_index(drop=True)
    result, tables = _evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, list(labels["image_id"]))
    row = tables["per_type_cluster"].set_index("cell_state").loc["lonely"]
    assert row["n_cells"] == 0 and row["f1"] == 0.0 and row["n_images"] == 4 and row["n_images_with_cells"] == 0
    assert result["fine"]["n_types_without_cells"] == 1 and result["primary"]["n_classes_without_cells"] == 0
    knn = tables["per_type"].set_index("cell_state").loc["lonely"]
    assert knn["f1"] == 0.0 and knn["n_cells"] == 0                     # the kNN level: a type without cells, F1 0
    assert result[f"{score.CLUSTER}_fine"] == pytest.approx(4 / 5)        # lonely 0; single (one micrograph) unscorable
    assert result[score.CLUSTER] == pytest.approx(2 / 3)                   # m2 = lonely (none) + single (unlabeled)
    assert result["coverage"]["n_images_with_cells"] == len(labels) - 4


def test_per_class_bootstrap_and_the_comparison(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    x, images = informative(labels, cells), list(labels["image_id"])
    rng = np.random.default_rng(3)
    _, good = _evaluate(cells, x, _by_type(labels, cells), root, images)
    _, bad = _evaluate(cells, random_embedding(cells), rng.integers(0, 3, len(cells)).astype(str), root, images)
    better, table = score.compare(good, bad)
    assert better["level"] == "morph_class" and better["n_classes"] == len(table) == 3
    assert better["verdict"] == "GAIN" and better["delta"] > 2 * better["se"] > 0
    assert better["delta"] == pytest.approx(better["value"] - better["control_value"])
    assert better["fine"]["n_types"] == 5 and better["fine"]["delta"] > 0 and better["fine"]["se"] > 0
    assert better["knn_transfer_accuracy"]["delta"] > 0 and better["knn_transfer_accuracy"]["se"] > 0
    assert score.compare(bad, good)[0]["verdict"] == "LOSS"
    same = score.compare(good, good)[0]
    assert same["verdict"] == "NO_RESOLVABLE_DIFFERENCE" and same["delta"] == 0 and same["verdict_note"] == "se_zero"
    one = {**good, "per_class": good["per_class"].iloc[:1]}
    lower = {**bad, "per_class": bad["per_class"].iloc[:1].assign(f1=0.0)}
    single = score.compare(one, lower)[0]          # one class: SE 0, never GAIN
    assert single["delta"] > 0 and single["verdict"] == "NO_RESOLVABLE_DIFFERENCE" and single["verdict_note"] == "se_zero"
    f1 = good["per_class"]["f1"].to_numpy() - bad["per_class"]["f1"].to_numpy()
    boot = score.type_bootstrap(f1)
    assert len(boot) == score.N_BOOT and np.array_equal(boot, score.type_bootstrap(f1))
    assert better["se"] == pytest.approx(np.std(boot, ddof=1))
    # the same difference for every class: float noise in the bootstrap is SE 0 (QC4's case)
    shifted = score.compare({**good, "per_class": good["per_class"].iloc[:3].assign(f1=[0.8, 0.5, 0.3])},
                            {**bad, "per_class": bad["per_class"].iloc[:3].assign(f1=[0.7, 0.4, 0.2])})[0]
    assert shifted["se"] == 0.0 and shifted["verdict"] == "NO_RESOLVABLE_DIFFERENCE" and shifted["verdict_note"] == "se_zero"


def test_linear_probe_at_both_levels_uses_the_fixed_folds(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels, n=3)
    many = pd.DataFrame({"cell_id": [f"img_0001_S001_{k:05d}" for k in range(4, 154)], "image_id": "img_0001"})
    cells = pd.concat([cells, many], ignore_index=True)
    first = score.sample_cells(cells)
    assert (cells["image_id"].to_numpy()[first] == "img_0001").sum() == score.MAX_CELLS
    shuffled = cells.sample(frac=1, random_state=3).reset_index(drop=True)
    assert sorted(cells["cell_id"].to_numpy()[first]) == sorted(shuffled["cell_id"].to_numpy()[score.sample_cells(shuffled)])
    small = set(cells["cell_id"].to_numpy()[score.sample_cells(cells, score.PROBE_CELLS)])
    assert small <= set(cells["cell_id"].to_numpy()[first])
    result, tables = _evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, list(labels["image_id"]))
    per_image = tables["per_image"].set_index("image_id")
    assert per_image.loc["img_0001", "probe_n_sampled"] == 100 and per_image.loc["img_0001", "n_cells"] == 153
    lonely = per_image[per_image["cell_state"] == "lonely"]
    for suffix in ("", "_fine"):           # lonely (morph class m2 too) sits in fold 0 only
        assert not lonely[f"probe_scorable{suffix}"].any() and lonely[f"probe_accuracy{suffix}"].isna().all()
        assert not per_image[per_image["cell_state"] == "single"][f"probe_scorable{suffix}"].any()
    for level in ("morph_class", "cell_state"):
        probe = result["secondary"][level]["linear_probe"]
        assert probe["n_unscorable_images"] == 4 and probe["n_scored_images"] == len(labels) - 6
    assert result["secondary"]["morph_class"]["linear_probe"]["n_classes"] == 2      # m0, m1 (m2 unscorable)
    assert result["sampling"]["linear_probe"]["n_cells"] == len(cells) - 153 + 100
    # the scIB-style re-clustering runs on the kNN sample (img_0001: 100 of its 153 cells)
    assert result["secondary"]["scib_n_cells"] == result["sampling"]["knn"]["n_cells"] == len(cells) - 53
    assert result["secondary"]["morph_class"]["scib"]["sampled"] is True and result["secondary"]["scib_sampled"]
    assert result["sampling"]["source_residue"]["n_cells"] == len(cells) - 153 + 20


def test_best_cluster_f1_on_a_known_clustering():
    truth = np.array(["a", "a", "a", "b", "b", "c"], dtype=object)
    clusterings = {0.1: np.array(["0"] * 6), 0.5: np.array(["0", "0", "1", "2", "2", "3"])}
    table = score.best_cluster_f1(truth, clusterings).set_index("label")
    # 0.1: one cluster of 6: a 6/9, b 4/8, c 2/7; 0.5: a's best cluster {a, a} 2*2/(2+3), b {b, b} 1, c {c} 1
    assert table.loc["a", "best_f1"] == pytest.approx(0.8) and table.loc["a", "resolution"] == 0.5
    assert table.loc["a", "cluster"] == "0" and table.loc["a", "n_cells"] == 3
    assert table.loc["b", "best_f1"] == 1.0 and table.loc["c", "best_f1"] == 1.0
    one = score.best_cluster_f1(truth, {0.1: clusterings[0.1]}).set_index("label")["best_f1"]
    assert one.tolist() == pytest.approx([6 / 9, 4 / 8, 2 / 7])


def test_isolated_label_f1_and_best_cluster_f1_in_the_result(root, fast_scib):
    """Isolated labels = the labels in the fewest sources of the expected images; their mean
    best-cluster F1. `single` (one micrograph, unscorable at the fine level) still gets its
    best-cluster F1: a clean cluster without a cross-micrograph partner."""
    labels = _labels(root)
    cells = _cells(labels)
    result, tables = _evaluate(cells, informative(labels, cells, signal=6.0), _by_type(labels, cells), root,
                               list(labels["image_id"]))
    best = tables["best_cluster_f1"]
    assert set(best["level"]) == {"morph_class", "cell_state"}
    fine = best[best["level"] == "cell_state"].set_index("label")
    n_sources = labels.groupby("cell_state")["source"].nunique()
    assert fine["n_sources"].to_dict() == n_sources.to_dict()
    isolated = sorted(n_sources[n_sources == n_sources.min()].index)
    block = result["secondary"]["cell_state"]
    assert block["isolated_label"]["isolated_labels"] == isolated == fine.index[fine["isolated"]].tolist()
    assert block["isolated_label_f1"] == pytest.approx(fine.loc[isolated, "best_f1"].mean())
    assert fine.loc["single", "best_f1"] > 0.9 and "single" not in set(tables["per_type"]["cell_state"])
    assert "Single-batch purity" in block["isolated_label"]["rule"]
    coarse = best[best["level"] == "morph_class"]
    assert list(coarse["label"]) == ["m0", "m1", "m2"]
    md = score.markdown({**result, "mode": "h5ad", "class_map": {}})
    assert "secondary `isolated_label_f1` (scIB; single-batch purity" in md and "secondary image-level 1-NN: NSC" in md


def test_image_nsc_and_nscb_remove_queries_without_a_partner():
    """A1, A2 (class a, source s1), B1 (b, s2), B2 (b, s3), C1 (c, s1); every image its own
    micrograph. C1 has no other c image: removed under both rules. NSCB also removes A1 and A2 (their
    only partner shares their source). B2's nearest image is C1: wrong under both rules."""
    unit = lambda v: np.asarray(v, float) / np.linalg.norm(v)              # noqa: E731
    profiles = np.stack([unit([1, 0]), unit([0.9, 0.1]), unit([0, 1]), unit([0.6, 0.8]), unit([0.7, 0.7])])
    labels = np.array(["a", "a", "b", "b", "c"], dtype=object)
    parents = np.array(["p1", "p2", "p3", "p4", "p5"], dtype=object)
    sources = np.array(["s1", "s1", "s2", "s3", "s1"], dtype=object)
    out = score.image_nsc(profiles, labels, parents, sources)
    assert out["nsc"] == {"accuracy": 0.75, "n_evaluated": 4, "n_removed": 1, "n_correct": 3}
    assert out["nscb"] == {"accuracy": 0.5, "n_evaluated": 2, "n_removed": 3, "n_correct": 1}
    # other micrographs only: a same-micrograph twin of a different class is never the neighbour
    same = score.image_nsc(np.stack([unit([1, 0]), unit([1, 0]), unit([0.9, 0.1])]),
                           np.array(["a", "b", "a"], dtype=object), np.array(["p1", "p1", "p2"], dtype=object),
                           np.array(["s1", "s2", "s3"], dtype=object))
    assert same["nsc"]["n_correct"] == 2 == same["nsc"]["n_evaluated"]      # 0 -> 2 (not its twin 1); 2 -> 0 (tie: lower)


def test_the_exact_cosine_graph_matches_brute_force():
    from sklearn.neighbors import NearestNeighbors

    x = np.random.default_rng(2).normal(size=(500, 12)).astype(np.float32)
    graph = score.ExactCosineNeighbors(15).fit_transform(x)
    assert graph.shape == (500, 500) and (np.diff(graph.indptr) == 15).all()
    indices = graph.indices.reshape(500, 15)
    distances = graph.data.reshape(500, 15)
    assert (indices[:, 0] == np.arange(500)).all() and (distances[:, 0] == 0).all()
    _, reference = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute").fit(x).kneighbors(x)
    assert (np.sort(indices, axis=1) == np.sort(reference, axis=1)).all()
    v = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert np.allclose(distances[:, 1:], 1 - np.einsum("ij,ikj->ik", v, v[indices[:, 1:]]), atol=1e-5)


def test_scib_reclusters_the_sample_on_an_exact_cosine_graph(root):
    labels = _labels(root)
    cells = _cells(labels, n=12)
    fine = _by_type(labels, cells)
    coarse = pd.Series(fine).map(SYNTHETIC_CLASSES).to_numpy(dtype=object)
    scib = score.scib_nmi_ari(informative(labels, cells, signal=6.0), cells["cell_id"].tolist(),
                              {"morph_class": coarse, "cell_state": fine})
    assert [e["resolution"] for e in scib["per_resolution"]] == list(score.SCIB_RESOLUTIONS)
    for level in ("morph_class", "cell_state"):
        assert scib[level]["nmi_max"] == max(e[f"nmi_{level}"] for e in scib["per_resolution"])
        assert scib[level]["resolution"] in score.SCIB_RESOLUTIONS
    assert scib["cell_state"]["nmi_max"] > 0.5 and scib["n_neighbors"] == 15
    assert scib["n_cells"] == len(cells) and set(scib["seconds"]) == {"graph", "leiden"}


def test_the_class_map_fails_closed(tmp_path, root, monkeypatch, fast_scib, class_map):
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = tmp_path / "run"
    full_chain(run, labels, {i: 3 for i in labels["image_id"]}, lambda c: informative(labels, c), root=root)
    check = score.class_map_check(root, "full")
    assert check["ok"] and check["sha256"] == digest(class_map) and check["classes"] == ["m0", "m1", "m2"]
    cases = {"missing": None, "no row": {k: v for k, v in SYNTHETIC_CLASSES.items() if k != "single"},
             "empty class": {**SYNTHETIC_CLASSES, "single": ""}}
    for case, mapping in cases.items():
        path = tmp_path / f"map_{case.replace(' ', '_')}.csv"
        if mapping is not None:
            write_class_map(path, mapping)
        monkeypatch.setattr(score, "CLASS_MAP", path)
        result = score.score_run(run, root, forbidden=[])
        assert result["status"] == score.NO_CLASS_MAP and "primary" not in result, case
        assert not result["class_map"]["ok"] and result["chain"]["ok"]
        assert not (run / "per_class.csv").exists()
        assert "Not scored (NO_CLASS_MAP)" in (run / "RESULT.md").read_text(encoding="utf-8")
    problems = {case: " ".join(score.load_class_map(tmp_path / f"map_{case.replace(' ', '_')}.csv")["problems"])
                for case in cases}
    assert "no class map at" in problems["missing"] and "empty morph_class" in problems["empty class"]
    assert "no morph_class row" in " ".join(score.class_map_check(
        root, "full", tmp_path / "map_no_row.csv")["problems"])
    duplicated = tmp_path / "dup.csv"
    pd.DataFrame({"cell_state": ["c0", "c0"], "morph_class": ["m0", "m1"], "note": ""}).to_csv(duplicated, index=False)
    assert "listed more than once" in " ".join(score.load_class_map(duplicated)["problems"])


def test_the_figure_renders_with_fixed_colours(tmp_path):
    rng = np.random.default_rng(0)
    all_classes = ["spindle", "round", "sheet", "branched"]
    n = 60
    classes = rng.choice(["round", "sheet"], n)
    used = figures.umap_scored(tmp_path / "a.png", rng.normal(size=(n, 2)), classes, rng.integers(0, 3, n).astype(str),
                               rng.choice(["correct", "wrong", "unlabeled"], n), rng.choice(["v1", "v2"], n),
                               all_classes=all_classes, all_sources=["v1", "v2", "v3"], title="test")
    other = figures.umap_scored(tmp_path / "b.png", rng.normal(size=(n, 2)), rng.choice(["sheet", "branched"], n),
                                ["0"] * n, ["correct"] * n, ["v3"] * n, all_classes=all_classes,
                                all_sources=["v3", "v1", "v2"])
    assert (tmp_path / "a.png").stat().st_size > 10_000 and (tmp_path / "b.png").exists()
    assert used["classes"]["sheet"] == other["classes"]["sheet"] == figures.type_palette(all_classes)["sheet"]
    assert not used["layout"]["classes_legend_below"]               # four short class names fit beside panel a
    assert used["status"] == figures.STATUS_COLOURS and other["source"]["v3"] == figures.source_palette(["v1", "v2", "v3"])["v3"]
    from PIL import Image

    width, height = Image.open(tmp_path / "a.png").size
    assert width == int(figures.WIDTH_IN * 300) and height <= 9 * 300


def test_palettes_have_no_repeats_and_long_legends_go_below(tmp_path):
    rng = np.random.default_rng(1)
    many = [f"type {n:02d} " + "x" * 26 for n in range(90)]                      # 35 characters each
    assert len(figures.distinct_colours()) >= 100
    palette = figures.type_palette(many)
    assert len(set(palette.values())) == 90 and list(palette) == sorted(many)
    states = rng.integers(0, 45, 4000).astype(str)
    by_size = figures.state_palette(states)
    counts = pd.Series(states).value_counts()
    assert list(by_size)[0] == counts.index[0] and len(set(by_size.values())) == 45
    used = figures.umap_scored(tmp_path / "many.png", rng.normal(size=(4000, 2)), rng.choice(many, 4000), states,
                               rng.choice(["correct", "wrong"], 4000), rng.choice(["v1", "v2"], 4000),
                               all_classes=many, all_sources=["v1", "v2"])
    layout = used["layout"]
    assert layout["height_in"] <= figures.MAX_HEIGHT_IN and layout["states_legend_below"]
    assert layout["classes_legend_below"] and layout["class_legend_columns"] >= 3 and layout["panel_in"] >= 1.8
    from PIL import Image

    assert Image.open(tmp_path / "many.png").size[0] == int(figures.WIDTH_IN * 300)


def _registry_stub(monkeypatch):
    levers = {"segment": {"diameter": None}, "crop": {"margin": None}, "embed": {"learning_rate": None},
              "batch_correct": {"theta": None}, "cluster": {"resolution": None}}
    monkeypatch.setattr(coverage, "_registry", lambda tools_dir: (tuple(levers), lambda tool, repo: levers[tool]))


def _round(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _tool_use(name, data):
    return {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t", "name": name, "input": data}]}}


def test_control_slug_from_run_meta(tmp_path, monkeypatch):
    names = ["NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 2080 Ti", "Tesla V100-SXM2-32GB", "NVIDIA A100-SXM4-40GB"]
    for name in names:
        monkeypatch.setattr(hardware, "_query", lambda fields, name=name: [name, "81559"])
        run = tmp_path / hardware.slug(name)
        run.mkdir()
        # what the launcher writes: nvidia-smi `name,memory.total` and hardware.gpu_slug() of the node
        (run / "RUN_META.json").write_text(json.dumps({"gpu": f"{name}, 81559 MiB", "gpu_slug": hardware.gpu_slug()}))
        which = score.control_slug(run)
        assert which["slug"] == hardware.gpu_slug() == which["run_meta_gpu_slug"]
        assert which["source"] == "RUN_META.json gpu"
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "scoring-node")
    which = score.control_slug(tmp_path / "no_meta")
    assert which["slug"] == "scoring-node" and "no RUN_META.json" in which["source"]
    assert score.control_dir(tmp_path, "pairs", "none", "x") == tmp_path / "control" / "pairs" / "none" / "x"


def test_score_run_end_to_end(tmp_path, root, monkeypatch, fast_scib, class_map):
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "scoring-node")       # not the run's GPU model
    labels = _labels(root)
    cells = {i: 6 for i in labels["image_id"]}

    control = score.control_dir(root, "full", "full", "test-gpu-1")
    full_chain(control, labels, cells, lambda c: random_embedding(c), root=root)
    result_c, tables_c, _ = score.score_choice(control, root, {"cluster_call": "K001", "resolution": 0.5},
                                               forbidden=[], control=None, lever_coverage=False)
    score.write_outputs(control, {**result_c, "control_chain": {"embed": "zero_shot", "rule": "test"}}, tables_c, None)

    run = root / "runs" / "claude-full-full-1"
    (root / "runs" / "claude-2").mkdir(parents=True)
    full_chain(run, labels, cells, lambda c: informative(labels, c), root=root)
    with open(run / "logs" / "tool_calls.jsonl", "a", encoding="utf-8") as handle:     # a refused peek
        handle.write(json.dumps({"event": "start", "call": "S099", "tool": "segment", "name": "peek",
                                 "call_dir": "calls/S099_peek",
                                 "argv": ["--images-file", f"{(root / 'private').as_posix()}/scorable.csv"]}) + "\n")
    tools = run / "workspace" / "tools"
    shutil.copytree(TEMPLATE_TOOLS, tools, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (tools / "_config.json").write_text(json.dumps({"repo": "/repo/SHAPE"}), encoding="utf-8")
    (run / "TOOLS_SHA256.json").write_text(json.dumps({"_config.json": digest(tools / "_config.json")}))
    (run / "AGENT_RUN.json").write_text(json.dumps({"wall_hours_used": 1.5}))
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "Test GPU 1, 81559 MiB", "gpu_slug": "test-gpu-1",
                                                   "prep_name": "full", "metadata": "full"}))
    secret = "sk-" + "a" * 30
    _round(run / "traces" / "round_01.jsonl", [
        _tool_use("Bash", {"command": f"ls {root.as_posix()}/private  # key {secret}"}),
        _tool_use("Read", {"file_path": "/repo/SHAPE/shape/agent/verification/base.py"}),
        _tool_use("Bash", {"command": f"cat {(root / 'runs' / 'claude-2').as_posix()}/RESULT.json"}),
        _tool_use("Bash", {"command": "python tools/segment.py --name ok"}),
        {"type": "result", "usage": {"input_tokens": 100, "output_tokens": 20}, "total_cost_usd": 0.5}])
    _round(run / "traces" / "round_02.jsonl", [
        _tool_use("Grep", {"pattern": "x", "path": "benchmarks/cell_state_HCMA"}),
        {"type": "result", "usage": {"input_tokens": 1, "output_tokens": 2}, "total_cost_usd": 0.25}])

    result = score.score_run(run, root, forbidden=["/raw/Human_Cell_Morphology_Atlas_HCMA"])
    assert result["status"] == "OK" and result["chain"]["ok"] and result["prep_name"] == "full"
    assert result["metadata"] == "full" and result["settings_from"]["prep_name"] == "RUN_META.json"
    assert result["class_map"]["sha256"] == digest(class_map) == result_c["class_map"]["sha256"]
    assert result["comparison"]["verdict"] == "GAIN" == result["knn_transfer_verdict"]
    full = result["full_metrics"]
    assert full["status"] == "OK" and set(result["verdict"]) == {"mAP", "macro_f1", "balanced_accuracy"}
    assert result["verdict"] == full["verdicts"] and (control / score.FULL_DRAWS).exists()
    assert result["verdict"] == {m: "GAIN" for m in ("mAP", "macro_f1", "balanced_accuracy")}  # informative vs random
    assert full["layout"]["splits"]["main"]["n_folds_chosen"] == 2 and full["n_units_without_cells"] == 0
    assert "knn_transfer verdict **GAIN**" in (run / "RESULT.md").read_text(encoding="utf-8")
    assert result["comparison"]["control"] == str(control) and result["comparison"]["fine"]["delta"] > 0
    assert result["control_slug"]["slug"] == "test-gpu-1" and result["control_slug"]["source"] == "RUN_META.json gpu"
    assert result["tool_intact"]["intact"]
    assert (run / "best_cluster_f1.csv").exists()
    assert {"setup", "cluster_transfer", "linear_probe", "scib", "knn", "source_residue", "load_delivery", "figure",
            "cost_tools_audit"} <= set(result["timings"])
    assert result["cost"]["tokens"] == {"input_tokens": 101, "output_tokens": 22}
    assert result["cost"]["cost_usd"] == 0.75 and result["cost"]["wall_hours"] == 1.5
    assert result["cost"]["tool_calls"]["embed"]["n_calls"] == 1
    assert result["cost"]["tool_calls"]["segment"]["statuses"] == {"done": 1, "not_ended": 1}
    by_needle = result["audit"]["hits_by_needle"]
    assert by_needle[(root / "private").as_posix()] == 2 and by_needle["scorable.csv"] == 1
    assert any(h.get("call") == "S099" for h in result["audit"]["hits"])
    assert by_needle["/repo/SHAPE"] == 1 and by_needle["shape/agent"] == 1
    assert by_needle[(root / "runs" / "claude-2").as_posix()] == 1
    assert by_needle["cell_state_HCMA"] == 1
    assert not any((run.as_posix() in n) for n in result["audit"]["needles"])
    assert all(secret not in hit["input"] and len(hit["input"]) <= score.INPUT_CHARS + 6
               for hit in result["audit"]["hits"])
    assert {h["round"] for h in result["audit"]["hits"] if "call" not in h} == {1, 2}
    assert result["lever_coverage"]["missing"]                 # nothing was changed in the synthetic run
    for name in ("RESULT.json", "RESULT.md", "per_image.csv", "per_class.csv", "per_type.csv",
                 "per_class_vs_control.csv", score.FIGURE):
        assert (run / name).exists()
    assert result["figure"]["path"] == str(run / score.FIGURE)
    md = (run / "RESULT.md").read_text(encoding="utf-8")
    lines = [line for line in md.splitlines() if line]
    assert lines[1].startswith(f"Primary `{score.PRIMARY}` (morph_class") and lines[4].startswith("Fine")
    assert lines[2].startswith(f"Cross-source guard (secondary) `{score.CROSS}`")       # right after the primary
    assert lines[3].startswith("Reference (same sample): oracle")                       # then the ceiling / chance
    assert "Fine-type cluster bootstrap of the primary delta: SE" in md and "State level (secondary" in md
    assert result["comparison"]["fine_type_bootstrap"]["se"] > 0 and result["comparison"]["fine_type_bootstrap"]["n_types"] == 6      # single is in m2
    assert "GAIN" in md and digest(class_map) in md
    assert f"Audit: {len(result['audit']['hits'])} hits" in md and "ledger S099" in md and score.FIGURE in md

    (tools / "segment.py").write_text("print('edited')\n", encoding="utf-8")
    assert score.tool_intact(run)["modified"] == ["segment.py"]

    # a control scored before the full metrics (no draws): the knn_transfer verdict stands, no full_metrics verdicts
    (control / score.FULL_DRAWS).unlink()
    older = score.score_run(run, root, forbidden=[])
    assert older["knn_transfer_verdict"] == "GAIN" and older["verdict"] is None
    assert "no full_metrics_draws.npz" in older["full_metrics"]["control_note"]
    assert older["full_metrics"]["verdicts"] is None



def test_the_comparison_needs_the_same_images_and_class_map(tmp_path, root, fast_scib, class_map):
    labels = _labels(root)
    cells = _cells(labels)
    result, tables = _evaluate(cells, informative(labels, cells), _by_type(labels, cells), root,
                               list(labels["image_id"]))
    result.update(mode="h5ad", status="OK", class_map=score._public(score.class_map_check(root, "full")))
    zero_shot = {"embed": "zero_shot", "rule": "the five tools at their defaults, except embed adaptation_depth=zero_shot"}
    control = tmp_path / "control"
    other = dict(result, expected_images_sha256=score.images_sha256(["img_0001"]))
    score.write_outputs(control, other, tables, None)
    mine = dict(result)
    assert score._with_control(mine, tables, control, tmp_path / "run") is None
    assert mine["comparison"] is None and "another expected image set" in mine["comparison_skipped"]
    score.write_outputs(control, dict(result, class_map={**result["class_map"], "sha256": "0" * 64}), tables, None)
    mine = dict(result)
    score._with_control(mine, tables, control, tmp_path / "run")
    assert mine["comparison"] is None and "another class map" in mine["comparison_skipped"]
    assert "not compared (the control at" in score.markdown(mine)
    # a control scored by the earlier scorer (cluster-level endpoint, no kNN columns): skipped, never raised
    old = dict(result, endpoint=score.CLUSTER)
    score.write_outputs(control, old, {**tables, "per_image": tables["per_image"].drop(
        columns=["n_knn_scored", "n_knn_correct"])}, None)
    mine = dict(result)
    assert score._with_control(mine, tables, control, tmp_path / "run") is None
    assert mine["comparison"] is None and "another endpoint" in mine["comparison_skipped"]
    score.write_outputs(control, dict(result, primary={**result["primary"], "k": 5}), tables, None)
    mine = dict(result)
    score._with_control(mine, tables, control, tmp_path / "run")
    assert "another endpoint" in mine["comparison_skipped"]
    plain = {k: v for k, v in result["primary"].items() if k not in ("vote", "temperature")}     # an older vote
    for primary in (plain, {**result["primary"], "vote": "pool_weighted"}, {**result["primary"], "temperature": 0.1},
                    {k: v for k, v in result["primary"].items() if k != "temperature"}):
        score.write_outputs(control, dict(result, primary=primary), tables, None)
        mine = dict(result)
        score._with_control(mine, tables, control, tmp_path / "run")
        assert mine["comparison"] is None and "another endpoint" in mine["comparison_skipped"]
    score.write_outputs(control, dict(result, control_chain=zero_shot),
                        {**tables, "per_image": tables["per_image"].drop(columns=["n_knn_scored"])},
                        None)
    mine = dict(result)
    assert score._with_control(mine, tables, control, tmp_path / "run") is None
    assert mine["comparison"] is None and "could not be compared: KeyError" in mine["comparison_skipped"]
    score.write_outputs(control, dict(result, expected_set="smoke"), tables, None)      # a smoke is never a control
    mine = dict(result)
    assert score._with_control(mine, tables, control, tmp_path / "run") is None
    assert mine["comparison"] is None and "is a smoke" in mine["comparison_skipped"]
    # a control without the zero-shot chain (the fine-tune control kept on the cluster) or another embed: skipped
    for chain_record in (None, {"embed": "trained", "rule": "fine-tune"}):
        score.write_outputs(control, dict(result) if chain_record is None else dict(result, control_chain=chain_record),
                            tables, None)
        mine = dict(result)
        assert score._with_control(mine, tables, control, tmp_path / "run") is None
        assert mine["comparison_skipped"] == (f"the control at {control} is a fine-tune (pre-2026-10-03) control; "
                                              "recompute the zero-shot control")
    score.write_outputs(control, dict(result, control_chain=zero_shot), tables, None)
    same = dict(result)
    score._with_control(same, tables, control, tmp_path / "run")
    assert same["comparison"]["verdict"] == "NO_RESOLVABLE_DIFFERENCE"


def test_a_subset_prep_is_the_expected_image_set(tmp_path, root, monkeypatch, fast_scib):
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    pick = labels.drop_duplicates("parent_id").groupby("cell_state").head(2)["image_id"].tolist()
    build_subset(root, "pairs", pick)
    run = root / "runs" / "claude-pairs-full-1"
    full_chain(run, labels, {i: 5 for i in pick}, lambda c: informative(labels, c), root=root, prep="prep_pairs")
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "", "prep_name": "pairs"}))
    _round(run / "traces" / "round_01.jsonl", [_tool_use("Bash", {"command": f"ls {(root / 'prep').as_posix()}/images"}),
                                               _tool_use("Bash", {"command": f"ls {(root / 'prep_pairs').as_posix()}/images"})])
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "OK", result["inputs"]["problems"]
    assert result["prep_name"] == "pairs" and result["coverage"]["n_images"] == len(pick)
    assert result["fine"]["n_types"] == labels["cell_state"].nunique() - 1 and result["primary"]["n_classes"] == 3
    assert [c["cell_state"] for c in result["unscorable_classes"]["cell_state"]] == ["single"]     # one micrograph
    assert result["audit"]["hits_by_needle"] == {(root / "prep").as_posix(): 1}
    (root / "prep_pairs" / "images" / f"{pick[0]}.png").write_bytes(b"changed")
    problems = score.inputs_check(run, root, "pairs")["problems"]
    assert len(problems) == 1 and problems[0].startswith("prep pairs: 1 problem(s)") and "missing or changed" in problems[0]


def test_metadata_none_is_checked_after_blanking(tmp_path, root, monkeypatch, fast_scib):
    """The workspace copy of a `none` run must equal the prep's images.csv after blank_metadata; the
    run's own prep tables become audit needles (they hold the withheld metadata)."""
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "gpu-x")
    labels = _labels(root)
    run = root / "runs" / "claude-full-none-1"
    full_chain(run, labels, {i: 3 for i in labels["image_id"]}, lambda c: informative(labels, c), root=root)
    from benchmarks.full_HCMA.test._synthetic_chain import workspace_data

    workspace_data(run, root, "prep", "none")
    data = run / "workspace" / "data" / "images.csv"
    blanked = pd.read_csv(data, dtype=str, keep_default_na=False)
    assert (blanked["source"] == "").all() and (blanked["parent_id"] == "").all()
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "", "prep_name": "full", "metadata": "none"}))
    assert score.inputs_check(run, root, "full", "none") == {"ok": True, "problems": []}
    wrong_level = score.inputs_check(run, root, "full", "full")["problems"]
    assert any("differ from the prep's images.csv after the run's metadata blanking (`full`)" in p for p in wrong_level)
    restored = blanked.copy()
    restored.loc[0, "source"] = labels.loc[0, "source"]          # an agent "restoring" a value
    restored.to_csv(data, index=False)
    problems = " ".join(score.inputs_check(run, root, "full", "none")["problems"])
    assert "1 row(s) differ" in problems and "images.csv missing or changed" in problems
    assert any("not comparable" in p for p in score.inputs_check(run, root, "full", "some")["problems"])
    blanked.to_csv(data, index=False)
    own_tables = [(root / "prep" / "images.csv").as_posix(), (root / "prep" / "PREPARE.json").as_posix()]
    _round(run / "traces" / "round_01.jsonl", [_tool_use("Bash", {"command": f"cat {own_tables[0]}"}),
                                               _tool_use("Read", {"file_path": own_tables[1]}),
                                               _tool_use("Bash", {"command": "cat data/images/../images.csv"})])
    hits = score.audit(run, root, [], "full", "none")["hits"]
    assert {h["needle"] for h in hits if h["line"] in (1, 2)} == set(own_tables)
    assert any(h["needle"] == score.CLIMB and h["line"] == 3 for h in hits)
    assert not any(h["needle"] in own_tables for h in score.audit(run, root, [], "full", "full")["hits"])
    full_run = root / "runs" / "claude-full-full-1"           # the same chain at full metadata
    full_chain(full_run, labels, {i: 3 for i in labels["image_id"]}, lambda c: informative(labels, c), root=root)
    reference = score.score_run(full_run, root, forbidden=[])
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "OK" and result["metadata"] == "none"
    # the scorer reads parent_id and source from the private labels, never the blanked workspace copy
    for key in (score.PRIMARY, f"{score.PRIMARY}_fine", f"{score.PRIMARY}_se", f"{score.PRIMARY}_fine_se"):
        assert result[key] == reference[key] and result[key] is not None
    assert result["secondary"]["source_residue"] == reference["secondary"]["source_residue"]
    assert result["settings_from"]["metadata"] == "RUN_META.json"
    assert "metadata `none`" in (run / "RESULT.md").read_text(encoding="utf-8")


def test_invalid_choice_scores_nothing(tmp_path, root, monkeypatch, fast_scib):
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, {i: 4 for i in labels["image_id"]}, lambda c: informative(labels, c),
                            root=root)
    chain.choose({"cluster_call": ids["K"], "resolution": 2.0})
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "NO_VALID_CHOICE"
    assert result[score.PRIMARY] == 0.0 == result[f"{score.PRIMARY}_fine"] and result["coverage"]["n_cells"] == 0
    assert "NOT VALID" in (run / "RESULT.md").read_text(encoding="utf-8")


def test_changed_inputs_make_the_run_invalid(tmp_path, root, monkeypatch, fast_scib):
    """A changed prep image, an added prep file, an edited images.csv: INVALID_INPUTS, scored empty."""
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = tmp_path / "run"
    full_chain(run, labels, {i: 4 for i in labels["image_id"]}, lambda c: informative(labels, c), root=root)
    assert score.inputs_check(run, root) == {"ok": True, "problems": []}
    image = root / "prep" / "images" / "img_0001.png"
    original = image.read_bytes()
    image.write_bytes(b"preprocessed by the agent")
    (root / "prep" / "images" / "img_9999.png").write_bytes(b"new")
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "INVALID_INPUTS" and result["chain"]["ok"]
    assert result[score.PRIMARY] == 0.0 and result["coverage"]["n_cells"] == 0
    problems = " ".join(result["inputs"]["problems"])
    assert "img_0001.png: missing or changed" in problems and "img_9999.png: added after the record" in problems
    assert "Inputs: INVALID" in (run / "RESULT.md").read_text(encoding="utf-8")
    image.write_bytes(original)
    (root / "prep" / "images" / "img_9999.png").unlink()
    images_csv = run / "workspace" / "data" / "images.csv"
    with open(images_csv, "a", encoding="utf-8") as handle:
        handle.write("img_7777,par_0001,src0,images/img_7777.png\n")
    result = score.score_run(run, root, forbidden=[])           # no exception
    problems = " ".join(result["inputs"]["problems"])
    assert result["status"] == "INVALID_INPUTS"
    assert "images.csv missing or changed" in problems and "1 unknown" in problems
    (run / "DATA_SHA256.json").unlink()
    assert any("no readable DATA_SHA256.json" in p for p in score.inputs_check(run, root)["problems"])


QC_PROBES = [                      # the QC's eight commands that the audit missed, then more evidence writes
    "cat ../../../private/labels.csv",
    "cat data/images/../../private/labels.csv",
    "python -c 'import benchmarks.full_HCMA.harness.score'",
    "python -c 'from shape.agent.verification.verifiers import crop'",
    "echo '{}' >> ../logs/tool_calls.jsonl",
    "FULL_HCMA_TOOL_STUBS=my.py python tools/embed.py --name x --crops C001",
    "ls $(dirname $(readlink data/images))/..",
    "bash ~/.claude-full-hcma/key_helper.sh",
    "rm -rf calls/E001_emb",
    "cp mine.npy calls/E001_emb/embedding.npy",
    "chmod u+w tools/_config.json",
    "mv data/images.csv old.csv",
    "sed -i 's/a/b/' tools/crop.py",
    "cat > data/images.csv <<'EOF'",
    "touch .deadline",
]
BENIGN = ["python tools/segment.py --name s1 --workers 2 > logs/s1.log 2>&1 &",
          "cat calls/K001_k/states.csv | head", "python my_analysis.py > notes/out.txt", "ls data/images | wc -l",
          "cp calls/K001_k/states.csv analysis/states.csv"]


def test_audit_catches_the_qc_probes(tmp_path, root):
    run = root / "runs" / "claude-9"
    run.mkdir(parents=True)
    ws = run.absolute().as_posix() + "/workspace"
    events = [_tool_use("Bash", {"command": c}) for c in QC_PROBES + BENIGN]
    events += [_tool_use("Write", {"file_path": f"{ws}/calls/E001_emb/cells.csv", "content": "x"}),
               _tool_use("Edit", {"file_path": "tools/_common.py", "old_string": "a", "new_string": "b"}),
               _tool_use("Write", {"file_path": "../logs/tool_calls.jsonl", "content": "{}"}),
               _tool_use("Read", {"file_path": "../../private/labels.csv"}),
               _tool_use("Glob", {"pattern": "*", "path": "data/images/../.."}),
               _tool_use("Write", {"file_path": f"{ws}/NOTES.md", "content": "notes"}),
               _tool_use("Read", {"file_path": "calls/K001_k/states.csv"})]
    _round(run / "traces" / "round_03.jsonl", events)
    hits = score.audit(run, root, [])["hits"]
    assert "morph_classes" in score.NAME_NEEDLES
    flagged = {h["line"] for h in hits}
    assert flagged == set(range(1, len(QC_PROBES) + 1)) | {len(QC_PROBES) + len(BENIGN) + k for k in range(1, 6)}
    needle_of = lambda line: {h["needle"] for h in hits if h["line"] == line}          # noqa: E731
    assert score.CLIMB in needle_of(1) and "labels.csv" in needle_of(1)
    assert "full_HCMA.harness" in needle_of(3) and "shape.agent" in needle_of(4)
    assert score.WRITE in needle_of(5) and "FULL_HCMA_TOOL_STUBS" in needle_of(6) and "readlink" in needle_of(7)
    assert {"key_helper", ".claude-full-hcma"} <= needle_of(8)
    assert all(score.WRITE in needle_of(line) for line in range(9, 15))
    assert needle_of(len(QC_PROBES) + len(BENIGN) + 1) == {score.WRITE}


def test_other_prep_needles_end_at_a_path_boundary(root):
    build_subset(root, "pairs", ["img_0001", "img_0003"])
    run = root / "runs" / "claude-pairs-2"
    run.mkdir(parents=True)
    full, pairs = (root / "prep").as_posix(), (root / "prep_pairs").as_posix()
    commands = [f"ls {full}", f"cd {full} && ls", f'ls "{full}"', f"ls {full}/images | head",
                f"ls {pairs}/images", f"ls {pairs}", f"ls {full}_other"]
    _round(run / "traces" / "round_01.jsonl", [_tool_use("Bash", {"command": c}) for c in commands])
    hits = score.audit(run, root, [], "pairs")["hits"]
    assert sorted(h["line"] for h in hits if h["needle"] == full) == [1, 2, 3, 4]
    assert not any(h["needle"] == pairs for h in hits)


def test_score_h5ad(tmp_path, root, monkeypatch, fast_scib):
    import anndata as ad

    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    cells = _cells(labels)
    x = informative(labels, cells)
    states = np.argmax(x, axis=1).astype(str)
    adata = ad.AnnData(X=np.zeros((len(cells), 2), dtype=np.float32),
                       obs=cells.assign(leiden=states).set_index("cell_id", drop=False))
    adata.obs.index = adata.obs.index.astype(str)
    adata.obsm["X_emb"] = x
    adata.obsm["X_umap"] = x[:, :2]
    path = tmp_path / "arm.h5ad"
    adata.write_h5ad(path)
    out = tmp_path / "scored"
    result = score.score_h5ad(path, root, out, state_key="leiden", embedding_key="X_emb", prep_name="full",
                              metadata="none")
    direct, _ = _evaluate(cells, x, states, root, list(labels["image_id"]))
    assert result["mode"] == "h5ad" and result[score.PRIMARY] == pytest.approx(direct[score.PRIMARY])
    assert result["metadata"] == "none" and result["secondary"]["n_clusters"] == len(set(states))
    for name in ("RESULT.json", "per_image.csv", "per_class.csv", "per_type.csv", score.FIGURE):
        assert (out / name).exists()


def test_needles_cover_the_results_layout(tmp_path):
    """Results layout: $ROOT = <bench>/cache, runs beside it: every other entry of <bench> is a needle,
    the run itself, cache and the run's own prep are not."""
    bench = tmp_path / "full_HCMA"
    root = bench / "cache"
    for name in ("prep", "prep_pairs", "private", "control", "smoke"):
        (root / name).mkdir(parents=True)
    run, other, logs = bench / "claude-pairs-full_abcdef12_20261005T000000Z_1", bench / "claude-2", bench / "logs"
    for d in (run, other, logs):
        d.mkdir()
    needles = score.needle_patterns(run, root, forbidden=[], prep_name="pairs")
    assert other.as_posix() in needles and logs.as_posix() in needles and (root / "prep").as_posix() in needles
    assert not any(n in needles for n in (run.as_posix(), root.as_posix(), bench.as_posix(),
                                          (root / "prep_pairs").as_posix()))


def test_recorded_root_scores_a_moved_run_against_its_own_root(tmp_path):
    """RUN_META.json `root` (the old $ROOT, a symlink to cache/ after the migration) wins only when it
    resolves to the given root; any other directory (an agent could rewrite RUN_META.json) does not."""
    other, new, run = tmp_path / "other", tmp_path / "cache", tmp_path / "run"
    for d in (other, new, run):
        d.mkdir()
    meta = run / "RUN_META.json"
    assert score.recorded_root(run, new) == new                                     # no RUN_META.json
    meta.write_text(json.dumps({"root": str(other)}), encoding="utf-8")
    assert score.recorded_root(run, new) == new                                     # unrelated dir: ignored
    meta.write_text(json.dumps({"root": str(tmp_path / "gone")}), encoding="utf-8")
    assert score.recorded_root(run, new) == new                                     # not a dir: the given root
    assert score._protected(meta.as_posix(), run.as_posix(), (run / "workspace").as_posix())
    alias = tmp_path / "old_root"
    try:
        alias.symlink_to(new, target_is_directory=True)
    except OSError:
        pytest.skip("cannot create a symlink here")
    meta.write_text(json.dumps({"root": str(alias)}), encoding="utf-8")
    assert score.recorded_root(run, new) == alias                                   # the old path, a symlink


def test_full_metrics_count_an_image_without_cells_as_misses(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    x = informative(labels, cells, signal=6.0)
    states = _by_type(labels, cells)
    whole, _ = _evaluate(cells, x, states, root, images)
    full = whole["full_metrics"]
    assert full["status"] == "OK" and full["n_units_without_cells"] == 0 and full["verdicts"] is None
    assert full["layout"]["splits"]["main"]["n_folds"] >= 2
    _, full_units = score.full_layout(score.expected_sets(root, images, dict(SYNTHETIC_CLASSES)))
    gone = full_units["image_id"].iloc[0]
    keep = (cells["image_id"] != gone).to_numpy()
    missing, _ = _evaluate(cells[keep].reset_index(drop=True), x[keep], states[keep], root, images)
    after = missing["full_metrics"]
    assert after["n_units_without_cells"] == 1 and after["layout"]["fingerprint"] == full["layout"]["fingerprint"]
    for metric in ("mAP", "macro_f1", "balanced_accuracy"):
        assert after["metrics"][metric]["main"]["value"] < full["metrics"][metric]["main"]["value"], metric
    # the same image's cells excluded by the agent instead: sampled, each a miss (as knn_transfer counts them)
    excluded, _ = _evaluate(cells[keep].reset_index(drop=True), x[keep], states[keep], root, images,
                            excluded=list(cells.loc[~keep, "cell_id"]))
    out = excluded["full_metrics"]
    assert out["n_excluded_sampled"] == int((~keep).sum()) > 0 and out["n_units_without_cells"] == 1
    assert out["metrics"]["balanced_accuracy"]["main"]["value"] < full["metrics"]["balanced_accuracy"]["main"]["value"]
