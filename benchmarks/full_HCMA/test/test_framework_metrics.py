"""harness/score.py, evaluation framework batch 1: the cross-source guard (A3.2), state replication
(A5.2), state-composition replicate consistency (A5.3), fine-level near misses (B2), the fine-type
cluster bootstrap (P0.5) and the scoring of every candidate of a run (P0.2: D3, A4.1, A5.5, E2)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import coverage, hardware, score
from benchmarks.full_HCMA.test._synthetic_chain import (SYNTHETIC_CLASSES, Chain, build_private, informative,
                                                        random_embedding, workspace_data, write_class_map)


@pytest.fixture()
def root(tmp_path):
    path = tmp_path / "root"
    build_private(path)
    return path


@pytest.fixture(autouse=True)
def class_map(tmp_path, monkeypatch):
    path = write_class_map(tmp_path / "morph_classes.csv")
    monkeypatch.setattr(score, "CLASS_MAP", path)
    return path


@pytest.fixture()
def fast_scib(monkeypatch):
    monkeypatch.setattr(score, "recluster", lambda x, ids, resolutions=score.SCIB_RESOLUTIONS, n_neighbors=15, timings=None: {
        r: np.argmax(np.asarray(x), axis=1).astype(str) for r in resolutions})


def _labels(root):
    return pd.read_csv(root / "private" / "labels.csv", dtype=str, keep_default_na=False)


def _cells(labels, n=8):
    return pd.DataFrame([(f"{i}_S001_{k:05d}", i) for i in labels["image_id"] for k in range(1, n + 1)],
                        columns=["cell_id", "image_id"])


def _by_type(labels, cells):
    return cells["image_id"].map(labels.set_index("image_id")["cell_state"]).to_numpy(dtype=object)


# ---------------------------------------------------------------- A3.2 the cross-source guard

def test_cross_source_neighbours_come_from_another_source():
    x = np.array([[1.0, 0.0], [1.0, 0.01], [1.0, 0.3]])
    labels = np.array(["q", "X", "Y"], dtype=object)
    micrographs = np.array(["p1", "p2", "p3"], dtype=object)
    sources = np.array(["a", "a", "b"], dtype=object)
    assert score.knn_predict(x, micrographs, labels, k=1)[0] == "X"          # the nearest other micrograph
    assert score.knn_predict(x, sources, labels, k=1)[0] == "Y"              # the nearest other source


def test_the_cross_source_guard_removes_a_source_shortcut(root, fast_scib):
    """m2 (lonely + single) is imaged by one source only (`srcX`). An embedding that encodes only
    the source lets m2's cells find m2 neighbours in other micrographs (the primary rewards the
    shortcut); the cross-source guard has no partner for m2, removes and lists it."""
    labels = _labels(root)
    m2 = labels["cell_state"].isin(["lonely", "single"])
    labels.loc[m2, "source"] = "srcX"
    labels.to_csv(root / "private" / "labels.csv", index=False)
    cells = _cells(labels)
    source = cells["image_id"].map(labels.set_index("image_id")["source"])
    centre = {s: 6.0 * np.eye(16)[n] for n, s in enumerate(sorted(set(source)))}
    x = (np.stack([centre[s] for s in source]) + np.random.default_rng(1).normal(size=(len(cells), 16))).astype(np.float32)
    result, tables = score.evaluate(cells, x, _by_type(labels, cells), root, list(labels["image_id"]),
                                    class_map=SYNTHETIC_CLASSES)
    assert tables["per_class"].set_index("morph_class").loc["m2", "f1"] > 0.9            # the shortcut pays here
    cross = result["cross_source"]
    assert cross["unscorable_classes"] == [{"morph_class": "m2", "n_images": 6, "n_sources": 1}]
    assert list(tables["per_class_cross_source"]["morph_class"]) == ["m0", "m1"] and cross["n_classes"] == 2
    assert result[score.CROSS] == cross["value"] == pytest.approx(tables["per_class_cross_source"]["f1"].mean())
    assert result[f"{score.CROSS}_se"] == cross["se"] and cross["k"] == 20 and cross["vote"] == "dino_weighted"
    assert result[score.CROSS] < result[score.PRIMARY]                                       # m0 / m1: near chance
    md = score.markdown({**result, "mode": "h5ad", "class_map": {}})
    assert "Cross-source guard (secondary)" in md and "m2 (1 source)" in md


# ---------------------------------------------------------------- A5.2 state replication

def test_state_replication_by_hand():
    states = ["s1"] * 20 + ["s2"] * 21 + ["s3"] * 5
    parents = ["p1"] * 10 + ["p2"] * 10 + ["p1"] * 20 + ["p2"] + ["p3"] * 5
    out = score.state_replication(np.array(states, dtype=object), np.array(parents, dtype=object))
    # s1: 50 / 50 -> replicated; s2: p2 has 1 / 21 < 5 % -> not; s3: one micrograph
    assert out["n_states"] == 3 and out["n_replicated_states"] == 1
    assert out["share_cells_in_replicated_states"] == pytest.approx(20 / 46)
    h2 = -(20 / 21 * np.log(20 / 21) + 1 / 21 * np.log(1 / 21))
    assert out["median_effective_micrographs"] == pytest.approx(np.exp(h2))      # median of 2, e^H(s2), 1


# ---------------------------------------------------------------- A5.3 state-composition replicate consistency

def test_js_similarity_matches_scipy():
    from scipy.spatial.distance import jensenshannon

    p, q = np.array([0.5, 0.3, 0.2, 0.0]), np.array([[0.1, 0.1, 0.4, 0.4], [0.5, 0.3, 0.2, 0.0], [0, 0, 0, 1.0]])
    expected = [1 - jensenshannon(p, row, base=2) for row in q]
    assert score.js_similarity(p, q) == pytest.approx(expected)
    assert score.js_similarity(p, q)[1] == 1.0 and score.js_similarity(p, q)[2] == pytest.approx(0.0, abs=1e-12)


def test_state_composition_auc_by_hand():
    images = pd.DataFrame({"image_id": ["A1", "A1b", "A2", "A3", "B1", "B2"], "cell_state": ["a", "a", "a", "a", "b", "b"],
                           "parent_id": ["p1", "p1", "p2", "p5", "p3", "p4"], "source": ["s1", "s1", "s2", "s1", "s1", "s2"]})
    cells = {"A1": ["x", "x", "y"], "A1b": ["x", "x", "y"], "A2": ["x", "x", "y"], "B1": ["z"], "B2": ["z"]}     # A3: none
    ids = np.array([i for i, v in cells.items() for _ in v], dtype=object)
    states = np.array([s for v in cells.values() for s in v], dtype=object)
    out = score.state_composition_auc(ids, states, images, "cell_state")
    # positives: A1-A2, A1b-A2, B1-B2 (A1-A1b: same micrograph, neither); negatives: 3 A x 2 B
    assert out["auc"] == 1.0 and out["n_positive_pairs"] == 3 and out["n_negative_pairs"] == 6
    assert out["n_images"] == 5 and out["n_images_left_out"] == 1
    # A2 now looks like B: positives {0, 0, 1}, negatives {0 x 4, 1 x 2} -> ties count 0.5
    cells["A2"] = ["z"]
    ids = np.array([i for i, v in cells.items() for _ in v], dtype=object)
    states = np.array([s for v in cells.values() for s in v], dtype=object)
    out = score.state_composition_auc(ids, states, images, "cell_state")
    pos, neg = np.array([0, 0, 1.0]), np.array([0, 0, 0, 0, 1.0, 1.0])
    by_hand = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    assert out["auc"] == pytest.approx(by_hand)
    # cross-source: positives must also come from another source (A1-A2, A1b-A2, B1-B2 all do; drop A2's source)
    images.loc[images["image_id"] == "A2", "source"] = "s1"
    cross = score.state_composition_auc(ids, states, images, "cell_state", cross_source=True)
    assert cross["n_positive_pairs"] == 1 and cross["cross_source"]              # only B1-B2 is cross-source


def test_state_metrics_in_the_result(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    result, _ = score.evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, list(labels["image_id"]),
                               class_map=SYNTHETIC_CLASSES)
    sec = result["secondary"]
    # states = the fine types: every type with 2 micrographs is replicated; single (one micrograph) is not
    assert sec["state_replication"]["share_cells_in_replicated_states"] == pytest.approx(1 - 16 / len(cells))
    comp = sec["state_composition"]
    assert comp["cell_state"]["auc"] == 1.0 and comp["morph_class"]["auc"] < 1.0      # 2 types per class
    assert comp["morph_class_cross_source"]["cross_source"] and comp["morph_class"]["n_images_left_out"] == 0


# ---------------------------------------------------------------- B2 near misses

def test_near_miss_share_by_hand():
    truth = np.array(["c0", "c0", "c2", "c0", "lonely"], dtype=object)
    pred = np.array(["c1", "c2", "c2", score.UNLABELED, "single"], dtype=object)
    out = score.near_miss(truth, pred, {**SYNTHETIC_CLASSES, "lonely": score.UNSCORED, "single": score.UNSCORED})
    # errors: c0->c1 (m0, near), c0->c2 (m0 vs m1), lonely->single (unscored is no class); no prediction: no error
    assert out["n_errors"] == 3 and out["n_near_miss"] == 1 and out["share"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- P0.5 the fine-type cluster bootstrap

def test_the_fine_type_bootstrap_reproduces_the_delta(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    _, good = score.evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, images,
                             class_map=SYNTHETIC_CLASSES)
    _, bad = score.evaluate(cells, random_embedding(cells), _by_type(labels, cells), root, images,
                            class_map=SYNTHETIC_CLASSES)
    out, _ = score.compare(good, bad)
    boot = out["fine_type_bootstrap"]
    assert boot["delta"] == pytest.approx(out["delta"])           # the type counts rebuild the class-level F1
    assert boot["n_types"] == 6 and boot["se"] > 0 and boot["ci95"][0] <= boot["delta"] <= boot["ci95"][1]
    assert out["verdict"] == "GAIN"                              # the verdict stays on the class bootstrap
    same = score.compare(good, good)[0]["fine_type_bootstrap"]
    assert same["delta"] == 0.0 and same["se"] == 0.0
    counts = good["knn_type_counts"]
    assert set(counts.columns) == {"cell_state", "morph_class", "predicted", "n"}
    assert counts["n"].sum() == len(cells)                            # every image fully sampled (8 < 100 cells)
    no_counts = score.compare({**good, "knn_type_counts": None}, bad)[0]["fine_type_bootstrap"]
    assert no_counts["se"] is None and "no knn_type_counts" in no_counts["note"]


def test_the_type_bootstrap_counts_missing_images(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    gone = labels.loc[labels["cell_state"] == "c0", "image_id"].iloc[0]
    kept = cells[cells["image_id"] != gone].reset_index(drop=True)
    _, tables = score.evaluate(kept, informative(labels, kept), _by_type(labels, kept), root, list(labels["image_id"]),
                               class_map=SYNTHETIC_CLASSES)
    counts = tables["knn_type_counts"]
    row = counts[(counts["cell_state"] == "c0") & (counts["predicted"] == score.NO_PREDICTION)]
    assert row["n"].sum() == score.MAX_CELLS and row["morph_class"].tolist() == ["m0"]


# ---------------------------------------------------------------- P0.2 every candidate of a run

def _registry_stub(monkeypatch):
    levers = {"segment": {"diameter": None}, "crop": {"margin": None}, "embed": {"learning_rate": None},
              "batch_correct": {"theta": None}, "cluster": {"resolution": None}}
    monkeypatch.setattr(coverage, "_registry", lambda tools_dir: (tuple(levers), lambda tool, repo: levers[tool]))


def _retime(run, ends):
    """Rewrite the ledger's end times ({call: seconds after a start at 1000})."""
    path = run / "logs" / "tool_calls.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        line["time"] = 1000.0 if line["event"] == "start" else 1000.0 + ends.get(line["call"], 1.0)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def test_all_candidates_regrets_and_the_anytime_curve(tmp_path, root, monkeypatch, fast_scib):
    """E001 (informative) -> B001 -> K001 (0.25, 0.5); E002 (noise) -> K002 (0.5), which is delivered."""
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = root / "runs" / "claude-full-full-1"
    chain = Chain(run)
    workspace_data(run, root)
    images = list(labels["image_id"])
    s = chain.segment(images)
    c = chain.crop(s, {i: 6 for i in images})
    e1 = chain.embed(c, lambda cells: informative(labels, cells, signal=6.0), name="good")
    b1 = chain.batch_correct(e1)
    k1 = chain.cluster(b1, resolutions=(0.25, 0.5))
    e2 = chain.embed(c, lambda cells: random_embedding(cells), name="noise")
    k2 = chain.cluster(e2, name="k2")
    chain.choose({"cluster_call": k2, "resolution": 0.5, "reason": "test", "skipped": {}})
    _retime(run, {k1: 100.0, k2: 300.0, e1: 50.0, e2: 200.0, b1: 60.0})
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "", "prep_name": "full", "metadata": "full"}))

    plain = score.score_run(run, root, forbidden=[])
    assert "candidates" not in plain and not (run / "candidates.csv").exists()          # opt-in
    result = score.score_run(run, root, forbidden=[], all_candidates=True)
    assert result["status"] == "OK", result["chain"]["problems"]
    frame = pd.read_csv(run / "candidates.csv")
    assert list(frame.columns) == list(score.CANDIDATE_COLUMNS)
    assert list(zip(frame["call"], frame["resolution"].fillna(-1))) == [
        (b1, -1), (e1, -1), (e2, -1), (k1, 0.25), (k1, 0.5), (k2, 0.5)]
    assert frame.loc[frame["call"] == b1, "parent_embedding_call"].item() == e1
    assert frame.loc[frame["call"] == k1, "parent_embedding_call"].tolist() == [b1, b1]
    delivered = frame[frame["delivered"]]
    assert sorted(delivered["call"]) == [e2, k2]
    # the delivered embedding scores what the delivered result scores
    assert frame.loc[frame["call"] == e2, score.PRIMARY].item() == pytest.approx(result[score.PRIMARY])
    assert frame.loc[frame["call"] == e2, score.CROSS].item() == pytest.approx(result[score.CROSS])
    k2_row = frame[(frame["call"] == k2)].iloc[0]
    assert k2_row[score.CLUSTER] == pytest.approx(result[score.CLUSTER])
    assert k2_row["state_composition_auc"] == pytest.approx(result["secondary"]["state_composition"]["morph_class"]["auc"])
    cand = result["candidates"]
    assert cand["n_embedding_candidates"] == 3 and cand["n_cluster_calls"] == 2 and cand["n_cluster_candidates"] == 3
    regret = cand["selection_regret"][score.PRIMARY]
    assert regret["best_call"] in (e1, b1) and regret["regret"] == pytest.approx(regret["best"] - result[score.PRIMARY])
    assert regret["regret"] > 0.3
    assert cand["selection_regret"][score.CLUSTER]["best_call"] == k1
    batch = cand["batch_decision_regret"]                       # no batch_correct on E002: E002 alone
    assert batch["n_batch_correct_candidates"] == 0 and batch["regret"] == 0.0 and batch["chosen"] == e2
    assert cand["resolution_regret"][score.CLUSTER]["regret"] == 0.0              # K002 has one resolution
    points = cand["anytime_curve"]["points"]
    assert [p["call"] for p in points] == [k1, k2] and [p["end_seconds"] for p in points] == [100.0, 300.0]
    assert points[0]["best_so_far"] == points[1]["best_so_far"] == pytest.approx(
        frame.loc[frame["call"] == b1, score.PRIMARY].item()) and points[1]["best_call"] == k1
    assert result["timings"]["candidates"] == cand["seconds"]


def test_batch_and_resolution_regret_on_the_delivered_family(tmp_path, root, monkeypatch, fast_scib):
    """Delivered: K001 on E001 (no correction) at 0.5; B001 on E001 is a candidate of the batch
    decision; K001's other resolution 0.25 is a candidate of the resolution decision (it merges
    single with lonely, so single's cells find an m2 cluster mate in another micrograph)."""
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = root / "runs" / "claude-full-full-2"
    chain = Chain(run)
    workspace_data(run, root)
    images = list(labels["image_id"])
    c = chain.crop(chain.segment(images), {i: 6 for i in images})
    e1 = chain.embed(c, lambda cells: informative(labels, cells, signal=6.0))
    b1 = chain.batch_correct(e1)
    k1 = chain.cluster(e1, resolutions=(0.25, 0.5))
    chain.choose({"cluster_call": k1, "resolution": 0.5, "reason": "test", "skipped": {}})
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "", "prep_name": "full", "metadata": "full"}))
    result = score.score_run(run, root, forbidden=[], all_candidates=True)
    cand = result["candidates"]
    batch = cand["batch_decision_regret"]
    assert batch["n_batch_correct_candidates"] == 1 and batch["chosen"] == e1 and batch["embed_call"] == e1
    assert batch["regret"] == pytest.approx(0.0)                 # the synthetic B copies E's embedding
    res = cand["resolution_regret"][score.CLUSTER]
    frame = pd.read_csv(run / "candidates.csv")
    at = frame[frame["call"] == k1].set_index("resolution")[score.CLUSTER]
    assert res["delivered"] == pytest.approx(at[0.5]) and res["regret"] == pytest.approx(max(at) - at[0.5])
    assert res["regret"] > 0 and res["best_resolution"] == 0.25
    assert cand["selection_regret"][score.PRIMARY]["regret"] == pytest.approx(0.0)


# ---------------------------------------------------------------- batch 2, C1: ceiling and chance

def test_oracle_and_chance_references(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    result, _ = score.evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, images,
                               class_map=SYNTHETIC_CLASSES)
    ref = result["reference"]
    assert ref["oracle"]["primary"] == 1.0 and ref["oracle"]["cross_source"] == 1.0    # no misses in this delivery
    assert abs(ref["chance"]["primary"] - 1 / 3) < 0.15                                # 3 classes
    assert ref["normalized_primary"] == pytest.approx((result[score.PRIMARY] - ref["chance"]["primary"])
                                                      / (1.0 - ref["chance"]["primary"]))
    # the oracle keeps the delivery's misses: an image left out caps it below 1
    gone = labels.loc[labels["cell_state"] == "c0", "image_id"].iloc[0]
    kept = cells[cells["image_id"] != gone].reset_index(drop=True)
    partial, _ = score.evaluate(kept, informative(labels, kept), _by_type(labels, kept), root, images,
                                class_map=SYNTHETIC_CLASSES)
    assert partial["reference"]["oracle"]["primary"] < 1.0
    md = score.markdown({**result, "mode": "h5ad", "class_map": {}})
    assert "Reference (same sample): oracle 1.0000 / chance" in md


def test_normalized_primary_is_null_without_headroom(root, fast_scib):
    """No delivered cell: oracle and chance both score 0 (every image a miss), the denominator is 0."""
    labels = _labels(root)
    empty = pd.DataFrame({"cell_id": [], "image_id": []})
    result, _ = score.evaluate(empty, np.zeros((0, 4)), np.array([], dtype=object), root, list(labels["image_id"]),
                               class_map=SYNTHETIC_CLASSES)
    ref = result["reference"]
    assert ref["oracle"]["primary"] == 0.0 == ref["chance"]["primary"] and ref["normalized_primary"] is None


# ---------------------------------------------------------------- batch 2, C2: strata

def test_macro_f1_of_rows_by_hand():
    rows = pd.DataFrame({"image_id": ["i1", "i1", "i2", "i1"], "morph_class": ["a", "a", "b", "a"],
                         "predicted": ["a", "b", "b", score.NO_PREDICTION], "n": [1, 1, 1, 1]})
    f1, accuracy, classes = score.macro_f1_of_rows(rows)
    # a: tp 1, predicted 1, actual 3 -> 2/4; b: tp 1, predicted 2, actual 1 -> 2/3
    assert f1 == pytest.approx((0.5 + 2 / 3) / 2) and accuracy == 0.5 and classes == ["a", "b"]


def test_strata_from_the_prep_metadata(root, fast_scib):
    labels = _labels(root)
    prep = pd.read_csv(root / "prep" / "images.csv", dtype=str)
    modality = {i: ("phase" if n % 2 else "brightfield") for n, i in enumerate(prep["image_id"])}
    prep["image_modality"] = prep["image_id"].map(modality)
    cells = _cells(labels)
    x = informative(labels, cells, signal=6.0)
    noisy = cells["image_id"].map(modality).eq("phase").to_numpy()
    x[noisy] = random_embedding(cells)[noisy] * 6                              # phase images carry no class signal
    result, tables = score.evaluate(cells, x, _by_type(labels, cells), root, list(labels["image_id"]),
                                    class_map=SYNTHETIC_CLASSES, strata_meta=prep)
    strata = result["strata"]
    assert strata["absent_columns"] == ["magnification"] and set(strata) >= {"image_modality", "source"}
    by = {r["value"]: r for r in strata["image_modality"]}
    assert by["brightfield"]["macro_f1"] > 0.9 > by["phase"]["macro_f1"]
    assert sum(r["n_images"] for r in strata["image_modality"]) == len(labels)
    assert sum(r["n_queries"] for r in strata["source"]) == result["primary"]["n_queries"]
    assert all(r["n_classes"] == 3 for r in strata["source"])
    table = tables["strata"]
    assert list(table.columns)[:3] == ["column", "value", "n_images"] and set(table["column"]) == {"image_modality", "source"}


# ---------------------------------------------------------------- batch 2, C3: proxy validity

def test_proxy_validity_over_the_candidates(tmp_path, root, monkeypatch, fast_scib):
    _registry_stub(monkeypatch)
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "none")
    labels = _labels(root)
    run = root / "runs" / "claude-full-full-3"
    chain = Chain(run)
    workspace_data(run, root)
    images = list(labels["image_id"])
    c = chain.crop(chain.segment(images), {i: 6 for i in images})
    calls = []
    for n, signal in enumerate((6.0, 2.0, 1.0, 0.0)):
        e = chain.embed(c, lambda cells, signal=signal: informative(labels, cells, signal=signal, seed=n), name=f"e{n}")
        (chain._dir(e) / "result.json").write_text(json.dumps({"summary": {
            "proxy": signal, "neg": -signal, "constant": 1, "flag": True, "nested": {"x": signal * 2}, "sizes": [1, 2]}}))
        k = chain.cluster(e, name=f"k{n}", resolutions=(0.25, 0.5))
        (chain._dir(k) / "result.json").write_text(json.dumps({"summary": {
            "n_cells": 100, "resolutions": {"0.25": {"n_states": 3 + n}, "0.5": {"n_states": 6 + n}}}}))
        calls.append((e, k))
    chain.choose({"cluster_call": calls[0][1], "resolution": 0.5, "reason": "test", "skipped": {}})
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "", "prep_name": "full", "metadata": "full"}))
    result = score.score_run(run, root, forbidden=[], all_candidates=True)
    validity = result["candidates"]["proxy_validity"]
    embed = {(e["field"]): e for e in validity if e["tool"] == "embed"}
    assert set(embed) == {"proxy", "neg", "nested.x"}                        # constant, bool and list fields dropped
    assert embed["proxy"]["spearman"] == pytest.approx(1.0) and embed["neg"]["spearman"] == pytest.approx(-1.0)
    assert embed["proxy"]["n"] == 4 and embed["proxy"]["metric"] == score.PRIMARY
    cluster = [e for e in validity if e["tool"] == "cluster"]
    assert {e["field"] for e in cluster} == {"resolution.n_states"}            # n_cells is constant
    assert cluster[0]["metric"] == "state_composition_auc" and cluster[0]["n"] == 8
    rhos = [abs(e["spearman"]) for e in validity if e["spearman"] is not None]
    assert rhos == sorted(rhos, reverse=True)
    assert "proxy_validity_rule" in result["candidates"]
    assert result["strata"]["absent_columns"] == ["image_modality", "magnification"]   # the synthetic prep has neither


# ---------------------------------------------------------------- the pinned verdict (2026-10-05)

def test_the_verdict_needs_the_cross_source_guard(root, fast_scib):
    labels = _labels(root)
    cells = _cells(labels)
    images = list(labels["image_id"])
    _, good = score.evaluate(cells, informative(labels, cells), _by_type(labels, cells), root, images,
                             class_map=SYNTHETIC_CLASSES)
    _, bad = score.evaluate(cells, random_embedding(cells), _by_type(labels, cells), root, images,
                            class_map=SYNTHETIC_CLASSES)
    gain = score.compare(good, bad)[0]
    assert gain["verdict"] == score.GAIN and gain["cross_source_guard"]["status"] == "available"
    assert gain["cross_source_guard"]["delta"] >= 0 and gain["verdict_rule"] == score.VERDICT_RULE
    # the primary gains, the guard loses: a source shortcut is suspected
    worse_guard = good["per_class_cross_source"].assign(f1=0.0)
    shortcut = score.compare({**good, "per_class_cross_source": worse_guard}, bad)[0]
    assert shortcut["verdict"] == score.SHORTCUT == "SOURCE_SHORTCUT_SUSPECTED"
    assert shortcut["cross_source_guard"]["delta"] < 0
    # an older control without the guard: unavailable, the verdict from the primary alone, with a note
    old = score.compare(good, {**bad, "per_class_cross_source": None})[0]
    assert old["verdict"] == score.GAIN and old["cross_source_guard"]["status"] == "unavailable"
    assert "guard unavailable" in old["verdict_note"]
    assert score.compare(bad, good)[0]["verdict"] == score.LOSS
