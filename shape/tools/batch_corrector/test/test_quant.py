"""Tests of the batch corrector's metric module (`quant.py`).

Pure pandas / numpy tests always run. The adaptive composite, views and bands are tested with
stand-ins for the scib-metrics / copairs metric functions (the composite logic runs unchanged); the
real metric values are tested only where scib-metrics and copairs are installed.
"""
from __future__ import annotations

import inspect
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shape.tools.batch_corrector import quant

try:
    import copairs  # noqa: F401
    import scib_metrics  # noqa: F401
    from sklearn.metrics import silhouette_score
    _HAS_HEAVY_DEPS = True
except ImportError:
    _HAS_HEAVY_DEPS = False

needs_heavy_deps = pytest.mark.skipif(not _HAS_HEAVY_DEPS, reason="scib-metrics / copairs not installed")
STAND_INS = ("silhouette_batch", "lisi_batch", "graph_conn", "lisi_label", "asw", "nmi_ari",
             "average_precision_negcon", "average_precision_nonrep", "replicate_retrieval")


def _between(feats, groups):
    x = np.asarray(feats, dtype=float)
    g = np.asarray(groups).astype(str)
    tot = ((x - x.mean(0)) ** 2).sum() + 1e-12
    return float(sum((g == u).sum() * ((x[g == u].mean(0) - x.mean(0)) ** 2).sum() for u in np.unique(g)) / tot)


@pytest.fixture()
def stand_ins(monkeypatch):
    """Between-group variance fractions in place of the real metrics (fast, deterministic)."""
    monkeypatch.setattr(quant, "silhouette_batch", lambda feats, labels, batch, metric="cosine": 1 - _between(feats, batch))
    monkeypatch.setattr(quant, "lisi_batch", lambda feats, batch, n_neighbors=25: 1 - _between(feats, batch))
    monkeypatch.setattr(quant, "graph_conn", lambda feats, labels, n_neighbors=25: _between(feats, labels))
    monkeypatch.setattr(quant, "lisi_label", lambda feats, labels, n_neighbors=25: _between(feats, labels))
    monkeypatch.setattr(quant, "asw", lambda feats, labels, metric="cosine": _between(feats, labels))
    monkeypatch.setattr(quant, "nmi_ari", lambda feats, labels, n_neighbors=25, resolution=1.0:
                        {"nmi": _between(feats, labels), "ari": _between(feats, labels)})

    def negcon(meta, feats, label_key, plate_types, **kw):
        ix = quant._map_index(meta, plate_types, label_key)
        return _between(np.asarray(feats)[ix], meta.loc[ix, label_key])

    def nonrep(meta, feats, label_key, plate_types, **kw):
        ix = quant._map_index(meta, plate_types, label_key, ignore_dmso=True)
        return _between(np.asarray(feats)[ix], meta.loc[ix, label_key])

    monkeypatch.setattr(quant, "average_precision_negcon", negcon)
    monkeypatch.setattr(quant, "average_precision_nonrep", nonrep)
    monkeypatch.setattr(quant, "replicate_retrieval",
                        lambda meta, feats, batch_key, plate_types, cross: 0.3 if cross else 0.6)


def synthetic_jump(n_sources=3, plates_per_source=2, n_trt=24, n_pos=2, n_dmso=6, seed=0):
    """A small JUMP-like design: TARGET2 plates shared by every source, source offsets."""
    rng = np.random.default_rng(seed)
    trt = [f"JCP2022_{i:06d}" for i in range(n_trt)]
    pos = [f"JCP2022_9{i:05d}" for i in range(n_pos)]
    effect = {c: rng.normal(0, 1.0, 8) for c in trt + pos}
    sources = [f"source_{s}" for s in (2, 6, 10, 3, 8)[:n_sources]]
    offset = {s: rng.normal(0, 2.0, 8) for s in sources}
    rows, feats = [], []
    for si, s in enumerate(sources):
        for p in range(plates_per_source):
            layout = trt + pos + ["DMSO"] * n_dmso
            for w, c in enumerate(layout):
                pert = "negcon" if c == "DMSO" else ("poscon" if c in pos else "trt")
                rows.append({"Metadata_Source": s, "Metadata_Batch": f"{s}_B{p}", "Metadata_Plate": f"{s}_P{p}",
                             "Metadata_Well": f"W{(w + 7 * (si * plates_per_source + p)) % len(layout):02d}",
                             "Metadata_JCP2022": c, "Metadata_PlateType": "TARGET2", "Metadata_PertType": pert})
                feats.append((0 if c == "DMSO" else effect[c]) + offset[s] + rng.normal(0, 0.3, 8))
    meta = pd.DataFrame(rows)
    ids = sorted(meta.loc[meta.Metadata_PertType == "trt", "Metadata_JCP2022"].unique())
    split_of = {c: ("val" if i % 2 else "test") for i, c in enumerate(ids)}
    meta["split"] = [split_of[c] if t == "trt" else t for c, t in zip(meta.Metadata_JCP2022, meta.Metadata_PertType)]
    return meta, np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------- always run
def test_rosters_and_weights():
    assert quant.BATCH_METRICS == ["silhouette_batch", "lisi_batch", "graph_conn"]
    assert quant.BIO_METRICS == ["lisi_label", "ari", "nmi", "asw", "negcon_mean_map", "nonrep_mean_map"]
    assert quant.WEIGHTS == {"batch": 0.4, "bio": 0.6}


def test_silhouette_subsample_is_a_fixed_function_of_n(monkeypatch):
    assert quant._sil_subsample_idx(quant.MAX_SIL_CELLS) is None
    idx = quant._sil_subsample_idx(50000, max_cells=1000)
    np.testing.assert_array_equal(idx, quant._sil_subsample_idx(50000, max_cells=1000))
    assert idx.shape == (1000,) and len(set(idx.tolist())) == 1000 and 0 <= idx.min() and idx.max() < 50000
    monkeypatch.setattr(quant, "MAX_SIL_CELLS", 10)      # looked up at call time
    assert quant._sil_subsample_idx(10) is None and quant._sil_subsample_idx(20).shape == (10,)


def test_silhouette_batch_filters_undefined_groups_and_reports_an_empty_view(monkeypatch):
    seen = {}

    def fake_metric(feats, labels, batch, metric):
        seen.update(feats=np.asarray(feats), labels=np.asarray(labels), batch=np.asarray(batch), metric=metric)
        return 0.75

    monkeypatch.setitem(sys.modules, "scib_metrics", types.SimpleNamespace(silhouette_batch=fake_metric))
    feats = np.arange(15, dtype=float).reshape(5, 3)
    labels = np.array(["valid", "valid", "valid", "singleton", "singleton"])
    batches = np.array(["a", "a", "b", "a", "b"])
    assert quant.silhouette_batch(feats, labels, batches) == pytest.approx(0.75)
    assert seen["labels"].tolist() == ["valid"] * 3
    assert seen["batch"].tolist() == ["a", "a", "b"]

    with pytest.raises(quant.MetricUnavailable, match="at least two batches"):
        quant.silhouette_batch(feats[-2:], labels[-2:], batches[-2:])


def test_composite_skips_an_unavailable_silhouette_without_hiding_other_failures(stand_ins, monkeypatch):
    meta = pd.DataFrame({"organ": ["a", "a", "b", "b"], "source": ["x", "y", "x", "y"]})
    feats = np.arange(12, dtype=float).reshape(4, 3)

    def unavailable(*args, **kwargs):
        raise quant.MetricUnavailable("no eligible label")

    monkeypatch.setattr(quant, "silhouette_batch", unavailable)
    score = quant.composite_all(meta, feats, "organ", "source", plate_types=())
    assert score["skipped"]["silhouette_batch"] == "unavailable: no eligible label"
    assert {"lisi_batch", "graph_conn"} <= set(score)

    monkeypatch.setattr(quant, "silhouette_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        quant.composite_all(meta, feats, "organ", "source", plate_types=())


def test_filter_dmso_drops_only_dmso_rows():
    meta = pd.DataFrame({"Metadata_JCP2022": ["cmpdA", "DMSO", "cmpdB", "DMSO"]})
    meta_f, feats_f = quant.filter_dmso(meta, np.arange(8).reshape(4, 2), "Metadata_JCP2022")
    assert list(meta_f["Metadata_JCP2022"]) == ["cmpdA", "cmpdB"]
    np.testing.assert_array_equal(feats_f, np.array([[0, 1], [4, 5]]))


def test_map_index_drops_singletons_poscon_and_the_outlier_compound():
    meta = pd.DataFrame({"Metadata_JCP2022": ["JCP2022_033954"] * 3 + ["cmpdA"] * 2 + ["cmpdSingleton", "p", "p"],
                         "Metadata_PlateType": ["TARGET2"] * 8,
                         "Metadata_PertType": ["trt"] * 6 + ["poscon"] * 2})
    keep = quant._map_index(meta, plate_types=["TARGET2"], label_key="Metadata_JCP2022")
    assert set(meta.loc[keep, "Metadata_JCP2022"]) == {"cmpdA"} and keep.sum() == 2
    dmso = pd.DataFrame({"Metadata_JCP2022": ["JCP2022_033954"] * 2 + ["DMSO"] * 4,
                         "Metadata_PlateType": ["TARGET2"] * 6, "Metadata_PertType": ["trt"] * 2 + ["negcon"] * 4})
    assert set(dmso.loc[quant._map_index(dmso, ["TARGET2"], "Metadata_JCP2022"), "Metadata_JCP2022"]) == {"DMSO"}


def test_adaptive_composite(stand_ins):
    meta, feats = synthetic_jump()
    full = quant.score_split(meta, feats, "test", batch_key="Metadata_Source", plate_types=["TARGET2"])
    assert full["skipped"] == {} and set(full["weights_used"]) == {"batch", "bio"}
    assert full["overall"] == pytest.approx(0.4 * full["batch_mean"] + 0.6 * full["bio_mean"])
    no_dmso = (meta.split != "negcon").to_numpy()             # no DMSO: negcon mAP is left out
    s = quant.composite(meta, feats, no_dmso & quant.split_mask(meta, "test"), batch_key="Metadata_Source",
                        plate_types=["TARGET2"])
    assert "negcon_mean_map" in s["skipped"] and "negcon_mean_map" not in s and "nonrep_mean_map" in s
    one = (meta.Metadata_Source == "source_2").to_numpy()      # one batch: the batch metrics are left out
    s = quant.composite(meta, feats, one & quant.split_mask(meta, "test"), batch_key="Metadata_Source",
                        plate_types=["TARGET2"])
    assert set(quant.BATCH_METRICS) <= set(s["skipped"]) and s["weights_used"] == {"bio": 1.0}
    assert s["overall"] == pytest.approx(s["bio_mean"])
    assert quant.aggregate({"lisi_batch": 0.2, "ari": 0.5, "nmi": 0.7}, {})["overall"] == pytest.approx(0.4 * 0.2 + 0.6 * 0.6)


def test_composite_without_plate_or_perturbation_columns_skips_the_maps(stand_ins):
    # a biological label and a batch, no JUMP plate / perturbation-type columns
    rng = np.random.default_rng(1)
    meta = pd.DataFrame({"organ": np.repeat(["a", "b", "c"], 20), "source": np.tile(["x", "y"], 30)})
    feats = rng.normal(size=(60, 5))
    s = quant.composite_all(meta, feats, "organ", "source", plate_types=())
    assert {"negcon_mean_map", "nonrep_mean_map"} <= set(s["skipped"])
    assert set(s) >= {"silhouette_batch", "lisi_batch", "graph_conn", "lisi_label", "ari", "nmi", "asw"}
    assert s["bio_mean"] == pytest.approx(np.mean([s[m] for m in ("lisi_label", "ari", "nmi", "asw")]))


def test_subset_drops_unused_categories():
    meta = pd.DataFrame({"label": pd.Categorical(["a", "b", "c"]), "v": [1, 2, 3]})
    assert list(quant.subset(meta, np.array([True, True, False]))["label"].cat.categories) == ["a", "b"]


def test_plate_halves_are_drawn_within_batch_groups():
    meta, _ = synthetic_jump()
    halves = quant.plate_halves(meta, 5, batch_key="Metadata_Source", seed=0)
    assert halves == quant.plate_halves(meta, 5, batch_key="Metadata_Source", seed=0)
    for h in halves:
        assert {p.split("_P")[0] for p in h} == set(meta.Metadata_Source)


def test_paired_band():
    ref, new = {"overall": 0.50}, {"overall": 0.53}
    ref_h = [{"overall": v} for v in (0.50, 0.51, 0.49, 0.50)]
    new_h = [{"overall": v} for v in (0.53, 0.55, 0.51, 0.53)]
    diffs = np.array([0.03, 0.04, 0.02, 0.03])
    band = quant.paired_band(ref, new, ref_h, new_h)
    assert band["delta"] == pytest.approx(0.03) and band["band_2se"] == pytest.approx(2 * diffs.std(ddof=1) / np.sqrt(2))
    assert band["significant"] is True and band["n_halves"] == 4
    assert quant.paired_band({"overall": None}, new, ref_h, new_h)["delta"] is None
    assert quant.paired_band(ref, new, ref_h[:1], new_h[:1])["band_2se"] is None


@pytest.mark.skipif(not os.environ.get("BATCHAGENTIC_ROOT"), reason="BATCHAGENTIC_ROOT not set")
def test_the_batchagentic_copy_has_the_same_definitions():
    """The external agentic benchmark's metric module copies these definitions (`composite_all` and
    `composite` differ only by this module's plate-column guard and `subset` helper)."""
    sys.path.insert(0, str(Path(os.environ["BATCHAGENTIC_ROOT"])))
    from agentic_jump.harness import metric as copy

    for name in ("_sil_subsample_idx", "filter_dmso", "_neighbors", *STAND_INS[:6], "_map_index", "_group_negcons",
                 "_copairs_map_sameby", "average_precision_negcon", "average_precision_nonrep", "_replicated", "_safe",
                 "aggregate", "split_mask", "score_split", "plate_halves", "score_halves", "paired_band",
                 "replicate_retrieval"):
        assert inspect.getsource(getattr(copy, name)) == inspect.getsource(getattr(quant, name)), name


# ---------------------------------------------------------------- real metric values
def _two_clouds(seed=0, n=20, dim=8, sep=20.0, jitter=0.3):
    rng = np.random.default_rng(seed)
    c1 = np.zeros(dim)
    c1[0] = sep
    x = np.vstack([rng.normal(scale=jitter, size=(n, dim)), c1 + rng.normal(scale=jitter, size=(n, dim))])
    return x, np.array(["g0"] * n + ["g1"] * n)


def _replicates(seed=0, n_compounds=3, n_replicates=3, n_dmso=10, dim=4):
    rng = np.random.default_rng(seed)
    rows, feats, well = [], [], 0
    for i in range(n_compounds):
        center = rng.normal(scale=10.0, size=(dim,))
        for _ in range(n_replicates):
            rows.append({"Metadata_JCP2022": f"cmpd{i}", "Metadata_Well": f"W{well}", "Metadata_Plate": "P1",
                         "Metadata_PlateType": "TARGET2", "Metadata_PertType": "trt"})
            feats.append(center + rng.normal(scale=0.05, size=(dim,)))
            well += 1
    for _ in range(n_dmso):
        rows.append({"Metadata_JCP2022": "DMSO", "Metadata_Well": f"W{well}", "Metadata_Plate": "P1",
                     "Metadata_PlateType": "TARGET2", "Metadata_PertType": "negcon"})
        feats.append(rng.normal(scale=10.0, size=(dim,)))
        well += 1
    return pd.DataFrame(rows), np.array(feats)


@needs_heavy_deps
def test_asw_is_the_rescaled_cosine_silhouette():
    x, labels = _two_clouds()
    assert quant.asw(x, labels) == pytest.approx((silhouette_score(x, labels, metric="cosine") + 1) / 2, abs=1e-6)


@needs_heavy_deps
def test_label_metrics_are_perfect_for_two_separated_clouds():
    x, labels = _two_clouds()
    assert quant.graph_conn(x, labels) == pytest.approx(1.0)
    assert quant.lisi_label(x, labels) == pytest.approx(1.0)
    result = quant.nmi_ari(x, labels)
    assert result["nmi"] == pytest.approx(1.0, abs=1e-6) and result["ari"] == pytest.approx(1.0, abs=1e-6)


@needs_heavy_deps
def test_batch_metrics_are_higher_when_batches_are_mixed():
    x, labels = _two_clouds()
    mixed = np.array((["A", "B"] * 10) * 2)
    rng = np.random.default_rng(0)
    offset = np.eye(8)[1] * 8.0
    c1 = np.eye(8)[0] * 20.0
    seg = np.vstack([rng.normal(scale=0.3, size=(10, 8)), offset + rng.normal(scale=0.3, size=(10, 8)),
                     c1 + rng.normal(scale=0.3, size=(10, 8)), c1 + offset + rng.normal(scale=0.3, size=(10, 8))])
    segregated = np.array((["A"] * 10 + ["B"] * 10) * 2)
    assert quant.silhouette_batch(x, labels, mixed) > quant.silhouette_batch(seg, labels, segregated)
    assert quant.lisi_batch(x, mixed) > quant.lisi_batch(seg, segregated)


@needs_heavy_deps
def test_maps_are_perfect_for_tight_replicate_clusters():
    meta, feats = _replicates()
    assert quant.average_precision_negcon(meta, feats, "Metadata_JCP2022", ["TARGET2"]) == pytest.approx(1.0)
    assert quant.average_precision_nonrep(meta, feats, "Metadata_JCP2022", ["TARGET2"]) == pytest.approx(1.0)


@needs_heavy_deps
def test_composite_overall_is_the_weighted_mean():
    meta, feats = _replicates(n_compounds=4, n_replicates=4, n_dmso=20, dim=8)
    meta["Metadata_Source"] = ["src1", "src2"] * (len(meta) // 2)
    s = quant.composite_all(meta, feats, "Metadata_JCP2022", "Metadata_Source", ["TARGET2"], n_neighbors=10)
    assert s["skipped"] == {}
    assert s["overall"] == pytest.approx(0.4 * s["batch_mean"] + 0.6 * s["bio_mean"])
