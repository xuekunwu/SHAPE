import numpy as np
import pandas as pd
import pytest

from benchmarks.full_CPJUMP1.harness.score import cosine, retrieval


def design():
    return pd.DataFrame([dict(well=f"{gene}{guide}{plate}", gene=gene, guide=f"{gene}{guide}", plate=plate,
              role="evaluation", pert_type="trt") for gene in ("A", "B", "C") for guide in (1, 2) for plate in (1, 2)])


def test_cross_guide_cross_plate_perfect_signal():
    meta = design()
    x = np.eye(3)[["ABC".index(g) for g in meta.gene]]
    result, table = retrieval(meta, x, cross_guide=True)
    assert result == {"n_queries": 12, "n_genes": 3, "mAP": 1., "MRR": 1., "top1": 1.}
    assert set(table.candidates) == {5}


def test_same_guide_shortcut_is_not_cross_guide_success():
    meta = design()
    mapping = {g: i for i, g in enumerate(meta.guide.unique())}
    x = np.eye(6)[[mapping[g] for g in meta.guide]]
    same, _ = retrieval(meta, x, cross_guide=False)
    different, _ = retrieval(meta, x, cross_guide=True)
    assert same["mAP"] == 1
    assert different["mAP"] == pytest.approx(0.2)
    assert different["top1"] == 0


def test_heldout_only_and_single_guides_excluded_from_primary():
    meta = design()
    meta.loc[meta.gene == "A", "role"] = "calibration"
    meta = meta[~((meta.gene == "C") & (meta.guide == "C2"))].reset_index(drop=True)
    x = np.eye(3)[["ABC".index(g) for g in meta.gene]]
    cross, table = retrieval(meta, x, cross_guide=True)
    assert cross["n_genes"] == 1 and cross["n_queries"] == 4
    assert set(table.gene) == {"B"}
    same, _ = retrieval(meta, x, cross_guide=False)
    assert same["n_genes"] == 2


def test_ties_and_zero_norm_are_not_success():
    result, _ = retrieval(design(), np.ones((12, 3)), cross_guide=True)
    assert result["mAP"] == pytest.approx(0.2)
    assert result["top1"] == 0
    with pytest.raises(ValueError, match="zero-norm"):
        cosine(np.zeros((4, 5)))


def complete_design(tmp_path, cell_noise=0.05, seed=0, drop=(), exclude=()):
    from benchmarks.full_CPJUMP1.harness.data import MANIFEST_SHA256, digest, write_json
    from .test_data import fixed_manifest
    prep, run = tmp_path / "prep", tmp_path / "run"
    run.mkdir()
    public, private, rows, matrix, cells, embedding, excluded = [], [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    m = fixed_manifest()
    for w in m["wells"]:
        well = w["plate"] + "_" + w["well"]
        rows.append({"well": well, "plate": w["plate"]})
        v = np.zeros(131)
        v[int(w["gene"][1:]) if w["pert_type"] == "trt" else 130] = 1
        matrix.append(v)
        for site in range(2):
            image = f"{well}_{site}"
            public.append({"image_id": image, "well": well})
            private.append({"image_id": image, "gene": w["gene"], "guide": w["guide_sequence"],
                            "role": w["role"], "pert_type": w["pert_type"]})
            for c in range(0 if well in drop else 2):   # two cells per site around the well's gene signal
                if well in exclude:                      # excluded by the agent: an id, no vector
                    excluded.append(f"default_{image}_cell_{c:04d}")
                    continue
                cells.append({"cell_id": f"{image}_{c}", "image_id": image, "well": well, "plate": w["plate"]})
                embedding.append(v + rng.normal(scale=cell_noise, size=131))
    write_json(prep / "public" / "images.json", public)
    write_json(prep / "private" / "labels.json", private)
    write_json(prep / "PREPARE.json", {"status": "complete", "manifest_sha256": MANIFEST_SHA256,
               "images_sha256": digest(prep / "public" / "images.json"),
               "labels_sha256": digest(prep / "private" / "labels.json")})
    pd.DataFrame(rows).to_csv(run / "profiles.csv", index=False)
    for name in ("profiles.npy", "corrected.npy"):
        np.save(run / name, np.array(matrix))
    write_json(run / "cells.json", cells)
    write_json(run / "excluded_cells.json", excluded)
    np.save(run / "embedding.npy", np.array(embedding, dtype=np.float32))
    write_json(run / "DELIVERY.json", {"status": "complete", "arm": "control", "provenance": {"commit": "test"},
        "prepare_sha256": digest(prep / "PREPARE.json"),
        "artifacts": {name: digest(run / name) for name in ("profiles.csv", "profiles.npy", "corrected.npy",
                                                            "cells.json", "excluded_cells.json",
                                                            "embedding.npy")}})
    return prep, run


def test_complete_fixed_design_scoring_and_tamper_guard(tmp_path):
    from benchmarks.full_CPJUMP1.harness.score import score
    prep, run = complete_design(tmp_path)
    result = score(prep, run)
    retrieval_reading = result["metrics"][result["retrieval_reading"]]
    assert retrieval_reading["n_genes"] == 92 and retrieval_reading["n_queries"] == 368 and retrieval_reading["mAP"] == 1
    assert result["primary"] == ["mAP", "macro_f1", "balanced_accuracy"]
    full = result["full_metrics"]
    assert full["layout"]["splits"]["main"]["n_genes"] == 110 and full["layout"]["splits"]["guard"]["n_genes"] == 92
    assert full["layout"]["splits"]["main"]["n_test_units"] == [202, 202]
    for metric in ("mAP", "macro_f1", "balanced_accuracy"):          # a gene-perfect representation
        assert full["metrics"][metric]["main"]["value"] == 1 and full["metrics"][metric]["guard"]["value"] == 1
        assert full["metrics"][metric]["vs_control"] is None
    assert full["verdicts"] is None
    assert "| mAP (unit profiles" in (run / "RESULT.md").read_text(encoding="utf-8")
    assert pd.read_csv(run / "evaluation" / "gene_similarity_cosine.csv", index_col=0).shape == (130, 130)
    with (run / "corrected.npy").open("ab") as h:
        h.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        score(prep, run)


def test_guard_split_never_trains_on_the_test_guide():
    from benchmarks.full_CPJUMP1.harness.score import full_layout
    meta = design()
    meta = pd.concat([meta, pd.DataFrame([dict(well="D1x", gene="D", guide="D1", plate=1, role="evaluation",
                                               pert_type="trt"),
                                          dict(well="D1y", gene="D", guide="D1", plate=2, role="evaluation",
                                               pert_type="trt")])], ignore_index=True)
    layout, units = full_layout(meta)
    assert layout.n_classes == 4
    for fold in layout.splits["guard"]:
        for train, test in fold:
            for unit in test:
                same_gene = units.gene[train] == units.gene[unit]
                assert same_gene.any() and not (units.guide[train][same_gene] == units.guide[unit]).any()
                assert (units.plate[train] != units.plate[unit]).all()
            assert "D" not in set(units.gene[test]) | set(units.gene[train])     # a single-guide gene: main only
    assert {units.gene[u] for fold in layout.splits["main"] for _, test in fold for u in test} == set("ABCD")


def test_verdicts_against_the_control_reading(tmp_path):
    from benchmarks.full_CPJUMP1.harness.score import score
    (tmp_path / "c").mkdir()
    (tmp_path / "n").mkdir()
    prep, control = complete_design(tmp_path / "c", cell_noise=0.05)
    _, noisy = complete_design(tmp_path / "n", cell_noise=2.0, seed=1)
    score(prep, control)
    result = score(prep, noisy, control=control)
    full = result["full_metrics"]
    assert set(full["verdicts"]) == {"mAP", "macro_f1", "balanced_accuracy"}
    assert full["verdicts"]["macro_f1"] == "LOSS" and full["verdicts"]["balanced_accuracy"] == "LOSS"
    assert full["verdicts"]["mAP"] == "NO_RESOLVABLE_DIFFERENCE"      # the same corrected well profiles


def test_a_well_without_cells_counts_as_cap_misses(tmp_path):
    from benchmarks.full_CPJUMP1.harness.score import score
    prep, run = complete_design(tmp_path, drop=("P1_W40",))          # an evaluation well (gene G20, guide 0)
    full = score(prep, run)["full_metrics"]
    assert full["n_units_without_cells"] == 1 and full["cell_cap"] == 100
    main = full["metrics"]["balanced_accuracy"]["main"]["value"]
    # plate P1 tested: G20 has 2 test wells, one of them 100 misses and one with 4 correct cells -> recall 4/104
    assert main == pytest.approx((1 - (1 - 4 / 104) / 110 + 1) / 2)
    assert full["metrics"]["mAP"]["main"]["value"] < 1
    assert full["reported"]["uncorrected_contract_profile_mAP"]["main"]["value"] < 1


def test_excluded_cells_are_sampled_misses(tmp_path):
    from benchmarks.full_CPJUMP1.harness.score import excluded_image, score
    assert excluded_image("default_P1_W40_0_cell_0001") == "P1_W40_0"
    prep, run = complete_design(tmp_path, exclude=("P1_W40",))       # every cell of G20's guide-0 well on P1
    full = score(prep, run)["full_metrics"]
    assert full["n_excluded_cells"] == 4 == full["n_excluded_sampled"] and full["n_units_without_cells"] == 1
    # plate P1 tested: G20's 8 test cells, its 4 excluded ones misses (not 100: the well has excluded cells)
    assert full["metrics"]["balanced_accuracy"]["main"]["value"] == pytest.approx((1 - 0.5 / 110 + 1) / 2)
