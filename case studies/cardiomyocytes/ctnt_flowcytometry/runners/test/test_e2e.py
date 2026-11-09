"""Leakage, variable-view aggregation and invalid-run rejection tests (no GPU required)."""
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
data = importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.dataset.preparation")
score = importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.evaluation.heldout_evaluation")
provenance = importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.runtime.source_snapshot")
predictor = importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.prediction.well_predictor")
from benchmarks.full_CPJUMP1.harness.data import well_profiles

def image_rows():
    return [dict(image_id=f"w{w}s{s}", well=f"w{w}", plate=f"p{w%7}", plate_id=f"p{w%7}",
                 experiment_date=f"d{w%7}", partition="held_out" if w%3 == 2 else "seen", site=f"s{s}",
                 usage=data.USAGES[w%3], shape=[1,2048,2048], dtype="uint16",
                 calibration_group="", negative_control=False)
            for w in range(87) for s in range(25 if w<12 else 4)]

def test_split_independent_of_row_order_and_labels():
    from collections import Counter
    rows = split_fixture()
    first = data.well_split(rows, 20261015)
    assert first == data.well_split([{**r,"ctnt_positive_percent":"99"} for r in reversed(rows)], 20261015)
    assert first == data.well_split(rows*3, 20261015)
    assert set(first.values()) == set(data.USAGES)
    assert Counter(first.values()) == dict(zip(data.USAGES, (35, 9, 43)))
    assert first != data.well_split(rows, 1)


def split_fixture():
    return [dict(plate_id=f"p{p}",well_id=f"w{w}",experiment_date=f"d{0 if p<4 else 1 if p<6 else 2}",
                 donor_id=str(w%11),ctnt_positive_percent=w)
            for p,n in enumerate([3,3,3,3,16,24,35]) for w in range(n)]


def test_split_balances_dates_then_plates_and_rejects_conflicts():
    from collections import Counter
    rows = split_fixture()
    splits = data.well_split(rows, 20261015)
    for date in {r["experiment_date"] for r in rows}:
        group = [r for r in rows if r["experiment_date"] == date]
        counts = Counter(splits[r["plate_id"],r["well_id"]] for r in group)
        for usage, total in zip(data.USAGES, (35,9,43)):
            assert abs(counts[usage]-len(group)*total/87) < 1
        for plate in {r["plate_id"] for r in group}:
            wells = [r for r in group if r["plate_id"] == plate]
            actual = Counter(splits[r["plate_id"],r["well_id"]] for r in wells)
            for usage in data.USAGES:
                assert abs(actual[usage]-len(wells)*counts[usage]/len(group)) < 1
    with pytest.raises(ValueError,match="conflicting"):
        data.well_split(rows+[{**rows[0],"experiment_date":"changed"}], 1)

def test_smoke_keeps_all_views_and_all_splits():
    rows = image_rows()
    data.validate_rows(rows)
    smoke = data.smoke_rows(rows)
    assert len(smoke) == 37  # nearest feasible image count to 5% with 2/1/1 complete wells
    assert {r["usage"] for r in smoke} == set(data.USAGES)
    assert smoke == [r for r in rows if r["well"] in {s["well"] for s in smoke}]

def test_missing_view_and_well_leakage_rejected():
    rows = image_rows()
    with pytest.raises(ValueError):
        data.validate_rows(rows[:-1])
    rows[1]["usage"] = data.USAGES[1]
    with pytest.raises(ValueError, match="crosses"):
        data.validate_rows(rows)

@pytest.mark.parametrize("sites", [2,4,25])
def test_aggregation_weights_views_equally(sites):
    images, cells, features = [], [], []
    for s in range(sites):
        images.append(dict(image_id=f"i{s}",well="w",plate="p",usage="development_train",
                           calibration_group="",negative_control=False))
        for c in range(1 if s==0 else 3):
            cells.append(dict(cell_id=f"i{s}c{c}",image_id=f"i{s}"))
            features.append([s])
    x, table = well_profiles(np.asarray(features),cells,images,allowed_sites=(sites,))
    assert x[0,0] == pytest.approx(np.mean(np.arange(sites)))
    assert len(table) == 1
    with pytest.raises(ValueError, match="every selected site"):
        well_profiles(np.asarray(features)[1:],cells[1:],images,allowed_sites=(sites,))

def test_default_two_site_contract_preserved():
    images = [dict(image_id=f"i{s}",well="w",plate="p",usage="development_train",
                   calibration_group="",negative_control=False) for s in range(4)]
    cells = [dict(cell_id=f"c{s}",image_id=f"i{s}") for s in range(4)]
    with pytest.raises(ValueError, match="site count"):
        well_profiles(np.ones((4,2)),cells,images)

def regression_fixture():
    x = np.array([[0.,1.],[1.,2.],[2.,3.],[1000.,2000.]])
    table = [{"well":str(i),"usage":u} for i,u in enumerate(
        ["development_train","development_train","development_val","evaluation_only"])]
    return x,table,{"0":10.,"1":20.,"2":30.}

def test_regression_scaling_uses_training_only():
    x,table,labels = regression_fixture()
    predictions,raw,mean,params,selected = predictor.fit_predict(x,table,labels)
    np.testing.assert_array_equal(params["mean"],[.5,1.5])
    assert mean == pytest.approx(15)
    assert np.all((predictions>=0)&(predictions<=100))
    assert selected["refit_on_validation"] is False
    assert np.isfinite(raw).all()

def test_evaluation_target_cannot_enter_fit():
    x,table,labels = regression_fixture()
    with pytest.raises(ValueError,match="development targets only"):
        predictor.fit_predict(x,table,{**labels,"3":.8})

def test_metrics_percentage_points_and_degenerate_r2():
    result = score.metrics([25,25],[35,15])
    assert result["mae_percentage_points"] == pytest.approx(10)
    assert result["r2"] is None
    with pytest.raises(ValueError):
        score.metrics([.5],[float("nan")])
    with pytest.raises(ValueError, match="percentages"):
        score.metrics([101], [50])


def test_old_preparation_rejected(tmp_path):
    data.write_json(tmp_path/"PREPARE.json", {"status":"complete", "split_unit":"donor"})
    with pytest.raises(ValueError, match="obsolete"):
        data.load_prepared(tmp_path)


def test_batch_metadata_survives_aggregation():
    images = [dict(image_id=f"i{s}",well="w",plate="p",usage="development_train",
                   calibration_group="",negative_control=False,experiment_date="d",plate_id="p",partition="seen")
              for s in range(4)]
    cells = [dict(cell_id=f"c{s}",image_id=f"i{s}") for s in range(4)]
    _, table = well_profiles(np.ones((4,2)),cells,images,allowed_sites=(4,))
    assert table.iloc[0].experiment_date == "d"
    assert table.iloc[0].plate_id == "p"
    images[1]["experiment_date"] = "other"
    with pytest.raises(ValueError, match="inconsistent experiment_date"):
        well_profiles(np.ones((4,2)),cells,images,allowed_sites=(4,))


def test_percent_export_and_batch_scores(tmp_path, monkeypatch):
    import csv
    prep, run = tmp_path/"prep", tmp_path/"run"
    rows = [dict(image_id=f"i{i}",well=str(i),usage=u,partition="held_out" if u==data.USAGES[2] else "seen",
                 experiment_date="d0",plate_id="p0")
            for i,u in enumerate([data.USAGES[0],data.USAGES[0],data.USAGES[1],data.USAGES[2]])]
    labels = {str(i):dict(usage=r["usage"],ctnt_positive_percent=10*(i+1),donor_id="donor") for i,r in enumerate(rows)}
    data.write_json(prep/"private"/"development_labels.json",{k:v for k,v in labels.items() if k!="3"})
    data.write_json(prep/"PREPARE.json",{"development_labels_sha256":data.digest(prep/"private"/"development_labels.json")})
    data.write_json(run/"CONFIG.json",{"images":[r["image_id"] for r in rows],"segmentation_regime":"fixture"})
    with (run/"profiles.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["well","usage"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in ("well","usage")} for r in rows)
    np.save(run/"corrected.npy",np.array([[0],[1],[2],[3]]))
    data.write_json(run/"cells.json",[{"well":r["well"]} for r in rows])
    monkeypatch.setattr(score,"load_prepared",lambda p:rows)
    monkeypatch.setattr(predictor,"load_prepared",lambda p:rows)
    monkeypatch.setattr(score,"verify_private",lambda p,r:labels)
    predictor.predict(prep,run)
    predictions=data.read_json(run/"predictions.json")
    assert predictions[-1]["predicted_ctnt_positive_percent"] == pytest.approx(40,abs=1)
    assert predictions[-1]["baseline_ctnt_positive_percent"] == 15
    data.write_json(run/"DELIVERY.json",{"artifacts":{"predictions.json":data.digest(run/"predictions.json")},
                    "prepare_sha256":data.digest(prep/"PREPARE.json"),"smoke":False,"source_sha256":"fixture"})
    result=score.score(prep,run)
    assert result["scope"] == "batch_balanced_well_holdout"
    assert result["target"] == "ctnt_positive_percent"
    assert result["training_mean_baseline"]["mae_percentage_points"] == 25
    assert result["batch_metrics"]["experiment_date"]["macro_mae_percentage_points"] == result["primary"]["mae_percentage_points"]
    with (run/"heldout_predictions.csv").open() as f:
        assert float(next(csv.DictReader(f))["ctnt_positive_percent"]) == 40
    data.write_json(run/"predictions.json",[])
    with pytest.raises(ValueError,match="artifact changed"):
        score.score(prep,run)

def test_source_mismatch_rejected(monkeypatch):
    monkeypatch.setattr(provenance,"source_manifest",lambda:{"source_sha256":"a"*64,"files":{}})
    with pytest.raises(RuntimeError,match="INVALID_SOURCE_SNAPSHOT"):
        provenance.verify_source("b"*64)

def test_dataset_adapter_preserves_other_datasets():
    from benchmarks.full_CPJUMP1.harness.pipeline import dataset
    assert dataset({}) .allowed_sites == (2,)
    flow = dataset({"dataset":"ctnt_flowcytometry"})
    assert flow.allowed_sites == (4,25)
    assert flow.image_shape == (2048,2048)
    assert flow.loader(None) is data.load_prepared

def test_outputs_cannot_overwrite_sources(tmp_path):
    with pytest.raises(ValueError):
        data.fresh_output(tmp_path/"new",tmp_path)
    with pytest.raises(FileExistsError):
        data.fresh_output(tmp_path)

def test_delivery_requires_primary_outputs_and_selected_masks(tmp_path):
    e2e = importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.experiment_runner")
    data.write_json(tmp_path/"segmentation.json", [{"mask":str(tmp_path/"mask.tif")}])
    data.write_json(tmp_path/"SSL_SELECTION.json", {})
    with pytest.raises(ValueError,match="missing delivery artifact"):
        e2e.delivery_files(tmp_path)
    names = ("cells.json","excluded_cells.json","embedding.npy","profiles.npy","profiles.csv",
             "corrected.npy","batch_SEARCH.json","CONFIG.json","PREFLIGHT.json","predictions.json",
             "REGRESSOR.json","regressor.npz","mask.tif")
    for n in names:
        (tmp_path/n).write_bytes(b"fixture")
    for s in e2e.STAGES:
        (tmp_path/f"{s}_COMPLETE.json").write_text("{}")
        if s != "predict":
            (tmp_path/f"{s}_USAGE.json").write_text("{}")
    (tmp_path/"display.png").write_bytes(b"preview")
    files=e2e.delivery_files(tmp_path)
    assert tmp_path/"mask.tif" in files and tmp_path/"predictions.json" in files
    assert tmp_path/"display.png" not in files
    data.write_json(tmp_path/"SSL_SELECTION.json", {"checkpoint":str(tmp_path/"missing.pt")})
    with pytest.raises(ValueError,match="missing delivery artifact"):
        e2e.delivery_files(tmp_path)
