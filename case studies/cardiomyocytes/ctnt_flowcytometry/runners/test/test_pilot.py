import importlib
reasoning=importlib.import_module('case studies.cardiomyocytes.ctnt_flowcytometry.runners.runtime.reasoning')
from collections import Counter
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
PKG = "case studies.cardiomyocytes.ctnt_flowcytometry.runners"
data = importlib.import_module(PKG+".dataset.preparation")
pilot = importlib.import_module(PKG+".dataset.sampling")


def public_rows():
    raw = [dict(plate_id=f"p{p}",well_id=f"w{w}",experiment_date=f"d{0 if p<4 else 1 if p<6 else 2}")
           for p,n in enumerate([3,3,3,3,16,24,35]) for w in range(n)]
    splits = data.well_split(raw,20261015)
    return [{**r,"well":f"{r['plate_id']}-{r['well_id']}","image_id":f"i{i}-{s}",
             "usage":splits[r['plate_id'],r['well_id']]}
            for i,r in enumerate(raw) for s in range(25 if i<12 else 4)]


def test_sample_preserves_splits_whole_wells_and_is_label_independent():
    rows = public_rows()
    chosen = pilot.select(rows)
    wells = {r['well']:r for r in chosen}
    assert Counter(r['usage'] for r in wells.values()) == dict(zip(data.USAGES,(7,2,9)))
    assert {r['image_id'] for r in chosen} == {r['image_id'] for r in rows if r['well'] in wells}
    altered = [{**r,"ctnt_positive_percent":99} for r in reversed(rows)]
    assert [r['image_id'] for r in chosen] == [r['image_id'] for r in pilot.select(altered)]
    assert [r['image_id'] for r in chosen] != [r['image_id'] for r in pilot.select(rows,1)]


def test_pilot_rejects_changed_parent_or_membership(tmp_path,monkeypatch):
    monkeypatch.setattr(data,"load_prepared",lambda p:public_rows())
    data.write_json(tmp_path/"PREPARE.json",{"fixed":"parent"})
    path=tmp_path/"PILOT.json"
    record=pilot.manifest(tmp_path)
    data.write_json(path,record)
    assert pilot.verify(tmp_path,path)==record
    data.write_json(path,{**record,"images":record['images'][:-1]})
    with pytest.raises(ValueError,match="differs"):
        pilot.verify(tmp_path,path)
    data.write_json(path,record)
    data.write_json(tmp_path/"PREPARE.json",{"changed":"parent"})
    with pytest.raises(ValueError,match="differs"):
        pilot.verify(tmp_path,path)


@pytest.mark.parametrize("stage,effort", [("segment","medium"),("crop","medium"),("embed","high"),("batch","high")])
def test_image_medium_reaches_engine_without_changing_frozen_pilot(stage, effort, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks._shared.harness import shape_arm
    from shape.agent.verification import vision_engine
    protocol = reasoning.reasoning_protocol("image-medium")
    for key,value in pipeline.stage_env("shape",stage,protocol).items():
        monkeypatch.setenv(key,value)
    raw = object()
    monkeypatch.setattr(vision_engine,"build_vision_engine",lambda m:raw)
    monkeypatch.setattr(shape_arm,"build_engine",lambda m,engine:engine)
    engine = pipeline.engine_for({"arm":"shape","model":"gpt-5.6-luna","reasoning":{"protocol":protocol}},stage)
    assert engine.engine is raw and engine.max_tokens == 16000
    assert pipeline.check_stage_env("shape",stage,protocol)["effort"] == effort
    assert reasoning.reasoning_protocol()["table"][stage]["effort"] == "high"
    e2e = importlib.import_module(PKG+".workflows.experiment_runner")
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="separate SHAPE"):
        e2e.requested_protocol(SimpleNamespace(arm="codex",reasoning_profile="image-medium"))


def test_codex_dataset_has_all_prompts_and_native_worker_parser():
    from benchmarks.full_CPJUMP1.harness.pipeline import dataset
    spec=dataset({"dataset":"ctnt_flowcytometry"})
    for stage in ("segment","crop","embed","batch"):
        assert spec.inputs[stage].format(n=114,cells=128,dtype="uint16",rule="SKIP")
    parser=importlib.import_module(PKG+".workflows.experiment_runner").parser()
    args=parser.parse_args(["native-trial","embed","--prep","prep","--run","run","--config","config"])
    assert args.config==Path("config")
    with pytest.raises(SystemExit):
        parser.parse_args(["run","--prep","p","--out","r","--assets-root","a","--expected-source","s",
                           "--smoke","--pilot-manifest","pilot"])


def test_comparison_rejects_different_pilots_and_efforts():
    from benchmarks.full_CPJUMP1.harness.pipeline import reasoning_record
    compare=importlib.import_module(PKG+".evaluation.arm_comparison")
    configs={a:{"arm":a,"images":["i1"],"pilot_sha256":"pilot","reasoning":reasoning_record(a,reasoning.reasoning_protocol())}
             for a in ("control","shape","codex")}
    compare.comparable(configs)
    configs['shape']['images']=['different']
    with pytest.raises(ValueError,match="differ"):
        compare.comparable(configs)
    configs['shape']['images']=['i1']
    configs['codex']['reasoning']=reasoning_record('codex')
    with pytest.raises(ValueError,match="all-high"):
        compare.comparable(configs)
