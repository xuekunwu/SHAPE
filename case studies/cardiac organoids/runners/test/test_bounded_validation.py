"""Replay gates reject changed costs, incomplete records and unstable artifact bytes."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location("bounded_validation",Path(__file__).parents[1]/"segmentation/validate_bounded_stage.py")
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def summary():
    d={k:0 for k in mod.STABLE};d.update(n_planned=24,n_recorded=24,n_delivered=24,tokens=1000,llm_calls=3)
    return d

@pytest.mark.parametrize("key",["n_recorded","n_delivered","unknown_usage","incomplete_tool_requests"])
def test_incomplete_or_unknown_run_cannot_advance(key):
    d=summary();d[key]+=1
    with pytest.raises(ValueError):mod.check_complete(d,24)

@pytest.mark.parametrize("key",["tokens","llm_calls","map_50_95"])
def test_replay_cannot_silently_rerun_or_change_results(key):
    before=summary();after=dict(before);after[key]+=1
    with pytest.raises(ValueError):mod.check_replay(before,after,b"csv",b"csv")

def test_replay_requires_identical_export_bytes():
    d=summary()
    with pytest.raises(ValueError):mod.check_replay(d,d,b"a,b",b"b,a")
    mod.check_complete(d,24);mod.check_replay(d,d,b"csv",b"csv")
