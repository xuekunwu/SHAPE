"""Repeated measured evidence is distinct from failed tools or invalid model plans."""
import pytest
import threading

from shape.agent.verification.base import DuplicateConfiguration
from shape.agent.verification.verifiers.test.test_crop import _Engine as _CropEngine, _bound, _move, _run
from shape.agent.verification.verifiers.crop import CropPlan


class _Engine(_CropEngine):
    """The offline judge recognizes the real compact and full repair formats."""
    def generate(self,content,response_format=None):
        if content[0].startswith('Structural repair only.'):
            assert response_format is CropPlan
            self.threads.append(threading.get_ident())
            self.prompts.append(content[0]);self.contents.append(content)
            return {'content':CropPlan(analysis='repair',moves=list(self.repair or [])),'usage':{}}
        return super().generate(content,response_format=response_format)


@pytest.fixture(autouse=True,params=['legacy','balanced_v1'])
def policy(request,monkeypatch):
    monkeypatch.setenv('SHAPE_VRR_POLICY',request.param)
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_CROP','legacy')


def test_repeated_finished_geometry_stops_without_false_execution_error(tmp_path):
    # The fixture's crops all fit below this cap, so x=100 is a different
    # parameter request with exactly the already-measured default geometry.
    duplicate=_move(crop_size=100)
    engine=_Engine(rounds=[[duplicate]],repair=[duplicate],final=['default'])
    calls=[]
    resolution,verifier,waves=_run(tmp_path,engine,budget=4,calls=calls)
    assert waves==[1] and len(calls)==1
    assert verifier.stop_reason=='no_new_configuration'
    assert verifier.reasoner_failures==0 and verifier.execution_errors==[]
    assert verifier.fallback_rounds==0 and resolution.result is not None
    assert resolution.details['call']=='default'
    events=resolution.details['reused_configurations']
    assert len(events)==2
    assert all(e['known_calls']==['default'] and e['native_rerun'] is False for e in events)


def test_known_duplicate_does_not_discard_a_novel_valid_move(tmp_path):
    engine=_Engine(rounds=[[_move(crop_size=100),_move(margin_mode='tight')]],final=['agent_1'])
    calls=[]
    resolution,verifier,waves=_run(tmp_path,engine,budget=4,cores=8,calls=calls)
    assert waves==[1,1] and len(calls)==2
    assert verifier.reasoner_failures==0 and verifier.execution_errors==[]
    assert resolution.details['reused_configurations']


def test_true_invalid_repair_is_not_hidden_by_a_reused_configuration(tmp_path):
    duplicate=_move(crop_size=100)
    engine=_Engine(rounds=[[duplicate]],repair=[duplicate,_move(margin_frac=-1)],final=['default'])
    resolution,verifier,waves=_run(tmp_path,engine,budget=4,cores=8)
    assert waves==[1]
    assert verifier.reasoner_failures==1
    assert verifier.execution_errors[0]['question']=='repair'
    assert verifier.stop_reason=='no_valid_move'


def test_failed_native_configuration_cannot_be_reused_as_healthy_evidence(tmp_path):
    failure=_move(margin_mode='tight')
    engine=_Engine(rounds=[[failure],[failure]],repair=[failure],final=['default'])
    calls=[]
    resolution,verifier,waves=_run(tmp_path,engine,budget=4,calls=calls,
        crash=lambda params:params.get('margin_mode')=='tight')
    assert sum(p.get('margin_mode')=='tight' for p in calls)==1
    assert len(resolution.details['failed'])==1
    assert verifier.reasoner_failures==1 and verifier.execution_errors


def test_validation_distinguishes_finished_failed_and_pending_configurations(tmp_path):
    verifier,attempts=_bound(tmp_path,[[_move(margin_mode='tight')]])
    inert=dict(parent='agent_1',changes=[dict(lever='margin_frac',value='0.4')])
    with pytest.raises(DuplicateConfiguration) as caught:
        verifier.validate(inert,attempts,set())
    assert caught.value.reusable and caught.value.calls==('agent_1',)
    request=verifier.validate(dict(parent='input',changes=[dict(lever='crop_size',value='80')]),attempts,set())
    with pytest.raises(DuplicateConfiguration) as pending:
        verifier.validate(dict(parent='input',changes=[dict(lever='crop_size',value='81')]),
                          attempts,{request['config_key']})
    assert pending.value.pending and not pending.value.reusable
    verifier.failed.append(dict(params=attempts['agent_1'][1]))
    with pytest.raises(DuplicateConfiguration) as failed:
        verifier.validate(inert,attempts,set())
    assert failed.value.failed and not failed.value.reusable
