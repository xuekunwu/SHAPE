"""Experimental policy keeps VRR/tool guards while removing reference privilege."""
from types import SimpleNamespace
import pytest
from shape.agent.verification.balanced_policy import balanced_override, repair_packet
from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier
from shape.utils.schemas import VLRRResult


def comparison(**changes):
    return dict(dict(preferred_call='candidate',baseline_call='default',confidence='high',
                reason='Visible adjacent cells are separated; the default merges them',
                cited_readouts=['paired overlays','tiny-mask fraction'],loss_explained=False,
                loss_explanation=''),**changes)


def test_equal_severity_and_problem_counts_do_not_automatically_veto():
    assessments={c:dict(severity='major') for c in ['candidate','default']}
    assert balanced_override('candidate','default',dict(cell_count=100),dict(cell_count=100),assessments,comparison())[0]
    assessments['candidate']['severity']='major';assessments['default']['severity']='minor'
    assert not balanced_override('candidate','default',{}, {},assessments,comparison())[0]


def test_count_or_coverage_collapse_requires_visible_explanation():
    reports={c:dict(severity='minor') for c in ['candidate','default']}
    args=('candidate','default',dict(cell_count=40,foreground_coverage=.1),dict(cell_count=100,foreground_coverage=.4),reports)
    assert not balanced_override(*args,comparison())[0]
    answer=comparison();answer.update(loss_explained=True,loss_explanation='Raw-image comparison shows the removed masks are debris')
    assert balanced_override(*args,answer)[0]
    answer['confidence']='low';assert not balanced_override(*args,answer)[0]


def test_native_guard_still_blocks_invalid_candidate(monkeypatch):
    monkeypatch.setenv('SHAPE_VRR_POLICY','balanced_v1')
    v=SegmentationAgenticVerifier()
    monkeypatch.setattr(v,'guard',lambda inp,result:['invalid overlay'])
    assert not v.override_allowed(None,{'candidate':(None,{},VLRRResult('PASS',.5,{}))},'candidate','default')[0]


def test_balanced_cohort_does_not_ask_model_to_rank_one_admissible_call(monkeypatch):
    monkeypatch.setenv('SHAPE_VRR_POLICY','balanced_v1')
    v=SegmentationAgenticVerifier();v.stop_reason='cohort_applied'
    monkeypatch.setattr(v,'_ask_optional',lambda *args:pytest.fail('ranking one candidate is redundant'))
    assert v.choose(None,{},['default'])==('default','only_admissible_call')
    assert v.choice['answers']==[]
    assert v.efficiency_events[-1]['reason']=='only_admissible_call'


def test_original_valid_default_is_the_fallback_only_when_enabled(monkeypatch):
    monkeypatch.setenv('SHAPE_VRR_POLICY','balanced_v1');v=SegmentationAgenticVerifier()
    v.rounds=[[1]];v._names[1]='default';v.reference_calls={'reference':{}}
    monkeypatch.setattr(v,'guard',lambda *args:[])
    attempts={c:(None,{},VLRRResult('PASS',.5,{})) for c in ['default','reference']}
    assert v.reference_default(None,attempts)=='default'
    monkeypatch.setenv('SHAPE_VRR_POLICY','legacy');legacy=SegmentationAgenticVerifier()
    legacy.rounds=[[1]];legacy._names[1]='default';legacy.reference_calls={'reference':{}}
    monkeypatch.setattr(legacy,'guard',lambda *args:[])
    assert legacy.reference_default(None,attempts)=='reference'


def test_structural_repair_packet_does_not_retransmit_images_or_full_history():
    verifier=SimpleNamespace(task_text=lambda:'whole cells',levers=['diameter'],_pending_focus=[],
                             narrowing=False,search_strategy='four_stage')
    plan=SimpleNamespace(model_dump=lambda:dict(moves=[dict(parent='default',changes=[])]))
    packet=repair_packet(verifier,plan,[],['diameter outside range'],1)
    assert len(packet)==1 and isinstance(packet[0],str)
    assert 'diameter outside range' in packet[0] and 'whole cells' in packet[0]


@pytest.mark.parametrize('strategy',['legacy','four_stage'])
def test_balanced_vrr_still_runs_real_quantify_reason_and_refine(image,tmp_path,monkeypatch,strategy):
    from shape.agent.verification.test.test_segmentation_golden_path import Engine,run,move
    from shape.agent.verification.verifiers.segmentation import BalancedSegChoice,BalancedSegPlan,SegPlan,SegChoice,CallAssessment
    monkeypatch.setenv('SHAPE_VRR_POLICY','balanced_v1')
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY',strategy)
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_SEGMENTATION',strategy)
    class BalancedEngine(Engine):
        def generate(self,content,response_format=None):
            if response_format is BalancedSegChoice:
                assessments=[CallAssessment(call=c,problem='some defects',severity='minor',confidence='high')
                             for c in ['default','reference','agent_1']]
                answer=comparison();answer['preferred_call']='agent_1'
                return dict(content=BalancedSegChoice(call='agent_1',reasoning='paired visible evidence',
                         assessments=assessments,cited_readouts=['overlays','shape QC'],comparison=answer))
            if isinstance(response_format,type) and issubclass(response_format,BalancedSegPlan):
                value=super().generate(content,response_format=SegPlan)['content']
                payload=value.model_dump()
                if strategy=='four_stage':
                    payload['search']=dict(phase='cover' if value.moves else 'confirm',reason='Measured candidate comparison')
                return dict(content=response_format.model_validate(payload))
            return super().generate(content,response_format=response_format)
    engine=BalancedEngine([[move('default',min_size=20)]],final=['agent_1'])
    resolution,verifier,calls=run(image,tmp_path,engine,budget=4,monkeypatch=monkeypatch)
    assert len(calls)>=2 and verifier.plans
    assert resolution.details['vrr_policy']=='balanced_v1'
    assert resolution.details['call'] in ['default','agent_1']


# Reuse the existing real-file fixture without duplicating its image construction.
from shape.agent.verification.test.test_segmentation_golden_path import image
