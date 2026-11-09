"""GT-free negative decisions, uncertain inputs, and per-image budget regressions."""
import time
import pytest
from shape.agent.verification.test.test_organoid_agentic import NativeOrganoidTool,image,managed,Engine
from shape.agent.verification.verifiers.organoid_segmentation import EmptyOrganoidReview,OrganoidSegmentationAgenticVerifier
from shape.tools.organoid_segmenter.runtime_limits import check_native_size
from shape.utils.schemas import QuestionFailed

class NegativeEngine(Engine):
    def __init__(self,presence='absent',confidence='high',interpretable=True,limitations=None):
        super().__init__();self.answer=EmptyOrganoidReview(target_presence=presence,confidence=confidence,interpretable=interpretable,
            evidence='The interpretable raw field contains background without any visible organoid.',limitations=limitations or [])
        self.review_calls=0
    def generate(self,content,response_format=None,**kwargs):
        if response_format is EmptyOrganoidReview:
            assert len(content)==2 and isinstance(content[1],bytes)
            assert 'ground truth' in content[0]
            self.review_calls+=1
            return {'content':self.answer,'usage':{'total_tokens':71}}
        return super().generate(content,response_format=response_format,**kwargs)


def test_confident_independent_negative_stops_after_default_without_forced_positive(image,tmp_path):
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    engine=NegativeEngine()
    resolutions,result,status=managed(image,tool,engine)
    assert status=='satisfied' and result['organoid_count']==0 and result['negative_confirmed']
    assert len(tool.calls)==1 and engine.review_calls==1
    assert resolutions[0].details['negative_confirmed']
    assert resolutions[0].details['quality_certified'] is False

@pytest.mark.parametrize('options',[{'presence':'uncertain'},{'confidence':'low'},{'interpretable':False},{'limitations':['clipped field']}])
def test_uncertain_empty_is_not_tuned_into_positive_or_certified(image,tmp_path,options):
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    resolutions,result,status=managed(image,tool,NegativeEngine(**options))
    assert status=='conceded' and len(tool.calls)==1
    assert not resolutions[0].details['negative_confirmed']
    assert resolutions[0].details['needs_review']


def test_no_reasoner_does_not_certify_or_expand_empty(image,tmp_path):
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    from shape.engine.errors import NoEngineError
    with pytest.raises(NoEngineError):managed(image,tool)
    assert len(tool.calls)==1


def test_reported_token_budget_stops_before_new_question():
    v=OrganoidSegmentationAgenticVerifier();v.tokens['total']=24000
    with pytest.raises(QuestionFailed,match='reported_token_budget'):
        v._ask('irrelevant',EmptyOrganoidReview,'plan')


def test_wall_budget_stops_before_new_question():
    v=OrganoidSegmentationAgenticVerifier();v.started=time.time()-301
    with pytest.raises(QuestionFailed,match='image_wall_budget'):
        v._ask('irrelevant',EmptyOrganoidReview,'plan')


def test_native_expansion_is_rejected_before_gpu_allocation():
    assert check_native_size((904,1224),None)==904*1224
    with pytest.raises(ValueError,match='native_pixel_budget'):
        check_native_size((904,1224),5)


def test_token_threshold_blocks_http_retry_after_a_malformed_expensive_reply(monkeypatch):
    from shape.agent.verification import base
    monkeypatch.setattr(base,'REASONER_RETRY_SECONDS',0)
    class ExpensiveMalformed:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            return {'content':'not valid JSON','usage':{'total_tokens':24000}}
    engine=ExpensiveMalformed();v=OrganoidSegmentationAgenticVerifier();v.llm_engine=engine
    with pytest.raises(QuestionFailed):v._ask('test',EmptyOrganoidReview,'plan')
    assert engine.calls==1 and v.organoid_budget_stop=='reported_token_budget'


def test_constant_input_cannot_be_certified_by_overconfident_negative_reply(tmp_path):
    import numpy as np
    from PIL import Image
    path=tmp_path/'black.png';Image.fromarray(np.zeros((600,600),dtype=np.uint8)).save(path)
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    resolutions,_,status=managed(str(path),tool,NegativeEngine())
    assert status=='conceded' and not resolutions[0].details['negative_confirmed']
    assert len(tool.calls)==1


def test_confirmed_negative_calibration_never_transfers_positive_levers():
    from types import SimpleNamespace
    from shape.agent.verification import segmentation_cohort as cohort
    from shape.utils.schemas import StageResolution,VLRRResult
    card=OrganoidSegmentationAgenticVerifier()
    specs=[dict(image_id=str(i),path=str(i)) for i in range(12)]
    def run_one(spec,options):
        res=StageResolution(result={'mask_path':'empty'},status='satisfied',params={'min_size':100},
            verdict=VLRRResult('PASS',0,{}),details={'call':'default','measured_problems':{'default':[]},'negative_confirmed':True})
        return [res],res.result,res.status,None
    _,record=cohort.run(specs,run_one,lambda items,fn:[fn(s) for s in items],
        lambda spec:SimpleNamespace(confidence='high',specimen_context='organoid',modality='brightfield'),
        card.defaults(),card.levers,tool_name='Organoid_Segmenter_Tool')
    assert record['counts']=={'calibration':3,'per_image_vrr':9}
    assert record['groups'][0]['calibration_eligible']==[]


def test_confirmed_negative_uses_raw_evidence_when_empty_overlay_is_dark(image,tmp_path):
    class DarkEmpty(NativeOrganoidTool):
        def execute(self,*args,**kwargs):
            value=super().execute(*args,**kwargs)
            value['overlay_qc']['overlay_is_black']=True
            return value
    tool=DarkEmpty(tmp_path,fail_default=True)
    resolutions,result,status=managed(image,tool,NegativeEngine())
    assert status=='satisfied' and result['negative_confirmed'] and len(tool.calls)==1
    assert result['overlay_qc']['overlay_is_black']  # native observation is not falsified


@pytest.mark.parametrize('budget',['tokens','native_calls'])
def test_budget_stop_retains_completed_native_candidate_without_final_question(image,tmp_path,budget):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    from shape.agent.verification.verifiers.segmentation import SegPlan,SegChoice
    class BudgetEngine(Engine):
        def generate(self,content,response_format=None,**kwargs):
            value=super().generate(content,response_format=response_format,**kwargs)
            if budget=='tokens' and response_format is SegPlan:
                value['usage']={'total_tokens':24000}
            return value
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    engine=BudgetEngine([[move(cellprob_threshold=-2)]])
    options={'max_native_calls':2} if budget=='native_calls' else {}
    resolutions,result,status=managed(image,tool,engine,**options)
    res=resolutions[0]
    assert result is not None and result['organoid_count']==1
    assert status=='conceded' and res.details['needs_review'] and res.details['accepted_with_flags']
    assert not res.details['quality_certified'] and res.details['delivery_complete']
    assert res.details['retained_candidate_basis']=='best completed native QC; not quality-certified'
    assert res.details['choice']=='budget_retained_candidate'
    assert res.details['organoid_budget_stop']==('native_call_budget' if budget=='native_calls' else 'reported_token_budget')
    assert len(tool.calls)==2 and not engine.asked(SegChoice)


def test_budget_retention_excludes_a_later_failed_trial(image,tmp_path):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    class LaterFailed(NativeOrganoidTool):
        def execute(self,image,cellprob_threshold=0,attempt_id=None,**kwargs):
            value=super().execute(image,cellprob_threshold=cellprob_threshold,attempt_id=attempt_id,**kwargs)
            if len(self.calls)>=3:
                value.update(status='failed',error='invalid native overlay')
                value['overlay_qc']['overlay_is_black']=True
            return value
    from shape.agent.verification.verifiers.segmentation import SegPlan
    class ExpensiveEngine(Engine):
        def generate(self,content,response_format=None,**kwargs):
            value=super().generate(content,response_format=response_format,**kwargs)
            if response_format is SegPlan:value['usage']={'total_tokens':12000}
            return value
    tool=LaterFailed(tmp_path,fail_default=True)
    engine=ExpensiveEngine([[move(cellprob_threshold=-2)],[move('agent_1',cellprob_threshold=-3)]])
    resolutions,result,status=managed(image,tool,engine,max_native_calls=3)
    assert status=='conceded' and result is not None and result['status']=='success'
    assert result['organoid_mask_path'].endswith('attempt_2.npy') and len(tool.calls)==3
    assert resolutions[0].details['selected']==2


def test_budget_retention_does_not_invent_a_candidate_when_all_native_trials_failed(image,tmp_path):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    tool=NativeOrganoidTool(tmp_path,fail_default=True,invalid_reference=True)
    engine=Engine([[move(cellprob_threshold=-2)]])
    resolutions,result,status=managed(image,tool,engine,max_native_calls=2)
    assert result is None and status=='conceded'
    assert not resolutions[0].details['delivery_complete']


def test_confident_positive_presence_with_boundary_limitations_still_allows_repair(image,tmp_path):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    engine=NegativeEngine(presence='present',limitations=['The visible target has a fuzzy boundary.'])
    engine.rounds=[[move(cellprob_threshold=-2)]]
    engine.final=['agent_1']
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    resolutions,result,status=managed(image,tool,engine)
    assert result['organoid_count']==1 and status=='satisfied' and len(tool.calls)==2
    review=resolutions[0].details['negative_reviews'][1]
    assert review['target_presence']=='present' and not review['accepted']


def test_reserve_stops_planning_but_allows_final_visual_selection(image,tmp_path):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    from shape.agent.verification.verifiers.segmentation import SegPlan,SegChoice
    class CostEngine(Engine):
        def generate(self,content,response_format=None,**kwargs):
            value=super().generate(content,response_format=response_format,**kwargs)
            if response_format is SegPlan:value['usage']={'total_tokens':7000}
            return value
    tool=NativeOrganoidTool(tmp_path,fail_default=True)
    engine=CostEngine([[move(cellprob_threshold=-2)],[move('agent_1',cellprob_threshold=-3)]],final=['agent_2'])
    resolutions,result,status=managed(image,tool,engine)
    res=resolutions[0]
    assert len(tool.calls)==3 and len(engine.asked(SegPlan))==2
    assert engine.asked(SegChoice) and result is not None
    assert res.details['search_budget_stop']=='final_selection_reserve'
    assert res.details['organoid_budget_stop'] is None
    assert res.details['choice']!='budget_retained_candidate'
    assert not res.details['quality_certified']


def test_search_retry_cannot_spend_final_selection_reserve(monkeypatch):
    from shape.agent.verification import base
    from shape.agent.verification.verifiers.segmentation import SegPlan
    monkeypatch.setattr(base,'REASONER_RETRY_SECONDS',0)
    class Malformed:
        calls=0
        def generate(self,*args,**kwargs):
            self.calls+=1
            return {'content':'not valid JSON','usage':{'total_tokens':15000}}
    engine=Malformed();v=OrganoidSegmentationAgenticVerifier();v.llm_engine=engine
    with pytest.raises(QuestionFailed):v._ask('test',SegPlan,'round')
    assert engine.calls==1 and v.search_budget_stop=='final_selection_reserve'
    assert v.organoid_budget_stop is None


def test_final_question_can_use_reserved_balance():
    from shape.agent.verification.verifiers.segmentation import SegChoice
    v=OrganoidSegmentationAgenticVerifier();v.tokens['total']=15000
    v._check_search_reserve('final')
    v._check_search_reserve('final_retry')
    with pytest.raises(QuestionFailed):v._check_search_reserve('repair')


def test_major_visual_defect_is_not_clean_cohort_calibration(image,tmp_path):
    from shape.agent.verification.test.test_segmentation_golden_path import move
    from shape.agent.verification.segmentation_cohort import problems_of
    engine=Engine([[move(cellprob_threshold=-2)]],final=['agent_1'],severity={'agent_1':'major'})
    resolutions,result,status=managed(image,NativeOrganoidTool(tmp_path,fail_default=True),engine)
    res=resolutions[0]
    assert result is not None and result['organoid_count']==1
    assert status=='conceded' and res.details['needs_review']
    assert res.details['selected_visual_assessment']['severity']=='major'
    assert res.details['visual_review_required'] and not res.details['quality_certified']
    assert problems_of(resolutions)  # cannot transfer its parameters as clean calibration


def test_small_remaining_budget_still_allows_a_search_before_reserving_final():
    v=OrganoidSegmentationAgenticVerifier();v.max_reported_tokens=6000
    v._check_search_reserve('round')
    v.tokens['total']=3000
    with pytest.raises(QuestionFailed):v._check_search_reserve('round')
    v._check_search_reserve('final')
