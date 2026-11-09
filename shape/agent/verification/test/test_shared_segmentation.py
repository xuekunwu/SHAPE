from copy import deepcopy
import threading
import time

import pytest

from shape.agent.verification.segmentation_shared import run, identity
from shape.utils.concurrency import ordered_map


def observed(score=.5, problems=('tiny_mask_fraction high',), count=100, guard=True):
    return dict(result={'mask_path':'native-mask'},score=score,problems=list(problems),guard_ok=guard,
                metrics=dict(cell_count=count,foreground_coverage=.4,density=20))


def rule(value, **kwargs):
    return dict(params={'min_size':value},reason='test measured fragment hypothesis',
                expected_effect={'cell_count':'preserve'},**kwargs)


def coerce(k,v):
    if k!='min_size' or not isinstance(v,int) or v<1:raise ValueError('invalid lever')
    return v


def test_ordered_map_surfaces_later_error_without_waiting_for_first_item():
    release=threading.Event();started=threading.Event()
    def worker(i):
        if i==0:started.set();release.wait(3);return 0
        started.wait(1);raise ValueError('native failure')
    begin=time.monotonic()
    try:
        with pytest.raises(ValueError,match='native failure'):ordered_map(worker,[0,1],2)
        assert time.monotonic()-begin<1
    finally:release.set()


def test_ordered_map_keeps_input_order():
    assert ordered_map(lambda i:(time.sleep(.02*(3-i)),i)[1],[0,1,2],3)==[0,1,2]


def test_cancelled_work_cannot_mask_the_original_failure_in_progress_callback():
    def worker(i):
        if i==0:raise ValueError('original failure')
        return (i,)
    with pytest.raises(ValueError,match='original failure'):
        ordered_map(worker,list(range(100)),4,on_result=lambda value:tuple(value))


def test_frozen_images_are_not_remeasured_and_conditional_rules_share_one_round():
    calls=[]
    def measure(i,p):
        calls.append((i['image_id'],p['min_size']))
        return observed(problems=() if i['image_id']=='clean' or p['min_size']==30 else ('tiny_mask_fraction high',))
    def propose(p):
        assert p['unresolved']==['dirty']
        return {'rules':[rule(30,issues=['tiny_mask_fraction'])]}
    selected,report=run([{'image_id':'clean'},{'image_id':'dirty'}],{'min_size':15},measure,propose,coerce)
    assert calls==[('clean',15),('dirty',15),('dirty',30)]
    assert len(report['rounds'])==1 and report['delivery_summary']['qc_pass']==2


def test_partial_improvement_survives_even_without_passing_threshold():
    def measure(i,p):return observed(score=.5+p['min_size']/100)
    def propose(p):return {'rules':[rule(15+p['round'])]}
    selected,report=run([{'image_id':'a'}],{'min_size':15},measure,propose,coerce)
    assert len(report['rounds'])==8 and selected['a']['params']['min_size']==23
    assert report['delivery_summary']['unresolved']==1 and report['budget_exhausted']


def test_bad_candidate_is_preserved_in_history_but_not_delivered():
    def measure(i,p):return observed(score=.8 if p['min_size']==15 else .1)
    selected,r=run([{'image_id':'a'}],{'min_size':15},measure,
        lambda p:{'rules':[rule(30)]} if p['round']==1 else {'stop':True},coerce)
    assert selected['a']['params']['min_size']==15 and len(r['history']['a'])==2
    assert r['rounds'][0]['reverted']==1


def test_malformed_decisions_consume_shared_budget_without_tool_execution():
    calls=[]
    selected,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:(calls.append(p) or observed()),
        lambda p:{'rules':[rule(0)]},coerce)
    assert len(r['rounds'])==8 and len(calls)==1


def test_stop_with_rules_is_rejected_atomically():
    selected,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:observed(),
        lambda p:{'stop':True,'rules':[rule(30)]},coerce,max_rounds=1)
    assert len(r['history']['a'])==1 and r['rounds'][0]['invalid_decision']


def test_native_failure_is_isolated_and_missing_delivery_is_explicit():
    def measure(i,p):
        if i['image_id']=='broken':raise OSError('unreadable image')
        return observed(problems=())
    best,r=run([{'image_id':'good'},{'image_id':'broken'}],{'min_size':15},measure,
               lambda p:pytest.fail('no unresolved valid image'),coerce,workers=2)
    assert set(best)=={'good'} and r['delivery_summary']['failed']==1 and len(r['errors'])==1


def test_large_count_loss_requires_bound_visual_review():
    def measure(i,p):return observed(score=.5 if p['min_size']==15 else .9,count=100 if p['min_size']==15 else 50)
    def propose(p):
        if p['round']==1:return {'rules':[rule(30)]}
        return dict(stop=True,reviews=[dict(image_id='a',candidate=identity({'min_size':30}),
            baseline=identity({'min_size':15}),confidence='high',observations=['fragments removed','cell boundaries retained'],
            reason='visually better',image_quality_sufficient=True,loss_explanation='baseline fragments were counted as cells')])
    best,r=run([{'image_id':'a'}],{'min_size':15},measure,propose,coerce)
    assert best['a']['params']['min_size']==30 and r['rounds'][0]['reverted']==1


def test_resume_reuses_baseline_and_keeps_original_round_budget():
    snapshots=[]
    _,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:observed(),
        lambda p:{'stop':True},coerce,checkpoint=lambda s:snapshots.append(deepcopy(s)))
    snapshot=snapshots[-1];snapshot['rounds']=[dict(round=i) for i in range(1,9)]
    best,res=run([{'image_id':'a'}],{'min_size':15},lambda *a:pytest.fail('baseline repeated'),
        lambda *a:pytest.fail('budget reset'),coerce,resume=snapshot)
    assert len(res['rounds'])==8 and best['a']['params']['min_size']==15


def test_unknown_readout_is_an_invalid_decision_not_a_controller_crash():
    _,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:observed(),
        lambda p:{'rules':[rule(30,when=[{'metric':'test_label','min':0}])]},coerce,max_rounds=1)
    assert r['rounds'][0]['invalid_decision']


def test_visual_flag_acceptance_cannot_omit_existing_evidence_guards():
    def propose(p):
        return {'reviews':[dict(image_id='a',candidate=identity({'min_size':15}),
            baseline=identity({'min_size':15}),confidence='high',observations=['normal small cells','normal large cells'],
            reason='looks heterogeneous',accept_with_flags=True)]}
    _,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:observed(),propose,coerce,max_rounds=1)
    assert r['rounds'][0]['invalid_decision'] and r['delivery_summary']['accepted_with_flags']==0


def test_expected_effect_is_checked_and_failed_hypothesis_remains_visible():
    def propose(p):
        r=rule(30);r['expected_effect']={'cell_count':'down'}
        return {'rules':[r]} if p['round']==1 else {'stop':True}
    _,r=run([{'image_id':'a'}],{'min_size':15},lambda i,p:observed(),propose,coerce)
    candidate=r['history']['a'][identity({'min_size':30})]
    assert not candidate['hypothesis_supported']
    assert candidate['hypothesis_checks']['cell_count']['after']==100


def test_partial_baseline_checkpoints_resume_only_unfinished_images():
    snapshots=[]
    run([{'image_id':'a'},{'image_id':'b'}],{'min_size':15},lambda i,p:observed(problems=()),
        lambda p:pytest.fail('clean baseline'),coerce,checkpoint=lambda s:snapshots.append(deepcopy(s)))
    partial=next(s for s in snapshots if len(s['selected'])==1 and not s['baseline_complete'])
    calls=[]
    _,r=run([{'image_id':'a'},{'image_id':'b'}],{'min_size':15},
        lambda i,p:(calls.append(i['image_id']) or observed(problems=())),
        lambda p:pytest.fail('clean baseline'),coerce,resume=partial)
    assert calls==['b'] and r['delivery_summary']['qc_pass']==2


def test_failed_native_configuration_is_not_retried_and_is_shown_to_agent():
    calls=[]
    def measure(i,p):
        calls.append(p['min_size'])
        if p['min_size']==30:raise RuntimeError('native candidate error')
        return observed()
    def propose(p):
        if p['round']>1:assert p['errors'][0]['message']=='native candidate error'
        return {'rules':[rule(30)]}
    _,r=run([{'image_id':'a'}],{'min_size':15},measure,propose,coerce)
    assert calls==[15,30] and len(r['errors'])==1


def test_invalid_baseline_cannot_be_replaced_on_proxy_improvement_alone():
    def measure(i,p):
        return observed(problems=('no cells detected',),count=0,guard=False) if p['min_size']==15 else observed(problems=())
    best,r=run([{'image_id':'a'}],{'min_size':15},measure,
        lambda p:{'rules':[rule(30)]} if p['round']==1 else {'stop':True},coerce)
    assert best['a']['params']['min_size']==15 and r['delivery_summary']['qc_pass']==0
    assert len(r['history']['a'])==2
