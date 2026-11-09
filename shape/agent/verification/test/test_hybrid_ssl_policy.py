"""HYBRID SSL policy through real VRR, native files, and unchanged delivery guards."""
import pytest

from shape.agent.verification.representation_policy import (
    POLICY, continuation_evidence, execution_audit, expert_start)
from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier
from shape.agent.verification.step_verifier import StepVerifier
from shape.agent.verification.verifiers.representation import RepresentationTask, frozen_params
from .test_representation_agentic import card, proxy, crops, Engine, StubTool, mv


def run_policy(tmp_path, engine, quality):
    meta = crops()
    native = StubTool(tmp_path, meta, quality)
    verifier = FixedEpochVerifier(dict(max_epochs=8, n_train=48), expert_config=expert_start(),
                                  search_evidence_policy=POLICY)
    verifier.llm_engine = engine
    task = RepresentationTask(cores=1, gpus=1, deadline=None, split_key='split', brief='validate hybrid SSL VRR')
    audits = []
    def tool(params):
        execution_audit(params)
        result = native(params)
        audits.append(execution_audit(params, result))
        return result
    initial = frozen_params(task)
    result = StepVerifier(max_iters=6).resolve_stage(verifier, verifier.make_adapter({'cell_metadata': meta}),
        tool, initial, tool(initial), upstream_ctx={'representation_task': task})
    return verifier, result, native, audits


@pytest.mark.parametrize('strategy', ['legacy', 'four_stage'])
def test_complete_inheritance_staged_unfreezing_and_uncertain_stop(card, proxy, tmp_path, monkeypatch, strategy):
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_REPRESENTATION', strategy)
    first, second = expert_start()
    engine = Engine([[mv(**first)], [mv('agent_1', **second)]], final=['agent_2'])
    if strategy == 'four_stage':
        from .test_four_stage import CoveringEngine
        engine = CoveringEngine(engine)
    def quality(params):
        value = .7 + (.00001 if params.get('init_checkpoint') else 0)
        return value, value
    verifier, result, tool, audits = run_policy(tmp_path, engine, quality)
    assert len(tool.calls) == 3  # frozen baseline + two candidates; no forced continuation
    one, two = tool.calls[1:]
    for key in ('view_pairing', 'contrastive_mask', 'batch_sampling', 'intensity_norm', 'augmentation', 'crop_scale_min'):
        assert two[key] == one[key] == first[key]
    assert one['trainable_pct'] == 50 and two['trainable_pct'] == 100
    assert two['learning_rate'] == 1e-5 and two['init_checkpoint']
    assert all(p['max_epochs'] == 8 for p in tool.calls[1:])
    assert verifier.forced_continuations == [] and verifier.stop_refusals == []
    assert verifier.stop_evidence[-1]['reason'] == 'within_2se'
    assert result.details['search_evidence_policy'] == POLICY
    assert result.details['decision']['use'] == 'agent_2'
    assert audits[-1]['requested_effective_kwargs']['intensity_norm'] == 'per_plate'
    assert audits[-1]['parent_checkpoint'] == two['init_checkpoint']


def test_strong_gain_still_requires_a_justified_continuation(card, proxy, tmp_path, monkeypatch):
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_REPRESENTATION', 'legacy')
    first, second = expert_start()
    engine = Engine([[mv(**first)], [mv('agent_1', **second)]], final=['agent_2'])
    def quality(params):
        value = .7 + (.1 if params.get('init_checkpoint') else 0)
        return value, value
    verifier, result, tool, _ = run_policy(tmp_path, engine, quality)
    assert len(tool.calls) == 4  # one continuation; its unchanged proxy then permits stop
    assert len(verifier.forced_continuations) == 1
    assert verifier.stop_evidence[0]['reason'] == 'beyond_2se'
    assert result.details['decision']['use'] == 'agent_2'


def test_policy_does_not_relax_the_frozen_baseline_delivery_guard(card, proxy, tmp_path, monkeypatch):
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_REPRESENTATION', 'legacy')
    first = expert_start()[0]
    verifier, result, tool, _ = run_policy(tmp_path, Engine([[mv(**first)]], final=['agent_1']),
                                          lambda params: (.499, .499))
    assert result.details['decision']['use'] == 'frozen'
    assert len(tool.calls) == 2 and not verifier.forced_continuations


@pytest.mark.parametrize('paired', [dict(delta=.1,se=None,n_wells=2),dict(delta=.1,se=.01,n_wells=1),
    dict(delta=float('nan'),se=.01,n_wells=3),dict(delta=.001,se=.01,n_wells=3),
    dict(delta=.02,se=.01,n_wells=3)])
def test_missing_uncertain_and_boundary_evidence_does_not_force_training(paired):
    assert continuation_evidence(paired)['continue_required'] is False


def test_incomplete_resume_recipe_fails_before_native_training():
    with pytest.raises(ValueError, match='complete effective parameters'):
        execution_audit(dict(init_checkpoint='parent.pt'))
    assert execution_audit(dict(adaptation_depth='zero_shot'))['execution'] == 'frozen'


def test_native_pipeline_wires_policy_audits_and_excludes_heldout(card, proxy, tmp_path, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.data import digest, write_json, read_json, technical
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool
    from .test_four_stage import CoveringEngine
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY_REPRESENTATION', 'four_stage')
    image = tmp_path / 'crop.bin'; image.write_bytes(b'input-integrity-fixture')
    cells = []
    for row in crops(n_plates=1):
        cells.append(dict(cell_id=row['cell_id'], image_id=row['site'], plate=row['plate'],
            well=row['well'], area=100, crop_path=str(image), sha256=digest(image),
            usage='development_val' if row['split']=='val' else 'development_train'))
    cells.append(dict(cell_id='held', image_id='held_site', plate='P0', well='held', area=100,
                      crop_path=str(image), sha256=digest(image), usage='evaluation_only'))
    write_json(tmp_path/'cells.json', cells)
    images = list({c['image_id']:dict(image_id=c['image_id'], usage=c['usage']) for c in cells}.values())
    first, second = expert_start()
    engine = CoveringEngine(Engine([[mv(**first)], [mv('agent_1', **second)]], final=['agent_2']))
    monkeypatch.setattr(pipeline, 'engine_for', lambda *a: engine)
    native = StubTool(tmp_path, [], lambda p: (.7 + bool(p.get('init_checkpoint')) * .00001,) * 2)
    def execute(self, **kwargs):
        assert all(m['cell_id']!='held' for m in kwargs['cell_metadata'])
        assert {m['split'] for m in kwargs['cell_metadata']} == {'train', 'val'}
        native.meta = kwargs['cell_metadata']
        return native({k:v for k,v in kwargs.items() if k not in ('cell_crops','cell_metadata','query_cache_dir','seed')})
    monkeypatch.setattr(Cell_State_Analyzer_Tool, 'execute', execute)
    config=dict(arm='shape', cores=1, deadline=None, ssl_search_deadline=None, seed=20261015,
                train_cells_per_site=128, fit_only=True, ssl_search_policy=POLICY)
    pipeline.representation(tmp_path, tmp_path, config, images)
    selection=read_json(tmp_path/'SSL_SELECTION.json')
    assert selection['mode']=='frozen_checkpoint'
    audits=[read_json(p) for p in sorted((tmp_path/'ssl').glob('*.audit.json'))]
    assert len(audits)==3 and audits[0]['execution']=='frozen'
    assert audits[1]['requested_effective_kwargs']['trainable_pct']==50
    assert audits[2]['requested_effective_kwargs']['trainable_pct']==100
    assert audits[2]['requested_effective_kwargs']['intensity_norm']=='per_plate'
    assert not (tmp_path/'embedding.npy').exists()
