"""Three-arm (control / shape / codex) SSL embedding of the Xenium lung cancer crops. The inputs are crops, so the only
tool stage is embed: the shared full_CPJUMP1 harness runs the analyzer, SHAPE's representation VRR and the Codex stage
protocol exactly as in the cardiac organoid time course; this module declares the dataset and the arm configuration."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):  # executed as a script: import as the study package from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = 'case studies.tissues.xenium_lung_ssl'

from . import data  # noqa: E402
from .snapshot import pack, verify_source  # noqa: E402

MODULE = 'case studies.tissues.xenium_lung_ssl.e2e'
FC = 'case studies.cardiac organoids.timecourse.e2e'
STAGE = 'embed'
MODEL = 'gpt-5.6-luna'


def _fc():
    """The cardiac organoid runner, whose reasoning protocol, agent settings and environment record this study reuses."""
    import importlib
    return importlib.import_module(FC)


def preflight(args) -> dict:
    """Runs in its own process (`run` starts it with `child`), so the driver never opens the (exclusive) GPU."""
    verify_source(args.expected_source)
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from benchmarks.full_CPJUMP1.harness import pipeline
    import torch
    import shape
    if not Path(shape.__file__).resolve().is_relative_to(data.REPO):
        raise RuntimeError('SHAPE loaded from another checkout')
    assets = asset_environment(args.assets_root)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    config = data.read_json(args.out / 'CONFIG.json')
    if config['arm'] != args.arm or config['source_sha256'] != args.expected_source:
        raise ValueError('preflight arguments do not match CONFIG.json')
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool  # noqa: F401
    model_preflight = {}
    protocol = config['reasoning']['protocol']
    if args.arm in ('shape', 'codex'):
        from benchmarks.full_CPJUMP1.harness.gateway_route import validate
        model_preflight['gateway_route'] = validate(os.environ.get('LLM_BASE_URL', ''))
        if not os.environ.get('LLM_API_KEY'):
            raise RuntimeError('missing LLM_API_KEY')
        os.environ.update(pipeline.stage_env(args.arm, STAGE, protocol))
        if args.arm == 'shape':
            from shape.engine import usage
            before = usage.total_tokens()
            pipeline.engine_for(dict(arm='shape', model=config['model'], reasoning={'protocol': protocol}), STAGE)
            model_preflight['total_tokens'] = usage.total_tokens() - before
            if model_preflight['total_tokens'] <= 0:
                raise RuntimeError('SHAPE model preflight has no observed token usage')
        else:
            _fc().agent_environment(config['model'])
            from benchmarks.full_CPJUMP1.harness.codex_arm import preflight as check
            model_preflight.update(check(args.out / 'codex_preflight', config['model'], effort='high'))
    _fc().environment_record(args.out)
    return dict(status='complete', source_sha256=args.expected_source, arm=args.arm, model=config['model'],
                gpu=torch.cuda.get_device_name(), assets=assets, reasoning_effort='high',
                channels=data.SPEC['channels'], model_preflight=model_preflight)


def run_stage(args) -> None:
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from shape.engine import usage
    config = data.read_json(args.run / 'CONFIG.json')
    verify_source(config['source_sha256'])
    _fc().agent_environment(config['model'])
    selected = set(config['images'])
    rows = [r for r in data.load_prepared(args.prep) if r['image_id'] in selected]
    protocol = config['reasoning']['protocol']
    started = time.time()
    usage.reset()
    asset_environment(Path(config['assets_root']))
    pipeline.check_stage_env(config['arm'], STAGE, protocol)
    try:
        if config['arm'] == 'codex':
            from benchmarks.full_CPJUMP1.harness.codex_arm import run_stage as native
            native(STAGE, args.prep, args.run, config)
        else:
            pipeline.STAGES[STAGE](args.prep, args.run, config, rows)
    finally:
        data.write_json(args.run / f'{STAGE}_USAGE.json', usage.snapshot())
    tokens = usage.snapshot()
    if config['arm'] == 'codex':
        tokens = data.read_json(args.run / 'agent' / STAGE / 'SELECTED.json')
    if config['arm'] != 'control' and tokens.get('total_tokens', 0) <= 0:
        raise RuntimeError('reasoner stage has no observed token usage')
    data.write_json(args.run / f'{STAGE}_USAGE.json', tokens)
    data.write_json(args.run / f'{STAGE}_COMPLETE.json',
                    dict(status='complete', stage=STAGE, seconds=time.time() - started, usage=tokens))


def run(args) -> None:
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline
    verify_source(args.expected_source)
    rows = data.load_prepared(args.prep, verify_pixels=True)
    if args.smoke:
        rows = data.smoke_images(rows)
    data.fresh_output(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir()
    fc = _fc()
    protocol = fc.reasoning_protocol()
    # Each crop is its own development site, so one crop per site enters SSL and the epoch rule counts crops.
    config = dict(dataset='xenium_lung_ssl', arm=args.arm, model=args.model, seed=args.seed,
                  source_sha256=args.expected_source, deadline=None, budget_hours=None, cores=args.cores,
                  assets_root=str(args.assets_root.resolve()), images=[r['image_id'] for r in rows],
                  smoke=bool(args.smoke), train_cells_per_site=1, inference_chunk=2048,
                  reasoning={'protocol': protocol, 'stages': {STAGE: pipeline.stage_reasoning(args.arm, STAGE, protocol)}},
                  ssl_epoch_budget=pipeline.ssl_epoch_budget(rows, 1), channels=data.SPEC['channels'],
                  agent_context_policy=fc.AGENT_CONTEXT_POLICY, agent_context_stages=[STAGE],
                  agent_decision_rounds=fc.AGENT_DECISION_ROUNDS, readout=None,
                  prepare_sha256=data.digest(args.prep / 'PREPARE.json'),
                  technical_smoke_images=len(rows) if args.smoke else None)
    if args.smoke:
        # A smoke is a technical check, not a tuning run. Codex gets a third round so that one rejected request
        # (user decision 2026-10-26, after the v1 smoke) still leaves a run round before the closing finish.
        config['smoke_policy'] = dict(shape_search_images=len(rows), shape_search_rounds=2, codex_rounds=3)
    data.write_json(args.out / 'CONFIG.json', config)
    data.write_json(args.out / 'cells.json', data.cells(args.prep, rows))
    meta = dict(status='preflight', config=config, stages={})
    try:
        child([sys.executable, '-m', MODULE, 'preflight', '--out', str(args.out), '--arm', args.arm,
               '--assets-root', str(args.assets_root), '--expected-source', args.expected_source],
              args.out / 'logs' / 'preflight.log', 1800)
        if not (args.out / 'PREFLIGHT.json').is_file():
            raise RuntimeError('preflight wrote no record')
        meta.update(status='running', current_stage=STAGE)
        data.write_json(args.out / 'RUN_META.json', meta)
        print('STAGE_START', STAGE, flush=True)
        env = dict(os.environ)
        env.update(pipeline.stage_env(args.arm, STAGE, protocol))
        child([sys.executable, '-m', MODULE, 'stage', '--prep', str(args.prep), '--run', str(args.out)],
              args.out / 'logs' / f'{STAGE}.log', None, env=env)
        meta['stages'][STAGE] = data.read_json(args.out / f'{STAGE}_COMPLETE.json')
        print('STAGE_COMPLETE', STAGE, flush=True)
        meta.update(status='complete')
        data.write_json(args.out / 'RUN_META.json', meta)
        marker = 'XENIUM_LUNG_SSL_SMOKE_COMPLETE' if args.smoke else 'XENIUM_LUNG_SSL_COMPLETE'
        (args.out / marker).write_text(args.expected_source + '\n')
    except BaseException as exc:
        meta.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        data.write_json(args.out / 'RUN_META.json', meta)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('prepare')
    c.add_argument('--crops', type=Path, required=True)
    c.add_argument('--cells', type=Path, required=True, help="the section's cells.parquet")
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--seed', type=int, default=None)
    c = sub.add_parser('pack')
    c.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('run')
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--arm', choices=['control', 'shape', 'codex'], required=True)
    c.add_argument('--assets-root', type=Path, required=True)
    c.add_argument('--expected-source', required=True)
    c.add_argument('--model', default=MODEL)
    c.add_argument('--cores', type=int, default=8)
    c.add_argument('--seed', type=int, default=20261025)
    c.add_argument('--smoke', action='store_true')
    c = sub.add_parser('preflight', help='internal: the arm preflight in its own process')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--arm', choices=['control', 'shape', 'codex'], required=True)
    c.add_argument('--assets-root', type=Path, required=True)
    c.add_argument('--expected-source', required=True)
    c = sub.add_parser('stage', help='internal: the embed stage worker')
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--run', type=Path, required=True)
    c = sub.add_parser('native-trial', help='internal: one Codex-requested analyzer call')
    c.add_argument('stage', choices=[STAGE])
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--run', type=Path, required=True)
    c.add_argument('--config', type=Path)
    args = p.parse_args(argv)
    if args.command == 'prepare':
        print(data.prepare(args.crops, args.cells, args.out, args.seed))
    elif args.command == 'pack':
        print(pack(args.out))
    elif args.command == 'run':
        run(args)
    elif args.command == 'preflight':
        data.write_json(args.out / 'PREFLIGHT.json', preflight(args))
    elif args.command == 'stage':
        run_stage(args)
    else:
        config = data.read_json(args.config or args.run / 'CONFIG.json')
        verify_source(config['source_sha256'])
        _fc().agent_environment(config['model'])
        from benchmarks.full_CPJUMP1.harness.cli import native_trial
        native_trial(args)


if __name__ == '__main__':
    main()
