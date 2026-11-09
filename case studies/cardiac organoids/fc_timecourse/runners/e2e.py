"""Three-arm (control / shape / codex) end-to-end stages on the FC time course; private GFP and labels are read
only by the trusted predict and score stages. Mirrors the cTnT staining runner: the shared full_CPJUMP1 harness
runs the native tools, SHAPE's verifiers and the Codex stage protocol; this module declares the dataset, the
arm configuration and the two trusted stages."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):  # executed as a script: import as the study package from the repository root
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = 'case studies.cardiac organoids.fc_timecourse.runners'

from . import data  # noqa: E402
from .snapshot import pack, verify_source  # noqa: E402

MODULE = 'case studies.cardiac organoids.fc_timecourse.runners.e2e'
STAGES = ('segment', 'crop', 'embed', 'predict', 'score')
TOOL_STAGES = ('segment', 'crop', 'embed')
AGENT_CONTEXT_POLICY = 'tool_api_v1'          # the staining pilot's neutral tool-API context
AGENT_DECISION_ROUNDS = 8
MODEL = 'gpt-5.6-luna'


def reasoning_protocol() -> dict:
    """The cardiomyocyte studies' `all-high` protocol: every reasoning stage of both reasoning arms at high effort."""
    return {'arms': ['shape', 'codex'], 'table': {
        stage: {'effort': 'high', 'codex_effort': 'high', 'max_tokens': 16000, 'timeout_seconds': 900}
        for stage in ('segment', 'crop', 'embed', 'batch')}}


def agent_environment(model: str) -> None:
    """The Codex backend reads its model and gateway route from these variables when it is imported."""
    os.environ['CTNT_CODEX_MODEL'] = model
    os.environ['CTNT_CODEX_BASE_URL'] = os.environ.get('LLM_BASE_URL', '').rstrip('/')


def organoid_environment(config: dict) -> None:
    path = Path(config['organoid_model'])
    if data.digest(path) != config['organoid_model_sha256']:
        raise RuntimeError('organoid checkpoint changed')
    os.environ['SHAPE_ORGANOID_MODEL_PATH'] = str(path)


def environment_record(out: Path) -> None:
    """The runtime facts (`shape.utils.runtime_environment.capture`); when the strict capture refuses the
    environment (the shared cluster env carries two dist-info records of one package), record every installed
    distribution as found, with the refusal, instead of stopping the run."""
    from shape.utils.runtime_environment import capture
    lock = data.REPO / 'environments/benchmark/linux-py312-cu128/requirements.lock.txt'
    try:
        record = capture(lock if lock.is_file() else None)
    except RuntimeError as refusal:
        import platform
        from importlib import metadata
        import torch
        installed = sorted((d.metadata['Name'], d.version) for d in metadata.distributions() if d.metadata['Name'])
        record = dict(capture_status=f'strict capture refused: {refusal}', python=platform.python_version(),
                      platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                      installed_distributions=[dict(name=n, version=v) for n, v in installed])
    data.write_json(out / 'ENVIRONMENT.json', record)


def preflight(args) -> dict:
    """Runs in its own process (`run` starts it with `child`): it opens a CUDA context, which on an exclusive GPU
    the driver process must never do, or the stage workers cannot open the device."""
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
    organoid_environment(config)
    from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool  # noqa: F401
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool  # noqa: F401
    tool = Organoid_Segmenter_Tool()
    if tool.device.type != 'cuda':
        raise RuntimeError('the organoid segmenter did not get the GPU')
    model_preflight = {}
    protocol = config['reasoning']['protocol']
    if args.arm in ('shape', 'codex'):
        from benchmarks.full_CPJUMP1.harness.gateway_route import validate
        model_preflight['gateway_route'] = validate(os.environ.get('LLM_BASE_URL', ''))
        if not os.environ.get('LLM_API_KEY'):
            raise RuntimeError('missing LLM_API_KEY')
        os.environ.update(pipeline.stage_env(args.arm, 'segment', protocol))
        if args.arm == 'shape':
            from shape.engine import usage
            before = usage.total_tokens()
            pipeline.engine_for(dict(arm='shape', model=config['model'], reasoning={'protocol': protocol}), 'segment')
            model_preflight['total_tokens'] = usage.total_tokens() - before
            if model_preflight['total_tokens'] <= 0:
                raise RuntimeError('SHAPE model preflight has no observed token usage')
        else:
            agent_environment(config['model'])
            from benchmarks.full_CPJUMP1.harness.codex_arm import preflight as check
            # per run: the Codex setup needs an empty configuration directory
            model_preflight.update(check(args.out / 'codex_preflight', config['model'], effort='high'))
    environment_record(args.out)
    return dict(status='complete', source_sha256=args.expected_source, arm=args.arm, model=config['model'],
                gpu=torch.cuda.get_device_name(), assets=assets, reasoning_effort='high',
                organoid_model_sha256=config['organoid_model_sha256'],
                variant=config.get('variant', 'phase_only'), stage_channels=config.get('stage_channels'),
                prediction_channels=config.get('prediction_channels', data.SPEC['prediction_channels']),
                evaluation_channels=data.SPEC['evaluation_channels'], model_preflight=model_preflight)


def run_stage(args) -> None:
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from shape.engine import usage
    config = data.read_json(args.run / 'CONFIG.json')
    verify_source(config['source_sha256'])
    agent_environment(config['model'])
    selected = set(config['images'])
    rows = [r for r in data.load_prepared(args.prep) if r['image_id'] in selected]
    protocol = config['reasoning']['protocol']
    started = time.time()
    usage.reset()
    stage = args.stage
    tokens = None
    if stage in TOOL_STAGES:
        asset_environment(Path(config['assets_root']))
        organoid_environment(config)
        pipeline.check_stage_env(config['arm'], stage, protocol)
        try:
            if config['arm'] == 'codex':
                from benchmarks.full_CPJUMP1.harness.codex_arm import run_stage as native
                native(stage, args.prep, args.run, config)
            else:
                pipeline.STAGES[stage](args.prep, args.run, config, rows)
        finally:
            data.write_json(args.run / f'{stage}_USAGE.json', usage.snapshot())
        tokens = usage.snapshot()
        if config['arm'] == 'codex':
            tokens = data.read_json(args.run / 'agent' / stage / 'SELECTED.json')
        if config['arm'] != 'control' and tokens.get('total_tokens', 0) <= 0:
            raise RuntimeError('reasoner stage has no observed token usage')
        data.write_json(args.run / f'{stage}_USAGE.json', tokens)
    elif stage == 'predict':
        from .score import predict
        predict(args.prep, args.run)
    elif stage == 'score':
        from .score import score
        score(args.prep, args.run)
    else:
        raise ValueError(stage)
    data.write_json(args.run / f'{stage}_COMPLETE.json',
                    dict(status='complete', stage=stage, seconds=time.time() - started, usage=tokens))


def run(args) -> None:
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline
    verify_source(args.expected_source)
    rows = data.load_prepared(args.prep, verify_pixels=True)
    private = data.private_images(args.prep)
    if args.smoke:
        rows = data.smoke_images(rows, private)
    data.fresh_output(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'logs').mkdir()
    protocol = reasoning_protocol()
    checkpoint = Path(args.organoid_model).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    prepared = data.variant(args.prep)
    split_protocol = data.protocol_of(args.prep)
    cv = split_protocol == data.CV_PROTOCOL
    config = dict(dataset='co_fc_timecourse', protocol=split_protocol, variant=prepared['variant'], ssl_channels=prepared['ssl_channels'],
                  stage_channels=prepared['stage_channels'], arm=args.arm, model=args.model, seed=args.seed,
                  source_sha256=args.expected_source, deadline=None, budget_hours=None, cores=args.cores,
                  assets_root=str(args.assets_root.resolve()), organoid_model=str(checkpoint),
                  organoid_model_sha256=data.digest(checkpoint), images=[r['image_id'] for r in rows],
                  smoke=bool(args.smoke), train_cells_per_site=128, inference_chunk=2048,
                  reasoning={'protocol': protocol, 'stages': {s: pipeline.stage_reasoning(args.arm, s, protocol)
                                                              for s in ('segment', 'crop', 'embed')}},
                  shape_policy={'crop_vrr_workers': args.cores},
                  ssl_epoch_budget=pipeline.ssl_epoch_budget(rows, 128),
                  prediction_channels=sorted(set(prepared['stage_channels']['segment']) | set(prepared['ssl_channels'])),
                  evaluation_channels=data.SPEC['evaluation_channels'],
                  # the cross-validation protocol was run with the shape arm's segmentation as one organoid cohort (the crop of a
                  # full SHAPE run is the harness's own crop cohort); the original protocol segments every image by itself
                  segmentation_regime='cohort' if cv and args.arm == 'shape' else 'per-image',
                  agent_context_policy=AGENT_CONTEXT_POLICY, agent_context_stages=list(TOOL_STAGES),
                  agent_decision_rounds=AGENT_DECISION_ROUNDS,
                  readout=('cross-validated ' if cv else '') + 'day classifier (primary), day ridge, condition logistic probe, GFP ridge probe',
                  prepare_sha256=data.digest(args.prep / 'PREPARE.json'),
                  technical_smoke_images=len(rows) if args.smoke else None)
    if args.smoke:
        # Every selected image is searched by both adaptive arms; a smoke is a technical check, not a tuning run.
        config['smoke_policy'] = dict(shape_search_images=len(rows), shape_search_rounds=2, codex_rounds=2)
    data.write_json(args.out / 'CONFIG.json', config)
    meta = dict(status='preflight', config=config, stages={})
    try:
        env = dict(os.environ)
        env['SHAPE_ORGANOID_MODEL_PATH'] = str(checkpoint)
        # The preflight opens CUDA; it runs in its own process so the driver never holds the (exclusive) GPU.
        child([sys.executable, '-m', MODULE, 'preflight', '--out', str(args.out), '--arm', args.arm,
               '--assets-root', str(args.assets_root), '--expected-source', args.expected_source],
              args.out / 'logs' / 'preflight.log', 1800, env=env)
        if not (args.out / 'PREFLIGHT.json').is_file():
            raise RuntimeError('preflight wrote no record')
        for stage in STAGES:
            meta.update(status='running', current_stage=stage)
            data.write_json(args.out / 'RUN_META.json', meta)
            print('STAGE_START', stage, flush=True)
            env = dict(os.environ)
            env['SHAPE_ORGANOID_MODEL_PATH'] = str(checkpoint)
            if stage in TOOL_STAGES:
                env.update(pipeline.stage_env(args.arm, stage, protocol))
            child([sys.executable, '-m', MODULE, 'stage', stage, '--prep', str(args.prep), '--run', str(args.out)],
                  args.out / 'logs' / f'{stage}.log', None, env=env)
            meta['stages'][stage] = data.read_json(args.out / f'{stage}_COMPLETE.json')
            print('STAGE_COMPLETE', stage, flush=True)
        meta.update(status='complete')
        data.write_json(args.out / 'RUN_META.json', meta)
        marker = 'CO_FC_TIMECOURSE_SMOKE_COMPLETE' if args.smoke else 'CO_FC_TIMECOURSE_COMPLETE'
        (args.out / marker).write_text(args.expected_source + '\n')
    except BaseException as exc:
        meta.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        data.write_json(args.out / 'RUN_META.json', meta)
        raise


def rescore(args) -> None:
    """Re-run the trusted predict and score stages of a finished run with the current code (the read-out definition
    changed after the tool stages ran). The tool stages, their source identity and their usage records are untouched;
    RESCORE.json binds the new tables to both the run's source sha and the rescoring code's sha."""
    from .snapshot import source_manifest
    config = data.read_json(args.run / 'CONFIG.json')
    if not any((args.run / m).is_file() for m in ('CO_FC_TIMECOURSE_COMPLETE', 'CO_FC_TIMECOURSE_SMOKE_COMPLETE')):
        raise RuntimeError('rescore needs a run that completed every stage')
    from .score import predict, score
    started = time.time()
    predict(args.prep, args.run)
    score(args.prep, args.run)
    data.write_json(args.run / 'RESCORE.json', dict(
        status='complete', run_source_sha256=config['source_sha256'], rescore_source_sha256=source_manifest()['source_sha256'],
        seconds=time.time() - started, reason=args.reason,
        rewritten=['predictions.json', 'CLASSIFIER.json', 'classifier.npz', 'PREDICTIONS_COMPLETE.json', 'metrics.json',
                   'per_image.csv', 'per_crop.csv', 'per_object_gfp.csv']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('prepare')
    c.add_argument('--source', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--seed', type=int, default=None)
    c.add_argument('--protocol', choices=list(data.PROTOCOL_USAGES), default=data.PROTOCOL,
                   help='split protocol: the well-stratified held-out protocol (default) or the leave-one-plate-column-out cross-validation')
    c.add_argument('--ssl-channels', default='PhaseContrast',
                   help='comma-separated SSL input channels: PhaseContrast (default) or PhaseContrast,GFP (two-channel SSL variant)')
    c = sub.add_parser('pack')
    c.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('run')
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--arm', choices=['control', 'shape', 'codex'], required=True)
    c.add_argument('--assets-root', type=Path, required=True)
    c.add_argument('--organoid-model', type=Path, required=True)
    c.add_argument('--expected-source', required=True)
    c.add_argument('--model', default=MODEL)
    c.add_argument('--cores', type=int, default=8)
    c.add_argument('--seed', type=int, default=20261024)
    c.add_argument('--smoke', action='store_true')
    c = sub.add_parser('preflight', help='internal: the arm preflight in its own process')
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--arm', choices=['control', 'shape', 'codex'], required=True)
    c.add_argument('--assets-root', type=Path, required=True)
    c.add_argument('--expected-source', required=True)
    c = sub.add_parser('stage')
    c.add_argument('stage', choices=STAGES)
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--run', type=Path, required=True)
    c = sub.add_parser('rescore', help='re-run predict and score of a finished run with the current code')
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--run', type=Path, required=True)
    c.add_argument('--reason', required=True)
    c = sub.add_parser('native-trial')
    c.add_argument('stage', choices=['segment', 'crop', 'embed', 'batch'])
    c.add_argument('--prep', type=Path, required=True)
    c.add_argument('--run', type=Path, required=True)
    c.add_argument('--config', type=Path)
    args = p.parse_args(argv)
    if args.command == 'prepare':
        print(data.prepare(args.source, args.out, args.seed, tuple(c.strip() for c in args.ssl_channels.split(',')), args.protocol))
    elif args.command == 'pack':
        print(pack(args.out))
    elif args.command == 'run':
        run(args)
    elif args.command == 'preflight':
        data.write_json(args.out / 'PREFLIGHT.json', preflight(args))
    elif args.command == 'stage':
        run_stage(args)
    elif args.command == 'rescore':
        rescore(args)
    else:
        config = data.read_json(args.config or args.run / 'CONFIG.json')
        verify_source(config['source_sha256'])
        agent_environment(config['model'])
        organoid_environment(config)
        from benchmarks.full_CPJUMP1.harness.cli import native_trial
        native_trial(args)


if __name__ == '__main__':
    main()
