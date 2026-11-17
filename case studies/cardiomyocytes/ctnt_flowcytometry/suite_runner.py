"""Arrange arms and predecessor waits; each arm uses the same experiment runner."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .artifacts import read_json
from .artifacts import write_json


def wait_for_predecessor(path, *, require_suite=False):
    """Preserve failure checks and require proof, not merely an existing output."""
    while True:
        meta = read_json(path / 'RUN_META.json')
        suite = read_json(path.parent / 'SUITE_META.json') if require_suite else {}
        if meta.get('status') == 'failed' or suite.get('status') == 'failed':
            raise RuntimeError('predecessor failed; dependent run not started')
        marker = path / 'CTNT_FLOW_COMPLETE'
        if marker.exists() and meta.get('status') == 'complete' and (not require_suite or suite.get('status') == 'complete'):
            config = read_json(path / 'CONFIG.json')
            if marker.read_text().strip() != config['source_sha256']:
                raise ValueError('predecessor completion source mismatch')
            return
        time.sleep(30)


def configure_private_environment(path, *, expected_base=None):
    """Read an existing host-private file; never transfer, log or persist its values."""
    from dotenv import dotenv_values
    values = dotenv_values(path)
    if not values.get('LLM_API_KEY'):
        raise ValueError('private model configuration is missing LLM_API_KEY')
    if expected_base and values.get('LLM_BASE_URL', '').rstrip('/') != expected_base:
        raise ValueError('private model route differs from the requested protocol')
    os.environ.update({key: value for key, value in values.items() if value is not None})


def main(argv=None):
    from . import experiment_runner as experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--prep', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--expected-source', required=True)
    parser.add_argument('--arms', nargs='+', choices=('shape', 'codex', 'control'), default=['shape', 'codex', 'control'])
    parser.add_argument('--mode', choices=('smoke', 'pilot', 'full'), required=True)
    parser.add_argument('--pilot-manifest', type=Path)
    parser.add_argument('--model', default=experiment.MODEL)
    parser.add_argument('--cores', type=int, default=8)
    parser.add_argument('--budget-hours', type=float, default=12)
    parser.add_argument('--no-time-limit', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms):parser.error('arms must be unique')
    if (args.mode == 'pilot') != bool(args.pilot_manifest):parser.error('pilot mode requires --pilot-manifest; other modes forbid it')
    if not 1 <= args.cores <= (os.cpu_count() or 1):parser.error('cores must fit the host')
    experiment.verify_source(args.expected_source)
    args.suite = args.suite.resolve()
    args.suite.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', arms=args.arms, mode=args.mode, source_sha256=args.expected_source, completed=[])
    marker = 'CTNT_FLOW_SMOKE_COMPLETE' if args.mode == 'smoke' else 'CTNT_FLOW_PILOT_COMPLETE' if args.mode == 'pilot' else 'CTNT_FLOW_COMPLETE'
    try:
        for arm in args.arms:
            state['current_arm'] = arm
            write_json(args.suite / 'SUITE_META.json', state)
            command = [sys.executable, '-m', experiment.MODULE, 'run', '--arm', arm,
                       '--prep', str(args.prep.resolve()), '--assets-root', str(args.assets_root.resolve()),
                       '--expected-source', args.expected_source, '--out', str(args.suite / arm),
                       '--model', args.model, '--cores', str(args.cores), '--budget-hours', str(args.budget_hours)]
            if args.mode == 'smoke':command.append('--smoke')
            if args.mode == 'pilot':command += ['--pilot-manifest', str(args.pilot_manifest.resolve())]
            if args.no_time_limit:command.append('--no-time-limit')
            with (args.suite / (arm + '.log')).open('x') as log:
                subprocess.run(command, cwd=experiment.data.REPO, env=dict(os.environ),
                               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            if (args.suite / arm / marker).read_text().strip() != args.expected_source:
                raise RuntimeError('arm completion source mismatch: ' + arm)
            state['completed'].append(arm)
        state.update(status='complete', current_arm=None)
        (args.suite / 'CTNT_SUITE_COMPLETE').write_text(args.expected_source + '\n')
    except BaseException as exc:
        state.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        write_json(args.suite / 'SUITE_META.json', state)
    print('CTNT_SUITE_COMPLETE', args.suite)


if __name__ == '__main__':main()
