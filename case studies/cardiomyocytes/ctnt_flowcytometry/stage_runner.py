"""One subprocess and completion-proof boundary for native stages and recovery."""
import os
from pathlib import Path
import sys

from .artifacts import read_json
from .artifacts import write_json


def invoke(module, stage, prep, run, *, env=None, timeout=None, log=None,
           style='stage', command=None):
    """Run one worker and require its completion record; never skip or retry it."""
    from benchmarks.full_CPJUMP1.harness.cli import child
    run = Path(run)
    if (run / (stage + '_COMPLETE.json')).exists():
        raise FileExistsError('stage already has completion proof; audit reuse before execution: ' + stage)
    if command is None:
        suffix = ['stage', '--name', stage] if style == 'named' else ['stage', stage]
        command = [sys.executable, '-m', module, *suffix, '--prep', str(prep), '--run', str(run)]
    child(command, Path(log) if log else run / 'logs' / (stage + '.log'), timeout,
          env=dict(os.environ) if env is None else env)
    record = read_json(run / (stage + '_COMPLETE.json'))
    if record.get('status') != 'complete':
        raise RuntimeError('worker exited without a valid completion record: ' + stage)
    return record
