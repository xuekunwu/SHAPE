"""Sequence configured scientific stages; persist state and stop on failure."""
import os
from pathlib import Path
from .artifacts import write_json
from .stage_runner import invoke

def run_stages(stages, module, prep, run, config, state, *, command_for=None):
    """Shared ordinary/recovery loop. Arm-specific recovery audits happen first."""
    from benchmarks.full_CPJUMP1.harness import pipeline
    for stage in stages:
        state['current_stage'] = stage
        write_json(Path(run) / 'RUN_META.json', state)
        print('STAGE_START', stage, flush=True)
        env = {**os.environ, **pipeline.stage_env(config['arm'], stage, config['reasoning']['protocol'])}
        command = command_for(stage) if command_for else None
        record = invoke(module, stage, prep, run, env=env,
                        timeout=pipeline.remaining(config), command=command)
        state.setdefault('stages', {})[stage] = record
        write_json(Path(run) / 'RUN_META.json', state)
        print('STAGE_COMPLETE', stage, flush=True)
