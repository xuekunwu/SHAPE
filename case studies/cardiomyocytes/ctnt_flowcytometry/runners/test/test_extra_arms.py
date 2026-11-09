import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

e2e = importlib.import_module('case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.experiment_runner')


def test_new_agents_reject_wrong_model_before_loading_gpu(tmp_path):
    for arm, model in [('claude','gpt-5.6-luna'), ('biomni','claude-sonnet-5')]:
        with pytest.raises(ValueError, match='requires'):
            e2e.preflight(SimpleNamespace(arm=arm, model=model, reasoning_profile='all-high'))
