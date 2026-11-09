"""Small code, dataset-budget and same-model Gateway probe, on the allocated node."""
import argparse
import json
import os
from pathlib import Path

import shape
import torch
from pydantic import BaseModel

from benchmarks.representation_RxRx1.runners import control, epoch_budget
from benchmarks.representation_RxRx1.runners.config import DEFAULT, BenchmarkConfig
from benchmarks.representation_RxRx1.runners.prepare import load
from shape.agent.verification.vision_engine import build_vision_engine


class Probe(BaseModel):
    status: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("shape", "react", "llm"), required=True)
    args = parser.parse_args()
    if Path(epoch_budget.__file__).resolve().parents[3] != control.REPO:
        raise SystemExit("INVALID_REMOTE_SYNC: wrong epoch-budget module")
    if control.REPO not in Path(shape.__file__).resolve().parents:
        raise SystemExit("INVALID_REMOTE_SYNC: wrong SHAPE import")
    if control.git_commit() != os.environ["RXRX_EXPECTED_COMMIT"]:
        raise SystemExit("INVALID_REMOTE_SYNC: wrong commit")
    if not torch.cuda.is_available():
        raise SystemExit("NOT_READY: no CUDA GPU")
    if args.stage != "llm":
        # the launcher's prep: CELL_TYPE (default HEPG2) and RXRX_PREP_SET (default v3), as the VM driver exports them
        root = Path(os.environ.get("SHAPE_RESULTS_ROOT", "/scratch/groups/your-group/your-user/shape_benchmark/results"))
        config = BenchmarkConfig(cell_type=os.environ.get("CELL_TYPE") or "HEPG2",
                                 prep_set=os.environ.get("RXRX_PREP_SET") or "v3")
        root = root / "representation_RxRx1" / "cache" / config.root().name
        summary, cells = load(root)
        budget = epoch_budget.shared_budget(root, summary, cells)
        verifier = epoch_budget.FixedEpochVerifier(budget)
        assert "max_epochs" not in verifier.levers
        assert verifier.trial_params({"max_epochs": None})["max_epochs"] == budget["max_epochs"]
        print("TRAINING_BUDGET=" + json.dumps(budget), flush=True)
    if args.stage == "react":                         # the shared ReAct runner and its imports (A14)
        from benchmarks._shared import react_arm as shared

        if Path(shared.__file__).resolve().parents[2] != control.REPO:
            raise SystemExit("INVALID_REMOTE_SYNC: wrong shared ReAct runner")
    engine = build_vision_engine(DEFAULT.model)
    if engine is None:
        raise SystemExit("NOT_READY: no reasoner engine")
    reply = engine.generate(["Reply with status ready."], response_format=Probe)
    if not isinstance(reply, dict) or not (reply.get("usage") or {}).get("total_tokens", 0) > 0:
        raise SystemExit("NOT_READY: Gateway preflight has no positive usage")
    content = reply.get("content")
    parsed = (Probe.model_validate_json(content) if isinstance(content, str)
              else Probe.model_validate(content) if isinstance(content, dict) else content)
    if not isinstance(parsed, Probe) or parsed.status != "ready":
        raise SystemExit("NOT_READY: Gateway structured preflight failed")
    print(f"PREFLIGHT_OK model={DEFAULT.model} stage={args.stage} commit={control.git_commit()}", flush=True)


if __name__ == "__main__":
    main()
