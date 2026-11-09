"""
Canonical configuration for batch_correction_JUMP.

Benchmarks batch correction on JUMP Cell Painting, the five scenarios of Arevalo et al. (2024):
raw plus Harmony, Scanorama, GPT-only, ReAct and SHAPE, scored on held-out compounds. The SHAPE arm
uses BatchCorrectionVerifier through StepVerifier. This benchmark BYPASSES solve(): it
calls the tool and StepVerifier.resolve_stage directly (the planner's managed batch stage has no
calibration view to verify separability with).

Data: the authors' inputs/ layout (profiles + metadata) under the `--base` given to `prepare`.
GPU: not required (CPU; large scenarios need a large-memory node).
"""

from dataclasses import dataclass
from typing import Optional


SCENARIO_DESIGN = {   # the five scenarios in words (the reasoner's task brief)
    "scenario_1": "one laboratory (source), several experimental batches of plates; the batch to correct is the "
                  "experimental batch",
    "scenario_2": "three laboratories with the same type of microscope, the shared TARGET2 plates; the batch to "
                  "correct is the laboratory (source)",
    "scenario_3": "three laboratories, the shared TARGET2 plates plus each laboratory's own COMPOUND plates (large, "
                  "few replicates per compound); the batch to correct is the source",
    "scenario_4": "five laboratories with different microscopes, the shared TARGET2 plates; the batch to correct is "
                  "the source",
    "scenario_5": "five laboratories with different microscopes, TARGET2 plus COMPOUND plates; the batch to "
                  "correct is the source",
}


@dataclass
class BenchmarkConfig:
    split_seed: int = 0                          # compounds ordered by sha256(f"{seed}:{id}"); half val, half test
    max_rounds: int = 8                          # rounds at most (each: up to cores // 8 concurrent calls)
    noise_band: float = 0.002                    # the task's stated noise (plate-level spread of the kNN metrics)
    model: str = "gpt-5.6-luna"                  # the reasoner (Stanford AI Gateway, LLM_BASE_URL / LLM_API_KEY)
    cores: Optional[int] = None                  # cores for the tool calls (None = the cores this process may use)


DEFAULT = BenchmarkConfig()
