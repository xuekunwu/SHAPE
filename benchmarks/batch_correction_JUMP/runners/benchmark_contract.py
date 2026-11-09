"""Public contract for the JUMP batch-correction comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


PILOT_SCENARIO = "scenario_2"                                  # the default scenario
SCENARIOS = tuple(f"scenario_{i}" for i in range(1, 6))        # the five scenarios of Arevalo et al.
LUNA_MODEL = "gpt-5.6-luna"
# Output-token limit of every Luna call (reasoning tokens included): the engine's generate() default,
# which SHAPE's verifier uses, so every arm gets the same per-call budget.
LUNA_MAX_TOKENS = 4000


@dataclass(frozen=True)
class BenchmarkArm:
    arm_id: str
    display_name: str
    family: str
    uses_luna: bool
    domain_tools: tuple[str, ...]


RAW_BASELINE = BenchmarkArm("raw", "Raw", "baseline", False, ())

OFFICIAL_ARMS = (
    BenchmarkArm("harmony", "Harmony", "domain_tool", False, ("harmony",)),
    BenchmarkArm("scanorama", "Scanorama", "domain_tool", False, ("scanorama",)),
    BenchmarkArm("gpt_luna", "GPT-5.6 luna", "llm_only", True, ()),
    BenchmarkArm(
        "react_luna_harmony",
        "ReAct (GPT-5.6 luna + Harmony)",
        "react",
        True,
        ("harmony",),
    ),
    BenchmarkArm(
        "shape_luna_harmony",
        "SHAPE (GPT-5.6 luna, Harmony)",
        "shape",
        True,
        ("harmony",),
    ),
)

REPORT_METRICS = (
    "batch_correction",
    "biometrics",
    "runtime_per_image",
    "tool_calls_per_image",
    "ap50_gain_per_tool_call",
    "token_per_image",
)


def arm_ids(*, include_raw: bool = False) -> tuple[str, ...]:
    arms = tuple(arm.arm_id for arm in OFFICIAL_ARMS)
    return (RAW_BASELINE.arm_id, *arms) if include_raw else arms


def metric_ids() -> tuple[str, ...]:
    return REPORT_METRICS


def validate_arm_roster(arms: Sequence[str] | None = None) -> tuple[str, ...]:
    expected = arm_ids(include_raw=True)
    observed = tuple(arms or expected)
    if observed != expected:
        raise ValueError(f"benchmark roster must be {expected}, got {observed}")
    return observed


def validate_report_row(row: Mapping[str, object]) -> None:
    missing = [name for name in ("scenario", "arm_id", *REPORT_METRICS) if name not in row]
    if missing:
        raise ValueError(f"benchmark report row is missing columns: {missing}")
    if row["scenario"] not in SCENARIOS:
        raise ValueError(f"benchmark scenario must be one of {SCENARIOS}, got {row['scenario']!r}")
    if row["arm_id"] not in arm_ids(include_raw=True):
        raise ValueError(f"unknown benchmark arm {row['arm_id']!r}")
