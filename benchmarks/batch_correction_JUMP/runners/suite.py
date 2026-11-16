"""Run and summarize the raw baseline plus the five official benchmark arms."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.batch_correction_JUMP.runners import benchmark_contract


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_scores(prep: Path, array_path: Path, control: dict) -> dict:
    from benchmarks.batch_correction_JUMP.runners.control import score_views
    from benchmarks.batch_correction_JUMP.runners.prepare import load

    summary, meta, _ = load(prep)
    scenario = summary["scenario"]
    array = np.load(array_path, mmap_mode="r")
    return score_views(
        meta, array, control["band"]["halves"],
        batch_key=scenario["batch_key"], plate_types=scenario["plate_types"],
    )


def run_suite(prep: Path, output: Path, *, model: str, max_rounds: int,
              noise_band: float, cores: int | None = None, engine=None) -> dict:
    """Run every missing stage in a fresh output directory and emit the report."""
    from benchmarks.batch_correction_JUMP.runners import control as control_mod
    from benchmarks.batch_correction_JUMP.runners import score
    from benchmarks.batch_correction_JUMP.runners.llm_arms import run_gpt_only, run_react
    from benchmarks.batch_correction_JUMP.runners.shape_arm import run as run_shape

    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "BENCHMARK_REPORT.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists; use a fresh output directory")
    control_path = prep / control_mod.CONTROL_DIR / "CONTROL.json"
    if not control_path.exists():
        control_mod.control(prep)
    else:
        existing = _read(control_path)
        if existing.get("provenance") != control_mod.provenance():
            raise SystemExit(
                f"CONTROL_STALE: move {control_path.parent} aside and rerun with the current code"
            )
        if tuple(existing.get("arms", {})) != ("raw", "harmony", "scanorama"):
            raise SystemExit(
                f"CONTROL_ROSTER_STALE: move {control_path.parent} aside and rerun controls"
            )
    if engine is None:
        from shape.agent.verification.vision_engine import build_vision_engine

        engine = build_vision_engine(model)
    if engine is None:
        raise SystemExit(f"Luna engine unavailable for {model}")

    run_gpt_only(prep, output / "gpt_luna", engine=engine, model=model)
    run_react(
        prep, output / "react_luna_harmony", engine=engine,
        model=model, noise_band=noise_band, cores=cores,
    )
    shape_dir = output / "shape_luna_harmony"
    run_shape(
        prep, shape_dir, model=model, max_rounds=max_rounds,
        noise_band=noise_band, cores=cores, engine=engine,
    )
    score.final(shape_dir, prep)
    return write_report(prep, output)


def run_shape_only(prep: Path, output: Path, *, model: str, max_rounds: int, noise_band: float,
                   cores: int | None = None, engine=None) -> dict:
    """Run and score JUST the SHAPE arm (raw/harmony/scanorama baselines included for context, gpt_luna and
    react_luna_harmony skipped) -- a model-matrix ablation, not the official 6-arm benchmark_contract roster
    (`write_report`/`validate_arm_roster` enforce that exact set; this writes SHAPE_ONLY_REPORT.json, never
    BENCHMARK_REPORT.json, so the two never collide or get confused for each other). react alone burns hours
    and ~1M tokens per scenario (observed on deepseek-ai/DeepSeek-V4.1-Flash, scenario_2: 92 calls, 1.18M
    tokens) that a model-comparison ablation doesn't need."""
    from benchmarks.batch_correction_JUMP.runners import control as control_mod
    from benchmarks.batch_correction_JUMP.runners import score as score_mod
    from benchmarks.batch_correction_JUMP.runners.shape_arm import run as run_shape

    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "SHAPE_ONLY_REPORT.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists; use a fresh output directory")
    control_path = prep / control_mod.CONTROL_DIR / "CONTROL.json"
    if not control_path.exists():
        control_mod.control(prep)
    else:
        existing = _read(control_path)
        if existing.get("provenance") != control_mod.provenance():
            raise SystemExit(f"CONTROL_STALE: move {control_path.parent} aside and rerun with the current code")
    if engine is None:
        from shape.agent.verification.vision_engine import build_vision_engine

        engine = build_vision_engine(model)
    if engine is None:
        raise SystemExit(f"engine unavailable for {model}")

    shape_dir = output / "shape_luna_harmony"
    run_shape(prep, shape_dir, model=model, max_rounds=max_rounds, noise_band=noise_band, cores=cores,
             engine=engine)
    score_mod.final(shape_dir, prep)
    return write_shape_only_report(prep, output)


def write_shape_only_report(prep: Path, output: Path) -> dict:
    """SHAPE_ONLY_REPORT.json: raw/harmony/scanorama (from the shared control, free) + shape_luna_harmony.
    No react_luna_harmony or gpt_luna rows, and no benchmark_contract.validate_arm_roster call (that enforces
    the full official 6-arm set, which this deliberately isn't) -- report_row's own per-row validation still
    applies, so a malformed row still raises."""
    from benchmarks.batch_correction_JUMP.runners.control import CONTROL_DIR, provenance
    from benchmarks.batch_correction_JUMP.runners.prepare import load

    summary, meta, _ = load(prep)
    scenario = summary["scenario"]["name"]
    commit = _git_commit()
    control_path = prep / CONTROL_DIR / "CONTROL.json"
    if not control_path.exists():
        raise SystemExit(f"missing controls: {control_path}")
    control = _read(control_path)
    if control.get("provenance") != provenance():
        raise SystemExit("CONTROL_STALE: controls were produced by different tool or metric code")

    score_sets: dict[str, dict] = {}
    runtime: dict[str, float] = {}
    calls: dict[str, int] = {}
    tokens: dict[str, int] = {}
    for arm_id in ("raw", "harmony", "scanorama"):
        arm = control["arms"][arm_id]
        score_sets[arm_id] = arm
        runtime[arm_id] = float(arm.get("tool_seconds") or 0.0)
        calls[arm_id] = int(arm.get("tool_calls") or 0)
        tokens[arm_id] = 0

    shape_dir = output / "shape_luna_harmony"
    shape_result = _read(shape_dir / "RESULT.json")
    shape_meta = _read(shape_dir / "RUN_META.json")
    shape_agent = _read(shape_dir / "AGENT_RUN.json")
    if shape_meta.get("commit") != commit or shape_meta.get("scenario") != scenario:
        raise SystemExit("INVALID_ARM_OUTPUT: SHAPE identity mismatch")
    score_sets["shape_luna_harmony"] = shape_result["scores"]["agent"]
    runtime["shape_luna_harmony"] = float(shape_agent["ended"] - shape_agent["started"])
    calls["shape_luna_harmony"] = int(shape_result["usage"]["tool_calls_started"])
    tokens["shape_luna_harmony"] = int((shape_agent.get("tokens") or {}).get("total_tokens") or 0)

    roster = ("raw", "harmony", "scanorama", "shape_luna_harmony")
    raw_test = score_sets["raw"]["test"]
    n_profiles = int(summary.get("n_wells", len(meta)))
    rows = [report_row(scenario, arm_id, score_sets[arm_id]["test"], raw_test, runtime=runtime[arm_id],
                       n_calls=calls[arm_id], tokens=tokens[arm_id], n_profiles=n_profiles)
            for arm_id in roster]

    report = {"scenario": scenario, "git_commit": commit, "model": shape_agent.get("model"),
              "roster": list(roster), "rows": rows,
              "note": "shape-only model-matrix ablation -- gpt_luna and react_luna_harmony deliberately not "
                      "run; see write_report/BENCHMARK_REPORT.json for the official 6-arm comparison."}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "SHAPE_ONLY_REPORT.json"
    report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    fieldnames = list(rows[0])
    with (output / "SHAPE_ONLY_REPORT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report


def report_row(scenario: str, arm_id: str, test: dict, raw_test: dict, *, runtime: float, n_calls: int,
               tokens: int, n_profiles: int) -> dict:
    """One report row: the arm's test-view scores and its cost per well profile."""
    nonrep_gain = (
        test.get("nonrep_mean_map") - raw_test.get("nonrep_mean_map")
        if test.get("nonrep_mean_map") is not None and raw_test.get("nonrep_mean_map") is not None
        else None
    )
    row = {
        "scenario": scenario,
        "arm_id": arm_id,
        "batch_correction": test.get("batch_mean"),
        "biometrics": test.get("bio_mean"),
        "runtime_per_image": runtime / n_profiles,
        "tool_calls_per_image": n_calls / n_profiles,
        "ap50_gain_per_tool_call": None,
        "token_per_image": tokens / n_profiles,
        "overall": test.get("overall"),
        "nonrep_mean_map": test.get("nonrep_mean_map"),
        "nonrep_map_gain_per_tool_call": (
            nonrep_gain / n_calls if nonrep_gain is not None and n_calls else None
        ),
        "runtime_seconds": runtime,
        "tool_calls": n_calls,
        "tokens": tokens,
        "n_images": n_profiles,
    }
    benchmark_contract.validate_report_row(row)
    return row


def write_report(prep: Path, output: Path) -> dict:
    """Score all six representations and write JSON, CSV, and Markdown summaries."""
    from benchmarks.batch_correction_JUMP.runners.control import CONTROL_DIR, provenance
    from benchmarks.batch_correction_JUMP.runners.prepare import load

    summary, meta, _ = load(prep)
    scenario = summary["scenario"]["name"]
    commit = _git_commit()
    control_path = prep / CONTROL_DIR / "CONTROL.json"
    if not control_path.exists():
        raise SystemExit(f"missing controls: {control_path}")
    control = _read(control_path)
    if control.get("provenance") != provenance():
        raise SystemExit("CONTROL_STALE: controls were produced by different tool or metric code")

    score_sets: dict[str, dict] = {}
    runtime: dict[str, float] = {}
    calls: dict[str, int] = {}
    tokens: dict[str, int] = {}
    for arm_id in ("raw", "harmony", "scanorama"):
        arm = control["arms"][arm_id]
        score_sets[arm_id] = arm
        runtime[arm_id] = float(arm.get("tool_seconds") or 0.0)
        calls[arm_id] = int(arm.get("tool_calls") or 0)
        tokens[arm_id] = 0

    for arm_id in ("gpt_luna", "react_luna_harmony"):
        arm_dir = output / arm_id
        record = _read(arm_dir / "ARM_OUTPUT.json")
        if record.get("arm_id") != arm_id or record.get("scenario") != scenario:
            raise SystemExit(f"INVALID_ARM_OUTPUT: {arm_id} identity mismatch")
        if record.get("git_commit") != commit:
            raise SystemExit(f"INVALID_ARM_OUTPUT: {arm_id} commit mismatch")
        array_path = Path(record["array"])
        if not array_path.is_file():
            raise SystemExit(f"INVALID_ARM_OUTPUT: missing {array_path}")
        score_sets[arm_id] = _arm_scores(prep, array_path, control)
        runtime[arm_id] = float(record["runtime_seconds"])
        calls[arm_id] = int(record["tool_calls"])
        tokens[arm_id] = int((record.get("tokens") or {}).get("total_tokens") or 0)

    shape_dir = output / "shape_luna_harmony"
    shape_result = _read(shape_dir / "RESULT.json")
    shape_meta = _read(shape_dir / "RUN_META.json")
    shape_agent = _read(shape_dir / "AGENT_RUN.json")
    if shape_meta.get("commit") != commit or shape_meta.get("scenario") != scenario:
        raise SystemExit("INVALID_ARM_OUTPUT: SHAPE identity mismatch")
    score_sets["shape_luna_harmony"] = shape_result["scores"]["agent"]
    runtime["shape_luna_harmony"] = float(shape_agent["ended"] - shape_agent["started"])
    calls["shape_luna_harmony"] = int(shape_result["usage"]["tool_calls_started"])
    tokens["shape_luna_harmony"] = int((shape_agent.get("tokens") or {}).get("total_tokens") or 0)

    roster = benchmark_contract.validate_arm_roster(tuple(score_sets))
    raw_test = score_sets["raw"]["test"]
    n_profiles = int(summary.get("n_wells", len(meta)))
    rows = []
    for arm_id in roster:
        rows.append(report_row(scenario, arm_id, score_sets[arm_id]["test"], raw_test, runtime=runtime[arm_id],
                               n_calls=calls[arm_id], tokens=tokens[arm_id], n_profiles=n_profiles))

    report = {
        "scenario": scenario, "git_commit": commit,
        "model": shape_agent.get("model") or benchmark_contract.LUNA_MODEL,
        "roster": list(roster), "rows": rows,
        "metric_note": (
            "One JUMP well profile is the denominator for the requested per-image fields. "
            "AP50 is undefined for embedding batch correction, so ap50_gain_per_tool_call is null; "
            "nonrep_map_gain_per_tool_call is reported as the task-relevant replacement. "
            "runtime_seconds is each arm's own wall time from the loaded prep to its output "
            "(LLM calls, tool calls and the arm's calibration scoring included; the harness's final "
            "scoring excluded); for Harmony and Scanorama it is the one correction call. "
            "The Harmony arm runs harmonypy's library defaults (adaptive nclust, dynamic lambda), "
            "not the configuration of the authors' pipeline."
        ),
        "shape_implementation": {
            "source": "main",
            "runner": "BatchCorrectionVerifier through StepVerifier wave mode",
            "tool": "shape.tools.batch_corrector.Batch_Corrector_Tool",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "BENCHMARK_REPORT.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8",
    )
    fieldnames = list(rows[0])
    with (output / "BENCHMARK_REPORT.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    by_id = {arm.arm_id: arm.display_name for arm in (benchmark_contract.RAW_BASELINE, *benchmark_contract.OFFICIAL_ARMS)}
    lines = [
        "# JUMP batch-correction benchmark",
        "",
        "| arm | batch correction | biometrics | seconds/profile | calls/profile | tokens/profile | nonrep mAP gain/call |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fmt = lambda value: "n/a" if value is None else f"{value:.6g}"
    for row in rows:
        lines.append(
            f"| {by_id[row['arm_id']]} | {fmt(row['batch_correction'])} | {fmt(row['biometrics'])} | "
            f"{fmt(row['runtime_per_image'])} | {fmt(row['tool_calls_per_image'])} | "
            f"{fmt(row['token_per_image'])} | {fmt(row['nonrep_map_gain_per_tool_call'])} |"
        )
    lines += ["", report["metric_note"], ""]
    (output / "BENCHMARK_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report
