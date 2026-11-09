"""Pilot comparison: matched input identities and budgets; descriptive deltas, no population verdict."""

from pathlib import Path
from ..runtime.reasoning import reasoning_protocol
from ..dataset import preparation as data
from ..dataset import sampling as pilot
from ..runtime.source_snapshot import verify_source
from .heldout_evaluation import score


def comparable(configs):
    fields = ("images", "pilot_sha256", "source_sha256", "seed", "model", "budget_hours", "cores",
              "train_cells_per_site", "inference_chunk", "ssl_epoch_budget", "protocol", "target")
    base = configs["control"]
    for arm, config in configs.items():
        if config.get("arm") != arm or config.get("smoke") or not config.get("pilot_sha256"):
            raise ValueError("comparison requires identified, non-smoke pilot arms")
        if any(config.get(k) != base.get(k) for k in fields):
            raise ValueError("arms differ in input, code or resource policy")
        if config["reasoning"]["protocol"] != reasoning_protocol():
            raise ValueError("reasoning protocol is not all-high")


def resumed_configs(configs, run, expected_source, codex_run=None):
    """Explicit descriptive recovery only; never silently relax the matched-budget comparison."""
    from copy import deepcopy
    lineage = data.read_json(run/"RESUME.json")
    if (lineage["policy"] != "resume-crop-no-walltime-v1" or
            configs["shape"]["source_sha256"] != lineage["source_sha256"] or
            configs["control"]["source_sha256"] != lineage["previous_source_sha256"] or
            configs["control"]["budget_hours"] != lineage["original_budget_hours"]):
        raise ValueError("invalid resumed comparison lineage")
    if lineage["source_sha256"] != expected_source:
        if codex_run is None:
            raise ValueError("new Codex source requires explicit restart lineage")
        restart = data.read_json(codex_run/"RESUME.json")
        if (restart["policy"] != "codex-crop-input-bound-v1" or
                restart["previous_source_sha256"] != lineage["source_sha256"] or
                restart["source_sha256"] != expected_source):
            raise ValueError("invalid Codex restart lineage")
        prior = Path(restart["backup"])
        if not prior.resolve().is_relative_to(codex_run.resolve()):
            raise ValueError("Codex backup outside run")
        for name, digest in restart["original_artifact_sha256"].items():
            if data.digest(data.within(prior, name)) != digest:
                raise ValueError("original Codex evidence changed")
        old = data.read_json(prior/"CONFIG.json")
        if old["source_sha256"] != restart["previous_source_sha256"]:
            raise ValueError("Codex original source mismatch")
        for key in ("images", "pilot_sha256", "seed", "model", "cores", "train_cells_per_site", "ssl_epoch_budget", "protocol", "target"):
            if old[key] != configs["codex"][key]:
                raise ValueError("Codex protocol changed during restart")
        lineage = {**lineage, "codex_restart":restart}
    backup = Path(lineage["backup"])
    if not backup.resolve().is_relative_to(run.resolve()):
        raise ValueError("recovery backup outside run")
    for name, digest in lineage["original_artifact_sha256"].items():
        if data.digest(data.within(backup, name)) != digest:
            raise ValueError("original timeout evidence changed")
    original = data.read_json(backup/"CONFIG.json")
    if original["source_sha256"] != lineage["previous_source_sha256"]:
        raise ValueError("original source differs from recovery lineage")
    normalized = deepcopy(configs)
    for arm in ("shape", "codex"):
        arm_source = lineage["source_sha256"] if arm == "shape" else expected_source
        if (configs[arm]["source_sha256"] != arm_source or
                configs[arm]["budget_hours"] is not None or configs[arm]["deadline"] is not None):
            raise ValueError("resumed arms must use intended source without a wall-clock deadline")
    for arm in normalized:
        normalized[arm]["source_sha256"] = expected_source
        normalized[arm]["budget_hours"] = None
    comparable(normalized)
    original.update(source_sha256=expected_source, budget_hours=None)
    comparable({"control": normalized["control"], "shape": original})
    return lineage


def compare(prep: Path, runs: dict, out: Path, expected_source: str, *, allow_resumed=False):
    verify_source(expected_source)
    data.fresh_output(out, prep)
    configs = {arm:data.read_json(run/"CONFIG.json") for arm,run in runs.items()}
    lineage = resumed_configs(configs, runs["shape"], expected_source, runs["codex"]) if allow_resumed else None
    if not allow_resumed:
        comparable(configs)
    results, stages = {}, {}
    for arm, run in runs.items():
        config = configs[arm]
        arm_source = lineage["previous_source_sha256"] if lineage and arm == "control" else expected_source
        if lineage and arm == "shape":
            arm_source = lineage["source_sha256"]
        if config["source_sha256"] != arm_source:
            raise ValueError("unexpected arm source snapshot")
        if (run/"CTNT_FLOW_PILOT_COMPLETE").read_text().strip() != arm_source:
            raise ValueError("pilot has no valid completion marker")
        if data.read_json(run/"RUN_META.json")["status"] != "complete":
            raise ValueError("pilot run is not complete")
        sampling = pilot.verify(prep,run/"PILOT.json")
        if (set(config["images"]) != set(sampling["images"])
                or data.digest(run/"PILOT.json") != config["pilot_sha256"]):
            raise ValueError("run image set differs from frozen pilot")
        saved = data.read_json(run/"RESULT.json")
        if data.digest(run/"STAGE_READOUTS.json") != saved["stage_readouts"]["sha256"]:
            raise ValueError("stage readouts changed after scoring")
        stages[arm] = data.read_json(run/"STAGE_READOUTS.json")
        results[arm] = score(prep,run)  # Rechecks every delivery hash and scores frozen predictions.
    rows = {}
    for arm in runs:
        result, baseline = results[arm]["primary"], results["control"]["primary"]
        rows[arm] = {"endpoint":result, "delta_vs_control":{
            k:result[k]-baseline[k] if result[k] is not None and baseline[k] is not None else None
            for k in ("mae_percentage_points","rmse_percentage_points","bias_percentage_points","r2")},
            "batch_metrics":results[arm]["batch_metrics"],
            "segment":stages[arm]["segment"]["well_macro"],
            "crop":stages[arm]["crop"]["well_macro_quality"],
            "embed":stages[arm]["embed"],"batch":stages[arm]["batch"],"process":stages[arm]["process"],
            "unavailable":stages[arm]["unavailable"]}
    report = {"scope":"pilot_descriptive_comparison", "source_sha256":expected_source,
              "pilot_sha256":configs["control"]["pilot_sha256"],"arms":rows,
              "interpretation":"MAE/RMSE delta below zero favors arm; QC proxies are descriptive; no GAIN/LOSS verdict"}
    if lineage:
        report.update(scope="pilot_resumed_descriptive_comparison", recovery=lineage,
                      resource_comparability="Not a matched wall-clock-budget experiment: control completed under 4h; SHAPE resumed and Codex unlimited.",
                      configs={a:{k:c[k] for k in ("source_sha256", "budget_hours", "cores")} for a,c in configs.items()})
    data.write_json(out/"COMPARISON.json",report)
    lines = ["# cTnT 20% pilot", "", "| Arm | MAE (pp) | RMSE (pp) | Bias (pp) | R2 |", "|---|---:|---:|---:|---:|"]
    for arm in runs:
        m=rows[arm]["endpoint"]
        lines.append(f"| {arm} | {m['mae_percentage_points']:.4f} | {m['rmse_percentage_points']:.4f} | {m['bias_percentage_points']:.4f} | {m['r2']} |")
    lines += ["", "Stage QC, embedding diagnostics, batch structure and resource usage: COMPARISON.json.",
              "Small pilot; no population or causal attribution claim."]
    if lineage:
        lines += ["", report["resource_comparability"],
                  "SHAPE reuses decisions made before its timeout. Crop token usage is a lower bound: original worker totals were lost on termination."]
    (out/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report
