"""Verify frozen delivery before opening held-out labels and reporting endpoints."""
from pathlib import Path
import csv
import numpy as np
from .preparation import TARGET
from .preparation import PROTOCOL
from .preparation import digest
from .preparation import read_json
from .preparation import verify_private
from .preparation import load_prepared
from .preparation import write_json
from .metrics import metrics


def score(prep: Path, run: Path, *, persist: bool = True) -> dict:
    """Open held-out labels only after the predictions and all upstream artifacts have been frozen."""
    delivery = read_json(run / "DELIVERY.json")
    for name, sha in delivery["artifacts"].items():
        if digest(run / name) != sha:
            raise ValueError(f"delivery artifact changed: {name}")
    if delivery["prepare_sha256"] != digest(prep / "PREPARE.json"):
        raise ValueError("preparation changed after delivery")
    labels = verify_private(prep, load_prepared(prep))
    predictions = read_json(run / "predictions.json")
    heldout = [p for p in predictions if p["usage"] == "evaluation_only"]
    y = [labels[p["well"]][TARGET] for p in heldout]
    p = [r["predicted_ctnt_positive_percent"] for r in heldout]
    baseline = [r["baseline_ctnt_positive_percent"] for r in heldout]
    donors = sorted({labels[r["well"]]["donor_id"] for r in heldout})
    donor_mae = []
    for donor in donors:
        keep = [i for i,r in enumerate(heldout) if labels[r["well"]]["donor_id"] == donor]
        donor_mae.append(metrics(np.array(y)[keep], np.array(p)[keep])["mae_percentage_points"])
    batch_metrics = {}
    for field in ("experiment_date", "plate_id"):
        groups = {}
        for value in sorted({r[field] for r in heldout}):
            keep = [i for i,r in enumerate(heldout) if r[field] == value]
            groups[value] = {"primary": metrics(np.array(y)[keep], np.array(p)[keep]),
                             "training_mean_baseline": metrics(np.array(y)[keep], np.array(baseline)[keep])}
        batch_metrics[field] = {"groups": groups,
            "macro_mae_percentage_points": float(np.mean([v["primary"]["mae_percentage_points"] for v in groups.values()])),
            "baseline_macro_mae_percentage_points": float(np.mean([v["training_mean_baseline"]["mae_percentage_points"] for v in groups.values()]))}
    result = {"status": "complete", "scope": "technical_smoke_only" if delivery["smoke"] else
              "pilot_batch_balanced_well_holdout" if delivery.get("pilot") else "batch_balanced_well_holdout",
              "protocol": PROTOCOL, "target": TARGET, "target_unit": "percent", "batch_metrics": batch_metrics,
              "segmentation_regime": read_json(run / "CONFIG.json")["segmentation_regime"],
              "primary": metrics(y, p), "training_mean_baseline": metrics(y, baseline),
              "heldout_donors": len(donors), "donor_macro_mae_percentage_points": float(np.mean(donor_mae)),
              "scheduled_wells": len(predictions), "scheduled_images": sum(len(r["image_ids"]) for r in predictions),
              "prediction_sha256": digest(run / "predictions.json"), "source_sha256": delivery["source_sha256"],
              "uncertainty": "No confidence interval: wells share donors and experimental batches.",
              "calibration": "Not estimated; raw and clipped percentage predictions retained.",
              "limitations": ["No cell-level cTnT ground truth", "No segmentation ground truth",
                              "Donors and batches may overlap across splits; not unseen-donor/batch generalization",
                              "Unknown differentiation-batch dependence may remain", "Single well split"]}
    if (run/"STAGE_READOUTS.json").exists():
        result["stage_readouts"] = {"file":"STAGE_READOUTS.json", "sha256":digest(run/"STAGE_READOUTS.json")}
    if not persist:
        return result
    write_json(run / "RESULT.json", result)
    with (run / "heldout_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["well", "experiment_date", "plate_id", TARGET, "predicted_ctnt_positive_percent", "baseline_ctnt_positive_percent"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in fields if k != TARGET}, TARGET: labels[r["well"]][TARGET]}
                         for r in heldout)
    return result
