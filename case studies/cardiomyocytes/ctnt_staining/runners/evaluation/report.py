"""Replay the primary and supplementary tables from curated public evidence."""
import statistics
from pathlib import Path
from evaluation_common import ARMS, SPLITS, csv_rows, read, sha, write, write_csv
from process import evaluate


def donor_macro(metrics, key):
    values = [v[key] for v in metrics.values() if v.get(key) is not None]
    return statistics.mean(values) if values else None


def build(result, output):
    root, out = Path(result), Path(output)
    evidence = root / "evidence"
    quality = csv_rows(evidence / "stage_quality.csv")
    lookup = {(r["arm"], r["metric"]): float(r["value"]) for r in quality}
    dapi = read(evidence / "dapi_crop_summary.json")["well_macro"]
    probe = read(evidence / "linear_probe.json")
    population = read(evidence / "population.json")
    costs = csv_rows(evidence / "stage_costs.csv")
    main, splits, donors = [], [], []
    for arm in ARMS:
        p = probe[arm]
        held = p["by_split"]["evaluation_only"]
        main.append(dict(
            arm=arm, delivered_embeddings=population[arm]["all_cells"],
            seg_bf_proxy_Q=lookup[arm, "seg_quant_score"],
            crop_R_minus_C=lookup[arm, "crop_retention_minus_contamination"],
            crop_DAPI_target_nuclear_retention=dapi[arm]["target_nuclear_retention"],
            crop_DAPI_other_nuclear_pixel_fraction=dapi[arm]["other_nuclear_pixel_fraction"],
            heldout_accuracy=held["accuracy"], heldout_eligible_cells=held["cells"],
            heldout_donor_equal_balanced_accuracy=donor_macro(p["by_donor"]["evaluation_only"], "balanced_accuracy"),
            native_stage_elapsed_sum_minutes=sum(float(r["minutes"]) for r in costs if r["arm"] == arm),
            total_tokens=sum(int(r["total_tokens"]) for r in costs if r["arm"] == arm),
        ))
        for split in SPLITS:
            splits.append(dict(arm=arm, split=split, **p["by_split"][split],
                               embeddings=population[arm]["counts_by_split"][split]))
            for donor, metrics in sorted(p["by_donor"][split].items()):
                donors.append(dict(arm=arm, split=split, donor=donor, **metrics))
    write_csv(out / "main_metrics.csv", main)
    write_csv(out / "split_metrics.csv", splits)
    write_csv(out / "donor_metrics.csv", donors)
    failure = evaluate(read(evidence / "native_attempts.json"), read(evidence / "controller_decisions.json"),
                       expected=[(arm, stage) for arm in ARMS for stage in ("segment", "crop", "embed", "batch")])
    write(out / "failure_efficiency.json", failure)
    write_csv(out / "failure_efficiency.csv", [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in failure["rows"]])
    write(out / "metrics.json", dict(
        schema="staining-primary-evaluation-v1", arms=main,
        quality_aggregation="objects -> image mean -> equal well mean; 114 wells",
        accuracy_aggregation="all eligible held-out own-arm cells, micro accuracy; supervised readout",
        donor_aggregation="equal donor mean; balanced accuracy only when both classes present",
        F3="failure_efficiency.json; explicit stage denominators, not a single undifferentiated count",
        limitations=["no full-cell segmentation ground truth", "different own-arm populations",
                     "exploratory fluorescence labels", "historical cost not a controlled speed comparison"],
    ))
    write(out / "REPLAY.json", dict(
        inputs=[dict(path=p.relative_to(root).as_posix(), sha256=sha(p)) for p in sorted(evidence.iterdir()) if p.is_file()],
        script_sha256=sha(Path(__file__)),
        outputs=[dict(path=p.name, sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name in {
            "main_metrics.csv", "split_metrics.csv", "donor_metrics.csv", "failure_efficiency.json", "failure_efficiency.csv", "metrics.json"}],
    ))


def manifest(result, verify=False):
    root = Path(result)
    target = root / "MANIFEST.json"
    if verify:
        for item in read(target)["files"]:
            if sha(root / item["path"]) != item["sha256"]:
                raise ValueError(f"Hash mismatch: {item['path']}")
        return
    write(target, dict(files=[dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                              for p in sorted(root.rglob("*")) if p.is_file() and p != target]))
