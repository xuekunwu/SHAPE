"""Replay frozen cardiac organoid CSVs; no image/model/LLM dependencies."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HF_REPO = "SHAPE-Morph/result"
HF_REVISION = "cd0ab63f74fc304d481dad55912cc6afca00e780"
HF_FOLDER = "cardiac_organoids_segmentation"
MANIFEST_SHA256 = "bbafaee99e0b918b6fc16b3b59eaaacbd82ba4a848004cc4423df79de0dec5d8"
BASELINES = ("shape_seg_co", "microsam", "cellpose_sam", "organoid_id", "stardist")
ARMS = (*BASELINES, "shape", "llm", "react")
SCORES = ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50")
COSTS = ("seconds", "native_seconds", "llm_seconds", "native_calls", "llm_calls", "tokens")
IDENTITY = ("image_sha256", "ground_truth_sha256", "image_height_px", "image_width_px")
LABELS = {"shape_seg_co": "Direct CO", "microsam": "microSAM",
          "cellpose_sam": "Cellpose-SAM", "organoid_id": "OrganoID",
          "stardist": "StarDist", "shape": "SHAPE", "llm": "Direct LLM", "react": "ReAct"}
COHORT_NOTES = {
    "full_baselines": "Complete frozen cohort; five non-LLM baselines.",
    "shape_observed": "151 recorded images of an interrupted 745-image SHAPE sample; provisional.",
    "llm_react_canary": "20-image development canary; SHAPE has no records on these images.",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def download():
    """Use the official HF client and its existing login; never read/print tokens."""
    from huggingface_hub import hf_hub_download
    manifest = Path(hf_hub_download(HF_REPO, f"{HF_FOLDER}/manifest.json",
                                   repo_type="dataset", revision=HF_REVISION))
    if sha256(manifest.read_bytes()) != MANIFEST_SHA256:
        raise ValueError("Pinned manifest hash mismatch")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    for name in meta["files"]:
        hf_hub_download(HF_REPO, f"{HF_FOLDER}/{name}",
                        repo_type="dataset", revision=HF_REVISION)
    return manifest.parent


def observed(row):
    return row["evaluation_status"] != "not_completed"


def validate_rows(arm, rows, meta):
    """Refuse denominator changes, nonfinite scores and missing -> zero conversion."""
    if len(rows) != meta["rows"] or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError(f"{arm}: row count or duplicate image ID")
    counts = Counter(r["evaluation_status"] for r in rows)
    expected = {"success": meta["delivered_masks"], "failed": meta["failed_predictions"],
                "not_completed": meta["not_completed"]}
    if any(status not in expected for status in counts) or any(counts[k] != n for k, n in expected.items()):
        raise ValueError(f"{arm}: unexpected status counts")
    if sum(observed(r) for r in rows) != meta["recorded_predictions"]:
        raise ValueError(f"{arm}: recorded count mismatch")
    for row in rows:
        for key, value in (("arm", arm), ("source_run", meta["source_run"]),
                           ("source_commit", meta["source_commit"]),
                           ("manifest_fingerprint", meta["manifest_fingerprint"]),
                           ("run_kind", meta["run_kind"]), ("run_status", meta["run_status"])):
            if row[key] != value:
                raise ValueError(f"{arm}: {key} mismatch")
        if any(not row[key] for key in IDENTITY):
            raise ValueError(f"{arm}: missing input identity")
        if not observed(row):
            metrics = (*SCORES, *COSTS, "ground_truth_cell_count", "predicted_cell_count",
                       "mean_matched_iou_at_50", "cell_count_error_fraction", "tool_calls",
                       "started_unix", "mask_sha256", "error", "delivered")
            if any(row[key] for key in metrics) or any(v for k, v in row.items() if k.startswith("iou_")):
                raise ValueError(f"{arm}: not_completed row has metrics")
            continue
        success = row["evaluation_status"] == "success"
        if row["delivered"] != ("true" if success else "false") or bool(row["error"]) == success:
            raise ValueError(f"{arm}: inconsistent delivery/error status")
        for key in (*SCORES, *COSTS):
            if row[key] == "" and key == "native_seconds":
                continue
            value = float(row[key])
            if not math.isfinite(value) or value < 0 or (key in SCORES and value > 1):
                raise ValueError(f"{arm}: invalid {key}")


def load_inputs(folder):
    folder = Path(folder)
    raw_manifest = (folder / "manifest.json").read_bytes()
    if sha256(raw_manifest) != MANIFEST_SHA256:
        raise ValueError("Input manifest does not match the published HF revision")
    meta = json.loads(raw_manifest)
    if set(meta["files"]) != {f"{a}.csv" for a in ARMS}:
        raise ValueError("Unexpected arm list")
    data = {}
    for name, entry in meta["files"].items():
        raw = (folder / name).read_bytes()
        if sha256(raw) != entry["sha256"] or len(raw) != entry["bytes"]:
            raise ValueError(f"{name}: source checksum/size mismatch")
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
        if reader.fieldnames != meta["columns"]:
            raise ValueError(f"{name}: CSV schema mismatch")
        rows = list(reader)
        validate_rows(entry["arm"], rows, entry)
        data[entry["arm"]] = {row["image_id"]: row for row in rows}
    return meta, data


def cohorts(data):
    sets = {arm: {key for key, row in rows.items() if observed(row)} for arm, rows in data.items()}
    full = sets[BASELINES[0]]
    if not full or any(sets[arm] != full for arm in BASELINES):
        raise ValueError("Full baseline cohorts differ")
    if sets["llm"] != sets["react"] or not sets["llm"]:
        raise ValueError("LLM/ReAct canary cohorts differ")
    if not sets["shape"] or not sets["shape"] <= full or not sets["llm"] <= full:
        raise ValueError("Agent images are absent from full baselines")
    for arm, rows in data.items():
        for key, row in rows.items():
            if key not in full or any(row[k] != data[BASELINES[0]][key][k] for k in IDENTITY):
                raise ValueError(f"{arm}: input/GT identity mismatch")
    co_weights = {row["checkpoint_sha256"] for arm in ("shape_seg_co", "shape", "react")
                  for row in data[arm].values()}
    if len(co_weights) != 1 or "" in co_weights:
        raise ValueError("CO checkpoint differs across arms")
    return {
        "full_baselines": (tuple(sorted(full)), BASELINES),
        "shape_observed": (tuple(sorted(sets["shape"])), (*BASELINES, "shape")),
        "llm_react_canary": (tuple(sorted(sets["llm"])), (*BASELINES, "llm", "react")),
    }


def mean(rows, key):
    values = [float(row[key]) for row in rows if row[key] != ""]
    return statistics.fmean(values) if len(values) == len(rows) and rows else None


def summarize(data, ids, arms):
    summaries, deltas = [], []
    baseline = data["shape_seg_co"]
    for arm in arms:
        rows = [data[arm][key] for key in ids]
        item = {"arm": arm, "n_images": len(rows),
                "n_delivered": sum(r["delivered"] == "true" for r in rows),
                "n_failed": sum(r["evaluation_status"] == "failed" for r in rows)}
        item.update({key: mean(rows, key) for key in SCORES})
        for key in COSTS:
            item[f"{key}_per_image"] = mean(rows, key)
        summaries.append(item)
        if arm == "shape_seg_co":
            continue
        delta = {"arm": arm, "reference": "shape_seg_co", "n_pairs": len(ids)}
        for key in ("ap50", "ap75", "map_50_95"):
            values = [float(data[arm][i][key]) - float(baseline[i][key]) for i in ids]
            delta[f"{key}_mean_delta"] = statistics.fmean(values)
            delta[f"{key}_wins"] = sum(v > 1e-12 for v in values)
            delta[f"{key}_ties"] = sum(abs(v) <= 1e-12 for v in values)
            delta[f"{key}_losses"] = sum(v < -1e-12 for v in values)
        deltas.append(delta)
    return summaries, deltas


def csv_bytes(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue().encode("utf-8")


def table(rows):
    lines = ["| Arm | N | Failed | AP50 | AP75 | mAP50–95 | Seconds/image | Tokens/image |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {LABELS[row['arm']]} | {row['n_images']} | {row['n_failed']} | "
                     f"{row['ap50']:.6f} | {row['ap75']:.6f} | {row['map_50_95']:.6f} | "
                     f"{row['seconds_per_image']:.3f} | {row['tokens_per_image']:.1f} |")
    return "\n".join(lines)


def build(folder):
    meta, data = load_inputs(folder)
    groups = cohorts(data)
    summaries, pairwise = [], []
    files, membership = {}, {}
    for name, (ids, arms) in groups.items():
        rows, deltas = summarize(data, ids, arms)
        summaries.extend({"cohort": name, **row} for row in rows)
        pairwise.extend({"cohort": name, **row} for row in deltas)
        files[f"{name}.csv"] = csv_bytes(rows)
        membership[name] = {"n_images": len(ids), "arms": list(arms),
                            "sorted_image_ids_sha256": sha256(("\n".join(ids) + "\n").encode()),
                            "scope": COHORT_NOTES[name]}
    coverage = []
    for arm in ARMS:
        entry = meta["files"][f"{arm}.csv"]
        coverage.append({key: entry[key] for key in (
            "arm", "run_kind", "run_status", "rows", "recorded_predictions",
            "delivered_masks", "failed_predictions", "not_completed", "source_commit", "source_run")})
    files["coverage.csv"] = csv_bytes(coverage)
    files["paired_vs_direct_co.csv"] = csv_bytes(pairwise)
    sets = [{i for i, r in data[a].items() if observed(r)} for a in ARMS]
    report = {
        "schema": 1,
        "source": {"repo_id": HF_REPO, "revision": HF_REVISION, "folder": HF_FOLDER,
                   "manifest_sha256": MANIFEST_SHA256, "files": meta["files"]},
        "cohorts": membership,
        "all_eight_arms_common_images": len(set.intersection(*sets)),
        "shape_canary_common_images": len(sets[ARMS.index("shape")] & sets[ARMS.index("llm")]),
        "summaries": summaries, "paired_vs_direct_co": pairwise,
        "score_definition": "Per-image AP(t)=TP/(TP+FP+FN); macro mean, IoU t=0.50:0.05:0.95. Not confidence-ranked COCO AP.",
        "failure_policy": "Recorded failures retain original scorer values and remain in paired denominators. not_completed rows are excluded, never zero-filled.",
        "cost_scope": "Recorded image attempts only; excludes preflight, unrecorded interrupted attempts, adapter construction, queue and data preparation. Mixed hardware: descriptive only.",
        "limitations": [
            "SHAPE 151/745 is an interrupted prefix, not a completed representative sample.",
            "LLM/ReAct are development canaries; no shared eight-arm image cohort exists.",
            "Baseline code commit 00fad589 differs from agent commit 8f59e6b5; matching inputs/checkpoints does not establish equal code or search budgets.",
            "Training/evaluation non-overlap and biological replicate independence have not been verified.",
            "No biological endpoint or downstream end-to-end study has been evaluated.",
            "HF package lists are partial observations, not full environment locks; original hardware is unavailable in this export.",
        ],
        "stardist_recovery": {"job_id": "48581417", "elapsed": "00:41:15",
            "allocated_cpus": 8, "requested_memory_gib": 16, "gpu": False,
            "initial_records_preserved": 6194, "status": meta["stardist_completion"]["status"],
            "recorded": 7582, "delivered": 7578, "native_worker_failures": 4,
            "mechanism": "Spawned process isolation around unchanged StarDist adapter; native abort becomes a failed prediction and the next worker continues.",
            "recovery_wrapper_in_export": False},
    }
    files["comparison.json"] = encoded(report)
    lines = [
        "# Cardiac organoid segmentation comparison", "",
        "Five non-LLM baselines cover all **7,582 images**. StarDist finished with four recorded native failures. "
        "SHAPE has **151/745 recorded images**; direct LLM and ReAct each have a separate **20-image development canary**. "
        "**There are no images shared by all eight arms**, so no eight-arm ranking is reported.", "",
        f"Per-image source: [SHAPE-Morph/result, fixed commit]("
        f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION}/{HF_FOLDER}). "
        "The private source CSVs, image IDs, raw images, masks, weights and logs remain outside Git. "
        "This directory contains aggregate comparisons only.", "",
        "## Matched-image results", "",
        "Scores are fractions, averaged over images. Recorded failures remain in each denominator. "
        "AP(t) = TP/(TP+FP+FN) after maximum-cardinality instance matching at each IoU threshold; "
        "mAP averages thresholds 0.50–0.95 in steps of 0.05. These are not confidence-ranked COCO AP.",
    ]
    for name, (ids, arms) in groups.items():
        rows = [r for r in summaries if r["cohort"] == name]
        lines.extend(["", f"### {name} (N={len(ids)})", "", COHORT_NOTES[name], "",
                      table(rows), "", f"Unrounded values: [{name}.csv]({name}.csv)."])
    delta = next(r for r in pairwise if r["cohort"] == "shape_observed" and r["arm"] == "shape")
    lines.extend([
        "", "## Interpretation and scope", "",
        f"On the same 151 recorded images, SHAPE minus direct CO is "
        f"{delta['ap50_mean_delta']:+.6f} AP50 and {delta['map_50_95_mean_delta']:+.6f} mAP. "
        "This is a provisional descriptive difference on an interrupted subset. "
        "microSAM has the highest mAP in the full baseline cohort and in the SHAPE subset; "
        "SHAPE has the highest AP50 in its observed subset. "
        "The 20-image canary is disjoint from the SHAPE records.", "",
        "The 594 unrecorded SHAPE images have empty metric cells and are excluded from observed scores. "
        "The one recorded SHAPE failure and four recorded StarDist failures retain the original scorer values. "
        "A delivered mask denotes operational success, not segmentation accuracy. "
        "Generic StarDist performs poorly against whole-organoid instance annotations.", "",
        "Baseline source: 00fad58934b50008bfe4186a2e5e4cfd616662cd; "
        "SHAPE/LLM/ReAct source: 8f59e6b53b84229236bd091b9bb252ba83b266d9. "
        "All CO arms use checkpoint SHA256 f125b9754521efa042ef1ee614d8d900843b2c933517f393daf2e37268d20c48. "
        "Agent model is gpt-5.6-luna; the native runner defaults to high reasoning. "
        "The CSV export does not independently record all runtime settings or prove identical search budgets.", "",
        "Training/evaluation non-overlap and biological replicate independence have not been verified. "
        "No held-out generalization, statistical significance or biological endpoint is claimed.", "",
        "## Runtime, failure recovery and provenance", "",
        "Times and tokens above cover recorded image attempts only. Preflight, interrupted attempts without a final record, "
        "adapter construction, preparation and queue time are unavailable here. "
        "Arms used different environments/hardware; these times are descriptive, not hardware speed comparisons. "
        "Partial observed package versions, source runs, checkpoint hashes and file SHA256 values are preserved in "
        "[comparison.json](comparison.json), together with cohort membership digests.", "",
        "StarDist recovery job 48581417 completed in 00:41:15 with 8 CPUs / 16 GiB requested RAM and no GPU. "
        "The initial 6,194 records were preserved. Native Clipper aborts were isolated in spawned workers around the unchanged adapter; "
        "the four failures remain scored. The executed external recovery wrapper and complete hardware/environment locks "
        "are not contained in the HF CSV export, so this publication does not claim a byte-exact native recovery replay.", "",
        "## Files and replay", "",
        "- [coverage.csv](coverage.csv): planned/observed/delivered/failed/missing counts and source identity for all eight arms.",
        "- [paired_vs_direct_co.csv](paired_vs_direct_co.csv): same-image mean deltas and win/tie/loss counts; ties use absolute tolerance 1e-12.",
        "- [comparison.json](comparison.json): unrounded metrics, costs, provenance, cohort digests and limitations.",
        "- [EXPORT.json](EXPORT.json): SHA256 and byte counts for every generated report file.", "",
        "Use [the runner recipes](../../runners/README.md) to reproduce these files from the pinned export, "
        "verify the published hashes, run contract tests, or launch the original inference implementation "
        "from its exact source commit. Report generation uses Python 3.10+ and the standard library; "
        "only authenticated downloading needs huggingface-hub. No inference or model calls occur during replay.", "",
    ])
    files["README.md"] = "\n".join(lines).encode("utf-8")
    files["EXPORT.json"] = encoded({"schema": 1, "source_revision": HF_REVISION,
        "files": {name: {"sha256": sha256(raw), "bytes": len(raw)} for name, raw in sorted(files.items())}})
    return files


def write(folder, output):
    output = Path(output)
    if output.resolve() == Path(folder).resolve():
        raise ValueError("Output must differ from the source CSV directory")
    files = build(folder)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    return files


def verify(folder, published):
    expected = build(folder)
    published = Path(published)
    for name, raw in expected.items():
        if (published / name).read_bytes() != raw:
            raise ValueError(f"Published file does not reproduce: {name}")
    return len(expected)
