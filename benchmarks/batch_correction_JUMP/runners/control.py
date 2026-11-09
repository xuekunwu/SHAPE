"""The non-agentic arms, computed once per scenario prep. Representations, all on
the prepared features in prep row order:
  raw            the uncorrected input
  harmony        harmonypy at its documented defaults (random_state 0)
  scanorama      Scanorama correction grouped by the scenario batch key
Each is scored on three views -- val (calibration compounds + controls), test (evaluation compounds +
controls; the primary view) and all wells -- and on the fixed plate halves of the test view. The stage
also records the evaluator's admission evidence (it uses every label and is never shown to the SHAPE
arm). Outputs go to <prep>/shape_control/ (CONTROL.json, control_<arm>.npy, read-only); scoring
refuses a control computed with other tool or metric code.
"""
from __future__ import annotations

import hashlib
import json
import stat
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

CONTROL_DIR = "shape_control"
N_HALVES = 20
HALF_SEED = 0
VIEWS = {"val": "val", "test": "test", "all": None}


def provenance() -> dict:
    from shape.tools.batch_corrector import quant, tool

    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    return {"tool_version": tool.TOOL_VERSION,
            "tool_sha256": hashlib.sha256(Path(tool.__file__).read_bytes()).hexdigest(),
            "quant_sha256": hashlib.sha256(Path(quant.__file__).read_bytes()).hexdigest(),
            "control_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "harmonypy": installed("harmonypy"), "scib_metrics": installed("scib-metrics"),
            "copairs": installed("copairs"), "scanorama": installed("scanorama")}


def run_harmony(features: np.ndarray, batch: np.ndarray, *, batch_key: str) -> tuple[np.ndarray, dict]:
    """Run the standalone Harmony baseline without SHAPE's detector or verifier."""
    import pandas as pd

    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG, _default_harmony_runner

    metadata = pd.DataFrame({batch_key: np.asarray(batch).astype(str)})
    return _default_harmony_runner(np.asarray(features, dtype=float), metadata, batch_key, dict(DEFAULT_CONFIG))


def run_scanorama(features: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Run the authors' non-PCA Scanorama arm and restore the original row order."""
    from scanorama import assemble

    values = np.asarray(features, dtype=float)
    groups = np.asarray(batch).astype(str)
    indices = [np.flatnonzero(groups == label) for label in sorted(set(groups))]
    corrected = assemble([values[index] for index in indices], batch_size=5000)
    joined = np.concatenate(corrected)
    original_order = np.argsort(np.concatenate(indices))
    output = np.asarray(joined[original_order], dtype=float)
    if output.shape != values.shape or not np.isfinite(output).all():
        raise ValueError("Scanorama returned an invalid corrected matrix")
    return output


def run_tool(features: np.ndarray, batch: np.ndarray, params: dict, *, batch_key: str) -> dict:
    """One call of the tool on every well: the batch column only as metadata (as strings)."""
    import pandas as pd

    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool

    metadata = pd.DataFrame({batch_key: np.asarray(batch).astype(str)})
    return Batch_Corrector_Tool().execute(features, metadata, batch_key=batch_key, harmony_params=dict(params))


def score_views(meta, array, halves, *, batch_key: str, plate_types) -> dict:
    from shape.tools.batch_corrector import quant

    kw = {"batch_key": batch_key, "plate_types": plate_types}
    started = time.time()
    entry = {view: quant.score_split(meta, array, split, **kw) for view, split in VIEWS.items()}
    entry["test_halves"] = quant.score_halves(meta, array, "test", halves, **kw)
    entry["eval_seconds"] = round(time.time() - started, 1)
    return entry


def admission_evidence(meta, raw, harmony, *, batch_key: str, plate_types) -> dict:
    """Evaluator-side (all labels): overlap of treatment compounds across batch groups, technical
    batch signal on all wells and on DMSO only, and same- vs cross-batch replicate retrieval."""
    import pandas as pd

    from shape.tools.batch_corrector import quant
    from shape.tools.batch_corrector.tool import detect_batch_effect

    trt = meta[meta["Metadata_PertType"].astype(str) == "trt"]
    per_group = trt.groupby(batch_key, observed=True)[quant.LABEL_KEY].apply(lambda s: set(s.astype(str)))
    groups = sorted(per_group.index.astype(str))
    sets = {str(g): per_group.loc[g] for g in per_group.index}
    union = set().union(*sets.values()) if sets else set()
    shared_all = set.intersection(*sets.values()) if sets else set()
    in_one = sum(1 for c in union if sum(c in s for s in sets.values()) == 1)
    batch = meta[batch_key].astype(str).to_numpy()
    dmso = (meta[quant.LABEL_KEY].astype(str) == quant.DMSO).to_numpy()

    def gate(rows):
        return asdict(detect_batch_effect(np.asarray(raw, dtype=float)[rows], pd.DataFrame({"batch": batch[rows]}),
                                          batch_key="batch"))

    evidence = {
        "batch_key": batch_key, "n_groups": len(groups),
        "compounds_per_group": {g: len(sets[g]) for g in groups},
        "compounds_total": len(union), "compounds_in_all_groups": len(shared_all),
        "compounds_in_one_group_only": in_one,
        "pairwise_shared": {f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(groups) for b in groups[i + 1:]},
        "controls_per_group": {str(g): {"negcon": int(((batch == str(g)) & dmso).sum()),
                                        "poscon": int(((batch == str(g)) & (meta["split"] == "poscon").to_numpy()).sum())}
                               for g in groups},
        "gate_all_wells": gate(np.ones(len(meta), bool)) if len(groups) >= 2 else None,
        "gate_dmso_only": gate(dmso) if len(groups) >= 2 and dmso.sum() >= 2 * len(groups) else None,
    }
    for name, array in (("raw", raw), ("harmony", harmony)):
        evidence[f"replicate_retrieval_{name}"] = {
            "same_batch": quant.replicate_retrieval(meta, array, batch_key=batch_key, plate_types=plate_types, cross=False),
            "cross_batch": quant.replicate_retrieval(meta, array, batch_key=batch_key, plate_types=plate_types, cross=True)}
    return evidence


def control(prep: Path) -> dict:
    from shape.tools.batch_corrector import quant
    from benchmarks.batch_correction_JUMP.runners.prepare import load, sha256_file

    out = prep / CONTROL_DIR
    out_path = out / "CONTROL.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; move {out} aside to recompute")
    out.mkdir(exist_ok=True)
    summary_prep, meta, feats = load(prep)
    batch_key, plate_types = summary_prep["scenario"]["batch_key"], summary_prep["scenario"]["plate_types"]
    halves = quant.plate_halves(meta.loc[quant.split_mask(meta, "test")], N_HALVES, batch_key=batch_key, seed=HALF_SEED)
    summary = {"provenance": provenance(), "band": {"n_halves": N_HALVES, "seed": HALF_SEED, "unit": "plate",
                                                    "halves": halves}, "arms": {}}
    arrays = {}
    for arm in ("raw", "harmony", "scanorama"):
        entry = {"config": None}
        started = time.time()
        if arm == "raw":
            array = np.asarray(feats, dtype=np.float32)
            entry.update(tool_status="UNCORRECTED_INPUT", tool_calls=0)
        elif arm == "harmony":
            from shape.tools.batch_corrector.tool import DEFAULT_CONFIG

            corrected, telemetry = run_harmony(feats, meta[batch_key].to_numpy(), batch_key=batch_key)
            array = np.asarray(corrected, dtype=np.float32)
            entry.update(config=dict(DEFAULT_CONFIG), tool_status="CORRECTED_HARMONY",
                         telemetry=telemetry, tool_calls=1)
        else:
            array = np.asarray(run_scanorama(feats, meta[batch_key].to_numpy()), dtype=np.float32)
            entry.update(tool_status="CORRECTED_SCANORAMA", tool_calls=1)
        entry["tool_seconds"] = 0.0 if arm == "raw" else time.time() - started
        if arm != "raw":
            path = out / f"control_{arm}.npy"
            np.save(path, array)
            path.chmod(stat.S_IRUSR | stat.S_IRGRP)
            entry.update(array=path.name, array_sha256=sha256_file(path))
        entry.update(score_views(meta, array, halves, batch_key=batch_key, plate_types=plate_types))
        arrays[arm] = array
        summary["arms"][arm] = entry
        print(f"control {arm}: {entry.get('tool_status', 'uncorrected')} test overall {entry['test']['overall']} "
              f"all-wells overall {entry['all']['overall']} (tool {entry.get('tool_seconds', 0):.0f} s, "
              f"metrics {entry['eval_seconds']:.0f} s)", flush=True)
    started = time.time()
    summary["admission_evidence"] = admission_evidence(meta, arrays["raw"], arrays["harmony"],
                                                       batch_key=batch_key, plate_types=plate_types)
    print(f"admission evidence ({time.time() - started:.0f} s): "
          f"{json.dumps({k: v for k, v in summary['admission_evidence'].items() if not isinstance(v, dict)}, default=str)}",
          flush=True)
    out_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    out_path.chmod(stat.S_IRUSR | stat.S_IRGRP)
    return summary
