#!/usr/bin/env python
"""control.control() for big scenes: the three arms (raw / harmony / scanorama) in parallel processes, each arm's
scored entry cached to <prep>/shape_control/_entry_<arm>.json so a killed run resumes instead of re-scoring
(scoring is 1.4-2.3 h per arm on scenario_3 / scenario_5). Writes the same CONTROL.json as control.control()
(same provenance, arms roster and admission evidence), so run_suite accepts it. control.py itself is NOT edited:
its file hash is part of the provenance that scenario_2 / scenario_4 were stamped with.

    .venv/bin/python benchmarks/batch_correction_JUMP/runners/control_parallel.py <prep_dir>
"""
import json, os, stat, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ARMS = ("raw", "harmony", "scanorama")


def _entry(prep_str: str, arm: str) -> dict:
    from benchmarks.batch_correction_JUMP.runners import control as C
    from benchmarks.batch_correction_JUMP.runners.prepare import load, sha256_file
    from shape.tools.batch_corrector import quant

    prep = Path(prep_str)
    out = prep / C.CONTROL_DIR
    cache = out / f"_entry_{arm}.json"
    if cache.exists():
        print(f"control {arm}: cached entry reused", flush=True)
        return json.loads(cache.read_text())
    summary_prep, meta, feats = load(prep)
    batch_key, plate_types = summary_prep["scenario"]["batch_key"], summary_prep["scenario"]["plate_types"]
    halves = quant.plate_halves(meta.loc[quant.split_mask(meta, "test")], C.N_HALVES, batch_key=batch_key,
                                seed=C.HALF_SEED)
    entry = {"config": None}
    started = time.time()
    if arm == "raw":
        array = np.asarray(feats, dtype=np.float32)
        entry.update(tool_status="UNCORRECTED_INPUT", tool_calls=0)
    elif arm == "harmony":
        from shape.tools.batch_corrector.tool import DEFAULT_CONFIG
        corrected, telemetry = C.run_harmony(feats, meta[batch_key].to_numpy(), batch_key=batch_key)
        array = np.asarray(corrected, dtype=np.float32)
        entry.update(config=dict(DEFAULT_CONFIG), tool_status="CORRECTED_HARMONY", telemetry=telemetry, tool_calls=1)
    else:
        array = np.asarray(C.run_scanorama(feats, meta[batch_key].to_numpy()), dtype=np.float32)
        entry.update(tool_status="CORRECTED_SCANORAMA", tool_calls=1)
    entry["tool_seconds"] = 0.0 if arm == "raw" else time.time() - started
    if arm != "raw":
        path = out / f"control_{arm}.npy"
        if path.exists():                      # a killed earlier run left it read-only (0444)
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        np.save(path, array)
        path.chmod(stat.S_IRUSR | stat.S_IRGRP)
        entry.update(array=path.name, array_sha256=sha256_file(path))
    entry.update(C.score_views(meta, array, halves, batch_key=batch_key, plate_types=plate_types))
    print(f"control {arm}: {entry.get('tool_status')} test overall {entry['test']['overall']} "
          f"all-wells overall {entry['all']['overall']} (tool {entry['tool_seconds']:.0f} s, "
          f"metrics {entry['eval_seconds']:.0f} s)", flush=True)
    cache.write_text(json.dumps(entry, indent=1, default=str))
    return entry


def main(prep: Path) -> None:
    from benchmarks.batch_correction_JUMP.runners import control as C
    from benchmarks.batch_correction_JUMP.runners.prepare import load
    from shape.tools.batch_corrector import quant

    out = prep / C.CONTROL_DIR
    out_path = out / "CONTROL.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; move {out} aside to recompute")
    out.mkdir(exist_ok=True)
    summary_prep, meta, feats = load(prep)
    batch_key, plate_types = summary_prep["scenario"]["batch_key"], summary_prep["scenario"]["plate_types"]
    halves = quant.plate_halves(meta.loc[quant.split_mask(meta, "test")], C.N_HALVES, batch_key=batch_key,
                                seed=C.HALF_SEED)
    summary = {"provenance": C.provenance(), "band": {"n_halves": C.N_HALVES, "seed": C.HALF_SEED, "unit": "plate",
                                                      "halves": halves}, "arms": {}}
    with ProcessPoolExecutor(max_workers=len(ARMS), mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        futures = {arm: ex.submit(_entry, str(prep), arm) for arm in ARMS}
        for arm in ARMS:                                   # roster order must be raw, harmony, scanorama
            summary["arms"][arm] = futures[arm].result()
    raw = np.asarray(feats, dtype=np.float32)
    harmony = np.load(out / "control_harmony.npy")
    started = time.time()
    summary["admission_evidence"] = C.admission_evidence(meta, raw, harmony, batch_key=batch_key,
                                                         plate_types=plate_types)
    print(f"admission evidence ({time.time() - started:.0f} s)", flush=True)
    out_path.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    out_path.chmod(stat.S_IRUSR | stat.S_IRGRP)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
