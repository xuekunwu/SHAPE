"""Prepare stage: the Arevalo scenario's paper preprocessing and a frozen compound split.

The preprocessing is the authors' own code, vendored unchanged in `vendor/2023_Arevalo_NatComm_BatchCorrection/`
(github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection @ cd9bcf99, BSD-3; Nat Commun
15, 6516 (2024)): raw -> neg_stats -> variant_feats -> mad -> mad_int -> mad_int_featselect, the
chain of its `rules/processing.smk`. The vendored `preprocessing.metadata` downloads a commit-pinned
microscope table at import time; the staged copy is served instead.

<arevalo cache>/<scenario>/*.parquet   the stage outputs (resumable: a stage whose `.done` marker holds the current
                                       fingerprint -- vendored code, scenario conf, microscope table,
                                       library versions, raw inputs path -- is reused)
prep/
  features.npy        float32 (n_wells, n_features), prep row order
  meta.parquet        all metadata + `split` (val | test | negcon | poscon)
  PREPARE.json        scenario, sizes, split rule, artifact and array sha256
Split: treatment compounds (Metadata_PertType == trt) are ordered by sha256(f"{seed}:{id}"); the
first half is val (calibration: the only compound labels the SHAPE arm scores with), the rest test
(evaluation). Controls (negcon = DMSO, poscon) belong to neither split and appear in both views.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from shape.tools.batch_corrector.quant import DMSO, LABEL_KEY

VENDOR = Path(__file__).resolve().parent / "vendor" / "2023_Arevalo_NatComm_BatchCorrection"
MICROSCOPE_CONFIG_URL = ("https://raw.githubusercontent.com/jump-cellpainting/datasets/"
                         "181fa0dc96b0d68511b437cf75a712ec782576aa/metadata/microscope_config.csv")
PIPELINE = "mad_int_featselect"
PROTECTED = ("features.npy", "meta.parquet")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(compounds, seed: int = 0) -> dict:
    """Treatment compounds -> val | test (controls must not be passed in)."""
    ids = sorted({str(c) for c in compounds if str(c) != DMSO},
                 key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
    half = len(ids) // 2
    return {c: ("val" if i < half else "test") for i, c in enumerate(ids)}


def load_scenario(name: str) -> dict:
    payload = json.loads((VENDOR / "inputs" / "conf" / f"{name}.json").read_text(encoding="utf-8"))
    if payload.get("scenario") != name:
        raise SystemExit(f"scenario identity mismatch for {name}")
    return {k: payload[k] for k in ("scenario", "sources", "plate_types", "batch_key", "label_key")} | {"name": name}


def vendor_modules(microscope_config: Path) -> dict:
    import pandas as pd

    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    original = pd.read_csv

    def offline_read_csv(path, *args, **kwargs):
        return original(microscope_config if str(path) == MICROSCOPE_CONFIG_URL else path, *args, **kwargs)

    pd.read_csv = offline_read_csv
    try:
        return {name: importlib.import_module(f"preprocessing.{name}")
                for name in ("io", "stats", "normalize", "transform", "feature_selection")}
    finally:
        pd.read_csv = original


@contextmanager
def _cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def stage_fingerprint(scenario: dict, base: Path, microscope_config: Path) -> dict:
    """What a cached stage is computed from: the scenario, the raw inputs' location (not their content),
    the vendored code (preprocessing, pycytominer operations, the scenario conf), the libraries it runs
    on and the microscope table."""
    from importlib.metadata import version
    digest = hashlib.sha256()
    code = sorted(path for folder in ("preprocessing", "pycytominer") for path in (VENDOR / folder).rglob("*.py"))
    for path in [*code, VENDOR / "inputs" / "conf" / f"{scenario['name']}.json"]:
        digest.update(path.relative_to(VENDOR).as_posix().encode())
        digest.update(path.read_bytes())
    return {"scenario": scenario["name"], "raw_inputs": str((Path(base) / "inputs").resolve()),
            "vendor_sha256": digest.hexdigest(), "microscope_config_sha256": sha256_file(Path(microscope_config)),
            "packages": {name: version(name) for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn")}}


def preprocess(base: Path, out: Path, scenario: dict, modules: dict, fingerprint: dict) -> dict:
    """The vendored chain; `base` holds the authors' `inputs/` layout (profiles + metadata). A stage is
    reused only when its `.done` marker (written after the stage completes) holds `fingerprint`; once a
    stage is recomputed, every later stage is recomputed too."""
    out.mkdir(parents=True, exist_ok=True)
    p = {name: str((out / f"{name}.parquet").resolve())
         for name in ("raw", "neg_stats", "variant_feats", "mad", "mad_int", PIPELINE)}
    stages = (
        ("raw", lambda: modules["io"].write_parquet(list(scenario["sources"]), list(scenario["plate_types"]), p["raw"])),
        ("neg_stats", lambda: modules["stats"].compute_negcon_stats(p["raw"], p["neg_stats"])),
        ("variant_feats", lambda: modules["stats"].select_variant_features(p["raw"], p["neg_stats"], p["variant_feats"])),
        ("mad", lambda: modules["normalize"].mad(p["variant_feats"], p["neg_stats"], p["mad"])),
        ("mad_int", lambda: modules["transform"].rank_int(p["mad"], p["mad_int"])),
        (PIPELINE, lambda: modules["feature_selection"].select_features(p["mad_int"], p[PIPELINE])),
    )
    reused, recomputing = [], False
    with _cwd(base):
        for name, run in stages:
            marker = out / f"{name}.done"
            if (not recomputing and marker.exists() and Path(p[name]).exists()
                    and json.loads(marker.read_text(encoding="utf-8")) == fingerprint):
                reused.append(name)
                print(f"preprocessing stage {name} reused", flush=True)
                continue
            recomputing = True
            marker.unlink(missing_ok=True)
            run()
            marker.write_text(json.dumps(fingerprint), encoding="utf-8")
            print(f"preprocessing stage {name} done", flush=True)
    return {"protocol": f"Arevalo-2024-{PIPELINE}", "vendor_commit": "cd9bcf99240880a5c9f9858debf70e94f5b4c0f7",
            "fingerprint": fingerprint, "reused_stages": reused,
            "artifacts": {name: {"path": path, "sha256": sha256_file(Path(path))} for name, path in p.items()}}


def prepare(base: Path, prep: Path, *, scenario: str, microscope_config: Path, arevalo_cache: Path,
            seed: int = 0) -> dict:
    spec = load_scenario(scenario)
    missing = [source for source in spec["sources"]
               if not (Path(base) / "inputs" / source / "workspace" / "profiles").is_dir()]
    if missing:
        raise SystemExit(f"MISSING_INPUTS: {scenario} needs the profiles of {missing} under "
                         f"{Path(base) / 'inputs'}/<source>/workspace/profiles")
    modules = vendor_modules(microscope_config)
    stamp = preprocess(Path(base), Path(arevalo_cache) / scenario, spec, modules,
                       stage_fingerprint(spec, base, microscope_config))
    meta, feats, features = modules["io"].split_parquet(stamp["artifacts"][PIPELINE]["path"])
    feats = np.asarray(feats, dtype=np.float32)
    if not np.isfinite(feats).all():
        raise SystemExit("non-finite values in the preprocessed features")
    meta = meta.reset_index(drop=True)
    labels = meta[LABEL_KEY].astype(str)
    pert = meta["Metadata_PertType"].astype(str)
    if not set(pert) <= {"trt", "negcon", "poscon"}:
        raise SystemExit(f"unexpected Metadata_PertType values: {sorted(set(pert))}")
    if ((pert == "negcon") != (labels == DMSO)).any():
        raise SystemExit("negcon wells and DMSO labels disagree")
    split_of = assign_split(labels[pert == "trt"].unique(), seed)
    meta["split"] = [split_of[c] if t == "trt" else t for c, t in zip(labels, pert)]
    np.save(prep / "features.npy", feats)
    meta.to_parquet(prep / "meta.parquet")
    summary = {
        "scenario": spec, "preprocessing": stamp, "raw_inputs": str((Path(base) / "inputs").resolve()),
        "n_wells": int(len(meta)), "n_features": int(feats.shape[1]), "feature_names_head": list(features[:5]),
        "n_compounds": {s: int(labels[meta.split == s].nunique()) for s in ("val", "test", "poscon")},
        "n_wells_by_split": {s: int((meta.split == s).sum()) for s in ("val", "test", "negcon", "poscon")},
        "n_wells_by_plate_type": {str(k): int(v) for k, v in meta["Metadata_PlateType"].value_counts().items()},
        "n_wells_by_source": {str(k): int(v) for k, v in meta[spec["batch_key"]].value_counts().sort_index().items()},
        "n_plates": int(meta["Metadata_Plate"].nunique()), "n_batches": int(meta["Metadata_Batch"].nunique()),
        "split_rule": f"treatment compounds ordered by sha256('{seed}:<Metadata_JCP2022>'); first half val "
                      "(calibration), rest test (evaluation); controls (negcon, poscon) in neither",
        "sha256": {name: sha256_file(prep / name) for name in PROTECTED},
    }
    (prep / "PREPARE.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    for name in (*PROTECTED, "PREPARE.json"):
        (prep / name).chmod(stat.S_IRUSR | stat.S_IRGRP)
    print(json.dumps({k: v for k, v in summary.items() if k != "preprocessing"}, indent=1, default=str))
    return summary


def load(prep: Path):
    import pandas as pd

    summary = json.loads((prep / "PREPARE.json").read_text(encoding="utf-8"))
    changed = [n for n, h in summary["sha256"].items() if sha256_file(prep / n) != h]
    if changed:
        raise SystemExit(f"INTEGRITY_FAILED: {changed} changed since the prepare stage")
    return summary, pd.read_parquet(prep / "meta.parquet"), np.load(prep / "features.npy")
