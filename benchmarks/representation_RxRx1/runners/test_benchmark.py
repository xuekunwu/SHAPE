"""CPU tests of the RxRx1 representation benchmark on a synthetic task-116 prep.

- leak: no siRNA reaches any tool input (prepare's table, every analyzer call of control, the SHAPE arm and score);
- packed-array input: prepare writes no crop file; every tool input is the FULL images.npy (crop_array) with each
  cell's own array_row; a copy of the array at another path gives the same fingerprint; a changed array is refused;
- scorer parity: the well mAP and the paired band equal the task-116 metric (golden values computed with
  agentic_rxrx1/harness/metric.py @ 84ec172; compared directly when that checkout is importable);
- per_plate embed_only parity: a delivered per_plate state embeds its training cells exactly as the training call did
  (the real analyzer with a tiny stub backbone, on CPU);
- stubbed end-to-end: prepare -> control -> the SHAPE arm (RepresentationAgenticVerifier through StepVerifier, a
  scripted reasoner) -> score, with a stand-in analyzer; the provisional and the final decision.json;
- the external arms (external.py): the OpenPhenom adapter's input (channels, size, dtype, row order), its 256 px
  guard and its offline snapshot load (a stub model package), the CellProfiler-style masks (the seeded watershed) on
  synthetic crops and the raw filtered features (a stub cp_measure package on disk, also through the spawn pool), the
  native / matched post-processing (score.post_process), score with the external arms, both tables and their
  comparisons, a missing arm skipped and a stale or changed one refused, the stage's pilot / keep / refuse rules, and
  no label in any of their inputs;
- the LLM comparison arms (llm_arms.py, contract amendments A4, A5) with a scripted engine: llm (A5, no analyzer) a
  tiny feature module run in its own process on every cell in cells.csv order (no credential in its environment), the
  static check's rejections sent back and corrected, a crash and a gateway error retried then delivered, 3 failed
  attempts failing the run, invalid outputs (shape, d, dtype, d across chunks), a module stopped at the end of a tiny
  time budget; no llm prompt carries val_history, a val-proxy value, SHAPE's readouts text, a label, the CARD, the
  analyzer or metadata (react, react_arm.py on the shared runner, amendment A14: test_react.py); score with both runs
  (both tables: llm as an external feature arm, react raw in the
  native one; the A4 / A5 comparisons with llm vs cellprofiler; the efficiency rows), a failed run reported, a stale
  or wrong run refused;
- amendment A7: a move from a finished call without init_from resumes it (the ledgers' `init` record), the
  `chance` row of both tables, the `random` arm (no comparison, no change to any existing one) and its well mAP
  against chance on RxRx1-sized layouts;
- the per-arm score cache (score.ArmScores, stage score-arm): a cold-cache final equals the pre-cache score.py's
  RESULT.json / RESULT.md, a warm one recomputes nothing, a changed checkpoint or score.py recomputes, score_arm fills
  what final reads, and the scoring embedding without UMAP / Leiden equals the clustered call's.
The real stack (DINOv3, OpenPhenom, cp_measure, the reasoner) runs on the cluster.
"""
from __future__ import annotations

import errno
import hashlib
import inspect
import itertools
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
anndata = pytest.importorskip("anndata")

from benchmarks.representation_RxRx1.runners import control, prepare as prep_mod, score, shape_arm  # noqa: E402

SIRNA_TAG = "SIRNAX"                      # every fixture siRNA starts with it: a leak is a substring hit
HUB_116 = Path(os.environ.get("RXRX_116_HUB", "D:/1-Data/Code/Worktree/codebase/SSL/hub"))


# ---- a synthetic task-116 v3 prep ----------------------------------------------------------------------------

PLATES = {"HEPG2-01_1": "HEPG2-01", "HEPG2-02_1": "HEPG2-02", "HEPG2-03_1": "HEPG2-03"}   # the last is held out
WELLS = [f"B{w:02d}" for w in range(2, 12)]     # 10 wells per plate
VAL_WELLS, TRAINING_WELLS = WELLS[:4], WELLS[4:6]   # the rest of the train wells are unused by the run prep
SITES, CELLS = 2, 3


def _write_116_prep(base: Path) -> tuple[Path, Path, pd.DataFrame]:
    """FULL prep (images.npy + cells.csv) and its run prep (cells.csv with source_row), the task-116 layout."""
    rng = np.random.default_rng(0)
    rows, images = [], []
    for p, (plate, experiment) in enumerate(PLATES.items()):
        for w, well in enumerate(WELLS):
            split = "heldout" if p == 2 else ("val" if well in VAL_WELLS else "train")
            used = split != "train" or well in TRAINING_WELLS
            for s in range(1, SITES + 1):
                site = f"{plate}_{well}_{s}"
                for c in range(CELLS):
                    rows.append({"cell_id": f"{site}_n{c:04d}", "site_id": site, "well_unit": f"{plate}/{well}",
                                 "sirna": f"{SIRNA_TAG}{w % 5:03d}", "plate": plate, "experiment": experiment,
                                 "split": split})
                    # plate-specific intensity; the cells the tool never sees are brighter (another plate statistic)
                    level = 40 + 35 * p + 12 * (w % 5) + (0 if used else 60)
                    images.append(np.clip(rng.normal(level, 12, size=(6, 64, 64)), 0, 255).astype(np.uint8))
    full = base / "prep_full"
    full.mkdir(parents=True)
    cells = pd.DataFrame(rows)
    cells.insert(0, "row", np.arange(len(cells)))
    cells.to_csv(full / "cells.csv", index=False)
    np.save(full / "images.npy", np.stack(images))
    training = cells.well_unit.str.split("/").str[1].isin(TRAINING_WELLS) & (cells.split == "train")
    keep = cells[(cells.split == "val") | training].copy()
    keep.insert(keep.columns.get_loc("split") + 1, "source_row", keep["row"])
    keep["row"] = np.arange(len(keep))
    run = base / "prep_run"
    run.mkdir()
    keep.to_csv(run / "cells.csv", index=False)
    return full, run, cells


@pytest.fixture()
def prepared(tmp_path):
    full, run, source = _write_116_prep(tmp_path / "agentic_ssl")
    root = tmp_path / "shape_vrr" / "HEPG2"
    summary = prep_mod.prepare(full, run, root)
    return root, summary, source


# ---- a stand-in analyzer (no torch model): the tool's result contract on synthetic embeddings -----------------

class StubAnalyzer:
    """Cell_State_Analyzer_Tool's execute() signature and result keys. The embedding is a fixed projection of the
    crop pixels (the crop_array rows); a training call pulls each crop toward the mean of its well's crops
    (label-free metadata) by its state's `alpha`, which a resume state carries on, then squashes it; embed_only
    reproduces the state's embedding."""

    calls: list = []
    fail_zero_shot = False

    def execute(self, **kw):
        type(self).calls.append(kw)
        if kw.get("adaptation_depth") == "zero_shot" and type(self).fail_zero_shot:
            raise RuntimeError("stub: the backbone did not load")
        images, meta = np.load(kw["crop_array"], mmap_mode="r"), kw["cell_metadata"]
        x = np.stack([images[m["array_row"]].astype(np.float64).reshape(6, 4, 16, 4, 16).mean((2, 4)).ravel() / 255.0
                      for m in meta])
        x = (x - 0.4) @ np.random.default_rng(1).normal(size=(x.shape[1], 32)) / np.sqrt(x.shape[1])
        parent = json.loads(Path(kw["init_checkpoint"]).read_text()) if kw.get("init_checkpoint") else {"alpha": 0.0}
        trains = kw.get("adaptation_depth") != "zero_shot" and not kw.get("embed_only")
        alpha = parent["alpha"]
        if trains:
            alpha = min(0.95, alpha + (0.5 if kw.get("view_pairing") == "same_well" else 0.25)
                        + 0.01 * int(kw.get("seed") or 0))
        wells = np.array([m["well"] for m in meta])
        means = {w: x[wells == w].mean(axis=0) for w in set(wells)}
        # the nonlinearity moves the well profiles too (a pure shrink toward the well mean would leave them in place)
        vectors = np.tanh((1 + alpha) * ((1 - alpha) * x + alpha * np.stack([means[w] for w in wells])))
        out = Path(kw["query_cache_dir"]) / "cell_state_analysis" / (kw.get("output_subdir") or "")
        out.mkdir(parents=True, exist_ok=True)
        ids = [m["cell_id"] for m in meta]
        obs = pd.DataFrame({"cell_id": ids}, index=ids)
        split_key = kw.get("split_key")
        if split_key and not kw.get("embed_only"):
            obs["split"] = [m[split_key] for m in meta]
        adata = anndata.AnnData(X=vectors.astype(np.float32), obs=obs)
        adata.write_h5ad(out / "embeddings.h5ad")
        resume = None
        if trains and split_key:
            resume = out / "resume_best.pt"
            resume.write_text(json.dumps({"alpha": alpha}))
        split = [m.get(split_key) for m in meta] if split_key else []
        by_proxy = "val" in split                   # the tool's rule: no val crop, no proxy, best by training loss
        return {"adata_path": str(out / "embeddings.h5ad"), "resume_best": str(resume) if resume else None,
                "resume_end": str(resume) if resume else None, "seconds": 2.0,
                "split": {"n_train": split.count("train"), "n_val": split.count("val"),
                          "n_val_wells": len({m["well"] for m, s in zip(meta, split) if s == "val"})},
                "training": {"epochs_trained": 1 if trains else 0, "best_epoch": 1 if trains else None,
                             "best_by": "val_proxy" if by_proxy else "train_loss", "batch_size": 16,
                             "trainable_blocks": 12 if trains else 0, "train_seconds": 1.0},
                # the real tool: a training call's evaluations; an untrained call with val crops, epoch 0 (no loss)
                "val_history": ([{"epoch": 1, "sibling_site_map": 0.5, "eval_loss": 1.0}] if trains and by_proxy else
                                [{"epoch": 0, "sibling_site_map": 0.5}] if by_proxy and not kw.get("embed_only")
                                else []),
                "training_config": {"alpha": alpha}}


def _stub_signature():
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    StubAnalyzer.execute.__signature__ = inspect.signature(Cell_State_Analyzer_Tool.execute)


@pytest.fixture()
def stub(monkeypatch):
    _stub_signature()
    StubAnalyzer.calls = []
    StubAnalyzer.fail_zero_shot = False
    monkeypatch.setattr(control, "analyzer", lambda: StubAnalyzer())
    monkeypatch.setenv("SHAPE_REQUIRE_AGENTIC", "")       # recorded first: the SHAPE arm with an engine sets it to 1
    monkeypatch.delenv("SHAPE_REQUIRE_AGENTIC")           # in os.environ; the teardown restores the session's value
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    return StubAnalyzer


def _leaks(value) -> list[str]:
    text = json.dumps(value, default=str)
    return [SIRNA_TAG] if SIRNA_TAG in text or '"sirna"' in text.lower() else []


# ---- prepare ---------------------------------------------------------------------------------------------------

def test_prepare_writes_the_tool_table_without_labels(prepared):
    root, summary, source = prepared
    assert sorted(f.name for f in root.rglob("*") if f.is_file()) == ["PREPARE.json", "cells.csv", "labels.csv"]
    cells = pd.read_csv(root / "prep" / "cells.csv", dtype=str, keep_default_na=False)
    assert tuple(cells.columns) == prep_mod.COLUMNS and "sirna" not in cells
    assert not _leaks(cells.to_dict("records"))
    counts = cells.split.value_counts().to_dict()
    n_site = SITES * CELLS
    assert counts == {"val": 2 * 4 * n_site, "train": 2 * 2 * n_site, "unused": 2 * 4 * n_site, "heldout": 10 * n_site}
    labels = pd.read_csv(root / "private" / "labels.csv", dtype=str)
    assert labels.set_index("cell_id").sirna.to_dict() == source.set_index("cell_id").sirna.to_dict()
    assert summary["tool_input"]["min_val_wells_per_plate"] == 4
    crop_array, meta = prep_mod.tool_input(summary, cells)
    assert len(meta) == summary["tool_input"]["cells"] and {m["split"] for m in meta} == {"train", "val"}
    assert not _leaks(meta) and "private" not in crop_array


def test_cli_prepare_stage(tmp_path, monkeypatch, capsys):
    from benchmarks.representation_RxRx1 import run_rxrx1
    from benchmarks.representation_RxRx1.runners.config import PREPS

    full, run, _ = _write_116_prep(tmp_path / "data")
    for source, name in zip((full, run), PREPS["HEPG2"]):
        source.rename(source.with_name(name))
    args = ["prepare", "--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out")]
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    with pytest.raises(SystemExit, match="pass --dinov3-repo and --weights"):
        run_rxrx1.main(args)
    assert (tmp_path / "out" / "HEPG2" / "prep" / "PREPARE.json").exists()
    with pytest.raises(SystemExit, match="pass --dinov3-repo"):
        run_rxrx1.main(args)                        # the prep is kept, not rebuilt
    assert "prep exists, kept" in capsys.readouterr().out
    monkeypatch.setenv("RXRX_EXPECTED_COMMIT", "0" * 40)
    with pytest.raises(SystemExit, match="STALE_CODE"):
        run_rxrx1.main(args)


def test_cli_prep_set_full_v4(tmp_path, monkeypatch, capsys):
    """--prep-set full_v4: the full-set prep of the cell type (U2OS among them) into <output root>/<cell type>_full_v4/;
    the v3 default is unchanged; a cell type without a prep in the prep set is refused before any work."""
    from benchmarks.representation_RxRx1 import run_rxrx1
    from benchmarks.representation_RxRx1.runners.config import FULL_V4_PREPS

    full, run, _ = _write_116_prep(tmp_path / "data")
    for source, name in zip((full, run), FULL_V4_PREPS["U2OS"]):
        source.rename(source.with_name(name))
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    args = ["prepare", "--prep-set", "full_v4", "--cell-type", "U2OS", "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="pass --dinov3-repo and --weights"):
        run_rxrx1.main(args)
    summary = json.loads((tmp_path / "out" / "U2OS_full_v4" / "prep" / "PREPARE.json").read_text())
    assert summary["sources"]["full"]["path"] == str(tmp_path / "data" / FULL_V4_PREPS["U2OS"][0])
    assert not (tmp_path / "out" / "U2OS").exists()
    for argv in (["control", "--cell-type", "U2OS"], ["control", "--prep-set", "v3", "--cell-type", "U2OS"],
                 ["control", "--prep-set", "v5"]):
        with pytest.raises(SystemExit) as raised:
            run_rxrx1.main([*argv, "--output-root", str(tmp_path / "out")])
        assert raised.value.code == 2
    assert "has no v3 prep" in capsys.readouterr().err


# ---- the packed-array input ----------------------------------------------------------------------------------

def test_the_tool_input_is_the_full_array_at_each_cells_row(prepared):
    root, summary, source = prepared
    _, cells = prep_mod.load(root)
    crop_array, meta = prep_mod.tool_input(summary, cells, splits=prep_mod.SPLITS)
    assert Path(crop_array).is_absolute() and Path(crop_array).name == "images.npy"
    assert summary["crop_array"]["shape"] == [len(source), *prep_mod.CROP_SHAPE]
    assert summary["crop_array"]["sha256"] == summary["sources"]["full"]["images_sha256"]
    row_of = dict(zip(source.cell_id, source.row))
    assert [m["array_row"] for m in meta] == [row_of[m["cell_id"]] for m in meta]
    assert all(type(m["array_row"]) is int for m in meta)
    assert crop_array == str((root.parents[1] / "agentic_ssl" / "prep_full" / "images.npy").absolute())


def test_a_copy_of_the_array_at_another_path_gives_the_same_fingerprint(prepared, tmp_path):
    pytest.importorskip("torch")
    from shape.tools.cell_state_analyzer.tool import array_crops, embedding_fingerprint

    root, summary, _ = prepared
    _, cells = prep_mod.load(root)
    crop_array, meta = prep_mod.tool_input(summary, cells)
    other = tmp_path / "node_b" / "mnt" / "images.npy"          # the same file under another mount
    other.parent.mkdir(parents=True)
    shutil.copyfile(crop_array, other)
    params = {"adaptation_depth": "zero_shot", "split_key": "split"}
    assert (embedding_fingerprint(params, array_crops(crop_array, meta))
            == embedding_fingerprint(params, array_crops(str(other), meta)))


def test_a_changed_array_or_an_old_prep_is_refused(prepared):
    root, summary, _ = prepared
    images = np.load(summary["crop_array"]["path"])
    images[0, 0, 0, 0] ^= 1
    np.save(summary["crop_array"]["path"], images)
    with pytest.raises(SystemExit, match="SOURCE_CHANGED"):
        prep_mod.load(root)
    path = root / "prep" / "PREPARE.json"
    old = json.loads(path.read_text(encoding="utf-8"))
    old.pop("crop_array")
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="NOT_CURRENT"):
        prep_mod.load(root)


def test_the_images_hash_runs_once_per_file_state(prepared, monkeypatch):
    """`load` hashes images.npy in full once per file state (prep/CROP_ARRAY_VERIFIED.json): a repeated load skips the
    hash, a touch hashes again, a rewrite of the same size is refused (and never stamped)."""
    root, summary, _ = prepared
    path = Path(summary["crop_array"]["path"])
    hashed, real = [], prep_mod.sha256_file

    def counting(p):
        hashed.append(Path(p))
        return real(p)

    monkeypatch.setattr(prep_mod, "sha256_file", counting)
    prep_mod.load(root)
    prep_mod.load(root)
    assert hashed.count(path) == 1
    stamp = json.loads((root / "prep" / prep_mod.VERIFIED).read_text(encoding="utf-8"))
    assert (stamp["path"], stamp["sha256"], stamp["bytes"]) == (str(path), summary["crop_array"]["sha256"],
                                                                path.stat().st_size)
    state = path.stat()
    os.utime(path, ns=(state.st_atime_ns, state.st_mtime_ns + 10 ** 9))      # the same bytes, another state
    prep_mod.load(root)
    prep_mod.load(root)
    assert hashed.count(path) == 2
    images = np.load(path)
    images[0, 0, 0, 0] ^= 1
    np.save(path, images)
    for _ in range(2):
        with pytest.raises(SystemExit, match="SOURCE_CHANGED"):
            prep_mod.load(root)
    assert hashed.count(path) == 4


def test_outputs_never_go_under_the_repository(tmp_path):
    with pytest.raises(SystemExit, match="inside the repository"):
        prep_mod.outside_repo(prep_mod.REPO / "benchmarks" / "representation_RxRx1" / "out")
    assert prep_mod.outside_repo(tmp_path) == tmp_path


# ---- transient filesystem failures ----------------------------------------------------------------------------

def test_fs_retry_survives_a_lustre_eviction(monkeypatch):
    monkeypatch.setattr(control, "_sleep", lambda s: None)
    tries = []

    def flaky():
        tries.append(1)
        if len(tries) == 1:
            raise BrokenPipeError(108, "Cannot send after transport endpoint shutdown")
        return "ok"

    assert control.fs_retry(flaky) == "ok" and len(tries) == 2
    with pytest.raises(PermissionError):
        control.fs_retry(lambda: (_ for _ in ()).throw(PermissionError(errno.EACCES, "no")))


def test_a_status_write_retries_an_io_error(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "_sleep", lambda s: None)
    replace, tries = Path.replace, []

    def flaky(self, target):
        tries.append(1)
        if len(tries) == 1:
            raise OSError(errno.EIO, "Input/output error")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    control.write_json(tmp_path / "decision.json", {"use": "frozen"})
    assert len(tries) == 2 and json.loads((tmp_path / "decision.json").read_text()) == {"use": "frozen"}


def test_build_assets_retries_and_never_leaves_assets_without_its_manifest(tmp_path, monkeypatch):
    from benchmarks.full_HCMA.harness import assets

    monkeypatch.setattr(control, "_sleep", lambda s: None)
    copies, renames, rename = [], [], Path.rename

    def copy_dinov3(repo, commit, dest):
        copies.append(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if len(copies) == 1:
            raise OSError(errno.EIO, "Input/output error")
        return {"path": str(dest), "commit": commit, "n_files": 1, "tree_sha256": "t"}

    def place_hf_weights(weights, hf_home, expected):
        target = hf_home / "weights.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"w")
        return {"path": str(target), "sha256": "ab" * 32}

    def checked_rename(self, target):
        renames.append((self / "ASSETS.json").exists())
        return rename(self, target)

    monkeypatch.setattr(assets, "copy_dinov3", copy_dinov3)
    monkeypatch.setattr(assets, "place_hf_weights", place_hf_weights)
    monkeypatch.setattr(assets, "check_hf_offline", lambda hf_home, path: str(path))
    monkeypatch.setattr(Path, "rename", checked_rename)
    path = prep_mod.build_assets(tmp_path, dinov3_repo=tmp_path / "dinov3", weights=tmp_path / "weights.pth")
    assert len(copies) == 2 and renames == [True]                   # the EIO is retried; the manifest precedes assets/
    assert path == tmp_path / "assets" / "ASSETS.json" and not (tmp_path / "assets.partial").exists()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in (manifest["dinov3_hub"], manifest["dinov3_weights"]):
        assert "assets.partial" not in record["path"] and Path(record["path"]).exists()


def test_ledger_write_failure_does_not_stop_the_search(tmp_path, monkeypatch):
    def broken(path, record):
        raise BrokenPipeError(108, "Cannot send after transport endpoint shutdown")

    monkeypatch.setattr(control, "append_jsonl", broken)
    log = shape_arm.AgentLog(tmp_path / "agent_log.jsonl")
    log("system", text="x")
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    verifier = RepresentationAgenticVerifier()      # the ledger names the calls with verifier.label
    verifier._names = {0: "input", 1: "agent_1"}
    ledger = shape_arm.Ledger(tmp_path, verifier, log, lambda ledger: None, set())
    ledger({"event": "wave", "index": None, "size": 1})
    ledger({"event": "trial_started", "index": 1, "parent": 0, "params": {}, "param_delta": {},
            "expected_effect": {}, "notes": ""})
    assert len(log.write_errors) == 2 and "errno 108" in log.write_errors[0]["error"].lower()
    assert len(ledger.write_errors) == 1 and ledger.write_errors[0]["what"] == "trial_started 1"


# ---- scorer parity with the task-116 metric --------------------------------------------------------------------

def _metric_fixture():
    rng = np.random.default_rng(7)
    wells = [f"P{p}/W{w:02d}" for p in range(3) for w in range(8)]
    sirna_of_well = {w: f"s{i % 4}" for i, w in enumerate(wells)}
    well = np.array([w for w in wells for _ in range(5)])
    group = np.array([sirna_of_well[w] for w in well])
    centres = {s: rng.normal(size=8) for s in sorted(set(group))}
    a = np.stack([0.5 * centres[g] + rng.normal(size=8) for g in group])
    b = np.stack([1.0 * centres[g] + rng.normal(size=8) for g in group])
    return a, b, group, well


def test_well_map_matches_the_116_metric_golden_values():
    a, b, group, well = _metric_fixture()
    ra, rb = score.well_map(a, group, well), score.well_map(b, group, well)
    # agentic_rxrx1/harness/metric.py @ 84ec172: score_level(x, group, well, well, "well"), paired_band(ra, rb)
    assert ra["value"] == pytest.approx(0.6457742534999736, rel=1e-12)
    assert ra["chance"] == pytest.approx(0.21739130434782608, rel=1e-12)
    assert rb["value"] == pytest.approx(0.9443268999518999, rel=1e-12)
    assert ra["per_unit"]["P0/W00"] == pytest.approx(0.8711111111111111, rel=1e-12)
    assert ra["per_unit"]["P2/W07"] == pytest.approx(1.0, rel=1e-12)
    band = score.paired_band(ra["per_unit"], rb["per_unit"], ra["unit_group"])
    assert band["delta"] == pytest.approx(0.2985526464519263, rel=1e-12)
    assert band["band_2se"] == pytest.approx(0.08387363919319406, rel=1e-12)
    assert (band["n_units_paired"], band["significant"]) == (24, True)
    assert score.verdict(band) == "SHAPE_BEATS_DEFAULT"


@pytest.mark.skipif(not (HUB_116 / "agentic_rxrx1" / "harness" / "metric.py").exists(),
                    reason="the task-116 checkout is not on this machine (RXRX_116_HUB)")
def test_well_map_equals_the_116_metric_bitwise(monkeypatch):
    monkeypatch.syspath_prepend(str(HUB_116))
    from agentic_rxrx1.harness import metric

    a, b, group, well = _metric_fixture()
    for x in (a, b):
        ours, theirs = score.well_map(x, group, well), metric.score_level(x, group, well, well, "well")
        theirs.pop("level")
        assert ours == theirs
    ra, rb = score.well_map(a, group, well), score.well_map(b, group, well)
    assert (score.paired_band(ra["per_unit"], rb["per_unit"], ra["unit_group"])
            == metric.paired_band(ra["per_unit"], rb["per_unit"], ra["unit_group"]))


# ---- the stubbed end-to-end ------------------------------------------------------------------------------------

BOOKKEEPING = {"cell_crops", "crop_array", "cell_metadata", "query_cache_dir", "output_subdir", "max_minutes"}


def _effective(kwargs: dict) -> dict:
    """A call's kwargs with every omitted one at its execute() default, without the bookkeeping kwargs."""
    defaults = {name: p.default for name, p in inspect.signature(StubAnalyzer().execute).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return {k: v for k, v in {**defaults, **kwargs}.items() if k not in BOOKKEEPING}


def test_end_to_end_with_a_stand_in_analyzer(prepared, stub, tmp_path, monkeypatch):
    root, summary, _ = prepared
    crop_array = summary["crop_array"]["path"]
    ctl = control.control(root, seeds=(0, 1), split_key="split")
    assert ctl["complete"] and set(ctl["arms"]) == {"frozen", "dinov3", "dinov3_s1"}
    assert ctl["arms"]["frozen"]["kwargs"] == {"adaptation_depth": "zero_shot", "split_key": "split"}
    cap = json.loads((root / "TRAINING_BUDGET.json").read_text())["max_epochs"]
    budget = control.default_epochs(cap)
    for seed in (0, 1):            # A23: E x the round budget, an explicit cap (no extension), no epoch floor
        assert ctl["arms"][control.dinov3_arm(seed)]["kwargs"] == {"seed": seed, "split_key": "split",
                                                                   "max_epochs": budget, "min_epochs": 1}
    assert [kw.get("max_epochs") for kw in stub.calls] == [None, budget, budget]
    n_control = len(stub.calls)

    decisions = []
    write_json = control.write_json

    def recording(path, payload):
        if Path(path).name == "decision.json":
            decisions.append(json.loads(json.dumps(payload, default=str)))
        write_json(path, payload)

    monkeypatch.setattr(control, "write_json", recording)
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "smoke"
    engine = _shape_engine(3)
    final = shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=3, engine=engine)
    arm_calls = stub.calls[n_control:]

    # trial 0: the frozen backbone with the task's split, in ssl_calls/input
    trial0 = arm_calls[0]
    assert (trial0["adaptation_depth"], trial0["split_key"], trial0["output_subdir"]) == ("zero_shot", "split",
                                                                                          "ssl_calls/input")
    # every call the agent's loop made reached the tool through SHAPE's managed invocation: init_from never does,
    # a continuation resumes its parent's resume_best
    assert all("init_from" not in kw for kw in arm_calls)
    best = {kw["output_subdir"]: str(Path(kw["query_cache_dir"]) / "cell_state_analysis" / kw["output_subdir"]
                                     / "resume_best.pt") for kw in arm_calls}
    resumed = [kw for kw in arm_calls if kw.get("init_checkpoint")]
    assert resumed and all(kw["init_checkpoint"] in best.values() for kw in resumed)
    # the agent's round-1 move from input (the expert start's round 1, A21 / A24) is the dinov3 call at the expert
    # levers but for its epoch budget (A23: E x the rounds, no floor) and its input (the train wells only)
    start, _ = shape_arm.EXPERT_CONFIG
    expert_move = arm_calls[1]
    assert _effective(expert_move) == _effective({**ctl["arms"]["dinov3"]["kwargs"], **start, "max_epochs": cap,
                                                  "min_epochs": None})

    # decision.json: provisional after every round, then the delivery
    assert len(decisions) >= 2 and all(d["provisional"] for d in decisions[:-1])
    assert decisions[-1]["provisional"] is False and decisions[-1] == json.loads(
        (run_dir / "submission" / "decision.json").read_text())
    assert final["use"] == "frozen" or final["checkpoint"] in best.values()
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    assert meta["provenance"]["quant_sha256"] and meta["shape_file"].startswith(str(prep_mod.REPO))
    ledger = score.moves(run_dir)
    starts = [r for r in jsonl(run_dir / "logs" / "tool_calls.jsonl") if r["event"] == "start"]
    assert [r["init"] for r in starts] == ["fresh", "resumed: agent_1", "resumed: agent_2"]                # A7
    # the scripted agent's calls: the expert start, then its two training moves; every script reply was asked
    assert set(ledger) == {"agent_1", "agent_2", "agent_3"} and len(arm_calls) == 4      # trial 0 + 3 calls
    assert engine.replies == []
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert agent["write_errors"] == [] and agent["model"] == "stub" and meta["model"] == "stub"

    assert meta["crop_array"] == crop_array and ctl["arms"]["frozen"]["crop_array"] == crop_array
    result = score.final(root, run_dir)
    assert set(result["reading"]) == {"seen", "heldout"}
    assert result["decision"]["kind"] == "final" and result["decision"]["use"] == final["use"]
    points = {p["call"]: p for p in result["diagnostic"]["points"]}
    assert set(points) == {"input", *ledger}
    for call, point in points.items():
        if call != "input":
            assert point["parent"] == ledger[call]["parent"] and "beats" in point["proxy_vs_frozen"]
            assert point["proxy_agrees_with_label"] in (True, False, None)
    if final["use"] != "frozen":                    # the delivered state, not frozen, was embedded and scored
        assert result["tables"]["shape"]["seen"]["value"] != result["tables"]["frozen"]["seen"]["value"]
    seen = result["groups"]["seen"]
    assert {"shape_vs_default_mean", "shape_vs_best_default_seed", "shape_vs_frozen",
            "default_mean_vs_frozen"} <= set(seen["comparisons"])
    assert (run_dir / "RESULT.md").read_text().startswith("# Result HEPG2")

    # no siRNA in any tool input: control, the SHAPE arm and the scorer's embeddings
    assert len(stub.calls) > len(arm_calls) + n_control
    for kw in stub.calls:
        assert not _leaks({k: v for k, v in kw.items()}), kw.get("output_subdir")
        assert "cell_crops" not in kw and kw["crop_array"] == crop_array and "private" not in kw["crop_array"]
        assert all(type(m["array_row"]) is int for m in kw["cell_metadata"])
    score_calls = stub.calls[n_control + len(arm_calls):]
    assert all("split" not in m for kw in score_calls for m in kw["cell_metadata"])   # embed_only reads no split


def test_a_frozen_only_control_scores_shape_against_frozen(prepared, stub, tmp_path):
    root, _, _ = prepared
    ctl = control.control(root, seeds=(), split_key="split")          # CONTROL_SEEDS=none
    assert ctl["complete"] and set(ctl["arms"]) == {"frozen"}
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "frozen_only"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    result = score.final(root, run_dir)
    assert result["default_arms"] == [] and set(result["reading_vs_frozen"]) == {"seen", "heldout"}
    assert all(v == "UNSCOREABLE" for v in result["reading"].values())
    seen = result["groups"]["seen"]
    assert seen["default_mean"] is None and seen["best_default_seed"] is None
    assert "shape_vs_frozen" in seen["comparisons"]
    assert not any("default" in name for name in seen["comparisons"])
    assert "no default arm: shape vs frozen only" in (run_dir / "RESULT.md").read_text().splitlines()[0]


def test_the_run_seed_reaches_every_shape_call_and_is_recorded(prepared, stub, tmp_path):
    """A18: shape_arm.run(seed=k) passes seed k to every analyzer call of the arm (trial 0 included) and records it
    in RUN_META.json; the default is 0; the control's dinov3 is untouched."""
    root, _, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    for seed in (0, 3):
        before = len(stub.calls)
        run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / f"seed{seed}"
        kwargs = {"seed": seed} if seed else {}
        shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2), **kwargs)
        calls = stub.calls[before:]
        assert len(calls) >= 2 and all(kw["seed"] == seed for kw in calls)
        assert json.loads((run_dir / "RUN_META.json").read_text(encoding="utf-8"))["seed"] == seed
    ctl = json.loads((root / control.CONTROL_DIR / "CONTROL.json").read_text(encoding="utf-8"))
    assert set(ctl["arms"]) == {"frozen", "dinov3"}


def test_result_has_a_chance_row_and_no_random_arm(prepared, stub, tmp_path):
    """Amendment A7 reporting addition: a `chance` row per group in both tables (the exact expected AP of a random
    ranking, identical across arms; `base_rate` = n_pos / n_cand kept in RESULT.json only); the empirical `random`
    arm is no longer scored (user decision 2026-10-11)."""
    from benchmarks.representation_RxRx1.runners.prepare import load

    root, _, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "chance"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    result = score.final(root, run_dir)
    summary, cells = load(root)
    columns = [*score.GROUPS, *score.AUDIT_GROUPS]
    text = (run_dir / "RESULT.md").read_text(encoding="utf-8")
    text = text.split("\n\n## Secondary metrics")[0]       # the well mAP tables (A9's section has rows of its own)
    for table in (result, result["matched"]):
        assert "random" not in table["tables"]
        for g in columns:
            chance, base_rate = table["groups"][g]["chance"], table["groups"][g]["base_rate"]
            assert chance is not None and base_rate is not None and 0 < base_rate < chance < 1, g
            assert all(t[g]["chance"] == chance for t in table["tables"].values()), g     # labels and layout only
            assert all(t[g]["base_rate"] == base_rate for t in table["tables"].values()), g
            assert "random" not in table["groups"][g]["values"]
            assert not any("random" in name for name in table["groups"][g]["comparisons"]), g
        assert ("| chance | " + " | ".join(f"{table['groups'][g]['chance']:.4f}" for g in columns) + " |") in text
    assert result["groups"]["seen"]["chance"] == result["matched"]["groups"]["seen"]["chance"]
    assert text.count("\n| random | ") == 0 and text.count("\n| chance | ") == 2 and "| base_rate |" not in text
    # the well mAP's own chance (n_pos / n_cand) is kept as base_rate, unchanged
    seen_cells = np.isin(cells.split.to_numpy(), score.GROUPS["seen"])
    labels = score.read_labels(root, cells, summary)
    own = score.well_map(np.random.default_rng(0).normal(size=(int(seen_cells.sum()), 4)), labels[seen_cells],
                         cells.well.to_numpy()[seen_cells])
    assert result["groups"]["seen"]["base_rate"] == own["chance"]
    assert "random_arm" not in result
    vectors = score.random_vectors(len(cells))
    assert vectors.shape == (len(cells), 384)
    np.testing.assert_array_equal(vectors, np.random.default_rng(0).standard_normal((len(cells), 384)))
    # adding the random arm changes no existing comparison, value, default mean or best seed
    sirna = score.read_labels(root, cells, summary)
    rng = np.random.default_rng(1)
    base_vectors = rng.normal(size=(len(cells), 8))
    tables = {arm: score.evaluate(base_vectors + shift * rng.normal(size=base_vectors.shape), cells, sirna)
              for arm, shift in (("frozen", 0.5), ("dinov3", 0.4), ("shape", 0.3))}
    without = score.compare(tables, ["dinov3"])
    with_random = score.compare({**tables, "random": score.evaluate(vectors, cells, sirna)}, ["dinov3"])
    for g in columns:
        assert with_random[g]["comparisons"] == without[g]["comparisons"], g
        assert {k: v for k, v in with_random[g]["values"].items() if k != "random"} == without[g]["values"]
        keys = ("default_mean", "best_default_seed", "chance", "base_rate")
        assert [with_random[g][k] for k in keys] == [without[g][k] for k in keys], g
    with pytest.raises(SystemExit, match="CHANCE_MISMATCH"):
        score.group_chance({"a": {"chance": 0.1}, "b": {"chance": 0.2}}, "seen")
    with pytest.raises(SystemExit, match="CHANCE_MISMATCH: seen: the arms' base_rate"):
        score.group_chance({"a": {"base_rate": 0.1}, "b": {"base_rate": 0.2}}, "seen", "base_rate")


@pytest.mark.parametrize("n_cand, n_pos", [(1, 1), (2, 1), (5, 1), (5, 2), (5, 5), (6, 3), (8, 2), (9, 4)])
def test_the_exact_random_ranking_ap_equals_brute_force_enumeration(n_cand, n_pos):
    """score.expected_random_ap against the mean AP over every placement of the positives in a ranking of n_cand (all
    rankings are equally likely, so every set of positive ranks is too)."""
    aps = []
    for ranks in itertools.combinations(range(1, n_cand + 1), n_pos):
        aps.append(sum((hit + 1) / rank for hit, rank in enumerate(ranks)) / n_pos)
    assert score.expected_random_ap(n_pos, n_cand) == pytest.approx(sum(aps) / len(aps), rel=1e-12, abs=1e-15)
    if n_pos < n_cand:
        assert score.expected_random_ap(n_pos, n_cand) > n_pos / n_cand              # above the base rate


def test_the_exact_random_ranking_ap_refuses_a_query_without_positives():
    for n_pos, n_cand in ((0, 5), (6, 5)):
        with pytest.raises(ValueError, match="must lie in"):
            score.expected_random_ap(n_pos, n_cand)


@pytest.mark.parametrize("wells_per_sirna, tolerance", [(36, 0.002), (8, 0.015)],
                         ids=["seen_like_1080_wells", "heldout_like_240_wells"])
def test_the_random_arm_well_map_matches_the_exact_chance(wells_per_sirna, tolerance):
    """The random arm's well mAP on RxRx1-sized layouts (30 siRNA; seen ~36 wells each, heldout / run_prep ~7-8, as in
    the HEPG2 prep), seed 0 as score uses it, matches `chance` (the exact expected AP of a random ranking, averaged
    like the well mAP) within 5 seed-to-seed SDs (measured over 30 seeds: SD 0.0004 at 36 wells per siRNA, 0.003 at
    8). `base_rate` (n_pos / n_cand) lies below it, by ~0.006 and ~0.02."""
    n_sirna, cells_per_well = 30, 4
    n_wells = n_sirna * wells_per_sirna
    well = np.repeat([f"w{i:04d}" for i in range(n_wells)], cells_per_well)
    group = np.repeat([f"s{i % n_sirna:02d}" for i in range(n_wells)], cells_per_well)
    result = score.well_map(score.random_vectors(len(well)), group, well)
    per_well = score.well_chance(group, well)
    chance = score.pooled(per_well["chance"], result["unit_group"])
    base_rate = score.pooled(per_well["base_rate"], result["unit_group"])
    assert base_rate == pytest.approx(result["chance"], rel=1e-12)                # the well mAP's own chance
    assert chance == pytest.approx(score.expected_random_ap(wells_per_sirna - 1, n_wells - 1), rel=1e-12)
    assert abs(result["value"] - chance) < tolerance
    assert chance - base_rate > (0.005 if wells_per_sirna == 36 else 0.015)


def test_the_audit_groups_pool_each_seen_wells_own_chance():
    """A7: an audit group's chance / base_rate pool the per-well values each well has in the seen group (its R among
    the seen wells), per siRNA then over siRNA -- not the seen group's value, and not values recomputed inside the
    subset. The siRNA have unequal well counts, so the per-well chance varies within seen, and val_wells lacks one."""
    sirna_of = {f"w{i:02d}": s for i, s in enumerate(["s0"] * 5 + ["s1"] * 4 + ["s2"] * 3)}
    split_of = {"w00": "val", "w01": "val", "w05": "val", "w02": "train", "w06": "train", "w10": "train"}
    rows = [(w, split_of.get(w, "unused")) for w in sirna_of for _ in range(2)]
    cells = pd.DataFrame(rows, columns=["well", "split"])
    sirna = np.array([sirna_of[w] for w, _ in rows])
    out = score.evaluate(np.random.default_rng(0).normal(size=(len(rows), 6)), cells, sirna)
    exact, n = score.expected_random_ap, 11                              # 12 seen wells: 11 candidates per query
    r_of = {"s0": 4, "s1": 3, "s2": 2}                                   # positives per query within seen
    assert out["seen"]["chance"] == pytest.approx(np.mean([exact(r, n) for r in r_of.values()]), rel=1e-12)
    for group, present in (("val_wells", ("s0", "s1")), ("trained_wells", ("s0", "s1", "s2")),
                           ("unused_wells", ("s0", "s1", "s2"))):
        assert out[group]["chance"] == pytest.approx(np.mean([exact(r_of[s], n) for s in present]), rel=1e-12)
        assert out[group]["base_rate"] == pytest.approx(np.mean([r_of[s] / n for s in present]), rel=1e-12)
    assert out["val_wells"]["chance"] != pytest.approx(out["seen"]["chance"])
    # run_prep is its own group (train + val wells: s0 x3, s1 x2, s2 x1 -> 5 candidates, s2 unscored)
    assert out["run_prep"]["chance"] == pytest.approx(np.mean([exact(2, 5), exact(1, 5)]), rel=1e-12)
    assert out["run_prep"]["base_rate"] == pytest.approx(np.mean([2 / 5, 1 / 5]), rel=1e-12)
    assert out["heldout"] is None


def test_a_failing_trial_0_is_initial_trial_failed_not_a_crash(prepared, stub, tmp_path):
    root, _, _ = prepared
    stub.fail_zero_shot = True
    run_dir = tmp_path / "run_fail"
    final = shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    assert (final["use"], final["provisional"], final["reason"]) == ("frozen", False, "initial_trial_failed")
    trial0 = json.loads((run_dir / "logs" / "input_trial.json").read_text())
    assert trial0["status"] == "crashed" and "did not load" in trial0["error"]
    assert len(stub.calls) == 1


def test_a_run_killed_in_round_1_leaves_a_provisional_frozen_record(prepared, stub, tmp_path, monkeypatch):
    class Killed(BaseException):                    # a Slurm kill: nothing after it runs
        pass

    real = stub.execute

    def killed_in_training(self, **kw):
        if kw.get("adaptation_depth") != "zero_shot":
            raise Killed()
        return real(self, **kw)

    killed_in_training.__signature__ = real.__signature__
    monkeypatch.setattr(stub, "execute", killed_in_training)
    root, _, _ = prepared
    run_dir = tmp_path / "run_killed"
    with pytest.raises(Killed):
        shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    decision = score.resolve_decision(run_dir, {})
    assert (decision["use"], decision["provisional"], decision["checkpoint"]) == ("frozen", True, None)


def test_the_control_fails_a_default_arm_that_trained_nothing(prepared, stub, monkeypatch):
    real = stub.execute

    def untrained(self, **kw):                      # e.g. no memory for a batch: the frozen embedding, no state
        return real(self, **{**kw, "adaptation_depth": "zero_shot"})

    untrained.__signature__ = real.__signature__
    monkeypatch.setattr(stub, "execute", untrained)
    root, _, _ = prepared
    with pytest.raises(SystemExit, match="CONTROL_FAILED: dinov3: trained no epoch"):
        control.control(root, seeds=(0,), split_key="split")
    kept = json.loads((root / control.CONTROL_DIR / "CONTROL.json").read_text(encoding="utf-8"))
    assert set(kept["arms"]) == {"frozen"} and not kept.get("complete")


def test_the_dinov3_control_trains_on_the_train_wells_only_and_selects_by_training_loss(prepared, stub):
    """A23: the dinov3 arm's tool input is the train wells' crops only (the cells the SHAPE / ReAct calls train on),
    so the tool resolves a split without a val crop: no validation, no proxy, best = the lowest training loss; frozen
    keeps the run prep. CONTROL.json records each arm's input; scoring embeds every cell with the arm's resume_best."""
    from shape.tools.cell_state_analyzer.tool import resolve_split

    root, _, _ = prepared
    cells = prep_mod.load(root)[1]
    ctl = control.control(root, seeds=(0,), split_key="split")
    frozen_call, dinov3_call = stub.calls
    train = set(cells.cell_id[cells.split == "train"])
    run_prep = set(cells.cell_id[cells.split.isin(("train", "val"))])
    assert {m["cell_id"] for m in dinov3_call["cell_metadata"]} == train < run_prep
    assert {m["cell_id"] for m in frozen_call["cell_metadata"]} == run_prep
    assert dinov3_call.get("val_fraction") is None and dinov3_call["min_epochs"] == 1
    split = resolve_split(dinov3_call["cell_metadata"], dinov3_call["split_key"], dinov3_call.get("val_fraction"))
    assert set(split) == {"train"}                  # no val row: the tool builds no validation, so no proxy
    entry = ctl["arms"]["dinov3"]
    assert entry["input"] == {"splits": ["train"], "n_cells": len(train)}
    assert ctl["arms"]["frozen"]["input"] == {"splits": ["train", "val"], "n_cells": len(run_prep)}
    assert entry["split"]["n_val"] == 0 and entry["training"]["best_by"] == "train_loss" and not entry["val_history"]
    assert score.control_state(ctl, "dinov3") == {"init_checkpoint": entry["resume_best"], "embed_only": True}


def test_the_arm_refuses_to_start_without_an_engine(prepared, stub, tmp_path):
    root, _, _ = prepared
    with pytest.raises(SystemExit, match="no reasoner engine"):
        shape_arm.run(root, tmp_path / "run", cell_type="HEPG2", model="gpt-5.6-luna", max_rounds=1)


def test_score_refuses_a_submission_without_a_checkpoint(tmp_path):
    (tmp_path / "submission").mkdir()
    (tmp_path / "submission" / "decision.json").write_text(json.dumps({"use": "agent_1", "provisional": True}))
    with pytest.raises(SystemExit, match="INVALID_SUBMISSION"):
        score.resolve_decision(tmp_path, {"agent_1": {"resume_best": str(tmp_path / "missing.pt")}})
    (tmp_path / "submission" / "decision.json").write_text(json.dumps({"use": "frozen", "provisional": True}))
    assert score.resolve_decision(tmp_path, {})["provisional"] is True


# ---- per_plate: embed_only applies the stored training statistics ---------------------------------------------

def _stub_backbone():
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, d_in, d_out):
            super().__init__()
            self.mlp = nn.Linear(d_in, d_out)
            self.sample_drop_ratio = 0.0

        def forward(self, x):
            return torch.tanh(self.mlp(x))

    class StubViT(nn.Module):
        """dinov3-style names (patch_embed, cls_token, blocks.i, norm); CLS = the mean patch token's path."""

        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Module()
            self.patch_embed.proj = nn.Conv2d(3, 16, 16, 16)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 16))
            self.blocks = nn.ModuleList([Block(16, 16), Block(16, 768)])
            self.norm = nn.LayerNorm(768)

        def forward_features(self, x):
            tokens = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
            x = torch.cat([self.cls_token + tokens.mean(1, keepdim=True), tokens], dim=1)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return {"x_norm_clstoken": x[:, 0], "x_norm_patchtokens": x[:, 1:]}

        def forward(self, x):
            return self.forward_features(x)["x_norm_clstoken"]

    torch.manual_seed(0)
    return StubViT


def test_a_per_plate_state_embeds_its_training_cells_as_in_training(prepared, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from shape.tools.cell_state_analyzer import tool as tool_module

    root, _, _ = prepared
    stub_vit = _stub_backbone()
    weights = tmp_path / "stub_backbone.pth"
    torch.save(stub_vit().state_dict(), weights)
    monkeypatch.setattr(tool_module, "hf_hub_download", lambda **kw: str(weights))
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: stub_vit())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SHAPE_FAST_FIGURES", "1")

    summary, cells = prep_mod.load(root)
    full = pd.read_csv(root / "prep" / "cells.csv", dtype=str, keep_default_na=False)
    trained = full[full.split == "train"]
    # the fixture's point: statistics recomputed from every FULL cell of a plate differ from the training ones
    images = np.load(summary["crop_array"]["path"])
    level = {cid: images[int(row)].mean() for cid, row in zip(full.cell_id, full.array_row)}
    for plate in trained.plate.unique():
        train_mean = np.mean([level[c] for c in trained.cell_id[trained.plate == plate]])
        all_mean = np.mean([level[c] for c in full.cell_id[full.plate == plate]])
        assert abs(all_mean - train_mean) > 10

    crop_array, meta = prep_mod.tool_input(summary, cells)
    result = control.analyzer().execute(crop_array=crop_array, cell_metadata=[dict(m) for m in meta],
                                        query_cache_dir=str(tmp_path / "train"), output_subdir="per_plate",
                                        split_key="split", intensity_norm="per_plate", max_epochs=1, batch_size=8)
    assert "error" not in result and result["resume_best"]
    in_training = score.aligned(anndata.read_h5ad(result["adata_path"]), [m["cell_id"] for m in meta])
    delivered = score.embed(summary, cells, tmp_path / "score" / "shape",
                            {"init_checkpoint": result["resume_best"], "embed_only": True})
    rows = {cid: i for i, cid in enumerate(cells.cell_id)}
    again = delivered[[rows[m["cell_id"]] for m in meta]]
    np.testing.assert_allclose(again, in_training, rtol=1e-5, atol=1e-5)
    assert len(delivered) == len(cells) and np.isfinite(delivered).all()


# ---- the external arms (external.py, contract amendment A2) ----------------------------------------------------

from benchmarks.representation_RxRx1.runners import external  # noqa: E402

FAKE_CP_BULK = '''"""A stand-in for cp_measure.bulk (the call contract: func(masks, pixels, **params) -> {name: per-object array})."""
import numpy as np


def _object(masks):
    assert masks.dtype.kind == "i" and set(np.unique(masks)) <= {0, 1}
    return masks == 1


def sizeshape(masks, pixels, **params):
    assert pixels is None
    m = _object(masks)
    rows, cols = np.nonzero(m)
    return {"AreaShape_Area": np.array([m.sum()], float), "AreaShape_Extent": np.array([m.sum() / m.size]),
            "AreaShape_Spread": np.array([rows.std() + cols.std()]), "AreaShape_EulerOne": np.array([1.0]),
            "Center_X": np.array([cols.mean()])}


def intensity(masks, pixels, **params):
    assert pixels.dtype == np.float64 and 0.0 <= pixels.min() and pixels.max() <= 1.0
    v = pixels[_object(masks)]
    return {"Intensity_MeanIntensity": np.array([v.mean()]), "Intensity_StdIntensity": np.array([v.std()])}


def texture(masks, pixels, scale=3, gray_levels=256):
    v = np.where(_object(masks), pixels, 0.0)
    return {f"Texture_Contrast_{scale}_00_{gray_levels}": np.array([np.abs(v[:, scale:] - v[:, :-scale]).mean()]),
            "Texture_Undefined": np.array([np.nan])}


def granularity(mask, pixels, subsample_size=0.25, image_sample_size=0.25, element_size=10, granular_spectrum_length=16):
    v = pixels[_object(mask)]
    return {f"Granularity_{i}": np.array([float((v ** i).mean())]) for i in (1, 2)}


def radial_distribution(labels, pixels, scaled=True, bin_count=4, maximum_radius=100):
    m = _object(labels)
    total = max(float(pixels[m].sum()), 1e-9)
    inner = m.copy()
    inner[:, : m.shape[1] // 2] = False
    return {f"RadialDistribution_FracAtD_1of{bin_count}": np.array([float(pixels[inner].sum()) / total])}


def get_core_measurements(legacy=False, sanitize=True):
    return {"radial_distribution": radial_distribution, "intensity": intensity, "sizeshape": sizeshape,
            "texture": texture, "granularity": granularity}
'''

FAKE_MAE = '''"""A stand-in for OpenPhenom's huggingface_mae (MAEConfig / MAEModel with the card's predict contract)."""
import json
import os

import torch
from torch import nn

from .normalizer import Normalizer


class MAEConfig:
    def __init__(self, **values):
        self.__dict__.update(values)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            return cls(**json.load(handle))


class MAEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = Normalizer()
        self.proj = nn.Linear(96, 16)
        self.return_channelwise_embeddings = True

    def predict(self, imgs):
        assert imgs.shape[1:] == (6, 256, 256) and not self.return_channelwise_embeddings
        x = nn.functional.adaptive_avg_pool2d(self.norm(imgs), 4).flatten(1)
        return torch.tanh(self.proj(x))
'''

FAKE_NORMALIZER = '''import torch


class Normalizer(torch.nn.Module):
    def forward(self, pixels):
        return pixels.float() / 255.0
'''


@pytest.fixture()
def fake_cp_measure(tmp_path, monkeypatch):
    """A cp_measure stand-in as a package on disk (so the spawn pool's workers import it too)."""
    base = tmp_path / "fake_packages"
    (base / "cp_measure").mkdir(parents=True)
    (base / "cp_measure" / "__init__.py").write_text("", encoding="utf-8")
    (base / "cp_measure" / "bulk.py").write_text(FAKE_CP_BULK, encoding="utf-8")
    monkeypatch.syspath_prepend(str(base))
    for name in [m for m in sys.modules if m == "cp_measure" or m.startswith("cp_measure.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(external, "_WORKER", {})
    return base


def _fake_openphenom_source(base: Path) -> tuple[Path, str]:
    """A local copy of an OpenPhenom snapshot with the stand-in model package; (its directory, the weights sha256)."""
    torch = pytest.importorskip("torch")
    base.mkdir(parents=True)
    for name in external.OPENPHENOM_FILES:
        (base / name).write_text("", encoding="utf-8")
    (base / "huggingface_mae.py").write_text(FAKE_MAE, encoding="utf-8")
    (base / "normalizer.py").write_text(FAKE_NORMALIZER, encoding="utf-8")
    (base / "config.json").write_text(json.dumps({"model_type": "MAE", "mask_ratio": 0.0}), encoding="utf-8")
    generator = torch.Generator().manual_seed(0)
    torch.save({"state_dict": {"proj.weight": torch.randn(16, 96, generator=generator), "proj.bias": torch.zeros(16)}},
               base / external.OPENPHENOM_WEIGHTS)
    return base, prep_mod.sha256_file(base / external.OPENPHENOM_WEIGHTS)


def _stage_fake_openphenom(output_root: Path, tmp_path: Path) -> dict:
    (output_root / "assets").mkdir(parents=True, exist_ok=True)
    (output_root / "assets" / "ASSETS.json").write_text("{}", encoding="utf-8")       # the DINOv3 assets (prepare)
    source, sha = _fake_openphenom_source(tmp_path / "openphenom_download")
    external.stage_openphenom(output_root, source=source, expected_weights_sha256=sha)
    return external.check_openphenom(output_root)


class RecordingPhenom:
    """model.predict's contract: records every input batch, returns its per-channel means (B, 6)."""

    def __init__(self):
        self.inputs = []

    def predict(self, x):
        self.inputs.append(x.detach().clone())
        return x.mean(dim=(2, 3))


def test_openphenom_adapter_feeds_six_channels_at_256_in_row_order():
    pytest.importorskip("torch")
    images = np.zeros((5, 6, 64, 64), dtype=np.uint8)
    for i in range(5):
        for c in range(6):
            images[i, c] = 10 * c + i + 1                  # channel c of crop i: a constant that names both
    rows = [3, 0, 4]
    model = RecordingPhenom()
    out = external.openphenom_vectors(model, images, rows, batch_size=2, progress=False)
    assert [tuple(x.shape) for x in model.inputs] == [(2, 6, 256, 256), (1, 6, 256, 256)]
    assert all(str(x.dtype) == "torch.float32" for x in model.inputs)
    expected = np.array([[10 * c + r + 1 for c in range(6)] for r in rows], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6)    # the stored channel order, one vector per cell in row order
    padded = RecordingPhenom()
    external.openphenom_vectors(padded, images, [2], fit="pad", progress=False)
    x = padded.inputs[0][0]
    assert float(x[:, :96].abs().sum()) == 0 and float(x[:, :, :96].abs().sum()) == 0     # zero canvas around
    np.testing.assert_array_equal(x[:, 96:160, 96:160].numpy(), images[2].astype(np.float32))   # the crop, centred
    with pytest.raises(ValueError, match="fit must be"):
        external.openphenom_vectors(RecordingPhenom(), images, [0], fit="crop", progress=False)


def test_openphenom_loads_the_staged_snapshot_offline(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")            # restored after the test
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="run the prepare stage first"):
        external.stage_openphenom(out, source=tmp_path)
    manifest = _stage_fake_openphenom(out, tmp_path)
    assert set(manifest["files"]) == set(external.OPENPHENOM_FILES) and manifest["revision"] == external.OPENPHENOM_REVISION
    assert Path(manifest["snapshot"]) == out.absolute() / "assets" / "openphenom" / "snapshot"
    assert not (out / "assets" / "openphenom.partial").exists() and "Non-Commercial" in manifest["license"]
    model = external.load_openphenom(Path(manifest["snapshot"]), "cpu")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert model.return_channelwise_embeddings is False and not model.training
    vectors = external.openphenom_vectors(model, np.zeros((3, 6, 64, 64), np.uint8), [0, 1, 2], progress=False)
    assert vectors.shape == (3, 16) and np.isfinite(vectors).all()
    with open(Path(manifest["snapshot"]) / external.OPENPHENOM_WEIGHTS, "rb") as handle:   # a path would pick the
        weights = torch.load(handle, weights_only=True)["state_dict"]            # safetensors loader (newer torch)
    assert torch.equal(model.proj.weight, weights["proj.weight"])            # the staged weights were loaded
    assert external.stage_openphenom(out) == out.absolute() / "assets" / "openphenom" / "ASSETS.json"   # kept
    (Path(manifest["snapshot"]) / "vit.py").write_text("changed", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed since staging"):
        external.check_openphenom(out)
    other = tmp_path / "out2"
    (other / "assets").mkdir(parents=True)
    (other / "assets" / "ASSETS.json").write_text("{}", encoding="utf-8")
    source, _ = _fake_openphenom_source(tmp_path / "other")
    with pytest.raises(SystemExit, match="is not the pinned"):                 # the real weights' sha256 is pinned
        external.stage_openphenom(other, source=source)


def _disk(size, centre, radius):
    yy, xx = np.mgrid[:size, :size]
    return np.hypot(yy - centre[0], xx - centre[1]) <= radius


def _cell_crop(seed=0, nucleus_at=(32, 32), other_dna=(8, 52), blank_other=False):
    rng = np.random.default_rng(seed)
    crop = rng.normal(15, 3, size=(6, 64, 64))
    crop[0] += 170 * _disk(64, nucleus_at, 8) + 170 * _disk(64, other_dna, 4)     # its nucleus + a neighbour's
    if not blank_other:
        crop[1:] += 100 * _disk(64, nucleus_at, 20)[None]
    else:
        crop[1:] = 0
    return np.clip(crop, 0, 255).astype(np.uint8), _disk(64, nucleus_at, 8), _disk(64, nucleus_at, 20)


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_cellprofiler_masks_find_the_central_disk_nucleus_and_its_cell():
    crop, nucleus_truth, cell_truth = _cell_crop()
    nucleus, cell, flags = external.crop_masks(crop)
    assert _iou(nucleus, nucleus_truth) > 0.85 and not nucleus[8, 52]          # the centre one, not the neighbour
    assert _iou(cell, cell_truth) > 0.85 and (cell | ~nucleus).all()           # the cell holds its nucleus
    assert flags == {"nucleus_fallback": False, "cell_fallback": False, "cell_touches_border": False}
    again = external.crop_masks(crop)
    assert (again[0] == nucleus).all() and (again[1] == cell).all()            # deterministic
    noisy = external.crop_masks(_cell_crop(seed=5)[0])
    assert _iou(noisy[0], nucleus) > 0.9 and _iou(noisy[1], cell) > 0.9        # stable under other pixel noise
    off, truth, _ = _cell_crop(nucleus_at=(32, 46))                            # the centre pixel is background
    assert _iou(external.crop_masks(off)[0], truth) > 0.85                     # the nearest component
    blank = _cell_crop(blank_other=True)[0]
    _, whole, flags = external.crop_masks(blank)
    assert whole.all() and flags["cell_fallback"] and not flags["nucleus_fallback"]
    flat = np.full((6, 64, 64), 7, np.uint8)
    nucleus, cell, flags = external.crop_masks(flat)
    assert nucleus.all() and cell.all()
    assert flags == {"nucleus_fallback": True, "cell_fallback": True, "cell_touches_border": False}
    no_dna = _cell_crop()[0]
    no_dna[0] = 7                                                              # cytoplasm, but no nucleus to seed
    nucleus, cell, flags = external.crop_masks(no_dna)
    assert nucleus.all() and cell.all() and flags["nucleus_fallback"] and flags["cell_fallback"]


def _two_cell_crop(seed=0, centre=(32, 30), neighbour=(32, 54), spread=9.0):
    """Two cells whose cytoplasm foregrounds join: the central one and a neighbour; the cytoplasm dims between them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:64, :64]
    crop = rng.normal(15, 3, size=(6, 64, 64))
    crop[0] += 170 * _disk(64, centre, 6) + 170 * _disk(64, neighbour, 6)
    blob = sum(np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * spread ** 2)) for c in (centre, neighbour))
    crop[1:] += 110 * blob[None]
    return np.clip(crop, 0, 255).astype(np.uint8)


def test_cellprofiler_cell_mask_is_a_seeded_watershed_that_keeps_the_central_cell():
    from skimage.measure import label

    crop = _two_cell_crop()
    dna = external._foreground(crop[0])
    joined = label(external._foreground(crop[1:].astype(np.float32).sum(axis=0)) | dna, connectivity=2)
    assert joined[32, 30] == joined[32, 54] != 0          # one foreground component: touching-components would merge
    nucleus, cell, flags = external.crop_masks(crop)
    assert _iou(nucleus, _disk(64, (32, 30), 6)) > 0.8
    assert cell[nucleus].all() and not cell[_disk(64, (32, 54), 6)].any()     # its nucleus, not the neighbour's
    assert cell[32, 20] and not cell[32, 50] and np.flatnonzero(cell.any(axis=0)).max() <= 45   # split between them
    assert flags == {"nucleus_fallback": False, "cell_fallback": False, "cell_touches_border": False}
    mirrored = external.crop_masks(_two_cell_crop(centre=(32, 34), neighbour=(32, 10)))[1]
    assert mirrored[32, 44] and not mirrored[_disk(64, (32, 10), 6)].any()    # the other side
    edge = _cell_crop()[0]
    edge[1:] = np.clip(np.random.default_rng(1).normal(15, 3, size=(5, 64, 64)), 0, 255).astype(np.uint8)
    edge[1:, :, :48] += 100                                                    # its cytoplasm runs off the crop
    _, cell, flags = external.crop_masks(edge)
    assert cell[:, 0].any() and flags["cell_touches_border"] and not flags["cell_fallback"]


def test_cellprofiler_features_are_filtered_raw_and_scaled_only_in_score(fake_cp_measure):
    crops, plates = [], []
    for i in range(40):
        crop = _cell_crop(seed=i, nucleus_at=(32 + i % 3, 31 + i % 4))[0].astype(np.int32)
        crop += 40 * (i % 2)                                                     # plate B is brighter
        crops.append(np.clip(crop, 0, 255).astype(np.uint8))
        plates.append("B" if i % 2 else "A")
    images, plates = np.stack(crops), np.array(plates)
    rows = np.arange(len(images))[::-1]                                          # any row order
    raw, names, counts = external.cellprofiler_features(images, rows, progress=False)
    assert raw.shape == (40, len(names)) and counts["errors"] == {}
    assert {k: counts[k] for k in external.CP_MASK_COUNTS} == dict.fromkeys(external.CP_MASK_COUNTS, 0)
    assert "Nuclei_AreaShape_Area" in names and "Cells_Intensity_MeanIntensity_wga" in names
    assert len([n for n in names if n.endswith("_hoechst")]) == 2 * 7            # 2 compartments x 7 per channel
    single, _, _ = external.cellprofiler_features(images, rows[:1], progress=False)
    np.testing.assert_array_equal(single[0], raw[0])                             # row i = images[rows[i]]
    kept_raw, kept, dropped = external.filter_features(raw, names)
    assert kept_raw.dtype == np.float32 and kept_raw.shape == (40, len(kept))
    np.testing.assert_array_equal(kept_raw, raw[:, [names.index(n) for n in kept]].astype(np.float32))   # raw values
    assert dropped["Nuclei_AreaShape_EulerOne"] == "constant" and dropped["Cells_AreaShape_EulerOne"] == "constant"
    assert dropped["Nuclei_Center_X"] == "absolute position in the crop"
    assert all(dropped[n].startswith("non-finite") for n in names if "Texture_Undefined" in n)
    assert set(kept) | set(dropped) == set(names) and not set(kept) & set(dropped)
    z = score.post_process("cellprofiler", kept_raw, plates[rows], "matched")   # the scaling lives in score.py
    assert np.isfinite(z).all() and np.abs(z).max() <= score.CLIP
    for plate in ("A", "B"):                                                     # robust z per plate
        block = z[plates[rows] == plate]
        np.testing.assert_allclose(np.median(block, axis=0), 0, atol=1e-6)
    pooled, _, _ = external.cellprofiler_features(images, rows, workers=2, progress=False)
    np.testing.assert_array_equal(pooled, raw)                                   # the spawn pool keeps the row order


def test_matched_is_one_per_plate_centre_scale_for_every_arm_and_native_keeps_the_analyzer_raw():
    rng = np.random.default_rng(0)
    plates = np.repeat(np.array(["P1", "P2", "P3"]), 50)
    x = rng.normal(size=(150, 6)) * np.arange(1, 7) + 10.0 * (plates == "P2")[:, None]     # P2 shifted
    x[:, 4] = 3.0                                                               # constant
    x[:, 5] = (np.arange(150) % 10 == 0).astype(float)                          # MAD 0 on every plate, SD > 0
    arms = ("frozen", "dinov3", "shape", "shape_top", "openphenom", "cellprofiler")
    reference = score.centre_scale(x, plates)
    for arm in arms:                                                            # the same transform for every arm
        np.testing.assert_array_equal(score.post_process(arm, x, plates, "matched"), reference)
    for plate in ("P1", "P2", "P3"):
        block, member = reference[plates == plate], x[plates == plate]
        np.testing.assert_allclose(np.median(block[:, :4], axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(score.MAD_TO_SD * np.median(np.abs(block[:, :4]), axis=0), 1, rtol=1e-5)
        assert (block[:, 4] == 0).all()
        np.testing.assert_allclose(block[:, 5], member[:, 5] / member[:, 5].std(), rtol=1e-5)   # the SD fallback
    assert np.abs(reference).max() <= score.CLIP and reference.dtype == np.float32
    for arm in ("frozen", "dinov3", "shape", "shape_top", "openphenom"):    # native: as embedded
        np.testing.assert_array_equal(score.post_process(arm, x, plates, "native"), x)
    native_cp = score.post_process("cellprofiler", x, plates, "native")       # one global scaling: no plate fit
    np.testing.assert_array_equal(native_cp, score.centre_scale(x, np.zeros(len(x))))
    assert not np.allclose(native_cp, reference)
    y = x.copy()
    y[0, 0] = np.nan
    assert np.isfinite(score.centre_scale(y, plates)).all() and score.centre_scale(y, plates)[0, 0] == 0
    with pytest.raises(ValueError, match="table must be"):
        score.post_process("frozen", x, plates, "pca")


def _external_run(root, tmp_path, arms):
    """The external stage on the prep under `root` with its labels moved away (no arm can read one)."""
    private = root / "private"
    hidden = tmp_path / "labels_elsewhere"
    private.rename(hidden)
    try:
        if "openphenom" in arms:
            _stage_fake_openphenom(root.parent, tmp_path)
        return external.run(root, root.parent, arms, workers=1)
    finally:
        hidden.rename(private)


def test_score_includes_the_external_arms_and_their_comparisons(prepared, stub, fake_cp_measure, tmp_path,
                                                                monkeypatch):
    root, _, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "external"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    n_tool_calls = len(stub.calls)
    records = _external_run(root, tmp_path, external.EXTERNAL_ARMS)
    assert len(stub.calls) == n_tool_calls                                       # the analyzer is not involved
    _, cells = prep_mod.load(root)
    for arm, record in records.items():
        assert record["n_cells"] == len(cells) and record["label_free"] and not record["pilot"]
        assert not _leaks(record) and not _leaks(json.loads((root / "external" / arm / "EMBED.json").read_text()))
        assert record["settings"] and set(record["packages"]) and record["external_sha256"]
    assert records["openphenom"]["settings"]["input_size"] == 256 and records["openphenom"]["dim"] == 16
    assert records["cellprofiler"]["features"] and records["cellprofiler"]["dropped_features"]
    assert set(records["cellprofiler"]["masks"]) == set(external.CP_MASK_COUNTS)
    raw_cp = np.load(root / "external" / "cellprofiler" / "embeddings.npy")      # stored raw, not scaled
    assert raw_cp.dtype == np.float32 and raw_cp.shape == (len(cells), len(records["cellprofiler"]["features"]))
    assert np.nanmax(np.abs(raw_cp)) > score.CLIP or not np.allclose(np.nanmedian(raw_cp, axis=0), 0)
    calls, real = [], score.post_process

    def recording(arm, raw, plates, table):
        out = real(arm, raw, plates, table)
        calls.append((arm, table, raw, plates, out))
        return out

    monkeypatch.setattr(score, "post_process", recording)
    result = score.final(root, run_dir)
    assert {a: s["status"] for a, s in result["external_arms"].items()} == {"openphenom": "scored",
                                                                            "cellprofiler": "scored"}
    for arm, status in result["external_arms"].items():
        assert status["embeddings_sha256"] == records[arm]["embeddings_sha256"]
        assert status["external_sha256"] == records[arm]["external_sha256"]
    assert status["masks"] == records["cellprofiler"]["masks"]
    # every arm is post-processed in both tables; matched = the same per-plate centre_scale; native analyzer arms raw
    assert {a for a, t, *_ in calls if t == "matched"} == {a for a, t, *_ in calls if t == "native"}
    assert {"frozen", "dinov3", "openphenom", "cellprofiler"} <= {a for a, *_ in calls}
    for arm, table, raw, plates, out in calls:
        np.testing.assert_array_equal(plates, cells.plate.to_numpy())
        if table == "matched":
            np.testing.assert_array_equal(out, score.centre_scale(raw, cells.plate.to_numpy()))
        elif arm != "cellprofiler":
            np.testing.assert_array_equal(out, raw)
    assert set(result["matched"]["tables"]) == set(result["tables"])           # both tables hold every arm
    assert result["postprocess"]["read"]["external_comparisons"] == "matched"
    assert set(result["matched"]["reading_vs_frozen"]) == {"seen", "heldout"}
    for group in ("seen", "heldout"):
        for groups in (result["groups"], result["matched"]["groups"]):
            comparisons = groups[group]["comparisons"]
            assert {"shape_vs_default_mean", "shape_vs_frozen", "shape_vs_openphenom", "shape_vs_cellprofiler",
                    "frozen_vs_openphenom", "frozen_vs_cellprofiler", "default_mean_vs_frozen"} <= set(comparisons)
        assert result["tables"]["openphenom"][group]["value"] is not None
        assert result["matched"]["tables"]["frozen"][group]["value"] is not None
    table = (run_dir / "RESULT.md").read_text(encoding="utf-8")
    assert "| openphenom |" in table and "| cellprofiler |" in table and "openphenom scored" in table
    assert "not the full CellProfiler pipeline" in table and external.OPENPHENOM_ATTRIBUTION in table
    assert "## Native table" in table and "## Matched table (read the external comparisons here)" in table
    assert "| seen | frozen_vs_openphenom |" in table and "Native vs matched:" in table


def test_a_missing_external_arm_is_skipped_and_a_stale_one_refused(prepared, stub, fake_cp_measure, tmp_path):
    root, _, _ = prepared
    control.control(root, seeds=(), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "skip"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    _external_run(root, tmp_path, ("cellprofiler",))
    result = score.final(root, run_dir)
    status = result["external_arms"]
    assert status["cellprofiler"]["status"] == "scored" and status["openphenom"]["status"] == "skipped"
    assert "run the external stage" in status["openphenom"]["note"] and "openphenom" not in result["tables"]
    seen = result["groups"]["seen"]["comparisons"]
    assert "shape_vs_cellprofiler" in seen and not any("openphenom" in name for name in seen)
    assert "openphenom skipped" in (run_dir / "RESULT.md").read_text(encoding="utf-8")
    path = root / "external" / "cellprofiler" / "EMBED.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    summary, cells = prep_mod.load(root)
    for change, match in (({"external_sha256": "0" * 64}, "another external.py"),
                          ({"crop_array_sha256": "0" * 64}, "another images.npy"),
                          ({"n_cells": len(cells) - 1}, "another prep")):
        path.write_text(json.dumps({**record, **change}), encoding="utf-8")
        with pytest.raises(SystemExit, match=f"EXTERNAL_STALE: .*{match}"):
            external.load_embeddings(root, summary, len(cells))
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(SystemExit, match="EXTERNAL_STALE: .*another prep"):      # n_cells vs score's cell count
        external.load_embeddings(root, summary, len(cells) + 1)
    embeddings = root / "external" / "cellprofiler" / "embeddings.npy"
    changed = np.load(embeddings)
    changed[0, 0] += 1.0
    np.save(embeddings, changed)
    with pytest.raises(SystemExit, match="INTEGRITY_FAILED"):
        score.final(root, run_dir)
    path.write_text(json.dumps({**record, "external_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="EXTERNAL_STALE"):
        score.final(root, run_dir)


def test_the_external_stage_pilots_keeps_and_refuses(prepared, fake_cp_measure, tmp_path, monkeypatch, capsys):
    from benchmarks.representation_RxRx1 import run_rxrx1

    root, _, _ = prepared
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    args = ["external", "--arms", "cellprofiler", "--output-root", str(root.parent)]
    run_rxrx1.main([*args, "--limit", "24"])
    first = root / "external_pilot" / "cellprofiler_n24" / "EMBED.json"
    pilot = json.loads(first.read_text(encoding="utf-8"))
    assert pilot["pilot"] and pilot["n_cells"] == 24
    assert {"value", "chance", "n_queries"} <= set(pilot["pilot_proxy"]["native"]) and pilot["pilot_proxy"]["matched"]
    assert not (root / "external").exists()                                       # a pilot is never an arm
    run_rxrx1.main([*args, "--limit", "12"])                                      # another setting: its own record
    assert json.loads((root / "external_pilot" / "cellprofiler_n12" / "EMBED.json").read_text())["n_cells"] == 12
    assert json.loads(first.read_text(encoding="utf-8")) == pilot                 # never overwritten
    run_rxrx1.main([*args, "--limit", "24"])
    assert "external cellprofiler: kept" in capsys.readouterr().out and json.loads(first.read_text()) == pilot
    assert external.pilot_name("openphenom", "pad", 2048) == "openphenom_pad_n2048"
    run_rxrx1.main(args)
    run_rxrx1.main(args)
    assert "external cellprofiler: kept" in capsys.readouterr().out
    monkeypatch.setattr(external, "CP_MAX_NONFINITE", 0.1)                        # other settings: refused
    with pytest.raises(SystemExit, match="EXTERNAL_STALE"):
        run_rxrx1.main(args)
    with pytest.raises(SystemExit, match="EXTERNAL_STALE"):                       # a pilot too
        run_rxrx1.main([*args, "--limit", "24"])
    with pytest.raises(SystemExit, match="unknown external arms"):
        run_rxrx1.main(["external", "--arms", "deepprofiler", "--output-root", str(root.parent)])
    with pytest.raises(SystemExit, match="positive number of cells"):
        run_rxrx1.main([*args, "--limit", "0"])


def test_openphenom_takes_256_px_only(prepared, tmp_path, monkeypatch):
    from benchmarks.representation_RxRx1 import run_rxrx1

    root, _, _ = prepared
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    assert external.check_input_size(256) == 256
    for size in (224, 64, 512):
        with pytest.raises(SystemExit, match="256 px only"):
            run_rxrx1.main(["external", "--arms", "openphenom", "--input-size", str(size),
                            "--output-root", str(root.parent)])
        with pytest.raises(SystemExit, match="256 px only"):
            external.openphenom_vectors(RecordingPhenom(), np.zeros((1, 6, 64, 64), np.uint8), [0], input_size=size,
                                        progress=False)
    assert not (root / "external").exists() and not (root / "external_pilot").exists()


# ---- the LLM comparison arms (llm_arms.py, contract amendment A4) ---------------------------------------------

from benchmarks.batch_correction_JUMP.runners.score import jsonl  # noqa: E402
from benchmarks.representation_RxRx1.runners import llm_arms, react_arm  # noqa: E402

PROXY_SENTINEL, EVAL_SENTINEL = 0.876543, 2.718281   # the stand-in's val-proxy / eval-loss values: never in a prompt
STUB_TOKENS = 110                                     # the scripted engine's tokens per reply


class ScriptedEngine:
    """A reasoner stand-in: the scripted replies in order (a dict of the schema's fields, returned as the engine returns
    a parsed reply, with its usage recorded in SHAPE's process-level count; {"error": ...}: a gateway error reply);
    every content, prompt and schema is recorded. The content must be a one-element list holding the prompt, as
    SHAPE's `AgenticVerifier._ask` sends it."""

    def __init__(self, replies):
        self.replies, self.contents, self.prompts, self.schemas = list(replies), [], [], []

    def generate(self, content, response_format=None, **kwargs):
        from shape.engine import usage

        assert isinstance(content, list) and len(content) == 1 and isinstance(content[0], str), content
        self.contents.append(content)
        self.prompts.append(content[0])
        self.schemas.append(response_format)
        reply = self.replies.pop(0)
        if "error" in reply:
            return reply
        tokens = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": STUB_TOKENS}
        usage.record(tokens)
        return {"content": response_format.model_validate(reply), "usage": tokens}


class ReactEngine(ScriptedEngine):
    """The react arm's reasoner stand-in: ReactAction replies (the shared runner sends the prompt as a string)."""

    def generate(self, content, response_format=None, **kwargs):
        return super().generate([content] if isinstance(content, str) else content, response_format, **kwargs)


def _shape_engine(max_rounds: int) -> ScriptedEngine:
    """The SHAPE arm's reasoner stand-in for a run of `max_rounds` (2 or 3) rounds, one call per round (the
    engine-less fallback chain is gone: a missing reasoner is a system error): rounds 1-2 the arm's two-round expert
    start (A24: agent_1 from the input at trainable_pct 33, agent_2 resuming agent_1 at trainable_pct 66);
    learning_rate 1e-5 from agent_2 (agent_3, resumed); then the final question delivers the last agent call."""
    start, second = shape_arm.EXPERT_CONFIG
    rounds = [("input", start), ("agent_1", {k: v for k, v in second.items() if k != "init_from"}),
              ("agent_2", {"learning_rate": 1e-5})][:max_rounds]
    plans = [{"analysis": "a", "moves": [{"parent": parent, "diagnosis": "d", "changes": [
        {"lever": k, "value": json.dumps(v), "basis": f"scripted basis for {k}"} for k, v in changes.items()]}]}
        for parent, changes in rounds]
    return ScriptedEngine([*plans, {"call": f"agent_{len(rounds)}", "reasoning": "r"}])


def react_call(**params) -> dict:
    return {"thought": "t", "action": "call",
            "calls": [{"params": [{"name": k, "value": v} for k, v in params.items()]}]}


def react_finish(call: str) -> dict:
    return {"thought": "done", "action": "finish", "deliver": call}


@pytest.fixture()
def llm_env(stub, monkeypatch):
    """The stand-in analyzer with the real tool's withheld outputs (val_history, also inside training_config), a
    per-epoch loss and a stop reason; a call with koleo_weight > 0 fails; a call with drop_path > 0 finishes without
    a resume state. The 30 s gateway wait is recorded, not slept."""
    real = stub.execute

    def execute(self, **kw):
        if (kw.get("koleo_weight") or 0) > 0:
            raise RuntimeError("stub: out of memory")
        out = real(self, **kw)
        if out.get("val_history"):
            history = [{**entry, "sibling_site_map": PROXY_SENTINEL,
                        **({"eval_loss": EVAL_SENTINEL} if "eval_loss" in entry else {})}
                       for entry in out["val_history"]]
            out.update(val_history=history)
            if out["training"]["epochs_trained"]:
                out.update(loss_history=[1.23456, 1.11111],
                           training_config={**out["training_config"], "stop_reason": "max_epochs_reached",
                                            "val_history": history})
        if (kw.get("drop_path") or 0) > 0:
            out.update(resume_best=None, resume_end=None)
        return out

    execute.__signature__ = real.__signature__
    monkeypatch.setattr(stub, "execute", execute)
    sleeps = []
    monkeypatch.setattr(llm_arms, "_sleep", sleeps.append)
    # the llm arm's job check (no Slurm deadline here, the budget's 16 CPUs and 1 GPU)
    monkeypatch.setattr(llm_arms, "feature_resources", lambda: {"deadline": None, "cores": 16, "gpus": 1})
    return sleeps


def _requests(run_dir: Path) -> list[dict]:
    return jsonl(run_dir / "logs" / "requests.jsonl")


def _shape_levers() -> list[str]:
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    return RepresentationAgenticVerifier().levers


# ---- the llm arm (contract amendment A5): LLM-written hand-crafted features --------------------------------------

TINY_MODULE = '''import numpy as np


def features(crops):
    x = crops.astype(np.float64)
    out = np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1)
    out[::7, 0] = np.nan                      # NaN is allowed
    return out
'''


def _tiny_features(crops: np.ndarray) -> np.ndarray:
    x = crops.astype(np.float64)
    out = np.concatenate([x.mean(axis=(2, 3)), x.std(axis=(2, 3))], axis=1)
    out[::7, 0] = np.nan
    return out


def _module(source: str, reasoning: str = "r") -> dict:
    return {"reasoning": reasoning, "module": source}


def _crops_in_order(root: Path) -> np.ndarray:
    summary, cells = prep_mod.load(root)
    return np.load(summary["crop_array"]["path"])[cells.array_row.astype(int).to_numpy()]


def _attempts(run_dir: Path) -> list[dict]:
    return jsonl(run_dir / "logs" / "attempts.jsonl")


@pytest.fixture()
def popen_calls(monkeypatch):
    """Every module process the llm arm starts: (args, kwargs) of subprocess.Popen."""
    calls, real = [], llm_arms.subprocess.Popen

    def recording(args, **kwargs):
        if "-I" in list(args):                                       # not git (control.git_commit)
            calls.append((list(args), kwargs))
        return real(args, **kwargs)

    monkeypatch.setattr(llm_arms.subprocess, "Popen", recording)
    return calls


def test_the_llm_arm_runs_its_module_on_every_cell_without_the_analyzer(prepared, stub, llm_env, tmp_path,
                                                                        popen_calls, monkeypatch):
    root, summary, _ = prepared
    monkeypatch.setenv("LLM_API_KEY", "sk-test-not-for-the-module")
    engine = ScriptedEngine([_module(TINY_MODULE)])
    run_dir = tmp_path / "llm_HEPG2_run"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert (final["use"], final["status"], final["case"], final["provisional"]) == ("attempt_1", "delivered",
                                                                                   "features", False)
    assert stub.calls == []                                            # no analyzer, no training, no backbone
    _, cells = prep_mod.load(root)
    vectors = np.load(run_dir / "submission" / "embeddings.npy")
    assert vectors.dtype == np.float32 and vectors.shape == (len(cells), 12) == (len(cells), final["dim"])
    np.testing.assert_array_equal(vectors, _tiny_features(_crops_in_order(root)).astype(np.float32))   # cells.csv order
    record = json.loads((run_dir / "submission" / "EMBED.json").read_text())
    code = hashlib.sha256(TINY_MODULE.encode()).hexdigest()
    assert (record["code_sha256"], record["dim"], record["n_cells"], record["attempt"]) == (code, 12, len(cells), 1)
    assert record["nonfinite_values"] == int(np.isnan(vectors).sum()) > 0 and record["seconds"] is not None
    assert record["embeddings_sha256"] == prep_mod.sha256_file(run_dir / "submission" / "embeddings.npy")
    assert record["cells_sha256"] == summary["cells_sha256"] and record["crop_array_sha256"] == summary["crop_array"][
        "sha256"]
    assert (run_dir / "modules" / "attempt_1.py").read_bytes() == TINY_MODULE.encode()
    assert final["code_sha256"] == code and not _leaks(record)
    (attempt,) = _attempts(run_dir)
    assert (attempt["stage"], attempt["code_sha256"], attempt["dim"]) == ("done", code, 12)
    requests = _requests(run_dir)
    assert len(requests) == 1 and requests[0]["reply"]["module"] == TINY_MODULE and requests[0]["error"] is None
    assert engine.schemas == [llm_arms.FeatureModule]
    # one separate process: isolated mode, the worker program only, no credential in its environment
    (args, kwargs), = popen_calls
    assert args[1:3] == ["-I", str(run_dir / "modules" / "worker.py")] and "LLM_API_KEY" not in kwargs["env"]
    assert set(kwargs["env"]) <= {*llm_arms.CHILD_ENV, *llm_arms.CHILD_TEMP_ENV} and kwargs["start_new_session"]
    cwd = Path(kwargs["cwd"])                          # a fresh temporary directory outside the run tree, removed
    assert kwargs["env"]["TMPDIR"] == str(cwd) and not cwd.exists() and run_dir not in cwd.parents
    config = json.loads(args[3])
    assert str(tmp_path.resolve()) in config["deny"] and config["allow"] == str(cwd)
    assert config["blocked"] == list(llm_arms.BLOCKED_EVENTS) and "socket.connect" in config["blocked"]
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    assert (meta["agent"], meta["model"], meta["max_attempts"]) == ("llm", "stub", 8)       # SHAPE's max_rounds
    assert meta["time_rule"] == llm_arms.TIME_RULE and "time_budget_seconds" not in meta
    assert meta["llm_arms_sha256"] == prep_mod.sha256_file(Path(llm_arms.__file__)) and "card_sha256" not in meta
    assert meta["provenance"]["prep_cells_sha256"] == summary["cells_sha256"]
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert (agent["tool_calls"], agent["attempts"], agent["tokens"]["total_tokens"]) == (0, 1, STUB_TOKENS)
    assert agent["write_errors"] == [] and agent["stopped_because"] == "features"
    assert agent["audit"]["banned_like_tokens"] == []                  # the post-hoc audit of the delivered text
    assert (meta["resources"]["cores"], meta["resources"]["gpus"]) == (16, 1) and meta["denied_roots"]
    assert json.loads((run_dir / "submission" / "decision.json").read_text()) == final
    # the task text: facts only (the data, the goal, the interface, the packages, the budget)
    prompt = engine.prompts[0]
    assert f"{len(cells)} single-cell crops of RxRx1 HEPG2" in prompt
    assert ", ".join(external.CHANNELS) in prompt and "0.1th to its 99.9th percentile" in prompt
    assert "features(crops) -> array" in prompt and "1 <= d <= 4096" in prompt
    assert "NaN\nvalues are allowed" in prompt
    assert "first 180 crops (a smoke run)" in prompt and "Compute: 16 CPUs + 1 GPU for the whole task" in prompt
    assert "Time: no time limit (the job has no deadline)." in prompt and "Attempts: at most 8." in prompt
    assert "numpy, scipy, scikit-image, pandas, and torch without pretrained weights" in prompt
    with pytest.raises(SystemExit, match="holds a run already"):
        llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=ScriptedEngine([]))
    with pytest.raises(SystemExit, match="NOT_READY"):
        llm_arms.run(root, tmp_path / "no_engine", arm="llm", cell_type="HEPG2", model="stub", engine=None)


def test_the_static_check_rejects_network_subprocess_and_file_access():
    rejected = {
        "import socket": "import socket (not an allowed module)",
        "import urllib.request": "import urllib.request (not an allowed module)",
        "from http import client": "import http (not an allowed module)",
        "import requests": "import requests (not an allowed module)",
        "import subprocess": "import subprocess (not an allowed module)",
        "import os\nos.system('ls')": "import os (not an allowed module)",
        "import ctypes": "import ctypes (not an allowed module)",
        "x = open('f')": "open (file access)",
        "import numpy as np\nx = np.load('f.npy')": ".load (file access)",
        "from numpy import save": "from numpy import save (file access)",
        "import pandas as pd\nx = pd.read_csv('f')": ".read_csv (file access)",
        "from skimage import io": "import skimage.io (file access)",
        "import torch\nm = torch.hub.load('a', 'b')": ".hub (pretrained weights)",
        "m = __import__('os')": "__import__ (a dynamic import)",
        "eval('1')": "eval (dynamic code)",
        "exec('x = 1')": "exec (dynamic code)",
        "from . import helper": "a relative import",
        "def features(:": "SyntaxError",
    }
    for source, expected in rejected.items():
        found = llm_arms.static_check(source)
        assert found and any(expected in line for line in found), (source, found)
    allowed = ("import numpy as np\nimport scipy.ndimage as ndi\nfrom skimage.measure import regionprops\n"
               "import torch\nimport math, re, warnings\npattern = re.compile('a')\n"
               "net = torch.nn.Linear(2, 2)\nnet.eval()\nx = np.zeros(3).tolist()\n")
    assert llm_arms.static_check(allowed) == [] and llm_arms.static_check(TINY_MODULE) == []


def test_the_llm_arm_sends_a_static_check_rejection_back_and_the_model_corrects_it(prepared, stub, llm_env, tmp_path,
                                                                                   popen_calls):
    root, _, _ = prepared
    bad = "import os\nimport numpy as np\n\n\ndef features(crops):\n    return np.load(open(os.devnull))\n"
    engine = ScriptedEngine([_module(bad), _module(TINY_MODULE)])
    run_dir = tmp_path / "llm_static"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert (final["use"], final["status"]) == ("attempt_2", "delivered")
    assert len(popen_calls) == 1                                       # the rejected module never ran
    error = engine.prompts[1].split("Your previous module could not be used: ")[1]
    assert error.startswith("the static check rejected the module: ")
    assert "line 1: import os (not an allowed module)" in error and "line 6: open (file access)" in error
    assert "line 6: .load (file access)" in error and "Write a corrected module." in error
    assert f"Your previous module:\n```python\n{bad}\n```\nWrite a corrected module." in error   # its own reply
    assert [a["stage"] for a in _attempts(run_dir)] == ["static_check", "done"]
    assert [r["kind"] if r["error"] else None for r in _requests(run_dir)] == ["static_check", None]
    assert (run_dir / "modules" / "attempt_1.py").read_text() == bad


def test_the_llm_arm_retries_a_crash_and_a_gateway_error_then_delivers(prepared, stub, llm_env, tmp_path):
    root, _, _ = prepared
    crash = ("import numpy as np\n\n\ndef features(crops):\n    total = crops.sum()\n"
             "    raise RuntimeError(f'boom after {total > 0}')\n")
    engine = ScriptedEngine([_module(crash), {"error": "rate_limit", "message": "slow down"}, _module(TINY_MODULE)])
    run_dir = tmp_path / "llm_retry"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert (final["use"], final["status"]) == ("attempt_3", "delivered") and llm_env == [30]
    error = engine.prompts[1].split("Your previous module could not be used: ")[1]
    assert error.startswith("the smoke run (crops 1-180) raised an error:")
    assert "RuntimeError: boom after True" in error and 'File "features_module.py", line 6' in error
    assert "raise RuntimeError" in error                               # the traceback shows the module's line
    assert len(error) < llm_arms.ERROR_CHARS + 200 and "gateway error: rate_limit" in engine.prompts[2]
    assert [r["kind"] if r["error"] else None for r in _requests(run_dir)] == ["crash", "gateway", None]
    assert [a["stage"] for a in _attempts(run_dir)] == ["crash", "done"]
    assert json.loads((run_dir / "submission" / "EMBED.json").read_text())["attempt"] == 3
    assert stub.calls == []


def test_three_failed_attempts_fail_the_llm_run_without_a_fallback(prepared, stub, llm_env, tmp_path):
    root, _, _ = prepared
    engine = ScriptedEngine([_module("import numpy as np\nraise ValueError('at import')\n"),
                             _module("import numpy as np\n\n\ndef feature(crops):\n    return crops\n"),
                             _module("import numpy as np\n\n\ndef features(crops):\n    return (\n")])
    run_dir = tmp_path / "llm_failed"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine, max_attempts=3)
    assert (final["use"], final["status"], final["case"]) == (None, "failed", "no_valid_module")
    assert "no usable module in 3 attempt(s)" in final["reason"] and "SyntaxError" in final["reason"]
    assert "the module import raised an error:" in engine.prompts[1] and "ValueError: at import" in engine.prompts[1]
    assert "the module defines no callable features(crops)" in engine.prompts[2]
    assert [a["stage"] for a in _attempts(run_dir)] == ["crash", "crash", "static_check"]
    assert not (run_dir / "submission" / "embeddings.npy").exists() and engine.replies == []
    agent = json.loads((run_dir / "AGENT_RUN.json").read_text())
    assert (agent["attempts"], agent["stopped_because"]) == (3, "no_valid_module")


def test_the_llm_arm_rejects_an_invalid_output(prepared, stub, llm_env, tmp_path, monkeypatch):
    root, _, _ = prepared
    returns = {"np.zeros((len(crops) - 1, 4))": "shape (179, 4), expected (n, d) = (180, d) for a chunk of 180 crops",
               "np.zeros((len(crops), 5000))": "d = 5000, expected 1 <= d <= 4096",
               "np.zeros((len(crops), 4), dtype=np.int64)": "dtype int64, expected a float array"}
    engine = ScriptedEngine([_module(f"import numpy as np\n\n\ndef features(crops):\n    return {r}\n")
                             for r in returns])
    run_dir = tmp_path / "llm_invalid"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine, max_attempts=3)
    assert (final["status"], final["case"]) == ("failed", "no_valid_module")
    errors = [r["error"] for r in _requests(run_dir)]
    for error, problem in zip(errors, returns.values()):
        assert error == f"the smoke run (crops 1-180) returned an invalid output: {problem}"
    assert [r["kind"] for r in _requests(run_dir)] == ["invalid_output"] * 3
    # in chunks: every chunk must return the d of the smoke run; the rows keep the cells.csv order across chunks
    monkeypatch.setattr(llm_arms, "FEATURE_CHUNK", 64)
    varying = ("import numpy as np\n\n\ndef features(crops):\n"
               "    return np.zeros((len(crops), 3 if len(crops) == 64 else 4))\n")
    engine = ScriptedEngine([_module(varying), _module(TINY_MODULE)])
    run_dir = tmp_path / "llm_chunks"
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert final["status"] == "delivered"
    assert ("the full run (crops 1-64 of 180) returned an invalid output: d = 3, while the earlier chunks returned "
            "d = 4") in engine.prompts[1]
    crops = _crops_in_order(root)
    by_chunk = np.concatenate([_tiny_features(crops[i:i + 64]) for i in range(0, len(crops), 64)])   # NaN per chunk
    np.testing.assert_array_equal(np.load(run_dir / "submission" / "embeddings.npy"), by_chunk.astype(np.float32))


def test_a_module_past_the_time_budget_is_stopped_and_fails_the_run(prepared, stub, llm_env, tmp_path, monkeypatch):
    """SHAPE's wall-clock rule (A5 as amended): a module runs until RESERVE_MINUTES before the job's deadline, then it
    is stopped and the run ends; no attempt starts after that."""
    import time as clock

    from shape.agent.verification.verifiers.representation import RESERVE_MINUTES

    root, _, _ = prepared
    assert llm_arms.RESERVE_MINUTES == RESERVE_MINUTES                 # SHAPE's per-call reserve
    # 90 s for the arm: the module process's start (PREIMPORTS: torch, scipy, ...) must end inside it on a loaded
    # host, so the time runs out in the smoke run; the module sleeps far past it, so the bound below shows the stop.
    end = clock.time() + RESERVE_MINUTES * 60 + 90
    monkeypatch.setattr(llm_arms, "feature_resources", lambda: {"deadline": end, "cores": 16, "gpus": 1})
    slow = "import time\n\n\ndef features(crops):\n    time.sleep(600)\n"
    engine = ScriptedEngine([_module(slow), _module(TINY_MODULE)])
    run_dir = tmp_path / "llm_timeout"
    started = clock.time()
    final = llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert clock.time() - started < 300                                # the process was stopped, not waited for
    assert (final["status"], final["case"]) == ("failed", "no_valid_module")
    assert "the smoke run (crops 1-180) was still running 15 min before the job's time limit" in final["reason"]
    assert final["reason"].endswith("then the time budget was spent")
    assert len(engine.prompts) == 1 and len(engine.replies) == 1       # no attempt starts after the budget
    assert [a["stage"] for a in _attempts(run_dir)] == ["timeout"]
    stated = re.search(r"Time: the job ends in (\d+\.\d\d) h; a module still running 15 min before then is stopped\.",
                       engine.prompts[0])
    assert stated and abs(float(stated.group(1)) - (RESERVE_MINUTES * 60 + 90) / 3600) <= 0.011
    meta = json.loads((run_dir / "RUN_META.json").read_text())
    assert meta["resources"]["deadline_utc"] and meta["max_attempts"] == 8


def test_the_static_check_rejects_interpreter_access_and_process_pools():
    for source, expected in {"x = getattr(np, 'zeros')": "getattr (interpreter access)",
                             "t = (1).__class__.__mro__": ".__class__ (interpreter access)",
                             "import concurrent.futures": "import concurrent.futures (not an allowed module)",
                             "from numpy import DataSource": "from numpy import DataSource (file access)",
                             "import torch\nx = torch.utils.cpp_extension": ".cpp_extension (native code)",
                             "import torch\nd = torch.distributed": ".distributed (a subprocess or network)",
                             "v = vars()": "vars (interpreter access)"}.items():
        assert any(expected in line for line in llm_arms.static_check(source)), (source,
                                                                                 llm_arms.static_check(source))
    assert "concurrent" not in llm_arms.ALLOWED_STANDARD
    assert llm_arms.audit_tokens(TINY_MODULE) == []
    assert llm_arms.audit_tokens("x = 1  # we never open files or call subprocess\n") == ["open", "subprocess"]


def test_the_module_process_audit_hook_blocks_benchmark_files_and_network_but_not_the_packages(tmp_path):
    results = tmp_path / "results_root"
    (results / "run").mkdir(parents=True)
    secret = results / "run" / "labels.csv"
    secret.write_text("cell_id,sirna\n", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_bytes(llm_arms.WORKER.encode("utf-8"))
    images = np.random.default_rng(0).integers(0, 255, size=(8, 6, 64, 64), dtype=np.uint8)

    def run(source: str):
        return llm_arms.run_module(source, images, np.arange(8), worker=worker, stderr_path=tmp_path / "stderr.log",
                                   deadline=__import__("time").time() + 120, deny=[results])

    reader = (f"import numpy as np\n\n\ndef features(crops):\n    with open({str(secret)!r}) as handle:\n"
              "        handle.read()\n    return np.zeros((len(crops), 2))\n")
    with pytest.raises(llm_arms.ModuleFailed, match="blocked by the harness: open"):
        run(reader)
    caller = ("import socket\nimport numpy as np\n\n\ndef features(crops):\n"
              "    socket.socket().connect(('127.0.0.1', 9))\n    return np.zeros((len(crops), 2))\n")
    with pytest.raises(llm_arms.ModuleFailed, match="blocked by the harness: socket.connect"):
        run(caller)
    ordinary = ("import numpy as np\nfrom scipy import ndimage\nfrom skimage import filters, measure\n\n\n"
                "def features(crops):\n    out = []\n    for crop in crops.astype(np.float64):\n"
                "        smooth = ndimage.gaussian_filter(crop[0], 2.0)\n"
                "        mask = smooth > filters.threshold_otsu(smooth)\n"
                "        regions = measure.regionprops(measure.label(mask))\n"
                "        out.append([crop.mean(), float(len(regions)), float(mask.sum())])\n"
                "    return np.array(out)\n")
    vectors, timing = run(ordinary)                                 # lazy imports of the packages keep working
    assert vectors.shape == (8, 3) and timing["dim"] == 3
    writer = ("import numpy as np\nimport tempfile\n\n\ndef features(crops):\n"
              "    with tempfile.TemporaryFile() as handle:\n        handle.write(b'x')\n"
              "    return np.zeros((len(crops), 1))\n")
    assert run(writer)[0].shape == (8, 1)                           # its own temporary directory stays usable


def test_the_llm_arm_refuses_a_job_without_its_budget(prepared, stub, llm_env, tmp_path, monkeypatch):
    import time as clock

    root, _, _ = prepared
    for resources, match in (({"deadline": clock.time() + 10 * 60, "cores": 16, "gpus": 1}, "the job ends"),
                             ({"deadline": None, "cores": 8, "gpus": 1}, "8 CPU cores visible"),
                             ({"deadline": None, "cores": 32, "gpus": 0}, "0 GPUs visible")):
        monkeypatch.setattr(llm_arms, "feature_resources", lambda resources=resources: resources)
        run_dir = tmp_path / f"llm_refused_{resources['cores']}_{resources['gpus']}"
        with pytest.raises(SystemExit, match=f"NOT_READY: {match}"):
            llm_arms.run(root, run_dir, arm="llm", cell_type="HEPG2", model="stub", engine=ScriptedEngine([]))
        assert not (run_dir / "RUN_META.json").exists()
    monkeypatch.setattr(llm_arms, "feature_resources",                 # any deadline past the reserve
                        lambda: {"deadline": clock.time() + 30 * 60, "cores": 16, "gpus": 1})
    final = llm_arms.run(root, tmp_path / "llm_ok", arm="llm", cell_type="HEPG2", model="stub",
                         engine=ScriptedEngine([_module(TINY_MODULE)]))
    assert final["status"] == "delivered"
    meta = json.loads((tmp_path / "llm_ok" / "RUN_META.json").read_text())
    assert meta["resources"]["deadline_utc"] and meta["resources"]["cores"] == 16


def test_the_llm_arm_prompts_withhold_val_history_proxy_values_and_shapes_readouts(prepared, stub, llm_env,
                                                                                  tmp_path):
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    from benchmarks.representation_RxRx1.runners.prepare import load

    root, _, source = prepared
    llm = ScriptedEngine([_module("import os\n"), _module(TINY_MODULE)])     # its only feedback: the error text
    llm_arms.run(root, tmp_path / "llm", arm="llm", cell_type="HEPG2", model="stub", engine=llm)
    summary, cells = load(root)
    shape_verifier = RepresentationAgenticVerifier()
    # what SHAPE's prompts carry and this arm must not: the stable metric set, the guard, the plateau, the input
    withheld = [paragraph.splitlines()[0] for paragraph in shape_verifier.readouts_text().split("\n\n")]
    withheld += [shape_verifier.final_rule_text().splitlines()[0], shape_verifier.admissible_text().splitlines()[0],
                 shape_verifier.plateau_text().splitlines()[0], "Input (trial 0", "ALARMS", "comparison draws",
                 "levers no call has changed yet"]
    # A12: SHAPE's goal and basis rule are its verifier's prompt text
    withheld += [shape_verifier.goal_text().splitlines()[0], shape_verifier.basis_text().splitlines()[0],
                 "Changing levers is the means"]
    for text in llm.prompts:
        assert isinstance(text, str)
        assert "val_history" not in text and "sibling_site_map" not in text and "eval_loss" not in text
        assert str(PROXY_SENTINEL) not in text and str(EVAL_SENTINEL) not in text and '"alpha"' not in text
        assert "resume_best" not in text and ".pt" not in text           # no checkpoint path
        assert not [w for w in withheld if w in text], [w for w in withheld if w in text]
        assert not _leaks(text)
    # llm: no CARD, no analyzer, no CellProfiler-arm code, no SHAPE task text, no metadata, no result
    task = shape_arm.task_brief(summary, "HEPG2", 1)
    identities = {*source.plate, *source.experiment, *source.site_id, *source.well_unit}
    assert len(llm.prompts) == 2 and "the static check rejected the module" in llm.prompts[1]
    for text in llm.prompts:
        assert "## Levers" not in text and "### " not in text and task not in text and "trainable_pct" not in text
        assert not [w for w in ("analyzer", "Cell_State", "DINOv3", "CellProfiler", "cp_measure", "val wells",
                                "proxy", "mAP", "plate") if w in text]
        assert not [i for i in identities if i in text]
    assert llm.prompts[0] in llm.prompts[1]                              # the same task text, the error appended


def test_score_with_the_llm_and_react_runs(prepared, stub, llm_env, fake_cp_measure, tmp_path):
    root, summary, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    shape_run = tmp_path / "shape_run"
    shape_arm.run(root, shape_run, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    shape_meta_path = shape_run / "RUN_META.json"
    shape_meta = json.loads(shape_meta_path.read_text())
    assert shape_meta["card_sha256"] == control.card_sha256() and shape_meta["model"] == "stub"   # the arms' model
    llm_run, react_run = tmp_path / "llm_run", tmp_path / "react_run"
    llm_arms.run(root, llm_run, arm="llm", cell_type="HEPG2", model="stub",
                 engine=ScriptedEngine([_module(TINY_MODULE)]), max_attempts=2)            # the SHAPE run's max_rounds
    react_arm.run(root, react_run, cell_type="HEPG2", model="stub", budget=2,               # the SHAPE run's max_rounds
                  engine=ReactEngine([react_call(view_pairing="same_site"), react_finish("agent_1")]))
    names = ["shape_vs_llm", "llm_vs_frozen", "llm_vs_default_mean", "shape_vs_react", "react_vs_default_mean",
             "react_vs_frozen"]

    without = score.final(root, shape_run)                         # neither given: unchanged
    assert not {"llm_arms", "llm_comparisons", "efficiency"} & set(without)
    assert not {"llm", "react"} & set(without["tables"])
    assert not any("llm" in c or "react" in c for c in without["groups"]["seen"]["comparisons"])
    assert "## LLM arms" not in (shape_run / "RESULT.md").read_text(encoding="utf-8")

    n_calls = len(stub.calls)
    result = score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    for tables in (result["tables"], result["matched"]["tables"]):
        assert tables["llm"]["seen"]["value"] is not None and tables["react"]["seen"]["value"] is not None
    for groups in (result["groups"], result["matched"]["groups"]):
        for group in ("seen", "heldout"):
            assert set(names) <= set(groups[group]["comparisons"])
    assert result["llm_comparisons"]["names"] == names
    assert result["llm_comparisons"]["read"] == {**dict.fromkeys(names[:3], "matched"), **dict.fromkeys(names[3:],
                                                                                                      "native")}
    for table, groups in (("native", result["groups"]), ("matched", result["matched"]["groups"])):
        assert result["llm_comparisons"][table]["seen"]["shape_vs_llm"] == groups["seen"]["comparisons"]["shape_vs_llm"]
    # llm is an external feature arm: its own embeddings.npy (no analyzer call), native = one global centre-scale,
    # matched = the per-plate one
    # react's delivery only, and only when its checkpoint's content is new to the score cache (the stand-in's state
    # is its alpha alone, so it can equal an arm scored before)
    react_sha = prep_mod.sha256_file(Path(result["llm_arms"]["react"]["checkpoint"]))
    scored_before = {prep_mod.sha256_file(Path(c["init_checkpoint"])) for c in stub.calls[:n_calls]
                     if c.get("embed_only")}
    assert len(stub.calls) == n_calls + (react_sha not in scored_before)
    assert all(c.get("embed_only") and c["init_checkpoint"] == result["llm_arms"]["react"]["checkpoint"]
               for c in stub.calls[n_calls:])
    _, cells = prep_mod.load(root)
    raw = np.load(llm_run / "submission" / "embeddings.npy")
    sirna = score.read_labels(root, cells, summary)
    plates = cells.plate.to_numpy()
    native = score.evaluate(score.centre_scale(raw, np.zeros(len(raw))), cells, sirna)
    matched = score.evaluate(score.centre_scale(raw, plates), cells, sirna)
    for group in ("seen", "heldout"):
        assert result["tables"]["llm"][group]["value"] == native[group]["value"]
        assert result["matched"]["tables"]["llm"][group]["value"] == matched[group]["value"]
    assert result["postprocess"]["read"]["llm_comparisons"] == "matched"
    assert result["reading"] == without["reading"]                 # the pre-registered reading is unchanged
    assert {a: r["status"] for a, r in result["llm_arms"].items()} == {"llm": "delivered", "react": "delivered"}
    assert result["llm_arms"]["react"]["use"] == "agent_1" and result["llm_arms"]["llm"]["use"] == "attempt_1"
    assert result["llm_arms"]["llm"]["embedding"]["dim"] == 12
    eff = result["efficiency"]
    assert set(eff) == {"shape", "llm", "react"}
    assert (eff["llm"]["tool_calls"], eff["react"]["tool_calls"]) == (0, 2)          # react: trial 0 and agent_1
    assert eff["shape"]["tool_calls"] == eff["shape"]["training_calls"] + 1          # trial 0 is an analyzer call
    assert (eff["llm"]["reasoner_tokens"], eff["react"]["reasoner_tokens"]) == (STUB_TOKENS, 2 * STUB_TOKENS)
    assert all(e["wall_hours"] is not None for e in eff.values())
    text = (shape_run / "RESULT.md").read_text(encoding="utf-8")
    assert "## LLM arms (contract amendments A4, A5; llm read on the matched table" in text and "## Efficiency" in text
    assert "| seen | shape_vs_llm | matched |" in text and "| heldout | react_vs_frozen | native |" in text
    assert "| llm |" in text and "llm: attempt_1 (case: features, d = 12)" in text
    assert "llm_arms_warnings" not in result and "Warning:" not in text

    # with the cellprofiler arm scored: llm vs cellprofiler too, read on the matched table
    _external_run(root, tmp_path, ("cellprofiler",))
    result = score.final(root, shape_run, llm_run=llm_run)
    assert result["llm_comparisons"]["names"] == ["shape_vs_llm", "llm_vs_frozen", "llm_vs_default_mean",
                                                  "llm_vs_cellprofiler"]
    assert result["llm_comparisons"]["read"]["llm_vs_cellprofiler"] == "matched"
    band = result["matched"]["groups"]["seen"]["comparisons"]["llm_vs_cellprofiler"]
    assert band["delta"] is not None and band["n_units_paired"] > 0
    shutil.rmtree(root / "external")

    # react delivering frozen: its comparisons are frozen's, exactly
    frozen_run = tmp_path / "react_frozen"
    react_arm.run(root, frozen_run, cell_type="HEPG2", model="stub", budget=2,
                  engine=ReactEngine([react_finish("default")]))
    result = score.final(root, shape_run, react_run=frozen_run)
    for groups in (result["groups"], result["matched"]["groups"]):
        for group in ("seen", "heldout"):
            c = groups[group]["comparisons"]
            assert c["react_vs_frozen"]["delta"] == 0.0 and c["react_vs_frozen"]["band_2se"] == 0.0
            assert c["shape_vs_react"] == c["shape_vs_frozen"]
            assert c["react_vs_default_mean"] == c["default_mean_vs_frozen"] | {
                "delta": -c["default_mean_vs_frozen"]["delta"]}

    # a failed run is reported, not scored
    failed_run = tmp_path / "llm_failed"
    llm_arms.run(root, failed_run, arm="llm", cell_type="HEPG2", model="stub",
                 engine=ScriptedEngine([_module("import os\n")] * 3), max_attempts=2)
    result = score.final(root, shape_run, llm_run=failed_run, react_run=react_run)
    assert result["llm_arms"]["llm"]["status"] == "failed" and "llm" not in result["tables"]
    assert result["llm_comparisons"]["names"] == ["shape_vs_react", "react_vs_default_mean", "react_vs_frozen"]
    assert "llm: failed (no_valid_module" in (shape_run / "RESULT.md").read_text(encoding="utf-8")

    # refused: another prep or code (provenance, react_arm.py, the shared runner), another model, react's budget or
    # CARD, another arm
    path = react_run / "RUN_META.json"
    meta = json.loads(path.read_text())
    path.write_text(json.dumps({**meta, "provenance": {**meta["provenance"], "tool_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="ARM_STALE: the react run .*tool_sha256"):
        score.final(root, shape_run, react_run=react_run)
    path.write_text(json.dumps({**meta, "provenance": {**meta["provenance"], "prep_cells_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="ARM_STALE: .*prep_cells_sha256"):
        score.final(root, shape_run, react_run=react_run)
    for change, match in (({"model": "gpt-other"}, "model"), ({"tool_budget": 8}, "tool_budget"),
                          ({"card_sha256": "0" * 64}, "card_sha256"),
                          ({"react_arm_sha256": "0" * 64}, "react_arm_sha256"),     # e.g. a run of another design
                          ({"react_arm_sha256": None}, "react_arm_sha256"),         # a run of the A4 / A11 react
                          ({"shared_react_sha256": "0" * 64}, "shared_react_sha256")):
        path.write_text(json.dumps({**meta, **change}))
        with pytest.raises(SystemExit, match=f"ARM_STALE: the react run .*{match}"):
            score.final(root, shape_run, react_run=react_run)
    path.write_text(json.dumps(meta))
    llm_path = llm_run / "RUN_META.json"
    llm_meta = json.loads(llm_path.read_text())
    llm_path.write_text(json.dumps({**llm_meta, "llm_arms_sha256": "0" * 64}))    # A11: the file alone is not
    score.final(root, shape_run, llm_run=llm_run, react_run=react_run)               # the llm design
    llm_path.write_text(json.dumps({**llm_meta, "worker_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="ARM_STALE: the llm run .*worker_sha256"):
        score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    llm_path.write_text(json.dumps({**llm_meta, "max_attempts": 8}))             # not the SHAPE run's max_rounds
    with pytest.raises(SystemExit, match="ARM_STALE: the llm run .*max_attempts"):
        score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    prov = llm_meta["provenance"]                       # A12: the analyzer's code does not bind the llm arm ...
    llm_path.write_text(json.dumps({**llm_meta, "provenance": {**prov, "tool_sha256": "0" * 64}}))
    score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    llm_path.write_text(json.dumps({**llm_meta, "provenance": {**prov, "prep_cells_sha256": "0" * 64}}))
    with pytest.raises(SystemExit, match="ARM_STALE: the llm run .*prep_cells_sha256"):   # ... its prep does
        score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    llm_path.write_text(json.dumps(llm_meta))
    embed_path = llm_run / "submission" / "EMBED.json"
    embed = json.loads(embed_path.read_text())
    embed_path.write_text(json.dumps({**embed, "cells_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="ARM_STALE: the llm run .*another prep"):
        score.final(root, shape_run, llm_run=llm_run)
    embed_path.write_text(json.dumps(embed))
    for change, match in (({"crop_array_sha256": "0" * 64}, "another images.npy"),
                          ({"n_cells": embed["n_cells"] - 1}, "another prep")):
        embed_path.write_text(json.dumps({**embed, **change}))
        with pytest.raises(SystemExit, match=f"ARM_STALE: the llm run .*{match}"):
            score.final(root, shape_run, llm_run=llm_run)
    embed_path.write_text(json.dumps(embed))
    changed = raw.copy()
    changed[0, 1] += 1.0
    np.save(llm_run / "submission" / "embeddings.npy", changed)
    with pytest.raises(SystemExit, match="INTEGRITY_FAILED"):
        score.final(root, shape_run, llm_run=llm_run)
    np.save(llm_run / "submission" / "embeddings.npy", raw)
    # a degenerate output: one well's cells all NaN (0 after either post-processing) leaves a well profile of zero
    # norm; llm is reported as unscorable, not scored, and the SHAPE run's RESULT is still written
    emb = llm_run / "submission" / "embeddings.npy"
    degenerate = raw.copy()
    degenerate[(cells.well == cells.well.iloc[0]).to_numpy()] = np.nan
    np.save(emb, degenerate)
    embed_path.write_text(json.dumps({**embed, "embeddings_sha256": prep_mod.sha256_file(emb)}))
    result = score.final(root, shape_run, llm_run=llm_run, react_run=react_run)
    llm_record = result["llm_arms"]["llm"]
    assert llm_record["status"] == "unscorable" and llm_record["zero_norm_wells"] == {"native": 1, "matched": 1}
    assert "llm" not in result["tables"] and "llm" not in result["matched"]["tables"]
    assert result["llm_comparisons"]["names"] == ["shape_vs_react", "react_vs_default_mean", "react_vs_frozen"]
    assert ("llm: unscorable (attempt_1: well profiles of zero norm after the post-processing, native 1, matched 1; "
            "not scored)") in (shape_run / "RESULT.md").read_text(encoding="utf-8")
    np.save(emb, raw)
    embed_path.write_text(json.dumps(embed))
    # A pre-correction SHAPE run, or a ReAct run with another fixed cap, is never compared.
    shape_meta_path.write_text(json.dumps({k: v for k, v in shape_meta.items() if k != "training_budget"}))
    with pytest.raises(SystemExit, match="ARM_STALE: SHAPE training_budget"):
        score.final(root, shape_run, react_run=react_run)
    shape_meta_path.write_text(json.dumps(shape_meta))
    path.write_text(json.dumps({**meta, "fixed_levers": {"max_epochs": 60}}))
    with pytest.raises(SystemExit, match="ARM_STALE: .*fixed_levers"):
        score.final(root, shape_run, react_run=react_run)
    path.write_text(json.dumps(meta))
    # the export of the llm run carries its modules (the sources it ran) next to its logs
    import tarfile

    from benchmarks.batch_correction_JUMP.runners.trace_md import export

    with tarfile.open(export(llm_run)) as tar:
        names = tar.getnames()
    assert f"{llm_run.name}/modules/attempt_1.py" in names and f"{llm_run.name}/modules/worker.py" in names
    assert f"{llm_run.name}/submission/EMBED.json" in names and not any(n.endswith(".npy") for n in names)
    with pytest.raises(SystemExit, match="WRONG_ARM"):
        score.final(root, shape_run, llm_run=react_run)
    # A7: with either LLM arm, a SHAPE run whose CARD.md sha256 is missing or not the current one (e.g. a 9b86927e
    # SHAPE run with A7 arms) is refused; without the LLM arms it is still scored
    for card in (None, "0" * 64):
        shape_meta_path.write_text(json.dumps({**shape_meta, "model": "stub", "card_sha256": card}))
        for arms in ({"llm_run": llm_run}, {"react_run": react_run}, {"llm_run": llm_run, "react_run": react_run}):
            with pytest.raises(SystemExit, match="ARM_STALE: the SHAPE run .*records CARD.md sha256"):
                score.final(root, shape_run, **arms)
        assert "llm_arms" not in score.final(root, shape_run)
    shape_meta_path.write_text(json.dumps({**shape_meta, "model": "stub"}))


def test_gpu_smoke_checks_the_cap_and_resume_before_its_success_marker(prepared, stub, tmp_path, monkeypatch):
    import torch
    from benchmarks.representation_RxRx1.runners import smoke_gpu

    root, _, _ = prepared
    monkeypatch.setenv("RXRX_EXPECTED_COMMIT", control.git_commit())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "diagnostic stand-in")
    real = stub.execute

    def execute(self, **kwargs):
        result = real(self, **kwargs)
        result["training_config"].update(max_epochs_hard_cap=kwargs["max_epochs"],
                                         resume={"epoch": 1} if kwargs.get("init_checkpoint") else {})
        return result

    monkeypatch.setattr(stub, "execute", execute)
    run_dir = tmp_path / "gpu_smoke"
    record = smoke_gpu.run(root, run_dir)
    assert (run_dir / "SMOKE_COMPLETE").read_text().strip() == control.git_commit()
    assert record["diagnostic_only"] and record["training_budget"]["n_train"] == 24
    first, second = record["calls"]
    cap = record["training_budget"]["max_epochs"]
    assert first["kwargs"]["max_epochs"] == second["kwargs"]["max_epochs"] == cap
    assert second["kwargs"]["init_checkpoint"] == first["resume_best"]
    assert not _leaks(record)

    def bad_cap(self, **kwargs):
        result = execute(self, **kwargs)
        result["training_config"]["max_epochs_hard_cap"] = 60
        return result

    monkeypatch.setattr(stub, "execute", bad_cap)
    failed_dir = tmp_path / "gpu_smoke_failed"
    with pytest.raises(SystemExit, match="SMOKE_FAILED"):
        smoke_gpu.run(root, failed_dir)
    assert not (failed_dir / "SMOKE_COMPLETE").exists()


def test_an_llm_run_is_stale_only_when_its_llm_design_differs(tmp_path):
    """A11: an llm run is matched by its own design (first prompt, worker, recorded settings), so a change of react's
    code in the shared llm_arms.py leaves it scorable; a changed prompt or setting makes it stale."""
    import hashlib
    from types import SimpleNamespace

    run = tmp_path / "llm_HEPG2_x"
    (run / "logs").mkdir(parents=True)
    meta = {"cell_type": "HEPG2", "n_cells": 123, "time_rule": llm_arms.TIME_RULE,
            "compute": {"cpus": llm_arms.FEATURE_CPUS, "gpus": llm_arms.FEATURE_GPUS}, "chunk": llm_arms.FEATURE_CHUNK,
            "smoke_crops": llm_arms.SMOKE_CROPS, "max_dim": llm_arms.MAX_DIM, "max_attempts": 8,
            "worker_sha256": hashlib.sha256(llm_arms.WORKER.encode("utf-8")).hexdigest(),
            "llm_arms_sha256": "an earlier file"}
    import time as clock

    brief = llm_arms.feature_brief(123, "HEPG2", 8, clock.time() + 3600)     # its clock line is masked
    prompt = llm_arms.Features.prompt(SimpleNamespace(brief=brief, last_source=None), None)
    requests = run / "logs" / "requests.jsonl"
    requests.write_text(json.dumps({"step": 1, "try": 1, "prompt": prompt}) + "\n", encoding="utf-8")
    assert llm_arms.llm_design_problems(run, meta) == {}
    assert set(llm_arms.llm_design_problems(run, {**meta, "max_dim": 1})) == {"max_dim"}
    assert set(llm_arms.llm_design_problems(run, {**meta, "max_attempts": 3})) == {"prompt"}   # its task text
    old = {k: v for k, v in meta.items() if k != "time_rule"}                    # a run of the fixed 5 h budget
    assert set(llm_arms.llm_design_problems(run, {**old, "time_budget_seconds": 5 * 3600})) == {"time_rule"}
    requests.write_text(json.dumps({"step": 1, "try": 1, "prompt": prompt + "x"}) + "\n", encoding="utf-8")
    assert set(llm_arms.llm_design_problems(run, meta)) == {"prompt"}


# ---- the per-arm score cache (score.ArmScores, stage score-arm) -----------------------------------------------

PRE_CACHE = "d11f5836d0a8d9882c4bb675cbeb4c30c15a8fdd"     # the last commit before the per-arm score cache


# A23 renamed the control family default_s<k> to dinov3 / dinov3_s<k> ("scoring unchanged"): the pre-cache score.py
# reads that family under the new name, so it scores the same arms as the current one.
A23_FAMILY = {'a.startswith("default_s")': "control.is_dinov3(a)"}


def _score_at(commit: str, monkeypatch, replace: dict | None = None):
    """score.py at `commit` (git show), loaded as its own module, each `replace` source (found exactly once) replaced;
    skipped without git."""
    import importlib.util
    import subprocess

    try:
        text = subprocess.run(["git", "show", f"{commit}:benchmarks/representation_RxRx1/score.py"],
                              cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
                              encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {commit[:8]} unavailable")
    for old, new in (replace or {}).items():
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    # its lazy imports name the modules before the move into runners/
    text = text.replace("benchmarks.representation_RxRx1", "benchmarks.representation_RxRx1.runners")
    name = f"score_{commit[:8]}"
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    monkeypatch.setitem(sys.modules, name, module)             # its dataclass looks its module up
    exec(compile(text, f"{name}.py", "exec"), module.__dict__)
    return module


def _outputs(run_dir: Path) -> tuple[str, str]:
    """RESULT.json (without `secondary.seconds`, the wall time of that step) and RESULT.md, as written."""
    record = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
    record["secondary"].pop("seconds")
    return json.dumps(record, indent=1), (run_dir / "RESULT.md").read_text(encoding="utf-8")


OLD_REACT_TEXT = "react: given the task and the CARD levers only."
NEW_REACT_TEXT = ("react: the shared ReAct runner -- SHAPE's task text, the tool's parameters (name, type, default) "
                  "and the tool's own outputs, no CARD text.")


def _without_a14(outputs: tuple[str, str]) -> tuple[dict, str]:
    """RESULT.json / RESULT.md without what amendment A14 changed in them alone: the react record's
    `react_arm_sha256` key and the react sentence of RESULT.md (the pre-cache score.py writes neither)."""
    record = json.loads(outputs[0])
    for arm in (record.get("llm_arms") or {}).values():
        arm.pop("react_arm_sha256", None)
    return record, outputs[1].replace(NEW_REACT_TEXT, OLD_REACT_TEXT)


def _refuse_recomputation(m):
    """Every embedding, well-mAP evaluation and probe fit raises (a warm cache needs none)."""
    from benchmarks.representation_RxRx1.runners import secondary

    def refuse(*args, **kwargs):
        raise AssertionError("recomputed with a warm score cache")

    for target, name in ((score, "embed"), (score, "evaluate"), (secondary.Probe, "run"),
                         (secondary, "knn_predict"), (secondary, "logreg_predict")):
        m.setattr(target, name, refuse)


def test_the_score_cache_reproduces_the_uncached_result_and_a_warm_final_recomputes_nothing(
        prepared, stub, llm_env, fake_cp_measure, tmp_path, monkeypatch):
    old_score = _score_at(PRE_CACHE, monkeypatch, replace=A23_FAMILY)
    root, _, _ = prepared
    # no default arm: the pre-cache score.py reads default_s<k>, which A23 renamed the dinov3 control arm, so with
    # one the two scorers would see different arms (the dinov3 arm's cache: the test below)
    control.control(root, seeds=(), split_key="split")
    shape_run = tmp_path / "shape_run"
    shape_arm.run(root, shape_run, cell_type="HEPG2", model="stub", max_rounds=3, engine=_shape_engine(3))
    meta_path = shape_run / "RUN_META.json"
    meta_path.write_text(json.dumps({**json.loads(meta_path.read_text()), "model": "stub"}))
    llm_run, react_run = tmp_path / "llm_run", tmp_path / "react_run"
    llm_arms.run(root, llm_run, arm="llm", cell_type="HEPG2", model="stub",
                 engine=ScriptedEngine([_module(TINY_MODULE)]), max_attempts=3)
    react_arm.run(root, react_run, cell_type="HEPG2", model="stub", budget=3,
                  engine=ReactEngine([react_call(view_pairing="same_site"), react_finish("agent_1")]))
    # the pre-cache score.py checks a react run by llm_arms.py (the A4 / A11 react); the current one ignores the key
    react_meta = react_run / "RUN_META.json"
    react_meta.write_text(json.dumps({**json.loads(react_meta.read_text()),
                                      "llm_arms_sha256": prep_mod.sha256_file(Path(llm_arms.__file__))}))
    _external_run(root, tmp_path, external.EXTERNAL_ARMS)
    runs = {"llm_run": llm_run, "react_run": react_run}

    old_score.final(root, shape_run, **runs)        # the uncached path: every arm embedded and scored in memory
    uncached = _without_a14(_outputs(shape_run))
    assert not (root / score.SCORE_CACHE).exists()
    result = score.final(root, shape_run, **runs)  # a cold cache: every arm scored into it
    cold = _outputs(shape_run)
    assert _without_a14(cold) == uncached and NEW_REACT_TEXT in cold[1]
    assert {"frozen", "shape", "react", "openphenom", "cellprofiler", "llm"} <= set(result["tables"])
    files = sorted((root / score.SCORE_CACHE).glob("*.json"))
    assert files and not list((root / score.SCORE_CACHE).glob("*.tmp"))
    n_calls = len(stub.calls)
    with monkeypatch.context() as m:                # a warm cache: no embedding, no evaluation, no probe fit
        _refuse_recomputation(m)
        score.final(root, shape_run, **runs)
    assert _outputs(shape_run) == cold and len(stub.calls) == n_calls
    assert sorted((root / score.SCORE_CACHE).glob("*.json")) == files
    # without the LLM arms the cached result is the uncached one too
    old_score.final(root, shape_run)
    uncached = _outputs(shape_run)
    with monkeypatch.context() as m:
        _refuse_recomputation(m)
        score.final(root, shape_run)
    assert _outputs(shape_run) == uncached


def test_the_score_cache_recomputes_after_a_changed_checkpoint_or_scoring_code(prepared, stub, tmp_path, monkeypatch):
    root, _, _ = prepared
    ctl = control.control(root, seeds=(0,), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "cache"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=2, engine=_shape_engine(2))
    first = score.final(root, run_dir)
    files = set((root / score.SCORE_CACHE).glob("*.json"))
    embedded, evaluated = [], []
    real_embed, real_evaluate = score.embed, score.evaluate

    def spy_embed(summary, cells, arm_dir, kwargs):
        embedded.append(arm_dir.name.split("-")[0])
        return real_embed(summary, cells, arm_dir, kwargs)

    def spy_evaluate(*args):
        evaluated.append(1)
        return real_evaluate(*args)

    monkeypatch.setattr(score, "embed", spy_embed)
    monkeypatch.setattr(score, "evaluate", spy_evaluate)
    assert score.final(root, run_dir)["tables"] == first["tables"] and embedded == evaluated == []
    # a changed checkpoint (another sha256 at the same path): that arm only is embedded and scored again
    checkpoint = Path(ctl["arms"]["dinov3"]["resume_best"])
    original = checkpoint.read_text()
    checkpoint.write_text(json.dumps({"alpha": json.loads(original)["alpha"] + 0.3}))
    n_calls = len(stub.calls)
    changed = score.final(root, run_dir)
    assert embedded == ["dinov3"] and len(evaluated) == 2 and len(stub.calls) == n_calls + 1
    assert changed["tables"]["dinov3"] != first["tables"]["dinov3"]
    assert changed["tables"]["frozen"] == first["tables"]["frozen"]
    assert len(set((root / score.SCORE_CACHE).glob("*.json")) - files) == 2      # one entry per table
    checkpoint.write_text(original)                 # the original content: its entries again, nothing recomputed
    embedded.clear(), evaluated.clear()
    assert score.final(root, run_dir)["tables"] == first["tables"] and embedded == evaluated == []
    # another score.py: every entry is recomputed (the embeddings, which do not depend on it, are reused)
    real_code = score.scoring_code
    monkeypatch.setattr(score, "scoring_code", lambda: {**real_code(), "score_sha256": "0" * 64})
    n_calls = len(stub.calls)
    again = score.final(root, run_dir)
    assert len(evaluated) == len(files) and len(stub.calls) == n_calls
    assert again["tables"] == first["tables"]
    assert len(set((root / score.SCORE_CACHE).glob("*.json"))) == 2 * len(files) + 2


def test_score_arm_fills_the_cache_that_final_reads(prepared, stub, tmp_path, monkeypatch, capsys):
    root, _, _ = prepared
    control.control(root, seeds=(0,), split_key="split")
    run_dir = tmp_path / "shape_vrr" / "HEPG2" / "runs" / "arms"
    shape_arm.run(root, run_dir, cell_type="HEPG2", model="stub", max_rounds=3, engine=_shape_engine(3))
    for retired in ("random", "default_s0"):        # default_s0: A22's control, retired by dinov3 (A23)
        with pytest.raises(SystemExit, match=f"unknown arm '{retired}'"):
            score.score_arm(root, retired)
    with pytest.raises(SystemExit, match="needs its run directory"):
        score.score_arm(root, "shape")
    capsys.readouterr()
    scored = {}
    for arm in ("frozen", "dinov3"):
        scored.update(score.score_arm(root, arm))
    scored.update(score.score_arm(root, "shape", run_dir))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("SCORE_ARM ")]
    assert [line.split(":")[0] for line in lines] == [f"SCORE_ARM {a}" for a in scored]
    assert {"frozen", "dinov3", "shape"} <= set(scored) <= {"frozen", "dinov3", "shape", "shape_top"}
    n_calls = len(stub.calls)
    with monkeypatch.context() as m:
        _refuse_recomputation(m)
        result = score.final(root, run_dir)
    assert len(stub.calls) == n_calls
    for arm, tables in scored.items():
        native = result["tables"][arm]
        assert native["seen"]["value"] == tables["native"]["evaluation"]["seen"]["value"]
        line = next(line for line in lines if line.startswith(f"SCORE_ARM {arm}:"))
        assert (f"well mAP (native) seen {native['seen']['value']:.4f} heldout {native['heldout']['value']:.4f}"
                in line)
        if arm != "shape_top":                       # analysis only: no probe
            ba = result["secondary"]["native"]["well"]["knn"]["arms"][arm]["balanced_accuracy"]["value"]
            assert line.endswith(f"balanced accuracy {ba:.4f}")
        else:                                        # no probe of its own; one shared with shape's equal state
            assert (line.endswith("balanced accuracy n/a")
                    or tables["native"] is scored["shape"]["native"] and "probe" in tables["native"])


def test_scoring_skips_the_clustering_and_embeds_the_same_vectors(prepared, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("scanpy")
    pytest.importorskip("leidenalg")
    from shape.tools.cell_state_analyzer import tool as tool_module

    root, _, _ = prepared
    stub_vit = _stub_backbone()
    weights = tmp_path / "stub_backbone.pth"
    torch.save(stub_vit().state_dict(), weights)
    monkeypatch.setattr(tool_module, "hf_hub_download", lambda **kw: str(weights))
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: stub_vit())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SHAPE_FAST_FIGURES", "1")
    summary, cells = prep_mod.load(root)
    crop_array, meta = prep_mod.tool_input(summary, cells)
    trained = control.analyzer().execute(crop_array=crop_array, cell_metadata=[dict(m) for m in meta],
                                         query_cache_dir=str(tmp_path / "train"), output_subdir="call",
                                         split_key="split", max_epochs=1, batch_size=8)
    assert "error" not in trained and trained["resume_best"]
    crop_array, meta = prep_mod.tool_input(summary, cells, splits=prep_mod.SPLITS, keys=score.EMBED_KEYS)
    for name, kwargs in (("frozen", score.FROZEN), ("delivered", {"init_checkpoint": trained["resume_best"],
                                                                  "embed_only": True})):
        clustered, scoring = (tool.execute(crop_array=crop_array, cell_metadata=[dict(m) for m in meta],
                                           query_cache_dir=str(tmp_path / which / name), **kwargs)
                              for which, tool in (("with", control.analyzer()), ("without", score.embedding_tool())))
        with_clusters, without = anndata.read_h5ad(clustered["adata_path"]), anndata.read_h5ad(scoring["adata_path"])
        assert clustered["cluster_key"] in with_clusters.obs and "X_umap" in with_clusters.obsm
        assert scoring["cluster_key"] is None and "X_umap" not in without.obsm and not without.obsp
        assert list(with_clusters.obs_names) == list(without.obs_names)
        np.testing.assert_array_equal(np.asarray(without.X), np.asarray(with_clusters.X))


# ---- A20: the SHAPE / ReAct reasoner (high effort, room for it in the token cap) -------------------------------

def test_the_reasoner_runs_at_high_effort_with_the_raised_token_cap(monkeypatch):
    """A20: `reasoner` sets SHAPE_REASONING_EFFORT=high for the engine and wraps it so every generate call carries
    max_tokens=MAX_COMPLETION_TOKENS unless the caller sets one; other attributes pass through; no engine is
    NOT_READY."""
    from benchmarks.representation_RxRx1 import run_rxrx1
    from shape.agent.verification import vision_engine

    seen = []

    class Raw:
        model_string = "gpt-5.6-luna"

        def generate(self, content, **kwargs):
            seen.append(kwargs)
            return {"ok": content}

    monkeypatch.delenv("SHAPE_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(vision_engine, "build_vision_engine", lambda model: Raw())
    engine = run_rxrx1.reasoner("gpt-5.6-luna")
    assert os.environ["SHAPE_REASONING_EFFORT"] == run_rxrx1.REASONING_EFFORT == "high"
    assert engine.max_tokens == run_rxrx1.MAX_COMPLETION_TOKENS == 16000 and engine.model_string == "gpt-5.6-luna"
    assert engine.generate("q", response_format=dict) == {"ok": "q"}
    engine.generate("q", max_tokens=500)
    assert seen == [{"response_format": dict, "max_tokens": 16000}, {"max_tokens": 500}]
    monkeypatch.setattr(vision_engine, "build_vision_engine", lambda model: None)
    with pytest.raises(SystemExit, match="NOT_READY"):
        run_rxrx1.reasoner("gpt-5.6-luna")


def test_every_llm_arm_gets_the_high_effort_reasoner_with_the_token_cap(tmp_path, monkeypatch):
    """The shape, react and llm stages each get `reasoner`'s engine: SHAPE_REASONING_EFFORT high when the arm starts
    and the MAX_COMPLETION_TOKENS cap on every generate call (the llm arm had the engine's default, medium, before)."""
    from benchmarks.representation_RxRx1 import run_rxrx1
    from benchmarks.representation_RxRx1.runners import llm_arms, react_arm
    from shape.agent.verification import vision_engine

    class Raw:
        model_string = "gpt-5.6-luna"

        def generate(self, content, **kwargs):
            return {"content": content, "kwargs": kwargs}

    seen = {}

    attempts = []

    def started(arm):
        def run(*args, engine, **kwargs):
            if arm == "llm":
                attempts.append(kwargs["max_attempts"])
            seen[arm] = (engine, os.environ.get("SHAPE_REASONING_EFFORT"), engine.generate(["q"]))
            return {"use": "frozen"}
        return run

    monkeypatch.delenv("SHAPE_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("RXRX_EXPECTED_COMMIT", raising=False)
    monkeypatch.setattr(vision_engine, "build_vision_engine", lambda model: Raw())
    monkeypatch.setattr(prep_mod, "check_assets", lambda output_root: {})
    for module, arm in ((shape_arm, "shape"), (react_arm, "react"), (llm_arms, "llm")):
        monkeypatch.setattr(module, "run", started(arm))
    for arm in ("shape", "react", "llm"):
        run_rxrx1.main([arm, "--run", str(tmp_path / arm), "--output-root", str(tmp_path / "out")])
    assert sorted(seen) == ["llm", "react", "shape"] and attempts == [8]      # llm: SHAPE's max_rounds
    for arm, (engine, effort, reply) in seen.items():
        assert isinstance(engine, run_rxrx1.Budgeted) and effort == run_rxrx1.REASONING_EFFORT == "high", arm
        assert engine.max_tokens == run_rxrx1.MAX_COMPLETION_TOKENS == 16000, arm
        assert reply["kwargs"] == {"max_tokens": 16000}, arm


def test_the_llm_run_records_its_reasoner_settings(prepared, stub, llm_env, tmp_path, monkeypatch):
    """RUN_META.json of the llm arm: reasoning_effort and max_completion_tokens, as shape_arm and react_arm record
    them (resume.STABLE_META holds both fixed across a resume)."""
    from benchmarks.representation_RxRx1 import run_rxrx1
    from benchmarks.representation_RxRx1.runners import resume

    root, _, _ = prepared
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", run_rxrx1.REASONING_EFFORT)
    engine = run_rxrx1.Budgeted(ScriptedEngine([_module(TINY_MODULE)]), run_rxrx1.MAX_COMPLETION_TOKENS)
    final = llm_arms.run(root, tmp_path / "llm", arm="llm", cell_type="HEPG2", model="stub", engine=engine)
    assert final["status"] == "delivered"
    meta = json.loads((tmp_path / "llm" / "RUN_META.json").read_text())
    assert (meta["reasoning_effort"], meta["max_completion_tokens"]) == ("high", 16000)
    assert {"reasoning_effort", "max_completion_tokens"} <= set(resume.STABLE_META)
