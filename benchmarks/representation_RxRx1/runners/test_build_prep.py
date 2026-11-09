"""CPU tests of build_prep.py (the RxRx1 prep builder) and its launchers, on a tiny fake RxRx1 zip.

- `prepare --workers N` gives the bytes of --workers 1, which is the pre-sharding algorithm (an in-test copy: one
  loop, np.stack, np.save; it pins the crop geometry and the seeded rank strings); the CLI module run as
  `python -m` with spawned workers gives the same bytes;
- `--well-types all` takes every well_type of metadata.csv, a list only those;
- the holdout guard: a held-out count that leaves no seen experiment is refused before anything is written;
- the seeded experiment order reproduces the held-out experiments of the task-116 v3 preps;
- `subsample --train-cells K` resolves the smallest k/n fraction reaching K training cells;
- compatibility: this benchmark's prepare stage accepts a built FULL prep and its run prep under the config.py
  full_v4 names (prep_set="full_v4"), and the default configuration is unchanged;
- full_v5: the well-level split (per plate floor(0.1 n + 1/2) seen wells, of them max(2, floor(0.2 n_seen + 1/2))
  val, the rest train; every other well held out; deterministic; no held-out experiment, no unused cell), `resplit
  --split wells` from a full_v4-like prep without reading an image, `subsample --train-well-fraction 1.0` (every train
  and val cell), and the config.py full_v5 names through the prepare stage and the score groups (no unused cell).
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("skimage")
pytest.importorskip("PIL")

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.representation_RxRx1.runners import build_prep  # noqa: E402
from benchmarks.representation_RxRx1.runners.build_prep import (SiteReader, _rank, assign_splits,  # noqa: E402
                                                        holdout_experiments, nucleus_centroids, stretch_site)

BENCH = REPO / "benchmarks" / "representation_RxRx1" / "runners"
TIMING = ("config", "workers", "seconds", "shard_seconds")


def _fake_rxrx1_zip(root: Path) -> tuple[Path, Path]:
    """A tiny RxRx1 zip (rxrx1/images/<experiment>/Plate<p>/<well>_s<site>_w<c>.png) and metadata.csv: 3 HUVEC
    experiments x 2 plates x 4 wells x 2 sites (and one RPE plate that is filtered out), well types treatment /
    positive_control / negative_control, 1-4 Hoechst nuclei in site 1, 0-4 in site 2."""
    from PIL import Image

    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:128, :128]
    centres = ((40, 40), (88, 88), (40, 88), (88, 40))
    wells = (("B02", "treatment", 11), ("B03", "positive_control", 1139), ("C02", "negative_control", "EMPTY"),
             ("C03", "treatment", 12))
    records = []
    archive = root / "rxrx1-images.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for cell_type, experiments, plates in (("HUVEC", (1, 2, 3), (1, 2)), ("RPE", (1,), (1,))):
            for e in experiments:
                experiment = f"{cell_type}-0{e}"
                for p in plates:
                    for well, well_type, sirna in wells:
                        for site in (1, 2):
                            n_nuclei = int(rng.integers(1 if site == 1 else 0, 5))   # every well has a cell
                            for channel in range(1, 7):
                                image = rng.integers(0, 30, size=(128, 128)).astype(np.uint8)
                                if channel == 1 and n_nuclei == 0:     # flat Hoechst: no nucleus
                                    image[:] = 10
                                elif channel == 1:
                                    for cy, cx in centres[:n_nuclei]:
                                        image[(yy - cy) ** 2 + (xx - cx) ** 2 <= 8 ** 2] = 200
                                buffer = io.BytesIO()
                                Image.fromarray(image).save(buffer, format="PNG")
                                zf.writestr(f"rxrx1/images/{experiment}/Plate{p}/{well}_s{site}_w{channel}.png",
                                            buffer.getvalue())
                            records.append({"site_id": f"{experiment}_{p}_{well}_{site}",
                                            "well_id": f"{experiment}_{p}_{well}", "cell_type": cell_type,
                                            "dataset": "train" if e < 3 else "test", "experiment": experiment,
                                            "plate": p, "well": well, "site": site, "well_type": well_type,
                                            "sirna": sirna})
    metadata = root / "metadata.csv"
    pd.DataFrame(records).sample(frac=1.0, random_state=0).to_csv(metadata, index=False)   # not in site order
    return metadata, archive


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    return _fake_rxrx1_zip(tmp_path_factory.mktemp("rxrx1"))


def _prepare_args(metadata, archive, out, *extra) -> list[str]:
    return ["prepare", "--metadata", str(metadata), "--images", str(archive), "--output", str(out),
            "--datasets", "train,test", "--n-experiments", "999", "--max-cells-per-site", "3", *extra]


def test_prepare_workers_give_the_single_process_output(fake, tmp_path, capsys):
    metadata, archive = fake
    outputs = {}
    for workers in (1, 3):
        out = tmp_path / f"prep_w{workers}"
        build_prep.main(_prepare_args(metadata, archive, out, "--well-types", "all", "--val-fraction", "0.25",
                                      "--holdout-experiments", "2", "--workers", str(workers)))
        outputs[workers] = out
        assert {p.name for p in out.iterdir()} == {"images.npy", "cells.csv", "PREPARE.json"}
        assert not (tmp_path / f"prep_w{workers}.partial").exists()
    assert "prepare: 48 sites in 3 shards of 16-16 sites" in capsys.readouterr().out
    # the pre-sharding algorithm on the same inputs
    meta = pd.read_csv(metadata)
    sites = meta[meta.cell_type == "HUVEC"].sort_values("site_id")
    reader, crops, records = SiteReader(archive), [], []
    for row in sites.itertuples(index=False):
        raw = reader.site(row)
        nuclei = sorted(nucleus_centroids(raw[0], min_area=60, max_area=2500, half=32),
                        key=lambda n: _rank(0, row.site_id, n[0]))[:3]
        image = stretch_site(raw)
        for label_id, r, c in nuclei:
            crops.append(np.ascontiguousarray(image[:, r - 32:r + 32, c - 32:c + 32]))
            records.append(f"{row.site_id}_n{label_id:04d}")
    expected = io.BytesIO()
    np.save(expected, np.stack(crops))
    reference = outputs[1]
    assert (reference / "images.npy").read_bytes() == expected.getvalue()
    cells = pd.read_csv(reference / "cells.csv")
    assert list(cells.columns) == ["row", "cell_id", "site_id", "well_unit", "sirna", "plate", "experiment", "split"]
    assert cells.cell_id.tolist() == records and len(cells) > 24 and cells.row.tolist() == list(range(len(cells)))
    assert (cells.well_unit == cells.plate + "/" + cells.site_id.str.rsplit("_", n=1).str[0]).all()
    held = holdout_experiments(["HUVEC-01", "HUVEC-02", "HUVEC-03"], 2, seed=0)
    assert cells.split.tolist() == assign_splits(cells.well_unit.tolist(), cells.plate.tolist(),
                                                 cells.experiment.tolist(), val_fraction=0.25,
                                                 holdout_experiments=held, seed=0)
    summary = json.loads((reference / "PREPARE.json").read_text())
    assert summary["n_sites"] == 48 and summary["experiments"] == ["HUVEC-01", "HUVEC-02", "HUVEC-03"]
    assert summary["holdout_experiments"] == held and summary["val_fraction"] == 0.25
    assert summary["sites_without_cells"] > 0 and summary["workers"] == 1 and len(summary["shard_seconds"]) == 1
    assert set(cells.split) == {"train", "val", "heldout"}
    other = json.loads((outputs[3] / "PREPARE.json").read_text())
    assert (outputs[3] / "images.npy").read_bytes() == (reference / "images.npy").read_bytes()
    assert (outputs[3] / "cells.csv").read_bytes() == (reference / "cells.csv").read_bytes()
    assert {k: v for k, v in other.items() if k not in TIMING} == {k: v for k, v in summary.items() if k not in TIMING}
    assert other["workers"] == 3 and len(other["shard_seconds"]) == 3 and other["config"]["workers"] == 3


def test_well_types_all_takes_every_well_type(fake, tmp_path):
    metadata, archive = fake
    build_prep.main(_prepare_args(metadata, archive, tmp_path / "all", "--well-types", "all",
                                  "--holdout-experiments", "1", "--val-fraction", "0.25"))
    summary = json.loads((tmp_path / "all" / "PREPARE.json").read_text())
    assert summary["well_types"] == ["negative_control", "positive_control", "treatment"] and summary["n_sites"] == 48
    build_prep.main(_prepare_args(metadata, archive, tmp_path / "two", "--well-types", "treatment,positive_control",
                                  "--holdout-experiments", "1", "--val-fraction", "0.25"))
    summary = json.loads((tmp_path / "two" / "PREPARE.json").read_text())
    assert summary["well_types"] == ["treatment", "positive_control"] and summary["n_sites"] == 36
    assert set(pd.read_csv(tmp_path / "two" / "cells.csv").sirna.astype(str)) <= {"11", "12", "1139"}
    meta = pd.read_csv(metadata)
    assert build_prep._well_types(meta, "positive_control") == ["positive_control"]     # the CLI default


def test_the_holdout_guard_refuses_before_anything_is_written(fake, tmp_path):
    metadata, archive = fake
    out = tmp_path / "prep"
    with pytest.raises(SystemExit, match="leaves no training experiment of 3"):
        build_prep.main(_prepare_args(metadata, archive, out, "--well-types", "all", "--holdout-experiments", "3",
                                      "--workers", "2"))
    assert not out.exists() and not (tmp_path / "prep.partial").exists()
    build_prep.main(_prepare_args(metadata, archive, out, "--well-types", "all", "--holdout-experiments", "1",
                                  "--val-fraction", "0.25"))
    with pytest.raises(SystemExit, match="leaves no training experiment of 3"):
        build_prep.resplit(out, tmp_path / "again", 0.25, 3)
    assert not (tmp_path / "again").exists()
    with pytest.raises(SystemExit, match="exists and is not empty"):
        build_prep.main(_prepare_args(metadata, archive, out, "--well-types", "all", "--holdout-experiments", "1"))


def test_the_seeded_order_reproduces_the_v3_held_out_experiments():
    """The task-116 v3 preps held out HUVEC-11 and HUVEC-13 (2 of the 16 train-set HUVEC experiments) and HEPG2-01
    and HEPG2-03 (2 of 11), seed 0 (the task-116 README and this benchmark's README)."""
    assert holdout_experiments([f"HUVEC-{i:02d}" for i in range(1, 17)], 2) == ["HUVEC-11", "HUVEC-13"]
    assert holdout_experiments([f"HEPG2-{i:02d}" for i in range(1, 12)], 2) == ["HEPG2-01", "HEPG2-03"]


def _synthetic_full(root: Path, n_heldout: int = 1) -> tuple[Path, pd.DataFrame]:
    """A full prep without images of its own crops: 4 experiments x 4 plates x 6 wells x 6 cells (val 0.1 per plate:
    2 val wells, 4 train), `n_heldout` experiments held out; random 2 x 2 images."""
    records = []
    for e in range(1, 5):
        for p in range(1, 5):
            plate = f"HUVEC-0{e}_{p}"
            for w in range(6):
                for c in range(6):
                    records.append({"cell_id": f"{plate}_W{w}_n{c:04d}", "site_id": f"{plate}_W{w}_1",
                                    "well_unit": f"{plate}/W{w}", "sirna": str(w), "plate": plate,
                                    "experiment": f"HUVEC-0{e}"})
    cells = pd.DataFrame.from_records(records)
    cells.insert(0, "row", np.arange(len(cells)))
    held = holdout_experiments(sorted(cells.experiment.unique()), n_heldout)
    cells["split"] = assign_splits(cells.well_unit.tolist(), cells.plate.tolist(), cells.experiment.tolist(),
                                   val_fraction=0.1, holdout_experiments=held)
    full = root / "prep_full"
    full.mkdir()
    cells.to_csv(full / "cells.csv", index=False)
    np.save(full / "images.npy", np.random.default_rng(0).integers(0, 255, size=(len(cells), 6, 2, 2), dtype=np.uint8))
    (full / "PREPARE.json").write_text(json.dumps({"experiments": sorted(cells.experiment.unique()),
                                                   "val_fraction": 0.1, "holdout_experiments": held,
                                                   "split_design": build_prep.split_design(0.1, held)}))
    return full, cells


def test_subsample_train_cells_resolves_the_fraction(tmp_path):
    full, cells = _synthetic_full(tmp_path)          # 12 seen plates x 4 train wells x 6 cells
    # f = 1/4: 12 wells, 72 cells < 100; f = 2/4: 24 wells, 144 cells
    block = build_prep.subsample(full, tmp_path / "k100", None, train_cells=100)["subsample"]
    assert block["train_well_fraction"] == 0.5 and block["n_train_cells"] == 144 and block["n_train_wells"] == 24
    assert block["train_cells"] == 100 and block["train_cells_rule"] == build_prep.TRAIN_CELLS_RULE
    build_prep.subsample(full, tmp_path / "f50", 0.5)
    assert (tmp_path / "k100" / "cells.csv").read_bytes() == (tmp_path / "f50" / "cells.csv").read_bytes()
    assert np.array_equal(np.load(tmp_path / "k100" / "images.npy"), np.load(tmp_path / "f50" / "images.npy"))
    run = pd.read_csv(tmp_path / "k100" / "cells.csv")
    assert list(run.columns) == [*cells.columns, "source_row"] and run.row.tolist() == list(range(len(run)))
    assert (run.split == "val").sum() == (cells.split == "val").sum() and not (run.split == "heldout").any()
    assert np.array_equal(np.load(tmp_path / "k100" / "images.npy"), np.load(full / "images.npy")[run.source_row])
    assert build_prep.subsample(full, tmp_path / "k1", None, train_cells=1)["subsample"]["n_train_cells"] == 72
    k72 = build_prep.subsample(full, tmp_path / "k72", None, train_cells=72)["subsample"]
    assert k72["train_well_fraction"] == 0.25
    everything = build_prep.subsample(full, tmp_path / "kall", None, train_cells=10**6)["subsample"]
    assert everything["train_well_fraction"] == 1.0 and everything["n_train_cells"] == (cells.split == "train").sum()
    with pytest.raises(ValueError, match="train_cells"):
        build_prep.subsample(full, tmp_path / "bad", None, train_cells=0)
    with pytest.raises(SystemExit, match="exists"):
        build_prep.subsample(full, tmp_path / "k100", None, train_cells=100)
    build_prep.main(["subsample", "--prep", str(full), "--output", str(tmp_path / "cli"), "--train-cells", "100"])
    assert (tmp_path / "cli" / "cells.csv").read_bytes() == (tmp_path / "k100" / "cells.csv").read_bytes()
    with pytest.raises(SystemExit):
        build_prep.main(["subsample", "--prep", str(full), "--output", str(tmp_path / "both"), "--train-cells", "5",
                         "--train-well-fraction", "0.1"])
    assert not (tmp_path / "both").exists()


def test_the_benchmark_prepare_accepts_a_built_full_v4_prep(fake, tmp_path):
    from benchmarks.representation_RxRx1.runners import prepare as prep_mod
    from benchmarks.representation_RxRx1.runners.config import (DEFAULT, FULL_V4_PREPS, PREP_SETS, PREPS,
                                                        BenchmarkConfig)

    assert DEFAULT.prep_set == "v3" and PREP_SETS["v3"] is PREPS
    assert DEFAULT.preps() == tuple(Path(DEFAULT.data_root) / n for n in PREPS["HEPG2"])
    assert DEFAULT.root() == Path(DEFAULT.output_root) / "HEPG2"
    assert sorted(FULL_V4_PREPS) == ["HEPG2", "HUVEC", "RPE", "U2OS"]
    assert all(v == (f"prep_{k.lower()}_full_v4", f"prep_{k.lower()}_full_v4_t5k") for k, v in FULL_V4_PREPS.items())
    metadata, archive = fake
    data = tmp_path / "agentic_ssl"
    cfg = BenchmarkConfig(cell_type="HUVEC", prep_set="full_v4", data_root=str(data),
                          output_root=str(tmp_path / "cache"))
    full, run = cfg.preps()
    build_prep.main(_prepare_args(metadata, archive, full, "--cell-type", "HUVEC", "--well-types", "all",
                                  "--holdout-experiments", "1", "--val-fraction", "0.25", "--workers", "2"))
    build_prep.main(["subsample", "--prep", str(full), "--output", str(run), "--train-cells", "10"])
    assert cfg.root() == tmp_path / "cache" / "HUVEC_full_v4"
    summary = prep_mod.prepare(full, run, cfg.root())
    source = pd.read_csv(full / "cells.csv")
    run_cells = pd.read_csv(run / "cells.csv")
    splits = summary["splits"]
    assert summary["n_cells"] == len(source)
    assert splits["heldout"]["cells"] == (source.split == "heldout").sum() > 0
    assert splits["val"]["cells"] == (source.split == "val").sum() == (run_cells.split == "val").sum() > 0
    assert splits["train"]["cells"] == (run_cells.split == "train").sum() >= 10
    assert splits["train"]["cells"] + splits["unused"]["cells"] == (source.split == "train").sum()
    assert summary["crop_array"]["shape"] == [len(source), 6, 64, 64] and summary["crop_array"]["dtype"] == "uint8"
    assert summary["sources"]["full"]["holdout_experiments"] == json.loads((full / "PREPARE.json").read_text())[
        "holdout_experiments"]
    assert summary["sources"]["run"]["subsample"]["train_cells"] == 10
    loaded, table = prep_mod.load(cfg.root())
    assert len(table) == len(source) and "sirna" not in table


def test_the_cli_module_runs_spawned_workers_and_shows_its_options(fake, tmp_path):
    """The launchers' stale-code probe (`prepare --help` shows --workers) and `python -m` with spawned workers (the
    workers unpickle _crop_shard from the module run as __main__)."""
    metadata, archive = fake
    module = [sys.executable, "-m", "benchmarks.representation_RxRx1.runners.build_prep"]
    shown = subprocess.run([*module, "prepare", "--help"], cwd=REPO, capture_output=True, text=True, check=True)
    assert "--workers" in shown.stdout and "--well-types" in shown.stdout
    shown = subprocess.run([*module, "subsample", "--help"], cwd=REPO, capture_output=True, text=True, check=True)
    assert "--train-cells" in shown.stdout
    args = _prepare_args(metadata, archive, tmp_path / "w2", "--well-types", "all", "--holdout-experiments", "1",
                         "--val-fraction", "0.25")
    subprocess.run([*module, *args, "--workers", "2"], cwd=REPO, capture_output=True, text=True, check=True)
    build_prep.main(_prepare_args(metadata, archive, tmp_path / "w1", "--well-types", "all",
                                  "--holdout-experiments", "1", "--val-fraction", "0.25"))
    for name in ("images.npy", "cells.csv"):
        assert (tmp_path / "w2" / name).read_bytes() == (tmp_path / "w1" / name).read_bytes()


def _bash_parses(text: str) -> None:
    bash = shutil.which("bash")
    found = subprocess.run([bash, "--version"], capture_output=True, text=True) if bash else None
    if found is None or "GNU bash" not in found.stdout:
        pytest.skip("no GNU bash")
    # bash -n reads the script from stdin: no path translation, LF line ends whatever the checkout's are
    checked = subprocess.run([bash, "-n"], input=text.encode("utf-8"), capture_output=True)
    assert checked.returncode == 0, checked.stderr.decode(errors="replace")


def _wells_layout(sizes: dict[str, int]) -> tuple[list[str], list[str]]:
    """(well_unit, plate) of 2 cells per well, `sizes` = {plate: n wells}."""
    units, plates = [], []
    for plate, n in sizes.items():
        for w in range(n):
            units += [f"{plate}/{plate}_W{w:03d}"] * 2
            plates += [plate] * 2
    return units, plates


def test_the_well_split_takes_per_plate_ten_percent_seen_and_twenty_percent_of_them_val():
    sizes = {"HUVEC-01_1": 308, "HUVEC-01_2": 305, "HUVEC-02_1": 304, "HUVEC-02_2": 40, "HUVEC-03_1": 25}
    units, plates = _wells_layout(sizes)
    split = build_prep.assign_well_splits(units, plates, seen_fraction=0.10, val_of_seen=0.20)
    table = pd.DataFrame({"well_unit": units, "plate": plates, "split": split})
    assert table.groupby("well_unit").split.nunique().max() == 1                     # whole wells, never cells
    counts = table.drop_duplicates("well_unit").groupby(["plate", "split"]).size().unstack(fill_value=0)
    # n_seen = floor(0.1 n + 1/2) (305 -> 31: half up), n_val = max(2, floor(0.2 n_seen + 1/2))
    expected = {"HUVEC-01_1": (6, 25, 277), "HUVEC-01_2": (6, 25, 274), "HUVEC-02_1": (6, 24, 274),
                "HUVEC-02_2": (2, 2, 36), "HUVEC-03_1": (2, 1, 22)}
    assert {p: (int(r.val), int(r.train), int(r.heldout)) for p, r in counts.iterrows()} == expected
    # the order: per plate the seeded sha256 order of its wells; val first, then train, then heldout
    of = dict(zip(table.well_unit, table.split))
    for plate, (n_val, n_train, n_heldout) in expected.items():
        ordered = sorted(set(table.well_unit[table.plate == plate]), key=lambda u: _rank(0, "seen", plate, u))
        assert [of[u] for u in ordered] == ["val"] * n_val + ["train"] * n_train + ["heldout"] * n_heldout
    assert split == build_prep.assign_well_splits(units, plates, seen_fraction=0.10, val_of_seen=0.20)   # determinism
    assert split != build_prep.assign_well_splits(units, plates, seen_fraction=0.10, val_of_seen=0.20, seed=1)
    assert set(split) == {"train", "val", "heldout"}                                 # no unused
    with pytest.raises(ValueError, match="too few wells"):                           # 24 wells: 2 seen, no train well
        build_prep.assign_well_splits(*_wells_layout({"P_1": 24}), seen_fraction=0.10, val_of_seen=0.20)
    with pytest.raises(ValueError, match="must be in"):
        build_prep.assign_well_splits(units, plates, seen_fraction=1.0, val_of_seen=0.20)


def _synthetic_v4(root: Path, n_wells: int = 30, side: int = 2) -> tuple[Path, pd.DataFrame, Path]:
    """A full_v4-like full prep (experiment-level split: 1 of 3 experiments held out) of 3 experiments x 2 plates x
    `n_wells` wells x 2 cells and its metadata.csv: well 0 of every plate EMPTY (negative_control), wells 1-3 the
    positive controls pc1-pc3 (on every plate), well w > 3 of plate p the treatment t<p>_<w> (one well per
    experiment); images uint8 (N, 6, side, side)."""
    records, meta = [], []
    for e in (1, 2, 3):
        for p in (1, 2):
            plate = f"HUVEC-0{e}_{p}"
            for w in range(n_wells):
                well_id = f"{plate}_W{w:02d}"
                sirna, kind = (("EMPTY", "negative_control") if w == 0 else (f"pc{w}", "positive_control") if w <= 3
                               else (f"t{p}_{w}", "treatment"))
                meta.append({"well_id": well_id, "well_type": kind, "site": 1})
                for c in range(2):
                    records.append({"cell_id": f"{well_id}_1_n{c:04d}", "site_id": f"{well_id}_1",
                                    "well_unit": f"{plate}/{well_id}", "sirna": sirna, "plate": plate,
                                    "experiment": f"HUVEC-0{e}"})
    cells = pd.DataFrame.from_records(records)
    cells.insert(0, "row", np.arange(len(cells)))
    held = holdout_experiments(sorted(cells.experiment.unique()), 1)
    cells["split"] = assign_splits(cells.well_unit.tolist(), cells.plate.tolist(), cells.experiment.tolist(),
                                   val_fraction=0.1, holdout_experiments=held)
    full = root / "prep_huvec_full_v4"
    full.mkdir(parents=True)
    cells.to_csv(full / "cells.csv", index=False)
    np.save(full / "images.npy", np.random.default_rng(0).integers(0, 255, size=(len(cells), 6, side, side),
                                                                   dtype=np.uint8))
    (full / "PREPARE.json").write_text(json.dumps({
        "experiments": sorted(cells.experiment.unique()), "val_fraction": 0.1, "holdout_experiments": held,
        "test_experiments": held, "split_design": build_prep.split_design(0.1, held), "n_cells": len(cells)}))
    metadata = root / "metadata.csv"
    pd.DataFrame(meta).to_csv(metadata, index=False)
    return full, cells, metadata


def test_resplit_wells_reads_no_image_and_records_the_design(tmp_path, monkeypatch, capsys):
    source, cells, _ = _synthetic_v4(tmp_path)
    before = (source / "images.npy").read_bytes()

    def no_image(*_, **__):
        raise AssertionError("resplit read images.npy or a raw image")

    monkeypatch.setattr(build_prep.np, "load", no_image)
    monkeypatch.setattr(build_prep, "SiteReader", no_image)
    out = tmp_path / "prep_huvec_full_v5"
    summary = build_prep.resplit(source, out, None, None, split="wells", seen_fraction=0.10, val_of_seen=0.20)
    monkeypatch.undo()
    linked = (out / "images.npy").samefile(source / "images.npy")                    # a hard link: one inode
    assert summary["resplit"] == {"source": str(source), "seed": 0, "split": "wells",
                                  "images": "linked" if linked else "copied"}
    assert (out / "images.npy").read_bytes() == before
    assert f'"images.npy": "{summary["resplit"]["images"]}"' in capsys.readouterr().out
    new = pd.read_csv(out / "cells.csv")
    old_cells = pd.read_csv(source / "cells.csv")
    assert new.drop(columns="split").equals(old_cells.drop(columns="split"))         # the same cells, a new split
    assert new.split.tolist() == build_prep.assign_well_splits(cells.well_unit.tolist(), cells.plate.tolist(),
                                                               seen_fraction=0.10, val_of_seen=0.20)
    assert all(s == {"train", "val", "heldout"} for s in new.groupby("experiment").split.agg(set))  # no held-out exp.
    written = json.loads((out / "PREPARE.json").read_text())
    assert written == summary and written["split_mode"] == "wells" and written["holdout_experiments"] == []
    assert (written["seen_fraction"], written["val_of_seen"], written["min_val_wells"]) == (0.10, 0.20, 2)
    assert "val_fraction" not in written and "test_experiments" not in written and written["n_cells"] == len(cells)
    assert written["split_rule"] == build_prep.WELL_SPLIT_RULE
    # 30 wells per plate: 3 seen (2 val, 1 train), 27 held out; 6 plates
    assert written["wells_per_plate"] == {"train": [1, 1], "val": [2, 2], "heldout": [27, 27]}
    assert written["wells_per_split"] == {"train": 6, "val": 12, "heldout": 162}
    assert written["cells_per_split"] == {"train": 12, "val": 24, "heldout": 324}
    assert build_prep.require_split(out) == written
    # the CLI gives the same split; refusals: an existing output, the options of the other mode
    build_prep.main(["resplit", "--source", str(source), "--output", str(tmp_path / "cli"), "--split", "wells",
                     "--seen-fraction", "0.10", "--val-of-seen", "0.20"])
    assert (tmp_path / "cli" / "cells.csv").read_bytes() == (out / "cells.csv").read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        build_prep.resplit(source, out, None, None, split="wells", seen_fraction=0.10, val_of_seen=0.20)
    for argv in (["--split", "wells", "--seen-fraction", "0.1"],
                 ["--split", "wells", "--seen-fraction", "0.1", "--val-of-seen", "0.2", "--val-fraction", "0.1"],
                 ["--val-fraction", "0.1"],
                 ["--val-fraction", "0.1", "--holdout-experiments", "1", "--seen-fraction", "0.1"]):
        with pytest.raises(SystemExit, match="--split wells takes"):
            build_prep.main(["resplit", "--source", str(source), "--output", str(tmp_path / "bad"), *argv])
        assert not (tmp_path / "bad").exists()
    # the experiment-level resplit (the default) writes what it wrote before: no well-split key, its resplit block
    old = build_prep.resplit(source, tmp_path / "v4again", 0.1, 1)
    assert not {"split_mode", "seen_fraction", "wells_per_plate", "split_rule"} & set(old)
    assert old["resplit"] == {"source": str(source), "seed": 0} and old["val_fraction"] == 0.1
    assert (tmp_path / "v4again" / "cells.csv").read_bytes() == (source / "cells.csv").read_bytes()


def test_the_full_v5_run_prep_takes_every_train_and_val_cell_and_scores_without_unused(tmp_path):
    from benchmarks.representation_RxRx1.runners import prepare as prep_mod
    from benchmarks.representation_RxRx1.runners import score
    from benchmarks.representation_RxRx1.runners.config import DEFAULT, FULL_V5_PREPS, PREP_SETS, BenchmarkConfig

    assert DEFAULT.prep_set == "v3" and PREP_SETS["full_v5"] is FULL_V5_PREPS
    assert all(v == (f"prep_{k.lower()}_full_v5", f"prep_{k.lower()}_full_v5_run") for k, v in FULL_V5_PREPS.items())
    assert sorted(FULL_V5_PREPS) == ["HEPG2", "HUVEC", "RPE", "U2OS"]
    data = tmp_path / "agentic_ssl"
    source, _, _ = _synthetic_v4(data, n_wells=60, side=64)
    cfg = BenchmarkConfig(cell_type="HUVEC", prep_set="full_v5", data_root=str(data),
                          output_root=str(tmp_path / "cache"))
    full, run = cfg.preps()
    assert cfg.root() == tmp_path / "cache" / "HUVEC_full_v5"
    build_prep.main(["resplit", "--source", str(source), "--output", str(full), "--split", "wells",
                     "--seen-fraction", "0.10", "--val-of-seen", "0.20"])
    build_prep.main(["subsample", "--prep", str(full), "--output", str(run), "--train-well-fraction", "1.0"])
    source_cells, run_cells = pd.read_csv(full / "cells.csv"), pd.read_csv(run / "cells.csv")
    seen = source_cells.split.isin(["train", "val"])
    assert run_cells.source_row.tolist() == source_cells.row[seen].tolist()          # every train and val cell
    assert np.array_equal(np.load(run / "images.npy"), np.load(full / "images.npy")[seen.to_numpy()])
    written = json.loads((run / "PREPARE.json").read_text())
    block = written["subsample"]
    assert block["train_well_fraction"] == 1.0 and block["n_train_cells"] == (source_cells.split == "train").sum()
    assert block["n_train_wells"] == source_cells.well_unit[source_cells.split == "train"].nunique()
    assert written["cells_per_split"]["heldout"] == 0 and written["wells_per_plate"]["heldout"] == [0, 0]
    assert written["split_mode"] == "wells"
    summary = prep_mod.prepare(full, run, cfg.root())
    splits = summary["splits"]
    assert splits["unused"]["cells"] == 0 and splits["train"]["cells"] == (source_cells.split == "train").sum()
    assert splits["heldout"]["experiments"] == 3 and splits["train"]["experiments"] == 3
    assert summary["sources"]["full"]["holdout_experiments"] == []
    assert summary["sources"]["full"]["val_fraction"] is None
    assert score.prep_set(summary) == "full_v5" and score.probe_units(summary)[0] == ("well",)
    # the score groups with no unused cell: seen = train + val, heldout = heldout, unused_wells empty (n/a)
    _, cells = prep_mod.load(cfg.root())
    # labels with positives among the few seen wells: well w > 0 of every plate the siRNA s<w mod 3> (well 0 EMPTY)
    sirna = np.array(["EMPTY" if s == "EMPTY" else f"s{int(u[-2:]) % 3}"
                      for s, u in zip(source_cells.sirna, source_cells.well_unit)])
    rng = np.random.default_rng(0)
    tables = {arm: score.evaluate(rng.normal(size=(len(cells), 8)), cells, sirna) for arm in ("frozen", "dinov3")}
    for out in tables.values():
        assert out["seen"]["n_rows"] == cells.well[cells.split.isin(["train", "val"]) & (sirna != "EMPTY")].nunique()
        assert out["heldout"]["value"] is not None and out["seen"]["value"] is not None
        assert out["unused_wells"]["value"] is None and out["unused_wells"]["per_unit"] == {}
    compared = score.compare(tables, ["dinov3"])
    assert compared["seen"]["values"]["frozen"] == tables["frozen"]["seen"]["value"]
    assert compared["unused_wells"]["default_mean"] is None
    assert compared["unused_wells"]["comparisons"]["default_mean_vs_frozen"]["delta"] is None


