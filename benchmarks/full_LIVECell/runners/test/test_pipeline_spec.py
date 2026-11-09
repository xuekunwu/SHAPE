"""The full_CPJUMP1 harness under the LIVECell dataset spec: texts, single-channel crops, mask labels, batch."""
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness import codex_arm, pipeline
from shape.agent.arms import ARM_MODELS, BASE_URL
from shape.agent.arms import codex as codex_backend
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_LIVECell.harness.spec import LIVECELL

DESIGN_TASKS = {
    "segment": "Segment whole cells, one instance per cell, in a label-free phase-contrast image (LIVECell: eight "
               "cell lines, sparse to confluent). The masks crop the same single-channel image at identical "
               "coordinates.",
    "crop": "Single-channel phase-contrast single-cell morphology.",
    "embed": "Single-channel phase-contrast cells of eight cell lines; six culture-time stages; two fields (sites) "
             "per unit. Only train and val units are visible.",
    "batch": "Unit profiles grouped into six culture-time stages (the batch key). Calibration references are opaque "
             "train/val cell-line ids; there are no negative controls. Joint unlabeled correction is transductive."}
CELL_TYPES = ("A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr3", "SKOV3")
LABELS = {3: (slice(20, 36), slice(20, 36)), 7: (slice(50, 70), slice(60, 80)), 12: (slice(20, 35), slice(90, 105))}


def test_cpjump1_spec_is_todays_values_and_the_default():
    spec = pipeline.CPJUMP1
    assert pipeline.dataset({}) is spec and pipeline.dataset({"dataset": "CPJUMP1"}) is spec
    assert (spec.n_images, spec.marker, spec.image_shape, spec.dtype, spec.plate_type, spec.mask_label) == (
        1000, "CPJUMP1_COMPLETE", (1080, 1080), "uint16", "CRISPR", False)
    assert spec.channels == ("Mito", "AGP", "RNA", "ER", "DNA") and spec.cli == "benchmarks.full_CPJUMP1.harness.cli"
    assert codex_arm.TASKS is spec.tasks and spec.loader(len) is len          # this harness's own data module
    assert spec.usages == ("development_train", "development_val", "evaluation_only")
    with pytest.raises(ValueError, match="unknown dataset"):
        pipeline.dataset({"dataset": "RxRx1"})


def test_livecell_spec_texts_are_the_designs_and_name_no_cell_type():
    assert pipeline.dataset({"dataset": "LIVECell"}) is LIVECELL
    assert LIVECELL.tasks == DESIGN_TASKS
    assert (LIVECELL.n_images, LIVECELL.marker, LIVECELL.image_shape, LIVECELL.channels, LIVECELL.dtype) == (
        1310, "LIVECELL_COMPLETE", (520, 704), ("Phase",), None)
    assert LIVECELL.mask_label and LIVECELL.cli == "benchmarks.full_LIVECell.harness.cli"
    assert LIVECELL.usages == ("train", "val", "heldout")
    lines = [t.format(n=1310, cells=128, dtype="uint8", rule=LIVECELL.rules["RUN"]) for t in LIVECELL.inputs.values()]
    assert lines[0].startswith("Inputs: 1310 images; each a 704 x 520 single-channel uint8 phase-contrast TIFF")
    texts = [*LIVECELL.tasks.values(), *lines, *LIVECELL.rules.values(), LIVECELL.withheld]
    assert not any(re.search(rf"\b{t}\b", text, re.IGNORECASE) for t in CELL_TYPES for text in texts)
    assert not any(word in text for word in ("gene", "CRISPR", "A549", "well", "{", "}") for text in texts)
    assert not any(word in text for word in ("development", "evaluation") for text in texts)       # train / val / heldout
    assert "Only train and val units are visible" in LIVECELL.tasks["embed"]
    assert "heldout units never enter a call" in LIVECELL.inputs["embed"]


def test_raw_stack_restores_the_single_channel_axis(tmp_path):
    pixels = np.arange(6 * 8, dtype=np.uint8).reshape(1, 6, 8)
    path = tmp_path / "img.ome.tif"
    tifffile.imwrite(path, pixels, ome=True, photometric="minisblack", metadata={"axes": "CYX"})
    assert tifffile.imread(path).shape == (6, 8)                          # tifffile drops C = 1
    assert np.array_equal(pipeline.raw_stack(path, LIVECELL), pixels)
    five = np.zeros((5, 6, 8), np.uint16)
    tifffile.imwrite(path, five, ome=True, photometric="minisblack", metadata={"axes": "CYX"})
    assert pipeline.raw_stack(path, pipeline.CPJUMP1).shape == (5, 6, 8)


def test_single_channel_raw_crop_is_lossless_in_its_native_dtype(tmp_path):
    raw = np.random.default_rng(0).integers(0, 256, (1, 40, 50), dtype=np.uint8)
    path = tmp_path / "crop.ome.tif"
    pipeline.materialize_raw_crop(raw, [5, 10, 25, 30], path, LIVECELL.channels, "uint8")
    back = tifffile.imread(path)
    assert back.dtype == np.uint8 and np.array_equal(back, raw[0, 5:25, 10:30])
    assert "Phase" in tifffile.TiffFile(path).ome_metadata
    with pytest.raises(ValueError, match="1 channel"):
        pipeline.materialize_raw_crop(raw.astype(np.uint16), [5, 10, 25, 30], path, LIVECELL.channels, "uint8")
    with pytest.raises(ValueError, match="1 channel"):
        pipeline.materialize_raw_crop(np.repeat(raw, 2, axis=0), [5, 10, 25, 30], path, LIVECELL.channels, "uint8")


def _livecell_image(tmp_path):
    """A 96 x 128 single-channel uint8 image (CYX, C = 1) and a mask whose labels 3, 7, 12 are not 1..n."""
    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public" / "images").mkdir(parents=True)
    run.mkdir()
    pixels = np.random.default_rng(1).integers(10, 250, (96, 128), dtype=np.uint8)
    tifffile.imwrite(prep / "public" / "images" / "img_0000.ome.tif", pixels[None], ome=True,
                     photometric="minisblack", metadata={"axes": "CYX", "Channel": {"Name": ["Phase"]}})
    mask = np.zeros((96, 128), np.uint32)
    for label, (rows, cols) in LABELS.items():
        mask[rows, cols] = label
    path = run / "cell_mask_img_0000.tif"
    tifffile.imwrite(path, mask)
    row = {"image_id": "img_0000", "file": "images/img_0000.ome.tif", "plate": "stage_2", "well": "unit_0000",
           "usage": "heldout", "dtype": "uint8", "shape": [1, 96, 128]}
    return prep, run, pixels, mask, path, row


def _check_cells(cells, pixels, mask):
    assert sorted(c["mask_label"] for c in cells) == sorted(LABELS)
    for c in cells:
        r0, c0, r1, c1 = c["crop_bbox"]
        rows, cols = LABELS[c["mask_label"]]
        assert r0 <= rows.start and rows.stop <= r1 and c0 <= cols.start and cols.stop <= c1   # its own cell
        crop = tifffile.imread(c["crop_path"])
        assert crop.dtype == np.uint8 and np.array_equal(crop, pixels[r0:r1, c0:c1])           # lossless


def test_real_cropper_records_each_cells_mask_label_on_a_single_channel_image(tmp_path):
    prep, run, pixels, mask, path, row = _livecell_image(tmp_path)
    write_json(run / "segmentation.json", [{"image_id": "img_0000", "mask": str(path), "sha256": digest(path),
                                            "min_area": 15}])
    pipeline.cropping(prep, run, {"arm": "control", "deadline": time.time() + 120, "cores": 1,
                                  "dataset": "LIVECell"}, [row])
    cells = read_json(run / "cells.json")
    _check_cells(cells, pixels, mask)
    # the scorer's input: every delivered cell's label is an instance of its delivered mask
    assert all(mask[tuple(np.argwhere(mask == c["mask_label"])[0])] == c["mask_label"] for c in cells)


def test_a_one_instance_mask_keeps_its_own_instance_id(tmp_path):
    """The cropper relabels a mask with one instance (skimage label: 1); mask_label is still the delivered id."""
    prep, run, pixels, mask, path, row = _livecell_image(tmp_path)
    single = np.where(mask == 7, 7, 0).astype(np.uint32)
    tifffile.imwrite(path, single)
    write_json(run / "segmentation.json", [{"image_id": "img_0000", "mask": str(path), "sha256": digest(path),
                                            "min_area": 15}])
    pipeline.cropping(prep, run, {"arm": "control", "deadline": time.time() + 120, "cores": 1,
                                  "dataset": "LIVECell"}, [row])
    cells = read_json(run / "cells.json")
    assert [c["mask_label"] for c in cells] == [7] and 7 in np.unique(tifffile.imread(path))


def test_cpjump1_cells_have_no_mask_label(tmp_path):
    prep, run, pixels, mask, path, row = _livecell_image(tmp_path)
    five = np.repeat(pixels[None].astype(np.uint16), 5, axis=0)
    tifffile.imwrite(prep / "public" / "images" / "img_0000.ome.tif", five, ome=True, photometric="minisblack",
                     metadata={"axes": "CYX"})
    write_json(run / "segmentation.json", [{"image_id": "img_0000", "mask": str(path), "sha256": digest(path),
                                            "min_area": 15}])
    pipeline.cropping(prep, run, {"arm": "control", "deadline": time.time() + 120, "cores": 1}, [row])
    cells = read_json(run / "cells.json")
    assert len(cells) == 3 and not any("mask_label" in c for c in cells)


def test_mask_labels_fail_closed_on_an_unmatched_or_ambiguous_cell():
    mask = np.zeros((20, 20), np.uint32)
    mask[2:6, 2:6] = 4
    crops = [{"cell_id": "a", "original_bbox": [2, 2, 6, 6], "area": 16.0}]
    assert pipeline.mask_labels(mask, crops) == {"a": 4}
    with pytest.raises(ValueError, match="matches 0 mask labels"):
        pipeline.mask_labels(mask, [{**crops[0], "area": 15}])
    checker = np.zeros((20, 20), np.uint32)   # two labels with the same box and pixel count
    checker[2:6, 2:6] = 4 + (np.add.outer(np.arange(4), np.arange(4)) % 2)
    with pytest.raises(ValueError, match="matches 2 mask labels"):
        pipeline.mask_labels(checker, [{"cell_id": "b", "original_bbox": [2, 2, 6, 6], "area": 8}])
    with pytest.raises(ValueError, match="delivered segmentation mask"):
        pipeline.mask_labels(None, crops)


@pytest.fixture
def gateway(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "offline-dummy-key-not-a-credential")
    monkeypatch.setenv("LLM_API_BASE", BASE_URL)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


def test_codex_crop_stage_delivers_mask_labels_and_states_the_livecell_facts(tmp_path, monkeypatch, gateway):
    """The Codex arm's crop stage with the real cropper: the LIVECell prompt, the native worker of the LIVECell CLI,
    and cells.json with mask_label."""
    from benchmarks.full_CPJUMP1.harness import cli
    from benchmarks.full_LIVECell.harness import data
    prep, run, pixels, mask, path, row = _livecell_image(tmp_path)
    (run / "logs").mkdir()
    write_json(run / "segmentation.json", [{"image_id": "img_0000", "mask": str(path), "sha256": digest(path),
                                            "min_area": 15, "n_cells": 3}])
    monkeypatch.setattr(data, "load_prepared", lambda prep: [row])
    modules, prompts = [], []

    def child(command, log, timeout, **kwargs):
        modules.append(command[command.index("-m") + 1])
        folder = Path(command[command.index("--run") + 1])
        config = read_json(folder / "CONFIG.json")
        log.write_text("", encoding="utf-8")
        pipeline.codex_calls("crop", prep, folder, config, [row], threads=True)

    def invoke(root, prompt, model, settings, deadline, index, session=None, images=(), marker=None, effort=None):
        prompts.append(prompt)
        return {"action": "finish", "request_json": json.dumps({"all": "c0"}), "reason": "r"}, {"total_tokens": 5}, None

    monkeypatch.setattr(cli, "child", child)
    monkeypatch.setattr(codex_backend, "invoke", invoke)
    config = {"arm": "codex", "model": ARM_MODELS["codex"], "deadline": time.time() + 600, "cores": 2,
              "images": ["img_0000"], "dataset": "LIVECell", "train_cells_per_site": 128}
    codex_arm.run_stage("crop", prep, run, config)
    assert modules == ["benchmarks.full_LIVECell.harness.cli"]
    assert "arm of the LIVECell benchmark, stage crop" in prompts[0]
    assert f"Task: {DESIGN_TASKS['crop']}" in prompts[0] and "Private labels are not available." in prompts[0]
    assert "Inputs: 1 images; each a 704 x 520 single-channel uint8 phase-contrast TIFF (Phase)" in prompts[0]
    assert not any(t in prompts[0] for t in ("A549", "CRISPR", "five-channel", "evaluation genes"))
    _check_cells(read_json(run / "cells.json"), pixels, mask)


def test_single_channel_inputs_reach_the_segmenter_and_analyzer_as_one_channel(tmp_path):
    """The segmenter's loader reads the YX view as one channel; the analyzer reads a C = 1 raw crop as one channel
    and adapts it by replicate_1_to_rgb; uint8 passes its display conversion unchanged."""
    from shape.tools.cell_state_analyzer.tool import ChannelAdapter, _normalized_channel_tensor, infer_channel_spec
    from shape.utils.image_processor import ImageProcessor
    pixels = np.random.default_rng(2).integers(0, 256, (52, 70), dtype=np.uint8)
    segment = tmp_path / "segment.ome.tif"
    tifffile.imwrite(segment, pixels, ome=True, photometric="minisblack", metadata={"axes": "YX"})
    loaded = ImageProcessor.load_image(str(segment))
    assert not loaded.is_multi_channel and loaded.num_channels == 1
    assert np.array_equal(np.squeeze(loaded.to_segmentation_input(channel_idx=0)), pixels)
    crop = tmp_path / "crop.ome.tif"
    pipeline.materialize_raw_crop(pixels[None], [2, 3, 32, 33], crop, LIVECELL.channels, "uint8")
    assert infer_channel_spec([str(crop)]) == (1, ["HW"])
    tensor = _normalized_channel_tensor(tifffile.imread(crop), str(crop))
    assert tensor.shape == (1, 30, 30) and np.allclose(tensor[0].numpy() * 255, pixels[2:32, 3:33])
    assert ChannelAdapter(1).init_name == "replicate_1_to_rgb"


def _profiles(tmp_path, n_per=3, offset=6.0):
    """Unit profiles: 8 references x 2 train/val stages (0 train, 1 val) and 4 heldout stages, n_per units each, a
    stage shift."""
    rng = np.random.default_rng(3)
    rows, x = [], []
    for k in range(8):
        for stage in range(6):
            for u in range(n_per):
                rows.append({"well": f"unit_{len(rows):04d}", "plate": f"stage_{stage}",
                             "usage": ("train", "val", "heldout")[min(stage, 2)],
                             "calibration_group": f"reference_{k:03d}" if stage < 2 else "",
                             "negative_control": False})
                x.append(rng.normal(size=12) + 3.0 * np.eye(12)[k] + offset * stage)
    np.save(tmp_path / "profiles.npy", np.asarray(x, np.float32))
    pd.DataFrame(rows).to_csv(tmp_path / "profiles.csv", index=False)
    return pd.read_csv(tmp_path / "profiles.csv", keep_default_na=False), np.asarray(x, np.float32)


@pytest.fixture
def scib_standins(monkeypatch):
    """scib-metrics is not installed in the CPU test environment: its six metrics return constants here; the
    composite's own logic (which metrics apply without negative controls) and copairs' mAP stay real."""
    from shape.tools.batch_corrector import quant
    for name in ("silhouette_batch", "lisi_batch", "graph_conn", "lisi_label", "asw"):
        monkeypatch.setattr(quant, name, lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(quant, "nmi_ari", lambda *args, **kwargs: {"nmi": 0.5, "ari": 0.5})


def test_batch_calibration_runs_without_negative_controls(tmp_path, scib_standins):
    """No negative-control rows: the view holds the train and val references only (no heldout unit: identity or
    label), the shared admission evidence
    and rule run (the concern batch_group_without_negative_controls holds), and the calibration score runs with
    negcon_mean_map skipped, not imputed."""
    from shape.agent.verification.verifiers.batch_correction_unified import (
        BatchTask, admission_concerns, admission_evidence, admission_rule, uncorrected_result)
    meta, x = _profiles(tmp_path)
    view = pipeline.calibration_view(meta, LIVECELL.plate_type)
    assert len(view.rows) == 8 * 2 * 3 and set(view.meta.Metadata_PertType) == {"trt"}
    usage = meta.usage.to_numpy()
    assert set(usage[view.rows]) == {"train", "val"} and not (usage[view.rows] == "heldout").any()
    heldout = set(meta.well[meta.usage == "heldout"])
    assert heldout and not heldout & set(view.meta.Metadata_Well)                      # no heldout identity
    assert set(view.meta.Metadata_Plate) == {"stage_0", "stage_1"} and set(meta.calibration_group[usage == "heldout"]) == {""}
    assert "DMSO" not in set(view.meta.Metadata_JCP2022) and set(view.meta.Metadata_PlateType) == {"LIVECell"}
    task = BatchTask(batch=meta.plate.to_numpy(), features=x, calibration=view, cores=1)
    evidence = admission_evidence(task, uncorrected_result(x, task.batch)["batch_effect"])
    assert evidence["gate_negative_controls_only"] is None and evidence["negative_controls_per_batch_group"] == {}
    assert evidence["reference_groups_in_2plus_batch_groups"] == 8
    assert admission_rule(evidence) == "RUN"
    assert admission_concerns(evidence)["batch_group_without_negative_controls"]
    scores = view.score(x)
    assert "negcon_mean_map" in scores["skipped"] and scores["overall"] is not None
    assert scores.get("nonrep_mean_map") is not None


def test_codex_batch_feedback_runs_without_negative_controls(tmp_path, monkeypatch, scib_standins):
    """The Codex arm's batch call (the control's path): rule, one correction and FEEDBACK's calibration scores.
    harmonypy is not installed in the CPU test environment: the corrector returns its input shifted."""
    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool
    monkeypatch.setattr(Batch_Corrector_Tool, "execute", lambda self, features, metadata, **kwargs: {
        "corrected_cls_output": np.asarray(features) + 1.0, "status": "CORRECTED_HARMONY"})
    _profiles(tmp_path)
    pipeline.batch_correction(tmp_path, tmp_path, {"arm": "codex", "cores": 2, "deadline": time.time() + 600,
                                                   "dataset": "LIVECell"}, [])
    feedback = read_json(tmp_path / "FEEDBACK.json")
    assert feedback["admission"] == "RUN" and feedback["calibration"]["overall"] is not None
    assert "negcon_mean_map" in feedback["uncorrected_calibration"]["skipped"]
    assert np.load(tmp_path / "corrected.npy").shape == (8 * 6 * 3, 12)


def test_ssl_budget_and_selection_follow_the_spec_usages():
    """The SSL epoch budget counts the spec's train sites and the SSL selection holds its train and val cells; heldout
    cells never enter (the run's embed stage passes `dataset(config).usages`)."""
    images = [{"usage": u} for u in ("train", "train", "val", "heldout", "heldout")]
    budget = pipeline.ssl_epoch_budget(images, 128, LIVECELL.usages)
    assert budget["n_train"] == 256 and budget["train_sites"] == 2 and "development_train_sites" not in budget
    with pytest.raises(ValueError, match="no development_train site"):      # CPJUMP1's names do not fit these rows
        pipeline.ssl_epoch_budget(images, 128)
    cells = [{"cell_id": f"{u}_{k}", "image_id": f"img_{u}", "crop_path": f"{u}_{k}.tif", "usage": u,
              "plate": "stage_0", "well": f"unit_{u}", "area": 1.0}
             for u in ("train", "val", "heldout") for k in range(4)]
    paths, meta = pipeline.select_development(cells, 2, LIVECELL.usages)
    assert sorted(m["cell_id"].split("_")[0] for m in meta) == ["train"] * 2 + ["val"] * 2 and len(paths) == 4
    assert {m["split"] for m in meta} == {"train", "val"}
    with pytest.raises(ValueError, match="unknown usage"):
        pipeline.select_development(cells, 2)
