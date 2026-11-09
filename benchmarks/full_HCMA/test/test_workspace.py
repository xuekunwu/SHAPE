"""setup_run (workspace layout, _config.json, TOOLS_SHA256.json, TASK.md placeholders) and the
node facts it reports, on a synthetic prep."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import hardware, prepare, subset, workspace
from benchmarks.full_HCMA.harness.cli import main

PLACEHOLDERS = ("N_IMAGES", "DESIGN", "SOURCES", "METADATA_NOTE", "BUDGET_HOURS", "GPU", "GPU_SLOTS", "CPUS",
                "MEM_GB", "TIMINGS", "FORBIDDEN")
REAL_TASK = Path(workspace.TEMPLATE) / "TASK.md"


@pytest.fixture()
def root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "nvidia-h100-80gb-hbm3")
    monkeypatch.setattr(hardware, "node_facts", lambda: {"gpu": "NVIDIA H100 80GB HBM3", "gpu_mem_gb": 79.6,
                                                         "gpu_compute_mode": "Exclusive_Process",
                                                         "gpu_exclusive": True, "cpus": 16, "mem_gb": 128.0})
    root = tmp_path / "full_HCMA"
    (root / "prep" / "images").mkdir(parents=True)
    rows = []
    for i, (source, mag) in enumerate((("ATCC", ""), ("ATCC", ""), ("ScienCell", "100x"))):
        image_id = f"img_{i + 1:04d}"
        (root / "prep" / "images" / f"{image_id}.jpg").write_bytes(b"jpeg bytes")
        rows.append({"image_id": image_id, "file": f"images/{image_id}.jpg", "parent_id": f"par_{i + 1:04d}",
                     "source": source, "image_modality": "Phase contrast", "magnification": mag,
                     "height": 10, "width": 12, "channels": 3})
    pd.DataFrame(rows, columns=list(prepare.IMAGES_COLUMNS)).to_csv(root / "prep" / "images.csv", index=False)
    (root / "prep" / "PREPARE.json").write_text(json.dumps({"n_images": 3}))
    (root / "private").mkdir()
    (root / "private" / "PREPARE_PRIVATE.json").write_text(json.dumps(
        {"hcma_root": "/data/Human_Cell_Morphology_Atlas_HCMA", "hcma_root_resolved": "/data/Human_Cell_Morphology_Atlas_HCMA"}))
    return root


def _template(tmp_path: Path, task: str) -> Path:
    template = tmp_path / "template"
    (template / "tools" / "sub").mkdir(parents=True)
    (template / "tools" / "segment.py").write_text("print('segment')\n")
    (template / "tools" / "sub" / "helper.txt").write_text("helper\n")
    (template / "tools" / "__pycache__").mkdir()
    (template / "tools" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (template / "TASK.md").write_text(task, encoding="utf-8")
    return template


def _ledger(root: Path, name: str = "full", metadata: str = "full", call_name: str = "defaults") -> None:
    logs = root / "control" / name / metadata / "nvidia-h100-80gb-hbm3" / "logs"
    logs.mkdir(parents=True)
    records = [{"event": "start", "time": 1.0, "call": "S001", "tool": "segment", "name": call_name},
               {"event": "end", "time": 2.0, "call": "S001", "tool": "segment", "status": "done", "seconds": 600.0,
                "summary": {"n_images": 3, "n_cells": 250, "mean_flow_error": 0.4, "n_failed": 0}},
               {"event": "start", "time": 3.0, "call": "E001", "tool": "embed", "name": "defaults_s0"},
               {"event": "end", "time": 4.0, "call": "E001", "tool": "embed", "status": "failed", "seconds": 5.0}]
    (logs / "tool_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records) + "not json\n")


def test_setup_run_builds_the_workspace(tmp_path, root):
    _ledger(root)
    task = "# Task\n" + "\n".join(f"{k}: {{{{{k}}}}}" for k in PLACEHOLDERS) + "\n"
    run, repo = tmp_path / "runs" / "claude-1-x", tmp_path / "SHAPE-agentic-vrr"
    before = time.time()
    ws = workspace.setup_run(run, root, repo, budget_hours=2.5, gpu_slots=2, template=_template(tmp_path, task))
    assert ws == run.absolute() / "workspace"
    assert (ws / "data" / "images").is_symlink()
    assert Path(ws / "data" / "images").resolve() == (root / "prep" / "images").resolve()
    assert sorted(p.name for p in (ws / "data" / "images").iterdir()) == ["img_0001.jpg", "img_0002.jpg", "img_0003.jpg"]
    for name in ("images.csv", "PREPARE.json"):
        assert (ws / "data" / name).read_bytes() == (root / "prep" / name).read_bytes()
    for name in ("calls", "submission", "logs"):
        assert (ws / name).is_dir() and not list((ws / name).iterdir())
    assert (run / "logs").is_dir() and (run / "traces").is_dir() and (ws / "NOTES.md").is_file()
    assert not (ws / "tools" / "__pycache__").exists()

    config = json.loads((ws / "tools" / "_config.json").read_text())
    assert config == {"run": str(run.absolute()), "ledger": str(run.absolute() / "logs" / "tool_calls.jsonl"),
                      "images_csv": "data/images.csv", "images_dir": "data/images", "gpu_slots": 2,
                      "deadline_file": str(run.absolute() / "logs" / ".deadline"), "has_deadline": True,
                      "min_start_seconds": 600, "repo": str(repo.absolute()), "prep_name": "full",
                      "prep": str(root / "prep"), "metadata": "full",
                      "gpu_exclusive": True, "env": {"TORCH_HOME": str(root / "assets" / "torch"), "HF_HOME": str(root / "assets" / "hf"),
                              "SHAPE_DINOV3_REPO": str(root / "assets" / "torch" / "hub" / "facebookresearch_dinov3_main"),
                              "HF_HUB_OFFLINE": "1", "CELLPOSE_LOCAL_MODELS_PATH": str(root / "assets" / "cellpose"),
                              "SHAPE_FAST_FIGURES": "1", "MPLBACKEND": "Agg"}}
    hashes = json.loads((run / "TOOLS_SHA256.json").read_text())
    assert set(hashes) == {"segment.py", "sub/helper.txt", "_config.json"}
    for rel, digest in hashes.items():
        assert prepare.sha256(ws / "tools" / rel) == digest
    deadline = float((run / "logs" / ".deadline").read_text())
    assert before + 2.5 * 3600 - 5 <= deadline <= time.time() + 2.5 * 3600 + 5
    assert float((ws / ".deadline").read_text()) == deadline
    data = json.loads((run / "DATA_SHA256.json").read_text())
    assert set(data) == {"images.csv", "PREPARE.json"}
    for name, digest in data.items():                             # read-only, hashed outside the workspace
        assert prepare.sha256(ws / "data" / name) == digest
        assert not (ws / "data" / name).stat().st_mode & 0o222
        with pytest.raises(PermissionError):
            open(ws / "data" / name, "a").close()

    text = (ws / "TASK.md").read_text(encoding="utf-8")
    assert "{{" not in text
    assert "DESIGN: The images are the whole atlas: every image of every cell type.\n" in text
    assert "N_IMAGES: 3\n" in text and "`ATCC` 2, `ScienCell` 1 (3 images)." in text
    assert "| `magnification` | 1 / 3 | ATCC 0/2, ScienCell 1/1 |" in text
    assert "| `parent_id` | 3 / 3 |" in text
    assert "BUDGET_HOURS: 2.5\n" in text
    assert "GPU: NVIDIA H100 80GB HBM3 (79.6 GB, compute mode Exclusive_Process: one process per GPU)\n" in text
    assert "GPU_SLOTS: 2\n" in text and "CPUS: 16\n" in text and "MEM_GB: 128\n" in text
    assert "  - `segment` (S001, `defaults`): 10.0 min; n_images 3, n_cells 250, n_failed 0" in text
    assert "E001" not in text                                   # failed control calls are not timings
    for path in ("/data/Human_Cell_Morphology_Atlas_HCMA", root / "private", root / "control", root / "smoke",
                 repo.absolute()):
        assert f"   - `{path}`" in text
    assert f"   - `{root / 'runs'}` (every run except your own, `{run.absolute()}`)" in text
    assert f"   - every entry of `{run.absolute().parent}` except your own run, `{run.absolute()}`\n" in text
    with pytest.raises(SystemExit, match="exists"):
        workspace.setup_run(run, root, repo, budget_hours=1, gpu_slots=1, template=_template(tmp_path / "t2", task))


def test_setup_run_on_a_subset_prep(tmp_path, root):
    pairs = root / "prep_pairs"
    (pairs / "images").mkdir(parents=True)
    images = pd.read_csv(root / "prep" / "images.csv", dtype=str)
    for file in images.file[:2]:
        (pairs / file).write_bytes((root / "prep" / file).read_bytes())
    images[:2].to_csv(pairs / "images.csv", index=False)
    (pairs / "PREPARE.json").write_text(json.dumps({"n_images": 2, "n_types": 1, "per_class": 2,
                                                    "design": subset.DESIGN.format(per_class=2)}))
    _ledger(root, "pairs")
    (root / "prep_old.partial").mkdir()
    task = "\n".join(f"{k}: {{{{{k}}}}}" for k in PLACEHOLDERS)
    run = tmp_path / "runs" / "claude-pairs-1-x"
    ws = workspace.setup_run(run, root, tmp_path / "repo", budget_hours=1, gpu_slots=1,
                             template=_template(tmp_path, task), prep=pairs)
    config = json.loads((ws / "tools" / "_config.json").read_text())
    assert config["prep_name"] == "pairs" and config["prep"] == str(pairs)
    assert Path(ws / "data" / "images").resolve() == (pairs / "images").resolve()
    text = (ws / "TASK.md").read_text(encoding="utf-8")
    assert "DESIGN: The images show 1 cell types, 2 images per type, each from a different micrograph." in text
    assert "N_IMAGES: 2" in text and "(S001, `defaults`)" in text            # the pairs control's ledger
    assert f"   - `{root / 'prep'}`" in text                                 # every other prep is forbidden
    assert f"`{pairs}`" not in text and "prep_old.partial" not in text
    cli_run = tmp_path / "runs" / "cli"                       # `harness setup` without --prep: pairs
    main(["setup", "--run", str(cli_run), "--root", str(root), "--repo", str(tmp_path / "repo"),
          "--gpu-slots", "1", "--template", str(tmp_path / "template")])
    assert json.loads((cli_run / "workspace" / "tools" / "_config.json").read_text())["prep_name"] == "pairs"


def test_blank_metadata():
    images = pd.DataFrame({"image_id": ["img_0001", "img_0002"], "file": ["images/img_0001.jpg", "images/img_0002.png"],
                           "parent_id": ["par_0001", "par_0002"], "source": ["ATCC", "LIVECell"],
                           "image_modality": ["Phase contrast", ""], "magnification": ["100x", ""],
                           "height": ["10", "20"], "width": ["12", "22"], "channels": ["3", "3"]})
    assert workspace.blank_metadata(images, "full").equals(images)
    blank = workspace.blank_metadata(images, "none")
    assert list(blank.columns) == list(images.columns) and len(blank) == 2
    assert (blank[list(workspace.METADATA_COLUMNS)] == "").all().all()
    kept = ["image_id", "file", "height", "width", "channels"]
    assert blank[kept].equals(images[kept]) and images.source.tolist() == ["ATCC", "LIVECell"]   # a copy
    with pytest.raises(ValueError, match="metadata"):
        workspace.blank_metadata(images, "partial")
    summary = {"n_images": 2, "n_parents": 2, "sources": {"ATCC": 1}, "images_with_value": {}, "n_types": 1,
               "per_class": 2, "n_types_cross_source": 1, "design": "d"}
    assert workspace.blank_summary(summary, "full") == summary
    assert workspace.blank_summary(summary, "none") == {"n_images": 2, "n_types": 1, "per_class": 2, "design": "d"}


def test_setup_run_without_metadata(tmp_path, root):
    prep = root / "prep"
    (prep / "PREPARE.json").write_text(json.dumps({"n_images": 3, "n_parents": 3, "sources": {"ATCC": 2, "ScienCell": 1},
                                                   "images_with_value": {"magnification": 1}, "height": [10, 10]}))
    _ledger(root)                                           # only the full-metadata control has run here
    with open(root / "control" / "full" / "full" / "nvidia-h100-80gb-hbm3" / "logs" / "tool_calls.jsonl", "a") as f:
        f.write(json.dumps({"event": "start", "time": 5.0, "call": "B001", "tool": "batch_correct", "name": "d"}) + "\n")
        f.write(json.dumps({"event": "end", "time": 6.0, "call": "B001", "tool": "batch_correct", "status": "done",
                            "seconds": 60.0, "summary": {"n_cells": 250, "n_batches": 4}}) + "\n")
    task = "\n".join(f"{k}: {{{{{k}}}}}" for k in PLACEHOLDERS)
    run = tmp_path / "runs" / "claude-full-none-1-x"
    ws = workspace.setup_run(run, root, tmp_path / "repo", budget_hours=1, gpu_slots=1,
                             template=_template(tmp_path, task), metadata="none")
    full = pd.read_csv(prep / "images.csv", dtype=str, keep_default_na=False)
    seen = pd.read_csv(ws / "data" / "images.csv", dtype=str, keep_default_na=False)
    assert (ws / "data" / "images.csv").read_text().splitlines()[0] == (prep / "images.csv").read_text().splitlines()[0]
    assert seen.equals(workspace.blank_metadata(full, "none"))           # what the scorer recomputes
    assert (seen[list(workspace.METADATA_COLUMNS)] == "").all().all() and seen.image_id.tolist() == full.image_id.tolist()
    assert json.loads((ws / "data" / "PREPARE.json").read_text()) == {"n_images": 3, "height": [10, 10]}
    data = json.loads((run / "DATA_SHA256.json").read_text())
    for name, digest in data.items():                             # the blanked copies, read-only
        assert prepare.sha256(ws / "data" / name) == digest
        assert not (ws / "data" / name).stat().st_mode & 0o222
    config = json.loads((ws / "tools" / "_config.json").read_text())
    assert config["metadata"] == "none" and config["prep_name"] == "full"
    text = (ws / "TASK.md").read_text(encoding="utf-8")
    assert "SOURCES: not given in this run\n" in text and "ATCC" not in text and "ScienCell" not in text
    assert "No technical metadata is given in this run" in text
    for column in workspace.METADATA_COLUMNS:
        assert f"| `{column}` | 0 / 3 |" in text
    assert "DESIGN: The images are the whole atlas: every image of every cell type.\n" in text
    assert "No control with metadata `none` on this GPU yet" in text and "(S001, `defaults`)" in text
    assert "batch_correct's does" in text and "(B001, `d`): 1.0 min; n_cells 250\n" in text
    assert "n_batches" not in text                                    # the full control's source count stays out
    assert f"   - `{prep / 'images.csv'}`" in text and f"   - `{prep / 'PREPARE.json'}`" in text
    # the none control's own timings once it exists; nothing about a fallback
    _ledger(root, metadata="none", call_name="defaults_none")
    cli_run = tmp_path / "runs" / "cli-none"
    main(["setup", "--run", str(cli_run), "--root", str(root), "--repo", str(tmp_path / "repo"), "--gpu-slots", "1",
          "--template", str(tmp_path / "template"), "--prep", str(prep), "--metadata", "none"])
    text = (cli_run / "workspace" / "TASK.md").read_text(encoding="utf-8")
    assert "(S001, `defaults_none`)" in text and "No control with metadata" not in text
    assert json.loads((cli_run / "workspace" / "tools" / "_config.json").read_text())["metadata"] == "none"
    with pytest.raises(SystemExit, match="metadata"):
        workspace.setup_run(tmp_path / "runs" / "bad", root, tmp_path / "repo", budget_hours=1, gpu_slots=1,
                            template=_template(tmp_path / "t3", task), metadata="partial")


def test_prep_names():
    root = Path("/x/full_HCMA")
    assert subset.prep_dir(root, "full") == root / "prep" and subset.prep_dir(root, "pairs") == root / "prep_pairs"
    assert subset.prep_name(root / "prep") == "full" and subset.prep_name(root / "prep_pairs") == "pairs"
    with pytest.raises(ValueError):
        subset.prep_name(root / "images")


def test_setup_run_without_budget_or_timings(tmp_path, root):
    task = "\n".join(f"{{{{{k}}}}}" for k in PLACEHOLDERS)
    ws = workspace.setup_run(tmp_path / "control", root, tmp_path / "repo", budget_hours=None, gpu_slots=1,
                             template=_template(tmp_path, task))
    assert not (ws / ".deadline").exists() and not (tmp_path / "control" / "logs" / ".deadline").exists()
    config = json.loads((ws / "tools" / "_config.json").read_text())
    assert config["has_deadline"] is False and Path(config["deadline_file"]).is_absolute()
    assert "No timing measured on this GPU yet." in (ws / "TASK.md").read_text()


def test_setup_run_fails_on_an_unfilled_placeholder(tmp_path, root):
    with pytest.raises(RuntimeError, match="UNKNOWN"):
        workspace.setup_run(tmp_path / "run", root, tmp_path / "repo", budget_hours=1, gpu_slots=1,
                            template=_template(tmp_path, "{{N_IMAGES}} {{UNKNOWN}}"))
    assert not (tmp_path / "run" / "workspace").exists()          # nothing half-built is left behind


@pytest.mark.skipif(not REAL_TASK.exists(), reason="workspace_template/TASK.md not written yet")
def test_the_real_task_template_is_filled(tmp_path, root):
    template = _template(tmp_path, REAL_TASK.read_text(encoding="utf-8"))
    ws = workspace.setup_run(tmp_path / "run", root, tmp_path / "repo", budget_hours=10, gpu_slots=1,
                             template=template)
    text = (ws / "TASK.md").read_text(encoding="utf-8")
    assert "{{" not in text and "**10 h**" in text


def test_gpu_line_shows_the_compute_mode():
    facts = {"gpu": "NVIDIA A100-SXM4-40GB", "gpu_mem_gb": 40.0, "gpu_compute_mode": "Default", "gpu_exclusive": False}
    assert workspace._gpu_line(facts) == "NVIDIA A100-SXM4-40GB (40 GB, compute mode Default)"
    assert workspace._gpu_line({"gpu": "none", "gpu_mem_gb": None, "gpu_compute_mode": "", "gpu_exclusive": False}) == "none"


def test_node_facts_and_gpu_slug(monkeypatch):
    monkeypatch.setattr(hardware, "_query", lambda fields: ["NVIDIA H100 80GB HBM3", "81559", "Exclusive_Process"]
                        if fields == "name,memory.total,compute_mode" else ["NVIDIA H100 80GB HBM3"])
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "65536")
    assert hardware.node_facts() == {"gpu": "NVIDIA H100 80GB HBM3", "gpu_mem_gb": 79.6,
                                     "gpu_compute_mode": "Exclusive_Process", "gpu_exclusive": True,
                                     "cpus": 8, "mem_gb": 64.0}
    assert hardware.gpu_slug() == "nvidia-h100-80gb-hbm3" == hardware.slug("NVIDIA H100 80GB HBM3")
    assert hardware.slug("NVIDIA GeForce RTX 2080 Ti") == "nvidia-geforce-rtx-2080-ti" and hardware.slug("") == "cpu"
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
    assert hardware.node_facts()["mem_gb"] == 32.0
    monkeypatch.setattr(hardware, "_query", lambda fields: [])
    assert hardware.gpu_slug() == "cpu" and hardware.gpu_slots() == 1
    facts = hardware.node_facts()
    assert facts["gpu"] == "none" and facts["gpu_compute_mode"] == "" and facts["gpu_exclusive"] is False


def test_setup_run_names_the_results_layout_siblings(tmp_path, root):
    """Results layout: the run sits beside $ROOT's dir; TASK.md names what score.needle_patterns flags."""
    _ledger(root)
    task = "# Task\n" + "\n".join(f"{k}: {{{{{k}}}}}" for k in PLACEHOLDERS) + "\n"
    run = tmp_path / "claude-full-full_abcdef12_20261005T000000Z_1"
    ws = workspace.setup_run(run, root, tmp_path / "repo", budget_hours=1, gpu_slots=1,
                             template=_template(tmp_path, task))
    text = (ws / "TASK.md").read_text(encoding="utf-8")
    assert (f"   - every entry of `{tmp_path.absolute()}` except your own run, `{run.absolute()}`, and "
            f"`{root.absolute()}` (its forbidden parts are listed above)") in text
