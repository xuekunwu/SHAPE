"""Build <run>/workspace from the template and the prep (agent runs, the control and the smoke).

<run>/workspace/
  TASK.md            the template with every {{PLACEHOLDER}} filled (an unfilled one is an error)
  NOTES.md           the agent's memory across rounds
  data/images        symlink to <prep>/images (anonymized names only); <prep> = $ROOT/prep (prep name
                     `full`) or a subset $ROOT/prep_<name> (harness/subset.py)
  data/images.csv, data/PREPARE.json   copies; with metadata `none` the images.csv columns
                     parent_id, source, image_modality, magnification are empty for every row
                     (blank_metadata, shared with the scorer) and PREPARE.json drops the counts
                     derived from them
  tools/             copied from the template + _config.json, then read-only; hashed into
                     <run>/TOOLS_SHA256.json
  calls/, submission/, logs/
  .deadline          epoch seconds (only with a budget; the tools read <run>/logs/.deadline)
<run>/DATA_SHA256.json   sha256 of data/images.csv and data/PREPARE.json (both read-only)
"""
from __future__ import annotations

import json
import shutil
import stat
import time
from pathlib import Path

import pandas as pd

from .assets import tool_env
from .prepare import sha256
from .subset import all_preps, prep_name

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "workspace_template"
NOT_METADATA = ("image_id", "file", "height", "width", "channels")
METADATA_LEVELS = ("full", "none")
METADATA_COLUMNS = ("parent_id", "source", "image_modality", "magnification")   # emptied under `none`
SUMMARY_METADATA_KEYS = ("n_parents", "sources", "images_with_value", "n_types_cross_source")
LEDGER_METADATA_KEYS = ("n_batches",)      # tool summary counts derived from the metadata columns


def blank_metadata(images: pd.DataFrame, metadata: str) -> pd.DataFrame:
    """The images table as a run at this metadata level sees it: `full` = unchanged; `none` = the
    columns parent_id, source, image_modality and magnification emptied ("") for every row, the header
    and image_id, file, height, width, channels kept. Returns a copy. `images` as read by
    `pd.read_csv(path, dtype=str, keep_default_na=False)` (the scorer compares the workspace copy,
    read the same way, with blank_metadata(<prep>/images.csv, <run metadata>))."""
    if metadata not in METADATA_LEVELS:
        raise ValueError(f"metadata must be one of {METADATA_LEVELS}, got {metadata!r}")
    out = images.copy()
    if metadata == "none":
        for column in METADATA_COLUMNS:
            if column in out.columns:
                out[column] = ""
    return out


def blank_summary(summary: dict, metadata: str) -> dict:
    """PREPARE.json as the workspace gets it: under `none` without the counts derived from the
    emptied columns (micrographs, sources, per-column counts, cross-source pairs)."""
    if metadata not in METADATA_LEVELS:
        raise ValueError(f"metadata must be one of {METADATA_LEVELS}, got {metadata!r}")
    return dict(summary) if metadata == "full" else {k: v for k, v in summary.items() if k not in SUMMARY_METADATA_KEYS}


def tool_hashes(directory: Path) -> dict[str, str]:
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _sources_line(images: pd.DataFrame) -> str:
    counts = images["source"].replace("", "(none)").value_counts()
    return ", ".join(f"`{name}` {n}" for name, n in counts.items()) + f" ({len(images)} images)."


def _metadata_note(images: pd.DataFrame, metadata: str = "full") -> str:
    """Per metadata column of images.csv: how many images have a value, overall and per source
    (under `none`: the empty columns, without a per-source split)."""
    columns = [c for c in images.columns if c not in NOT_METADATA]
    if metadata == "none":
        lines = ["No technical metadata is given in this run: these columns of `data/images.csv` are empty "
                 "for every image.", "", "| column | images with a value |", "| --- | --- |"]
        lines += [f"| `{c}` | {int((images[c].astype(str) != '').sum())} / {len(images)} |" for c in columns]
        return "\n".join(lines)
    sources = images["source"].replace("", "(none)")
    lines = ["| column | images with a value | per source |", "| --- | --- | --- |"]
    for column in columns:
        has = images[column].astype(str) != ""
        per = ", ".join(f"{name} {int(has[sources == name].sum())}/{int((sources == name).sum())}"
                        for name in sources.value_counts().index)
        lines.append(f"| `{column}` | {int(has.sum())} / {len(images)} | {per} |")
    return "\n".join(lines)


def _timings(root: Path, name: str, metadata: str, slug: str) -> str:
    """Each completed tool call of the control of this prep, metadata level and GPU model: seconds and
    the summary's counts. Without that control, the full-metadata control of the same prep and GPU
    model, saying so and without the counts derived from metadata (LEDGER_METADATA_KEYS)."""
    def ledger_of(level: str) -> Path:
        return root / "control" / name / level / slug / "logs" / "tool_calls.jsonl"

    ledger, note = ledger_of(metadata), None
    if not ledger.exists() and metadata != "full" and ledger_of("full").exists():
        ledger = ledger_of("full")
        note = (f"  - (No control with metadata `{metadata}` on this GPU yet: these are the timings of the "
                "full-metadata control of the same images. The segment, crop, embed and cluster timings do not "
                "depend on metadata; batch_correct's does: without metadata it finds a single batch and returns "
                "its input unchanged.)")
    hidden = LEDGER_METADATA_KEYS if note else ()
    starts, lines = {}, []
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8", errors="replace").split("\n"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("event") == "start":
                starts[rec.get("call")] = rec
            elif rec.get("event") == "end" and rec.get("status") in ("done", "partial"):
                start = starts.get(rec.get("call"), {})
                counts = ", ".join(f"{k} {v}" for k, v in (rec.get("summary") or {}).items()
                                   if k.startswith("n_") and k not in hidden and isinstance(v, int)
                                   and not isinstance(v, bool))
                seconds = rec.get("seconds")
                lines.append(f"  - `{rec.get('tool')}` ({rec.get('call')}, `{start.get('name')}`): "
                             + (f"{seconds / 60:.1f} min" if isinstance(seconds, (int, float)) else "n/a")
                             + (f"; {counts}" if counts else ""))
    if lines and note:
        lines.insert(0, note)
    return "\n".join(lines) if lines else "  - No timing measured on this GPU yet."


def _gpu_line(facts: dict) -> str:
    """e.g. 'NVIDIA H100 80GB HBM3 (79.6 GB, compute mode Exclusive_Process: one process per GPU)'."""
    details = [f"{facts['gpu_mem_gb']:g} GB"] if facts.get("gpu_mem_gb") else []
    if facts.get("gpu_compute_mode"):
        details.append(f"compute mode {facts['gpu_compute_mode']}"
                       + (": one process per GPU" if facts.get("gpu_exclusive") else ""))
    return f"{facts['gpu']} ({', '.join(details)})" if details else str(facts["gpu"])


def _design(summary: dict) -> str:
    """TASK.md's {{DESIGN}} sentence, from the prep's PREPARE.json."""
    if "n_types" in summary and "per_class" in summary:
        return (f"The images show {summary['n_types']} cell types, {summary['per_class']} images per type, "
                "each from a different micrograph.")
    return "The images are the whole atlas: every image of every cell type."


def _forbidden(run: Path, root: Path, repo: Path, private: dict, prep: Path, metadata: str = "full") -> str:
    raw = list(dict.fromkeys(p for p in (private.get("hcma_root"), private.get("hcma_root_resolved")) if p))
    others = [str(p) for p in all_preps(root) if p.absolute() != prep]      # every prep but the run's own
    # under `none` the run's own prep tables hold the withheld metadata (data/images/.. leads there)
    own = [str(prep / "images.csv"), str(prep / "PREPARE.json")] if metadata == "none" else []
    paths = [*raw, str(root / "private"), str(root / "control"), str(root / "smoke"), *others, *own]
    lines = [f"   - `{p}`" for p in paths]
    lines.append(f"   - `{root / 'runs'}` (every run except your own, `{run}`)")
    if run.parent != root / "runs":     # the results layout: runs beside $ROOT (score.needle_patterns)
        held = [p for p in (root, *root.parents) if p.parent == run.parent]      # the entry holding $ROOT
        lines.append(f"   - every entry of `{run.parent}` except your own run, `{run}`"
                     + (f", and `{held[0]}` (its forbidden parts are listed above)" if held else ""))
    lines.append(f"   - `{repo}`")
    return "\n".join(lines)


def setup_run(run: Path, root: Path, repo: Path, *, budget_hours: float | None, gpu_slots: int,
              template: Path | None = None, prep: Path | None = None, metadata: str = "full") -> Path:
    """Create <run>/{workspace,logs,traces} and TOOLS_SHA256.json; return the workspace. `metadata`:
    `full` (the prep's technical metadata) or `none` (blank_metadata: no per-image metadata at all)."""
    from .hardware import gpu_slug, node_facts

    run, root, repo = Path(run).absolute(), Path(root).absolute(), Path(repo).absolute()
    template = Path(template) if template is not None else TEMPLATE
    prep = Path(prep).absolute() if prep is not None else root / "prep"
    workspace = run / "workspace"
    if workspace.exists():
        raise SystemExit(f"{workspace} exists; use a fresh run directory")
    if metadata not in METADATA_LEVELS:
        raise SystemExit(f"metadata must be one of {METADATA_LEVELS}, got {metadata!r}")
    private = json.loads((root / "private" / "PREPARE_PRIVATE.json").read_text(encoding="utf-8"))
    images = blank_metadata(pd.read_csv(prep / "images.csv", dtype=str, keep_default_na=False), metadata)
    summary = json.loads((prep / "PREPARE.json").read_text(encoding="utf-8"))
    this_prep = prep_name(prep)

    facts = node_facts()
    fill = {
        "N_IMAGES": str(len(images)),
        "DESIGN": _design(summary),                     # the image set, not per-image metadata
        "SOURCES": _sources_line(images) if metadata == "full" else "not given in this run",
        "METADATA_NOTE": _metadata_note(images, metadata),
        "BUDGET_HOURS": f"{budget_hours:g}" if budget_hours is not None else "n/a",
        "GPU": _gpu_line(facts),
        "GPU_SLOTS": str(int(gpu_slots)),
        "CPUS": str(facts["cpus"]),
        "MEM_GB": f"{facts['mem_gb']:g}" if facts["mem_gb"] is not None else "unknown",
        "TIMINGS": _timings(root, this_prep, metadata, gpu_slug()),
        "FORBIDDEN": _forbidden(run, root, repo, private, prep, metadata),
    }
    task = (template / "TASK.md").read_text(encoding="utf-8")
    for key, value in fill.items():
        task = task.replace("{{" + key + "}}", value)
    if "{{" in task:
        left = sorted(set(part.split("}}")[0] for part in task.split("{{")[1:]))
        raise RuntimeError(f"unfilled TASK.md placeholder(s): {left}")

    for sub in ("data", "calls", "submission", "logs"):
        (workspace / sub).mkdir(parents=True)
    for sub in ("logs", "traces"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "images").symlink_to(prep / "images", target_is_directory=True)
    if metadata == "full":                                 # the prep's own files, byte for byte
        for name in ("images.csv", "PREPARE.json"):
            shutil.copyfile(prep / name, workspace / "data" / name)
    else:
        images.to_csv(workspace / "data" / "images.csv", index=False)
        (workspace / "data" / "PREPARE.json").write_text(json.dumps(blank_summary(summary, metadata), indent=1),
                                                         encoding="utf-8")
    for name in ("images.csv", "PREPARE.json"):            # read-only copies, hashed outside the workspace
        (workspace / "data" / name).chmod(stat.S_IRUSR | stat.S_IRGRP)
    (run / "DATA_SHA256.json").write_text(json.dumps(
        {name: sha256(workspace / "data" / name) for name in ("images.csv", "PREPARE.json")}, indent=1),
        encoding="utf-8")

    tools = workspace / "tools"
    shutil.copytree(template / "tools", tools, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    config = {"run": str(run), "ledger": str(run / "logs" / "tool_calls.jsonl"),
              "images_csv": "data/images.csv", "images_dir": "data/images", "gpu_slots": int(gpu_slots),
              "deadline_file": str(run / "logs" / ".deadline"), "has_deadline": budget_hours is not None,
              "min_start_seconds": 600, "repo": str(repo), "prep_name": this_prep, "prep": str(prep),
              "metadata": metadata,
              "gpu_exclusive": bool(facts.get("gpu_exclusive", False)), "env": tool_env(root)}
    (tools / "_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    (run / "TOOLS_SHA256.json").write_text(json.dumps(tool_hashes(tools), indent=1), encoding="utf-8")
    read_only_dir = stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
    for path in sorted(tools.rglob("*"), reverse=True):        # read-only: the agent may not edit them
        path.chmod(read_only_dir if path.is_dir() else stat.S_IRUSR | stat.S_IRGRP)
    tools.chmod(read_only_dir)
    if budget_hours is not None:      # logs/.deadline is the one the tools read (outside the workspace)
        deadline = f"{time.time() + budget_hours * 3600:.0f}\n"
        for path in (workspace / ".deadline", run / "logs" / ".deadline"):
            path.write_text(deadline, encoding="utf-8")
    (workspace / "TASK.md").write_text(task, encoding="utf-8")
    (workspace / "NOTES.md").write_text("# NOTES\n\n", encoding="utf-8")
    return workspace
