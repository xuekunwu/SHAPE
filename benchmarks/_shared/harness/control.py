"""The control arm: the five workspace tools at their defaults in a fixed chain, no agent.

    segment --name defaults -> crop --name defaults --masks S001 -> per seed s:
    embed --name defaults_s<s> --crops C001 --set seed=<s>
    -> batch_correct --name defaults_s<s> --embedding E00k --batch-key source
    -> cluster --name defaults_s<s> --embedding B00k

in a workspace built exactly like an agent run's (`setup_run` on the prep `prep_name`: $ROOT/prep
for `full`, $ROOT/prep_<name> otherwise; at the metadata level `metadata`; no deadline), one per
prep, metadata level and GPU model ($ROOT/control/<prep-name>/<metadata>/<gpu-slug>/), or a smoke
run on the prep's first N images ($ROOT/smoke/<prep-name>/<metadata>/<utc>/). Under metadata `none`
the chain is the same: `batch_correct --batch-key source` sees one (empty) batch, has no calibration
view and returns its input unchanged (as on HCMA at any metadata level: no label-free reference groups
replicated across batches, workspace_template/tools/batch_correct.py). The class map is checked before any
tool runs (NO_CLASS_MAP stops the control at once). Each tool
runs in its own process (the harness process never touches the GPU), with the environment the agent's
processes get (the keys agents.py scrubs removed, e.g. SHAPE_SAM_SERVER_URL, so every arm segments
with the same backend); each command is logged in logs/control_commands.jsonl, the provenance the
chain checks for a control run. submission/choice.json names
seed 0's cluster call; every seed is scored (RESULT_s<s>.json, per_type_s<s>.csv, per_image_s<s>.csv);
RESULT.json / per_class.csv / per_type.csv / per_image.csv / umap_scored.png = seed 0, RESULT.json with
the seed spread of the primary endpoint.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .ledger import COMPLETED, CONTROL_COMMANDS, calls, state_columns

Runner = Callable[[Path, str, list, Path], int]


def subprocess_runner(repo: Path) -> Runner:
    """Run `python tools/<tool>.py ARGS` in the workspace, output to `log`; returns the exit code."""
    from .agents import _SCRUB

    def run(workspace: Path, tool: str, args: list, log: Path) -> int:
        env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), os.environ.get("PYTHONPATH", "")) if p)
        with open(log, "w", encoding="utf-8") as handle:
            return subprocess.run([sys.executable, f"tools/{tool}.py", *args], cwd=workspace, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
    return run


def _tail(log: Path, n: int = 20) -> str:
    if not log.exists():
        return "(no log)"
    return "\n".join([line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                      if line.strip()][-n:])


def _call(run: Path, runner: Runner, tool: str, name: str, args: list, statuses: tuple = COMPLETED) -> str:
    """One tool call; returns its call id. Stops the control when it fails."""
    workspace = run / "workspace"
    log = run / "logs" / f"control_{tool}_{name}.log"
    print(f"control: {tool} --name {name} {' '.join(args)}", flush=True)
    started = time.time()
    argv = ["--name", name, *args]
    code = runner(workspace, tool, argv, log)
    with open(run / "logs" / CONTROL_COMMANDS, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": time.time(), "tool": tool, "argv": argv, "returncode": code}) + "\n")
    found = [(cid, r) for cid, r in calls(run).items()
             if (r["start"] or {}).get("tool") == tool and (r["start"] or {}).get("name") == name]
    status = (found[-1][1]["end"] or {}).get("status") if found else None
    if code != 0 or len(found) != 1 or status not in statuses:
        raise SystemExit(f"CONTROL_FAILED at {tool} --name {name}: exit code {code}, "
                         f"{'call ' + found[-1][0] if found else 'no ledger entry'}, status {status} "
                         f"(needs {' or '.join(statuses)}); log {log}:\n{_tail(log)}")
    print(f"control: {found[0][0]} {status} in {time.time() - started:.0f}s", flush=True)
    return found[0][0]


SMOKE_RULE = ("subset prep: whole type pairs from SUBSET_<name>.json, the types round-robin over the sorted "
              "morph_classes (unscored last), each class's types in the selection order, both images of each "
              "chosen type, while the images fit in N; full prep: the first N image ids")


CONTROL_CHAIN = {"embed": "zero_shot",
                 "rule": "the five tools at their defaults, except embed adaptation_depth=zero_shot"}


def smoke_selection(root: Path, prep_name: str, n_images: int, class_map: dict[str, str]) -> dict:
    """The smoke's images: on a subset prep whole type pairs covering as many morph_classes as possible
    (SMOKE_RULE); on the full prep the first `n_images` image ids. {"images", "types" (the chosen
    cell_states), "morph_classes" ({cell_state: morph_class}), "rule"}."""
    from .score import UNSCORED
    from .subset import prep_dir

    if prep_name == "full":
        ids = pd.read_csv(prep_dir(root, prep_name) / "images.csv", dtype=str)["image_id"].head(n_images).tolist()
        labels = pd.read_csv(Path(root) / "private" / "labels.csv", dtype=str, keep_default_na=False)
        types = list(dict.fromkeys(labels.set_index("image_id")["cell_state"].reindex(ids).dropna()))
        return {"images": ids, "types": types, "morph_classes": {t: class_map.get(t) for t in types},
                "rule": SMOKE_RULE}
    record = json.loads((Path(root) / "private" / f"SUBSET_{prep_name}.json").read_text(encoding="utf-8"))
    selection = pd.DataFrame(record["selection"])
    queues: dict[str, list[str]] = {}
    for cell_state in dict.fromkeys(selection["cell_state"]):          # the selection order
        queues.setdefault(class_map.get(cell_state, UNSCORED), []).append(cell_state)
    classes = sorted(c for c in queues if c != UNSCORED) + ([UNSCORED] if UNSCORED in queues else [])
    chosen, n = [], 0
    while any(queues[c] for c in classes):
        for c in classes:
            if not queues[c]:
                continue
            cell_state = queues[c].pop(0)
            size = int((selection["cell_state"] == cell_state).sum())
            if n + size > n_images and chosen:
                queues = {c: [] for c in classes}
                break
            chosen.append(cell_state)
            n += size
    images = selection.loc[selection["cell_state"].isin(chosen), "image_id"].tolist()
    return {"images": images, "types": chosen, "morph_classes": {t: class_map.get(t) for t in chosen},
            "rule": SMOKE_RULE}


def _spread(values: dict) -> dict:
    v = [x for x in values.values() if x is not None]
    return {"values": values, "mean": float(np.mean(v)) if v else None,
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "min": float(min(v)) if v else None, "max": float(max(v)) if v else None}


def run_control(root: Path, repo: Path, *, seeds: list[int], n_images: int | None = None, out: Path | None = None,
                runner: Runner | None = None, prep_name: str = "pairs", metadata: str = "full") -> Path:
    """Build the control workspace on the prep `prep_name` at the metadata level `metadata`, run the
    default chain, score every seed; returns the run dir. A smoke (`n_images`) is scored over its own
    selected images only (score_choice smoke_images). `runner` replaces the subprocess call of a
    tool (tests)."""
    from .hardware import gpu_slots, gpu_slug, node_facts
    from .score import NO_CLASS_MAP, class_map_check, control_dir, markdown, score_choice, write_outputs
    from .subset import prep_dir
    from .workspace import setup_run

    root, repo = Path(root), Path(repo)
    seeds = list(seeds)
    if not seeds:
        raise SystemExit("control: no seeds")
    check = class_map_check(root, prep_name)
    if not check["ok"]:
        raise SystemExit(f"CONTROL_FAILED: {NO_CLASS_MAP}: " + "; ".join(check["problems"]))
    if out is None:
        out = control_dir(root, prep_name, metadata, gpu_slug()) if n_images is None else \
            root / "smoke" / prep_name / metadata / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run = Path(out)
    if run.exists():
        raise SystemExit(f"{run} exists; move it aside to recompute the control")
    workspace = setup_run(run, root, repo, budget_hours=None, gpu_slots=gpu_slots(), prep=prep_dir(root, prep_name),
                          metadata=metadata)
    facts = node_facts()
    tool_config = json.loads((workspace / "tools" / "_config.json").read_text(encoding="utf-8"))
    # an exclusive-process GPU admits one CUDA context: one segment worker
    seg_workers = 1 if tool_config.get("gpu_exclusive") else max(1, min(4, int((facts.get("gpu_mem_gb") or 0) // 6)))
    crop_workers = max(1, min(16, int(facts.get("cpus") or 1)))
    images, smoke = [], None
    if n_images is not None:
        smoke = smoke_selection(root, prep_name, n_images, check["map"])
        images = ["--images", ",".join(smoke["images"])]
    runner = runner or subprocess_runner(repo)

    s = _call(run, runner, "segment", "defaults", [*images, "--workers", str(seg_workers)])
    c = _call(run, runner, "crop", "defaults", ["--masks", s, "--workers", str(crop_workers)])
    clusters = {}
    for seed in seeds:
        name = f"defaults_s{seed}"
        e = _call(run, runner, "embed", name, ["--crops", c, "--set", f"seed={seed}", "adaptation_depth=zero_shot"])
        b = _call(run, runner, "batch_correct", name, ["--embedding", e, "--batch-key", "source"])
        clusters[seed] = _call(run, runner, "cluster", name, ["--embedding", b], ("done",))

    k_dir = calls(run)[clusters[seeds[0]]]["dir"]
    resolution = next(iter(state_columns(pd.read_csv(k_dir / "states.csv", nrows=0).columns)), None)
    choices = {seed: {"cluster_call": k, "resolution": resolution, "skipped": {},
                      "reason": f"control: {CONTROL_CHAIN['rule']}, embed seed {seed}"}
               for seed, k in clusters.items()}
    (workspace / "submission" / "choice.json").write_text(json.dumps(choices[seeds[0]], indent=1), encoding="utf-8")

    results = {}
    for seed in seeds:
        result, table, _ = score_choice(run, root, choices[seed], forbidden=[], control=None, lever_coverage=False,
                                        prep_name=prep_name, metadata=metadata, figure=seed == seeds[0],
                                        smoke_images=smoke["images"] if smoke else None)
        result.update(arm="control", seed=seed, n_images_requested=n_images, smoke=smoke, control_chain=dict(CONTROL_CHAIN))
        if result["status"] != "OK":
            raise SystemExit(f"CONTROL_FAILED: seed {seed} scored {result['status']}: "
                             f"{result['class_map']['problems'] + result['inputs']['problems'] + result['chain']['problems']}")
        write_outputs(run, result, table, None, suffix=f"_s{seed}")
        results[seed] = (result, table)
        print(f"control seed {seed}: {result['endpoint']} {result['primary']['value']}", flush=True)
    result, table = results[seeds[0]]
    result = {**result, "seed_spread": _spread({s: r["primary"]["value"] for s, (r, _) in results.items()})}
    write_outputs(run, result, table, None)
    print(markdown(result), flush=True)
    return run
