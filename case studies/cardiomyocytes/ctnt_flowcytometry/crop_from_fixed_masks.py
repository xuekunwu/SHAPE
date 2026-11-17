"""The crop stage of the cTnT pilot as one cohort VRR (`shape.agent.verification.crop_cohort`) on a fixed input bundle.

The bundle (`--inputs`) is the one the earlier crop runs took: `MANIFEST.json` (every file with its sha256 and source
name), `PILOT.json` (the images), `images/<id>.tif`, `masks/<id>.tif` and `traces/<id>.json` (the segmentation
trace with its `selected_parameters`). Every file is checked against the manifest; each mask is linked under its
source name (the tool's defaults read the mask file name). The run writes into a fresh `--root`: the tool calls under
`cache/tool_cache`, `RESOLUTION.json` (the stage's details) and `SUMMARY.json` (the delivered crops' readouts, the
calls, the model use and the time). Model: the SHAPE vision engine of `--model` through the OpenAI-compatible gateway
(`LLM_BASE_URL`, `LLM_API_KEY`), `SHAPE_REASONING_EFFORT` as set, each reply capped at `--max-tokens` (the pipeline's
`CappedEngine`); search strategy `SHAPE_SEARCH_STRATEGY_CROP`, else `SHAPE_SEARCH_STRATEGY`.
"""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry'

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=1, default=str) + "\n")


def bundle(inputs: Path, root: Path) -> tuple[list[dict], dict]:
    """The checked images (original_image, nuclei_mask under its source name, source_image_id) and the shared
    kwargs (min_area and margin when every trace selected the same)."""
    manifest = json.loads((inputs / "MANIFEST.json").read_text())
    for entry in manifest["entries"]:
        path = inputs / entry["relative"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"input differs from the manifest: {entry['relative']}")
    if sha256(inputs / "PILOT.json") != manifest["pilot_sha256"]:
        raise ValueError("PILOT.json differs from the manifest")
    ids = json.loads((inputs / "PILOT.json").read_text())["images"]
    masks = {Path(e["relative"]).stem: e for e in manifest["entries"] if e["relative"].startswith("masks/")}
    folder = root / "masks"
    folder.mkdir()
    images, selected = [], []
    for ident in ids:
        settings=json.loads((inputs/'traces'/f'{ident}.json').read_text())["selected_parameters"]
        alias = folder / Path(masks[ident]["source"]).name
        os.link(inputs / masks[ident]["relative"], alias)
        images.append({"original_image": str(inputs / "images" / f"{ident}.tif"), "nuclei_mask": str(alias),
                       "source_image_id": ident,
                       **{k:settings[k] for k in ('min_area','margin') if settings.get(k) is not None}})
        selected.append(settings)
    shared = {}
    for key in ("min_area", "margin"):
        values = {json.dumps(p.get(key)) for p in selected}
        if len(values) == 1 and selected[0].get(key) is not None:
            shared[key] = selected[0][key]
    return images, shared


def delivered_readouts(paths: dict) -> dict:
    """Means of the delivered crops' native readouts over every image's delivered metadata."""
    rows = [m.get("qc") or {} for p in paths.values() for m in json.loads(Path(p).read_text())["cell_metadata"]]
    out = {"n_crops": len(rows)}
    for key in ("neighbor_contamination", "target_dominance", "target_retention", "centroid_offset"):
        values = [float(q[key]) for q in rows if key in q]
        if values:
            out[f"mean_{key}"] = sum(values) / len(values)
    out["min_target_retention"] = min((float(q["target_retention"]) for q in rows if "target_retention" in q), default=None)
    return out


def usage(llm_calls: list) -> dict:
    """The reasoner's reported tokens summed over its questions (`AgenticVerifier.llm_calls`; the mask check's
    montage questions are not among them)."""
    out = {"input": 0, "output": 0, "reasoning": 0, "total": 0}
    for call in llm_calls:
        u = call.get("usage") or {}
        out["input"] += int(u.get("input_tokens", u.get("prompt_tokens")) or 0)
        out["output"] += int(u.get("output_tokens", u.get("completion_tokens")) or 0)
        out["reasoning"] += int(u.get("reasoning_tokens", u.get("thinking_tokens")) or 0)
        out["total"] += int(u.get("total_tokens") or 0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=8, help="images cropped in parallel within one call")
    parser.add_argument("--hours", type=float, default=None, help="deadline from the start")
    parser.add_argument("--max-tokens", type=int, default=16000,
                        help="per-reply token cap, reasoning included (the pipeline's high-effort stages: 16000)")
    parser.add_argument("--no-mask-check", action="store_true",
                        help="deliver without the vision mask check (CropTask.enable_mask_check=False, as crop v10)")
    parser.add_argument('--crop-policy',choices=['rule-first-balanced','legacy'],default='rule-first-balanced')
    args = parser.parse_args()

    from benchmarks.full_CPJUMP1.harness.pipeline import engine_for,stage_env
    from .reasoning import reasoning_protocol
    from shape.agent.verification import crop_cohort
    from shape.agent.verification.four_stage import strategy_name
    from shape.agent.verification.verifiers.crop import CropTask
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
    from shape.engine import usage as engine_usage

    started = time.time()
    protocol=reasoning_protocol('all-high');protocol['table']['crop']['max_tokens']=args.max_tokens
    os.environ.update(stage_env('shape','crop',protocol))
    repaired=args.crop_policy=='rule-first-balanced'
    args.root.mkdir(parents=True, exist_ok=False)
    images, shared = bundle(args.inputs, args.root)
    meta = {"status": "running", "started": started, "model": args.model, "images": len(images), "shared": shared,
            "reasoning_effort": os.environ.get("SHAPE_REASONING_EFFORT"), "max_tokens": args.max_tokens,
            "search_strategy": strategy_name("crop"),
            "keep_passing": True, "delivery_policy": crop_cohort.DELIVERY_POLICY, "mask_check": not args.no_mask_check,
            'crop_policy':args.crop_policy,'global_execution_wave_limit':8,
            "cores": args.cores, "workers": args.workers, "job": os.environ.get("SLURM_JOB_ID")}
    write(args.root / "RUN_META.json", meta)
    tool, cache = Single_Cell_Cropper_Tool(), args.root / "cache"
    if repaired:
        from shape.tools.single_cell_cropper.geometry_cache import configure
        configure(tool)

    def run_image(image, params):
        return tool.execute(query_cache_dir=str(cache), build_archive=False, **image, **params)

    try:
        engine=engine_for(dict(arm='shape',model=args.model,reasoning={'protocol':protocol}),'crop')
        resolution = crop_cohort.run(
            run_image, images, llm_engine=engine,
            workers=args.workers,
            rule_first=repaired,delivery_objective='balanced_v2' if repaired else 'pass_then_dominance',
            progress_path=str(args.root/'CROP_VRR_PROGRESS.json'),
            task=CropTask(cores=args.cores, deadline=started + args.hours * 3600 if args.hours else None,
                          brief=f"crop every segmented object of {len(images)} cTnT cardiomyocyte images",
                          enable_mask_check=not args.no_mask_check),
            **shared)
        details = resolution.details or {}
        write(args.root / "RESOLUTION.json", {"status": resolution.status, "reason": resolution.reason,
                                              "params": resolution.params, "details": details})
        calls = details.get("calls") or {}
        defaults = next((c for c in calls if c == "default"), None)
        llm = details.get("llm_calls") or []
        summary = {
            "status": resolution.status, "reason": resolution.reason, "choice": details.get("choice"),
            "chosen_call": details.get("call"), "rounds": details.get("rounds"), "calls": len(calls),
            "admission": (details.get("admission") or {}).get("decision"),
            "defaults_readouts": (calls.get(defaults) or {}).get("readouts") if defaults else None,
            "chosen_readouts": (calls.get(details.get("call")) or {}).get("readouts"),
            "best_observed": details.get("best_observed"),
            "delivered": delivered_readouts(details.get("delivered_image_metadata") or {}),
            "llm_calls": len(llm), "tokens": usage(llm),
            "tokens_scope":"verifier decisions only; total_stage_usage also includes preflight and screening",
            "total_stage_usage":engine_usage.snapshot(),
            "seconds": round(time.time() - started, 1)}
        write(args.root / "SUMMARY.json", summary)
        write(args.root / "RUN_META.json", {**meta, "status": "complete", "seconds": summary["seconds"]})
        print("CROP_COHORT_DONE", json.dumps(summary["delivered"]), flush=True)
    except Exception as exc:
        write(args.root / "RUN_META.json", {**meta, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                            "seconds": round(time.time() - started, 1)})
        raise


if __name__ == "__main__":
    main()
