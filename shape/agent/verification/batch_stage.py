"""The system's batch-correction step of solve(): Harmony on the analyzer's embedding, routed by the metadata profile.

After a successful Cell_State_Analyzer_Tool step, when the run's `metadata_profile.MetadataProfile` has batch
labels (`batch_routed`), `solver._batch_followup` calls `run_followup`. The planner never chooses this step
(`stage_map.SYSTEM_ROUTED_TOOLS`). The follow-up:

1. reads the analyzer's h5ad (`integration.build_base_kwargs`' batch branch, `batch_inputs_from_h5ad`), checks,
   fail-closed, that each image's `obs.batch` is the profile's batch label, and that the h5ad's cells still come
   from at least two batch groups of at least `MIN_IMAGES_PER_BATCH` images each (`design_problem`); a full tier
   whose cells no longer carry `MIN_REPLICATED_GROUPS` replicated groups runs as the partial tier
   (`cells_profile`);
2. builds the `BatchTask` from the profile (`batch_task_for`): a calibration view on the cells of the declared
   groups and of the control images in the full tier, none in the partial tier (the frozen admission then
   decides SKIP or ABSTAIN);
3. runs the frozen agentic loop of `BatchCorrectionVerifier` through `Verifier.resolve_managed_stage`, trial 0
   being the uncorrected input (`batch_input_record`), every later trial a `Batch_Corrector_Tool` call through
   `Executor.run_managed_tool`;
4. on CORRECTED_HARMONY, writes a corrected h5ad next to the analyzer's (`write_corrected_h5ad`: X = the
   corrected matrix, obsm["X_dinov3"] = the uncorrected one, the uncorrected states kept), re-clusters it with
   the analyzer's own re-clustering (`Cell_State_Analyzer_Tool.recluster`) at the uncorrected run's resolution,
   and has the existing MorphologicalStateVerifier assess it once (`recluster_corrected`).

The batch loop runs once per embedding (`embedding_digest`: its cells and the bytes of its X): a later analyzer
step on the same embedding records the same decision again and, after a correction, re-clusters the corrected
states with its own states (`_reuse_followup`); a re-run whose X differs is a new embedding. A SKIP, an ABSTAIN or
a step that does not run (`record_not_routed`) keeps the uncorrected states; the follow-up records what happened in
memory. A failure is recorded and raised (a hard error of the step). `cell_states_deliverable` /
`write_cell_states` report both state sets (CELL_STATES.json).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Any, Optional

import numpy as np

from shape.utils.logger import logger

from .metadata_profile import MIN_IMAGES_PER_BATCH, MIN_REPLICATED_GROUPS, MetadataProfile, batch_routed

TOOL = "Batch_Corrector_Tool"
ANALYZER = "Cell_State_Analyzer_Tool"
BATCH_SUB_GOAL = "batch correction (system step, metadata profile)"
RECLUSTER_SUB_GOAL = "re-cluster the batch-corrected embedding (system step, metadata profile)"
CORRECTED_H5AD = "cell_state_analyzed_harmony.h5ad"
DELIVERABLE_JSON = "CELL_STATES.json"
UNCORRECTED_EMBEDDING = "X_dinov3"
CORRECTED = "CORRECTED_HARMONY"
UNCORRECTED = "UNCORRECTED_INPUT"
NOT_ROUTED = "not_routed"
CONTROL_LABEL = "control"          # the calibration label of a control image's cells without a declared group
ASSESSED_ONCE = "assessed_once_resolution_locked"
_OBS_COLUMNS = ("batch", "group", "image_name", "technical_group")
_UNDECLARED = ("default", "", "nan", "None")
_NO_CALL_REASONS = ("initial_trial_failed", "no_call_finished")
BRIEF = ("Cell-level embeddings from the Cell_State_Analyzer_Tool: {n_cells} cells from {n_images} images in "
         "{n_batches} batch groups (batch column: {batch_column}); {n_replicated} reference groups occur in 2 or "
         "more batch groups; {n_controls} control images.")


# ---- inputs ------------------------------------------------------------------------------------------------
def batch_inputs_from_h5ad(path: str) -> dict:
    """The batch corrector's inputs from the analyzer's h5ad: `cls_output` = X (float), `metadata` = the obs
    columns batch / group / image_name / technical_group (strings), `batch_key` "batch", `row_ids` = obs_names
    (unique), and, for provenance, `adata_path` and `fingerprint` (uns["embedding_fingerprint"])."""
    import anndata as ad
    import pandas as pd

    adata = ad.read_h5ad(path)
    if "batch" not in adata.obs:
        raise ValueError(f"{path} has no obs['batch']")
    x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    row_ids = [str(name) for name in adata.obs_names]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"{path}: obs_names are not unique")
    metadata = pd.DataFrame({c: adata.obs[c].astype(str).to_numpy() for c in _OBS_COLUMNS if c in adata.obs})
    return {"cls_output": np.asarray(x, dtype=float), "metadata": metadata, "batch_key": "batch",
            "row_ids": row_ids, "adata_path": str(path), "fingerprint": adata.uns.get("embedding_fingerprint")}


def check_batch_labels(inputs: dict, profile: MetadataProfile) -> None:
    """Fail-closed: every image's obs.batch in the h5ad is the profile's batch label for it. Raises ValueError."""
    meta = inputs["metadata"]
    if "image_name" not in meta:
        raise ValueError("the analyzer's h5ad has no obs['image_name']: its batch labels cannot be checked")
    labels = profile.batch.get("labels") or {}
    bad = []
    for image, values in meta.groupby("image_name")["batch"]:
        found, expected = sorted(set(values)), labels.get(image)
        if expected is None or found != [expected]:
            bad.append(f"{image}: obs.batch {found}, profile {expected!r}")
    if bad:
        raise ValueError("obs.batch does not match the metadata profile's batch labels: " + "; ".join(bad[:5]))


def design_problem(inputs: dict) -> Optional[str]:
    """Why the analyzer's h5ad cannot carry the batch step although the image specs could (images without cells
    drop out): fewer than two batch groups, or a batch group with fewer than `MIN_IMAGES_PER_BATCH` images with
    cells (the gate's per-cell cross-validation would read image identity as batch). None when it can."""
    meta = inputs["metadata"]
    images_per_batch = meta.groupby("batch")["image_name"].nunique()
    if len(images_per_batch) < 2:
        return f"the analyzer's h5ad holds cells from {len(images_per_batch)} batch group(s)"
    small = sorted(str(b) for b, n in images_per_batch.items() if n < MIN_IMAGES_PER_BATCH)
    if small:
        return (f"batch group(s) with fewer than {MIN_IMAGES_PER_BATCH} images with cells in the analyzer's h5ad: "
                + ", ".join(small))
    return None


def _cell_groups(meta, profile: MetadataProfile):
    """Per cell of the h5ad: (image names, groups, control image?, declared group?)."""
    import pandas as pd

    images = meta["image_name"].astype(str) if "image_name" in meta else pd.Series("", index=meta.index)
    group = meta["group"].astype(str) if "group" in meta else pd.Series("default", index=meta.index)
    control = images.isin(set(profile.controls.get("images") or [])).to_numpy()
    return images, group, control, ~group.isin(_UNDECLARED).to_numpy()


def replicated_groups_with_cells(inputs: dict, profile: MetadataProfile) -> list:
    """The non-control groups whose cells in the h5ad come from two or more batch groups."""
    import pandas as pd

    meta = inputs["metadata"]
    _, group, control, declared = _cell_groups(meta, profile)
    reference = declared & ~control
    batch = meta[inputs.get("batch_key", "batch")].to_numpy()
    per_group = pd.Series(batch[reference]).groupby(group.to_numpy()[reference]).nunique()
    return sorted(str(g) for g, n in per_group.items() if n >= 2)


def cells_profile(profile: MetadataProfile, inputs: dict) -> MetadataProfile:
    """The profile the batch step runs under. The tier is decided on the image specs; a full tier whose produced
    cells carry fewer than `MIN_REPLICATED_GROUPS` replicated non-control groups (e.g. every image of a group
    yielded no cell) runs as the partial tier, with the reason (`reasons[-1]`): no calibration view, so the frozen
    admission ABSTAINs (biology preservation cannot be verified)."""
    from dataclasses import replace

    if profile.tier != "full":
        return profile
    replicated = replicated_groups_with_cells(inputs, profile)
    if len(replicated) >= MIN_REPLICATED_GROUPS:
        return profile
    return replace(profile, tier="partial", reasons=profile.reasons + (
        f"after segmentation and cropping only {len(replicated)} non-control groups have cells in 2 or more batch "
        f"groups (at least {MIN_REPLICATED_GROUPS} needed): biology preservation cannot be verified",))


def batch_input_record(base_kwargs: dict) -> dict:
    """Trial 0, the uncorrected input (`uncorrected_result`): a measurement, not a tool call."""
    from .verifiers.batch_correction_unified import uncorrected_result

    meta = base_kwargs["metadata"]
    return uncorrected_result(base_kwargs["cls_output"], meta[base_kwargs.get("batch_key", "batch")].to_numpy())


def batch_task_for(profile: MetadataProfile, inputs: dict):
    """The `BatchTask` of the run: the per-row batch labels, the tool input, the stated noise, the brief and, in
    the full tier, the calibration view: the cells of the declared groups and of the control images (label = the
    group, `CONTROL_LABEL` for a control image without one; Metadata_PertType "negcon" for the control images'
    cells, else "trt"). The partial tier has no calibration view. The brief counts what the h5ad holds (its cells,
    their images and batch groups), not the image specs."""
    import pandas as pd

    from .verifiers.batch_correction_unified import NOISE_BAND, BatchTask, CalibrationView

    meta = inputs["metadata"]
    batch = meta[inputs.get("batch_key", "batch")].to_numpy()
    batch_column = profile.batch.get("key") or "batch"
    images, group, control, declared = _cell_groups(meta, profile)
    calibration = None
    if profile.tier == "full":
        rows = np.flatnonzero(declared | control)
        if len(rows):
            view = pd.DataFrame({"group": np.where(declared, group.to_numpy(), CONTROL_LABEL)[rows],
                                 "batch": batch[rows],
                                 "Metadata_PertType": np.where(control[rows], "negcon", "trt")})
            calibration = CalibrationView(rows=rows, meta=view, batch_key="batch", label_key="group")
    brief = BRIEF.format(n_cells=len(batch), n_images=int(images.nunique()), n_batches=len(set(batch)),
                         batch_column=batch_column, n_replicated=len(replicated_groups_with_cells(inputs, profile)),
                         n_controls=int(images[control].nunique()))
    return BatchTask(batch=batch, features=inputs["cls_output"], calibration=calibration, noise_band=NOISE_BAND,
                     brief=brief, batch_column=batch_column)


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def embedding_digest(inputs: dict) -> str:
    """sha256 of the embedding the batch step reads: its cells (`row_ids`) and the bytes of its whole X as float32
    (a full hash, about a second per GB: a re-run of the analyzer whose fine-tune is not bit-reproducible is a new
    embedding)."""
    digest = hashlib.sha256(json.dumps(inputs.get("row_ids"), default=str).encode())
    digest.update(memoryview(np.ascontiguousarray(inputs["cls_output"], dtype=np.float32)).cast("B"))
    return digest.hexdigest()


def _tool_runner(executor, base_kwargs: dict, trials_dir: str):
    """One Batch_Corrector_Tool call through `Executor.run_managed_tool`; its corrected matrix is saved as float32
    to `trials_dir/<sha256(config)[:12]>.npy` (short names: Windows' path length) and replaced by a memory map."""
    def runner(params):
        result = executor.run_managed_tool(TOOL, base_kwargs, params)
        if isinstance(result, dict) and result.get("corrected_cls_output") is not None:
            os.makedirs(trials_dir, exist_ok=True)
            path = os.path.join(trials_dir, f"{_sha256(dict(params.get('harmony_params') or {}))[:12]}.npy")
            np.save(path, np.asarray(result["corrected_cls_output"], dtype=np.float32))
            result["corrected_cls_output"] = np.load(path, mmap_mode="r")
            result["corrected_cls_output_path"] = path
        return result
    return runner


def _llm_engine(executor):
    return (getattr(executor, "verifier_llm_engine_mm", None)
            if getattr(executor, "enable_llm_verifier", False) else None)


def run_batch_stage(executor, base_kwargs: dict, profile: MetadataProfile, request: Optional[str] = None,
                    embedding: Optional[str] = None):
    """The frozen batch loop on the analyzer's embedding: `Verifier.resolve_managed_stage` with trial 0 the
    uncorrected input and the profile's `BatchTask`, under the verifier's threads per call (`blas_limits`). The trials' matrices
    go to a directory of their own embedding (`batch_correction/trials/<embedding_digest[:12]>/`). Returns the
    StageResolution."""
    from shape.agent.verifier import Verifier

    from .stage_map import stage_by_name
    from .verifiers.batch_correction_unified import blas_limits

    task = batch_task_for(profile, base_kwargs)
    trials_dir = os.path.join(getattr(executor, "query_cache_dir", None) or "solver_cache", "batch_correction",
                              "trials", (embedding or embedding_digest(base_kwargs))[:12])
    try:
        limits = blas_limits(task.features, task.cores)
    except ImportError:
        limits = contextlib.nullcontext()
    with limits:
        return Verifier().resolve_managed_stage(
            stage_by_name("batch_correction"), _tool_runner(executor, base_kwargs, trials_dir), {},
            batch_input_record(base_kwargs), base_kwargs=base_kwargs, llm_engine=_llm_engine(executor),
            tool_name=TOOL, request=request, upstream_ctx={"batch_task": task}, metadata_profile=profile)


# ---- the corrected states -----------------------------------------------------------------------------------
def write_corrected_h5ad(adata_path: str, corrected, state_key: Optional[str], provenance: dict) -> str:
    """A sibling of the analyzer's h5ad, `cell_state_analyzed_harmony.h5ad`: X = the corrected matrix,
    obsm["X_dinov3"] = the uncorrected X, obs[f"{state_key}_uncorrected"] and obsm["X_umap_uncorrected"] = the
    uncorrected states; the neighbours, obsp, the other clusterings, PCA and X_umap dropped (they describe the
    uncorrected X); uns["batch_correction"] = the provenance (JSON); the embedding fingerprint kept. The analyzer's
    own h5ad is not modified."""
    import anndata as ad
    import pandas as pd

    adata = ad.read_h5ad(adata_path)
    corrected = np.asarray(corrected, dtype=np.float32)
    if corrected.ndim != 2 or corrected.shape[0] != adata.n_obs:
        raise ValueError(f"corrected matrix {corrected.shape} does not match {adata.n_obs} cells")
    obs = adata.obs.copy()
    if state_key and state_key in obs:
        obs[f"{state_key}_uncorrected"] = obs[state_key]
    obs = obs.drop(columns=[c for c in obs.columns if str(c).startswith("leiden_")
                            and not str(c).endswith("_uncorrected")])
    # the delivered call may reduce the input (`input_pca`, e.g. the deterministic fallback without a reasoner):
    # its corrected matrix then has fewer columns, named harmony_<i>
    var = (adata.var.copy() if corrected.shape[1] == adata.n_vars
           else pd.DataFrame(index=[f"harmony_{i}" for i in range(corrected.shape[1])]))
    out = ad.AnnData(X=corrected, obs=obs, var=var)
    out.obsm[UNCORRECTED_EMBEDDING] = np.asarray(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X)
    if "X_umap" in adata.obsm:
        out.obsm["X_umap_uncorrected"] = np.asarray(adata.obsm["X_umap"])
    for key, value in adata.uns.items():
        if key in ("neighbors", "umap", "pca", "clustering_representation") or str(key).startswith("leiden") \
                or str(key).endswith("_colors"):
            continue
        out.uns[key] = value
    out.uns["batch_correction"] = json.dumps(provenance, sort_keys=True, default=str)
    path = os.path.join(os.path.dirname(os.path.abspath(adata_path)), CORRECTED_H5AD)
    out.write(path)
    return path


def recluster_corrected(corrected_path: str, fingerprint, start_params: dict):
    """Re-cluster the corrected h5ad with the analyzer's own re-clustering (`recluster`, fail-closed: it never
    trains) at the uncorrected run's resolution and neighbours, then have the existing MorphologicalStateVerifier
    assess it ONCE. The resolution is locked (a cleaner comparison with the uncorrected states), so a retune could
    only re-run the same clustering: there is none. The verifier's verdict is recorded as it is, and the stage ends
    as the linear loop ends a stage on that verdict with no retune left (reason `ASSESSED_ONCE`): accepted ->
    satisfied; conceded -> conceded with the verifier's fault; a retune it asks for -> conceded, ceiling. So the
    same verdict reads the same on the corrected and the uncorrected states. Returns (StageResolution, the locked
    resolution)."""
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool
    from shape.utils.schemas import StageResolution

    from .stage_map import stage_by_name, verifier_cls_for
    from .verifiers.morphological_state import adapt_morphological

    params = {"leiden_resolution": float(start_params["leiden_resolution"]),
              "n_neighbors": int(start_params["n_neighbors"])}
    result = Cell_State_Analyzer_Tool().recluster(corrected_path, fingerprint, params["leiden_resolution"],
                                                  params["n_neighbors"])
    if result is None:
        raise RuntimeError(f"re-clustering refused {corrected_path}: unreadable, or its fingerprint differs")
    spec = stage_by_name("morphological_state")
    verifier = verifier_cls_for(spec, ANALYZER)(**(spec.verifier_kwargs or {}))
    vres = verifier.verify(*adapt_morphological(result["adata_path"], resolution=params["leiden_resolution"]))
    if vres.action in ("accept", "exclude_samples"):
        status, fault = ("satisfied" if vres.action == "accept" else "exclude_samples"), None
    else:
        status, fault = "conceded", (vres.fault or "ceiling") if vres.action == "conceded" else "ceiling"
    res = StageResolution(result=result, status=status, params=dict(params), verdict=vres,
                          trials=[(dict(params), vres, result)],
                          upstream_target=vres.upstream_target if fault == "upstream" else None,
                          fault=fault, reason=ASSESSED_ONCE)
    return res, params["leiden_resolution"]


# ---- the follow-up ------------------------------------------------------------------------------------------
def _analyzer_entry(result) -> Optional[dict]:
    if not isinstance(result, dict):
        return None
    if isinstance(result.get("per_image"), list) and result["per_image"]:
        return result["per_image"][0] if isinstance(result["per_image"][0], dict) else None
    return result


def _batch_record(res, profile: MetadataProfile, inputs: dict) -> dict:
    """The memory record of the batch stage: no matrix, only what was decided and measured (and where the
    delivered corrected matrix is saved)."""
    from .base import INPUT

    details = getattr(res, "details", None) or {}
    names = [INPUT] + [c for r in details.get("rounds") or [] for c in r]     # trial index -> its name
    selected = details.get("selected")
    inp = res.trials[0][2] if res.trials and isinstance(res.trials[0][2], dict) else {}
    delivered = res.result if isinstance(res.result, dict) else None
    chosen = delivered if delivered and selected not in (None, 0) else None
    corrected = bool(chosen and chosen.get("status") == CORRECTED)
    failed = res.reason in _NO_CALL_REASONS
    record = {
        "status": "failed" if failed else (CORRECTED if corrected else UNCORRECTED),
        "decision": (details.get("admission") or {}).get("decision"),
        "resolution": {"status": res.status, "reason": res.reason, "fault": res.fault},
        "tier": profile.tier,
        "gate": inp.get("batch_effect"),
        "gt_free_metrics": (delivered or inp).get("gt_free_metrics"),
        "harmony": delivered.get("harmony") if corrected else None,
        "selected_call": names[selected] if isinstance(selected, int) and selected < len(names) else None,
        "failed_calls": list(details.get("failed") or []),
        "uncorrected_adata_path": inputs.get("adata_path"),
        "corrected_adata_path": None,
        "corrected_matrix_path": chosen.get("corrected_cls_output_path") if corrected else None,
        "embedding_fingerprint": inputs.get("fingerprint"),
    }
    if chosen and not corrected and not failed:
        record["selected_status"] = chosen.get("status")
        record["note"] = (f"the selected call {record['selected_call']} returned {chosen.get('status')} (its gate "
                          "found no batch effect after its input processing): not a Harmony correction, so the "
                          "uncorrected states are kept")
    if failed:
        errors = [f.get("error") for f in record["failed_calls"] if f.get("error")]
        record["error"] = (f"every {TOOL} call failed ({res.reason})"
                           + (f": {str(errors[0])[-300:]}" if errors else ""))
    return record


def _failed_record(profile: MetadataProfile, error: str, adata_path=None) -> dict:
    return {"status": "failed", "error": error, "decision": None, "resolution": None, "tier": profile.tier,
            "uncorrected_adata_path": adata_path, "corrected_adata_path": None}


def _add(memory, step, tool_name, sub_goal, command, result, params=None, verification=None,
         planner_reasoning="system step (metadata profile), not a planner step") -> None:
    memory.add_action(step_count=step, tool_name=tool_name, sub_goal=sub_goal, command=command, result=result,
                      params=params, verification=verification, planner_reasoning=planner_reasoning)


def record_not_routed(memory, step, reason: str, sub_goal: str = BATCH_SUB_GOAL,
                      planner_reasoning: Optional[str] = None) -> dict:
    """Record that the batch step did not run here, and why (the deliverable reports the reason)."""
    record = {"status": NOT_ROUTED, "reason": reason,
              "summary": (f"{TOOL} is a system step that follows the analyzer when batch labels exist; not run: "
                          f"{reason}.")}
    _add(memory, step, TOOL, sub_goal, "[not executed: system-routed tool]", record,
         planner_reasoning=planner_reasoning or "system step (metadata profile), not a planner step")
    return record


def analyzer_wait_reason(memory) -> Optional[str]:
    """Why the batch step waits for the analyzer's re-run, or None: the newest analyzer step conceded upstream for
    another reason than batch dominance (the representation stage's collapse / size / consistency concedes, to the
    image loading or the crop), so the planner re-runs upstream and this embedding is about to be replaced. A
    concede as batch-dominated (the morphology stage's batch_driven or "significant but batch-dominated" gate,
    upstream_target "representation") is the case Harmony exists for: no wait."""
    from .integration import _latest_analyzer_action

    verification = (_latest_analyzer_action(memory) or {}).get("verification")
    if isinstance(verification, dict) and isinstance(verification.get("per_image"), list) \
            and verification["per_image"]:
        verification = verification["per_image"][0]
    if not isinstance(verification, dict) or verification.get("status") != "conceded":
        return None
    stages = [s for s in verification.get("stages") or [] if isinstance(s, dict)]
    last = stages[-1] if stages else {}
    target = verification.get("upstream_target")
    if last.get("fault") != "upstream" or target == "representation":
        return None
    return (f"the analyzer conceded upstream ({last.get('stage')} -> {target}): its embedding is about to be "
            "replaced")


def _latest_analyzer_entry(memory, adata_path) -> dict:
    """The newest analyzer result entry that delivered `adata_path` (its cluster key and resolution)."""
    return next((_analyzer_entry(a.get("result")) for a in reversed(memory.get_actions(llm_safe=False))
                 if a.get("tool_name") == ANALYZER
                 and (_analyzer_entry(a.get("result")) or {}).get("adata_path") == adata_path), None) or {}


def _provenance(record: dict, profile: MetadataProfile, adata_path: str) -> dict:
    return {"decision": record.get("decision"), "selected_call": record.get("selected_call"), "tier": profile.tier,
            "harmony": record.get("harmony"), "source_adata_path": adata_path}


def run_followup(executor, memory, profile: Optional[MetadataProfile], step: int, request: Optional[str] = None,
                 cache: Optional[dict] = None) -> Optional[dict]:
    """The batch follow-up of the newest analyzer step (the module docstring). Records the batch action and, after
    a correction, the re-cluster action in memory; returns the batch record, or None when not routed or when this
    analyzer step was already handled. `cache` maps (h5ad, embedding fingerprint) to the embedding's digest
    (`embedding_digest`), its batch record and the analyzer steps handled: the batch loop runs once per embedding,
    and every later analyzer step on it gets the same decision recorded again, with its corrected states
    re-clustered (`_reuse_followup`); another digest under the same key is a new embedding, decided again.
    A failure (e.g. a reasoner failure of the batch loop) is recorded as a failed batch record and then raised: the
    caller reports it as a hard error."""
    if not batch_routed(profile):
        return None
    from .integration import _latest_analyzer_action, _latest_analyzer_h5ad, build_base_kwargs, \
        make_verification_record

    adata_path, done = None, None
    try:
        adata_path = _latest_analyzer_h5ad(memory)
        if not adata_path:
            raise ValueError("no Cell_State_Analyzer_Tool h5ad in memory")
        action = _latest_analyzer_action(memory) or {}
        analyzer_step = action.get("step_id") or id(action)
        inputs = build_base_kwargs(TOOL, None, memory)
        analyzer = _latest_analyzer_entry(memory, adata_path)
        key, embedding = (os.path.abspath(adata_path), inputs.get("fingerprint")), embedding_digest(inputs)
        done = cache.get(key) if cache is not None else None
        if done is not None and done["embedding"] == embedding:
            if analyzer_step in done["analyzer_steps"]:
                logger.info("batch follow-up: this analyzer step was already handled (%s)", key[0])
                return None
            done["analyzer_steps"].add(analyzer_step)
            return _reuse_followup(memory, step, done["record"], analyzer, adata_path, profile)
        if cache is not None:     # a first embedding, or another one under the same file and fingerprint
            cache[key] = done = {"embedding": embedding, "record": None, "analyzer_steps": {analyzer_step}}
        check_batch_labels(inputs, profile)
        problem = design_problem(inputs)
        if problem:
            record = record_not_routed(memory, step, problem)
            if done is not None:
                done["record"] = record
            return record
        step_profile = cells_profile(profile, inputs)
        res = run_batch_stage(executor, inputs, step_profile, request, embedding)
        record = _batch_record(res, step_profile, inputs)
        if step_profile is not profile:
            record["tier_reason"] = step_profile.reasons[-1]
        vrecord = make_verification_record(TOOL, [res], res.status)
        # a system follow-up: its readouts stay in `stages`; the latest stage metrics (`latest_stage_metrics`,
        # the next stage's upstream context and the final review) remain the analyzer's
        vrecord["metrics"] = {}
    except Exception as error:      # recorded, then raised: a failed batch step is an error
        logger.error(f"batch follow-up failed ({error})")
        record = _failed_record(profile, f"{type(error).__name__}: {error}", adata_path)
        _add(memory, step, TOOL, BATCH_SUB_GOAL, f"[system] {TOOL} (failed)", record)
        if done is not None and done["record"] is None:
            done["record"] = record
        raise

    if done is not None:
        done["record"] = record
    corrected = res.result.get("corrected_cls_output") if record["status"] == CORRECTED else None
    if corrected is not None:
        try:
            record["corrected_adata_path"] = write_corrected_h5ad(
                adata_path, corrected, analyzer.get("cluster_key"), _provenance(record, profile, adata_path))
        except Exception as error:  # noqa: BLE001
            logger.warning(f"corrected h5ad not written ({error})")
            record["corrected_h5ad_error"] = f"{type(error).__name__}: {error}"
    _add(memory, step, TOOL, BATCH_SUB_GOAL, f"[system] {TOOL} (decision {record['decision']})", record,
         params=dict(res.params or {}), verification=vrecord)
    if record.get("corrected_adata_path"):
        _recluster_followup(memory, step, record, analyzer)
    return record


def _reuse_followup(memory, step, cached: Optional[dict], analyzer: dict, adata_path: str,
                    profile: MetadataProfile) -> Optional[dict]:
    """A later analyzer step on an embedding the batch loop already decided on (the same h5ad, fingerprint and
    cells; e.g. a re-run of the analyzer, at the same or a new cluster resolution): the same decision, recorded
    again (`reused`); after a correction, the corrected h5ad is rebuilt from the embedding's own saved corrected
    matrix with this step's uncorrected states, and re-clustered at this step's resolution."""
    if cached is None:
        return None
    record = dict(cached, reused=True)
    if cached.get("status") != NOT_ROUTED:
        record["uncorrected_adata_path"] = adata_path
    record.pop("corrected_h5ad_error", None)
    if cached.get("corrected_adata_path"):
        try:
            corrected = np.load(cached["corrected_matrix_path"])
            record["corrected_adata_path"] = write_corrected_h5ad(
                adata_path, corrected, analyzer.get("cluster_key"), _provenance(record, profile, adata_path))
        except Exception as error:  # noqa: BLE001
            logger.warning(f"corrected h5ad not rebuilt ({error})")
            record["corrected_adata_path"] = None
            record["corrected_h5ad_error"] = f"{type(error).__name__}: {error}"
    _add(memory, step, TOOL, BATCH_SUB_GOAL,
         f"[system] {TOOL} (decision {record.get('decision')}, reused for this embedding)", record)
    if record.get("corrected_adata_path"):
        _recluster_followup(memory, step, record, analyzer)
    return record


def _recluster_followup(memory, step, record: dict, analyzer: dict) -> None:
    """Re-cluster the corrected h5ad and record it (a Cell_State_Analyzer_Tool action with `batch_corrected`); a
    failure is a failed re-cluster record (the uncorrected states stay the deliverable). The action keeps its
    verification metrics: after a correction, the latest stage metrics (`latest_stage_metrics`: the next stage's
    upstream context and the final vision review) are the delivered corrected states' morphology readouts."""
    from .integration import make_verification_record
    from .stage_map import stage_by_name

    start = {"leiden_resolution": float(analyzer.get("cluster_resolution", 0.5) or 0.5),
             "n_neighbors": int(analyzer.get("n_neighbors", 15) or 15)}
    marks = {"batch_corrected": True, "embedding": "harmony",
             "uncorrected_adata_path": record["uncorrected_adata_path"]}
    try:
        res, resolution = recluster_corrected(record["corrected_adata_path"], record.get("embedding_fingerprint"),
                                              start)
        result = dict(res.result or {}, **marks, resolution_locked=resolution)
        vrecord = make_verification_record(ANALYZER, [res], res.status, stages=[stage_by_name("morphological_state")])
        _add(memory, step, ANALYZER, RECLUSTER_SUB_GOAL, f"[system] {ANALYZER}.recluster(**{res.params})", result,
             params=dict(res.params), verification=vrecord)
    except Exception as error:      # noqa: BLE001
        logger.warning(f"re-clustering the corrected embedding failed ({error})")
        _add(memory, step, ANALYZER, RECLUSTER_SUB_GOAL, f"[system] {ANALYZER}.recluster (failed)",
             {"status": "failed", "error": f"{type(error).__name__}: {error}", **marks,
              "corrected_adata_path": record["corrected_adata_path"]})


# ---- the deliverable ----------------------------------------------------------------------------------------
def _n_clusters(adata_path, state_key) -> Optional[int]:
    try:
        import anndata as ad

        adata = ad.read_h5ad(adata_path, backed="r")
        try:
            return int(adata.obs[state_key].nunique()) if state_key in adata.obs else None
        finally:
            adata.file.close()
    except Exception:               # noqa: BLE001 -- a missing count is reported as None
        return None


def _qc(verification) -> dict:
    """A state set's morphology outcome as recorded, the same fields for both sets: the morphology stage's status,
    reason, decision (PASS / WARNING / FAIL), action and notes (the record's own status without that stage)."""
    if isinstance(verification, dict) and isinstance(verification.get("per_image"), list) \
            and verification["per_image"]:
        verification = verification["per_image"][0]
    if not isinstance(verification, dict):
        verification = {}
    stage = next((s for s in verification.get("stages") or [] if s.get("stage") == "morphological_state"), {})
    return {"qc_status": stage.get("status") or verification.get("status"), "qc_reason": stage.get("reason"),
            "qc_decision": stage.get("decision"), "qc_action": stage.get("action"), "qc_notes": stage.get("notes")}


def _states(action, adata_key="adata_path") -> dict:
    entry = _analyzer_entry(action.get("result")) or {}
    if entry.get("status") == "failed":
        return {"adata_path": entry.get("corrected_adata_path"), "state_key": None, "embedding_key": "X",
                "n_clusters": None, "qc_status": "failed", "qc_reason": None, "qc_decision": None,
                "qc_action": None, "qc_notes": None, "error": entry.get("error")}
    path, key = entry.get(adata_key), entry.get("cluster_key")
    return {"adata_path": path, "state_key": key, "embedding_key": "X",
            "n_clusters": _n_clusters(path, key) if path and key else None, **_qc(action.get("verification"))}


def cell_states_deliverable(memory, profile: Optional[MetadataProfile]) -> dict:
    """Both state sets of the run: the newest uncorrected analyzer states, the corrected ones re-clustered after
    them (if any), which is delivered (corrected only when its re-cluster succeeded), and the batch step after
    them (a not_routed record only when there is no other, for its reason)."""
    actions = memory.get_actions(llm_safe=False) if hasattr(memory, "get_actions") else []
    index = next((i for i in range(len(actions) - 1, -1, -1)
                  if actions[i].get("tool_name") == ANALYZER
                  and not (_analyzer_entry(actions[i].get("result")) or {}).get("batch_corrected")
                  and (_analyzer_entry(actions[i].get("result")) or {}).get("adata_path")), None)
    later = actions[index + 1:] if index is not None else []
    batch_actions = [a for a in later if a.get("tool_name") == TOOL]
    batch_action = next((a for a in reversed(batch_actions) if (a.get("result") or {}).get("status") != NOT_ROUTED),
                        batch_actions[-1] if batch_actions else None)
    recluster = next((a for a in reversed(later) if a.get("tool_name") == ANALYZER
                      and (_analyzer_entry(a.get("result")) or {}).get("batch_corrected")), None)
    uncorrected = _states(actions[index]) if index is not None else None
    corrected = _states(recluster) if recluster is not None else None
    batch = (batch_action or {}).get("result") or {}
    harmony = batch.get("harmony") or {}
    delivered = ("corrected" if corrected and corrected["qc_status"] != "failed"
                 else "uncorrected" if uncorrected else None)
    return {
        "delivered": delivered,
        "uncorrected": uncorrected,
        "corrected": corrected,
        "batch_correction": {
            "routed": batch_routed(profile), "tier": batch.get("tier") or getattr(profile, "tier", None),
            "tier_reason": batch.get("tier_reason"),
            "decision": batch.get("decision"), "status": batch.get("status"),
            "reason": (batch.get("error") or batch.get("reason") or batch.get("corrected_h5ad_error")
                       or batch.get("note") or (batch.get("resolution") or {}).get("reason")),
            "gate": batch.get("gate"), "selected_call": batch.get("selected_call"),
            "harmony_params": harmony.get("parameters"), "failed_calls": batch.get("failed_calls") or []},
    }


def write_cell_states(memory, profile: Optional[MetadataProfile]) -> dict:
    """`cell_states_deliverable`, also written to CELL_STATES.json next to the uncorrected h5ad (when there is
    one)."""
    deliverable = cell_states_deliverable(memory, profile)
    path = (deliverable.get("uncorrected") or {}).get("adata_path")
    if path:
        out = os.path.join(os.path.dirname(os.path.abspath(path)), DELIVERABLE_JSON)
        with open(out, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(deliverable, handle, indent=2, sort_keys=True, default=str)
    return deliverable
