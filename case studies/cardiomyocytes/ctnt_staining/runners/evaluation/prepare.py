"""Curate aggregate evidence from the immutable private export, without labels."""
from pathlib import Path
import json
import tarfile

from evaluation_common import ARMS, csv_rows, read, sha, write, write_csv


def prepare(export_root, codex_trajectory, output):
    root, trajectory, out = Path(export_root), Path(codex_trajectory), Path(output)
    evidence = out / "evidence"
    provenance = out / "provenance"
    sources = {}

    def load(relative):
        path = root / relative
        sources[relative] = sha(path)
        return read(path)

    def table(relative):
        path = root / relative
        sources[relative] = sha(path)
        return csv_rows(path)

    quality = table("multilayer_evaluation/STAGE_QUALITY.csv")
    write_csv(evidence / "stage_quality.csv", quality)
    write_csv(evidence / "quality_by_well.csv", table("multilayer_evaluation/QUALITY_BY_WELL.csv"))
    costs = [r for r in table("multilayer_evaluation/COSTS.csv") if r["arm"] in ARMS]
    write_csv(evidence / "stage_costs.csv", costs)
    process = load("multilayer_evaluation/PROCESS.json")
    # This is an adapted mapping, not a native harness process score.
    write(evidence / "agentic_mapping.json", process)
    dapi = load("analysis_all_cells_umap_v2/DAPI_CROP.json")
    write(evidence / "dapi_crop_summary.json", {k: v for k, v in dapi.items() if k != "per_image"})
    write(evidence / "probe_protocol.json", load("analysis_own_cells_probe_v1/PROTOCOL.json"))
    write(evidence / "umap_protocol.json", load("analysis_all_cells_umap_v2/PROTOCOL.json"))
    write(evidence / "seg_Q_counterexample.json", load("analysis_own_cells_probe_v1/evidence/Q_PROXY_COUNTEREXAMPLE.json"))
    environment = load("analysis_all_cells_umap_v2/evidence/ENVIRONMENT.json")
    for gpu in environment["hardware"]["gpus"]:
        gpu.pop("uuid", None)
    environment["python"].pop("executable", None)
    write(provenance / "environment.json", environment)
    (provenance / "requirements.observed.txt").write_text(
        "# Observed remote environment; not a tested fresh-install lock.\n"
        + "\n".join(f"{k}=={v}" for k, v in sorted(environment["packages"].items()) if k != "shape") + "\n",
        encoding="utf-8", newline="\n")

    population, probe = {}, {}
    for arm in ARMS:
        probe[arm] = load(f"analysis_own_cells_probe_v1/{arm}/PROBE.json")
        population[arm] = load(f"analysis_all_cells_umap_v2/{arm}/COMPLETE.json")
    write(evidence / "linear_probe.json", probe)
    write(evidence / "population.json", population)

    # Post-freeze diagnostics support optimization hypotheses, not agent inputs.
    mixture = load("analysis_own_cells_probe_v1/evidence/OWN_LABEL_MIXTURE_AUDIT.json")["shape"]
    analysis = load("analysis/SUMMARY.json")
    selection = load("shape/SSL_SELECTION.json")
    search = load("shape/representation_SEARCH.json")
    batch = load("shape/batch_SEARCH.json")
    batch_evidence = batch["details"]["evidence"]
    params = {k: v for k, v in search["selected_params"].items()
              if k not in {"init_checkpoint", "output_subdir"}}
    write(evidence / "shape_optimization_evidence.json", dict(
        status="retrospective_diagnostics_not_a_new_experiment",
        label_access="Fluorescence mixture is post-freeze diagnostic evidence; never passed to native SSL or agents.",
        training_well_mixture={k: mixture[k] for k in (
            "training_wells", "mixed_label_wells", "well_macro_opposite_label_pair_probability")},
        pair_probability_definition="equal-well mean of 2*p*(1-p) among confident training labels; independent with-replacement pairing estimate, not measured actual SSL pair noise",
        ssl_trials=analysis["ssl_trials"], selected_ssl_params=params,
        ssl_validation_scope=selection["validation_scope"],
        ssl_selection_mode=selection["mode"],
        ssl_stop_evidence=analysis["ssl_stop_evidence"],
        batch_admission={
            "decision": batch["admission"]["decision"],
            "reason_code": batch["admission"]["reasoner"]["reason_code"],
            "batch_predictability": batch_evidence["gate_all_rows"]["batch_predictability"],
            "chance_accuracy": batch_evidence["gate_all_rows"]["chance_accuracy"],
            "reference_groups_in_2plus_batch_groups": batch_evidence["reference_groups_in_2plus_batch_groups"],
            "negative_controls_only_gate": batch_evidence["gate_negative_controls_only"],
        },
        changed_segmentation_masks_vs_control=analysis["changed_masks_vs_control"]["shape"],
        interpretation="Proxy improvement, quality improvement and predictive improvement are distinct; proposed gains require new controlled ablations."))

    # Records contain opaque image IDs and scalar outcomes; no cell labels,
    # original filenames, donor identities, arrays, secret config or raw logs.
    events, decisions = [], []
    cp = load("shape/segment/shared_vrr/CHECKPOINT.json")
    for unit, history in cp["history"].items():
        for call, item in history.items():
            events.append(dict(arm="shape", stage="segment", unit=unit, call=call,
                               status="complete", error=None, params=item["params"],
                               is_tool_attempt=True, selected=cp["selected"][unit]["candidate"] == call))
    for i, item in enumerate(cp["errors"]):
        events.append(dict(arm="shape", stage="segment", unit=item["image_id"], call=f"error{i}",
                           status="error", error=item["message"], params=item["params"],
                           is_tool_attempt=True, selected=False))
    if len(events) != len(cp["selected"]) + sum(r["native_calls"] for r in cp["rounds"]):
        raise ValueError("SHAPE completed history plus errors does not reconcile to the attempt ledger")
    write(evidence / "shape_seg_rounds.json", cp["rounds"])
    for path in sorted((root / "shape/segment/shared_vrr").glob("RECEIPT_*.json")):
        relative = path.relative_to(root).as_posix()
        item = load(relative)
        answer = item.get("raw_answer", {})
        # Audits place rationales on reviews, global rules may place them on rules.
        def has_reason(value):
            if isinstance(value, dict):
                return bool(value.get("reason")) or any(has_reason(v) for v in value.values())
            return isinstance(value, list) and any(has_reason(v) for v in value)
        decisions.append(dict(arm="shape", stage="segment", round=int(path.stem.split("_")[-1]),
                              action="cohort_review", outcome=item.get("status", "unknown"),
                              message="", has_reason=bool(has_reason(answer))))

    crop = load("shape/CROP_DECISIONS.json")
    write(evidence / "shape_crop_rounds.json", crop.get("round_progress", []))
    for item, plan in zip(crop["llm_calls"], crop["plans"], strict=True):
        decisions.append(dict(arm="shape", stage="crop", round=item["round"],
                              action="cohort_review", outcome=item["outcome"], message="",
                              has_reason=bool(plan.get("analysis")),
                              tokens=item["usage"].get("total_tokens")))
    for stage in ("segment", "crop", "embed", "batch"):
        call_path = trajectory / stage / "CALLS_COMPACT.json"
        round_path = trajectory / stage / "ROUNDS.json"
        select_path = trajectory / stage / "SELECTED.json"
        for path in (call_path, round_path, select_path):
            sources["codex_trajectory/" + path.relative_to(trajectory).as_posix()] = sha(path)
        calls, rounds, selected = read(call_path)["calls"], read(round_path), read(select_path)
        selected = {unit: item["call"] for unit, item in selected["deliveries"].items()}
        if set(selected) != set(calls) or any(selected[u] not in calls[u] for u in selected):
            raise ValueError(f"Missing selected native call: {stage}")
        for unit, ledger in calls.items():
            for call, item in ledger.items():
                input_only = "not a tool call" in str(item.get("scalar_readouts", {}).get("status", ""))
                events.append(dict(arm="codex", stage=stage, unit=unit, call=call,
                                   status=item["status"], error=item.get("error"), params=item["params"],
                                   is_tool_attempt=not input_only, selected=selected.get(unit) == call))
        for item in rounds:
            decisions.append(dict(arm="codex", stage=stage, round=item["round"],
                                  action=item["action"], outcome=item["outcome"],
                                  message=item.get("message", "") if item["outcome"] == "rejected" else "",
                                  has_reason=bool(item.get("reason")), tokens=item.get("total_tokens"),
                                  requested_model=item.get("requested_model"),
                                  response_model_attested=item.get("response_model_attested")))
    write(evidence / "native_attempts.json", events)
    write(evidence / "controller_decisions.json", decisions)

    configs = {}
    for arm in ARMS:
        relative = "shape/CONFIG.json" if arm == "shape" else f"historical/{arm}/CONFIG.json"
        config = load(relative)
        configs[arm] = {k: config.get(k) for k in ("dataset", "arm", "model", "seed", "source_sha256", "cores")}
        configs[arm]["images"] = len(config["images"])
        configs[arm]["historical_reuse"] = arm != "shape"
    write(provenance / "run_identity.json", dict(
        arms=configs, model_reasoning_effort="high", split_seed=20261017, pilot_seed=20261016,
        full_split_by_well={"train": 0.4, "val": 0.1, "held_out": 0.5},
        pilot={"fraction": 0.2, "wells": 114, "images": 456,
               "wells_by_split": {"train": 46, "val": 11, "held_out": 57}},
        evaluation="own-arm cells; no shared matching; separate supervised frozen-embedding probe",
        native_inputs="brightfield only; DAPI/cTnT used by trusted independent evaluation",
        timing="four native stage elapsed sums; historical resources/concurrency differ"))
    snapshot = root / "source_independent_pilot.tar.gz"
    sources[snapshot.name] = sha(snapshot)
    with tarfile.open(snapshot) as archive:
        files = []
        for member in archive.getmembers():
            if member.isfile() and member.name != "SOURCE_PACK.json":
                import hashlib
                files.append(dict(path=member.name, sha256=hashlib.sha256(archive.extractfile(member).read()).hexdigest()))
    pack = load("SOURCE_PACK.json")
    write(provenance / "native_source_snapshot.json", dict(
        base_commit=pack["base_commit"], scientific_source_sha256=pack["source_sha256"],
        archive_sha256=sha(snapshot), includes_uncommitted_edits=True,
        policy="External immutable archive retained privately; per-file hashes permit exact source verification. Publication commit is analysis code, not native run identity.",
        files=files))
    write(provenance / "private_input_hashes.json", sources)
    print(json.dumps({"curated": str(out), "native_events": len(events), "decision_records": len(decisions)}))
