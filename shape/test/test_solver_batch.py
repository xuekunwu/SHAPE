"""The system's batch step inside solve() (round 5 part 2): a synthetic two-plate chain through the real solve()
loop -- a fake planner, a real Executor with a fake analyzer that writes the analyzer's h5ad (2 plates x 2 groups x
2 images x 40 cells; X = group signal + plate offset + noise), the real Batch_Corrector_Tool, the calibration
scoring replaced by a cheap sklearn function and enable_llm_verifier=False. With harmonypy in the environment the
real Harmony runs; without it a stand-in of `harmonypy.run_harmony` is installed (the tool, the loop and the
readouts are the real ones): per-batch centering, or, for a design on which harmonypy 2.0.0 leaves the input
unchanged, the input unchanged."""
from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool, resolve_technical_metadata

FINGERPRINT = "synthetic-fingerprint"
N_CELLS = 40
DIMS = 16


def _stub_harmonypy(raises: bool = False, unchanged: bool = False):
    """A stand-in for `harmonypy.run_harmony` with its signature: per-batch centering (each batch moved to the
    global mean), the input unchanged (`unchanged`), or a RuntimeError on every call."""
    module = types.ModuleType("harmonypy")

    def run_harmony(data_mat, meta_data, vars_use, theta=None, lamb=None, sigma=0.1, nclust=None, tau=0,
                    block_size=0.05, max_iter_harmony=10, max_iter_kmeans=20, epsilon_cluster=1e-5,
                    epsilon_harmony=1e-4, random_state=0, verbose=True):
        if raises:
            raise RuntimeError("harmony failed")
        x = np.asarray(data_mat, dtype=float)
        batch = np.asarray(meta_data[vars_use]).astype(str)
        out = x.copy()
        if unchanged:
            return SimpleNamespace(Z_corr=out)
        for b in np.unique(batch):
            out[batch == b] += x.mean(axis=0) - x[batch == b].mean(axis=0)
        return SimpleNamespace(Z_corr=out)

    module.run_harmony = run_harmony
    return module


@pytest.fixture
def harmony(monkeypatch):
    """`harmony()`: the real harmonypy when it is installed, else the per-batch-centering stand-in;
    `harmony(unchanged=True)`: the real harmonypy, else a stand-in that returns the input unchanged (what harmonypy
    2.0.0 returns on a design where every soft cluster holds the cells of one plate: its ridge correction skips a
    cluster without a second batch above `batch_prop_cutoff`); `harmony(raises=True)`: every call raises."""
    def install(raises=False, unchanged=False):
        if raises:
            monkeypatch.setitem(sys.modules, "harmonypy", _stub_harmonypy(raises=True))
            return
        try:
            import harmonypy  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "harmonypy", _stub_harmonypy(unchanged=unchanged))
    return install


def _cheap_composite(meta, feats, label_key, batch_key, plate_types, n_neighbors=25):
    """The calibration readouts, cheaply: label silhouette (bio; left out with one label, as `composite_all`
    does) and 1 - |batch silhouette| (batch)."""
    from sklearn.metrics import silhouette_score

    feats = np.asarray(feats, dtype=float)
    batch = 1 - abs(silhouette_score(feats, meta[batch_key].astype(str)))
    if meta[label_key].nunique() < 2:
        return {"silhouette_batch": batch, "batch_mean": batch, "bio_mean": None, "overall": batch,
                "skipped": {"asw": "1 label"}, "weights_used": {"batch": 1.0}}
    bio = (silhouette_score(feats, meta[label_key].astype(str)) + 1) / 2
    return {"asw": bio, "silhouette_batch": batch, "batch_mean": batch, "bio_mean": bio,
            "overall": 0.4 * batch + 0.6 * bio, "skipped": {}, "weights_used": {"batch": 0.4, "bio": 0.6}}


def _specs(tmp_path, plates=True, groups=("ctrl", "drug")):
    specs = []
    for p in (1, 2):
        for g in groups:
            for i in (1, 2):
                image_id = f"p{p}_{g}_{i}"
                path = tmp_path / f"{image_id}.png"
                path.write_bytes(b"not an LFS pointer")
                spec = {"path": str(path), "image_id": image_id, "group": g}
                if plates:
                    spec["plate"] = f"P{p}"
                specs.append(spec)
    return specs


def _fake_analyzer(specs, offset, calls, batch_override=None, area_signal=False, group_signal=3.0, drift=False,
                   no_cells=(), common=0.0):
    """The analyzer's h5ad from synthetic features (with `area_signal`, one feature tracks the cell area, so the
    first PC is size-dominated; with `drift`, each training draws new noise, as a fine-tune that is not
    bit-reproducible would, under the same fingerprint and cells; the images of the `no_cells` groups yield no
    cell; `common`, a mean shared by every cell, on every feature, as real embeddings have one: it leaves the
    Euclidean geometry as it is and brings the cells closer in Harmony's cosine space); the real clustering (`_compute_umap_and_clustering`) and the real fail-closed
    re-clustering on reuse."""
    import anndata as ad
    import pandas as pd

    class FakeAnalyzer(Cell_State_Analyzer_Tool):
        def execute(self, query_cache_dir=None, cluster_resolution=0.5, n_neighbors=15, reuse_embedding=False,
                    **kwargs):
            path = os.path.join(query_cache_dir, "cell_state_analysis", "cell_state_analyzed.h5ad")
            if reuse_embedding and os.path.exists(path):
                reused = self._reuse_cached_embedding(path, FINGERPRINT, cluster_resolution, n_neighbors)
                if reused is not None:
                    return reused
            calls.append(dict(kwargs, cluster_resolution=cluster_resolution))
            rng = np.random.default_rng(len(calls) if drift else 0)
            groups = sorted({s["group"] for s in specs})
            rows, obs = [], []
            for spec in (s for s in specs if s["group"] not in no_cells):
                technical = {k: spec[k] for k in ("plate",) if k in spec}
                _, batches, _ = resolve_technical_metadata([technical], [spec["image_id"]])
                plate = 0 if spec.get("plate") in (None, "P1") else 1
                signal = np.full(DIMS, float(common))
                signal[:4] = group_signal if spec["group"] == groups[-1] else -group_signal
                signal[4:8] += offset * (1 if plate else -1)
                cells = signal + rng.normal(size=(N_CELLS, DIMS))
                areas = rng.uniform(60, 140, N_CELLS)
                if area_signal:
                    cells[:, 8] += (areas - 100) / 2
                rows.append(cells)
                batch = batch_override(spec, batches[0]) if batch_override else batches[0]
                obs += [{"batch": batch, "group": spec["group"], "image_name": spec["image_id"],
                         "technical_group": spec["image_id"], "area": float(area)} for area in areas]
            adata = ad.AnnData(np.vstack(rows).astype(np.float32), obs=pd.DataFrame(obs))
            adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
            adata.uns["embedding_fingerprint"] = FINGERPRINT
            key = self._compute_umap_and_clustering(adata, cluster_resolution, None, n_neighbors)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            adata.write(path)
            return {"summary": "synthetic analysis", "cell_count": int(adata.n_obs), "adata_path": path,
                    "cluster_key": key, "cluster_resolution": cluster_resolution, "n_neighbors": n_neighbors,
                    "analyzer_kwargs": {"augment_color": True}}

    return FakeAnalyzer


class _Planner:
    """Picks the given tools in order (a tuple: a tool and its recommended params). Each pick goes through the
    real planner's name check (`normalize_tool_name`) against the planner's toolbox, which `construct_solver`
    builds without the system-routed tools."""
    recommended_params = {"leiden_resolution": 0.5, "n_neighbors": 15}
    task_spec = None

    def __init__(self, picks, toolbox):
        from shape.agent.verification.stage_map import SYSTEM_ROUTED_TOOLS

        self.picks = list(picks)
        self.toolbox = [t for t in toolbox if t not in SYSTEM_ROUTED_TOOLS]

    def analyze_query(self, **kwargs):
        return "cell-state comparison"

    def verificate_memory(self, **kwargs):
        return SimpleNamespace(stop_signal=not self.picks, analysis="")

    def generate_next_step(self, **kwargs):
        pick = self.picks.pop(0)
        tool, params = pick if isinstance(pick, tuple) else (pick, None)
        if params:
            self.recommended_params = params
        from shape.utils.agent_io import normalize_tool_name

        return SimpleNamespace(tool_name=normalize_tool_name(tool, self.toolbox), sub_goal="analyze the cell states",
                               context="", justification="")

    def generate_final_output(self, **kwargs):
        return "done"


class _BatchAgent:
    """The batch loop's reasoner (a stub): the admission keeps the rule's outcome; round 1 runs the tool at its
    defaults, round 2 stops; the final choice is the first admissible call, else none (keep the input)."""

    def generate(self, content, response_format=None):
        import ast
        import re

        from shape.agent.verification.verifiers.batch_correction_unified import (
            AdmissionVerdict, FinalChoice, WaveMove, WavePlan)

        prompt = content[0]
        if response_format is AdmissionVerdict:
            rule = re.search(r"Deterministic rule outcome: (\w+)\.", prompt).group(1)
            return {"content": AdmissionVerdict(decision=rule, reason_code="rule", reasoning="r")}
        if response_format is WavePlan:
            first = '{"call": "default"' not in prompt
            return {"content": WavePlan(analysis="a", moves=[WaveMove(parent="input", changes=[])] if first else [])}
        if response_format is FinalChoice:
            admissible = ast.literal_eval(re.search(r"Admissible calls \(.*?\): (\[[^\]]*\])", prompt,
                                                    re.DOTALL).group(1))
            return {"content": FinalChoice(call=admissible[0] if admissible else None, reasoning="r")}
        raise RuntimeError(f"no answer for {response_format}")


def _solve(tmp_path, monkeypatch, *, plates=True, groups=("ctrl", "drug"), offset=1.5, full=True,
           picks=("Cell_State_Analyzer_Tool",), batch_override=None, area_signal=False, group_signal=3.0,
           batch_enabled=True, drift=False, no_cells=(), common=0.0):
    import shape.solver as solver_module
    from shape.agent.executor import Executor
    from shape.agent.memory import Memory
    from shape.agent.verification import metadata_profile
    from shape.agent.verifier import Verdict, Verifier
    from shape.tools.batch_corrector import quant

    monkeypatch.setattr(metadata_profile, "batch_metrics_problem",
                        lambda: None if full else "ModuleNotFoundError: No module named 'jax'")
    monkeypatch.setattr(quant, "composite_all", _cheap_composite)
    specs = _specs(tmp_path, plates, groups)
    calls = []
    executor = Executor("unused-model", query_cache_dir=str(tmp_path / "cache"),
                        initializer=SimpleNamespace(_tool_classes_cache={
                            "Cell_State_Analyzer_Tool": _fake_analyzer(specs, offset, calls, batch_override, area_signal,
                                                                         group_signal, drift, no_cells, common)}))
    llm_path = []
    monkeypatch.setattr(executor, "generate_tool_command", lambda **kw: llm_path.append(kw) or 1 / 0)
    tools = ["Cell_State_Analyzer_Tool"] + (["Batch_Corrector_Tool"] if batch_enabled else [])
    solver = {"planner": _Planner(picks, tools), "executor": executor, "memory": Memory(), "initializer": None,
              "available_tools": tools}
    monkeypatch.setattr(solver_module, "construct_solver", lambda **kw: solver)
    # the analyzer's linear loop (RepresentationVerifier): the batch step's routing reads its verdicts (e.g. a
    # batch-dominated concede); the analyzer's agentic loop on the solve() path is tested in test_solve_orchestration
    from shape.agent.verification.stage_map import stage_by_name
    monkeypatch.delitem(stage_by_name("representation").verifier_by_tool, "Cell_State_Analyzer_Tool")
    # ... without its vision tie-breaker (the run has no engine; with `uses_llm` it would raise)
    from shape.agent.verification.verifiers.representation import RepresentationVerifier
    monkeypatch.setattr(RepresentationVerifier, "uses_llm", False)
    # the batch loop's reasoner (the run has no LLM verifier engine): a stub, there is no deterministic fallback
    from shape.agent.verification import batch_stage
    monkeypatch.setattr(batch_stage, "_llm_engine", lambda executor: _BatchAgent())
    # the plan gate (the analyzer needs the cropper first) is not under test here
    monkeypatch.setattr(solver_module, "_verify_plan",
                        lambda next_step, memory, task_spec=None: (Verifier(), Verdict(scope="plan", status="pass")))
    out = solver_module.solve("Compare the cell states of ctrl and drug.", images=specs, max_steps=len(picks),
                              query_cache_dir=str(tmp_path / "cache"), enable_llm_verifier=False)
    return out, calls, llm_path


def _ndarrays(value, path="root"):
    if isinstance(value, np.ndarray):
        return [path]
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in _ndarrays(v, f"{path}.{k}")]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in _ndarrays(v, f"{path}[{i}]")]
    return []


def _actions(out):
    return [(a["tool_name"], bool((a.get("result") or {}).get("batch_corrected"))) for a in out["steps"]]


def _batch_action(out):
    return next(a for a in out["steps"] if a["tool_name"] == "Batch_Corrector_Tool")


def test_a_two_plate_set_ends_corrected_with_reclustered_states(tmp_path, monkeypatch, harmony):
    import anndata as ad
    from sklearn.metrics import silhouette_score

    harmony()
    out, calls, llm_path = _solve(tmp_path, monkeypatch)
    assert out["metadata_profile"]["tier"] == "full" and len(calls) == 1 and not llm_path
    assert out["metadata_profile"]["batch_metrics"] == "usable"         # checked when the batch step first ran
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Cell_State_Analyzer_Tool", True)]
    batch = _batch_action(out)["result"]
    assert (batch["status"], batch["decision"], batch["tier"]) == ("CORRECTED_HARMONY", "RUN", "full")
    assert batch["selected_call"] == "default"  # the scripted reasoner proposed the tool's defaults
    assert batch["harmony"]["applied"] is True
    trials = os.path.join(str(tmp_path / "cache"), "batch_correction", "trials")
    assert os.path.dirname(os.path.dirname(batch["corrected_matrix_path"])) == trials      # trials/<embedding>/

    corrected_path = batch["corrected_adata_path"]
    assert os.path.basename(corrected_path) == "cell_state_analyzed_harmony.h5ad"
    corrected = ad.read_h5ad(corrected_path)
    uncorrected = ad.read_h5ad(batch["uncorrected_adata_path"])
    assert {"X_dinov3", "X_umap", "X_umap_uncorrected"} <= set(corrected.obsm)
    assert {"leiden_0.5", "leiden_0.5_uncorrected"} <= set(corrected.obs.columns)
    assert corrected.uns["embedding_fingerprint"] == FINGERPRINT
    assert json.loads(corrected.uns["batch_correction"])["decision"] == "RUN"
    np.testing.assert_allclose(corrected.obsm["X_dinov3"], uncorrected.X)
    assert list(corrected.obs["leiden_0.5_uncorrected"]) == list(uncorrected.obs["leiden_0.5"])
    assert "batch_correction" not in uncorrected.uns      # the analyzer's own h5ad is untouched
    plate = corrected.obs["batch"].astype(str)
    assert silhouette_score(corrected.X, plate) < silhouette_score(corrected.obsm["X_dinov3"], plate)

    recluster = out["steps"][2]
    stage = recluster["verification"]["stages"]
    assert [s["stage"] for s in stage] == ["morphological_state"]
    assert (stage[0]["reason"], stage[0]["n_trials"]) == ("assessed_once_resolution_locked", 1)
    assert recluster["result"]["embedding"] == "harmony" and recluster["result"]["resolution_locked"] == 0.5
    assert recluster["params"]["leiden_resolution"] == 0.5

    states = out["cell_states"]
    assert states["delivered"] == "corrected"
    assert states["corrected"]["adata_path"] == corrected_path and states["corrected"]["state_key"] == "leiden_0.5"
    assert states["uncorrected"]["adata_path"] == batch["uncorrected_adata_path"]
    assert states["batch_correction"]["status"] == "CORRECTED_HARMONY"
    on_disk = os.path.join(os.path.dirname(corrected_path), "CELL_STATES.json")
    assert json.load(open(on_disk, encoding="utf-8"))["delivered"] == "corrected"

    assert not _ndarrays([a for a in out["memory"].get_actions(llm_safe=False)])
    assert not _ndarrays(out["verification_summary"])
    trace = json.load(open(tmp_path / "cache" / "vlrr_trace.json", encoding="utf-8"))
    assert len(trace["steps"]) == 3


@pytest.mark.parametrize("groups, full, offset, reason", [
    # batch labels only, with a plate offset the gate detects
    (("default",), True, 0.5, "fewer than 2 non-control groups"),
    (("ctrl", "drug"), False, 1.5, "scib-metrics"),        # replicated groups, no batch-metrics extra: downgraded
])
def test_the_partial_tier_abstains_and_keeps_the_uncorrected_states(tmp_path, monkeypatch, harmony, groups, full,
                                                                    offset, reason):
    from shape.agent.planner import Planner

    harmony()
    out, calls, _ = _solve(tmp_path, monkeypatch, groups=groups, full=full, offset=offset)
    profile = out["metadata_profile"]
    assert profile["tier"] == "partial" and any(reason in r for r in profile["reasons"])
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    action = _batch_action(out)
    batch = action["result"]
    assert (batch["status"], batch["decision"], batch["corrected_adata_path"]) == ("UNCORRECTED_INPUT", "ABSTAIN", None)
    assert batch["gate"]["detected"] is True and "error" not in batch
    stage = action["verification"]["stages"][0]
    assert (stage["kept_input"], stage["no_output"]) == ("separability_unverifiable", False)
    digest = Planner._format_memory(object.__new__(Planner), out["memory"])
    assert "INPUT KEPT unchanged, deliberately (separability_unverifiable)" in digest
    assert "NO OUTPUT produced" not in digest
    assert out["cell_states"]["delivered"] == "uncorrected" and out["cell_states"]["corrected"] is None
    assert out["cell_states"]["batch_correction"]["decision"] == "ABSTAIN"


def test_without_a_plate_offset_the_batch_step_skips(tmp_path, monkeypatch, harmony):
    harmony()
    out, _, _ = _solve(tmp_path, monkeypatch, offset=0.0)
    action = _batch_action(out)
    batch = action["result"]
    assert (batch["status"], batch["decision"], batch["selected_call"]) == ("UNCORRECTED_INPUT", "SKIP", "input")
    stage = action["verification"]["stages"][0]
    assert (stage["kept_input"], stage["no_output"]) == ("no_batch_effect", False)
    assert out["cell_states"]["delivered"] == "uncorrected"


def test_without_batch_labels_there_is_no_batch_step(tmp_path, monkeypatch, harmony):
    harmony()
    out, _, _ = _solve(tmp_path, monkeypatch, plates=False)
    assert out["metadata_profile"]["tier"] == "none"
    assert out["metadata_profile"]["batch_metrics"] is None               # no batch step: never checked
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False)]
    assert out["cell_states"]["delivered"] == "uncorrected"
    assert out["cell_states"]["batch_correction"]["routed"] is False


def test_harmony_failing_on_every_call_is_a_failed_batch_step_and_the_run_continues(tmp_path, monkeypatch,
                                                                                     harmony):
    harmony(raises=True)
    # the second pick is the batch tool itself (the planner never sees it): the embedding was already handled
    out, calls, llm_path = _solve(tmp_path, monkeypatch, picks=("Cell_State_Analyzer_Tool", "Batch_Corrector_Tool"))
    assert len(calls) == 1 and not llm_path and out["stop_reason"] == "max_steps_reached"
    batch = _batch_action(out)
    assert batch["result"]["status"] == "failed" and "no_call_finished" in batch["result"]["error"]
    assert batch["result"]["failed_calls"] and batch["verification"]["stages"][0]["no_output"] is True
    assert batch["verification"]["stages"][0]["kept_input"] is None
    last = out["steps"][-1]
    assert (last["tool_name"], last["result"]["status"]) == ("Batch_Corrector_Tool", "not_routed")
    assert out["cell_states"]["delivered"] == "uncorrected"
    assert out["cell_states"]["batch_correction"]["status"] == "failed"


def test_a_batch_label_mismatch_is_a_failed_batch_step_that_stops_the_run(tmp_path, monkeypatch, harmony):
    harmony()
    out, calls, llm_path = _solve(tmp_path, monkeypatch,
                                  batch_override=lambda spec, batch: "P1")      # every cell on plate P1
    assert len(calls) == 1 and not llm_path
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    batch = _batch_action(out)["result"]
    assert batch["status"] == "failed" and "does not match the metadata profile" in batch["error"]
    assert out["stop_reason"] == "hard_error"                          # a failed batch step is an error
    assert out["cell_states"]["delivered"] == "uncorrected"


def test_a_corrupt_h5ad_is_a_failed_batch_step_that_stops_the_run(tmp_path, monkeypatch, harmony):
    from shape.agent.verification import integration

    harmony()
    corrupt = tmp_path / "corrupt.h5ad"
    corrupt.write_bytes(b"not an h5ad")
    monkeypatch.setattr(integration, "_latest_analyzer_h5ad", lambda memory: str(corrupt))
    out, calls, llm_path = _solve(tmp_path, monkeypatch)
    assert len(calls) == 1 and not llm_path
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    batch = _batch_action(out)["result"]
    assert batch["status"] == "failed" and batch["uncorrected_adata_path"] == str(corrupt)
    assert out["stop_reason"] == "hard_error"                          # a failed batch step is an error
    assert out["cell_states"]["delivered"] == "uncorrected"


def test_a_batch_dominated_concede_runs_the_batch_step(tmp_path, monkeypatch, harmony):
    harmony()
    # one declared group and a strong plate offset: the morphology stage concedes upstream as batch-dominated (the
    # case Harmony exists for), so the batch step runs; with one group the tier is partial (biology preservation
    # cannot be verified), so the frozen admission ABSTAINs
    out, calls, _ = _solve(tmp_path, monkeypatch, groups=("a",), offset=3.0)
    record = out["steps"][0]["verification"]["per_image"][0]
    stage = record["stages"][-1]
    assert (stage["stage"], stage["status"], stage["fault"]) == ("morphological_state", "conceded", "upstream")
    assert record["upstream_target"] == "representation"
    assert out["metadata_profile"]["tier"] == "partial"
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    assert (_batch_action(out)["result"]["status"], _batch_action(out)["result"]["decision"]) == (
        "UNCORRECTED_INPUT", "ABSTAIN")
    assert out["cell_states"]["delivered"] == "uncorrected"


def test_a_batch_dominated_concede_runs_the_batch_step_and_records_its_outcome(tmp_path, monkeypatch, harmony):
    harmony(unchanged=True)
    # two replicated groups with a weak signal under a strong plate offset, centred on the origin: batch-dominated,
    # so the batch step runs; but every soft cluster of Harmony (cosine space, sigma 0.1) holds one plate, so
    # harmonypy 2.0.0 returns the input unchanged, no call beats the input, and the step keeps it, recorded. The
    # tool's input processing (input_scale / input_pca) is held to the identity too: with the plateau counted in
    # rounds the search reaches those levers, and a changed input would no longer be "no call beats the input"
    from shape.tools.batch_corrector import tool as batch_tool

    monkeypatch.setattr(batch_tool, "preprocess_input", lambda features, params: (features, {}))
    out, calls, _ = _solve(tmp_path, monkeypatch, offset=3.0, group_signal=0.8)
    record = out["steps"][0]["verification"]["per_image"][0]
    assert (record["status"], record["upstream_target"]) == ("conceded", "representation")
    assert out["metadata_profile"]["tier"] == "full"
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    action = _batch_action(out)
    batch = action["result"]
    assert (batch["status"], batch["decision"], batch["resolution"]["reason"]) == (
        "UNCORRECTED_INPUT", "RUN", "no_admissible_call")
    stage = action["verification"]["stages"][0]
    assert (stage["kept_input"], stage["no_output"], stage["details"]["admissible"]) == (
        "no_admissible_call", False, [])
    inp, calls_made = stage["trials"][0], stage["trials"][1:]
    assert calls_made and all(t["result"]["status"] == "CORRECTED_HARMONY" for t in calls_made)
    assert all(abs(t["metrics"]["overall"] - inp["metrics"]["overall"]) < 1e-6 for t in calls_made)
    states = out["cell_states"]
    assert states["delivered"] == "uncorrected" and states["corrected"] is None
    assert (states["batch_correction"]["status"], states["batch_correction"]["reason"]) == (
        "UNCORRECTED_INPUT", "no_admissible_call")


def test_a_batch_dominated_concede_in_the_full_tier_is_corrected(tmp_path, monkeypatch, harmony):
    harmony()
    # the same design with a mean shared by every cell (as real embeddings have): the Euclidean geometry, so the
    # batch-dominated concede, is the same, while Harmony's cosine-space clusters now span both plates
    out, calls, _ = _solve(tmp_path, monkeypatch, offset=3.0, group_signal=0.8, common=3.0)
    record = out["steps"][0]["verification"]["per_image"][0]
    assert (record["status"], record["upstream_target"]) == ("conceded", "representation")
    assert out["metadata_profile"]["tier"] == "full"
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Cell_State_Analyzer_Tool", True)]
    batch = _batch_action(out)["result"]
    assert (batch["status"], batch["decision"]) == ("CORRECTED_HARMONY", "RUN")
    stage = _batch_action(out)["verification"]["stages"][0]
    chosen = stage["trials"][stage["details"]["selected"]]["metrics"]
    assert chosen["overall"] > stage["trials"][0]["metrics"]["overall"] + 0.002          # beyond the noise
    assert out["cell_states"]["delivered"] == "corrected"


def test_another_upstream_concede_waits_for_the_rerun_and_so_does_a_planner_pick(tmp_path, monkeypatch, harmony):
    harmony()
    # the first PC tracks the cell area: the representation stage concedes upstream to the crop, so the planner
    # re-runs upstream and this embedding is about to be replaced; a planner pick of the batch tool waits too
    out, calls, _ = _solve(tmp_path, monkeypatch, area_signal=True,
                           picks=("Cell_State_Analyzer_Tool", "Batch_Corrector_Tool"))
    record = out["steps"][0]["verification"]["per_image"][0]
    assert (record["stages"][-1]["stage"], record["stages"][-1]["fault"]) == ("representation", "upstream")
    assert record["upstream_target"] == "crop"
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Batch_Corrector_Tool", False)] and len(calls) == 1
    wait = "the analyzer conceded upstream (representation -> crop)"
    assert all(a["result"]["status"] == "not_routed" and a["result"]["reason"].startswith(wait)
               for a in out["steps"][1:])
    assert out["steps"][2]["sub_goal"] == "analyze the cell states"          # the planner's pick
    # the batch loop never ran, so the batch-metrics extra was never checked (no jax import)
    assert (out["metadata_profile"]["tier"], out["metadata_profile"]["batch_metrics"]) == ("full", None)
    states = out["cell_states"]
    assert states["delivered"] == "uncorrected" and states["batch_correction"]["routed"]
    assert states["batch_correction"]["status"] == "not_routed"
    assert states["batch_correction"]["reason"].startswith(wait)


def test_a_planner_pick_of_the_batch_tool_does_not_end_the_run(tmp_path, monkeypatch, harmony):
    harmony()
    # the pick comes back from the planner's name check as "No matched tool given: Batch_Corrector_Tool"
    out, calls, llm_path = _solve(tmp_path, monkeypatch, picks=(
        "Cell_State_Analyzer_Tool", "Batch_Corrector_Tool", ("Cell_State_Analyzer_Tool", {
            "leiden_resolution": 0.8, "n_neighbors": 15})))
    assert out["stop_reason"] == "max_steps_reached" and len(calls) == 2 and not llm_path
    pick = out["steps"][3]
    assert (pick["tool_name"], pick["result"]["status"]) == ("Batch_Corrector_Tool", "not_routed")
    assert pick["result"]["reason"].startswith("this analyzer step was already handled")
    assert out["steps"][4]["tool_name"] == "Cell_State_Analyzer_Tool"          # the next planned step ran
    assert out["cell_states"]["delivered"] == "corrected"


def test_a_rerun_at_the_same_resolution_keeps_the_corrected_states(tmp_path, monkeypatch, harmony):
    harmony()
    out, calls, _ = _solve(tmp_path, monkeypatch, picks=("Cell_State_Analyzer_Tool", "Cell_State_Analyzer_Tool"))
    assert len(calls) == 2
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Cell_State_Analyzer_Tool", True)] * 2
    first, second = out["steps"][1]["result"], out["steps"][4]["result"]
    assert second["status"] == first["status"] == "CORRECTED_HARMONY" and second["reused"] is True
    assert second["uncorrected_adata_path"] == first["uncorrected_adata_path"]
    states = out["cell_states"]
    assert states["delivered"] == "corrected" and states["batch_correction"]["status"] == "CORRECTED_HARMONY"
    assert states["corrected"]["state_key"] == states["uncorrected"]["state_key"]


def test_a_disabled_batch_tool_is_not_routed(tmp_path, monkeypatch, harmony):
    harmony()
    out, _, _ = _solve(tmp_path, monkeypatch, batch_enabled=False)
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False)]
    batch = _batch_action(out)["result"]
    assert (batch["status"], batch["reason"]) == (
        "not_routed", "Batch_Corrector_Tool is not among this run's enabled tools")
    assert out["cell_states"]["delivered"] == "uncorrected"


def test_the_corrected_states_are_assessed_once(tmp_path, monkeypatch):
    """The resolution is locked, so the morphology verifier assesses the corrected re-cluster once: one
    re-clustering, the verdict as it is (here a retune it asks for), no retune."""
    import anndata as ad
    import pandas as pd

    from shape.agent.verification import batch_stage
    from shape.agent.verification.verifiers.morphological_state import MorphologicalStateVerifier
    from shape.utils.schemas import VLRRResult

    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.normal(size=(60, 4)).astype(np.float32),
                       obs=pd.DataFrame({"group": ["a", "b"] * 30, "batch": ["P1"] * 30 + ["P2"] * 30,
                                         "image_name": [f"i{k % 4}" for k in range(60)]}))
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.uns["embedding_fingerprint"] = FINGERPRINT
    path = str(tmp_path / "cell_state_analyzed_harmony.h5ad")
    adata.write(path)
    reclusters = []
    real = Cell_State_Analyzer_Tool.recluster
    monkeypatch.setattr(Cell_State_Analyzer_Tool, "recluster",
                        lambda self, *a, **k: reclusters.append(a) or real(self, *a, **k))
    retune = VLRRResult("WARNING", 0.4, {}, action="retune_here", param_delta={"leiden_resolution": "down"},
                        notes="Resolution instability/fragmentation -> lower leiden_resolution.")
    monkeypatch.setattr(MorphologicalStateVerifier, "adjudicate", lambda self, *a: retune)
    res, resolution = batch_stage.recluster_corrected(path, FINGERPRINT, {"leiden_resolution": 0.5,
                                                                          "n_neighbors": 10})
    assert len(reclusters) == 1 and resolution == 0.5 and len(res.trials) == 1
    # the linear loop ends a stage on a retune verdict with no retune left the same way: conceded, ceiling
    assert (res.status, res.reason, res.fault) == ("conceded", "assessed_once_resolution_locked", "ceiling")
    assert res.trials[0][1].action == "retune_here" and res.trials[0][1].param_delta == {"leiden_resolution": "down"}
    assert res.verdict is res.trials[0][1] and res.verdict.decision == "WARNING"
    assert res.params == {"leiden_resolution": 0.5, "n_neighbors": 10}


def test_a_rerun_at_a_new_resolution_reclusters_the_corrected_states(tmp_path, monkeypatch, harmony):
    import anndata as ad

    harmony()
    out, calls, _ = _solve(tmp_path, monkeypatch, picks=(
        "Cell_State_Analyzer_Tool", ("Cell_State_Analyzer_Tool", {"leiden_resolution": 0.8, "n_neighbors": 15})))
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Cell_State_Analyzer_Tool", True)] * 2
    first, second = out["steps"][1]["result"], out["steps"][4]["result"]
    assert first["status"] == second["status"] == "CORRECTED_HARMONY"
    assert second["reused"] is True and "reused" not in first          # the batch loop ran once
    assert second["selected_call"] == first["selected_call"] and out["steps"][4]["verification"] is None
    resolution = out["steps"][3]["result"]["per_image"][0]["cluster_resolution"]
    assert resolution != out["steps"][0]["result"]["per_image"][0]["cluster_resolution"]
    assert out["steps"][5]["result"]["resolution_locked"] == resolution
    corrected = ad.read_h5ad(second["corrected_adata_path"])
    assert {f"leiden_{resolution}", f"leiden_{resolution}_uncorrected"} <= set(corrected.obs.columns)
    states = out["cell_states"]
    assert states["delivered"] == "corrected" and states["corrected"]["state_key"] == f"leiden_{resolution}"
    assert states["uncorrected"]["state_key"] == f"leiden_{resolution}"


def test_a_rerun_whose_embedding_differs_is_decided_again(tmp_path, monkeypatch, harmony):
    harmony()
    # the same fingerprint and cells, but the second training's X differs: a new embedding, so the batch loop runs
    # again instead of reusing the first embedding's correction
    out, calls, _ = _solve(tmp_path, monkeypatch, drift=True,
                           picks=("Cell_State_Analyzer_Tool", "Cell_State_Analyzer_Tool"))
    assert len(calls) == 2
    assert _actions(out) == [("Cell_State_Analyzer_Tool", False), ("Batch_Corrector_Tool", False),
                             ("Cell_State_Analyzer_Tool", True)] * 2
    first, second = out["steps"][1], out["steps"][4]
    assert "reused" not in second["result"] and second["verification"] is not None      # the loop ran again
    assert first["result"]["corrected_matrix_path"] != second["result"]["corrected_matrix_path"]
    assert (os.path.dirname(first["result"]["corrected_matrix_path"])
            != os.path.dirname(second["result"]["corrected_matrix_path"]))               # one directory each


def test_the_same_morphology_verdict_reads_the_same_on_both_state_sets(tmp_path, monkeypatch, harmony):
    from shape.agent.verification.verifiers.morphological_state import MorphologicalStateVerifier
    from shape.utils.schemas import VLRRResult

    harmony()
    notes = "Resolution instability/fragmentation -> lower leiden_resolution."
    monkeypatch.setattr(MorphologicalStateVerifier, "adjudicate", lambda self, *a: VLRRResult(
        "WARNING", 0.4, {}, action="retune_here", param_delta={"leiden_resolution": "down"}, notes=notes))
    out, _, _ = _solve(tmp_path, monkeypatch)
    states = out["cell_states"]
    assert states["delivered"] == "corrected"
    fields = ("qc_status", "qc_decision", "qc_action", "qc_notes")
    assert ({k: states["uncorrected"][k] for k in fields} == {k: states["corrected"][k] for k in fields}
            == {"qc_status": "conceded", "qc_decision": "WARNING", "qc_action": "retune_here", "qc_notes": notes})
    assert states["corrected"]["qc_reason"] == "assessed_once_resolution_locked"
    on_disk = json.load(open(os.path.join(os.path.dirname(states["uncorrected"]["adata_path"]), "CELL_STATES.json"),
                             encoding="utf-8"))
    assert on_disk["corrected"]["qc_status"] == on_disk["uncorrected"]["qc_status"] == "conceded"


def test_a_full_design_whose_cells_lost_a_group_abstains_with_the_reason(tmp_path, monkeypatch, harmony):
    harmony()
    # the specs are full (ctrl and drug on both plates), but every drug image yields no cell
    out, calls, _ = _solve(tmp_path, monkeypatch, no_cells=("drug",))
    assert out["metadata_profile"]["tier"] == "full"                  # the specs' tier
    batch = _batch_action(out)["result"]
    reason = ("after segmentation and cropping only 1 non-control groups have cells in 2 or more batch groups "
              "(at least 2 needed): biology preservation cannot be verified")
    assert (batch["tier"], batch["tier_reason"]) == ("partial", reason)
    assert (batch["status"], batch["decision"]) == ("UNCORRECTED_INPUT", "ABSTAIN")
    states = out["cell_states"]
    assert states["delivered"] == "uncorrected"
    assert (states["batch_correction"]["tier"], states["batch_correction"]["tier_reason"]) == ("partial", reason)
