from types import SimpleNamespace

import numpy as np
import pandas as pd

from shape.agent.verification import integration
from shape.agent.verifier import Verifier
from shape.agent.verification.initial_params import _source_reliable
from shape.agent.verification.verifiers.morphological_state import _replicate_consistency
from shape.agent.verification.verifiers.representation import trial_context
from shape.tools.cell_state_analyzer.tool import (
    embedding_fingerprint,
    resolve_plate_nuisance,
    resolve_technical_metadata,
)
from shape.solver import _normalize_image_specs


def test_morphology_rerun_forces_embedding_reuse(monkeypatch):
    calls = []

    class FakeExecutor:
        enable_llm_verifier = False

        def run_managed_tool(self, tool_name, base_kwargs, params):
            calls.append(dict(params))
            return {"adata_path": "cached.h5ad", "cluster_resolution": 0.5}

    def fake_resolve(self, spec, runner, initial_params, initial_result, **kwargs):
        params = dict(initial_params)
        result = initial_result
        if spec.stage == "morphological_state":
            params["leiden_resolution"] = 0.4
            result = runner(params)
        return SimpleNamespace(result=result, params=params, status="satisfied")

    monkeypatch.setattr(Verifier, "resolve_managed_stage", fake_resolve)
    integration.verify_managed_tool(
        FakeExecutor(), "Cell_State_Analyzer_Tool", {},
        {"learning_rate": 3e-5, "leiden_resolution": 0.5, "n_neighbors": 15},
    )

    assert len(calls) == 2
    assert "reuse_embedding" not in calls[0]
    assert calls[1]["reuse_embedding"] is True


def test_image_spec_technical_metadata_reaches_cropper_kwargs():
    spec = _normalize_image_specs([{
        "path": "image.tif", "image_id": "field-1", "group": "treated",
        "plate_id": "P1", "well_id": "A01", "site_id": "S1",
        "ignored": "must-not-cross-the-boundary",
    }])[0]
    assert "ignored" not in spec
    kwargs = integration.build_base_kwargs(
        "Single_Cell_Cropper_Tool",
        spec,
        memory=None,
    )
    assert kwargs["technical_metadata"] == {
        "plate_id": "P1", "well_id": "A01", "site_id": "S1",
    }


def test_technical_aliases_enable_plate_route_and_reliability_report():
    metadata = [
        {"plate_id": "P1", "well_id": "A01", "site_id": "S1"},
        {"plate_id": "P1", "well_id": "A01", "site_id": "S2"},
        {"plate_id": "P2", "well_id": "B01", "site_id": "S1"},
        {"plate_id": "P2", "well_id": "B01", "site_id": "S2"},
    ]
    route = resolve_plate_nuisance(metadata, 1.0, 0.1)
    assert route["eligible"] is True
    technical_groups, batches, report = resolve_technical_metadata(
        metadata, ["i1", "i2", "i3", "i4"])
    # plate-qualified well and site (contract revision 2, 3.1: technical_group identities carry the plate)
    assert technical_groups == ["P1/A01:P1/S1", "P1/A01:P1/S2", "P2/B01:P2/S1", "P2/B01:P2/S2"]
    assert batches == ["P1", "P1", "P2", "P2"]
    assert report["scale_and_balance"]["metadata_availability"]["plate_id"] == "available"


def test_representation_context_receives_production_reliability_report():
    report = {
        "scale_and_balance": {
            "metadata_availability": {"plate_id": "available"},
            "median_crops_per_source": 2.0,
        }
    }
    context = trial_context(
        "not-loaded-without-base-kwargs.h5ad", None,
        trial_kwargs={"field_aware": False}, report=report)
    assert context["report"] == report
    assert context["field_aware"] is False
    assert _source_reliable(context["report"]) is True


def test_replicate_consistency_is_computed_within_biological_group():
    # Each group has perfectly agreeing replicate images, while the two groups have a
    # deliberately different cluster composition. Pooling all images would score this low.
    obs = pd.DataFrame({
        "image_name": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "group": ["control"] * 4 + ["treated"] * 4,
    })
    adata = SimpleNamespace(obs=obs)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert _replicate_consistency(labels, adata) == 1.0


def test_embedding_fingerprint_changes_when_same_path_content_changes(tmp_path):
    crop = tmp_path / "cell.png"
    crop.write_bytes(b"first-image")
    first = embedding_fingerprint({"seed": 0}, [crop])
    crop.write_bytes(b"second-image")
    second = embedding_fingerprint({"seed": 0}, [crop])
    assert first != second
