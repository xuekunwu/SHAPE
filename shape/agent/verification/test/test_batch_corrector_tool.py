from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shape.tools.batch_corrector.tool import (
    DIAGNOSTIC_KEYS,
    TOOL_VERSION,
    Batch_Corrector_Tool,
    SKIPPED_NO_BATCH_EFFECT,
    resolve_library_default_params,
)


def _data():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 8))
    metadata = pd.DataFrame({"batch": np.repeat(["a", "b", "c"], 20)})
    return features, metadata


def test_tool_skips_when_detector_finds_no_effect():
    features, metadata = _data()
    calls = []

    def detector(*args, **kwargs):
        return type("Report", (), {
            "batch_key": "batch", "n_samples": 60, "n_batches": 3,
            "chance_accuracy": 1 / 3, "batch_predictability": 1 / 3,
            "effect_margin": 0.0, "detected": False,
            "reason": "batch_predictability_not_above_margin",
        })()

    def runner(*args, **kwargs):
        calls.append(True)
        return features + 1

    result = Batch_Corrector_Tool().execute(features, metadata, detector=detector,
                                            harmony_runner=runner)
    assert result["status"] == SKIPPED_NO_BATCH_EFFECT
    np.testing.assert_array_equal(result["corrected_cls_output"], features)
    assert result["provenance"]["detect_margin"] == 0.10
    assert result["provenance"]["row_identity_validated"] is False
    assert calls == []


def test_tool_freezes_margin_and_validates_row_identity():
    features, metadata = _data()
    with pytest.raises(ValueError, match="frozen"):
        Batch_Corrector_Tool().execute(features, metadata, detect_margin=-0.1)
    with pytest.raises(ValueError, match="frozen"):
        Batch_Corrector_Tool().execute(features, metadata, detect_margin=float("nan"))
    with pytest.raises(ValueError, match="same row count"):
        Batch_Corrector_Tool().execute(features, metadata, row_ids=["only-one"])
    with pytest.raises(ValueError, match="unique"):
        Batch_Corrector_Tool().execute(features, metadata, row_ids=["same"] * len(features))


def test_runtime_default_is_sample_aware_and_rejects_overclustered_override():
    assert resolve_library_default_params(10)["nclust"] == 1
    assert resolve_library_default_params(219)["nclust"] == 7
    assert resolve_library_default_params(1094)["nclust"] == 36
    features, metadata = _data()
    with pytest.raises(ValueError, match="nclust"):
        Batch_Corrector_Tool().execute(
            features, metadata,
            harmony_params={
                "theta": 2.0, "lambda_value": None, "sigma": 0.1,
                "nclust": len(features), "random_state": 0,
            },
        )

def test_tool_calls_harmony_only_after_batch_effect_detection(tmp_path):
    features = np.zeros((60, 8), dtype=float)
    features[:, 0] = np.repeat([0.0, 5.0, 10.0], 20)
    metadata = pd.DataFrame({"batch": np.repeat(["a", "b", "c"], 20)})
    seen = []

    def runner(values, meta, batch_key, params):
        seen.append((batch_key, dict(params)))
        return values + 1.0

    result = Batch_Corrector_Tool().execute(
        features, metadata, row_ids=[f"row-{i}" for i in range(len(features))],
        output_dir=str(tmp_path), run_id="synthetic", harmony_runner=runner,
    )
    assert result["status"] == "CORRECTED_HARMONY"
    assert len(seen) == 1
    assert seen[0][0] == "batch"
    assert seen[0][1] == {
        "theta": 2.0, "lambda_value": None, "sigma": 0.1,
        "nclust": 2, "random_state": 0,
    }
    assert result["provenance"]["row_identity_validated"] is True
    assert result["harmony"]["parameter_source"] == "harmonypy_library_default"
    assert result["provenance"]["harmony_backend"]["name"] == "harmonypy"
    assert result["provenance"]["detector"]["version"] == TOOL_VERSION
    assert (tmp_path / "corrected_cls_output.npy").exists()
    assert (tmp_path / "BATCH_CORRECTOR_RESULT.json").exists()


def _detected(*args, **kwargs):
    return type("Report", (), {
        "batch_key": "batch", "n_samples": 60, "n_batches": 3, "chance_accuracy": 1 / 3,
        "batch_predictability": 0.9, "effect_margin": 0.57, "detected": True,
        "reason": "batch_predictability_above_margin"})()


def test_tool_returns_gt_free_readouts_with_every_call():
    features, metadata = _data()
    result = Batch_Corrector_Tool().execute(features, metadata, batch_key="batch", detector=_detected,
                                            harmony_runner=lambda x, *a: x * 0.5)
    readouts = result["gt_free_metrics"]
    assert set(DIAGNOSTIC_KEYS) <= set(readouts) and readouts["batch_effect_detected"] is True
    assert readouts["displacement"] == pytest.approx(0.5 * np.linalg.norm(features)
                                                     / np.linalg.norm(features - features.mean(axis=0)))
    assert readouts["effective_rank_ratio"] == pytest.approx(1.0)


def test_tool_accepts_optional_solver_and_input_processing_keys():
    features, metadata = _data()
    seen = []
    params = {**resolve_library_default_params(60), "epsilon_harmony": 1e-4, "input_pca": 4,
              "input_scale": "zscore"}
    result = Batch_Corrector_Tool().execute(features, metadata, batch_key="batch", harmony_params=params,
                                            detector=_detected,
                                            harmony_runner=lambda x, meta, key, p: seen.append(p) or x)
    assert seen[0]["epsilon_harmony"] == 1e-4 and seen[0]["max_iter_harmony"] is None
    assert result["corrected_cls_output"].shape == (60, 4)
    assert result["preprocess"]["pca_components"] == 4
    readouts = result["gt_free_metrics"]         # relative to the raw 8-feature input
    assert readouts["displacement"] is None and readouts["within_batch_neighbour_overlap"] is not None


def test_tool_rejects_unknown_runtime_keys():
    features, metadata = _data()
    params = {**resolve_library_default_params(60), "unknown": 1}
    with pytest.raises(ValueError, match="invalid schema"):
        Batch_Corrector_Tool().execute(features, metadata, batch_key="batch", harmony_params=params,
                                       detector=_detected)


def test_tool_reports_harmony_telemetry_and_effective_defaults():
    from shape.agent.verification.stage_map import INDEPENDENT_STAGES
    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG, _telemetry, effective_defaults

    features = np.zeros((60, 8), dtype=float)
    features[:, 0] = np.repeat([0.0, 5.0, 10.0], 20)
    metadata = pd.DataFrame({"batch": np.repeat(["a", "b", "c"], 20)})
    ho = type("Harmony", (), {"objective_harmony": [3.0, 2.0], "R": np.full((4, 60), 0.25)})()

    def runner(values, meta, batch_key, params):          # a runner may return (matrix, telemetry)
        return values - values.mean(axis=0), _telemetry(ho)

    seen = []

    def recording_runner(values, meta, batch_key, params):
        seen.append(dict(params))
        return runner(values, meta, batch_key, params)

    result = Batch_Corrector_Tool().execute(features, metadata, harmony_params=dict(DEFAULT_CONFIG),
                                            harmony_runner=recording_runner)
    assert seen[0]["nclust"] == 2
    assert result["telemetry"]["n_clusters"] == 4 and result["telemetry"]["objective_harmony"] == [3.0, 2.0]
    assert result["telemetry"]["cluster_mass_min_max"] == [0.25, 0.25] and result["seconds"] >= 0
    effective = effective_defaults(result)
    assert effective["n_rows"] == 60 and effective["input_pca"] == 8 and effective["nclust"] == 4
    assert effective["lambda_value"] == "dynamic"          # harmonypy 2.0: lamb=None is a per-cluster estimate
    plain = Batch_Corrector_Tool().execute(features, metadata, harmony_params=dict(DEFAULT_CONFIG),
                                           harmony_runner=lambda values, *a: values * 0.5)
    assert plain["telemetry"] == {}                        # a runner returning only the matrix still works
    assert INDEPENDENT_STAGES[0].default_params == {"harmony_params": DEFAULT_CONFIG}


def test_real_harmonypy_handles_ten_profile_default_config():
    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG

    pytest.importorskip("harmonypy")
    rng = np.random.default_rng(7)
    metadata = pd.DataFrame({"batch": ["a"] * 5 + ["b"] * 5})
    features = rng.normal(size=(10, 6))
    features[5:, 0] += 5.0
    result = Batch_Corrector_Tool().execute(
        features,
        metadata,
        harmony_params=dict(DEFAULT_CONFIG),
        detector=_detected,
    )

    assert result["status"] == "CORRECTED_HARMONY"
    assert result["harmony"]["parameters"]["nclust"] == 1
    assert np.asarray(result["corrected_cls_output"]).shape == features.shape
