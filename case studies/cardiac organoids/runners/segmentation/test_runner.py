"""Contract tests with synthetic images; no weights download, gateway or GPU is used."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common
import arms
import audit
import run_organoid_segmentation as runner


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


@pytest.fixture
def dataset(tmp_path):
    images, gt = tmp_path / "images", tmp_path / "gt"
    images.mkdir()
    gt.mkdir()
    mask = np.zeros((32, 48), dtype=np.uint16)
    mask[5:20, 9:35] = 400
    for stem in ("a", "b"):
        tifffile.imwrite(images / f"{stem}.tif", mask * 100)
        tifffile.imwrite(gt / f"{stem}.tif", mask)
    return images, gt, mask


def manifest_for(tmp_path, dataset):
    images, gt, _ = dataset
    path = tmp_path / "results" / "pairs.json"
    common.prepare(SimpleNamespace(data_root=str(images), image_root=None, label_root=str(gt),
                                   label_suffix="_seg", sample_size=None, seed=112, manifest=str(path)))
    return path


def test_user_defaults():
    assert common.DATA_ROOT.endswith("/Inhouse_Organoid_Seg")
    assert common.LABEL_ROOT.endswith("/Inhouse_Organoid_Seg_GT")
    assert common.WEIGHTS.endswith("/CO_cpsam_finetune/shape_seg_co")
    assert common.ARMS == ("cellpose_sam", "shape_seg_co", "microsam", "stardist", "llm", "react", "shape")


def test_pairing_and_input_tamper(tmp_path, dataset):
    path = manifest_for(tmp_path, dataset)
    manifest = common.read_manifest(path)
    assert len(manifest["pairs"]) == 2
    tifffile.imwrite(dataset[0] / "a.tif", np.zeros((32, 48), np.uint16))
    with pytest.raises(ValueError, match="Input changed"):
        common.read_manifest(path)


def test_missing_ambiguous_and_rgb_gt(dataset):
    images, gt, mask = dataset
    extra = gt / "a_seg.npy"
    np.save(extra, {"masks": mask})
    with pytest.raises(ValueError, match="exactly one GT"):
        common.discover(images, label_root=gt)
    extra.unlink()
    (gt / "a.tif").unlink()
    with pytest.raises(ValueError, match="exactly one GT"):
        common.discover(images, label_root=gt)
    tifffile.imwrite(gt / "a.tif", np.stack([mask] * 3, axis=-1), photometric="rgb")
    with pytest.raises(ValueError, match="2D integer"):
        common.discover(images, label_root=gt)


def test_flat_cellpose_seg_dict_and_readonly_inputs(tmp_path):
    tifffile.imwrite(tmp_path / "raw.tif", np.ones((8, 9), np.uint16))
    np.save(tmp_path / "raw_seg.npy", {"masks": np.ones((8, 9), np.int32)})
    pairs = common.discover(tmp_path)
    assert len(pairs) == 1
    with pytest.raises(ValueError, match="outside source"):
        common.protect_output(tmp_path / "outputs", pairs)


def test_failure_inclusive_scoring(tmp_path, dataset):
    pairs = common.discover(dataset[0], label_root=dataset[1])
    good = runner.evaluate_pair(pairs[0], dataset[2], tmp_path)
    failed = runner.evaluate_pair(pairs[1], None, tmp_path, error="inference failed")
    assert good["ap50"] == 1
    assert failed["ap50"] == 0 and not failed["delivered"]
    invalid = runner.evaluate_pair(pairs[1], np.zeros((4, 4)), tmp_path)
    assert invalid["ap50"] == 0 and "Invalid prediction" in invalid["error"]


def test_llm_polygon_scaling_empty_and_invalid(tmp_path, monkeypatch):
    image = tmp_path / "large.tif"
    tifffile.imwrite(image, np.zeros((1025, 2049), np.uint16))

    class Engine:
        polygons = [[[10, 10], [100, 10], [100, 100], [10, 100]]]

        def generate(self, content, response_format, **kwargs):
            assert "whole cardiac organoid" in content[0]
            return {"content": response_format(organoids=[{"polygon": p} for p in self.polygons])}

    engine = Engine()
    arm = arms.LLMPolygons(engine)
    result = arm(image, tmp_path)
    assert result["mask"].shape == (1025, 2049) and result["tool_calls"] == 0
    assert result["mask"][150, 150] == 1
    engine.polygons = []
    assert not arm(image, tmp_path)["mask"].any()
    engine.polygons = [[[0, 0], [float("nan"), 1], [1, 2]]]
    with pytest.raises(ValueError, match="Invalid organoid"):
        arm(image, tmp_path)


def test_react_fixed_weights_and_single_native_attempt(tmp_path):
    seen = []

    class Tool:
        def execute(self, **kwargs):
            seen.append(kwargs)
            return {"error": "zero detections"}

    verifier = SimpleNamespace(levers=["diameter"], concurrency=lambda: 1, admission_call=True)
    adapter = arms.OrganoidAdapter({"a": tmp_path / "a.tif"}, {"a": tmp_path}, tool=Tool(), verifier=verifier)
    adapter.call("a", {"diameter": 120}, "c2")
    assert seen[0]["adaptive_rescue"] is False and seen[0]["attempt_id"] == "c2"
    assert adapter.round_cap() == 1 and adapter.default_counts_as_round()
    assert adapter.call("a", {"model_path": "other"}, "c3")["error"]
    assert len(seen) == 1


def test_native_call_audit_snapshots_and_weights(tmp_path, dataset, monkeypatch):
    from cellpose.models import CellposeModel
    def recursive_eval(self, data, **kwargs):
        return self.eval(data[0], **kwargs) if isinstance(data, list) else dataset[2]
    monkeypatch.setattr(CellposeModel, "eval", recursive_eval)
    weights = tmp_path / "co_model"
    weights.write_bytes(b"synthetic model, never loaded")
    model = object.__new__(CellposeModel)
    model.pretrained_model, model.device = str(weights), "cpu"

    class Tool:
        def execute(self, **kwargs):
            result = model.eval([dataset[2]], diameter=100)
            path = tmp_path / "native.npy"
            np.save(path, result)
            return {"mask_path": str(path)}

    with audit.tool_audit(tmp_path, Tool) as (calls, native):
        Tool().execute(image=str(dataset[0] / "a.tif"), adaptive_rescue=False)
    assert len(calls) == len(native) == 1
    assert calls[0]["native_requests"] == 1
    assert native[0]["weights_sha256"] == common.sha256(weights)
    assert np.array_equal(common.labels(calls[0]["snapshot"]), dataset[2])


def test_llm_audit_pins_high_records_usage_and_redacts(tmp_path, monkeypatch):
    from shape.engine.openai import ChatOpenAI
    from shape.engine.errors import NoEngineError
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "high")
    monkeypatch.setenv("LLM_API_KEY", "fake-secret-for-unit-test")
    monkeypatch.setattr(ChatOpenAI, "generate", lambda *a, **kw: {
        "content": "OK", "usage": {"total_tokens": 12}})
    engine = object.__new__(ChatOpenAI)
    engine.model_string, engine.is_multimodal = common.MODEL, True
    engine._extra_body, engine._sanitize_messages = {}, False
    request = {}
    engine._apply_backend_quirks(request)
    assert request["reasoning_effort"] == "high"
    with audit.llm_audit(tmp_path) as calls:
        engine.generate(["fake-secret-for-unit-test", b"not an actual image"])
    assert calls[0]["tokens"] == 12 and calls[0]["reasoning_effort"] == "high"
    assert "fake-secret-for-unit-test" not in (tmp_path / "LLM_CALLS.jsonl").read_text()
    monkeypatch.setattr(ChatOpenAI, "generate", lambda *a, **kw: {"content": "OK"})
    with audit.llm_audit(tmp_path), pytest.raises(NoEngineError, match="usage"):
        engine.generate("test")


def test_preflight_prohibits_plain_completion_fallback(tmp_path, monkeypatch):
    from shape.engine.openai import ChatOpenAI
    from shape.engine.errors import NoEngineError
    engine = object.__new__(ChatOpenAI)
    def fail(*args, **kwargs):
        raise ValueError("bad structured endpoint response")
    monkeypatch.setattr(ChatOpenAI, "_parse_structured", fail)
    with audit.llm_audit(tmp_path, strict_structured=True), pytest.raises(NoEngineError, match="fallback is disabled"):
        engine._parse_structured({}, None)


def test_real_engine_wrapper_sends_pinned_model_and_high_reasoning(tmp_path, monkeypatch):
    """Exercise the real tenacity-decorated generate/structured paths with a fake HTTP transport."""
    from pydantic import BaseModel
    from shape.engine.openai import ChatOpenAI
    monkeypatch.setenv("SHAPE_REASONING_EFFORT", "high")
    engine = ChatOpenAI(model_string=common.MODEL, is_multimodal=True,
                        api_key="synthetic-not-used", base_url="https://llm-gateway.example.edu/v1")
    sent = []

    class Answer(BaseModel):
        answer: str

    def transport(**kwargs):
        sent.append(kwargs)
        message = SimpleNamespace(parsed=Answer(answer="OK"), content='{"answer":"OK"}', refusal=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)],
                                   usage=SimpleNamespace(total_tokens=15, prompt_tokens=9, completion_tokens=6))
        return SimpleNamespace(parse=lambda: response)

    engine.client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(parse=transport)))))
    png, _ = common.cellseg.display_png(np.zeros((8, 8), np.uint8), 1024)
    with audit.llm_audit(tmp_path, strict_structured=True) as calls:
        reply = engine.generate(["Return OK", png], response_format=Answer)
    assert reply["content"].answer == "OK" and calls[0]["tokens"] == 15
    assert sent[0]["model"] == common.MODEL and sent[0]["reasoning_effort"] == "high"
    assert "max_completion_tokens" in sent[0] and "temperature" not in sent[0]


def test_configure_pins_vrr_policy_and_completion_cap(monkeypatch):
    monkeypatch.setenv("SHAPE_MAX_COMPLETION_TOKENS", "999")          # a stale shell value must not leak in
    settings = common.configure("legacy")
    assert settings["SHAPE_VRR_POLICY"] == "legacy" and "SHAPE_MAX_COMPLETION_TOKENS" not in settings
    assert "SHAPE_MAX_COMPLETION_TOKENS" not in os.environ
    settings = common.configure("four_stage", "balanced_v1", 16000)
    assert settings["SHAPE_VRR_POLICY"] == os.environ["SHAPE_VRR_POLICY"] == "balanced_v1"
    assert settings["SHAPE_MAX_COMPLETION_TOKENS"] == os.environ["SHAPE_MAX_COMPLETION_TOKENS"] == "16000"
    assert settings["SHAPE_SEARCH_STRATEGY_SEGMENTATION"] == "four_stage"
    assert common.configure("legacy")["SHAPE_REASONING_EFFORT"] == "high"          # the default stays high
    settings = common.configure("legacy", reasoning_effort="medium")
    assert os.environ["SHAPE_REASONING_EFFORT"] == os.environ["ORGANOID_PINNED_REASONING_EFFORT"] == "medium"
    with pytest.raises(ValueError, match="unsupported reasoning effort"):
        common.configure("legacy", reasoning_effort="xhigh")
    common.configure("legacy")


def test_organoid_id_arm_keeps_border_organoids_at_original_size(tmp_path, monkeypatch):
    from PIL import Image
    repo = tmp_path / "OrganoID"
    (repo / "Core").mkdir(parents=True)
    (repo / "OptimizedModel").write_bytes(b"pinned tflite")
    (repo / "Core" / "__init__.py").write_text("")
    (repo / "Core" / "Model.py").write_text(
        "import numpy as np\n"
        "def LoadLiteModel(path): return 'model'\n"
        "def PrepareImagesForModel(images, model, verbose=True): return np.zeros((1, 32, 32), np.uint8)\n"
        "def Detect(model, prepared, batch):\n"
        "    b = np.zeros((1, 32, 32), np.float32); b[0, :10, :10] = 1; b[0, 15:25, 15:25] = 1; return b\n")
    (repo / "Core" / "Identification.py").write_text(
        "import numpy as np, skimage.measure\n"
        "CALLS = []\n"
        "def DetectEdges(b, s, lo, hi, t): return np.zeros(b.shape, bool)\n"
        "def SeparateContours(b, e, t, s): return np.stack([skimage.measure.label(x >= t) for x in b]).astype(np.uint16)\n"
        "def Cleanup(labels, area, remove_border, fill): CALLS.append((area, remove_border, fill)); return labels\n")
    monkeypatch.syspath_prepend(str(repo))
    for name in [m for m in sys.modules if m == "Core" or m.startswith("Core.")]:
        monkeypatch.delitem(sys.modules, name)
    image = tmp_path / "organoid.tif"
    Image.fromarray(np.zeros((40, 50), np.uint16)).save(image)
    with pytest.raises(RuntimeError, match="pinned OptimizedModel"):
        arms.OrganoID(repo)
    monkeypatch.setattr(arms, "ORGANOID_ID_MODEL_SHA256", common.sha256(repo / "OptimizedModel"))
    result = arms.OrganoID(repo)(image, tmp_path)
    import Core.Identification as identification
    assert identification.CALLS == [(100, False, True)]        # border organoids kept, published defaults
    assert result["mask"].shape == (40, 50) and np.issubdtype(result["mask"].dtype, np.integer)
    assert result["trace"]["organoids"] == 2 and result["trace"]["remove_border"] is False
    assert "organoid_id" in common.EXTRA_ARMS and "organoid_id" not in common.ARMS


def test_dependency_probe_uses_current_organoid_checkout():
    origins = arms.probe_dependencies("shape_seg_co")
    assert common.REPO in Path(origins["shape.tools.organoid_segmenter.tool"]).parents


def test_recovered_failed_tool_call_does_not_invalidate_agent_delivery(tmp_path, dataset, monkeypatch):
    from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
    from cellpose.models import CellposeModel
    from shape.engine.openai import ChatOpenAI
    common.configure("legacy")
    weights = tmp_path / "weights"
    weights.write_bytes(b"synthetic CO")
    monkeypatch.setenv("ORGANOID_EXPECTED_SHA256", common.sha256(weights))
    model = object.__new__(CellposeModel)
    model.pretrained_model, model.device = str(weights), "cpu"
    monkeypatch.setattr(CellposeModel, "eval", lambda *a, **kw: dataset[2])
    tool = object.__new__(Organoid_Segmenter_Tool)

    def execute(self, **kwargs):
        if kwargs["diameter"] < 0:
            return {"error": "invalid diameter"}  # rejected before GPU work
        path = tmp_path / "prediction.npy"
        np.save(path, model.eval(dataset[2]))
        return {"mask_path": str(path)}

    monkeypatch.setattr(Organoid_Segmenter_Tool, "execute", execute)
    monkeypatch.setattr(ChatOpenAI, "generate", lambda *a, **kw: {"content": "OK", "usage": {"total_tokens": 13}})
    engine = object.__new__(ChatOpenAI)
    engine.model_string, engine.is_multimodal = common.MODEL, True

    def arm(image, out):
        tool.execute(diameter=-1)
        engine.generate("Choose a corrected diameter")
        result = tool.execute(diameter=100)
        return {"mask": common.labels(result["mask_path"]), "trace": {}}

    pair = common.discover(dataset[0], label_root=dataset[1])[0]
    record = runner.run_image(arm, pair, tmp_path / "attempt", "shape")
    assert record["delivered"] and not record["error"] and record["ap50"] == 1
    assert record["tool_calls"] == 2 and record["native_calls"] == 1
    assert record["call_trajectory"][0]["error"] == "invalid diameter"


def test_shared_react_session_delivers_agent_choice_without_gt(tmp_path, dataset):
    image = dataset[0] / "a.tif"
    seen = []

    class Tool:
        def execute(self, **kwargs):
            seen.append(kwargs)
            mask = tmp_path / "co.npy"
            np.save(mask, dataset[2])
            return {"mask_path": str(mask), "organoid_count": 1}

    class Engine:
        is_multimodal = True

        def generate(self, content, response_format, **kwargs):
            text = content[0] if isinstance(content, list) else content
            assert str(dataset[1]) not in text
            assert "scale_error" not in text and "CARD" not in text
            return {"content": response_format(thought="Deliver this call", action="finish", deliver="default"),
                    "usage": {"total_tokens": 19}}

    arm = object.__new__(arms.React)
    arm.tool, arm.engine = Tool(), Engine()
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    arm.verifier = OrganoidSegmentationAgenticVerifier()
    result = arm(image, tmp_path)
    assert np.array_equal(result["mask"], dataset[2])
    assert len(seen) == 1 and seen[0]["adaptive_rescue"] is False
    selected = result["trace"]["selected"]["a"]
    assert selected["rounds"] == 1 and selected["call_id"] == "default"
    assert selected["tokens"]["total_tokens"] == 19


def test_shape_uses_organoid_vrr_and_rejects_failed_delivery(tmp_path, dataset, monkeypatch):
    import shape.solver
    path = tmp_path / "chosen.npy"
    np.save(path, dataset[2])
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    chosen = {"image_id": "a", "mask_path": str(path)}
    seen = []

    def solve(**kwargs):
        seen.append(kwargs)
        return {"memory": SimpleNamespace(get_actions=lambda **kw: [{"result": chosen.copy()}]),
                "verification_summary": [{"audit": "preserved"}], "stop_reason": "delivered"}

    monkeypatch.setattr(shape.solver, "solve", solve)
    arm = arms.Shape(max_steps=10)
    result = arm(dataset[0] / "a.tif", tmp_path)
    assert np.array_equal(result["mask"], dataset[2])
    assert seen[0]["enabled_tools"] == ["Organoid_Segmenter_Tool"]
    assert seen[0]["enable_verification"] and seen[0]["enable_llm_verifier"]
    assert seen[0]["segmentation_cohort"] is False
    chosen["status"] = "failed"
    assert arm(dataset[0] / "a.tif", tmp_path)["mask"] is None


def test_resume_compare_and_changed_code_rejected(tmp_path, dataset, monkeypatch):
    path = manifest_for(tmp_path, dataset)
    source = {"source_sha256": "fake-source", "commit": "test", "dirty": True}
    monkeypatch.setattr(runner, "source_identity", lambda: source.copy())
    monkeypatch.setattr(arms, "probe_dependencies", lambda name: {"synthetic_backend": "test"})
    monkeypatch.setattr(arms, "make_arm", lambda *a: lambda image, out: {
        "mask": dataset[2] if image.stem == "a" else None, "tool_calls": 1, "trace": {}})
    args = runner.parser().parse_args(["run", "--arm", "stardist", "--manifest", str(path),
                                      "--run-dir", str(tmp_path / "results")])
    assert runner.run(args)["status"] == "COMPLETED_WITH_FAILURES"
    args.resume = True
    assert runner.run(args)["n_recorded"] == 2
    comp = runner.parser().parse_args(["compare", "--arms", "stardist", "--manifest", str(path),
                                      "--run-dir", str(tmp_path / "results")])
    summary = runner.compare(comp)["summary"]["stardist"]
    assert summary["ap50"] == .5 and summary["n_errors"] == 1
    orphan = tmp_path / "results/stardist/images/a/attempt_interrupted"
    orphan.mkdir()
    common.append(orphan / "LLM_CALLS.jsonl", {"tokens": 31, "seconds": .1})
    summary = runner.compare(comp)["summary"]["stardist"]
    assert summary["interrupted_attempt_tokens"] == 31
    assert summary["total_tokens_including_preflight"] == 31
    source["source_sha256"] = "changed-source"
    with pytest.raises(ValueError, match="identical"):
        runner.run(args)


def test_cross_arm_incomplete_or_mismatched_manifest_rejected(tmp_path, dataset):
    path = manifest_for(tmp_path, dataset)
    root = tmp_path / "results"
    folder = root / "shape"
    folder.mkdir()
    common.dump(folder / "run_meta.json", {"status": "PREFLIGHT_ONLY"})
    args = SimpleNamespace(run_dir=str(root), manifest=str(path), arms=["shape"])
    with pytest.raises(ValueError, match="Incomplete arm"):
        runner.compare(args)
    common.dump(folder / "run_meta.json", {"status": "COMPLETE", "identity": {"manifest": "different"}})
    with pytest.raises(ValueError, match="Different manifest"):
        runner.compare(args)
