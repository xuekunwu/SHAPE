"""Guards for the cached-embedding reuse path and the fingerprint that gates it.

Reuse trades a DINOv3 fine-tune for a re-clustering, which is the whole point -- and also
the whole risk: re-clustering a STALE embedding is a silently wrong answer, far worse than
the retrain it saves. So the tests below assert both directions: it must reuse when the
embedding-affecting configuration is identical, and it must refuse in every ambiguous case.
"""

import numpy as np
import pytest

ad = pytest.importorskip("anndata")

from shape.tools.cell_state_analyzer.tool import (  # noqa: E402
    Cell_State_Analyzer_Tool,
    _ADAPTATION_DEPTH,
    admissible_adaptation_depths,
    embedding_fingerprint,
    resolve_adaptation_depth,
)

BASE = {"adaptation_depth": "trained", "learning_rate": 3e-5, "temperature": 0.1,
        "augment_color": False, "field_aware": False}
CROPS = ["a.png", "b.png", "c.png"]


# --------------------------------------------------------------------- fingerprint

def test_clustering_knobs_do_not_change_the_fingerprint():
    """The load-bearing property: two runs differing only in clustering share an embedding."""
    base = embedding_fingerprint(BASE, CROPS)
    for knob in ("cluster_resolution", "n_neighbors"):
        assert embedding_fingerprint({**BASE, knob: 99}, CROPS) == base


@pytest.mark.parametrize("field,value", [
    ("adaptation_depth", "zero_shot"),
    ("learning_rate", 1e-4),
    ("temperature", 0.2),
    ("augment_color", True),
    ("field_aware", True),
    ("max_epochs", 3),
])
def test_every_embedding_param_changes_the_fingerprint(field, value):
    """Anything that alters the feature space must invalidate reuse."""
    assert embedding_fingerprint({**BASE, field: value}, CROPS) != embedding_fingerprint(BASE, CROPS)


def test_crop_set_matters_but_crop_order_does_not():
    base = embedding_fingerprint(BASE, CROPS)
    assert embedding_fingerprint(BASE, list(reversed(CROPS))) == base
    assert embedding_fingerprint(BASE, CROPS[:2]) != base


# --------------------------------------------------------------------- reuse gate

def _cached(tmp_path, fingerprint, n=12):
    rng = np.random.default_rng(0)
    a = ad.AnnData(X=rng.normal(size=(n, 8)).astype("float32"))
    a.obs["group"] = ["g0", "g1"] * (n // 2)
    a.obs["image_name"] = [f"img{i // 2}" for i in range(n)]
    a.obs.index = [f"cell_{i}" for i in range(n)]
    if fingerprint is not None:
        a.uns["embedding_fingerprint"] = fingerprint
    p = tmp_path / "cell_state_analyzed.h5ad"
    a.write(p)
    return str(p)


def test_reuse_refuses_on_fingerprint_mismatch(tmp_path):
    tool = Cell_State_Analyzer_Tool()
    path = _cached(tmp_path, "some-other-fingerprint")
    assert tool._reuse_cached_embedding(path, "expected", 0.5, 15) is None


def test_reuse_refuses_when_the_cache_carries_no_fingerprint(tmp_path):
    """A pre-fingerprint artifact must never be reused: we cannot tell what produced it."""
    tool = Cell_State_Analyzer_Tool()
    path = _cached(tmp_path, None)
    assert tool._reuse_cached_embedding(path, "expected", 0.5, 15) is None


def test_reuse_refuses_on_an_unreadable_cache(tmp_path):
    """Falling through costs a retrain; reusing garbage costs a wrong answer."""
    tool = Cell_State_Analyzer_Tool()
    bad = tmp_path / "cell_state_analyzed.h5ad"
    bad.write_bytes(b"not an h5ad")
    assert tool._reuse_cached_embedding(str(bad), "expected", 0.5, 15) is None


def test_reuse_reclusters_on_an_exact_match(tmp_path):
    tool = Cell_State_Analyzer_Tool()
    fp = embedding_fingerprint(BASE, CROPS)
    path = _cached(tmp_path, fp)
    out = tool._reuse_cached_embedding(path, fp, 0.5, 5)
    if out is None:                      # scanpy absent in this environment
        pytest.skip("clustering unavailable")
    assert out["mode"] == "reused-embedding"
    assert out["epochs_trained"] == 0
    assert out["training_config"]["stop_reason"] == "reused_cached_embedding"
    assert out["embedding_fingerprint"] == fp
    assert out["adata_path"] == path


# ------------------------------------------- stable-identity cache gate (task 087 SS6)

def test_reuse_refuses_a_legacy_cache_when_identity_is_now_requested(tmp_path):
    """A cache built before this contract (fingerprint matches, but no
    identity_sequence_digest) must never be reused once the caller asks for stable
    identity -- reusing it would silently hand back rows with no cell_id/crop_id."""
    tool = Cell_State_Analyzer_Tool()
    fp = embedding_fingerprint(BASE, CROPS)
    path = _cached(tmp_path, fp)   # legacy cache: no identity_sequence_digest in uns
    out = tool._reuse_cached_embedding(
        path, fp, 0.5, 15, expected_identity_digest="some-digest")
    assert out is None


def test_reuse_refuses_on_identity_digest_mismatch(tmp_path):
    """Same crop SET in a different order (or a different id sequence entirely) must not
    reuse a row-associated cache incorrectly, even though the fingerprint (which sorts the
    crop set) still matches."""
    tool = Cell_State_Analyzer_Tool()
    fp = embedding_fingerprint(BASE, CROPS)
    path = _cached(tmp_path, fp)
    a = ad.read_h5ad(path)
    a.uns["identity_sequence_digest"] = "digest-for-order-A"
    a.write(path)
    out = tool._reuse_cached_embedding(
        path, fp, 0.5, 15, expected_identity_digest="digest-for-order-B")
    assert out is None


def test_reuse_allows_an_identity_carrying_cache_on_an_exact_digest_match(tmp_path):
    tool = Cell_State_Analyzer_Tool()
    fp = embedding_fingerprint(BASE, CROPS)
    path = _cached(tmp_path, fp)
    a = ad.read_h5ad(path)
    a.uns["identity_sequence_digest"] = "digest-for-order-A"
    a.write(path)
    out = tool._reuse_cached_embedding(
        path, fp, 0.5, 15, expected_identity_digest="digest-for-order-A")
    if out is None:
        pytest.skip("clustering unavailable")
    assert out["mode"] == "reused-embedding"


def test_reuse_ignores_identity_digest_when_not_requested(tmp_path):
    """Legacy behavior preserved: a caller that does not request identity
    (expected_identity_digest=None) reuses exactly as before this task, even against a
    legacy cache with no identity schema at all."""
    tool = Cell_State_Analyzer_Tool()
    fp = embedding_fingerprint(BASE, CROPS)
    path = _cached(tmp_path, fp)
    out = tool._reuse_cached_embedding(path, fp, 0.5, 15, expected_identity_digest=None)
    if out is None:
        pytest.skip("clustering unavailable")
    assert out["mode"] == "reused-embedding"


# --------------------------------------------------------------------- adaptation depth

def test_adaptation_depth_excludes_the_meaningless_corner():
    """zero_shot AND freeze_backbone is not a state the categorical can express."""
    pairs = {resolve_adaptation_depth(d) for d in _ADAPTATION_DEPTH}
    assert (True, True) not in pairs
    assert resolve_adaptation_depth("zero_shot") == (True, False)
    assert resolve_adaptation_depth("frozen_head") == (False, True)
    assert resolve_adaptation_depth("trained") == (False, False)


def test_more_than_three_channels_use_deterministic_adapter_at_every_depth():
    """The unified C-to-3 adapter removes the old random patch-embed restriction."""
    assert set(admissible_adaptation_depths(4)) == set(_ADAPTATION_DEPTH)
    assert set(admissible_adaptation_depths(1)) == set(_ADAPTATION_DEPTH)
    assert set(admissible_adaptation_depths(None)) == set(_ADAPTATION_DEPTH)
    assert resolve_adaptation_depth("zero_shot", n_channels=5) == (True, False)


def test_unknown_depth_is_rejected():
    with pytest.raises(ValueError, match="unknown adaptation_depth"):
        resolve_adaptation_depth("finetune_everything")


# --------------------------------------------------------------------- warmup

def test_warmup_ramps_linearly_then_holds_the_base_lr():
    from shape.tools.cell_state_analyzer.tool import warmup_lr
    base = 1e-4
    ramp = [warmup_lr(i, 4, base) for i in range(4)]
    assert ramp == [base * 0.25, base * 0.5, base * 0.75, base]
    assert ramp == sorted(ramp)                      # monotonically increasing
    assert all(warmup_lr(i, 4, base) == base for i in (4, 5, 100))


def test_warmup_is_a_no_op_when_disabled():
    """warmup_fraction=0 must reproduce the previous flat-LR behaviour exactly."""
    from shape.tools.cell_state_analyzer.tool import warmup_lr
    assert warmup_lr(0, 0, 3e-5) == 3e-5


def test_warmup_never_exceeds_the_base_lr():
    from shape.tools.cell_state_analyzer.tool import warmup_lr
    assert all(warmup_lr(i, 10, 3e-5) <= 3e-5 for i in range(20))


# --------------------------------------------------------------------- backbone weights

def test_unavailable_backbone_is_refused_rather_than_silently_randomised():
    """The old code loaded the vitb16 checkpoint for ANY backbone_name with strict=False,
    yielding a partly-random backbone and a correct-looking feature dim -- invisible to
    every test. Selecting an architecture with no checkpoint must now fail loudly."""
    from shape.tools.cell_state_analyzer.tool import DinoV3Projector
    with pytest.raises(ValueError, match="no checkpoint available"):
        DinoV3Projector(backbone_name="dinov3_vitl16")


def test_explicit_local_dinov3_repo_never_uses_github(stub_backbone, monkeypatch, tmp_path):
    from shape.tools.cell_state_analyzer import tool as tool_mod

    repo = tmp_path / "dinov3"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# pinned test checkout\n", encoding="ascii")
    monkeypatch.setenv("SHAPE_DINOV3_REPO", str(repo))
    calls = []
    stub_load = tool_mod.torch.hub.load

    def spy_load(*args, **kwargs):
        calls.append((args, kwargs))
        return stub_load(*args, **kwargs)

    monkeypatch.setattr(tool_mod.torch.hub, "load", spy_load)
    tool_mod.DinoV3Projector(input_channels=6)

    assert calls == [((str(repo.resolve()), "dinov3_vitb16"),
                      {"pretrained": False, "source": "local"})]


# --------------------------------------------------------- mid-loop retune vocabulary invariant

def test_every_midloop_rep_lever_maps_to_a_real_execute_kwarg():
    """The invariant this task exists to enforce, at the mid-loop layer QC found uncovered:
    a lever the representation recommend can propose must reach a real analyzer kwarg through
    map_tool_params -- otherwise it is a silent no-op retune (validates, then is dropped),
    the exact dead-lever class being removed. `freeze_backbone` regressed here: retired from
    stage_map's param_map but left in the recommend vocabulary."""
    import inspect
    import typing

    from shape.agent.verification.llm_recommend import RepresentationVerdict
    from shape.agent.verification.stage_map import map_tool_params
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    # the levers the mid-loop schema can name
    param_field = RepresentationVerdict.model_fields["param"]
    levers = [a for a in typing.get_args(param_field.annotation) if a != "none"]
    assert levers, "representation retune vocabulary is empty"

    execute_kwargs = set(inspect.signature(Cell_State_Analyzer_Tool.execute).parameters)
    for lever in levers:
        mapped = map_tool_params(
            "Cell_State_Analyzer_Tool", {lever: True}, execute_kwargs)
        assert mapped, (
            f"mid-loop lever {lever!r} maps to no analyzer kwarg -- a silent no-op retune")

    # and the retired one must NOT be proposable
    assert "freeze_backbone" not in levers
    assert "adaptation_depth" not in levers   # pre-flight only, never mid-loop


# ------------------------------- managed defaults: no retired freeze_backbone lever

def test_managed_defaults_use_adaptation_depth_not_freeze_backbone():
    """QC (diff-review b1317813) found the freeze_backbone->adaptation_depth migration done in the
    tool but not in stage_map default_params. This guards the managed surface so the retired lever
    cannot silently return."""
    # managed-tool default_params expose adaptation_depth, not the dropped freeze_backbone
    from shape.agent.verification.stage_map import default_params_for_tool
    dp_defaults = default_params_for_tool("Cell_State_Analyzer_Tool")
    assert "freeze_backbone" not in dp_defaults
    assert dp_defaults.get("adaptation_depth") == "trained"


# --------------------- shared managed layer: default_params filtered to each tool's real kwargs

def test_default_params_never_advertise_a_lever_the_tool_cannot_accept():
    """Every unified managed default must map to a real execute kwarg."""
    import importlib
    import inspect

    from shape.agent.verification.stage_map import (
        default_params_for_tool, stages_for_tool)

    for tool_name in ("Cell_State_Analyzer_Tool",):
        tool_dir = "_".join(p.lower() for p in tool_name.replace("_Tool", "").split("_"))
        cls = getattr(importlib.import_module(f"shape.tools.{tool_dir}.tool"), tool_name)
        kwargs = set(inspect.signature(cls.execute).parameters)
        combined = {}
        for s in stages_for_tool(tool_name):
            combined.update(s.param_map)
        for key in default_params_for_tool(tool_name):
            mapped = combined.get(key, key)
            assert mapped in kwargs, (
                f"{tool_name}: default {key!r} maps to {mapped!r}, not an execute kwarg "
                f"-- a silent no-op default")

    # The unified analyzer exposes adaptation_depth for every channel count.
    assert "adaptation_depth" in default_params_for_tool("Cell_State_Analyzer_Tool")
