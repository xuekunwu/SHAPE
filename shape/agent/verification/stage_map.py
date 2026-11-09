"""Stage configuration: the contract between SHAPE tools and VLRR verifiers.

This module is pure configuration/data (no heavy deps, no tool imports). It maps
each managed pipeline stage to:
  - the verifier class that scores it (+ default constructor kwargs),
  - the parameter-name mapping between VLRR's vocabulary and the tool's actual
    `execute(...)` keyword arguments (resolving the naming mismatches documented
    in the integration plan),
  - the tool(s) that produce the stage and the upstream stage to escalate to,
  - the adapter key used to turn a tool result into the verifier's input shape.

The fixed bioimage DAG is:

    image_loading -> segmentation -> crop
                  -> representation -> morphological_state

`representation` and `morphological_state` are both produced by the
Cell_State_Analyzer_* tools (one tool, two verification sub-stages), so they map
to the same tool name with different adapters and parameter subsets.

NOTE: some tool kwargs referenced in `param_map` (e.g. `min_size`, `crop_size`,
`n_neighbors`, `correction_strength`, `input_strategy`) are introduced by the
Phase 1 tool edits. Until a given tool exposes them, the adapter must skip
unknown kwargs rather than crash -- see `map_params`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type

from .base import BaseVerifier
from .verifiers import (
    CropAgenticVerifier,
    CropVerifier,
    ImageLoadingVerifier,
    MorphologicalStateVerifier,
    OrganoidSegmentationAgenticVerifier,
    RepresentationAgenticVerifier,
    RepresentationVerifier,
    SegmentationAgenticVerifier,
    SegmentationVerifier,
    BatchCorrectionVerifier,
)


@dataclass(frozen=True)
class StageSpec:
    """One verification stage in the bioimage DAG."""

    stage: str
    verifier_cls: Type[BaseVerifier]
    # tool class names (as the planner/executor know them) that produce this stage
    tool_names: tuple[str, ...]
    # VLRR-param-name -> tool-execute-kwarg-name
    param_map: dict[str, str] = field(default_factory=dict)
    # sensible starting parameters in VLRR vocabulary (planner may override)
    default_params: dict = field(default_factory=dict)
    # stage to escalate to when a fault is localized upstream (None = pipeline head)
    upstream: Optional[str] = None
    # default constructor kwargs for the verifier (runtime may add e.g. bit_range)
    verifier_kwargs: dict = field(default_factory=dict)
    # a tool's own verifier, when it differs from the stage's (tool name -> verifier class): a tool
    # tuned by the agentic VRR (`base.AgenticVerifier`) while the stage's other tools keep theirs
    verifier_by_tool: dict = field(default_factory=dict)


# Independent managed workflows are registered separately from the bioimage DAG. Keeping
# them in the same ordered tuple made scope handling pretend batch correction was an
# upstream image stage and then special-case it back out again.
INDEPENDENT_STAGES: tuple[StageSpec, ...] = (
    StageSpec(stage="batch_correction", verifier_cls=BatchCorrectionVerifier,
              tool_names=("Batch_Corrector_Tool",),
              param_map={"harmony_params": "harmony_params"},
              # the tool's DEFAULT_CONFIG (tool.py), spelled out: this module imports no tool code
              default_params={"harmony_params": {
                  "theta": 2.0, "lambda_value": None, "sigma": 0.1, "nclust": None, "random_state": 0,
                  "max_iter_harmony": None, "max_iter_kmeans": None, "epsilon_cluster": None,
                  "epsilon_harmony": None, "block_size": None, "tau": None,
                  "input_scale": "none", "input_pca": 0}},
              upstream=None),
)

# Tools the system routes itself and the planner never sees (`construct_solver` filters them out of the
# planner's toolbox): batch correction follows every successful Cell_State_Analyzer_Tool step when the run's
# metadata profile has batch labels (`solver._batch_followup`).
SYSTEM_ROUTED_TOOLS: tuple[str, ...] = ("Batch_Corrector_Tool",)

# Canonical, ordered bioimage DAG. The list order is the pipeline order.
STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        stage="image_loading",
        verifier_cls=ImageLoadingVerifier,
        tool_names=(),  # loading is not a "tool" call; handled by the loader wrapper
        param_map={},
        default_params={},
        upstream=None,
        verifier_kwargs={"bit_range": 255.0, "expected_channels": 1},
    ),
    StageSpec(
        stage="segmentation",
        verifier_cls=SegmentationVerifier,
        tool_names=("Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"),
        # Both segmenters use the shared agentic VRR with their own CARD/model action space.
        verifier_by_tool={"Cell_Segmenter_Tool": SegmentationAgenticVerifier,
                          "Organoid_Segmenter_Tool": OrganoidSegmentationAgenticVerifier},
        param_map={
            "diameter": "diameter",
            "flow_threshold": "flow_threshold",
            "cellprob_threshold": "cellprob_threshold",
            "min_size": "min_size",  # Phase 1: new tool kwarg
            "niter": "niter",        # max dynamics iterations (large/elongated cells)
            "tile_overlap": "tile_overlap",  # tile overlap (boundary cells)
            "augment": "augment",    # test-time augmentation (faint-cell recall)
            "resample": "resample",  # full-resolution dynamics
            # Verifier-invoked color/class filter: a POST-seg retune action (drop a non-target
            # color population from the mask via an Otsu B-R cut) -- unlike the re-seg knobs
            # above, the tool applies it to the produced mask, not to Cellpose. Value is a
            # categorical directive (keep_high_br/keep_low_br) passed through by apply_bounded.
            "color_class_filter": "color_class_filter",
            "_vlrr_attempt_index": "attempt_id",  # audit-only retry artifact suffix
            # NOTE: bsize is NOT tunable -- Cellpose-SAM only accepts bsize=256.
        },
        default_params={"diameter": 30.0, "flow_threshold": 0.4,
                        "cellprob_threshold": 0.0, "min_size": 15},
        upstream="image_loading",
    ),
    StageSpec(
        stage="crop",
        verifier_cls=CropVerifier,
        tool_names=("Single_Cell_Cropper_Tool",),
        # Single_Cell_Cropper_Tool is tuned by the agentic VRR (CARD.md "Agentic VRR"); the linear
        # CropVerifier stays the stage's own verifier (the benchmarks' linear arm runs it)
        verifier_by_tool={"Single_Cell_Cropper_Tool": CropAgenticVerifier},
        param_map={
            "padding": "margin",
            "crop_size": "crop_size",  # Phase 1: new tool kwarg
            # the agentic loop's calls name the tool's own kwargs; without these the geometry of a
            # retune never reached the tool on the solve() path (the linear loop's cell_overrides too)
            "margin": "margin",
            "margin_frac": "margin_frac",
            "margin_mode": "margin_mode",
            "recenter": "recenter",
            "tight_buffer_px": "tight_buffer_px",
            "cell_overrides": "cell_overrides",
            "min_area": "min_area",
            "output_subdir": "output_subdir",   # each agentic call's own directory inside tool_cache
        },
        # no margin ("padding"): the tool's own rule by mask file name (`default_min_area_margin`) is the default
        default_params={"crop_size": 256},
        upstream="segmentation",
    ),
    StageSpec(
        stage="representation",
        verifier_cls=RepresentationVerifier,
        tool_names=("Cell_State_Analyzer_Tool",),
        # Cell_State_Analyzer_Tool is tuned by the agentic VRR (CARD.md "Agentic VRR"; trial 0 the frozen
        # backbone, `RepresentationAgenticVerifier.trial0_call`); the linear RepresentationVerifier stays the
        # stage's own verifier
        verifier_by_tool={"Cell_State_Analyzer_Tool": RepresentationAgenticVerifier},
        param_map={
            # lr/epoch are confirmed by the pre-flight calibration (CARD.md); identity-mapped
            # so the managed stage can carry them. Defaults = the tool defaults (no behavior
            # change unless the calibration sets them).
            "learning_rate": "learning_rate",
            "max_epochs": "max_epochs",
            # task 008 P3: up-front-only knobs (recommend_initial_analyzer_config); the
            # mid-loop retune vocabulary stays "input" only (categorical). Identity-mapped
            # (same name in the tool's execute() already) so the up-front recommend's config
            # actually reaches the analyzer -- defaults = today's tool defaults, so a run with
            # no up-front override behaves exactly as before.
            "augment_color": "augment_color",
            "field_aware": "field_aware",
            # Not in the linear verifier's mid-loop retune vocabulary; its up-front depth and the
            # agentic VRR's trial 0 (adaptation_depth "zero_shot", the frozen backbone) reach the tool
            # through this entry. The agentic VRR's depth lever is trainable_pct below.
            "adaptation_depth": "adaptation_depth",
            # the agentic VRR's levers (CARD.md "## Levers"; learning_rate / max_epochs / augment_color
            # above) and the non-lever kwargs of its calls and of an embedding-only call: identity-mapped,
            # since map_tool_params drops unmapped keys silently. init_from is a verifier lever and never
            # reaches the tool (it becomes init_checkpoint).
            "view_pairing": "view_pairing",
            "contrastive_mask": "contrastive_mask",
            "batch_sampling": "batch_sampling",
            "trainable_pct": "trainable_pct",
            "head_learning_rate": "head_learning_rate",
            "patch_embed_lr_mult": "patch_embed_lr_mult",
            "weight_decay": "weight_decay",
            "clip_grad_norm": "clip_grad_norm",
            "warmup_fraction": "warmup_fraction",
            "head_warmup_epochs": "head_warmup_epochs",
            "gram_weight": "gram_weight",
            "koleo_weight": "koleo_weight",
            "plate_mean_weight": "plate_mean_weight",
            "plate_adversarial_weight": "plate_adversarial_weight",
            "intensity_norm": "intensity_norm",
            "batch_size": "batch_size",
            "augmentation": "augmentation",
            "crop_scale_min": "crop_scale_min",
            "drop_path": "drop_path",
            "nn_k": "nn_k",
            "nn_space": "nn_space",
            "nn_mutual": "nn_mutual",
            "nn_refresh_epochs": "nn_refresh_epochs",
            "init_checkpoint": "init_checkpoint",
            "output_subdir": "output_subdir",
            "split_key": "split_key",
            "val_fraction": "val_fraction",
            "max_minutes": "max_minutes",
            "eval_every_epochs": "eval_every_epochs",
            "embed_only": "embed_only",
            "crop_array": "crop_array",         # the packed-array pixel source (amendment A1); not a lever
        },
        default_params={"learning_rate": 3e-5, "max_epochs": None,
                        "augment_color": True, "field_aware": False, "adaptation_depth": "trained"},
        upstream="image_loading",
    ),
    StageSpec(
        stage="morphological_state",
        verifier_cls=MorphologicalStateVerifier,
        tool_names=("Cell_State_Analyzer_Tool",),
        param_map={
            "leiden_resolution": "cluster_resolution",
            "n_neighbors": "n_neighbors",  # Phase 1: new tool kwarg
            # Internal integration flag: morphology reruns must re-cluster the accepted
            # representation, never train a second SSL feature space.
            "reuse_embedding": "reuse_embedding",
        },
        default_params={"leiden_resolution": 0.5, "n_neighbors": 15},
        upstream="representation",
        verifier_kwargs={"multi_group": False},
    ),
)

MANAGED_STAGES: tuple[StageSpec, ...] = (*INDEPENDENT_STAGES, *STAGES)

# Lookups -------------------------------------------------------------------

_STAGE_BY_NAME: dict[str, StageSpec] = {s.stage: s for s in MANAGED_STAGES}
_TOOL_DEFAULT_OVERRIDES: dict[str, dict] = {
    "Organoid_Segmenter_Tool": {"diameter": None},   # native scale (Cellpose resizes by 30 / diameter)
}


def stage_by_name(name: str) -> Optional[StageSpec]:
    return _STAGE_BY_NAME.get(name)


def verifier_cls_for(spec: StageSpec, tool_name: Optional[str]) -> Type[BaseVerifier]:
    """The verifier class `tool_name` is verified with in `spec` (its own, else the stage's)."""
    return (spec.verifier_by_tool or {}).get(tool_name, spec.verifier_cls)


def stages_for_tool(tool_name: str) -> list[StageSpec]:
    """All verification stages produced by a given tool, in pipeline order.

    Returns [] for tools that are not part of the managed VLRR pipeline (search,
    visualization, etc.) -- those bypass verification entirely.
    """
    return [s for s in MANAGED_STAGES if tool_name in s.tool_names]


def is_managed_tool(tool_name: str) -> bool:
    return any(tool_name in s.tool_names for s in MANAGED_STAGES)


# Task boundary (045 Phase 3, Fix 0) ---------------------------------------------
#
# The planner is the final authority, but its authority must be BOUNDED by the query:
# it must not run beyond what the query asks (the vicious cycle -- a segment/count
# query wandering crop -> analyze -> crash). We make that boundary explicit, derived
# from THIS DAG (single source of truth), not hard-coded in a prompt.
#
# The one bit of task knowledge is which stage's OWN output answers a query type; the
# in-/out-of-scope stage sets are then DERIVED deterministically from the topological
# STAGES order (never guessed by the LLM).


@dataclass(frozen=True)
class TaskSpec:
    """The query's stage boundary. `required_terminal_stage` is the stage whose own
    output answers the ask; `in_scope_stages` is the DAG prefix needed to reach it;
    `out_of_scope_stages` must NOT run for this query. A None terminal = could not
    classify -> fail-open (everything in scope, the scope guard is inert)."""

    query_type: str
    required_terminal_stage: Optional[str]
    in_scope_stages: frozenset
    out_of_scope_stages: frozenset


def _terminal_for_query_type(query_type: Optional[str]) -> Optional[str]:
    """Map an `analyze_query` query-type label to the DAG stage that answers it.

    counting / pure segmentation -> `segmentation` is terminal (crop/analysis are out
    of scope). morphology / cell-state / comparison / clustering -> the full pipeline
    (`morphological_state`). Unknown -> None (fail-open: no boundary enforced)."""
    s = (query_type or "").strip().lower()
    if not s:
        return None
    if "count" in s or ("segment" in s and "state" not in s and "morph" not in s):
        return "segmentation"
    if any(k in s for k in ("morph", "state", "compar", "cluster", "phenotype")):
        return "morphological_state"
    return None


def _stages_up_to(terminal: str) -> frozenset:
    """The set of stages at or before `terminal` in the topological STAGES order --
    i.e. the stages needed to reach it. STAGES is a valid topo order of the DAG."""
    names = [s.stage for s in STAGES]
    if terminal not in names:
        return frozenset(names)          # fail-open
    return frozenset(names[: names.index(terminal) + 1])


def task_spec_for(query_type: Optional[str]) -> TaskSpec:
    """Derive the query's stage boundary from its type. Deterministic; DAG-derived."""
    all_stages = frozenset(s.stage for s in STAGES)
    terminal = _terminal_for_query_type(query_type)
    if terminal is None:
        # could not classify -> everything in scope, out-of-scope empty (guard inert)
        return TaskSpec(query_type or "unknown", None, all_stages, frozenset())
    in_scope = _stages_up_to(terminal)
    return TaskSpec(query_type or "unknown", terminal, in_scope, all_stages - in_scope)


def tool_in_scope(tool_name: str, spec: Optional[TaskSpec]) -> bool:
    """Is `tool_name` allowed for this query's boundary?

    A MANAGED tool whose (every) stage is out of scope is blocked (e.g. the cropper /
    analyzer on a segment/count query). Unmanaged tools (search / visualize) and any
    tool with no TaskSpec (fail-open) always pass -- the scope guard governs the
    managed pipeline only. Returns True = in scope (allowed)."""
    if spec is None or not spec.out_of_scope_stages:
        return True
    if tool_name == "Batch_Corrector_Tool":
        return True
    stages = {s.stage for s in stages_for_tool(tool_name)}
    if not stages:
        return True                      # unmanaged tool: not scope-gated here
    return not stages.issubset(spec.out_of_scope_stages)


def _tool_execute_kwargs(tool_name: str) -> Optional[set]:
    """The set of execute() kwarg names for `tool_name`, or None if the class cannot be
    resolved. Mirrors executor's name->dir convention; failure degrades to None (no filtering)
    rather than raising, so a resolution problem never breaks the planner path."""
    import ast
    import importlib
    import inspect
    from pathlib import Path

    tool_dir = "_".join(p.lower() for p in tool_name.replace("_Tool", "").split("_"))
    for mod_path in (f"shape.tools.{tool_dir}.tool", f"tools.{tool_dir}.tool"):
        try:
            module = importlib.import_module(mod_path)
            cls = getattr(module, tool_name)
            return set(inspect.signature(cls.execute).parameters)
        except Exception:
            continue
    source_path = Path(__file__).resolve().parents[2] / "tools" / tool_dir / "tool.py"
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != tool_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "execute":
                continue
            args = [
                *item.args.posonlyargs,
                *item.args.args,
                *item.args.kwonlyargs,
            ]
            names = {arg.arg for arg in args}
            if item.args.kwarg:
                names.add(item.args.kwarg.arg)
            return names
    return None


def default_params_for_tool(tool_name: str) -> dict:
    """Planner-facing: recommended initial VLRR params for a managed tool.

    Merges the default params of every stage the tool produces (the analyzer maps
    to both representation and morphological_state). Returns {} for unmanaged tools.

    A default is only offered if it maps (via the stage param_map) to a kwarg the tool's
    execute() actually accepts -- the SAME gate map_tool_params applies. A default that no
    longer maps to the unified tool is filtered here, preventing a stale
    planner lever from becoming a silent no-op. If the tool class cannot be resolved, defaults
    pass through unfiltered.
    """
    params: dict = {}
    combined: dict = {}
    for s in stages_for_tool(tool_name):
        params.update(s.default_params)
        combined.update(s.param_map)
    params.update(_TOOL_DEFAULT_OVERRIDES.get(tool_name, {}))
    available = _tool_execute_kwargs(tool_name)
    if available is None:
        return params
    return {k: v for k, v in params.items()
            if (combined.get(k) or k) in available}


def map_tool_params(tool_name: str, vlrr_params: dict, available_kwargs: set[str]) -> dict:
    """Translate VLRR params to a tool's execute() kwargs, merging across the
    tool's stages (the analyzer maps to representation + morphological_state).

    Only keys whose mapped tool-kwarg exists in `available_kwargs` are forwarded.
    """
    combined: dict[str, str] = {}
    for s in stages_for_tool(tool_name):
        combined.update(s.param_map)
    out: dict = {}
    for vlrr_name, value in vlrr_params.items():
        tool_kwarg = combined.get(vlrr_name)
        if tool_kwarg and tool_kwarg in available_kwargs:
            out[tool_kwarg] = value
    return out


def map_params(spec: StageSpec, vlrr_params: dict, available_kwargs: set[str]) -> dict:
    """Translate VLRR-vocabulary params to a tool's execute() kwargs.

    Only keys whose mapped tool-kwarg actually exists in `available_kwargs`
    (the tool's real signature) are forwarded; unmapped/not-yet-added kwargs are
    skipped so a tool that hasn't gained a new param yet degrades gracefully.
    """
    out: dict = {}
    for vlrr_name, value in vlrr_params.items():
        tool_kwarg = spec.param_map.get(vlrr_name)
        if tool_kwarg and tool_kwarg in available_kwargs:
            out[tool_kwarg] = value
    return out


def invoke_managed_tool(tool, tool_name: str, base_kwargs: dict,
                        vlrr_params: Optional[dict] = None,
                        query_cache_dir: Optional[str] = None):
    """Deterministically run a managed pipeline tool INSTANCE.

    The single managed-tool invocation shared by the core ``Executor`` and the
    benchmark harness (one orchestrator -- no duplicate runner). Builds execute()
    kwargs by mapping VLRR params to the tool's real signature, injecting
    ``query_cache_dir``, and dropping unknown kwargs unless the tool takes
    ``**kwargs``. No LLM, no exec-string. Errors propagate to the caller.
    """
    import inspect

    sig = inspect.signature(tool.execute)
    available = set(sig.parameters.keys())
    accepts_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    kwargs = dict(base_kwargs or {})
    if (query_cache_dir is not None and "query_cache_dir" in available
            and "query_cache_dir" not in kwargs):
        kwargs["query_cache_dir"] = query_cache_dir
    if vlrr_params:
        kwargs.update(map_tool_params(tool_name, vlrr_params, available))
    if not accepts_var_kw:
        kwargs = {k: v for k, v in kwargs.items() if k in available}
    return tool.execute(**kwargs)
