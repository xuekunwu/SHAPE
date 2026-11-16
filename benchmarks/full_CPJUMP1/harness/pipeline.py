"""Native SHAPE tools and shared VRR loops; each heavy stage runs in a fresh process."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import contextlib
import gc
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import tifffile

from .data import CHANNELS, CPJUMP1_USAGES, development, digest, read_json, technical, well_profiles, within, write_json


@dataclass(frozen=True)
class Dataset:
    """The dataset-specific values the stages, the agent arms and the CLI use. A run's CONFIG.json names its
    dataset (`dataset`; absent: CPJUMP1), so every stage worker, native worker and crop worker process finds it."""
    name: str                     # the benchmark's name in the agent prompts
    cli: str                      # the CLI module whose stage / preflight / native-trial workers run this dataset
    data: str | None              # the module with load_prepared(prep, *, verify_pixels); None: this harness's
    score: str                    # the module with score(prep, run, control=None)
    n_images: int                 # prepared views of a full run
    marker: str                   # a full run's completion marker
    image_shape: tuple            # (rows, columns) of every image and segmentation mask
    channels: tuple               # the raw CYX stack's channel names
    dtype: str | None             # the raw stack's dtype; None: each prepared row's own `dtype`
    plate_type: str               # the calibration view's Metadata_PlateType
    mask_label: bool              # cells.json records each cell's delivered mask instance id (`mask_label`)
    tasks: dict                   # stage -> SHAPE's verifier request / brief, also the Codex prompt's "Task:" line
    inputs: dict                  # stage -> the Codex prompt's input line (format fields n, cells, dtype, rule)
    input_channels: dict          # image stage -> the channels of its raw input, as the Codex inspect labels name them
    rules: dict                   # the batch calibration rule's decisions, as the Codex prompt states them
    withheld: str                 # the Codex prompt's statement of what is not available
    usages: tuple = CPJUMP1_USAGES  # the (train, val, held-out) values of an image row's `usage`
    allowed_sites: tuple = (2,)   # complete views per well; dataset admission verifies the exact source set
    allow_empty_masks: bool = False  # explicit dataset contract; zero detections are not biological empty-field GT
    # The SHAPE tool directory of the segment stage: "cell_segmenter" (Cellpose-SAM, the benchmarks) or
    # "organoid_segmenter" (the fixed custom CO model; `SHAPE_ORGANOID_MODEL_PATH`). Its CARD declares the levers
    # every arm sets; `segmenter(spec)` builds the tool, its fixed kwargs and its registry.
    segment_tool: str = "cell_segmenter"

    def loader(self, own):
        """The dataset's load_prepared; `own` is the caller's imported one (this harness's data) when data is None."""
        return own if self.data is None else importlib.import_module(self.data).load_prepared

    def raw_dtype(self, row: dict) -> str:
        return self.dtype or row["dtype"]


SEGMENT_REQUEST = ("Segment whole A549 cells, one instance per cell. Fluorescence channels are DNA, AGP; "
                   "the resulting masks crop a separate five-channel raw image at identical coordinates.")
CROP_BRIEF = "Five-channel fluorescence single-cell morphology: preserve all channels."
EMBED_BRIEF = ("Five-channel A549 CRISPR; two plates, two sites per well. Only development wells are visible. "
               "Held-out genes are excluded entirely; frozen inference follows parameter selection.")
BATCH_BRIEF = ("Two technical plates of CRISPR well profiles. "
               "Calibration references are opaque development guide IDs; negative controls are NO_SITE. "
               "Joint unlabeled correction is transductive.")
CPJUMP1 = Dataset(
    name="CPJUMP1", cli="benchmarks.full_CPJUMP1.harness.cli", data=None, score="benchmarks.full_CPJUMP1.harness.score",
    n_images=1000, marker="CPJUMP1_COMPLETE", image_shape=(1080, 1080), channels=CHANNELS, dtype="uint16",
    plate_type="CRISPR", mask_label=False,
    tasks={"segment": SEGMENT_REQUEST, "crop": CROP_BRIEF, "embed": EMBED_BRIEF, "batch": BATCH_BRIEF},
    inputs={"segment": "Inputs: {n} images (sites) of A549 Cas9 CRISPR cells on two plates; each a 1080 x 1080 "
                       "two-channel uint16 TIFF (DNA, AGP) taken from the site's five-channel image "
                       "(Mito, AGP, RNA, ER, DNA).",
            "crop": "Inputs: {n} images; each a 1080 x 1080 five-channel uint16 TIFF (Mito, AGP, RNA, ER, DNA) "
                    "and its cell mask delivered by this arm's segment stage (min_area is that segmentation's "
                    "min_size). The delivered crop boxes are cut from the raw five-channel image as the "
                    "embedding stage's input.",
            "embed": "Inputs: the single-cell crops of this arm's crop stage. Calls fit and validate on development "
                     "wells only: at most {cells} cells per development site; "
                     "development_train wells train, development_val wells validate; evaluation wells never enter "
                     "a call. input is the pretrained backbone (zero_shot) on those cells, not a call. After "
                     "finish, a fresh worker embeds every retained cell with the delivered call's frozen "
                     "checkpoint (the pretrained backbone for input), within the same arm deadline.",
            "batch": "Inputs: the well profiles of this arm's embed stage (median cell embedding per site, mean "
                     "of a well's two sites), batch key plate (two plates). Each call runs the tool on all rows "
                     "with harmony_params = the tool's DEFAULT_CONFIG overridden by your parameters. input is "
                     "the uncorrected profiles, not a call. Batch calibration rule (shared by all arms): "
                     "correction runs only when the tool's batch-effect detector reports a batch effect on all "
                     "rows and at least one calibration reference group occurs on two or more plates. "
                     "Here the rule's decision is {rule}."},
    input_channels={"segment": "DNA, AGP", "crop": "Mito, AGP, RNA, ER, DNA"},
    rules={"RUN": "RUN (correction admitted)", "SKIP": "SKIP (no batch effect detected on all rows)",
           "ABSTAIN": "ABSTAIN (no calibration reference group occurs on two or more plates)"},
    withheld="Private labels and evaluation genes are not available.", usages=CPJUMP1_USAGES)

MODALITY_PROBE_SEGMENT_REQUEST = ("Segment whole cells, one instance per cell. Fluorescence channels are "
                                  "DNA, AGP; the resulting masks crop a separate five-channel raw image at "
                                  "identical coordinates.")
MODALITY_PROBE_EMBED_BRIEF = ("Five-channel Cell Painting images spanning two cell types (A549, U2OS) and "
                              "three perturbation modalities (CRISPR knockout, small-molecule compound, "
                              "ORF overexpression), one plate per (cell type, modality) combination -- not "
                              "CPJUMP1's two-plate same-biology replicate design. Only development wells "
                              "are visible; held-out wells are excluded entirely.")
MODALITY_PROBE_BATCH_BRIEF = ("Six (cell type, modality) combinations, each its own single plate -- plate "
                              "is confounded with the biological condition here, unlike CPJUMP1's two "
                              "technical-replicate plates of the same knockouts. No calibration reference "
                              "spans two or more plates by construction, so correction is expected to "
                              "ABSTAIN (the existing admission rule, unmodified): batch correction must "
                              "never erase a real biological difference it was never designed to separate "
                              "from a technical one.")
MODALITY_PROBE = Dataset(
    name="ModalityProbe", cli="benchmarks.full_CPJUMP1.harness.cli",
    data="benchmarks.full_CPJUMP1.harness.modality_probe_data",
    score="benchmarks.full_CPJUMP1.harness.modality_probe_score",
    n_images=600, marker="MODALITY_PROBE_COMPLETE", image_shape=(1080, 1080), channels=CHANNELS, dtype="uint16",
    plate_type="Mixed", mask_label=False,
    tasks={"segment": MODALITY_PROBE_SEGMENT_REQUEST, "crop": CROP_BRIEF, "embed": MODALITY_PROBE_EMBED_BRIEF,
           "batch": MODALITY_PROBE_BATCH_BRIEF},
    inputs={"segment": "Inputs: {n} images (sites) of mixed cell-type/perturbation Cell Painting wells on "
                       "six plates; each a 1080 x 1080 two-channel uint16 TIFF (DNA, AGP) taken from the "
                       "site's five-channel image (Mito, AGP, RNA, ER, DNA).",
            "crop": "Inputs: {n} images; each a 1080 x 1080 five-channel uint16 TIFF (Mito, AGP, RNA, ER, DNA) "
                    "and its cell mask delivered by this arm's segment stage (min_area is that segmentation's "
                    "min_size). The delivered crop boxes are cut from the raw five-channel image as the "
                    "embedding stage's input.",
            "embed": "Inputs: the single-cell crops of this arm's crop stage. Calls fit and validate on "
                     "development wells only: at most {cells} cells per development site; "
                     "development_train wells train, development_val wells validate; evaluation wells never "
                     "enter a call. input is the pretrained backbone (zero_shot) on those cells, not a call. "
                     "After finish, a fresh worker embeds every retained cell with the delivered call's "
                     "frozen checkpoint (the pretrained backbone for input), within the same arm deadline.",
            "batch": "Inputs: the well profiles of this arm's embed stage (median cell embedding per site, "
                     "mean of a well's two sites), batch key plate (six plates, each its own cell type x "
                     "modality combination). Each call runs the tool on all rows with harmony_params = the "
                     "tool's DEFAULT_CONFIG overridden by your parameters. input is the uncorrected profiles, "
                     "not a call. Batch calibration rule (shared by all arms): correction runs only when the "
                     "tool's batch-effect detector reports a batch effect on all rows and at least one "
                     "calibration reference group occurs on two or more plates. Here the rule's decision is "
                     "{rule}."},
    input_channels={"segment": "DNA, AGP", "crop": "Mito, AGP, RNA, ER, DNA"},
    rules={"RUN": "RUN (correction admitted)", "SKIP": "SKIP (no batch effect detected on all rows)",
           "ABSTAIN": "ABSTAIN (no calibration reference group occurs on two or more plates)"},
    withheld="Per-well treatment identity (compound/guide/construct) and private labels are not available; "
             "only each image's (cell type, perturbation) combination is known.", usages=CPJUMP1_USAGES,
    allow_empty_masks=True, allowed_sites=(1,))

# ModalityProbe2 (MODALITY_PROBE2_PLAN.md): same 6 groups and 600-image scale as MODALITY_PROBE, but 2
# plates per group (12 total) at each modality's LONGEST available timepoint, 50 shared well positions per
# plate -- every well has a guaranteed cross-plate identity twin (private/labels.json's pair_plate/pair_well)
# instead of probe1's identity-blind random sample. Reuses MODALITY_PROBE's loader/scorer modules unmodified
# (both are prep-path-generic, not hardcoded to probe1's directory); only the Dataset name, image count
# source description and plate count in the prompts differ. The retrieval scorer upgrade that actually uses
# pair_well (plan step 7) is not written yet -- until then this dataset's retrieval_metrics will just report
# "nothing eligible" (no gene/compound_target fetched for probe2 yet either; a later, separate step).
MODALITY_PROBE2 = Dataset(
    name="ModalityProbe2", cli="benchmarks.full_CPJUMP1.harness.cli",
    data="benchmarks.full_CPJUMP1.harness.modality_probe_data",
    score="benchmarks.full_CPJUMP1.harness.modality_probe_score",
    n_images=600, marker="MODALITY_PROBE2_COMPLETE", image_shape=(1080, 1080), channels=CHANNELS, dtype="uint16",
    plate_type="Mixed", mask_label=False,
    tasks={"segment": MODALITY_PROBE_SEGMENT_REQUEST, "crop": CROP_BRIEF, "embed": MODALITY_PROBE_EMBED_BRIEF,
           "batch": MODALITY_PROBE_BATCH_BRIEF},
    inputs={"segment": "Inputs: {n} images (sites) of mixed cell-type/perturbation Cell Painting wells on "
                       "twelve plates (2 per cell-type/modality combination, each modality's longest "
                       "available timepoint); each a 1080 x 1080 two-channel uint16 TIFF (DNA, AGP) taken "
                       "from the site's five-channel image (Mito, AGP, RNA, ER, DNA).",
            "crop": "Inputs: {n} images; each a 1080 x 1080 five-channel uint16 TIFF (Mito, AGP, RNA, ER, DNA) "
                    "and its cell mask delivered by this arm's segment stage (min_area is that segmentation's "
                    "min_size). The delivered crop boxes are cut from the raw five-channel image as the "
                    "embedding stage's input.",
            "embed": "Inputs: the single-cell crops of this arm's crop stage. Calls fit and validate on "
                     "development wells only: at most {cells} cells per development site; "
                     "development_train wells train, development_val wells validate; evaluation wells never "
                     "enter a call. input is the pretrained backbone (zero_shot) on those cells, not a call. "
                     "After finish, a fresh worker embeds every retained cell with the delivered call's "
                     "frozen checkpoint (the pretrained backbone for input), within the same arm deadline.",
            "batch": "Inputs: the well profiles of this arm's embed stage (median cell embedding per site, "
                     "mean of a well's two sites), batch key plate (twelve plates, two per cell type x "
                     "modality combination). Each call runs the tool on all rows with harmony_params = the "
                     "tool's DEFAULT_CONFIG overridden by your parameters. input is the uncorrected profiles, "
                     "not a call. Batch calibration rule (shared by all arms): correction runs only when the "
                     "tool's batch-effect detector reports a batch effect on all rows and at least one "
                     "calibration reference group occurs on two or more plates. Here the rule's decision is "
                     "{rule}."},
    input_channels={"segment": "DNA, AGP", "crop": "Mito, AGP, RNA, ER, DNA"},
    rules={"RUN": "RUN (correction admitted)", "SKIP": "SKIP (no batch effect detected on all rows)",
           "ABSTAIN": "ABSTAIN (no calibration reference group occurs on two or more plates)"},
    withheld="Per-well treatment identity (compound/guide/construct) and private labels are not available; "
             "only each image's (cell type, perturbation) combination is known.", usages=CPJUMP1_USAGES,
    allow_empty_masks=True, allowed_sites=(1,))

# ModalityProbe3: same 6 groups, same 12 plates as probe2, but 25 shared well positions (half of probe2's
# 50) with 2 REAL sites per (plate, well) instead of 1 -- same plate, same well, same imaging session, two
# different fields of view. This is deliberately the SAME structure original CPJUMP1 already has (2
# sites/well), so SHAPE's own SSL validation (sibling_site_map / tier_of's "replicate" tier,
# shape/tools/cell_state_analyzer/quant.py) works natively here: no SHAPE_FORCE_SSL_RUN override, no code
# change to that shared infrastructure. allowed_sites=(2,) (CPJUMP1's own default) replaces probe1/probe2's
# (1,) override. Cross-plate identity twins (pair_plate/pair_well) are preserved at the (plate, well) level.
MODALITY_PROBE3 = Dataset(
    name="ModalityProbe3", cli="benchmarks.full_CPJUMP1.harness.cli",
    data="benchmarks.full_CPJUMP1.harness.modality_probe_data",
    score="benchmarks.full_CPJUMP1.harness.modality_probe_score",
    n_images=600, marker="MODALITY_PROBE3_COMPLETE", image_shape=(1080, 1080), channels=CHANNELS, dtype="uint16",
    plate_type="Mixed", mask_label=False,
    tasks={"segment": MODALITY_PROBE_SEGMENT_REQUEST, "crop": CROP_BRIEF, "embed": MODALITY_PROBE_EMBED_BRIEF,
           "batch": MODALITY_PROBE_BATCH_BRIEF},
    inputs={"segment": "Inputs: {n} images (sites) of mixed cell-type/perturbation Cell Painting wells on "
                       "twelve plates (2 per cell-type/modality combination, each modality's longest "
                       "available timepoint, 2 sites per well); each a 1080 x 1080 two-channel uint16 TIFF "
                       "(DNA, AGP) taken from the site's five-channel image (Mito, AGP, RNA, ER, DNA).",
            "crop": "Inputs: {n} images; each a 1080 x 1080 five-channel uint16 TIFF (Mito, AGP, RNA, ER, DNA) "
                    "and its cell mask delivered by this arm's segment stage (min_area is that segmentation's "
                    "min_size). The delivered crop boxes are cut from the raw five-channel image as the "
                    "embedding stage's input.",
            "embed": "Inputs: the single-cell crops of this arm's crop stage. Calls fit and validate on "
                     "development wells only: at most {cells} cells per development site; "
                     "development_train wells train, development_val wells validate; evaluation wells never "
                     "enter a call. input is the pretrained backbone (zero_shot) on those cells, not a call. "
                     "After finish, a fresh worker embeds every retained cell with the delivered call's "
                     "frozen checkpoint (the pretrained backbone for input), within the same arm deadline.",
            "batch": "Inputs: the well profiles of this arm's embed stage (median cell embedding per site, "
                     "mean of a well's two sites), batch key plate (twelve plates, two per cell type x "
                     "modality combination). Each call runs the tool on all rows with harmony_params = the "
                     "tool's DEFAULT_CONFIG overridden by your parameters. input is the uncorrected profiles, "
                     "not a call. Batch calibration rule (shared by all arms): correction runs only when the "
                     "tool's batch-effect detector reports a batch effect on all rows and at least one "
                     "calibration reference group occurs on two or more plates. Here the rule's decision is "
                     "{rule}."},
    input_channels={"segment": "DNA, AGP", "crop": "Mito, AGP, RNA, ER, DNA"},
    rules={"RUN": "RUN (correction admitted)", "SKIP": "SKIP (no batch effect detected on all rows)",
           "ABSTAIN": "ABSTAIN (no calibration reference group occurs on two or more plates)"},
    withheld="Per-well treatment identity (compound/guide/construct) and private labels are not available; "
             "only each image's (cell type, perturbation) combination is known.", usages=CPJUMP1_USAGES,
    allow_empty_masks=True, allowed_sites=(2,))


def dataset(config: dict) -> Dataset:
    """The run's dataset: CONFIG.json's `dataset` (absent: CPJUMP1)."""
    name = config.get("dataset", "CPJUMP1")
    if name == "CPJUMP1":
        return CPJUMP1
    if name == "LIVECell":
        from benchmarks.full_LIVECell.harness.spec import LIVECELL
        return LIVECELL
    if name == "co_fc_timecourse":
        return importlib.import_module("case studies.cardiac organoids.fc_timecourse.runners.spec").for_config(config)
    if name == "ctnt_flowcytometry":
        return importlib.import_module("case studies.cardiomyocytes.ctnt_flowcytometry.runners.dataset.contract").for_config(config)
    if name == "ctnt_staining":
        return importlib.import_module("case studies.cardiomyocytes.ctnt_staining.runners.spec").for_config(config)
    if name == "ModalityProbe":
        return MODALITY_PROBE
    if name == "ModalityProbe2":
        return MODALITY_PROBE2
    if name == "ModalityProbe3":
        return MODALITY_PROBE3
    raise ValueError(f"unknown dataset {name!r}")


def raw_stack(path: Path, spec: Dataset) -> np.ndarray:
    """An image's raw CYX stack; tifffile drops the channel axis of a one-channel stack, which this restores."""
    raw = tifffile.imread(path)
    return raw[None] if raw.ndim == 2 and len(spec.channels) == 1 else raw


def check_result(result: dict, field: str) -> dict:
    if not isinstance(result, dict) or result.get("error") or result.get(field) is None:
        raise RuntimeError(f"tool did not deliver {field}: {(result or {}).get('error')}")
    return result


def check_segment_result(result: dict, spec: Dataset) -> dict:
    """check_result(result, "mask_path"), except on a dataset with allow_empty_masks: a visual-QC
    rejection caused purely by cell_count==0 (overlay rendered fine -- overlay_is_black is False, so this
    is not a broken render or a connection failure, just truly no cells) is accepted as a valid zero-cell
    delivery instead of raising. Any other failure (no mask_path at all, overlay_is_black, an exception)
    still raises -- this does not mask real tool failures, on this dataset or any other."""
    overlay_qc = (result or {}).get("overlay_qc") or {}
    if (spec.allow_empty_masks and isinstance(result, dict) and result.get("error")
            and result.get("mask_path") and result.get("cell_count") == 0
            and not overlay_qc.get("overlay_is_black")):
        return result
    return check_result(result, "mask_path")


def assert_search_clean(record: dict) -> dict | None:
    """Raise for a search that cannot be delivered: a rule fallback round, a failed or unstarted search, or a
    record without its health counters. Basic LLM failures (no engine, auth, model, connection) already raise in
    the engine. Any other failed call (a rejected search plan, a failed repair or vision question) is not fatal:
    return its facts for the run's warnings (None: the search had no failed call)."""
    if type(record.get("fallback_rounds")) is not int or record["fallback_rounds"] != 0:
        raise RuntimeError("invalid VRR run: missing or nonzero fallback_rounds")
    if type(record.get("reasoner_failures")) is not int or not isinstance(record.get("execution_errors"), list):
        raise RuntimeError("invalid VRR run: missing reasoner_failures or execution_errors")
    if record.get("status") in ("failed", "not_started") or record.get("stop"):
        raise RuntimeError("VRR search failed")
    if (record["reasoner_failures"] or record["execution_errors"]
            or record.get("stop_reason") in ("invalid_search_plan", "execution_error")):
        return {key: record.get(key) for key in ("reasoner_failures", "execution_errors", "stop_reason")}
    return None


_warnings_lock = threading.Lock()


def record_search(run: Path, stage: str, image_id: str | None, record: dict) -> None:
    """`assert_search_clean`, then one line in the run's SEARCH_WARNINGS.jsonl for a search with failed calls."""
    warning = assert_search_clean(record)
    if warning is None:
        return
    with _warnings_lock, open(run / "SEARCH_WARNINGS.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "image_id": image_id, **warning}, default=str) + "\n")


def search_record(verifier, resolution) -> dict:
    record = {"admission": verifier.admission, "stop_reason": verifier.stop_reason,
              "reasoner_failures": verifier.reasoner_failures, "fallback_rounds": verifier.fallback_rounds,
              "execution_errors": verifier.execution_errors, "selected_params": resolution.params,
              "resolution": {"status": resolution.status, "reason": resolution.reason, "fault": resolution.fault},
              "details": resolution.details}
    return record


# The reasoner settings of each stage (user decision 2026-10-13), the one place they are set: the SHAPE arm's
# reasoning effort (SHAPE_REASONING_EFFORT, read by shape/engine/openai.py per request), its per-reply token cap
# (max_tokens, reasoning included; `CappedEngine`) and request timeout (SHAPE_LLM_TIMEOUT), and the Codex arm's
# model_reasoning_effort (None: not written, the CLI's own default). embed and batch run high, both with
# representation_RxRx1 A20's cap of 16000 tokens (batch too, not batch_correction_JUMP's effort-sweep cap of 32000:
# user decision 2026-10-13); segment and crop keep the engine's defaults (medium, 4000, 120 s) and Codex runs them at
# medium too, set explicitly rather than left to the CLI's default, so both reasoners run every stage at the same
# effort. The table applies to REASONING_ARMS; the other
# agent arms (claude, biomni) keep their backends' own settings until a round adds them here (with a column).
REASONING = {
    "segment": {"effort": "medium", "max_tokens": 4000, "timeout_seconds": 120, "codex_effort": "medium"},
    "crop": {"effort": "medium", "max_tokens": 4000, "timeout_seconds": 120, "codex_effort": "medium"},
    "embed": {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900, "codex_effort": "high"},
    "batch": {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900, "codex_effort": "high"},
}
REASONING_ARMS = ("shape", "codex")
# The SHAPE arm's SSL expert start (user decisions 2026-10-13), the verifier's `expert_config` (one entry per round;
# SHAPE only), two rounds of progressive unfreezing: round 1 is representation_RxRx1's A21 `EXPERT_CONFIG` training
# the last third of the backbone (trainable_pct 33); round 2 resumes a round-1 call (init_from parent) and deepens to
# trainable_pct 66, no other lever change; from round 3 the loop is the agent's. On LIVECell `plate` is the
# culture-time stage.
SSL_EXPERT_CONFIG = ({"view_pairing": "same_well", "contrastive_mask": "same_well", "batch_sampling": "per_plate",
                      "intensity_norm": "per_plate", "augmentation": "light", "trainable_pct": 33},
                     {"init_from": "parent", "trainable_pct": 66})
from shape.agent.arms.protocol import EFFORTS       # the reasoning efforts, now shared by the arm backends


def reasoning_protocol() -> dict:
    """The reasoning protocol a run records (CONFIG.json `reasoning.protocol`); compare refuses arms that differ."""
    return {"table": {stage: dict(entry) for stage, entry in REASONING.items()}, "arms": list(REASONING_ARMS)}


def stage_reasoning(arm: str, stage: str, protocol: dict | None = None) -> dict | None:
    """The reasoner settings `arm` runs `stage` with under `protocol` (default: this code's). None: the table does
    not apply (an arm outside its arms, a stage without a reasoner such as cluster)."""
    protocol = protocol or reasoning_protocol()
    entry = protocol["table"].get(stage)
    if arm not in protocol["arms"] or entry is None:
        return None
    if arm == "shape":
        return {"effort": entry["effort"], "max_tokens": entry["max_tokens"],
                "timeout_seconds": entry["timeout_seconds"]}
    if arm == "codex":
        return {"model_reasoning_effort": entry["codex_effort"]}
    raise ValueError(f"the reasoning table has no column for the {arm} arm")


def reasoning_record(arm: str, protocol: dict | None = None) -> dict:
    """CONFIG.json's `reasoning`: the protocol and the settings each of this arm's stages runs with."""
    protocol = protocol or reasoning_protocol()
    return {"protocol": protocol, "stages": {stage: stage_reasoning(arm, stage, protocol) for stage in STAGES}}


def stage_env(arm: str, stage: str, protocol: dict | None = None) -> dict:
    """The environment a stage worker of `arm` gets for its reasoner (empty: none applies)."""
    settings = stage_reasoning(arm, stage, protocol)
    if arm != "shape" or settings is None:
        return {}
    return {"SHAPE_REASONING_EFFORT": settings["effort"], "SHAPE_LLM_TIMEOUT": str(settings["timeout_seconds"])}


def check_stage_env(arm: str, stage: str, protocol: dict | None = None) -> dict | None:
    """The stage's reasoner settings, after checking this process's environment carries them (a stage worker
    started without its stage's environment raises)."""
    settings = stage_reasoning(arm, stage, protocol)
    wrong = {k: os.environ.get(k) for k, v in stage_env(arm, stage, protocol).items() if os.environ.get(k) != v}
    if wrong:
        raise RuntimeError(f"stage {stage} runs without its reasoning environment {stage_env(arm, stage, protocol)}: {wrong}")
    return settings


class CappedEngine:
    """A reasoner engine whose every generate call carries the stage's per-reply token cap (reasoning included)
    unless the caller sets one (representation_RxRx1's `Budgeted`)."""

    def __init__(self, engine, max_tokens: int):
        self.engine, self.max_tokens = engine, int(max_tokens)

    def generate(self, content, *args, **kwargs):
        kwargs.setdefault("max_tokens", self.max_tokens)
        return self.engine.generate(content, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.engine, name)


def engine_for(config: dict, stage: str):
    """The SHAPE arm's reasoner for `stage`: the vision engine under the stage's REASONING settings (the effort and
    timeout from this process's environment, which `check_stage_env` checks; the cap by `CappedEngine`); None for
    any other arm."""
    if config["arm"] != "shape":
        return None
    from benchmarks.full_HCMA.harness.shape_arm import build_engine
    from shape.agent.verification.vision_engine import build_vision_engine
    settings = check_stage_env(config["arm"], stage, config.get("reasoning", {}).get("protocol"))
    raw = build_vision_engine(config["model"]) if config.get("model") else None
    engine = build_engine(config["model"], engine=None if raw is None else CappedEngine(raw, settings["max_tokens"]))
    print(f"REASONING stage={stage} effort={settings['effort']} max_tokens={settings['max_tokens']} "
          f"timeout_seconds={settings['timeout_seconds']}", flush=True)
    return engine


def remaining(config: dict) -> float:
    if config["deadline"] is None:
        return float("inf")
    seconds = config["deadline"] - time.time()
    if seconds <= 0:
        raise TimeoutError("arm budget exhausted")
    return seconds


# The share of the arm's remaining time the representation search may use; the rest is kept for frozen inference.
SSL_SEARCH_SHARE = 0.65


def ssl_search_deadline(config: dict) -> float | None:
    """The representation search's deadline (its RepresentationTask deadline), taken when the stage starts."""
    return None if config["deadline"] is None else time.time() + remaining(config) * SSL_SEARCH_SHARE


def ssl_epoch_budget(images: list[dict], train_cells_per_site: int, usages: tuple = CPJUMP1_USAGES) -> dict:
    """The fixed epoch count of every arm's SSL training call (user decision 2026-10-12, as representation_RxRx1's
    epoch_budget): the analyzer's own automatic rule (`resolve_training_config`) on the arm-independent
    train cap, train_cells_per_site x the train sites (`usages[0]`) of the run's images. It is not a
    lever: SHAPE's verifier drops it (`FixedEpochVerifier`), the Codex arm's parameter table leaves it out, and every
    training call gets it."""
    from shape.utils.adaptive_training import resolve_training_config

    train = usages[0]
    sites = sum(1 for r in images if r["usage"] == train)
    n_train = int(train_cells_per_site) * sites
    if n_train < 1:
        raise ValueError(f"no {train} site: the SSL epoch budget is undefined")
    return {"policy": "cpjump1-fixed-v1", "rule": "clamp(round(8 + 6 * sqrt(128 / N_train)), 8, 20)",
            "n_train": n_train, "train_cells_per_site": int(train_cells_per_site), f"{train}_sites": sites,
            "max_epochs": resolve_training_config(n_train, 1)["max_epochs"]}


def crop_workers(config: dict, n_images: int) -> tuple[int, int]:
    """(images cropped at a time, cores each image's crop VRR gets). SHAPE uses
    shape_policy['crop_vrr_workers'] (its per-image crop VRR shares cores + the LLM). The
    control/native arms have NO per-image LLM, so they fan out up to SEGMENT_WORKERS_CAP (the
    same import-storm-bounded cap the segment stage uses) -- otherwise crop_vrr_workers defaults
    to 1 and crops run serially at ~7/min."""
    policy = config.get("shape_policy") or {}
    # control has no per-image crop LLM -> fan out up to the cap; SHAPE keeps its policy's
    # crop_vrr_workers (default 1: its crop VRR shares cores + the LLM per image).
    default_workers = SEGMENT_WORKERS_CAP if config.get("arm") == "control" else 1
    workers = min(max(1, int(policy.get("crop_vrr_workers", default_workers))),
                  max(1, int(config["cores"])), n_images or 1)
    return workers, max(1, int(config["cores"]) // workers)


def shape_search_policy(config: dict, n_images: int) -> tuple[int, int | None]:
    policy = config.get("smoke_policy") or {}
    images = int(policy.get("shape_search_images", n_images))
    rounds = policy.get("shape_search_rounds")
    if not 0 <= images <= n_images:
        raise ValueError("shape search image limit is outside this stage")
    if rounds is not None and int(rounds) < 2:
        raise ValueError("SHAPE smoke search needs at least two rounds")
    return images, None if rounds is None else int(rounds)


def _parallel(items, fn, workers: int):
    """fn over items, results in item order; a failure cancels the items not started yet."""
    from shape.utils.concurrency import ordered_map
    return ordered_map(fn, items, workers)


_IN_WORKER = False      # True in a worker process of `_in_processes`
_WORKER_ENGINE: dict = {}
# Cap on spawned segment workers (control/native path): each cold-imports shape (~10s), so too
# many simultaneous imports thrash the box. 8 processes keep the remote SAM pool busy while
# bounding the import storm; override with $FULL_CPJUMP1_SEGMENT_WORKERS.
SEGMENT_WORKERS_CAP = int(os.environ.get("FULL_CPJUMP1_SEGMENT_WORKERS", "8"))


def _worker_start(cores: int) -> None:
    """A worker process's share of the cores for BLAS/OpenMP (libraries loaded later read the environment)."""
    global _IN_WORKER, _WORKER_LIMITS
    _IN_WORKER = True
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(cores)
    from threadpoolctl import threadpool_limits
    _WORKER_LIMITS = threadpool_limits(limits=cores)


def _in_processes(fn, items, workers: int, cores: int, *, threads: bool = False):
    """fn over items, `workers` at a time, each in a spawned worker process with `cores` cores (separate file
    descriptor tables and no shared GIL: a library's double close in one image's work cannot close another's
    files); results in item order, a failure cancels the items not started yet. One worker runs in this process.
    threads=True runs the same in threads of this process: only for tests whose stubs live in this process."""
    if threads or workers <= 1:
        return _parallel(items, fn, workers)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_worker_start, initargs=(cores,))
    try:
        return list(pool.map(fn, items))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _require_segmenter_gpu(tool) -> None:
    """Cellpose-SAM must run on a GPU -- EXCEPT in shared-server mode, where this process
    delegates every eval to a remote GPU-resident SAM server ($SHAPE_SAM_SERVER_URL) and is
    deliberately CPU-only (see Cell_Segmenter_Tool's shared-server branch). Then the GPU lives
    on the server, so a CPU device here is correct, not a silent fallback. This lets the GPU
    stage run on a CPU node that points at a remote SAM service (the cell_state_HCMA pattern)."""
    if getattr(tool, "_sam_server_url", None):
        return
    if tool.device.type != "cuda":
        raise RuntimeError("Cellpose-SAM fell back to CPU; this GPU benchmark must not continue "
                           "silently (set SHAPE_SAM_SERVER_URL to delegate to a remote SAM service)")


SEGMENTERS = {"cell_segmenter": ("Cell_Segmenter_Tool", {"model_type": "cpsam", "bsize": 256}),
              "organoid_segmenter": ("Organoid_Segmenter_Tool", {"adaptive_rescue": False})}


def segment_registry(spec: Dataset) -> dict:
    """{lever: LeverInfo} of the dataset's segmenter: the template registry for Cellpose-SAM, the organoid CARD's
    `## Levers` (parsed the same way) for the organoid segmenter."""
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    C = tools_common()
    repo = Path(__file__).resolve().parents[3]
    if spec.segment_tool == "cell_segmenter":
        return C.tool_levers("segment", repo)
    from shape.utils.schemas import card_lever_fields, lever_spec
    fields = card_lever_fields(str(repo / "shape" / "tools" / spec.segment_tool), "## Levers")
    if not fields:
        raise RuntimeError(f"no levers under '## Levers' in the {spec.segment_tool} CARD")
    return {name: C.LeverInfo(name, *lever_spec(fields_)) for name, fields_ in fields.items()}


def segmenter(spec: Dataset, tool_params: dict | None = None) -> dict:
    """The dataset's segmenter on the GPU: {tool, name, fixed (the kwargs no arm sets), registry, defaults
    (the CARD defaults, then `tool_params`)}. Raises when the tool fell back to the CPU."""
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    if spec.segment_tool not in SEGMENTERS:
        raise ValueError(f"unknown segment tool {spec.segment_tool!r}")
    name, fixed = SEGMENTERS[spec.segment_tool]
    if spec.segment_tool == "cell_segmenter":
        from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool as Tool
    else:
        from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool as Tool
    C = tools_common()
    registry = segment_registry(spec)
    defaults = {k: C.normal_default(v) for k, v in registry.items()}
    defaults.update(tool_params or {})
    tool = Tool()
    tool.allow_empty_masks = spec.allow_empty_masks
    if spec.segment_tool == "cell_segmenter":
        _require_segmenter_gpu(tool)             # v2.0: shared-server mode (SHAPE_SAM_SERVER_URL) keeps this process CPU-only
    elif tool.device.type != "cuda":
        raise RuntimeError(f"{name} fell back to CPU; this GPU benchmark must not continue silently")
    return {"tool": tool, "name": name, "fixed": dict(fixed), "registry": registry, "defaults": defaults}


def _segmentation_per_image(prep: Path, run: Path, config: dict, images: list[dict], engine) -> None:
    from benchmarks.full_HCMA.harness.shape_arm import segment_one, tools_common

    spec = dataset(config)
    C = tools_common()
    seg = segmenter(spec, config.get("tool_params", {}))
    tool, registry, defaults = seg["tool"], seg["registry"], seg["defaults"]
    adaptive = config.get('segmentation_adaptive')
    verifier_options = {}
    if adaptive is not None:
        if engine is None:
            raise ValueError('adaptive segmentation requires a SHAPE reasoner')
        from shape.tools.cell_segmenter.cache_runtime import configure
        identity = configure(tool, run/'segment'/'native_trials', seed=adaptive['native_seed'])
        write_json(run/'segment_NATIVE_CACHE.json', identity)
        verifier_options['gain_policy'] = adaptive['gain_policy']
    model_lock = threading.Lock()
    outputs = []
    search_images, search_rounds = shape_search_policy(config, len(images))

    def _one(indexed_row):
        image_index, row = indexed_row
        remaining(config)
        ident = row["image_id"]
        folder = run / "segment" / ident
        folder.mkdir(parents=True, exist_ok=True)
        image = within(prep / "public", row["segment_file"])
        initial = tool.execute(image=str(image), image_id=ident, **seg["fixed"],
                               query_cache_dir=str(folder / "defaults"), **defaults)
        qc_error = None
        if spec.segment_tool != "cell_segmenter" and isinstance(initial, dict) and initial.get("mask_path") and initial.get("error"):
            # The organoid segmenter reports a failed visual QC (zero organoids, implausible coverage) as `error`
            # while still delivering its mask. The call at the defaults keeps that mask as its delivery, with the
            # complaint recorded (the baseline convention of the cardiac organoid study: a baseline mask remains
            # evaluable); SHAPE's VRR sees the complaint in S001 and searches from it.
            qc_error = str(initial["error"])
        else:
            initial = check_segment_result(initial, spec)
        write_json(folder / "defaults.json", initial)
        mask = np.load(initial["mask_path"])
        target = folder / f"cell_mask_{ident}.tif"
        tifffile.imwrite(target, mask.astype(np.uint32))
        params = dict(defaults)
        if engine is not None and image_index < search_images:
            item = {"image_id": ident, "path": str(image), "s001_record": initial,
                    "s001_mask": str(target), "s001_overlay": initial.get("overlay_path"),
                    "request": spec.tasks["segment"]}
            record = segment_one(C, tool, item, list(registry), defaults, engine, folder / "search",
                                 run / "logs" / "segment", model_lock, budget=search_rounds,
                                 tool_name=seg["name"], fixed=seg["fixed"],
                                 **({'verifier_options': verifier_options} if adaptive is not None else {}))
            write_json(folder / "SEARCH.json", record)
            record_search(run, "segment", ident, record)
            params.update(record["levers"])
            if record["levers"]:
                if not record.get("mask_npy"):
                    raise RuntimeError("selected segmentation has no mask")
                mask = np.load(record["mask_npy"])
                tifffile.imwrite(target, mask.astype(np.uint32))
        if mask.shape != spec.image_shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"invalid segmentation mask: {ident}")
        n_cells = int(np.count_nonzero(np.unique(mask) > 0))
        if not n_cells and not spec.allow_empty_masks:
            raise RuntimeError(f"empty segmentation: {ident}; recorded failure, not silently omitted")
        return {"image_id": ident, "mask": str(target.resolve()), "sha256": digest(target),
                "min_area": params["min_size"], "n_cells": n_cells,
                **({"defaults_qc_error": qc_error} if qc_error else {})}

    # The organoid segmenter (any arm) also stays serial: one model object, one call at a time.
    # The SHAPE search path (engine set) stays serial: segment_one shares model_lock + records
    # search order. The control/native path (engine is None) has no per-image LLM and no
    # cross-image state, so it fans out over SPAWNED PROCESSES (_in_processes, the same mechanism
    # the crop stage uses) -- each worker has its own GIL + file descriptors, so the remote-SAM
    # client cost (pickling the ~30 MB image/mask, np.load, tifffile.imwrite) overlaps instead of
    # contending (a ThreadPool here was measured SLOWER; SAM_SERVICE.md: "scale processes, not
    # threads"). _in_processes preserves item order, so segmentation.json is byte-identical.
    if engine is None and spec.segment_tool == "cell_segmenter":
        # Spawned workers each cold-import shape (~10s: torch/cellpose/scanpy). With one worker
        # per core their simultaneous imports thrash CPU/disk and stall; cap the pool at
        # SEGMENT_WORKERS_CAP so imports are paid a bounded number of times and then AMORTISED as
        # each worker reuses its process across many images (the remote SAM pool, not local CPU,
        # is the real concurrency). Cores are split across the pool for each worker's BLAS.
        workers = max(1, min(int(config["cores"]), SEGMENT_WORKERS_CAP, len(images)))
        cores = max(1, int(config["cores"]) // workers)
        tasks = [(prep, run, config, row) for row in images]
        outputs = _in_processes(_segment_image_worker, tasks, workers, cores)
        for i, o in enumerate(outputs, 1):
            print(f"SEGMENT {i}/{len(images)} {o['image_id']} cells={o['n_cells']}", flush=True)
    else:
        for image_index, row in enumerate(images):
            outputs.append(_one((image_index, row)))
            print(f"SEGMENT {len(outputs)}/{len(images)} {outputs[-1]['image_id']} cells={outputs[-1]['n_cells']}", flush=True)
    write_json(run / "segmentation.json", outputs)


def _segment_image_worker(task) -> dict:
    """One image's default-parameter segmentation (the control/native path: no LLM search),
    top-level + picklable so a spawned worker process can run it. Reconstructs the tool in the
    worker (like _crop_image) and delegates to the remote SAM service via $SHAPE_SAM_SERVER_URL,
    which the spawned process inherits from the parent's environment."""
    prep, run, config, row = task
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    spec = dataset(config)
    C = tools_common()
    registry = C.tool_levers("segment", Path(__file__).resolve().parents[3])
    defaults = {k: C.normal_default(v) for k, v in registry.items()}
    defaults.update(config.get("tool_params", {}))
    tool = Cell_Segmenter_Tool()
    _require_segmenter_gpu(tool)
    tool.allow_empty_masks = spec.allow_empty_masks
    ident = row["image_id"]
    folder = run / "segment" / ident
    folder.mkdir(parents=True, exist_ok=True)
    image = within(prep / "public", row["segment_file"])
    initial = check_segment_result(tool.execute(image=str(image), image_id=ident, model_type="cpsam",
                           bsize=256, query_cache_dir=str(folder / "defaults"), **defaults), spec)
    write_json(folder / "defaults.json", initial)
    mask = np.load(initial["mask_path"])
    target = folder / f"cell_mask_{ident}.tif"
    tifffile.imwrite(target, mask.astype(np.uint32))
    if mask.shape != spec.image_shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError(f"invalid segmentation mask: {ident}")
    n_cells = int(np.count_nonzero(np.unique(mask) > 0))
    if not n_cells and not spec.allow_empty_masks:
        raise RuntimeError(f"empty segmentation: {ident}; recorded failure, not silently omitted")
    return {"image_id": ident, "mask": str(target.resolve()), "sha256": digest(target),
            "min_area": defaults["min_size"], "n_cells": n_cells}


COHORT_RECORD = "segment_COHORT.json"


def _profile_reader(engine, env: str):
    """Preflight reader for `segmentation_cohort.run`. $<env>="<specimen_context>:<modality>" puts an image with no
    usable preflight read into that one group (a homogeneous dataset whose preflight the gateway refuses); every
    other image keeps its read."""
    from types import SimpleNamespace
    from shape.agent.verification import segmentation_cohort
    fallback = os.environ.get(env)

    def read_profile(s):
        profile = segmentation_cohort.preflight_read(s["path"], engine, s["image_id"])
        if fallback and segmentation_cohort.group_key(profile) is None:
            print(f"PREFLIGHT_FALLBACK {env} {s['image_id']} read={profile!r}", flush=True)
            specimen, modality = fallback.split(":")
            return SimpleNamespace(specimen_context=specimen, modality=modality, confidence="high")
        return profile
    return read_profile


def _segmentation_cohort(prep: Path, run: Path, config: dict, images: list[dict], engine) -> None:
    """The full SHAPE arm's segmentation: SHAPE's shared segmenter cohort (`segmentation_cohort.run`, the one
    solve() runs over a set of images): groups by each image's preflight read, a progressive calibration sample per
    group runs the per-image VRR, the consensus of the agent-tuned levers (not diameter / color_class_filter) is
    applied to the rest in the verifier's cohort mode, and only the images the admission standard flags get their
    own VRR from the consensus; groups at the same time, `config["cores"]` images at once. Each image's per-image run
    is HCMA's `segment_resolution` on its S001 (the call at the defaults, made once per image). Records: every pass
    with the image's own VRR at segment/<id>/SEARCH.json (calibration, ungrouped, small or unstable group, routed),
    every cohort-mode pass at segment/<id>/COHORT_SEARCH.json, the shared record at COHORT_RECORD, each image's
    `parameter_source` in segmentation.json."""
    if (config.get('segmentation_adaptive') or {}).get('routing_policy') == 'conditional_cohort_v2':
        return _segmentation_shared(prep, run, config, images, engine)
    from benchmarks.full_HCMA.harness.shape_arm import segment_resolution, tools_common
    from shape.agent.verification import segmentation_cohort
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool

    dataset_spec = dataset(config)
    C = tools_common()
    registry = C.tool_levers("segment", Path(__file__).resolve().parents[3])
    defaults = {k: C.normal_default(v) for k, v in registry.items()}
    defaults.update(config.get("tool_params", {}))
    tool = Cell_Segmenter_Tool()
    _require_segmenter_gpu(tool)
    # The cohort's batch phase runs `config["cores"]` images at once through run_one, each of
    # which calls tool.execute (default_call's S001 + segment_resolution's passes). With a REMOTE
    # SAM service the lock is not only unnecessary but actively the cohort bottleneck: the server
    # holds its OWN thread-safe pool of K replicas ("distinct model per thread -> safe concurrency",
    # sam_server.py) and remote-mode execute() touches no shared tool state (_segment_channel's
    # remote branch only reads the immutable url list + random.choice, writes no self.*), so a
    # client lock would re-serialize evals that the pool is built to run concurrently -- the GIL/
    # serial behaviour that made cohort slow and prone to piling up. Keep a REAL lock only for a
    # LOCAL single model (one GPU model object is not safe under concurrent eval); use a no-op
    # context when delegating to the remote pool.
    model_lock = contextlib.nullcontext() if getattr(tool, "_sam_server_url", None) else threading.Lock()
    tool.allow_empty_masks = dataset(config).allow_empty_masks
    adaptive = config.get('segmentation_adaptive')
    adaptive_options = {}
    if adaptive is not None:
        from shape.agent.verification.segmentation_adaptive import GroupingPolicy, POLICY, native_features
        from shape.agent.verification.segmentation_triage import VisualTriage
        from shape.tools.cell_segmenter.cache_runtime import configure
        if adaptive.get('routing_policy') != POLICY:
            raise ValueError('unknown adaptive segmentation routing policy')
        identity = configure(tool, run/'segment'/'native_trials', seed=adaptive['native_seed'])
        write_json(run/'segment_NATIVE_CACHE.json', identity)
        adaptive_options = dict(routing_policy=POLICY,
            grouping_policy=GroupingPolicy(**adaptive['grouping']),
            gain_policy=adaptive['gain_policy'],
            triage=VisualTriage(engine, run/'segment'/'triage', task=dataset_spec.tasks['segment'],
                model_identity=dict(requested_model=config['model'],
                    reasoning=stage_reasoning('shape', 'segment', config.get('reasoning', {}).get('protocol'))),
                batch_size=adaptive['triage_batch_size']))
    workers = max(1, int(config["cores"]))
    rows = {r["image_id"]: r for r in images}
    s001, s001_lock = {}, threading.Lock()

    def default_call(ident: str) -> tuple[dict, Path]:
        """The image's call at the defaults (S001), once per image (a routed image's second pass reuses it)."""
        with s001_lock:
            if ident in s001:
                return s001[ident]
        folder = run / "segment" / ident / "defaults"
        folder.mkdir(parents=True, exist_ok=True)
        image = within(prep / "public", rows[ident]["segment_file"])
        with model_lock:
            initial = check_result(tool.execute(image=str(image), image_id=ident, model_type="cpsam", bsize=256,
                                   query_cache_dir=str(folder), **defaults), "mask_path")
        write_json(run / "segment" / ident / "defaults.json", initial)
        target = folder / f"cell_mask_{ident}.tif"
        tifffile.imwrite(target, np.load(initial["mask_path"]).astype(np.uint32))
        with s001_lock:
            s001[ident] = (initial, target)
        return initial, target

    def run_one(spec: dict, options: dict | None) -> tuple:
        remaining(config)
        ident = spec["image_id"]
        mode = "cohort" if (options or {}).get("cohort_levers") is not None else (
            "routed" if (options or {}).get("start_params") is not None else "search")
        initial, s001_mask = default_call(ident)
        item = {"image_id": ident, "path": spec["path"], "s001_record": initial, "s001_mask": str(s001_mask),
                "s001_overlay": initial.get("overlay_path"), "request": dataset_spec.tasks["segment"]}
        folder = run / "segment" / ident
        record, res = segment_resolution(C, tool, item, list(registry), defaults, engine, folder / mode,
                                         run / "logs" / "segment" / mode, model_lock, verifier_options=options)
        record = {**record, "mode": mode, "verifier_options": options}
        write_json(folder / ("COHORT_SEARCH.json" if mode == "cohort" else "SEARCH.json"), record)
        record_search(run, "segment", ident, record)
        print(f"SEGMENT_{mode.upper()} {ident} calls={record['n_calls']} levers={record['levers']}", flush=True)
        return [res], record, res.status, res.fault

    fields = adaptive['grouping']['batch_fields'] if adaptive is not None else []
    specs = [{"image_id": r["image_id"], "path": str(within(prep / "public", r["segment_file"])),
              **{field: r[field] for field in fields}} for r in images]
    if adaptive is not None:
        def read_features(spec):
            initial, _ = default_call(spec['image_id'])
            mask = np.load(initial['mask_path'], allow_pickle=False)
            ids, areas = np.unique(mask, return_counts=True)
            areas = areas[ids > 0]
            if not len(areas):
                return {}  # explicit ungrouped individual VRR, never shared calibration
            return native_features(len(areas), float(np.median(areas)), mask.shape)
        adaptive_options['read_features'] = read_features
    runs, cohort = segmentation_cohort.run(
        specs, run_one, lambda items, fn: _parallel(items, fn, min(workers, len(items) or 1)),
        _profile_reader(engine, "FULL_CPJUMP1_SEGMENT_FALLBACK_GROUP"),
        defaults, list(registry), workers=workers, **adaptive_options)
    write_json(run / COHORT_RECORD, {**cohort, "implementation": "shape.agent.verification.segmentation_cohort.run",
                                     "workers": workers, "defaults": defaults})
    outputs = []
    for spec, (_, record, _, _) in zip(specs, runs):
        ident = spec["image_id"]
        target = run / "segment" / ident / f"cell_mask_{ident}.tif"
        if record["levers"]:
            if not record.get("mask_npy"):
                raise RuntimeError(f"selected segmentation has no mask: {ident}")
            mask = np.load(record["mask_npy"])
        else:
            mask = tifffile.imread(s001[ident][1])
        tifffile.imwrite(target, mask.astype(np.uint32))
        if mask.shape != dataset_spec.image_shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"invalid segmentation mask: {ident}")
        n_cells = int(np.count_nonzero(np.unique(mask) > 0))
        if not n_cells and not dataset(config).allow_empty_masks:
            raise RuntimeError(f"empty segmentation: {ident}; recorded failure, not silently omitted")
        params = {**defaults, **record["levers"]}
        outputs.append({"image_id": ident, "mask": str(target.resolve()), "sha256": digest(target),
                        "min_area": params["min_size"], "n_cells": n_cells,
                        "parameter_source": cohort["parameter_source"][ident]})
    write_json(run / "segmentation.json", outputs)


def _segmentation_cohort_organoid(prep: Path, run: Path, config: dict, images: list[dict], engine) -> None:
    """The organoid segmenter's cohort (config segmentation_regime "cohort"; SHAPE's `segmentation_cohort.run` with
    tool Organoid_Segmenter_Tool and its legacy routing): images are grouped by their preflight read, a calibration
    sample per group (at least MIN_CALIBRATION, groups of at least MIN_GROUP) runs the per-image VRR, the consensus of
    the tuned levers is applied to the rest in the verifier's cohort mode, and only the flagged images get their own
    VRR from the consensus. Same records and files as `_segmentation_cohort` (the Cellpose-SAM cohort, untouched), and
    `_segmentation_per_image`'s handling of the organoid segmenter's failed visual QC (the mask is kept, the complaint
    recorded as defaults_qc_error)."""
    from benchmarks.full_HCMA.harness.shape_arm import segment_resolution, tools_common
    from shape.agent.verification import segmentation_cohort

    spec = dataset(config)
    if spec.segment_tool != "organoid_segmenter" or engine is None:
        raise ValueError("the organoid segmentation cohort needs the organoid segmenter and a SHAPE reasoner")
    C = tools_common()
    seg = segmenter(spec, config.get("tool_params", {}))
    tool, registry, defaults = seg["tool"], seg["registry"], seg["defaults"]
    model_lock = threading.Lock()
    workers = max(1, int(config["cores"]))
    rows = {r["image_id"]: r for r in images}
    s001, qc_errors, s001_lock = {}, {}, threading.Lock()

    def default_call(ident: str) -> tuple[dict, Path]:
        """The image's call at the defaults (S001), once per image (a routed image's second pass reuses it)."""
        with s001_lock:
            if ident in s001:
                return s001[ident]
        folder = run / "segment" / ident / "defaults"
        folder.mkdir(parents=True, exist_ok=True)
        image = within(prep / "public", rows[ident]["segment_file"])
        with model_lock:
            initial = tool.execute(image=str(image), image_id=ident, **seg["fixed"], query_cache_dir=str(folder), **defaults)
        if isinstance(initial, dict) and initial.get("mask_path") and initial.get("error"):
            qc_errors[ident] = str(initial["error"])
        else:
            initial = check_result(initial, "mask_path")
        write_json(run / "segment" / ident / "defaults.json", initial)
        target = folder / f"cell_mask_{ident}.tif"
        tifffile.imwrite(target, np.load(initial["mask_path"]).astype(np.uint32))
        with s001_lock:
            s001[ident] = (initial, target)
        return initial, target

    def run_one(spec_: dict, options: dict | None) -> tuple:
        remaining(config)
        ident = spec_["image_id"]
        mode = "cohort" if (options or {}).get("cohort_levers") is not None else (
            "routed" if (options or {}).get("start_params") is not None else "search")
        initial, s001_mask = default_call(ident)
        item = {"image_id": ident, "path": spec_["path"], "s001_record": initial, "s001_mask": str(s001_mask),
                "s001_overlay": initial.get("overlay_path"), "request": spec.tasks["segment"]}
        folder = run / "segment" / ident
        record, res = segment_resolution(C, tool, item, list(registry), defaults, engine, folder / mode,
                                         run / "logs" / "segment" / mode, model_lock, verifier_options=options,
                                         tool_name=seg["name"], fixed=seg["fixed"])
        record = {**record, "mode": mode, "verifier_options": options}
        write_json(folder / ("COHORT_SEARCH.json" if mode == "cohort" else "SEARCH.json"), record)
        record_search(run, "segment", ident, record)
        print(f"SEGMENT_{mode.upper()} {ident} calls={record['n_calls']} levers={record['levers']}", flush=True)
        return [res], record, res.status, res.fault

    specs = [{"image_id": r["image_id"], "path": str(within(prep / "public", r["segment_file"]))} for r in images]
    runs, cohort = segmentation_cohort.run(
        specs, run_one, lambda items, fn: _parallel(items, fn, min(workers, len(items) or 1)),
        lambda spec_: segmentation_cohort.preflight_read(spec_["path"], engine, spec_["image_id"]),
        defaults, list(registry), workers=workers, tool_name=seg["name"])
    write_json(run / COHORT_RECORD, {**cohort, "implementation": "shape.agent.verification.segmentation_cohort.run",
                                     "workers": workers, "defaults": defaults})
    outputs = []
    for spec_, (_, record, _, _) in zip(specs, runs):
        ident = spec_["image_id"]
        target = run / "segment" / ident / f"cell_mask_{ident}.tif"
        if record["levers"]:
            if not record.get("mask_npy"):
                raise RuntimeError(f"selected segmentation has no mask: {ident}")
            mask = np.load(record["mask_npy"])
        else:
            mask = tifffile.imread(s001[ident][1])
        tifffile.imwrite(target, mask.astype(np.uint32))
        if mask.shape != spec.image_shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"invalid segmentation mask: {ident}")
        n_cells = int(np.count_nonzero(np.unique(mask) > 0))
        if not n_cells and not spec.allow_empty_masks:
            raise RuntimeError(f"empty segmentation: {ident}; recorded failure, not silently omitted")
        params = {**defaults, **record["levers"]}
        outputs.append({"image_id": ident, "mask": str(target.resolve()), "sha256": digest(target),
                        "min_area": params["min_size"], "n_cells": n_cells,
                        "parameter_source": cohort["parameter_source"][ident],
                        **({"defaults_qc_error": qc_errors[ident]} if ident in qc_errors else {})})
    write_json(run / "segmentation.json", outputs)


def _segmentation_shared(prep, run, config, images, engine):
    from shape.agent.verification.segmentation_shared_native import run_native
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    from shape.tools.cell_segmenter.cache_runtime import configure
    from benchmarks.full_HCMA.harness.shape_arm import tools_common
    C = tools_common(); registry = C.tool_levers('segment', Path(__file__).resolve().parents[3])
    defaults = {k:C.normal_default(v) for k,v in registry.items()}
    defaults.update(config.get('tool_params',{}))
    tool = Cell_Segmenter_Tool(); tool.allow_empty_masks = dataset(config).allow_empty_masks
    if tool.device.type != 'cuda': raise RuntimeError('shared segmentation requires CUDA')
    policy = config['segmentation_adaptive']
    cache = configure(tool,run/'segment'/'native_trials',seed=policy['native_seed'])
    write_json(run/'segment_NATIVE_CACHE.json',cache)
    items = [dict(image_id=r['image_id'],path=str(within(prep/'public',r['segment_file']))) for r in images]
    selected, record = run_native(items,tool,defaults,engine,run/'segment'/'shared_vrr',
        task=dataset(config).tasks['segment'],model_identity=dict(model=config['model'],
            reasoning=config.get('reasoning'),native_cache_identity=cache),
        workers=config['cores'],max_rounds=policy.get('max_rounds',8))
    write_json(run/COHORT_RECORD,record)
    outputs=[]
    for row in images:
        ident=row['image_id']; observation=selected.get(ident)
        if observation is None: continue
        native=observation['result']; mask=np.load(native['mask_path'],allow_pickle=False)
        if mask.shape!=dataset(config).image_shape or not np.issubdtype(mask.dtype,np.integer) or np.any(mask<0):
            raise ValueError('invalid selected native mask: '+ident)
        if digest(Path(native['mask_path']))!=observation['mask_sha256']:
            raise ValueError('selected native mask changed: '+ident)
        target=run/'segment'/ident/f'cell_mask_{ident}.tif';target.parent.mkdir(parents=True,exist_ok=True)
        tifffile.imwrite(target,mask.astype(np.uint32))
        outputs.append(dict(image_id=ident,mask=str(target.resolve()),sha256=digest(target),
            min_area=observation['params']['min_size'],n_cells=int(np.count_nonzero(np.unique(mask)>0)),
            parameter_source='conditional_cohort_v2',selected_candidate=observation['candidate'],
            selected_parameters=observation['params'],residual_problems=observation['problems'],
            qc_pass=observation['guard_ok'] and not observation['problems']))
    write_json(run/'segmentation.json',outputs)
    if len(outputs)!=len(images):
        raise RuntimeError('shared segmentation preserved partial outputs; unresolved native failures: '+
                           ','.join(sorted(set(r['image_id'] for r in images)-set(selected))))


def segmentation(prep: Path, run: Path, config: dict, images: list[dict]) -> None:
    """An arm without a SHAPE engine (control, Codex's native path) and a SHAPE smoke: the per-image path; a full
    SHAPE run: the shared segmenter cohort (`_segmentation_cohort`)."""
    engine = engine_for(config, "segment")
    if (engine is not None and config.get("segmentation_regime") == "cohort"
            and dataset(config).segment_tool == "organoid_segmenter"):
        return _segmentation_cohort_organoid(prep, run, config, images, engine)
    if dataset(config).segment_tool != "cell_segmenter":
        # The cohort and shared paths are Cellpose-SAM's (its native cache, grouping and triage); another segmenter
        # runs the per-image VRR for every image.
        return _segmentation_per_image(prep, run, config, images, engine)
    if engine is not None and (config.get('segmentation_adaptive') or {}).get('routing_policy')=='conditional_cohort_v2':
        return _segmentation_shared(prep,run,config,images,engine)
    if engine is None or config.get("smoke_policy"):
        return _segmentation_per_image(prep, run, config, images, engine)
    return _segmentation_cohort(prep, run, config, images, engine)


def _crop_vrr_options(config: dict) -> dict:
    """The crop VRR options a run sets: `crop_mask_check: false` switches the crop VRR's mask-error check off."""
    return {"mask_check": False} if config.get("crop_mask_check") is False else {}


def _crop_image(task) -> tuple[list, list, dict | None, dict | None]:
    """One image's crop work (its crop VRR when searched, the delivery call, the box and rejection checks, the
    retained crops); top-level so a worker process can run it. Returns (kept, excluded, search record, the worker
    process's token usage or None in this process)."""
    from benchmarks.full_HCMA.harness.shape_arm import crop_one, tools_common
    from shape.engine import usage
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    image_index, row, prep, run, config, mask, search_images, search_rounds, cores = task
    if _IN_WORKER:
        usage.reset()
    remaining(config)
    spec = dataset(config)
    ident = row["image_id"]
    folder = run / "crop" / ident
    folder.mkdir(parents=True, exist_ok=True)
    if digest(Path(mask["mask"])) != mask["sha256"]:
        raise ValueError(f"segmentation mask changed before cropping: {ident}")
    image = within(prep / "public", row["file"])
    params, rejected, expected_boxes, record = dict(config.get("tool_params", {})), set(), {}, None
    if config["arm"] == "shape" and image_index < search_images and not (spec.allow_empty_masks and mask.get("n_cells") == 0):
        if "engine" not in _WORKER_ENGINE:                       # once per process
            _WORKER_ENGINE["engine"] = engine_for(config, "crop")
        C = tools_common()
        registry = C.tool_levers("crop", Path(__file__).resolve().parents[3])
        defaults = {k: C.normal_default(v) for k, v in registry.items() if v.kind != "per_cell"}
        record = crop_one(C, Single_Cell_Cropper_Tool,
            {"image_id": ident, "path": str(image), "mask": mask["mask"], "min_area": mask["min_area"],
             "defaults": defaults, "brief": spec.tasks["crop"]},
            _WORKER_ENGINE["engine"], config["deadline"], cores, folder / "search", run / "logs" / "crop",
            budget=search_rounds, **_crop_vrr_options(config))
        write_json(folder / "SEARCH.json", record)
        assert_search_clean(record)
        params = {**record["levers"], "cell_overrides": record["cell_overrides"]}
        rejected = set(record["rejected_tool_cells"])
        expected_boxes = record["boxes"]
    if config.get('reuse_verified_crop_delivery') and record is not None and record.get('verified_delivery'):
        from shape.tools.single_cell_cropper.delivery import reuse
        result = check_result(reuse(record['verified_delivery'], image=image, mask=mask['mask'],
            source_image_id=ident, group='default', min_area=mask['min_area'],
            parameters={**defaults, **record['levers'], 'cell_overrides': record['cell_overrides']},
            rejected_cells=rejected), 'cell_crops_metadata_path')
    else:
        result = check_result(Single_Cell_Cropper_Tool().execute(original_image=str(image), nuclei_mask=mask["mask"],
                 source_image_id=ident, group="default", min_area=mask["min_area"], output_format="tif",
                 query_cache_dir=str(folder / "delivered"), **params), "cell_crops_metadata_path")
    write_json(folder / "result.json", result)
    crops = read_json(Path(result["cell_crops_metadata_path"]))["cell_metadata"]
    if expected_boxes:
        boxes = {c["cell_id"]: c.get("crop_bbox") for c in crops}
        if boxes != expected_boxes:
            raise RuntimeError(f"crop delivery differs from VRR selection: {ident}")
    if rejected - {c["cell_id"] for c in crops}:
        raise RuntimeError(f"crop rejection IDs do not match delivered cells: {ident}")
    labels = tifffile.imread(mask["mask"]) if spec.mask_label else None
    # the scientific crops are cut from the row's raw stack (`raw_file`, when a dataset separates the crop decision
    # image from the stack the SSL stage sees; otherwise the decision image itself) at the selected boxes
    raw = raw_stack(within(prep / "public", row.get("raw_file", row["file"])), spec)
    kept, excluded = retained_crops(raw, row, crops, rejected, folder, spec, labels)
    print(f"CROP {ident} retained={len(kept)} excluded={len(rejected)}", flush=True)
    return kept, excluded, record, usage.snapshot() if _IN_WORKER else None


def cropping(prep: Path, run: Path, config: dict, images: list[dict], *, threads: bool = False) -> None:
    """An arm without a SHAPE engine (control, Codex's native path) and a SHAPE smoke: every image cropped by itself
    (per-image crop VRR with SHAPE); a full SHAPE run: the shared crop cohort (`_cropping_cohort`, the crop analogue
    of the segmentation cohort -- calibration subset's crop VRR -> consensus crop levers -> the rest cropped at the
    consensus with no LLM, outliers routed back to the VRR)."""
    engine = engine_for(config, "crop")
    # $FULL_CPJUMP1_CROP_PER_IMAGE=1 forces the per-image crop VRR even for a full SHAPE run -- used to
    # A/B the crop cohort against per-image crop through the full pipeline (same segmentation masks).
    force_per_image = os.environ.get("FULL_CPJUMP1_CROP_PER_IMAGE") == "1"
    if (config["arm"] == "shape" and not config.get("smoke_policy") and engine is not None
            and not threads and not force_per_image):
        return _cropping_cohort(prep, run, config, images, engine)
    return _cropping_per_image(prep, run, config, images, threads=threads)


def _cropping_per_image(prep: Path, run: Path, config: dict, images: list[dict], *, threads: bool = False) -> None:
    """Every image is cropped by itself (with SHAPE: its own crop VRR, no cohort), `crop_vrr_workers` images at a
    time, each in its own worker process with its share of the cores (`_in_processes`; threads=True: tests only);
    outputs are gathered in image order."""
    from shape.engine import usage

    masks = {r["image_id"]: r for r in read_json(run / "segmentation.json")}
    search_images, search_rounds = shape_search_policy(config, len(images))
    workers, cores = crop_workers(config, len(images))
    tasks = [(i, row, prep, run, config, masks[row["image_id"]], search_images, search_rounds, cores)
             for i, row in enumerate(images)]
    in_process = threads or workers <= 1
    if in_process:                    # one engine for the stage; a worker process builds its own once
        _WORKER_ENGINE.clear()
        if config["arm"] == "shape" and search_images:
            _WORKER_ENGINE["engine"] = engine_for(config, "crop")
    all_cells, exclusions = [], []
    try:
        results = _in_processes(_crop_image, tasks, workers, cores, threads=threads)
    finally:
        if in_process:
            _WORKER_ENGINE.clear()
    for row, (kept, excluded, record, spent) in zip(images, results):
        if spent is not None:
            usage.absorb(spent)
        all_cells.extend(kept)
        exclusions.extend(excluded)
        if record is not None:
            record_search(run, "crop", row["image_id"], record)
    if not all_cells:
        raise RuntimeError(f"crop retained no cells across all {len(images)} images")
    write_json(run / "cells.json", all_cells)
    write_json(run / "excluded_cells.json", exclusions)


def _crop_materialize(task) -> tuple[list, list, dict | None, dict | None]:
    """Phase 2 of the crop cohort: one image's delivered crop at its cohort-decided params (NO LLM) and its retained
    crops. Top-level so a spawned worker can run it (the pure cropper is CPU-bound; processes beat the GIL)."""
    from shape.engine import usage
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    row, prep, run, config, mask, delivered = task
    if _IN_WORKER:
        usage.reset()
    remaining(config)
    spec = dataset(config)
    ident = row["image_id"]
    folder = run / "crop" / ident
    folder.mkdir(parents=True, exist_ok=True)
    if digest(Path(mask["mask"])) != mask["sha256"]:
        raise ValueError(f"segmentation mask changed before cropping: {ident}")
    image = within(prep / "public", row["file"])
    params = dict(delivered["params"])
    if delivered["cell_overrides"]:
        params["cell_overrides"] = delivered["cell_overrides"]
    result = check_result(Single_Cell_Cropper_Tool().execute(
        original_image=str(image), nuclei_mask=mask["mask"], source_image_id=ident, group="default",
        min_area=mask["min_area"], output_format="tif", query_cache_dir=str(folder / "delivered"), **params),
        "cell_crops_metadata_path")
    crops = read_json(Path(result["cell_crops_metadata_path"]))["cell_metadata"]
    labels = tifffile.imread(mask["mask"]) if spec.mask_label else None
    # the scientific crops are cut from the row's raw stack (`raw_file`, see `_crop_image`) at the delivered boxes
    raw = raw_stack(within(prep / "public", row.get("raw_file", row["file"])), spec)
    kept, excluded = retained_crops(raw, row, crops, delivered["rejected"], folder, spec, labels)
    print(f"CROP_MATERIALIZE {ident} retained={len(kept)} excluded={len(delivered['rejected'])}", flush=True)
    return kept, excluded, delivered["record"], usage.snapshot() if _IN_WORKER else None


def _crop_reroute_empty(prep: Path, run: Path, config: dict, defaults: dict, engine, search_rounds: int,
                         per_vrr_cores: int, pending: list) -> dict:
    """Re-crop, with a full per-image VRR search (no cohort consensus), every image whose cohort-delivered crop
    retained zero cells -- a legitimate but rare cohort-consensus miss (e.g. this image's own segmentation found
    only small/edge cells that the shared margin/recenter levers happen to trim away). This mirrors the
    admission-routed path (crop_one seeded per image) but triggers on "delivered zero cells", a condition the
    cohort's upstream preflight grouping cannot see ahead of time. `pending`: [(row, mask)]. Returns
    {image_id: (kept, excluded, record)}; if the full search ALSO retains nothing, kept is empty (a genuinely
    cell-less image), left for the caller's aggregate check to surface."""
    from benchmarks.full_HCMA.harness.shape_arm import crop_one, tools_common
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    C = tools_common()
    spec = dataset(config)
    out = {}
    for row, mask in pending:
        ident = row["image_id"]
        image = within(prep / "public", row["file"])
        folder = run / "crop" / ident
        record = crop_one(C, Single_Cell_Cropper_Tool,
            {"image_id": ident, "path": str(image), "mask": mask["mask"], "min_area": mask["min_area"],
             "defaults": defaults, "brief": spec.tasks["crop"]},
            engine, config["deadline"], per_vrr_cores, folder / "reroute", run / "logs" / "crop",
            budget=search_rounds)
        write_json(folder / "COHORT_REROUTE.json", record)
        assert_search_clean(record)
        params = {**record["levers"], "cell_overrides": record["cell_overrides"]}
        rejected = set(record["rejected_tool_cells"])
        result = check_result(Single_Cell_Cropper_Tool().execute(
            original_image=str(image), nuclei_mask=mask["mask"], source_image_id=ident, group="default",
            min_area=mask["min_area"], output_format="tif", query_cache_dir=str(folder / "reroute_delivered"),
            **params), "cell_crops_metadata_path")
        crops = read_json(Path(result["cell_crops_metadata_path"]))["cell_metadata"]
        labels = tifffile.imread(mask["mask"]) if spec.mask_label else None
        kept, excluded = retained_crops(raw_stack(image, spec), row, crops, rejected, folder, spec, labels)
        print(f"CROP_REROUTE {ident} retained={len(kept)} excluded={len(rejected)} (cohort delivered 0)",
              flush=True)
        out[ident] = (kept, excluded, {**record, "mode": "reroute_empty"})
    return out


def _cropping_cohort(prep: Path, run: Path, config: dict, images: list[dict], engine) -> None:
    """The full SHAPE arm's cropping as a cohort -- the crop analogue of `_segmentation_cohort`, over the shared
    `segmentation_cohort.run`: images group by their preflight read, a calibration sample per group runs the full
    per-image crop VRR, the consensus of the agent-tuned crop levers (margin/recenter/... the CARD's non-per-cell
    levers) applies to the rest at a single NO-LLM cropper call, and any image the admission flags is routed back to
    its own VRR. Phase 1 decides each image's crop params (above); phase 2 materialises every image's retained crops
    at its params in spawned workers (the cropper is CPU-bound). Records each image's SEARCH/COHORT record and the
    shared cohort record (COHORT_RECORD.json), like the segmentation cohort."""
    from types import SimpleNamespace
    from benchmarks.full_HCMA.harness.shape_arm import crop_one, tools_common
    from shape.agent.verification import segmentation_cohort
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
    from shape.engine import usage

    spec = dataset(config)
    masks = {r["image_id"]: r for r in read_json(run / "segmentation.json")}
    rows = {r["image_id"]: r for r in images}
    C = tools_common()
    registry = C.tool_levers("crop", Path(__file__).resolve().parents[3])
    crop_levers = [k for k, v in registry.items() if v.kind != "per_cell"]
    defaults = {k: C.normal_default(registry[k]) for k in crop_levers}
    defaults.update({k: v for k, v in config.get("tool_params", {}).items() if k in defaults})
    workers = max(1, int(config["cores"]))
    _, search_rounds = shape_search_policy(config, len(images))
    per_vrr_cores = max(1, workers // max(1, min(workers, len(images))))
    deliveries, lock = {}, threading.Lock()                 # image_id -> {params, cell_overrides, rejected, record}

    def run_one(spec_c, options):
        remaining(config)
        ident = spec_c["image_id"]
        row, mask = rows[ident], masks[ident]
        image = within(prep / "public", row["file"])
        folder = run / "crop" / ident
        folder.mkdir(parents=True, exist_ok=True)
        opts = options or {}
        if opts.get("cohort_levers") is not None:
            # cohort mode: no LLM, no cropper here -- just carry the consensus params to phase 2.
            params = {k: v for k, v in opts["cohort_levers"].items() if k in defaults}
            # a cohort-mode image runs no VRR; give the record the health counters a clean search has
            # (fallback_rounds / reasoner_failures / execution_errors) so record_search's assert passes.
            rec = {"mode": "cohort", "levers": params, "cell_overrides": {}, "rejected_tool_cells": [],
                   "n_calls": 0, "status": "searched", "resolution": {"status": "satisfied"},
                   "admission": {"rule": "RUN"}, "fallback_rounds": 0, "reasoner_failures": 0,
                   "execution_errors": [], "stop_reason": None}
            with lock:
                deliveries[ident] = {"params": params, "cell_overrides": {}, "rejected": set(), "record": rec}
            write_json(folder / "COHORT_SEARCH.json", rec)
            res = SimpleNamespace(params=params, result=object(), status="satisfied", reason="cohort",
                                  details={"admission": {"rule": "RUN"}, "call": "delivered",
                                           "measured_problems": {"delivered": []}})
            return [res], res.result, "satisfied", None
        mode = "routed" if opts.get("start_params") is not None else "search"
        d = dict(defaults)
        if opts.get("start_params"):
            d.update({k: v for k, v in opts["start_params"].items() if k in defaults})
        record = crop_one(C, Single_Cell_Cropper_Tool,
            {"image_id": ident, "path": str(image), "mask": mask["mask"], "min_area": mask["min_area"],
             "defaults": d, "brief": spec.tasks["crop"]},
            engine, config["deadline"], per_vrr_cores, folder / mode, run / "logs" / "crop", budget=search_rounds,
            **_crop_vrr_options(config))
        write_json(folder / ("COHORT_SEARCH.json" if mode == "routed" else "SEARCH.json"), record)
        assert_search_clean(record)
        with lock:
            deliveries[ident] = {"params": dict(record["levers"]), "cell_overrides": dict(record["cell_overrides"]),
                                 "rejected": set(record["rejected_tool_cells"]), "record": {**record, "mode": mode}}
        status = record["resolution"]["status"]
        res = SimpleNamespace(
            params=record["levers"], result=(object() if record.get("delivered") is not None else None),
            status="satisfied" if status in ("satisfied", "resolved", "searched") else status,
            reason=record["resolution"].get("reason"),
            details={"admission": record.get("admission") or {}, "call": "delivered",
                     "measured_problems": {"delivered": []}})
        print(f"CROP_{mode.upper()} {ident} calls={record['n_calls']} levers={record['levers']}", flush=True)
        return [res], res.result, res.status, record["resolution"].get("fault")

    # a dataset that allows empty masks has images without an object: no crop VRR for them (as in the per-image path);
    # the cropper still runs on them at the defaults in phase 2
    empty = [r for r in images if spec.allow_empty_masks and masks[r["image_id"]].get("n_cells") == 0]
    for r in empty:
        deliveries[r["image_id"]] = {"params": dict(defaults), "cell_overrides": {}, "rejected": set(), "record": None}
    scheduled = [r for r in images if r not in empty]
    specs = [{"image_id": r["image_id"], "path": str(within(prep / "public", r["file"]))} for r in scheduled]
    runs, cohort = segmentation_cohort.run(
        specs, run_one, lambda items, fn: _parallel(items, fn, min(workers, len(items) or 1)),
        _profile_reader(engine, "FULL_CPJUMP1_CROP_FALLBACK_GROUP"), defaults, crop_levers,
        workers=min(workers, len(scheduled) or 1))
    (run / "crop").mkdir(parents=True, exist_ok=True)
    write_json(run / "crop" / "COHORT_RECORD.json", cohort)
    print(f"CROP_COHORT counts={cohort.get('counts')}", flush=True)

    # Phase 2: materialise every image's retained crops at its delivered params, spawned (CPU-bound cropper).
    tasks = [(rows[r["image_id"]], prep, run, config, masks[r["image_id"]], deliveries[r["image_id"]])
             for r in images]
    mat_workers = max(1, min(workers, SEGMENT_WORKERS_CAP, len(images)))
    results = _in_processes(_crop_materialize, tasks, mat_workers, max(1, workers // mat_workers))
    all_cells, exclusions, empty = [], [], []
    for row, (kept, excluded, record, spent) in zip(images, results):
        if spent is not None:
            usage.absorb(spent)
        if not kept:
            empty.append((row, masks[row["image_id"]]))
            continue
        all_cells.extend(kept)
        exclusions.extend(excluded)
        if record is not None:
            record_search(run, "crop", row["image_id"], record)
    if empty:
        print(f"CROP_COHORT {len(empty)} image(s) delivered zero cells -- rerouting to per-image VRR: "
              f"{[r['image_id'] for r, _ in empty]}", flush=True)
        rerouted = _crop_reroute_empty(prep, run, config, defaults, engine, search_rounds, per_vrr_cores, empty)
        for row, _ in empty:
            kept, excluded, record = rerouted[row["image_id"]]
            all_cells.extend(kept)
            exclusions.extend(excluded)
            record_search(run, "crop", row["image_id"], record)
    if not all_cells:
        raise RuntimeError(f"crop cohort retained no cells across all {len(images)} images")
    write_json(run / "cells.json", all_cells)
    write_json(run / "excluded_cells.json", exclusions)


def retained_crops(raw: np.ndarray, row: dict, crops: list[dict], rejected: set, folder: Path,
                   spec: Dataset = CPJUMP1, mask: np.ndarray | None = None) -> tuple[list, list]:
    """The scientific crops of one image: every crop not in `rejected`, cut from the raw CYX stack at the
    cropper's selected box into folder/raw_crops; (kept cell records, excluded cell IDs). With spec.mask_label
    each record also has `mask_label`, its instance id in `mask` (the delivered segmentation mask)."""
    ident = row["image_id"]
    labels = mask_labels(mask, crops) if spec.mask_label else {}
    kept, excluded = [], []
    for index, c in enumerate(crops):
        if c["cell_id"] in rejected:
            excluded.append(c["cell_id"])
            continue
        # The shared cropper's display crops are uint8. Its selected geometry, not those
        # display pixels, defines the scientific input to SSL in this benchmark.
        raw_path = folder / "raw_crops" / f"cell_{index:06d}.ome.tif"
        materialize_raw_crop(raw, c["crop_bbox"], raw_path, spec.channels, spec.raw_dtype(row))
        kept.append({"cell_id": c["cell_id"], "crop_path": str(raw_path.resolve()), "area": c["area"],
                     "sha256": digest(raw_path),
                     "display_crop_path": c["crop_path"], "crop_bbox": c["crop_bbox"],
                     **{k: row[k] for k in ("image_id", "plate", "well", "usage")},
                     **({"mask_label": labels[c["cell_id"]]} if spec.mask_label else {})})
    if not kept:
        if not spec.allow_empty_masks:
            raise RuntimeError(f"no retained crops for {ident}")
        # On a dataset that allows empty masks: either the segmentation mask was genuinely empty, or every
        # cell of this image was rejected by the (cohort or per-image) crop VRR. Both are legitimate
        # per-image outcomes here -- the image contributes no cells rather than crashing the whole stage.
        # The aggregate guard in the caller (before writing cells.json) still fails if NO image retained
        # anything.
        print(f"CROP_EMPTY {ident}: all {len(crops)} crop(s) rejected -> image contributes no cells", flush=True)
    return kept, excluded


def mask_labels(mask: np.ndarray | None, crops: list[dict]) -> dict:
    """{cell_id: the cell's instance id in the delivered mask}. The cropper's cells are the regionprops regions of
    that mask (one region per label, in label order); a cell's metadata keeps its region's `original_bbox` and
    `area`, and the label is the one whose bounding box and pixel count are those. Raises when none or more than one
    label matches a cell."""
    from scipy import ndimage
    if mask is None:
        raise ValueError("mask labels need the delivered segmentation mask")
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
        raise ValueError("invalid segmentation mask")
    mask = mask.astype(np.int64)
    areas = np.bincount(mask.ravel())
    regions = {}
    for label, box in enumerate(ndimage.find_objects(mask), 1):
        if box is not None:
            key = (box[0].start, box[1].start, box[0].stop, box[1].stop, int(areas[label]))
            regions.setdefault(key, []).append(label)
    out = {}
    for c in crops:
        found = regions.get((*(int(v) for v in c["original_bbox"]), int(c["area"])), [])
        if len(found) != 1:
            raise ValueError(f"cell {c['cell_id']} matches {len(found)} mask labels by its bbox and area")
        out[c["cell_id"]] = found[0]
    return out


def _tool_dict(result) -> dict:
    """A tool's returned dict as stored for the Codex arm: arrays and in-memory crop objects left out."""
    if not isinstance(result, dict):
        return {"error": f"tool returned {type(result).__name__}, not a dict"}
    return {k: v for k, v in result.items() if k != "cell_crop_objects" and not isinstance(v, np.ndarray)}


def codex_calls(stage: str, prep: Path, run: Path, config: dict, images: list[dict], *, threads: bool = False) -> None:
    """The Codex arm's image-level tool calls (one native worker per request wave). config["calls"] lists
    {"image_id", "call", "params"}; each runs once, on its own image, with SHAPE's tool kwargs: segment
    `Cell_Segmenter_Tool.execute(image, image_id, cpsam, bsize 256, CARD defaults + params)`, crop
    `Single_Cell_Cropper_Tool.execute(image, the delivered segmentation mask, its min_area, group "default", tif,
    params)`. Calls run in parallel over the cores: segmentation calls in threads sharing one GPU model under a lock,
    crop calls in worker processes (`_in_processes`; threads=True: tests only). A call the tool fails is recorded as
    failed, not fatal. Writes run/<image>/<call>/{result.json, CALL.json, tool/...}."""
    from benchmarks.full_HCMA.harness.shape_arm import tools_common

    rows = {r["image_id"]: r for r in images}
    calls = list(config["calls"])
    if not calls or any(c["image_id"] not in rows for c in calls):
        raise ValueError("Codex calls name images outside this worker")
    workers = max(1, min(int(config["cores"]), len(calls)))
    if stage == "segment":
        seg = segmenter(dataset(config))
        tool, defaults = seg["tool"], seg["defaults"]
        # With a REMOTE SAM service the per-call lock only re-serializes evals the server's replica pool runs
        # concurrently; keep a real lock only for a LOCAL single model (v2.0).
        lock = contextlib.nullcontext() if getattr(tool, "_sam_server_url", None) else threading.Lock()
        tool.allow_empty_masks = dataset(config).allow_empty_masks
        _parallel(calls, lambda call: _codex_call(
            (stage, prep, run, config, rows[call["image_id"]], None, call), (tool, lock, defaults, seg["fixed"])), workers)
    elif stage == "crop":
        masks = {r["image_id"]: r for r in read_json(run / "segmentation.json")}
        _in_processes(_codex_call, [("crop", prep, run, config, rows[c["image_id"]], masks[c["image_id"]], c)
                                    for c in calls], workers, max(1, int(config["cores"]) // workers), threads=threads)
    else:
        raise ValueError(f"no image-level Codex calls for stage {stage}")


def _codex_call(task, segment=None) -> None:
    """One Codex arm call (`codex_calls`); top-level so a worker process can run a crop call. `segment`: the shared
    (segmenter, its lock, the CARD defaults) of a segmentation call."""
    from benchmarks.full_HCMA.harness.shape_arm import CROP_GROUP
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    stage, prep, run, config, row, mask, call = task
    tool, lock, defaults, fixed = segment or (None, None, None, None)
    remaining(config)
    ident, params = call["image_id"], dict(call["params"])
    folder = run / ident / call["call"]
    folder.mkdir(parents=True)
    started, delivery, compiled = time.time(), None, None
    try:
        if stage == "segment":
            image = within(prep / "public", row["segment_file"])
            with lock:
                result = tool.execute(image=str(image), image_id=ident, query_cache_dir=str(folder / "tool"),
                                      **fixed, **{**defaults, **params})
        else:
            if digest(Path(mask["mask"])) != mask["sha256"]:
                raise ValueError(f"segmentation mask changed before cropping: {ident}")
            image = within(prep / "public", row["file"])
            rules = params.pop("cell_rules", None)
            params.pop("rules_parent", None)
            if rules:
                params["cell_overrides"], compiled = compile_cell_rules(
                    image, mask, ident, params, rules, call.get("rules_parent_folder"))
            result = Single_Cell_Cropper_Tool().execute(
                original_image=str(image), nuclei_mask=mask["mask"], source_image_id=ident, group=CROP_GROUP,
                min_area=mask["min_area"], output_format="tif", query_cache_dir=str(folder / "tool"), **params)
        error = result.get("error") if isinstance(result, dict) else "tool returned no dict"
    except Exception as failure:  # noqa: BLE001 -- the tool's failure is this call's recorded outcome
        result, error = {"error": f"{type(failure).__name__}: {failure}"}, f"{type(failure).__name__}: {failure}"
    stored = _tool_dict(result)
    write_json(folder / "result.json", stored)
    if not error and stage == "segment":
        if not stored.get("mask_path"):
            error = "tool did not deliver mask_path"
        else:
            labels = np.load(stored["mask_path"])
            height, width = tifffile.TiffFile(image).series[0].shape[-2:]
            if labels.shape != (height, width) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
                error = "invalid segmentation mask"
            else:
                target = folder / f"cell_mask_{ident}.tif"
                tifffile.imwrite(target, labels.astype(np.uint32))
                delivery = {"mask": str(target.resolve()), "sha256": digest(target),
                            "min_area": {**defaults, **params}["min_size"],
                            "n_cells": int(np.count_nonzero(np.unique(labels) > 0))}
    elif not error:
        metadata = Path(stored.get("cell_crops_metadata_path") or "")
        if not metadata.is_file():
            error = "tool did not deliver cell_crops_metadata_path"
        else:
            crops = read_json(metadata)["cell_metadata"]
            delivery = {"metadata": str(metadata.resolve()), "sha256": digest(metadata), "n_crops": len(crops),
                        "cell_ids": [c["cell_id"] for c in crops]}
    write_json(folder / "CALL.json", {"image_id": ident, "call": call["call"], "params": call["params"],
               "status": "error" if error else "complete", "error": error, "delivery": delivery,
               **({"cell_rules_compiled": compiled} if compiled is not None else {}),
               "seconds": round(time.time() - started, 1)})
    print(f"CODEX_CALL {stage} {ident} {call['call']} {'error' if error else 'complete'}", flush=True)


def compile_cell_rules(image, mask: dict, ident: str, params: dict, rules: list, parent_folder) -> tuple[dict, list]:
    """SHAPE's cell rules for one Codex crop call, by SHAPE's own compiler (`CropAgenticVerifier.extend_move`): the
    conditions read the per-cell input measurements of the image's delivered mask (`input_record`, the cell ids the
    tool gives at group CROP_GROUP) and, with a rules parent (that call's folder), its per-cell readouts from its own
    crop metadata (`CropAgenticVerifier.quantify`, as SHAPE reads a parent call); the rules apply in order on top of
    the call's own cell_overrides. Returns (the compiled cell_overrides, the rules as compiled with the cells each
    matched). Raises ValueError (a rule that matches no cell, auto_tighten outside mask_bbox, an unreadable parent)."""
    from benchmarks.full_HCMA.harness.shape_arm import CROP_GROUP
    from shape.agent.verification.base import INPUT
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier, CropTask, input_record

    verifier = CropAgenticVerifier(log=lambda *args, **kwargs: None)
    verifier.begin(CropTask(cores=1))
    verifier.input = input_record({"original_image": str(image), "nuclei_mask": mask["mask"],
                                   "min_area": mask["min_area"], "source_image_id": ident, "group": CROP_GROUP})
    parent = INPUT
    if parent_folder:
        parent = "rules_parent"
        verifier._labels[parent] = 1
        verifier.quantify(read_json(Path(parent_folder) / "result.json"), {"trial_index": 1})
        if 1 not in verifier._cells:
            raise ValueError(f"rules_parent has no per-cell readouts: {verifier._unusable.get(1)}")
    target = dict(params)
    extra = verifier.extend_move({"cell_rules": rules}, target, parent, {})
    return target.get("cell_overrides") or {}, list(extra.get("cell_rules") or [])


def materialize_raw_crop(raw: np.ndarray, box, path: Path, channels=CHANNELS, dtype="uint16") -> None:
    if raw.ndim != 3 or raw.shape[0] != len(channels) or raw.dtype != np.dtype(dtype):
        raise ValueError(f"raw crop source must be CYX, {len(channels)} channel(s) {list(channels)}, {dtype}")
    if len(box) != 4 or any(int(v) != v for v in box):
        raise ValueError("crop bounds must be four integers")
    r0, c0, r1, c1 = map(int, box)
    if not (0 <= r0 < r1 <= raw.shape[1] and 0 <= c0 < c1 <= raw.shape[2]) or r1 - r0 != c1 - c0:
        raise ValueError("invalid or non-square crop bounds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, raw[:, r0:r1, c0:c1], ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": list(channels)}})


def select_development(cells: list[dict], per_site: int, usages: tuple = CPJUMP1_USAGES) -> tuple[list[str], list[dict]]:
    selected = []
    table = pd.DataFrame(cells)
    for _, group in table.groupby("image_id", sort=True):
        records = group.to_dict("records")
        records.sort(key=lambda c: hashlib.sha256(c["cell_id"].encode()).hexdigest())
        selected.extend(records[:per_site])
    return development(selected, usages)


def read_embedding(result: dict, meta: list[dict]) -> np.ndarray:
    import anndata as ad
    check_result(result, "adata_path")
    data = ad.read_h5ad(result["adata_path"])
    ids = [str(c) for c in data.obs["cell_id"]]
    wanted = [c["cell_id"] for c in meta]
    if len(set(ids)) != len(ids) or set(ids) != set(wanted) or len(ids) != len(wanted):
        raise ValueError("embedding changed cell identities")
    position = {c: i for i, c in enumerate(ids)}
    x = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
    x = x[[position[c] for c in wanted]].astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("non-finite embedding")
    return x


def representation(prep: Path, run: Path, config: dict, images: list[dict]) -> None:
    from shape.agent.verification.representation_policy import POLICY as SSL_POLICY, expert_start, validate_policy, execution_audit
    search_policy = validate_policy(config.get('ssl_search_policy', 'legacy'))
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier, RepresentationTask, frozen_params
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool
    from benchmarks.full_HCMA.harness.shape_arm import _log_to
    from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier

    # The embed stage (DINOv3 inference + SSL training) uses the GPU when CUDA is available and
    # SILENTLY falls back to CPU otherwise (Cell_State_Analyzer_Tool resolves cuda-if-available) --
    # no loud failure like the segmenter's _require_segmenter_gpu. A CPU embed is correct but
    # ruinously slow, and the usual cause is running this stage on the CPU orchestration node by
    # mistake instead of the GPU node. Fail fast unless CPU is explicitly allowed.
    import torch
    if not torch.cuda.is_available() and os.environ.get("FULL_CPJUMP1_ALLOW_CPU_EMBED") != "1":
        from shape.utils.device import resolve_torch_device
        _, warn = resolve_torch_device()
        raise RuntimeError(
            f"embed stage: no usable CUDA GPU ({warn}). This stage must run on the GPU node -- it "
            "would otherwise fall back to CPU silently (hours, not minutes). Run it on the GPU node, "
            "or set FULL_CPJUMP1_ALLOW_CPU_EMBED=1 to force a CPU embed on purpose.")

    cells = read_json(run / "cells.json")
    from .input_integrity import verify_crop_inputs
    last_progress = [time.monotonic()]
    def integrity_progress(count):
        now = time.monotonic()
        if count == len(cells) or now-last_progress[0] >= 10:
            print(f"VERIFY_CROPS {count}/{len(cells)}", flush=True)
            last_progress[0] = now
    integrity = verify_crop_inputs(cells, min(32, int(config.get('cores', 1))), integrity_progress)
    write_json(run / 'CROP_INPUT_VERIFICATION.json', integrity)
    engine = engine_for(config, "embed")
    budget = ssl_epoch_budget(images, config["train_cells_per_site"], dataset(config).usages)
    if config.get("ssl_epoch_budget") not in (None, budget):
        raise ValueError(f"SSL epoch budget changed: config {config['ssl_epoch_budget']}, images {budget}")
    selected_checkpoint = None
    selection = {"mode": "zero_shot", "reason": "control frozen-backbone baseline"}
    if config.get("ssl_selection") is not None:
        selection = config["ssl_selection"]
        selected_checkpoint = selection.get("checkpoint")
        if selected_checkpoint and digest(Path(selected_checkpoint)) != selection["checkpoint_sha256"]:
            raise ValueError("selected SSL checkpoint changed")
    elif config["arm"] == "codex":
        # One Codex call: the tool's own result is its only feedback (no SHAPE readouts or gate).
        paths, meta = select_development(cells, config["train_cells_per_site"], dataset(config).usages)
        params = {**config.get("tool_params", {}), "split_key": "split"}
        if "max_epochs" in params:
            raise ValueError(f"max_epochs is fixed at {budget['max_epochs']} for every call; it is not a parameter")
        if params.get("adaptation_depth") != "zero_shot":
            params["max_epochs"] = budget["max_epochs"]
            # SHAPE's call time cap: `RepresentationAgenticVerifier.max_minutes` against the stage's search
            # deadline (`ssl_search_deadline`, taken when the Codex embed stage started).
            cap = RepresentationAgenticVerifier()
            cap.deadline = config["ssl_search_deadline"]
            params["max_minutes"] = cap.max_minutes()
        if config.get("resume_checkpoint"):
            params["init_checkpoint"] = config["resume_checkpoint"]
        result = Cell_State_Analyzer_Tool().execute(cell_crops=paths, cell_metadata=meta, seed=config["seed"],
                    query_cache_dir=str(run / "ssl"), **params)
        check_result(result, "adata_path")
        read_embedding(result, meta)
        selected_checkpoint = result.get("resume_best")
        if params.get("adaptation_depth") != "zero_shot" and not selected_checkpoint:
            raise RuntimeError("trained selection lacks resume_best")
        selection = {"mode": "frozen_checkpoint" if selected_checkpoint else "zero_shot",
                     "checkpoint": selected_checkpoint,
                     "checkpoint_sha256": digest(Path(selected_checkpoint)) if selected_checkpoint else None,
                     "n_development_cells": len(meta), "params": params, "training_budget": budget}
        write_json(run / "ssl_result.json", result)
    elif engine is not None:
        paths, meta = select_development(cells, config["train_cells_per_site"], dataset(config).usages)
        write_json(run / "development_cells.json", meta)
        # Optional label-free early stopping (default OFF), for the SSL early-stop-vs-baseline A/B on the SAME
        # cohort crops: $FULL_CPJUMP1_EARLY_STOP=1 turns on the tool's within-call proxy stop (es_tool, the
        # tool's EARLY_STOP_DEFAULTS) and the search-level stop (es_search). Unset = byte-identical to the
        # existing shape run (None passed to both, the pre-existing behaviour).
        early_stop_on = os.environ.get("FULL_CPJUMP1_EARLY_STOP") == "1"
        es_tool = ({"stop_by": "val_proxy", "patience": 2, "overfit_gap": True,
                    "gap_evals": 2, "train_wells": 264} if early_stop_on else None)
        es_search = {"plateau_rounds": 2} if early_stop_on else None
        task = RepresentationTask(cores=config["cores"], gpus=1,
                 deadline=ssl_search_deadline(config), split_key="split", brief=dataset(config).tasks["embed"])
        counter, lock = [0], threading.Lock()

        def runner(params):
            remaining(config)
            with lock:
                index = counter[0]
                counter[0] += 1
            write_json(run / "ssl" / f"call_{index:03d}.params.json", params)
            if search_policy == SSL_POLICY:
                write_json(run / "ssl" / f"call_{index:03d}.audit.json", execution_audit(params))
            result = Cell_State_Analyzer_Tool().execute(cell_crops=paths, cell_metadata=meta, seed=config["seed"],
                          query_cache_dir=str(run / "ssl"), early_stop=es_tool, **params)
            write_json(run / "ssl" / f"call_{index:03d}.result.json", result)
            if search_policy == SSL_POLICY:
                write_json(run / "ssl" / f"call_{index:03d}.audit.json", execution_audit(params, result))
            check_result(result, "adata_path")
            return result

        initial = frozen_params(task)
        baseline = runner(initial)
        verifier = FixedEpochVerifier(budget, log=_log_to(run / "logs" / "representation.jsonl"),
                                      expert_config=[dict(r) for r in SSL_EXPERT_CONFIG], search_evidence_policy=search_policy,
                                      early_stop=es_search)
        if search_policy == SSL_POLICY:
            verifier.expert_config = [dict(r) for r in expert_start()]
        verifier.llm_engine = engine
        if config.get("smoke_policy"):
            verifier.budget = int(config["smoke_policy"]["shape_search_rounds"])
        res = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(verifier,
              verifier.make_adapter({"cell_metadata": meta}), runner, initial, baseline,
              upstream_ctx={"representation_task": task})
        record = search_record(verifier, res)
        write_json(run / "representation_SEARCH.json", record)
        record_search(run, "embed", None, record)
        selected = res.result or baseline
        selected_checkpoint = selected.get("resume_best")
        if not selected_checkpoint and (res.params or initial).get("adaptation_depth") != "zero_shot":
            raise RuntimeError("trained selection lacks resume_best; cannot substitute a frozen backbone")
        if selected_checkpoint and not Path(selected_checkpoint).is_file():
            raise FileNotFoundError(selected_checkpoint)
        selection = {"mode": "frozen_checkpoint" if selected_checkpoint else "zero_shot",
                     "checkpoint": selected_checkpoint,
                     "checkpoint_sha256": digest(Path(selected_checkpoint)) if selected_checkpoint else None,
                     "n_development_cells": len(meta), "gate": verifier.admission, "training_budget": budget}
    write_json(run / "SSL_SELECTION.json", selection)
    if config.get("fit_only"):
        return
    # A bounded inference chunk keeps all evaluation pixels out of fitting and avoids an all-cell RAM spike.
    matrix = None
    import math
    ranges = np.array_split(np.arange(len(cells)), max(1, math.ceil(len(cells) / config["inference_chunk"])))
    for indices in ranges:
        remaining(config)
        start = int(indices[0])
        chunk = cells[start:int(indices[-1]) + 1]
        meta = [technical(c) for c in chunk]
        params = {"init_checkpoint": selected_checkpoint, "embed_only": True} if selected_checkpoint else {"adaptation_depth": "zero_shot"}
        result = Cell_State_Analyzer_Tool().execute(cell_crops=[c["crop_path"] for c in chunk], cell_metadata=meta,
                   seed=config["seed"], query_cache_dir=str(run / "inference" / f"chunk_{start:08d}"), **params)
        write_json(run / "inference" / f"chunk_{start:08d}.json", result)
        x = read_embedding(result, meta)
        if matrix is None:
            matrix = np.lib.format.open_memmap(run / "embedding.npy", mode="w+", dtype=np.float32, shape=(len(cells), x.shape[1]))
        matrix[start:start + len(chunk)] = x
        matrix.flush()
        gc.collect()
        print(f"EMBED {start + len(chunk)}/{len(cells)}", flush=True)
    if matrix is None:
        raise ValueError("no cells to embed")
    if selected_checkpoint and digest(Path(selected_checkpoint)) != selection["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint changed during frozen inference")
    profiles, table = well_profiles(matrix, cells, images, allowed_sites=dataset(config).allowed_sites,
                                    allow_empty_sites=dataset(config).allow_empty_masks)
    np.save(run / "profiles.npy", profiles)
    table.to_csv(run / "profiles.csv", index=False)


def calibration_view(meta: pd.DataFrame, plate_type: str = CPJUMP1.plate_type):
    """The calibration reference rows (opaque development groups) and the negative-control rows; a dataset
    without negative controls gives reference rows only."""
    from shape.agent.verification.verifiers.batch_correction_unified import CalibrationView
    keep = (meta.calibration_group != "") | meta.negative_control
    rows = np.flatnonzero(keep.to_numpy())
    chosen = meta.iloc[rows]
    # DMSO is the shared metric's negative-control sentinel, NOT this experiment's reagent.
    table = pd.DataFrame({"Metadata_JCP2022": ["DMSO" if r.negative_control else r.calibration_group
                        for r in chosen.itertuples()], "Metadata_Plate": chosen.plate.to_numpy(),
                        "Metadata_Well": chosen.well.to_numpy(), "Metadata_PlateType": plate_type,
                        "Metadata_PertType": ["negcon" if v else "trt" for v in chosen.negative_control],
                        "batch": chosen.plate.to_numpy()})
    return CalibrationView(rows=rows, meta=table, batch_key="batch", plate_types=(plate_type,))


def batch_correction(prep: Path, run: Path, config: dict, images: list[dict]) -> None:
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.batch_correction_unified import (
        BatchCorrectionVerifier, BatchTask, adapt_batch_correction, uncorrected_result, admission_evidence, admission_rule,
        blas_limits)
    from shape.tools.batch_corrector.tool import Batch_Corrector_Tool, DEFAULT_CONFIG
    from benchmarks.full_HCMA.harness.shape_arm import _log_to

    spec = dataset(config)
    engine = engine_for(config, "batch")
    x = np.load(run / "profiles.npy")
    meta = pd.read_csv(run / "profiles.csv", keep_default_na=False)
    view = calibration_view(meta, spec.plate_type)
    task = BatchTask(batch=meta.plate.to_numpy(), features=x, calibration=view, cores=config["cores"],
                    deadline=config["deadline"], brief=spec.tasks["batch"])
    initial = uncorrected_result(x, task.batch)
    counter, lock = [0], threading.Lock()

    def runner(params):
        remaining(config)
        with lock:
            index = counter[0]
            counter[0] += 1
        folder = run / "batch" / f"call_{index:03d}"
        result = Batch_Corrector_Tool().execute(x, pd.DataFrame({"batch": task.batch}), batch_key="batch",
                           harmony_params=params["harmony_params"], output_dir=str(folder))
        check_result(result, "corrected_cls_output")
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "embedding.npy", np.asarray(result["corrected_cls_output"], dtype=np.float32))
        write_json(folder / "params.json", params)
        write_json(folder / "result.json", {k: v for k, v in result.items() if k != "corrected_cls_output"})
        return result

    with blas_limits(x, config["cores"]):          # the verifier's threads per call, for every arm
        if engine is None:
            evidence = admission_evidence(task, initial["batch_effect"])
            decision = admission_rule(evidence)
            params = {**DEFAULT_CONFIG, **config.get("tool_params", {})}
            chosen = runner({"harmony_params": params}) if decision == "RUN" and not config.get("uncorrected") else initial
            write_json(run / "batch_SEARCH.json", {"admission": {"decision": decision}, "evidence": evidence,
                       "policy": "native tool, shared admission rule", "params": params})
            if config["arm"] == "codex":
                write_json(run / "FEEDBACK.json", {"admission": decision, "evidence": evidence,
                           "calibration": view.score(chosen["corrected_cls_output"]),
                           "uncorrected_calibration": view.score(x)})
        else:
            verifier = BatchCorrectionVerifier(log=_log_to(run / "logs" / "batch.jsonl"))
            verifier.llm_engine = engine
            if config.get("smoke_policy"):
                verifier.budget = int(config["smoke_policy"]["shape_search_rounds"])
            res = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(verifier, adapt_batch_correction,
                       runner, {}, initial, upstream_ctx={"batch_task": task})
            record = search_record(verifier, res)
            write_json(run / "batch_SEARCH.json", record)
            record_search(run, "batch", None, record)
            chosen = res.result or initial
    output = np.asarray(chosen["corrected_cls_output"], dtype=np.float32)
    if output.ndim != 2 or len(output) != len(x) or not np.isfinite(output).all():
        raise ValueError("invalid corrected profiles")
    np.save(run / "corrected.npy", output)


def cluster(prep: Path, run: Path, config: dict, images: list[dict]) -> None:
    import anndata as ad
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool
    x = np.load(run / "corrected.npy")
    meta = pd.read_csv(run / "profiles.csv", keep_default_na=False)
    data = ad.AnnData(X=x, obs=meta.set_index("well"))
    tool = Cell_State_Analyzer_Tool()
    key = tool._compute_umap_and_clustering(data, 0.5, None, 15, n_pcs=None)
    data.write_h5ad(run / "profiles.h5ad")
    data.obs[[key]].rename(columns={key: "state"}).to_csv(run / "states.csv")
    if "X_umap" in data.obsm:
        pd.DataFrame(data.obsm["X_umap"], index=data.obs_names, columns=["UMAP1", "UMAP2"]).to_csv(run / "umap.csv")


STAGES = {"segment": segmentation, "crop": cropping, "embed": representation, "batch": batch_correction, "cluster": cluster}
