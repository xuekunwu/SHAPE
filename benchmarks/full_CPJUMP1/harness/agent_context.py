"""Opt-in tool API documentation without SHAPE's search-policy commentary.

This is a benchmark context ablation, not a change to the executable tools,
observations, task, action schema, or resource budgets. Historical runs retain
the default full-card policy. Semantics must cover the entire actual registry.
"""
from __future__ import annotations

import json

LEGACY = "shape_card_v1"
API_ONLY = "tool_api_v1"
POLICIES = (LEGACY, API_ONLY)


def policy(config: dict) -> str:
    value = config.get("agent_context_policy", LEGACY)
    if value not in POLICIES:
        raise ValueError(f"unknown agent_context_policy: {value!r}")
    return value


# Functional meanings only: no symptom-to-action advice, recommended recipes,
# empirical benchmark outcomes, SHAPE observations, or suggested search order.
SEMANTICS = {
    "segment": {
        "diameter": "Expected cell diameter in pixels; determines the model's image scale.",
        "cellprob_threshold": "Foreground cell-probability threshold.",
        "flow_threshold": "Maximum per-mask flow error accepted by the model.",
        "min_size": "Minimum retained mask area in pixels.",
        "niter": "Number of flow-dynamics iterations; null uses the native default.",
        "tile_overlap": "Fractional overlap between inference tiles.",
        "augment": "Enable test-time flipped-tile augmentation and averaging.",
        "resample": "Run mask dynamics at original resolution when true; at rescaled resolution otherwise.",
        "color_class_filter": "Optional post-segmentation population filter: keep_high_br retains the higher blue-minus-red class; keep_low_br the lower class; null disables it.",
    },
    "crop": {
        "margin_mode": "fraction: bbox-based square plus max(margin, round(margin_frac * equivalent diameter)); tight: bbox-based square plus tight_buffer_px; mask_bbox: bbox square plus tight_buffer_px, independent of recenter, crop_size and min_containment.",
        "margin_frac": "Fraction of equivalent mask diameter used for the fraction-mode margin.",
        "margin": "Minimum fraction-mode margin in pixels; null resolves by mask filename: cell_mask=1, organoid_mask=10, otherwise 25 pixels.",
        "recenter": "Center fraction/tight crops on the mask centroid instead of the bounding-box midpoint, subject to image bounds and containment.",
        "tight_buffer_px": "Pixel buffer around the bounding box in tight and mask_bbox modes.",
        "crop_size": "Maximum fraction/tight crop side length in pixels; null means no cap. Crops that fail required containment are omitted.",
        "cell_overrides": "Per-cell settings keyed by cell_id: margin_mode, recenter, auto_tighten, min_containment, margin_frac, tight_buffer_px, or reset. auto_tighten applies only in mask_bbox mode. min_containment is in [0.90, 1], defaults to 1, and applies in fraction/tight modes. reset=true removes the cell's overrides.",
    },
    "embed": {
        "view_pairing": "Positive training view: another augmentation of this crop, a random crop of its site, or a random crop of its well. The crop itself is used when it is alone. Site/well pairing requires plate-qualified metadata.",
        "contrastive_mask": "Exclude negatives sharing the anchor's site or well, or exclude none (off); retain the positive pair. Site/well modes require metadata.",
        "batch_sampling": "Draw batches from all training crops, one plate, or one experiment; grouped modes require the corresponding metadata.",
        "trainable_pct": "Percentage of final backbone transformer blocks to train, rounded to whole blocks. 0 trains adapter/projector only; 100 also trains patch embeddings/tokens; null uses full fine-tuning. Backpropagation still traverses the backbone.",
        "init_from": "pretrained initializes a fresh training state; parent resumes the named parent's best validation-proxy state, including weights, optimizer, random streams and counters. See the action contract for parent selection.",
        "learning_rate": "AdamW learning rate for adapter and unfrozen backbone; also for projector unless head_learning_rate is set.",
        "head_learning_rate": "Separate projector learning rate; null uses learning_rate.",
        "weight_decay": "AdamW decoupled weight decay.",
        "clip_grad_norm": "Global trainable-parameter gradient norm cap; 0 disables clipping.",
        "warmup_fraction": "Fraction of this call's steps used for linear warmup; null uses native fresh/resumed-call defaults.",
        "koleo_weight": "Weight of the nearest-neighbor spreading loss on L2-normalized backbone CLS tokens; 0 disables it.",
        "intensity_norm": "off uses stored intensities; per_plate matches plate channel mean/spread to training crops and stores the training statistics with the model. For an unseen plate, the native tool uses that plate's own statistics.",
        "augmentation": "strong: resized crop, flips, continuous rotation, intensity scaling, blur and erasing. light: resized crop, right-angle rotation/mirror, per-channel brightness/contrast/gamma jitter, light blur and additive noise. Frozen embedding transforms are identical.",
        "crop_scale_min": "Minimum area fraction retained by the training random resized crop; no change to frozen embedding transform.",
        "drop_path": "Training-only stochastic-depth rate, increasing by block depth; disabled for embedding.",
    },
    "batch": {
        "theta": "Harmony batch-diversity penalty.",
        "sigma": "Soft k-means kernel width.",
        "lambda_value": "Per-cluster batch-regression ridge penalty; null uses the native dynamic estimate.",
        "nclust": "Number of Harmony clusters, also constrained by input row count; null uses the native rule.",
        "input_pca": "Input PCA dimension before detection/correction; 0 disables PCA.",
        "input_scale": "Input scaling: none, per-feature z-score, or per-row L2 normalization.",
        "max_iter_harmony": "Maximum outer Harmony iterations; null uses the native default.",
        "epsilon_harmony": "Outer Harmony objective convergence tolerance; null uses the native default.",
        "max_iter_kmeans": "Maximum k-means iterations per Harmony round; null uses the native default.",
        "epsilon_cluster": "Per-round k-means convergence tolerance; null uses the native default.",
        "block_size": "Fraction of rows updated per k-means block; null uses the native default.",
        "tau": "Small-batch protection parameter in the batch-diversity penalty; null uses the native default.",
        "random_state": "k-means initialization seed.",
    },
}

FIXED = {
    "segment": ["The model is Cellpose-SAM CPSAM; bsize is fixed at 256. Task-specific target semantics come from Task, not a stained-cell default convention."],
    "crop": [
        "Every output crop is a square of original image pixels, shifted within image bounds; neighboring pixels are not masked or padded.",
        "Objects below the delivering segmentation's min_area are excluded. No crop retains less than 0.90 of its target mask.",
        "auto_tighten cuts a mask_bbox square while preserving at least 0.90 target area; fixed internal limits are contamination 0.2, improvement 0.02 and 20 steps. A mode change away from mask_bbox clears inherited auto_tighten.",
    ],
    "embed": ["The separately stated fixed epoch budget is binding. Training uses supplied train wells; validation selects checkpoints by the native validation proxy. Held-out wells never fit weights or select candidates."],
    "batch": ["Shared admission and calibration constraints remain binding. Parameters do not override an abstention decision."],
}


CROP_OBJECTIVES = [
    "Every crop is square.",
    "Contain one whole target cell without cutting it off.",
    "Center the target where image bounds and containment permit.",
    "The target occupies the main area of the frame.",
    "Minimize neighboring cells, without sacrificing the preceding requirements.",
]


# Datasets whose segment stage runs another SHAPE segmenter than Cellpose-SAM (`Dataset.segment_tool`): the same
# functional meanings for its own levers, and its own fixed behaviour.
SEMANTICS_BY_TOOL = {
    "organoid_segmenter": {
        "diameter": "Cellpose rescale reference in pixels: the image is resized by 30 / diameter before inference; null keeps the checkpoint's native scale.",
        "flow_threshold": "Maximum per-mask flow error accepted by the model.",
        "cellprob_threshold": "Foreground probability threshold.",
        "min_size": "Minimum retained mask area in pixels.",
        "niter": "Number of flow-dynamics iterations; null uses the native default.",
        "tile_overlap": "Fractional overlap between inference tiles; null uses the native default.",
        "augment": "Enable test-time augmentation; null uses the native default.",
        "resample": "Run mask dynamics at original resolution when true; at rescaled resolution when false; null uses the native default.",
    },
}
FIXED_BY_TOOL = {
    "organoid_segmenter": ["The model is a fixed custom-trained CO Cellpose checkpoint; the input is segmented as one grayscale channel at its original intensity range; adaptive rescue is disabled. Task-specific target semantics come from Task."],
}


def tool_api(stage: str, registry: dict, tool: str | None = None) -> str:
    """Render neutral semantics; never silently hide an executable lever. `tool`: the stage's SHAPE tool directory
    when it is not the benchmark default (a dataset's own segmenter)."""
    semantics = SEMANTICS_BY_TOOL.get(tool, SEMANTICS[stage]) if tool else SEMANTICS[stage]
    fixed = FIXED_BY_TOOL.get(tool, FIXED[stage]) if tool else FIXED[stage]
    missing, stale = set(registry) - set(semantics), set(semantics) - set(registry)
    if missing or stale:
        raise ValueError(f"tool API documentation drift for {stage}: missing={sorted(missing)}, stale={sorted(stale)}")
    contract = {"parameters": {k: semantics[k] for k in registry}, "fixed_behavior": fixed}
    if stage == "crop":
        # These five task objectives live inside the historical Levers section.
        # They must survive the removal of search advice and tuning commentary.
        contract["quality_objectives_in_order"] = CROP_OBJECTIVES
    return "Tool API semantics:\n" + json.dumps(contract, sort_keys=True, ensure_ascii=False) + "\n"
