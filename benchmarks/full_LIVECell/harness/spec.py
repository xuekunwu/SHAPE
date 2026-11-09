"""The full_LIVECell dataset spec for the full_CPJUMP1 harness."""
from __future__ import annotations

from benchmarks.full_CPJUMP1.harness.pipeline import Dataset

from .data import CHANNELS, IMAGE_SHAPE, N_UNITS, USAGES

LIVECELL = Dataset(
    name="LIVECell", cli="benchmarks.full_LIVECell.harness.cli", data="benchmarks.full_LIVECell.harness.data",
    score="benchmarks.full_LIVECell.harness.score", n_images=2 * N_UNITS, marker="LIVECELL_COMPLETE",
    image_shape=tuple(IMAGE_SHAPE), channels=tuple(CHANNELS), dtype=None, plate_type="LIVECell", mask_label=True,
    tasks={"segment": "Segment whole cells, one instance per cell, in a label-free phase-contrast image (LIVECell: "
                      "eight cell lines, sparse to confluent). The masks crop the same single-channel image at "
                      "identical coordinates.",
           "crop": "Single-channel phase-contrast single-cell morphology.",
           "embed": "Single-channel phase-contrast cells of eight cell lines; six culture-time stages; two fields "
                    "(sites) per unit. Only train and val units are visible.",
           "batch": "Unit profiles grouped into six culture-time stages (the batch key). Calibration references are "
                    "opaque train/val cell-line ids; there are no negative controls. Joint unlabeled correction "
                    "is transductive."},
    inputs={"segment": "Inputs: {n} images; each a 704 x 520 single-channel {dtype} phase-contrast TIFF (Phase), the "
                       "site's raw image.",
            "crop": "Inputs: {n} images; each a 704 x 520 single-channel {dtype} phase-contrast TIFF (Phase) and its "
                    "cell mask delivered by this arm's segment stage (min_area is that segmentation's min_size). The "
                    "delivered crop boxes are cut from the same raw image as the embedding stage's input.",
            "embed": "Inputs: the single-cell crops of this arm's crop stage. Calls fit and validate on train and val "
                     "units only: at most {cells} cells per train or val site; train units train, "
                     "val units validate; heldout units never enter a call. input is the pretrained "
                     "backbone (zero_shot) on those cells, not a call. After finish, a fresh worker embeds every "
                     "retained cell with the delivered call's frozen checkpoint (the pretrained backbone for input), "
                     "within the same arm deadline.",
            "batch": "Inputs: the unit profiles of this arm's embed stage (median cell embedding per site, mean of a "
                     "unit's two sites), batch key plate (the six culture-time stages). Each call runs the tool on "
                     "all rows with harmony_params = the tool's DEFAULT_CONFIG overridden by your parameters. input "
                     "is the uncorrected profiles, not a call. Batch calibration rule (shared by all arms): "
                     "correction runs only when the tool's batch-effect detector reports a batch effect on all rows "
                     "and at least one calibration reference group occurs on two or more stages. Here the rule's "
                     "decision is {rule}."},
    input_channels={"segment": "Phase", "crop": "Phase"},
    rules={"RUN": "RUN (correction admitted)", "SKIP": "SKIP (no batch effect detected on all rows)",
           "ABSTAIN": "ABSTAIN (no calibration reference group occurs on two or more stages)"},
    withheld="Private labels are not available.", usages=USAGES)
