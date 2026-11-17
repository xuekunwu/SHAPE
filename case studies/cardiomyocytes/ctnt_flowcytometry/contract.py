"""Brightfield data contract for the shared SHAPE image-processing stages."""

from benchmarks.full_CPJUMP1.harness.pipeline import Dataset
from dataclasses import replace

MODULE = "case studies.cardiomyocytes.ctnt_flowcytometry"
FLOW = Dataset(
    name="ctnt_flowcytometry", cli=MODULE + ".experiment_runner", data=MODULE + ".preparation", score=MODULE + ".heldout_evaluation",
    n_images=600, marker="CTNT_FLOW_COMPLETE", image_shape=(2048, 2048), channels=("Brightfield",),
    dtype="uint16", plate_type="iPSC-CM", mask_label=True, allowed_sites=(4, 25),
    tasks={
        "segment": "Segment whole live iPSC-derived cardiomyocytes, one instance per cell, in a single-channel "
                   "brightfield image. Preserve boundaries in confluent cultures; no stained channel is available.",
        "crop": "Single-channel brightfield iPSC-derived cardiomyocyte morphology. Preserve complete cells and "
                "use the verified boxes to cut the original uint16 pixels.",
        "embed": "Brightfield cardiomyocytes, multiple views of each culture well. Train and validate only on "
                 "the supplied seen train/validation wells; held-out wells are inference-only. Flow measurements are withheld.",
        "batch": "Well profiles from brightfield cardiomyocytes. experiment_date and plate_id are candidate batch factors; "
                 "plate_id is nested within experiment_date in this dataset. No validated "
                 "cross-plate technical reference groups or negative controls are supplied; donor identity alone "
                 "is not a technical replicate. Abstain from correction without calibration support."},
    inputs={
        "segment": "Inputs: {n} single-channel {dtype} brightfield images; one mask instance per cell.",
        "crop": "Inputs: {n} original {dtype} brightfield images and the masks selected by this arm. "
                "Return complete single-cell crop boxes; final scientific crops use original pixels.",
        "embed": "Inputs: retained single-cell crops; up to {cells} cells per seen training/validation view. "
                 "Only seen train wells may fit weights; seen val wells select. Held-out wells are inference-only.",
        "batch": "Inputs: well profiles (median cells per view, mean views per well). Candidate batch fields "
                 "are experiment_date and plate_id. Shared admission decision: {rule}. "
                 "No validated technical references or negative controls; preserve input when correction is not admitted."},
    input_channels={"segment": "Brightfield", "crop": "Brightfield"}, rules={},
    withheld="Flow percentages, donor identities and original filenames are not exposed to the SHAPE stages.")


def for_config(config):
    if config.get("technical_smoke_images") == 5:
        if config.get("smoke") is not True:
            raise ValueError("one-view aggregation is restricted to the five-image technical smoke")
        return replace(FLOW, allowed_sites=(1,))
    return FLOW
