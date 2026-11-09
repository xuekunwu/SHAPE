"""Recorded, independently reversible segmentation efficiency switches."""
import os


def option(name, explicit, default, choices):
    value = str(explicit if explicit is not None else os.environ.get(name, default))
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")
    return value


def calibration_mode(explicit=None):
    return option("SHAPE_SEGMENTATION_CALIBRATION", explicit, "full", ("full", "skip"))


# The defaults of the Cellpose-SAM verifier are the historical behaviour (a stop is confirmed, an admissible reference with
# zero problems is still searched); the organoid verifier defaults to the cost-saving settings (its CARD). Each is
# switched per run by its environment variable.
LEGACY = {"zero_problem_reference": "0", "stop_confirmation": "always", "compact_prompts": "0"}
EFFICIENT = {"zero_problem_reference": "1", "stop_confirmation": "valid_choice", "compact_prompts": "0"}


def verifier_policy(defaults=None):
    defaults = LEGACY if defaults is None else defaults
    return {
        "zero_problem_reference": option("SHAPE_SEGMENTATION_EARLY_STOP", None, defaults["zero_problem_reference"], ("0", "1")),
        "stop_confirmation": option("SHAPE_SEGMENTATION_CONFIRM_STOP", None, defaults["stop_confirmation"], ("always", "valid_choice")),
        "compact_prompts": option("SHAPE_SEGMENTATION_COMPACT_PROMPTS", None, defaults["compact_prompts"], ("0", "1")),
    }
