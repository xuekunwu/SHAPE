"""SHAPE agentic verification (VLRR) subsystem.

VLRR = Verify -> Localize -> Recommend -> Rerun. Each pipeline step is wrapped by
a deterministic verifier that scores the tool's output, localizes any fault, and
recommends one of five actions. In the SHAPE framework the roles split as:

  - the **planner** recommends sensible *initial* parameters for a tool;
  - the **verifier** (this subsystem) judges the execution result and, when a
    local fault is found, adjusts parameters and reruns the tool (via the
    Executor) until it converges or escalates.

Public surface:
  - ``base``: shared primitives (Verifier, normalize, weighted_score, reconcile,
    apply_bounded, PARAM_BOUNDS). ``VLRRResult``/``StageResolution`` live in
    ``shape.utils.schemas`` (single schema home) and are re-exported here.
  - ``verifiers``: the five per-stage verifiers (image loading -> morphological state).
"""

from shape.utils.schemas import StageResolution, VLRRResult

from .base import (
    Action,
    Decision,
    Locus,
    BaseVerifier,
    PARAM_BOUNDS,
    apply_bounded,
    normalize,
    reconcile,
    weighted_score,
)
from .verifiers import (
    CropVerifier,
    ImageLoadingVerifier,
    MorphologicalStateVerifier,
    RepresentationVerifier,
    SegmentationVerifier,
    BatchCorrectionVerifier,
)
from .step_verifier import StepVerifier

__all__ = [
    # primitives
    "VLRRResult",
    "BaseVerifier",
    "Decision",
    "Locus",
    "Action",
    "normalize",
    "weighted_score",
    "reconcile",
    "apply_bounded",
    "PARAM_BOUNDS",
    # verifiers
    "ImageLoadingVerifier",
    "SegmentationVerifier",
    "BatchCorrectionVerifier",
    "CropVerifier",
    "RepresentationVerifier",
    "MorphologicalStateVerifier",
    # framework component
    "StepVerifier",
    "StageResolution",
]
