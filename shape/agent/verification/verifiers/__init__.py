"""Per-stage VLRR verifiers for the SHAPE bioimage pipeline.

Each verifier is pure and deterministic: it imports only numpy and the shared
primitives in ``shape.agent.verification.base``. It has no dependency on the SHAPE
tools themselves -- the impedance match (tool result dict -> verifier input)
lives in ``shape.agent.verification.loaders``.
"""

from .image_loading import ImageLoadingVerifier
from .segmentation import SegmentationAgenticVerifier, SegmentationVerifier
from .organoid_segmentation import OrganoidSegmentationAgenticVerifier
from .crop import CropAgenticVerifier, CropVerifier
from .representation import RepresentationAgenticVerifier, RepresentationVerifier
from .morphological_state import MorphologicalStateVerifier
from .batch_correction_unified import BatchCorrectionVerifier

__all__ = [
    "ImageLoadingVerifier",
    "SegmentationVerifier",
    "SegmentationAgenticVerifier",
    "OrganoidSegmentationAgenticVerifier",
    "CropVerifier",
    "CropAgenticVerifier",
    "RepresentationVerifier",
    "RepresentationAgenticVerifier",
    "MorphologicalStateVerifier",
    "BatchCorrectionVerifier",
]
