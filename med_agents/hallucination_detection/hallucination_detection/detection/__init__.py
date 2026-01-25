"""Detection package initialization."""

from hallucination_detection.detection.classifier import (
    HallucinationDetector,
    EnsembleDetector,
)
from hallucination_detection.detection.features import (
    FeatureExtractor,
    prepare_classification_dataset,
)

__all__ = [
    "HallucinationDetector",
    "EnsembleDetector",
    "FeatureExtractor",
    "prepare_classification_dataset",
]
