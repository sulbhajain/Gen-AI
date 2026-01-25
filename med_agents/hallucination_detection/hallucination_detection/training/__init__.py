"""Training package initialization."""

from hallucination_detection.training.trainer import (
    BatchEnsembleTrainer,
    train_hallucination_detector,
)
from hallucination_detection.training.utils import (
    TextDataset,
    SQuADDataset,
    MMLUDataset,
    compute_metrics,
)

__all__ = [
    "BatchEnsembleTrainer",
    "train_hallucination_detector",
    "TextDataset",
    "SQuADDataset",
    "MMLUDataset",
    "compute_metrics",
]
