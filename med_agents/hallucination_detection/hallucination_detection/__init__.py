"""
Hallucination Detection Package

A fast and memory-efficient package for detecting hallucinations in LLMs
using BatchEnsemble with LoRA fine-tuning.
"""

__version__ = "0.1.0"

from hallucination_detection.models.batch_ensemble import BatchEnsembleModel
from hallucination_detection.models.ollama_ensemble import OllamaBatchEnsemble
from hallucination_detection.uncertainty.estimator import UncertaintyEstimator
from hallucination_detection.detection.classifier import HallucinationDetector

__all__ = [
    "BatchEnsembleModel",
    "OllamaBatchEnsemble",
    "UncertaintyEstimator",
    "HallucinationDetector",
]
