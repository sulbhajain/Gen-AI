# Hallucination Detection in LLMs

A fast and memory-efficient Python package for detecting hallucinations in Large Language Models (LLMs) using BatchEnsemble with LoRA fine-tuning.

Based on the paper: ["Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models"](https://arxiv.org/html/2409.02976v1)

## Overview

This package implements a novel method for detecting both **faithfulness** and **factual** hallucinations in LLMs through:

- **Memory-Efficient Ensembles**: Uses BatchEnsemble with shared pre-trained weights and rank-one fast weights
- **LoRA Integration**: Low-rank adaptation for efficient fine-tuning
- **Uncertainty Estimation**: Predictive entropy, aleatoric, and epistemic uncertainty metrics
- **Binary Classification**: Trains classifiers to detect hallucinations using uncertainty features

### Key Features

- **Two Backend Options**:
  - **HuggingFace Transformers**: Full BatchEnsemble with LoRA fine-tuning
  - **Ollama** (NEW): Lightweight API-based ensemble for local inference
- **Fast Training & Inference**: Requires only a single GPU (tested on A40)
- **High Accuracy**: 97.8% accuracy on faithfulness hallucination detection, 68% on factual hallucinations
- **Low Memory Footprint**: O(mn + M(m+n)) complexity vs O(Mmn) for vanilla ensembles
- **Task Agnostic**: Works across different downstream tasks

## Installation

### Package Installation

```bash
pip install -e .
```

### Ollama Installation (Optional - Recommended for Mac)

For lightweight, local inference without HuggingFace cache issues:

```bash
# Install Ollama
brew install ollama

# Start Ollama server
ollama serve

# Pull a model (in another terminal)
ollama pull gemma:2b
```

Or download from: https://ollama.ai

## Quick Start

### Option 1: Ollama Backend (Recommended for Local Use)

```python
from hallucination_detection import OllamaBatchEnsemble

# Create Ollama-based ensemble (no model download needed)
model = OllamaBatchEnsemble.from_pretrained(
    model_name="gemma:2b",  # Any Ollama model
    num_ensemble_members=5
)

# Generate ensemble predictions
prompt = "Answer: What is the capital of France?"
outputs = model.generate_ensemble(prompt, max_tokens=100)

# Outputs contain 5 predictions with varying temperatures
print(f"Generated {len(outputs)} ensemble predictions")
```

### Option 2: HuggingFace Backend (Full BatchEnsemble + LoRA)

```python
from hallucination_detection import BatchEnsembleModel, HallucinationDetector
from hallucination_detection.uncertainty import UncertaintyEstimator

# Load and create ensemble
model = BatchEnsembleModel.from_pretrained(
    "google/gemma-2b",
    num_ensemble_members=5
)

# Fine-tune on your dataset
from hallucination_detection.training import BatchEnsembleTrainer
trainer = BatchEnsembleTrainer(model, train_dataset)
trainer.train()

# Estimate uncertainty
estimator = UncertaintyEstimator(model)
uncertainties = estimator.compute_uncertainties(prompts)

# Detect hallucinations
detector = HallucinationDetector()
detector.train(uncertainty_features, labels)
is_hallucinated = detector.predict(uncertainties)
```

## Package Structure

```
hallucination_detection/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── batch_ensemble.py       # BatchEnsemble implementation (HuggingFace)
│   ├── ollama_ensemble.py      # Ollama-based ensemble (NEW)
│   ├── lora_adapter.py          # LoRA integration
│   └── fast_weights.py          # Rank-one fast weight matrices
├── uncertainty/
│   ├── __init__.py
│   ├── estimator.py             # Uncertainty computation
│   └── metrics.py               # Entropy, aleatoric, epistemic
├── detection/
│   ├── __init__.py
│   ├── classifier.py            # Hallucination detector
│   └── features.py              # Feature extraction
├── training/
│   ├── __init__.py
│   ├── trainer.py               # Training loop
│   └── utils.py                 # Training utilities
└── utils/
    ├── __init__.py
    ├── data.py                  # Dataset processing
    └── initialization.py        # Weight initialization
```

## Supported Hallucination Types

### Faithfulness Hallucinations
When the LLM deviates from provided instructions or context. Example: Answering when the answer is not in the provided context.

### Factual Hallucinations
When the LLM generates content that contradicts verifiable facts. Example: Incorrect answers in knowledge-based questions.

## Backend Comparison

| Feature | Ollama | HuggingFace |
|---------|--------|-------------|
| **Installation** | Simple (`brew install ollama`) | Requires pip packages |
| **Model Loading** | ✅ Fast (API-based) | ❌ Slow (downloads ~2GB) |
| **Memory Usage** | ✅ Low (server managed) | ❌ High (loads in Python) |
| **Cache Issues** | ✅ None | ⚠️ Possible corruption |
| **LoRA Fine-tuning** | ❌ Not supported | ✅ Full support |
| **Fast Weights** | ❌ Not supported | ✅ Full BatchEnsemble |
| **Ensemble Method** | Temperature variation | Rank-one fast weights |
| **Best For** | Quick inference, Mac users | Training, research |

**Recommendation**: Use Ollama for quick experiments and inference, especially on Mac. Use HuggingFace for full training pipeline and research.

## Citation

If you use this package, please cite the original paper:

```bibtex
@article{arteaga2024hallucination,
  title={Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models},
  author={Arteaga, Gabriel Y. and Sch{\"o}n, Thomas B. and Pielawski, Nicolas},
  journal={arXiv preprint arXiv:2409.02976},
  year={2024}
}
```

## License

MIT License

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30
- peft >= 0.4.0
- scikit-learn >= 1.0
- numpy >= 1.20
