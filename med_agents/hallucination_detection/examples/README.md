# Example Usage

This directory contains examples demonstrating different use cases of the hallucination detection package.

## Examples

### 1. Quick Start (`quickstart.py`)

Minimal example showing basic usage:
- Loading a BatchEnsemble model
- Computing uncertainty estimates
- Generating predictions from ensemble members

```bash
python examples/quickstart.py
```

### 2. Faithfulness Hallucination Detection (`faithfulness_detection.py`)

Complete example for detecting faithfulness hallucinations on SQuAD:
- Fine-tuning BatchEnsemble on SQuAD 2.0
- Computing uncertainty estimates
- Training hallucination detector
- Evaluating performance

```bash
python examples/faithfulness_detection.py
```

**Expected Results** (based on paper):
- Accuracy: ~97.8% for faithfulness hallucination detection
- Requires: Single A40 GPU or equivalent
- Training time: ~2-3 hours for 3 epochs

### 3. Factual Hallucination Detection (`factual_detection.py`)

Example for detecting factual hallucinations on MMLU:
- Fine-tuning on MMLU dataset
- Multiple-choice question answering
- Detecting incorrect answers

```bash
python examples/factual_detection.py
```

**Expected Results** (based on paper):
- Accuracy: ~68% for factual hallucination detection

## Requirements

All examples require:
- PyTorch with CUDA support
- At least 16GB GPU memory (24GB recommended)
- Internet connection for downloading models and datasets

## Notes

- The first run will download the base model (~15GB for Mistral-7B)
- Fine-tuning can take several hours depending on your hardware
- You can adjust batch size and other hyperparameters based on your GPU memory
- For faster experimentation, reduce the dataset sizes in the examples
