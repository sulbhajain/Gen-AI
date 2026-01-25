# Hallucination Detection Package - Complete Guide

## Package Overview

This Python package implements the methods from the paper "Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models" (arXiv:2409.02976v1). It provides:

✅ **BatchEnsemble with LoRA**: Memory-efficient ensemble training  
✅ **Uncertainty Estimation**: Predictive entropy, aleatoric, and epistemic uncertainty  
✅ **Hallucination Detection**: Binary classifiers for detecting hallucinations  
✅ **Training Utilities**: Complete pipeline for fine-tuning and evaluation  
✅ **Pre-configured Datasets**: SQuAD and MMLU dataset loaders  

## Project Structure

```
hallucination_detection/
├── README.md                          # Main documentation
├── USAGE.md                           # Detailed usage guide
├── LICENSE                            # MIT License
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
│
├── hallucination_detection/           # Main package
│   ├── __init__.py                    # Package exports
│   │
│   ├── models/                        # BatchEnsemble models
│   │   ├── __init__.py
│   │   ├── batch_ensemble.py          # Main BatchEnsemble class
│   │   ├── fast_weights.py            # Rank-one fast weights
│   │   └── lora_adapter.py            # LoRA integration
│   │
│   ├── uncertainty/                   # Uncertainty estimation
│   │   ├── __init__.py
│   │   ├── estimator.py               # UncertaintyEstimator class
│   │   └── metrics.py                 # Entropy calculations
│   │
│   ├── detection/                     # Hallucination detection
│   │   ├── __init__.py
│   │   ├── classifier.py              # HallucinationDetector class
│   │   └── features.py                # Feature extraction
│   │
│   ├── training/                      # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py                 # BatchEnsembleTrainer
│   │   └── utils.py                   # Datasets and helpers
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       └── data.py                    # Data loading
│
└── examples/                          # Usage examples
    ├── README.md                      # Examples documentation
    ├── quickstart.py                  # Quick start guide
    └── faithfulness_detection.py      # Full training example
```

## Installation

### Step 1: Clone or Navigate to Directory

```bash
cd /Users/sulbhajain/Documents/Personal/genAI_projects/hallucination_detection
```

### Step 2: Install Package

```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"
```

### Step 3: Verify Installation

```python
import hallucination_detection
print(hallucination_detection.__version__)  # Should print: 0.1.0
```

## Quick Start (5 Minutes)

```python
from hallucination_detection import (
    BatchEnsembleModel,
    UncertaintyEstimator,
    HallucinationDetector
)

# 1. Load model
model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5
)

# 2. Compute uncertainty
estimator = UncertaintyEstimator(model)
prompt = "What is the capital of France?"
uncertainties = estimator.compute_uncertainties(prompt)

print(f"Predictive Entropy: {uncertainties['predictive_entropy']:.4f}")
```

Run the quickstart example:
```bash
python examples/quickstart.py
```

## Core Components

### 1. BatchEnsembleModel

The main model class that implements memory-efficient ensembles.

**Key Features:**
- Shared pre-trained weights with LoRA fine-tuning
- Rank-one fast weights for each ensemble member
- Memory complexity: O(mn + M(m+n)) vs O(Mmn) for vanilla ensembles

**Usage:**
```python
model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5,
    lora_r=8,
    lora_alpha=32,
)

# Fine-tune (adds LoRA to shared weights)
trainer = BatchEnsembleTrainer(model, train_dataset)
trainer.train()

# Use ensemble for inference
outputs = model.generate_ensemble(prompt)
```

### 2. UncertaintyEstimator

Computes uncertainty metrics from ensemble predictions.

**Metrics:**
- **Predictive Entropy**: Overall uncertainty H[P(x|D)]
- **Aleatoric**: Data uncertainty (irreducible)
- **Epistemic**: Model uncertainty (reducible with more data)

**Usage:**
```python
estimator = UncertaintyEstimator(model)

# Compute uncertainties
uncertainties = estimator.compute_uncertainties(
    prompts,
    return_tokens=True,  # Per-token uncertainties
    batch_size=8
)

# Analyze snowballing effect
analysis = estimator.analyze_snowballing_effect(prompt)
```

### 3. HallucinationDetector

Binary classifier for detecting hallucinations using uncertainty features.

**Supported Classifiers:**
- Logistic Regression (lr)
- Decision Tree (dt)
- Support Vector Classifier (svc)
- Random Forest (rf) - recommended
- k-Nearest Neighbors (knn)

**Usage:**
```python
detector = HallucinationDetector(
    classifier_type="rf",
    scale_features=True
)

# Train
metrics = detector.train(uncertainties, labels)

# Predict
is_hallucinated = detector.predict(new_uncertainties)

# Save/load
detector.save("detector.pkl")
detector = HallucinationDetector.load("detector.pkl")
```

## Complete Training Pipeline

### Step 1: Load Dataset

```python
from hallucination_detection.utils import load_squad_dataset

data = load_squad_dataset(
    include_unanswerable=True,
    train_answerable=28000,
    train_unanswerable=14000,
)
```

### Step 2: Create Model

```python
from hallucination_detection import BatchEnsembleModel

model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5
)
```

### Step 3: Fine-tune Model

```python
from hallucination_detection.training import BatchEnsembleTrainer, SQuADDataset

# Prepare dataset
train_dataset = SQuADDataset(
    contexts=train_contexts,
    questions=train_questions,
    answers=train_answers,
    tokenizer=model.tokenizer,
)

# Train
trainer = BatchEnsembleTrainer(
    model=model,
    train_dataset=train_dataset,
    num_epochs=3,
    batch_size=4,
)

history = trainer.train()
```

### Step 4: Compute Uncertainties

```python
from hallucination_detection import UncertaintyEstimator

estimator = UncertaintyEstimator(model)
uncertainties = estimator.compute_uncertainties(
    test_prompts,
    return_tokens=True
)
```

### Step 5: Train Detector

```python
from hallucination_detection import HallucinationDetector

detector = HallucinationDetector(classifier_type="rf")
metrics = detector.train(uncertainties, labels)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Step 6: Evaluate

```python
# Generate predictions
predictions = []
for prompt in test_prompts:
    outputs = model.generate_ensemble(prompt)
    predictions.append(outputs[0])

# Compute uncertainties
test_uncertainties = estimator.compute_uncertainties(test_prompts)

# Detect hallucinations
is_hallucinated = detector.predict(test_uncertainties)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_labels, is_hallucinated)
print(f"Detection Accuracy: {accuracy:.4f}")
```

## Expected Results

Based on the paper (Mistral-7B, single A40 GPU):

### Faithfulness Hallucination Detection (SQuAD 2.0)
- **Accuracy**: 97.8%
- **Dataset**: Unanswerable questions from SQuAD 2.0
- **Task**: Detect when model answers instead of saying "I don't know"

### Factual Hallucination Detection (MMLU)
- **Accuracy**: 68.0%
- **Dataset**: Multiple-choice questions
- **Task**: Detect incorrect answers

### Predictive Performance
- **SQuAD F1**: 93.4
- **SQuAD Exact Match**: 85.9
- **MMLU Accuracy**: 56.7%

### Computational Efficiency
- **Hardware**: Single A40 GPU
- **Training Time**: 2-3 hours for 3 epochs
- **Memory**: ~16-24GB GPU memory
- **Inference**: Faster than sample-based methods (single forward pass)

## Key Implementation Details

### 1. Weight Initialization

Fast weights initialized with **mean=1** (not 0) to preserve pre-trained knowledge:

```python
# He initialization with mean=1
self.r_vectors.normal_(mean=1.0, std=np.sqrt(2.0 / in_features))
self.s_vectors.normal_(mean=1.0, std=np.sqrt(2.0 / out_features))
```

### 2. LoRA Configuration

Applied to **all modules** (not just attention) for better diversity:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]
```

### 3. Feature Extraction

Features for classification (epistemic excluded due to correlation):

```python
features = [
    first_token_predictive_entropy,
    first_token_aleatoric,
    avg_predictive_entropy,
    avg_aleatoric,
]
```

### 4. Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 2e-4
- **Batch Size**: 4 (with gradient accumulation)
- **Epochs**: 3
- **Warmup**: 10% of total steps
- **Weight Decay**: 0.01

## Advanced Topics

### Snowballing Effect

Analyzing how uncertainty drops after generating a hallucinated token:

```python
analysis = estimator.analyze_snowballing_effect(
    prompt="Answer: What is the population of fictional city X?",
    max_new_tokens=50
)

# Plot entropy over tokens
import matplotlib.pyplot as plt
plt.plot(analysis['predictive_entropy'])
plt.title('Snowballing Effect: Uncertainty Drops After Hallucination')
plt.xlabel('Token Position')
plt.ylabel('Predictive Entropy (bits)')
plt.show()
```

### Anchored Ensembling

For training stability with noise injection:

```python
model = BatchEnsembleModel.from_pretrained(
    model_name,
    use_noise_injection=True  # Enable anchored ensembling
)
```

### Ensemble of Detectors

Combining multiple classifier types:

```python
from hallucination_detection.detection import EnsembleDetector

ensemble = EnsembleDetector(
    classifier_types=["lr", "dt", "svc", "rf", "knn"]
)
ensemble.train(uncertainties, labels)
predictions = ensemble.predict(test_uncertainties)
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `batch_size=2`
- Increase gradient accumulation: `gradient_accumulation_steps=8`
- Use smaller model or reduce ensemble size

### Slow Training
- Use fewer training examples for experimentation
- Reduce `max_length` in datasets
- Use gradient checkpointing (enabled by default)

### Poor Detection Performance
- Collect more training data
- Increase ensemble size
- Use ensemble of detectors
- Try different classifiers

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

## Additional Resources

- **Paper**: https://arxiv.org/html/2409.02976v1
- **Examples**: See `examples/` directory
- **Usage Guide**: See `USAGE.md`
- **API Documentation**: See docstrings in source code

## Next Steps

1. ✅ **Run Quick Start**: `python examples/quickstart.py`
2. ✅ **Read USAGE.md**: Detailed usage instructions
3. ✅ **Try Full Example**: `python examples/faithfulness_detection.py`
4. ✅ **Experiment**: Modify hyperparameters, try different datasets
5. ✅ **Contribute**: Extend functionality, add new features

---

**Package Version**: 0.1.0  
**Python**: >=3.8  
**PyTorch**: >=2.0  
**License**: MIT
