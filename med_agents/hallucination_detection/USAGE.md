# Installation and Usage Guide

## Installation

### From Source

```bash
cd hallucination_detection
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Basic Uncertainty Estimation

```python
from hallucination_detection import BatchEnsembleModel, UncertaintyEstimator

# Load model
model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5
)

# Create estimator
estimator = UncertaintyEstimator(model)

# Compute uncertainties
prompt = "What is the capital of France?"
uncertainties = estimator.compute_uncertainties(prompt)

print(f"Predictive Entropy: {uncertainties['predictive_entropy']:.4f}")
print(f"Aleatoric: {uncertainties['aleatoric']:.4f}")
print(f"Epistemic: {uncertainties['epistemic']:.4f}")
```

### 2. Fine-tuning on Custom Data

```python
from hallucination_detection import BatchEnsembleModel
from hallucination_detection.training import BatchEnsembleTrainer, SQuADDataset

# Load model
model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5,
    lora_r=8,
    lora_alpha=32,
)

# Create dataset
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
    output_dir="./output",
    num_epochs=3,
    batch_size=4,
)

history = trainer.train()
```

### 3. Training Hallucination Detector

```python
from hallucination_detection import HallucinationDetector

# Collect uncertainty estimates and labels
uncertainties = [...]  # List of uncertainty dicts
labels = [...]  # Binary labels (0=correct, 1=hallucinated)

# Create and train detector
detector = HallucinationDetector(
    classifier_type="rf",  # Random Forest
    scale_features=True,
)

metrics = detector.train(uncertainties, labels)

# Use detector
is_hallucinated = detector.predict(new_uncertainties)
```

### 4. Complete Pipeline

```python
from hallucination_detection import (
    BatchEnsembleModel,
    UncertaintyEstimator,
    HallucinationDetector,
)

# 1. Load and fine-tune model
model = BatchEnsembleModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
# ... fine-tune model ...

# 2. Compute uncertainties
estimator = UncertaintyEstimator(model)
uncertainties = estimator.compute_uncertainties(prompts)

# 3. Train detector
detector = HallucinationDetector()
detector.train(uncertainties, labels)

# 4. Detect hallucinations
new_uncertainties = estimator.compute_uncertainties(new_prompts)
predictions = detector.predict(new_uncertainties)
```

## Advanced Usage

### Analyzing Snowballing Effect

```python
# Analyze per-token uncertainty during generation
analysis = estimator.analyze_snowballing_effect(
    prompt="What is the capital of Mars?",
    max_new_tokens=50,
)

# Plot uncertainty over tokens
import matplotlib.pyplot as plt
plt.plot(analysis['predictive_entropy'])
plt.xlabel('Token Position')
plt.ylabel('Predictive Entropy')
plt.show()
```

### Using Different Classifiers

```python
# Compare multiple classifier types
from hallucination_detection import EnsembleDetector

# Train ensemble of classifiers
ensemble = EnsembleDetector(
    classifier_types=["lr", "dt", "svc", "rf", "knn"]
)
ensemble.train(uncertainties, labels)

# Predict with ensemble
predictions = ensemble.predict(new_uncertainties, voting="soft")
```

### Custom Feature Extraction

```python
from hallucination_detection.detection import FeatureExtractor

# Create custom feature extractor
extractor = FeatureExtractor(
    include_epistemic=True,
    use_max_features=True,
)

# Use with detector
detector = HallucinationDetector(
    feature_extractor=extractor
)
```

## Datasets

### Loading SQuAD

```python
from hallucination_detection.utils import load_squad_dataset

data = load_squad_dataset(
    include_unanswerable=True,
    train_answerable=28000,
    train_unanswerable=14000,
)

train_data = data['train']
val_data = data['validation']
test_data = data['test']
```

### Loading MMLU

```python
from hallucination_detection.utils import load_mmlu_dataset

data = load_mmlu_dataset(
    train_size=40000,
    val_size=2000,
    test_size=5000,
)
```

## Configuration

### Model Configuration

```python
model = BatchEnsembleModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    num_ensemble_members=5,      # Number of ensemble members
    lora_r=8,                     # LoRA rank
    lora_alpha=32,                # LoRA scaling
    lora_dropout=0.1,             # LoRA dropout
    use_noise_injection=False,    # Anchored ensembling
)
```

### Training Configuration

```python
trainer = BatchEnsembleTrainer(
    model=model,
    train_dataset=train_dataset,
    output_dir="./output",
    learning_rate=2e-4,           # Learning rate
    num_epochs=3,                 # Training epochs
    batch_size=4,                 # Batch size
    gradient_accumulation_steps=4, # Gradient accumulation
    warmup_steps=100,             # Warmup steps
    weight_decay=0.01,            # Weight decay
)
```

### Detector Configuration

```python
detector = HallucinationDetector(
    classifier_type="rf",         # lr, dt, svc, rf, knn
    scale_features=True,          # Standardize features
    n_estimators=100,             # RF: number of trees
    max_depth=None,               # RF: max depth
)
```

## Best Practices

1. **GPU Memory**: Requires ~16-24GB GPU memory for Mistral-7B
2. **Batch Size**: Adjust based on available memory
3. **Ensemble Size**: 5 members is a good balance (paper uses 5)
4. **LoRA Rank**: 8 works well for most tasks
5. **Fine-tuning**: Apply LoRA to all modules for better diversity
6. **Features**: Use first token + average features (exclude epistemic)

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
trainer = BatchEnsembleTrainer(
    batch_size=2,
    gradient_accumulation_steps=8,
)

# Use gradient checkpointing (enabled by default)
model.model.gradient_checkpointing_enable()
```

### Slow Training

```python
# Use mixed precision training (fp16)
# Enabled by default in BatchEnsembleTrainer

# Reduce dataset size for experimentation
data = load_squad_dataset(train_answerable=5000)
```

### Poor Performance

```python
# Increase ensemble size
model = BatchEnsembleModel.from_pretrained(
    model_name,
    num_ensemble_members=10,
)

# Train for more epochs
trainer = BatchEnsembleTrainer(num_epochs=5)

# Use ensemble of detectors
detector = EnsembleDetector()
```

## Performance Benchmarks

Based on the paper (Mistral-7B, single A40 GPU):

| Task | Metric | Score |
|------|--------|-------|
| Faithfulness Hallucination | Accuracy | 97.8% |
| Factual Hallucination | Accuracy | 68.0% |
| SQuAD F1 | Score | 93.4 |
| MMLU | Accuracy | 56.7% |

Training Time (3 epochs):
- SQuAD: ~2-3 hours
- MMLU: ~3-4 hours

Inference Speed:
- BatchEnsemble is faster than sample-based methods
- Single forward pass for all ensemble members

## References

```bibtex
@article{arteaga2024hallucination,
  title={Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models},
  author={Arteaga, Gabriel Y. and Sch{\"o}n, Thomas B. and Pielawski, Nicolas},
  journal={arXiv preprint arXiv:2409.02976},
  year={2024}
}
```

## Support

For issues and questions:
- GitHub Issues: [your-repo-url]
- Paper: https://arxiv.org/html/2409.02976v1
