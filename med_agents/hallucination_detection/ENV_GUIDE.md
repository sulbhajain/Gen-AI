# Environment Configuration Guide

This package uses `.env` files to manage configuration and secrets.

## Quick Start

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your values:**
   ```bash
   nano .env
   # or
   code .env
   ```

3. **Add your Hugging Face token** (if needed):
   ```bash
   HF_TOKEN=your_actual_token_here
   ```

4. **Use in your code:**
   ```python
   from hallucination_detection.utils import config
   
   print(config.model_name)  # mistralai/Mistral-7B-Instruct-v0.2
   print(config.num_epochs)  # 3
   ```

## Files

### `.env.example`
- Template file with all available settings
- Safe to commit to git
- Contains placeholder values and documentation

### `.env`
- Your actual configuration file
- **NEVER commit to git** (already in .gitignore)
- Contains your real API keys and settings

### `hallucination_detection/utils/config.py`
- Python module for loading and accessing configuration
- Type-safe property access
- Provides sensible defaults

## Configuration Sections

### 1. Model Configuration
```bash
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
NUM_ENSEMBLE_MEMBERS=5
HF_TOKEN=your_token_here
```

### 2. Training Configuration
```bash
LEARNING_RATE=2e-4
NUM_EPOCHS=3
BATCH_SIZE=4
OUTPUT_DIR=./output
```

### 3. LoRA Configuration
```bash
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1
```

### 4. Dataset Configuration
```bash
SQUAD_TRAIN_ANSWERABLE=28000
SQUAD_TRAIN_UNANSWERABLE=14000
MMLU_TRAIN_SIZE=40000
```

### 5. Compute Configuration
```bash
CUDA_VISIBLE_DEVICES=0
DEVICE=cuda
USE_FP16=true
```

### 6. Logging Configuration
```bash
WANDB_ENABLED=false
WANDB_API_KEY=your_wandb_key
LOG_LEVEL=INFO
```

## Usage Examples

### Basic Usage

```python
from hallucination_detection.utils import config

# Access configuration
model_name = config.model_name
learning_rate = config.learning_rate
use_fp16 = config.use_fp16  # Returns bool
```

### Using with BatchEnsemble

```python
from hallucination_detection import BatchEnsembleModel
from hallucination_detection.utils import config

model = BatchEnsembleModel.from_pretrained(
    config.model_name,
    num_ensemble_members=config.num_ensemble_members,
    lora_r=config.lora_r,
    lora_alpha=config.lora_alpha,
)
```

### Using with Trainer

```python
from hallucination_detection.training import BatchEnsembleTrainer
from hallucination_detection.utils import config

trainer = BatchEnsembleTrainer(
    model=model,
    train_dataset=dataset,
    output_dir=config.output_dir,
    learning_rate=config.learning_rate,
    num_epochs=config.num_epochs,
    batch_size=config.batch_size,
)
```

### Loading Custom Config File

```python
from hallucination_detection.utils import get_config

# Load from custom file
config = get_config("production.env")
print(config.model_name)
```

### Environment Variables Override

Environment variables always take precedence over `.env` file:

```bash
# Command line override
MODEL_NAME=gpt2 python examples/quickstart.py

# Or export
export NUM_ENSEMBLE_MEMBERS=10
python examples/quickstart.py
```

## Required Settings

### Minimum Configuration

For basic usage, you only need:
```bash
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

### For Gated Models

If using gated models (like Llama), you need:
```bash
HF_TOKEN=your_huggingface_token
```

Get your token at: https://huggingface.co/settings/tokens

### For Training

Recommended settings for training:
```bash
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
NUM_ENSEMBLE_MEMBERS=5
OUTPUT_DIR=./output
BATCH_SIZE=4
NUM_EPOCHS=3
USE_FP16=true
```

## Security Best Practices

1. **Never commit `.env`** - It's in `.gitignore` by default
2. **Use `.env.example`** - Update it when adding new settings
3. **Rotate tokens** - Change API keys periodically
4. **Use environment-specific files**:
   - `.env.development`
   - `.env.production`
   - `.env.test`

## Common Settings

### For Development
```bash
DEBUG=true
LOG_LEVEL=DEBUG
BATCH_SIZE=2
NUM_EPOCHS=1
WANDB_ENABLED=false
```

### For Production
```bash
DEBUG=false
LOG_LEVEL=INFO
BATCH_SIZE=8
NUM_EPOCHS=3
WANDB_ENABLED=true
USE_FP16=true
```

### For Testing
```bash
DEBUG=true
BATCH_SIZE=1
NUM_EPOCHS=1
SQUAD_TRAIN_ANSWERABLE=100
SQUAD_TRAIN_UNANSWERABLE=50
```

## Troubleshooting

### Config not loading
```python
# Check if .env exists
import os
print(os.path.exists('.env'))  # Should be True

# Load manually
from dotenv import load_dotenv
load_dotenv('.env')
```

### Values not updating
```python
# Environment variables take precedence
import os
print(os.getenv('MODEL_NAME'))  # Check env var

# Clear and reload
from hallucination_detection.utils import Config
config = Config('.env')  # Force reload
```

### Token not recognized
```bash
# Make sure no extra spaces
HF_TOKEN=hf_xxxxx  # Correct
# HF_TOKEN = hf_xxxxx  # Wrong (spaces)

# Check in Python
from hallucination_detection.utils import config
print(f"Token: {config.hf_token}")
```

## Example Configuration

See `examples/config_usage.py` for a complete example:

```bash
python examples/config_usage.py
```

## Advanced Usage

### Multiple Environments

```python
# Development
dev_config = get_config('.env.development')

# Production  
prod_config = get_config('.env.production')

# Use appropriate config
config = prod_config if IS_PRODUCTION else dev_config
```

### Dynamic Configuration

```python
from hallucination_detection.utils import config
import os

# Override at runtime
os.environ['BATCH_SIZE'] = '8'
config = Config()  # Will use BATCH_SIZE=8
```

### Type Safety

```python
# Config provides type-safe access
config.batch_size         # int
config.learning_rate      # float
config.use_fp16          # bool
config.model_name        # str
```

## Reference

All available configuration options are documented in `.env.example`.

For the complete API, see `hallucination_detection/utils/config.py`.
