# ✅ Environment Configuration - Setup Complete!

I've created a complete environment configuration system for your hallucination detection package.

## 📁 Files Created

### 1. `.env.example` (Template)
- **Purpose**: Template with all available settings and documentation
- **Status**: ✅ Safe to commit to git
- **Contents**: 
  - Model configuration
  - Training hyperparameters
  - LoRA settings
  - Dataset configuration
  - Compute settings
  - API keys (with placeholders)
  - Logging options

### 2. `.env` (Your Local Config)
- **Purpose**: Your actual configuration file
- **Status**: 🔒 **NEVER commit to git** (protected by .gitignore)
- **Contents**: Pre-filled with default values, ready to customize

### 3. `hallucination_detection/utils/config.py` (Config Loader)
- **Purpose**: Type-safe configuration access
- **Features**:
  - Automatic .env loading
  - Type conversions (int, float, bool, string)
  - Sensible defaults
  - Environment variable override support
  - ~60 configuration properties

### 4. `ENV_GUIDE.md` (Documentation)
- Complete guide for using environment configuration
- Examples and best practices
- Troubleshooting tips

### 5. `examples/config_usage.py` (Example)
- Demonstrates how to use the config system
- Shows all configuration categories

## 🚀 Quick Start

### Step 1: Install Dependencies (including python-dotenv)

```bash
cd /Users/sulbhajain/Documents/Personal/genAI_projects/hallucination_detection
pip install -e .
```

### Step 2: Configure Your Environment

The `.env` file is already created. Edit it to add your settings:

```bash
# Edit the .env file
nano .env
# or
code .env
```

**Important**: Add your Hugging Face token if you plan to use gated models:
```bash
HF_TOKEN=hf_your_actual_token_here
```

Get your token at: https://huggingface.co/settings/tokens

### Step 3: Use Configuration in Your Code

```python
from hallucination_detection.utils import config

# Access any configuration value
print(config.model_name)          # mistralai/Mistral-7B-Instruct-v0.2
print(config.num_ensemble_members) # 5
print(config.learning_rate)       # 0.0002
print(config.batch_size)          # 4
```

### Step 4: Test the Configuration

```bash
# Run the example
python examples/config_usage.py
```

## 📋 Available Configuration Categories

### 1. **Model Configuration**
- `MODEL_NAME` - Base model to use
- `NUM_ENSEMBLE_MEMBERS` - Ensemble size (default: 5)
- `HF_TOKEN` - Hugging Face API token

### 2. **Training Configuration**
- `LEARNING_RATE` - Learning rate (default: 2e-4)
- `NUM_EPOCHS` - Training epochs (default: 3)
- `BATCH_SIZE` - Batch size (default: 4)
- `GRADIENT_ACCUMULATION_STEPS` - Gradient accumulation (default: 4)
- `OUTPUT_DIR` - Output directory

### 3. **LoRA Configuration**
- `LORA_R` - LoRA rank (default: 8)
- `LORA_ALPHA` - LoRA alpha (default: 32)
- `LORA_DROPOUT` - LoRA dropout (default: 0.1)

### 4. **Dataset Configuration**
- `SQUAD_TRAIN_ANSWERABLE` - SQuAD answerable questions
- `SQUAD_TRAIN_UNANSWERABLE` - SQuAD unanswerable questions
- `MMLU_TRAIN_SIZE` - MMLU training size

### 5. **Compute Configuration**
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `DEVICE` - Device (cuda/cpu)
- `USE_FP16` - Mixed precision training
- `USE_GRADIENT_CHECKPOINTING` - Memory optimization

### 6. **Logging & Monitoring**
- `WANDB_ENABLED` - Enable Weights & Biases
- `WANDB_API_KEY` - W&B API key
- `LOG_LEVEL` - Logging level

## 💡 Usage Examples

### Example 1: Basic Usage

```python
from hallucination_detection import BatchEnsembleModel
from hallucination_detection.utils import config

# Use config values automatically
model = BatchEnsembleModel.from_pretrained(
    config.model_name,
    num_ensemble_members=config.num_ensemble_members,
    lora_r=config.lora_r,
    lora_alpha=config.lora_alpha,
)
```

### Example 2: Training with Config

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

### Example 3: Override from Command Line

```bash
# Override specific settings
MODEL_NAME=gpt2 NUM_EPOCHS=1 python examples/quickstart.py

# Or export
export BATCH_SIZE=8
python examples/faithfulness_detection.py
```

## 🔧 Customization

### For Development
Edit `.env`:
```bash
DEBUG=true
LOG_LEVEL=DEBUG
BATCH_SIZE=2
NUM_EPOCHS=1
```

### For Production
Edit `.env`:
```bash
DEBUG=false
LOG_LEVEL=INFO
BATCH_SIZE=8
NUM_EPOCHS=3
WANDB_ENABLED=true
USE_FP16=true
```

### For Quick Testing
Edit `.env`:
```bash
SQUAD_TRAIN_ANSWERABLE=100
SQUAD_TRAIN_UNANSWERABLE=50
NUM_EPOCHS=1
```

## 🔒 Security Notes

1. ✅ `.env` is already in `.gitignore` - won't be committed
2. ✅ `.env.example` is safe to commit - no secrets
3. ⚠️ Never share your `.env` file or commit it
4. 🔑 Store API tokens securely in `.env`

## 📚 Documentation

- **Quick Reference**: See `.env.example` for all settings
- **Complete Guide**: Read `ENV_GUIDE.md`
- **API Reference**: Check `hallucination_detection/utils/config.py`
- **Example**: Run `examples/config_usage.py`

## ✨ Key Features

1. **Type-Safe**: Automatic type conversion (int, float, bool)
2. **Defaults**: Sensible defaults for all settings
3. **Override**: Environment variables take precedence
4. **Flexible**: Load from custom files
5. **Documented**: Every setting explained in `.env.example`

## 🎯 Next Steps

1. ✅ Install dependencies: `pip install -e .`
2. ✅ Edit `.env` with your settings
3. ✅ Add your HF token if needed
4. ✅ Run example: `python examples/config_usage.py`
5. ✅ Start using in your code!

## 📖 Related Files

- `.env.example` - Template (commit this)
- `.env` - Your config (DON'T commit)
- `ENV_GUIDE.md` - Full documentation
- `examples/config_usage.py` - Usage example
- `hallucination_detection/utils/config.py` - Implementation
- `.gitignore` - Updated to exclude `.env`
- `requirements.txt` - Updated with `python-dotenv`

---

**Status**: ✅ Complete and ready to use!  
**Security**: 🔒 Protected by .gitignore  
**Documentation**: 📚 Comprehensive guides included
