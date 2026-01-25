"""
Example: Using Environment Configuration

Demonstrates how to use the .env file and Config class.
"""

from hallucination_detection.utils import config, get_config

def main():
    print("="*60)
    print("Environment Configuration Example")
    print("="*60)
    print()
    
    # Use global config instance
    print("1. Model Configuration:")
    print(f"   Model Name: {config.model_name}")
    print(f"   Ensemble Members: {config.num_ensemble_members}")
    print(f"   HF Token: {'Set' if config.hf_token else 'Not Set'}")
    print()
    
    print("2. Training Configuration:")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Output Dir: {config.output_dir}")
    print()
    
    print("3. LoRA Configuration:")
    print(f"   Rank (r): {config.lora_r}")
    print(f"   Alpha: {config.lora_alpha}")
    print(f"   Dropout: {config.lora_dropout}")
    print()
    
    print("4. Compute Configuration:")
    print(f"   Device: {config.device}")
    print(f"   Use FP16: {config.use_fp16}")
    print(f"   Gradient Checkpointing: {config.use_gradient_checkpointing}")
    print()
    
    print("5. Detection Configuration:")
    print(f"   Classifier: {config.classifier_type}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Scale Features: {config.scale_features}")
    print()
    
    # You can also load from a custom .env file
    print("6. Loading from custom .env file:")
    custom_config = get_config("custom.env")
    print(f"   (Will use default values if file doesn't exist)")
    print()
    
    # Using config in your code
    print("="*60)
    print("Usage Example:")
    print("="*60)
    print("""
from hallucination_detection import BatchEnsembleModel
from hallucination_detection.utils import config

# Use config values
model = BatchEnsembleModel.from_pretrained(
    config.model_name,
    num_ensemble_members=config.num_ensemble_members,
    lora_r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
)

# Training with config
from hallucination_detection.training import BatchEnsembleTrainer

trainer = BatchEnsembleTrainer(
    model=model,
    train_dataset=dataset,
    output_dir=config.output_dir,
    learning_rate=config.learning_rate,
    num_epochs=config.num_epochs,
    batch_size=config.batch_size,
)
    """)


if __name__ == "__main__":
    main()
