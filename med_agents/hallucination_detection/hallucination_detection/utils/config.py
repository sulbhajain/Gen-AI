"""
Environment Configuration Loader

Loads environment variables from .env file and provides type-safe access.
"""

import os
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv


class Config:
    """
    Configuration manager for hallucination detection package.
    
    Loads settings from .env file and environment variables.
    Environment variables take precedence over .env file.
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            env_file: Path to .env file (default: .env in package root)
        """
        if env_file is None:
            # Look for .env in package root
            package_root = Path(__file__).parent.parent
            env_file = package_root / ".env"
        
        # Load .env file if it exists
        if os.path.exists(env_file):
            load_dotenv(env_file)
    
    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get string value from environment."""
        return os.getenv(key, default)
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer value from environment."""
        value = os.getenv(key)
        return int(value) if value is not None else default
    
    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get float value from environment."""
        value = os.getenv(key)
        return float(value) if value is not None else default
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean value from environment."""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # Model Configuration
    @property
    def model_name(self) -> str:
        return self.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
    
    @property
    def num_ensemble_members(self) -> int:
        return self.get_int("NUM_ENSEMBLE_MEMBERS", 5)
    
    @property
    def hf_token(self) -> Optional[str]:
        return self.get("HF_TOKEN")
    
    @property
    def hf_home(self) -> str:
        return self.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    # Training Configuration
    @property
    def output_dir(self) -> str:
        return self.get("OUTPUT_DIR", "./output")
    
    @property
    def checkpoint_dir(self) -> str:
        return self.get("CHECKPOINT_DIR", "./checkpoints")
    
    @property
    def detector_output_dir(self) -> str:
        return self.get("DETECTOR_OUTPUT_DIR", "./detector_output")
    
    @property
    def learning_rate(self) -> float:
        return self.get_float("LEARNING_RATE", 2e-4)
    
    @property
    def num_epochs(self) -> int:
        return self.get_int("NUM_EPOCHS", 3)
    
    @property
    def batch_size(self) -> int:
        return self.get_int("BATCH_SIZE", 4)
    
    @property
    def gradient_accumulation_steps(self) -> int:
        return self.get_int("GRADIENT_ACCUMULATION_STEPS", 4)
    
    @property
    def max_length(self) -> int:
        return self.get_int("MAX_LENGTH", 512)
    
    # LoRA Configuration
    @property
    def lora_r(self) -> int:
        return self.get_int("LORA_R", 8)
    
    @property
    def lora_alpha(self) -> int:
        return self.get_int("LORA_ALPHA", 32)
    
    @property
    def lora_dropout(self) -> float:
        return self.get_float("LORA_DROPOUT", 0.1)
    
    # Dataset Configuration
    @property
    def datasets_cache(self) -> str:
        return self.get("DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
    
    @property
    def squad_train_answerable(self) -> int:
        return self.get_int("SQUAD_TRAIN_ANSWERABLE", 28000)
    
    @property
    def squad_train_unanswerable(self) -> int:
        return self.get_int("SQUAD_TRAIN_UNANSWERABLE", 14000)
    
    @property
    def squad_val_size(self) -> int:
        return self.get_int("SQUAD_VAL_SIZE", 2000)
    
    @property
    def squad_test_size(self) -> int:
        return self.get_int("SQUAD_TEST_SIZE", 5000)
    
    @property
    def mmlu_train_size(self) -> int:
        return self.get_int("MMLU_TRAIN_SIZE", 40000)
    
    @property
    def mmlu_val_size(self) -> int:
        return self.get_int("MMLU_VAL_SIZE", 2000)
    
    @property
    def mmlu_test_size(self) -> int:
        return self.get_int("MMLU_TEST_SIZE", 5000)
    
    # Compute Configuration
    @property
    def cuda_visible_devices(self) -> Optional[str]:
        return self.get("CUDA_VISIBLE_DEVICES")
    
    @property
    def device(self) -> str:
        return self.get("DEVICE", "cuda")
    
    @property
    def use_fp16(self) -> bool:
        return self.get_bool("USE_FP16", True)
    
    @property
    def use_gradient_checkpointing(self) -> bool:
        return self.get_bool("USE_GRADIENT_CHECKPOINTING", True)
    
    # Logging
    @property
    def wandb_project(self) -> str:
        return self.get("WANDB_PROJECT", "hallucination-detection")
    
    @property
    def wandb_api_key(self) -> Optional[str]:
        return self.get("WANDB_API_KEY")
    
    @property
    def wandb_enabled(self) -> bool:
        return self.get_bool("WANDB_ENABLED", False)
    
    @property
    def log_level(self) -> str:
        return self.get("LOG_LEVEL", "INFO")
    
    # Detection Configuration
    @property
    def classifier_type(self) -> str:
        return self.get("CLASSIFIER_TYPE", "rf")
    
    @property
    def scale_features(self) -> bool:
        return self.get_bool("SCALE_FEATURES", True)
    
    @property
    def temperature(self) -> float:
        return self.get_float("TEMPERATURE", 1.0)
    
    @property
    def validation_split(self) -> float:
        return self.get_float("VALIDATION_SPLIT", 0.2)
    
    # Development
    @property
    def debug(self) -> bool:
        return self.get_bool("DEBUG", False)
    
    @property
    def random_seed(self) -> int:
        return self.get_int("RANDOM_SEED", 42)
    
    @property
    def num_workers(self) -> int:
        return self.get_int("NUM_WORKERS", 4)


# Global config instance
config = Config()


def get_config(env_file: Optional[str] = None) -> Config:
    """
    Get configuration instance.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Config instance
    """
    return Config(env_file) if env_file else config
