"""
Trainer for BatchEnsemble Models

Main training class for fine-tuning BatchEnsemble models with LoRA.
"""

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer as HFTrainer
from typing import Optional, Dict, List
import os
import json

from hallucination_detection.models.batch_ensemble import BatchEnsembleModel
from hallucination_detection.training.utils import (
    create_optimizer_and_scheduler,
    train_epoch,
    evaluate,
)


class BatchEnsembleTrainer:
    """
    Trainer for BatchEnsemble models.
    
    Handles:
    1. LoRA fine-tuning of shared weights
    2. Merging LoRA weights
    3. Adding fast weights for ensemble members
    
    Args:
        model: BatchEnsembleModel to train
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        output_dir: Directory to save checkpoints
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for regularization
        save_steps: Steps between checkpoints
        logging_steps: Steps between logging
    """
    
    def __init__(
        self,
        model: BatchEnsembleModel,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./output",
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        logging_steps: int = 10,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare model for training
        self.model.prepare_for_training()
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Returns:
            Dictionary with training history
        """
        device = self.model.device
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            eval_loader = None
        
        # Calculate total training steps
        num_training_steps = (
            len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        )
        
        # Create optimizer and scheduler
        optimizer, scheduler = create_optimizer_and_scheduler(
            self.model.model,
            learning_rate=self.learning_rate,
            num_training_steps=num_training_steps,
            num_warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
        )
        
        # Training history
        history = {
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
        }
        
        print(f"\n{'='*50}")
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"{'='*50}\n")
        
        # Training loop
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = train_epoch(
                self.model.model,
                train_loader,
                optimizer,
                scheduler,
                device,
                self.gradient_accumulation_steps,
            )
            
            history["train_loss"].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            if eval_loader is not None:
                eval_metrics = evaluate(self.model.model, eval_loader, device)
                history["eval_loss"].append(eval_metrics["eval_loss"])
                history["perplexity"].append(eval_metrics["perplexity"])
                
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"Perplexity: {eval_metrics['perplexity']:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch+1}")
            self.model.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved to {checkpoint_dir}")
        
        print(f"\n{'='*50}")
        print("Training complete!")
        print(f"{'='*50}\n")
        
        # Save training history
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
        # Plot training curves if matplotlib is available
        try:
            self._plot_training_curves(history)
        except ImportError:
            print("matplotlib not installed, skipping curve plotting")
        
        # Merge LoRA and add fast weights
        print("Preparing model for ensemble inference...")
        self.model.merge_lora_and_add_fast_weights()
        
        # Save final model
        final_dir = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_dir)
        print(f"Final model saved to {final_dir}")
        
        return history
    
    def _plot_training_curves(self, history: Dict[str, List[float]]):
        """Plot and save training curves."""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot training loss
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if history['eval_loss']:
            axes[0].plot(epochs, history['eval_loss'], 'r-', label='Eval Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Evaluation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot perplexity
        if history['perplexity']:
            axes[1].plot(epochs, history['perplexity'], 'g-', label='Perplexity')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Perplexity')
            axes[1].set_title('Evaluation Perplexity')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {plot_path}")
        plt.close()
    
    def train_with_huggingface(self) -> HFTrainer:
        """
        Train using Hugging Face Trainer (alternative method).
        
        Returns:
            Trained HuggingFace Trainer
        """
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps or 100,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            save_total_limit=3,
            fp16=True,
            gradient_checkpointing=True,
        )
        
        # Create trainer
        trainer = HFTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        
        # Train
        trainer.train()
        
        # Merge LoRA and add fast weights
        self.model.merge_lora_and_add_fast_weights()
        
        # Save final model
        final_dir = os.path.join(self.output_dir, "final")
        self.model.save_pretrained(final_dir)
        
        return trainer


def train_hallucination_detector(
    model: BatchEnsembleModel,
    train_contexts: List[str],
    train_questions: List[str],
    train_answers: List[str],
    eval_contexts: Optional[List[str]] = None,
    eval_questions: Optional[List[str]] = None,
    eval_answers: Optional[List[str]] = None,
    output_dir: str = "./detector_output",
    **trainer_kwargs,
):
    """
    Train a model for hallucination detection on SQuAD-style data.
    
    Args:
        model: BatchEnsembleModel
        train_contexts: Training contexts
        train_questions: Training questions
        train_answers: Training answers
        eval_contexts: Evaluation contexts
        eval_questions: Evaluation questions
        eval_answers: Evaluation answers
        output_dir: Output directory
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Training history
    """
    from hallucination_detection.training.utils import SQuADDataset
    
    # Create datasets
    train_dataset = SQuADDataset(
        contexts=train_contexts,
        questions=train_questions,
        answers=train_answers,
        tokenizer=model.tokenizer,
    )
    
    eval_dataset = None
    if eval_contexts is not None:
        eval_dataset = SQuADDataset(
            contexts=eval_contexts,
            questions=eval_questions,
            answers=eval_answers,
            tokenizer=model.tokenizer,
        )
    
    # Create trainer
    trainer = BatchEnsembleTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        **trainer_kwargs,
    )
    
    # Train
    history = trainer.train()
    
    return history
