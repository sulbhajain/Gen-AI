"""
Training Utilities

Helper functions for training BatchEnsemble models and hallucination detectors.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


class TextDataset(Dataset):
    """
    Simple text dataset for fine-tuning.
    
    Args:
        texts: List of input texts
        labels: Optional list of labels (for supervised tasks)
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        tokenizer=None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Encode input
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        # Add label if available
        if self.labels is not None:
            label = self.labels[idx]
            label_encoding = self.tokenizer(
                label,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item["labels"] = label_encoding["input_ids"].squeeze()
        
        return item


class SQuADDataset(Dataset):
    """
    Dataset for SQuAD-style question answering.
    
    Format: [INST] Context: {context} Question: {question} [/INST] {answer}
    """
    
    def __init__(
        self,
        contexts: List[str],
        questions: List[str],
        answers: List[str],
        tokenizer,
        max_length: int = 512,
        instruction_template: Optional[str] = None,
    ):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default instruction from paper
        if instruction_template is None:
            instruction_template = (
                "[INST] Answer the question based only on the given context. "
                "Keep the answer short. If the answer is not in the context or "
                "if you are unsure, respond with 'I don't know'. "
                "Context: {context} Question: {question} [/INST]"
            )
        self.instruction_template = instruction_template
    
    def __len__(self) -> int:
        return len(self.contexts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Format prompt
        prompt = self.instruction_template.format(
            context=context,
            question=question,
        )
        full_text = prompt + " " + answer
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels (same as input_ids for language modeling)
        labels = encoding["input_ids"].clone()
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class MMLUDataset(Dataset):
    """
    Dataset for MMLU-style multiple choice questions.
    
    Format: [INST] Question: {question} Options: A) ... B) ... C) ... D) ... [/INST] {answer}
    """
    
    def __init__(
        self,
        questions: List[str],
        choices: List[List[str]],
        answers: List[str],
        tokenizer,
        max_length: int = 512,
    ):
        self.questions = questions
        self.choices = choices
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        question = self.questions[idx]
        choices = self.choices[idx]
        answer = self.answers[idx]
        
        # Format prompt
        options_str = "\n".join([
            f"{chr(65+i)}) {choice}"
            for i, choice in enumerate(choices)
        ])
        
        prompt = (
            f"[INST] Question: {question}\n"
            f"Options:\n{options_str}\n"
            f"[/INST]"
        )
        full_text = prompt + " " + answer
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Compute perplexity
    loss = torch.nn.functional.cross_entropy(
        torch.tensor(predictions).view(-1, predictions.shape[-1]),
        torch.tensor(labels).view(-1),
        ignore_index=-100,
    )
    perplexity = torch.exp(loss).item()
    
    return {
        "perplexity": perplexity,
    }


def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    num_training_steps: int,
    num_warmup_steps: Optional[int] = None,
    weight_decay: float = 0.01,
) -> Tuple[torch.optim.Optimizer, object]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        learning_rate: Peak learning rate
        num_training_steps: Total training steps
        num_warmup_steps: Warmup steps (default: 10% of total)
        weight_decay: Weight decay for regularization
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    if num_warmup_steps is None:
        num_warmup_steps = int(0.1 * num_training_steps)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    return optimizer, scheduler


def train_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    gradient_accumulation_steps: int = 1,
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        gradient_accumulation_steps: Steps to accumulate gradients
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for i, batch in enumerate(progress_bar):
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    return total_loss / num_batches


def evaluate(
    model,
    eval_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        eval_loader: Evaluation data loader
        device: Device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    
    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
    }
