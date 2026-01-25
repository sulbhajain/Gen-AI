"""
Data Loading Utilities

Functions for loading and preprocessing datasets for hallucination detection.
"""

from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import random


def load_squad_dataset(
    include_unanswerable: bool = True,
    train_answerable: int = 28000,
    train_unanswerable: int = 14000,
    val_size: int = 2000,
    test_size: int = 5000,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Load and preprocess SQuAD dataset for hallucination detection.
    
    Following the paper's setup:
    - Train: 28k answerable + 14k unanswerable questions
    - Val: 2k mixed
    - Test: 5k unanswerable (for faithfulness hallucination detection)
    
    Args:
        include_unanswerable: Whether to include unanswerable questions
        train_answerable: Number of answerable questions for training
        train_unanswerable: Number of unanswerable questions for training
        val_size: Validation set size
        test_size: Test set size
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    random.seed(seed)
    
    if include_unanswerable:
        # Load SQuAD 2.0
        dataset = load_dataset("squad_v2")
    else:
        # Load SQuAD 1.1
        dataset = load_dataset("squad")
    
    train_data = dataset["train"]
    val_data = dataset["validation"]
    
    # Separate answerable and unanswerable
    train_answerable_data = []
    train_unanswerable_data = []
    
    for example in train_data:
        is_answerable = len(example["answers"]["text"]) > 0
        
        item = {
            "context": example["context"],
            "question": example["question"],
            "answer": example["answers"]["text"][0] if is_answerable else "I don't know",
            "is_answerable": is_answerable,
        }
        
        if is_answerable:
            train_answerable_data.append(item)
        else:
            train_unanswerable_data.append(item)
    
    # Shuffle and sample
    random.shuffle(train_answerable_data)
    random.shuffle(train_unanswerable_data)
    
    train_answerable_sample = train_answerable_data[:train_answerable]
    train_unanswerable_sample = train_unanswerable_data[:train_unanswerable]
    
    # Combine and shuffle training data
    train_combined = train_answerable_sample + train_unanswerable_sample
    random.shuffle(train_combined)
    
    # Split into train and validation
    train_split = train_combined[:-val_size]
    val_split = train_combined[-val_size:]
    
    # Process test data (unanswerable questions only)
    test_data = []
    for example in val_data:
        is_answerable = len(example["answers"]["text"]) > 0
        
        if not is_answerable:
            test_data.append({
                "context": example["context"],
                "question": example["question"],
                "answer": "I don't know",
                "is_answerable": False,
            })
    
    random.shuffle(test_data)
    test_split = test_data[:test_size]
    
    return {
        "train": train_split,
        "validation": val_split,
        "test": test_split,
    }


def load_mmlu_dataset(
    train_size: int = 40000,
    val_size: int = 2000,
    test_size: int = 5000,
    seed: int = 42,
) -> Dict[str, List]:
    """
    Load and preprocess MMLU dataset for factual hallucination detection.
    
    Following the paper's setup:
    - Train: 40k from 'all' and 'auxiliary_train'
    - Val: 2k
    - Test: 5k from test split
    
    Args:
        train_size: Training set size
        val_size: Validation set size
        test_size: Test set size
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    random.seed(seed)
    
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all")
    
    # Combine training data
    train_data = []
    
    if "train" in dataset:
        for example in dataset["train"]:
            train_data.append({
                "question": example["question"],
                "choices": example["choices"],
                "answer": example["answer"],
            })
    
    if "auxiliary_train" in dataset:
        for example in dataset["auxiliary_train"]:
            train_data.append({
                "question": example["question"],
                "choices": example["choices"],
                "answer": example["answer"],
            })
    
    # Shuffle and split
    random.shuffle(train_data)
    
    train_split = train_data[:train_size]
    val_split = train_data[train_size:train_size + val_size]
    
    # Process test data
    test_data = []
    if "test" in dataset:
        for example in dataset["test"]:
            test_data.append({
                "question": example["question"],
                "choices": example["choices"],
                "answer": example["answer"],
            })
    
    random.shuffle(test_data)
    test_split = test_data[:test_size]
    
    return {
        "train": train_split,
        "validation": val_split,
        "test": test_split,
    }


def prepare_squad_for_training(
    data: List[Dict],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Prepare SQuAD data for training.
    
    Args:
        data: List of SQuAD examples
        
    Returns:
        Tuple of (contexts, questions, answers)
    """
    contexts = []
    questions = []
    answers = []
    
    for example in data:
        contexts.append(example["context"])
        questions.append(example["question"])
        answers.append(example["answer"])
    
    return contexts, questions, answers


def create_hallucination_labels(
    data: List[Dict],
    predictions: List[str],
) -> List[int]:
    """
    Create binary labels for hallucination detection.
    
    For SQuAD:
    - Answerable question but model says "I don't know": not hallucinated (0)
    - Unanswerable question but model gives answer: hallucinated (1)
    - Correct answer: not hallucinated (0)
    - Incorrect answer: hallucinated (1)
    
    Args:
        data: List of examples with ground truth
        predictions: List of model predictions
        
    Returns:
        Binary labels (0=correct, 1=hallucinated)
    """
    labels = []
    
    for example, prediction in zip(data, predictions):
        is_answerable = example.get("is_answerable", True)
        ground_truth = example["answer"]
        
        if is_answerable:
            # Check if answer is correct
            if prediction.lower().strip() == ground_truth.lower().strip():
                labels.append(0)  # Correct
            else:
                labels.append(1)  # Hallucinated (wrong answer)
        else:
            # Unanswerable question
            if "don't know" in prediction.lower() or "i don't know" in prediction.lower():
                labels.append(0)  # Correct (refused to answer)
            else:
                labels.append(1)  # Hallucinated (answered when shouldn't)
    
    return labels
