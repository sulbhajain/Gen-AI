"""
Example: Faithfulness Hallucination Detection on SQuAD

This example demonstrates how to:
1. Load and fine-tune a BatchEnsemble model on SQuAD
2. Compute uncertainty estimates
3. Train a hallucination detector
4. Evaluate on faithfulness hallucination detection

Based on the paper's experiments on SQuAD 2.0.
"""

import torch
from hallucination_detection import BatchEnsembleModel, UncertaintyEstimator, HallucinationDetector
from hallucination_detection.utils import load_squad_dataset, prepare_squad_for_training
from hallucination_detection.training import BatchEnsembleTrainer, SQuADDataset

# Configuration
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_NAME = "google/gemma-2b"
NUM_ENSEMBLE_MEMBERS = 5
OUTPUT_DIR = "./squad_output"
DETECTOR_OUTPUT = "./detector_squad.pkl"

def main():
    print("="*60)
    print("Faithfulness Hallucination Detection Example")
    print("="*60)
    
    # Step 1: Load Dataset
    print("\n[1/6] Loading SQuAD dataset...")
    data = load_squad_dataset(
        include_unanswerable=True,
        train_answerable=28000,
        train_unanswerable=14000,
        val_size=2000,
        test_size=5000,
    )
    
    print(f"Train size: {len(data['train'])}")
    print(f"Val size: {len(data['validation'])}")
    print(f"Test size: {len(data['test'])}")
    
    # Step 2: Create BatchEnsemble Model
    print(f"\n[2/6] Loading BatchEnsemble model: {MODEL_NAME}")
    model = BatchEnsembleModel.from_pretrained(
        MODEL_NAME,
        num_ensemble_members=NUM_ENSEMBLE_MEMBERS,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    # Step 3: Fine-tune Model
    print("\n[3/6] Fine-tuning model on SQuAD...")
    
    # Prepare datasets
    train_contexts, train_questions, train_answers = prepare_squad_for_training(data['train'])
    val_contexts, val_questions, val_answers = prepare_squad_for_training(data['validation'])
    
    train_dataset = SQuADDataset(
        contexts=train_contexts,
        questions=train_questions,
        answers=train_answers,
        tokenizer=model.tokenizer,
    )
    
    val_dataset = SQuADDataset(
        contexts=val_contexts,
        questions=val_questions,
        answers=val_answers,
        tokenizer=model.tokenizer,
    )
    
    # Create trainer
    trainer = BatchEnsembleTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        num_epochs=3,
        batch_size=4,
        gradient_accumulation_steps=4,
    )
    
    # Train
    history = trainer.train()
    
    # Step 4: Compute Uncertainty Estimates
    print("\n[4/6] Computing uncertainty estimates on test set...")
    
    # Prepare test prompts
    test_contexts, test_questions, test_answers = prepare_squad_for_training(data['test'])
    
    test_prompts = [
        f"[INST] Answer the question based only on the given context. "
        f"Keep the answer short. If the answer is not in the context or "
        f"if you are unsure, respond with 'I don't know'. "
        f"Context: {ctx} Question: {q} [/INST]"
        for ctx, q in zip(test_contexts, test_questions)
    ]
    
    # Compute uncertainties
    estimator = UncertaintyEstimator(model)
    uncertainties = estimator.compute_uncertainties(
        test_prompts,
        return_tokens=True,
        batch_size=8,
    )
    
    # Step 5: Generate Predictions and Create Labels
    print("\n[5/6] Generating predictions...")
    
    predictions = []
    for prompt in test_prompts[:100]:  # Sample for demo
        outputs = model.generate_ensemble(prompt, max_length=512)
        # Use first ensemble member's output
        predictions.append(outputs[0])
    
    # Create hallucination labels
    # For unanswerable questions:
    # - If model says "I don't know": not hallucinated (0)
    # - If model gives an answer: hallucinated (1)
    labels = []
    for pred in predictions:
        if "don't know" in pred.lower() or "i don't know" in pred.lower():
            labels.append(0)  # Correct (refused to answer)
        else:
            labels.append(1)  # Hallucinated
    
    print(f"Hallucination rate: {sum(labels) / len(labels) * 100:.2f}%")
    
    # Step 6: Train Hallucination Detector
    print("\n[6/6] Training hallucination detector...")
    
    # Use uncertainties from first 100 examples
    detector = HallucinationDetector(
        classifier_type="rf",  # Random Forest
        scale_features=True,
    )
    
    metrics = detector.train(
        uncertainties[:100],
        labels,
        validation_split=0.2,
    )
    
    print("\nFinal Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    # Save detector
    detector.save(DETECTOR_OUTPUT)
    
    print(f"\n✓ Training complete! Model saved to {OUTPUT_DIR}")
    print(f"✓ Detector saved to {DETECTOR_OUTPUT}")
    
    # Example: Analyze snowballing effect
    print("\n" + "="*60)
    print("Analyzing Snowballing Effect")
    print("="*60)
    
    example_prompt = test_prompts[0]
    analysis = estimator.analyze_snowballing_effect(
        example_prompt,
        max_new_tokens=30,
    )
    
    print("\nPer-token uncertainty:")
    for i, entropy in enumerate(analysis['predictive_entropy'][:10]):
        print(f"Token {i}: {entropy:.4f}")


if __name__ == "__main__":
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No GPU detected. Training will be slow.")
    
    main()
