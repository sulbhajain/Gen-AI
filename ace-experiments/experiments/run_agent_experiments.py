"""
Main experiment runner for ACE framework.

Supports offline and online adaptation experiments on various datasets.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from datasets import load_dataset
from src.ace import ACEFramework
from src.utils.metrics import evaluate_predictions, save_results


def load_data(dataset_name: str, split: str = "train", 
              max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset for experiments.
    
    Args:
        dataset_name: Name of dataset (hotpotqa, financial_phrasebank, etc.)
        split: Dataset split (train/validation/test)
        max_samples: Maximum number of samples to load
        
    Returns:
        List of samples with 'task' and optionally 'answer' fields
    """
    print(f"Loading {dataset_name} dataset ({split} split)...")
    
    if dataset_name == "hotpotqa":
        dataset = load_dataset("hotpot_qa", "fullwiki", split=split)
        
        samples = []
        for item in dataset:
            if max_samples and len(samples) >= max_samples:
                break
            
            samples.append({
                'task': item['question'],
                'answer': item['answer'],
                'task_type': 'qa',
                'context': ' '.join([' '.join(sent) for sent in item['context']['sentences']])
            })
    
    elif dataset_name == "financial_phrasebank":
        dataset = load_dataset("financial_phrasebank", "sentences_allagree", split="train")
        
        # Split into train/test manually
        total = len(dataset)
        train_size = int(0.8 * total)
        
        if split == "train":
            dataset = dataset.select(range(train_size))
        else:
            dataset = dataset.select(range(train_size, total))
        
        samples = []
        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        
        for item in dataset:
            if max_samples and len(samples) >= max_samples:
                break
            
            samples.append({
                'task': f"Classify the sentiment of this financial statement: {item['sentence']}",
                'answer': label_map[item['label']],
                'task_type': 'classification'
            })
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    print(f"Loaded {len(samples)} samples")
    return samples


def run_offline_experiment(args):
    """Run offline adaptation experiment."""
    print("=" * 80)
    print("OFFLINE ADAPTATION EXPERIMENT")
    print("=" * 80)
    
    # Load data
    train_data = load_data(args.dataset, "train", args.num_samples)
    test_data = load_data(args.dataset, "test", args.test_samples)
    
    # Initialize ACE
    ace = ACEFramework(
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model
    )
    
    # Baseline evaluation (no playbook)
    print("\n--- Baseline Evaluation (No Playbook) ---")
    baseline_results = ace.evaluate(test_data, use_playbook=False)
    print(f"Baseline Accuracy: {baseline_results['accuracy']:.3f}")
    
    # Offline adaptation
    print("\n--- Offline Adaptation ---")
    evolved_playbook = ace.offline_adaptation(
        train_data=train_data,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save playbook
    playbook_path = Path(args.output_dir) / f"playbook_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ace.save_playbook(str(playbook_path))
    print(f"\nPlaybook saved to: {playbook_path}")
    
    # Evaluation with evolved playbook
    print("\n--- Evaluation with Evolved Playbook ---")
    ace_results = ace.evaluate(test_data, use_playbook=True)
    print(f"ACE Accuracy: {ace_results['accuracy']:.3f}")
    print(f"Improvement: {(ace_results['accuracy'] - baseline_results['accuracy']):.3f}")
    
    # Save results
    results = {
        'experiment_type': 'offline',
        'dataset': args.dataset,
        'baseline': baseline_results,
        'ace': ace_results,
        'improvement': ace_results['accuracy'] - baseline_results['accuracy'],
        'metrics': ace.get_metrics(),
        'playbook_stats': evolved_playbook.get_stats()
    }
    
    results_path = Path(args.output_dir) / f"results_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def run_online_experiment(args):
    """Run online adaptation experiment."""
    print("=" * 80)
    print("ONLINE ADAPTATION EXPERIMENT")
    print("=" * 80)
    
    # Load data
    test_data = load_data(args.dataset, "test", args.num_samples)
    
    # Initialize ACE
    ace = ACEFramework(
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model
    )
    
    # Optional: warmup with offline playbook
    warmup_playbook = None
    if args.warmup_playbook:
        print(f"Loading warmup playbook from: {args.warmup_playbook}")
        ace.load_playbook(args.warmup_playbook)
        warmup_playbook = ace.playbook
    
    # Online adaptation
    print("\n--- Online Adaptation ---")
    results, evolved_playbook = ace.online_adaptation(
        test_data=test_data,
        warmup_playbook=warmup_playbook,
        update_frequency=args.update_frequency
    )
    
    # Calculate accuracy
    correct = sum(1 for r in results if r.get('ground_truth') and 
                  r['prediction'].lower().strip() == r['ground_truth'].lower().strip())
    accuracy = correct / len(results) if results else 0
    
    print(f"\nOnline Adaptation Accuracy: {accuracy:.3f}")
    
    # Save playbook
    playbook_path = Path(args.output_dir) / f"playbook_online_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ace.save_playbook(str(playbook_path))
    print(f"Playbook saved to: {playbook_path}")
    
    # Save results
    results_summary = {
        'experiment_type': 'online',
        'dataset': args.dataset,
        'accuracy': accuracy,
        'num_samples': len(results),
        'metrics': ace.get_metrics(),
        'playbook_stats': evolved_playbook.get_stats()
    }
    
    results_path = Path(args.output_dir) / f"results_online_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Run ACE experiments")
    
    # Experiment settings
    parser.add_argument("--mode", type=str, choices=["offline", "online"], 
                       default="offline", help="Adaptation mode")
    parser.add_argument("--dataset", type=str, default="hotpotqa",
                       help="Dataset to use (hotpotqa, financial_phrasebank)")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples to use")
    parser.add_argument("--test_samples", type=int, default=50,
                       help="Number of test samples (offline mode)")
    
    # Model settings
    parser.add_argument("--generator_model", type=str, default="ollama:gemma3:1b",
                       help="Model for Generator")
    parser.add_argument("--reflector_model", type=str, default="ollama:qwen2.5:0.5b",
                       help="Model for Reflector")
    parser.add_argument("--curator_model", type=str, default="ollama:qwen2.5:0.5b",
                       help="Model for Curator")
    
    # Training settings (offline)
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs (offline mode)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for delta updates")
    
    # Online settings
    parser.add_argument("--update_frequency", type=int, default=1,
                       help="Update frequency for online mode")
    parser.add_argument("--warmup_playbook", type=str, default=None,
                       help="Path to warmup playbook for online mode")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    if args.mode == "offline":
        results = run_offline_experiment(args)
    else:
        results = run_online_experiment(args)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
