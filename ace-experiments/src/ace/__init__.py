"""
ACE Framework: Main orchestrator for Agentic Context Engineering.

Coordinates Generator, Reflector, and Curator for context adaptation.
"""

from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import time

from src.ace.playbook import Playbook
from src.ace.generator import Generator
from src.ace.reflector import Reflector
from src.ace.curator import Curator


class ACEFramework:
    """
    Main ACE (Agentic Context Engineering) framework.
    
    Coordinates the three components (Generator, Reflector, Curator)
    to evolve contexts over time through offline or online adaptation.
    """
    
    def __init__(self, 
                 generator_model: Optional[str] = None,
                 reflector_model: Optional[str] = None,
                 curator_model: Optional[str] = None,
                 playbook_sections: Optional[List[str]] = None):
        """
        Initialize ACE framework.
        
        Args:
            generator_model: Model for Generator (defaults to Ollama if not provided)
            reflector_model: Model for Reflector (defaults to Ollama if not provided)
            curator_model: Model for Curator (defaults to Ollama if not provided)
            playbook_sections: Custom playbook sections
        """
        self.playbook = Playbook(sections=playbook_sections)
        self.generator = Generator(model=generator_model)
        self.reflector = Reflector(model=reflector_model)
        self.curator = Curator(model=curator_model)
        
        self.metrics = {
            'total_samples': 0,
            'total_time': 0.0,
            'delta_updates': 0,
            'reflections': 0
        }
    
    def offline_adaptation(self,
                          train_data: List[Dict[str, Any]],
                          num_epochs: int = 5,
                          batch_size: int = 1,
                          deduplicate_frequency: int = 10) -> Playbook:
        """
        Offline context adaptation on training data.
        
        Args:
            train_data: List of training samples with 'task' and 'answer' fields
            num_epochs: Number of training epochs
            batch_size: Samples per delta update
            deduplicate_frequency: How often to deduplicate playbook
            
        Returns:
            Evolved playbook
        """
        print(f"Starting offline adaptation: {len(train_data)} samples, {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
            
            # Process in batches
            for batch_start in tqdm(range(0, len(train_data), batch_size),
                                   desc=f"Epoch {epoch + 1}"):
                batch_end = min(batch_start + batch_size, len(train_data))
                batch = train_data[batch_start:batch_end]
                
                # Generate trajectories
                trajectories = []
                for sample in batch:
                    start_time = time.time()
                    
                    trajectory = self.generator.generate(
                        task=sample['task'],
                        playbook=self.playbook,
                        task_type=sample.get('task_type', 'qa')
                    )
                    
                    self.metrics['total_time'] += time.time() - start_time
                    self.metrics['total_samples'] += 1
                    
                    trajectories.append({
                        **trajectory,
                        'ground_truth': sample.get('answer')
                    })
                
                # Reflect on trajectories
                reflections = []
                for i, trajectory in enumerate(trajectories):
                    reflection = self.reflector.reflect(
                        trajectory=trajectory,
                        ground_truth=trajectory.get('ground_truth'),
                        used_bullets=trajectory.get('used_bullets', [])
                    )
                    reflections.append(reflection)
                    self.metrics['reflections'] += 1
                
                # Curate and update playbook
                deltas = self.curator.curate(
                    insights=reflections,
                    current_playbook=self.playbook,
                    task_context=batch[0]['task'] if batch else None
                )
                
                self.playbook.update(deltas)
                self.metrics['delta_updates'] += len(deltas)
                
                # Apply bullet feedback
                for reflection in reflections:
                    if 'bullet_feedback' in reflection:
                        self.curator.apply_bullet_feedback(
                            self.playbook,
                            reflection['bullet_feedback']
                        )
                
                # Periodic deduplication
                if (batch_start // batch_size) % deduplicate_frequency == 0:
                    self.playbook.deduplicate()
            
            # End of epoch deduplication and pruning
            self.playbook.deduplicate()
            self.curator.prune_harmful_bullets(self.playbook)
            
            stats = self.playbook.get_stats()
            print(f"Epoch {epoch + 1} complete. Playbook stats: {stats}")
        
        return self.playbook
    
    def online_adaptation(self,
                         test_data: List[Dict[str, Any]],
                         warmup_playbook: Optional[Playbook] = None,
                         update_frequency: int = 1) -> Tuple[List[Dict[str, Any]], Playbook]:
        """
        Online context adaptation during test-time evaluation.
        
        Args:
            test_data: Test samples (evaluated sequentially)
            warmup_playbook: Optional pre-trained playbook
            update_frequency: How often to update playbook (in samples)
            
        Returns:
            Tuple of (results, evolved_playbook)
        """
        if warmup_playbook:
            self.playbook = warmup_playbook
        
        print(f"Starting online adaptation: {len(test_data)} samples")
        
        results = []
        reflection_buffer = []
        
        for i, sample in enumerate(tqdm(test_data, desc="Online evaluation")):
            start_time = time.time()
            
            # Generate prediction
            trajectory = self.generator.generate(
                task=sample['task'],
                playbook=self.playbook,
                task_type=sample.get('task_type', 'qa')
            )
            
            # Reflect (with or without ground truth)
            reflection = self.reflector.reflect(
                trajectory=trajectory,
                ground_truth=sample.get('answer'),
                execution_feedback=sample.get('feedback'),
                used_bullets=trajectory.get('used_bullets', []),
                refine=False  # Skip refinement for speed in online mode
            )
            
            reflection_buffer.append(reflection)
            
            # Store result
            results.append({
                'task': sample['task'],
                'prediction': trajectory['answer'],
                'ground_truth': sample.get('answer'),
                'reasoning': trajectory['reasoning'],
                'time': time.time() - start_time
            })
            
            self.metrics['total_samples'] += 1
            self.metrics['total_time'] += time.time() - start_time
            
            # Periodic playbook update
            if (i + 1) % update_frequency == 0 and reflection_buffer:
                deltas = self.curator.curate(
                    insights=reflection_buffer,
                    current_playbook=self.playbook
                )
                
                self.playbook.update(deltas)
                self.metrics['delta_updates'] += len(deltas)
                
                # Apply feedback
                for reflection in reflection_buffer:
                    if 'bullet_feedback' in reflection:
                        self.curator.apply_bullet_feedback(
                            self.playbook,
                            reflection['bullet_feedback']
                        )
                
                reflection_buffer = []
        
        # Final update if buffer has items
        if reflection_buffer:
            deltas = self.curator.curate(
                insights=reflection_buffer,
                current_playbook=self.playbook
            )
            self.playbook.update(deltas)
        
        # Final cleanup
        self.playbook.deduplicate()
        
        return results, self.playbook
    
    def evaluate(self, test_data: List[Dict[str, Any]], 
                 use_playbook: bool = True) -> Dict[str, Any]:
        """
        Evaluate on test data without adaptation.
        
        Args:
            test_data: Test samples
            use_playbook: Whether to use current playbook
            
        Returns:
            Evaluation metrics
        """
        print(f"Evaluating on {len(test_data)} samples")
        
        correct = 0
        total = 0
        
        for sample in tqdm(test_data, desc="Evaluation"):
            trajectory = self.generator.generate(
                task=sample['task'],
                playbook=self.playbook if use_playbook else None,
                task_type=sample.get('task_type', 'qa')
            )
            
            if 'answer' in sample:
                is_correct = self._check_correctness(
                    trajectory['answer'],
                    sample['answer']
                )
                if is_correct:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'playbook_stats': self.playbook.get_stats()
        }
    
    def _check_correctness(self, prediction: Optional[str], ground_truth: Optional[str]) -> bool:
        """Check if prediction matches ground truth."""
        if not prediction or not ground_truth:
            return False

        pred_norm = prediction.strip().lower()
        gt_norm = ground_truth.strip().lower()

        return pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
    
    def save_playbook(self, filepath: str):
        """Save current playbook to file."""
        self.playbook.save(filepath)
    
    def load_playbook(self, filepath: str):
        """Load playbook from file."""
        self.playbook = Playbook.load(filepath)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get framework metrics."""
        return {
            **self.metrics,
            'avg_time_per_sample': self.metrics['total_time'] / max(self.metrics['total_samples'], 1),
            'avg_deltas_per_sample': self.metrics['delta_updates'] / max(self.metrics['total_samples'], 1)
        }
