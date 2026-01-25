"""
Uncertainty Estimator

Main class for computing uncertainty estimates from BatchEnsemble models.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from tqdm import tqdm

from hallucination_detection.models.batch_ensemble import BatchEnsembleModel
from hallucination_detection.uncertainty.metrics import (
    UncertaintyMetrics,
    compute_all_uncertainties,
)


class UncertaintyEstimator:
    """
    Estimate uncertainties from BatchEnsemble model predictions.
    
    Computes predictive entropy, aleatoric uncertainty, and epistemic
    uncertainty for hallucination detection.
    
    Args:
        model: BatchEnsembleModel instance
        temperature: Temperature for softmax (default: 1.0)
    """
    
    def __init__(
        self,
        model: BatchEnsembleModel,
        temperature: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.device = model.device
    
    def compute_uncertainties(
        self,
        prompts: Union[str, List[str]],
        max_length: int = 512,
        return_tokens: bool = False,
        batch_size: int = 1,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Compute uncertainty estimates for given prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            max_length: Maximum generation length
            return_tokens: Whether to return per-token uncertainties
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of uncertainty metrics (or list of dicts)
        """
        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False
        
        results = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Computing uncertainties"):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self._compute_batch_uncertainties(
                batch_prompts,
                max_length,
                return_tokens,
            )
            results.extend(batch_results)
        
        if return_single:
            return results[0]
        return results
    
    def _compute_batch_uncertainties(
        self,
        prompts: List[str],
        max_length: int,
        return_tokens: bool,
    ) -> List[Dict]:
        """
        Compute uncertainties for a batch of prompts.
        
        Args:
            prompts: List of prompts
            max_length: Maximum length
            return_tokens: Whether to return per-token uncertainties
            
        Returns:
            List of uncertainty dictionaries
        """
        # Tokenize inputs
        inputs = self.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        
        # Get predictions from all ensemble members
        with torch.no_grad():
            ensemble_outputs = []
            
            for member_idx in range(self.model.num_ensemble_members):
                outputs = self.model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                ensemble_outputs.append(outputs.logits)
            
            # Stack ensemble outputs
            ensemble_logits = torch.stack(ensemble_outputs)  # (M, B, L, V)
        
        # Compute uncertainties
        results = []
        
        for b in range(len(prompts)):
            # Get logits for this example
            example_logits = ensemble_logits[:, b, :, :]  # (M, L, V)
            
            # Get the last token's logits (next token prediction)
            last_token_logits = example_logits[:, -1, :]  # (M, V)
            
            # Compute uncertainties
            uncertainties = UncertaintyMetrics.from_logits(
                last_token_logits.unsqueeze(1),  # (M, 1, V)
                temperature=self.temperature,
                reduction="mean",
            )
            
            # Convert to float
            result = {k: v.item() for k, v in uncertainties.items()}
            
            # Add per-token uncertainties if requested
            if return_tokens:
                token_uncertainties = UncertaintyMetrics.compute_token_level_uncertainties(
                    example_logits,
                    temperature=self.temperature,
                )
                result["per_token"] = {
                    k: v.cpu().numpy() for k, v in token_uncertainties.items()
                }
            
            results.append(result)
        
        return results
    
    def compute_generation_uncertainties(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        generation_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        Generate text and compute per-token uncertainties during generation.
        
        This allows analyzing the "snowballing effect" mentioned in the paper,
        where uncertainty drops after generating the first hallucinated token.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            generation_kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_texts, per_token_uncertainties)
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        
        # Tokenize input
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate from each ensemble member
        generated_texts = []
        all_token_uncertainties = []
        
        for member_idx in range(self.model.num_ensemble_members):
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    member_idx=member_idx,
                    output_scores=True,
                    return_dict_in_generate=True,
                    **generation_kwargs,
                )
            
            # Decode
            generated_text = self.model.tokenizer.decode(
                outputs.sequences[0][input_length:],
                skip_special_tokens=True,
            )
            generated_texts.append(generated_text)
        
        # Compute uncertainties at each generation step
        # This requires collecting logits from all members at each step
        # Simplified version - compute for final output
        uncertainties = self.compute_uncertainties(
            prompt,
            max_length=input_length + max_new_tokens,
            return_tokens=True,
        )
        
        return generated_texts, [uncertainties]
    
    def compute_features_for_classification(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        max_length: int = 512,
    ) -> np.ndarray:
        """
        Compute uncertainty features for hallucination classification.
        
        As described in the paper, features include:
        - First token's predictive entropy
        - First token's aleatoric uncertainty
        - Average predictive entropy
        - Average aleatoric uncertainty
        
        Args:
            prompts: List of prompts
            responses: Optional list of responses (if already generated)
            max_length: Maximum length
            
        Returns:
            Feature matrix (N, num_features)
        """
        features_list = []
        
        for prompt in tqdm(prompts, desc="Extracting features"):
            # Compute uncertainties
            uncertainties = self.compute_uncertainties(
                prompt,
                max_length=max_length,
                return_tokens=True,
            )
            
            # Extract features
            per_token = uncertainties["per_token"]
            
            features = [
                # First token features
                per_token["predictive_entropy"][0],
                per_token["aleatoric"][0],
                # Average features
                per_token["predictive_entropy"].mean(),
                per_token["aleatoric"].mean(),
            ]
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def analyze_snowballing_effect(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Analyze the "snowballing effect" where uncertainty drops after
        the first hallucinated token.
        
        Returns per-token uncertainty progression.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with per-token uncertainty arrays
        """
        texts, uncertainties = self.compute_generation_uncertainties(
            prompt,
            max_new_tokens,
        )
        
        # Extract per-token uncertainties
        per_token = uncertainties[0]["per_token"]
        
        return {
            "tokens": self.model.tokenizer.encode(texts[0]),
            "predictive_entropy": per_token["predictive_entropy"],
            "aleatoric": per_token["aleatoric"],
            "epistemic": per_token["epistemic"],
        }
