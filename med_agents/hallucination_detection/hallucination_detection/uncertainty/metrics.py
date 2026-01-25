"""
Uncertainty Metrics

Implements entropy-based uncertainty metrics:
- Predictive entropy
- Aleatoric uncertainty
- Epistemic uncertainty
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import torch.nn.functional as F


def compute_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute entropy of a probability distribution.
    
    H[P(x)] = -∑ P(x) log P(x)
    
    Args:
        probs: Probability distribution
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy values
    """
    # Add small epsilon for numerical stability
    probs = probs + 1e-10
    entropy = -(probs * torch.log(probs)).sum(dim=dim)
    return entropy


def compute_predictive_entropy(
    ensemble_probs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute predictive entropy of an ensemble.
    
    H[P(x_t | x_<t; D)] = -∑ P(x_t | x_<t; D) log P(x_t | x_<t; D)
    
    where P(x_t | x_<t; D) ≈ (1/M) ∑ P(x_t | x_<t; θ_m)
    
    Args:
        ensemble_probs: Probability distributions from ensemble members
                       Shape: (num_members, batch_size, vocab_size)
        reduction: How to reduce over batch ('mean', 'sum', 'none')
        
    Returns:
        Predictive entropy
    """
    # Average probabilities across ensemble members
    mean_probs = ensemble_probs.mean(dim=0)  # (batch_size, vocab_size)
    
    # Compute entropy of averaged distribution
    predictive_entropy = compute_entropy(mean_probs, dim=-1)
    
    if reduction == "mean":
        return predictive_entropy.mean()
    elif reduction == "sum":
        return predictive_entropy.sum()
    else:
        return predictive_entropy


def compute_aleatoric_uncertainty(
    ensemble_probs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute aleatoric (data) uncertainty.
    
    Aleatoric = E_p(θ|D)[H[P(x_t | x_<t; θ)]]
              = (1/M) ∑ H[P(x_t | x_<t; θ_m)]
    
    This represents the irreducible uncertainty due to noise in the data.
    
    Args:
        ensemble_probs: Probability distributions from ensemble members
                       Shape: (num_members, batch_size, vocab_size)
        reduction: How to reduce over batch
        
    Returns:
        Aleatoric uncertainty
    """
    # Compute entropy for each ensemble member
    entropies = compute_entropy(ensemble_probs, dim=-1)  # (num_members, batch_size)
    
    # Average over ensemble members
    aleatoric = entropies.mean(dim=0)  # (batch_size,)
    
    if reduction == "mean":
        return aleatoric.mean()
    elif reduction == "sum":
        return aleatoric.sum()
    else:
        return aleatoric


def compute_epistemic_uncertainty(
    ensemble_probs: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute epistemic (model) uncertainty.
    
    Epistemic = H[P(x_t | x_<t; D)] - E_p(θ|D)[H[P(x_t | x_<t; θ)]]
              = Predictive Entropy - Aleatoric Uncertainty
    
    This represents the reducible uncertainty due to lack of knowledge,
    which decreases with more data.
    
    Args:
        ensemble_probs: Probability distributions from ensemble members
                       Shape: (num_members, batch_size, vocab_size)
        reduction: How to reduce over batch
        
    Returns:
        Epistemic uncertainty
    """
    predictive = compute_predictive_entropy(ensemble_probs, reduction="none")
    aleatoric = compute_aleatoric_uncertainty(ensemble_probs, reduction="none")
    
    epistemic = predictive - aleatoric
    
    if reduction == "mean":
        return epistemic.mean()
    elif reduction == "sum":
        return epistemic.sum()
    else:
        return epistemic


def compute_all_uncertainties(
    ensemble_probs: torch.Tensor,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """
    Compute all uncertainty metrics at once.
    
    Args:
        ensemble_probs: Probability distributions from ensemble members
        reduction: How to reduce over batch
        
    Returns:
        Dictionary with all uncertainty metrics
    """
    return {
        "predictive_entropy": compute_predictive_entropy(ensemble_probs, reduction),
        "aleatoric": compute_aleatoric_uncertainty(ensemble_probs, reduction),
        "epistemic": compute_epistemic_uncertainty(ensemble_probs, reduction),
    }


class UncertaintyMetrics:
    """
    Container for computing and storing uncertainty metrics.
    """
    
    @staticmethod
    def from_logits(
        ensemble_logits: torch.Tensor,
        temperature: float = 1.0,
        reduction: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainties from ensemble logits.
        
        Args:
            ensemble_logits: Logits from ensemble members
                            Shape: (num_members, batch_size, vocab_size)
            temperature: Temperature for softmax
            reduction: How to reduce over batch
            
        Returns:
            Dictionary of uncertainty metrics
        """
        # Convert logits to probabilities
        ensemble_probs = F.softmax(ensemble_logits / temperature, dim=-1)
        
        return compute_all_uncertainties(ensemble_probs, reduction)
    
    @staticmethod
    def compute_token_level_uncertainties(
        ensemble_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute per-token uncertainties (no reduction).
        
        Useful for analyzing uncertainty at each generation step,
        as shown in Table 3 of the paper.
        
        Args:
            ensemble_logits: Logits from ensemble members
            temperature: Temperature for softmax
            
        Returns:
            Dictionary of per-token uncertainties
        """
        ensemble_probs = F.softmax(ensemble_logits / temperature, dim=-1)
        return compute_all_uncertainties(ensemble_probs, reduction="none")
    
    @staticmethod
    def compute_sequence_level_uncertainties(
        token_uncertainties: Dict[str, torch.Tensor],
        aggregation: str = "mean",
    ) -> Dict[str, float]:
        """
        Aggregate token-level uncertainties to sequence level.
        
        Args:
            token_uncertainties: Per-token uncertainty dict
            aggregation: How to aggregate ('mean', 'max', 'first')
            
        Returns:
            Sequence-level uncertainties
        """
        result = {}
        
        for key, values in token_uncertainties.items():
            if aggregation == "mean":
                result[key] = values.mean().item()
            elif aggregation == "max":
                result[key] = values.max().item()
            elif aggregation == "first":
                result[key] = values[0].item()
            elif aggregation == "sum":
                result[key] = values.sum().item()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return result


def compute_sample_based_uncertainty(
    samples: List[str],
    tokenizer,
    reduction: str = "mean",
) -> Dict[str, float]:
    """
    Compute uncertainty from multiple samples (baseline method).
    
    This is the sample-based baseline mentioned in the paper that uses
    stochastic sampling (temperature=0.5, top-p=0.99, top-k=5).
    
    Args:
        samples: List of sampled outputs
        tokenizer: Tokenizer for encoding
        reduction: How to reduce
        
    Returns:
        Uncertainty metrics
    """
    # Tokenize all samples
    token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in samples]
    
    # Compute token-level agreement
    max_len = max(len(ids) for ids in token_ids)
    
    # Pad sequences
    padded = []
    for ids in token_ids:
        padded.append(ids + [tokenizer.pad_token_id] * (max_len - len(ids)))
    
    # Convert to tensor
    token_tensor = torch.tensor(padded)  # (num_samples, max_len)
    
    # Compute entropy at each position
    entropies = []
    for i in range(max_len):
        tokens_at_pos = token_tensor[:, i]
        
        # Count unique tokens
        unique, counts = torch.unique(tokens_at_pos, return_counts=True)
        probs = counts.float() / len(tokens_at_pos)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        entropies.append(entropy.item())
    
    if reduction == "mean":
        uncertainty = np.mean(entropies)
    elif reduction == "max":
        uncertainty = np.max(entropies)
    elif reduction == "first":
        uncertainty = entropies[0] if entropies else 0.0
    else:
        uncertainty = entropies
    
    return {
        "sample_entropy": uncertainty,
    }
