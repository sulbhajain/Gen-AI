"""
Feature Extraction for Hallucination Detection

Extracts uncertainty-based features for training classifiers.
"""

import numpy as np
from typing import Dict, List, Optional, Union


class FeatureExtractor:
    """
    Extract features from uncertainty estimates for classification.
    
    As described in the paper, features include:
    - First token's predictive entropy
    - First token's aleatoric uncertainty  
    - Average predictive entropy
    - Average aleatoric uncertainty
    
    Epistemic uncertainty is excluded due to correlation with other features.
    """
    
    def __init__(
        self,
        include_epistemic: bool = False,
        use_max_features: bool = False,
    ):
        """
        Initialize feature extractor.
        
        Args:
            include_epistemic: Whether to include epistemic uncertainty
            use_max_features: Whether to use max instead of mean for aggregation
        """
        self.include_epistemic = include_epistemic
        self.use_max_features = use_max_features
    
    def extract_from_uncertainties(
        self,
        uncertainties: Dict[str, Union[float, np.ndarray]],
    ) -> np.ndarray:
        """
        Extract feature vector from uncertainty estimates.
        
        Args:
            uncertainties: Dictionary with uncertainty metrics
            
        Returns:
            Feature vector
        """
        features = []
        
        # Check if per-token uncertainties are available
        if "per_token" in uncertainties:
            per_token = uncertainties["per_token"]
            
            # First token features
            features.append(per_token["predictive_entropy"][0])
            features.append(per_token["aleatoric"][0])
            
            # Aggregated features
            if self.use_max_features:
                features.append(per_token["predictive_entropy"].max())
                features.append(per_token["aleatoric"].max())
            else:
                features.append(per_token["predictive_entropy"].mean())
                features.append(per_token["aleatoric"].mean())
            
            # Optional: epistemic uncertainty
            if self.include_epistemic:
                features.append(per_token["epistemic"][0])
                if self.use_max_features:
                    features.append(per_token["epistemic"].max())
                else:
                    features.append(per_token["epistemic"].mean())
        else:
            # Use scalar values
            features.append(uncertainties["predictive_entropy"])
            features.append(uncertainties["aleatoric"])
            
            if self.include_epistemic:
                features.append(uncertainties["epistemic"])
        
        return np.array(features)
    
    def extract_batch(
        self,
        uncertainty_list: List[Dict],
    ) -> np.ndarray:
        """
        Extract features for a batch of uncertainty estimates.
        
        Args:
            uncertainty_list: List of uncertainty dictionaries
            
        Returns:
            Feature matrix (N, num_features)
        """
        features = []
        
        for uncertainties in uncertainty_list:
            feature_vec = self.extract_from_uncertainties(uncertainties)
            features.append(feature_vec)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of extracted features.
        
        Returns:
            List of feature names
        """
        names = [
            "first_token_predictive_entropy",
            "first_token_aleatoric",
            "avg_predictive_entropy",
            "avg_aleatoric",
        ]
        
        if self.include_epistemic:
            names.extend([
                "first_token_epistemic",
                "avg_epistemic",
            ])
        
        return names


def prepare_classification_dataset(
    uncertainty_estimates: List[Dict],
    labels: List[int],
    feature_extractor: Optional[FeatureExtractor] = None,
) -> tuple:
    """
    Prepare dataset for hallucination classification.
    
    Args:
        uncertainty_estimates: List of uncertainty dictionaries
        labels: Binary labels (0=correct, 1=hallucinated)
        feature_extractor: Optional custom feature extractor
        
    Returns:
        (X, y) tuple for training classifiers
    """
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    
    # Extract features
    X = feature_extractor.extract_batch(uncertainty_estimates)
    y = np.array(labels)
    
    return X, y


def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = "mutual_info",
) -> Dict[str, float]:
    """
    Compute feature importance for understanding which uncertainty
    metrics are most predictive of hallucinations.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: Names of features
        method: Method for computing importance
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    
    if method == "mutual_info":
        # Mutual information
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == "random_forest":
        # Random forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        scores = rf.feature_importances_
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create dictionary
    importance = {
        name: score for name, score in zip(feature_names, scores)
    }
    
    return importance
