"""
Hallucination Detector Classifier

Binary classifier for detecting hallucinations using uncertainty features.
Supports multiple classifier types as evaluated in the paper.
"""

import numpy as np
import pickle
from typing import Dict, List, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from hallucination_detection.detection.features import (
    FeatureExtractor,
    prepare_classification_dataset,
)


class HallucinationDetector:
    """
    Binary classifier for detecting hallucinations.
    
    As described in the paper, this classifier is trained on uncertainty
    estimates to distinguish between hallucinated and accurate content.
    
    Supports multiple classifier types:
    - Logistic Regression (LR)
    - Decision Tree (DT)
    - Support Vector Classifier (SVC)
    - Random Forest (RF)
    - k-Nearest Neighbors (kNN)
    
    Args:
        classifier_type: Type of classifier ('lr', 'dt', 'svc', 'rf', 'knn')
        feature_extractor: Optional custom feature extractor
        scale_features: Whether to standardize features
        **classifier_kwargs: Additional arguments for the classifier
    """
    
    def __init__(
        self,
        classifier_type: str = "rf",
        feature_extractor: Optional[FeatureExtractor] = None,
        scale_features: bool = True,
        **classifier_kwargs,
    ):
        self.classifier_type = classifier_type.lower()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.scale_features = scale_features
        
        # Initialize scaler
        self.scaler = StandardScaler() if scale_features else None
        
        # Initialize classifier
        self.classifier = self._create_classifier(classifier_kwargs)
        
        # Training metadata
        self.is_trained = False
        self.feature_names = self.feature_extractor.get_feature_names()
    
    def _create_classifier(self, kwargs: Dict) -> object:
        """
        Create classifier based on type.
        
        Args:
            kwargs: Classifier-specific arguments
            
        Returns:
            Classifier instance
        """
        # Default parameters from scikit-learn (as mentioned in paper)
        if self.classifier_type == "lr":
            return LogisticRegression(**kwargs)
        elif self.classifier_type == "dt":
            return DecisionTreeClassifier(**kwargs)
        elif self.classifier_type == "svc":
            return SVC(**kwargs)
        elif self.classifier_type == "rf":
            return RandomForestClassifier(**kwargs)
        elif self.classifier_type == "knn":
            return KNeighborsClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def train(
        self,
        uncertainty_estimates: List[Dict],
        labels: List[int],
        validation_split: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train the hallucination detector.
        
        Args:
            uncertainty_estimates: List of uncertainty dictionaries
            labels: Binary labels (0=correct, 1=hallucinated)
            validation_split: Fraction of data for validation
            random_state: Random seed
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare dataset
        X, y = prepare_classification_dataset(
            uncertainty_estimates,
            labels,
            self.feature_extractor,
        )
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=random_state,
            stratify=y,
        )
        
        # Scale features
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        y_pred = self.classifier.predict(X_val)
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }
        
        print(f"\nTraining complete!")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}")
        print(f"Validation Recall: {metrics['recall']:.4f}")
        print(f"Validation F1: {metrics['f1']:.4f}")
        
        return metrics
    
    def predict(
        self,
        uncertainty_estimates: Union[Dict, List[Dict]],
    ) -> Union[int, np.ndarray]:
        """
        Predict whether outputs are hallucinated.
        
        Args:
            uncertainty_estimates: Single or list of uncertainty dictionaries
            
        Returns:
            Prediction(s): 0=correct, 1=hallucinated
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before prediction")
        
        # Handle single input
        single_input = isinstance(uncertainty_estimates, dict)
        if single_input:
            uncertainty_estimates = [uncertainty_estimates]
        
        # Extract features
        X = self.feature_extractor.extract_batch(uncertainty_estimates)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        predictions = self.classifier.predict(X)
        
        if single_input:
            return int(predictions[0])
        return predictions
    
    def predict_proba(
        self,
        uncertainty_estimates: Union[Dict, List[Dict]],
    ) -> np.ndarray:
        """
        Predict probability of hallucination.
        
        Args:
            uncertainty_estimates: Single or list of uncertainty dictionaries
            
        Returns:
            Probabilities for each class
        """
        if not self.is_trained:
            raise RuntimeError("Detector must be trained before prediction")
        
        # Handle single input
        single_input = isinstance(uncertainty_estimates, dict)
        if single_input:
            uncertainty_estimates = [uncertainty_estimates]
        
        # Extract features
        X = self.feature_extractor.extract_batch(uncertainty_estimates)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict probabilities
        if hasattr(self.classifier, 'predict_proba'):
            probas = self.classifier.predict_proba(X)
        else:
            # For classifiers without predict_proba (e.g., SVC with default kernel)
            probas = np.column_stack([
                1 - self.classifier.decision_function(X),
                self.classifier.decision_function(X),
            ])
        
        if single_input:
            return probas[0]
        return probas
    
    def evaluate(
        self,
        uncertainty_estimates: List[Dict],
        labels: List[int],
    ) -> Dict[str, float]:
        """
        Evaluate detector on test set.
        
        Args:
            uncertainty_estimates: List of uncertainty dictionaries
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(uncertainty_estimates)
        
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision_score(labels, predictions, zero_division=0),
            "recall": recall_score(labels, predictions, zero_division=0),
            "f1": f1_score(labels, predictions, zero_division=0),
        }
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save detector to file.
        
        Args:
            filepath: Path to save to
        """
        state = {
            "classifier_type": self.classifier_type,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "feature_extractor": self.feature_extractor,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "HallucinationDetector":
        """
        Load detector from file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded HallucinationDetector
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        detector = cls(
            classifier_type=state["classifier_type"],
            feature_extractor=state["feature_extractor"],
            scale_features=(state["scaler"] is not None),
        )
        
        detector.classifier = state["classifier"]
        detector.scaler = state["scaler"]
        detector.is_trained = state["is_trained"]
        detector.feature_names = state["feature_names"]
        
        print(f"Detector loaded from {filepath}")
        return detector


class EnsembleDetector:
    """
    Ensemble of multiple detectors for improved performance.
    
    Combines predictions from multiple classifier types.
    """
    
    def __init__(
        self,
        classifier_types: List[str] = ["lr", "dt", "svc", "rf", "knn"],
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        """
        Initialize ensemble detector.
        
        Args:
            classifier_types: List of classifier types to ensemble
            feature_extractor: Shared feature extractor
        """
        self.detectors = [
            HallucinationDetector(
                classifier_type=clf_type,
                feature_extractor=feature_extractor,
            )
            for clf_type in classifier_types
        ]
    
    def train(
        self,
        uncertainty_estimates: List[Dict],
        labels: List[int],
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all detectors in the ensemble.
        
        Args:
            uncertainty_estimates: Uncertainty estimates
            labels: Labels
            **kwargs: Training arguments
            
        Returns:
            Metrics for each detector
        """
        results = {}
        
        for detector in self.detectors:
            print(f"\nTraining {detector.classifier_type.upper()} detector...")
            metrics = detector.train(uncertainty_estimates, labels, **kwargs)
            results[detector.classifier_type] = metrics
        
        return results
    
    def predict(
        self,
        uncertainty_estimates: Union[Dict, List[Dict]],
        voting: str = "hard",
    ) -> Union[int, np.ndarray]:
        """
        Predict using ensemble.
        
        Args:
            uncertainty_estimates: Uncertainty estimates
            voting: 'hard' (majority) or 'soft' (average probabilities)
            
        Returns:
            Predictions
        """
        predictions = []
        
        for detector in self.detectors:
            if voting == "hard":
                pred = detector.predict(uncertainty_estimates)
            else:
                pred = detector.predict_proba(uncertainty_estimates)[:, 1]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if voting == "hard":
            # Majority voting
            final_pred = (predictions.sum(axis=0) > len(self.detectors) / 2).astype(int)
        else:
            # Average probabilities
            avg_proba = predictions.mean(axis=0)
            final_pred = (avg_proba > 0.5).astype(int)
        
        return final_pred
