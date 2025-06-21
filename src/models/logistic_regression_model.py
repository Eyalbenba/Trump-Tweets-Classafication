"""
Logistic Regression implementation for Trump tweets classification.
Algorithm 1: sklearn.linear_model.LogisticRegression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Tuple
import pickle
import os

class LogisticRegressionClassifier:
    """Logistic Regression classifier for tweet authorship attribution."""
    
    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.best_params = None
        self.cv_scores = None
        
    def train(self, X: np.ndarray, y: np.ndarray, tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train the logistic regression model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            Dict[str, Any]: Training results
        """
        if tune_hyperparameters:
            # Hyperparameter tuning
            param_grid = {
                'C': [0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42),
                param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            print(f"Best parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model.fit(X, y)
        
        # Cross-validation evaluation
        self.cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        
        results = {
            'model': self.model,
            'best_params': self.best_params,
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
            'cv_scores': self.cv_scores
        }
        
        print(f"Cross-validation accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels
            
        Returns:
            Dict[str, Any]: Evaluation metrics
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance (coefficients).
        
        Args:
            feature_names (list): Names of features
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        coefficients = self.model.coef_[0]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data.get('best_params')
        self.cv_scores = model_data.get('cv_scores')
        
        print(f"Model loaded from {filepath}")


def train_logistic_regression(X: np.ndarray, y: np.ndarray, 
                            tune_hyperparameters: bool = True) -> LogisticRegressionClassifier:
    """
    Convenience function to train a logistic regression model.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        tune_hyperparameters (bool): Whether to tune hyperparameters
        
    Returns:
        LogisticRegressionClassifier: Trained model
    """
    classifier = LogisticRegressionClassifier()
    classifier.train(X, y, tune_hyperparameters=tune_hyperparameters)
    return classifier