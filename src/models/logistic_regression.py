"""
Algorithm 1: Logistic Regression Model for Trump Tweets Classification

This module implements the Logistic Regression classifier with hyperparameter tuning
for the Trump tweets authorship attribution task.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrumpTweetsLogisticRegression:
    """
    Logistic Regression classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 1 as specified in the assignment requirements.
    It uses sklearn.linear_model.LogisticRegression with hyperparameter tuning.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Logistic Regression classifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        
    def train(self, X, y, cv_folds=5, verbose=True):
        """
        Train the Logistic Regression model with hyperparameter tuning.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels (0=Trump, 1=Staffer)
            cv_folds (int): Number of cross-validation folds
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with model and performance metrics
        """
        if verbose:
            print("Training Algorithm 1: Logistic Regression")
            print(f"Feature matrix shape: {X.shape}")
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000]
        }
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            LogisticRegression(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Comprehensive cross-validation evaluation
        cv_accuracy = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(self.model, X, y, cv=cv, scoring='precision')
        cv_recall = cross_val_score(self.model, X, y, cv=cv, scoring='recall')
        cv_f1 = cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        
        self.cv_scores = {
            'accuracy': cv_accuracy,
            'precision': cv_precision,
            'recall': cv_recall,
            'f1': cv_f1
        }
        
        if verbose:
            print(f"\nBest parameters: {self.best_params}")
            print(f"Cross-validation accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std() * 2:.4f})")
            print(f"Cross-validation precision: {cv_precision.mean():.4f} (±{cv_precision.std() * 2:.4f})")
            print(f"Cross-validation recall: {cv_recall.mean():.4f} (±{cv_recall.std() * 2:.4f})")
            print(f"Cross-validation F1: {cv_f1.mean():.4f} (±{cv_f1.std() * 2:.4f})")
        
        return {
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'algorithm': 'Logistic Regression'
        }
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Feature matrix for prediction
            
        Returns:
            np.array: Predictions (0s and 1s)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Feature matrix for prediction
            
        Returns:
            np.array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """
        Get feature importance (coefficients) from the trained model.
        
        Returns:
            np.array: Feature coefficients
        """
        if self.model is None:
            raise ValueError("Model must be trained before accessing feature importance")
        
        return self.model.coef_[0]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.array): Test feature matrix
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }


def train_algorithm_1_logistic_regression(X, y):
    """
    Convenience function for training Algorithm 1: Logistic Regression.
    
    This function maintains compatibility with the notebook implementation.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        
    Returns:
        dict: Training results with model and performance metrics
    """
    lr_classifier = TrumpTweetsLogisticRegression()
    return lr_classifier.train(X, y)