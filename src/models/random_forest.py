"""
Algorithm 4: Random Forest Model for Trump Tweets Classification

This module implements a Random Forest classifier as the fourth classifier choice
for the Trump tweets authorship attribution task. It combines different types of features
and provides feature importance analysis.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class TrumpTweetsRandomForest:
    """
    Random Forest classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 4 as specified in the assignment requirements.
    It uses sklearn.ensemble.RandomForestClassifier with hyperparameter tuning
    and can handle combining different types of features effectively.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Random Forest classifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.cv_scores = None
        self.feature_importance = None
        self.feature_names = None
        
    def train(self, X, y, cv_folds=5, feature_names=None, verbose=True):
        """
        Train the Random Forest model with hyperparameter tuning.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels (0=Trump, 1=Staffer)
            cv_folds (int): Number of cross-validation folds
            feature_names (list): Optional list of feature names for importance analysis
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with model and performance metrics
        """
        if verbose:
            print("Training Algorithm 4: Random Forest")
            print(f"Feature matrix shape: {X.shape}")
        
        self.feature_names = feature_names
        
        # Comprehensive hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Reduce parameter grid for computational efficiency while maintaining effectiveness
        reduced_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        if verbose:
            print("Performing hyperparameter tuning...")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
            reduced_param_grid,
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
        
        # Extract feature importance
        self.feature_importance = self.model.feature_importances_
        
        if verbose:
            print(f"\nBest parameters: {self.best_params}")
            print(f"Cross-validation accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std() * 2:.4f})")
            print(f"Cross-validation precision: {cv_precision.mean():.4f} (±{cv_precision.std() * 2:.4f})")
            print(f"Cross-validation recall: {cv_recall.mean():.4f} (±{cv_recall.std() * 2:.4f})")
            print(f"Cross-validation F1: {cv_f1.mean():.4f} (±{cv_f1.std() * 2:.4f})")
            
            self._print_feature_importance(top_k=10)
        
        return {
            'model': self.model,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'feature_importance': self.feature_importance,
            'algorithm': 'Random Forest'
        }
    
    def _print_feature_importance(self, top_k=10):
        """Print top k most important features."""
        if self.feature_importance is None:
            return
        
        print(f"\nTop {top_k} most important features:")
        top_features = np.argsort(self.feature_importance)[-top_k:]
        
        for i, feat_idx in enumerate(reversed(top_features)):
            if self.feature_names and feat_idx < len(self.feature_names):
                feature_name = self.feature_names[feat_idx]
            else:
                feature_name = f"Feature {feat_idx}"
            
            print(f"  {i+1}. {feature_name}: {self.feature_importance[feat_idx]:.4f}")
    
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
    
    def get_feature_importance(self, top_k=None):
        """
        Get feature importance scores.
        
        Args:
            top_k (int): Number of top features to return (None for all)
            
        Returns:
            dict: Feature importance information
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained before accessing feature importance")
        
        if top_k is None:
            indices = np.arange(len(self.feature_importance))
        else:
            indices = np.argsort(self.feature_importance)[-top_k:]
        
        importance_data = {
            'indices': indices,
            'scores': self.feature_importance[indices],
            'feature_names': [self.feature_names[i] if self.feature_names and i < len(self.feature_names) 
                            else f"Feature {i}" for i in indices] if self.feature_names else None
        }
        
        return importance_data
    
    def plot_feature_importance(self, top_k=20, figsize=(10, 8), save_path=None):
        """
        Plot feature importance.
        
        Args:
            top_k (int): Number of top features to plot
            figsize (tuple): Figure size
            save_path (str): Path to save the plot (optional)
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained before plotting feature importance")
        
        importance_data = self.get_feature_importance(top_k)
        
        plt.figure(figsize=figsize)
        y_pos = np.arange(len(importance_data['indices']))
        
        plt.barh(y_pos, importance_data['scores'])
        plt.ylabel('Features')
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance - Random Forest')
        
        if importance_data['feature_names']:
            plt.yticks(y_pos, importance_data['feature_names'])
        else:
            plt.yticks(y_pos, [f"Feature {i}" for i in importance_data['indices']])
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_tree_info(self):
        """
        Get information about the trees in the forest.
        
        Returns:
            dict: Tree information
        """
        if self.model is None:
            raise ValueError("Model must be trained before accessing tree information")
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'max_features': self.model.max_features,
            'oob_score': getattr(self.model, 'oob_score_', None)
        }
    
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
        y_proba = self.predict_proba(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'prediction_confidence': {
                'mean_max_proba': np.mean(np.max(y_proba, axis=1)),
                'std_max_proba': np.std(np.max(y_proba, axis=1))
            }
        }
    
    def analyze_predictions(self, X, y, threshold=0.6):
        """
        Analyze prediction confidence and potential misclassifications.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): True labels
            threshold (float): Confidence threshold for analysis
            
        Returns:
            dict: Prediction analysis
        """
        if self.model is None:
            raise ValueError("Model must be trained before analysis")
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        max_proba = np.max(y_proba, axis=1)
        
        # Identify high and low confidence predictions
        high_confidence = max_proba >= threshold
        low_confidence = max_proba < threshold
        
        # Identify correct and incorrect predictions
        correct = (y_pred == y)
        incorrect = (y_pred != y)
        
        analysis = {
            'total_samples': len(y),
            'high_confidence_samples': np.sum(high_confidence),
            'low_confidence_samples': np.sum(low_confidence),
            'correct_predictions': np.sum(correct),
            'incorrect_predictions': np.sum(incorrect),
            'high_confidence_correct': np.sum(high_confidence & correct),
            'high_confidence_incorrect': np.sum(high_confidence & incorrect),
            'low_confidence_correct': np.sum(low_confidence & correct),
            'low_confidence_incorrect': np.sum(low_confidence & incorrect),
            'mean_confidence': np.mean(max_proba),
            'confidence_threshold': threshold
        }
        
        # Calculate percentages
        analysis['high_confidence_accuracy'] = (
            analysis['high_confidence_correct'] / analysis['high_confidence_samples'] 
            if analysis['high_confidence_samples'] > 0 else 0
        )
        
        analysis['low_confidence_accuracy'] = (
            analysis['low_confidence_correct'] / analysis['low_confidence_samples']
            if analysis['low_confidence_samples'] > 0 else 0
        )
        
        return analysis


def train_algorithm_4_random_forest(X, y):
    """
    Convenience function for training Algorithm 4: Random Forest.
    
    This function maintains compatibility with the notebook implementation.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        
    Returns:
        dict: Training results with model and performance metrics
    """
    rf_classifier = TrumpTweetsRandomForest()
    return rf_classifier.train(X, y)