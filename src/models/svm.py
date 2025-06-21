"""
Algorithm 2: Support Vector Machine Model for Trump Tweets Classification

This module implements both linear and nonlinear (RBF) SVM classifiers with hyperparameter tuning
for the Trump tweets authorship attribution task.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrumpTweetsSVM:
    """
    Support Vector Machine classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 2 as specified in the assignment requirements.
    It uses sklearn.svm.SVC with both linear and nonlinear (RBF) kernels.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the SVM classifier.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.linear_model = None
        self.rbf_model = None
        self.best_model = None
        self.best_kernel = None
        self.best_params = None
        self.cv_scores = None
        self.all_results = None
        
    def train(self, X, y, cv_folds=5, verbose=True):
        """
        Train both linear and RBF SVM models and select the best one.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels (0=Trump, 1=Staffer)
            cv_folds (int): Number of cross-validation folds
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with best model and performance metrics
        """
        if verbose:
            print("Training Algorithm 2: Support Vector Machine")
            print(f"Feature matrix shape: {X.shape}")
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {}
        
        # 1. Linear SVM
        if verbose:
            print("\nTraining Linear SVM...")
        
        linear_param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear']
        }
        
        linear_grid = GridSearchCV(
            SVC(random_state=self.random_state),
            linear_param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        linear_grid.fit(X, y)
        self.linear_model = linear_grid.best_estimator_
        linear_cv_scores = cross_val_score(self.linear_model, X, y, cv=cv, scoring='accuracy')
        
        results['linear'] = {
            'model': self.linear_model,
            'params': linear_grid.best_params_,
            'cv_accuracy': linear_cv_scores.mean(),
            'cv_std': linear_cv_scores.std(),
            'cv_scores': linear_cv_scores
        }
        
        if verbose:
            print(f"Linear SVM - Best params: {linear_grid.best_params_}")
            print(f"Linear SVM - CV accuracy: {linear_cv_scores.mean():.4f} (±{linear_cv_scores.std() * 2:.4f})")
        
        # 2. RBF (Nonlinear) SVM
        if verbose:
            print("\nTraining RBF SVM...")
        
        rbf_param_grid = {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf']
        }
        
        rbf_grid = GridSearchCV(
            SVC(random_state=self.random_state),
            rbf_param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        rbf_grid.fit(X, y)
        self.rbf_model = rbf_grid.best_estimator_
        rbf_cv_scores = cross_val_score(self.rbf_model, X, y, cv=cv, scoring='accuracy')
        
        results['rbf'] = {
            'model': self.rbf_model,
            'params': rbf_grid.best_params_,
            'cv_accuracy': rbf_cv_scores.mean(),
            'cv_std': rbf_cv_scores.std(),
            'cv_scores': rbf_cv_scores
        }
        
        if verbose:
            print(f"RBF SVM - Best params: {rbf_grid.best_params_}")
            print(f"RBF SVM - CV accuracy: {rbf_cv_scores.mean():.4f} (±{rbf_cv_scores.std() * 2:.4f})")
        
        # Select best kernel
        if results['linear']['cv_accuracy'] >= results['rbf']['cv_accuracy']:
            self.best_kernel = 'linear'
            self.best_model = results['linear']['model']
            self.best_params = results['linear']['params']
            best_cv_scores = linear_cv_scores
        else:
            self.best_kernel = 'rbf'
            self.best_model = results['rbf']['model']
            self.best_params = results['rbf']['params']
            best_cv_scores = rbf_cv_scores
        
        self.cv_scores = {'accuracy': best_cv_scores}
        self.all_results = results
        
        if verbose:
            print(f"\nBest kernel: {self.best_kernel}")
            print(f"Best SVM accuracy: {best_cv_scores.mean():.4f} (±{best_cv_scores.std() * 2:.4f})")
        
        return {
            'model': self.best_model,
            'best_kernel': self.best_kernel,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'algorithm': f'SVM ({self.best_kernel})',
            'all_results': self.all_results
        }
    
    def predict(self, X):
        """
        Make predictions on new data using the best model.
        
        Args:
            X (np.array): Feature matrix for prediction
            
        Returns:
            np.array: Predictions (0s and 1s)
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities using the best model.
        Note: SVM doesn't naturally output probabilities, so this uses decision function.
        
        Args:
            X (np.array): Feature matrix for prediction
            
        Returns:
            np.array: Decision function scores
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.best_model.decision_function(X)
    
    def get_support_vectors(self):
        """
        Get support vectors from the trained model.
        
        Returns:
            np.array: Support vectors
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before accessing support vectors")
        
        return self.best_model.support_vectors_
    
    def get_model_comparison(self):
        """
        Get comparison results between linear and RBF kernels.
        
        Returns:
            dict: Comparison results
        """
        if self.all_results is None:
            raise ValueError("Models must be trained before accessing comparison results")
        
        return {
            'linear_accuracy': self.all_results['linear']['cv_accuracy'],
            'rbf_accuracy': self.all_results['rbf']['cv_accuracy'],
            'best_kernel': self.best_kernel,
            'performance_difference': abs(
                self.all_results['linear']['cv_accuracy'] - 
                self.all_results['rbf']['cv_accuracy']
            )
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the best model on test data.
        
        Args:
            X_test (np.array): Test feature matrix
            y_test (np.array): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'kernel_used': self.best_kernel
        }


def train_algorithm_2_svm(X, y):
    """
    Convenience function for training Algorithm 2: SVM.
    
    This function maintains compatibility with the notebook implementation.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        
    Returns:
        dict: Training results with best model and performance metrics
    """
    svm_classifier = TrumpTweetsSVM()
    return svm_classifier.train(X, y)