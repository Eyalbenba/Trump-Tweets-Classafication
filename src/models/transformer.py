"""
Algorithm 5: Transformer-based Model for Trump Tweets Classification

This module implements a transformer-based classifier (BERT/RoBERTa) with fallback to
Naive Bayes for the Trump tweets authorship attribution task.
"""

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Try to import transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TrumpTweetsTransformer:
    """
    Transformer-based classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 5 as specified in the assignment requirements.
    It attempts to use transformer models (BERT/RoBERTa) and falls back to
    an enhanced Naive Bayes classifier if transformers are not available.
    """
    
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                 random_state=42, fallback_to_nb=True):
        """
        Initialize the Transformer classifier.
        
        Args:
            model_name (str): Hugging Face model name for transformer
            random_state (int): Random state for reproducibility
            fallback_to_nb (bool): Whether to fallback to Naive Bayes if transformers fail
        """
        self.model_name = model_name
        self.random_state = random_state
        self.fallback_to_nb = fallback_to_nb
        
        self.model = None
        self.scaler = None
        self.transformer_pipeline = None
        self.algorithm_used = None
        self.best_params = None
        self.cv_scores = None
        
    def train(self, X, y, texts=None, cv_folds=5, verbose=True):
        """
        Train the transformer-based model with fallback to Naive Bayes.
        
        Args:
            X (np.array): Feature matrix (for fallback)
            y (np.array): Labels (0=Trump, 1=Staffer)
            texts (list): Original tweet texts (for transformer)
            cv_folds (int): Number of cross-validation folds
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with model and performance metrics
        """
        if verbose:
            print("Training Algorithm 5: Transformer-based Classifier")
        
        # Try transformer approach first
        if TRANSFORMERS_AVAILABLE and texts is not None:
            try:
                return self._train_transformer(X, y, texts, cv_folds, verbose)
            except Exception as e:
                if verbose:
                    print(f"Transformer approach failed: {e}")
                    print("Falling back to Naive Bayes...")
        
        # Fallback to enhanced Naive Bayes
        if self.fallback_to_nb:
            return self._train_naive_bayes(X, y, cv_folds, verbose)
        else:
            raise RuntimeError("Transformer training failed and fallback disabled")
    
    def _train_transformer(self, X, y, texts, cv_folds, verbose):
        """
        Train using transformer approach.
        
        Args:
            X (np.array): Feature matrix (combined with transformer features)
            y (np.array): Labels
            texts (list): Tweet texts
            cv_folds (int): CV folds
            verbose (bool): Verbose output
            
        Returns:
            dict: Training results
        """
        if verbose:
            print("Using BERT/RoBERTa-based approach...")
        
        # Initialize transformer pipeline
        self.transformer_pipeline = pipeline(
            "sentiment-analysis", 
            model=self.model_name,
            return_all_scores=True
        )
        
        if verbose:
            print("Extracting transformer features...")
        
        # Extract transformer features from texts
        transformer_features = self._extract_transformer_features(texts, verbose)
        
        # Combine with existing features
        if transformer_features is not None:
            # Ensure transformer features match the size of X and y
            min_size = min(len(X), len(y), len(transformer_features))
            X_subset = X[:min_size]
            y_subset = y[:min_size]
            transformer_features_subset = transformer_features[:min_size]
            
            combined_features = np.hstack([X_subset, transformer_features_subset])
            
            if verbose:
                print(f"Combined features shape: {combined_features.shape}")
            
            # Use Random Forest on combined features
            from sklearn.ensemble import RandomForestClassifier
            
            # Hyperparameter tuning for the combined model
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 20],
                'min_samples_split': [2, 5],
                'max_features': ['sqrt', 'log2']
            }
            
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=self.random_state),
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(combined_features, y_subset)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            # Cross-validation evaluation
            cv_scores = cross_val_score(self.model, combined_features, y_subset, 
                                      cv=cv, scoring='accuracy')
            self.cv_scores = {'accuracy': cv_scores}
            self.algorithm_used = 'Transformer-Enhanced RF'
            
            if verbose:
                print(f"Best parameters: {self.best_params}")
                print(f"Transformer-enhanced model accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
            
            return {
                'model': self.model,
                'cv_scores': self.cv_scores,
                'algorithm': self.algorithm_used,
                'features_used': 'TF-IDF + Stylistic + Transformer',
                'best_params': self.best_params,
                'transformer_pipeline': self.transformer_pipeline
            }
        else:
            raise ValueError("Failed to extract transformer features")
    
    def _extract_transformer_features(self, texts, verbose=False, batch_size=50):
        """
        Extract features using the transformer pipeline.
        
        Args:
            texts (list): List of tweet texts
            verbose (bool): Verbose output
            batch_size (int): Batch size for processing
            
        Returns:
            np.array: Transformer features
        """
        transformer_features = []
        
        # Process in batches to avoid memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_features = []
            
            for text in batch_texts:
                try:
                    # Truncate long texts
                    text = str(text)[:512] if len(str(text)) > 512 else str(text)
                    
                    # Get sentiment scores
                    result = self.transformer_pipeline(text)
                    
                    # Convert to feature vector
                    # Assuming 3-class sentiment: negative, neutral, positive
                    if len(result) >= 3:
                        feature_vector = [r['score'] for r in result]
                    else:
                        # If fewer classes, pad or adjust
                        feature_vector = [r['score'] for r in result] + [0.0] * (3 - len(result))
                    
                    batch_features.append(feature_vector)
                    
                except Exception as e:
                    if verbose and len(batch_features) < 5:  # Only print first few errors
                        print(f"Error processing text: {e}")
                    # Default feature vector
                    batch_features.append([0.33, 0.33, 0.34])
            
            transformer_features.extend(batch_features)
            
            if verbose and i % (batch_size * 10) == 0:
                print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return np.array(transformer_features)
    
    def _train_naive_bayes(self, X, y, cv_folds, verbose):
        """
        Train using Naive Bayes fallback approach.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            cv_folds (int): CV folds
            verbose (bool): Verbose output
            
        Returns:
            dict: Training results
        """
        if verbose:
            print("Using Naive Bayes as transformer alternative...")
        
        # Scale features to be non-negative for Multinomial NB
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X) + 1e-10  # Add small constant
        
        # Hyperparameter tuning for Naive Bayes
        param_grid = {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]}
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            MultinomialNB(),
            param_grid,
            cv=cv,
            scoring='accuracy'
        )
        
        grid_search.fit(X_scaled, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        self.cv_scores = {'accuracy': cv_scores}
        self.algorithm_used = 'Naive Bayes (Transformer Alternative)'
        
        if verbose:
            print(f"Best alpha: {self.best_params['alpha']}")
            print(f"Naive Bayes accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'algorithm': self.algorithm_used
        }
    
    def predict(self, X, texts=None):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Feature matrix for prediction
            texts (list): Original texts (needed for transformer approach)
            
        Returns:
            np.array: Predictions (0s and 1s)
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.algorithm_used == 'Transformer-Enhanced RF':
            if texts is None:
                raise ValueError("Texts required for transformer-based predictions")
            
            # Extract transformer features
            transformer_features = self._extract_transformer_features(texts)
            
            # Combine features
            min_size = min(len(X), len(transformer_features))
            X_subset = X[:min_size]
            transformer_features_subset = transformer_features[:min_size]
            
            combined_features = np.hstack([X_subset, transformer_features_subset])
            return self.model.predict(combined_features)
        
        elif self.algorithm_used == 'Naive Bayes (Transformer Alternative)':
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X) + 1e-10
                return self.model.predict(X_scaled)
            else:
                return self.model.predict(X)
        
        else:
            return self.model.predict(X)
    
    def predict_proba(self, X, texts=None):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Feature matrix for prediction
            texts (list): Original texts (needed for transformer approach)
            
        Returns:
            np.array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if self.algorithm_used == 'Transformer-Enhanced RF':
            if texts is None:
                raise ValueError("Texts required for transformer-based predictions")
            
            # Extract transformer features
            transformer_features = self._extract_transformer_features(texts)
            
            # Combine features
            min_size = min(len(X), len(transformer_features))
            X_subset = X[:min_size]
            transformer_features_subset = transformer_features[:min_size]
            
            combined_features = np.hstack([X_subset, transformer_features_subset])
            return self.model.predict_proba(combined_features)
        
        elif self.algorithm_used == 'Naive Bayes (Transformer Alternative)':
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X) + 1e-10
                return self.model.predict_proba(X_scaled)
            else:
                return self.model.predict_proba(X)
        
        else:
            return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, texts_test=None):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.array): Test feature matrix
            y_test (np.array): Test labels
            texts_test (list): Test texts (for transformer approach)
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test, texts_test)
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'algorithm_used': self.algorithm_used
        }
    
    def get_model_info(self):
        """Get information about the trained model."""
        return {
            'algorithm_used': self.algorithm_used,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'model_name': self.model_name if self.algorithm_used == 'Transformer-Enhanced RF' else None,
            'best_params': self.best_params,
            'cv_accuracy': self.cv_scores['accuracy'].mean() if self.cv_scores else None
        }


def train_algorithm_5_transformer(X, y, texts=None):
    """
    Convenience function for training Algorithm 5: Transformer-based classifier.
    
    This function maintains compatibility with the notebook implementation.
    
    Args:
        X (np.array): Feature matrix (for fallback)
        y (np.array): Labels
        texts (list): Original tweet texts (for transformer)
        
    Returns:
        dict: Training results with model and performance metrics
    """
    transformer_classifier = TrumpTweetsTransformer()
    return transformer_classifier.train(X, y, texts=texts)