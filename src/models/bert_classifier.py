"""
Algorithm 5: BERT/RoBERTa Fine-tuning for Trump Tweets Classification

This module implements a proper transformer-based classifier using BERT or RoBERTa
fine-tuning for the Trump tweets authorship attribution task.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import transformers and torch
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class TweetDataset(Dataset):
    """Custom dataset for tweet classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTTrumpClassifier:
    """
    BERT/RoBERTa-based classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 5 as specified in the assignment requirements.
    It fine-tunes pre-trained BERT/RoBERTa models for the binary classification task.
    """
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=128, 
                 random_state=42, fallback_to_nb=True):
        """
        Initialize the BERT classifier.
        
        Args:
            model_name (str): Hugging Face model name (e.g., 'distilbert-base-uncased', 
                            'roberta-base', 'bert-base-uncased')
            max_length (int): Maximum token length for input texts
            random_state (int): Random state for reproducibility
            fallback_to_nb (bool): Whether to fallback to Naive Bayes if BERT fails
        """
        self.model_name = model_name
        self.max_length = max_length
        self.random_state = random_state
        self.fallback_to_nb = fallback_to_nb
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.algorithm_used = None
        self.training_results = None
        
        # Fallback components
        self.nb_model = None
        self.scaler = None
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    def train(self, X, y, texts, validation_split=0.2, num_epochs=3, 
              batch_size=16, learning_rate=2e-5, verbose=True):
        """
        Train the BERT classifier with fine-tuning.
        
        Args:
            X (np.array): Feature matrix (for fallback only)
            y (np.array): Labels (0=Trump, 1=Staffer)
            texts (list): Original tweet texts for BERT training
            validation_split (float): Fraction of data to use for validation
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for fine-tuning
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with model and performance metrics
        """
        if verbose:
            print("Training Algorithm 5: BERT/RoBERTa Fine-tuning Classifier")
            print(f"Model: {self.model_name}")
            print(f"Training samples: {len(texts)}")
        
        # Try BERT approach first
        if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
            try:
                return self._train_bert(X, y, texts, validation_split, num_epochs, 
                                     batch_size, learning_rate, verbose)
            except Exception as e:
                if verbose:
                    print(f"BERT training failed: {e}")
                    print("Falling back to CPU training...")
                
                # Try CPU training
                try:
                    return self._train_bert_cpu(X, y, texts, validation_split, 
                                              num_epochs, batch_size, learning_rate, verbose)
                except Exception as e2:
                    if verbose:
                        print(f"CPU BERT training also failed: {e2}")
                        print("Falling back to Naive Bayes...")
        
        elif TRANSFORMERS_AVAILABLE:
            try:
                return self._train_bert_cpu(X, y, texts, validation_split, 
                                          num_epochs, batch_size, learning_rate, verbose)
            except Exception as e:
                if verbose:
                    print(f"BERT training failed: {e}")
                    print("Falling back to Naive Bayes...")
        
        # Fallback to Naive Bayes
        if self.fallback_to_nb:
            return self._train_naive_bayes_fallback(X, y, verbose)
        else:
            raise RuntimeError("BERT training failed and fallback disabled")
    
    def _train_bert(self, X, y, texts, validation_split, num_epochs, 
                   batch_size, learning_rate, verbose):
        """Train using BERT with GPU acceleration."""
        return self._train_bert_common(X, y, texts, validation_split, num_epochs, 
                                     batch_size, learning_rate, verbose, use_gpu=True)
    
    def _train_bert_cpu(self, X, y, texts, validation_split, num_epochs, 
                       batch_size, learning_rate, verbose):
        """Train using BERT with CPU only."""
        return self._train_bert_common(X, y, texts, validation_split, num_epochs, 
                                     batch_size, learning_rate, verbose, use_gpu=False)
    
    def _train_bert_common(self, X, y, texts, validation_split, num_epochs, 
                          batch_size, learning_rate, verbose, use_gpu=True):
        """Common BERT training logic."""
        if verbose:
            device_str = "GPU" if use_gpu and torch.cuda.is_available() else "CPU"
            print(f"Using {device_str} for BERT training...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2,
            id2label={0: "Trump", 1: "Staffer"},
            label2id={"Trump": 0, "Staffer": 1}
        )
        
        # Split data for training and validation
        texts_train, texts_val, y_train, y_val = train_test_split(
            texts, y, test_size=validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        if verbose:
            print(f"Training set: {len(texts_train)} samples")
            print(f"Validation set: {len(texts_val)} samples")
        
        # Create datasets
        train_dataset = TweetDataset(texts_train, y_train, self.tokenizer, self.max_length)
        val_dataset = TweetDataset(texts_val, y_val, self.tokenizer, self.max_length)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            learning_rate=learning_rate,
            seed=self.random_state,
            no_cuda=not use_gpu
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions)
            }
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        if verbose:
            print("Starting BERT fine-tuning...")
        
        train_result = self.trainer.train()
        
        # Evaluate on validation set
        eval_results = self.trainer.evaluate()
        
        self.algorithm_used = f'BERT Fine-tuned ({self.model_name})'
        self.training_results = {
            'train_loss': train_result.training_loss,
            'eval_results': eval_results
        }
        
        if verbose:
            print(f"Training completed!")
            print(f"Final validation accuracy: {eval_results['eval_accuracy']:.4f}")
            print(f"Final validation F1: {eval_results['eval_f1']:.4f}")
        
        return {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'trainer': self.trainer,
            'algorithm': self.algorithm_used,
            'features_used': 'BERT Token Embeddings',
            'training_results': self.training_results,
            'eval_accuracy': eval_results['eval_accuracy'],
            'eval_f1': eval_results['eval_f1']
        }
    
    def _train_naive_bayes_fallback(self, X, y, verbose):
        """Fallback training using Naive Bayes."""
        if verbose:
            print("Using Naive Bayes as BERT fallback...")
        
        # Scale features for Multinomial NB
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X) + 1e-10
        
        # Train Naive Bayes
        self.nb_model = MultinomialNB(alpha=1.0)
        self.nb_model.fit(X_scaled, y)
        
        # Evaluate with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.nb_model, X_scaled, y, cv=cv, scoring='accuracy')
        
        self.algorithm_used = 'Naive Bayes (BERT Fallback)'
        
        if verbose:
            print(f"Naive Bayes accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return {
            'model': self.nb_model,
            'scaler': self.scaler,
            'algorithm': self.algorithm_used,
            'features_used': 'TF-IDF + Stylistic Features',
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X, texts):
        """
        Make predictions on new data.
        
        Args:
            X (np.array): Feature matrix (for fallback)
            texts (list): Original texts (for BERT)
            
        Returns:
            np.array: Predictions (0s and 1s)
        """
        if self.algorithm_used is None:
            raise ValueError("Model must be trained before making predictions")
        
        if 'BERT' in self.algorithm_used and self.model is not None:
            return self._predict_bert(texts)
        elif self.nb_model is not None:
            return self._predict_naive_bayes(X)
        else:
            raise ValueError("No trained model available for predictions")
    
    def _predict_bert(self, texts):
        """Make predictions using BERT model."""
        predictions = []
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _predict_naive_bayes(self, X):
        """Make predictions using Naive Bayes fallback."""
        X_scaled = self.scaler.transform(X) + 1e-10
        return self.nb_model.predict(X_scaled)
    
    def predict_proba(self, X, texts):
        """
        Get prediction probabilities.
        
        Args:
            X (np.array): Feature matrix (for fallback)
            texts (list): Original texts (for BERT)
            
        Returns:
            np.array: Prediction probabilities
        """
        if self.algorithm_used is None:
            raise ValueError("Model must be trained before making predictions")
        
        if 'BERT' in self.algorithm_used and self.model is not None:
            return self._predict_proba_bert(texts)
        elif self.nb_model is not None:
            return self._predict_proba_naive_bayes(X)
        else:
            raise ValueError("No trained model available for predictions")
    
    def _predict_proba_bert(self, texts):
        """Get prediction probabilities using BERT model."""
        probabilities = []
        
        # Process in batches
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get probabilities
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                probabilities.extend(batch_probs)
        
        return np.array(probabilities)
    
    def _predict_proba_naive_bayes(self, X):
        """Get prediction probabilities using Naive Bayes fallback."""
        X_scaled = self.scaler.transform(X) + 1e-10
        return self.nb_model.predict_proba(X_scaled)
    
    def evaluate(self, X_test, y_test, texts_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.array): Test feature matrix
            y_test (np.array): Test labels
            texts_test (list): Test texts
            
        Returns:
            dict: Evaluation metrics
        """
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
            'model_name': self.model_name,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'torch_available': torch.cuda.is_available() if 'torch' in globals() else False,
            'max_length': self.max_length,
            'training_results': self.training_results
        }


def train_algorithm_5_bert(X, y, texts):
    """
    Convenience function for training Algorithm 5: BERT/RoBERTa classifier.
    
    Args:
        X (np.array): Feature matrix (for fallback)
        y (np.array): Labels
        texts (list): Original tweet texts
        
    Returns:
        dict: Training results with model and performance metrics
    """
    # Try DistilBERT first (faster), then RoBERTa if needed
    models_to_try = [
        "distilbert-base-uncased",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "roberta-base"
    ]
    
    for model_name in models_to_try:
        try:
            bert_classifier = BERTTrumpClassifier(
                model_name=model_name,
                max_length=128,
                random_state=42
            )
            
            result = bert_classifier.train(
                X, y, texts,
                validation_split=0.2,
                num_epochs=2,  # Reduced for faster training
                batch_size=16,
                learning_rate=2e-5,
                verbose=True
            )
            
            # Add the classifier instance to the result
            result['classifier_instance'] = bert_classifier
            return result
            
        except Exception as e:
            print(f"Failed to train with {model_name}: {e}")
            continue
    
    # If all BERT models fail, fallback to naive bayes
    print("All BERT models failed, using Naive Bayes fallback")
    bert_classifier = BERTTrumpClassifier(fallback_to_nb=True)
    result = bert_classifier.train(X, y, texts, verbose=True)
    result['classifier_instance'] = bert_classifier
    return result