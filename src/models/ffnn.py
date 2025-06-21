"""
Algorithm 3: Feed-Forward Neural Network Model for Trump Tweets Classification

This module implements a PyTorch-based Feed-Forward Neural Network with at least one hidden layer
for the Trump tweets authorship attribution task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TweetFFNN(nn.Module):
    """
    Feed-Forward Neural Network for tweet classification.
    
    This network has at least one hidden layer as required by the assignment.
    Architecture: Input -> Hidden1 -> Hidden2 -> Output (2 classes)
    """
    
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout_rate=0.3):
        """
        Initialize the neural network.
        
        Args:
            input_size (int): Number of input features
            hidden_size1 (int): Size of first hidden layer
            hidden_size2 (int): Size of second hidden layer
            dropout_rate (float): Dropout rate for regularization
        """
        super(TweetFFNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, 2)  # Binary classification
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
        # Store architecture info
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_rate = dropout_rate
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output logits
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def get_architecture_info(self):
        """Get information about the network architecture."""
        return {
            'input_size': self.input_size,
            'hidden_size1': self.hidden_size1,
            'hidden_size2': self.hidden_size2,
            'dropout_rate': self.dropout_rate,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }


class TrumpTweetsFFNN:
    """
    Feed-Forward Neural Network classifier for Trump tweets authorship attribution.
    
    This class implements Algorithm 3 as specified in the assignment requirements.
    It uses PyTorch to build an FFNN with at least one hidden layer.
    """
    
    def __init__(self, hidden_size1=128, hidden_size2=64, dropout_rate=0.3, random_state=42):
        """
        Initialize the FFNN classifier.
        
        Args:
            hidden_size1 (int): Size of first hidden layer
            hidden_size2 (int): Size of second hidden layer
            dropout_rate (float): Dropout rate for regularization
            random_state (int): Random state for reproducibility
        """
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.training_history = None
        self.cv_scores = None
        
    def train(self, X, y, epochs=50, batch_size=32, learning_rate=0.001, 
              validation_split=0.2, patience=10, cv_folds=5, verbose=True):
        """
        Train the FFNN model with early stopping and cross-validation.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels (0=Trump, 1=Staffer)
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            validation_split (float): Fraction of data to use for validation
            patience (int): Early stopping patience
            cv_folds (int): Number of cross-validation folds
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training results with model and performance metrics
        """
        if verbose:
            print("Training Algorithm 3: Feed-Forward Neural Network (PyTorch)")
            print(f"Feature matrix shape: {X.shape}")
            print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Split into train/validation for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[1]
        self.model = TweetFFNN(input_size, self.hidden_size1, self.hidden_size2, self.dropout_rate)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop with validation and early stopping
        train_losses = []
        val_accuracies = []
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total
            train_losses.append(epoch_loss / len(train_loader))
            val_accuracies.append(val_acc)
            
            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.training_history = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': best_val_acc
        }
        
        # Cross-validation evaluation
        if verbose:
            print("\nPerforming cross-validation...")
        
        cv_scores = self._cross_validate(X, y, cv_folds, epochs=20, batch_size=batch_size, 
                                       learning_rate=learning_rate, verbose=verbose)
        
        self.cv_scores = {'accuracy': cv_scores}
        
        if verbose:
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        return {
            'model': self.model,
            'best_val_accuracy': best_val_acc,
            'training_history': self.training_history,
            'cv_scores': self.cv_scores,
            'algorithm': 'FFNN (PyTorch)',
            'hyperparameters': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'input_size': input_size,
                'architecture': self.model.get_architecture_info()
            }
        }
    
    def _cross_validate(self, X, y, cv_folds, epochs=20, batch_size=32, learning_rate=0.001, verbose=False):
        """
        Perform cross-validation evaluation.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Labels
            cv_folds (int): Number of CV folds
            epochs (int): Number of epochs for each fold
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            verbose (bool): Verbose output
            
        Returns:
            np.array: Cross-validation accuracy scores
        """
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        input_size = X.shape[1]
        criterion = nn.CrossEntropyLoss()
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Create and train model for this fold
            fold_model = TweetFFNN(input_size, self.hidden_size1, self.hidden_size2, self.dropout_rate)
            fold_optimizer = optim.Adam(fold_model.parameters(), lr=learning_rate)
            
            # Quick training for CV (fewer epochs)
            fold_model.train()
            for _ in range(epochs):
                fold_optimizer.zero_grad()
                outputs = fold_model(torch.FloatTensor(X_fold_train))
                loss = criterion(outputs, torch.LongTensor(y_fold_train))
                loss.backward()
                fold_optimizer.step()
            
            # Evaluate
            fold_model.eval()
            with torch.no_grad():
                outputs = fold_model(torch.FloatTensor(X_fold_val))
                _, predicted = torch.max(outputs.data, 1)
                fold_acc = (predicted == torch.LongTensor(y_fold_val)).float().mean().item()
                cv_scores.append(fold_acc)
            
            if verbose:
                print(f"  Fold {fold + 1}: {fold_acc:.4f}")
        
        return np.array(cv_scores)
    
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
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
    
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
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.numpy()
    
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
    
    def get_training_history(self):
        """Get training history for visualization."""
        return self.training_history


def train_algorithm_3_ffnn(X, y, epochs=50, batch_size=32, learning_rate=0.001):
    """
    Convenience function for training Algorithm 3: FFNN.
    
    This function maintains compatibility with the notebook implementation.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        
    Returns:
        dict: Training results with model and performance metrics
    """
    ffnn_classifier = TrumpTweetsFFNN()
    return ffnn_classifier.train(X, y, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)