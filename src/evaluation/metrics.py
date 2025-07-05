"""
Enhanced evaluation metrics for Trump Tweets Classification project.
Provides detailed classification reports, confusion matrices, and performance visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def generate_classification_report(y_true, y_pred, algorithm_name, target_names=None):
    """
    Generate comprehensive classification report with all metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels  
        algorithm_name (str): Name of the algorithm
        target_names (list): Names for each class
        
    Returns:
        dict: Comprehensive classification metrics
    """
    if target_names is None:
        target_names = ['Trump (Android)', 'Staffer (iPhone/other)']
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION REPORT: {algorithm_name}")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall:    {recall:.4f} (weighted)")
    print(f"  F1-Score:  {f1:.4f} (weighted)")
    
    print(f"\nPer-Class Metrics:")
    for i, class_name in enumerate(target_names):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    print(f"{'':>12} {'Predicted':>20}")
    print(f"{'':>12} {target_names[0][:8]:>8} {target_names[1][:8]:>8}")
    print(f"Actual {target_names[0][:8]:>6} {cm[0,0]:>8d} {cm[0,1]:>8d}")
    print(f"       {target_names[1][:8]:>6} {cm[1,0]:>8d} {cm[1,1]:>8d}")
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"  Sensitivity (True Positive Rate):  {sensitivity:.4f}")
    print(f"  Specificity (True Negative Rate):  {specificity:.4f}")
    print(f"  False Positive Rate: {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "  False Positive Rate: 0.0000")
    print(f"  False Negative Rate: {fn/(fn+tp):.4f}" if (fn+tp) > 0 else "  False Negative Rate: 0.0000")
    
    return {
        'algorithm': algorithm_name,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'target_names': target_names
    }


def plot_confusion_matrix(cm, algorithm_name, target_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix with enhanced visualization.
    
    Args:
        cm (array): Confusion matrix
        algorithm_name (str): Name of the algorithm
        target_names (list): Names for each class
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Confusion matrix plot
    """
    if target_names is None:
        target_names = ['Trump\n(Android)', 'Staffer\n(iPhone/other)']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                ax=ax, cbar_kws={'label': 'Number of Predictions'})
    
    ax.set_title(f'Confusion Matrix: {algorithm_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Add accuracy information
    accuracy = np.trace(cm) / np.sum(cm)
    ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.4f}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, algorithm_name, figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true (array-like): True labels
        y_prob (array-like): Predicted probabilities for positive class
        algorithm_name (str): Name of the algorithm
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {algorithm_name}', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, roc_auc


def create_performance_comparison_table(results_dict):
    """
    Create comprehensive performance comparison table for all algorithms.
    
    Args:
        results_dict (dict): Dictionary with algorithm results
        
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    comparison_data = []
    
    for alg_id, metrics in results_dict.items():
        if isinstance(metrics, dict) and 'algorithm' in metrics:
            row = {
                'Algorithm': metrics['algorithm'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision (Weighted)': f"{metrics['precision_weighted']:.4f}",
                'Recall (Weighted)': f"{metrics['recall_weighted']:.4f}",
                'F1-Score (Weighted)': f"{metrics['f1_weighted']:.4f}",
                'Trump Precision': f"{metrics['precision_per_class'][0]:.4f}",
                'Trump Recall': f"{metrics['recall_per_class'][0]:.4f}",
                'Trump F1': f"{metrics['f1_per_class'][0]:.4f}",
                'Staffer Precision': f"{metrics['precision_per_class'][1]:.4f}",
                'Staffer Recall': f"{metrics['recall_per_class'][1]:.4f}",
                'Staffer F1': f"{metrics['f1_per_class'][1]:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}",
                'Sensitivity': f"{metrics['sensitivity']:.4f}"
            }
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*120)
    print("COMPREHENSIVE PERFORMANCE COMPARISON TABLE")
    print("="*120)
    print(df.to_string(index=False))
    
    return df


def evaluate_model_comprehensive(model, X_test, y_test, algorithm_name, 
                                target_names=None, plot_figures=True):
    """
    Comprehensive evaluation of a single model with all metrics and visualizations.
    
    Args:
        model: Trained model (sklearn or pytorch)
        X_test (array): Test features  
        y_test (array): Test labels
        algorithm_name (str): Name of the algorithm
        target_names (list): Names for each class
        plot_figures (bool): Whether to generate plots
        
    Returns:
        dict: Complete evaluation results
    """
    # Make predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        y_pred = model.predict(X_test)
    elif hasattr(model, 'decision_function'):
        decision_scores = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-decision_scores))  # Convert to probabilities
        y_pred = model.predict(X_test)
    else:
        # For models without probability estimates
        y_pred = model.predict(X_test)
        y_prob = None
    
    # Generate classification report
    metrics = generate_classification_report(y_test, y_pred, algorithm_name, target_names)
    
    if plot_figures:
        # Plot confusion matrix
        cm_fig = plot_confusion_matrix(metrics['confusion_matrix'], algorithm_name, target_names)
        metrics['confusion_matrix_plot'] = cm_fig
        
        # Plot ROC curve if probabilities available
        if y_prob is not None:
            roc_fig, roc_auc = plot_roc_curve(y_test, y_prob, algorithm_name)
            metrics['roc_curve_plot'] = roc_fig
            metrics['roc_auc'] = roc_auc
        
        plt.show()
    
    return metrics


def cross_validate_with_detailed_metrics(model, X, y, cv_folds=5, 
                                       algorithm_name="Unknown", random_state=42):
    """
    Perform cross-validation with detailed metrics collection.
    
    Args:
        model: Model to evaluate
        X (array): Features
        y (array): Labels  
        cv_folds (int): Number of CV folds
        algorithm_name (str): Name of the algorithm
        random_state (int): Random seed
        
    Returns:
        dict: Detailed cross-validation results
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Calculate multiple metrics
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}
    
    for metric in scoring_metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'ci': scores.std() * 2  # 95% confidence interval approximation
            }
        except Exception as e:
            print(f"Warning: Could not calculate {metric} for {algorithm_name}: {e}")
            cv_results[metric] = {'mean': 0.0, 'std': 0.0, 'ci': 0.0}
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION RESULTS: {algorithm_name}")
    print(f"{'='*50}")
    
    for metric, results in cv_results.items():
        if 'mean' in results:
            print(f"{metric.upper():>12}: {results['mean']:.4f} (±{results['ci']:.4f})")
    
    return cv_results
