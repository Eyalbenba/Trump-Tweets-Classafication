"""
Visualization utilities for Trump Tweets Classification project.
Provides comprehensive visualization of model performance, comparisons, and insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_confusion_matrices_comparison(results_dict, figsize=(15, 10), save_path=None):
    """
    Plot confusion matrices for all models in a comparison grid.
    
    Args:
        results_dict (dict): Dictionary with model results containing confusion matrices
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    n_models = len(results_dict)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if n_models > 1 else [axes]
    
    target_names = ['Trump\n(Android)', 'Staffer\n(iPhone/other)']
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        cm = results.get('confusion_matrix', np.array([[0, 0], [0, 0]]))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names,
                   ax=ax, cbar=False)
        
        ax.set_title(f'{model_name}\nAccuracy: {results.get("accuracy", 0):.3f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Add performance metrics as text
        if 'precision_weighted' in results and 'recall_weighted' in results:
            metrics_text = f"Precision: {results['precision_weighted']:.3f}\nRecall: {results['recall_weighted']:.3f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(results_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves_comparison(models_dict, X_test, y_test, figsize=(10, 8), save_path=None):
    """
    Plot ROC curves for all models in a single comparison plot.
    
    Args:
        models_dict (dict): Dictionary of trained models
        X_test (array): Test features
        y_test (array): Test labels
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: ROC comparison plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_dict)))
    roc_aucs = {}
    
    for (model_name, model), color in zip(models_dict.items(), colors):
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-decision_scores))  # Convert to probabilities
            else:
                print(f"Cannot plot ROC for {model_name}: no probability estimates available")
                continue
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            roc_aucs[model_name] = roc_auc
            
            # Plot
            ax.plot(fpr, tpr, color=color, lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        except Exception as e:
            print(f"Error plotting ROC for {model_name}: {e}")
            continue
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
           label='Random Classifier (AUC = 0.5)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, roc_aucs


def plot_precision_recall_curves(models_dict, X_test, y_test, figsize=(10, 8), save_path=None):
    """
    Plot Precision-Recall curves for all models.
    
    Args:
        models_dict (dict): Dictionary of trained models
        X_test (array): Test features  
        y_test (array): Test labels
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: PR curves comparison plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(models_dict)))
    avg_precisions = {}
    
    # Calculate baseline (random classifier)
    baseline_precision = np.sum(y_test) / len(y_test)
    
    for (model_name, model), color in zip(models_dict.items(), colors):
        try:
            # Get prediction probabilities
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                y_prob = 1 / (1 + np.exp(-decision_scores))
            else:
                continue
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
            avg_precisions[model_name] = avg_precision
            
            # Plot
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{model_name} (AP = {avg_precision:.3f})')
        
        except Exception as e:
            print(f"Error plotting PR curve for {model_name}: {e}")
            continue
    
    # Plot baseline
    ax.axhline(y=baseline_precision, color='gray', linestyle='--', lw=1,
              label=f'Random Classifier (AP = {baseline_precision:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, avg_precisions


def plot_performance_comparison_bar(results_dict, metrics=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], 
                                  figsize=(12, 8), save_path=None):
    """
    Create comprehensive bar plot comparing model performance across multiple metrics.
    
    Args:
        results_dict (dict): Dictionary with model results
        metrics (list): List of metrics to compare
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Performance comparison plot
    """
    # Prepare data
    model_names = list(results_dict.keys())
    n_metrics = len(metrics)
    n_models = len(model_names)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize) if n_metrics == 4 else plt.subplots(1, n_metrics, figsize=figsize)
    axes = axes.ravel() if n_metrics > 1 else [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract metric values
        metric_values = []
        for model_name in model_names:
            value = results_dict[model_name].get(metric, 0)
            metric_values.append(value)
        
        # Create bar plot
        bars = ax.bar(model_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Formatting
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-labels for better readability
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0, ha='center')
        
        # Highlight best performance
        best_idx = np.argmax(metric_values)
        bars[best_idx].set_facecolor('gold')
        bars[best_idx].set_edgecolor('orange')
        bars[best_idx].set_linewidth(2)
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance_comparison(models_dict, feature_names=None, top_k=15, figsize=(12, 10), save_path=None):
    """
    Plot feature importance comparison for models that support it.
    
    Args:
        models_dict (dict): Dictionary of trained models
        feature_names (list): Names of features
        top_k (int): Number of top features to show
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Feature importance comparison plot
    """
    # Find models with feature importance
    importance_models = {}
    
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            importance_models[model_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficient values
            importance_models[model_name] = np.abs(model.coef_[0])
    
    if not importance_models:
        print("No models with feature importance found")
        return None
    
    n_models = len(importance_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(figsize[0], figsize[1] * n_models // 2))
    axes = [axes] if n_models == 1 else axes
    
    for idx, (model_name, importance) in enumerate(importance_models.items()):
        ax = axes[idx]
        
        # Get top k features
        top_indices = np.argsort(importance)[-top_k:]
        top_importance = importance[top_indices]
        
        if feature_names is not None:
            top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in top_indices]
        else:
            top_features = [f'Feature_{i}' for i in top_indices]
        
        # Create horizontal bar plot
        bars = ax.barh(range(top_k), top_importance, color=plt.cm.viridis(np.linspace(0, 1, top_k)))
        
        # Formatting
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_k} Features: {model_name}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, importance_val in zip(bars, top_importance):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance_val:.3f}', ha='left', va='center', fontsize=9)
    
    plt.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_performance_heatmap(results_dict, metrics=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                             figsize=(10, 6), save_path=None):
    """
    Create heatmap showing performance of all models across all metrics.
    
    Args:
        results_dict (dict): Dictionary with model results
        metrics (list): List of metrics to include
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Performance heatmap
    """
    # Prepare data matrix
    model_names = list(results_dict.keys())
    data_matrix = []
    
    for model_name in model_names:
        row = []
        for metric in metrics:
            value = results_dict[model_name].get(metric, 0)
            row.append(value)
        data_matrix.append(row)
    
    data_matrix = np.array(data_matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    
    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', rotation=270, labelpad=20)
    
    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_analysis(models_dict, X_test, y_test, feature_names=None, figsize=(15, 10), save_path=None):
    """
    Create comprehensive error analysis visualization.
    
    Args:
        models_dict (dict): Dictionary of trained models
        X_test (array): Test features
        y_test (array): Test labels
        feature_names (list): Names of features
        figsize (tuple): Figure size
        save_path (str): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: Error analysis plot
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, n_models, figsize=figsize)
    axes = axes.reshape(2, -1) if n_models > 1 else axes.reshape(2, 1)
    
    target_names = ['Trump (Android)', 'Staffer (iPhone/other)']
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        if idx >= axes.shape[1]:
            break
            
        try:
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate prediction confidence if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                confidence = np.max(y_prob, axis=1)
            elif hasattr(model, 'decision_function'):
                decision_scores = model.decision_function(X_test)
                confidence = np.abs(decision_scores)
            else:
                confidence = np.ones(len(y_pred))  # Default confidence
            
            # Error analysis - top plot: error distribution by confidence
            ax1 = axes[0, idx]
            errors = (y_pred != y_test).astype(int)
            
            # Bin by confidence levels
            conf_bins = np.linspace(0, 1, 11)
            error_rates = []
            bin_centers = []
            
            for i in range(len(conf_bins) - 1):
                mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i+1])
                if np.sum(mask) > 0:
                    error_rate = np.mean(errors[mask])
                    error_rates.append(error_rate)
                    bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
            
            ax1.bar(bin_centers, error_rates, width=0.08, alpha=0.7, color='red')
            ax1.set_xlabel('Prediction Confidence')
            ax1.set_ylabel('Error Rate')
            ax1.set_title(f'{model_name}: Error vs Confidence')
            ax1.grid(True, alpha=0.3)
            
            # Error analysis - bottom plot: confusion by class
            ax2 = axes[1, idx]
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                       xticklabels=target_names, yticklabels=target_names,
                       ax=ax2, cbar=False)
            
            ax2.set_title(f'{model_name}: Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
            
        except Exception as e:
            # Handle errors gracefully
            axes[0, idx].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                             transform=axes[0, idx].transAxes, ha='center', va='center')
            axes[1, idx].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                             transform=axes[1, idx].transAxes, ha='center', va='center')
    
    # Hide unused subplots
    for idx in range(len(models_dict), axes.shape[1]):
        axes[0, idx].set_visible(False)
        axes[1, idx].set_visible(False)
    
    plt.suptitle('Error Analysis Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_all_performance_plots(results_dict, models_dict, X_test, y_test, 
                             output_dir='results/figures', prefix='performance'):
    """
    Generate and save all performance visualization plots.
    
    Args:
        results_dict (dict): Dictionary with model evaluation results
        models_dict (dict): Dictionary of trained models
        X_test (array): Test features
        y_test (array): Test labels
        output_dir (str): Directory to save plots
        prefix (str): Prefix for file names
        
    Returns:
        dict: Dictionary of saved plot paths
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    saved_plots = {}
    
    try:
        # 1. Confusion matrices comparison
        fig1 = plot_confusion_matrices_comparison(results_dict)
        path1 = os.path.join(output_dir, f'{prefix}_confusion_matrices.png')
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        saved_plots['confusion_matrices'] = path1
        plt.close(fig1)
        
        # 2. ROC curves comparison
        fig2, roc_aucs = plot_roc_curves_comparison(models_dict, X_test, y_test)
        path2 = os.path.join(output_dir, f'{prefix}_roc_curves.png')
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        saved_plots['roc_curves'] = path2
        plt.close(fig2)
        
        # 3. Performance comparison bars
        fig3 = plot_performance_comparison_bar(results_dict)
        path3 = os.path.join(output_dir, f'{prefix}_performance_bars.png')
        fig3.savefig(path3, dpi=300, bbox_inches='tight')
        saved_plots['performance_bars'] = path3
        plt.close(fig3)
        
        # 4. Performance heatmap
        fig4 = create_performance_heatmap(results_dict)
        path4 = os.path.join(output_dir, f'{prefix}_heatmap.png')
        fig4.savefig(path4, dpi=300, bbox_inches='tight')
        saved_plots['heatmap'] = path4
        plt.close(fig4)
        
        print(f"All performance plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    return saved_plots