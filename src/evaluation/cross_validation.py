"""
Cross-validation utilities for Trump Tweets Classification project.
Provides comprehensive cross-validation with detailed reporting and comparison.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, 
    validation_curve, learning_curve
)
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def perform_detailed_cross_validation(models_dict, X, y, cv_folds=5, random_state=42):
    """
    Perform detailed cross-validation on multiple models with comprehensive reporting.
    
    Args:
        models_dict (dict): Dictionary of {model_name: model} pairs
        X (array): Feature matrix
        y (array): Labels
        cv_folds (int): Number of cross-validation folds
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Comprehensive cross-validation results for all models
    """
    print(f"\n{'='*70}")
    print(f"DETAILED CROSS-VALIDATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Models to evaluate: {len(models_dict)}")
    print(f"Cross-validation folds: {cv_folds}")
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall', 
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    all_results = {}
    
    for model_name, model in models_dict.items():
        print(f"\n{'-'*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'-'*50}")
        
        try:
            # Perform cross-validation with multiple metrics
            cv_results = cross_validate(
                model, X, y, cv=cv, scoring=scoring, 
                return_train_score=True, n_jobs=-1
            )
            
            # Calculate statistics for each metric
            results = {}
            for metric in scoring.keys():
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                results[metric] = {
                    'test_mean': test_scores.mean(),
                    'test_std': test_scores.std(),
                    'test_scores': test_scores,
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std(),
                    'train_scores': train_scores,
                    'overfitting': train_scores.mean() - test_scores.mean()
                }
                
                print(f"{metric.upper():>12}: Test={test_scores.mean():.4f}(±{test_scores.std()*2:.4f}) "
                      f"Train={train_scores.mean():.4f}(±{train_scores.std()*2:.4f}) "
                      f"Gap={train_scores.mean() - test_scores.mean():.4f}")
            
            # Store fit times
            results['fit_time'] = {
                'mean': cv_results['fit_time'].mean(),
                'std': cv_results['fit_time'].std()
            }
            
            results['model_name'] = model_name
            all_results[model_name] = results
            
            print(f"Average fit time: {cv_results['fit_time'].mean():.3f}s (±{cv_results['fit_time'].std():.3f}s)")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    return all_results


def create_cv_comparison_plot(cv_results, metric='accuracy', figsize=(12, 8)):
    """
    Create comparison plot for cross-validation results.
    
    Args:
        cv_results (dict): Results from perform_detailed_cross_validation
        metric (str): Metric to plot ('accuracy', 'f1', etc.)
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    model_names = list(cv_results.keys())
    test_means = [cv_results[name][metric]['test_mean'] for name in model_names]
    test_stds = [cv_results[name][metric]['test_std'] for name in model_names]
    train_means = [cv_results[name][metric]['train_mean'] for name in model_names]
    overfitting = [cv_results[name][metric]['overfitting'] for name in model_names]
    
    # Plot 1: Test vs Train Performance
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, test_means, width, 
                    yerr=[s*2 for s in test_stds], label='Test', 
                    capsize=5, alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x_pos + width/2, train_means, width, 
                    label='Train', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel(f'{metric.capitalize()} Score')
    ax1.set_title(f'Cross-Validation {metric.capitalize()} Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Overfitting Analysis
    colors = ['green' if gap < 0.02 else 'orange' if gap < 0.05 else 'red' for gap in overfitting]
    bars3 = ax2.bar(x_pos, overfitting, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Train - Test Gap')
    ax2.set_title('Overfitting Analysis (Train-Test Gap)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace(' ', '\n') for name in model_names], rotation=0)
    ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Caution (2%)')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Concern (5%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, gap in zip(bars3, overfitting):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{gap:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_learning_curves(models_dict, X, y, cv_folds=5, train_sizes=None, random_state=42):
    """
    Plot learning curves for multiple models to analyze training efficiency.
    
    Args:
        models_dict (dict): Dictionary of {model_name: model} pairs
        X (array): Feature matrix
        y (array): Labels
        cv_folds (int): Number of cross-validation folds
        train_sizes (array): Training set sizes to evaluate
        random_state (int): Random seed
        
    Returns:
        matplotlib.figure.Figure: Learning curves plot
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
    axes = axes.ravel() if n_models > 1 else [axes]
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        ax = axes[idx] if idx < len(axes) else None
        if ax is None:
            break
            
        try:
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X, y, cv=cv, train_sizes=train_sizes,
                scoring='accuracy', n_jobs=-1, random_state=random_state
            )
            
            train_mean = train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            test_mean = test_scores.mean(axis=1)
            test_std = test_scores.std(axis=1)
            
            ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                           alpha=0.1, color='blue')
            ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                           alpha=0.1, color='red')
            
            ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
            ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Cross-validation score')
            
            ax.set_title(f'Learning Curve: {model_name}')
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy Score')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Learning Curve: {model_name} (Error)')
    
    # Hide unused subplots
    for idx in range(len(models_dict), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def generate_cv_summary_table(cv_results):
    """
    Generate summary table from cross-validation results.
    
    Args:
        cv_results (dict): Results from perform_detailed_cross_validation
        
    Returns:
        pd.DataFrame: Formatted summary table
    """
    summary_data = []
    
    for model_name, results in cv_results.items():
        row = {
            'Model': model_name,
            'Test Accuracy': f"{results['accuracy']['test_mean']:.4f} ± {results['accuracy']['test_std']*2:.4f}",
            'Test F1': f"{results['f1']['test_mean']:.4f} ± {results['f1']['test_std']*2:.4f}",
            'Test Precision': f"{results['precision']['test_mean']:.4f} ± {results['precision']['test_std']*2:.4f}",
            'Test Recall': f"{results['recall']['test_mean']:.4f} ± {results['recall']['test_std']*2:.4f}",
            'Test AUC': f"{results['roc_auc']['test_mean']:.4f} ± {results['roc_auc']['test_std']*2:.4f}",
            'Overfitting (Acc)': f"{results['accuracy']['overfitting']:.4f}",
            'Fit Time (s)': f"{results['fit_time']['mean']:.3f} ± {results['fit_time']['std']:.3f}"
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Sort by test accuracy (descending)
    accuracy_values = [float(acc.split(' ±')[0]) for acc in df['Test Accuracy']]
    df['_sort_key'] = accuracy_values
    df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
    
    print(f"\n{'='*120}")
    print("CROSS-VALIDATION SUMMARY TABLE")
    print(f"{'='*120}")
    print(df.to_string(index=False))
    
    return df


def analyze_cv_statistical_significance(cv_results, alpha=0.05):
    """
    Analyze statistical significance of performance differences between models.
    
    Args:
        cv_results (dict): Results from perform_detailed_cross_validation
        alpha (float): Significance level
        
    Returns:
        dict: Statistical significance analysis
    """
    from scipy import stats
    
    model_names = list(cv_results.keys())
    significance_results = {}
    
    print(f"\n{'='*60}")
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Significance level (α): {alpha}")
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            
            # Get accuracy scores for both models
            scores1 = cv_results[model1]['accuracy']['test_scores']
            scores2 = cv_results[model2]['accuracy']['test_scores']
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            # Determine significance
            is_significant = p_value < alpha
            
            result = {
                'model1': model1,
                'model2': model2,
                'mean_diff': scores1.mean() - scores2.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'better_model': model1 if scores1.mean() > scores2.mean() else model2
            }
            
            significance_results[f"{model1}_vs_{model2}"] = result
            
            significance_marker = "*" if is_significant else ""
            print(f"{model1:25} vs {model2:25}: "
                  f"Δ={result['mean_diff']:+.4f}, p={p_value:.4f}{significance_marker}")
    
    # Count significant differences
    significant_count = sum(1 for r in significance_results.values() if r['is_significant'])
    total_comparisons = len(significance_results)
    
    print(f"\nSignificant differences: {significant_count}/{total_comparisons}")
    
    return significance_results


def select_best_model_from_cv(cv_results, criteria='accuracy', consider_overfitting=True):
    """
    Select the best model based on cross-validation results with multiple criteria.
    
    Args:
        cv_results (dict): Results from perform_detailed_cross_validation
        criteria (str): Primary selection criteria ('accuracy', 'f1', etc.)
        consider_overfitting (bool): Whether to penalize overfitting
        
    Returns:
        dict: Best model selection results
    """
    print(f"\n{'='*50}")
    print("BEST MODEL SELECTION")
    print(f"{'='*50}")
    print(f"Primary criteria: {criteria}")
    print(f"Consider overfitting: {consider_overfitting}")
    
    model_scores = {}
    
    for model_name, results in cv_results.items():
        base_score = results[criteria]['test_mean']
        overfitting_penalty = 0
        
        if consider_overfitting:
            # Penalize models with high overfitting
            overfitting = results[criteria]['overfitting']
            if overfitting > 0.05:  # 5% threshold
                overfitting_penalty = overfitting * 0.5  # 50% penalty weight
            elif overfitting > 0.02:  # 2% threshold
                overfitting_penalty = overfitting * 0.2  # 20% penalty weight
        
        final_score = base_score - overfitting_penalty
        
        model_scores[model_name] = {
            'base_score': base_score,
            'overfitting': results[criteria]['overfitting'],
            'overfitting_penalty': overfitting_penalty,
            'final_score': final_score,
            'std': results[criteria]['test_std'],
            'fit_time': results['fit_time']['mean']
        }
        
        print(f"{model_name:25}: {base_score:.4f} - {overfitting_penalty:.4f} = {final_score:.4f}")
    
    # Select best model
    best_model_name = max(model_scores.keys(), key=lambda x: model_scores[x]['final_score'])
    best_score = model_scores[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Final score: {best_score['final_score']:.4f}")
    print(f"Base {criteria}: {best_score['base_score']:.4f} (±{best_score['std']*2:.4f})")
    print(f"Overfitting: {best_score['overfitting']:.4f}")
    print(f"Fit time: {best_score['fit_time']:.3f}s")
    
    return {
        'best_model': best_model_name,
        'best_score': best_score['final_score'],
        'selection_criteria': criteria,
        'all_scores': model_scores
    }