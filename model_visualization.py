# model_visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap

def create_plots_directory():
    """
    Create a directory to store visualization plots.
    """
    plots_dir = 'model_visualizations'
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def plot_predicted_vs_actual(model_results, plots_dir):
    """
    Create scatter plots of predicted vs actual performance scores.
    
    Args:
        model_results: Dictionary containing prediction results for different models
        plots_dir: Directory to save the plots
    """
    plt.figure(figsize=(15, 12))
    
    # Create a 2x2 grid for the four models
    for i, (model_name, results) in enumerate(model_results.items(), 1):
        predictions = results['predictions']
        true_values = results['true_values']
        
        # Plot setup
        plt.subplot(2, 2, i)
        
        # Create scatter plot
        scatter = plt.scatter(
            true_values, 
            predictions, 
            alpha=0.6, 
            edgecolors='w', 
            linewidth=0.5
        )
        
        # Add reference line (perfect predictions)
        min_val = min(min(true_values), min(predictions))
        max_val = max(max(true_values), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate and display metrics
        rmse = np.sqrt(mean_squared_error(true_values, predictions)) * 1000
        
        # Add annotations and formatting
        plt.title(f'{model_name} (RMSE: {rmse:.2f})', fontsize=12)
        plt.xlabel('Actual Score', fontsize=10)
        plt.ylabel('Predicted Score', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a colorbar to represent density
        plt.colorbar(scatter, label='Density')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'predicted_vs_actual.png'), dpi=300)
    plt.close()

def plot_residuals(model_results, plots_dir):
    """
    Create residual plots to analyze error patterns.
    
    Args:
        model_results: Dictionary containing prediction results for different models
        plots_dir: Directory to save the plots
    """
    plt.figure(figsize=(15, 12))
    
    # Create a 2x2 grid for the four models
    for i, (model_name, results) in enumerate(model_results.items(), 1):
        predictions = results['predictions']
        true_values = results['true_values']
        
        # Calculate residuals
        residuals = true_values - predictions
        
        # Plot setup
        plt.subplot(2, 2, i)
        
        # Create scatter plot with custom colormap for better visualization
        colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", 
                 "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
        cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
        
        scatter = plt.scatter(
            predictions, 
            residuals, 
            c=residuals, 
            cmap=cmap,
            alpha=0.7, 
            edgecolors='w', 
            linewidth=0.5
        )
        
        # Add zero reference line
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        # Add a trend line to visualize systematic bias
        z = np.polyfit(predictions, residuals, 1)
        p = np.poly1d(z)
        plt.plot(predictions, p(predictions), "k--", alpha=0.5, linewidth=2)
        
        # Calculate and display metrics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Add annotations and formatting
        plt.title(f'{model_name}\nMean Residual: {mean_residual:.4f}, Std: {std_residual:.4f}', fontsize=12)
        plt.xlabel('Predicted Score', fontsize=10)
        plt.ylabel('Residual (Actual - Predicted)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Residual Magnitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residual_plots.png'), dpi=300)
    plt.close()

def plot_learning_curves(model_history, plots_dir):
    """Plot learning curves with dynamic grid sizing"""
    valid_models = {k: v for k, v in model_history.items() if v is not None}
    n_models = len(valid_models)
    
    if n_models == 0:
        return
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(n_models)))
    cols = int(np.ceil(n_models / rows))
    
    plt.figure(figsize=(cols*6, rows*4))
    
    for i, (model_name, history) in enumerate(valid_models.items(), 1):
        plt.subplot(rows, cols, i)
        
        plt.plot(history['loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='red')
        
        # Add annotations and formatting
        plt.title(f'{model_name} Learning Curve', fontsize=12)
        plt.xlabel('Epochs', fontsize=10)
        plt.ylabel('Loss', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Add markers for the best epoch
        best_epoch = np.argmin(history['val_loss'])
        best_loss = history['val_loss'][best_epoch]
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
        plt.scatter(best_epoch, best_loss, s=100, color='green', zorder=5)
        plt.annotate(f'Best: {best_loss:.4f} @ epoch {best_epoch}',
                    (best_epoch, best_loss),
                    xytext=(best_epoch + 5, best_loss),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'learning_curves.png'), dpi=300)
    plt.close()

def plot_cross_validation_performance(cv_results, plots_dir):
    """
    Create a violin plot showing performance distribution across cross-validation folds.
    
    Args:
        cv_results: Dictionary containing cross-validation results for different models
        plots_dir: Directory to save the plots
    """
    # Prepare data for plotting
    model_names = []
    rmse_values = []
    mae_values = []
    r2_values = []
    
    for model_name, results in cv_results.items():
        for fold, metrics in results.items():
            model_names.append(model_name)
            rmse_values.append(metrics['rmse'])
            mae_values.append(metrics['mae'])
            r2_values.append(metrics['r2'])
    
    # Create DataFrame for easier plotting
    import pandas as pd
    cv_df = pd.DataFrame({
        'Model': model_names,
        'RMSE': rmse_values,
        'MAE': mae_values,
        'R²': r2_values
    })
    
    # Create separate plots for each metric
    metrics = ['RMSE', 'MAE', 'R²']
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 8))
    
    for i, metric in enumerate(metrics):
        # Create violin plot
        sns.violinplot(x='Model', y=metric, data=cv_df, ax=axes[i], 
                      inner='quartile', palette='Blues')
        
        # Add individual points
        sns.stripplot(x='Model', y=metric, data=cv_df, ax=axes[i], 
                     size=6, jitter=True, edgecolor='gray', linewidth=1, 
                     palette='Blues', alpha=0.5)
        
        # Add annotations and formatting
        axes[i].set_title(f'Cross-Validation {metric} Distribution', fontsize=12)
        axes[i].set_ylabel(metric, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Rotate x labels for better readability
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cross_validation_performance.png'), dpi=300)
    plt.close()

def collect_training_history(hybrid_models):
    """
    Collect training history from hybrid models for learning curves.
    
    Args:
        hybrid_models: HybridModels instance containing trained models
        
    Returns:
        Dictionary containing training history for each model
    """
    # Use the get_training_history method from HybridModels
    return hybrid_models.get_training_history()

def collect_cv_results(hybrid_models):
    """
    Collect cross-validation results from hybrid models.
    
    Args:
        hybrid_models: HybridModels instance containing trained models
        
    Returns:
        Dictionary containing cross-validation results for each model
    """
    # Use the get_cv_results method from HybridModels
    cv_results = hybrid_models.get_cv_results()
    
    # Extract fold results from the nested structure
    processed_cv_results = {}
    
    for model_name, model_results in cv_results.items():
        if not model_results:
            processed_cv_results[model_name] = {}
            continue
            
        # Extract fold-specific results if they exist
        if 'fold_results' in model_results:
            processed_cv_results[model_name] = model_results['fold_results']
        else:
            # Otherwise use the full model_results dictionary
            processed_cv_results[model_name] = model_results
    
    return processed_cv_results

def visualize_models(model_results, hybrid_models):
    """
    Generate and save all visualization plots.
    
    Args:
        model_results: Dictionary containing prediction results for different models
        hybrid_models: HybridModels instance containing trained models
    """
    # Create directory for plots
    plots_dir = create_plots_directory()
    
    # Generate all plots
    plot_predicted_vs_actual(model_results, plots_dir)
    plot_residuals(model_results, plots_dir)
    
    # Collect training history for learning curves
    model_history = collect_training_history(hybrid_models)
    plot_learning_curves(model_history, plots_dir)
    
    # Collect cross-validation results
    cv_results = collect_cv_results(hybrid_models)
    plot_cross_validation_performance(cv_results, plots_dir)
    
    print(f"All visualization plots have been saved to: {plots_dir}")