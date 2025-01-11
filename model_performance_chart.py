# model_performance_chart.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_fold_metrics(predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    r2 = r2_score(true_values, predictions)
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    return mae, rmse, r2, mape

def plot_model_performance(model_results):

    # Create charts directory if it doesn't exist
    import os
    charts_dir = 'charts'
    os.makedirs(charts_dir, exist_ok=True)
    models = list(model_results.keys())
    metrics = {}
    
    # Calculate metrics for each model
    for model_name in models:
        predictions = model_results[model_name]['predictions']
        true_values = model_results[model_name]['true_values']
        
        mae, rmse, r2, mape = calculate_fold_metrics(predictions, true_values)
        metrics[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

    # Helper function for adding value labels
    def autolabel(rects, ax, is_percentage=False):
        for rect in rects:
            height = rect.get_height()
            if is_percentage:
                ax.annotate(f'{height:.2f}%',  # Format as percentage with 2 decimal places
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)
            else:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width()/2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=8)

    # MAE and RMSE plot
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    mae_values = [metrics[model]['mae'] * 1000 for model in models]
    rmse_values = [metrics[model]['rmse'] * 1000 for model in models]
    
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='red')
    rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE', color='blue')

    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance (MAE and RMSE) with stratified K-fold CV')
    ax.set_xticks(x)
    display_names = [model.replace('_', '/').upper() for model in models]
    ax.set_xticklabels(display_names, rotation=45)    
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'performance_mae_rmse_kfold.png'))
    plt.close()

    # R² plot
    fig, ax = plt.subplots(figsize=(12, 6))
    r2_values = [metrics[model]['r2'] * 100 for model in models]
    rects = ax.bar(models, r2_values, color='limegreen')

    ax.set_ylabel('R² Value (%)')
    ax.set_title('Model Performance (R²) with stratified K-fold CV')
    display_names = [model.replace('_', '/').upper() for model in models]
    ax.set_xticklabels(display_names, rotation=45)    
    
    autolabel(rects, ax, is_percentage=True)

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'performance_r2_kfold.png'))
    plt.close()

    # MAPE plot
    fig, ax = plt.subplots(figsize=(12, 6))
    mape_values = [metrics[model]['mape'] for model in models]
    rects = ax.bar(models, mape_values, color='indigo')

    ax.set_ylabel('MAPE (%)')
    ax.set_title('Model Performance (MAPE) with stratified K-fold CV')
    display_names = [model.replace('_', '/').upper() for model in models]
    ax.set_xticklabels(display_names, rotation=45)    
    
    autolabel(rects, ax, is_percentage=True)

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, 'performance_mape_kfold.png'))
    plt.close()
    
    # Return metrics dictionary for further use if needed
    return metrics