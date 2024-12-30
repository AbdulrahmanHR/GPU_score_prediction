import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(xgboost_lstm_metrics, lightgbm_lstm_metrics, 
                         xgboost_cnn_metrics, lightgbm_cnn_metrics):
    """
    Plot model performance comparison.
    Each metrics tuple contains (MAE, RMSE, R²)
    """
    # Prepare data for plotting
    models = ['XGBoost/LSTM', 'LightGBM/LSTM', 'XGBoost/CNN', 'LightGBM/CNN']
    
    # Each metric: MAE, RMSE, R²
    mae_values = [xgboost_lstm_metrics[0], lightgbm_lstm_metrics[0],
                 xgboost_cnn_metrics[0], lightgbm_cnn_metrics[0]]
    rmse_values = [xgboost_lstm_metrics[1], lightgbm_lstm_metrics[1],
                  xgboost_cnn_metrics[1], lightgbm_cnn_metrics[1]]
    r2_values = [xgboost_lstm_metrics[2], lightgbm_lstm_metrics[2],
                xgboost_cnn_metrics[2], lightgbm_cnn_metrics[2]]

    # Set up the first plot for MAE and RMSE
    x = np.arange(len(models))
    width = 0.35  # Adjusted width for two bars (MAE and RMSE)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot MAE and RMSE for each model
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='red')
    rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE',color='blue')

    # Customize the first plot
    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance Comparison (MAE and RMSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('model_performance_mae_rmse.png')
    plt.close()

    # Set up the second plot for R²
    fig, ax2 = plt.subplots(figsize=(12, 6))

    # Plot R² for each model
    ax2.bar(models, r2_values, color='lightgreen', label='R²')

    # Customize the second plot
    ax2.set_ylabel('R² Value')
    ax2.set_title('Model Performance Comparison (R²)')
    ax2.legend()

    # Add value labels on top of bars
    def autolabel_r2(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel_r2(ax2.patches)

    plt.tight_layout()
    plt.savefig('model_performance_r2.png')
    plt.close()