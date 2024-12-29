# model_performance_chart.py
import matplotlib.pyplot as plt
import numpy as np

def plot_model_performance(xgboost_lstm_metrics, lightgbm_lstm_metrics, 
                         xgboost_cnn_metrics, lightgbm_cnn_metrics):
    """
    Plot model performance comparison.
    Each metrics tuple contains (MAE, RMSE)
    """
    # Prepare data for plotting
    models = ['XGBoost/LSTM', 'LightGBM/LSTM', 'XGBoost/CNN', 'LightGBM/CNN']
    mae_values = [xgboost_lstm_metrics[0], lightgbm_lstm_metrics[0],
                 xgboost_cnn_metrics[0], lightgbm_cnn_metrics[0]]
    rmse_values = [xgboost_lstm_metrics[1], lightgbm_lstm_metrics[1],
                  xgboost_cnn_metrics[1], lightgbm_cnn_metrics[1]]

    # Set up the plot
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE')
    rects2 = ax.bar(x + width/2, rmse_values, width, label='RMSE')

    # Customize the plot
    ax.set_ylabel('Error Value')
    ax.set_title('Model Performance Comparison')
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
    plt.savefig('model_performance.png')
    plt.close()