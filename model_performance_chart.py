# model_performance_chart.py
import matplotlib.pyplot as plt

def plot_model_performance(xgb_lstm, lgb_lstm, xgb_cnn, lgb_cnn):
    metrics = ["MAE", "RMSE"]
    models = ["XGBoost/LSTM", "LightGBM/LSTM", "XGBoost/CNN", "LightGBM/CNN"]

    for i, metric in enumerate(metrics):
        values = [xgb_lstm[i], lgb_lstm[i], xgb_cnn[i], lgb_cnn[i]]
        plt.figure()
        plt.bar(models, values)
        plt.title(f"{metric} across models")
        plt.ylabel(metric)
        plt.savefig(f"{metric}_performance.png")