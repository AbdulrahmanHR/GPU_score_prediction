# main.py
import numpy as np
from data_preparation import DataPreparation
from models import HybridModels
from inference import InferencePipeline
from model_performance_chart import plot_model_performance

if __name__ == '__main__':
    # Data preparation
    data_prep = DataPreparation('gpu_specs_v6_score.csv')
    data = data_prep.preprocess_data()

    # Model training and evaluation
    hybrid_models = HybridModels(data)

    xgboost_lstm_preds = hybrid_models.xgboost_lstm()
    lightgbm_lstm_preds = hybrid_models.lightgbm_lstm()
    xgboost_cnn_preds = hybrid_models.xgboost_cnn()
    lightgbm_cnn_preds = hybrid_models.lightgbm_cnn()

    xgboost_lstm_mae = np.mean(abs(xgboost_lstm_preds - data["score"]))
    xgboost_lstm_rmse = np.sqrt(np.mean((xgboost_lstm_preds - data["score"]) ** 2))
    xgboost_lstm_r2 = 1 - (np.sum((xgboost_lstm_preds - data["score"]) ** 2) / np.sum((data["score"] - np.mean(data["score"])) ** 2))

    lightgbm_lstm_mae = np.mean(abs(lightgbm_lstm_preds - data["score"]))
    lightgbm_lstm_rmse = np.sqrt(np.mean((lightgbm_lstm_preds - data["score"]) ** 2))
    lightgbm_lstm_r2 = 1 - (np.sum((lightgbm_lstm_preds - data["score"]) ** 2) / np.sum((data["score"] - np.mean(data["score"])) ** 2))

    xgboost_cnn_mae = np.mean(abs(xgboost_cnn_preds - data["score"]))
    xgboost_cnn_rmse = np.sqrt(np.mean((xgboost_cnn_preds - data["score"]) ** 2))
    xgboost_cnn_r2 = 1 - (np.sum((xgboost_cnn_preds - data["score"]) ** 2) / np.sum((data["score"] - np.mean(data["score"])) ** 2))

    lightgbm_cnn_mae = np.mean(abs(lightgbm_cnn_preds - data["score"]))
    lightgbm_cnn_rmse = np.sqrt(np.mean((lightgbm_cnn_preds - data["score"]) ** 2))
    lightgbm_cnn_r2 = 1 - (np.sum((lightgbm_cnn_preds - data["score"]) ** 2) / np.sum((data["score"] - np.mean(data["score"])) ** 2))

    # Print model performance
    print("XGBoost/LSTM MAE:", xgboost_lstm_mae)
    print("XGBoost/LSTM RMSE:", xgboost_lstm_rmse)
    print("XGBoost/LSTM R²:", xgboost_lstm_r2)

    print("LightGBM/LSTM MAE:", lightgbm_lstm_mae)
    print("LightGBM/LSTM RMSE:", lightgbm_lstm_rmse)
    print("LightGBM/LSTM R²:", lightgbm_lstm_r2)

    print("XGBoost/CNN MAE:", xgboost_cnn_mae)
    print("XGBoost/CNN RMSE:", xgboost_cnn_rmse)
    print("XGBoost/CNN R²:", xgboost_cnn_r2)

    print("LightGBM/CNN MAE:", lightgbm_cnn_mae)
    print("LightGBM/CNN RMSE:", lightgbm_cnn_rmse)
    print("LightGBM/CNN R²:", lightgbm_cnn_r2)

    # Plot model performance
    plot_model_performance(
        (xgboost_lstm_mae, xgboost_lstm_rmse),
        (lightgbm_lstm_mae, lightgbm_lstm_rmse),
        (xgboost_cnn_mae, xgboost_cnn_rmse),
        (lightgbm_cnn_mae, lightgbm_cnn_rmse)
    )

    # Inference pipeline
    inference_pipeline = InferencePipeline('xgboost_lstm', 'gpu_specs_v6_score.csv')
    new_data = inference_pipeline.preprocess_new_data()
    predictions = inference_pipeline.predict(new_data)