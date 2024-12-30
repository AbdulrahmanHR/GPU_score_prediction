# main.py
import numpy as np
from data_preparation import DataPreparation
from models import HybridModels
from inference import InferencePipeline
from model_performance_chart import plot_model_performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __name__ == '__main__':
    # Data preparation
    data_prep = DataPreparation('gpu_specs_v6_score.csv')
    data = data_prep.preprocess_data()

    # Model training and evaluation
    hybrid_models = HybridModels(data)

    # Dictionary to store all predictions and metrics
    model_results = {}

    # Train and evaluate each model
    models = {
        'XGBoost/LSTM': hybrid_models.xgboost_lstm,
        'LightGBM/LSTM': hybrid_models.lightgbm_lstm,
        'XGBoost/CNN': hybrid_models.xgboost_cnn,
        'LightGBM/CNN': hybrid_models.lightgbm_cnn
    }

    for model_name, model_func in models.items():
        # Get predictions and true values
        predictions, true_values = model_func()
        
        # Calculate metrics
        mae = mean_absolute_error(true_values, predictions)
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        r2 = r2_score(true_values, predictions)
        
        # Store results
        model_results[model_name] = {
            'predictions': predictions,
            'true_values': true_values,
            'metrics': {
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2
            }
        }

        # Print model performance
        print(f"\n{model_name} Performance:")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")

    # Plot model performance
    plot_model_performance(
        (model_results['XGBoost/LSTM']['metrics']['MAE'], 
         model_results['XGBoost/LSTM']['metrics']['RMSE']),
        (model_results['LightGBM/LSTM']['metrics']['MAE'], 
         model_results['LightGBM/LSTM']['metrics']['RMSE']),
        (model_results['XGBoost/CNN']['metrics']['MAE'], 
         model_results['XGBoost/CNN']['metrics']['RMSE']),
        (model_results['LightGBM/CNN']['metrics']['MAE'], 
         model_results['LightGBM/CNN']['metrics']['RMSE'])
    )

    # Inference pipeline, This part will be tottaly changed
    inference_pipeline = InferencePipeline('xgboost_lstm', 'gpu_specs_v6_score.csv')
    new_data = inference_pipeline.preprocess_new_data()
    predictions = inference_pipeline.predict(new_data)