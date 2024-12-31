# main.py
import numpy as np
from data_preparation import DataPreparation
from models import HybridModels
from inference import InferencePipeline
from model_performance_chart import plot_model_performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

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
    # Get model results including both models and predictions
    results = model_func()
    
    # Calculate metrics
    mae = mean_absolute_error(results['true_values'], results['predictions'])
    rmse = np.sqrt(mean_squared_error(results['true_values'], results['predictions']))
    r2 = r2_score(results['true_values'], results['predictions'])
    
    # Store results
    model_results[model_name] = {
        'feature_extractor': results['feature_extractor'],
        'predictor': results['predictor'],
        'predictions': results['predictions'],
        'true_values': results['true_values'],
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
        model_results['XGBoost/LSTM']['metrics']['RMSE'],
        model_results['XGBoost/LSTM']['metrics']['R²']),
    (model_results['LightGBM/LSTM']['metrics']['MAE'], 
        model_results['LightGBM/LSTM']['metrics']['RMSE'],
        model_results['LightGBM/LSTM']['metrics']['R²']),
    (model_results['XGBoost/CNN']['metrics']['MAE'], 
        model_results['XGBoost/CNN']['metrics']['RMSE'],
        model_results['XGBoost/CNN']['metrics']['R²']),
    (model_results['LightGBM/CNN']['metrics']['MAE'], 
        model_results['LightGBM/CNN']['metrics']['RMSE'],
        model_results['LightGBM/CNN']['metrics']['R²'])
)

# Inference pipeline

# Create directories if they don't exist
import os
os.makedirs('models/transformers/label_encoders', exist_ok=True)

# Save transformers
transformer_paths = {
    'label_encoders': {
        'gpuChip': 'models/transformers/label_encoders/le_gpuChip.pkl',
        'bus': 'models/transformers/label_encoders/le_bus.pkl',
        'memType': 'models/transformers/label_encoders/le_memType.pkl'
    },
    'knn_imputer': 'models/transformers/knn_imputer.pkl',
    'scaler': 'models/transformers/scaler.pkl'
}

# Save label encoders
for column, le in data_prep.label_encoders.items():
    joblib.dump(le, transformer_paths['label_encoders'][column])

# Save KNN imputer and scaler
joblib.dump(data_prep.knn_imputer, transformer_paths['knn_imputer'])
joblib.dump(data_prep.scaler, transformer_paths['scaler'])

# Save models
model_paths = {
    'xgboost_lstm': {
        'feature_extractor': 'models/xgboost_lstm_feature_extractor.pkl',
        'predictor': 'models/xgboost_lstm_predictor.h5'
    },
    'lightgbm_lstm': {
        'feature_extractor': 'models/lightgbm_lstm_feature_extractor.pkl',
        'predictor': 'models/lightgbm_lstm_predictor.h5'
    },
    'xgboost_cnn': {
        'feature_extractor': 'models/xgboost_cnn_feature_extractor.pkl',
        'predictor': 'models/xgboost_cnn_predictor.h5'
    },
    'lightgbm_cnn': {
        'feature_extractor': 'models/lightgbm_cnn_feature_extractor.pkl',
        'predictor': 'models/lightgbm_cnn_predictor.h5'
    },
    'transformers': transformer_paths
}

# # Save models after training
# for model_name, model_objects in model_results.items():
#     feature_extractor_path = f'models/{model_name.lower()}_feature_extractor.pkl'
#     predictor_path = f'models/{model_name.lower()}_predictor.h5'
    
#     joblib.dump(model_objects['feature_extractor'], feature_extractor_path)
#     model_objects['predictor'].save(predictor_path)