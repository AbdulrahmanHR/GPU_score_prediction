# main.py
import numpy as np
from data_preparation import DataPreparation
from models import HybridModels
from model_performance_chart import plot_model_performance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


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

train_metrics = {}
test_metrics = {}

for model_name, model_func in models.items():
    # Get model results including both models and predictions
    results = model_func()
    
    # Calculate training metrics using the new return values
    train_mae = mean_absolute_error(results['train_true_values'], results['train_predictions'])
    train_rmse = np.sqrt(mean_squared_error(results['train_true_values'], results['train_predictions']))
    train_r2 = r2_score(results['train_true_values'], results['train_predictions'])
    
    # Calculate test metrics
    test_mae = mean_absolute_error(results['true_values'], results['predictions'])
    test_rmse = np.sqrt(mean_squared_error(results['true_values'], results['predictions']))
    test_r2 = r2_score(results['true_values'], results['predictions'])
    
    # Store metrics
    train_metrics[model_name] = (train_mae, train_rmse, train_r2)
    test_metrics[model_name] = (test_mae, test_rmse, test_r2)
    
    # Print model performance
    print(f"\n{model_name} Performance:")
    print("Training Metrics:")
    print(f"MAE: {train_mae:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    print(f"R²: {train_r2:.4f}")
    print("\nTesting Metrics:")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R²: {test_r2:.4f}")

# Plot model performance
plot_model_performance(train_metrics, test_metrics)

# Hyperparameters values
# xgb_lstm_result = hybrid_models.xgboost_lstm()
# print("Best XGBoost Parameters:", xgb_lstm_result['feature_extractor'].get_params())

# Inference pipeline, this part is not fully functional

# Create directories if they don't exist
os.makedirs('models/data_processing/label_encoders', exist_ok=True)

# Save data processing
data_processing_paths = {
    'label_encoders': {
        'gpuChip': 'models/data_processing/label_encoders/le_gpuChip.pkl',
        'bus': 'models/data_processing/label_encoders/le_bus.pkl',
        'memType': 'models/data_processing/label_encoders/le_memType.pkl'
    },
    'knn_imputer': 'models/data_processing/knn_imputer.pkl',
    'scaler': 'models/data_processing/scaler.pkl'
}

# Save label encoders
for column, le in data_prep.label_encoders.items():
    joblib.dump(le, data_processing_paths['label_encoders'][column])

# Save KNN imputer and scaler
joblib.dump(data_prep.knn_imputer, data_processing_paths['knn_imputer'])
joblib.dump(data_prep.scaler, data_processing_paths['scaler'])

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
    'data_processing': data_processing_paths
}

# Save models after training
for model_name, model_objects in model_results.items():
    # Create directory structure
    model_dir = f"models/{model_name.replace('/', '_').lower()}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Define file paths
    feature_extractor_path = f"{model_dir}/feature_extractor.pkl"
    predictor_path = f"{model_dir}/predictor.h5"
    
    # Save feature extractor and predictor
    joblib.dump(model_objects['feature_extractor'], feature_extractor_path)
    model_objects['predictor'].save(predictor_path)

# Save data processing (already handled earlier in your script)
data_processing_paths = {
    'label_encoders': {
        'gpuChip': 'models/data_processing/label_encoders/le_gpuChip.pkl',
        'bus': 'models/data_processing/label_encoders/le_bus.pkl',
        'memType': 'models/data_processing/label_encoders/le_memType.pkl'
    },
    'knn_imputer': 'models/data_processing/knn_imputer.pkl',
    'scaler': 'models/data_processing/scaler.pkl'
} 

# Ensure directories for data processing
os.makedirs('models/data_processing/label_encoders', exist_ok=True)

# Save label encoders
for column, le in data_prep.label_encoders.items():
    joblib.dump(le, data_processing_paths['label_encoders'][column])

# Save KNN imputer and scaler
joblib.dump(data_prep.knn_imputer, data_processing_paths['knn_imputer'])
joblib.dump(data_prep.scaler, data_processing_paths['scaler'])

# Update `model_paths` dictionary to match structure in `InferencePipeline`
model_paths = {
    'data_processing': data_processing_paths,
    'xgboost_lstm': {
        'feature_extractor': 'models/xgboost_lstm/feature_extractor.pkl',
        'predictor': 'models/xgboost_lstm/predictor.h5'
    },
    'lightgbm_lstm': {
        'feature_extractor': 'models/lightgbm_lstm/feature_extractor.pkl',
        'predictor': 'models/lightgbm_lstm/predictor.h5'
    },
    'xgboost_cnn': {
        'feature_extractor': 'models/xgboost_cnn/feature_extractor.pkl',
        'predictor': 'models/xgboost_cnn/predictor.h5'
    },
    'lightgbm_cnn': {
        'feature_extractor': 'models/lightgbm_cnn/feature_extractor.pkl',
        'predictor': 'models/lightgbm_cnn/predictor.h5'
    }
}

# Save `model_paths` for easy loading later
joblib.dump(model_paths, 'models/model_paths.pkl')