# main.py
import json
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
    'xgboost_lstm': hybrid_models.xgboost_lstm,
    'xgboost_cnn': hybrid_models.xgboost_cnn,
    'lightgbm_lstm': hybrid_models.lightgbm_lstm,
    'lightgbm_cnn': hybrid_models.lightgbm_cnn
}

for model_name, model_func in models.items():
    print(f"\nTraining {model_name}...")
    # Get model results
    results = model_func()
    model_results[model_name] = results

# Plot model performance
metrics = plot_model_performance(model_results)

# Print detailed metrics
for model_name, model_metrics in metrics.items():
    print(f"\n{model_name} Performance:")
    print(f"MAE: {model_metrics['mae']:.4f}")
    print(f"RMSE: {model_metrics['rmse']:.4f}")
    print(f"R²: {model_metrics['r2']:.4f}")

# Create base directory structure
base_dir = 'models'
data_processing_dir = os.path.join(base_dir, 'data_processing')
label_encoders_dir = os.path.join(data_processing_dir, 'label_encoders')

# Create all necessary directories
for directory in [base_dir, data_processing_dir, label_encoders_dir]:
    os.makedirs(directory, exist_ok=True)

# Define paths for data processing components
data_processing_paths = {
    'label_encoders': {
        'gpuChip': os.path.join(label_encoders_dir, 'le_gpuChip.pkl'),
        'bus': os.path.join(label_encoders_dir, 'le_bus.pkl'),
        'memType': os.path.join(label_encoders_dir, 'le_memType.pkl')
    },
    'knn_imputer': os.path.join(data_processing_dir, 'knn_imputer.pkl'),
    'scaler': os.path.join(data_processing_dir, 'scaler.pkl'),
    'known_categories': os.path.join(data_processing_dir, 'known_categories.json')
}

# Save label encoders and their known categories
known_categories = {}
for column, le in data_prep.label_encoders.items():
    # Save label encoder
    joblib.dump(le, data_processing_paths['label_encoders'][column])
    # Store known categories
    known_categories[column] = list(le.classes_)

# Save known categories for future reference
with open(data_processing_paths['known_categories'], 'w') as f:
    json.dump(known_categories, f)

# Save KNN imputer and scaler
joblib.dump(data_prep.knn_imputer, data_processing_paths['knn_imputer'])
joblib.dump(data_prep.scaler, data_processing_paths['scaler'])

# Define model structure and save models
model_structure = {}
for model_name in model_results.keys():
    # Create model directory
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define paths for this model
    model_paths = {
        'feature_extractor': os.path.join(model_dir, 'feature_extractor.pkl'),
        'predictor': os.path.join(model_dir, 'predictor.h5')
    }
    
    # Save the models
    print(f"\nSaving {model_name} models...")
    joblib.dump(
        model_results[model_name]['feature_extractor'],
        model_paths['feature_extractor']
    )
    model_results[model_name]['predictor'].save(
        model_paths['predictor']
    )
    
    # Store paths in model_structure
    model_structure[model_name] = model_paths

# Create final model paths dictionary
model_paths = {
    'data_processing': data_processing_paths,
    **model_structure
}

# Save model paths for easy loading
paths_file = os.path.join(base_dir, 'model_paths.pkl')
print(f"\nSaving model paths to {paths_file}")
joblib.dump(model_paths, paths_file)

print("\nAll models and data processing components have been saved successfully!")