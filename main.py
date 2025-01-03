# main.py
import json
import numpy as np
from data_preparation import DataPreparation
from models import HybridModels
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from model_performance_chart import plot_model_performance
import os
import tensorflow as tf

def create_directory_structure():
    """Create necessary directories for model saving"""
    base_dir = 'models'
    data_processing_dir = os.path.join(base_dir, 'data_processing')
    encoders_dir = os.path.join(data_processing_dir, 'encoders')
    
    for directory in [base_dir, data_processing_dir, encoders_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return base_dir, data_processing_dir, encoders_dir

def save_data_processing_components(data_prep, encoders_dir, data_processing_dir):
    """Save all data preprocessing components"""
    # Save label encoders
    for column, encoder in data_prep.label_encoders.items():
        joblib.dump(encoder, os.path.join(encoders_dir, f'le_{column}.pkl'))
    
    # Save KNN imputer and scaler
    joblib.dump(data_prep.knn_imputer, os.path.join(data_processing_dir, 'knn_imputer.pkl'))
    joblib.dump(data_prep.scaler, os.path.join(data_processing_dir, 'scaler.pkl'))

def calculate_metrics(predictions, true_values):
    """Calculate comprehensive model metrics"""
    mae = mean_absolute_error(true_values, predictions) * 1000
    rmse = np.sqrt(mean_squared_error(true_values, predictions)) * 1000
    r2 = r2_score(true_values, predictions) * 100
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

def main():
    # Enable mixed precision training for better performance
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create directory structure
    base_dir, data_processing_dir, encoders_dir = create_directory_structure()
    
    # Prepare data
    print("\nPreparing data...")
    data_prep = DataPreparation('gpu_specs_v6_score.csv')
    data = data_prep.preprocess_data()
    
    # Save data processing components
    print("\nSaving data processing components...")
    save_data_processing_components(data_prep, encoders_dir, data_processing_dir)
    
    # Save known categories for inference
    known_categories = {
        'gpuChip': data['gpuChip'].unique().tolist(),
        'bus': data['bus'].unique().tolist(),
        'memType': data['memType'].unique().tolist(),
        'manufacturer': data['manufacturer'].unique().tolist()
    }
    with open(os.path.join(data_processing_dir, 'known_categories.json'), 'w') as f:
        json.dump(known_categories, f)
    
    # Initialize and train models
    print("\nTraining models...")
    hybrid_models = HybridModels(data)
    
    # Dictionary to store model results and paths
    model_results = {}
    metrics = {}
    model_paths = {
        'data_processing': {
            'known_categories': os.path.join(data_processing_dir, 'known_categories.json'),
            'label_encoders': {
                'gpuChip': os.path.join(encoders_dir, 'le_gpuChip.pkl'),
                'bus': os.path.join(encoders_dir, 'le_bus.pkl'),
                'memType': os.path.join(encoders_dir, 'le_memType.pkl'),
                'manufacturer': os.path.join(encoders_dir, 'le_manufacturer.pkl')
            },
            'knn_imputer': os.path.join(data_processing_dir, 'knn_imputer.pkl'),
            'scaler': os.path.join(data_processing_dir, 'scaler.pkl')
        }
    }
    
    # Train all model combinations
    model_types = {
        'xgboost_lstm': hybrid_models.xgboost_lstm,
        'lightgbm_lstm': hybrid_models.lightgbm_lstm,
        'xgboost_cnn': hybrid_models.xgboost_cnn,
        'lightgbm_cnn': hybrid_models.lightgbm_cnn
    }
    
    for model_name, model_func in model_types.items():
        print(f"\nTraining {model_name}...")
        results = model_func()
        model_results[model_name] = results
        
        # Calculate metrics
        metrics[model_name] = calculate_metrics(
            results['predictions'],
            results['true_values']
        )
        
        # Create model directory
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models and update paths
        print(f"Saving {model_name} models...")
        feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
        predictor_path = os.path.join(model_dir, 'predictor.keras')
        
        joblib.dump(
            results['feature_extractor'],
            feature_extractor_path
        )
        results['predictor'].save(predictor_path)
        
        # Store paths for inference
        model_paths[model_name] = {
            'feature_extractor': feature_extractor_path,
            'predictor': predictor_path
        }
    
    # Save model paths for inference
    joblib.dump(model_paths, os.path.join(base_dir, 'model_paths.pkl'))
    
    # Plot performance metrics
    plot_model_performance(model_results)
    
    # Print metrics
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"MAE: {model_metrics['mae']:.2f} # Lower is better")
        print(f"RMSE: {model_metrics['rmse']:.2f} # Lower is better")
        print(f"R²: {model_metrics['r2']:.2f}% # Higher is better")
        print(f"MAPE: {model_metrics['mape']:.2f}% # Lower is better")
    
    # Save metrics
    with open(os.path.join(base_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nTraining and evaluation completed successfully!")
    print(f"All models and analysis files have been saved to: {base_dir}")

if __name__ == "__main__":
    main()