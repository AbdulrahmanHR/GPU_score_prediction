# main.py
import json
import os
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from data_preparation import DataPreparation
from models import HybridModels
from model_performance_chart import plot_model_performance
from model_visualization import visualize_models 
import time

def create_directory_structure():
    """    
    Creates a hierarchical directory structure for organizing model artifacts:
    - base_dir: Root directory for all model-related files
    - data_processing_dir: Directory for data preprocessing components
    - encoders_dir: Directory for label encoders
    """
    base_dir = 'models'
    data_processing_dir = os.path.join(base_dir, 'data_processing')
    
    for directory in [base_dir, data_processing_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return base_dir, data_processing_dir

def save_data_processing_components(data_prep, data_processing_dir):
    """
    Save all data preprocessing components for later use in inference.
    
    Args:
        data_prep: DataPreparation instance containing preprocessing components
        data_processing_dir: Directory path for saving preprocessing components
    """
    # Save feature order for consistent preprocessing
    data_prep.save_feature_order(os.path.join(data_processing_dir, 'feature_order.json'))

def calculate_metrics(predictions, true_values):
    """
    Calculate comprehensive model performance metrics.
    
    Args:
        predictions: Model predictions
        true_values: Actual target values
    
    Returns:
        dict: Dictionary containing various performance metrics:
            - mae: Mean Absolute Error (scaled by 1000)
            - rmse: Root Mean Square Error (scaled by 1000)
            - r2: R-squared score (as percentage)
            - smape:  
    """
    mae = mean_absolute_error(true_values, predictions) * 1000
    rmse = np.sqrt(mean_squared_error(true_values, predictions)) * 1000
    r2 = r2_score(true_values, predictions) * 100
    smape = np.mean(2 * np.abs(predictions - true_values) / (np.abs(predictions) + np.abs(true_values))) * 100   
     
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'smape': smape
    }

def main():
    """
    Main execution function that orchestrates the entire model training pipeline:
    1. Sets up mixed precision training
    2. Creates directory structure
    3. Prepares and preprocesses data
    4. Trains multiple hybrid models
    5. Evaluates and saves model results
    """
    start_time = time.time()

    # Enable mixed precision training for better performance on compatible GPUs
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Create directory structure for saving models and artifacts
    base_dir, data_processing_dir = create_directory_structure()
    
    # Prepare and preprocess the data
    print("\nPreparing data...")
    data_prep = DataPreparation('gpu_specs_v6_score.csv')
    
    train_data, test_data = data_prep.preprocess_train_test_split()
    
    # Save preprocessing components for inference
    print("\nSaving data processing components...")
    save_data_processing_components(data_prep, data_processing_dir)
    
    # Save unique categories for categorical variables
    known_categories = {}
    for col in data_prep.categorical_columns:
        known_categories[col] = data_prep.ordinal_encoder.categories_[data_prep.categorical_columns.index(col)].tolist()
    
    with open(os.path.join(data_processing_dir, 'known_categories.json'), 'w') as f:
        json.dump(known_categories, f)
    
    # Initialize hybrid models with both train and test data
    print("\nTraining models...")
    hybrid_models = HybridModels(train_data, test_data)  # Updated to pass both datasets
    
    # Initialize dictionaries to store results and file paths
    model_results = {}
    metrics = {}
    model_paths = {
        'data_processing': {
            'known_categories': os.path.join(data_processing_dir, 'known_categories.json')
        }
    }
    
    # Update model paths for the new encoder structure
    model_paths['data_processing']['ordinal_encoder'] = os.path.join(data_processing_dir, 'ordinal_encoder.pkl')
    model_paths['data_processing']['feature_scaler'] = os.path.join(data_processing_dir, 'feature_scaler.pkl')
    model_paths['data_processing']['score_scaler'] = os.path.join(data_processing_dir, 'score_scaler.pkl')
    
    # Define all model combinations to train
    model_types = {
        'xgboost_lstm': hybrid_models.xgboost_lstm,
        'lightgbm_lstm': hybrid_models.lightgbm_lstm,
        'xgboost_cnn': hybrid_models.xgboost_cnn,
        'lightgbm_cnn': hybrid_models.lightgbm_cnn
    }
    
    # Train and evaluate each model type
    for model_name, model_func in model_types.items():
        print(f"\nTraining {model_name}...")
        results = model_func()
        model_results[model_name] = results
        
        # Calculate performance metrics
        metrics[model_name] = calculate_metrics(
            results['predictions'],
            results['true_values']
        )
        
        # Create model-specific directory and save models
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"Saving {model_name} models...")
        feature_extractor_path = os.path.join(model_dir, 'feature_extractor.pkl')
        predictor_path = os.path.join(model_dir, 'predictor.keras')
        
        # Save feature extractor and predictor models
        joblib.dump(
            results['feature_extractor'],
            feature_extractor_path
        )
        results['predictor'].save(predictor_path)
        
        # Store model paths for inference
        model_paths[model_name] = {
            'feature_extractor': feature_extractor_path,
            'predictor': predictor_path
        }
    
    # Save model paths for future reference
    joblib.dump(model_paths, os.path.join(base_dir, 'model_paths.pkl'))
    
    # Generate and save performance visualization
    plot_model_performance(model_results)
    visualize_models(model_results, hybrid_models)
    
    # Print performance metrics for all models
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"MAE: {model_metrics['mae']:.2f} # Lower is better")
        print(f"RMSE: {model_metrics['rmse']:.2f} # Lower is better")
        print(f"R²: {model_metrics['r2']:.2f}% # Higher is better")
        print(f"SMAPE: {model_metrics['smape']:.2f}% # Lower is better")
    
    # Save metrics to JSON file
    with open(os.path.join(base_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nTraining and evaluation completed successfully!")
    print(f"All models and analysis files have been saved to: {base_dir}")
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.4f} seconds")
    training_time_min= training_time/60
    print(f"Training completed in {training_time_min:.4f} minutes")

if __name__ == "__main__":
    main()