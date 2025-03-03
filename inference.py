# inference.py
import joblib
import numpy as np
import pandas as pd
import json
import os
import tensorflow as tf
from tensorflow import keras

class GPUPredictor:
    def __init__(self, models_dir='models'):
        """Initialize the GPU predictor with necessary models and preprocessing components."""
        self.models_dir = models_dir
        self._load_components()
        
    def _load_components(self):
        """Load all necessary components for prediction."""
        try:
            # Load model paths
            self.model_paths = joblib.load(os.path.join(self.models_dir, 'model_paths.pkl'))
            
            # Load feature order and column information
            data_processing_dir = os.path.join(self.models_dir, 'data_processing')
            with open(os.path.join(data_processing_dir, 'feature_order.json'), 'r') as f:
                feature_info = json.load(f)
                self.feature_order = feature_info['feature_order']
                self.categorical_columns = feature_info['categorical_columns']
                self.numerical_columns = feature_info['numerical_columns']
            
            # Load known categories for categorical features
            with open(os.path.join(data_processing_dir, 'known_categories.json'), 'r') as f:
                self.known_categories = json.load(f)
            
            # Load ordinal encoder and scalers
            self.ordinal_encoder = joblib.load(os.path.join(data_processing_dir, 'ordinal_encoder.pkl'))
            self.feature_scaler = joblib.load(os.path.join(data_processing_dir, 'feature_scaler.pkl'))
            self.score_scaler = joblib.load(os.path.join(data_processing_dir, 'score_scaler.pkl'))
            
            # Load models
            self.models = {}
            for model_type in ['xgboost_lstm', 'lightgbm_lstm', 'xgboost_cnn', 'lightgbm_cnn']:
                if model_type in self.model_paths:
                    model_dir = os.path.join(self.models_dir, model_type)
                    self.models[model_type] = {
                        'feature_extractor': joblib.load(self.model_paths[model_type]['feature_extractor']),
                        'predictor': keras.models.load_model(self.model_paths[model_type]['predictor'])
                    }
                
        except Exception as e:
            raise Exception(f"Error loading models and components: {str(e)}")
    
    def get_categories(self):
        """Return the known categories for categorical features."""
        try:
            return self.known_categories
        except Exception as e:
            raise Exception(f"Error getting categories: {str(e)}")
    
    def _prepare_input(self, input_data):
        """Prepare input data for prediction."""
        try:
            # Create DataFrame with correct feature order
            df = pd.DataFrame([input_data])
            
            # Store original values
            original_values = {col: df[col].iloc[0] if col in df else None 
                             for col in self.feature_order}
            
            # Encode categorical variables
            categorical_data = df[self.categorical_columns].copy()
            encoded_cats = self.ordinal_encoder.transform(categorical_data)
            df[self.categorical_columns] = encoded_cats
            
            # Scale numerical features
            df[self.numerical_columns] = self.feature_scaler.transform(df[self.numerical_columns])
            
            # Ensure correct column order
            df = df[self.feature_order]
            
            return df.values, original_values
            
        except Exception as e:
            raise Exception(f"Error preparing input data: {str(e)}")
    
    def predict(self, input_data):
        """Make predictions using all models."""
        try:
            prepared_input, original_values = self._prepare_input(input_data)
            
            predictions = {}
            for model_type, model_dict in self.models.items():
                # Get tree model predictions
                tree_predictions = model_dict['feature_extractor'].predict(prepared_input)
                
                # Reshape for deep model depending on model type
                if 'lstm' in model_type:
                    deep_input = tree_predictions.reshape(-1, 1, 1)  # LSTM expects (samples, timesteps, features)
                else:  # CNN
                    deep_input = tree_predictions.reshape(-1, 1, 1, 1)  # CNN expects (samples, height, width, channels)
                
                # Get final prediction
                final_prediction = model_dict['predictor'].predict(deep_input).flatten()[0]
                
                # Inverse transform the prediction using score_scaler
                scaled_prediction = self.score_scaler.inverse_transform([[final_prediction]])[0][0]
                predictions[model_type] = scaled_prediction
            
            return predictions, original_values
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")