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
            
            # Load label encoders
            encoders_dir = os.path.join(data_processing_dir, 'encoders')
            self.label_encoders = {}
            for column in self.categorical_columns:
                self.label_encoders[column] = joblib.load(
                    os.path.join(encoders_dir, f'le_{column}.pkl')
                )
            
            # Load KNN imputer and scaler
            self.knn_imputer = joblib.load(os.path.join(data_processing_dir, 'knn_imputer.pkl'))
            self.scaler = joblib.load(os.path.join(data_processing_dir, 'scaler.pkl'))
            
            # Load models
            self.models = {}
            for model_type in ['xgboost_lstm', 'lightgbm_lstm', 'xgboost_cnn', 'lightgbm_cnn']:
                model_dir = os.path.join(self.models_dir, model_type)
                self.models[model_type] = {
                    'feature_extractor': joblib.load(os.path.join(model_dir, 'feature_extractor.pkl')),
                    'predictor': keras.models.load_model(os.path.join(model_dir, 'predictor.keras'))
                }
                
            # Load score scaler
            self.score_scaler = joblib.load(os.path.join(data_processing_dir, 'score_scaler.pkl'))
                
        except Exception as e:
            raise Exception(f"Error loading models and components: {str(e)}")
    
    def get_categories(self):
        """Return the known categories for categorical features."""
        try:
            categories = {}
            for column in self.categorical_columns:
                categories[column] = self.label_encoders[column].classes_.tolist()
            return categories
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
            
            # Add dummy score column for scaler
            df['score'] = 0
            
            # Encode categorical variables
            for column in self.categorical_columns:
                df[column] = self.label_encoders[column].transform(df[column].astype(str))
            
            # Handle missing values with KNN imputation
            impute_columns = ["memSize", "memBusWidth", "memClock"]
            impute_data = df[impute_columns].copy()
            df[impute_columns] = self.knn_imputer.transform(impute_data)
            
            # Scale numerical features
            scale_columns = self.numerical_columns + ['score']
            scale_data = df[scale_columns].copy()
            df[scale_columns] = self.scaler.transform(scale_data)
            
            # Ensure correct column order and drop score
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
                tree_predictions = model_dict['feature_extractor'].predict(prepared_input)
                deep_input = tree_predictions.reshape(-1, 1, 1)
                final_prediction = model_dict['predictor'].predict(deep_input).flatten()[0]
                
                # Inverse transform the prediction using score_scaler
                scaled_prediction = self.score_scaler.inverse_transform([[final_prediction]])[0][0]
                predictions[model_type] = scaled_prediction
            
            return predictions, original_values
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")