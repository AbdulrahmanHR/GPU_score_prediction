# inference.py
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import os
import json

class GPUPredictor:
    def __init__(self, base_dir='models'):
        """Initialize the predictor with model directory path"""
        self.base_dir = base_dir
        self._load_components()
    
    def _load_components(self):
        """Load all necessary components for prediction"""
        try:
            # Load encoders
            self.label_encoders = {}
            encoders_dir = os.path.join(self.base_dir, 'data_processing', 'encoders')
            categorical_columns = ['gpuChip', 'bus', 'memType', 'manufacturer']
            for col in categorical_columns:
                encoder_path = os.path.join(encoders_dir, f'le_{col}.pkl')
                self.label_encoders[col] = joblib.load(encoder_path)
                
                # Store the classes for each encoder
                classes = self.label_encoders[col].classes_
                setattr(self, f'{col}_classes', classes)
            
            # Load imputer and scaler
            data_processing_dir = os.path.join(self.base_dir, 'data_processing')
            self.knn_imputer = joblib.load(os.path.join(data_processing_dir, 'knn_imputer.pkl'))
            self.scaler = joblib.load(os.path.join(data_processing_dir, 'scaler.pkl'))
            
            # Load models
            self.models = {}
            model_types = ['xgboost_lstm', 'lightgbm_lstm', 'xgboost_cnn', 'lightgbm_cnn']
            
            for model_type in model_types:
                model_dir = os.path.join(self.base_dir, model_type)
                if os.path.exists(model_dir):
                    self.models[model_type] = {
                        'feature_extractor': joblib.load(os.path.join(model_dir, 'feature_extractor.pkl')),
                        'predictor': keras.models.load_model(os.path.join(model_dir, 'predictor.keras'))
                    }
        
        except Exception as e:
            raise Exception(f"Error loading model components: {str(e)}")
    
    def get_categories(self):
        """Return the available categories for each categorical field"""
        return {
            'manufacturer': self.manufacturer_classes.tolist(),
            'gpuChip': self.gpuChip_classes.tolist(),
            'memType': self.memType_classes.tolist(),
            'bus': self.bus_classes.tolist()
        }
    
    def prepare_input(self, input_data):
        """Prepare input data for prediction"""
        try:
            # Create DataFrame with the correct column order
            columns = [
                'manufacturer', 'gpuChip', 'memType', 'bus', 'memSize', 
                'memBusWidth', 'gpuClock', 'memClock', 'unifiedShader', 
                'tmu', 'rop', 'releaseYear'
            ]
            
            df = pd.DataFrame([input_data], columns=columns)
            
            # Store original categorical values
            categorical_columns = ['manufacturer', 'gpuChip', 'memType', 'bus']
            original_values = {col: df[col].values[0] for col in categorical_columns}
            
            # Encode categorical variables
            for column, encoder in self.label_encoders.items():
                df[column] = encoder.transform(df[column].astype(str))
            
            # Apply imputation
            impute_columns = ["memSize", "memBusWidth", "memClock"]
            df[impute_columns] = self.knn_imputer.transform(df[impute_columns])
            
            # Scale numerical features
            numerical_columns = [
                "memSize", "memBusWidth", "gpuClock", "memClock",
                "unifiedShader", "tmu", "rop", "releaseYear"
            ]
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])
            
            return df.values, original_values
            
        except Exception as e:
            raise Exception(f"Error preparing input data: {str(e)}")
    
    def predict(self, input_data):
        """Make predictions using all available models"""
        try:
            # Prepare input
            prepared_input, original_values = self.prepare_input(input_data)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                # Get features from tree model
                features = model['feature_extractor'].predict(prepared_input).reshape(-1, 1)
                
                # Get prediction from deep model
                pred = model['predictor'].predict(features, verbose=0).flatten()[0]
                predictions[model_name] = pred
            
            return predictions, original_values
            
        except Exception as e:
            raise Exception(f"Error making predictions: {str(e)}")