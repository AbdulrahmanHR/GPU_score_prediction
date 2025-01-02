# inference
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import json
from tensorflow.keras.losses import MeanSquaredError  # type: ignore


class InferencePipeline:
    def __init__(self):
        self.models = {}
        self.transformers = {}
        self.known_categories = {}
    
    def load_models(self, model_paths):
        """Load all necessary models and preprocessors"""
        # Load known categories
        with open(model_paths['data_processing']['known_categories'], 'r') as f:
            self.known_categories = json.load(f)
        
        # Load transformers
        self.transformers = {
            'label_encoders': {
                'gpuChip': joblib.load(model_paths['data_processing']['label_encoders']['gpuChip']),
                'bus': joblib.load(model_paths['data_processing']['label_encoders']['bus']),
                'manufacturer': joblib.load(model_paths['data_processing']['label_encoders']['manufacturer']),
                'memType': joblib.load(model_paths['data_processing']['label_encoders']['memType'])
            },
            'knn_imputer': joblib.load(model_paths['data_processing']['knn_imputer']),
            'scaler': joblib.load(model_paths['data_processing']['scaler'])
        }
        
        # Load models
        for model_name, paths in model_paths.items():
            if model_name not in ['data_processing']:
                self.models[model_name] = {
                    'feature_extractor': joblib.load(paths['feature_extractor']),
                    'predictor': load_model(paths['predictor'], 
                                         custom_objects={'mse': MeanSquaredError()})
                }

    def handle_unknown_categories(self, data, column):
        """Handle unknown categories in categorical columns"""
        known_cats = self.known_categories[column]
        unknown_cats = set(data[column].unique()) - set(known_cats)
        
        if unknown_cats:
            print(f"Warning: Unknown categories in {column}: {unknown_cats}")
            print(f"Known categories are: {known_cats}")
            # Replace unknown categories with the most similar known category
            for unknown in unknown_cats:
                # Find most similar known category using string similarity
                similarities = [self._string_similarity(unknown, known) 
                              for known in known_cats]
                most_similar = known_cats[np.argmax(similarities)]
                data[column] = data[column].replace(unknown, most_similar)
                print(f"Replaced '{unknown}' with '{most_similar}'")
        
        return data

    def _string_similarity(self, s1, s2):
        """Calculate string similarity using Levenshtein distance"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()
    
    def preprocess_input(self, input_data):
        """Preprocess the input data using the saved transformers"""
        # Make a copy of input data
        data = input_data.copy()
        
        # Drop productName if exists
        if 'productName' in data.columns:
            data.drop(columns=['productName'], inplace=True)
        
        # Handle unknown categories and label encode
        for column in ['gpuChip', 'bus', 'memType', 'manufacturer']:
            data = self.handle_unknown_categories(data, column)
            data[column] = self.transformers['label_encoders'][column].transform(data[column])
        
        # Apply KNN Imputation
        impute_columns = ['memSize', 'memBusWidth', 'memClock']
        data[impute_columns] = self.transformers['knn_imputer'].transform(data[impute_columns])
        
        # Standardize numerical features
        numerical_features = ['releaseYear', 'memSize', 'memBusWidth', 'gpuClock', 
                            'memClock', 'unifiedShader', 'tmu', 'rop']
        data[numerical_features] = self.transformers['scaler'].transform(data[numerical_features])
        
        return data
    
    def predict(self, input_data, model_name):
        """Make predictions using the specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        # Preprocess input data
        preprocessed_data = self.preprocess_input(input_data)
        
        # Extract features using the first stage model
        feature_extractor = self.models[model_name]['feature_extractor']
        predictor = self.models[model_name]['predictor']
        
        if 'xgboost' in model_name:
            features = feature_extractor.apply(preprocessed_data)
        else:  # lightgbm
            features = feature_extractor.predict(preprocessed_data).reshape(-1, 1)
        
        # Reshape features for LSTM/CNN input
        if 'lstm' in model_name or 'cnn' in model_name:
            features = features.reshape((features.shape[0], features.shape[1], 1))
            
        # Make final prediction
        predictions = predictor.predict(features)
        
        return predictions.flatten()
    
    def predict_all(self, input_data):
        """Make predictions using all available models"""
        predictions = {}
        for model_name in self.models.keys():
            predictions[model_name] = self.predict(input_data, model_name)
        return predictions