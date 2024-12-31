# inference
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

class InferencePipeline:
    def __init__(self):
        self.models = {}
        self.transformers = {}
    
    def load_models(self, model_paths):
        """
        Load all necessary models and preprocessors
        
        Args:
            model_paths (dict): Dictionary containing paths to saved models and transformers
            Example:
            {
                'xgboost_lstm': {
                    'feature_extractor': 'path/to/xgboost.pkl',
                    'predictor': 'path/to/lstm.h5'
                },
                'lightgbm_lstm': {
                    'feature_extractor': 'path/to/lightgbm.pkl',
                    'predictor': 'path/to/lstm.h5'
                },
                'xgboost_cnn': {
                    'feature_extractor': 'path/to/xgboost.pkl',
                    'predictor': 'path/to/cnn.h5'
                },
                'lightgbm_cnn': {
                    'feature_extractor': 'path/to/lightgbm.pkl',
                    'predictor': 'path/to/cnn.h5'
                },
                'transformers': {
                    'label_encoders': {
                        'gpuChip': 'path/to/le_gpuChip.pkl',
                        'bus': 'path/to/le_bus.pkl',
                        'memType': 'path/to/le_memType.pkl'
                    },
                    'knn_imputer': 'path/to/knn_imputer.pkl',
                    'scaler': 'path/to/scaler.pkl'
                }
            }
        """
        # Load transformers
        self.transformers = {
            'label_encoders': {
                'gpuChip': joblib.load(model_paths['transformers']['label_encoders']['gpuChip']),
                'bus': joblib.load(model_paths['transformers']['label_encoders']['bus']),
                'memType': joblib.load(model_paths['transformers']['label_encoders']['memType'])
            },
            'knn_imputer': joblib.load(model_paths['transformers']['knn_imputer']),
            'scaler': joblib.load(model_paths['transformers']['scaler'])
        }
        
        # Load models
        for model_name, paths in model_paths.items():
            if model_name not in ['transformers']:
                self.models[model_name] = {
                    'feature_extractor': joblib.load(paths['feature_extractor']),
                    'predictor': load_model(paths['predictor'])
                }
    
    def preprocess_input(self, input_data):
        """
        Preprocess the input data using the saved transformers
        
        Args:
            input_data (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Make a copy of input data
        data = input_data.copy()
        
        # Drop productName if exists
        if 'productName' in data.columns:
            data.drop(columns=['productName'], inplace=True)
            
        # One-hot encode manufacturer
        if 'manufacturer' in data.columns:
            data = pd.get_dummies(data, columns=['manufacturer'], drop_first=True)
        
        # Label encode categorical columns
        data['gpuChip'] = self.transformers['label_encoders']['gpuChip'].transform(data['gpuChip'])
        data['bus'] = self.transformers['label_encoders']['bus'].transform(data['bus'])
        data['memType'] = self.transformers['label_encoders']['memType'].transform(data['memType'])
        
        # Apply KNN Imputation
        impute_columns = ['memSize', 'memBusWidth', 'memClock']
        data[impute_columns] = self.transformers['knn_imputer'].transform(data[impute_columns])
        
        # Standardize numerical features
        numerical_features = ['releaseYear', 'memSize', 'memBusWidth', 'gpuClock', 
                            'memClock', 'unifiedShader', 'tmu', 'rop']
        data[numerical_features] = self.transformers['scaler'].transform(data[numerical_features])
        
        return data
    
    def predict(self, input_data, model_name):
        """
        Make predictions using the specified model
        
        Args:
            input_data (pd.DataFrame): Input data to make predictions on
            model_name (str): Name of the model to use ('xgboost_lstm', 'lightgbm_lstm', etc.)
            
        Returns:
            np.array: Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        # Preprocess input data
        preprocessed_data = self.preprocess_input(input_data)
        
        # Extract features using the first stage model (XGBoost or LightGBM)
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
        """
        Make predictions using all available models
        
        Args:
            input_data (pd.DataFrame): Input data to make predictions on
            
        Returns:
            dict: Dictionary containing predictions from all models
        """
        predictions = {}
        for model_name in self.models.keys():
            predictions[model_name] = self.predict(input_data, model_name)
        return predictions