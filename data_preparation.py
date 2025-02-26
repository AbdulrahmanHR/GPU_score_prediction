# data_preparation.py
import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.reg_imputer = None
        self.scaler = None
        self.score_scaler = None
        
        # Define consistent column ordering for processing
        self.feature_order = [
            'gpuChip', 'memSize', 'memBusWidth', 'gpuClock', 'memClock',
            'memType', 'bus', 'unifiedShader', 'tmu', 'rop',
            'manufacturer', 'releaseYear'
        ]
        
        # Separate features by type for appropriate processing
        self.categorical_columns = ['gpuChip', 'bus', 'memType', 'manufacturer']
        self.numerical_columns = [
            'memSize', 'memBusWidth', 'gpuClock', 'memClock',
            'unifiedShader', 'tmu', 'rop', 'releaseYear'
        ]

    def handle_outliers(self, columns, method="percentile", n_std=3):
        for col in columns:
            if method == "percentile":
                lower_bound = self.data[col].quantile(0.01) * 0.95
                upper_bound = self.data[col].quantile(0.99)
            elif method == "std":
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
            else:
                raise ValueError("Invalid method. Choose 'percentile' or 'std'.")
            
        if col == 'score':
            lower_bound = max(lower_bound, 10)
            
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

    def preprocess_data(self):
        # Load data from CSV
        self.data = pd.read_csv(self.file_path)

        # Remove product name if present (not used for prediction)
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)

        # Handle outliers first
        self.handle_outliers(self.numerical_columns)
        self.handle_outliers(['score'])

        # Encode categorical features
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Remove missing values and capture count
        initial_rows = self.data.shape[0]
        self.data = self.data.dropna()
        removed_rows = initial_rows - self.data.shape[0]
        print(f"Removed {removed_rows} rows with missing values.")

        # Capture original scores AFTER cleaning
        original_score = self.data['score'].copy()

        # Initialize scalers
        self.scaler = MinMaxScaler()
        self.score_scaler = MinMaxScaler()

        # Scale features and target
        self.data[self.numerical_columns] = self.scaler.fit_transform(self.data[self.numerical_columns])
        self.data['score'] = self.score_scaler.fit_transform(self.data[['score']])

        # Verify scaling with aligned data
        score_inverse = self.score_scaler.inverse_transform(self.data[['score']]).flatten()
        relative_diff = np.abs(score_inverse - original_score) / np.maximum(np.abs(original_score), 1e-10)
        
        if np.max(relative_diff) > 0.3:
            warnings.warn(f"Large relative difference in score scaling detected: {np.max(relative_diff)}")

        # Return data with consistent column ordering
        return self.data[self.feature_order + ['score']]

    def save_feature_order(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save feature configuration
        with open(path, 'w') as f:
            json.dump({
                'feature_order': self.feature_order,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns
            }, f, indent=4)
        
        # Save score scaler for inference
        score_scaler_path = os.path.join(os.path.dirname(path), 'score_scaler.pkl')
        joblib.dump(self.score_scaler, score_scaler_path)
        
        # Save verification data for scaling checks
        verification_data = {
            'original_scale_example': float(self.score_scaler.inverse_transform([[1.0]])[0][0]),
            'scaled_example': float(self.score_scaler.transform([[1000.0]])[0][0])
        }
        verification_path = os.path.join(os.path.dirname(path), 'scaling_verification.json')
        with open(verification_path, 'w') as f:
            json.dump(verification_data, f, indent=4)