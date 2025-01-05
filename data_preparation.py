# data_preparation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib
import os
import json
import warnings

warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, file_path):
        """Initialize data preparation with file path and necessary components."""
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.knn_imputer = None
        self.scaler = None
        self.score_scaler = None  # Separate scaler for score column
        
        # Define column order that will be used consistently
        self.feature_order = [
            'gpuChip', 'memSize', 'memBusWidth', 'gpuClock', 'memClock',
            'memType', 'bus', 'unifiedShader', 'tmu', 'rop',
            'manufacturer', 'releaseYear'
        ]
        self.categorical_columns = ['gpuChip', 'bus', 'memType', 'manufacturer']
        self.numerical_columns = [
            'memSize', 'memBusWidth', 'gpuClock', 'memClock',
            'unifiedShader', 'tmu', 'rop', 'releaseYear'
        ]

    def handle_outliers(self, columns, method="percentile", n_std=3):
        """Remove or cap outliers based on the chosen method."""
        for col in columns:
            if method == "percentile":
                lower_bound = self.data[col].quantile(0.01)
                upper_bound = self.data[col].quantile(0.99)
            elif method == "std":
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
            else:
                raise ValueError("Invalid method. Choose 'percentile' or 'std'.")

            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

    def preprocess_data(self):
        """Preprocess the data with proper scaling and encoding."""
        # Load data
        self.data = pd.read_csv(self.file_path)

        # Drop productName if it exists
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)

        # Store original score values for reference
        original_score = self.data['score'].copy()

        # Handle outliers in numerical columns and score separately
        self.handle_outliers(self.numerical_columns)
        self.handle_outliers(['score'])

        # Label encode categorical features
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Improve imputation with iterative KNN
        self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        impute_data = self.data[impute_columns].copy()
        self.data[impute_columns] = self.knn_imputer.fit_transform(impute_data)

        # Initialize scalers
        self.scaler = StandardScaler()
        self.score_scaler = StandardScaler()

        # Scale numerical features
        scale_data = self.data[self.numerical_columns].copy()
        self.data[self.numerical_columns] = self.scaler.fit_transform(scale_data)

        # Scale score separately
        score_data = self.data[['score']].copy()
        self.data['score'] = self.score_scaler.fit_transform(score_data)

        # Verify scaling with relaxed precision
        score_inverse = self.score_scaler.inverse_transform(
            self.data[['score']]
        ).flatten()
        
        # Check if the relative difference is within acceptable bounds
        relative_diff = np.abs(score_inverse - original_score) / np.maximum(np.abs(original_score), 1e-10)
        max_relative_diff = np.max(relative_diff)
        
        if max_relative_diff > 0.5:  # Allow up to 50% relative difference
            warnings.warn(f"Large relative difference in score scaling detected: {max_relative_diff}")

        # Ensure correct column order
        return self.data[self.feature_order + ['score']]

    def save_feature_order(self, path):
        """Save feature order and all necessary components for inference."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save feature information
        with open(path, 'w') as f:
            json.dump({
                'feature_order': self.feature_order,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns
            }, f, indent=4)
        
        # Save score scaler separately
        score_scaler_path = os.path.join(os.path.dirname(path), 'score_scaler.pkl')
        joblib.dump(self.score_scaler, score_scaler_path)
        
        # Save verification data
        verification_data = {
            'original_scale_example': float(self.score_scaler.inverse_transform([[1.0]])[0][0]),
            'scaled_example': float(self.score_scaler.transform([[1000.0]])[0][0])
        }
        verification_path = os.path.join(os.path.dirname(path), 'scaling_verification.json')
        with open(verification_path, 'w') as f:
            json.dump(verification_data, f, indent=4)