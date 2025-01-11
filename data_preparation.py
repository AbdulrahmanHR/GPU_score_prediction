# data_preparation.py
import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # This import is required
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, file_path):
        """
        Initialize the data preparation pipeline.
        
        Args:
            file_path: Path to the CSV file containing GPU specifications data
        """
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
        """
        Remove or cap outliers in specified columns using selected method.
        """
        for col in columns:
            if method == "percentile":
                # Use 1st and 99th percentiles as bounds
                lower_bound = self.data[col].quantile(0.01)
                upper_bound = self.data[col].quantile(0.99)
            elif method == "std":
                # Use standard deviation based bounds
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
            else:
                raise ValueError("Invalid method. Choose 'percentile' or 'std'.")

            # Clip values to bounds
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

    def preprocess_data(self):
        """
        Execute the complete data preprocessing pipeline.
        
        Steps:
        1. Load and clean data
        2. Handle outliers
        3. Encode categorical features
        4. Impute missing values using regression
        5. Scale numerical features
        6. Verify scaling accuracy
        """
        # Load data from CSV
        self.data = pd.read_csv(self.file_path)

        # Remove product name if present (not used for prediction)
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)

        # Store original scores for verification
        original_score = self.data['score'].copy()

        # Handle outliers separately for features and target
        self.handle_outliers(self.numerical_columns)
        self.handle_outliers(['score'])

        # Encode categorical features using label encoding
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Initialize regression imputer with RandomForestRegressor
        self.reg_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=200, random_state=32),
            random_state=32,
            max_iter=10,
            initial_strategy='mean'
        )
        
        # Impute missing values using regression
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        impute_data = self.data[impute_columns].copy()
        self.data[impute_columns] = self.reg_imputer.fit_transform(impute_data)

        # Initialize scalers for features and target
        self.scaler = StandardScaler()
        self.score_scaler = StandardScaler()

        # Scale numerical features
        scale_data = self.data[self.numerical_columns].copy()
        self.data[self.numerical_columns] = self.scaler.fit_transform(scale_data)

        # Scale target variable separately
        score_data = self.data[['score']].copy()
        self.data['score'] = self.score_scaler.fit_transform(score_data)

        # Verify scaling accuracy
        score_inverse = self.score_scaler.inverse_transform(
            self.data[['score']]
        ).flatten()
        
        # Check relative difference between original and reconstructed values
        relative_diff = np.abs(score_inverse - original_score) / np.maximum(np.abs(original_score), 1e-10)
        max_relative_diff = np.max(relative_diff)
        
        # Warn if scaling error is too large
        if max_relative_diff > 0.3:  # 30% threshold
            warnings.warn(f"Large relative difference in score scaling detected: {max_relative_diff}")

        # Return data with consistent column ordering
        return self.data[self.feature_order + ['score']]

    def save_feature_order(self, path):
        """
        Save all preprocessing components and verification data for inference.
        
        Args:
            path: Path to save the feature order JSON file
            
        Saves:
        - Feature order and column types
        - Score scaler for target variable transformation
        - Verification data for scaling checks
        """
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