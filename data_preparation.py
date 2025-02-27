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
        
        
    def load_data(self):
        """Load data from CSV without any processing"""
        self.data = pd.read_csv(self.file_path)
        
        # Remove product name if present (not used for prediction)
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)
            
        return self.data
    
    def handle_outliers(self, df, columns, method="percentile", n_std=3):
        """Handle outliers in specified columns - returns a NEW dataframe"""
        result_df = df.copy()
        
        for col in columns:
            if method == "percentile":
                lower_bound = result_df[col].quantile(0.01) * 0.95
                upper_bound = result_df[col].quantile(0.99)
            elif method == "std":
                mean = result_df[col].mean()
                std = result_df[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
            else:
                raise ValueError("Invalid method. Choose 'percentile' or 'std'.")
            
            if col == 'score':
                lower_bound = max(lower_bound, 10)
            
            result_df[col] = result_df[col].clip(lower_bound, upper_bound)
            
        return result_df

    def fit_transformers(self, train_df):
        """
        Fit label encoders and scalers on training data only.
        This should be called only on the training set.
        """
        # Initialize transformers
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.score_scaler = MinMaxScaler()
        
        # Fit label encoders on categorical features
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].fit(train_df[column])
        
        # Fit scalers on numerical features and target
        self.scaler.fit(train_df[self.numerical_columns])
        
        if 'score' in train_df.columns:
            self.score_scaler.fit(train_df[['score']])
    
    def transform_data(self, df, transform_target=True):
        """
        Apply transformations using pre-fitted transformers.
        Can be applied to both training and test data.
        """
        result_df = df.copy()
        
        # Apply label encoders to categorical features
        for column in self.categorical_columns:
            # Handle unseen categories by mapping them to -1
            result_df[column] = result_df[column].map(
                lambda x: self.label_encoders[column].transform([x])[0] 
                if x in self.label_encoders[column].classes_ else -1
            )
        
        # Apply scalers to numerical features
        result_df[self.numerical_columns] = self.scaler.transform(result_df[self.numerical_columns])
        
        # Apply target scaling if needed and if target exists
        if transform_target and 'score' in result_df.columns:
            result_df['score'] = self.score_scaler.transform(result_df[['score']])
        
        # Return data with consistent column ordering
        return result_df[self.feature_order + (
            ['score'] if 'score' in result_df.columns else []
        )]
        
    def preprocess_train_test_split(self, test_size=0.15, random_state=42):
        """
        Load, clean, split and preprocess data while avoiding data leakage.
        Returns train and test datasets properly processed.
        """
        from sklearn.model_selection import train_test_split
        
        # Step 1: Load raw data
        raw_data = self.load_data()
        
        # Step 2: Handle missing values and capture count
        initial_rows = raw_data.shape[0]
        cleaned_data = raw_data.dropna()
        removed_rows = initial_rows - cleaned_data.shape[0]
        print(f"Removed {removed_rows} rows with missing values.")
        
        # Step 3: Split the data BEFORE any transformations
        train_data, test_data = train_test_split(
            cleaned_data, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Step 4: Handle outliers in training data only
        train_data = self.handle_outliers(
            train_data, 
            self.numerical_columns + ['score']
        )
        
        # Step 5: Fit transformers on training data only
        self.fit_transformers(train_data)
        
        # Step 6: Apply transformations to both sets
        train_processed = self.transform_data(train_data)
        test_processed = self.transform_data(test_data)
        
        return train_processed, test_processed
    # data_preparation.py - Fixed preprocessing method
    def preprocess_data(self):
        """
        Complete preprocessing method for a single dataset
        (without splitting into train/test)
        """
        # Step 1: Load raw data
        raw_data = self.load_data()
        
        # Step 2: Handle missing values
        initial_rows = raw_data.shape[0]
        cleaned_data = raw_data.dropna()
        removed_rows = initial_rows - cleaned_data.shape[0]
        print(f"Removed {removed_rows} rows with missing values.")
        
        # Step 3: Handle outliers
        processed_data = self.handle_outliers(
            cleaned_data, 
            self.numerical_columns + (['score'] if 'score' in cleaned_data.columns else [])
        )
                
        # Step 4: Fit transformers on the entire dataset
        self.fit_transformers(processed_data)
        
        # Step 5: Apply transformations
        processed_data = self.transform_data(processed_data)
        
        return processed_data

    def save_feature_order(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save feature configuration
        with open(path, 'w') as f:
            json.dump({
                'feature_order': self.feature_order,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns,
            }, f, indent=4)
        
        # Save transformers for inference
        transformers_dir = os.path.dirname(path)
        joblib.dump(self.score_scaler, os.path.join(transformers_dir, 'score_scaler.pkl'))
        joblib.dump(self.scaler, os.path.join(transformers_dir, 'feature_scaler.pkl'))
        
        # Save label encoders
        for col, encoder in self.label_encoders.items():
            joblib.dump(encoder, os.path.join(transformers_dir, f'{col}_encoder.pkl'))
        
        # Save verification data for scaling checks
        if self.score_scaler is not None:
            verification_data = {
                'original_scale_example': float(self.score_scaler.inverse_transform([[1.0]])[0][0]),
                'scaled_example': float(self.score_scaler.transform([[1000.0]])[0][0])
            }
            verification_path = os.path.join(transformers_dir, 'scaling_verification.json')
            with open(verification_path, 'w') as f:
                json.dump(verification_data, f, indent=4)