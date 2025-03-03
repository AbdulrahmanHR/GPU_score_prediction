# data_preparation.py
import os
import json
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataPreparation:
    """
    A class for preparing GPU data for machine learning models.
    Handles loading, preprocessing, encoding categorical features,
    scaling numerical features, and managing train/test splits.
    """
    def __init__(self, file_path):
        """
        Initialize the DataPreparation class.
        
        Parameters
        ----------
        file_path : str
            Path to the CSV file containing the GPU data.
        """
        self.file_path = file_path
        self.data = None
        self.ordinal_encoder = None
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
        """
        Load data from CSV file without any processing.
        
        Returns
        -------
        pandas.DataFrame
            The raw data loaded from the CSV file.
        """
        self.data = pd.read_csv(self.file_path)
        
        # Remove product name if present (not used for prediction)
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)
            
        return self.data
    
    def fit_transformers(self, train_df):
        """
        Fit encoders and scalers on training data to prevent data leakage.
        This should be called only on the training set or a single dataset
        that will be used for fitting a model.
        
        Parameters
        ----------
        train_df : pandas.DataFrame
            The training data to fit the transformers on.
        """
        # Initialize transformers
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = MinMaxScaler()
        self.score_scaler = MinMaxScaler()
        
        # Fit ordinal encoder on categorical features
        self.ordinal_encoder.fit(train_df[self.categorical_columns])
        
        # Fit scalers on numerical features and target
        self.scaler.fit(train_df[self.numerical_columns])
        
        if 'score' in train_df.columns:
            self.score_scaler.fit(train_df[['score']])
    
    def transform_data(self, df, transform_target=True):
        """
        Apply transformations using pre-fitted transformers.
        Can be applied to both training and test data.
        
        Parameters
        ----------
        df : pandas.DataFrame
            The data to transform.
        transform_target : bool, default=True
            Whether to transform the target variable 'score' if present.
            
        Returns
        -------
        pandas.DataFrame
            The transformed data with encoded categorical features and scaled numerical features.
        """
        result_df = df.copy()
        
        # Apply ordinal encoder to categorical features
        result_df[self.categorical_columns] = self.ordinal_encoder.transform(result_df[self.categorical_columns])
        
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
        
        Parameters
        ----------
        test_size : float, default=0.15
            The proportion of the data to include in the test split.
        random_state : int, default=42
            Controls the shuffling applied to the data before splitting.
            
        Returns
        -------
        tuple
            A tuple containing two pandas.DataFrame objects:
            (train_processed, test_processed)
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
        
        # Step 4: Fit transformers on training data only
        self.fit_transformers(train_data)
        
        # Step 5: Apply transformations to both sets
        train_processed = self.transform_data(train_data)
        test_processed = self.transform_data(test_data)
        
        return train_processed, test_processed
    
    def preprocess_data(self):
        """
        Complete preprocessing method for a single dataset
        (without splitting into train/test).
        
        Returns
        -------
        pandas.DataFrame
            The processed data with encoded categorical features and scaled numerical features.
        """
        # Step 1: Load raw data
        raw_data = self.load_data()
        
        # Step 2: Handle missing values
        initial_rows = raw_data.shape[0]
        cleaned_data = raw_data.dropna()
        removed_rows = initial_rows - cleaned_data.shape[0]
        print(f"Removed {removed_rows} rows with missing values.")
        
        # Step 3: Fit transformers on the cleaned dataset
        self.fit_transformers(cleaned_data)
        
        # Step 4: Apply transformations to the cleaned dataset
        processed_data = self.transform_data(cleaned_data)
        
        return processed_data

    def save_feature_order(self, path):
        """
        Save feature configuration and transformers to disk for later use in inference.
        
        Parameters
        ----------
        path : str
            Path where the feature configuration JSON file should be saved.
            Transformers will be saved in the same directory.
        """
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
        joblib.dump(self.ordinal_encoder, os.path.join(transformers_dir, 'ordinal_encoder.pkl'))
        
        # Save verification data for scaling checks
        if self.score_scaler is not None:
            verification_data = {
                'original_scale_example': float(self.score_scaler.inverse_transform([[1.0]])[0][0]),
                'scaled_example': float(self.score_scaler.transform([[1000.0]])[0][0])
            }
            verification_path = os.path.join(transformers_dir, 'scaling_verification.json')
            with open(verification_path, 'w') as f:
                json.dump(verification_data, f, indent=4)