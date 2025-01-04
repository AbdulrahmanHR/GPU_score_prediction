import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings('ignore')

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.knn_imputer = None
        self.scaler = None
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
        # Load data
        self.data = pd.read_csv(self.file_path)

        # Drop productName if it exists
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)

        # Handle outliers in numerical columns
        all_numerical = self.numerical_columns + ['score']
        self.handle_outliers(all_numerical)

        # Label encode categorical features
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Improve imputation with iterative KNN
        self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        impute_data = self.data[impute_columns].copy()
        self.data[impute_columns] = self.knn_imputer.fit_transform(impute_data)

        # Standardize numerical features
        self.scaler = StandardScaler()
        scale_data = self.data[all_numerical].copy()
        self.data[all_numerical] = self.scaler.fit_transform(scale_data)

        # Ensure correct column order
        return self.data[self.feature_order + ['score']]

    def save_feature_order(self, path):
        """Save feature order for inference"""
        import json
        with open(path, 'w') as f:
            json.dump({
                'feature_order': self.feature_order,
                'categorical_columns': self.categorical_columns,
                'numerical_columns': self.numerical_columns
            }, f)