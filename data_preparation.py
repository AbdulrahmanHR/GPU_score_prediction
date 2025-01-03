# data_preparation.py
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

    def handle_outliers(self, columns, method="percentile", n_std=3):
        """Remove or cap outliers based on the chosen method: percentile or standard deviation."""
        for col in columns:
            if method == "percentile":
                lower_bound = self.data[col].quantile(0.01)  # 1st percentile
                upper_bound = self.data[col].quantile(0.99)  # 99th percentile
            elif method == "std":
                mean = self.data[col].mean()
                std = self.data[col].std()
                lower_bound = mean - n_std * std
                upper_bound = mean + n_std * std
            else:
                raise ValueError("Invalid method. Choose 'percentile' or 'std'.")

            # Cap the outliers
            self.data[col] = self.data[col].clip(lower_bound, upper_bound)

    def preprocess_data(self):
        # Load data
        self.data = pd.read_csv(self.file_path)

        # Drop productName if it exists
        if "productName" in self.data.columns:
            self.data.drop(columns=["productName"], inplace=True)

        # Handle outliers in numerical columns
        numerical_columns = [
            "memSize", "memBusWidth", "gpuClock", "memClock",
            "unifiedShader", "tmu", "rop", "score", "releaseYear"
        ]
        self.handle_outliers(numerical_columns)

        # Label encode all categorical features
        categorical_columns = ['gpuChip', 'bus', 'memType', 'manufacturer']
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Improve imputation with iterative KNN
        self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        self.data[impute_columns] = self.knn_imputer.fit_transform(self.data[impute_columns])

        # Add imputation confidence scores
        for col in impute_columns:
            missing_mask = self.data[col].isna()
            if missing_mask.any():
                self.data[f'{col}_imputed'] = missing_mask.astype(int)

        # Standardize numerical features
        self.scaler = StandardScaler()
        self.data[numerical_columns] = self.scaler.fit_transform(self.data[numerical_columns])

        return self.data
