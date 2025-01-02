# data_preparation.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.knn_imputer = None
        self.scaler = None

    def preprocess_data(self):
        # Load data
        self.data = pd.read_csv(self.file_path)

        # Drop productName
        self.data.drop(columns=["productName"], inplace=True)

        # Label encode gpuChip, bus, memType, manufacturer
        self.label_encoders = {}
        for column in ['gpuChip', 'bus', 'memType', 'manufacturer']:
            self.label_encoders[column] = LabelEncoder()
            self.data[column] = self.label_encoders[column].fit_transform(self.data[column])

        # Apply KNN Imputation
        self.knn_imputer = KNNImputer(n_neighbors=5)
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        self.data[impute_columns] = self.knn_imputer.fit_transform(self.data[impute_columns])

        # Standardize numerical features
        self.scaler = StandardScaler()
        numerical_features = ["releaseYear", "memSize", "memBusWidth", "gpuClock", 
                            "memClock", "unifiedShader", "tmu", "rop"]
        self.data[numerical_features] = self.scaler.fit_transform(self.data[numerical_features])

        return self.data