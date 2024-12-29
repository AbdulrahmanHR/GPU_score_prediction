# data_preparation.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import KNNImputer

class DataPreparation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def preprocess_data(self):
        # Load data
        self.data = pd.read_csv(self.file_path)

        # Drop productName
        self.data.drop(columns=["productName"], inplace=True)

        # One-hot encode manufacturer
        self.data = pd.get_dummies(self.data, columns=["manufacturer"], drop_first=True)

        # Label encode gpuChip, bus, and memType
        le = LabelEncoder()
        self.data["gpuChip"] = le.fit_transform(self.data["gpuChip"])
        self.data["bus"] = le.fit_transform(self.data["bus"])
        self.data["memType"] = le.fit_transform(self.data["memType"])

        # Apply KNN Imputation to memSize, memBusWidth, memClock
        knn_imputer = KNNImputer(n_neighbors=5)  # Using 5 nearest neighbors
        impute_columns = ["memSize", "memBusWidth", "memClock"]
        self.data[impute_columns] = knn_imputer.fit_transform(self.data[impute_columns])

        # Standardize numerical features
        numerical_features = ["releaseYear", "memSize", "memBusWidth", "gpuClock", "memClock", "unifiedShader", "tmu", "rop"]
        scaler = StandardScaler()
        self.data[numerical_features] = scaler.fit_transform(self.data[numerical_features])

        return self.data
