# predict.py
from inference import InferencePipeline
import pandas as pd
import joblib

# Load model paths
model_paths = joblib.load('models/model_paths.pkl')

# Initialize the inference pipeline
pipeline = InferencePipeline()

# Load models and transformers
pipeline.load_models(model_paths)

# Prepare input data
input_data = pd.DataFrame([{
    'manufacturer': 'AMD',
    'releaseYear': 2021,
    'memSize': 8,
    'memBusWidth': 128,
    'gpuClock': 1700,
    'memClock': 12000,
    'unifiedShader': 2048,
    'tmu': 128,
    'rop': 64,
    'bus': 'PCIe 4.0 x8',
    'memType': 'GDDR6',
    'gpuChip': 'Navi 23'
}])

# Predict with all models
all_predictions = pipeline.predict_all(input_data)

# Display results
for model_name, preds in all_predictions.items():
    print(f"Predictions from {model_name}: {preds}")
