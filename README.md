# GPU Performance Predictor

## Overview
GPU Performance Predictor is a hybrid machine learning and deep learning application that predicts GPU performance scores based on hardware specifications. The system leverages a combination of models such as XGBoost, LightGBM, LSTM, and CNN to provide accurate and insightful predictions.

## Features
- **Hybrid Models**: Combines XGBoost/LightGBM with LSTM/CNN architectures for enhanced prediction accuracy.
- **Interactive Web Interface**: Provides an intuitive interface using Streamlit for users to input GPU specifications and get predictions.
- **Customizable Components**: Includes modular data preparation, model training, and inference pipelines for easy customization.

## Directory Structure
```
.
├── app.py                 # Streamlit application for the web interface
├── main.py                # Main script for training models and saving components
├── inference.py           # Handles inference logic for predictions
├── models.py              # Defines hybrid model architectures and training pipelines
├── data_preparation.py    # Handles data preprocessing, encoding, and scaling
├── model_performance_chart.py # Generates performance metrics visualizations
├── requirements.txt       # List of Python dependencies
└── models/                # Directory for saving trained models and preprocessing components
```

## Setup and Installation
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## How to Run the Code

### Training the Models
To train the models and save preprocessing components:
```bash
python main.py
```

### Running the Web Application
To launch the interactive web interface:
```bash
streamlit run app.py
```

### Making Predictions Programmatically
You can use the `GPUPredictor` class in `inference.py` to make predictions programmatically:
```python
from inference import GPUPredictor

predictor = GPUPredictor()
input_data = {
    'manufacturer': 'NVIDIA',
    'gpuChip': 'AD106',
    'memType': 'GDDR6',
    'bus': 'PCIe 4.0 x16',
    'releaseYear': 2024,
    'memSize': 8,
    'memBusWidth': 128,
    'gpuClock': 1925,
    'memClock': 2250,
    'unifiedShader': 3840,
    'tmu': 120,
    'rop': 48
}

predictions, original_values = predictor.predict(input_data)
print(predictions)
```

## Usage

### Training Models
Run `main.py` to train the hybrid models and save preprocessing components and trained models to the `models/` directory:
```bash
python main.py
```

### Web Interface
Use the Streamlit interface to input GPU specifications and view predictions:
```bash
streamlit run app.py
```

## Model Architectures

### Tree-Based Models
- **XGBoost** and **LightGBM** models are used for feature extraction, leveraging gradient boosting algorithms.

### Deep Learning Models
- **LSTM**: Long Short-Term Memory networks are used for sequence modeling.
- **CNN**: Convolutional Neural Networks are applied for feature extraction.

### Hybrid Models
- **XGBoost + LSTM**
- **LightGBM + LSTM**
- **XGBoost + CNN**
- **LightGBM + CNN**

## Performance Metrics
Model performance is evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**
- **Mean Absolute Percentage Error (MAPE)**

Performance metrics are visualized as bar charts and saved as PNG files in the project directory.

## Key Files and Their Functions

### `main.py`
- Orchestrates model training and evaluation.
- Saves trained models and preprocessing components.

### `data_preparation.py`
- Preprocesses data by handling outliers, scaling, and encoding categorical variables.
- Implements KNN imputation for missing values.

### `models.py`
- Defines hybrid model architectures.
- Trains XGBoost/LightGBM models and combines them with LSTM/CNN models.

### `inference.py`
- Handles inference logic, including data preparation and prediction using trained models.

### `app.py`
- Provides a Streamlit-based web interface for user interaction and predictions.

### `requirements.txt`
- Lists the Python dependencies required for the project.

### `model_performance_chart.py`
- Visualizes performance metrics as bar charts.

## Dependencies
The project requires the following Python libraries (specified in `requirements.txt`):
- joblib
- keras
- lightgbm
- matplotlib
- numpy
- pandas
- scikit-learn
- streamlit
- tensorflow
- tensorflow-intel
- xgboost

---
**Developed by Abdulrahman Rahmouni**

