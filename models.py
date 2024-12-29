import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input
import numpy as np


class HybridModels:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values

    def _train_lstm(self, features, epochs=20, batch_size=62):
        # Ensure the input is reshaped for LSTM
        lstm_input = features.reshape((features.shape[0], features.shape[1], 1))
        
        lstm_model = Sequential([
            Input(shape=(features.shape[1], 1)),  # Explicit Input layer
            LSTM(128, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(lstm_input, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        return lstm_model.predict(lstm_input).flatten()

    def xgboost_lstm(self):
        xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=20, learning_rate=0.01, colsample_bytree=0.75)
        xgb_model.fit(self.X, self.y)
        xgb_features = xgb_model.apply(self.X)  # Extract tree leaf indices
        
        return self._train_lstm(xgb_features, epochs=20, batch_size=62)

    def lightgbm_lstm(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=20, max_depth=20, learning_rate=0.01, min_split_gain=0.01)
        lgb_model.fit(self.X, self.y)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)  # Continuous predictions
        
        return self._train_lstm(lgb_features, epochs=20, batch_size=62)
    
    
    
    def _train_cnn(self, features, epochs=20, batch_size=32):
       # Ensure the input is reshaped for CNN
        cnn_input = features.reshape((features.shape[0], features.shape[1], 1))
        
        cnn_model = Sequential([
            Input(shape=(features.shape[1], 1)),  # Explicit Input layer
            Conv1D(128, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        cnn_model.fit(cnn_input, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        return cnn_model.predict(cnn_input).flatten()

    def xgboost_cnn(self):
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=40, learning_rate=0.001, colsample_bytree=1, subsample = 0.9)
        xgb_model.fit(self.X, self.y)
        xgb_features = xgb_model.apply(self.X)  # Extract tree leaf indices
        
        return self._train_cnn(xgb_features, epochs=20, batch_size=62)

    def lightgbm_cnn(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=20, learning_rate=0.01, min_split_gain=0.01)
        lgb_model.fit(self.X, self.y)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)  # Continuous predictions
        
        return self._train_cnn(lgb_features, epochs=20, batch_size=62)
    
'''import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

class HybridModels:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        
        # Implement train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Store results for comparison
        self.model_results = {}
        
    def evaluate_model(self, y_pred, model_name):
        """
        Comprehensive model evaluation with multiple metrics
        """
        # Calculate various metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        mape = mean_absolute_percentage_error(self.y_test, y_pred) * 100
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate custom accuracy metric (predictions within 10% of actual)
        accuracy = np.mean(np.abs((self.y_test - y_pred) / self.y_test) <= 0.1) * 100
        
        # Store results
        self.model_results[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Accuracy_10%': accuracy,
            'predictions': y_pred
        }
        
        # Print detailed results
        print(f"\n{model_name} Performance Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"R2 Score: {r2:.4f}")
        print(f"Accuracy (within 10%): {accuracy:.2f}%")
        
        # Visualize predictions vs actual
        self.plot_predictions(y_pred, model_name)
        
        return self.model_results[model_name]

    def plot_predictions(self, y_pred, model_name):
        """
        Create scatter plot of predicted vs actual values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Predicted vs Actual Values')
        plt.tight_layout()
        plt.show()

    def compare_models(self):
        """
        Compare all trained models' performance
        """
        if not self.model_results:
            print("No models have been trained yet.")
            return
        
        metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2', 'Accuracy_10%']
        
        print("\nModel Comparison:")
        print("-" * 100)
        print(f"{'Model':<15} {'MSE':>12} {'RMSE':>12} {'MAE':>12} {'MAPE':>12} {'R2':>12} {'Accuracy':>12}")
        print("-" * 100)
        
        for model_name, results in self.model_results.items():
            print(f"{model_name:<15}", end="")
            for metric in metrics:
                if metric in ['MAPE', 'Accuracy_10%']:
                    print(f"{results[metric]:>11.2f}%", end="")
                else:
                    print(f"{results[metric]:>12.4f}", end="")
            print()

    def _train_lstm(self, features, epochs=20, batch_size=62):
        # Split features for training
        features_train = features[:len(self.X_train)]
        features_test = features[len(self.X_train):]
        
        # Reshape input for LSTM
        lstm_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        lstm_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        lstm_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            LSTM(128, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Add history for tracking training progress
        history = lstm_model.fit(lstm_input_train, self.y_train, epochs=epochs, 
                               batch_size=batch_size, verbose=0, validation_split=0.1)
        
        # Plot training history
        self.plot_training_history(history, 'LSTM')
        
        return lstm_model.predict(lstm_input_test).flatten()

    def _train_cnn(self, features, epochs=20, batch_size=32):
        # Split features for training
        features_train = features[:len(self.X_train)]
        features_test = features[len(self.X_train):]
        
        # Reshape input for CNN
        cnn_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        cnn_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        cnn_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            Conv1D(128, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        
        # Add history for tracking training progress
        history = cnn_model.fit(cnn_input_train, self.y_train, epochs=epochs, 
                              batch_size=batch_size, verbose=0, validation_split=0.1)
        
        # Plot training history
        self.plot_training_history(history, 'CNN')
        
        return cnn_model.predict(cnn_input_test).flatten()

    def plot_training_history(self, history, model_type):
        """
        Plot training and validation loss
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type} Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def xgboost_lstm(self):
        xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=20, 
                                    learning_rate=0.01, colsample_bytree=0.75)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_lstm(xgb_features, epochs=20, batch_size=62)
        return self.evaluate_model(predictions, "XGBoost-LSTM")

    def lightgbm_lstm(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=20, max_depth=20, 
                                     learning_rate=0.01, min_split_gain=0.01)
        lgb_model.fit(self.X_train, self.y_train)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_lstm(lgb_features, epochs=20, batch_size=62)
        return self.evaluate_model(predictions, "LightGBM-LSTM")

    def xgboost_cnn(self):
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=40, 
                                    learning_rate=0.001, colsample_bytree=1, subsample=0.9)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_cnn(xgb_features, epochs=20, batch_size=62)
        return self.evaluate_model(predictions, "XGBoost-CNN")

    def lightgbm_cnn(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=20, 
                                     learning_rate=0.01, min_split_gain=0.01)
        lgb_model.fit(self.X_train, self.y_train)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_cnn(lgb_features, epochs=20, batch_size=62)
        return self.evaluate_model(predictions, "LightGBM-CNN")

# Example usage:
# hybrid_models = HybridModels(your_data)
# hybrid_models.xgboost_lstm()
# hybrid_models.lightgbm_lstm()
# hybrid_models.xgboost_cnn()
# hybrid_models.lightgbm_cnn()
# hybrid_models.compare_models()  # Compare all trained models'''