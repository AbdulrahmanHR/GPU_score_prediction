# models.py
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout  # type: ignore
from keras.callbacks import EarlyStopping  # type: ignore
import numpy as np

class HybridModels:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        
        # Calculate split sizes
        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.2)
        test_size = len(data) - train_size - val_size
        
        # Split the data
        self.X_train = self.X[:train_size]
        self.y_train = self.y[:train_size]
        
        self.X_val = self.X[train_size:train_size + val_size]
        self.y_val = self.y[train_size:train_size + val_size]
        
        self.X_test = self.X[train_size + val_size:train_size + val_size + test_size]
        self.y_test = self.y[train_size + val_size:train_size + val_size + test_size]
        

    def train_lstm(self, features, model_name, epochs=50, batch_size=32):
        features_train = features[:self.X_train.shape[0]]
        features_val = features[self.X_train.shape[0]:self.X_train.shape[0] + self.X_val.shape[0]]
        features_test = features[self.X_train.shape[0] + self.X_val.shape[0]:]

        lstm_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        lstm_input_val = features_val.reshape((features_val.shape[0], features_val.shape[1], 1))
        lstm_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        lstm_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            LSTM(128, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with validation data
        history = lstm_model.fit(
            lstm_input_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(lstm_input_val, self.y_val),
            callbacks=[early_stopping]
        )
        
        # Evaluate model
        mse = lstm_model.evaluate(lstm_input_val, self.y_val, verbose=0)
        rmse = np.sqrt(mse)
        print(f'{model_name} Validation MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        
        test_predictions = lstm_model.predict(lstm_input_test).flatten()
        return lstm_model, test_predictions

    def train_cnn(self, features, model_name, epochs=50, batch_size=32):
        features_train = features[:self.X_train.shape[0]]
        features_val = features[self.X_train.shape[0]:self.X_train.shape[0] + self.X_val.shape[0]]
        features_test = features[self.X_train.shape[0] + self.X_val.shape[0]:]
        
        cnn_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        cnn_input_val = features_val.reshape((features_val.shape[0], features_val.shape[1], 1))
        cnn_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        cnn_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            Conv1D(128, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train with validation data
        history = cnn_model.fit(
            cnn_input_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(cnn_input_val, self.y_val),
            callbacks=[early_stopping]
        )
        
        # Evaluate model
        mse = cnn_model.evaluate(cnn_input_val, self.y_val, verbose=0)
        rmse = np.sqrt(mse)
        print(f'{model_name} Validation MSE: {mse:.4f}, RMSE: {rmse:.4f}')
        
        test_predictions = cnn_model.predict(cnn_input_test).flatten()
        return cnn_model, test_predictions

    def xgboost_lstm(self):
        param_grid = {
            'n_estimators': [40, 50, 100, 150],
            'max_depth': [20, 40, 50, 60],
            'learning_rate': [0.001, 0.01, 0.05, 0.1],
            'colsample_bytree': [0.6, 0.7, 0.8]
        }
        best_params = None
        best_score = float('inf')

        for n in param_grid['n_estimators']:
            for d in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for col in param_grid['colsample_bytree']:
                        model = xgb.XGBRegressor(
                            n_estimators=n,
                            max_depth=d,
                            learning_rate=lr,
                            colsample_bytree=col,
                            early_stopping_rounds=10,
                            eval_metric='rmse'
                        )
                        model.fit(
                            self.X_train, self.y_train,
                            eval_set=[(self.X_val, self.y_val)],
                            verbose=0
                        )
                        evals_result = model.evals_result()
                        score = evals_result['validation_0']['rmse'][-1]
                        if score < best_score:
                            best_score = score
                            best_params = (n, d, lr, col)

        xgb_model = xgb.XGBRegressor(
            n_estimators=best_params[0],
            max_depth=best_params[1],
            learning_rate=best_params[2],
            colsample_bytree=best_params[3],
            early_stopping_rounds=10,
            eval_metric='rmse'
        )
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=0
        )
        
        xgb_features = xgb_model.predict(self.X).reshape(-1, 1)
        lstm_model, predictions = self.train_lstm(xgb_features, 'xgboost_lstm')
        
        # Generate training predictions
        train_features = xgb_model.predict(self.X_train).reshape(-1, 1)
        train_input = train_features.reshape((train_features.shape[0], train_features.shape[1], 1))
        train_predictions = lstm_model.predict(train_input).flatten()
        
        return {
            'feature_extractor': xgb_model,
            'predictor': lstm_model,
            'predictions': predictions,
            'true_values': self.y_test,
            'train_predictions': train_predictions,
            'train_true_values': self.y_train
        }

    def lightgbm_lstm(self):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=25,
            learning_rate=0.1,
            min_split_gain=0.01
        )
        lgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(10, verbose=0)],
        )
        
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        lstm_model, predictions = self.train_lstm(lgb_features, 'lightgbm_lstm')
        
        # Generate training predictions
        train_features = lgb_model.predict(self.X_train).reshape(-1, 1)
        train_input = train_features.reshape((train_features.shape[0], 1, 1))
        train_predictions = lstm_model.predict(train_input).flatten()
        
        return {
            'feature_extractor': lgb_model,
            'predictor': lstm_model,
            'predictions': predictions,
            'true_values': self.y_test,
            'train_predictions': train_predictions,
            'train_true_values': self.y_train
        }

    def xgboost_cnn(self):
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=45,
            learning_rate=0.001,
            colsample_bytree=0.7,
            early_stopping_rounds=10
        )
        xgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=0
        )
        
        xgb_features = xgb_model.apply(self.X)
        cnn_model, predictions = self.train_cnn(xgb_features, 'xgboost_cnn')
        
        # Generate training predictions
        train_features = xgb_model.apply(self.X_train)
        train_input = train_features.reshape((train_features.shape[0], train_features.shape[1], 1))
        train_predictions = cnn_model.predict(train_input).flatten()
        
        return {
            'feature_extractor': xgb_model,
            'predictor': cnn_model,
            'predictions': predictions,
            'true_values': self.y_test,
            'train_predictions': train_predictions,
            'train_true_values': self.y_train
        }

    def lightgbm_cnn(self):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=25,
            learning_rate=0.1,
            min_split_gain=0.01
        )
        lgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.early_stopping(10, verbose=0)],
        )
        
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        cnn_model, predictions = self.train_cnn(lgb_features, 'lightgbm_cnn')
        
        # Generate training predictions
        train_features = lgb_model.predict(self.X_train).reshape(-1, 1)
        train_input = train_features.reshape((train_features.shape[0], 1, 1))
        train_predictions = cnn_model.predict(train_input).flatten()
        
        return {
            'feature_extractor': lgb_model,
            'predictor': cnn_model,
            'predictions': predictions,
            'true_values': self.y_test,
            'train_predictions': train_predictions,
            'train_true_values': self.y_train
        }
