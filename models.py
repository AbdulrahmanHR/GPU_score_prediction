import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input
import numpy as np
from sklearn.model_selection import train_test_split

class HybridModels:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        
        # Implement train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def _train_lstm(self, features, epochs=20, batch_size=62):
        features_train = features[:len(self.X_train)]
        features_test = features[len(self.X_train):]
        
        lstm_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        lstm_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        lstm_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            LSTM(128, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(lstm_input_train, self.y_train, epochs=epochs, 
                      batch_size=batch_size, verbose=0, validation_split=0.1)
        
        return lstm_model.predict(lstm_input_test).flatten()

    def _train_cnn(self, features, epochs=20, batch_size=62):
        features_train = features[:len(self.X_train)]
        features_test = features[len(self.X_train):]
        
        cnn_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        cnn_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        cnn_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            Conv1D(128, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        cnn_model.fit(cnn_input_train, self.y_train, epochs=epochs, 
                     batch_size=batch_size, verbose=0, validation_split=0.1)
        
        return cnn_model.predict(cnn_input_test).flatten()

    def xgboost_lstm(self):
        xgb_model = xgb.XGBRegressor(n_estimators=20, max_depth=20, 
                                    learning_rate=0.001, colsample_bytree=0.75)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_lstm(xgb_features, epochs=20, batch_size=62)
        return predictions, self.y_test

    def lightgbm_lstm(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=20, max_depth=20, 
                                    learning_rate=0.001, min_split_gain=0.01)
        lgb_model.fit(self.X_train, self.y_train)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_lstm(lgb_features, epochs=20, batch_size=62)
        return predictions, self.y_test

    def xgboost_cnn(self):
        xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=40, 
                                    learning_rate=0.001, colsample_bytree=1, subsample=0.9)
        xgb_model.fit(self.X_train, self.y_train)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_cnn(xgb_features, epochs=20, batch_size=62)
        return predictions, self.y_test

    def lightgbm_cnn(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=20, 
                                    learning_rate=0.001, min_split_gain=0.01)
        lgb_model.fit(self.X_train, self.y_train)
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_cnn(lgb_features, epochs=20, batch_size=62)
        return predictions, self.y_test