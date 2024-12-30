# models.py
import xgboost as xgb
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.model_selection import train_test_split

class HybridModels:
    def __init__(self, data):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        
        # Implement train-test split (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=32
        )

    def _train_lstm(self, features, epochs=50, batch_size=32):
        features_train = features[:self.X_train.shape[0]]
        features_test = features[self.X_train.shape[0]:]

        lstm_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        lstm_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        lstm_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            LSTM(128, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mse')
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lstm_model.fit(lstm_input_train, self.y_train, epochs=epochs, 
                    batch_size=batch_size, verbose=0, validation_split=0.1, callbacks=[early_stopping])
        
        return lstm_model.predict(lstm_input_test).flatten()

    def _train_cnn(self, features, epochs=50, batch_size=32):
        features_train = features[:self.X_train.shape[0]]
        features_test = features[self.X_train.shape[0]:]
        
        cnn_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
        cnn_input_test = features_test.reshape((features_test.shape[0], features_test.shape[1], 1))
        
        cnn_model = Sequential([
            Input(shape=(features.shape[1], 1)),
            Conv1D(128, kernel_size=1, activation='relu'),
            Flatten(),
            Dense(1)
        ])
        cnn_model.compile(optimizer='adam', loss='mse')
        
        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        cnn_model.fit(cnn_input_train, self.y_train, epochs=epochs, 
                    batch_size=batch_size, verbose=0, validation_split=0.1, callbacks=[early_stopping])
    
        return cnn_model.predict(cnn_input_test).flatten()

    def xgboost_lstm(self):
        xgb_model = xgb.XGBRegressor(n_estimators=30, max_depth=40,
                                    learning_rate=0.01, colsample_bytree=0.8,
                                    early_stopping_rounds=10 )
        xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=0)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_lstm(xgb_features)
        return predictions, self.y_test

    def lightgbm_lstm(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=40, max_depth=40, 
                                    learning_rate=0.01, min_split_gain=0.01)
        lgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(10, verbose=0)],
        )
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_lstm(lgb_features)
        return predictions, self.y_test

    def xgboost_cnn(self):
        xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=40,
                                    learning_rate=0.01, colsample_bytree=0.8,
                                    early_stopping_rounds=10) 
        xgb_model.fit(self.X_train, self.y_train, eval_set=[(self.X_test, self.y_test)], verbose=0)
        xgb_features = xgb_model.apply(self.X)
        predictions = self._train_cnn(xgb_features)
        return predictions, self.y_test

    def lightgbm_cnn(self):
        lgb_model = lgb.LGBMRegressor(n_estimators=40, max_depth=20, 
                                    learning_rate=0.001, min_split_gain=0.01)
        lgb_model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            callbacks=[lgb.early_stopping(10, verbose=0)],
        ) 
        lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
        predictions = self._train_cnn(lgb_features)
        return predictions, self.y_test