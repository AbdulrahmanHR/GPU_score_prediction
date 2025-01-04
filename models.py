# models.py
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout, BatchNormalization # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.optimizers import Adam # type: ignore
import numpy as np
from sklearn.model_selection import KFold

class HybridModels:
    def __init__(self, data, n_splits=5):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=32)
    
    def create_lstm_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            LSTM(256, return_sequences=True, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')  # Huber loss is more robust to outliers
        return model
    
    def create_cnn_model(self, input_shape):
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.3),
            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            Flatten(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model
    
    def train_deep_model(self, model, features_train, features_val, y_train, y_val, model_name, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-4
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                min_delta=1e-4
            )
        ]
        
        history = model.fit(
            features_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(features_val, y_val),
            callbacks=callbacks
        )
        
        predictions = model.predict(features_val).flatten()
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2)) * 1000
        print(f'{model_name} RMSE: {rmse:.2f}')
        
        return model, predictions, y_val
    
    def train_xgboost_model(self, X_train, X_val, y_train, y_val):
        params = {
            'n_estimators': 500,
            'max_depth': 25,
            'learning_rate': 0.01,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0
        )
        return model
    
    def train_lightgbm_model(self, X_train, X_val, y_train, y_val):
        params = {
            'n_estimators': 500,
            'max_depth': 15,
            'learning_rate': 0.01,
            'num_leaves': 32,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=0)]
        )
        return model
    
    def xgboost_lstm(self):
        return self._train_hybrid_model('xgboost', self.train_xgboost_model, self.create_lstm_model)
    
    def lightgbm_lstm(self):
        return self._train_hybrid_model('lightgbm', self.train_lightgbm_model, self.create_lstm_model)
    
    def xgboost_cnn(self):
        return self._train_hybrid_model('xgboost', self.train_xgboost_model, self.create_cnn_model)
    
    def lightgbm_cnn(self):
        return self._train_hybrid_model('lightgbm', self.train_lightgbm_model, self.create_cnn_model)
    
    def _train_hybrid_model(self, name, tree_model_func, deep_model_func):
        fold_predictions = []
        fold_true_values = []
        best_score = float('inf')
        best_models = None
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X)):
            X_train = self.X[train_idx]
            X_val = self.X[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            # Train tree model
            tree_model = tree_model_func(X_train, X_val, y_train, y_val)
            
            # Generate features
            features_train = tree_model.predict(X_train).reshape(-1, 1, 1)  # Change reshape
            features_val = tree_model.predict(X_val).reshape(-1, 1, 1)      # Change reshape
            
            # Create and train deep model
            deep_model = deep_model_func((1, 1))
            deep_model, predictions, true_values = self.train_deep_model(
                deep_model,
                features_train,
                features_val,
                y_train,
                y_val,
                f'{name}_fold_{fold+1}'
            )
            
            # Calculate fold score
            fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
            
            if fold_score < best_score:
                best_score = fold_score
                best_models = {
                    'feature_extractor': tree_model,
                    'predictor': deep_model
                }
            
            fold_predictions.append(predictions)
            fold_true_values.append(true_values)
        
        return {
            'predictions': np.concatenate(fold_predictions),
            'true_values': np.concatenate(fold_true_values),
            'feature_extractor': best_models['feature_extractor'],
            'predictor': best_models['predictor']
        }