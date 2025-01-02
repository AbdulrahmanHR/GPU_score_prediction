# models.py
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
import numpy as np
from sklearn.model_selection import KFold

class HybridModels:
    def __init__(self, data, n_splits=5):
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        self.n_splits = n_splits
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def train_lstm(self, features, model_name, epochs=50, batch_size=32):
        fold_predictions = []
        fold_true_values = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(features)):
            features_train = features[train_idx]
            features_val = features[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]

            lstm_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
            lstm_input_val = features_val.reshape((features_val.shape[0], features_val.shape[1], 1))
            
            lstm_model = Sequential([
                Input(shape=(features.shape[1], 1)),
                LSTM(128, activation='relu'),
                Dropout(0.1),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer='adam', loss='mse')
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = lstm_model.fit(
                lstm_input_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(lstm_input_val, y_val),
                callbacks=[early_stopping]
            )
            
            # Evaluate model
            mse = lstm_model.evaluate(lstm_input_val, y_val, verbose=0)
            rmse = np.sqrt(mse)
            print(f'{model_name} Fold {fold+1} RMSE: {rmse:.4f}')
            
            # Store predictions and true values for this fold
            fold_predictions.append(lstm_model.predict(lstm_input_val).flatten())
            fold_true_values.append(y_val)
            
        # Combine all fold predictions and true values
        all_predictions = np.concatenate(fold_predictions)
        all_true_values = np.concatenate(fold_true_values)
        
        # Calculate overall RMSE
        overall_rmse = np.sqrt(np.mean((all_predictions - all_true_values) ** 2))
        print(f'{model_name} Overall RMSE: {overall_rmse:.4f}')
        
        return lstm_model, all_predictions, all_true_values

    def train_cnn(self, features, model_name, epochs=50, batch_size=32):
        fold_predictions = []
        fold_true_values = []
        
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(features)):
            features_train = features[train_idx]
            features_val = features[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]

            cnn_input_train = features_train.reshape((features_train.shape[0], features_train.shape[1], 1))
            cnn_input_val = features_val.reshape((features_val.shape[0], features_val.shape[1], 1))
            
            cnn_model = Sequential([
                Input(shape=(features.shape[1], 1)),
                Conv1D(128, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.GlobalAveragePooling1D(),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dense(1)
            ])
            
            cnn_model.compile(optimizer='adam', loss='mse')
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = cnn_model.fit(
                cnn_input_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=(cnn_input_val, y_val),
                callbacks=[early_stopping]
            )
            
            # Evaluate model
            mse = cnn_model.evaluate(cnn_input_val, y_val, verbose=0)
            rmse = np.sqrt(mse)
            print(f'{model_name} Fold {fold+1} RMSE: {rmse:.4f}')
            
            # Store predictions and true values for this fold
            fold_predictions.append(cnn_model.predict(cnn_input_val).flatten())
            fold_true_values.append(y_val)
            
        # Combine all fold predictions and true values
        all_predictions = np.concatenate(fold_predictions)
        all_true_values = np.concatenate(fold_true_values)
        
        # Calculate overall RMSE
        overall_rmse = np.sqrt(np.mean((all_predictions - all_true_values) ** 2))
        print(f'{model_name} Overall RMSE: {overall_rmse:.4f}')
        
        return cnn_model, all_predictions, all_true_values

    def xgboost_lstm(self):
            fold_predictions = []
            fold_true_values = []
            best_score = float('inf')
            best_models = None
            
            # Improved XGBoost parameters
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.01,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'early_stopping_rounds': 20,
                'eval_metric': 'rmse'
            }
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X)):
                X_train = self.X[train_idx]
                X_val = self.X[val_idx]
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                xgb_model = xgb.XGBRegressor(**xgb_params)
                
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )
                
                # Generate features for LSTM
                xgb_features = xgb_model.predict(self.X).reshape(-1, 1)
                lstm_model, predictions, true_values = self.train_lstm(xgb_features, f'xgboost_lstm_fold_{fold+1}')
                
                # Calculate fold score
                fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
                
                # Save best models
                if fold_score < best_score:
                    best_score = fold_score
                    best_models = {
                        'feature_extractor': xgb_model,
                        'predictor': lstm_model
                    }
                
                fold_predictions.append(predictions)
                fold_true_values.append(true_values)
            
            # Combine results from all folds
            all_predictions = np.concatenate(fold_predictions)
            all_true_values = np.concatenate(fold_true_values)
            
            return {
                'predictions': all_predictions,
                'true_values': all_true_values,
                'feature_extractor': best_models['feature_extractor'],
                'predictor': best_models['predictor']
            }

    def lightgbm_lstm(self):
            fold_predictions = []
            fold_true_values = []
            best_score = float('inf')
            best_models = None
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X)):
                X_train = self.X[train_idx]
                X_val = self.X[val_idx]
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=50,
                    max_depth=25,
                    learning_rate=0.1,
                    min_split_gain=0.01
                )
                
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10, verbose=0)]
                )
                
                # Generate features for LSTM
                lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
                lstm_model, predictions, true_values = self.train_lstm(lgb_features, f'lightgbm_lstm_fold_{fold+1}')
                
                # Calculate fold score
                fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
                
                # Save best models
                if fold_score < best_score:
                    best_score = fold_score
                    best_models = {
                        'feature_extractor': lgb_model,
                        'predictor': lstm_model
                    }
                
                fold_predictions.append(predictions)
                fold_true_values.append(true_values)
            
            # Combine results from all folds
            all_predictions = np.concatenate(fold_predictions)
            all_true_values = np.concatenate(fold_true_values)
            
            return {
                'predictions': all_predictions,
                'true_values': all_true_values,
                'feature_extractor': best_models['feature_extractor'],
                'predictor': best_models['predictor']
            }

    def xgboost_cnn(self):
            fold_predictions = []
            fold_true_values = []
            best_score = float('inf')
            best_models = None
            
            # Improved XGBoost parameters
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.01,
                'colsample_bytree': 0.8,
                'subsample': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'early_stopping_rounds': 20,
                'eval_metric': 'rmse'
            }
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X)):
                X_train = self.X[train_idx]
                X_val = self.X[val_idx]
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                xgb_model = xgb.XGBRegressor(**xgb_params)
                
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=0
                )
                
                # Generate features for LSTM
                xgb_features = xgb_model.predict(self.X).reshape(-1, 1)
                cnn_model, predictions, true_values = self.train_cnn(xgb_features, f'xgboost_cnn_fold_{fold+1}')
                
                # Calculate fold score
                fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
                
                # Save best models
                if fold_score < best_score:
                    best_score = fold_score
                    best_models = {
                        'feature_extractor': xgb_model,
                        'predictor': cnn_model
                    }
                
                fold_predictions.append(predictions)
                fold_true_values.append(true_values)
            
            # Combine results from all folds
            all_predictions = np.concatenate(fold_predictions)
            all_true_values = np.concatenate(fold_true_values)
            
            return {
                'predictions': all_predictions,
                'true_values': all_true_values,
                'feature_extractor': best_models['feature_extractor'],
                'predictor': best_models['predictor']
            }
        
    def lightgbm_cnn(self):
            fold_predictions = []
            fold_true_values = []
            best_score = float('inf')
            best_models = None
            
            for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X)):
                X_train = self.X[train_idx]
                X_val = self.X[val_idx]
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=50,
                    max_depth=25,
                    learning_rate=0.1,
                    min_split_gain=0.01
                )
                
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10, verbose=0)]
                )
                
                # Generate features for CNN
                lgb_features = lgb_model.predict(self.X).reshape(-1, 1)
                cnn_model, predictions, true_values = self.train_cnn(lgb_features, f'lightgbm_cnn_fold_{fold+1}')
                
                # Calculate fold score
                fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
                
                # Save best models
                if fold_score < best_score:
                    best_score = fold_score
                    best_models = {
                        'feature_extractor': lgb_model,
                        'predictor': cnn_model
                    }
                
                fold_predictions.append(predictions)
                fold_true_values.append(true_values)
            
            # Combine results from all folds
            all_predictions = np.concatenate(fold_predictions)
            all_true_values = np.concatenate(fold_true_values)
            
            return {
                'predictions': all_predictions,
                'true_values': all_true_values,
                'feature_extractor': best_models['feature_extractor'],
                'predictor': best_models['predictor']
            }