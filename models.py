# models.py
from matplotlib import pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout, BatchNormalization # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.optimizers import Adam # type: ignore

class HybridModels:
    def __init__(self, train_data, test_data=None, n_splits=5):
        """
        Initialize the hybrid models with training and test data.
        
        Args:
            train_data: DataFrame containing GPU specifications and performance scores
                        (Should be pre-processed training data)
            test_data: DataFrame containing test data (optional)
            n_splits: Number of folds for k-fold cross-validation (default: 5)
        """
        self.train_data = train_data
        self.test_data = test_data
        self.X = train_data.drop(columns=["score"]).values
        self.y = train_data["score"].values
        self.n_splits = int(n_splits)
        self.feature_names = train_data.drop(columns=["score"]).columns.tolist()
        
        # Create directories for saving plots
        os.makedirs("Importance_plots", exist_ok=True)  # Folder for importance plots
               
        # Define importance weights for key GPU features
        self.feature_weights = {
            'gpuClock': 3.0,
            'unifiedShader': 2.9,
            'gpuChip': 1.5,
            'memSize': 1.3,
            'rop': 1.0,
            'memClock': 1.3,          
            'releaseYear': 0.8,       
            'manufacturer': 0.6, 
            'memBusWidth': 0.6,
            'tmu': 0.6,             
            'memType': 1.0,          
            'bus': 0.6               
        }
        
        # Convert feature weights to list matching feature order
        self.feature_weight_list = [self.feature_weights.get(feat, 1.0) 
                                  for feat in self.feature_names]
        
        self.training_history = {
            'xgboost_lstm': {},
            'lightgbm_lstm': {},
            'xgboost_cnn': {},
            'lightgbm_cnn': {}
        }
        
        self.cv_results = {
            'xgboost_lstm': {},
            'lightgbm_lstm': {},
            'xgboost_cnn': {},
            'lightgbm_cnn': {}
        }
        
        # Store best models
        self.best_models = {}
        
    def prepare_stratified_folds(self):
        """
        Prepare stratified folds for cross-validation using only training data.
        This avoids data leakage from using information about the entire dataset.
        """
        # Ensure n_splits is an integer
        n_splits = int(self.n_splits)
        
        # Create stratification bins for performance scores from training data only
        stratifier = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        y_binned = stratifier.fit_transform(self.y.reshape(-1, 1)).ravel()
        
        # Initialize stratified k-fold
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
        
        # Return fold indices
        return list(kf.split(self.X, y_binned))
        
    def create_lstm_model(self, input_shape):
        inputs = Input(shape=input_shape)

        # Apply feature-wise attention using pre-defined feature weights
        attention_weights = tf.constant(self.feature_weight_list, dtype=tf.float32)
        attention_weights = tf.reshape(attention_weights, (1, 1, -1))
        weighted_inputs = tf.keras.layers.Multiply()([inputs, attention_weights])
        
        # Deep LSTM architecture with batch normalization and dropout
        x = LSTM(256, return_sequences=True, activation='relu')(weighted_inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        # Compile model with Huber loss for robustness to outliers
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model
    
    def create_cnn_model(self, input_shape):
        """
        Create a CNN-based deep learning model with attention mechanism.
        """
        inputs = Input(shape=input_shape)

        # Apply feature-wise attention using pre-defined feature weights
        attention_weights = tf.constant(self.feature_weight_list, dtype=tf.float32)
        attention_weights = tf.reshape(attention_weights, (1, 1, -1))
        weighted_inputs = tf.keras.layers.Multiply()([inputs, attention_weights])
        
        # Deep CNN architecture with batch normalization and dropout
        x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(weighted_inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        # Compile model with Huber loss for robustness to outliers
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model
    
    def train_deep_model(self, model, features_train, features_val, y_train, y_val, 
                        model_name, epochs=200, batch_size=16):
        """
        Train a deep learning model with early stopping and learning rate reduction.
        """
        # Define callbacks for training optimization
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4  
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=6,
                min_lr=1e-6,
                min_delta=1e-5
            )
        ]
        
        # Train the model
        history = model.fit(
            features_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(features_val, y_val),
            callbacks=callbacks,
            shuffle=True 
        )
        
        # Calculate and print performance metrics
        predictions = model.predict(features_val, verbose=0).flatten()
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2)) * 1000
        mae = np.mean(np.abs(predictions - y_val)) * 1000
        r2 = 1 - np.sum((y_val - predictions) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
        
        print(f'{model_name} RMSE: {rmse:.2f}')
        
        # Store history for visualization
        base_model_name = model_name.split('_fold_')[0]
        fold_num = int(model_name.split('_fold_')[1]) if '_fold_' in model_name else 0
        
        # Store the model's history
        self.training_history[base_model_name][f'fold_{fold_num}'] = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        # Store fold metrics
        self.cv_results[base_model_name][f'fold_{fold_num}'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2 * 100  # Convert to percentage
        }
        
        return model, predictions, y_val
    
    def train_xgboost_model(self, X_train, X_val, y_train, y_val):
        """
        Train an XGBoost model with grid search for hyperparameter optimization.
        """
        # Define parameter grid for XGBoost
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [15, 20, 25],
            'learning_rate': [0.001, 0.01, 0.05],
            'colsample_bytree': [0.7, 0.8],
            'subsample': [0.8],
            'min_child_weight': [2, 3, 5],
            'gamma': [0.1, 0.2],
            'reg_alpha': [0.1, 0.2, 0.5],
            'reg_lambda': [1.0, 2.0]
        }
        
        best_model = None
        best_rmse = float('inf')
        best_params = {}
        
        print(f"Starting XGBoost grid search...")
        
        # Nested loops for grid search
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        for subsample in param_grid['subsample']:
                            # Use a subset of parameters for the inner loops to reduce computational load
                            for min_child_weight in param_grid['min_child_weight']:
                                for gamma in param_grid['gamma']:
                                    for reg_alpha in param_grid['reg_alpha']:
                                        for reg_lambda in param_grid['reg_lambda']:
                                            # Current parameter set
                                            params = {
                                                'n_estimators': n_estimators,
                                                'max_depth': max_depth,
                                                'learning_rate': learning_rate,
                                                'colsample_bytree': colsample_bytree,
                                                'subsample': subsample,
                                                'min_child_weight': min_child_weight,
                                                'gamma': gamma,
                                                'reg_alpha': reg_alpha,
                                                'reg_lambda': reg_lambda,
                                                'early_stopping_rounds': 20,
                                                'eval_metric': ['rmse', 'mae']
                                            }
                                            
                                            # Train model with current parameters
                                            model = xgb.XGBRegressor(**params)
                                            model.fit(
                                                X_train, y_train,
                                                eval_set=[(X_val, y_val)],
                                                verbose=0,
                                                feature_weights=self.feature_weight_list
                                            )
                                            
                                            # Evaluate on validation set
                                            predictions = model.predict(X_val)
                                            rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
                                            
                                            # Update best model if this one is better
                                            if rmse < best_rmse:
                                                best_rmse = rmse
                                                best_model = model
                                                best_params = params
                                                print(f"New best XGBoost model: RMSE={best_rmse:.6f}")
                                                print(f"Parameters: {best_params}")
        
        print(f"XGBoost grid search complete. Best RMSE: {best_rmse:.6f}")
        print(f"Best parameters: {best_params}")
        
        # Create and save feature importance plot for the best model
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(best_model, importance_type='weight', max_num_features=20)
        plt.title("XGBoost Feature Importance (Grid Search)")
        plt.tight_layout()
        plt.savefig("Importance_plots/xgboost_grid_search_feature_importance.png")
        plt.close()
        
        return best_model

    
    def train_lightgbm_model(self, X_train, X_val, y_train, y_val):
        """
        Train a LightGBM model with grid search for hyperparameter optimization.
        """
        # Define parameter grid for LightGBM
        param_grid = {
            'n_estimators': [500, 800],
            'max_depth': [15, 20, 25],
            'learning_rate': [0.001, 0.005, 0.01],
            'num_leaves': [10, 20, 35],
            'feature_fraction': [0.8],
            'bagging_fraction': [0.8],
            'bagging_freq': [3, 5],
            'min_child_samples': [15, 25, 35],
            'reg_alpha': [0.4, 0.6],
            'reg_lambda': [1.0, 1.5]
        }
        
        best_model = None
        best_rmse = float('inf')
        best_params = {}
        
        print(f"Starting LightGBM grid search...")
        
        # Nested loops for grid search
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                for learning_rate in param_grid['learning_rate']:
                    for num_leaves in param_grid['num_leaves']:
                        for feature_fraction in param_grid['feature_fraction']:
                            # Use a subset of parameters for the inner loops to reduce computational load
                            for bagging_fraction in param_grid['bagging_fraction']:
                                for bagging_freq in param_grid['bagging_freq']:
                                    for min_child_samples in param_grid['min_child_samples']:
                                        for reg_alpha in param_grid['reg_alpha']:
                                            for reg_lambda in param_grid['reg_lambda']:
                                                # Current parameter set
                                                params = {
                                                    'n_estimators': n_estimators,
                                                    'max_depth': max_depth,
                                                    'learning_rate': learning_rate,
                                                    'num_leaves': num_leaves,
                                                    'feature_fraction': feature_fraction,
                                                    'bagging_fraction': bagging_fraction,
                                                    'bagging_freq': bagging_freq,
                                                    'min_child_samples': min_child_samples,
                                                    'reg_alpha': reg_alpha,
                                                    'reg_lambda': reg_lambda,
                                                    'feature_contrib': self.feature_weight_list
                                                }
                                                
                                                # Train model with current parameters
                                                model = lgb.LGBMRegressor(**params)
                                                model.fit(
                                                    X_train, y_train,
                                                    eval_set=[(X_val, y_val)],
                                                    callbacks=[lgb.early_stopping(20, verbose=0)]
                                                )
                                                
                                                # Evaluate on validation set
                                                predictions = model.predict(X_val)
                                                rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
                                                
                                                # Update best model if this one is better
                                                if rmse < best_rmse:
                                                    best_rmse = rmse
                                                    best_model = model
                                                    best_params = params
                                                    print(f"New best LightGBM model: RMSE={best_rmse:.6f}")
                                                    print(f"Parameters: {best_params}")
        
        print(f"LightGBM grid search complete. Best RMSE: {best_rmse:.6f}")
        print(f"Best parameters: {best_params}")
        
        # Create and save feature importance plot for the best model
        plt.figure(figsize=(10, 6))
        lgb.plot_importance(best_model, importance_type='split', max_num_features=20)
        plt.title("LightGBM Feature Importance (Grid Search)")
        plt.tight_layout()
        plt.savefig("Importance_plots/lightgbm_grid_search_feature_importance.png")
        plt.close()
        
        return best_model
    
    def evaluate_on_test_set(self, test_data, model_name='best'):
        """
        Evaluate a trained model on the test set.
        
        Args:
            test_data: DataFrame containing test set (features and target)
            model_name: Which model to evaluate ('xgboost_lstm', 'lightgbm_cnn', etc.)
                       or 'best' to use the best performing model
        
        Returns:
            Dictionary with evaluation metrics
        """
        if model_name == 'best':
            # Find best performing model based on validation RMSE
            best_model_name = None
            best_rmse = float('inf')
            
            for name, results in self.cv_results.items():
                if not results:
                    continue
                    
                avg_rmse = np.mean([fold['rmse'] for fold in results.values()])
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_model_name = name
                    
            model_name = best_model_name
            
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' has not been trained yet")
            
        # Get feature extractor and predictor
        feature_extractor = self.best_models[model_name]['feature_extractor']
        predictor = self.best_models[model_name]['predictor']
        
        # Prepare test data
        X_test = test_data.drop(columns=["score"]).values
        y_test = test_data["score"].values
        
        # Generate features using feature extractor
        features_test = feature_extractor.predict(X_test).reshape(-1, 1, 1)
        
        # Get predictions
        predictions = predictor.predict(features_test, verbose=0).flatten()
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2)) * 1000
        mae = np.mean(np.abs(predictions - y_test)) * 1000
        r2 = 1 - np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
        
        return {
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2 * 100,  # Convert to percentage
            'predictions': predictions,
            'true_values': y_test
        }
    
    def xgboost_lstm(self):
        return self._train_hybrid_model('xgboost_lstm', self.train_xgboost_model, self.create_lstm_model)
    
    def lightgbm_lstm(self):
        return self._train_hybrid_model('lightgbm_lstm', self.train_lightgbm_model, self.create_lstm_model) 
    
    def xgboost_cnn(self):
        return self._train_hybrid_model('xgboost_cnn', self.train_xgboost_model, self.create_cnn_model) 
    
    def lightgbm_cnn(self):
        return self._train_hybrid_model('lightgbm_cnn', self.train_lightgbm_model, self.create_cnn_model)  
    
    def _train_hybrid_model(self, name, tree_model_func, deep_model_func):
        """
        Internal method to train a hybrid model using stratified k-fold cross validation.
        
        Args:
            name: Name of the hybrid model (e.g., 'xgboost_lstm')
            tree_model_func: Function to train the tree-based model
            deep_model_func: Function to create the deep learning model
            
        Returns:
            dict containing:
                - predictions: Concatenated predictions from all folds
                - true_values: Concatenated true values from all folds
                - feature_extractor: Best performing tree model
                - predictor: Best performing deep model
        """
        fold_predictions = []
        fold_true_values = []
        best_score = float('inf')
        best_models = None
        
        if name not in self.training_history:
            self.training_history[name] = {}
    
        if name not in self.cv_results:
            self.cv_results[name] = {}
        
        # Get stratified folds using only training data
        folds = self.prepare_stratified_folds()
        
        # Perform stratified k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(folds):
            # Split data for current fold
            X_train_fold = self.X[train_idx]
            X_val_fold = self.X[val_idx]
            y_train_fold = self.y[train_idx]
            y_val_fold = self.y[val_idx]
            
            # Train tree-based feature extractor
            tree_model = tree_model_func(X_train_fold, X_val_fold, y_train_fold, y_val_fold)
            
            # Generate features for deep learning model
            features_train = tree_model.predict(X_train_fold).reshape(-1, 1, 1)
            features_val = tree_model.predict(X_val_fold).reshape(-1, 1, 1)
            
            # Train deep learning predictor
            deep_model = deep_model_func((1, 1))
            deep_model, predictions, true_values = self.train_deep_model(
                deep_model,
                features_train,
                features_val,
                y_train_fold,
                y_val_fold,
                f'{name}_fold_{fold+1}'
            )
            
            # Track best performing models
            fold_score = np.sqrt(np.mean((predictions - true_values) ** 2))
            if fold_score < best_score:
                best_score = fold_score
                best_models = {
                    'feature_extractor': tree_model,
                    'predictor': deep_model
                }
            
            fold_predictions.append(predictions)
            fold_true_values.append(true_values)
        
        # Store best models for later use
        self.best_models[name] = best_models
        
        return {
            'predictions': np.concatenate(fold_predictions),
            'true_values': np.concatenate(fold_true_values),
            'feature_extractor': best_models['feature_extractor'],
            'predictor': best_models['predictor']
        }
        
    def get_training_history(self):
        """
        Get the training history for visualization.
        
        Returns:
            Dictionary containing training history for each model
        """
        # Process the training history to make it suitable for plotting
        processed_history = {}
        
        for model_name, fold_histories in self.training_history.items():
            # Average the loss across all folds for each epoch
            if not fold_histories:
                processed_history[model_name] = None
                continue
                
            # Find the maximum number of epochs across all folds
            max_epochs = max(len(history['loss']) for history in fold_histories.values())
            
            # Initialize arrays for averaging
            avg_loss = np.zeros(max_epochs)
            avg_val_loss = np.zeros(max_epochs)
            counts = np.zeros(max_epochs)
            
            # Sum up losses for each epoch
            for fold_history in fold_histories.values():
                for epoch, (loss, val_loss) in enumerate(zip(fold_history['loss'], fold_history['val_loss'])):
                    avg_loss[epoch] += loss
                    avg_val_loss[epoch] += val_loss
                    counts[epoch] += 1
            
            # Calculate averages
            for epoch in range(max_epochs):
                if counts[epoch] > 0:
                    avg_loss[epoch] /= counts[epoch]
                    avg_val_loss[epoch] /= counts[epoch]
            
            # Store averaged history
            processed_history[model_name] = {
                'loss': avg_loss.tolist(),
                'val_loss': avg_val_loss.tolist()
            }
        
        return processed_history

    def get_cv_results(self):
        """
        Get the cross-validation results for visualization.
        
        Returns:
            Dictionary containing CV results for each model
        """
        # Calculate average metrics across all folds for each model
        avg_results = {}
        
        for model_name, fold_results in self.cv_results.items():
            if not fold_results:
                avg_results[model_name] = None
                continue
                
            avg_rmse = 0
            avg_mae = 0
            avg_r2 = 0
            fold_count = 0
            
            for fold_data in fold_results.values():
                avg_rmse += fold_data['rmse']
                avg_mae += fold_data['mae']
                avg_r2 += fold_data['r2']
                fold_count += 1
            
            if fold_count > 0:
                avg_results[model_name] = {
                    'rmse': avg_rmse / fold_count,
                    'mae': avg_mae / fold_count,
                    'r2': avg_r2 / fold_count,
                    'fold_results': fold_results
                }
        
        return avg_results