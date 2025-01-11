# models.py
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
import tensorflow as tf
from keras.layers import Dense, LSTM, Conv1D, Flatten, Input, Dropout, BatchNormalization # type: ignore
from keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.optimizers import Adam # type: ignore

class HybridModels:
    def __init__(self, data, n_splits=5):
        """
        Initialize the hybrid models with data and configuration.
        
        Args:
            data: DataFrame containing GPU specifications and performance scores
            n_splits: Number of folds for k-fold cross-validation (default: 5)
        """
        self.data = data
        self.X = data.drop(columns=["score"]).values
        self.y = data["score"].values
        self.n_splits = n_splits
        self.feature_names = data.drop(columns=["score"]).columns.tolist()
        
        # Create stratification bins for performance scores
        self.stratifier = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        self.y_binned = self.stratifier.fit_transform(self.y.reshape(-1, 1)).ravel()
        
        # Initialize stratified k-fold
        self.kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=32)
        
        # Define importance weights for key GPU features
        self.feature_weights = {
            'memSize': 2.3,
            'gpuClock': 2.4,
            'unifiedShader': 1.8,
            'gpuChip': 1.6,
            'memBusWidth': 1.5,
            'memClock': 1.5
        }
        
        # Convert feature weights to list matching feature order
        self.feature_weight_list = [self.feature_weights.get(feat, 1.0) 
                                  for feat in self.feature_names]
    
    def create_lstm_model(self, input_shape):
        """
        Create an LSTM-based deep learning model with attention mechanism.
        """
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
                min_delta=1e-5
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=6,
                min_lr=1e-7,
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
            callbacks=callbacks
        )
        
        # Calculate and print performance metrics
        predictions = model.predict(features_val, verbose=0).flatten()
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2)) * 1000
        print(f'{model_name} RMSE: {rmse:.2f}')
        
        return model, predictions, y_val
    
    def train_xgboost_model(self, X_train, X_val, y_train, y_val):
        """
        Train an XGBoost model with optimized hyperparameters.
        """
        params = {
            'n_estimators': 1000,
            'max_depth': 20,
            'learning_rate': 0.001,
            'colsample_bytree': 0.8,
            'subsample': 0.9,
            'min_child_weight': 3,
            'gamma': 0.2,
            'reg_alpha': 0.2,
            'reg_lambda': 1,
            'early_stopping_rounds': 40,
            'eval_metric': ['rmse', 'mae']
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=0,
            feature_weights=self.feature_weight_list
        )
        return model
    
    def train_lightgbm_model(self, X_train, X_val, y_train, y_val):
        """
        Train a LightGBM model with optimized hyperparameters.
        """
        params = {
            'n_estimators': 1000,
            'max_depth': 20,
            'learning_rate': 0.005,
            'num_leaves': 35,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 3,
            'min_child_samples': 15,
            'reg_alpha': 0.2,
            'reg_lambda': 1,
            'feature_contrib': self.feature_weight_list
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(40, verbose=0)]
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
        """
        Internal method to train a hybrid model using stratified k-fold cross validation.
        
        Args:
            name: Name of the tree-based model ('xgboost' or 'lightgbm')
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
        
        # Perform stratified k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.X, self.y_binned)):
            # Split data for current fold
            X_train = self.X[train_idx]
            X_val = self.X[val_idx]
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            # Train tree-based feature extractor
            tree_model = tree_model_func(X_train, X_val, y_train, y_val)
            
            # Generate features for deep learning model
            features_train = tree_model.predict(X_train).reshape(-1, 1, 1)
            features_val = tree_model.predict(X_val).reshape(-1, 1, 1)
            
            # Train deep learning predictor
            deep_model = deep_model_func((1, 1))
            deep_model, predictions, true_values = self.train_deep_model(
                deep_model,
                features_train,
                features_val,
                y_train,
                y_val,
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
        
        return {
            'predictions': np.concatenate(fold_predictions),
            'true_values': np.concatenate(fold_true_values),
            'feature_extractor': best_models['feature_extractor'],
            'predictor': best_models['predictor']
        }