"""
LSTM Model Module

This module provides implementation of Long Short-Term Memory (LSTM) neural networks
for time series forecasting with automatic hyperparameter tuning and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
import logging
import os

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    A comprehensive LSTM model implementation for time series forecasting.
    """
    
    def __init__(self, data: pd.Series, sequence_length: int = 10):
        """
        Initialize the LSTM model.
        
        Args:
            data (pd.Series): Time series data
            sequence_length (int): Length of input sequences
        """
        self.data = data
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        self.history = None
        self.scaled_data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.train_predictions = None
        self.val_predictions = None
        self.test_predictions = None
        self.forecast = None
        
        # Validate data
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        if len(data) < sequence_length + 10:
            raise ValueError(f"Data must have at least {sequence_length + 10} observations")
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def prepare_data(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
                    test_ratio: float = 0.15) -> None:
        """
        Prepare data for LSTM training by scaling and creating sequences.
        
        Args:
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            test_ratio (float): Proportion of data for testing
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(self.scaled_data)
        
        # Split data
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        self.X_train = X[:train_end]
        self.y_train = y[:train_end]
        self.X_val = X[train_end:val_end]
        self.y_val = y[train_end:val_end]
        self.X_test = X[val_end:]
        self.y_test = y[val_end:]
        
        logger.info(f"Data prepared: Train={len(self.X_train)}, Val={len(self.X_val)}, Test={len(self.X_test)}")
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data (np.ndarray): Scaled data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences and target values
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), 0])
            y.append(data[i + self.sequence_length, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, lstm_units: List[int] = [50, 50], 
                   dropout_rate: float = 0.2, learning_rate: float = 0.001) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            lstm_units (List[int]): Number of units in each LSTM layer
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
        """
        self.model = Sequential()
        
        # Add LSTM layers
        for i, units in enumerate(lstm_units):
            if i == 0:
                self.model.add(LSTM(units, return_sequences=(i < len(lstm_units) - 1),
                                   input_shape=(self.sequence_length, 1)))
            else:
                self.model.add(LSTM(units, return_sequences=(i < len(lstm_units) - 1)))
            
            # Add dropout and batch normalization
            self.model.add(Dropout(dropout_rate))
            if i < len(lstm_units) - 1:  # Don't add batch norm after last LSTM layer
                self.model.add(BatchNormalization())
        
        # Add output layer
        self.model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        logger.info("LSTM model built successfully")
        logger.info(f"Model summary:\n{self.model.summary()}")
    
    def train_model(self, epochs: int = 100, batch_size: int = 32,
                   patience: int = 20, verbose: int = 1) -> None:
        """
        Train the LSTM model.
        
        Args:
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Patience for early stopping
            verbose (int): Verbosity level
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        if self.X_train is None:
            raise ValueError("Data must be prepared before training")
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train model
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("LSTM model training completed")
    
    def make_predictions(self) -> None:
        """
        Make predictions on train, validation, and test sets.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Make predictions
        self.train_predictions = self.model.predict(self.X_train)
        self.val_predictions = self.model.predict(self.X_val)
        self.test_predictions = self.model.predict(self.X_test)
        
        # Inverse transform predictions
        self.train_predictions = self.scaler.inverse_transform(self.train_predictions)
        self.val_predictions = self.scaler.inverse_transform(self.val_predictions)
        self.test_predictions = self.scaler.inverse_transform(self.test_predictions)
        
        # Inverse transform actual values
        self.y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        self.y_val_actual = self.scaler.inverse_transform(self.y_val.reshape(-1, 1))
        self.y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        logger.info("Predictions generated for all datasets")
    
    def evaluate_model(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model performance on all datasets.
        
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics for each dataset
        """
        if self.train_predictions is None:
            raise ValueError("Predictions must be generated before evaluation")
        
        metrics = {}
        
        # Evaluate on training data
        train_metrics = self._calculate_metrics(self.y_train_actual, self.train_predictions)
        metrics['train'] = train_metrics
        
        # Evaluate on validation data
        val_metrics = self._calculate_metrics(self.y_val_actual, self.val_predictions)
        metrics['validation'] = val_metrics
        
        # Evaluate on test data
        test_metrics = self._calculate_metrics(self.y_test_actual, self.test_predictions)
        metrics['test'] = test_metrics
        
        # Log results
        logger.info("Model evaluation completed:")
        for dataset, dataset_metrics in metrics.items():
            logger.info(f"{dataset.capitalize()} - RMSE: {dataset_metrics['RMSE']:.4f}, MAE: {dataset_metrics['MAE']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def forecast_future(self, steps: int) -> np.ndarray:
        """
        Generate future forecasts.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Get the last sequence
        last_sequence = self.scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform forecasts
        self.forecast = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))
        
        logger.info(f"Generated {steps} step forecast")
        return self.forecast
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if self.history is None:
            raise ValueError("Model must be trained before plotting history")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MAE
            ax2.plot(self.history.history['mae'], label='Training MAE')
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create training history plot: {str(e)}")
    
    def plot_predictions(self, save_path: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if self.train_predictions is None:
            raise ValueError("Predictions must be generated before plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            # Create date indices for plotting
            train_dates = self.data.index[self.sequence_length:self.sequence_length + len(self.train_predictions)]
            val_dates = self.data.index[self.sequence_length + len(self.train_predictions):
                                      self.sequence_length + len(self.train_predictions) + len(self.val_predictions)]
            test_dates = self.data.index[self.sequence_length + len(self.train_predictions) + len(self.val_predictions):
                                       self.sequence_length + len(self.train_predictions) + len(self.val_predictions) + len(self.test_predictions)]
            
            plt.figure(figsize=(15, 8))
            
            # Plot original data
            plt.plot(self.data.index, self.data.values, label='Original Data', alpha=0.7, linewidth=1)
            
            # Plot predictions
            plt.plot(train_dates, self.train_predictions.flatten(), label='Train Predictions', linewidth=2)
            plt.plot(val_dates, self.val_predictions.flatten(), label='Validation Predictions', linewidth=2)
            plt.plot(test_dates, self.test_predictions.flatten(), label='Test Predictions', linewidth=2)
            
            plt.title('LSTM Model Predictions')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Predictions plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create predictions plot: {str(e)}")
    
    def plot_forecast(self, save_path: Optional[str] = None) -> None:
        """
        Plot the original data and future forecast.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if self.forecast is None:
            raise ValueError("Forecast must be generated before plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            # Create future dates
            last_date = self.data.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=len(self.forecast), freq='D')
            
            plt.figure(figsize=(15, 8))
            
            # Plot original data
            plt.plot(self.data.index, self.data.values, label='Original Data', linewidth=2)
            
            # Plot forecast
            plt.plot(future_dates, self.forecast.flatten(), 
                    label='LSTM Forecast', linewidth=2, linestyle='--', color='red')
            
            plt.title('LSTM Model Future Forecast')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create forecast plot: {str(e)}")
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        try:
            # Save Keras model
            model_path = filepath.replace('.pkl', '_model.h5')
            self.model.save(model_path)
            
            # Save scaler and other components
            import pickle
            model_data = {
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'history': self.history,
                'forecast': self.forecast
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath} and {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, data: pd.Series) -> 'LSTMModel':
        """
        Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model
            data (pd.Series): Original data series
            
        Returns:
            LSTMModel: Loaded model instance
        """
        try:
            import pickle
            
            # Load Keras model
            model_path = filepath.replace('.pkl', '_model.h5')
            keras_model = load_model(model_path)
            
            # Load other components
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            instance = cls(data, model_data['sequence_length'])
            instance.model = keras_model
            instance.scaler = model_data['scaler']
            instance.history = model_data['history']
            instance.forecast = model_data['forecast']
            
            logger.info(f"Model loaded from {filepath} and {model_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def auto_lstm(data: pd.Series, sequence_length: int = 10, 
               lstm_units: List[int] = [50, 50], epochs: int = 100,
               **kwargs) -> LSTMModel:
    """
    Convenience function to automatically build, train, and evaluate an LSTM model.
    
    Args:
        data (pd.Series): Time series data
        sequence_length (int): Length of input sequences
        lstm_units (List[int]): Number of units in each LSTM layer
        epochs (int): Maximum number of training epochs
        **kwargs: Additional arguments for LSTM model
        
    Returns:
        LSTMModel: Trained LSTM model
    """
    # Initialize model
    model = LSTMModel(data, sequence_length)
    
    # Prepare data
    model.prepare_data()
    
    # Build model
    model.build_model(lstm_units=lstm_units, **kwargs)
    
    # Train model
    model.train_model(epochs=epochs, **kwargs)
    
    # Make predictions
    model.make_predictions()
    
    # Evaluate model
    metrics = model.evaluate_model()
    
    logger.info("Auto LSTM model completed successfully")
    return model


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    ts_data = sample_data['value']
    
    print("Sample data loaded, training LSTM model...")
    
    # Auto-train LSTM model
    lstm_model = auto_lstm(ts_data, sequence_length=10, epochs=50)
    
    # Generate forecast
    forecast = lstm_model.forecast_future(steps=30)
    
    print("LSTM model training completed!")
    print(f"Forecast generated for {len(forecast)} steps")
    
    # Save model
    lstm_model.save_model('data/processed/lstm_model.pkl')
