"""
LSTM Model Module

This module provides implementation of Long Short-Term Memory (LSTM) neural networks
for time series forecasting with automatic hyperparameter tuning and validation.
Updated for PyTorch compatibility with Python 3.13.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import logging
import os
import pickle

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    A comprehensive LSTM model implementation for time series forecasting using PyTorch.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            output_size (int): Number of output features
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def forward(self, x):
        """
        Forward pass through the LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out


class LSTMForecaster:
    """
    A comprehensive LSTM forecaster implementation for time series forecasting.
    """
    
    def __init__(self, data: pd.Series, sequence_length: int = 10):
        """
        Initialize the LSTM forecaster.
        
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate data
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        if len(data) < sequence_length + 10:
            raise ValueError(f"Data must have at least {sequence_length + 10} observations")
        
        logger.info(f"Using device: {self.device}")
    
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
            data (np.ndarray): Scaled time series data
            
        Returns:
            Tuple of (X, y) where X contains sequences and y contains targets
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def build_model(self, hidden_size: int = 50, num_layers: int = 2, 
                   dropout: float = 0.2, learning_rate: float = 0.001) -> None:
        """
        Build the LSTM model.
        
        Args:
            hidden_size (int): Number of hidden units in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            learning_rate (float): Learning rate for optimizer
        """
        self.model = LSTMModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        logger.info(f"Model built: hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}")
    
    def train(self, epochs: int = 100, batch_size: int = 32, patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the LSTM model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
            
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(self.X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(self.y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(self.X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(self.y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor).item()
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.history = history
        logger.info("Training completed")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Input sequences
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        self.test_predictions = self.predict(self.X_test)
        
        # Inverse transform predictions and actual values
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_test_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_test_pred)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test_actual - y_test_pred) / y_test_actual)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
        
        logger.info(f"Test Metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
        return metrics
    
    def forecast_future(self, steps: int) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps (int): Number of steps to forecast
            
        Returns:
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting")
        
        self.model.eval()
        
        # Start with the last sequence from training data
        last_sequence = self.scaled_data[-self.sequence_length:].reshape(1, -1, 1)
        last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
        
        forecast_values = []
        
        with torch.no_grad():
            for _ in range(steps):
                # Make prediction
                prediction = self.model(last_sequence_tensor)
                forecast_values.append(prediction.cpu().numpy()[0, 0])
                
                # Update sequence for next prediction
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = prediction.cpu().numpy()[0, 0]
                last_sequence_tensor = torch.FloatTensor(last_sequence).to(self.device)
        
        # Inverse transform forecast
        forecast_scaled = np.array(forecast_values).reshape(-1, 1)
        self.forecast = self.scaler.inverse_transform(forecast_scaled).flatten()
        
        logger.info(f"Forecast completed for {steps} steps")
        return self.forecast
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state
        torch.save(self.model.state_dict(), filepath)
        
        # Save scaler
        scaler_path = filepath.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to the saved model
        """
        # Load model state
        self.model = LSTMModel()
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        
        # Load scaler
        scaler_path = filepath.replace('.pth', '_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self) -> None:
        """
        Plot the training history.
        """
        if self.history is None:
            logger.warning("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_predictions(self) -> None:
        """
        Plot the predictions vs actual values.
        """
        if self.test_predictions is None:
            logger.warning("No test predictions available")
            return
        
        import matplotlib.pyplot as plt
        
        # Inverse transform for plotting
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        y_test_pred = self.scaler.inverse_transform(self.test_predictions.reshape(-1, 1)).flatten()
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_actual, label='Actual', alpha=0.7)
        plt.plot(y_test_pred, label='Predicted', alpha=0.7)
        plt.title('Test Predictions vs Actual Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_forecast(self) -> None:
        """
        Plot the forecast.
        """
        if self.forecast is None:
            logger.warning("No forecast available")
            return
        
        import matplotlib.pyplot as plt
        
        # Get the last part of the original data for context
        last_data = self.data.tail(50)
        
        plt.figure(figsize=(12, 6))
        plt.plot(last_data.index, last_data.values, label='Historical Data', alpha=0.7)
        
        # Create future index
        future_index = pd.date_range(start=last_data.index[-1], periods=len(self.forecast) + 1, freq='D')[1:]
        plt.plot(future_index, self.forecast, label='Forecast', alpha=0.7, linestyle='--')
        
        plt.title('Time Series Forecast')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def create_lstm_model(data: pd.Series, sequence_length: int = 10, 
                     hidden_size: int = 50, num_layers: int = 2, 
                     dropout: float = 0.2, learning_rate: float = 0.001,
                     epochs: int = 100, batch_size: int = 32) -> LSTMForecaster:
    """
    Factory function to create and train an LSTM model.
    
    Args:
        data (pd.Series): Time series data
        sequence_length (int): Length of input sequences
        hidden_size (int): Number of hidden units in LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
    Returns:
        Trained LSTMForecaster instance
    """
    # Create forecaster
    forecaster = LSTMForecaster(data, sequence_length)
    
    # Prepare data
    forecaster.prepare_data()
    
    # Build and train model
    forecaster.build_model(hidden_size, num_layers, dropout, learning_rate)
    forecaster.train(epochs, batch_size)
    
    return forecaster


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    ts_data = sample_data['value']
    
    print("Sample data loaded, training LSTM model...")
    
    # Auto-train LSTM model
    lstm_model = create_lstm_model(ts_data, sequence_length=10, epochs=50)
    
    # Generate forecast
    forecast = lstm_model.forecast_future(steps=30)
    
    print("LSTM model training completed!")
    print(f"Forecast generated for {len(forecast)} steps")
    
    # Save model
    lstm_model.save_model('data/processed/lstm_model.pth')
