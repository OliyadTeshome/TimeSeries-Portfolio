"""
TimeSeries-Portfolio Package

A comprehensive framework for time series analysis and portfolio optimization.
"""

__version__ = "1.0.0"
__author__ = "TimeSeries-Portfolio Team"
__description__ = "Time series analysis and portfolio optimization framework"

# Import main modules
from .data_loader import DataLoader, load_sample_data
from .preprocessing import TimeSeriesPreprocessor, preprocess_pipeline
from .eda import TimeSeriesEDA, run_complete_eda
from .arima_model import ARIMAModel, auto_arima
from .lstm_model import LSTMModel, auto_lstm
from .forecasting import ForecastingEngine, run_complete_forecasting
from .portfolio_optimization import PortfolioOptimizer, run_portfolio_optimization
from .backtesting import Backtester, run_complete_backtesting

__all__ = [
    # Data management
    'DataLoader',
    'load_sample_data',
    
    # Preprocessing
    'TimeSeriesPreprocessor',
    'preprocess_pipeline',
    
    # Exploratory data analysis
    'TimeSeriesEDA',
    'run_complete_eda',
    
    # Forecasting models
    'ARIMAModel',
    'auto_arima',
    'LSTMModel',
    'auto_lstm',
    
    # Forecasting engine
    'ForecastingEngine',
    'run_complete_forecasting',
    
    # Portfolio optimization
    'PortfolioOptimizer',
    'run_portfolio_optimization',
    
    # Backtesting
    'Backtester',
    'run_complete_backtesting',
]
