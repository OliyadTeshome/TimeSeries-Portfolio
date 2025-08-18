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
from .lstm_model import LSTMForecaster, create_lstm_model
from .forecasting import ForecastingEngine, run_complete_forecasting
from .portfolio_optimization import PortfolioOptimizer, run_portfolio_optimization
from .backtesting import Backtester, run_complete_backtesting

# Import enhanced framework modules (Phase 1)
from .config import (
    ConfigManager, ModelConfig, DataConfig, PortfolioConfig, 
    BacktestingConfig, SystemConfig, get_config, get_model_config,
    get_data_config, get_portfolio_config, get_backtesting_config,
    get_system_config
)
from .base import (
    BaseModel, BaseDataProcessor, BasePortfolioOptimizer, BaseBacktester,
    ModelRegistry, model_registry, ModelType, OptimizationMethod, StrategyType,
    ModelResult, PerformanceMetrics, PortfolioResult, BacktestResult
)
from .exceptions import (
    TimeSeriesPortfolioError, DataError, DataValidationError, DataNotFoundError,
    ModelError, ModelTrainingError, ModelPredictionError, ModelNotTrainedError,
    PortfolioError, OptimizationError, BacktestingError, ConfigurationError,
    ValidationError, ResourceError, PerformanceError, handle_exception
)
from .validation import (
    DataValidator, ValidationRule, ValidationResult, ValidationReport,
    ValidationSeverity, ValidationStatus, validate_financial_data
)
from .logging_config import (
    setup_logging, log_function_call, log_performance, 
    PerformanceLogger, ContextLogger, performance_logger, context_logger
)

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
    'LSTMForecaster',
    'create_lstm_model',
    
    # Forecasting engine
    'ForecastingEngine',
    'run_complete_forecasting',
    
    # Portfolio optimization
    'PortfolioOptimizer',
    'run_portfolio_optimization',
    
    # Backtesting
    'Backtester',
    'run_complete_backtesting',
    
    # Enhanced framework modules (Phase 1)
    # Configuration
    'ConfigManager', 'ModelConfig', 'DataConfig', 'PortfolioConfig',
    'BacktestingConfig', 'SystemConfig', 'get_config', 'get_model_config',
    'get_data_config', 'get_portfolio_config', 'get_backtesting_config',
    'get_system_config',
    
    # Base classes and interfaces
    'BaseModel', 'BaseDataProcessor', 'BasePortfolioOptimizer', 'BaseBacktester',
    'ModelRegistry', 'model_registry', 'ModelType', 'OptimizationMethod', 'StrategyType',
    'ModelResult', 'PerformanceMetrics', 'PortfolioResult', 'BacktestResult',
    
    # Exception handling
    'TimeSeriesPortfolioError', 'DataError', 'DataValidationError', 'DataNotFoundError',
    'ModelError', 'ModelTrainingError', 'ModelPredictionError', 'ModelNotTrainedError',
    'PortfolioError', 'OptimizationError', 'BacktestingError', 'ConfigurationError',
    'ValidationError', 'ResourceError', 'PerformanceError', 'handle_exception',
    
    # Data validation
    'DataValidator', 'ValidationRule', 'ValidationResult', 'ValidationReport',
    'ValidationSeverity', 'ValidationStatus', 'validate_financial_data',
    
    # Enhanced logging
    'setup_logging', 'log_function_call', 'log_performance',
    'PerformanceLogger', 'ContextLogger', 'performance_logger', 'context_logger',
]
