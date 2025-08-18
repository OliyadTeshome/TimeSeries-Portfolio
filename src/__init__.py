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
    BaseModel, BaseDataProcessor, BasePortfolioOptimizer, BaseBacktestingStrategy,
    ModelFactory, StrategyFactory, ModelType, OptimizationMethod, StrategyType,
    ModelResult, PerformanceMetrics
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
    setup_logging, get_logger, 
    PerformanceLogger, ErrorTracker, TimerContext
)

# Phase 2: Advanced Features
from .dashboard import FinancialDashboard
from .explainability import ModelExplainer, ModelDiagnostics, create_explainer, create_diagnostics
from .risk_management import RiskManager, create_risk_manager
from .portfolio_analytics import PortfolioAnalytics, create_portfolio_analytics
from .real_time_data import RealTimeDataManager, create_real_time_manager

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
    'BaseModel', 'BaseDataProcessor', 'BasePortfolioOptimizer', 'BaseBacktestingStrategy',
    'ModelFactory', 'StrategyFactory', 'ModelType', 'OptimizationMethod', 'StrategyType',
    'ModelResult', 'PerformanceMetrics',
    
    # Exception handling
    'TimeSeriesPortfolioError', 'DataError', 'DataValidationError', 'DataNotFoundError',
    'ModelError', 'ModelTrainingError', 'ModelPredictionError', 'ModelNotTrainedError',
    'PortfolioError', 'OptimizationError', 'BacktestingError', 'ConfigurationError',
    'ValidationError', 'ResourceError', 'PerformanceError', 'handle_exception',
    
    # Data validation
    'DataValidator', 'ValidationRule', 'ValidationResult', 'ValidationReport',
    'ValidationSeverity', 'ValidationStatus', 'validate_financial_data',
    
    # Enhanced logging
    'setup_logging', 'get_logger',
    'PerformanceLogger', 'ErrorTracker', 'TimerContext',

# Phase 2: Advanced Features
# Interactive Dashboard
'FinancialDashboard',

# Model Explainability
'ModelExplainer', 'ModelDiagnostics', 'create_explainer', 'create_diagnostics',

# Advanced Risk Management
'RiskManager', 'create_risk_manager',

# Enhanced Portfolio Analytics
'PortfolioAnalytics', 'create_portfolio_analytics',

# Real-time Data Integration
'RealTimeDataManager', 'create_real_time_manager',
]
