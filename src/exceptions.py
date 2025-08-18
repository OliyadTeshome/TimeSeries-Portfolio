"""
Custom Exceptions Module

This module defines custom exception classes for the TimeSeries-Portfolio
framework, providing specific error types for different failure scenarios.
"""

from typing import Optional, Any, Dict


class TimeSeriesPortfolioError(Exception):
    """Base exception class for TimeSeries-Portfolio framework."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception.
        
        Args:
            message (str): Error message
            error_code (str, optional): Error code for categorization
            details (dict, optional): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataError(TimeSeriesPortfolioError):
    """Exception raised for data-related errors."""
    
    def __init__(self, message: str, data_source: Optional[str] = None, 
                 data_type: Optional[str] = None, **kwargs):
        """
        Initialize data error.
        
        Args:
            message (str): Error message
            data_source (str, optional): Source of the data
            data_type (str, optional): Type of data
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="DATA_ERROR", **kwargs)
        self.data_source = data_source
        self.data_type = data_type


class DataValidationError(DataError):
    """Exception raised for data validation failures."""
    
    def __init__(self, message: str, validation_rule: Optional[str] = None, 
                 invalid_data: Optional[Any] = None, **kwargs):
        """
        Initialize data validation error.
        
        Args:
            message (str): Error message
            validation_rule (str, optional): Rule that failed validation
            invalid_data (Any, optional): Data that failed validation
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="DATA_VALIDATION_ERROR", **kwargs)
        self.validation_rule = validation_rule
        self.invalid_data = invalid_data


class DataNotFoundError(DataError):
    """Exception raised when data cannot be found."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 data_id: Optional[str] = None, **kwargs):
        """
        Initialize data not found error.
        
        Args:
            message (str): Error message
            file_path (str, optional): Path to the missing file
            data_id (str, optional): ID of the missing data
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="DATA_NOT_FOUND", **kwargs)
        self.file_path = file_path
        self.data_id = data_id


class ModelError(TimeSeriesPortfolioError):
    """Exception raised for model-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_type: Optional[str] = None, **kwargs):
        """
        Initialize model error.
        
        Args:
            message (str): Error message
            model_name (str, optional): Name of the model
            model_type (str, optional): Type of the model
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_name = model_name
        self.model_type = model_type


class ModelTrainingError(ModelError):
    """Exception raised for model training failures."""
    
    def __init__(self, message: str, training_data_info: Optional[Dict[str, Any]] = None, 
                 hyperparameters: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize model training error.
        
        Args:
            message (str): Error message
            training_data_info (dict, optional): Information about training data
            hyperparameters (dict, optional): Model hyperparameters
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="MODEL_TRAINING_ERROR", **kwargs)
        self.training_data_info = training_data_info
        self.hyperparameters = hyperparameters


class ModelPredictionError(ModelError):
    """Exception raised for model prediction failures."""
    
    def __init__(self, message: str, input_data_info: Optional[Dict[str, Any]] = None, 
                 prediction_params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize model prediction error.
        
        Args:
            message (str): Error message
            input_data_info (dict, optional): Information about input data
            prediction_params (dict, optional): Prediction parameters
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="MODEL_PREDICTION_ERROR", **kwargs)
        self.input_data_info = input_data_info
        self.prediction_params = prediction_params


class ModelNotTrainedError(ModelError):
    """Exception raised when trying to use an untrained model."""
    
    def __init__(self, message: str = "Model has not been trained yet", 
                 model_name: Optional[str] = None, **kwargs):
        """
        Initialize model not trained error.
        
        Args:
            message (str): Error message
            model_name (str, optional): Name of the untrained model
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="MODEL_NOT_TRAINED", **kwargs)
        self.model_name = model_name


class PortfolioError(TimeSeriesPortfolioError):
    """Exception raised for portfolio-related errors."""
    
    def __init__(self, message: str, portfolio_name: Optional[str] = None, 
                 optimization_method: Optional[str] = None, **kwargs):
        """
        Initialize portfolio error.
        
        Args:
            message (str): Error message
            portfolio_name (str, optional): Name of the portfolio
            optimization_method (str, optional): Optimization method used
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="PORTFOLIO_ERROR", **kwargs)
        self.portfolio_name = portfolio_name
        self.optimization_method = optimization_method


class OptimizationError(PortfolioError):
    """Exception raised for portfolio optimization failures."""
    
    def __init__(self, message: str, constraints: Optional[Dict[str, Any]] = None, 
                 objective_function: Optional[str] = None, **kwargs):
        """
        Initialize optimization error.
        
        Args:
            message (str): Error message
            constraints (dict, optional): Optimization constraints
            objective_function (str, optional): Objective function used
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="OPTIMIZATION_ERROR", **kwargs)
        self.constraints = constraints
        self.objective_function = objective_function


class BacktestingError(TimeSeriesPortfolioError):
    """Exception raised for backtesting-related errors."""
    
    def __init__(self, message: str, strategy_name: Optional[str] = None, 
                 backtest_period: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize backtesting error.
        
        Args:
            message (str): Error message
            strategy_name (str, optional): Name of the strategy
            backtest_period (dict, optional): Backtesting period information
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="BACKTESTING_ERROR", **kwargs)
        self.strategy_name = strategy_name
        self.backtest_period = backtest_period


class ConfigurationError(TimeSeriesPortfolioError):
    """Exception raised for configuration-related errors."""
    
    def __init__(self, message: str, config_section: Optional[str] = None, 
                 config_key: Optional[str] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message (str): Error message
            config_section (str, optional): Configuration section
            config_key (str, optional): Configuration key
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="CONFIGURATION_ERROR", **kwargs)
        self.config_section = config_section
        self.config_key = config_key


class ValidationError(TimeSeriesPortfolioError):
    """Exception raised for general validation failures."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, 
                 expected_value: Optional[Any] = None, actual_value: Optional[Any] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message (str): Error message
            field_name (str, optional): Name of the field that failed validation
            expected_value (Any, optional): Expected value
            actual_value (Any, optional): Actual value
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field_name = field_name
        self.expected_value = expected_value
        self.actual_value = actual_value


class ResourceError(TimeSeriesPortfolioError):
    """Exception raised for resource-related errors."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, 
                 resource_path: Optional[str] = None, **kwargs):
        """
        Initialize resource error.
        
        Args:
            message (str): Error message
            resource_type (str, optional): Type of resource
            resource_path (str, optional): Path to the resource
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type
        self.resource_path = resource_path


class PerformanceError(TimeSeriesPortfolioError):
    """Exception raised for performance-related errors."""
    
    def __init__(self, message: str, performance_metric: Optional[str] = None, 
                 threshold: Optional[float] = None, actual_value: Optional[float] = None, **kwargs):
        """
        Initialize performance error.
        
        Args:
            message (str): Error message
            performance_metric (str, optional): Performance metric that failed
            threshold (float, optional): Expected threshold
            actual_value (float, optional): Actual value
            **kwargs: Additional error details
        """
        super().__init__(message, error_code="PERFORMANCE_ERROR", **kwargs)
        self.performance_metric = performance_metric
        self.threshold = threshold
        self.actual_value = actual_value


def handle_exception(func):
    """
    Decorator for handling exceptions and providing consistent error handling.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with exception handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TimeSeriesPortfolioError:
            # Re-raise custom exceptions as-is
            raise
        except Exception as e:
            # Convert generic exceptions to framework exceptions
            raise TimeSeriesPortfolioError(
                message=f"Unexpected error in {func.__name__}: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={'original_exception': str(e), 'function_name': func.__name__}
            )
    return wrapper
