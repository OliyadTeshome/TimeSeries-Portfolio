"""
Base Classes and Abstract Interfaces

This module provides abstract base classes and interfaces for the TimeSeries-Portfolio
framework, implementing design patterns for better modularity and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of supported model types."""
    ARIMA = "arima"
    LSTM = "lstm"
    PROPHET = "prophet"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"


class OptimizationMethod(Enum):
    """Enumeration of portfolio optimization methods."""
    SHARPE_RATIO = "sharpe_ratio"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    MAX_DIVERSIFICATION = "max_diversification"
    BLACK_LITTERMAN = "black_litterman"


class StrategyType(Enum):
    """Enumeration of backtesting strategy types."""
    BUY_HOLD = "buy_hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_TARGETING = "volatility_targeting"
    RISK_PARITY = "risk_parity"


@dataclass
class ModelResult:
    """Standardized result container for model outputs."""
    
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    model_parameters: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the result data."""
        if not isinstance(self.predictions, np.ndarray):
            raise ValueError("Predictions must be a numpy array")
        if len(self.predictions) == 0:
            raise ValueError("Predictions cannot be empty")


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics container."""
    
    mse: float
    mae: float
    mape: float
    rmse: float
    r2_score: Optional[float] = None
    directional_accuracy: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate the metrics data."""
        if self.mse < 0:
            raise ValueError("MSE cannot be negative")
        if self.mae < 0:
            raise ValueError("MAE cannot be negative")
        if self.mape < 0:
            raise ValueError("MAPE cannot be negative")


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.
    
    This class defines the interface that all forecasting models must implement,
    ensuring consistency and enabling polymorphic behavior.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            name (str): Model name/identifier
            config (Dict[str, Any], optional): Model configuration
        """
        self.name = name
        self.config = config or {}
        self.is_trained = False
        self.training_time = None
        self.model_parameters = {}
        
        # Validate configuration
        self._validate_config()
    
    @abstractmethod
    def train(self, data: pd.Series, **kwargs) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            data (pd.Series): Training data
            **kwargs: Additional training parameters
            
        Returns:
            BaseModel: Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> ModelResult:
        """
        Generate predictions for the specified number of steps.
        
        Args:
            steps (int): Number of steps to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            ModelResult: Prediction results
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.Series) -> PerformanceMetrics:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data (pd.Series): Test data for evaluation
            
        Returns:
            PerformanceMetrics: Performance metrics
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        try:
            self._save_implementation(filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to load the model from
        """
        try:
            self._load_implementation(filepath)
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'model_parameters': self.model_parameters,
            'config': self.config
        }
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        # Override in subclasses for specific validation
        pass
    
    @abstractmethod
    def _save_implementation(self, filepath: str) -> None:
        """Implementation-specific model saving."""
        pass
    
    @abstractmethod
    def _load_implementation(self, filepath: str) -> None:
        """Implementation-specific model loading."""
        pass


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processing operations.
    
    This class defines the interface for data preprocessing, feature engineering,
    and data validation operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config (Dict[str, Any], optional): Processing configuration
        """
        self.config = config or {}
        self.feature_names = []
        self.processing_stats = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseDataProcessor':
        """
        Fit the processor on the provided data.
        
        Args:
            data (pd.DataFrame): Training data
            
        Returns:
            BaseDataProcessor: Self for method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted processor.
        
        Args:
            data (pd.DataFrame): Data to transform
            
        Returns:
            pd.DataFrame: Transformed data
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the processor and transform the data in one step.
        
        Args:
            data (pd.DataFrame): Data to process
            
        Returns:
            pd.DataFrame: Processed data
        """
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features created by the processor.
        
        Returns:
            List[str]: Feature names
        """
        return self.feature_names.copy()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the processing operations.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return self.processing_stats.copy()


class BasePortfolioOptimizer(ABC):
    """
    Abstract base class for portfolio optimization strategies.
    
    This class defines the interface for different portfolio optimization
    approaches, enabling easy comparison and strategy selection.
    """
    
    def __init__(self, returns: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio optimizer.
        
        Args:
            returns (pd.DataFrame): Asset returns data
            config (Dict[str, Any], optional): Optimization configuration
        """
        self.returns = returns
        self.config = config or {}
        self.optimal_weights = None
        self.optimization_result = None
        
        # Validate data
        self._validate_data()
    
    @abstractmethod
    def optimize(self, **kwargs) -> Dict[str, Any]:
        """
        Perform portfolio optimization.
        
        Args:
            **kwargs: Optimization parameters
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        pass
    
    @abstractmethod
    def get_efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier points.
        
        Args:
            num_portfolios (int): Number of portfolios to generate
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        pass
    
    def get_optimal_weights(self) -> np.ndarray:
        """
        Get the optimal portfolio weights.
        
        Returns:
            np.ndarray: Optimal weights
        """
        if self.optimal_weights is None:
            raise RuntimeError("Optimization must be performed first")
        return self.optimal_weights.copy()
    
    def get_portfolio_metrics(self, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights (np.ndarray, optional): Portfolio weights (uses optimal if None)
            
        Returns:
            Dict[str, float]: Portfolio metrics
        """
        if weights is None:
            weights = self.get_optimal_weights()
        
        return self._calculate_metrics(weights)
    
    def _validate_data(self) -> None:
        """Validate input data."""
        if not isinstance(self.returns, pd.DataFrame):
            raise ValueError("Returns must be a pandas DataFrame")
        if self.returns.empty:
            raise ValueError("Returns data cannot be empty")
        if self.returns.isnull().any().any():
            raise ValueError("Returns data cannot contain missing values")
    
    @abstractmethod
    def _calculate_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics for given weights."""
        pass


class BaseBacktestingStrategy(ABC):
    """
    Abstract base class for backtesting strategies.
    
    This class defines the interface for different trading strategies,
    enabling consistent backtesting and performance comparison.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtesting strategy.
        
        Args:
            name (str): Strategy name
            config (Dict[str, Any], optional): Strategy configuration
        """
        self.name = name
        self.config = config or {}
        self.positions = None
        self.returns = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            pd.DataFrame: Trading signals
        """
        pass
    
    @abstractmethod
    def calculate_positions(self, signals: pd.DataFrame, 
                          initial_capital: float) -> pd.DataFrame:
        """
        Calculate position sizes based on signals.
        
        Args:
            signals (pd.DataFrame): Trading signals
            initial_capital (float): Initial capital
            
        Returns:
            pd.DataFrame: Position sizes
        """
        pass
    
    def backtest(self, data: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """
        Perform backtesting of the strategy.
        
        Args:
            data (pd.DataFrame): Market data
            initial_capital (float): Initial capital
            
        Returns:
            Dict[str, Any]: Backtesting results
        """
        try:
            # Generate signals
            signals = self.generate_signals(data)
            
            # Calculate positions
            positions = self.calculate_positions(signals, initial_capital)
            
            # Calculate returns
            self.returns = self._calculate_returns(data, positions)
            
            # Calculate performance metrics
            performance = self._calculate_performance()
            
            return {
                'strategy_name': self.name,
                'signals': signals,
                'positions': positions,
                'returns': self.returns,
                'performance': performance
            }
        
        except Exception as e:
            logger.error(f"Backtesting failed for strategy {self.name}: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dict[str, float]: Performance summary
        """
        if self.returns is None:
            raise RuntimeError("Strategy must be backtested first")
        
        return self._calculate_performance()
    
    @abstractmethod
    def _calculate_returns(self, data: pd.DataFrame, 
                          positions: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns."""
        pass
    
    @abstractmethod
    def _calculate_performance(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        pass


class ModelFactory:
    """
    Factory class for creating model instances.
    
    This class implements the Factory pattern to create different types
    of forecasting models based on configuration.
    """
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type) -> None:
        """
        Register a new model type.
        
        Args:
            model_type (str): Model type identifier
            model_class (type): Model class to register
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class must inherit from BaseModel")
        cls._models[model_type] = model_class
        logger.info(f"Registered model type: {model_type}")
    
    @classmethod
    def create_model(cls, model_type: str, name: str, 
                    config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """
        Create a model instance of the specified type.
        
        Args:
            model_type (str): Type of model to create
            name (str): Model name
            config (Dict[str, Any], optional): Model configuration
            
        Returns:
            BaseModel: Model instance
            
        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        return model_class(name, config)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        Get list of available model types.
        
        Returns:
            List[str]: Available model types
        """
        return list(cls._models.keys())


class StrategyFactory:
    """
    Factory class for creating backtesting strategy instances.
    
    This class implements the Factory pattern to create different types
    of trading strategies based on configuration.
    """
    
    _strategies: Dict[str, type] = {}
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: type) -> None:
        """
        Register a new strategy type.
        
        Args:
            strategy_type (str): Strategy type identifier
            strategy_class (type): Strategy class to register
        """
        if not issubclass(strategy_class, BaseBacktestingStrategy):
            raise ValueError(f"Strategy class must inherit from BaseBacktestingStrategy")
        cls._strategies[strategy_type] = strategy_class
        logger.info(f"Registered strategy type: {strategy_type}")
    
    @classmethod
    def create_strategy(cls, strategy_type: str, name: str,
                       config: Optional[Dict[str, Any]] = None) -> BaseBacktestingStrategy:
        """
        Create a strategy instance of the specified type.
        
        Args:
            strategy_type (str): Type of strategy to create
            name (str): Strategy name
            config (Dict[str, Any], optional): Strategy configuration
            
        Returns:
            BaseBacktestingStrategy: Strategy instance
            
        Raises:
            ValueError: If strategy type is not registered
        """
        if strategy_type not in cls._strategies:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_class = cls._strategies[strategy_type]
        return strategy_class(name, config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get list of available strategy types.
        
        Returns:
            List[str]: Available strategy types
        """
        return list(cls._strategies.keys())


# Register built-in model types
def register_builtin_models():
    """Register built-in model types with the factory."""
    try:
        from .arima_model import ARIMAModel
        from .lstm_model import LSTMForecaster
        
        ModelFactory.register_model(ModelType.ARIMA.value, ARIMAModel)
        ModelFactory.register_model(ModelType.LSTM.value, LSTMForecaster)
        
        logger.info("Built-in models registered successfully")
    
    except ImportError as e:
        logger.warning(f"Could not register all built-in models: {e}")


# Register built-in strategy types
def register_builtin_strategies():
    """Register built-in strategy types with the factory."""
    try:
        from .backtesting import BuyHoldStrategy, MomentumStrategy, MeanReversionStrategy
        
        StrategyFactory.register_strategy(StrategyType.BUY_HOLD.value, BuyHoldStrategy)
        StrategyFactory.register_strategy(StrategyType.MOMENTUM.value, MomentumStrategy)
        StrategyFactory.register_strategy(StrategyType.MEAN_REVERSION.value, MeanReversionStrategy)
        
        logger.info("Built-in strategies registered successfully")
    
    except ImportError as e:
        logger.warning(f"Could not register all built-in strategies: {e}")


# Auto-register built-in components
register_builtin_models()
register_builtin_strategies()
