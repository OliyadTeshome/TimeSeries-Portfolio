"""
Unit tests for base classes and abstract interfaces.

This module tests the abstract base classes, design patterns, and interfaces
that form the foundation of the TimeSeries-Portfolio framework.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from base import (
    BaseModel, BaseDataProcessor, BasePortfolioOptimizer, BaseBacktestingStrategy,
    ModelFactory, StrategyFactory, ModelType, OptimizationMethod, StrategyType,
    ModelResult, PerformanceMetrics
)


class TestModelResult:
    """Test ModelResult dataclass."""
    
    def test_model_result_creation(self):
        """Test ModelResult creation with required fields."""
        predictions = np.array([1.0, 2.0, 3.0])
        result = ModelResult(predictions=predictions)
        
        assert np.array_equal(result.predictions, predictions)
        assert result.confidence_intervals is None
        assert result.model_parameters is None
        assert result.training_time is None
        assert result.prediction_time is None
        assert result.metadata is None
    
    def test_model_result_with_optional_fields(self):
        """Test ModelResult creation with optional fields."""
        predictions = np.array([1.0, 2.0, 3.0])
        confidence_intervals = (np.array([0.8, 1.8, 2.8]), np.array([1.2, 2.2, 3.2]))
        model_parameters = {'p': 1, 'd': 1, 'q': 1}
        
        result = ModelResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            model_parameters=model_parameters,
            training_time=1.5,
            prediction_time=0.1,
            metadata={'model_type': 'arima'}
        )
        
        assert np.array_equal(result.predictions, predictions)
        assert result.confidence_intervals == confidence_intervals
        assert result.model_parameters == model_parameters
        assert result.training_time == 1.5
        assert result.prediction_time == 0.1
        assert result.metadata == {'model_type': 'arima'}
    
    def test_model_result_validation(self):
        """Test ModelResult validation."""
        # Test with non-array predictions
        with pytest.raises(ValueError, match="Predictions must be a numpy array"):
            ModelResult(predictions=[1, 2, 3])
        
        # Test with empty predictions
        with pytest.raises(ValueError, match="Predictions cannot be empty"):
            ModelResult(predictions=np.array([]))


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation with required fields."""
        metrics = PerformanceMetrics(
            mse=0.1,
            mae=0.2,
            mape=5.0,
            rmse=0.32
        )
        
        assert metrics.mse == 0.1
        assert metrics.mae == 0.2
        assert metrics.mape == 5.0
        assert metrics.rmse == 0.32
        assert metrics.r2_score is None
        assert metrics.directional_accuracy is None
        assert metrics.additional_metrics is None
    
    def test_performance_metrics_with_optional_fields(self):
        """Test PerformanceMetrics creation with optional fields."""
        metrics = PerformanceMetrics(
            mse=0.1,
            mae=0.2,
            mape=5.0,
            rmse=0.32,
            r2_score=0.85,
            directional_accuracy=0.72,
            additional_metrics={'aic': 150.5, 'bic': 160.2}
        )
        
        assert metrics.r2_score == 0.85
        assert metrics.directional_accuracy == 0.72
        assert metrics.additional_metrics == {'aic': 150.5, 'bic': 160.2}
    
    def test_performance_metrics_validation(self):
        """Test PerformanceMetrics validation."""
        # Test with negative MSE
        with pytest.raises(ValueError, match="MSE cannot be negative"):
            PerformanceMetrics(mse=-0.1, mae=0.2, mape=5.0, rmse=0.32)
        
        # Test with negative MAE
        with pytest.raises(ValueError, match="MAE cannot be negative"):
            PerformanceMetrics(mse=0.1, mae=-0.2, mape=5.0, rmse=0.32)
        
        # Test with negative MAPE
        with pytest.raises(ValueError, match="MAPE cannot be negative"):
            PerformanceMetrics(mse=0.1, mae=0.2, mape=-5.0, rmse=0.32)


class TestEnums:
    """Test enumeration classes."""
    
    def test_model_type_enum(self):
        """Test ModelType enum values."""
        assert ModelType.ARIMA.value == "arima"
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.PROPHET.value == "prophet"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.ENSEMBLE.value == "ensemble"
    
    def test_optimization_method_enum(self):
        """Test OptimizationMethod enum values."""
        assert OptimizationMethod.SHARPE_RATIO.value == "sharpe_ratio"
        assert OptimizationMethod.MIN_VARIANCE.value == "min_variance"
        assert OptimizationMethod.MAX_RETURN.value == "max_return"
        assert OptimizationMethod.MAX_DIVERSIFICATION.value == "max_diversification"
        assert OptimizationMethod.BLACK_LITTERMAN.value == "black_litterman"
    
    def test_strategy_type_enum(self):
        """Test StrategyType enum values."""
        assert StrategyType.BUY_HOLD.value == "buy_hold"
        assert StrategyType.MOMENTUM.value == "momentum"
        assert StrategyType.MEAN_REVERSION.value == "mean_reversion"
        assert StrategyType.VOLATILITY_TARGETING.value == "volatility_targeting"
        assert StrategyType.RISK_PARITY.value == "risk_parity"


class TestBaseModel:
    """Test BaseModel abstract base class."""
    
    def test_base_model_initialization(self):
        """Test BaseModel initialization."""
        class ConcreteModel(BaseModel):
            def train(self, data, **kwargs):
                self.is_trained = True
                return self
            
            def predict(self, steps, **kwargs):
                return ModelResult(predictions=np.random.randn(steps))
            
            def evaluate(self, test_data):
                return PerformanceMetrics(
                    mse=0.1, mae=0.2, mape=5.0, rmse=0.32
                )
            
            def _save_implementation(self, filepath):
                pass
            
            def _load_implementation(self, filepath):
                pass
        
        model = ConcreteModel("test_model", {"param": "value"})
        
        assert model.name == "test_model"
        assert model.config == {"param": "value"}
        assert model.is_trained is False
        assert model.training_time is None
        assert model.model_parameters == {}
    
    def test_base_model_abstract_methods(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseModel("test", {})
    
    def test_base_model_save_before_training(self):
        """Test that model cannot be saved before training."""
        class ConcreteModel(BaseModel):
            def train(self, data, **kwargs):
                self.is_trained = True
                return self
            
            def predict(self, steps, **kwargs):
                return ModelResult(predictions=np.random.randn(steps))
            
            def evaluate(self, test_data):
                return PerformanceMetrics(
                    mse=0.1, mae=0.2, mape=5.0, rmse=0.32
                )
            
            def _save_implementation(self, filepath):
                pass
            
            def _load_implementation(self, filepath):
                pass
        
        model = ConcreteModel("test_model")
        
        with pytest.raises(RuntimeError, match="Model must be trained before saving"):
            model.save_model("test.pkl")
    
    def test_base_model_get_model_info(self):
        """Test get_model_info method."""
        class ConcreteModel(BaseModel):
            def train(self, data, **kwargs):
                self.is_trained = True
                self.training_time = 1.5
                self.model_parameters = {'p': 1, 'd': 1, 'q': 1}
                return self
            
            def predict(self, steps, **kwargs):
                return ModelResult(predictions=np.random.randn(steps))
            
            def evaluate(self, test_data):
                return PerformanceMetrics(
                    mse=0.1, mae=0.2, mape=5.0, rmse=0.32
                )
            
            def _save_implementation(self, filepath):
                pass
            
            def _load_implementation(self, filepath):
                pass
        
        model = ConcreteModel("test_model", {"param": "value"})
        model.train(pd.Series([1, 2, 3]))
        
        info = model.get_model_info()
        
        assert info['name'] == "test_model"
        assert info['type'] == "ConcreteModel"
        assert info['is_trained'] is True
        assert info['training_time'] == 1.5
        assert info['model_parameters'] == {'p': 1, 'd': 1, 'q': 1}
        assert info['config'] == {"param": "value"}


class TestBaseDataProcessor:
    """Test BaseDataProcessor abstract base class."""
    
    def test_base_data_processor_initialization(self):
        """Test BaseDataProcessor initialization."""
        class ConcreteProcessor(BaseDataProcessor):
            def fit(self, data):
                self.feature_names = ['feature_1', 'feature_2']
                self.processing_stats = {'rows_processed': len(data)}
                return self
            
            def transform(self, data):
                return data.copy()
        
        processor = ConcreteProcessor({"param": "value"})
        
        assert processor.config == {"param": "value"}
        assert processor.feature_names == []
        assert processor.processing_stats == {}
    
    def test_base_data_processor_abstract_methods(self):
        """Test that BaseDataProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDataProcessor({})
    
    def test_base_data_processor_fit_transform(self):
        """Test fit_transform method."""
        class ConcreteProcessor(BaseDataProcessor):
            def fit(self, data):
                self.feature_names = ['feature_1', 'feature_2']
                self.processing_stats = {'rows_processed': len(data)}
                return self
            
            def transform(self, data):
                return data.copy()
        
        processor = ConcreteProcessor()
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        
        result = processor.fit_transform(data)
        
        assert processor.feature_names == ['feature_1', 'feature_2']
        assert processor.processing_stats == {'rows_processed': 3}
        assert isinstance(result, pd.DataFrame)
    
    def test_base_data_processor_get_feature_names(self):
        """Test get_feature_names method."""
        class ConcreteProcessor(BaseDataProcessor):
            def fit(self, data):
                self.feature_names = ['feature_1', 'feature_2']
                return self
            
            def transform(self, data):
                return data.copy()
        
        processor = ConcreteProcessor()
        processor.fit(pd.DataFrame({'A': [1, 2, 3]}))
        
        feature_names = processor.get_feature_names()
        assert feature_names == ['feature_1', 'feature_2']
        
        # Test that returned list is a copy
        feature_names.append('feature_3')
        assert processor.feature_names == ['feature_1', 'feature_2']


class TestBasePortfolioOptimizer:
    """Test BasePortfolioOptimizer abstract base class."""
    
    def test_base_portfolio_optimizer_initialization(self):
        """Test BasePortfolioOptimizer initialization."""
        returns = pd.DataFrame({
            'SPY': [0.01, -0.02, 0.03],
            'TSLA': [0.05, -0.01, 0.02]
        })
        
        class ConcreteOptimizer(BasePortfolioOptimizer):
            def optimize(self, **kwargs):
                self.optimal_weights = np.array([0.6, 0.4])
                return {'status': 'success'}
            
            def get_efficient_frontier(self, num_portfolios=100):
                return pd.DataFrame({
                    'return': [0.05, 0.06, 0.07],
                    'risk': [0.15, 0.16, 0.17]
                })
            
            def _calculate_metrics(self, weights):
                return {'return': 0.06, 'risk': 0.16}
        
        optimizer = ConcreteOptimizer(returns, {"param": "value"})
        
        assert optimizer.returns is returns
        assert optimizer.config == {"param": "value"}
        assert optimizer.optimal_weights is None
        assert optimizer.optimization_result is None
    
    def test_base_portfolio_optimizer_abstract_methods(self):
        """Test that BasePortfolioOptimizer cannot be instantiated directly."""
        returns = pd.DataFrame({'A': [0.01, 0.02]})
        
        with pytest.raises(TypeError):
            BasePortfolioOptimizer(returns, {})
    
    def test_base_portfolio_optimizer_data_validation(self):
        """Test data validation in BasePortfolioOptimizer."""
        # Test with non-DataFrame
        with pytest.raises(ValueError, match="Returns must be a pandas DataFrame"):
            class ConcreteOptimizer(BasePortfolioOptimizer):
                def optimize(self, **kwargs):
                    pass
                
                def get_efficient_frontier(self, num_portfolios=100):
                    pass
                
                def _calculate_metrics(self, weights):
                    pass
            
            ConcreteOptimizer([0.01, 0.02])
        
        # Test with empty DataFrame
        with pytest.raises(ValueError, match="Returns data cannot be empty"):
            class ConcreteOptimizer(BasePortfolioOptimizer):
                def optimize(self, **kwargs):
                    pass
                
                def get_efficient_frontier(self, num_portfolios=100):
                    pass
                
                def _calculate_metrics(self, weights):
                    pass
            
            ConcreteOptimizer(pd.DataFrame())
        
        # Test with missing values
        returns_with_nan = pd.DataFrame({
            'SPY': [0.01, np.nan, 0.03],
            'TSLA': [0.05, -0.01, 0.02]
        })
        
        with pytest.raises(ValueError, match="Returns data cannot contain missing values"):
            class ConcreteOptimizer(BasePortfolioOptimizer):
                def optimize(self, **kwargs):
                    pass
                
                def get_efficient_frontier(self, num_portfolios=100):
                    pass
                
                def _calculate_metrics(self, weights):
                    pass
            
            ConcreteOptimizer(returns_with_nan)
    
    def test_base_portfolio_optimizer_get_optimal_weights(self):
        """Test get_optimal_weights method."""
        returns = pd.DataFrame({
            'SPY': [0.01, -0.02, 0.03],
            'TSLA': [0.05, -0.01, 0.02]
        })
        
        class ConcreteOptimizer(BasePortfolioOptimizer):
            def optimize(self, **kwargs):
                self.optimal_weights = np.array([0.6, 0.4])
                return {'status': 'success'}
            
            def get_efficient_frontier(self, num_portfolios=100):
                return pd.DataFrame()
            
            def _calculate_metrics(self, weights):
                return {'return': 0.06, 'risk': 0.16}
        
        optimizer = ConcreteOptimizer(returns)
        
        # Test before optimization
        with pytest.raises(RuntimeError, match="Optimization must be performed first"):
            optimizer.get_optimal_weights()
        
        # Test after optimization
        optimizer.optimize()
        weights = optimizer.get_optimal_weights()
        
        assert np.array_equal(weights, np.array([0.6, 0.4]))
        
        # Test that returned array is a copy
        weights[0] = 0.8
        assert optimizer.optimal_weights[0] == 0.6
    
    def test_base_portfolio_optimizer_get_portfolio_metrics(self):
        """Test get_portfolio_metrics method."""
        returns = pd.DataFrame({
            'SPY': [0.01, -0.02, 0.03],
            'TSLA': [0.05, -0.01, 0.02]
        })
        
        class ConcreteOptimizer(BasePortfolioOptimizer):
            def optimize(self, **kwargs):
                self.optimal_weights = np.array([0.6, 0.4])
                return {'status': 'success'}
            
            def get_efficient_frontier(self, num_portfolios=100):
                return pd.DataFrame()
            
            def _calculate_metrics(self, weights):
                return {'return': 0.06, 'risk': 0.16}
        
        optimizer = ConcreteOptimizer(returns)
        optimizer.optimize()
        
        # Test with default weights (optimal)
        metrics = optimizer.get_portfolio_metrics()
        assert metrics == {'return': 0.06, 'risk': 0.16}
        
        # Test with custom weights
        custom_weights = np.array([0.5, 0.5])
        metrics = optimizer.get_portfolio_metrics(custom_weights)
        assert metrics == {'return': 0.06, 'risk': 0.16}


class TestBaseBacktestingStrategy:
    """Test BaseBacktestingStrategy abstract base class."""
    
    def test_base_backtesting_strategy_initialization(self):
        """Test BaseBacktestingStrategy initialization."""
        class ConcreteStrategy(BaseBacktestingStrategy):
            def generate_signals(self, data):
                return pd.DataFrame({'signal': [1, 0, -1]}, index=data.index)
            
            def calculate_positions(self, signals, initial_capital):
                return pd.DataFrame({'position': [100, 0, -100]}, index=signals.index)
            
            def _calculate_returns(self, data, positions):
                return pd.Series([0.01, 0.0, -0.01], index=data.index)
            
            def _calculate_performance(self):
                return {'total_return': 0.1, 'sharpe_ratio': 0.5}
        
        strategy = ConcreteStrategy("test_strategy", {"param": "value"})
        
        assert strategy.name == "test_strategy"
        assert strategy.config == {"param": "value"}
        assert strategy.positions is None
        assert strategy.returns is None
    
    def test_base_backtesting_strategy_abstract_methods(self):
        """Test that BaseBacktestingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBacktestingStrategy("test", {})
    
    def test_base_backtesting_strategy_backtest(self):
        """Test backtest method."""
        class ConcreteStrategy(BaseBacktestingStrategy):
            def generate_signals(self, data):
                return pd.DataFrame({'signal': [1, 0, -1]}, index=data.index)
            
            def calculate_positions(self, signals, initial_capital):
                return pd.DataFrame({'position': [100, 0, -100]}, index=signals.index)
            
            def _calculate_returns(self, data, positions):
                return pd.Series([0.01, 0.0, -0.01], index=data.index)
            
            def _calculate_performance(self):
                return {'total_return': 0.1, 'sharpe_ratio': 0.5}
        
        strategy = ConcreteStrategy("test_strategy")
        data = pd.DataFrame({'close': [100, 101, 99]}, index=pd.date_range('2020-01-01', periods=3))
        
        result = strategy.backtest(data, 1000)
        
        assert result['strategy_name'] == "test_strategy"
        assert 'signals' in result
        assert 'positions' in result
        assert 'returns' in result
        assert 'performance' in result
        assert strategy.returns is not None
    
    def test_base_backtesting_strategy_get_performance_summary(self):
        """Test get_performance_summary method."""
        class ConcreteStrategy(BaseBacktestingStrategy):
            def generate_signals(self, data):
                return pd.DataFrame({'signal': [1, 0, -1]}, index=data.index)
            
            def calculate_positions(self, signals, initial_capital):
                return pd.DataFrame({'position': [100, 0, -100]}, index=signals.index)
            
            def _calculate_returns(self, data, positions):
                return pd.Series([0.01, 0.0, -0.01], index=data.index)
            
            def _calculate_performance(self):
                return {'total_return': 0.1, 'sharpe_ratio': 0.5}
        
        strategy = ConcreteStrategy("test_strategy")
        
        # Test before backtesting
        with pytest.raises(RuntimeError, match="Strategy must be backtested first"):
            strategy.get_performance_summary()
        
        # Test after backtesting
        data = pd.DataFrame({'close': [100, 101, 99]}, index=pd.date_range('2020-01-01', periods=3))
        strategy.backtest(data, 1000)
        
        performance = strategy.get_performance_summary()
        assert performance == {'total_return': 0.1, 'sharpe_ratio': 0.5}


class TestModelFactory:
    """Test ModelFactory class."""
    
    def test_model_factory_registration(self):
        """Test model registration."""
        class TestModel(BaseModel):
            def train(self, data, **kwargs):
                self.is_trained = True
                return self
            
            def predict(self, steps, **kwargs):
                return ModelResult(predictions=np.random.randn(steps))
            
            def evaluate(self, test_data):
                return PerformanceMetrics(
                    mse=0.1, mae=0.2, mape=5.0, rmse=0.32
                )
            
            def _save_implementation(self, filepath):
                pass
            
            def _load_implementation(self, filepath):
                pass
        
        # Test registration
        ModelFactory.register_model("test_model", TestModel)
        assert "test_model" in ModelFactory.get_available_models()
        
        # Test creating model
        model = ModelFactory.create_model("test_model", "test_instance")
        assert isinstance(model, TestModel)
        assert model.name == "test_instance"
        
        # Test invalid model type
        with pytest.raises(ValueError, match="Unknown model type"):
            ModelFactory.create_model("invalid_type", "test")
        
        # Test registering non-BaseModel class
        with pytest.raises(ValueError, match="Model class must inherit from BaseModel"):
            ModelFactory.register_model("invalid", str)
    
    def test_model_factory_get_available_models(self):
        """Test get_available_models method."""
        available = ModelFactory.get_available_models()
        assert isinstance(available, list)
        assert "test_model" in available  # From previous test


class TestStrategyFactory:
    """Test StrategyFactory class."""
    
    def test_strategy_factory_registration(self):
        """Test strategy registration."""
        class TestStrategy(BaseBacktestingStrategy):
            def generate_signals(self, data):
                return pd.DataFrame({'signal': [1, 0, -1]}, index=data.index)
            
            def calculate_positions(self, signals, initial_capital):
                return pd.DataFrame({'position': [100, 0, -100]}, index=signals.index)
            
            def _calculate_returns(self, data, positions):
                return pd.Series([0.01, 0.0, -0.01], index=data.index)
            
            def _calculate_performance(self):
                return {'total_return': 0.1, 'sharpe_ratio': 0.5}
        
        # Test registration
        StrategyFactory.register_strategy("test_strategy", TestStrategy)
        assert "test_strategy" in StrategyFactory.get_available_strategies()
        
        # Test creating strategy
        strategy = StrategyFactory.create_strategy("test_strategy", "test_instance")
        assert isinstance(strategy, TestStrategy)
        assert strategy.name == "test_instance"
        
        # Test invalid strategy type
        with pytest.raises(ValueError, match="Unknown strategy type"):
            StrategyFactory.create_strategy("invalid_type", "test")
        
        # Test registering non-BaseBacktestingStrategy class
        with pytest.raises(ValueError, match="Strategy class must inherit from BaseBacktestingStrategy"):
            StrategyFactory.register_strategy("invalid", str)
    
    def test_strategy_factory_get_available_strategies(self):
        """Test get_available_strategies method."""
        available = StrategyFactory.get_available_strategies()
        assert isinstance(available, list)
        assert "test_strategy" in available  # From previous test


class TestIntegration:
    """Integration tests for base classes."""
    
    def test_model_factory_integration(self):
        """Test integration between ModelFactory and BaseModel."""
        class TestModel(BaseModel):
            def train(self, data, **kwargs):
                self.is_trained = True
                self.training_time = 1.0
                return self
            
            def predict(self, steps, **kwargs):
                return ModelResult(predictions=np.random.randn(steps))
            
            def evaluate(self, test_data):
                return PerformanceMetrics(
                    mse=0.1, mae=0.2, mape=5.0, rmse=0.32
                )
            
            def _save_implementation(self, filepath):
                pass
            
            def _load_implementation(self, filepath):
                pass
        
        # Register and create model
        ModelFactory.register_model("integration_test", TestModel)
        model = ModelFactory.create_model("integration_test", "test", {"param": "value"})
        
        # Test model lifecycle
        assert not model.is_trained
        
        data = pd.Series([1, 2, 3, 4, 5])
        model.train(data)
        assert model.is_trained
        assert model.training_time == 1.0
        
        result = model.predict(3)
        assert isinstance(result, ModelResult)
        assert len(result.predictions) == 3
        
        metrics = model.evaluate(data)
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.mse == 0.1
    
    def test_strategy_factory_integration(self):
        """Test integration between StrategyFactory and BaseBacktestingStrategy."""
        class TestStrategy(BaseBacktestingStrategy):
            def generate_signals(self, data):
                return pd.DataFrame({'signal': [1, 0, -1]}, index=data.index)
            
            def calculate_positions(self, signals, initial_capital):
                return pd.DataFrame({'position': [100, 0, -100]}, index=signals.index)
            
            def _calculate_returns(self, data, positions):
                return pd.Series([0.01, 0.0, -0.01], index=data.index)
            
            def _calculate_performance(self):
                return {'total_return': 0.1, 'sharpe_ratio': 0.5}
        
        # Register and create strategy
        StrategyFactory.register_strategy("integration_test", TestStrategy)
        strategy = StrategyFactory.create_strategy("integration_test", "test", {"param": "value"})
        
        # Test strategy lifecycle
        data = pd.DataFrame({'close': [100, 101, 99]}, index=pd.date_range('2020-01-01', periods=3))
        
        result = strategy.backtest(data, 1000)
        assert result['strategy_name'] == "test"
        assert strategy.returns is not None
        
        performance = strategy.get_performance_summary()
        assert performance['total_return'] == 0.1
        assert performance['sharpe_ratio'] == 0.5
