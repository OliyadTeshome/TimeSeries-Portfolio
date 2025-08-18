"""
Pytest Configuration and Shared Fixtures

This module provides shared fixtures and configuration for all tests
in the TimeSeries-Portfolio framework.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import ConfigManager, get_config
from base import ModelType, OptimizationMethod, StrategyType
from logging_config import setup_logging


@pytest.fixture(scope="session")
def test_config():
    """Create a test configuration instance."""
    config = ConfigManager()
    
    # Override with test-specific settings
    config.update({
        'data': {
            'data_path': 'tests/data/',
            'raw_data_path': 'tests/data/raw/',
            'processed_data_path': 'tests/data/processed/',
            'cache_path': 'tests/data/cache/',
            'min_data_length': 10,
            'max_missing_ratio': 0.2
        },
        'system': {
            'log_level': 'WARNING',
            'enable_caching': False,
            'max_workers': 1
        }
    })
    
    return config


@pytest.fixture(scope="session")
def test_logger():
    """Create a test logger instance."""
    return setup_logging(
        log_level="WARNING",
        log_file=None,
        enable_performance_logging=False,
        enable_error_tracking=False
    )


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    # Create date range
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample data with trend, seasonality, and noise
    n = len(dates)
    trend = np.linspace(100, 150, n)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    
    # Create multiple assets
    data = pd.DataFrame({
        'SPY': trend + seasonality + noise,
        'TSLA': trend * 1.5 + seasonality * 1.2 + noise * 2,
        'BND': trend * 0.8 + seasonality * 0.5 + noise * 0.5
    }, index=dates)
    
    # Add some missing values for testing
    data.loc[data.sample(frac=0.05).index, 'SPY'] = np.nan
    
    return data


@pytest.fixture
def sample_returns_data(sample_time_series_data):
    """Generate sample returns data for testing."""
    return sample_time_series_data.pct_change().dropna()


@pytest.fixture
def sample_forecast_data(sample_time_series_data):
    """Generate sample data for forecasting tests."""
    # Use only SPY data for forecasting tests
    return sample_time_series_data['SPY'].dropna()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_data_loader():
    """Create a mock data loader for testing."""
    class MockDataLoader:
        def __init__(self):
            self.data_path = "tests/data/"
            self.raw_path = "tests/data/raw/"
            self.processed_path = "tests/data/processed/"
        
        def load_csv(self, filename, **kwargs):
            # Return sample data
            dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
            data = pd.DataFrame({
                'close': np.random.randn(len(dates)).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)
            return data
        
        def save_data(self, data, filename, **kwargs):
            # Mock save operation
            pass
    
    return MockDataLoader()


@pytest.fixture
def sample_portfolio_weights():
    """Generate sample portfolio weights for testing."""
    return np.array([0.4, 0.4, 0.2])


@pytest.fixture
def sample_correlation_matrix():
    """Generate sample correlation matrix for testing."""
    return pd.DataFrame({
        'SPY': [1.0, 0.7, 0.3],
        'TSLA': [0.7, 1.0, 0.2],
        'BND': [0.3, 0.2, 1.0]
    }, index=['SPY', 'TSLA', 'BND'])


@pytest.fixture
def sample_covariance_matrix():
    """Generate sample covariance matrix for testing."""
    return pd.DataFrame({
        'SPY': [0.04, 0.028, 0.012],
        'TSLA': [0.028, 0.09, 0.018],
        'BND': [0.012, 0.018, 0.01]
    }, index=['SPY', 'TSLA', 'BND'])


@pytest.fixture
def sample_expected_returns():
    """Generate sample expected returns for testing."""
    return pd.Series({
        'SPY': 0.08,
        'TSLA': 0.15,
        'BND': 0.04
    })


@pytest.fixture
def sample_backtest_data(sample_time_series_data):
    """Generate sample data for backtesting tests."""
    # Use last 252 days (1 year) for backtesting
    return sample_time_series_data.tail(252)


@pytest.fixture
def sample_strategy_signals(sample_backtest_data):
    """Generate sample strategy signals for testing."""
    signals = pd.DataFrame(index=sample_backtest_data.index)
    signals['SPY_signal'] = np.random.choice([-1, 0, 1], size=len(signals))
    signals['TSLA_signal'] = np.random.choice([-1, 0, 1], size=len(signals))
    signals['BND_signal'] = np.random.choice([-1, 0, 1], size=len(signals))
    return signals


@pytest.fixture
def sample_performance_metrics():
    """Generate sample performance metrics for testing."""
    return {
        'total_return': 0.15,
        'annualized_return': 0.08,
        'volatility': 0.12,
        'sharpe_ratio': 0.67,
        'max_drawdown': -0.08,
        'var_95': -0.15,
        'cvar_95': -0.20
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    class MockModel:
        def __init__(self, name="mock_model"):
            self.name = name
            self.is_trained = False
            self.training_time = None
            self.model_parameters = {}
        
        def train(self, data, **kwargs):
            self.is_trained = True
            self.training_time = 1.0
            return self
        
        def predict(self, steps, **kwargs):
            if not self.is_trained:
                raise RuntimeError("Model must be trained first")
            return np.random.randn(steps)
        
        def evaluate(self, test_data):
            return {
                'mse': 0.1,
                'mae': 0.2,
                'mape': 5.0,
                'rmse': 0.32
            }
    
    return MockModel()


@pytest.fixture
def mock_optimizer():
    """Create a mock portfolio optimizer for testing."""
    class MockOptimizer:
        def __init__(self):
            self.optimal_weights = np.array([0.4, 0.4, 0.2])
            self.optimization_result = {'status': 'success'}
        
        def optimize(self, **kwargs):
            return self.optimization_result
        
        def get_efficient_frontier(self, num_portfolios=100):
            return pd.DataFrame({
                'return': np.linspace(0.04, 0.15, num_portfolios),
                'risk': np.linspace(0.08, 0.25, num_portfolios)
            })
    
    return MockOptimizer()


@pytest.fixture
def mock_strategy():
    """Create a mock backtesting strategy for testing."""
    class MockStrategy:
        def __init__(self, name="mock_strategy"):
            self.name = name
            self.positions = None
            self.returns = None
        
        def generate_signals(self, data):
            signals = pd.DataFrame(index=data.index)
            signals['signal'] = np.random.choice([-1, 0, 1], size=len(signals))
            return signals
        
        def calculate_positions(self, signals, initial_capital):
            positions = pd.DataFrame(index=signals.index)
            positions['position'] = signals['signal'] * initial_capital / 100
            return positions
        
        def backtest(self, data, initial_capital):
            signals = self.generate_signals(data)
            positions = self.calculate_positions(signals, initial_capital)
            self.returns = pd.Series(np.random.randn(len(data)) * 0.01, index=data.index)
            
            return {
                'strategy_name': self.name,
                'signals': signals,
                'positions': positions,
                'returns': self.returns,
                'performance': {
                    'total_return': 0.1,
                    'sharpe_ratio': 0.5,
                    'max_drawdown': -0.05
                }
            }
    
    return MockStrategy()


# Test utilities
def assert_dataframe_equal(df1, df2, **kwargs):
    """Assert that two dataframes are equal with custom tolerance."""
    pd.testing.assert_frame_equal(df1, df2, **kwargs)


def assert_series_equal(s1, s2, **kwargs):
    """Assert that two series are equal with custom tolerance."""
    pd.testing.assert_series_equal(s1, s2, **kwargs)


def assert_array_equal(arr1, arr2, **kwargs):
    """Assert that two arrays are equal with custom tolerance."""
    np.testing.assert_array_equal(arr1, arr2, **kwargs)


def assert_array_almost_equal(arr1, arr2, **kwargs):
    """Assert that two arrays are almost equal with custom tolerance."""
    np.testing.assert_array_almost_equal(arr1, arr2, **kwargs)


def create_test_data(n_samples=100, n_features=3, random_state=42):
    """Create test data with specified parameters."""
    np.random.seed(random_state)
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    data = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    return data


def create_test_returns(n_assets=3, n_periods=252, random_state=42):
    """Create test returns data with specified parameters."""
    np.random.seed(random_state)
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
    returns = pd.DataFrame(
        np.random.randn(n_periods, n_assets) * 0.02,
        index=dates,
        columns=[f'asset_{i}' for i in range(n_assets)]
    )
    
    return returns


# Performance testing utilities
def benchmark_function(func, *args, iterations=100, **kwargs):
    """Benchmark a function's performance."""
    import time
    
    times = []
    for _ in range(iterations):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'iterations': iterations,
        'result': result
    }


# Memory testing utilities
def get_memory_usage():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def assert_memory_efficient(func, *args, max_memory_increase=50, **kwargs):
    """Assert that a function doesn't use excessive memory."""
    initial_memory = get_memory_usage()
    result = func(*args, **kwargs)
    final_memory = get_memory_usage()
    
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < max_memory_increase, \
        f"Memory usage increased by {memory_increase:.2f}MB, " \
        f"exceeding limit of {max_memory_increase}MB"
    
    return result
