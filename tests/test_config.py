"""
Unit tests for configuration management module.

This module tests the ConfigManager class and related functionality.
"""

import pytest
import tempfile
import os
import yaml
from pathlib import Path

from config import (
    ConfigManager, 
    get_config, 
    ModelConfig, 
    DataConfig, 
    PortfolioConfig,
    BacktestingConfig,
    SystemConfig
)


class TestModelConfig:
    """Test ModelConfig dataclass."""
    
    def test_model_config_defaults(self):
        """Test that ModelConfig has correct default values."""
        config = ModelConfig()
        
        assert config.arima_max_p == 5
        assert config.arima_max_q == 5
        assert config.arima_max_d == 2
        assert config.arima_seasonal is False
        assert config.arima_seasonal_periods == 12
        assert config.lstm_sequence_length == 10
        assert config.lstm_hidden_size == 50
        assert config.lstm_num_layers == 2
        assert config.lstm_dropout == 0.2
        assert config.lstm_learning_rate == 0.001
        assert config.lstm_epochs == 100
        assert config.lstm_batch_size == 32
        assert config.ensemble_method == "weighted_average"
        assert config.ensemble_weights is None
    
    def test_model_config_custom_values(self):
        """Test that ModelConfig accepts custom values."""
        config = ModelConfig(
            arima_max_p=10,
            lstm_epochs=200,
            ensemble_method="simple_average"
        )
        
        assert config.arima_max_p == 10
        assert config.lstm_epochs == 200
        assert config.ensemble_method == "simple_average"


class TestDataConfig:
    """Test DataConfig dataclass."""
    
    def test_data_config_defaults(self):
        """Test that DataConfig has correct default values."""
        config = DataConfig()
        
        assert config.data_path == "data/"
        assert config.raw_data_path == "data/raw/"
        assert config.processed_data_path == "data/processed/"
        assert config.cache_path == "data/cache/"
        assert config.min_data_length == 20
        assert config.max_missing_ratio == 0.1
        assert config.outlier_threshold == 3.0
        assert config.create_lags is True
        assert config.max_lags == 10
        assert config.create_rolling_features is True
        assert config.rolling_windows == [5, 10, 20]


class TestPortfolioConfig:
    """Test PortfolioConfig dataclass."""
    
    def test_portfolio_config_defaults(self):
        """Test that PortfolioConfig has correct default values."""
        config = PortfolioConfig()
        
        assert config.risk_free_rate == 0.02
        assert config.optimization_method == "sharpe_ratio"
        assert config.max_portfolio_weight == 0.4
        assert config.min_portfolio_weight == 0.0
        assert config.enable_short_selling is False
        assert config.enable_leverage is False
        assert config.max_leverage == 1.0
        assert config.var_confidence_level == 0.05
        assert config.cvar_confidence_level == 0.05
        assert config.max_drawdown_threshold == 0.2


class TestBacktestingConfig:
    """Test BacktestingConfig dataclass."""
    
    def test_backtesting_config_defaults(self):
        """Test that BacktestingConfig has correct default values."""
        config = BacktestingConfig()
        
        assert config.initial_capital == 100000
        assert config.transaction_cost == 0.001
        assert config.slippage == 0.0005
        assert config.rebalance_frequency == 21
        assert config.benchmark == "SPY"
        assert config.risk_free_rate == 0.02
        assert config.momentum_lookback == 20
        assert config.mean_reversion_lookback == 20
        assert config.volatility_lookback == 20


class TestSystemConfig:
    """Test SystemConfig dataclass."""
    
    def test_system_config_defaults(self):
        """Test that SystemConfig has correct default values."""
        config = SystemConfig()
        
        assert config.log_level == "INFO"
        assert config.log_format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert config.log_file is None
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.max_workers == 4
        assert config.output_format == "csv"
        assert config.save_plots is True
        assert config.plot_format == "png"
        assert config.plot_dpi == 300


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization with defaults."""
        config = ConfigManager()
        
        assert isinstance(config.get_model_config(), ModelConfig)
        assert isinstance(config.get_data_config(), DataConfig)
        assert isinstance(config.get_portfolio_config(), PortfolioConfig)
        assert isinstance(config.get_backtesting_config(), BacktestingConfig)
        assert isinstance(config.get_system_config(), SystemConfig)
    
    def test_config_manager_with_file(self):
        """Test ConfigManager initialization with config file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'models': {
                    'arima_max_p': 10,
                    'lstm_epochs': 200
                },
                'data': {
                    'data_path': 'custom_data/'
                }
            }, f)
            config_file = f.name
        
        try:
            config = ConfigManager(config_file)
            
            assert config.get('models', 'arima_max_p') == 10
            assert config.get('models', 'lstm_epochs') == 200
            assert config.get('data', 'data_path') == 'custom_data/'
        
        finally:
            os.unlink(config_file)
    
    def test_config_manager_environment_variables(self, monkeypatch):
        """Test ConfigManager loading from environment variables."""
        # Set environment variables
        monkeypatch.setenv('TS_DATA_PATH', 'env_data/')
        monkeypatch.setenv('TS_LOG_LEVEL', 'DEBUG')
        monkeypatch.setenv('TS_RISK_FREE_RATE', '0.03')
        monkeypatch.setenv('TS_INITIAL_CAPITAL', '200000')
        monkeypatch.setenv('TS_TRANSACTION_COST', '0.002')
        
        config = ConfigManager()
        
        assert config.get('data', 'data_path') == 'env_data/'
        assert config.get('system', 'log_level') == 'DEBUG'
        assert config.get('portfolio', 'risk_free_rate') == 0.03
        assert config.get('backtesting', 'initial_capital') == 200000
        assert config.get('backtesting', 'transaction_cost') == 0.002
    
    def test_config_manager_validation(self):
        """Test ConfigManager configuration validation."""
        config = ConfigManager()
        
        # Test invalid LSTM epochs
        with pytest.raises(ValueError, match="LSTM epochs must be positive"):
            config.set('models', 'lstm_epochs', 0)
        
        # Test invalid LSTM batch size
        with pytest.raises(ValueError, match="LSTM batch size must be positive"):
            config.set('models', 'lstm_batch_size', -1)
        
        # Test invalid risk-free rate
        with pytest.raises(ValueError, match="Risk-free rate cannot be negative"):
            config.set('portfolio', 'risk_free_rate', -0.01)
        
        # Test invalid portfolio weight
        with pytest.raises(ValueError, match="Maximum portfolio weight cannot exceed 1.0"):
            config.set('portfolio', 'max_portfolio_weight', 1.5)
    
    def test_config_manager_get_set(self):
        """Test ConfigManager get and set methods."""
        config = ConfigManager()
        
        # Test getting entire section
        model_config = config.get('models')
        assert isinstance(model_config, ModelConfig)
        
        # Test getting specific key
        assert config.get('models', 'arima_max_p') == 5
        
        # Test setting value
        config.set('models', 'arima_max_p', 15)
        assert config.get('models', 'arima_max_p') == 15
        
        # Test getting non-existent section
        with pytest.raises(KeyError, match="Unknown configuration section"):
            config.get('nonexistent_section')
        
        # Test getting non-existent key
        with pytest.raises(KeyError, match="Unknown configuration key"):
            config.get('models', 'nonexistent_key')
    
    def test_config_manager_update(self):
        """Test ConfigManager update method."""
        config = ConfigManager()
        
        updates = {
            'models': {
                'arima_max_p': 12,
                'lstm_epochs': 150
            },
            'portfolio': {
                'risk_free_rate': 0.025
            }
        }
        
        config.update(updates)
        
        assert config.get('models', 'arima_max_p') == 12
        assert config.get('models', 'lstm_epochs') == 150
        assert config.get('portfolio', 'risk_free_rate') == 0.025
    
    def test_config_manager_save_load(self):
        """Test ConfigManager save and load functionality."""
        config = ConfigManager()
        
        # Modify some values
        config.set('models', 'arima_max_p', 20)
        config.set('portfolio', 'risk_free_rate', 0.03)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name
        
        try:
            config.save_to_file(config_file)
            
            # Create new config manager and load from file
            new_config = ConfigManager(config_file)
            
            assert new_config.get('models', 'arima_max_p') == 20
            assert new_config.get('portfolio', 'risk_free_rate') == 0.03
        
        finally:
            os.unlink(config_file)
    
    def test_config_manager_directory_creation(self, tmp_path):
        """Test that ConfigManager creates necessary directories."""
        data_path = tmp_path / "test_data"
        raw_path = data_path / "raw"
        processed_path = data_path / "processed"
        cache_path = data_path / "cache"
        
        # These directories should not exist initially
        assert not data_path.exists()
        assert not raw_path.exists()
        assert not processed_path.exists()
        assert not cache_path.exists()
        
        # Create config with these paths
        config = ConfigManager()
        config.set('data', 'data_path', str(data_path))
        config.set('data', 'raw_data_path', str(raw_path))
        config.set('data', 'processed_data_path', str(processed_path))
        config.set('data', 'cache_path', str(cache_path))
        
        # Validate config (this should create directories)
        config._validate_config()
        
        # Directories should now exist
        assert data_path.exists()
        assert raw_path.exists()
        assert processed_path.exists()
        assert cache_path.exists()


class TestConfigFunctions:
    """Test configuration utility functions."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()
        
        assert config1 is config2
    
    def test_get_config_with_file(self):
        """Test get_config with specific config file."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'models': {
                    'arima_max_p': 25
                }
            }, f)
            config_file = f.name
        
        try:
            config = get_config(config_file)
            assert config.get('models', 'arima_max_p') == 25
        
        finally:
            os.unlink(config_file)
    
    def test_convenience_functions(self):
        """Test convenience functions for config access."""
        config = get_config()
        
        # Test convenience functions
        assert get_model_config() is config.get_model_config()
        assert get_data_config() is config.get_data_config()
        assert get_portfolio_config() is config.get_portfolio_config()
        assert get_backtesting_config() is config.get_backtesting_config()
        assert get_system_config() is config.get_system_config()


class TestConfigEdgeCases:
    """Test configuration edge cases and error handling."""
    
    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        # Create temporary invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_file = f.name
        
        try:
            # Should not raise exception, just log warning
            config = ConfigManager(config_file)
            
            # Should use default values
            assert config.get('models', 'arima_max_p') == 5
        
        finally:
            os.unlink(config_file)
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        config = ConfigManager("nonexistent_file.yaml")
        
        # Should use default values
        assert config.get('models', 'arima_max_p') == 5
    
    def test_invalid_environment_variables(self, monkeypatch):
        """Test handling of invalid environment variable values."""
        # Set invalid environment variables
        monkeypatch.setenv('TS_RISK_FREE_RATE', 'invalid_number')
        monkeypatch.setenv('TS_INITIAL_CAPITAL', 'not_a_number')
        
        # Should not raise exception, just log warning and use defaults
        config = ConfigManager()
        
        assert config.get('portfolio', 'risk_free_rate') == 0.02
        assert config.get('backtesting', 'initial_capital') == 100000
    
    def test_unknown_config_keys(self):
        """Test handling of unknown configuration keys."""
        config = ConfigManager()
        
        # Test setting unknown key (should log warning)
        config.set('models', 'unknown_key', 'value')
        
        # Test updating with unknown keys (should log warning)
        updates = {
            'models': {
                'unknown_key': 'value'
            }
        }
        config.update(updates)
    
    def test_config_persistence_across_instances(self):
        """Test that configuration changes persist across instances."""
        config1 = ConfigManager()
        config1.set('models', 'arima_max_p', 30)
        
        config2 = ConfigManager()
        assert config2.get('models', 'arima_max_p') == 30
