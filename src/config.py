"""
Configuration Management Module

This module provides centralized configuration management for the TimeSeries-Portfolio
framework, including model parameters, data paths, and system settings.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for forecasting models."""
    
    # ARIMA parameters
    arima_max_p: int = 5
    arima_max_q: int = 5
    arima_max_d: int = 2
    arima_seasonal: bool = False
    arima_seasonal_periods: int = 12
    
    # LSTM parameters
    lstm_sequence_length: int = 10
    lstm_hidden_size: int = 50
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    
    # Ensemble parameters
    ensemble_method: str = 'weighted_average'
    ensemble_weights: Optional[Dict[str, float]] = None


@dataclass
class DataConfig:
    """Configuration for data handling."""
    
    data_path: str = "data/"
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    cache_path: str = "data/cache/"
    
    # Data validation
    min_data_length: int = 20
    max_missing_ratio: float = 0.1
    outlier_threshold: float = 3.0
    
    # Feature engineering
    create_lags: bool = True
    max_lags: int = 10
    create_rolling_features: bool = True
    rolling_windows: list = field(default_factory=lambda: [5, 10, 20])


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization."""
    
    risk_free_rate: float = 0.02
    optimization_method: str = 'sharpe_ratio'
    max_portfolio_weight: float = 0.4
    min_portfolio_weight: float = 0.0
    
    # Risk metrics
    var_confidence_level: float = 0.05
    cvar_confidence_level: float = 0.05
    max_drawdown_threshold: float = 0.2
    
    # Optimization constraints
    enable_short_selling: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0


@dataclass
class BacktestingConfig:
    """Configuration for backtesting."""
    
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    rebalance_frequency: int = 21
    
    # Performance metrics
    benchmark: str = 'SPY'
    risk_free_rate: float = 0.02
    
    # Strategy parameters
    momentum_lookback: int = 20
    mean_reversion_lookback: int = 20
    volatility_lookback: int = 20


@dataclass
class SystemConfig:
    """Configuration for system settings."""
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    max_workers: int = 4
    
    # Output
    output_format: str = "csv"
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 300


class ConfigManager:
    """
    Centralized configuration manager for the TimeSeries-Portfolio framework.
    
    This class provides a unified interface for accessing configuration settings
    across all modules, with support for environment variables, YAML files,
    and runtime configuration updates.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (str, optional): Path to YAML configuration file
        """
        self.config_file = config_file
        self._config: Dict[str, Any] = {}
        
        # Initialize default configurations
        self._init_default_config()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Load from environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
    
    def _init_default_config(self) -> None:
        """Initialize default configuration values."""
        self._config = {
            'models': ModelConfig(),
            'data': DataConfig(),
            'portfolio': PortfolioConfig(),
            'backtesting': BacktestingConfig(),
            'system': SystemConfig()
        }
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file (str): Path to YAML configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
            
            if file_config:
                self._update_config(file_config)
                logger.info(f"Configuration loaded from {config_file}")
        
        except Exception as e:
            logger.warning(f"Could not load configuration from {config_file}: {e}")
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        env_mappings = {
            'TS_DATA_PATH': ('data', 'data_path'),
            'TS_LOG_LEVEL': ('system', 'log_level'),
            'TS_RISK_FREE_RATE': ('portfolio', 'risk_free_rate'),
            'TS_INITIAL_CAPITAL': ('backtesting', 'initial_capital'),
            'TS_TRANSACTION_COST': ('backtesting', 'transaction_cost'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert string values to appropriate types
                    if key in ['risk_free_rate', 'transaction_cost', 'slippage']:
                        value = float(value)
                    elif key in ['initial_capital']:
                        value = float(value)
                    elif key in ['log_level']:
                        value = str(value)
                    
                    setattr(self._config[section], key, value)
                    logger.debug(f"Loaded {env_var}={value} -> {section}.{key}")
                
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not set {env_var}={value}: {e}")
    
    def _update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config (Dict[str, Any]): New configuration values
        """
        for section, section_config in new_config.items():
            if section in self._config:
                if isinstance(section_config, dict):
                    for key, value in section_config.items():
                        if hasattr(self._config[section], key):
                            setattr(self._config[section], key, value)
                        else:
                            logger.warning(f"Unknown config key: {section}.{key}")
                else:
                    logger.warning(f"Invalid section config for {section}")
    
    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate data paths
        for path_key in ['data_path', 'raw_data_path', 'processed_data_path']:
            path = getattr(self._config['data'], path_key)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")
        
        # Validate model parameters
        if self._config['models'].lstm_epochs <= 0:
            raise ValueError("LSTM epochs must be positive")
        
        if self._config['models'].lstm_batch_size <= 0:
            raise ValueError("LSTM batch size must be positive")
        
        # Validate portfolio parameters
        if self._config['portfolio'].risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")
        
        if self._config['portfolio'].max_portfolio_weight > 1:
            raise ValueError("Maximum portfolio weight cannot exceed 1.0")
        
        logger.info("Configuration validation completed successfully")
    
    def get(self, section: str, key: Optional[str] = None) -> Any:
        """
        Get configuration value(s).
        
        Args:
            section (str): Configuration section name
            key (str, optional): Specific configuration key
            
        Returns:
            Any: Configuration value or section
        """
        if section not in self._config:
            raise KeyError(f"Unknown configuration section: {section}")
        
        if key is None:
            return self._config[section]
        
        if not hasattr(self._config[section], key):
            raise KeyError(f"Unknown configuration key: {section}.{key}")
        
        return getattr(self._config[section], key)
    
    def set(self, section: str, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            section (str): Configuration section name
            key (str): Configuration key
            value (Any): Configuration value
        """
        if section not in self._config:
            raise KeyError(f"Unknown configuration section: {section}")
        
        if not hasattr(self._config[section], key):
            raise KeyError(f"Unknown configuration key: {section}.{key}")
        
        setattr(self._config[section], key, value)
        logger.debug(f"Configuration updated: {section}.{key} = {value}")
    
    def update(self, updates: Dict[str, Dict[str, Any]]) -> None:
        """
        Update multiple configuration values.
        
        Args:
            updates (Dict[str, Dict[str, Any]]): Configuration updates
        """
        self._update_config(updates)
        self._validate_config()
        logger.info("Configuration updated successfully")
    
    def save_to_file(self, config_file: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            config_file (str): Path to output YAML file
        """
        try:
            # Convert dataclasses to dictionaries
            config_dict = {}
            for section, section_config in self._config.items():
                config_dict[section] = section_config.__dict__
            
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
        
        except Exception as e:
            logger.error(f"Could not save configuration to {config_file}: {e}")
            raise
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self._config['models']
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self._config['data']
    
    def get_portfolio_config(self) -> PortfolioConfig:
        """Get portfolio configuration."""
        return self._config['portfolio']
    
    def get_backtesting_config(self) -> BacktestingConfig:
        """Get backtesting configuration."""
        return self._config['backtesting']
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration."""
        return self._config['system']


# Global configuration instance
@lru_cache(maxsize=1)
def get_config(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration instance.
    
    Args:
        config_file (str, optional): Path to configuration file
        
    Returns:
        ConfigManager: Global configuration instance
    """
    return ConfigManager(config_file)


# Convenience functions for common configuration access
def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return get_config().get_model_config()


def get_data_config() -> DataConfig:
    """Get data configuration."""
    return get_config().get_data_config()


def get_portfolio_config() -> PortfolioConfig:
    """Get portfolio configuration."""
    return get_config().get_portfolio_config()


def get_backtesting_config() -> BacktestingConfig:
    """Get backtesting configuration."""
    return get_config().get_backtesting_config()


def get_system_config() -> SystemConfig:
    """Get system configuration."""
    return get_config().get_system_config()
