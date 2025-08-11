"""
Data Preprocessing Module

This module provides functionality for cleaning, transforming, and preparing
time series data for analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any, Tuple
from scipy import stats
from scipy.signal import detrend
import logging

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    A class for preprocessing time series data.
    """
    
    def __init__(self):
        """Initialize the TimeSeriesPreprocessor."""
        pass
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            method: str = 'interpolate',
                            **kwargs) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            method (str): Method to handle missing values ('drop', 'interpolate', 'fill')
            **kwargs: Additional arguments for the chosen method
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        data_clean = data.copy()
        
        if method == 'drop':
            data_clean = data_clean.dropna()
            logger.info(f"Dropped {len(data) - len(data_clean)} rows with missing values")
            
        elif method == 'interpolate':
            data_clean = data_clean.interpolate(method='time' if isinstance(data_clean.index, pd.DatetimeIndex) else 'linear', **kwargs)
            logger.info("Interpolated missing values")
            
        elif method == 'fill':
            fill_value = kwargs.get('fill_value', 0)
            data_clean = data_clean.fillna(fill_value)
            logger.info(f"Filled missing values with {fill_value}")
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return data_clean
    
    def remove_outliers(self, data: pd.DataFrame, columns: Optional[list] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): Columns to process. If None, process all numeric columns
            method (str): Method to detect outliers ('iqr', 'zscore', 'isolation_forest')
            threshold (float): Threshold for outlier detection
            
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        data_clean = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_mask = pd.Series([False] * len(data), index=data.index)
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    col_outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[col].dropna()))
                    col_outliers = pd.Series([False] * len(data), index=data.index)
                    col_outliers[data[col].dropna().index] = z_scores > threshold
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                outliers_mask |= col_outliers
                logger.info(f"Found {col_outliers.sum()} outliers in column {col}")
        
        data_clean = data_clean[~outliers_mask]
        logger.info(f"Removed {outliers_mask.sum()} rows with outliers")
        
        return data_clean
    
    def normalize_data(self, data: pd.DataFrame, columns: Optional[list] = None,
                      method: str = 'zscore') -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize the data using various methods.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): Columns to normalize. If None, normalize all numeric columns
            method (str): Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Normalized data and normalization parameters
        """
        data_normalized = data.copy()
        normalization_params = {}
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in data.columns:
                if method == 'zscore':
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    data_normalized[col] = (data[col] - mean_val) / std_val
                    normalization_params[col] = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
                    
                elif method == 'minmax':
                    min_val = data[col].min()
                    max_val = data[col].max()
                    data_normalized[col] = (data[col] - min_val) / (max_val - min_val)
                    normalization_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
                    
                elif method == 'robust':
                    median_val = data[col].median()
                    mad_val = stats.median_abs_deviation(data[col].dropna())
                    data_normalized[col] = (data[col] - median_val) / mad_val
                    normalization_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Normalized {len(columns)} columns using {method} method")
        return data_normalized, normalization_params
    
    def detrend_data(self, data: pd.DataFrame, columns: Optional[list] = None,
                     method: str = 'linear') -> pd.DataFrame:
        """
        Remove trend from the data.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): Columns to detrend. If None, detrend all numeric columns
            method (str): Detrending method ('linear', 'polynomial', 'spline')
            
        Returns:
            pd.DataFrame: Detrended data
        """
        data_detrended = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in data.columns:
                if method == 'linear':
                    trend = np.polyfit(range(len(data)), data[col], 1)
                    trend_line = np.polyval(trend, range(len(data)))
                    data_detrended[col] = data[col] - trend_line
                    
                elif method == 'polynomial':
                    degree = 2  # Can be made configurable
                    trend = np.polyfit(range(len(data)), data[col], degree)
                    trend_line = np.polyval(trend, range(len(data)))
                    data_detrended[col] = data[col] - trend_line
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Detrended {len(columns)} columns using {method} method")
        return data_detrended
    
    def create_lags(self, data: pd.DataFrame, columns: Optional[list] = None,
                    lag_periods: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): Columns to create lags for. If None, use all numeric columns
            lag_periods (list): List of lag periods to create
            
        Returns:
            pd.DataFrame: Data with lagged features
        """
        data_with_lags = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in data.columns:
                for lag in lag_periods:
                    lag_col_name = f"{col}_lag_{lag}"
                    data_with_lags[lag_col_name] = data[col].shift(lag)
        
        # Remove rows with NaN values from lag creation
        data_with_lags = data_with_lags.dropna()
        
        logger.info(f"Created lagged features for {len(columns)} columns with {len(lag_periods)} lag periods")
        return data_with_lags
    
    def create_rolling_features(self, data: pd.DataFrame, columns: Optional[list] = None,
                               windows: list = [5, 10, 20], functions: list = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            data (pd.DataFrame): Input data
            columns (list, optional): Columns to create rolling features for
            windows (list): List of window sizes
            functions (list): List of functions to apply ('mean', 'std', 'min', 'max', 'sum')
            
        Returns:
            pd.DataFrame: Data with rolling features
        """
        data_with_rolling = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in data.columns:
                for window in windows:
                    for func in functions:
                        if hasattr(data[col].rolling(window), func):
                            rolling_feature = getattr(data[col].rolling(window), func)()
                            feature_name = f"{col}_rolling_{func}_{window}"
                            data_with_rolling[feature_name] = rolling_feature
        
        # Remove rows with NaN values from rolling features
        data_with_rolling = data_with_rolling.dropna()
        
        logger.info(f"Created rolling features for {len(columns)} columns with {len(windows)} window sizes")
        return data_with_rolling
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8,
                   val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data (pd.DataFrame): Input data
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            test_ratio (float): Proportion of data for testing
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test sets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        logger.info(f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data
    
    def get_preprocessing_summary(self, original_data: pd.DataFrame, 
                                processed_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing operations.
        
        Args:
            original_data (pd.DataFrame): Original data
            processed_data (pd.DataFrame): Processed data
            
        Returns:
            Dict[str, Any]: Summary of preprocessing operations
        """
        summary = {
            'original_shape': original_data.shape,
            'processed_shape': processed_data.shape,
            'rows_removed': len(original_data) - len(processed_data),
            'columns_added': len(processed_data.columns) - len(original_data.columns),
            'missing_values_original': original_data.isnull().sum().sum(),
            'missing_values_processed': processed_data.isnull().sum().sum(),
            'duplicates_original': original_data.duplicated().sum(),
            'duplicates_processed': processed_data.duplicated().sum()
        }
        
        return summary


def preprocess_pipeline(data: pd.DataFrame, 
                       handle_missing: bool = True,
                       remove_outliers: bool = True,
                       normalize: bool = False,
                       detrend: bool = False,
                       create_lags: bool = False,
                       create_rolling: bool = False,
                       **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline.
    
    Args:
        data (pd.DataFrame): Input data
        handle_missing (bool): Whether to handle missing values
        remove_outliers (bool): Whether to remove outliers
        normalize (bool): Whether to normalize data
        detrend (bool): Whether to detrend data
        create_lags (bool): Whether to create lag features
        create_rolling (bool): Whether to create rolling features
        **kwargs: Additional arguments for preprocessing methods
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Processed data and preprocessing summary
    """
    preprocessor = TimeSeriesPreprocessor()
    processed_data = data.copy()
    preprocessing_summary = {}
    
    # Handle missing values
    if handle_missing:
        processed_data = preprocessor.handle_missing_values(processed_data, **kwargs)
    
    # Remove outliers
    if remove_outliers:
        processed_data = preprocessor.remove_outliers(processed_data, **kwargs)
    
    # Detrend data
    if detrend:
        processed_data = preprocessor.detrend_data(processed_data, **kwargs)
    
    # Create lag features
    if create_lags:
        processed_data = preprocessor.create_lags(processed_data, **kwargs)
    
    # Create rolling features
    if create_rolling:
        processed_data = preprocessor.create_rolling_features(processed_data, **kwargs)
    
    # Normalize data (do this last)
    if normalize:
        processed_data, norm_params = preprocessor.normalize_data(processed_data, **kwargs)
        preprocessing_summary['normalization_params'] = norm_params
    
    # Get preprocessing summary
    preprocessing_summary.update(preprocessor.get_preprocessing_summary(data, processed_data))
    
    logger.info("Preprocessing pipeline completed successfully")
    return processed_data, preprocessing_summary


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    print("Original data shape:", sample_data.shape)
    
    # Run preprocessing pipeline
    processed_data, summary = preprocess_pipeline(
        sample_data,
        handle_missing=True,
        remove_outliers=True,
        normalize=True,
        create_lags=True,
        create_rolling=True
    )
    
    print("Processed data shape:", processed_data.shape)
    print("Preprocessing summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
