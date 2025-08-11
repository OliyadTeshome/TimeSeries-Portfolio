"""
Data Loader Module

This module provides functionality for loading and managing time series data
from various sources including CSV files, APIs, and databases.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class for loading and managing time series data from various sources.
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (str): Path to the data directory
        """
        self.data_path = data_path
        self.raw_path = os.path.join(data_path, "raw")
        self.processed_path = os.path.join(data_path, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.processed_path, exist_ok=True)
    
    def load_csv(self, filename: str, date_column: str = "date", 
                 index_col: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            filename (str): Name of the CSV file
            date_column (str): Name of the date column
            index_col (str, optional): Column to use as index
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            file_path = os.path.join(self.raw_path, filename)
            if not os.path.exists(file_path):
                file_path = os.path.join(self.processed_path, filename)
            
            data = pd.read_csv(file_path, **kwargs)
            
            # Convert date column to datetime if it exists
            if date_column in data.columns:
                data[date_column] = pd.to_datetime(data[date_column])
                if index_col is None:
                    data = data.set_index(date_column)
            
            logger.info(f"Successfully loaded {filename} with shape {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            raise
    
    def load_multiple_csvs(self, filenames: list, date_column: str = "date",
                          **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files.
        
        Args:
            filenames (list): List of CSV filenames
            date_column (str): Name of the date column
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of loaded dataframes
        """
        data_dict = {}
        for filename in filenames:
            try:
                data_dict[filename] = self.load_csv(filename, date_column, **kwargs)
            except Exception as e:
                logger.warning(f"Could not load {filename}: {str(e)}")
                continue
        
        return data_dict
    
    def save_data(self, data: pd.DataFrame, filename: str, 
                  to_processed: bool = True, **kwargs) -> None:
        """
        Save data to a file.
        
        Args:
            data (pd.DataFrame): Data to save
            filename (str): Name of the output file
            to_processed (bool): Whether to save to processed directory
            **kwargs: Additional arguments for pd.to_csv
        """
        try:
            if to_processed:
                output_path = os.path.join(self.processed_path, filename)
            else:
                output_path = os.path.join(self.raw_path, filename)
            
            data.to_csv(output_path, **kwargs)
            logger.info(f"Successfully saved {filename} to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving {filename}: {str(e)}")
            raise
    
    def get_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, Any]: Dictionary containing data information
        """
        info = {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'duplicates': data.duplicated().sum()
        }
        
        # Add date range if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            info['date_range'] = {
                'start': data.index.min(),
                'end': data.index.max(),
                'frequency': pd.infer_freq(data.index)
            }
        
        # Add numeric column statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            info['numeric_stats'] = data[numeric_cols].describe().to_dict()
        
        return info
    
    def validate_data(self, data: pd.DataFrame, required_columns: list = None,
                     date_column: str = "date") -> bool:
        """
        Validate the loaded data.
        
        Args:
            data (pd.DataFrame): Data to validate
            required_columns (list): List of required columns
            date_column (str): Name of the date column
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        # Check if data is empty
        if data.empty:
            logger.error("Data is empty")
            return False
        
        # Check required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
        
        # Check date column
        if date_column in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
                logger.error(f"Date column {date_column} is not datetime type")
                return False
        
        # Check for all-null columns
        null_cols = data.columns[data.isnull().all()].tolist()
        if null_cols:
            logger.warning(f"Columns with all null values: {null_cols}")
        
        logger.info("Data validation passed")
        return True


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for testing and demonstration purposes.
    
    Returns:
        pd.DataFrame: Sample time series data
    """
    # Generate sample time series data
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Generate sample data with trend and seasonality
    trend = np.linspace(100, 120, 252)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(252) / 252)
    noise = np.random.normal(0, 2, 252)
    
    sample_data = pd.DataFrame({
        'value': trend + seasonality + noise,
        'volume': np.random.randint(1000, 5000, 252),
        'volatility': np.random.exponential(1, 252)
    }, index=dates)
    
    return sample_data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Save sample data
    loader.save_data(sample_data, "sample_data.csv")
    
    # Load the saved data
    loaded_data = loader.load_csv("sample_data.csv")
    
    # Get data info
    info = loader.get_data_info(loaded_data)
    print("Data Information:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Validate data
    is_valid = loader.validate_data(loaded_data, required_columns=['value'])
    print(f"Data is valid: {is_valid}")
