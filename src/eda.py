"""
Exploratory Data Analysis Module

This module provides comprehensive tools for analyzing time series data
including statistical summaries, visualizations, and pattern detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TimeSeriesEDA:
    """
    A comprehensive class for exploratory data analysis of time series data.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the TimeSeriesEDA with data.
        
        Args:
            data (pd.DataFrame): Time series data to analyze
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def basic_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistical measures for the dataset.
        
        Returns:
            Dict[str, Any]: Dictionary containing basic statistics
        """
        stats_dict = {
            'data_shape': self.data.shape,
            'data_types': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        # Add date range if index is datetime
        if isinstance(self.data.index, pd.DatetimeIndex):
            stats_dict['date_range'] = {
                'start': self.data.index.min(),
                'end': self.data.index.max(),
                'frequency': pd.infer_freq(self.data.index),
                'total_days': (self.data.index.max() - self.data.index.min()).days
            }
        
        # Add numeric column statistics
        if self.numeric_columns:
            stats_dict['numeric_summary'] = self.data[self.numeric_columns].describe().to_dict()
            
            # Add skewness and kurtosis
            stats_dict['skewness'] = self.data[self.numeric_columns].skew().to_dict()
            stats_dict['kurtosis'] = self.data[self.numeric_columns].kurtosis().to_dict()
        
        return stats_dict
    
    def stationarity_tests(self, columns: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Perform stationarity tests on time series data.
        
        Args:
            columns (List[str], optional): Columns to test. If None, test all numeric columns
            
        Returns:
            Dict[str, Dict]: Dictionary containing test results for each column
        """
        if columns is None:
            columns = self.numeric_columns
        
        results = {}
        
        for col in columns:
            if col in self.data.columns:
                series = self.data[col].dropna()
                if len(series) > 0:
                    col_results = {}
                    
                    # Augmented Dickey-Fuller test
                    try:
                        adf_result = adfuller(series)
                        col_results['adf'] = {
                            'statistic': adf_result[0],
                            'p_value': adf_result[1],
                            'critical_values': adf_result[4],
                            'is_stationary': adf_result[1] < 0.05
                        }
                    except:
                        col_results['adf'] = {'error': 'Could not compute ADF test'}
                    
                    # KPSS test
                    try:
                        kpss_result = kpss(series)
                        col_results['kpss'] = {
                            'statistic': kpss_result[0],
                            'p_value': kpss_result[1],
                            'critical_values': kpss_result[3],
                            'is_stationary': kpss_result[1] > 0.05
                        }
                    except:
                        col_results['kpss'] = {'error': 'Could not compute KPSS test'}
                    
                    results[col] = col_results
        
        return results
    
    def seasonal_decomposition(self, columns: Optional[List[str]] = None, 
                             period: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform seasonal decomposition of time series.
        
        Args:
            columns (List[str], optional): Columns to decompose. If None, use all numeric columns
            period (int, optional): Period for seasonal decomposition. If None, auto-detect
            
        Returns:
            Dict[str, Any]: Dictionary containing decomposition results
        """
        if columns is None:
            columns = self.numeric_columns
        
        if period is None:
            # Try to infer period from data
            if isinstance(self.data.index, pd.DatetimeIndex):
                freq = pd.infer_freq(self.data.index)
                if freq == 'D':
                    period = 7  # Weekly seasonality for daily data
                elif freq == 'M':
                    period = 12  # Monthly seasonality for monthly data
                else:
                    period = 12  # Default to 12
        
        decompositions = {}
        
        for col in columns:
            if col in self.data.columns:
                series = self.data[col].dropna()
                if len(series) > period * 2:
                    try:
                        decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
                        decompositions[col] = {
                            'trend': decomposition.trend,
                            'seasonal': decomposition.seasonal,
                            'residual': decomposition.resid,
                            'period': period
                        }
                    except Exception as e:
                        logger.warning(f"Could not decompose {col}: {str(e)}")
                        continue
        
        return decompositions
    
    def correlation_analysis(self, method: str = 'pearson') -> Dict[str, Any]:
        """
        Perform correlation analysis on numeric columns.
        
        Args:
            method (str): Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict[str, Any]: Dictionary containing correlation results
        """
        if not self.numeric_columns:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = self.data[self.numeric_columns].corr(method=method)
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'column1': corr_matrix.columns[i],
                        'column2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'method': method
        }
    
    def autocorrelation_analysis(self, columns: Optional[List[str]] = None, 
                                max_lags: int = 40) -> Dict[str, Any]:
        """
        Perform autocorrelation analysis.
        
        Args:
            columns (List[str], optional): Columns to analyze. If None, use all numeric columns
            max_lags (int): Maximum number of lags to compute
            
        Returns:
            Dict[str, Any]: Dictionary containing autocorrelation results
        """
        if columns is None:
            columns = self.numeric_columns
        
        acf_results = {}
        
        for col in columns:
            if col in self.data.columns:
                series = self.data[col].dropna()
                if len(series) > max_lags:
                    try:
                        # Calculate ACF
                        acf_values = pd.Series(series).autocorr(lag=1)
                        acf_results[col] = {
                            'lag_1_autocorr': acf_values,
                            'series_length': len(series)
                        }
                    except Exception as e:
                        logger.warning(f"Could not compute ACF for {col}: {str(e)}")
                        continue
        
        return acf_results
    
    def create_summary_plots(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive summary plots for the dataset.
        
        Args:
            save_path (str, optional): Path to save plots. If None, display plots
        """
        n_cols = len(self.numeric_columns)
        if n_cols == 0:
            logger.warning("No numeric columns found for plotting")
            return
        
        # Calculate grid dimensions
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(self.numeric_columns):
            row = i // 2
            col_idx = i % 2
            
            # Time series plot
            axes[row, col_idx].plot(self.data.index, self.data[col], linewidth=1)
            axes[row, col_idx].set_title(f'{col} - Time Series')
            axes[row, col_idx].set_xlabel('Date')
            axes[row, col_idx].set_ylabel('Value')
            axes[row, col_idx].tick_params(axis='x', rotation=45)
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, n_rows * 2):
            row = i // 2
            col_idx = i % 2
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        else:
            plt.show()
    
    def create_distribution_plots(self, save_path: Optional[str] = None) -> None:
        """
        Create distribution plots for numeric columns.
        
        Args:
            save_path (str, optional): Path to save plots. If None, display plots
        """
        if not self.numeric_columns:
            logger.warning("No numeric columns found for plotting")
            return
        
        n_cols = len(self.numeric_columns)
        n_rows = (n_cols + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(self.numeric_columns):
            row = i // 2
            col_idx = i % 2
            
            # Histogram with KDE
            axes[row, col_idx].hist(self.data[col].dropna(), bins=30, alpha=0.7, 
                                   edgecolor='black', density=True)
            axes[row, col_idx].set_title(f'{col} - Distribution')
            axes[row, col_idx].set_xlabel('Value')
            axes[row, col_idx].set_ylabel('Density')
            axes[row, col_idx].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_cols, n_rows * 2):
            row = i // 2
            col_idx = i % 2
            axes[row, col_idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Distribution plots saved to {save_path}")
        else:
            plt.show()
    
    def create_correlation_heatmap(self, save_path: Optional[str] = None) -> None:
        """
        Create correlation heatmap for numeric columns.
        
        Args:
            save_path (str, optional): Path to save plot. If None, display plot
        """
        if not self.numeric_columns:
            logger.warning("No numeric columns found for plotting")
            return
        
        corr_matrix = self.data[self.numeric_columns].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()
    
    def create_decomposition_plots(self, columns: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Create seasonal decomposition plots.
        
        Args:
            columns (List[str], optional): Columns to plot. If None, use all numeric columns
            save_path (str, optional): Path to save plots. If None, display plots
        """
        if columns is None:
            columns = self.numeric_columns
        
        decompositions = self.seasonal_decomposition(columns)
        
        for col, decomp in decompositions.items():
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
            
            # Original series
            ax1.plot(self.data.index, self.data[col])
            ax1.set_title(f'{col} - Original Series')
            ax1.grid(True, alpha=0.3)
            
            # Trend
            ax2.plot(self.data.index, decomp['trend'])
            ax2.set_title(f'{col} - Trend')
            ax2.grid(True, alpha=0.3)
            
            # Seasonal
            ax3.plot(self.data.index, decomp['seasonal'])
            ax3.set_title(f'{col} - Seasonal')
            ax3.grid(True, alpha=0.3)
            
            # Residual
            ax4.plot(self.data.index, decomp['residual'])
            ax4.set_title(f'{col} - Residual')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                col_save_path = save_path.replace('.png', f'_{col}.png')
                plt.savefig(col_save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Decomposition plots for {col} saved to {col_save_path}")
            else:
                plt.show()
    
    def generate_eda_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive EDA report.
        
        Args:
            save_path (str, optional): Path to save the report. If None, return report as string
            
        Returns:
            str: EDA report
        """
        report = []
        report.append("# Exploratory Data Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Basic statistics
        report.append("## Basic Statistics")
        basic_stats = self.basic_statistics()
        for key, value in basic_stats.items():
            if key != 'numeric_summary':  # Skip detailed numeric summary
                report.append(f"- **{key}**: {value}")
        report.append("")
        
        # Stationarity tests
        report.append("## Stationarity Tests")
        stationarity_results = self.stationarity_tests()
        for col, results in stationarity_results.items():
            report.append(f"### {col}")
            if 'adf' in results and 'is_stationary' in results['adf']:
                report.append(f"- ADF Test: {'Stationary' if results['adf']['is_stationary'] else 'Non-stationary'}")
                report.append(f"- ADF p-value: {results['adf']['p_value']:.4f}")
            if 'kpss' in results and 'is_stationary' in results['kpss']:
                report.append(f"- KPSS Test: {'Stationary' if results['kpss']['is_stationary'] else 'Non-stationary'}")
                report.append(f"- KPSS p-value: {results['kpss']['p_value']:.4f}")
            report.append("")
        
        # Correlation analysis
        report.append("## Correlation Analysis")
        corr_results = self.correlation_analysis()
        if 'high_correlation_pairs' in corr_results:
            report.append("### High Correlation Pairs (|r| > 0.7)")
            for pair in corr_results['high_correlation_pairs']:
                report.append(f"- {pair['column1']} vs {pair['column2']}: {pair['correlation']:.4f}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("Based on the analysis, consider the following:")
        
        # Check for non-stationary series
        non_stationary = []
        for col, results in stationarity_results.items():
            if 'adf' in results and 'is_stationary' in results['adf']:
                if not results['adf']['is_stationary']:
                    non_stationary.append(col)
        
        if non_stationary:
            report.append(f"- **Non-stationary series detected**: {', '.join(non_stationary)}")
            report.append("  - Consider differencing or transformation")
            report.append("  - Use ARIMA models with appropriate differencing")
        
        # Check for high correlations
        if 'high_correlation_pairs' in corr_results and corr_results['high_correlation_pairs']:
            report.append("- **High correlations detected**: Consider feature selection to avoid multicollinearity")
        
        # Check for missing values
        missing_cols = [col for col, missing in basic_stats['missing_values'].items() if missing > 0]
        if missing_cols:
            report.append(f"- **Missing values detected**: {', '.join(missing_cols)}")
            report.append("  - Consider imputation strategies")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"EDA report saved to {save_path}")
        
        return report_text


def run_complete_eda(data: pd.DataFrame, output_dir: str = "reports/figures/") -> Dict[str, Any]:
    """
    Run complete exploratory data analysis.
    
    Args:
        data (pd.DataFrame): Input data
        output_dir (str): Directory to save outputs
        
    Returns:
        Dict[str, Any]: Dictionary containing all EDA results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize EDA
    eda = TimeSeriesEDA(data)
    
    # Run all analyses
    results = {
        'basic_statistics': eda.basic_statistics(),
        'stationarity_tests': eda.stationarity_tests(),
        'seasonal_decomposition': eda.seasonal_decomposition(),
        'correlation_analysis': eda.correlation_analysis(),
        'autocorrelation_analysis': eda.autocorrelation_analysis()
    }
    
    # Generate plots
    eda.create_summary_plots(os.path.join(output_dir, 'summary_plots.png'))
    eda.create_distribution_plots(os.path.join(output_dir, 'distribution_plots.png'))
    eda.create_correlation_heatmap(os.path.join(output_dir, 'correlation_heatmap.png'))
    eda.create_decomposition_plots(save_path=os.path.join(output_dir, 'decomposition_plots.png'))
    
    # Generate report
    report_path = os.path.join(output_dir, 'eda_report.md')
    eda.generate_eda_report(report_path)
    
    logger.info("Complete EDA analysis finished")
    return results


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    print("Sample data loaded, running EDA...")
    
    # Run complete EDA
    results = run_complete_eda(sample_data)
    
    print("EDA completed successfully!")
    print(f"Basic statistics: {len(results['basic_statistics'])} metrics computed")
    print(f"Stationarity tests: {len(results['stationarity_tests'])} columns tested")
    print(f"Correlation analysis: {len(results['correlation_analysis'])} correlations found")
