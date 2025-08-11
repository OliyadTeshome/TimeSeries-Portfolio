"""
ARIMA Model Module

This module provides implementation of ARIMA (AutoRegressive Integrated Moving Average)
models for time series forecasting with automatic parameter selection and validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    A comprehensive ARIMA model implementation for time series forecasting.
    """
    
    def __init__(self, data: pd.Series):
        """
        Initialize the ARIMA model.
        
        Args:
            data (pd.Series): Time series data
        """
        self.data = data
        self.model = None
        self.fitted_model = None
        self.forecast = None
        self.forecast_ci = None
        self.model_params = {}
        self.diagnostics = {}
        
        # Validate data
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        if len(data) < 10:
            raise ValueError("Data must have at least 10 observations")
    
    def check_stationarity(self, method: str = 'adf') -> Dict[str, Any]:
        """
        Check if the time series is stationary.
        
        Args:
            method (str): Method to use ('adf' or 'kpss')
            
        Returns:
            Dict[str, Any]: Stationarity test results
        """
        series = self.data.dropna()
        
        if method == 'adf':
            result = adfuller(series)
            is_stationary = result[1] < 0.05
            test_name = 'Augmented Dickey-Fuller'
        elif method == 'kpss':
            result = kpss(series)
            is_stationary = result[1] > 0.05
            test_name = 'KPSS'
        else:
            raise ValueError("Method must be 'adf' or 'kpss'")
        
        return {
            'test_name': test_name,
            'statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4] if method == 'adf' else result[3],
            'is_stationary': is_stationary,
            'method': method
        }
    
    def find_optimal_differencing(self, max_diff: int = 2) -> int:
        """
        Find the optimal number of differences to make the series stationary.
        
        Args:
            max_diff (int): Maximum number of differences to try
            
        Returns:
            int: Optimal number of differences
        """
        series = self.data.copy()
        optimal_diff = 0
        
        for d in range(max_diff + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
            
            if len(test_series) < 10:
                break
            
            # Perform ADF test
            try:
                adf_result = adfuller(test_series)
                if adf_result[1] < 0.05:  # p-value < 0.05 indicates stationarity
                    optimal_diff = d
                    break
            except:
                continue
        
        logger.info(f"Optimal differencing order: {optimal_diff}")
        return optimal_diff
    
    def find_optimal_parameters(self, max_p: int = 5, max_q: int = 5, 
                               max_d: int = 2, seasonal: bool = False,
                               seasonal_periods: int = 12) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA parameters using grid search with AIC.
        
        Args:
            max_p (int): Maximum AR order
            max_q (int): Maximum MA order
            max_d (int): Maximum differencing order
            seasonal (bool): Whether to use seasonal ARIMA
            seasonal_periods (int): Seasonal period for SARIMA
            
        Returns:
            Tuple[int, int, int]: Optimal (p, d, q) parameters
        """
        best_aic = np.inf
        best_params = (0, 0, 0)
        
        # Determine differencing order
        if max_d > 0:
            d = self.find_optimal_differencing(max_d)
        else:
            d = 0
        
        # Grid search for p and q
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    if seasonal:
                        model = ARIMA(self.data, order=(p, d, q), 
                                    seasonal_order=(p, d, q, seasonal_periods))
                    else:
                        model = ARIMA(self.data, order=(p, d, q))
                    
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        
                except:
                    continue
        
        logger.info(f"Optimal parameters: p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
        logger.info(f"Best AIC: {best_aic:.2f}")
        
        return best_params
    
    def fit_model(self, order: Tuple[int, int, int], 
                  seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                  **kwargs) -> 'ARIMAModel':
        """
        Fit the ARIMA model with specified parameters.
        
        Args:
            order (Tuple[int, int, int]): (p, d, q) parameters
            seasonal_order (Tuple[int, int, int, int], optional): Seasonal parameters
            **kwargs: Additional arguments for ARIMA model
            
        Returns:
            ARIMAModel: Self for method chaining
        """
        try:
            if seasonal_order:
                self.model = ARIMA(self.data, order=order, seasonal_order=seasonal_order)
            else:
                self.model = ARIMA(self.data, order=order)
            
            self.fitted_model = self.model.fit(**kwargs)
            self.model_params = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'hqic': self.fitted_model.hqic
            }
            
            logger.info(f"ARIMA{order} model fitted successfully")
            if seasonal_order:
                logger.info(f"Seasonal ARIMA{seasonal_order} model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
        
        return self
    
    def auto_fit(self, max_p: int = 5, max_q: int = 5, max_d: int = 2,
                 seasonal: bool = False, seasonal_periods: int = 12,
                 **kwargs) -> 'ARIMAModel':
        """
        Automatically find optimal parameters and fit the model.
        
        Args:
            max_p (int): Maximum AR order
            max_q (int): Maximum MA order
            max_d (int): Maximum differencing order
            seasonal (bool): Whether to use seasonal ARIMA
            seasonal_periods (int): Seasonal period for SARIMA
            **kwargs: Additional arguments for ARIMA model
            
        Returns:
            ARIMAModel: Self for method chaining
        """
        # Find optimal parameters
        optimal_params = self.find_optimal_parameters(
            max_p, max_q, max_d, seasonal, seasonal_periods
        )
        
        # Fit model with optimal parameters
        if seasonal:
            seasonal_order = (optimal_params[0], optimal_params[1], optimal_params[2], seasonal_periods)
            return self.fit_model(optimal_params, seasonal_order, **kwargs)
        else:
            return self.fit_model(optimal_params, **kwargs)
    
    def forecast_future(self, steps: int, alpha: float = 0.05) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Generate future forecasts with confidence intervals.
        
        Args:
            steps (int): Number of steps to forecast
            alpha (float): Significance level for confidence intervals
            
        Returns:
            Tuple[pd.Series, pd.DataFrame]: Forecast values and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        try:
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            
            # Extract forecast values and confidence intervals
            self.forecast = forecast_result.predicted_mean
            self.forecast_ci = forecast_result.conf_int(alpha=alpha)
            
            logger.info(f"Generated {steps} step forecast")
            
            return self.forecast, self.forecast_ci
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate_model(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model performance on test data.
        
        Args:
            test_data (pd.Series): Test data for evaluation
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Generate in-sample predictions
        predictions = self.fitted_model.predict(
            start=test_data.index[0],
            end=test_data.index[-1]
        )
        
        # Calculate metrics
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        logger.info("Model evaluation completed")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def perform_diagnostics(self) -> Dict[str, Any]:
        """
        Perform model diagnostics including residual analysis.
        
        Returns:
            Dict[str, Any]: Diagnostic results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before diagnostics")
        
        residuals = self.fitted_model.resid
        
        # Basic residual statistics
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }
        
        # Ljung-Box test for autocorrelation
        try:
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            lb_result = {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1],
                'is_white_noise': lb_test['lb_pvalue'].iloc[-1] > 0.05
            }
        except:
            lb_result = {'error': 'Could not compute Ljung-Box test'}
        
        # Normality test
        try:
            from scipy.stats import shapiro
            shapiro_stat, shapiro_p = shapiro(residuals.dropna())
            normality_test = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        except:
            normality_test = {'error': 'Could not compute normality test'}
        
        self.diagnostics = {
            'residual_stats': residual_stats,
            'ljung_box_test': lb_result,
            'normality_test': normality_test
        }
        
        logger.info("Model diagnostics completed")
        return self.diagnostics
    
    def plot_diagnostics(self, save_path: Optional[str] = None) -> None:
        """
        Create diagnostic plots for the fitted model.
        
        Args:
            save_path (str, optional): Path to save plots
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before creating diagnostic plots")
        
        try:
            # Create diagnostic plots
            fig = self.fitted_model.plot_diagnostics(figsize=(15, 10))
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Diagnostic plots saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create diagnostic plots: {str(e)}")
    
    def plot_forecast(self, save_path: Optional[str] = None) -> None:
        """
        Plot the original data and forecast.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if self.forecast is None:
            raise ValueError("Forecast must be generated before plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 8))
            
            # Plot original data
            plt.plot(self.data.index, self.data.values, label='Original Data', linewidth=2)
            
            # Plot forecast
            plt.plot(self.forecast.index, self.forecast.values, 
                    label='Forecast', linewidth=2, linestyle='--')
            
            # Plot confidence intervals
            if self.forecast_ci is not None:
                plt.fill_between(self.forecast.index, 
                               self.forecast_ci.iloc[:, 0], 
                               self.forecast_ci.iloc[:, 1], 
                               alpha=0.3, label='95% Confidence Interval')
            
            plt.title('ARIMA Model Forecast')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecast plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create forecast plot: {str(e)}")
    
    def get_model_summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns:
            str: Model summary
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        
        return str(self.fitted_model.summary())
    
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before saving")
        
        try:
            import pickle
            model_data = {
                'fitted_model': self.fitted_model,
                'model_params': self.model_params,
                'diagnostics': self.diagnostics,
                'forecast': self.forecast,
                'forecast_ci': self.forecast_ci
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str, data: pd.Series) -> 'ARIMAModel':
        """
        Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model
            data (pd.Series): Original data series
            
        Returns:
            ARIMAModel: Loaded model instance
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create new instance
            instance = cls(data)
            instance.fitted_model = model_data['fitted_model']
            instance.model_params = model_data['model_params']
            instance.diagnostics = model_data['diagnostics']
            instance.forecast = model_data['forecast']
            instance.forecast_ci = model_data['forecast_ci']
            
            logger.info(f"Model loaded from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def auto_arima(data: pd.Series, max_p: int = 5, max_q: int = 5, max_d: int = 2,
                seasonal: bool = False, seasonal_periods: int = 12,
                **kwargs) -> ARIMAModel:
    """
    Convenience function to automatically fit an ARIMA model.
    
    Args:
        data (pd.Series): Time series data
        max_p (int): Maximum AR order
        max_q (int): Maximum MA order
        max_d (int): Maximum differencing order
        seasonal (bool): Whether to use seasonal ARIMA
        seasonal_periods (int): Seasonal period for SARIMA
        **kwargs: Additional arguments for ARIMA model
        
    Returns:
        ARIMAModel: Fitted ARIMA model
    """
    model = ARIMAModel(data)
    return model.auto_fit(max_p, max_q, max_d, seasonal, seasonal_periods, **kwargs)


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    ts_data = sample_data['value']
    
    print("Sample data loaded, fitting ARIMA model...")
    
    # Auto-fit ARIMA model
    arima_model = auto_arima(ts_data, max_p=3, max_q=3, max_d=1)
    
    # Generate forecast
    forecast, ci = arima_model.forecast_future(steps=30)
    
    # Evaluate model (using last 20% as test)
    test_size = int(len(ts_data) * 0.2)
    test_data = ts_data.iloc[-test_size:]
    metrics = arima_model.evaluate_model(test_data)
    
    # Perform diagnostics
    diagnostics = arima_model.perform_diagnostics()
    
    print("ARIMA model analysis completed!")
    print(f"Model parameters: {arima_model.model_params}")
    print(f"Forecast generated for {len(forecast)} steps")
    print(f"Model performance: {metrics}")
    
    # Save model
    arima_model.save_model('data/processed/arima_model.pkl')
