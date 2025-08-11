"""
Forecasting Module

This module provides a unified interface for time series forecasting using
multiple models (ARIMA, LSTM) with ensemble methods and model comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import logging

from .arima_model import ARIMAModel, auto_arima
from .lstm_model import LSTMModel, auto_lstm

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ForecastingEngine:
    """
    A comprehensive forecasting engine that combines multiple models.
    """
    
    def __init__(self, data: pd.Series):
        """
        Initialize the forecasting engine.
        
        Args:
            data (pd.Series): Time series data
        """
        self.data = data
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        self.ensemble_forecast = None
        
        # Validate data
        if not isinstance(data, pd.Series):
            raise ValueError("Data must be a pandas Series")
        if len(data) < 20:
            raise ValueError("Data must have at least 20 observations")
    
    def add_arima_model(self, model_name: str = 'arima', 
                        max_p: int = 5, max_q: int = 5, max_d: int = 2,
                        seasonal: bool = False, seasonal_periods: int = 12,
                        **kwargs) -> 'ForecastingEngine':
        """
        Add an ARIMA model to the forecasting engine.
        
        Args:
            model_name (str): Name for the model
            max_p (int): Maximum AR order
            max_q (int): Maximum MA order
            max_d (int): Maximum differencing order
            seasonal (bool): Whether to use seasonal ARIMA
            seasonal_periods (int): Seasonal period for SARIMA
            **kwargs: Additional arguments for ARIMA model
            
        Returns:
            ForecastingEngine: Self for method chaining
        """
        try:
            logger.info(f"Adding ARIMA model: {model_name}")
            arima_model = auto_arima(
                self.data, max_p, max_q, max_d, seasonal, seasonal_periods, **kwargs
            )
            self.models[model_name] = arima_model
            logger.info(f"ARIMA model {model_name} added successfully")
            
        except Exception as e:
            logger.error(f"Error adding ARIMA model {model_name}: {str(e)}")
            raise
        
        return self
    
    def add_lstm_model(self, model_name: str = 'lstm', 
                       sequence_length: int = 10, lstm_units: List[int] = [50, 50],
                       epochs: int = 100, **kwargs) -> 'ForecastingEngine':
        """
        Add an LSTM model to the forecasting engine.
        
        Args:
            model_name (str): Name for the model
            sequence_length (int): Length of input sequences
            lstm_units (List[int]): Number of units in each LSTM layer
            epochs (int): Maximum number of training epochs
            **kwargs: Additional arguments for LSTM model
            
        Returns:
            ForecastingEngine: Self for method chaining
        """
        try:
            logger.info(f"Adding LSTM model: {model_name}")
            lstm_model = auto_lstm(
                self.data, sequence_length, lstm_units, epochs, **kwargs
            )
            self.models[model_name] = lstm_model
            logger.info(f"LSTM model {model_name} added successfully")
            
        except Exception as e:
            logger.error(f"Error adding LSTM model {model_name}: {str(e)}")
            raise
        
        return self
    
    def generate_forecasts(self, steps: int, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Generate forecasts from all models.
        
        Args:
            steps (int): Number of steps to forecast
            alpha (float): Significance level for confidence intervals
            
        Returns:
            Dict[str, Any]: Dictionary containing forecasts from all models
        """
        if not self.models:
            raise ValueError("No models have been added to the engine")
        
        logger.info(f"Generating {steps} step forecasts from {len(self.models)} models")
        
        for model_name, model in self.models.items():
            try:
                if isinstance(model, ARIMAModel):
                    forecast, ci = model.forecast_future(steps, alpha)
                    self.forecasts[model_name] = {
                        'forecast': forecast,
                        'confidence_intervals': ci,
                        'model_type': 'ARIMA'
                    }
                elif isinstance(model, LSTMModel):
                    forecast = model.forecast_future(steps)
                    self.forecasts[model_name] = {
                        'forecast': forecast,
                        'confidence_intervals': None,
                        'model_type': 'LSTM'
                    }
                else:
                    logger.warning(f"Unknown model type for {model_name}")
                    continue
                
                logger.info(f"Forecast generated for {model_name}")
                
            except Exception as e:
                logger.error(f"Error generating forecast for {model_name}: {str(e)}")
                continue
        
        return self.forecasts
    
    def create_ensemble_forecast(self, method: str = 'weighted_average',
                                weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create an ensemble forecast from all models.
        
        Args:
            method (str): Ensemble method ('simple_average', 'weighted_average', 'median')
            weights (Dict[str, float], optional): Weights for weighted average
            
        Returns:
            pd.Series: Ensemble forecast
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Generate forecasts first.")
        
        logger.info(f"Creating ensemble forecast using {method} method")
        
        # Collect all forecasts
        all_forecasts = []
        model_names = []
        
        for model_name, forecast_data in self.forecasts.items():
            if forecast_data['forecast'] is not None:
                all_forecasts.append(forecast_data['forecast'])
                model_names.append(model_name)
        
        if not all_forecasts:
            raise ValueError("No valid forecasts available for ensemble")
        
        # Create ensemble based on method
        if method == 'simple_average':
            ensemble_values = np.mean(all_forecasts, axis=0)
        elif method == 'weighted_average':
            if weights is None:
                # Equal weights if none provided
                weights = {name: 1.0/len(model_names) for name in model_names}
            
            # Ensure weights sum to 1
            weight_sum = sum(weights.values())
            normalized_weights = {name: weight/weight_sum for name, weight in weights.items()}
            
            ensemble_values = np.zeros_like(all_forecasts[0])
            for i, model_name in enumerate(model_names):
                ensemble_values += normalized_weights[model_name] * all_forecasts[i]
        
        elif method == 'median':
            ensemble_values = np.median(all_forecasts, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Create ensemble forecast series
        if isinstance(all_forecasts[0], pd.Series):
            self.ensemble_forecast = pd.Series(ensemble_values, index=all_forecasts[0].index)
        else:
            # Create default index if not a pandas series
            self.ensemble_forecast = pd.Series(ensemble_values)
        
        logger.info(f"Ensemble forecast created using {method} method")
        return self.ensemble_forecast
    
    def evaluate_models(self, test_data: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on test data.
        
        Args:
            test_data (pd.Series): Test data for evaluation
            
        Returns:
            Dict[str, Dict[str, float]]: Performance metrics for all models
        """
        if not self.models:
            raise ValueError("No models have been added to the engine")
        
        logger.info("Evaluating all models on test data")
        
        for model_name, model in self.models.items():
            try:
                if isinstance(model, ARIMAModel):
                    metrics = model.evaluate_model(test_data)
                elif isinstance(model, LSTMModel):
                    # For LSTM, we need to use the test predictions that were already generated
                    if hasattr(model, 'y_test_actual') and hasattr(model, 'test_predictions'):
                        metrics = model._calculate_metrics(model.y_test_actual, model.test_predictions)
                    else:
                        logger.warning(f"LSTM model {model_name} test predictions not available")
                        continue
                else:
                    logger.warning(f"Unknown model type for {model_name}")
                    continue
                
                self.performance_metrics[model_name] = metrics
                logger.info(f"Evaluation completed for {model_name}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return self.performance_metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare performance of all models.
        
        Returns:
            pd.DataFrame: Comparison table of model performance
        """
        if not self.performance_metrics:
            raise ValueError("No performance metrics available. Evaluate models first.")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.performance_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        if 'RMSE' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('RMSE')
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def get_best_model(self, metric: str = 'RMSE') -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison ('RMSE', 'MAE', 'MAPE')
            
        Returns:
            Tuple[str, Dict[str, float]]: Best model name and its metrics
        """
        if not self.performance_metrics:
            raise ValueError("No performance metrics available. Evaluate models first.")
        
        if metric not in ['RMSE', 'MAE', 'MAPE']:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Find best model
        best_model = None
        best_value = float('inf')
        
        for model_name, metrics in self.performance_metrics.items():
            if metric in metrics:
                if metrics[metric] < best_value:
                    best_value = metrics[metric]
                    best_model = model_name
        
        if best_model is None:
            raise ValueError(f"No model found with metric {metric}")
        
        logger.info(f"Best model by {metric}: {best_model} ({best_value:.4f})")
        return best_model, self.performance_metrics[best_model]
    
    def plot_forecasts(self, save_path: Optional[str] = None) -> None:
        """
        Plot forecasts from all models and ensemble.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Generate forecasts first.")
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 8))
            
            # Plot original data
            plt.plot(self.data.index, self.data.values, label='Original Data', 
                    linewidth=2, color='black')
            
            # Plot individual model forecasts
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.forecasts)))
            for i, (model_name, forecast_data) in enumerate(self.forecasts.items()):
                if forecast_data['forecast'] is not None:
                    if isinstance(forecast_data['forecast'], pd.Series):
                        plt.plot(forecast_data['forecast'].index, forecast_data['forecast'].values,
                               label=f'{model_name} ({forecast_data["model_type"]})', 
                               linewidth=2, linestyle='--', color=colors[i])
                    else:
                        # Handle numpy array forecasts
                        plt.plot(forecast_data['forecast'], 
                               label=f'{model_name} ({forecast_data["model_type"]})', 
                               linewidth=2, linestyle='--', color=colors[i])
            
            # Plot ensemble forecast if available
            if self.ensemble_forecast is not None:
                plt.plot(self.ensemble_forecast.index, self.ensemble_forecast.values,
                        label='Ensemble Forecast', linewidth=3, color='red')
            
            plt.title('Model Forecasts Comparison')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Forecasts plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create forecasts plot: {str(e)}")
    
    def save_results(self, filepath: str) -> None:
        """
        Save all forecasting results to a file.
        
        Args:
            filepath (str): Path to save the results
        """
        try:
            import pickle
            
            results_data = {
                'forecasts': self.forecasts,
                'performance_metrics': self.performance_metrics,
                'ensemble_forecast': self.ensemble_forecast,
                'data_info': {
                    'data_length': len(self.data),
                    'data_range': (self.data.index[0], self.data.index[-1])
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Forecasting results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def generate_forecast_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive forecast report.
        
        Args:
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Forecast report
        """
        report = []
        report.append("# Time Series Forecasting Report")
        report.append("=" * 50)
        report.append("")
        
        # Data information
        report.append("## Data Information")
        report.append(f"- **Data length**: {len(self.data)} observations")
        report.append(f"- **Date range**: {self.data.index[0]} to {self.data.index[-1]}")
        report.append("")
        
        # Models information
        report.append("## Models Used")
        for model_name, model in self.models.items():
            model_type = type(model).__name__
            report.append(f"- **{model_name}**: {model_type}")
        report.append("")
        
        # Performance comparison
        if self.performance_metrics:
            report.append("## Model Performance Comparison")
            comparison_df = self.compare_models()
            report.append(comparison_df.to_string(index=False))
            report.append("")
            
            # Best model
            try:
                best_model, best_metrics = self.get_best_model('RMSE')
                report.append(f"**Best Model**: {best_model} (RMSE: {best_metrics['RMSE']:.4f})")
                report.append("")
            except:
                pass
        
        # Forecasts information
        if self.forecasts:
            report.append("## Forecasts Generated")
            for model_name, forecast_data in self.forecasts.items():
                forecast_length = len(forecast_data['forecast'])
                report.append(f"- **{model_name}**: {forecast_length} steps")
            report.append("")
            
            if self.ensemble_forecast is not None:
                report.append(f"**Ensemble Forecast**: {len(self.ensemble_forecast)} steps")
                report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.performance_metrics:
            report.append("- Consider using the best performing model for production")
            report.append("- Ensemble methods can improve forecast stability")
        else:
            report.append("- Evaluate models on test data to compare performance")
        
        report.append("- Monitor forecast accuracy over time")
        report.append("- Consider retraining models with new data")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Forecast report saved to {save_path}")
        
        return report_text


def run_complete_forecasting(data: pd.Series, forecast_steps: int = 30,
                            output_dir: str = "data/processed/") -> ForecastingEngine:
    """
    Run complete forecasting pipeline with multiple models.
    
    Args:
        data (pd.Series): Time series data
        forecast_steps (int): Number of steps to forecast
        output_dir (str): Directory to save outputs
        
    Returns:
        ForecastingEngine: Configured forecasting engine with results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize forecasting engine
    engine = ForecastingEngine(data)
    
    # Add models
    try:
        engine.add_arima_model('arima_basic', max_p=3, max_q=3, max_d=1)
        logger.info("ARIMA model added successfully")
    except Exception as e:
        logger.warning(f"Could not add ARIMA model: {str(e)}")
    
    try:
        engine.add_lstm_model('lstm_basic', sequence_length=10, epochs=50)
        logger.info("LSTM model added successfully")
    except Exception as e:
        logger.warning(f"Could not add LSTM model: {str(e)}")
    
    # Generate forecasts
    engine.generate_forecasts(forecast_steps)
    
    # Create ensemble forecast
    try:
        engine.create_ensemble_forecast(method='weighted_average')
        logger.info("Ensemble forecast created successfully")
    except Exception as e:
        logger.warning(f"Could not create ensemble forecast: {str(e)}")
    
    # Evaluate models (using last 20% as test)
    test_size = int(len(data) * 0.2)
    test_data = data.iloc[-test_size:]
    
    try:
        engine.evaluate_models(test_data)
        logger.info("Model evaluation completed")
    except Exception as e:
        logger.warning(f"Could not evaluate models: {str(e)}")
    
    # Generate plots
    try:
        engine.plot_forecasts(os.path.join(output_dir, 'forecasts_comparison.png'))
    except Exception as e:
        logger.warning(f"Could not create forecasts plot: {str(e)}")
    
    # Generate report
    try:
        report_path = os.path.join(output_dir, 'forecast_report.md')
        engine.generate_forecast_report(report_path)
    except Exception as e:
        logger.warning(f"Could not generate forecast report: {str(e)}")
    
    # Save results
    try:
        engine.save_results(os.path.join(output_dir, 'forecasting_results.pkl'))
    except Exception as e:
        logger.warning(f"Could not save results: {str(e)}")
    
    logger.info("Complete forecasting pipeline finished")
    return engine


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data
    sample_data = load_sample_data()
    ts_data = sample_data['value']
    
    print("Sample data loaded, running complete forecasting pipeline...")
    
    # Run complete forecasting
    engine = run_complete_forecasting(ts_data, forecast_steps=30)
    
    print("Forecasting pipeline completed successfully!")
    print(f"Models used: {list(engine.models.keys())}")
    print(f"Forecasts generated: {list(engine.forecasts.keys())}")
    
    if engine.performance_metrics:
        print("Model performance:")
        for model, metrics in engine.performance_metrics.items():
            print(f"  {model}: RMSE = {metrics.get('RMSE', 'N/A'):.4f}")
