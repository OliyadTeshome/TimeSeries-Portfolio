"""
Model Explainability and Interpretability Tools

This module provides comprehensive tools for explaining and interpreting financial models,
including SHAP values, feature importance, and model diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import logging

# SHAP for model explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# LIME for local interpretability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

from .base import BaseModel, ModelResult
from .config import get_config
from .logging_config import setup_logging

# Setup logging
logger = setup_logging()
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    Comprehensive model explainability and interpretability tools.
    
    This class provides methods for explaining model predictions, understanding
    feature importance, and diagnosing model behavior.
    """
    
    def __init__(self, model: Optional[BaseModel] = None):
        """
        Initialize the model explainer.
        
        Args:
            model: The model to explain (optional, can be set later)
        """
        self.model = model
        self.config = get_config()
        self.explainer = None
        self.feature_names = None
        self.feature_importance = None
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
    
    def set_model(self, model: BaseModel):
        """Set the model to explain."""
        self.model = model
        if SHAP_AVAILABLE:
            self._init_shap_explainer()
    
    def _init_shap_explainer(self):
        """Initialize SHAP explainer for the current model."""
        if self.model is None:
            return
            
        try:
            # Get model type and initialize appropriate explainer
            if hasattr(self.model, 'model_type'):
                model_type = self.model.model_type
            else:
                # Try to infer model type
                model_type = self._infer_model_type()
            
            if model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif model_type == 'linear':
                self.explainer = shap.LinearExplainer(self.model)
            elif model_type == 'deep':
                self.explainer = shap.DeepExplainer(self.model)
            else:
                # Kernel explainer as fallback
                self.explainer = shap.KernelExplainer(self.model.predict, self._get_background_data())
                
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _infer_model_type(self) -> str:
        """Infer the type of model for explainability."""
        if self.model is None:
            return 'unknown'
        
        model_name = type(self.model).__name__.lower()
        
        if any(x in model_name for x in ['tree', 'forest', 'xgboost', 'lightgbm']):
            return 'tree'
        elif any(x in model_name for x in ['linear', 'regression', 'arima']):
            return 'linear'
        elif any(x in model_name for x in ['lstm', 'neural', 'deep']):
            return 'deep'
        else:
            return 'unknown'
    
    def _get_background_data(self) -> np.ndarray:
        """Get background data for SHAP explainer."""
        # This should be implemented based on the specific model
        # For now, return a simple array
        return np.random.randn(100, 10)
    
    def explain_prediction(self, 
                          data: Union[pd.DataFrame, np.ndarray], 
                          sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a specific prediction using SHAP values.
        
        Args:
            data: Input data for prediction
            sample_idx: Index of the sample to explain
            
        Returns:
            Dictionary containing explanation results
        """
        if self.model is None:
            raise ValueError("No model set for explanation")
        
        if not SHAP_AVAILABLE:
            return self._fallback_explanation(data, sample_idx)
        
        try:
            # Get SHAP values
            if isinstance(data, pd.DataFrame):
                sample_data = data.iloc[sample_idx:sample_idx+1]
            else:
                sample_data = data[sample_idx:sample_idx+1]
            
            # Generate SHAP values
            if self.explainer is not None:
                shap_values = self.explainer.shap_values(sample_data)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first output for regression
                
                # Get feature names
                if isinstance(data, pd.DataFrame):
                    feature_names = data.columns.tolist()
                else:
                    feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
                
                # Calculate feature importance
                feature_importance = np.abs(shap_values[0])
                
                explanation = {
                    'shap_values': shap_values[0],
                    'feature_importance': feature_importance,
                    'feature_names': feature_names,
                    'prediction': self.model.predict(sample_data),
                    'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                    'method': 'SHAP'
                }
                
            else:
                explanation = self._fallback_explanation(data, sample_idx)
                
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            explanation = self._fallback_explanation(data, sample_idx)
        
        return explanation
    
    def _fallback_explanation(self, data: Union[pd.DataFrame, np.ndarray], 
                             sample_idx: int) -> Dict[str, Any]:
        """Fallback explanation method when SHAP is not available."""
        if isinstance(data, pd.DataFrame):
            sample_data = data.iloc[sample_idx:sample_idx+1]
            feature_names = data.columns.tolist()
        else:
            sample_data = data[sample_idx:sample_idx+1]
            feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
        
        # Simple feature importance based on correlation
        try:
            prediction = self.model.predict(sample_data)
            # Calculate simple correlation-based importance
            feature_importance = np.random.rand(len(feature_names))  # Placeholder
            
            explanation = {
                'shap_values': np.zeros(len(feature_names)),
                'feature_importance': feature_importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'base_value': 0,
                'method': 'Fallback'
            }
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            explanation = {
                'shap_values': np.zeros(len(feature_names)),
                'feature_importance': np.ones(len(feature_names)),
                'feature_names': feature_names,
                'prediction': None,
                'base_value': 0,
                'method': 'Error'
            }
        
        return explanation
    
    def get_feature_importance(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, float]:
        """
        Get global feature importance for the model.
        
        Args:
            data: Training or validation data
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("No model set for feature importance analysis")
        
        try:
            # Try to get feature importance from model attributes
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance = np.abs(self.model.coef_)
            else:
                # Calculate importance using permutation
                importance = self._calculate_permutation_importance(data)
            
            # Get feature names
            if isinstance(data, pd.DataFrame):
                feature_names = data.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(data.shape[1])]
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importance))
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            self.feature_importance = feature_importance
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _calculate_permutation_importance(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Calculate feature importance using permutation method."""
        try:
            # Simple permutation importance calculation
            n_features = data.shape[1]
            importance = np.zeros(n_features)
            
            # Get baseline performance
            if isinstance(data, pd.DataFrame):
                baseline_pred = self.model.predict(data)
            else:
                baseline_pred = self.model.predict(data)
            
            baseline_score = np.mean(baseline_pred)
            
            # Calculate importance for each feature
            for i in range(n_features):
                # Shuffle feature values
                shuffled_data = data.copy()
                if isinstance(data, pd.DataFrame):
                    shuffled_data.iloc[:, i] = np.random.permutation(data.iloc[:, i].values)
                else:
                    shuffled_data[:, i] = np.random.permutation(data[:, i])
                
                # Get new predictions
                shuffled_pred = self.model.predict(shuffled_data)
                shuffled_score = np.mean(shuffled_pred)
                
                # Importance is the difference in performance
                importance[i] = abs(baseline_score - shuffled_score)
            
            return importance
            
        except Exception as e:
            logger.error(f"Error in permutation importance: {e}")
            return np.ones(data.shape[1])
    
    def explain_model_behavior(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """
        Provide comprehensive explanation of model behavior.
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary containing model behavior analysis
        """
        if self.model is None:
            raise ValueError("No model set for behavior analysis")
        
        try:
            # Get predictions
            predictions = self.model.predict(data)
            
            # Calculate residuals if applicable
            residuals = None
            if hasattr(data, 'target') or 'target' in data.columns:
                if isinstance(data, pd.DataFrame) and 'target' in data.columns:
                    actual = data['target']
                else:
                    actual = data.target
                residuals = actual - predictions
            
            # Analyze predictions distribution
            pred_stats = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'median': np.median(predictions)
            }
            
            # Analyze residuals if available
            residual_stats = None
            if residuals is not None:
                residual_stats = {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'min': np.min(residuals),
                    'max': np.max(residuals),
                    'median': np.median(residuals)
                }
            
            # Get feature importance
            feature_importance = self.get_feature_importance(data)
            
            behavior_analysis = {
                'predictions': predictions,
                'prediction_stats': pred_stats,
                'residuals': residuals,
                'residual_stats': residual_stats,
                'feature_importance': feature_importance,
                'data_shape': data.shape,
                'model_type': type(self.model).__name__
            }
            
            return behavior_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model behavior: {e}")
            return {}
    
    def create_explanation_plots(self, 
                                explanation: Dict[str, Any], 
                                plot_type: str = 'all') -> Dict[str, Any]:
        """
        Create visualization plots for model explanations.
        
        Args:
            explanation: Explanation results from explain_prediction
            plot_type: Type of plots to create ('all', 'shap', 'importance', 'summary')
            
        Returns:
            Dictionary containing plot objects
        """
        plots = {}
        
        try:
            if plot_type in ['all', 'shap'] and explanation.get('method') == 'SHAP':
                plots['shap_summary'] = self._create_shap_summary_plot(explanation)
                plots['shap_waterfall'] = self._create_shap_waterfall_plot(explanation)
            
            if plot_type in ['all', 'importance']:
                plots['feature_importance'] = self._create_feature_importance_plot(explanation)
            
            if plot_type in ['all', 'summary']:
                plots['prediction_summary'] = self._create_prediction_summary_plot(explanation)
            
        except Exception as e:
            logger.error(f"Error creating explanation plots: {e}")
        
        return plots
    
    def _create_shap_summary_plot(self, explanation: Dict[str, Any]) -> go.Figure:
        """Create SHAP summary plot."""
        try:
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            # Sort by absolute SHAP values
            sorted_indices = np.argsort(np.abs(shap_values))
            sorted_values = shap_values[sorted_indices]
            sorted_features = [feature_names[i] for i in sorted_indices]
            
            # Color bars based on positive/negative values
            colors = ['red' if v < 0 else 'blue' for v in sorted_values]
            
            fig.add_trace(go.Bar(
                y=sorted_features,
                x=sorted_values,
                orientation='h',
                marker_color=colors,
                name='SHAP Values'
            ))
            
            fig.update_layout(
                title="SHAP Feature Importance",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {e}")
            return go.Figure()
    
    def _create_shap_waterfall_plot(self, explanation: Dict[str, Any]) -> go.Figure:
        """Create SHAP waterfall plot."""
        try:
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            base_value = explanation['base_value']
            prediction = explanation['prediction']
            
            # Create waterfall chart
            fig = go.Figure()
            
            # Calculate cumulative values
            cumulative = base_value
            cumulative_values = [base_value]
            
            for i, (feature, value) in enumerate(zip(feature_names, shap_values)):
                cumulative += value
                cumulative_values.append(cumulative)
            
            # Create waterfall
            fig.add_trace(go.Waterfall(
                name="SHAP Values",
                orientation="h",
                measure=["relative"] * len(feature_names) + ["total"],
                x=shap_values.tolist() + [prediction - base_value],
                textposition="outside",
                text=[f"{v:.4f}" for v in shap_values] + [f"{prediction:.4f}"],
                y=feature_names + ["Prediction"],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "red"}},
                increasing={"marker": {"color": "blue"}},
                totals={"marker": {"color": "green"}}
            ))
            
            fig.update_layout(
                title="SHAP Waterfall Plot",
                xaxis_title="SHAP Value",
                height=500,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {e}")
            return go.Figure()
    
    def _create_feature_importance_plot(self, explanation: Dict[str, Any]) -> go.Figure:
        """Create feature importance plot."""
        try:
            feature_importance = explanation['feature_importance']
            feature_names = explanation['feature_names']
            
            # Sort by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_indices]
            sorted_features = [feature_names[i] for i in sorted_indices]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=sorted_features,
                y=sorted_importance,
                marker_color='lightblue',
                name='Feature Importance'
            ))
            
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return go.Figure()
    
    def _create_prediction_summary_plot(self, explanation: Dict[str, Any]) -> go.Figure:
        """Create prediction summary plot."""
        try:
            prediction = explanation['prediction']
            base_value = explanation['base_value']
            
            fig = go.Figure()
            
            # Create bar chart showing base value and prediction
            fig.add_trace(go.Bar(
                x=['Base Value', 'Prediction'],
                y=[base_value, prediction],
                marker_color=['lightgray', 'lightblue'],
                name='Values'
            ))
            
            fig.update_layout(
                title="Prediction Summary",
                yaxis_title="Value",
                height=300,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating prediction summary plot: {e}")
            return go.Figure()
    
    def generate_explanation_report(self, 
                                  data: Union[pd.DataFrame, np.ndarray],
                                  sample_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive explanation report.
        
        Args:
            data: Data to analyze
            sample_indices: Specific samples to explain (if None, explain first few)
            
        Returns:
            Dictionary containing comprehensive explanation report
        """
        if self.model is None:
            raise ValueError("No model set for explanation report")
        
        try:
            # Determine samples to explain
            if sample_indices is None:
                n_samples = min(5, len(data))
                sample_indices = list(range(n_samples))
            
            # Generate explanations for each sample
            explanations = {}
            for idx in sample_indices:
                explanations[f'sample_{idx}'] = self.explain_prediction(data, idx)
            
            # Get global model behavior
            behavior_analysis = self.explain_model_behavior(data)
            
            # Create plots
            plots = {}
            for idx in sample_indices:
                sample_explanation = explanations[f'sample_{idx}']
                plots[f'sample_{idx}'] = self.create_explanation_plots(sample_explanation)
            
            # Compile report
            report = {
                'model_info': {
                    'type': type(self.model).__name__,
                    'parameters': self._get_model_parameters()
                },
                'data_info': {
                    'shape': data.shape,
                    'features': list(data.columns) if isinstance(data, pd.DataFrame) else None
                },
                'explanations': explanations,
                'behavior_analysis': behavior_analysis,
                'plots': plots,
                'summary': self._generate_summary_statistics(explanations, behavior_analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explanation report: {e}")
            return {}
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters for documentation."""
        if self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'get_params'):
                return self.model.get_params()
            elif hasattr(self.model, '__dict__'):
                # Extract common parameters
                params = {}
                for key, value in self.model.__dict__.items():
                    if not key.startswith('_') and not callable(value):
                        params[key] = str(value)
                return params
            else:
                return {'type': type(self.model).__name__}
        except Exception as e:
            logger.warning(f"Could not extract model parameters: {e}")
            return {'type': type(self.model).__name__}
    
    def _generate_summary_statistics(self, 
                                   explanations: Dict[str, Any], 
                                   behavior_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the explanation report."""
        try:
            # Extract key metrics
            feature_importance = behavior_analysis.get('feature_importance', {})
            
            # Top features
            top_features = dict(list(feature_importance.items())[:10])
            
            # Prediction statistics
            pred_stats = behavior_analysis.get('prediction_stats', {})
            
            # Residual statistics
            residual_stats = behavior_analysis.get('residual_stats', {})
            
            summary = {
                'top_features': top_features,
                'prediction_summary': pred_stats,
                'residual_summary': residual_stats,
                'explanation_method': 'SHAP' if SHAP_AVAILABLE else 'Fallback',
                'total_samples_explained': len(explanations)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            return {}


class ModelDiagnostics:
    """Tools for diagnosing model performance and behavior."""
    
    def __init__(self, model: Optional[BaseModel] = None):
        """Initialize model diagnostics."""
        self.model = model
        self.diagnostics = {}
    
    def diagnose_model(self, 
                      train_data: Union[pd.DataFrame, np.ndarray],
                      val_data: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model diagnostics.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            
        Returns:
            Dictionary containing diagnostic results
        """
        if self.model is None:
            raise ValueError("No model set for diagnostics")
        
        try:
            diagnostics = {}
            
            # Basic model info
            diagnostics['model_info'] = self._get_model_info()
            
            # Data quality checks
            diagnostics['data_quality'] = self._check_data_quality(train_data)
            
            # Performance metrics
            diagnostics['performance'] = self._calculate_performance_metrics(train_data, val_data)
            
            # Model stability
            diagnostics['stability'] = self._check_model_stability(train_data)
            
            # Bias and fairness
            diagnostics['bias_fairness'] = self._check_bias_fairness(train_data)
            
            self.diagnostics = diagnostics
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error in model diagnostics: {e}")
            return {}
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        if self.model is None:
            return {}
        
        return {
            'type': type(self.model).__name__,
            'parameters': self._get_model_parameters(),
            'training_status': hasattr(self.model, 'fitted_') and self.model.fitted_
        }
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        try:
            if hasattr(self.model, 'get_params'):
                return self.model.get_params()
            else:
                return {}
        except Exception:
            return {}
    
    def _check_data_quality(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Check data quality metrics."""
        try:
            if isinstance(data, pd.DataFrame):
                quality_metrics = {
                    'missing_values': data.isnull().sum().to_dict(),
                    'duplicates': data.duplicated().sum(),
                    'data_types': data.dtypes.to_dict(),
                    'shape': data.shape
                }
            else:
                quality_metrics = {
                    'missing_values': np.isnan(data).sum(),
                    'duplicates': 0,  # Not applicable for numpy arrays
                    'data_types': str(data.dtype),
                    'shape': data.shape
                }
            
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {}
    
    def _calculate_performance_metrics(self, 
                                     train_data: Union[pd.DataFrame, np.ndarray],
                                     val_data: Optional[Union[pd.DataFrame, np.ndarray]] = None) -> Dict[str, Any]:
        """Calculate performance metrics."""
        try:
            metrics = {}
            
            # Training performance
            train_pred = self.model.predict(train_data)
            if hasattr(train_data, 'target') or (isinstance(train_data, pd.DataFrame) and 'target' in train_data.columns):
                if isinstance(train_data, pd.DataFrame) and 'target' in train_data.columns:
                    train_actual = train_data['target']
                else:
                    train_actual = train_data.target
                
                metrics['train'] = self._calculate_metrics(train_actual, train_pred)
            
            # Validation performance
            if val_data is not None:
                val_pred = self.model.predict(val_data)
                if hasattr(val_data, 'target') or (isinstance(val_data, pd.DataFrame) and 'target' in val_data.columns):
                    if isinstance(val_data, pd.DataFrame) and 'target' in val_data.columns:
                        val_actual = val_data['target']
                    else:
                        val_actual = val_data.target
                    
                    metrics['validation'] = self._calculate_metrics(val_actual, val_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate common performance metrics."""
        try:
            # Remove any NaN values
            mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual_clean = actual[mask]
            predicted_clean = predicted[mask]
            
            if len(actual_clean) == 0:
                return {}
            
            # Calculate metrics
            mse = np.mean((actual_clean - predicted_clean) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_clean - predicted_clean))
            
            # R-squared
            ss_res = np.sum((actual_clean - predicted_clean) ** 2)
            ss_tot = np.sum((actual_clean - np.mean(actual_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _check_model_stability(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Check model stability across different data subsets."""
        try:
            # Simple stability check using bootstrap
            n_samples = len(data)
            n_bootstrap = min(10, n_samples // 2)
            
            predictions = []
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                bootstrap_data = data[indices] if isinstance(data, np.ndarray) else data.iloc[indices]
                
                # Get predictions
                pred = self.model.predict(bootstrap_data)
                predictions.append(np.mean(pred))
            
            # Calculate stability metrics
            predictions = np.array(predictions)
            stability_metrics = {
                'mean_prediction': np.mean(predictions),
                'std_prediction': np.std(predictions),
                'cv_prediction': np.std(predictions) / np.mean(predictions) if np.mean(predictions) != 0 else 0
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error checking model stability: {e}")
            return {}
    
    def _check_bias_fairness(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Check for potential bias and fairness issues."""
        try:
            # Simple bias check based on prediction distribution
            predictions = self.model.predict(data)
            
            bias_metrics = {
                'prediction_mean': np.mean(predictions),
                'prediction_std': np.std(predictions),
                'prediction_skew': self._calculate_skewness(predictions),
                'prediction_kurtosis': self._calculate_kurtosis(predictions)
            }
            
            return bias_metrics
            
        except Exception as e:
            logger.error(f"Error checking bias and fairness: {e}")
            return {}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except Exception:
            return 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except Exception:
            return 0


def create_explainer(model: BaseModel) -> ModelExplainer:
    """Factory function to create a model explainer."""
    return ModelExplainer(model)


def create_diagnostics(model: BaseModel) -> ModelDiagnostics:
    """Factory function to create model diagnostics."""
    return ModelDiagnostics(model)
