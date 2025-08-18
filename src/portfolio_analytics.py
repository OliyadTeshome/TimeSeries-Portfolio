"""
Enhanced Portfolio Analytics Module

This module provides advanced portfolio analytics including performance attribution,
factor analysis, risk decomposition, and sophisticated portfolio evaluation tools.
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
from scipy import stats
from scipy.optimize import minimize
import cvxpy as cp
from datetime import datetime, timedelta

from .config import get_config
from .logging_config import setup_logging
from .exceptions import PortfolioError as PortfolioAnalyticsError

# Setup logging
logger = setup_logging()
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PortfolioAnalytics:
    """
    Advanced portfolio analytics and performance evaluation system.
    
    This class provides comprehensive tools for analyzing portfolio performance,
    attributing returns to various factors, and evaluating portfolio efficiency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the portfolio analytics system.
        
        Args:
            config: Configuration dictionary for analytics parameters
        """
        self.config = config or get_config()
        self.analytics_results = {}
        self.performance_metrics = {}
        self.attribution_results = {}
        
        # Default parameters
        self.default_benchmarks = ['SPY', '^GSPC', '^VIX']  # S&P 500, VIX
        self.default_factors = ['MKT', 'SMB', 'HML', 'MOM', 'RMW', 'CMA']  # Fama-French factors
        self.default_periods = ['1D', '1W', '1M', '3M', '6M', '1Y', 'YTD']
    
    def calculate_comprehensive_metrics(self, 
                                     portfolio_returns: Union[pd.Series, np.ndarray],
                                     benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                     risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio performance metrics.
        
        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary containing comprehensive performance metrics
        """
        try:
            # Ensure returns are pandas Series with datetime index
            if isinstance(portfolio_returns, np.ndarray):
                portfolio_returns = pd.Series(portfolio_returns)
            
            if benchmark_returns is not None and isinstance(benchmark_returns, np.ndarray):
                benchmark_returns = pd.Series(benchmark_returns)
            
            # Basic return metrics
            return_metrics = self._calculate_return_metrics(portfolio_returns)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_returns)
            
            # Risk-adjusted return metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(
                portfolio_returns, risk_free_rate
            )
            
            # Benchmark comparison metrics
            benchmark_metrics = {}
            if benchmark_returns is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(
                    portfolio_returns, benchmark_returns, risk_free_rate
                )
            
            # Drawdown analysis
            drawdown_metrics = self._calculate_drawdown_metrics(portfolio_returns)
            
            # Rolling metrics
            rolling_metrics = self._calculate_rolling_metrics(portfolio_returns)
            
            # Compile all metrics
            all_metrics = {
                'return_metrics': return_metrics,
                'risk_metrics': risk_metrics,
                'risk_adjusted_metrics': risk_adjusted_metrics,
                'benchmark_metrics': benchmark_metrics,
                'drawdown_metrics': drawdown_metrics,
                'rolling_metrics': rolling_metrics,
                'data_info': {
                    'total_observations': len(portfolio_returns),
                    'date_range': f"{portfolio_returns.index[0]} to {portfolio_returns.index[-1]}" if len(portfolio_returns) > 0 else "No data"
                }
            }
            
            self.performance_metrics = all_metrics
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive metrics: {e}")
            raise PortfolioAnalyticsError(f"Comprehensive metrics calculation failed: {str(e)}")
    
    def _calculate_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic return metrics."""
        try:
            # Remove NaN values
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                return {}
            
            # Calculate metrics
            total_return = (1 + clean_returns).prod() - 1
            arithmetic_mean = clean_returns.mean()
            geometric_mean = (1 + total_return) ** (1 / len(clean_returns)) - 1
            
            # Annualized metrics (assuming daily data)
            trading_days = 252
            annualized_return = (1 + total_return) ** (trading_days / len(clean_returns)) - 1
            annualized_arithmetic = arithmetic_mean * trading_days
            
            # Period returns
            period_returns = {}
            for period in self.default_periods:
                if period == '1D':
                    period_returns[period] = clean_returns.iloc[-1] if len(clean_returns) > 0 else 0
                elif period == '1W':
                    period_returns[period] = (1 + clean_returns.tail(5)).prod() - 1 if len(clean_returns) >= 5 else 0
                elif period == '1M':
                    period_returns[period] = (1 + clean_returns.tail(21)).prod() - 1 if len(clean_returns) >= 21 else 0
                elif period == '3M':
                    period_returns[period] = (1 + clean_returns.tail(63)).prod() - 1 if len(clean_returns) >= 63 else 0
                elif period == '6M':
                    period_returns[period] = (1 + clean_returns.tail(126)).prod() - 1 if len(clean_returns) >= 126 else 0
                elif period == '1Y':
                    period_returns[period] = (1 + clean_returns.tail(252)).prod() - 1 if len(clean_returns) >= 252 else 0
                elif period == 'YTD':
                    # Calculate YTD return
                    current_year = datetime.now().year
                    ytd_returns = clean_returns[clean_returns.index.year == current_year]
                    period_returns[period] = (1 + ytd_returns).prod() - 1 if len(ytd_returns) > 0 else 0
            
            return {
                'total_return': total_return,
                'arithmetic_mean': arithmetic_mean,
                'geometric_mean': geometric_mean,
                'annualized_return': annualized_return,
                'annualized_arithmetic': annualized_arithmetic,
                'period_returns': period_returns
            }
            
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                return {}
            
            # Basic risk metrics
            volatility = clean_returns.std()
            annualized_volatility = volatility * np.sqrt(252)
            
            # Downside deviation
            downside_returns = clean_returns[clean_returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            annualized_downside_deviation = downside_deviation * np.sqrt(252)
            
            # Value at Risk and Conditional VaR
            var_95 = np.percentile(clean_returns, 5)
            var_99 = np.percentile(clean_returns, 1)
            
            cvar_95 = clean_returns[clean_returns <= var_95].mean() if var_95 < 0 else 0
            cvar_99 = clean_returns[clean_returns <= var_99].mean() if var_99 < 0 else 0
            
            # Skewness and kurtosis
            skewness = stats.skew(clean_returns)
            kurtosis = stats.kurtosis(clean_returns)
            
            # Maximum drawdown
            cumulative = (1 + clean_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            return {
                'volatility': volatility,
                'annualized_volatility': annualized_volatility,
                'downside_deviation': downside_deviation,
                'annualized_downside_deviation': annualized_downside_deviation,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series, risk_free_rate: float) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics."""
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                return {}
            
            # Annualized metrics
            annualized_return = (1 + clean_returns).prod() ** (252 / len(clean_returns)) - 1
            annualized_volatility = clean_returns.std() * np.sqrt(252)
            daily_risk_free = risk_free_rate / 252
            
            # Sharpe ratio
            excess_returns = clean_returns - daily_risk_free
            sharpe_ratio = excess_returns.mean() / clean_returns.std() if clean_returns.std() != 0 else 0
            annualized_sharpe = sharpe_ratio * np.sqrt(252)
            
            # Sortino ratio
            downside_returns = clean_returns[clean_returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0
            annualized_sortino = sortino_ratio * np.sqrt(252)
            
            # Calmar ratio
            max_dd = abs(self._calculate_max_drawdown(clean_returns))
            calmar_ratio = annualized_return / max_dd if max_dd != 0 else 0
            
            # Information ratio (if benchmark available)
            information_ratio = np.nan  # Will be calculated in benchmark metrics
            
            # Treynor ratio
            # Beta calculation would be needed here
            treynor_ratio = np.nan
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'annualized_sharpe': annualized_sharpe,
                'sortino_ratio': sortino_ratio,
                'annualized_sortino': annualized_sortino,
                'calmar_ratio': calmar_ratio,
                'information_ratio': information_ratio,
                'treynor_ratio': treynor_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}
    
    def _calculate_benchmark_metrics(self, 
                                   portfolio_returns: pd.Series, 
                                   benchmark_returns: pd.Series,
                                   risk_free_rate: float) -> Dict[str, float]:
        """Calculate benchmark comparison metrics."""
        try:
            # Align returns
            aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
            portfolio_aligned = aligned_returns.iloc[:, 0]
            benchmark_aligned = aligned_returns.iloc[:, 1]
            
            if len(aligned_returns) == 0:
                return {}
            
            # Calculate excess returns
            excess_returns = portfolio_aligned - benchmark_aligned
            
            # Beta calculation
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # Alpha calculation
            portfolio_mean = portfolio_aligned.mean()
            benchmark_mean = benchmark_aligned.mean()
            daily_risk_free = risk_free_rate / 252
            alpha = portfolio_mean - (daily_risk_free + beta * (benchmark_mean - daily_risk_free))
            annualized_alpha = alpha * 252
            
            # Information ratio
            tracking_error = excess_returns.std()
            information_ratio = excess_returns.mean() / tracking_error if tracking_error != 0 else 0
            annualized_information_ratio = information_ratio * np.sqrt(252)
            
            # Treynor ratio
            treynor_ratio = (portfolio_mean - daily_risk_free) / beta if beta != 0 else np.nan
            
            # Jensen's alpha
            jensen_alpha = alpha
            
            # R-squared
            correlation = np.corrcoef(portfolio_aligned, benchmark_aligned)[0, 1]
            r_squared = correlation ** 2 if not np.isnan(correlation) else 0
            
            return {
                'beta': beta,
                'alpha': alpha,
                'annualized_alpha': annualized_alpha,
                'information_ratio': information_ratio,
                'annualized_information_ratio': annualized_information_ratio,
                'treynor_ratio': treynor_ratio,
                'jensen_alpha': jensen_alpha,
                'r_squared': r_squared,
                'tracking_error': tracking_error,
                'annualized_tracking_error': tracking_error * np.sqrt(252)
            }
            
        except Exception as e:
            logger.error(f"Error calculating benchmark metrics: {e}")
            return {}
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive drawdown metrics."""
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) == 0:
                return {}
            
            # Calculate cumulative returns and drawdowns
            cumulative = (1 + clean_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdown.min()
            max_drawdown_date = drawdown.idxmin() if hasattr(drawdown, 'idxmin') else None
            
            # Drawdown duration
            underwater_periods = drawdown < 0
            underwater_duration = underwater_periods.sum()
            underwater_percentage = underwater_periods.mean()
            
            # Recovery periods
            recovery_periods = []
            current_dd = 0
            recovery_count = 0
            
            for dd in drawdown:
                if dd < 0:
                    current_dd += 1
                elif current_dd > 0:
                    recovery_periods.append(current_dd)
                    current_dd = 0
                    recovery_count += 1
            
            if current_dd > 0:
                recovery_periods.append(current_dd)
            
            avg_recovery_period = np.mean(recovery_periods) if recovery_periods else 0
            
            # Drawdown distribution
            dd_distribution = {
                'shallow': (drawdown >= -0.05).sum(),      # < 5%
                'moderate': ((drawdown < -0.05) & (drawdown >= -0.15)).sum(),  # 5-15%
                'deep': ((drawdown < -0.15) & (drawdown >= -0.25)).sum(),      # 15-25%
                'severe': (drawdown < -0.25).sum()         # > 25%
            }
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_drawdown_date,
                'underwater_duration': underwater_duration,
                'underwater_percentage': underwater_percentage,
                'avg_recovery_period': avg_recovery_period,
                'recovery_count': recovery_count,
                'drawdown_distribution': dd_distribution,
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating drawdown metrics: {e}")
            return {}
    
    def _calculate_rolling_metrics(self, returns: pd.Series, window: int = 252) -> Dict[str, pd.Series]:
        """Calculate rolling performance metrics."""
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) < window:
                return {}
            
            # Rolling returns
            rolling_returns = clean_returns.rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1
            )
            
            # Rolling volatility
            rolling_volatility = clean_returns.rolling(window=window).std() * np.sqrt(252)
            
            # Rolling Sharpe ratio
            daily_risk_free = 0.02 / 252  # Assuming 2% annual risk-free rate
            rolling_excess = clean_returns.rolling(window=window).mean() - daily_risk_free
            rolling_sharpe = rolling_excess / (clean_returns.rolling(window=window).std() * np.sqrt(252))
            
            # Rolling maximum drawdown
            rolling_dd = pd.Series(index=clean_returns.index, dtype=float)
            for i in range(window, len(clean_returns)):
                window_returns = clean_returns.iloc[i-window:i]
                window_cumulative = (1 + window_returns).cumprod()
                window_max = window_cumulative.expanding().max()
                window_dd = (window_cumulative - window_max) / window_max
                rolling_dd.iloc[i] = window_dd.min()
            
            return {
                'rolling_returns': rolling_returns,
                'rolling_volatility': rolling_volatility,
                'rolling_sharpe': rolling_sharpe,
                'rolling_max_drawdown': rolling_dd
            }
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except Exception:
            return 0.0
    
    def perform_return_attribution(self, 
                                 portfolio_returns: pd.Series,
                                 asset_returns: Dict[str, pd.Series],
                                 portfolio_weights: Dict[str, float],
                                 method: str = 'brinson') -> Dict[str, Any]:
        """
        Perform return attribution analysis.
        
        Args:
            portfolio_returns: Portfolio returns
            asset_returns: Dictionary of asset returns
            portfolio_weights: Portfolio weights
            method: Attribution method ('brinson', 'factor', 'custom')
            
        Returns:
            Dictionary containing attribution results
        """
        try:
            if method == 'brinson':
                return self._brinson_attribution(portfolio_returns, asset_returns, portfolio_weights)
            elif method == 'factor':
                return self._factor_attribution(portfolio_returns, asset_returns, portfolio_weights)
            else:
                return self._custom_attribution(portfolio_returns, asset_returns, portfolio_weights)
                
        except Exception as e:
            logger.error(f"Error performing return attribution: {e}")
            raise PortfolioAnalyticsError(f"Return attribution failed: {str(e)}")
    
    def _brinson_attribution(self, 
                            portfolio_returns: pd.Series,
                            asset_returns: Dict[str, pd.Series],
                            portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Perform Brinson attribution analysis."""
        try:
            # Align all return series
            all_returns = pd.concat([portfolio_returns] + list(asset_returns.values()), axis=1)
            all_returns.columns = ['portfolio'] + list(asset_returns.keys())
            aligned_returns = all_returns.dropna()
            
            if len(aligned_returns) == 0:
                return {}
            
            # Calculate attribution components
            attribution_results = {}
            
            for asset_name in asset_returns.keys():
                asset_weight = portfolio_weights.get(asset_name, 0.0)
                asset_return = aligned_returns[asset_name]
                portfolio_return = aligned_returns['portfolio']
                
                # Allocation effect
                allocation_effect = (asset_weight - asset_weight) * asset_return.mean()
                
                # Selection effect
                selection_effect = asset_weight * (asset_return.mean() - asset_return.mean())
                
                # Interaction effect
                interaction_effect = (asset_weight - asset_weight) * (asset_return.mean() - asset_return.mean())
                
                attribution_results[asset_name] = {
                    'allocation_effect': allocation_effect,
                    'selection_effect': selection_effect,
                    'interaction_effect': interaction_effect,
                    'total_effect': allocation_effect + selection_effect + interaction_effect
                }
            
            # Calculate total attribution
            total_allocation = sum(result['allocation_effect'] for result in attribution_results.values())
            total_selection = sum(result['selection_effect'] for result in attribution_results.values())
            total_interaction = sum(result['interaction_effect'] for result in attribution_results.values())
            
            return {
                'asset_attribution': attribution_results,
                'total_attribution': {
                    'allocation': total_allocation,
                    'selection': total_selection,
                    'interaction': total_interaction,
                    'total': total_allocation + total_selection + total_interaction
                },
                'method': 'Brinson'
            }
            
        except Exception as e:
            logger.error(f"Error in Brinson attribution: {e}")
            return {}
    
    def _factor_attribution(self, 
                           portfolio_returns: pd.Series,
                           asset_returns: Dict[str, pd.Series],
                           portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Perform factor-based attribution analysis."""
        try:
            # This is a simplified factor attribution
            # In practice, you would use actual factor data
            
            attribution_results = {
                'factor_attribution': {
                    'market': 0.0,
                    'size': 0.0,
                    'value': 0.0,
                    'momentum': 0.0
                },
                'method': 'Factor'
            }
            
            return attribution_results
            
        except Exception as e:
            logger.error(f"Error in factor attribution: {e}")
            return {}
    
    def _custom_attribution(self, 
                           portfolio_returns: pd.Series,
                           asset_returns: Dict[str, pd.Series],
                           portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """Perform custom attribution analysis."""
        try:
            # Simple weight-based attribution
            attribution_results = {}
            
            for asset_name, asset_return in asset_returns.items():
                weight = portfolio_weights.get(asset_name, 0.0)
                contribution = weight * asset_return.mean()
                
                attribution_results[asset_name] = {
                    'weight': weight,
                    'return': asset_return.mean(),
                    'contribution': contribution,
                    'percentage_contribution': contribution / portfolio_returns.mean() if portfolio_returns.mean() != 0 else 0
                }
            
            return {
                'asset_attribution': attribution_results,
                'method': 'Custom'
            }
            
        except Exception as e:
            logger.error(f"Error in custom attribution: {e}")
            return {}
    
    def calculate_risk_decomposition(self, 
                                   portfolio_weights: Dict[str, float],
                                   asset_returns: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        Calculate risk decomposition for portfolio.
        
        Args:
            portfolio_weights: Portfolio weights
            asset_returns: Asset returns
            
        Returns:
            Dictionary containing risk decomposition results
        """
        try:
            # Align all return series
            all_returns = pd.concat(list(asset_returns.values()), axis=1)
            all_returns.columns = list(asset_returns.keys())
            aligned_returns = all_returns.dropna()
            
            if len(aligned_returns) == 0:
                return {}
            
            # Calculate covariance matrix
            covariance_matrix = aligned_returns.cov()
            
            # Calculate portfolio variance
            weights_array = np.array([portfolio_weights.get(asset, 0.0) for asset in asset_returns.keys()])
            portfolio_variance = weights_array.T @ covariance_matrix.values @ weights_array
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate risk contributions
            risk_contributions = {}
            marginal_risk = covariance_matrix.values @ weights_array
            
            for i, asset_name in enumerate(asset_returns.keys()):
                weight = portfolio_weights.get(asset_name, 0.0)
                risk_contributions[asset_name] = {
                    'weight': weight,
                    'marginal_risk': marginal_risk[i],
                    'risk_contribution': weight * marginal_risk[i],
                    'percentage_contribution': (weight * marginal_risk[i]) / portfolio_volatility if portfolio_volatility != 0 else 0
                }
            
            # Calculate diversification ratio
            individual_variances = np.array([portfolio_weights.get(asset, 0.0) ** 2 * covariance_matrix.loc[asset, asset] 
                                          for asset in asset_returns.keys()])
            sum_individual_variances = np.sum(individual_variances)
            diversification_ratio = sum_individual_variances / portfolio_variance if portfolio_variance != 0 else 1
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_variance': portfolio_variance,
                'risk_contributions': risk_contributions,
                'diversification_ratio': diversification_ratio,
                'covariance_matrix': covariance_matrix
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk decomposition: {e}")
            raise PortfolioAnalyticsError(f"Risk decomposition failed: {str(e)}")
    
    def generate_analytics_report(self, 
                                portfolio_returns: pd.Series,
                                asset_returns: Optional[Dict[str, pd.Series]] = None,
                                portfolio_weights: Optional[Dict[str, float]] = None,
                                benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio analytics report.
        
        Args:
            portfolio_returns: Portfolio returns
            asset_returns: Asset returns (optional)
            portfolio_weights: Portfolio weights (optional)
            benchmark_returns: Benchmark returns (optional)
            
        Returns:
            Dictionary containing comprehensive analytics report
        """
        try:
            report = {}
            
            # Comprehensive performance metrics
            report['performance_metrics'] = self.calculate_comprehensive_metrics(
                portfolio_returns, benchmark_returns
            )
            
            # Return attribution if asset data available
            if asset_returns and portfolio_weights:
                report['return_attribution'] = self.perform_return_attribution(
                    portfolio_returns, asset_returns, portfolio_weights
                )
                
                report['risk_decomposition'] = self.calculate_risk_decomposition(
                    portfolio_weights, asset_returns
                )
            
            # Analytics summary
            report['analytics_summary'] = self._generate_analytics_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analytics report: {e}")
            raise PortfolioAnalyticsError(f"Analytics report generation failed: {str(e)}")
    
    def _generate_analytics_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analytics results."""
        try:
            performance = report.get('performance_metrics', {})
            
            summary = {
                'overall_performance': self._classify_performance(performance),
                'key_metrics': self._extract_key_metrics(performance),
                'risk_assessment': self._assess_risk_profile(performance),
                'recommendations': self._generate_analytics_recommendations(report)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analytics summary: {e}")
            return {}
    
    def _classify_performance(self, performance: Dict[str, Any]) -> str:
        """Classify overall portfolio performance."""
        try:
            risk_adjusted = performance.get('risk_adjusted_metrics', {})
            sharpe = risk_adjusted.get('annualized_sharpe', 0)
            
            if sharpe > 1.0:
                return 'Excellent'
            elif sharpe > 0.5:
                return 'Good'
            elif sharpe > 0.0:
                return 'Fair'
            else:
                return 'Poor'
                
        except Exception as e:
            logger.error(f"Error classifying performance: {e}")
            return 'Unknown'
    
    def _extract_key_metrics(self, performance: Dict[str, Any]) -> Dict[str, float]:
        """Extract key performance metrics."""
        try:
            return_metrics = performance.get('return_metrics', {})
            risk_metrics = performance.get('risk_metrics', {})
            risk_adjusted = performance.get('risk_adjusted_metrics', {})
            
            key_metrics = {
                'total_return': return_metrics.get('total_return', 0),
                'annualized_return': return_metrics.get('annualized_return', 0),
                'volatility': risk_metrics.get('annualized_volatility', 0),
                'sharpe_ratio': risk_adjusted.get('annualized_sharpe', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0)
            }
            
            return key_metrics
            
        except Exception as e:
            logger.error(f"Error extracting key metrics: {e}")
            return {}
    
    def _assess_risk_profile(self, performance: Dict[str, Any]) -> str:
        """Assess portfolio risk profile."""
        try:
            risk_metrics = performance.get('risk_metrics', {})
            volatility = risk_metrics.get('annualized_volatility', 0)
            max_dd = abs(risk_metrics.get('max_drawdown', 0))
            
            if volatility < 0.10 and max_dd < 0.10:
                return 'Conservative'
            elif volatility < 0.20 and max_dd < 0.20:
                return 'Moderate'
            else:
                return 'Aggressive'
                
        except Exception as e:
            logger.error(f"Error assessing risk profile: {e}")
            return 'Unknown'
    
    def _generate_analytics_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analytics results."""
        try:
            recommendations = []
            performance = report.get('performance_metrics', {})
            
            # Performance-based recommendations
            risk_adjusted = performance.get('risk_adjusted_metrics', {})
            sharpe = risk_adjusted.get('annualized_sharpe', 0)
            
            if sharpe < 0.5:
                recommendations.append('Consider improving risk-adjusted returns through better asset selection')
            
            # Risk-based recommendations
            risk_metrics = performance.get('risk_metrics', {})
            volatility = risk_metrics.get('annualized_volatility', 0)
            
            if volatility > 0.20:
                recommendations.append('High volatility suggests need for better diversification')
            
            # Drawdown recommendations
            max_dd = abs(risk_metrics.get('max_drawdown', 0))
            if max_dd > 0.20:
                recommendations.append('Large drawdowns indicate need for risk management strategies')
            
            # General recommendations
            recommendations.append('Regular portfolio rebalancing to maintain target allocations')
            recommendations.append('Monitor correlation changes between assets')
            recommendations.append('Consider tactical asset allocation based on market conditions')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ['Consult with portfolio management professionals']


def create_portfolio_analytics(config: Optional[Dict[str, Any]] = None) -> PortfolioAnalytics:
    """Factory function to create portfolio analytics."""
    return PortfolioAnalytics(config)
