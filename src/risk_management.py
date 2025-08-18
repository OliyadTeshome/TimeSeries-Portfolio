"""
Advanced Risk Management Module

This module provides comprehensive risk management tools including VaR, CVaR,
stress testing, scenario analysis, and other advanced risk metrics for financial portfolios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import warnings
import logging
from scipy import stats
from scipy.optimize import minimize
import cvxpy as cp

from .config import get_config
from .logging_config import setup_logging
from .exceptions import PortfolioError as RiskManagementError

# Setup logging
logger = setup_logging()
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class RiskManager:
    """
    Comprehensive risk management system for financial portfolios.
    
    This class provides advanced risk metrics, stress testing, scenario analysis,
    and risk optimization capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk manager.
        
        Args:
            config: Configuration dictionary for risk parameters
        """
        self.config = config or get_config()
        self.risk_metrics = {}
        self.stress_test_results = {}
        self.scenario_results = {}
        
        # Default risk parameters
        self.default_confidence_levels = [0.90, 0.95, 0.99]
        self.default_time_horizons = [1, 5, 10, 21]  # days
        self.default_scenarios = self._initialize_default_scenarios()
    
    def _initialize_default_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Initialize default stress test scenarios."""
        return {
            'market_crash': {
                'description': 'Global market crash (-30% across all assets)',
                'equity_shock': -0.30,
                'bond_shock': -0.15,
                'commodity_shock': -0.25,
                'currency_shock': -0.10
            },
            'interest_rate_spike': {
                'description': 'Sharp increase in interest rates',
                'equity_shock': -0.15,
                'bond_shock': -0.25,
                'commodity_shock': -0.10,
                'currency_shock': 0.05
            },
            'inflation_surge': {
                'description': 'Unexpected inflation surge',
                'equity_shock': -0.10,
                'bond_shock': -0.20,
                'commodity_shock': 0.15,
                'currency_shock': -0.05
            },
            'liquidity_crisis': {
                'description': 'Market liquidity crisis',
                'equity_shock': -0.20,
                'bond_shock': -0.30,
                'commodity_shock': -0.35,
                'currency_shock': -0.15
            },
            'geopolitical_crisis': {
                'description': 'Major geopolitical event',
                'equity_shock': -0.25,
                'bond_shock': -0.10,
                'commodity_shock': 0.20,
                'currency_shock': -0.20
            }
        }
    
    def calculate_var(self, 
                     returns: Union[pd.Series, np.ndarray], 
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Asset or portfolio returns
            confidence_level: Confidence level for VaR calculation
            method: Method for VaR calculation ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            VaR value at specified confidence level
        """
        try:
            if isinstance(returns, pd.Series):
                returns = returns.dropna().values
            else:
                returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                raise RiskManagementError("No valid returns data provided")
            
            if method == 'historical':
                var = np.percentile(returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                var = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
            elif method == 'monte_carlo':
                var = self._calculate_monte_carlo_var(returns, confidence_level)
            else:
                raise ValueError(f"Unknown VaR method: {method}")
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            raise RiskManagementError(f"VaR calculation failed: {str(e)}")
    
    def calculate_cvar(self, 
                      returns: Union[pd.Series, np.ndarray], 
                      confidence_level: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.
        
        Args:
            returns: Asset or portfolio returns
            confidence_level: Confidence level for CVaR calculation
            method: Method for CVaR calculation ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            CVaR value at specified confidence level
        """
        try:
            if isinstance(returns, pd.Series):
                returns = returns.dropna().values
            else:
                returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                raise RiskManagementError("No valid returns data provided")
            
            if method == 'historical':
                var = self.calculate_var(returns, confidence_level, 'historical')
                cvar = np.mean(returns[returns <= var])
            elif method == 'parametric':
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                var = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
                cvar = mean_return - std_return * stats.norm.pdf(stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
            elif method == 'monte_carlo':
                cvar = self._calculate_monte_carlo_cvar(returns, confidence_level)
            else:
                raise ValueError(f"Unknown CVaR method: {method}")
            
            return cvar
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            raise RiskManagementError(f"CVaR calculation failed: {str(e)}")
    
    def _calculate_monte_carlo_var(self, 
                                 returns: np.ndarray, 
                                 confidence_level: float,
                                 n_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation."""
        try:
            # Fit distribution to returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate Monte Carlo samples
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            return var
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR: {e}")
            return np.nan
    
    def _calculate_monte_carlo_cvar(self, 
                                  returns: np.ndarray, 
                                  confidence_level: float,
                                  n_simulations: int = 10000) -> float:
        """Calculate CVaR using Monte Carlo simulation."""
        try:
            # Calculate VaR first
            var = self._calculate_monte_carlo_var(returns, confidence_level, n_simulations)
            
            # Fit distribution to returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate Monte Carlo samples
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate CVaR
            cvar = np.mean(simulated_returns[simulated_returns <= var])
            
            return cvar
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo CVaR: {e}")
            return np.nan
    
    def calculate_portfolio_risk_metrics(self, 
                                       portfolio_returns: Union[pd.Series, np.ndarray],
                                       confidence_levels: Optional[List[float]] = None,
                                       time_horizons: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_returns: Portfolio returns time series
            confidence_levels: List of confidence levels for risk calculations
            time_horizons: List of time horizons for risk calculations
            
        Returns:
            Dictionary containing comprehensive risk metrics
        """
        try:
            if confidence_levels is None:
                confidence_levels = self.default_confidence_levels
            if time_horizons is None:
                time_horizons = self.default_time_horizons
            
            # Basic statistics
            returns = portfolio_returns.dropna() if isinstance(portfolio_returns, pd.Series) else portfolio_returns[~np.isnan(portfolio_returns)]
            
            basic_stats = {
                'mean_return': np.mean(returns),
                'volatility': np.std(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns)
            }
            
            # Risk metrics at different confidence levels
            risk_metrics = {}
            for conf_level in confidence_levels:
                risk_metrics[f'var_{int(conf_level*100)}'] = self.calculate_var(returns, conf_level)
                risk_metrics[f'cvar_{int(conf_level*100)}'] = self.calculate_cvar(returns, conf_level)
            
            # Time horizon risk metrics
            horizon_metrics = {}
            for horizon in time_horizons:
                if len(returns) >= horizon:
                    horizon_returns = returns[-horizon:]
                    horizon_metrics[f'horizon_{horizon}'] = {
                        'var_95': self.calculate_var(horizon_returns, 0.95),
                        'volatility': np.std(horizon_returns),
                        'max_drawdown': self._calculate_max_drawdown(horizon_returns)
                    }
            
            # Additional risk metrics
            additional_metrics = {
                'max_drawdown': self._calculate_max_drawdown(returns),
                'calmar_ratio': self._calculate_calmar_ratio(returns),
                'sortino_ratio': self._calculate_sortino_ratio(returns),
                'var_ratio': self._calculate_var_ratio(returns),
                'tail_dependence': self._calculate_tail_dependence(returns)
            }
            
            # Compile all metrics
            all_metrics = {
                'basic_statistics': basic_stats,
                'risk_metrics': risk_metrics,
                'horizon_metrics': horizon_metrics,
                'additional_metrics': additional_metrics,
                'data_info': {
                    'total_observations': len(returns),
                    'date_range': f"{len(returns)} periods"
                }
            }
            
            self.risk_metrics = all_metrics
            return all_metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk metrics: {e}")
            raise RiskManagementError(f"Portfolio risk metrics calculation failed: {str(e)}")
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        try:
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return np.min(drawdown)
        except Exception:
            return np.nan
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        try:
            annualized_return = np.mean(returns) * 252
            max_dd = abs(self._calculate_max_drawdown(returns))
            return annualized_return / max_dd if max_dd != 0 else np.nan
        except Exception:
            return np.nan
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio."""
        try:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns)
            return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else np.nan
        except Exception:
            return np.nan
    
    def _calculate_var_ratio(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate VaR ratio (VaR / volatility)."""
        try:
            var = self.calculate_var(returns, confidence_level)
            volatility = np.std(returns)
            return abs(var) / volatility if volatility != 0 else np.nan
        except Exception:
            return np.nan
    
    def _calculate_tail_dependence(self, returns: np.ndarray, threshold: float = 0.05) -> float:
        """Calculate tail dependence coefficient."""
        try:
            # Calculate exceedances above threshold
            threshold_value = np.percentile(returns, (1 - threshold) * 100)
            exceedances = returns[returns > threshold_value]
            return len(exceedances) / len(returns)
        except Exception:
            return np.nan
    
    def run_stress_test(self, 
                       portfolio_weights: Dict[str, float],
                       asset_returns: Dict[str, np.ndarray],
                       scenarios: Optional[Dict[str, Dict[str, float]]] = None,
                       custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Run comprehensive stress testing on portfolio.
        
        Args:
            portfolio_weights: Dictionary mapping asset names to weights
            asset_returns: Dictionary mapping asset names to return arrays
            scenarios: Predefined stress test scenarios
            custom_scenarios: User-defined custom scenarios
            
        Returns:
            Dictionary containing stress test results
        """
        try:
            if scenarios is None:
                scenarios = self.default_scenarios
            
            if custom_scenarios:
                scenarios.update(custom_scenarios)
            
            stress_test_results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                logger.info(f"Running stress test: {scenario_name}")
                
                # Apply scenario shocks to asset returns
                shocked_returns = self._apply_scenario_shocks(
                    asset_returns, portfolio_weights, scenario_params
                )
                
                # Calculate portfolio impact
                portfolio_impact = self._calculate_portfolio_impact(
                    portfolio_weights, shocked_returns
                )
                
                # Calculate risk metrics for shocked scenario
                shocked_risk_metrics = self.calculate_portfolio_risk_metrics(
                    shocked_returns['portfolio']
                )
                
                stress_test_results[scenario_name] = {
                    'description': scenario_params.get('description', ''),
                    'portfolio_impact': portfolio_impact,
                    'risk_metrics': shocked_risk_metrics,
                    'scenario_params': scenario_params,
                    'shocked_returns': shocked_returns
                }
            
            self.stress_test_results = stress_test_results
            return stress_test_results
            
        except Exception as e:
            logger.error(f"Error running stress test: {e}")
            raise RiskManagementError(f"Stress testing failed: {str(e)}")
    
    def _apply_scenario_shocks(self, 
                              asset_returns: Dict[str, np.ndarray],
                              portfolio_weights: Dict[str, np.ndarray],
                              scenario_params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Apply scenario shocks to asset returns."""
        try:
            shocked_returns = {}
            
            for asset_name, returns in asset_returns.items():
                # Determine asset type and apply appropriate shock
                if 'equity' in asset_name.lower() or asset_name in ['SPY', 'AAPL', 'MSFT']:
                    shock = scenario_params.get('equity_shock', 0.0)
                elif 'bond' in asset_name.lower() or asset_name in ['BND', 'TLT']:
                    shock = scenario_params.get('bond_shock', 0.0)
                elif 'commodity' in asset_name.lower() or asset_name in ['GLD', 'SLV']:
                    shock = scenario_params.get('commodity_shock', 0.0)
                else:
                    shock = scenario_params.get('equity_shock', 0.0)  # Default to equity shock
                
                # Apply shock to returns
                shocked_returns[asset_name] = returns * (1 + shock)
            
            # Calculate portfolio returns
            portfolio_returns = np.zeros(len(list(asset_returns.values())[0]))
            for asset_name, returns in shocked_returns.items():
                weight = portfolio_weights.get(asset_name, 0.0)
                portfolio_returns += weight * returns
            
            shocked_returns['portfolio'] = portfolio_returns
            
            return shocked_returns
            
        except Exception as e:
            logger.error(f"Error applying scenario shocks: {e}")
            raise RiskManagementError(f"Scenario shock application failed: {str(e)}")
    
    def _calculate_portfolio_impact(self, 
                                  portfolio_weights: Dict[str, float],
                                  shocked_returns: Dict[str, np.ndarray]) -> float:
        """Calculate portfolio impact from stress test scenario."""
        try:
            # Calculate cumulative portfolio return
            portfolio_returns = shocked_returns['portfolio']
            cumulative_return = np.prod(1 + portfolio_returns) - 1
            
            return cumulative_return
            
        except Exception as e:
            logger.error(f"Error calculating portfolio impact: {e}")
            return np.nan
    
    def run_scenario_analysis(self, 
                            portfolio_weights: Dict[str, float],
                            asset_returns: Dict[str, np.ndarray],
                            scenarios: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Run scenario analysis with custom market conditions.
        
        Args:
            portfolio_weights: Dictionary mapping asset names to weights
            asset_returns: Dictionary mapping asset names to return arrays
            scenarios: Dictionary of scenario definitions
            
        Returns:
            Dictionary containing scenario analysis results
        """
        try:
            if scenarios is None:
                scenarios = {
                    'bull_market': {'equity_shock': 0.20, 'bond_shock': 0.05},
                    'bear_market': {'equity_shock': -0.20, 'bond_shock': 0.10},
                    'high_volatility': {'volatility_multiplier': 2.0},
                    'low_volatility': {'volatility_multiplier': 0.5},
                    'correlation_breakdown': {'correlation_shock': 0.3}
                }
            
            scenario_results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                logger.info(f"Running scenario analysis: {scenario_name}")
                
                # Apply scenario parameters
                modified_returns = self._apply_scenario_parameters(
                    asset_returns, portfolio_weights, scenario_params
                )
                
                # Calculate portfolio performance
                portfolio_performance = self._calculate_scenario_performance(
                    portfolio_weights, modified_returns
                )
                
                # Calculate risk metrics
                risk_metrics = self.calculate_portfolio_risk_metrics(
                    modified_returns['portfolio']
                )
                
                scenario_results[scenario_name] = {
                    'scenario_params': scenario_params,
                    'portfolio_performance': portfolio_performance,
                    'risk_metrics': risk_metrics,
                    'modified_returns': modified_returns
                }
            
            self.scenario_results = scenario_results
            return scenario_results
            
        except Exception as e:
            logger.error(f"Error running scenario analysis: {e}")
            raise RiskManagementError(f"Scenario analysis failed: {str(e)}")
    
    def _apply_scenario_parameters(self, 
                                 asset_returns: Dict[str, np.ndarray],
                                 portfolio_weights: Dict[str, np.ndarray],
                                 scenario_params: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Apply scenario parameters to asset returns."""
        try:
            modified_returns = {}
            
            for asset_name, returns in asset_returns.items():
                modified_returns[asset_name] = returns.copy()
                
                # Apply volatility multiplier if specified
                if 'volatility_multiplier' in scenario_params:
                    vol_mult = scenario_params['volatility_multiplier']
                    mean_return = np.mean(returns)
                    modified_returns[asset_name] = mean_return + (returns - mean_return) * vol_mult
                
                # Apply correlation breakdown if specified
                if 'correlation_shock' in scenario_params:
                    # This is a simplified implementation
                    # In practice, you would modify the correlation matrix
                    pass
            
            # Calculate portfolio returns
            portfolio_returns = np.zeros(len(list(asset_returns.values())[0]))
            for asset_name, returns in modified_returns.items():
                weight = portfolio_weights.get(asset_name, 0.0)
                portfolio_returns += weight * returns
            
            modified_returns['portfolio'] = portfolio_returns
            
            return modified_returns
            
        except Exception as e:
            logger.error(f"Error applying scenario parameters: {e}")
            raise RiskManagementError(f"Scenario parameter application failed: {str(e)}")
    
    def _calculate_scenario_performance(self, 
                                      portfolio_weights: Dict[str, float],
                                      modified_returns: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate portfolio performance metrics for scenario."""
        try:
            portfolio_returns = modified_returns['portfolio']
            
            performance = {
                'total_return': np.prod(1 + portfolio_returns) - 1,
                'annualized_return': np.mean(portfolio_returns) * 252,
                'volatility': np.std(portfolio_returns) * np.sqrt(252),
                'sharpe_ratio': np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) != 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio_returns)
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating scenario performance: {e}")
            return {}
    
    def optimize_risk_budget(self, 
                           expected_returns: np.ndarray,
                           covariance_matrix: np.ndarray,
                           target_return: float,
                           risk_budget_method: str = 'equal') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize portfolio weights using risk budgeting approach.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            target_return: Target portfolio return
            risk_budget_method: Risk budgeting method ('equal', 'proportional', 'custom')
            
        Returns:
            Tuple of optimal weights and optimization results
        """
        try:
            n_assets = len(expected_returns)
            
            if risk_budget_method == 'equal':
                # Equal risk contribution
                weights = self._optimize_equal_risk_contribution(covariance_matrix, target_return)
            elif risk_budget_method == 'proportional':
                # Proportional to expected returns
                weights = self._optimize_proportional_risk_budget(expected_returns, covariance_matrix, target_return)
            else:
                # Custom risk budget
                weights = self._optimize_custom_risk_budget(expected_returns, covariance_matrix, target_return)
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(weights, covariance_matrix)
            
            results = {
                'optimal_weights': weights,
                'portfolio_return': portfolio_return,
                'portfolio_volatility': portfolio_volatility,
                'risk_contributions': risk_contributions,
                'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            }
            
            return weights, results
            
        except Exception as e:
            logger.error(f"Error optimizing risk budget: {e}")
            raise RiskManagementError(f"Risk budget optimization failed: {str(e)}")
    
    def _optimize_equal_risk_contribution(self, 
                                        covariance_matrix: np.ndarray,
                                        target_return: float) -> np.ndarray:
        """Optimize for equal risk contribution."""
        try:
            n_assets = covariance_matrix.shape[0]
            
            # Define optimization problem
            weights = cp.Variable(n_assets)
            
            # Objective: minimize portfolio variance
            portfolio_variance = cp.quad_form(weights, covariance_matrix)
            objective = cp.Minimize(portfolio_variance)
            
            # Constraints
            constraints = [
                cp.sum(weights) == 1,  # Weights sum to 1
                weights >= 0,  # Long-only portfolio
                weights <= 1   # Maximum 100% in any asset
            ]
            
            # Solve optimization problem
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                return weights.value
            else:
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets
                
        except Exception as e:
            logger.error(f"Error in equal risk contribution optimization: {e}")
            return np.ones(covariance_matrix.shape[0]) / covariance_matrix.shape[0]
    
    def _optimize_proportional_risk_budget(self, 
                                         expected_returns: np.ndarray,
                                         covariance_matrix: np.ndarray,
                                         target_return: float) -> np.ndarray:
        """Optimize for proportional risk budget."""
        try:
            n_assets = len(expected_returns)
            
            # Simple proportional allocation based on expected returns
            # This is a simplified implementation
            weights = expected_returns / np.sum(expected_returns)
            
            # Normalize to sum to 1
            weights = weights / np.sum(weights)
            
            return weights
            
        except Exception as e:
            logger.error(f"Error in proportional risk budget optimization: {e}")
            return np.ones(n_assets) / n_assets
    
    def _optimize_custom_risk_budget(self, 
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   target_return: float) -> np.ndarray:
        """Optimize for custom risk budget."""
        try:
            n_assets = len(expected_returns)
            
            # This is a placeholder for custom risk budget optimization
            # In practice, you would implement specific risk budgeting logic
            
            # For now, return equal weights
            return np.ones(n_assets) / n_assets
            
        except Exception as e:
            logger.error(f"Error in custom risk budget optimization: {e}")
            return np.ones(n_assets) / n_assets
    
    def _calculate_risk_contributions(self, 
                                    weights: np.ndarray,
                                    covariance_matrix: np.ndarray) -> np.ndarray:
        """Calculate risk contributions for each asset."""
        try:
            portfolio_variance = weights.T @ covariance_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Marginal risk contribution
            marginal_risk = covariance_matrix @ weights / portfolio_volatility
            
            # Risk contribution
            risk_contributions = weights * marginal_risk
            
            return risk_contributions
            
        except Exception as e:
            logger.error(f"Error calculating risk contributions: {e}")
            return np.zeros(len(weights))
    
    def generate_risk_report(self, 
                           portfolio_returns: Union[pd.Series, np.ndarray],
                           portfolio_weights: Optional[Dict[str, float]] = None,
                           asset_returns: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk management report.
        
        Args:
            portfolio_returns: Portfolio returns time series
            portfolio_weights: Portfolio weights (optional, for stress testing)
            asset_returns: Individual asset returns (optional, for stress testing)
            
        Returns:
            Dictionary containing comprehensive risk report
        """
        try:
            report = {}
            
            # Basic risk metrics
            report['risk_metrics'] = self.calculate_portfolio_risk_metrics(portfolio_returns)
            
            # Stress testing if weights and asset returns provided
            if portfolio_weights and asset_returns:
                report['stress_testing'] = self.run_stress_test(
                    portfolio_weights, asset_returns
                )
                
                report['scenario_analysis'] = self.run_scenario_analysis(
                    portfolio_weights, asset_returns
                )
            
            # Risk summary
            report['risk_summary'] = self._generate_risk_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise RiskManagementError(f"Risk report generation failed: {str(e)}")
    
    def _generate_risk_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of risk metrics."""
        try:
            risk_metrics = report.get('risk_metrics', {})
            
            summary = {
                'total_risk_score': self._calculate_total_risk_score(risk_metrics),
                'risk_level': self._classify_risk_level(risk_metrics),
                'key_risk_factors': self._identify_key_risk_factors(risk_metrics),
                'recommendations': self._generate_risk_recommendations(risk_metrics)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating risk summary: {e}")
            return {}
    
    def _calculate_total_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score."""
        try:
            # Simple risk scoring based on key metrics
            score = 0.0
            
            # Volatility component
            volatility = risk_metrics.get('basic_statistics', {}).get('volatility', 0)
            score += min(volatility * 10, 40)  # Cap at 40 points
            
            # VaR component
            var_95 = risk_metrics.get('risk_metrics', {}).get('var_95', 0)
            score += min(abs(var_95) * 20, 30)  # Cap at 30 points
            
            # Drawdown component
            max_dd = risk_metrics.get('additional_metrics', {}).get('max_drawdown', 0)
            score += min(abs(max_dd) * 15, 30)  # Cap at 30 points
            
            return min(score, 100)  # Cap total score at 100
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0  # Default middle score
    
    def _classify_risk_level(self, risk_metrics: Dict[str, Any]) -> str:
        """Classify portfolio risk level."""
        try:
            risk_score = self._calculate_total_risk_score(risk_metrics)
            
            if risk_score < 30:
                return 'Low'
            elif risk_score < 60:
                return 'Medium'
            else:
                return 'High'
                
        except Exception as e:
            logger.error(f"Error classifying risk level: {e}")
            return 'Unknown'
    
    def _identify_key_risk_factors(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Identify key risk factors for the portfolio."""
        try:
            risk_factors = []
            
            # Check volatility
            volatility = risk_metrics.get('basic_statistics', {}).get('volatility', 0)
            if volatility > 0.02:  # 2% daily volatility
                risk_factors.append('High volatility')
            
            # Check VaR
            var_95 = risk_metrics.get('risk_metrics', {}).get('var_95', 0)
            if abs(var_95) > 0.03:  # 3% daily VaR
                risk_factors.append('High Value at Risk')
            
            # Check drawdown
            max_dd = risk_metrics.get('additional_metrics', {}).get('max_drawdown', 0)
            if abs(max_dd) > 0.20:  # 20% max drawdown
                risk_factors.append('Large maximum drawdown')
            
            # Check skewness
            skewness = risk_metrics.get('basic_statistics', {}).get('skewness', 0)
            if abs(skewness) > 1.0:
                risk_factors.append('Significant return skewness')
            
            if not risk_factors:
                risk_factors.append('Moderate risk profile')
            
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return ['Risk assessment error']
    
    def _generate_risk_recommendations(self, risk_metrics: Dict[str, Any]) -> List[str]:
        """Generate risk management recommendations."""
        try:
            recommendations = []
            
            # Volatility recommendations
            volatility = risk_metrics.get('basic_statistics', {}).get('volatility', 0)
            if volatility > 0.02:
                recommendations.append('Consider reducing portfolio volatility through diversification')
            
            # VaR recommendations
            var_95 = risk_metrics.get('risk_metrics', {}).get('var_95', 0)
            if abs(var_95) > 0.03:
                recommendations.append('Implement position sizing to limit daily losses')
            
            # Drawdown recommendations
            max_dd = risk_metrics.get('additional_metrics', {}).get('max_drawdown', 0)
            if abs(max_dd) > 0.20:
                recommendations.append('Consider stop-loss strategies to limit drawdowns')
            
            # General recommendations
            recommendations.append('Regular portfolio rebalancing to maintain target risk levels')
            recommendations.append('Monitor correlation changes between assets')
            recommendations.append('Consider hedging strategies for extreme market conditions')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ['Consult with risk management professionals']


def create_risk_manager(config: Optional[Dict[str, Any]] = None) -> RiskManager:
    """Factory function to create a risk manager."""
    return RiskManager(config)
