"""
Portfolio Optimization Module

This module provides comprehensive portfolio optimization tools including
risk-return optimization, asset allocation strategies, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy.optimize import minimize
from scipy import stats
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    A comprehensive portfolio optimization class.
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
        """
        Initialize the PortfolioOptimizer.
        
        Args:
            returns (pd.DataFrame): Asset returns data
            risk_free_rate (float): Risk-free rate for calculations
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.covariance_matrix = None
        self.optimal_weights = None
        self.optimal_metrics = None
        self.efficient_frontier = None
        
        # Validate data
        if not isinstance(returns, pd.DataFrame):
            raise ValueError("Returns must be a pandas DataFrame")
        if returns.empty:
            raise ValueError("Returns data cannot be empty")
        
        # Calculate expected returns and covariance matrix
        self._calculate_statistics()
    
    def _calculate_statistics(self) -> None:
        """Calculate expected returns and covariance matrix."""
        # Annualized expected returns
        self.expected_returns = self.returns.mean() * 252
        
        # Annualized covariance matrix
        self.covariance_matrix = self.returns.cov() * 252
        
        logger.info("Portfolio statistics calculated successfully")
    
    def calculate_portfolio_metrics(self, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio return, risk, and Sharpe ratio.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            Dict[str, float]: Portfolio metrics
        """
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Portfolio return
        portfolio_return = np.sum(self.expected_returns * weights)
        
        # Portfolio risk
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Additional metrics
        var_95 = self._calculate_var(weights, 0.05)
        cvar_95 = self._calculate_cvar(weights, 0.05)
        
        return {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_var(self, weights: np.ndarray, confidence_level: float) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            weights (np.ndarray): Portfolio weights
            confidence_level (float): Confidence level (e.g., 0.05 for 95%)
            
        Returns:
            float: Value at Risk
        """
        portfolio_returns = np.dot(self.returns, weights)
        return np.percentile(portfolio_returns, confidence_level * 100)
    
    def _calculate_cvar(self, weights: np.ndarray, confidence_level: float) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            weights (np.ndarray): Portfolio weights
            confidence_level (float): Confidence level (e.g., 0.05 for 95%)
            
        Returns:
            float: Conditional Value at Risk
        """
        portfolio_returns = np.dot(self.returns, weights)
        var = np.percentile(portfolio_returns, confidence_level * 100)
        return portfolio_returns[portfolio_returns <= var].mean()
    
    def optimize_sharpe_ratio(self, constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights to maximize Sharpe ratio.
        
        Args:
            constraints (Dict, optional): Additional optimization constraints
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        n_assets = len(self.returns.columns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'no_short_selling': True,
                'max_weight': 1.0,
                'min_weight': 0.0
            }
        
        # Objective function (negative Sharpe ratio for minimization)
        def objective(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics['sharpe_ratio']
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds
        if constraints.get('no_short_selling', True):
            bounds = tuple((constraints.get('min_weight', 0), constraints.get('max_weight', 1)) 
                          for _ in range(n_assets))
        else:
            bounds = tuple((-1, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            self.optimal_weights = result.x
            self.optimal_metrics = self.calculate_portfolio_metrics(result.x)
            
            logger.info("Portfolio optimization completed successfully")
            logger.info(f"Optimal Sharpe ratio: {self.optimal_metrics['sharpe_ratio']:.4f}")
            
            return {
                'weights': self.optimal_weights,
                'metrics': self.optimal_metrics,
                'success': True
            }
        else:
            logger.error(f"Portfolio optimization failed: {result.message}")
            return {
                'weights': None,
                'metrics': None,
                'success': False,
                'message': result.message
            }
    
    def optimize_minimum_variance(self, target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights to minimize variance.
        
        Args:
            target_return (float, optional): Target return constraint
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        n_assets = len(self.returns.columns)
        
        # Objective function (portfolio variance)
        def objective(weights):
            return np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        if target_return is not None:
            constraint_list.append({
                'type': 'eq', 
                'fun': lambda x: np.sum(self.expected_returns * x) - target_return
            })
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = result.x
            metrics = self.calculate_portfolio_metrics(weights)
            
            logger.info("Minimum variance optimization completed successfully")
            
            return {
                'weights': weights,
                'metrics': metrics,
                'success': True
            }
        else:
            logger.error(f"Minimum variance optimization failed: {result.message}")
            return {
                'weights': None,
                'metrics': None,
                'success': False,
                'message': result.message
            }
    
    def generate_efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.
        
        Args:
            num_portfolios (int): Number of portfolios to generate
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        logger.info(f"Generating efficient frontier with {num_portfolios} portfolios")
        
        # Generate target returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                result = self.optimize_minimum_variance(target_return)
                if result['success']:
                    portfolio_data = {
                        'target_return': target_return,
                        'actual_return': result['metrics']['return'],
                        'risk': result['metrics']['risk'],
                        'sharpe_ratio': result['metrics']['sharpe_ratio'],
                        'weights': result['weights']
                    }
                    efficient_portfolios.append(portfolio_data)
            except:
                continue
        
        if efficient_portfolios:
            self.efficient_frontier = pd.DataFrame(efficient_portfolios)
            logger.info(f"Efficient frontier generated with {len(efficient_portfolios)} portfolios")
            return self.efficient_frontier
        else:
            logger.warning("Could not generate efficient frontier")
            return pd.DataFrame()
    
    def calculate_equal_weight_portfolio(self) -> Dict[str, Any]:
        """
        Calculate metrics for equal weight portfolio.
        
        Returns:
            Dict[str, Any]: Equal weight portfolio metrics
        """
        n_assets = len(self.returns.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        metrics = self.calculate_portfolio_metrics(equal_weights)
        
        return {
            'weights': equal_weights,
            'metrics': metrics
        }
    
    def calculate_value_weighted_portfolio(self, market_caps: pd.Series) -> Dict[str, Any]:
        """
        Calculate metrics for value-weighted portfolio.
        
        Args:
            market_caps (pd.Series): Market capitalization for each asset
            
        Returns:
            Dict[str, Any]: Value-weighted portfolio metrics
        """
        # Normalize market caps to weights
        value_weights = market_caps / market_caps.sum()
        
        metrics = self.calculate_portfolio_metrics(value_weights.values)
        
        return {
            'weights': value_weights.values,
            'metrics': metrics
        }
    
    def calculate_max_diversification_portfolio(self) -> Dict[str, Any]:
        """
        Calculate maximum diversification portfolio.
        
        Returns:
            Dict[str, Any]: Maximum diversification portfolio metrics
        """
        n_assets = len(self.returns.columns)
        
        # Objective function (negative diversification ratio)
        def objective(weights):
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            weighted_asset_risk = np.sum(weights * np.sqrt(np.diag(self.covariance_matrix)))
            diversification_ratio = weighted_asset_risk / portfolio_risk
            return -diversification_ratio
        
        # Constraints
        constraint_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            objective, initial_weights, method='SLSQP',
            bounds=bounds, constraints=constraint_list
        )
        
        if result.success:
            weights = result.x
            metrics = self.calculate_portfolio_metrics(weights)
            
            logger.info("Maximum diversification portfolio calculated successfully")
            
            return {
                'weights': weights,
                'metrics': metrics,
                'success': True
            }
        else:
            logger.error(f"Maximum diversification optimization failed: {result.message}")
            return {
                'weights': None,
                'metrics': None,
                'success': False,
                'message': result.message
            }
    
    def calculate_portfolio_contribution(self, weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate risk and return contribution of each asset.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            pd.DataFrame: Asset contribution analysis
        """
        # Return contribution
        return_contribution = weights * self.expected_returns
        
        # Risk contribution
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
        risk_contribution = weights * np.dot(self.covariance_matrix, weights) / portfolio_risk
        
        # Create contribution dataframe
        contribution_df = pd.DataFrame({
            'Weight': weights,
            'Expected_Return': self.expected_returns,
            'Return_Contribution': return_contribution,
            'Risk_Contribution': risk_contribution
        }, index=self.returns.columns)
        
        # Calculate percentages
        contribution_df['Return_Contribution_Pct'] = contribution_df['Return_Contribution'] / contribution_df['Return_Contribution'].sum() * 100
        contribution_df['Risk_Contribution_Pct'] = contribution_df['Risk_Contribution'] / contribution_df['Risk_Contribution'].sum() * 100
        
        return contribution_df
    
    def calculate_portfolio_statistics(self, weights: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio statistics.
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            Dict[str, Any]: Portfolio statistics
        """
        metrics = self.calculate_portfolio_metrics(weights)
        
        # Calculate additional statistics
        portfolio_returns = np.dot(self.returns, weights)
        
        # Skewness and kurtosis
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = metrics['return'] / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = (metrics['return'] - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Information ratio (assuming benchmark is risk-free rate)
        excess_returns = portfolio_returns - self.risk_free_rate/252
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Add to metrics
        additional_metrics = {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio
        }
        
        metrics.update(additional_metrics)
        
        return metrics
    
    def generate_portfolio_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive portfolio report.
        
        Args:
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Portfolio report
        """
        report = []
        report.append("# Portfolio Optimization Report")
        report.append("=" * 50)
        report.append("")
        
        # Portfolio information
        report.append("## Portfolio Information")
        report.append(f"- **Number of assets**: {len(self.returns.columns)}")
        report.append(f"- **Time period**: {len(self.returns)} observations")
        report.append(f"- **Risk-free rate**: {self.risk_free_rate:.2%}")
        report.append("")
        
        # Asset information
        report.append("## Asset Information")
        asset_info = pd.DataFrame({
            'Expected_Return': self.expected_returns,
            'Volatility': np.sqrt(np.diag(self.covariance_matrix))
        })
        report.append(asset_info.to_string())
        report.append("")
        
        # Optimal portfolio
        if self.optimal_weights is not None:
            report.append("## Optimal Portfolio (Maximum Sharpe Ratio)")
            report.append(f"- **Return**: {self.optimal_metrics['return']:.4f}")
            report.append(f"- **Risk**: {self.optimal_metrics['risk']:.4f}")
            report.append(f"- **Sharpe Ratio**: {self.optimal_metrics['sharpe_ratio']:.4f}")
            report.append("")
            
            # Asset weights
            report.append("### Asset Weights")
            weights_df = pd.DataFrame({
                'Asset': self.returns.columns,
                'Weight': self.optimal_weights
            })
            weights_df = weights_df.sort_values('Weight', ascending=False)
            report.append(weights_df.to_string(index=False))
            report.append("")
        
        # Efficient frontier
        if self.efficient_frontier is not None:
            report.append("## Efficient Frontier")
            report.append(f"- **Number of portfolios**: {len(self.efficient_frontier)}")
            report.append(f"- **Return range**: {self.efficient_frontier['actual_return'].min():.4f} to {self.efficient_frontier['actual_return'].max():.4f}")
            report.append(f"- **Risk range**: {self.efficient_frontier['risk'].min():.4f} to {self.efficient_frontier['risk'].max():.4f}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if self.optimal_weights is not None:
            report.append("- Consider the optimal portfolio for maximum risk-adjusted returns")
            report.append("- Monitor portfolio performance and rebalance periodically")
        
        report.append("- Diversify across different asset classes and sectors")
        report.append("- Consider transaction costs and tax implications")
        report.append("- Regularly review and update the optimization model")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Portfolio report saved to {save_path}")
        
        return report_text
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to a file.
        
        Args:
            filepath (str): Path to save the results
        """
        try:
            import pickle
            
            results_data = {
                'optimal_weights': self.optimal_weights,
                'optimal_metrics': self.optimal_metrics,
                'efficient_frontier': self.efficient_frontier,
                'expected_returns': self.expected_returns,
                'covariance_matrix': self.covariance_matrix,
                'risk_free_rate': self.risk_free_rate
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Portfolio optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


def run_portfolio_optimization(returns: pd.DataFrame, 
                              risk_free_rate: float = 0.02,
                              output_dir: str = "data/processed/") -> PortfolioOptimizer:
    """
    Run complete portfolio optimization pipeline.
    
    Args:
        returns (pd.DataFrame): Asset returns data
        risk_free_rate (float): Risk-free rate
        output_dir (str): Directory to save outputs
        
    Returns:
        PortfolioOptimizer: Configured optimizer with results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns, risk_free_rate)
    
    # Run optimizations
    logger.info("Running portfolio optimizations...")
    
    # Maximum Sharpe ratio optimization
    sharpe_result = optimizer.optimize_sharpe_ratio()
    if sharpe_result['success']:
        logger.info("Maximum Sharpe ratio optimization completed")
    
    # Minimum variance optimization
    min_var_result = optimizer.optimize_minimum_variance()
    if min_var_result['success']:
        logger.info("Minimum variance optimization completed")
    
    # Maximum diversification optimization
    max_div_result = optimizer.calculate_max_diversification_portfolio()
    if max_div_result['success']:
        logger.info("Maximum diversification optimization completed")
    
    # Generate efficient frontier
    optimizer.generate_efficient_frontier()
    
    # Generate report
    try:
        report_path = os.path.join(output_dir, 'portfolio_report.md')
        optimizer.generate_portfolio_report(report_path)
    except Exception as e:
        logger.warning(f"Could not generate portfolio report: {str(e)}")
    
    # Save results
    try:
        optimizer.save_results(os.path.join(output_dir, 'portfolio_optimization_results.pkl'))
    except Exception as e:
        logger.warning(f"Could not save results: {str(e)}")
    
    logger.info("Portfolio optimization pipeline completed")
    return optimizer


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data and create returns
    sample_data = load_sample_data()
    
    # Create sample returns for multiple assets
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Generate sample returns for 4 assets
    returns_data = pd.DataFrame({
        'Asset_A': np.random.normal(0.001, 0.02, 252),
        'Asset_B': np.random.normal(0.0008, 0.025, 252),
        'Asset_C': np.random.normal(0.0012, 0.018, 252),
        'Asset_D': np.random.normal(0.0009, 0.022, 252)
    }, index=dates)
    
    print("Sample returns data created, running portfolio optimization...")
    
    # Run portfolio optimization
    optimizer = run_portfolio_optimization(returns_data, risk_free_rate=0.02)
    
    print("Portfolio optimization completed successfully!")
    
    if optimizer.optimal_weights is not None:
        print(f"Optimal Sharpe ratio: {optimizer.optimal_metrics['sharpe_ratio']:.4f}")
        print(f"Optimal return: {optimizer.optimal_metrics['return']:.4f}")
        print(f"Optimal risk: {optimizer.optimal_metrics['risk']:.4f}")
