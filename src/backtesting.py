"""
Backtesting Module

This module provides comprehensive backtesting tools for evaluating trading strategies,
portfolio performance, and risk management approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class Backtester:
    """
    A comprehensive backtesting framework for trading strategies.
    """
    
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize the Backtester.
        
        Args:
            data (pd.DataFrame): Market data with OHLCV information
            initial_capital (float): Initial capital for backtesting
        """
        self.data = data
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
        self.strategies = {}
        self.results = {}
        
        # Validate data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if data.empty:
            raise ValueError("Data cannot be empty")
        
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index")
    
    def add_strategy(self, name: str, strategy_func: Callable, **kwargs) -> 'Backtester':
        """
        Add a trading strategy to the backtester.
        
        Args:
            name (str): Name of the strategy
            strategy_func (Callable): Strategy function that returns signals
            **kwargs: Additional arguments for the strategy
            
        Returns:
            Backtester: Self for method chaining
        """
        self.strategies[name] = {
            'function': strategy_func,
            'parameters': kwargs
        }
        logger.info(f"Strategy '{name}' added successfully")
        return self
    
    def run_backtest(self, strategy_name: str, **kwargs) -> Dict[str, Any]:
        """
        Run backtest for a specific strategy.
        
        Args:
            strategy_name (str): Name of the strategy to backtest
            **kwargs: Additional backtest parameters
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        logger.info(f"Running backtest for strategy: {strategy_name}")
        
        # Reset backtester state
        self._reset_state()
        
        # Get strategy
        strategy = self.strategies[strategy_name]
        strategy_func = strategy['function']
        strategy_params = strategy['parameters']
        
        # Generate signals
        signals = strategy_func(self.data, **strategy_params)
        
        # Execute trades based on signals
        self._execute_strategy(signals, **kwargs)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # Store results
        self.results[strategy_name] = {
            'signals': signals,
            'portfolio_values': self.portfolio_values.copy(),
            'trades': self.trades.copy(),
            'performance_metrics': performance_metrics
        }
        
        logger.info(f"Backtest completed for strategy: {strategy_name}")
        return self.results[strategy_name]
    
    def _reset_state(self) -> None:
        """Reset backtester state for new backtest."""
        self.capital = self.initial_capital
        self.positions = {}
        self.portfolio_values = []
        self.trades = []
    
    def _execute_strategy(self, signals: pd.DataFrame, 
                         transaction_costs: float = 0.001,
                         slippage: float = 0.0005) -> None:
        """
        Execute trading strategy based on signals.
        
        Args:
            signals (pd.DataFrame): Trading signals
            transaction_costs (float): Transaction costs as percentage
            slippage (float): Slippage as percentage
        """
        for i, (timestamp, row) in enumerate(signals.iterrows()):
            current_portfolio_value = self._calculate_portfolio_value(timestamp)
            self.portfolio_values.append({
                'timestamp': timestamp,
                'value': current_portfolio_value,
                'cash': self.capital,
                'positions': self.positions.copy()
            })
            
            # Execute trades based on signals
            for asset in signals.columns:
                if asset in row and pd.notna(row[asset]):
                    signal = row[asset]
                    
                    if signal > 0:  # Buy signal
                        self._execute_buy(timestamp, asset, signal, transaction_costs, slippage)
                    elif signal < 0:  # Sell signal
                        self._execute_sell(timestamp, asset, abs(signal), transaction_costs, slippage)
    
    def _execute_buy(self, timestamp: pd.Timestamp, asset: str, 
                     signal_strength: float, transaction_costs: float, slippage: float) -> None:
        """
        Execute buy order.
        
        Args:
            timestamp (pd.Timestamp): Trade timestamp
            asset (str): Asset to buy
            signal_strength (float): Signal strength (0-1)
            transaction_costs (float): Transaction costs
            slippage (float): Slippage
        """
        if asset not in self.data.columns:
            return
        
        # Get current price
        current_price = self.data.loc[timestamp, asset]
        if pd.isna(current_price):
            return
        
        # Apply slippage
        execution_price = current_price * (1 + slippage)
        
        # Calculate position size based on signal strength and available capital
        position_value = self.capital * signal_strength * 0.95  # Use 95% of available capital
        shares = position_value / execution_price
        
        # Calculate total cost including transaction costs
        total_cost = position_value * (1 + transaction_costs)
        
        if total_cost <= self.capital:
            # Execute trade
            if asset in self.positions:
                self.positions[asset] += shares
            else:
                self.positions[asset] = shares
            
            self.capital -= total_cost
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'asset': asset,
                'action': 'BUY',
                'shares': shares,
                'price': execution_price,
                'value': position_value,
                'costs': position_value * transaction_costs
            })
    
    def _execute_sell(self, timestamp: pd.Timestamp, asset: str, 
                      signal_strength: float, transaction_costs: float, slippage: float) -> None:
        """
        Execute sell order.
        
        Args:
            timestamp (pd.Timestamp): Trade timestamp
            asset (str): Asset to sell
            signal_strength (float): Signal strength (0-1)
            transaction_costs (float): Transaction costs
            slippage (float): Slippage
        """
        if asset not in self.data.columns or asset not in self.positions:
            return
        
        # Get current price
        current_price = self.data.loc[timestamp, asset]
        if pd.isna(current_price):
            return
        
        # Apply slippage
        execution_price = current_price * (1 - slippage)
        
        # Calculate shares to sell based on signal strength
        current_shares = self.positions[asset]
        shares_to_sell = current_shares * signal_strength
        
        if shares_to_sell > 0:
            # Execute trade
            self.positions[asset] -= shares_to_sell
            
            # Calculate proceeds
            proceeds = shares_to_sell * execution_price
            net_proceeds = proceeds * (1 - transaction_costs)
            
            self.capital += net_proceeds
            
            # Record trade
            self.trades.append({
                'timestamp': timestamp,
                'asset': asset,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': execution_price,
                'value': proceeds,
                'costs': proceeds * transaction_costs
            })
    
    def _calculate_portfolio_value(self, timestamp: pd.Timestamp) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            
        Returns:
            float: Portfolio value
        """
        portfolio_value = self.capital
        
        for asset, shares in self.positions.items():
            if asset in self.data.columns:
                current_price = self.data.loc[timestamp, asset]
                if pd.notna(current_price):
                    portfolio_value += shares * current_price
        
        return portfolio_value
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        if not self.portfolio_values:
            return {}
        
        # Extract portfolio values
        portfolio_df = pd.DataFrame(self.portfolio_values)
        portfolio_df = portfolio_df.set_index('timestamp')
        
        # Calculate returns
        portfolio_returns = portfolio_df['value'].pct_change().dropna()
        
        # Basic metrics
        total_return = (portfolio_df['value'].iloc[-1] / portfolio_df['value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_df)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_df['value'])
        var_95 = portfolio_returns.quantile(0.05)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade analysis
        if self.trades:
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['action'] == 'SELL' and t['value'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
        else:
            total_trades = 0
            win_rate = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'total_trades': total_trades,
            'win_rate': win_rate
        }
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            portfolio_values (pd.Series): Portfolio values over time
            
        Returns:
            float: Maximum drawdown
        """
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
    
    def run_multiple_strategies(self, **kwargs) -> Dict[str, Any]:
        """
        Run backtest for all strategies.
        
        Args:
            **kwargs: Additional backtest parameters
            
        Returns:
            Dict[str, Any]: Results for all strategies
        """
        logger.info(f"Running backtests for {len(self.strategies)} strategies")
        
        for strategy_name in self.strategies.keys():
            try:
                self.run_backtest(strategy_name, **kwargs)
            except Exception as e:
                logger.error(f"Error running backtest for {strategy_name}: {str(e)}")
                continue
        
        return self.results
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare performance of all strategies.
        
        Returns:
            pd.DataFrame: Strategy comparison table
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        comparison_data = []
        for strategy_name, result in self.results.items():
            row = {'Strategy': strategy_name}
            row.update(result['performance_metrics'])
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by Sharpe ratio (higher is better)
        if 'sharpe_ratio' in comparison_df.columns:
            comparison_df = comparison_df.sort_values('sharpe_ratio', ascending=False)
        
        logger.info("Strategy comparison completed")
        return comparison_df
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing strategy based on a specific metric.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            Tuple[str, Dict[str, float]]: Best strategy name and its metrics
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        # Find best strategy
        best_strategy = None
        best_value = float('-inf') if metric in ['sharpe_ratio', 'total_return', 'annualized_return'] else float('inf')
        
        for strategy_name, result in self.results.items():
            if metric in result['performance_metrics']:
                current_value = result['performance_metrics'][metric]
                
                if metric in ['sharpe_ratio', 'total_return', 'annualized_return']:
                    if current_value > best_value:
                        best_value = current_value
                        best_strategy = strategy_name
                else:
                    if current_value < best_value:
                        best_value = current_value
                        best_strategy = strategy_name
        
        if best_strategy is None:
            raise ValueError(f"No strategy found with metric {metric}")
        
        logger.info(f"Best strategy by {metric}: {best_strategy} ({best_value:.4f})")
        return best_strategy, self.results[best_strategy]['performance_metrics']
    
    def plot_portfolio_values(self, save_path: Optional[str] = None) -> None:
        """
        Plot portfolio values for all strategies.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 8))
            
            # Plot portfolio values for each strategy
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))
            for i, (strategy_name, result) in enumerate(self.results.items()):
                portfolio_df = pd.DataFrame(result['portfolio_values'])
                portfolio_df = portfolio_df.set_index('timestamp')
                
                plt.plot(portfolio_df.index, portfolio_df['value'], 
                        label=strategy_name, linewidth=2, color=colors[i])
            
            # Plot benchmark (buy and hold)
            if len(self.results) > 0:
                first_result = list(self.results.values())[0]
                portfolio_df = pd.DataFrame(first_result['portfolio_values'])
                portfolio_df = portfolio_df.set_index('timestamp')
                
                # Calculate buy and hold performance
                first_price = portfolio_df['value'].iloc[0]
                buy_hold_values = first_price * (1 + 0)  # No change for simplicity
                plt.axhline(y=buy_hold_values, color='black', linestyle='--', 
                           label='Buy & Hold', alpha=0.7)
            
            plt.title('Portfolio Values Comparison')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Portfolio values plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create portfolio values plot: {str(e)}")
    
    def plot_drawdowns(self, save_path: Optional[str] = None) -> None:
        """
        Plot drawdowns for all strategies.
        
        Args:
            save_path (str, optional): Path to save plot
        """
        if not self.results:
            raise ValueError("No backtest results available. Run backtests first.")
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 8))
            
            # Plot drawdowns for each strategy
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))
            for i, (strategy_name, result) in enumerate(self.results.items()):
                portfolio_df = pd.DataFrame(result['portfolio_values'])
                portfolio_df = portfolio_df.set_index('timestamp')
                
                # Calculate drawdown
                peak = portfolio_df['value'].expanding().max()
                drawdown = (portfolio_df['value'] - peak) / peak
                
                plt.plot(drawdown.index, drawdown.values, 
                        label=strategy_name, linewidth=2, color=colors[i])
            
            plt.title('Portfolio Drawdowns')
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Drawdowns plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create drawdowns plot: {str(e)}")
    
    def generate_backtest_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Backtest report
        """
        if not self.results:
            return "No backtest results available. Run backtests first."
        
        report = []
        report.append("# Backtesting Report")
        report.append("=" * 50)
        report.append("")
        
        # Backtest information
        report.append("## Backtest Information")
        report.append(f"- **Initial capital**: ${self.initial_capital:,.2f}")
        report.append(f"- **Number of strategies**: {len(self.strategies)}")
        report.append(f"- **Data period**: {self.data.index[0]} to {self.data.index[-1]}")
        report.append("")
        
        # Strategy comparison
        report.append("## Strategy Performance Comparison")
        comparison_df = self.compare_strategies()
        report.append(comparison_df.to_string(index=False))
        report.append("")
        
        # Best strategy
        try:
            best_strategy, best_metrics = self.get_best_strategy('sharpe_ratio')
            report.append(f"**Best Strategy by Sharpe Ratio**: {best_strategy}")
            report.append(f"- Sharpe Ratio: {best_metrics['sharpe_ratio']:.4f}")
            report.append(f"- Annualized Return: {best_metrics['annualized_return']:.2%}")
            report.append(f"- Volatility: {best_metrics['volatility']:.2%}")
            report.append(f"- Max Drawdown: {best_metrics['max_drawdown']:.2%}")
            report.append("")
        except:
            pass
        
        # Individual strategy details
        report.append("## Individual Strategy Details")
        for strategy_name, result in self.results.items():
            report.append(f"### {strategy_name}")
            metrics = result['performance_metrics']
            
            report.append(f"- **Total Return**: {metrics['total_return']:.2%}")
            report.append(f"- **Annualized Return**: {metrics['annualized_return']:.2%}")
            report.append(f"- **Volatility**: {metrics['volatility']:.2%}")
            report.append(f"- **Sharpe Ratio**: {metrics['sharpe_ratio']:.4f}")
            report.append(f"- **Max Drawdown**: {metrics['max_drawdown']:.2%}")
            report.append(f"- **Total Trades**: {metrics['total_trades']}")
            report.append(f"- **Win Rate**: {metrics['win_rate']:.2%}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        if len(self.results) > 1:
            report.append("- Compare strategies across different market conditions")
            report.append("- Consider transaction costs and slippage in real trading")
            report.append("- Monitor strategy performance and adjust parameters")
        
        report.append("- Diversify across multiple strategies")
        report.append("- Implement proper risk management")
        report.append("- Regularly review and update strategies")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Backtest report saved to {save_path}")
        
        return report_text
    
    def save_results(self, filepath: str) -> None:
        """
        Save backtest results to a file.
        
        Args:
            filepath (str): Path to save the results
        """
        try:
            import pickle
            
            results_data = {
                'results': self.results,
                'strategies': self.strategies,
                'initial_capital': self.initial_capital,
                'data_info': {
                    'start_date': self.data.index[0],
                    'end_date': self.data.index[-1],
                    'data_length': len(self.data)
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results_data, f)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise


# Predefined strategy functions
def buy_and_hold_strategy(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Buy and hold strategy.
    
    Args:
        data (pd.DataFrame): Market data
        **kwargs: Additional parameters
        
    Returns:
        pd.DataFrame: Trading signals
    """
    signals = pd.DataFrame(index=data.index, columns=data.columns)
    signals.iloc[0] = 1  # Buy at the beginning
    signals = signals.fillna(0)
    return signals


def momentum_strategy(data: pd.DataFrame, lookback_period: int = 20, 
                     threshold: float = 0.0, **kwargs) -> pd.DataFrame:
    """
    Momentum-based strategy.
    
    Args:
        data (pd.DataFrame): Market data
        lookback_period (int): Period for momentum calculation
        threshold (float): Threshold for signal generation
        **kwargs: Additional parameters
        
    Returns:
        pd.DataFrame: Trading signals
    """
    signals = pd.DataFrame(index=data.index, columns=data.columns)
    
    for asset in data.columns:
        # Calculate momentum (rolling returns)
        momentum = data[asset].pct_change(lookback_period)
        
        # Generate signals
        signals[asset] = np.where(momentum > threshold, 1, 0)
    
    return signals


def mean_reversion_strategy(data: pd.DataFrame, lookback_period: int = 20,
                           std_dev: float = 2.0, **kwargs) -> pd.DataFrame:
    """
    Mean reversion strategy.
    
    Args:
        data (pd.DataFrame): Market data
        lookback_period (int): Period for mean calculation
        std_dev (float): Standard deviation threshold
        **kwargs: Additional parameters
        
    Returns:
        pd.DataFrame: Trading signals
    """
    signals = pd.DataFrame(index=data.index, columns=data.columns)
    
    for asset in data.columns:
        # Calculate rolling mean and standard deviation
        rolling_mean = data[asset].rolling(lookback_period).mean()
        rolling_std = data[asset].rolling(lookback_period).std()
        
        # Calculate z-score
        z_score = (data[asset] - rolling_mean) / rolling_std
        
        # Generate signals
        signals[asset] = np.where(z_score > std_dev, -1, 0)  # Sell when overvalued
        signals[asset] = np.where(z_score < -std_dev, 1, signals[asset])  # Buy when undervalued
    
    return signals


def run_complete_backtesting(data: pd.DataFrame, 
                            initial_capital: float = 100000,
                            output_dir: str = "data/processed/") -> Backtester:
    """
    Run complete backtesting pipeline with predefined strategies.
    
    Args:
        data (pd.DataFrame): Market data
        initial_capital (float): Initial capital
        output_dir (str): Directory to save outputs
        
    Returns:
        Backtester: Configured backtester with results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize backtester
    backtester = Backtester(data, initial_capital)
    
    # Add predefined strategies
    backtester.add_strategy('Buy_andHold', buy_and_hold_strategy)
    backtester.add_strategy('Momentum', momentum_strategy, lookback_period=20)
    backtester.add_strategy('MeanReversion', mean_reversion_strategy, lookback_period=20, std_dev=2.0)
    
    # Run backtests
    logger.info("Running backtests for all strategies...")
    backtester.run_multiple_strategies(transaction_costs=0.001, slippage=0.0005)
    
    # Generate plots
    try:
        backtester.plot_portfolio_values(os.path.join(output_dir, 'portfolio_values.png'))
        backtester.plot_drawdowns(os.path.join(output_dir, 'drawdowns.png'))
    except Exception as e:
        logger.warning(f"Could not create plots: {str(e)}")
    
    # Generate report
    try:
        report_path = os.path.join(output_dir, 'backtest_report.md')
        backtester.generate_backtest_report(report_path)
    except Exception as e:
        logger.warning(f"Could not generate backtest report: {str(e)}")
    
    # Save results
    try:
        backtester.save_results(os.path.join(output_dir, 'backtest_results.pkl'))
    except Exception as e:
        logger.warning(f"Could not save results: {str(e)}")
    
    logger.info("Complete backtesting pipeline finished")
    return backtester


if __name__ == "__main__":
    # Example usage
    from data_loader import load_sample_data
    
    # Load sample data and create market data
    sample_data = load_sample_data()
    
    # Create sample market data for multiple assets
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Generate sample OHLCV data for 4 assets
    market_data = pd.DataFrame({
        'Asset_A': np.random.randn(252).cumsum() + 100,
        'Asset_B': np.random.randn(252).cumsum() + 95,
        'Asset_C': np.random.randn(252).cumsum() + 110,
        'Asset_D': np.random.randn(252).cumsum() + 105
    }, index=dates)
    
    print("Sample market data created, running complete backtesting...")
    
    # Run complete backtesting
    backtester = run_complete_backtesting(market_data, initial_capital=100000)
    
    print("Backtesting completed successfully!")
    print(f"Strategies tested: {list(backtester.strategies.keys())}")
    
    # Compare strategies
    comparison = backtester.compare_strategies()
    print("\nStrategy Comparison:")
    print(comparison[['Strategy', 'sharpe_ratio', 'annualized_return', 'max_drawdown']].to_string(index=False))
