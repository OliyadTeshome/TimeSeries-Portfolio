"""
Interactive Streamlit Dashboard for TimeSeries-Portfolio

This module provides a professional financial dashboard with real-time data visualization,
portfolio analytics, and interactive tools for financial analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Optional, Tuple, Any
import logging

# Use absolute imports for direct script execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.portfolio_optimization import PortfolioOptimizer
from src.backtesting import Backtester
from src.forecasting import ForecastingEngine
from src.validation import DataValidator
from src.logging_config import setup_logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logger = setup_logging()
logger = logging.getLogger(__name__)

class FinancialDashboard:
    """Professional financial dashboard with interactive visualizations."""
    
    def __init__(self):
        """Initialize the dashboard with configuration and components."""
        self.config = get_config()
        # Defer heavy component initialization until data is available
        self.portfolio_optimizer = None
        self.backtester = None
        self.forecaster = None
        self.validator = DataValidator()
        
        # Initialize session state
        if 'portfolio_data' not in st.session_state:
            st.session_state.portfolio_data = {}
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = {}
        if 'backtest_results' not in st.session_state:
            st.session_state.backtest_results = {}
        
        # Setup page configuration
        st.set_page_config(
            page_title="TimeSeries Portfolio Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the main dashboard application."""
        try:
            # Header
            self._render_header()
            
            # Sidebar
            self._render_sidebar()
            
            # Main content
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Portfolio Overview", 
                "🔮 Forecasting & Analysis",
                "⚖️ Portfolio Optimization",
                "📈 Backtesting & Performance",
                "⚠️ Risk Management"
            ])
            
            with tab1:
                self._render_portfolio_overview()
            
            with tab2:
                self._render_forecasting_analysis()
            
            with tab3:
                self._render_portfolio_optimization()
            
            with tab4:
                self._render_backtesting_performance()
            
            with tab5:
                self._render_risk_management()
                
        except Exception as e:
            logger.error(f"Dashboard error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
    
    def _render_header(self):
        """Render the dashboard header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("📈 TimeSeries Portfolio Dashboard")
            st.markdown("**Professional Financial Analysis & Portfolio Management**")
            st.markdown("---")
        
        # Market status indicator
        with col3:
            market_status = self._get_market_status()
            if market_status['is_open']:
                st.success(f"🟢 Market Open - {market_status['time']}")
            else:
                st.error(f"🔴 Market Closed - {market_status['time']}")
    
    def _render_sidebar(self):
        """Render the sidebar with configuration options."""
        st.sidebar.header("⚙️ Configuration")
        
        # Asset selection
        st.sidebar.subheader("📈 Assets")
        default_assets = ['SPY', 'TSLA', 'BND', 'AAPL', 'MSFT', 'GOOGL']
        selected_assets = st.sidebar.multiselect(
            "Select Assets:",
            default_assets,
            default=default_assets[:3]
        )
        
        # Date range
        st.sidebar.subheader("📅 Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_range = st.sidebar.date_input(
            "Select Date Range:",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        
        # Analysis parameters
        st.sidebar.subheader("🔧 Analysis Parameters")
        risk_free_rate = st.sidebar.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.5,
            step=0.1
        )
        
        confidence_level = st.sidebar.slider(
            "VaR Confidence Level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1
        )
        
        # Update session state
        st.session_state['selected_assets'] = selected_assets
        st.session_state['date_range'] = (start_date, end_date)
        st.session_state['risk_free_rate'] = risk_free_rate / 100
        st.session_state['confidence_level'] = confidence_level / 100
        
        # Load data button
        if st.sidebar.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading market data..."):
                self._load_market_data(selected_assets, start_date, end_date)
                st.success("Data loaded successfully!")
    
    def _render_portfolio_overview(self):
        """Render the portfolio overview tab."""
        st.header("📊 Portfolio Overview")
        
        if not st.session_state.get('portfolio_data'):
            st.info("Please load data from the sidebar to view portfolio overview.")
            return
        
        # Portfolio summary metrics
        self._render_portfolio_metrics()
        
        # Price charts
        self._render_price_charts()
        
        # Correlation matrix
        self._render_correlation_matrix()
        
        # Returns distribution
        self._render_returns_distribution()
    
    def _render_forecasting_analysis(self):
        """Render the forecasting and analysis tab."""
        st.header("🔮 Forecasting & Analysis")
        
        if not st.session_state.get('portfolio_data'):
            st.info("Please load data from the sidebar to perform forecasting analysis.")
            return
        
        # Forecasting parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_horizon = st.selectbox(
                "Forecast Horizon:",
                [30, 60, 90, 180, 365],
                index=2
            )
        
        with col2:
            forecast_model = st.selectbox(
                "Forecast Model:",
                ["Ensemble", "ARIMA", "LSTM", "Prophet"],
                index=0
            )
        
        with col3:
            if st.button("🚀 Generate Forecasts", type="primary"):
                with st.spinner("Generating forecasts..."):
                    self._generate_forecasts(forecast_horizon, forecast_model)
        
        # Display forecast results
        if st.session_state.get('forecast_results'):
            self._render_forecast_results()
    
    def _render_portfolio_optimization(self):
        """Render the portfolio optimization tab."""
        st.header("⚖️ Portfolio Optimization")
        
        if not st.session_state.get('portfolio_data'):
            st.info("Please load data from the sidebar to perform portfolio optimization.")
            return
        
        # Optimization parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method:",
                ["Sharpe Ratio", "Minimum Variance", "Maximum Return", "Risk Parity"],
                index=0
            )
        
        with col2:
            target_return = st.number_input(
                "Target Return (%)",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5
            )
        
        with col3:
            if st.button("⚖️ Optimize Portfolio", type="primary"):
                with st.spinner("Optimizing portfolio..."):
                    self._optimize_portfolio(optimization_method, target_return / 100)
        
        # Display optimization results
        if st.session_state.get('optimization_results'):
            self._render_optimization_results()
    
    def _render_backtesting_performance(self):
        """Render the backtesting and performance tab."""
        st.header("📈 Backtesting & Performance")
        
        if not st.session_state.get('portfolio_data'):
            st.info("Please load data from the sidebar to perform backtesting.")
            return
        
        # Backtesting parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Strategy:",
                ["Buy & Hold", "Momentum", "Mean Reversion", "Custom"],
                index=0
            )
        
        with col2:
            rebalance_frequency = st.selectbox(
                "Rebalance Frequency:",
                ["Daily", "Weekly", "Monthly", "Quarterly"],
                index=2
            )
        
        with col3:
            if st.button("📊 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    self._run_backtest(strategy, rebalance_frequency)
        
        # Display backtest results
        if st.session_state.get('backtest_results'):
            self._render_backtest_results()
    
    def _render_risk_management(self):
        """Render the risk management tab."""
        st.header("⚠️ Risk Management")
        
        if not st.session_state.get('portfolio_data'):
            st.info("Please load data from the sidebar to perform risk analysis.")
            return
        
        # Risk metrics
        self._render_risk_metrics()
        
        # Stress testing
        self._render_stress_testing()
        
        # Scenario analysis
        self._render_scenario_analysis()
    
    def _get_market_status(self) -> Dict[str, Any]:
        """Get current market status."""
        try:
            # Simple market hours check (9:30 AM - 4:00 PM ET, weekdays)
            now = datetime.now()
            is_weekday = now.weekday() < 5
            is_market_hours = 9.5 <= now.hour + now.minute / 60 <= 16
            
            return {
                'is_open': is_weekday and is_market_hours,
                'time': now.strftime("%H:%M:%S")
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {'is_open': False, 'time': 'Unknown'}
    
    def _load_market_data(self, assets: List[str], start_date: datetime, end_date: datetime):
        """Load market data for selected assets."""
        try:
            data = {}
            progress_bar = st.progress(0)
            
            for i, asset in enumerate(assets):
                progress_bar.progress((i + 1) / len(assets))
                
                # Load data using yfinance
                ticker = yf.Ticker(asset)
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate returns
                    hist['Returns'] = hist['Close'].pct_change()
                    hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
                    
                    data[asset] = hist
                else:
                    st.warning(f"No data available for {asset}")
            
            progress_bar.empty()
            st.session_state['portfolio_data'] = data
            
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            st.error(f"Error loading market data: {str(e)}")
    
    def _render_portfolio_metrics(self):
        """Render portfolio summary metrics."""
        data = st.session_state['portfolio_data']
        
        # Calculate metrics
        metrics = {}
        for asset, hist in data.items():
            returns = hist['Returns'].dropna()
            metrics[asset] = {
                'Total Return': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100,
                'Annualized Return': returns.mean() * 252 * 100,
                'Volatility': returns.std() * np.sqrt(252) * 100,
                'Sharpe Ratio': (returns.mean() * 252 - st.session_state['risk_free_rate']) / (returns.std() * np.sqrt(252)),
                'Max Drawdown': self._calculate_max_drawdown(hist['Close']) * 100
            }
        
        # Display metrics in columns
        cols = st.columns(len(metrics))
        for i, (asset, metric) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(
                    label=asset,
                    value=f"{metric['Total Return']:.2f}%",
                    delta=f"{metric['Annualized Return']:.2f}%"
                )
                st.caption(f"Vol: {metric['Volatility']:.2f}% | Sharpe: {metric['Sharpe Ratio']:.2f}")
    
    def _render_price_charts(self):
        """Render interactive price charts."""
        data = st.session_state['portfolio_data']
        
        # Create subplot for price charts
        fig = make_subplots(
            rows=len(data), cols=1,
            subplot_titles=list(data.keys()),
            vertical_spacing=0.05
        )
        
        for i, (asset, hist) in enumerate(data.items()):
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    name=asset,
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(data),
            showlegend=False,
            title="Asset Price Performance"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_correlation_matrix(self):
        """Render correlation matrix heatmap."""
        data = st.session_state['portfolio_data']
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame({
            asset: hist['Returns'].dropna() 
            for asset, hist in data.items()
        })
        
        correlation_matrix = returns_df.corr()
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Asset Returns Correlation Matrix"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_returns_distribution(self):
        """Render returns distribution charts."""
        data = st.session_state['portfolio_data']
        
        # Create subplots for returns distribution
        fig = make_subplots(
            rows=1, cols=len(data),
            subplot_titles=[f"{asset} Returns Distribution" for asset in data.keys()]
        )
        
        for i, (asset, hist) in enumerate(data.items()):
            returns = hist['Returns'].dropna()
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name=asset,
                    nbinsx=30,
                    opacity=0.7
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            height=400,
            title="Asset Returns Distribution",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_forecasts(self, horizon: int, model: str):
        """Generate forecasts for selected assets."""
        try:
            data = st.session_state['portfolio_data']
            forecast_results = {}
            
            for asset, hist in data.items():
                # Prepare data for forecasting
                prices = hist['Close'].values
                
                # Simple forecasting (placeholder for actual model integration)
                if model == "Ensemble":
                    # Simple trend-based forecast
                    trend = np.polyfit(range(len(prices)), prices, 1)[0]
                    forecast = prices[-1] + trend * np.arange(1, horizon + 1)
                else:
                    # Random walk forecast
                    returns = hist['Returns'].dropna().values
                    forecast = prices[-1] * np.cumprod(1 + np.random.choice(returns, horizon))
                
                forecast_results[asset] = {
                    'forecast': forecast,
                    'confidence_interval': self._calculate_confidence_interval(forecast, hist['Returns'].std())
                }
            
            st.session_state['forecast_results'] = forecast_results
            
        except Exception as e:
            logger.error(f"Error generating forecasts: {e}")
            st.error(f"Error generating forecasts: {str(e)}")
    
    def _render_forecast_results(self):
        """Render forecast results."""
        forecast_results = st.session_state['forecast_results']
        data = st.session_state['portfolio_data']
        
        # Create forecast charts
        fig = make_subplots(
            rows=len(forecast_results), cols=1,
            subplot_titles=[f"{asset} Forecast" for asset in forecast_results.keys()],
            vertical_spacing=0.05
        )
        
        for i, (asset, result) in enumerate(forecast_results.items()):
            hist = data[asset]
            forecast_dates = pd.date_range(
                start=hist.index[-1] + pd.Timedelta(days=1),
                periods=len(result['forecast']),
                freq='D'
            )
            
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    name=f"{asset} (Historical)",
                    line=dict(color='blue', width=2)
                ),
                row=i+1, col=1
            )
            
            # Forecast
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=result['forecast'],
                    name=f"{asset} (Forecast)",
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=300 * len(forecast_results),
            title="Asset Price Forecasts",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _optimize_portfolio(self, method: str, target_return: float):
        """Optimize portfolio using selected method."""
        try:
            data = st.session_state['portfolio_data']
            
            # Prepare returns data
            returns_df = pd.DataFrame({
                asset: hist['Returns'].dropna() 
                for asset, hist in data.items()
            })
            
            # Calculate expected returns and covariance
            expected_returns = returns_df.mean() * 252
            covariance_matrix = returns_df.cov() * 252
            
            # Portfolio optimization (placeholder - integrate with actual optimizer)
            if method == "Sharpe Ratio":
                weights = self._optimize_sharpe_ratio(expected_returns, covariance_matrix)
            elif method == "Minimum Variance":
                weights = self._optimize_minimum_variance(covariance_matrix)
            else:
                weights = np.ones(len(data)) / len(data)  # Equal weight
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(weights.T @ covariance_matrix @ weights)
            sharpe_ratio = (portfolio_return - st.session_state['risk_free_rate']) / portfolio_volatility
            
            st.session_state['optimization_results'] = {
                'weights': dict(zip(data.keys(), weights)),
                'metrics': {
                    'Expected Return': portfolio_return,
                    'Volatility': portfolio_volatility,
                    'Sharpe Ratio': sharpe_ratio
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            st.error(f"Error optimizing portfolio: {str(e)}")
    
    def _render_optimization_results(self):
        """Render portfolio optimization results."""
        results = st.session_state['optimization_results']
        
        # Display weights
        st.subheader("Optimal Portfolio Weights")
        weights_df = pd.DataFrame(
            list(results['weights'].items()),
            columns=['Asset', 'Weight']
        )
        weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
        st.dataframe(weights_df, use_container_width=True)
        
        # Display metrics
        st.subheader("Portfolio Metrics")
        metrics_df = pd.DataFrame(
            list(results['metrics'].items()),
            columns=['Metric', 'Value']
        )
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Weight pie chart
        fig = px.pie(
            values=list(results['weights'].values()),
            names=list(results['weights'].keys()),
            title="Optimal Portfolio Allocation"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _run_backtest(self, strategy: str, rebalance_freq: str):
        """Run backtest with selected strategy."""
        try:
            data = st.session_state['portfolio_data']
            
            # Simple backtesting implementation (placeholder)
            if strategy == "Buy & Hold":
                returns = self._buy_and_hold_strategy(data)
            elif strategy == "Momentum":
                returns = self._momentum_strategy(data, rebalance_freq)
            else:
                returns = self._equal_weight_strategy(data, rebalance_freq)
            
            # Calculate performance metrics
            total_return = (returns.iloc[-1] - 1) * 100
            annualized_return = (returns.iloc[-1] ** (252 / len(returns)) - 1) * 100
            volatility = returns.pct_change().std() * np.sqrt(252) * 100
            sharpe_ratio = (returns.pct_change().mean() * 252 - st.session_state['risk_free_rate']) / (returns.pct_change().std() * np.sqrt(252))
            max_drawdown = self._calculate_max_drawdown(returns) * 100
            
            st.session_state['backtest_results'] = {
                'returns': returns,
                'metrics': {
                    'Total Return': total_return,
                    'Annualized Return': annualized_return,
                    'Volatility': volatility,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown': max_drawdown
                }
            }
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            st.error(f"Error running backtest: {str(e)}")
    
    def _render_backtest_results(self):
        """Render backtesting results."""
        results = st.session_state['backtest_results']
        
        # Display metrics
        st.subheader("Backtest Performance Metrics")
        metrics_df = pd.DataFrame(
            list(results['metrics'].items()),
            columns=['Metric', 'Value']
        )
        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Performance chart
        fig = px.line(
            x=results['returns'].index,
            y=results['returns'],
            title="Cumulative Returns",
            labels={'x': 'Date', 'y': 'Cumulative Return'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_metrics(self):
        """Render risk management metrics."""
        data = st.session_state['portfolio_data']
        
        # Calculate VaR and CVaR
        risk_metrics = {}
        for asset, hist in data.items():
            returns = hist['Returns'].dropna()
            confidence_level = st.session_state['confidence_level']
            
            var = np.percentile(returns, (1 - confidence_level) * 100)
            cvar = returns[returns <= var].mean()
            
            risk_metrics[asset] = {
                'VaR': var * 100,
                'CVaR': cvar * 100,
                'Volatility': returns.std() * np.sqrt(252) * 100
            }
        
        # Display risk metrics
        st.subheader("Risk Metrics")
        risk_df = pd.DataFrame(risk_metrics).T
        st.dataframe(risk_df, use_container_width=True)
    
    def _render_stress_testing(self):
        """Render stress testing results."""
        st.subheader("Stress Testing")
        
        # Market crash scenario
        st.write("**Market Crash Scenario (-20% across all assets)**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Stress Test"):
                self._run_stress_test()
        
        with col2:
            if st.session_state.get('stress_test_results'):
                results = st.session_state['stress_test_results']
                st.metric("Portfolio Impact", f"{results['portfolio_impact']:.2f}%")
    
    def _render_scenario_analysis(self):
        """Render scenario analysis."""
        st.subheader("Scenario Analysis")
        
        # Define scenarios
        scenarios = {
            'Bull Market': 1.2,
            'Bear Market': 0.8,
            'High Volatility': 1.5,
            'Low Volatility': 0.7
        }
        
        if st.button("Run Scenario Analysis"):
            self._run_scenario_analysis(scenarios)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_confidence_interval(self, forecast: np.ndarray, std: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence intervals for forecasts."""
        z_score = 1.96  # 95% confidence
        margin = z_score * std * np.sqrt(np.arange(1, len(forecast) + 1))
        return forecast - margin, forecast + margin
    
    def _optimize_sharpe_ratio(self, expected_returns: pd.Series, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio for maximum Sharpe ratio."""
        # Simple optimization (placeholder)
        n_assets = len(expected_returns)
        weights = np.ones(n_assets) / n_assets
        return weights
    
    def _optimize_minimum_variance(self, covariance_matrix: pd.DataFrame) -> np.ndarray:
        """Optimize portfolio for minimum variance."""
        # Simple optimization (placeholder)
        n_assets = len(covariance_matrix)
        weights = np.ones(n_assets) / n_assets
        return weights
    
    def _buy_and_hold_strategy(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """Buy and hold strategy implementation."""
        # Equal weight buy and hold
        returns = pd.Series(1.0, index=list(data.values())[0].index)
        for hist in data.values():
            returns *= (1 + hist['Returns'].fillna(0))
        return returns
    
    def _momentum_strategy(self, data: Dict[str, pd.DataFrame], rebalance_freq: str) -> pd.Series:
        """Momentum strategy implementation."""
        # Simple momentum strategy (placeholder)
        return self._buy_and_hold_strategy(data)
    
    def _equal_weight_strategy(self, data: Dict[str, pd.DataFrame], rebalance_freq: str) -> pd.Series:
        """Equal weight strategy implementation."""
        # Equal weight rebalancing (placeholder)
        return self._buy_and_hold_strategy(data)
    
    def _run_stress_test(self):
        """Run stress test scenario."""
        # Simple stress test (placeholder)
        st.session_state['stress_test_results'] = {
            'portfolio_impact': -15.5
        }
    
    def _run_scenario_analysis(self, scenarios: Dict[str, float]):
        """Run scenario analysis."""
        # Simple scenario analysis (placeholder)
        st.write("Scenario analysis completed (placeholder implementation)")


def main():
    """Main function to run the dashboard."""
    dashboard = FinancialDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
