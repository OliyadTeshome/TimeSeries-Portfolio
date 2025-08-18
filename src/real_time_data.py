"""
Real-Time Data Integration Module

This module provides real-time market data feeds, portfolio monitoring,
alerts, and live data streaming capabilities for financial applications.
"""

import asyncio
import websockets
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import warnings
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field
from queue import Queue, Empty
import schedule

from .config import get_config
from .logging_config import setup_logging
from .exceptions import DataError as RealTimeDataError

# Setup logging
logger = setup_logging()
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class MarketData:
    """Market data structure for real-time information."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None


@dataclass
class Alert:
    """Alert structure for market conditions."""
    id: str
    symbol: str
    condition: str
    threshold: float
    current_value: float
    timestamp: datetime
    triggered: bool = False
    message: str = ""
    priority: str = "medium"  # low, medium, high, critical


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot for real-time monitoring."""
    timestamp: datetime
    total_value: float
    total_return: float
    total_return_percent: float
    daily_pnl: float
    daily_pnl_percent: float
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)


class RealTimeDataManager:
    """
    Real-time data management system for financial applications.
    
    This class provides live market data feeds, portfolio monitoring,
    and alert systems for real-time financial analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time data manager.
        
        Args:
            config: Configuration dictionary for data sources and parameters
        """
        self.config = config or get_config()
        self.data_sources = {}
        self.data_streams = {}
        self.portfolio_monitor = None
        self.alert_system = None
        self.is_running = False
        
        # Data storage
        self.market_data_cache = {}
        self.portfolio_cache = {}
        self.alert_cache = {}
        
        # Callback functions
        self.data_callbacks = []
        self.alert_callbacks = []
        self.portfolio_callbacks = []
        
        # Initialize components
        self._initialize_data_sources()
        self._initialize_alert_system()
        self._initialize_portfolio_monitor()
    
    def _initialize_data_sources(self):
        """Initialize data sources for real-time feeds."""
        try:
            # Yahoo Finance as primary source
            self.data_sources['yahoo'] = YahooFinanceSource()
            
            # Alpha Vantage as secondary source (if API key available)
            if self.config.get('alpha_vantage_api_key'):
                self.data_sources['alpha_vantage'] = AlphaVantageSource(
                    self.config['alpha_vantage_api_key']
                )
            
            # IEX Cloud as tertiary source (if API key available)
            if self.config.get('iex_api_key'):
                self.data_sources['iex'] = IEXCloudSource(
                    self.config['iex_api_key']
                )
            
            logger.info(f"Initialized {len(self.data_sources)} data sources")
            
        except Exception as e:
            logger.error(f"Error initializing data sources: {e}")
    
    def _initialize_alert_system(self):
        """Initialize the alert system."""
        try:
            self.alert_system = AlertSystem()
            logger.info("Alert system initialized")
        except Exception as e:
            logger.error(f"Error initializing alert system: {e}")
    
    def _initialize_portfolio_monitor(self):
        """Initialize the portfolio monitoring system."""
        try:
            self.portfolio_monitor = PortfolioMonitor()
            logger.info("Portfolio monitor initialized")
        except Exception as e:
            logger.error(f"Error initializing portfolio monitor: {e}")
    
    def start_data_streams(self, symbols: List[str], update_frequency: int = 5):
        """
        Start real-time data streams for specified symbols.
        
        Args:
            symbols: List of symbols to monitor
            update_frequency: Update frequency in seconds
        """
        try:
            if not symbols:
                raise RealTimeDataError("No symbols provided for data streams")
            
            logger.info(f"Starting data streams for {len(symbols)} symbols")
            
            # Start data collection threads
            for symbol in symbols:
                if symbol not in self.data_streams:
                    stream = DataStream(symbol, self.data_sources, update_frequency)
                    self.data_streams[symbol] = stream
                    stream.start()
            
            self.is_running = True
            logger.info("Data streams started successfully")
            
        except Exception as e:
            logger.error(f"Error starting data streams: {e}")
            raise RealTimeDataError(f"Failed to start data streams: {str(e)}")
    
    def stop_data_streams(self):
        """Stop all data streams."""
        try:
            logger.info("Stopping data streams")
            
            for stream in self.data_streams.values():
                stream.stop()
            
            self.data_streams.clear()
            self.is_running = False
            
            logger.info("Data streams stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping data streams: {e}")
    
    def get_latest_data(self, symbol: str) -> Optional[MarketData]:
        """
        Get latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Latest market data or None if not available
        """
        try:
            if symbol in self.market_data_cache:
                return self.market_data_cache[symbol]
            
            # Try to get from active streams
            if symbol in self.data_streams:
                return self.data_streams[symbol].get_latest_data()
            
            # Fallback to direct API call
            for source_name, source in self.data_sources.items():
                try:
                    data = source.get_quote(symbol)
                    if data:
                        self.market_data_cache[symbol] = data
                        return data
                except Exception as e:
                    logger.debug(f"Source {source_name} failed for {symbol}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {e}")
            return None
    
    def get_portfolio_snapshot(self) -> Optional[PortfolioSnapshot]:
        """
        Get current portfolio snapshot.
        
        Returns:
            Current portfolio snapshot or None if not available
        """
        try:
            if self.portfolio_monitor:
                return self.portfolio_monitor.get_current_snapshot()
            return None
        except Exception as e:
            logger.error(f"Error getting portfolio snapshot: {e}")
            return None
    
    def add_alert(self, 
                  symbol: str, 
                  condition: str, 
                  threshold: float,
                  priority: str = "medium") -> str:
        """
        Add a new market alert.
        
        Args:
            symbol: Symbol to monitor
            condition: Alert condition ('above', 'below', 'change')
            threshold: Threshold value for alert
            priority: Alert priority
            
        Returns:
            Alert ID
        """
        try:
            if self.alert_system:
                alert_id = self.alert_system.add_alert(symbol, condition, threshold, priority)
                logger.info(f"Added alert {alert_id} for {symbol}")
                return alert_id
            else:
                raise RealTimeDataError("Alert system not initialized")
                
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            raise RealTimeDataError(f"Failed to add alert: {str(e)}")
    
    def remove_alert(self, alert_id: str):
        """
        Remove an alert.
        
        Args:
            alert_id: ID of alert to remove
        """
        try:
            if self.alert_system:
                self.alert_system.remove_alert(alert_id)
                logger.info(f"Removed alert {alert_id}")
            else:
                raise RealTimeDataError("Alert system not initialized")
                
        except Exception as e:
            logger.error(f"Error removing alert: {e}")
            raise RealTimeDataError(f"Failed to remove alert: {str(e)}")
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        try:
            if self.alert_system:
                return self.alert_system.get_active_alerts()
            return []
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    def register_data_callback(self, callback: Callable[[str, MarketData], None]):
        """
        Register callback for market data updates.
        
        Args:
            callback: Function to call when data updates
        """
        try:
            if callback not in self.data_callbacks:
                self.data_callbacks.append(callback)
                logger.info("Registered data callback")
        except Exception as e:
            logger.error(f"Error registering data callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """
        Register callback for alert notifications.
        
        Args:
            callback: Function to call when alerts trigger
        """
        try:
            if callback not in self.alert_callbacks:
                self.alert_callbacks.append(callback)
                logger.info("Registered alert callback")
        except Exception as e:
            logger.error(f"Error registering alert callback: {e}")
    
    def register_portfolio_callback(self, callback: Callable[[PortfolioSnapshot], None]):
        """
        Register callback for portfolio updates.
        
        Args:
            callback: Function to call when portfolio updates
        """
        try:
            if callback not in self.portfolio_callbacks:
                self.portfolio_callbacks.append(callback)
                logger.info("Registered portfolio callback")
        except Exception as e:
            logger.error(f"Error registering portfolio callback: {e}")
    
    def _notify_data_callbacks(self, symbol: str, data: MarketData):
        """Notify all registered data callbacks."""
        try:
            for callback in self.data_callbacks:
                try:
                    callback(symbol, data)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying data callbacks: {e}")
    
    def _notify_alert_callbacks(self, alert: Alert):
        """Notify all registered alert callbacks."""
        try:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying alert callbacks: {e}")
    
    def _notify_portfolio_callbacks(self, snapshot: PortfolioSnapshot):
        """Notify all registered portfolio callbacks."""
        try:
            for callback in self.portfolio_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in portfolio callback: {e}")
        except Exception as e:
            logger.error(f"Error notifying portfolio callbacks: {e}")


class DataStream:
    """Individual data stream for a single symbol."""
    
    def __init__(self, symbol: str, data_sources: Dict[str, Any], update_frequency: int):
        """
        Initialize data stream.
        
        Args:
            symbol: Symbol to monitor
            data_sources: Available data sources
            update_frequency: Update frequency in seconds
        """
        self.symbol = symbol
        self.data_sources = data_sources
        self.update_frequency = update_frequency
        self.is_running = False
        self.thread = None
        self.latest_data = None
        self.data_queue = Queue()
        
    def start(self):
        """Start the data stream."""
        try:
            if not self.is_running:
                self.is_running = True
                self.thread = threading.Thread(target=self._run_stream, daemon=True)
                self.thread.start()
                logger.info(f"Started data stream for {self.symbol}")
        except Exception as e:
            logger.error(f"Error starting data stream for {self.symbol}: {e}")
    
    def stop(self):
        """Stop the data stream."""
        try:
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=5)
            logger.info(f"Stopped data stream for {self.symbol}")
        except Exception as e:
            logger.error(f"Error stopping data stream for {self.symbol}: {e}")
    
    def _run_stream(self):
        """Main stream loop."""
        try:
            while self.is_running:
                try:
                    # Get data from sources
                    data = self._fetch_data()
                    if data:
                        self.latest_data = data
                        self.data_queue.put(data)
                    
                    # Wait for next update
                    time.sleep(self.update_frequency)
                    
                except Exception as e:
                    logger.error(f"Error in data stream for {self.symbol}: {e}")
                    time.sleep(self.update_frequency)
                    
        except Exception as e:
            logger.error(f"Fatal error in data stream for {self.symbol}: {e}")
    
    def _fetch_data(self) -> Optional[MarketData]:
        """Fetch data from available sources."""
        try:
            for source_name, source in self.data_sources.items():
                try:
                    data = source.get_quote(self.symbol)
                    if data:
                        return data
                except Exception as e:
                    logger.debug(f"Source {source_name} failed for {self.symbol}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {e}")
            return None
    
    def get_latest_data(self) -> Optional[MarketData]:
        """Get latest data from stream."""
        return self.latest_data
    
    def get_data_queue(self) -> Queue:
        """Get data queue for stream."""
        return self.data_queue


class YahooFinanceSource:
    """Yahoo Finance data source."""
    
    def __init__(self):
        """Initialize Yahoo Finance source."""
        self.name = "Yahoo Finance"
        self.rate_limit = 1.0  # seconds between requests
    
    def get_quote(self, symbol: str) -> Optional[MarketData]:
        """
        Get quote data for symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Market data or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get real-time quote
            hist = ticker.history(period="1d", interval="1m")
            if hist.empty:
                return None
            
            latest = hist.iloc[-1]
            
            # Create market data object
            data = MarketData(
                symbol=symbol,
                price=float(latest['Close']),
                volume=int(latest['Volume']),
                timestamp=datetime.now(),
                high=float(latest['High']) if 'High' in latest else None,
                low=float(latest['Low']) if 'Low' in latest else None,
                open=float(latest['Open']) if 'Open' in latest else None,
                previous_close=info.get('previousClose', None)
            )
            
            # Calculate change
            if data.previous_close:
                data.change = data.price - data.previous_close
                data.change_percent = (data.change / data.previous_close) * 100
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance quote for {symbol}: {e}")
            return None


class AlphaVantageSource:
    """Alpha Vantage data source."""
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage source.
        
        Args:
            api_key: API key for Alpha Vantage
        """
        self.name = "Alpha Vantage"
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit = 12.0  # seconds between requests (5 requests per minute)
    
    def get_quote(self, symbol: str) -> Optional[MarketData]:
        """
        Get quote data for symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Market data or None if failed
        """
        try:
            # This is a placeholder implementation
            # In practice, you would make actual API calls to Alpha Vantage
            
            logger.debug(f"Alpha Vantage quote request for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage quote for {symbol}: {e}")
            return None


class IEXCloudSource:
    """IEX Cloud data source."""
    
    def __init__(self, api_key: str):
        """
        Initialize IEX Cloud source.
        
        Args:
            api_key: API key for IEX Cloud
        """
        self.name = "IEX Cloud"
        self.api_key = api_key
        self.base_url = "https://cloud.iexapis.com/stable"
        self.rate_limit = 1.0  # seconds between requests
    
    def get_quote(self, symbol: str) -> Optional[MarketData]:
        """
        Get quote data for symbol.
        
        Args:
            symbol: Symbol to get quote for
            
        Returns:
            Market data or None if failed
        """
        try:
            # This is a placeholder implementation
            # In practice, you would make actual API calls to IEX Cloud
            
            logger.debug(f"IEX Cloud quote request for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting IEX Cloud quote for {symbol}: {e}")
            return None


class AlertSystem:
    """System for managing market alerts."""
    
    def __init__(self):
        """Initialize alert system."""
        self.alerts = {}
        self.alert_counter = 0
        self.is_monitoring = False
        self.monitor_thread = None
    
    def add_alert(self, symbol: str, condition: str, threshold: float, priority: str = "medium") -> str:
        """
        Add a new alert.
        
        Args:
            symbol: Symbol to monitor
            condition: Alert condition
            threshold: Threshold value
            priority: Alert priority
            
        Returns:
            Alert ID
        """
        try:
            alert_id = f"alert_{self.alert_counter:06d}"
            self.alert_counter += 1
            
            alert = Alert(
                id=alert_id,
                symbol=symbol,
                condition=condition,
                threshold=threshold,
                current_value=0.0,
                timestamp=datetime.now(),
                priority=priority
            )
            
            self.alerts[alert_id] = alert
            
            # Start monitoring if not already running
            if not self.is_monitoring:
                self._start_monitoring()
            
            logger.info(f"Added alert {alert_id} for {symbol}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error adding alert: {e}")
            raise RealTimeDataError(f"Failed to add alert: {str(e)}")
    
    def remove_alert(self, alert_id: str):
        """
        Remove an alert.
        
        Args:
            alert_id: ID of alert to remove
        """
        try:
            if alert_id in self.alerts:
                del self.alerts[alert_id]
                logger.info(f"Removed alert {alert_id}")
                
                # Stop monitoring if no alerts left
                if not self.alerts and self.is_monitoring:
                    self._stop_monitoring()
            else:
                logger.warning(f"Alert {alert_id} not found")
                
        except Exception as e:
            logger.error(f"Error removing alert {alert_id}: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get list of active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.alerts.values())
    
    def _start_monitoring(self):
        """Start monitoring alerts."""
        try:
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitor_thread = threading.Thread(target=self._monitor_alerts, daemon=True)
                self.monitor_thread.start()
                logger.info("Started alert monitoring")
        except Exception as e:
            logger.error(f"Error starting alert monitoring: {e}")
    
    def _stop_monitoring(self):
        """Stop monitoring alerts."""
        try:
            if self.is_monitoring:
                self.is_monitoring = False
                if self.monitor_thread:
                    self.monitor_thread.join(timeout=5)
                logger.info("Stopped alert monitoring")
        except Exception as e:
            logger.error(f"Error stopping alert monitoring: {e}")
    
    def _monitor_alerts(self):
        """Monitor alerts for triggering conditions."""
        try:
            while self.is_monitoring:
                try:
                    # Check each alert
                    for alert in self.alerts.values():
                        if not alert.triggered:
                            self._check_alert(alert)
                    
                    # Wait before next check
                    time.sleep(10)  # Check every 10 seconds
                    
                except Exception as e:
                    logger.error(f"Error in alert monitoring: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            logger.error(f"Fatal error in alert monitoring: {e}")
    
    def _check_alert(self, alert: Alert):
        """
        Check if an alert should be triggered.
        
        Args:
            alert: Alert to check
        """
        try:
            # This is a simplified implementation
            # In practice, you would get real-time data and check conditions
            
            # Placeholder logic
            if alert.condition == "above" and alert.current_value > alert.threshold:
                alert.triggered = True
                alert.message = f"{alert.symbol} is above {alert.threshold}"
                self._trigger_alert(alert)
            elif alert.condition == "below" and alert.current_value < alert.threshold:
                alert.triggered = True
                alert.message = f"{alert.symbol} is below {alert.threshold}"
                self._trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking alert {alert.id}: {e}")
    
    def _trigger_alert(self, alert: Alert):
        """
        Trigger an alert.
        
        Args:
            alert: Alert to trigger
        """
        try:
            logger.info(f"Alert triggered: {alert.message}")
            
            # Here you would implement notification logic
            # (email, SMS, push notification, etc.)
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.id}: {e}")


class PortfolioMonitor:
    """Real-time portfolio monitoring system."""
    
    def __init__(self):
        """Initialize portfolio monitor."""
        self.portfolio_data = {}
        self.current_snapshot = None
        self.is_monitoring = False
        self.monitor_thread = None
        self.update_frequency = 30  # seconds
    
    def start_monitoring(self, portfolio_data: Dict[str, Any]):
        """
        Start monitoring portfolio.
        
        Args:
            portfolio_data: Initial portfolio data
        """
        try:
            self.portfolio_data = portfolio_data
            
            if not self.is_monitoring:
                self.is_monitoring = True
                self.monitor_thread = threading.Thread(target=self._monitor_portfolio, daemon=True)
                self.monitor_thread.start()
                logger.info("Started portfolio monitoring")
                
        except Exception as e:
            logger.error(f"Error starting portfolio monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring portfolio."""
        try:
            if self.is_monitoring:
                self.is_monitoring = False
                if self.monitor_thread:
                    self.monitor_thread.join(timeout=5)
                logger.info("Stopped portfolio monitoring")
        except Exception as e:
            logger.error(f"Error stopping portfolio monitoring: {e}")
    
    def get_current_snapshot(self) -> Optional[PortfolioSnapshot]:
        """
        Get current portfolio snapshot.
        
        Returns:
            Current portfolio snapshot or None
        """
        return self.current_snapshot
    
    def _monitor_portfolio(self):
        """Monitor portfolio for updates."""
        try:
            while self.is_monitoring:
                try:
                    # Update portfolio snapshot
                    self._update_snapshot()
                    
                    # Wait before next update
                    time.sleep(self.update_frequency)
                    
                except Exception as e:
                    logger.error(f"Error in portfolio monitoring: {e}")
                    time.sleep(self.update_frequency)
                    
        except Exception as e:
            logger.error(f"Fatal error in portfolio monitoring: {e}")
    
    def _update_snapshot(self):
        """Update portfolio snapshot."""
        try:
            # This is a simplified implementation
            # In practice, you would calculate real-time portfolio metrics
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=100000.0,  # Placeholder
                total_return=5000.0,   # Placeholder
                total_return_percent=5.0,  # Placeholder
                daily_pnl=100.0,       # Placeholder
                daily_pnl_percent=0.1, # Placeholder
                positions={},           # Placeholder
                risk_metrics={}         # Placeholder
            )
            
            self.current_snapshot = snapshot
            
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")


def create_real_time_manager(config: Optional[Dict[str, Any]] = None) -> RealTimeDataManager:
    """Factory function to create real-time data manager."""
    return RealTimeDataManager(config)
