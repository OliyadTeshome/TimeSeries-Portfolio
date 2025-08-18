"""
Enhanced Logging and Error Handling Configuration

This module provides comprehensive logging configuration, error handling,
and performance monitoring for the TimeSeries-Portfolio framework.
"""

import logging
import logging.handlers
import sys
import traceback
import time
import functools
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json
from datetime import datetime
import warnings

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class PerformanceLogger:
    """Performance monitoring and timing utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timings: Dict[str, float] = {}
    
    def time_function(self, func_name: Optional[str] = None):
        """
        Decorator to time function execution.
        
        Args:
            func_name (str, optional): Custom name for the function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = func_name or func.__name__
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.timings[name] = execution_time
                    
                    self.logger.info(
                        f"Function '{name}' completed in {execution_time:.4f}s",
                        extra={'extra_fields': {
                            'function_name': name,
                            'execution_time': execution_time,
                            'status': 'success'
                        }}
                    )
                    
                    return result
                
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.timings[name] = execution_time
                    
                    self.logger.error(
                        f"Function '{name}' failed after {execution_time:.4f}s: {str(e)}",
                        extra={'extra_fields': {
                            'function_name': name,
                            'execution_time': execution_time,
                            'status': 'failed',
                            'error': str(e)
                        }}
                    )
                    raise
            
            return wrapper
        return decorator
    
    def start_timer(self, name: str) -> None:
        """Start a timer for a named operation."""
        self.timings[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str) -> float:
        """
        End a timer and return elapsed time.
        
        Args:
            name (str): Timer name
            
        Returns:
            float: Elapsed time in seconds
        """
        start_key = f"{name}_start"
        if start_key not in self.timings:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.timings[start_key]
        self.timings[name] = elapsed
        
        self.logger.info(
            f"Operation '{name}' completed in {elapsed:.4f}s",
            extra={'extra_fields': {
                'operation_name': name,
                'elapsed_time': elapsed
            }}
        )
        
        return elapsed
    
    def get_timings(self) -> Dict[str, float]:
        """Get all recorded timings."""
        return {k: v for k, v in self.timings.items() if not k.endswith('_start')}
    
    def print_summary(self) -> None:
        """Print a summary of all timings."""
        timings = self.get_timings()
        if not timings:
            self.logger.info("No timing data available")
            return
        
        self.logger.info("Performance Summary:")
        for name, time_taken in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {name}: {time_taken:.4f}s")


class ErrorTracker:
    """Error tracking and reporting utilities."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.error_count = 0
        self.error_summary: Dict[str, int] = {}
    
    def track_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Track an error occurrence.
        
        Args:
            error (Exception): The error that occurred
            context (Dict[str, Any], optional): Additional context
        """
        self.error_count += 1
        error_type = type(error).__name__
        self.error_summary[error_type] = self.error_summary.get(error_type, 0) + 1
        
        # Log the error with context
        extra_fields = {
            'error_type': error_type,
            'error_message': str(error),
            'error_count': self.error_count,
            'context': context or {}
        }
        
        self.logger.error(
            f"Error {self.error_count}: {error_type} - {str(error)}",
            extra={'extra_fields': extra_fields},
            exc_info=True
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all tracked errors."""
        return {
            'total_errors': self.error_count,
            'error_types': self.error_summary,
            'most_common_error': max(self.error_summary.items(), key=lambda x: x[1]) if self.error_summary else None
        }
    
    def print_summary(self) -> None:
        """Print a summary of all errors."""
        summary = self.get_error_summary()
        self.logger.info("Error Summary:")
        self.logger.info(f"  Total Errors: {summary['total_errors']}")
        self.logger.info("  Error Types:")
        for error_type, count in summary['error_types'].items():
            self.logger.info(f"    {error_type}: {count}")
        
        if summary['most_common_error']:
            error_type, count = summary['most_common_error']
            self.logger.info(f"  Most Common: {error_type} ({count} occurrences)")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "colored",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_performance_logging: bool = True,
    enable_error_tracking: bool = True
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file
        log_format (str): Log format ('colored', 'structured', or 'standard')
        max_file_size (int): Maximum log file size in bytes
        backup_count (int): Number of backup log files to keep
        enable_performance_logging (bool): Enable performance monitoring
        enable_error_tracking (bool): Enable error tracking
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if log_format == "colored":
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    elif log_format == "structured":
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        if log_format == "structured":
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Create framework logger
    framework_logger = logging.getLogger('timeseries_portfolio')
    framework_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Add performance logger if enabled
    if enable_performance_logging:
        framework_logger.performance_logger = PerformanceLogger(framework_logger)
    
    # Add error tracker if enabled
    if enable_error_tracking:
        framework_logger.error_tracker = ErrorTracker(framework_logger)
    
    # Log configuration
    framework_logger.info(f"Logging configured - Level: {log_level}, Format: {log_format}")
    if log_file:
        framework_logger.info(f"Log file: {log_file}")
    
    return framework_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(f'timeseries_portfolio.{name}')


def log_execution_time(logger: logging.Logger, operation_name: Optional[str] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger (logging.Logger): Logger instance
        operation_name (str, optional): Custom operation name
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.info(
                    f"Operation '{name}' completed in {execution_time:.4f}s",
                    extra={'extra_fields': {
                        'operation_name': name,
                        'execution_time': execution_time,
                        'status': 'success'
                    }}
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                
                logger.error(
                    f"Operation '{name}' failed after {execution_time:.4f}s: {str(e)}",
                    extra={'extra_fields': {
                        'operation_name': name,
                        'execution_time': execution_time,
                        'status': 'failed',
                        'error': str(e)
                    }}
                )
                raise
        
        return wrapper
    return decorator


def log_errors(logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
    """
    Decorator to log errors with context.
    
    Args:
        logger (logging.Logger): Logger instance
        context (Dict[str, Any], optional): Additional context
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                # Add function context
                error_context = {
                    'function_name': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                if context:
                    error_context.update(context)
                
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    extra={'extra_fields': error_context},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# Default logging setup
def setup_default_logging() -> logging.Logger:
    """
    Set up default logging configuration.
    
    Returns:
        logging.Logger: Configured logger
    """
    return setup_logging(
        log_level="INFO",
        log_file="logs/timeseries_portfolio.log",
        log_format="colored",
        enable_performance_logging=True,
        enable_error_tracking=True
    )


# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            elapsed = time.time() - self.start_time
            status = "failed" if exc_type else "completed"
            
            self.logger.info(
                f"Operation '{self.operation_name}' {status} in {elapsed:.4f}s",
                extra={'extra_fields': {
                    'operation_name': self.operation_name,
                    'elapsed_time': elapsed,
                    'status': status
                }}
            )


# Utility function to create timer context
def timer(logger: logging.Logger, operation_name: str):
    """
    Create a timer context for timing operations.
    
    Args:
        logger (logging.Logger): Logger instance
        operation_name (str): Name of the operation to time
        
    Returns:
        TimerContext: Timer context manager
    """
    return TimerContext(logger, operation_name)
