"""
Data Validation Module

This module provides comprehensive data validation for financial time series data,
including data quality checks, schema validation, and business rule validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import DataValidationError, ValidationError
from .config import get_data_config
from .logging_config import setup_logging, get_logger

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Enumeration of validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Enumeration of validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Definition of a validation rule."""
    
    name: str
    description: str
    severity: ValidationSeverity
    rule_function: callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def __post_init__(self):
        """Validate the validation rule."""
        if not callable(self.rule_function):
            raise ValueError("rule_function must be callable")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    
    rule_name: str
    status: ValidationStatus
    message: str
    severity: ValidationSeverity
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate the validation result."""
        if not isinstance(self.status, ValidationStatus):
            raise ValueError("status must be a ValidationStatus enum")


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    
    data_source: str
    validation_timestamp: datetime
    total_rules: int
    passed_rules: int
    failed_rules: int
    warning_rules: int
    skipped_rules: int
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate summary statistics."""
        self.summary = {
            'pass_rate': self.passed_rules / self.total_rules if self.total_rules > 0 else 0,
            'fail_rate': self.failed_rules / self.total_rules if self.total_rules > 0 else 0,
            'overall_status': self._get_overall_status()
        }
    
    def _get_overall_status(self) -> ValidationStatus:
        """Determine overall validation status."""
        if self.failed_rules > 0:
            return ValidationStatus.FAILED
        elif self.warning_rules > 0:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.PASSED
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the report."""
        self.results.append(result)
        
        # Update counts
        if result.status == ValidationStatus.PASSED:
            self.passed_rules += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed_rules += 1
        elif result.status == ValidationStatus.WARNING:
            self.warning_rules += 1
        elif result.status == ValidationStatus.SKIPPED:
            self.skipped_rules += 1
        
        # Recalculate summary
        self.total_rules = len(self.results)
        self.summary = {
            'pass_rate': self.passed_rules / self.total_rules if self.total_rules > 0 else 0,
            'fail_rate': self.failed_rules / self.total_rules if self.total_rules > 0 else 0,
            'overall_status': self._get_overall_status()
        }
    
    def get_failed_rules(self) -> List[ValidationResult]:
        """Get all failed validation rules."""
        return [r for r in self.results if r.status == ValidationStatus.FAILED]
    
    def get_warning_rules(self) -> List[ValidationResult]:
        """Get all warning validation rules."""
        return [r for r in self.results if r.status == ValidationStatus.WARNING]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'data_source': self.data_source,
            'validation_timestamp': self.validation_timestamp.isoformat(),
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'warning_rules': self.warning_rules,
            'skipped_rules': self.skipped_rules,
            'overall_status': self.summary['overall_status'].value,
            'pass_rate': self.summary['pass_rate'],
            'fail_rate': self.summary['fail_rate'],
            'results': [
                {
                    'rule_name': r.rule_name,
                    'status': r.status.value,
                    'message': r.message,
                    'severity': r.severity.value,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }


class DataValidator:
    """Comprehensive data validator for financial time series data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.
        
        Args:
            config (dict, optional): Validation configuration
        """
        self.config = config or get_data_config().__dict__
        self.validation_rules: List[ValidationRule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        # Data structure rules
        self.add_rule(
            ValidationRule(
                name="check_dataframe_type",
                description="Verify data is a pandas DataFrame",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_dataframe_type
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_non_empty_data",
                description="Verify data is not empty",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_non_empty_data
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_required_columns",
                description="Verify required columns are present",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_required_columns,
                parameters={'required_columns': ['date', 'close']}
            )
        )
        
        # Data quality rules
        self.add_rule(
            ValidationRule(
                name="check_missing_values",
                description="Check for missing values",
                severity=ValidationSeverity.WARNING,
                rule_function=self._check_missing_values,
                parameters={'max_missing_ratio': self.config.get('max_missing_ratio', 0.1)}
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_duplicate_dates",
                description="Check for duplicate dates",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_duplicate_dates
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_date_order",
                description="Verify dates are in chronological order",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_date_order
            )
        )
        
        # Financial data specific rules
        self.add_rule(
            ValidationRule(
                name="check_positive_prices",
                description="Verify price values are positive",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_positive_prices,
                parameters={'price_columns': ['open', 'high', 'low', 'close']}
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_price_consistency",
                description="Verify price consistency (high >= low, etc.)",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_price_consistency
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_outliers",
                description="Check for statistical outliers",
                severity=ValidationSeverity.WARNING,
                rule_function=self._check_outliers,
                parameters={'threshold': self.config.get('outlier_threshold', 3.0)}
            )
        )
        
        # Data completeness rules
        self.add_rule(
            ValidationRule(
                name="check_minimum_data_length",
                description="Verify minimum required data length",
                severity=ValidationSeverity.ERROR,
                rule_function=self._check_minimum_data_length,
                parameters={'min_length': self.config.get('min_data_length', 20)}
            )
        )
        
        self.add_rule(
            ValidationRule(
                name="check_trading_days",
                description="Check for reasonable number of trading days",
                severity=ValidationSeverity.WARNING,
                rule_function=self._check_trading_days
            )
        )
    
    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a custom validation rule.
        
        Args:
            rule (ValidationRule): Validation rule to add
        """
        self.validation_rules.append(rule)
        logger.info(f"Added validation rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a validation rule.
        
        Args:
            rule_name (str): Name of the rule to remove
            
        Returns:
            bool: True if rule was removed, False otherwise
        """
        for i, rule in enumerate(self.validation_rules):
            if rule.name == rule_name:
                del self.validation_rules[i]
                logger.info(f"Removed validation rule: {rule_name}")
                return True
        return False
    
    def validate_data(self, data: pd.DataFrame, data_source: str = "unknown") -> ValidationReport:
        """
        Validate financial time series data.
        
        Args:
            data (pd.DataFrame): Data to validate
            data_source (str): Source identifier for the data
            
        Returns:
            ValidationReport: Comprehensive validation report
        """
        logger.info(f"Starting data validation for {data_source}")
        
        # Initialize validation report
        report = ValidationReport(
            data_source=data_source,
            validation_timestamp=datetime.utcnow(),
            total_rules=len(self.validation_rules),
            passed_rules=0,
            failed_rules=0,
            warning_rules=0,
            skipped_rules=0
        )
        
        # Run all validation rules
        for rule in self.validation_rules:
            if not rule.enabled:
                result = ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.SKIPPED,
                    message=f"Rule {rule.name} is disabled",
                    severity=rule.severity
                )
                report.add_result(result)
                continue
            
            try:
                # Execute validation rule
                rule_result = rule.rule_function(data, **rule.parameters)
                
                if isinstance(rule_result, dict):
                    status = rule_result.get('status', ValidationStatus.PASSED)
                    message = rule_result.get('message', f"Rule {rule.name} passed")
                    details = rule_result.get('details', {})
                else:
                    status = ValidationStatus.PASSED if rule_result else ValidationStatus.FAILED
                    message = f"Rule {rule.name} {'passed' if rule_result else 'failed'}"
                    details = {}
                
                result = ValidationResult(
                    rule_name=rule.name,
                    status=status,
                    message=message,
                    severity=rule.severity,
                    details=details
                )
                
            except Exception as e:
                logger.error(f"Error executing validation rule {rule.name}: {e}")
                result = ValidationResult(
                    rule_name=rule.name,
                    status=ValidationStatus.FAILED,
                    message=f"Rule execution failed: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    details={'error': str(e), 'error_type': type(e).__name__}
                )
            
            report.add_result(result)
        
        # Log validation summary
        logger.info(f"Validation completed for {data_source}: "
                   f"{report.passed_rules} passed, {report.failed_rules} failed, "
                   f"{report.warning_rules} warnings")
        
        return report
    
    # Default validation rule implementations
    def _check_dataframe_type(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check if data is a pandas DataFrame."""
        if not isinstance(data, pd.DataFrame):
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Data must be a pandas DataFrame, got {type(data)}",
                'details': {'actual_type': str(type(data))}
            }
        return {'status': ValidationStatus.PASSED, 'message': "Data is a pandas DataFrame"}
    
    def _check_non_empty_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check if data is not empty."""
        if data.empty:
            return {
                'status': ValidationStatus.FAILED,
                'message': "Data is empty",
                'details': {'rows': 0, 'columns': 0}
            }
        return {
            'status': ValidationStatus.PASSED,
            'message': f"Data has {len(data)} rows and {len(data.columns)} columns",
            'details': {'rows': len(data), 'columns': len(data.columns)}
        }
    
    def _check_required_columns(self, data: pd.DataFrame, required_columns: List[str], **kwargs) -> Dict[str, Any]:
        """Check if required columns are present."""
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Missing required columns: {missing_columns}",
                'details': {'missing_columns': missing_columns, 'available_columns': list(data.columns)}
            }
        return {
            'status': ValidationStatus.PASSED,
            'message': f"All required columns present: {required_columns}",
            'details': {'required_columns': required_columns}
        }
    
    def _check_missing_values(self, data: pd.DataFrame, max_missing_ratio: float, **kwargs) -> Dict[str, Any]:
        """Check for missing values."""
        missing_counts = data.isnull().sum()
        missing_ratios = missing_counts / len(data)
        
        high_missing_columns = missing_ratios[missing_ratios > max_missing_ratio]
        
        if len(high_missing_columns) > 0:
            return {
                'status': ValidationStatus.WARNING,
                'message': f"High missing values in columns: {list(high_missing_columns.index)}",
                'details': {
                    'high_missing_columns': high_missing_columns.to_dict(),
                    'max_allowed_ratio': max_missing_ratio
                }
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': f"Missing values within acceptable limits (max ratio: {max_missing_ratio})",
            'details': {'missing_ratios': missing_ratios.to_dict()}
        }
    
    def _check_duplicate_dates(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check for duplicate dates."""
        if 'date' not in data.columns:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "No date column found, skipping duplicate date check",
                'details': {}
            }
        
        duplicate_dates = data[data.duplicated(subset=['date'], keep=False)]
        
        if len(duplicate_dates) > 0:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Found {len(duplicate_dates)} duplicate dates",
                'details': {
                    'duplicate_count': len(duplicate_dates),
                    'duplicate_dates': duplicate_dates['date'].tolist()
                }
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': "No duplicate dates found",
            'details': {}
        }
    
    def _check_date_order(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check if dates are in chronological order."""
        if 'date' not in data.columns:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "No date column found, skipping date order check",
                'details': {}
            }
        
        # Convert to datetime if needed
        try:
            dates = pd.to_datetime(data['date'])
        except Exception as e:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Failed to parse dates: {str(e)}",
                'details': {'error': str(e)}
            }
        
        if not dates.is_monotonic_increasing:
            return {
                'status': ValidationStatus.FAILED,
                'message': "Dates are not in chronological order",
                'details': {'first_date': dates.iloc[0], 'last_date': dates.iloc[-1]}
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': "Dates are in chronological order",
            'details': {'first_date': dates.iloc[0], 'last_date': dates.iloc[-1]}
        }
    
    def _check_positive_prices(self, data: pd.DataFrame, price_columns: List[str], **kwargs) -> Dict[str, Any]:
        """Check if price values are positive."""
        available_price_columns = [col for col in price_columns if col in data.columns]
        
        if not available_price_columns:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "No price columns found for positive price check",
                'details': {}
            }
        
        negative_prices = {}
        for col in available_price_columns:
            negative_mask = data[col] <= 0
            if negative_mask.any():
                negative_prices[col] = negative_mask.sum()
        
        if negative_prices:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Found negative or zero prices in columns: {list(negative_prices.keys())}",
                'details': {'negative_price_counts': negative_prices}
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': f"All price values are positive in columns: {available_price_columns}",
            'details': {'checked_columns': available_price_columns}
        }
    
    def _check_price_consistency(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check price consistency (high >= low, etc.)."""
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in data.columns]
        
        if len(available_columns) < 2:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "Insufficient price columns for consistency check",
                'details': {}
            }
        
        inconsistencies = []
        
        # Check high >= low
        if 'high' in available_columns and 'low' in available_columns:
            high_low_violations = (data['high'] < data['low']).sum()
            if high_low_violations > 0:
                inconsistencies.append(f"High < Low: {high_low_violations} violations")
        
        # Check high >= open, close
        if 'high' in available_columns:
            for col in ['open', 'close']:
                if col in available_columns:
                    violations = (data['high'] < data[col]).sum()
                    if violations > 0:
                        inconsistencies.append(f"High < {col.capitalize()}: {violations} violations")
        
        # Check low <= open, close
        if 'low' in available_columns:
            for col in ['open', 'close']:
                if col in available_columns:
                    violations = (data['low'] > data[col]).sum()
                    if violations > 0:
                        inconsistencies.append(f"Low > {col.capitalize()}: {violations} violations")
        
        if inconsistencies:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Price consistency violations found: {len(inconsistencies)} types",
                'details': {'inconsistencies': inconsistencies}
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': "All price consistency checks passed",
            'details': {'checked_columns': available_columns}
        }
    
    def _check_outliers(self, data: pd.DataFrame, threshold: float, **kwargs) -> Dict[str, Any]:
        """Check for statistical outliers."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "No numeric columns found for outlier detection",
                'details': {}
            }
        
        outliers = {}
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': outlier_count / len(data) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        if outliers:
            return {
                'status': ValidationStatus.WARNING,
                'message': f"Found outliers in {len(outliers)} columns",
                'details': {'outliers': outliers, 'threshold': threshold}
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': f"No outliers detected using threshold {threshold}",
            'details': {'checked_columns': list(numeric_columns), 'threshold': threshold}
        }
    
    def _check_minimum_data_length(self, data: pd.DataFrame, min_length: int, **kwargs) -> Dict[str, Any]:
        """Check if data meets minimum length requirement."""
        if len(data) < min_length:
            return {
                'status': ValidationStatus.FAILED,
                'message': f"Data length {len(data)} is below minimum requirement {min_length}",
                'details': {'actual_length': len(data), 'required_length': min_length}
            }
        
        return {
            'status': ValidationStatus.PASSED,
            'message': f"Data length {len(data)} meets minimum requirement {min_length}",
            'details': {'actual_length': len(data), 'required_length': min_length}
        }
    
    def _check_trading_days(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Check for reasonable number of trading days."""
        if 'date' not in data.columns:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': "No date column found, skipping trading days check",
                'details': {}
            }
        
        try:
            dates = pd.to_datetime(data['date'])
            date_range = dates.max() - dates.min()
            trading_days = len(dates)
            
            # Calculate expected trading days (assuming 252 trading days per year)
            years = date_range.days / 365.25
            expected_trading_days = years * 252
            
            # Allow for some flexibility (80-120% of expected)
            min_expected = expected_trading_days * 0.8
            max_expected = expected_trading_days * 1.2
            
            if trading_days < min_expected or trading_days > max_expected:
                return {
                    'status': ValidationStatus.WARNING,
                    'message': f"Trading days count may be unusual",
                    'details': {
                        'actual_trading_days': trading_days,
                        'expected_range': [min_expected, max_expected],
                        'date_range_years': years
                    }
                }
            
            return {
                'status': ValidationStatus.PASSED,
                'message': f"Trading days count is reasonable",
                'details': {
                    'actual_trading_days': trading_days,
                    'expected_range': [min_expected, max_expected],
                    'date_range_years': years
                }
            }
            
        except Exception as e:
            return {
                'status': ValidationStatus.SKIPPED,
                'message': f"Could not analyze trading days: {str(e)}",
                'details': {'error': str(e)}
            }


# Convenience function for quick validation
def validate_financial_data(data: pd.DataFrame, data_source: str = "unknown", 
                           config: Optional[Dict[str, Any]] = None) -> ValidationReport:
    """
    Quick validation function for financial data.
    
    Args:
        data (pd.DataFrame): Data to validate
        data_source (str): Source identifier for the data
        config (dict, optional): Validation configuration
        
    Returns:
        ValidationReport: Validation report
    """
    validator = DataValidator(config)
    return validator.validate_data(data, data_source)
