"""
Custom exception hierarchy for the trading application.
Provides specific exceptions for different error scenarios.
"""

from typing import Optional, Any, Dict


class IBKRError(Exception):
    """Base exception for IBKR-related errors"""
    
    def __init__(self, message: str, error_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        error_str = f"IBKRError: {self.message}"
        if self.error_code:
            error_str += f" (Code: {self.error_code})"
        if self.details:
            error_str += f" Details: {self.details}"
        return error_str


class ConnectionError(IBKRError):
    """Raised when connection to IBKR fails"""
    
    def __init__(self, message: str = "Failed to connect to IBKR", **kwargs):
        super().__init__(message, **kwargs)


class AuthenticationError(IBKRError):
    """Raised when authentication with IBKR fails"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class OrderError(IBKRError):
    """Raised when order execution fails"""
    
    def __init__(self, message: str, order_id: Optional[int] = None, **kwargs):
        self.order_id = order_id
        if order_id:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['order_id'] = order_id
        super().__init__(message, **kwargs)


class OrderRejectionError(OrderError):
    """Raised when an order is rejected by the broker"""
    
    def __init__(self, message: str, reason: Optional[str] = None, **kwargs):
        self.reason = reason
        if reason:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['rejection_reason'] = reason
        super().__init__(message, **kwargs)


class InsufficientFundsError(OrderError):
    """Raised when there are insufficient funds for an order"""
    
    def __init__(self, message: str = "Insufficient funds for order", required_amount: Optional[float] = None, **kwargs):
        self.required_amount = required_amount
        if required_amount:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['required_amount'] = required_amount
        super().__init__(message, **kwargs)


class DataError(IBKRError):
    """Raised when data retrieval fails"""
    
    def __init__(self, message: str, symbol: Optional[str] = None, request_id: Optional[int] = None, **kwargs):
        self.symbol = symbol
        self.request_id = request_id
        if symbol:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['symbol'] = symbol
        if request_id:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['request_id'] = request_id
        super().__init__(message, **kwargs)


class MarketDataError(DataError):
    """Raised when market data retrieval fails"""
    
    def __init__(self, message: str, data_type: Optional[str] = None, **kwargs):
        self.data_type = data_type
        if data_type:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['data_type'] = data_type
        super().__init__(message, **kwargs)


class HistoricalDataError(DataError):
    """Raised when historical data retrieval fails"""
    
    def __init__(self, message: str, start_date: Optional[str] = None, end_date: Optional[str] = None, **kwargs):
        self.start_date = start_date
        self.end_date = end_date
        if start_date or end_date:
            kwargs['details'] = kwargs.get('details', {})
            if start_date:
                kwargs['details']['start_date'] = start_date
            if end_date:
                kwargs['details']['end_date'] = end_date
        super().__init__(message, **kwargs)


class ValidationError(IBKRError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        self.field = field
        self.value = value
        if field or value is not None:
            kwargs['details'] = kwargs.get('details', {})
            if field:
                kwargs['details']['field'] = field
            if value is not None:
                kwargs['details']['value'] = str(value)
        super().__init__(message, **kwargs)


class ContractValidationError(ValidationError):
    """Raised when contract validation fails"""
    
    def __init__(self, message: str, contract_field: Optional[str] = None, **kwargs):
        self.contract_field = contract_field
        if contract_field:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['contract_field'] = contract_field
        super().__init__(message, **kwargs)


class OrderValidationError(ValidationError):
    """Raised when order validation fails"""
    
    def __init__(self, message: str, order_field: Optional[str] = None, **kwargs):
        self.order_field = order_field
        if order_field:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['order_field'] = order_field
        super().__init__(message, **kwargs)


class RateLimitError(IBKRError):
    """Raised when API rate limits are exceeded"""
    
    def __init__(self, message: str = "API rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        self.retry_after = retry_after
        if retry_after:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['retry_after'] = retry_after
        super().__init__(message, **kwargs)


class TimeoutError(IBKRError):
    """Raised when operations timeout"""
    
    def __init__(self, message: str, operation: Optional[str] = None, timeout_duration: Optional[float] = None, **kwargs):
        self.operation = operation
        self.timeout_duration = timeout_duration
        if operation or timeout_duration:
            kwargs['details'] = kwargs.get('details', {})
            if operation:
                kwargs['details']['operation'] = operation
            if timeout_duration:
                kwargs['details']['timeout_duration'] = timeout_duration
        super().__init__(message, **kwargs)


class ConfigurationError(IBKRError):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        self.config_key = config_key
        if config_key:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['config_key'] = config_key
        super().__init__(message, **kwargs)


class DatabaseError(IBKRError):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, table: Optional[str] = None, **kwargs):
        self.operation = operation
        self.table = table
        if operation or table:
            kwargs['details'] = kwargs.get('details', {})
            if operation:
                kwargs['details']['operation'] = operation
            if table:
                kwargs['details']['table'] = table
        super().__init__(message, **kwargs)


class PortfolioError(IBKRError):
    """Raised when portfolio operations fail"""
    
    def __init__(self, message: str, account: Optional[str] = None, **kwargs):
        self.account = account
        if account:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['account'] = account
        super().__init__(message, **kwargs)


class CalculationError(IBKRError):
    """Raised when mathematical calculations fail"""
    
    def __init__(self, message: str, calculation_type: Optional[str] = None, **kwargs):
        self.calculation_type = calculation_type
        if calculation_type:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['calculation_type'] = calculation_type
        super().__init__(message, **kwargs)


class RiskManagementError(IBKRError):
    """Raised when risk management rules are violated"""
    
    def __init__(self, message: str, risk_type: Optional[str] = None, current_value: Optional[float] = None, 
                 limit_value: Optional[float] = None, **kwargs):
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value
        
        if any([risk_type, current_value, limit_value]):
            kwargs['details'] = kwargs.get('details', {})
            if risk_type:
                kwargs['details']['risk_type'] = risk_type
            if current_value is not None:
                kwargs['details']['current_value'] = current_value
            if limit_value is not None:
                kwargs['details']['limit_value'] = limit_value
        
        super().__init__(message, **kwargs)


class PositionSizeError(RiskManagementError):
    """Raised when position size limits are exceeded"""
    
    def __init__(self, message: str = "Position size limit exceeded", **kwargs):
        super().__init__(message, risk_type="position_size", **kwargs)


class DrawdownError(RiskManagementError):
    """Raised when maximum drawdown is exceeded"""
    
    def __init__(self, message: str = "Maximum drawdown exceeded", **kwargs):
        super().__init__(message, risk_type="max_drawdown", **kwargs)


class ConcentrationRiskError(RiskManagementError):
    """Raised when concentration risk limits are exceeded"""
    
    def __init__(self, message: str = "Concentration risk limit exceeded", **kwargs):
        super().__init__(message, risk_type="concentration", **kwargs)


class MarketClosedError(IBKRError):
    """Raised when attempting to trade while market is closed"""
    
    def __init__(self, message: str = "Market is closed", market: Optional[str] = None, **kwargs):
        self.market = market
        if market:
            kwargs['details'] = kwargs.get('details', {})
            kwargs['details']['market'] = market
        super().__init__(message, **kwargs)


class CircuitBreakerError(IBKRError):
    """Raised when circuit breaker is triggered"""
    
    def __init__(self, message: str = "Circuit breaker activated", 
                 failure_count: Optional[int] = None, threshold: Optional[int] = None, **kwargs):
        self.failure_count = failure_count
        self.threshold = threshold
        
        if failure_count is not None or threshold is not None:
            kwargs['details'] = kwargs.get('details', {})
            if failure_count is not None:
                kwargs['details']['failure_count'] = failure_count
            if threshold is not None:
                kwargs['details']['threshold'] = threshold
        
        super().__init__(message, **kwargs)


# Convenience functions for common error scenarios
def raise_connection_error(details: Optional[str] = None) -> None:
    """Raise a connection error with optional details"""
    message = "Failed to establish connection to IBKR"
    if details:
        message += f": {details}"
    raise ConnectionError(message)


def raise_order_error(order_id: int, reason: str) -> None:
    """Raise an order error with specific order ID and reason"""
    raise OrderError(f"Order {order_id} failed: {reason}", order_id=order_id)


def raise_data_error(symbol: str, data_type: str, reason: str) -> None:
    """Raise a data error for specific symbol and data type"""
    raise MarketDataError(f"Failed to get {data_type} for {symbol}: {reason}", 
                         symbol=symbol, data_type=data_type)


def raise_validation_error(field: str, value: Any, expected: str) -> None:
    """Raise a validation error for specific field"""
    raise ValidationError(f"Invalid value for {field}: {value}. Expected: {expected}", 
                         field=field, value=value)
