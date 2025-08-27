"""
Type definitions for the trading application.
Provides type safety through enums, TypedDicts, and protocols.
"""

from enum import Enum
from typing import Protocol, TypedDict, Optional, Dict, Any, List, Union
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
import pandas as pd


# Order and Trading Types
class OrderAction(Enum):
    """Order action types"""
    BUY = "BUY"
    SELL = "SELL"


class SecurityType(Enum):
    """Security types supported by IBKR"""
    STOCK = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"
    CASH = "CASH"
    BOND = "BOND"
    CRYPTO = "CRYPTO"
    CFD = "CFD"
    FUND = "FUND"
    WAR = "WAR"  # Warrant
    IOPT = "IOPT"  # Dutch Warrant
    BAG = "BAG"  # Combo
    IND = "IND"  # Index
    NEWS = "NEWS"
    CMDTY = "CMDTY"  # Commodity
    INDEX = "IND"  # Index (alias)
    COMMODITY = "CMDTY"  # Commodity (alias)
    ETF = "STK"  # ETF (treated as stock)


class OrderType(Enum):
    """Order types supported"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"
    MARKET_TO_LIMIT = "MTL"
    MARKET_IF_TOUCHED = "MIT"
    LIMIT_IF_TOUCHED = "LIT"
    PEGGED_TO_MARKET = "PEG MKT"
    RELATIVE = "REL"
    VOLATILITY = "VOL"
    BOX_TOP = "BOX TOP"
    LIMIT_ON_CLOSE = "LOC"
    LIMIT_ON_OPEN = "LOO"
    MARKET_ON_CLOSE = "MOC"
    MARKET_ON_OPEN = "MOO"
    PEGGED_TO_MIDPOINT = "PEG MID"
    AUCTION = "AUC"


class TimeInForce(Enum):
    """Time in force options"""
    DAY = "DAY"
    GOOD_TILL_CANCELLED = "GTC"
    IMMEDIATE_OR_CANCEL = "IOC"
    FILL_OR_KILL = "FOK"
    GOOD_TILL_DATE = "GTD"
    OPENING = "OPG"
    AUCTION = "AUC"


class OrderStatus(Enum):
    """Order status types"""
    PENDING_SUBMIT = "PendingSubmit"
    PENDING_CANCEL = "PendingCancel"
    PRE_SUBMITTED = "PreSubmitted"
    SUBMITTED = "Submitted"
    CANCELLED = "Cancelled"
    FILLED = "Filled"
    INACTIVE = "Inactive"


class TickType(Enum):
    """Market data tick types"""
    BID_SIZE = 0
    BID_PRICE = 1
    ASK_PRICE = 2
    ASK_SIZE = 3
    LAST_PRICE = 4
    LAST_SIZE = 5
    HIGH = 6
    LOW = 7
    VOLUME = 8
    CLOSE_PRICE = 9
    BID_OPTION_COMPUTATION = 10
    ASK_OPTION_COMPUTATION = 11
    LAST_OPTION_COMPUTATION = 12
    MODEL_OPTION = 13
    OPEN = 14


# Data Structures
class TickData(TypedDict):
    """Tick data structure for real-time market data"""
    timestamp: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    spread: Optional[float]
    mid_price: Optional[float]


class BarData(TypedDict):
    """Historical bar data structure"""
    timestamp: str
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class OrderData(TypedDict):
    """Order information structure"""
    order_id: int
    symbol: str
    order_type: str
    action: str
    quantity: Decimal
    limit_price: Optional[float]
    stop_price: Optional[float]
    status: str
    filled_quantity: Decimal
    avg_fill_price: Optional[float]
    commission: Optional[float]
    created_at: datetime
    updated_at: datetime


class PositionData(TypedDict):
    """Position information structure"""
    symbol: str
    position: Decimal
    market_price: float
    market_value: float
    average_cost: float
    unrealized_pnl: float
    realized_pnl: float


class AccountData(TypedDict):
    """Account information structure"""
    account_id: str
    net_liquidation: float
    total_cash: float
    buying_power: float
    gross_position_value: float
    unrealized_pnl: float
    realized_pnl: float


class PerformanceMetrics(TypedDict):
    """Portfolio performance metrics"""
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    omega_ratio: float
    cvar: tuple[float, float]  # (ratio, dollar_amount)
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    alpha: Optional[float]
    beta: Optional[float]


class RiskMetrics(TypedDict):
    """Risk management metrics"""
    value_at_risk: float
    expected_shortfall: float
    maximum_drawdown: float
    daily_vol: float
    portfolio_beta: float
    concentration_risk: float


# Protocols for dependency injection
class MarketDataProvider(Protocol):
    """Protocol for market data providers"""
    async def get_tick_data(self, symbol: str) -> TickData: ...
    async def get_historical_data(self, symbol: str, duration: str, bar_size: str) -> pd.DataFrame: ...
    async def subscribe_to_ticks(self, symbol: str) -> None: ...
    async def unsubscribe_from_ticks(self, symbol: str) -> None: ...


class OrderExecutionProvider(Protocol):
    """Protocol for order execution"""
    async def place_order(self, symbol: str, order_type: str, action: str, quantity: Decimal, **kwargs) -> int: ...
    async def cancel_order(self, order_id: int) -> bool: ...
    async def get_order_status(self, order_id: int) -> OrderStatus: ...
    async def get_open_orders(self) -> List[OrderData]: ...


class PortfolioProvider(Protocol):
    """Protocol for portfolio management"""
    async def get_positions(self) -> Dict[str, PositionData]: ...
    async def get_account_summary(self) -> AccountData: ...
    async def get_portfolio_value(self) -> float: ...


class DataStorageProvider(Protocol):
    """Protocol for data storage"""
    async def store_tick_data(self, tick: TickData) -> None: ...
    async def store_order_data(self, order: OrderData) -> None: ...
    async def get_historical_ticks(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...


class EventListener(Protocol):
    """Protocol for event listeners"""
    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> None: ...


# Algorithm and Strategy Types
class AlgorithmType(Enum):
    """Algorithm types for order execution"""
    VWAP = "Vwap"
    TWAP = "Twap"
    ARRIVAL_PX = "ArrivalPx"
    DARK_ICE = "DarkIce"
    BALANCE_IMPACT_RISK = "BalanceImpactRisk"
    MIN_IMPACT = "MinImpact"
    ADAPTIVE = "Adaptive"
    CLOSE_PX = "ClosePx"


class StrategyType(Enum):
    """Trading strategy types"""
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    PAIRS_TRADING = "pairs_trading"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"


# Contract and Instrument Types
class Exchange(Enum):
    """Common exchanges"""
    SMART = "SMART"
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    CBOE = "CBOE"
    CME = "CME"
    NYMEX = "NYMEX"
    EUREX = "EUREX"
    LSE = "LSE"
    TSE = "TSE"
    IDEALPRO = "IDEALPRO"


class Currency(Enum):
    """Common currencies"""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"


class OptionRight(Enum):
    """Option rights"""
    CALL = "C"
    PUT = "P"


# Event Types
class EventType(Enum):
    """System event types"""
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    POSITION_UPDATED = "position_updated"
    ACCOUNT_UPDATED = "account_updated"
    MARKET_DATA_UPDATED = "market_data_updated"
    CONNECTION_STATUS_CHANGED = "connection_status_changed"
    ERROR_OCCURRED = "error_occurred"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"


# Configuration Types
class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Type aliases for common types
Price = Union[float, Decimal]
Quantity = Union[int, Decimal]
ContractId = Union[int, str]
RequestId = int
OrderId = int
SymbolList = List[str]
PriceDict = Dict[str, Price]
VolumeDict = Dict[str, Quantity]
ReturnSeries = pd.Series


# Additional types for testing and analytics
class MarketDataType(Enum):
    """Market data subscription types"""
    REAL_TIME = "realtime"
    DELAYED = "delayed"
    FROZEN = "frozen"


@dataclass
class PositionData:
    """Position data for portfolio analytics"""
    account: str
    symbol: str
    sec_type: str
    exchange: str
    currency: str
    position: float
    avg_cost: float
    timestamp: datetime


@dataclass  
class AccountValue:
    """Account value data"""
    key: str
    value: str
    currency: str
    account: str
    timestamp: datetime
