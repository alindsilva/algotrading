# IBKR API Refactoring Plan

This document outlines a comprehensive refactoring strategy that combines the sophisticated IBKR API capabilities with the practical legacy functionality, while modernizing the codebase for production use.

## Project Goals

1. **Preserve Legacy Features**: Maintain SQLite storage, performance metrics, and core functionality
2. **Leverage IBKR Sophistication**: Incorporate advanced order types, algorithms, and contract support
3. **Modernize Architecture**: Implement async patterns, proper error handling, and type safety
4. **Enhance Security**: Remove hardcoded secrets and implement secure credential management
5. **Improve Maintainability**: Add comprehensive testing, logging, and documentation

## Architecture Overview

### New Modular Structure

```
src/
├── core/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── exceptions.py          # Custom exceptions
│   └── types.py              # Type definitions and enums
├── data/
│   ├── __init__.py
│   ├── storage.py            # SQLite operations (enhanced)
│   ├── cache.py              # Data caching with LRU
│   └── models.py             # Data models and schemas
├── api/
│   ├── __init__.py
│   ├── client.py             # Async IBKR client
│   ├── wrapper.py            # Event handlers
│   ├── connection.py         # Connection management
│   └── rate_limiter.py       # API rate limiting
├── contracts/
│   ├── __init__.py
│   ├── factory.py            # Enhanced contract factory
│   ├── validator.py          # Contract validation
│   └── samples.py            # All contract types from IBKR
├── orders/
│   ├── __init__.py
│   ├── factory.py            # Enhanced order factory
│   ├── manager.py            # Order lifecycle management
│   ├── algorithms.py         # Algo trading parameters
│   └── samples.py            # All order types from IBKR
├── portfolio/
│   ├── __init__.py
│   ├── manager.py            # Portfolio management
│   ├── analytics.py          # Performance metrics (preserved)
│   ├── optimization.py       # Portfolio optimization
│   └── risk.py               # Risk management
├── streaming/
│   ├── __init__.py
│   ├── market_data.py        # Real-time market data
│   ├── tick_storage.py       # Enhanced stream_to_sqlite
│   └── events.py             # Event-driven architecture
└── app/
    ├── __init__.py
    ├── application.py        # Main application
    ├── cli.py               # Command-line interface
    └── dashboard.py         # Optional web dashboard
```

## Phase 1: Foundation & Security (Week 1-2)

### 1.1 Configuration Management
**Goal**: Remove all hardcoded values and secrets

```python
# src/core/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    path: Path = field(default_factory=lambda: Path("data/trading.db"))
    timeout: float = 30.0
    max_connections: int = 5
    enable_wal_mode: bool = True
    backup_interval: int = 3600  # seconds

@dataclass
class IBConfig:
    host: str = field(default_factory=lambda: os.getenv("IB_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.getenv("IB_PORT", "7497")))
    client_id: int = field(default_factory=lambda: int(os.getenv("IB_CLIENT_ID", "1")))
    account: Optional[str] = field(default_factory=lambda: os.getenv("IB_ACCOUNT"))
    
    # Timeouts and limits
    connection_timeout: float = 30.0
    request_timeout: float = 10.0
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    rate_limit_per_second: int = 50
    max_concurrent_requests: int = 10

@dataclass
class TradingConfig:
    max_position_size: float = 0.1  # 10% of portfolio
    max_daily_trades: int = 100
    risk_free_rate: float = 0.05
    benchmark_symbol: str = "SPY"
    performance_calculation_interval: int = 300  # 5 minutes

@dataclass
class Config:
    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    ibkr: IBConfig = field(default_factory=IBConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    
    def __post_init__(self):
        # Validate required environment variables in production
        if self.environment == Environment.PRODUCTION:
            required_vars = ["IB_ACCOUNT", "IB_CLIENT_ID"]
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                raise ValueError(f"Missing required environment variables: {missing}")
```

### 1.2 Enhanced Type System
```python
# src/core/types.py
from enum import Enum
from typing import Protocol, TypedDict, Literal
from decimal import Decimal

class OrderAction(Enum):
    BUY = "BUY"
    SELL = "SELL"

class SecurityType(Enum):
    STOCK = "STK"
    OPTION = "OPT"
    FUTURE = "FUT"
    CASH = "CASH"
    BOND = "BOND"
    CRYPTO = "CRYPTO"

class OrderType(Enum):
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAIL = "TRAIL"
    TRAIL_LIMIT = "TRAIL LIMIT"

class TickType(TypedDict):
    timestamp: str
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int

class PerformanceMetrics(TypedDict):
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    omega_ratio: float
    cvar: tuple[float, float]  # (ratio, dollar_amount)

class OrderExecutionProtocol(Protocol):
    async def execute_order(self, contract, order) -> int: ...
    async def cancel_order(self, order_id: int) -> bool: ...
```

### 1.3 Custom Exceptions
```python
# src/core/exceptions.py
class IBKRError(Exception):
    """Base exception for IBKR-related errors"""
    pass

class ConnectionError(IBKRError):
    """Raised when connection to IBKR fails"""
    pass

class OrderError(IBKRError):
    """Raised when order execution fails"""
    pass

class DataError(IBKRError):
    """Raised when data retrieval fails"""
    pass

class ValidationError(IBKRError):
    """Raised when input validation fails"""
    pass

class RateLimitError(IBKRError):
    """Raised when API rate limits are exceeded"""
    pass
```

## Phase 2: Data Layer Enhancement (Week 2-3)

### 2.1 Enhanced SQLite Storage (Preserving stream_to_sqlite functionality)
```python
# src/data/storage.py
import asyncio
import aiosqlite
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Optional
import pandas as pd
from datetime import datetime

class AsyncSQLiteStorage:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_path = config.path
        
    async def initialize(self):
        """Initialize database with optimized settings"""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrent access
            if self.config.enable_wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")
            
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            await db.execute("PRAGMA temp_store=memory")
            
            # Create enhanced tables
            await self._create_tables(db)
            await db.commit()
    
    async def _create_tables(self, db):
        """Create all required tables with proper indexing"""
        tables = {
            'bid_ask_data': '''
                CREATE TABLE IF NOT EXISTS bid_ask_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    bid_price REAL,
                    ask_price REAL,
                    bid_size INTEGER,
                    ask_size INTEGER,
                    spread REAL GENERATED ALWAYS AS (ask_price - bid_price) STORED,
                    mid_price REAL GENERATED ALWAYS AS ((bid_price + ask_price) / 2) STORED,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, symbol)
                )
            ''',
            'portfolio_snapshots': '''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    positions_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'performance_metrics': '''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    volatility REAL,
                    omega_ratio REAL,
                    cvar_ratio REAL,
                    cvar_dollar REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'orders': '''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    limit_price REAL,
                    stop_price REAL,
                    status TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    avg_fill_price REAL,
                    commission REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            '''
        }
        
        # Create tables
        for table_name, create_sql in tables.items():
            await db.execute(create_sql)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_bid_ask_symbol_time ON bid_ask_data(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_portfolio_time ON portfolio_snapshots(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_time ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)"
        ]
        
        for index_sql in indexes:
            await db.execute(index_sql)
    
    async def stream_to_sqlite(self, ticks: AsyncGenerator[TickType, None], 
                              batch_size: int = 1000, 
                              max_duration: int = 23400) -> int:
        """Enhanced version of legacy stream_to_sqlite with batching and async"""
        records_inserted = 0
        batch = []
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async for tick in ticks:
                    # Check time limit
                    if asyncio.get_event_loop().time() - start_time > max_duration:
                        break
                    
                    batch.append((
                        tick['timestamp'],
                        tick['symbol'],
                        tick['bid_price'],
                        tick['ask_price'],
                        tick['bid_size'],
                        tick['ask_size']
                    ))
                    
                    # Batch insert for performance
                    if len(batch) >= batch_size:
                        await self._insert_batch(db, batch)
                        records_inserted += len(batch)
                        batch.clear()
                
                # Insert remaining records
                if batch:
                    await self._insert_batch(db, batch)
                    records_inserted += len(batch)
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Error in stream_to_sqlite: {e}")
            raise DataError(f"Failed to stream data to SQLite: {e}")
        
        return records_inserted
    
    async def _insert_batch(self, db, batch):
        """Insert batch of tick data"""
        await db.executemany(
            '''INSERT OR REPLACE INTO bid_ask_data 
               (timestamp, symbol, bid_price, ask_price, bid_size, ask_size) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            batch
        )
    
    async def get_tick_data(self, symbol: str, 
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Retrieve tick data as pandas DataFrame"""
        query = """
            SELECT timestamp, symbol, bid_price, ask_price, bid_size, ask_size, 
                   spread, mid_price
            FROM bid_ask_data 
            WHERE symbol = ?
        """
        params = [symbol]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp"
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
        return pd.DataFrame([dict(row) for row in rows])
```

## Phase 3: Async API Layer (Week 3-4)

### 3.1 Async IBKR Client
```python
# src/api/client.py
import asyncio
from typing import Optional, Dict, Any, AsyncGenerator
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order

from ..core.config import IBConfig
from ..core.exceptions import ConnectionError, OrderError, DataError
from .wrapper import AsyncIBWrapper
from .connection import ConnectionManager
from .rate_limiter import RateLimiter

class AsyncIBClient:
    def __init__(self, config: IBConfig, wrapper: AsyncIBWrapper):
        self.config = config
        self.wrapper = wrapper
        self.connection_manager = ConnectionManager(config)
        self.rate_limiter = RateLimiter(config.rate_limit_per_second)
        self._client: Optional[EClient] = None
        self._request_counter = 0
    
    async def connect(self) -> bool:
        """Connect to IBKR with retry logic"""
        return await self.connection_manager.connect_with_retry()
    
    async def disconnect(self):
        """Gracefully disconnect"""
        if self._client and self._client.isConnected():
            self._client.disconnect()
    
    async def send_order(self, contract: Contract, order: Order) -> int:
        """Send order with async handling"""
        await self.rate_limiter.acquire()
        
        try:
            order_id = await self.wrapper.get_next_order_id()
            
            # Place order
            self._client.placeOrder(order_id, contract, order)
            
            # Wait for order acknowledgment or timeout
            result = await asyncio.wait_for(
                self.wrapper.wait_for_order_status(order_id),
                timeout=self.config.request_timeout
            )
            
            return order_id
            
        except asyncio.TimeoutError:
            raise OrderError(f"Order placement timed out")
        except Exception as e:
            raise OrderError(f"Failed to place order: {e}")
    
    async def get_market_data(self, contract: Contract, 
                             tick_types: Optional[List[int]] = None) -> Dict[int, float]:
        """Get real-time market data"""
        await self.rate_limiter.acquire()
        
        request_id = self._get_next_request_id()
        
        try:
            # Request market data
            self._client.reqMktData(
                reqId=request_id,
                contract=contract,
                genericTickList="",
                snapshot=True,
                regulatorySnapshot=False,
                mktDataOptions=[]
            )
            
            # Wait for data
            data = await asyncio.wait_for(
                self.wrapper.wait_for_market_data(request_id),
                timeout=self.config.request_timeout
            )
            
            return data
            
        except asyncio.TimeoutError:
            raise DataError(f"Market data request timed out for {contract.symbol}")
        finally:
            self._client.cancelMktData(request_id)
    
    async def get_streaming_data(self, contract: Contract) -> AsyncGenerator[TickType, None]:
        """Get streaming tick data"""
        request_id = self._get_next_request_id()
        
        try:
            # Start streaming
            self._client.reqTickByTickData(
                reqId=request_id,
                contract=contract,
                tickType="BidAsk",
                numberOfTicks=0,
                ignoreSize=False
            )
            
            # Yield streaming data
            async for tick in self.wrapper.stream_tick_data(request_id):
                yield {
                    'timestamp': tick.timestamp.isoformat(),
                    'symbol': contract.symbol,
                    'bid_price': tick.bid_price,
                    'ask_price': tick.ask_price,
                    'bid_size': tick.bid_size,
                    'ask_size': tick.ask_size
                }
                
        finally:
            self._client.cancelTickByTickData(request_id)
    
    # High-level order methods (preserving legacy functionality)
    async def order_target_percent(self, contract: Contract, 
                                  order_type: str, target: float, **kwargs) -> int:
        """Order to reach target percentage of portfolio"""
        # Get current portfolio value
        portfolio_value = await self._get_portfolio_value()
        
        # Calculate target value
        target_value = portfolio_value * target
        
        # Get current position
        current_position = await self._get_position(contract)
        current_value = current_position * await self._get_last_price(contract)
        
        # Calculate required trade value
        trade_value = target_value - current_value
        
        if abs(trade_value) < 100:  # Minimum trade threshold
            return None
        
        # Create and send order
        quantity = int(trade_value / await self._get_last_price(contract))
        action = OrderAction.BUY if quantity > 0 else OrderAction.SELL
        
        order = self._create_order(order_type, action, abs(quantity), **kwargs)
        return await self.send_order(contract, order)
    
    def _get_next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter
```

### 3.2 Connection Manager
```python
# src/api/connection.py
import asyncio
import logging
from typing import Optional
from ibapi.client import EClient
from ..core.config import IBConfig
from ..core.exceptions import ConnectionError

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self, config: IBConfig):
        self.config = config
        self.client: Optional[EClient] = None
        self.connected = False
        self.connection_event = asyncio.Event()
    
    async def connect_with_retry(self) -> bool:
        """Connect with exponential backoff retry"""
        for attempt in range(self.config.max_reconnect_attempts):
            try:
                await self._connect()
                
                # Wait for connection to be established
                await asyncio.wait_for(
                    self.connection_event.wait(),
                    timeout=self.config.connection_timeout
                )
                
                self.connected = True
                logger.info(f"Connected to IBKR on attempt {attempt + 1}")
                return True
                
            except (ConnectionError, asyncio.TimeoutError) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.max_reconnect_attempts - 1:
                    delay = self.config.reconnect_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        logger.error("Failed to connect after maximum attempts")
        return False
    
    async def _connect(self):
        """Single connection attempt"""
        try:
            self.client.connect(
                self.config.host,
                self.config.port,
                self.config.client_id
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect: {e}")
    
    def on_connected(self):
        """Called when connection is established"""
        self.connection_event.set()
    
    def on_disconnected(self):
        """Called when connection is lost"""
        self.connected = False
        self.connection_event.clear()
```

## Phase 4: Enhanced Portfolio Analytics (Week 4-5)

### 4.1 Performance Metrics (Preserving Legacy Properties)
```python
# src/portfolio/analytics.py
import pandas as pd
import empyrical as ep
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from ..core.types import PerformanceMetrics
from ..core.config import TradingConfig

@dataclass
class PortfolioAnalytics:
    config: TradingConfig
    returns: Optional[pd.Series] = None
    
    def update_returns(self, returns: pd.Series):
        """Update portfolio returns series"""
        self.returns = returns
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio (preserving legacy functionality)"""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        return ep.sharpe_ratio(
            self.returns,
            risk_free=self.config.risk_free_rate,
            annualization=252
        )
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown (preserving legacy functionality)"""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        return ep.max_drawdown(self.returns)
    
    @property
    def volatility(self) -> float:
        """Calculate annualized volatility (preserving legacy functionality)"""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        return self.returns.std(ddof=1) * np.sqrt(252)
    
    @property
    def omega_ratio(self) -> float:
        """Calculate omega ratio (preserving legacy functionality)"""
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        
        return ep.omega_ratio(self.returns, risk_free=self.config.risk_free_rate)
    
    @property
    def cvar(self) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (preserving legacy functionality)"""
        if self.returns is None or len(self.returns) < 2:
            return (0.0, 0.0)
        
        cvar_ratio = ep.conditional_value_at_risk(self.returns, cutoff=0.05)
        
        # Need portfolio value to calculate dollar amount
        # This would be injected from portfolio manager
        portfolio_value = self._get_current_portfolio_value()
        cvar_dollar = cvar_ratio * portfolio_value if portfolio_value else 0.0
        
        return (cvar_ratio, cvar_dollar)
    
    def get_all_metrics(self) -> PerformanceMetrics:
        """Get all performance metrics as typed dict"""
        return PerformanceMetrics(
            sharpe_ratio=self.sharpe_ratio,
            max_drawdown=self.max_drawdown,
            volatility=self.volatility,
            omega_ratio=self.omega_ratio,
            cvar=self.cvar
        )
    
    @property
    def cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns (preserving legacy functionality)"""
        if self.returns is None:
            return pd.Series(dtype=float)
        
        return ep.cum_returns(self.returns, starting_value=1.0)
    
    async def calculate_benchmark_metrics(self, benchmark_symbol: str = None) -> dict:
        """Calculate metrics relative to benchmark"""
        if not benchmark_symbol:
            benchmark_symbol = self.config.benchmark_symbol
        
        # This would integrate with the data client to get benchmark returns
        # Implementation would fetch benchmark data and calculate relative metrics
        return {
            'alpha': 0.0,  # To be implemented
            'beta': 1.0,   # To be implemented
            'information_ratio': 0.0,  # To be implemented
            'tracking_error': 0.0      # To be implemented
        }
    
    def _get_current_portfolio_value(self) -> Optional[float]:
        """Get current portfolio value - to be implemented by portfolio manager"""
        # This is a placeholder that would be injected by the portfolio manager
        return None
```

### 4.2 Enhanced Contract Factory
```python
# src/contracts/factory.py
from typing import Optional, Dict, Any
from ibapi.contract import Contract
from ..core.types import SecurityType
from ..core.exceptions import ValidationError

class EnhancedContractFactory:
    """Enhanced contract factory combining legacy simplicity with IBKR sophistication"""
    
    # Legacy-style simple methods (preserving existing API)
    @staticmethod
    def stock(symbol: str, exchange: str = "SMART", currency: str = "USD",
              primary_exchange: Optional[str] = None) -> Contract:
        """Create stock contract (enhanced legacy method)"""
        if not symbol or not symbol.strip():
            raise ValidationError("Symbol cannot be empty")
        
        contract = Contract()
        contract.symbol = symbol.upper().strip()
        contract.secType = SecurityType.STOCK.value
        contract.exchange = exchange
        contract.currency = currency.upper()
        
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        
        return contract
    
    @staticmethod
    def option(symbol: str, exchange: str, contract_month: str,
               strike: float, right: str, multiplier: str = "100") -> Contract:
        """Create option contract (enhanced legacy method)"""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.OPTION.value
        contract.exchange = exchange
        contract.lastTradeDateOrContractMonth = contract_month
        contract.strike = strike
        contract.right = right.upper()
        contract.multiplier = multiplier
        
        return contract
    
    @staticmethod
    def future(symbol: str, exchange: str, contract_month: str,
               multiplier: Optional[str] = None) -> Contract:
        """Create future contract (enhanced legacy method)"""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.FUTURE.value
        contract.exchange = exchange
        contract.lastTradeDateOrContractMonth = contract_month
        
        if multiplier:
            contract.multiplier = multiplier
        
        return contract
    
    # Advanced methods leveraging IBKR samples
    @staticmethod
    def crypto(symbol: str, exchange: str = "PAXOS", currency: str = "USD") -> Contract:
        """Create cryptocurrency contract"""
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = SecurityType.CRYPTO.value
        contract.exchange = exchange
        contract.currency = currency.upper()
        
        return contract
    
    @staticmethod
    def by_contract_id(con_id: int, exchange: str = "SMART") -> Contract:
        """Create contract by contract ID"""
        contract = Contract()
        contract.conId = con_id
        contract.exchange = exchange
        
        return contract
    
    @staticmethod
    def by_isin(isin: str, exchange: str = "SMART", 
                currency: str = "USD", sec_type: str = "STK") -> Contract:
        """Create contract by ISIN"""
        contract = Contract()
        contract.secIdType = "ISIN"
        contract.secId = isin
        contract.exchange = exchange
        contract.currency = currency
        contract.secType = sec_type
        
        return contract
    
    # Validation and utility methods
    @staticmethod
    def validate_contract(contract: Contract) -> bool:
        """Validate contract has required fields"""
        required_fields = ['secType']
        
        for field in required_fields:
            if not hasattr(contract, field) or not getattr(contract, field):
                return False
        
        # Specific validation based on security type
        if contract.secType == SecurityType.STOCK.value:
            return bool(contract.symbol and contract.exchange)
        elif contract.secType == SecurityType.OPTION.value:
            return bool(contract.symbol and contract.exchange and 
                       contract.lastTradeDateOrContractMonth and 
                       contract.strike and contract.right)
        
        return True
```

## Phase 5: Integration & Testing (Week 5-6)

### 5.1 Main Application
```python
# src/app/application.py
import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from ..core.config import Config
from ..data.storage import AsyncSQLiteStorage
from ..api.client import AsyncIBClient
from ..api.wrapper import AsyncIBWrapper
from ..portfolio.analytics import PortfolioAnalytics
from ..portfolio.manager import PortfolioManager
from ..streaming.tick_storage import TickStorageManager

logger = logging.getLogger(__name__)

class TradingApplication:
    def __init__(self, config: Config):
        self.config = config
        self.storage = AsyncSQLiteStorage(config.database)
        self.wrapper = AsyncIBWrapper()
        self.client = AsyncIBClient(config.ibkr, self.wrapper)
        self.analytics = PortfolioAnalytics(config.trading)
        self.portfolio_manager = PortfolioManager(self.client, self.analytics)
        self.tick_storage = TickStorageManager(self.storage)
        
        self._running = False
    
    async def startup(self):
        """Initialize application"""
        logger.info("Starting trading application...")
        
        # Initialize database
        await self.storage.initialize()
        
        # Connect to IBKR
        connected = await self.client.connect()
        if not connected:
            raise ConnectionError("Failed to connect to IBKR")
        
        # Start background tasks
        self._running = True
        asyncio.create_task(self._performance_update_loop())
        
        logger.info("Trading application started successfully")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down trading application...")
        
        self._running = False
        await self.client.disconnect()
        
        logger.info("Trading application shut down successfully")
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for application lifecycle"""
        try:
            await self.startup()
            yield self
        finally:
            await self.shutdown()
    
    # Preserve legacy interface methods
    async def order_target_percent(self, symbol: str, target: float, 
                                  order_type: str = "MKT", **kwargs) -> Optional[int]:
        """Legacy-compatible order method"""
        contract = ContractFactory.stock(symbol)
        return await self.client.order_target_percent(contract, order_type, target, **kwargs)
    
    async def stream_to_sqlite(self, symbol: str, duration: int = 23400) -> int:
        """Enhanced version of legacy stream_to_sqlite"""
        contract = ContractFactory.stock(symbol)
        tick_stream = self.client.get_streaming_data(contract)
        
        return await self.storage.stream_to_sqlite(
            tick_stream, 
            max_duration=duration
        )
    
    # Performance metrics (preserving legacy properties)
    @property
    def sharpe_ratio(self) -> float:
        return self.analytics.sharpe_ratio
    
    @property
    def max_drawdown(self) -> float:
        return self.analytics.max_drawdown
    
    @property
    def volatility(self) -> float:
        return self.analytics.volatility
    
    @property
    def omega_ratio(self) -> float:
        return self.analytics.omega_ratio
    
    @property
    def cvar(self) -> tuple[float, float]:
        return self.analytics.cvar
    
    async def _performance_update_loop(self):
        """Background task to update performance metrics"""
        while self._running:
            try:
                # Update portfolio returns
                returns = await self.portfolio_manager.get_returns()
                self.analytics.update_returns(returns)
                
                # Store metrics to database
                metrics = self.analytics.get_all_metrics()
                await self._store_performance_metrics(metrics)
                
            except Exception as e:
                logger.error(f"Error updating performance metrics: {e}")
            
            await asyncio.sleep(self.config.trading.performance_calculation_interval)
```

## Additional Recommendations

### 1. **Rate Limiting & Circuit Breaker**
```python
# src/api/rate_limiter.py
import asyncio
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = "HALF_OPEN"
            else:
                raise RateLimitError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
```

### 2. **Monitoring & Observability**
```python
# src/monitoring/metrics.py
import time
from contextlib import contextmanager
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, list] = {}
    
    def increment(self, name: str, value: int = 1):
        self.counters[name] = self.counters.get(name, 0) + value
    
    @contextmanager
    def time_operation(self, name: str):
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if name not in self.timers:
                self.timers[name] = []
            self.timers[name].append(duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'counters': self.counters,
            'timers': {
                name: {
                    'count': len(times),
                    'avg': sum(times) / len(times) if times else 0,
                    'max': max(times) if times else 0,
                    'min': min(times) if times else 0
                }
                for name, times in self.timers.items()
            }
        }
```

### 3. **Configuration via Environment Variables**
Create a `.env.example` file:
```bash
# IBKR Connection
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1
IB_ACCOUNT=DU1234567

# Database
DB_PATH=./data/trading.db

# Trading Parameters
MAX_POSITION_SIZE=0.10
MAX_DAILY_TRADES=100
RISK_FREE_RATE=0.05
BENCHMARK_SYMBOL=SPY

# Logging
LOG_LEVEL=INFO
```

### 4. **Docker Support**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY config/ config/

CMD ["python", "-m", "src.app.cli"]
```

### 5. **CLI Interface**
```python
# src/app/cli.py
import asyncio
import click
from .application import TradingApplication
from ..core.config import Config, Environment

@click.group()
@click.option('--env', default='development', 
              type=click.Choice(['development', 'testing', 'production']))
@click.pass_context
def cli(ctx, env):
    ctx.ensure_object(dict)
    ctx.obj['config'] = Config(environment=Environment(env))

@cli.command()
@click.option('--symbol', required=True, help='Stock symbol to trade')
@click.option('--target', required=True, type=float, help='Target percentage')
@click.pass_context
async def order_target_percent(ctx, symbol, target):
    config = ctx.obj['config']
    async with TradingApplication(config).lifespan() as app:
        order_id = await app.order_target_percent(symbol, target)
        click.echo(f"Order placed: {order_id}")

if __name__ == '__main__':
    cli()
```

## Security & Deployment

### 1. **Secrets Management**
- Use environment variables for all sensitive data
- Consider HashiCorp Vault or AWS Secrets Manager for production
- Implement credential rotation

### 2. **Monitoring**
- Application metrics (Prometheus/Grafana)
- Error tracking (Sentry)
- Performance monitoring (APM)
- Health checks and alerts

### 3. **Testing Strategy**
- Unit tests with pytest and async support
- Integration tests with mock IBKR API
- Performance tests for database operations
- End-to-end tests for trading workflows

This refactoring plan maintains all legacy functionality while modernizing the architecture and adding sophisticated IBKR API capabilities. The modular structure allows for incremental implementation and easy testing.
