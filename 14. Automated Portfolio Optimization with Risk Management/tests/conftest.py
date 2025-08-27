"""
Pytest configuration and shared fixtures for the test suite.
"""

import asyncio
import pytest
import pytest_asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd
import numpy as np

from src.core.config import IBConfig, DatabaseConfig
from src.core.types import PositionData, AccountValue, MarketDataType
from src.data.storage import AsyncDataStorage
from src.analytics.portfolio import PortfolioAnalytics
from src.contracts.factory import ContractFactory
from ibapi.contract import Contract


# Test markers
pytestmark = pytest.mark.asyncio


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration."""
    db_path = temp_dir / "test_database.sqlite"
    
    config = IBConfig(
        host="127.0.0.1",
        port=7497,
        client_id=999,  # Use a test client ID
        connection_timeout=5,
        request_timeout=5,
        max_reconnect_attempts=2,
        reconnect_delay=1.0,
        database=DatabaseConfig(
            path=str(db_path),
            pool_size=2
        ),
        risk_free_rate=0.02
    )
    return config


@pytest_asyncio.fixture
async def test_storage(test_config):
    """Create a test database storage instance."""
    storage = AsyncDataStorage(test_config.database)
    await storage.initialize()
    yield storage
    await storage.close()


@pytest.fixture
def mock_ibkr_client():
    """Create a mock IBKR client for testing."""
    client = AsyncMock()
    client.is_connected = True
    client.connection_info = {
        'state': 'connected',
        'connected': True,
        'host': '127.0.0.1',
        'port': 7497,
        'client_id': 999,
        'uptime_seconds': 100
    }
    return client


@pytest.fixture
def sample_positions():
    """Create sample position data for testing."""
    return [
        PositionData(
            account="DU123456",
            symbol="AAPL",
            sec_type="STK",
            exchange="NASDAQ",
            currency="USD",
            position=100.0,
            avg_cost=150.0,
            timestamp=datetime.now()
        ),
        PositionData(
            account="DU123456",
            symbol="GOOGL",
            sec_type="STK",
            exchange="NASDAQ",
            currency="USD",
            position=50.0,
            avg_cost=2500.0,
            timestamp=datetime.now()
        ),
        PositionData(
            account="DU123456",
            symbol="TSLA",
            sec_type="STK",
            exchange="NASDAQ",
            currency="USD",
            position=-25.0,  # Short position
            avg_cost=800.0,
            timestamp=datetime.now()
        )
    ]


# Additional position data for the new test methods
@pytest.fixture
def position_data_for_orders():
    """Create position data specifically for order manager tests."""
    return [
        PositionData(
            account="DU123456",
            symbol="AAPL",
            sec_type="STK",
            exchange="NASDAQ",
            currency="USD",
            position=100,
            avg_cost=145.0,
            timestamp=datetime.now()
        ),
        PositionData(
            account="DU123456",
            symbol="GOOGL",
            sec_type="STK",
            exchange="NASDAQ",
            currency="USD",
            position=25,
            avg_cost=2300.0,
            timestamp=datetime.now()
        )
    ]


@pytest.fixture
def sample_account_values():
    """Create sample account values for testing."""
    return [
        AccountValue(
            key="NetLiquidation",
            value="100000.00",
            currency="USD",
            account="DU123456",
            timestamp=datetime.now()
        ),
        AccountValue(
            key="TotalCashValue",
            value="50000.00",
            currency="USD",
            account="DU123456",
            timestamp=datetime.now()
        ),
        AccountValue(
            key="BuyingPower",
            value="200000.00",
            currency="USD",
            account="DU123456",
            timestamp=datetime.now()
        ),
        AccountValue(
            key="GrossPositionValue",
            value="50000.00",
            currency="USD",
            account="DU123456",
            timestamp=datetime.now()
        )
    ]


@pytest.fixture
def sample_returns():
    """Create sample return series for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic daily returns
    n_days = 252  # One year of trading days
    returns = np.random.normal(0.0008, 0.02, n_days)  # ~20% annual return, ~20% volatility
    
    # Add some realistic patterns
    returns[50:60] = -0.05  # Market crash period
    returns[200:210] = 0.03  # Bull run period
    
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=n_days),
        periods=n_days,
        freq='D'
    )
    
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_prices(sample_returns):
    """Create sample price series from returns."""
    initial_price = 100.0
    prices = initial_price * (1 + sample_returns).cumprod()
    return prices


@pytest.fixture
def portfolio_analytics():
    """Create a portfolio analytics instance for testing."""
    return PortfolioAnalytics(
        risk_free_rate=0.02,
        window_size=21  # Smaller window for faster tests
    )


@pytest.fixture
def sample_contracts():
    """Create sample IBKR contracts for testing."""
    return {
        'stock': ContractFactory.create_stock('AAPL', 'SMART', 'USD'),
        'option': ContractFactory.create_option('AAPL', '20231215', 150.0, 'C'),
        'future': ContractFactory.create_future('ES', '202312', 'CME'),
        'forex': ContractFactory.create_forex('EUR', 'USD')
    }


@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    return [
        {
            'request_id': 1,
            'tick_type': 1,  # Bid price
            'price': 150.25,
            'timestamp': datetime.now(),
            'data_type': 'price'
        },
        {
            'request_id': 1,
            'tick_type': 2,  # Ask price
            'price': 150.27,
            'timestamp': datetime.now(),
            'data_type': 'price'
        },
        {
            'request_id': 1,
            'tick_type': 0,  # Bid size
            'size': 100,
            'timestamp': datetime.now(),
            'data_type': 'size'
        }
    ]


# Utility functions for tests

def assert_almost_equal(actual, expected, tolerance=1e-6):
    """Assert that two float values are approximately equal."""
    assert abs(actual - expected) < tolerance, f"Expected {expected}, got {actual}"


def create_mock_async_generator(data_list):
    """Create a mock async generator for testing streaming data."""
    async def mock_generator():
        for item in data_list:
            yield item
    return mock_generator()


class MockIBWrapper:
    """Mock IBKR wrapper for testing."""
    
    def __init__(self):
        self.market_data_queue = asyncio.Queue()
        self.order_status_queue = asyncio.Queue()
        self.position_queue = asyncio.Queue()
        self.account_queue = asyncio.Queue()
        self.error_queue = asyncio.Queue()
        self.streaming_subscriptions = {}
    
    async def add_market_data(self, data):
        """Add market data to the queue."""
        await self.market_data_queue.put(data)
    
    async def add_position_data(self, data):
        """Add position data to the queue."""
        await self.position_queue.put(data)
    
    async def add_account_data(self, data):
        """Add account data to the queue."""
        await self.account_queue.put(data)


@pytest.fixture
def mock_ib_wrapper():
    """Create a mock IB wrapper for testing."""
    return MockIBWrapper()


# Integration test helpers

@pytest.fixture
def skip_if_no_ibkr():
    """Skip test if IBKR is not available."""
    def _skip_if_no_ibkr():
        # This would check if IBKR connection is available
        # For now, we'll always skip integration tests in CI
        import os
        if os.getenv('CI') or os.getenv('GITHUB_ACTIONS'):
            pytest.skip("IBKR integration tests skipped in CI environment")
    
    return _skip_if_no_ibkr


# Performance testing utilities

@pytest.fixture
def benchmark_timer():
    """Utility for benchmarking test performance."""
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = datetime.now()
        
        def stop(self):
            self.end_time = datetime.now()
            return self.elapsed
        
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return None
    
    return Timer()


# Mock external dependencies

@pytest.fixture
def mock_ibapi():
    """Mock the entire ibapi package."""
    with patch('src.api.client.EClient'), \
         patch('src.api.client.EWrapper'), \
         patch('ibapi.contract.Contract'), \
         patch('ibapi.order.Order'):
        yield


# Database fixtures for specific test scenarios

@pytest_asyncio.fixture
async def populated_test_storage(test_storage, sample_returns, sample_positions):
    """Create a test storage with sample data."""
    # Add historical returns
    for date, return_val in sample_returns.items():
        await test_storage.store_portfolio_value(
            timestamp=date,
            total_value=100000 * (1 + return_val),
            positions=[]
        )
    
    # Add market data
    for i, (date, _) in enumerate(sample_returns.items()):
        await test_storage.store_market_data(
            symbol="AAPL",
            timestamp=date,
            price=150.0 + i * 0.1,
            size=100,
            tick_type=1
        )
    
    return test_storage
