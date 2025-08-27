"""
Unit and integration tests for the async data storage module.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.storage import AsyncDataStorage
from src.core.types import PositionData


class TestAsyncDataStorage:
    """Test AsyncDataStorage initialization and basic operations."""
    
    @pytest.mark.asyncio
    async def test_initialize_database(self, test_storage, temp_dir):
        """Test database initialization."""
        # Database should be created and tables should exist
        db_path = Path(test_storage.config.path)
        assert db_path.exists()
        
        # Test that we can connect and tables are created
        async with test_storage.get_connection() as conn:
            # Check that tables exist
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in await cursor.fetchall()]
            
        expected_tables = [
            'bid_ask_data', 'portfolio_snapshots', 'positions'
        ]
        for table in expected_tables:
            assert table in tables
    
    @pytest.mark.asyncio
    async def test_connection_pool(self, test_storage):
        """Test connection pooling functionality."""
        # Test multiple concurrent connections
        async def test_connection():
            async with test_storage.get_connection() as conn:
                cursor = await conn.execute("SELECT 1")
                result = await cursor.fetchone()
                return result[0]
        
        # Run multiple concurrent operations
        tasks = [test_connection() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should return 1
        assert all(result == 1 for result in results)


class TestMarketDataStorage:
    """Test market data storage functionality."""
    
    @pytest.mark.asyncio
    async def test_store_market_data_basic(self, test_storage):
        """Test basic market data storage."""
        timestamp = datetime.now()
        
        await test_storage.store_market_data(
            symbol="AAPL",
            timestamp=timestamp,
            price=150.25,
            size=100,
            tick_type=1
        )
        
        # Verify data was stored
        async with test_storage.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM bid_ask_data WHERE symbol = ?",
                ("AAPL",)
            )
            row = await cursor.fetchone()
            
            assert row is not None
            assert row[2] == "AAPL"  # symbol
            assert abs(row[3] - 150.25) < 1e-6  # bid_price
            assert row[5] == 100  # bid_size
    
    @pytest.mark.asyncio
    async def test_store_market_data_batch(self, test_storage):
        """Test storing multiple market data points."""
        base_time = datetime.now()
        
        # Store multiple data points
        for i in range(10):
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                price=150.0 + i * 0.1,
                size=100 + i,
                tick_type=1
            )
        
        # Verify all data was stored
        async with test_storage.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM bid_ask_data WHERE symbol = ?",
                ("AAPL",)
            )
            count = (await cursor.fetchone())[0]
            assert count == 10
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, test_storage):
        """Test retrieving market data."""
        base_time = datetime.now()
        
        # Store test data
        test_data = []
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            price = 150.0 + i * 0.5
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp=timestamp,
                price=price,
                size=100,
                tick_type=1
            )
            test_data.append((timestamp, price))
        
        # Retrieve data
        retrieved_data = await test_storage.get_market_data(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=10)
        )
        
        assert len(retrieved_data) == 5
        
        # Verify data ordering (should be chronological)
        for i in range(len(retrieved_data) - 1):
            assert retrieved_data[i][0] <= retrieved_data[i + 1][0]  # timestamp ordering
    
    @pytest.mark.asyncio
    async def test_get_market_data_with_limit(self, test_storage):
        """Test retrieving market data with limit."""
        base_time = datetime.now()
        
        # Store 10 data points
        for i in range(10):
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                price=150.0 + i * 0.1,
                size=100,
                tick_type=1
            )
        
        # Retrieve with limit
        retrieved_data = await test_storage.get_market_data(
            symbol="AAPL",
            limit=5
        )
        
        assert len(retrieved_data) <= 5
    
    @pytest.mark.asyncio
    async def test_get_latest_price(self, test_storage):
        """Test getting latest price for a symbol."""
        base_time = datetime.now()
        latest_price = 155.75
        
        # Store multiple prices with latest being the highest timestamp
        prices = [150.0, 152.5, 154.0, latest_price]
        for i, price in enumerate(prices):
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp=base_time + timedelta(minutes=i),
                price=price,
                size=100,
                tick_type=1
            )
        
        # Get latest price
        result = await test_storage.get_latest_price("AAPL")
        
        assert result is not None
        assert abs(result - latest_price) < 1e-6


class TestPortfolioDataStorage:
    """Test portfolio data storage functionality."""
    
    @pytest.mark.asyncio
    async def test_store_portfolio_value(self, test_storage, sample_positions):
        """Test storing portfolio value."""
        timestamp = datetime.now()
        total_value = 100000.0
        
        await test_storage.store_portfolio_value(
            timestamp=timestamp,
            total_value=total_value,
            positions=sample_positions
        )
        
        # Verify portfolio value was stored
        async with test_storage.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            
            assert row is not None
            assert abs(row[3] - total_value) < 1e-6  # total_value is at index 3
    
    @pytest.mark.asyncio
    async def test_store_portfolio_value_with_positions(self, test_storage, sample_positions):
        """Test storing portfolio value with position history."""
        timestamp = datetime.now()
        
        await test_storage.store_portfolio_value(
            timestamp=timestamp,
            total_value=100000.0,
            positions=sample_positions
        )
        
        # Verify positions were stored in history
        async with test_storage.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM positions WHERE timestamp = ?",
                (timestamp.isoformat(),)
            )
            count = (await cursor.fetchone())[0]
            assert count == len(sample_positions)
    
    @pytest.mark.asyncio
    async def test_get_portfolio_history(self, test_storage):
        """Test retrieving portfolio history."""
        base_time = datetime.now()
        
        # Store historical portfolio values
        historical_data = []
        for i in range(10):
            timestamp = base_time + timedelta(days=i)
            value = 100000 + i * 1000
            await test_storage.store_portfolio_value(
                timestamp=timestamp,
                total_value=value,
                positions=[]
            )
            historical_data.append((timestamp, value))
        
        # Retrieve history
        history = await test_storage.get_portfolio_history(
            start_time=base_time,
            end_time=base_time + timedelta(days=15),
            limit=5
        )
        
        assert len(history) <= 5
        
        # Verify data integrity
        for timestamp, value in history:
            assert isinstance(timestamp, datetime)
            assert isinstance(value, float)
    
    @pytest.mark.asyncio
    async def test_get_portfolio_returns(self, test_storage):
        """Test calculating portfolio returns from stored data."""
        base_time = datetime.now().replace(microsecond=0)  # Remove microseconds for consistent results
        
        # Store portfolio values that create known returns
        values = [100000, 101000, 99500, 102000, 103000]
        for i, value in enumerate(values):
            # Use smaller time intervals to ensure all records are captured
            timestamp = base_time + timedelta(hours=i)  # Spread them out by 1 hour each
            await test_storage.store_portfolio_value(
                timestamp=timestamp,
                total_value=value,
                positions=[]
            )
        
        # Get returns
        returns = await test_storage.get_portfolio_returns(days=10)
        
        assert returns is not None
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(values) - 1  # Returns are n-1 of values
        
        # Verify return calculations
        expected_first_return = (values[1] / values[0]) - 1
        assert abs(returns.iloc[0] - expected_first_return) < 1e-6
    
    @pytest.mark.asyncio
    async def test_get_portfolio_returns_insufficient_data(self, test_storage):
        """Test portfolio returns with insufficient data."""
        # Store only one data point
        await test_storage.store_portfolio_value(
            timestamp=datetime.now(),
            total_value=100000.0,
            positions=[]
        )
        
        # Should return None for insufficient data
        returns = await test_storage.get_portfolio_returns(days=30)
        assert returns is None


class TestDataCleanup:
    """Test data cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, test_storage):
        """Test cleaning up old data."""
        current_time = datetime.now()
        
        # Store data with various ages
        old_data_time = current_time - timedelta(days=100)
        recent_data_time = current_time - timedelta(days=10)
        
        # Store old market data
        await test_storage.store_market_data(
            symbol="AAPL",
            timestamp=old_data_time,
            price=100.0,
            size=100,
            tick_type=1
        )
        
        # Store recent market data
        await test_storage.store_market_data(
            symbol="AAPL",
            timestamp=recent_data_time,
            price=150.0,
            size=100,
            tick_type=1
        )
        
        # Store old portfolio value
        await test_storage.store_portfolio_value(
            timestamp=old_data_time,
            total_value=50000.0,
            positions=[]
        )
        
        # Store recent portfolio value
        await test_storage.store_portfolio_value(
            timestamp=recent_data_time,
            total_value=100000.0,
            positions=[]
        )
        
        # Perform cleanup (keep 30 days)
        deleted_count = await test_storage.cleanup_old_data(days_to_keep=30)
        
        assert deleted_count > 0
        
        # Verify old data was deleted and recent data remains
        async with test_storage.get_connection() as conn:
            # Check market data
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM bid_ask_data"
            )
            market_data_count = (await cursor.fetchone())[0]
            
            # Check portfolio values
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM portfolio_snapshots"
            )
            portfolio_count = (await cursor.fetchone())[0]
            
            # Should have at least the recent data
            assert market_data_count >= 1
            assert portfolio_count >= 1


class TestConcurrentOperations:
    """Test concurrent database operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_market_data_storage(self, test_storage):
        """Test storing market data concurrently."""
        base_time = datetime.now()
        
        async def store_data_batch(symbol, start_index, count):
            for i in range(count):
                await test_storage.store_market_data(
                    symbol=symbol,
                    timestamp=base_time + timedelta(seconds=start_index + i),
                    price=150.0 + i * 0.1,
                    size=100 + i,
                    tick_type=1
                )
        
        # Run concurrent storage operations
        tasks = [
            store_data_batch("AAPL", 0, 10),
            store_data_batch("GOOGL", 100, 10),
            store_data_batch("MSFT", 200, 10),
        ]
        
        await asyncio.gather(*tasks)
        
        # Verify all data was stored correctly
        async with test_storage.get_connection() as conn:
            cursor = await conn.execute(
                "SELECT symbol, COUNT(*) FROM bid_ask_data GROUP BY symbol"
            )
            results = await cursor.fetchall()
            
            symbol_counts = dict(results)
            assert symbol_counts.get("AAPL", 0) == 10
            assert symbol_counts.get("GOOGL", 0) == 10
            assert symbol_counts.get("MSFT", 0) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_read_write(self, test_storage):
        """Test concurrent read and write operations."""
        # Store initial data
        base_time = datetime.now()
        for i in range(5):
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp=base_time + timedelta(seconds=i),
                price=150.0 + i,
                size=100,
                tick_type=1
            )
        
        async def write_data():
            for i in range(5, 10):
                await test_storage.store_market_data(
                    symbol="AAPL",
                    timestamp=base_time + timedelta(seconds=i),
                    price=150.0 + i,
                    size=100,
                    tick_type=1
                )
        
        async def read_data():
            return await test_storage.get_market_data(
                symbol="AAPL",
                start_time=base_time,
                end_time=base_time + timedelta(seconds=15)
            )
        
        # Run concurrent read and write
        write_task = asyncio.create_task(write_data())
        read_task = asyncio.create_task(read_data())
        
        await asyncio.gather(write_task, read_task)
        
        # Final verification
        final_data = await test_storage.get_market_data(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(seconds=15)
        )
        
        assert len(final_data) == 10


class TestLegacyCompatibility:
    """Test legacy functionality compatibility."""
    
    @pytest.mark.asyncio
    async def test_stream_to_sqlite_equivalent(self, test_storage):
        """Test that async storage provides equivalent functionality to legacy stream_to_sqlite."""
        # Simulate streaming data like the legacy implementation
        symbol = "AAPL"
        base_time = datetime.now()
        
        # Stream multiple data points (like legacy streaming)
        streaming_data = []
        for i in range(100):
            timestamp = base_time + timedelta(milliseconds=i * 100)
            price = 150.0 + (i % 10) * 0.1
            size = 100 + (i % 5) * 10
            
            await test_storage.store_market_data(
                symbol=symbol,
                timestamp=timestamp,
                price=price,
                size=size,
                tick_type=1
            )
            streaming_data.append((timestamp, price, size))
        
        # Verify data integrity (equivalent to legacy verification)
        stored_data = await test_storage.get_market_data(
            symbol=symbol,
            start_time=base_time,
            end_time=base_time + timedelta(seconds=20)
        )
        
        assert len(stored_data) == 100
        
        # Verify data ordering and content
        for i, (timestamp, price, size, tick_type) in enumerate(stored_data):
            expected_timestamp, expected_price, expected_size = streaming_data[i]
            assert timestamp == expected_timestamp
            assert abs(price - expected_price) < 1e-6
            assert size == expected_size


class TestErrorHandling:
    """Test error handling in database operations."""
    
    @pytest.mark.asyncio
    async def test_invalid_data_handling(self, test_storage):
        """Test handling of invalid data."""
        # Test with invalid timestamp
        with pytest.raises((ValueError, TypeError)):
            await test_storage.store_market_data(
                symbol="AAPL",
                timestamp="invalid_timestamp",  # Should be datetime
                price=150.0,
                size=100,
                tick_type=1
            )
    
    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self, temp_dir):
        """Test handling of database connection errors."""
        from src.core.config import DatabaseConfig
        
        # Use invalid database path
        invalid_config = DatabaseConfig(path="/invalid/path/database.sqlite")
        
        # Should handle initialization errors gracefully
        storage = AsyncDataStorage(invalid_config)
        
        with pytest.raises(Exception):  # Should raise some kind of database error
            await storage.initialize()
    
    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, test_storage):
        """Test behavior when connection pool is exhausted."""
        # This test depends on the pool size being small
        # We'll create more concurrent operations than pool size
        
        async def long_running_operation():
            async with test_storage.get_connection() as conn:
                # Simulate a long-running operation
                await asyncio.sleep(0.1)
                cursor = await conn.execute("SELECT 1")
                return await cursor.fetchone()
        
        # Create more tasks than pool size (pool size is 2 in test config)
        tasks = [long_running_operation() for _ in range(5)]
        
        # Should complete without errors (might take longer due to queuing)
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        assert all(result[0] == 1 for result in results)


class TestStorageIntegration:
    """Integration tests for storage functionality."""
    
    @pytest.mark.asyncio
    async def test_full_trading_day_simulation(self, test_storage):
        """Test storage with a full trading day simulation."""
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        base_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)  # Market open
        
        # Simulate 6.5 hours of trading (390 minutes)
        total_minutes = 390
        
        # Store market data for multiple symbols
        for minute in range(0, total_minutes, 5):  # Every 5 minutes
            current_time = base_time + timedelta(minutes=minute)
            
            for symbol in symbols:
                # Simulate price movement
                base_price = {"AAPL": 150, "GOOGL": 2500, "MSFT": 300, "TSLA": 800}[symbol]
                price_change = np.random.normal(0, base_price * 0.001)  # Small random change
                price = base_price + price_change
                
                await test_storage.store_market_data(
                    symbol=symbol,
                    timestamp=current_time,
                    price=price,
                    size=np.random.randint(100, 1000),
                    tick_type=1
                )
        
        # Store portfolio snapshots every hour
        for hour in range(7):  # 7 snapshots during the day
            snapshot_time = base_time + timedelta(hours=hour)
            portfolio_value = 100000 + np.random.normal(0, 1000)  # Random portfolio changes
            
            await test_storage.store_portfolio_value(
                timestamp=snapshot_time,
                total_value=portfolio_value,
                positions=[]
            )
        
        # Verify data integrity
        market_data_end = base_time + timedelta(minutes=total_minutes)
        
        for symbol in symbols:
            data = await test_storage.get_market_data(
                symbol=symbol,
                start_time=base_time,
                end_time=market_data_end
            )
            
            # Should have approximately total_minutes/5 data points per symbol
            expected_points = total_minutes // 5
            assert len(data) >= expected_points - 5  # Allow some tolerance
        
        # Verify portfolio history
        portfolio_history = await test_storage.get_portfolio_history(
            start_time=base_time,
            end_time=base_time + timedelta(hours=8)
        )
        
        assert len(portfolio_history) >= 7
        
        # Calculate and verify returns
        returns = await test_storage.get_portfolio_returns(days=1)
        assert returns is not None
        assert len(returns) >= 6  # Should have returns between snapshots
