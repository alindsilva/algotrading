"""
Unit tests for the IBKR API client.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime

from src.api.client import IBClient
from src.core.config import IBConfig
from src.core.exceptions import ConnectionError, ValidationError


class TestIBClientInitialization:
    """Test IBClient initialization."""
    
    def test_init_with_config(self, test_config):
        """Test client initialization with config."""
        client = IBClient(test_config)
        
        assert client.config == test_config
        assert not client.is_connected
        assert client.connection_info['connected'] is False
        assert client.next_request_id == 1
    
    def test_init_without_config(self):
        """Test client initialization without config raises error."""
        with pytest.raises(ValueError, match="Config cannot be None"):
            IBClient(None)


class TestConnectionManagement:
    """Test connection management functionality."""
    
    @pytest.fixture
    def mock_client(self, test_config):
        """Fixture for client with mocked IB API."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            # Mock the underlying IB client
            client.connect = Mock()
            client.disconnect = Mock()
            client.run = Mock()
            return client
    
    @pytest.mark.asyncio
    async def test_start_connection(self, mock_client):
        """Test starting connection."""
        # Mock the connection manager's connect_with_retry method
        mock_client.connection_manager.connect_with_retry = AsyncMock(return_value=True)
        
        success = await mock_client.start()
        
        assert success
        mock_client.connection_manager.connect_with_retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_connection_failure(self, mock_client):
        """Test connection failure handling."""
        # Mock the connection manager's connect_with_retry method to fail
        mock_client.connection_manager.connect_with_retry = AsyncMock(return_value=False)
        
        success = await mock_client.start()
        
        assert not success
        assert not mock_client.is_connected
    
    @pytest.mark.asyncio
    async def test_stop_connection(self, mock_client):
        """Test stopping connection."""
        # Replace connection manager with a mock that allows setting is_connected
        mock_connection_manager = Mock()
        mock_connection_manager.disconnect = AsyncMock()
        mock_connection_manager.is_connected = True
        mock_client.connection_manager = mock_connection_manager
        
        # Set the client as running so stop() will actually do work
        mock_client._running = True
        
        await mock_client.stop()
        
        mock_connection_manager.disconnect.assert_called_once()
        # After stopping, client should not be running
        assert not mock_client._running


class TestMarketDataRequests:
    """Test market data request functionality."""
    
    @pytest.fixture
    def connected_client(self, test_config):
        """Fixture for connected client."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.connection_manager = Mock()
            client.connection_manager.is_connected = True
            client._market_data_responses = {}
            client._request_futures = {}
            return client
    
    @pytest.mark.asyncio
    async def test_get_market_data(self, connected_client):
        """Test getting market data."""
        client = connected_client
        
        # Mock contract
        mock_contract = Mock()
        mock_contract.symbol = "AAPL"
        
        # Mock the request market data method
        client.reqMktData = Mock()
        
        # Simulate market data response
        future = asyncio.Future()
        market_data = {
            'bid': 149.50,
            'ask': 150.50,
            'last': 150.00,
            'volume': 1000000
        }
        future.set_result(market_data)
        
        request = Mock()
        request.future = future
        
        with patch.object(client, '_create_request', return_value=request):
            result = await client.get_market_data(mock_contract)
            
            assert result == market_data
            client.reqMktData.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_market_data_timeout(self, connected_client):
        """Test market data request timeout."""
        client = connected_client
        mock_contract = Mock()
        
        client.reqMktData = Mock()
        
        # Create a future that never completes (timeout)
        future = asyncio.Future()
        
        request = Mock()
        request.future = future
        
        with patch.object(client, '_create_request', return_value=request):
            result = await client.get_market_data(mock_contract, timeout=0.1)
            
            # Should return None on timeout
            assert result is None
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connected_client):
        """Test getting historical data."""
        client = connected_client
        
        mock_contract = Mock()
        mock_contract.symbol = "AAPL"
        
        client.reqHistoricalData = Mock()
        
        # Mock historical data response
        historical_data = [
            {
                'timestamp': '2024-01-01 09:30:00',
                'open': 148.00,
                'high': 152.00,
                'low': 147.50,
                'close': 150.00,
                'volume': 1000000
            },
            {
                'timestamp': '2024-01-02 09:30:00',
                'open': 150.00,
                'high': 153.00,
                'low': 149.00,
                'close': 152.00,
                'volume': 1200000
            }
        ]
        
        future = asyncio.Future()
        future.set_result(historical_data)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.get_historical_data(
                contract=mock_contract,
                duration="2 D",
                bar_size="1 day"
            )
            
            assert result == historical_data
            assert len(result) == 2
            client.reqHistoricalData.assert_called_once()


class TestAccountDataRequests:
    """Test account data request functionality."""
    
    @pytest.fixture
    def account_client(self, test_config):
        """Fixture for client with account data mocks."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.connection_manager = Mock()
            client.connection_manager.is_connected = True
            return client
    
    @pytest.mark.asyncio
    async def test_get_positions(self, account_client):
        """Test getting account positions."""
        client = account_client
        
        client.reqPositions = Mock()
        
        # Mock positions response
        from src.core.types import PositionData
        positions = [
            PositionData(
                account="DU123456",
                symbol="AAPL",
                sec_type="STK",
                exchange="NASDAQ",
                currency="USD",
                position=100.0,
                avg_cost=145.0,
                timestamp=datetime.now()
            ),
            PositionData(
                account="DU123456",
                symbol="GOOGL",
                sec_type="STK",
                exchange="NASDAQ",
                currency="USD",
                position=25.0,
                avg_cost=2300.0,
                timestamp=datetime.now()
            )
        ]
        
        future = asyncio.Future()
        future.set_result(positions)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.get_positions()
            
            assert len(result) == 2
            assert result[0].symbol == "AAPL"
            assert result[1].symbol == "GOOGL"
            client.reqPositions.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_account_values(self, account_client):
        """Test getting account values."""
        client = account_client
        
        client.reqAccountSummary = Mock()
        
        # Mock account values response
        from src.core.types import AccountValue
        account_values = [
            AccountValue(
                key="NetLiquidation", 
                value="100000.0", 
                currency="USD",
                account="DU123456",
                timestamp=datetime.now()
            ),
            AccountValue(
                key="BuyingPower", 
                value="50000.0", 
                currency="USD",
                account="DU123456",
                timestamp=datetime.now()
            ),
            AccountValue(
                key="TotalCashValue", 
                value="25000.0", 
                currency="USD",
                account="DU123456",
                timestamp=datetime.now()
            )
        ]
        
        future = asyncio.Future()
        future.set_result(account_values)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.get_account_values()
            
            assert len(result) == 3
            assert result[0].key == "NetLiquidation"
            assert result[1].key == "BuyingPower"
            client.reqAccountSummary.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_pnl(self, account_client):
        """Test getting P&L data."""
        client = account_client
        
        client.reqPnL = Mock()
        
        # Mock P&L response
        pnl_data = {
            "daily_pnl": 1250.75,
            "unrealized_pnl": 850.25,
            "realized_pnl": 400.50,
            "account": "DU123456"
        }
        
        future = asyncio.Future()
        future.set_result(pnl_data)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.get_pnl("DU123456")
            
            assert result["daily_pnl"] == 1250.75
            assert result["account"] == "DU123456"
            client.reqPnL.assert_called_once()


class TestOrderManagement:
    """Test order management functionality."""
    
    @pytest.fixture
    def order_client(self, test_config):
        """Fixture for client with order management mocks."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.connection_manager = Mock()
            client.connection_manager.is_connected = True
            client.placeOrder = Mock()
            client.cancelOrder = Mock()
            client.reqGlobalCancel = Mock()
            return client
    
    @pytest.mark.asyncio
    async def test_place_order(self, order_client):
        """Test placing an order."""
        client = order_client
        
        # Mock order and contract
        mock_contract = Mock()
        mock_order = Mock()
        mock_order.orderId = 123
        
        # Mock order placement response
        order_result = {
            "order_id": 123,
            "status": "submitted",
            "timestamp": datetime.now()
        }
        
        future = asyncio.Future()
        future.set_result(order_result)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.place_order(mock_contract, mock_order)
            
            assert result["order_id"] == 123
            assert result["status"] == "submitted"
            client.placeOrder.assert_called_once_with(123, mock_contract, mock_order)
    
    def test_cancel_order_by_id(self, order_client):
        """Test cancelling order by ID."""
        client = order_client
        
        client.cancel_order_by_id(123)
        
        client.cancelOrder.assert_called_once_with(123)
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, order_client):
        """Test cancelling all orders."""
        client = order_client
        
        # Mock cancel all response
        cancel_result = {
            "status": "completed",
            "total_cancelled": 5
        }
        
        future = asyncio.Future()
        future.set_result(cancel_result)
        
        with patch.object(client, '_create_request_future', return_value=future):
            result = await client.cancel_all_orders()
            
            assert result["status"] == "completed"
            assert result["total_cancelled"] == 5
            client.reqGlobalCancel.assert_called_once()


class TestStreamingData:
    """Test streaming data functionality."""
    
    @pytest.fixture
    def streaming_client(self, test_config):
        """Fixture for client with streaming mocks."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.connection_manager = Mock()
            client.connection_manager.is_connected = True
            client._streaming_data = {}
            return client
    
    @pytest.mark.asyncio
    async def test_get_streaming_data(self, streaming_client):
        """Test getting streaming market data."""
        client = streaming_client
        
        mock_contract = Mock()
        mock_contract.symbol = "AAPL"
        
        # Create mock streaming data generator
        async def mock_stream():
            for i in range(3):
                yield {
                    'symbol': 'AAPL',
                    'price': 150.0 + i,
                    'size': 100,
                    'timestamp': datetime.now()
                }
        
        with patch.object(client, '_start_streaming_data', return_value=mock_stream()):
            stream_gen = client.get_streaming_data(mock_contract)
            
            # Collect streaming data
            data_points = []
            async for data in stream_gen:
                data_points.append(data)
                if len(data_points) >= 3:
                    break
            
            assert len(data_points) == 3
            assert data_points[0]['symbol'] == 'AAPL'
            assert data_points[0]['price'] == 150.0
            assert data_points[2]['price'] == 152.0


class TestErrorHandling:
    """Test error handling in client."""
    
    @pytest.fixture
    def error_client(self, test_config):
        """Fixture for client with error handling tests."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            return client
    
    @pytest.mark.asyncio
    async def test_request_when_not_connected(self, error_client):
        """Test making requests when not connected."""
        client = error_client
        client.connection_manager = Mock()
        client.connection_manager.is_connected = False
        
        mock_contract = Mock()
        
        # Should raise ConnectionError
        with pytest.raises(ConnectionError, match="Not connected to IBKR"):
            await client.get_market_data(mock_contract)
    
    @pytest.mark.asyncio
    async def test_invalid_contract(self, error_client):
        """Test handling invalid contract."""
        client = error_client
        client.connection_manager = Mock()
        client.connection_manager.is_connected = True
        
        # Should raise ValidationError for None contract
        with pytest.raises(ValidationError, match="Contract cannot be None"):
            await client.get_market_data(None)
    
    @pytest.mark.asyncio
    async def test_api_error_response(self, error_client):
        """Test handling API error responses."""
        client = error_client
        client.connection_manager = Mock()
        client.connection_manager.is_connected = True
        
        mock_contract = Mock()
        
        # Mock an API error
        future = asyncio.Future()
        future.set_exception(Exception("API Error: Invalid symbol"))
        
        with patch.object(client, '_create_request_future', return_value=future):
            with pytest.raises(Exception, match="API Error: Invalid symbol"):
                await client.get_market_data(mock_contract)


class TestConnectionInfo:
    """Test connection info functionality."""
    
    def test_connection_info_disconnected(self, test_config):
        """Test connection info when disconnected."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            
            info = client.connection_info
            
            assert info['connected'] is False
            assert info['host'] == test_config.host
            assert info['port'] == test_config.port
            assert info['client_id'] == test_config.client_id
            assert 'connection_time' not in info
    
    def test_connection_info_connected(self, test_config):
        """Test connection info when connected."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.connection_manager = Mock()
            client.connection_manager.is_connected = True
            client._connection_time = datetime.now()
            
            info = client.connection_info
            
            assert info['connected'] is True
            assert 'connection_time' in info
            assert 'uptime_seconds' in info


class TestRequestIdManagement:
    """Test request ID management."""
    
    def test_get_next_request_id(self, test_config):
        """Test getting next request ID."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            
            # Test sequential request IDs
            id1 = client._get_next_request_id()
            id2 = client._get_next_request_id()
            id3 = client._get_next_request_id()
            
            assert id1 == 1
            assert id2 == 2
            assert id3 == 3
            assert client.next_request_id == 4


class TestDataResponseHandling:
    """Test data response handling."""
    
    @pytest.fixture
    def response_client(self, test_config):
        """Fixture for client with response handling tests."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            client.is_connected = True
            client._request_futures = {}
            client._market_data_responses = {}
            client._account_data_responses = {}
            return client
    
    def test_handle_market_data_response(self, response_client):
        """Test handling market data responses."""
        client = response_client
        
        # Mock request future
        request_id = 1
        future = asyncio.Future()
        client._request_futures[request_id] = future
        
        # Simulate market data response
        market_data = {
            'bid': 149.50,
            'ask': 150.50,
            'last': 150.00
        }
        
        # This would typically be called by the IB API wrapper
        client._handle_market_data_response(request_id, market_data)
        
        # Future should be completed with market data
        assert future.done()
        assert not future.cancelled()
        assert future.result() == market_data
    
    def test_handle_error_response(self, response_client):
        """Test handling error responses."""
        client = response_client
        
        # Mock request future
        request_id = 1
        future = asyncio.Future()
        client._request_futures[request_id] = future
        
        # Simulate error response
        error_msg = "Invalid symbol"
        
        # This would typically be called by the IB API wrapper
        client._handle_error_response(request_id, error_msg)
        
        # Future should be completed with exception
        assert future.done()
        assert future.exception() is not None
        assert str(future.exception()) == error_msg


class TestClientIntegration:
    """Integration tests for IBClient."""
    
    @pytest.mark.asyncio
    async def test_complete_client_workflow(self, test_config):
        """Test complete client workflow."""
        with patch('src.api.client.EClient'), \
             patch('src.api.client.EWrapper'):
            
            client = IBClient(test_config)
            
            # Mock successful connection
            client.connect = Mock(return_value=True)
            client.disconnect = Mock()
            
            try:
                # Start connection
                success = await client.start()
                assert success
                
                # Mock connected state
                client.connection_manager = Mock()
                client.connection_manager.is_connected = True
                
                # Test market data request
                mock_contract = Mock()
                future = asyncio.Future()
                future.set_result({'last': 150.0})
                
                with patch.object(client, '_create_request_future', return_value=future):
                    data = await client.get_market_data(mock_contract)
                    assert data['last'] == 150.0
                
                # Test account data request
                future = asyncio.Future()
                future.set_result([])
                
                with patch.object(client, '_create_request_future', return_value=future):
                    positions = await client.get_positions()
                    assert isinstance(positions, list)
                
            finally:
                # Stop connection
                await client.stop()
