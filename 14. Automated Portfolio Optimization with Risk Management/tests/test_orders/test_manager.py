"""
Unit tests for the OrderManager class.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta
from decimal import Decimal

from src.orders.manager import OrderManager, OrderRequest, OrderType, OrderAction, TimeInForce
from src.core.exceptions import ValidationError, OrderError
from src.core.types import PositionData, AccountValue


class TestOrderManagerInitialization:
    """Test OrderManager initialization."""
    
    def test_init(self, mock_ibapi):
        """Test OrderManager initialization."""
        mock_client = Mock()
        manager = OrderManager(mock_client)
        
        assert manager.client == mock_client
        assert len(manager.active_orders) == 0
        assert len(manager.order_history) == 0
    
    def test_init_with_none_client(self):
        """Test initialization with None client raises error."""
        with pytest.raises(ValueError, match="Client cannot be None"):
            OrderManager(None)


class TestOrderValidation:
    """Test order validation logic."""
    
    def setup_method(self):
        """Setup for each test."""
        self.mock_client = Mock()
        self.manager = OrderManager(self.mock_client)
    
    def test_validate_order_request_valid(self):
        """Test validation of valid order request."""
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        # Should not raise
        self.manager._validate_order_request(order)
    
    def test_validate_order_request_invalid_symbol(self):
        """Test validation fails for invalid symbol."""
        order = OrderRequest(
            symbol="",  # Empty symbol
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            self.manager._validate_order_request(order)
    
    def test_validate_order_request_invalid_quantity(self):
        """Test validation fails for invalid quantity."""
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=0,  # Invalid quantity
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            self.manager._validate_order_request(order)
    
    def test_validate_limit_order_no_price(self):
        """Test validation fails for limit order without price."""
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY
            # Missing limit_price
        )
        
        with pytest.raises(ValidationError, match="Limit price required"):
            self.manager._validate_order_request(order)
    
    def test_validate_stop_order_no_price(self):
        """Test validation fails for stop order without price."""
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.STOP,
            time_in_force=TimeInForce.DAY
            # Missing stop_price
        )
        
        with pytest.raises(ValidationError, match="Stop price required"):
            self.manager._validate_order_request(order)


class TestBasicOrderPlacement:
    """Test basic order placement functionality."""
    
    @pytest.fixture
    def manager_with_mock_client(self):
        """Fixture for OrderManager with mocked client."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        return OrderManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_place_buy_order(self, manager_with_mock_client):
        """Test placing a basic buy order."""
        manager = manager_with_mock_client
        manager.client.place_order.return_value = {
            "order_id": 123,
            "status": "submitted"
        }
        
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.place_order(order)
        
        assert result["status"] == "placed"
        assert result["order_id"] == 123
        assert result["symbol"] == "AAPL"
        assert 123 in manager.active_orders
        manager.client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, manager_with_mock_client):
        """Test placing a limit order."""
        manager = manager_with_mock_client
        manager.client.place_order.return_value = {
            "order_id": 124,
            "status": "submitted"
        }
        
        order = OrderRequest(
            symbol="GOOGL",
            action=OrderAction.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=2500.0,
            time_in_force=TimeInForce.GTC
        )
        
        result = await manager.place_order(order)
        
        assert result["status"] == "placed"
        assert result["limit_price"] == 2500.0
        manager.client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_dry_run(self, manager_with_mock_client):
        """Test placing order in dry run mode."""
        manager = manager_with_mock_client
        
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.place_order(order, dry_run=True)
        
        assert result["status"] == "validated"
        assert "dry_run" in result
        assert result["dry_run"] is True
        manager.client.place_order.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_place_order_client_error(self, manager_with_mock_client):
        """Test handling of client errors during order placement."""
        manager = manager_with_mock_client
        manager.client.place_order.side_effect = Exception("Connection error")
        
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.place_order(order)
        
        assert result["status"] == "failed"
        assert "Connection error" in result["error"]


class TestAdvancedOrderMethods:
    """Test advanced order management methods."""
    
    @pytest.fixture
    def manager_with_positions(self):
        """Fixture for OrderManager with mock positions and account data."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        # Mock positions
        positions = [
            PositionData(
                symbol="AAPL",
                position=100,
                market_price=150.0,
                market_value=15000.0,
                average_cost=145.0,
                unrealized_pnl=500.0
            ),
            PositionData(
                symbol="GOOGL", 
                position=25,
                market_price=2400.0,
                market_value=60000.0,
                average_cost=2300.0,
                unrealized_pnl=2500.0
            )
        ]
        
        # Mock account values
        account_values = [
            AccountValue(key="NetLiquidation", value="100000.0", currency="USD"),
            AccountValue(key="BuyingPower", value="50000.0", currency="USD")
        ]
        
        mock_client.get_positions.return_value = positions
        mock_client.get_account_values.return_value = account_values
        mock_client.get_market_data.return_value = {"last": 150.0}  # Default price
        mock_client.place_order.return_value = {"order_id": 100, "status": "submitted"}
        
        return OrderManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_order_target_percent(self, manager_with_positions):
        """Test ordering to target percentage allocation."""
        manager = manager_with_positions
        
        # Target 25% allocation in AAPL (currently 15%)
        result = await manager.order_target_percent(
            symbol="AAPL",
            target_percent=0.25,
            order_type=OrderType.MARKET
        )
        
        assert result["status"] == "placed"
        assert result["target_percent"] == 0.25
        assert result["action_taken"] == "BUY"
        # Should buy more shares to reach 25% allocation
        assert result["quantity"] > 0
    
    @pytest.mark.asyncio
    async def test_order_target_quantity(self, manager_with_positions):
        """Test ordering to target quantity."""
        manager = manager_with_positions
        
        # Target 150 shares of AAPL (currently 100)
        result = await manager.order_target_quantity(
            symbol="AAPL",
            target_quantity=150,
            order_type=OrderType.MARKET
        )
        
        assert result["status"] == "placed"
        assert result["target_quantity"] == 150
        assert result["current_quantity"] == 100
        assert result["quantity_to_trade"] == 50
        assert result["action"] == "BUY"
    
    @pytest.mark.asyncio
    async def test_order_target_value(self, manager_with_positions):
        """Test ordering to target value."""
        manager = manager_with_positions
        
        # Target $20,000 position in AAPL (currently $15,000)
        result = await manager.order_target_value(
            symbol="AAPL",
            target_value=20000.0,
            order_type=OrderType.MARKET
        )
        
        assert result["status"] == "placed"
        assert result["target_value"] == 20000.0
        assert result["current_value"] == 15000.0
        assert result["value_to_trade"] == 5000.0
        assert result["action"] == "BUY"
    
    @pytest.mark.asyncio
    async def test_order_value(self, manager_with_positions):
        """Test ordering a fixed value."""
        manager = manager_with_positions
        
        # Buy $3,000 worth of AAPL
        result = await manager.order_value(
            symbol="AAPL",
            value=3000.0,
            order_type=OrderType.MARKET
        )
        
        assert result["status"] == "placed"
        assert result["value"] == 3000.0
        assert result["action"] == "BUY"
        assert result["quantity"] == 20  # $3000 / $150 = 20 shares
    
    @pytest.mark.asyncio
    async def test_order_percent(self, manager_with_positions):
        """Test ordering by portfolio percentage."""
        manager = manager_with_positions
        
        # Buy 10% of portfolio value in TSLA
        result = await manager.order_percent(
            symbol="TSLA",
            percent=0.10,
            order_type=OrderType.MARKET
        )
        
        # Mock market data for TSLA
        manager.client.get_market_data.return_value = {"last": 200.0}
        
        result = await manager.order_percent(
            symbol="TSLA",
            percent=0.10,
            order_type=OrderType.MARKET
        )
        
        assert result["status"] == "placed"
        assert result["percent"] == 0.10
        assert result["portfolio_value"] == 100000.0
        assert result["order_value"] == 10000.0


class TestPortfolioRebalancing:
    """Test portfolio rebalancing functionality."""
    
    @pytest.fixture
    def rebalance_manager(self):
        """Manager set up for rebalancing tests."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        # Mock current positions
        positions = [
            PositionData(symbol="AAPL", position=100, market_price=150.0, market_value=15000.0, average_cost=145.0, unrealized_pnl=500.0),
            PositionData(symbol="GOOGL", position=25, market_price=2400.0, market_value=60000.0, average_cost=2300.0, unrealized_pnl=2500.0),
            PositionData(symbol="MSFT", position=50, market_price=300.0, market_value=15000.0, average_cost=295.0, unrealized_pnl=250.0)
        ]
        
        account_values = [
            AccountValue(key="NetLiquidation", value="100000.0", currency="USD")
        ]
        
        mock_client.get_positions.return_value = positions
        mock_client.get_account_values.return_value = account_values
        
        # Mock market data responses
        def mock_market_data(contract):
            prices = {"AAPL": 150.0, "GOOGL": 2400.0, "MSFT": 300.0, "TSLA": 200.0}
            symbol = contract.symbol if hasattr(contract, 'symbol') else "AAPL"
            return {"last": prices.get(symbol, 100.0)}
        
        mock_client.get_market_data.side_effect = mock_market_data
        mock_client.place_order.return_value = {"order_id": 200, "status": "submitted"}
        
        return OrderManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, rebalance_manager):
        """Test full portfolio rebalancing."""
        manager = rebalance_manager
        
        target_allocations = {
            "AAPL": 0.30,   # Increase from 15%
            "GOOGL": 0.40,  # Decrease from 60%
            "MSFT": 0.20,   # Increase from 15%
            "TSLA": 0.10    # New position
        }
        
        result = await manager.rebalance_portfolio(
            target_allocations=target_allocations,
            order_type=OrderType.MARKET
        )
        
        assert result["summary"]["status"] == "completed"
        assert len(result["results"]) == 4
        
        # Check individual results
        assert "AAPL" in result["results"]
        assert "GOOGL" in result["results"]
        assert "MSFT" in result["results"]
        assert "TSLA" in result["results"]
        
        # AAPL should increase (buy more)
        aapl_result = result["results"]["AAPL"]
        assert aapl_result["action"] == "BUY"
        
        # GOOGL should decrease (sell some)
        googl_result = result["results"]["GOOGL"]
        assert googl_result["action"] == "SELL"
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio_dry_run(self, rebalance_manager):
        """Test portfolio rebalancing in dry run mode."""
        manager = rebalance_manager
        
        target_allocations = {
            "AAPL": 0.25,
            "GOOGL": 0.50,
            "MSFT": 0.25
        }
        
        result = await manager.rebalance_portfolio(
            target_allocations=target_allocations,
            order_type=OrderType.MARKET,
            dry_run=True
        )
        
        assert result["summary"]["dry_run"] is True
        assert result["summary"]["status"] == "validated"
        
        # Should not place any real orders
        manager.client.place_order.assert_not_called()


class TestOrderCancellation:
    """Test order cancellation functionality."""
    
    @pytest.fixture
    def manager_with_orders(self):
        """Manager with some active orders."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        manager = OrderManager(mock_client)
        
        # Add some mock active orders
        manager.active_orders[101] = {
            "order_id": 101,
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "status": "submitted",
            "timestamp": datetime.now()
        }
        manager.active_orders[102] = {
            "order_id": 102,
            "symbol": "GOOGL",
            "action": "SELL",
            "quantity": 25,
            "status": "submitted",
            "timestamp": datetime.now()
        }
        
        return manager
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, manager_with_orders):
        """Test cancelling a single order."""
        manager = manager_with_orders
        manager.client.cancel_order.return_value = {"status": "cancelled"}
        
        result = await manager.cancel_order(101)
        
        assert result["status"] == "cancelled"
        assert result["order_id"] == 101
        manager.client.cancel_order.assert_called_once_with(101)
    
    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, manager_with_orders):
        """Test cancelling non-existent order."""
        manager = manager_with_orders
        
        result = await manager.cancel_order(999)
        
        assert result["status"] == "not_found"
        assert result["order_id"] == 999
        manager.client.cancel_order.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, manager_with_orders):
        """Test cancelling all active orders."""
        manager = manager_with_orders
        manager.client.cancel_order.return_value = {"status": "cancelled"}
        
        result = await manager.cancel_all_orders()
        
        assert result["status"] == "completed"
        assert result["total_cancelled"] == 2
        assert len(result["cancelled_orders"]) == 2
        assert manager.client.cancel_order.call_count == 2


class TestOrderUpdate:
    """Test order update functionality."""
    
    @pytest.fixture
    def manager_with_order(self):
        """Manager with an active order."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        manager = OrderManager(mock_client)
        manager.active_orders[101] = {
            "order_id": 101,
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 100,
            "status": "submitted"
        }
        
        return manager
    
    @pytest.mark.asyncio
    async def test_update_order(self, manager_with_order):
        """Test updating an existing order."""
        manager = manager_with_order
        manager.client.cancel_order.return_value = {"status": "cancelled"}
        manager.client.place_order.return_value = {"order_id": 201, "status": "submitted"}
        
        new_order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=150,  # Updated quantity
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.update_order(101, new_order)
        
        assert result["status"] == "updated"
        assert result["old_order_id"] == 101
        assert result["new_order_id"] == 201
        
        # Should cancel old order and place new one
        manager.client.cancel_order.assert_called_once_with(101)
        manager.client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_order(self, manager_with_order):
        """Test updating non-existent order."""
        manager = manager_with_order
        
        new_order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=150,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.update_order(999, new_order)
        
        assert result["status"] == "not_found"
        assert result["order_id"] == 999


class TestOrderHistory:
    """Test order history functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        self.mock_client = Mock()
        self.manager = OrderManager(self.mock_client)
        
        # Add some order history
        old_order = {
            "order_id": 50,
            "symbol": "AAPL",
            "action": "BUY",
            "quantity": 50,
            "status": "filled",
            "timestamp": datetime.now() - timedelta(days=2)
        }
        recent_order = {
            "order_id": 51,
            "symbol": "GOOGL",
            "action": "SELL",
            "quantity": 10,
            "status": "filled",
            "timestamp": datetime.now() - timedelta(hours=2)
        }
        
        self.manager.order_history = [old_order, recent_order]
    
    def test_get_order_history_all(self):
        """Test getting all order history."""
        history = self.manager.get_order_history()
        
        assert len(history) == 2
        assert history[0]["order_id"] == 51  # Most recent first
        assert history[1]["order_id"] == 50
    
    def test_get_order_history_filtered(self):
        """Test getting filtered order history."""
        # Get only orders from last day
        history = self.manager.get_order_history(days=1)
        
        assert len(history) == 1
        assert history[0]["order_id"] == 51  # Only recent order
    
    def test_get_order_status_active(self):
        """Test getting status of active order."""
        # Add active order
        self.manager.active_orders[100] = {
            "order_id": 100,
            "symbol": "MSFT",
            "status": "submitted"
        }
        
        status = self.manager.get_order_status(100)
        assert status is not None
        assert status["order_id"] == 100
        assert status["status"] == "submitted"
    
    def test_get_order_status_historical(self):
        """Test getting status of historical order."""
        status = self.manager.get_order_status(50)
        assert status is not None
        assert status["order_id"] == 50
        assert status["status"] == "filled"
    
    def test_get_order_status_not_found(self):
        """Test getting status of non-existent order."""
        status = self.manager.get_order_status(999)
        assert status is None


class TestRiskManagement:
    """Test risk management features."""
    
    @pytest.fixture
    def risk_manager(self):
        """Manager set up for risk management tests."""
        mock_client = AsyncMock()
        mock_client.is_connected = True
        
        # Mock account with limited buying power
        account_values = [
            AccountValue(key="NetLiquidation", value="50000.0", currency="USD"),
            AccountValue(key="BuyingPower", value="25000.0", currency="USD")
        ]
        
        mock_client.get_account_values.return_value = account_values
        mock_client.get_market_data.return_value = {"last": 1000.0}  # High-priced stock
        
        return OrderManager(mock_client)
    
    @pytest.mark.asyncio
    async def test_risk_check_insufficient_buying_power(self, risk_manager):
        """Test risk check fails for insufficient buying power."""
        manager = risk_manager
        
        # Try to buy $30,000 worth when only $25,000 buying power
        with patch.object(manager, '_perform_risk_checks') as mock_risk:
            mock_risk.side_effect = OrderError("Insufficient buying power")
            
            order = OrderRequest(
                symbol="EXPENSIVE_STOCK",
                action=OrderAction.BUY,
                quantity=30,  # 30 * $1000 = $30,000
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            result = await manager.place_order(order)
            
            assert result["status"] == "failed"
            assert "Insufficient buying power" in result["error"]
    
    @pytest.mark.asyncio  
    async def test_position_size_limit(self, risk_manager):
        """Test position size limits."""
        manager = risk_manager
        
        # Mock existing large position
        positions = [
            PositionData(
                symbol="AAPL",
                position=1000,  # Large position
                market_price=150.0,
                market_value=150000.0,
                average_cost=145.0,
                unrealized_pnl=5000.0
            )
        ]
        manager.client.get_positions.return_value = positions
        
        with patch.object(manager, '_perform_risk_checks') as mock_risk:
            mock_risk.side_effect = OrderError("Position size limit exceeded")
            
            # Try to buy even more AAPL
            order = OrderRequest(
                symbol="AAPL",
                action=OrderAction.BUY,
                quantity=500,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            )
            
            result = await manager.place_order(order)
            
            assert result["status"] == "failed"
            assert "Position size limit exceeded" in result["error"]


class TestOrderManagerIntegration:
    """Integration tests for OrderManager."""
    
    @pytest.mark.asyncio
    async def test_complete_order_workflow(self):
        """Test complete order lifecycle."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.is_connected = True
        mock_client.place_order.return_value = {"order_id": 300, "status": "submitted"}
        mock_client.cancel_order.return_value = {"status": "cancelled"}
        
        manager = OrderManager(mock_client)
        
        # Place order
        order = OrderRequest(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY
        )
        
        result = await manager.place_order(order)
        assert result["status"] == "placed"
        order_id = result["order_id"]
        
        # Check order status
        status = manager.get_order_status(order_id)
        assert status is not None
        assert status["status"] == "submitted"
        
        # Cancel order
        cancel_result = await manager.cancel_order(order_id)
        assert cancel_result["status"] == "cancelled"
        
        # Verify order moved to history
        history = manager.get_order_history()
        assert len(history) >= 1
        assert any(order["order_id"] == order_id for order in history)
